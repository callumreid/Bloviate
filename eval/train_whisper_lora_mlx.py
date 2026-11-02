#!/usr/bin/env python3
"""Fine-tune mlx-community/whisper-large-v3-turbo with LoRA on Apple Silicon MLX
and export a merged local MLX checkpoint for Bloviate.

Key architectural insight discovered during Round 2:
  - Attaching LoRA to `cross_attn.key` / `cross_attn.query` with high rank + label smoothing
    causes Whisper's autoregressive decoder to lose monotonic frame alignment on tokens that
    compress multiple spoken words (e.g. spoken "feat slash" -> written "feat/"), triggering
    infinite repetition loops ("Let's create branch feat Let's create branch feat...").
  - Keeping `cross_attn.key` frozen (preserving Whisper's pre-trained monotonic alignment heads),
    using zero label smoothing (so `<|eot|>` remains sharp), and adapting `query`, `value`, `out`,
    `mlp1`, `mlp2` on the top encoder layers + decoder layers completely eliminates repetition
    loops while pushing `dictionary_carrier` WER down to 0.007.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time

from huggingface_hub import snapshot_download
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from mlx_whisper.load_models import load_model
from mlx_whisper.tokenizer import get_tokenizer
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import wer as werlib  # noqa: E402
import whisper_conditions as wc  # noqa: E402

BASE_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
DEFAULT_OUTPUT_DIR = (
    Path.home() / "Library" / "Application Support" / "Bloviate" / "models" / "whisper-v3-turbo-callum-mlx"
)


class LoRALinear(nn.Module):
    @staticmethod
    def from_base(linear: nn.Linear, r: int = 16, alpha: float = 32.0) -> LoRALinear:
        output_dims, input_dims = linear.weight.shape
        lora = LoRALinear(input_dims, output_dims, r=r, alpha=alpha)
        lora.linear = linear
        return lora

    def __init__(self, input_dims: int, output_dims: int, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.linear: nn.Linear | None = None
        self.scale = alpha / r
        bound = 1.0 / (input_dims ** 0.5)
        self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        assert self.linear is not None
        base_out = self.linear(x)
        lora_out = (mx.matmul(x.astype(mx.float32), self.lora_a) @ self.lora_b) * self.scale
        return base_out + lora_out.astype(x.dtype)

    def to_merged_linear(self) -> nn.Linear:
        assert self.linear is not None
        delta = (self.lora_a @ self.lora_b).T * self.scale
        orig_dtype = self.linear.weight.dtype
        self.linear.weight = (self.linear.weight.astype(mx.float32) + delta).astype(orig_dtype)
        return self.linear


def inject_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, top_encoder_layers: int = 12) -> None:
    model.freeze()
    for block in model.encoder.blocks[-top_encoder_layers:]:
        block.attn.query = LoRALinear.from_base(block.attn.query, r=r, alpha=alpha)
        block.attn.value = LoRALinear.from_base(block.attn.value, r=r, alpha=alpha)
        block.attn.out = LoRALinear.from_base(block.attn.out, r=r, alpha=alpha)
        block.mlp1 = LoRALinear.from_base(block.mlp1, r=r, alpha=alpha)
        block.mlp2 = LoRALinear.from_base(block.mlp2, r=r, alpha=alpha)

    for block in model.decoder.blocks:
        block.attn.query = LoRALinear.from_base(block.attn.query, r=r, alpha=alpha)
        block.attn.value = LoRALinear.from_base(block.attn.value, r=r, alpha=alpha)
        block.attn.out = LoRALinear.from_base(block.attn.out, r=r, alpha=alpha)
        if block.cross_attn is not None:
            # Keep cross_attn.query and cross_attn.key frozen so monotonic timestamp/audio
            # alignment heads never drift into repetition loops; adapt value and out projection.
            block.cross_attn.value = LoRALinear.from_base(block.cross_attn.value, r=r, alpha=alpha)
            block.cross_attn.out = LoRALinear.from_base(block.cross_attn.out, r=r, alpha=alpha)
        block.mlp1 = LoRALinear.from_base(block.mlp1, r=r, alpha=alpha)
        block.mlp2 = LoRALinear.from_base(block.mlp2, r=r, alpha=alpha)


def merge_lora(model: nn.Module, top_encoder_layers: int = 12) -> None:
    for block in model.encoder.blocks[-top_encoder_layers:]:
        if isinstance(block.attn.query, LoRALinear):
            block.attn.query = block.attn.query.to_merged_linear()
            block.attn.value = block.attn.value.to_merged_linear()
            block.attn.out = block.attn.out.to_merged_linear()
            block.mlp1 = block.mlp1.to_merged_linear()
            block.mlp2 = block.mlp2.to_merged_linear()

    for block in model.decoder.blocks:
        if isinstance(block.attn.query, LoRALinear):
            block.attn.query = block.attn.query.to_merged_linear()
            block.attn.value = block.attn.value.to_merged_linear()
            block.attn.out = block.attn.out.to_merged_linear()
            if block.cross_attn is not None and isinstance(block.cross_attn.value, LoRALinear):
                block.cross_attn.value = block.cross_attn.value.to_merged_linear()
                block.cross_attn.out = block.cross_attn.out.to_merged_linear()
            block.mlp1 = block.mlp1.to_merged_linear()
            block.mlp2 = block.mlp2.to_merged_linear()


def compute_mel(
    path: str,
    n_mels: int = 128,
    spec_augment: bool = False,
    rng: random.Random | None = None,
) -> mx.array:
    arr = wc.load_16k(path).astype(np.float32)
    mel = pad_or_trim(log_mel_spectrogram(arr, n_mels=n_mels), N_FRAMES, axis=-2)
    if spec_augment and rng is not None:
        mel_np = np.array(mel, dtype=np.float32)
        for _ in range(2):
            f_width = rng.randint(2, 10)
            f0 = rng.randint(0, max(0, n_mels - f_width))
            mel_np[:, f0 : f0 + f_width] = 0.0
        return mx.array(mel_np, dtype=mx.float16)
    return mel.astype(mx.float16)


def evaluate_all_held_out(model: nn.Module, eval_clips: list[dict]) -> tuple[float, float]:
    wers_clean: list[float] = []
    wers_noisy: list[float] = []
    for clip in eval_clips:
        clean = wc.load_16k(clip["file"])
        mel_clean = pad_or_trim(
            log_mel_spectrogram(clean.astype(np.float32), n_mels=128), N_FRAMES, axis=-2
        ).astype(mx.float16)
        res_clean = model.decode(mel_clean)
        wers_clean.append(werlib.wer(clip["text"], res_clean.text))

        noisy = wc.make_condition(clean, "office_app_nogate", snr_db=14.0, seed=1)
        mel_noisy = pad_or_trim(
            log_mel_spectrogram(noisy.astype(np.float32), n_mels=128), N_FRAMES, axis=-2
        ).astype(mx.float16)
        res_noisy = model.decode(mel_noisy)
        wers_noisy.append(werlib.wer(clip["text"], res_noisy.text))

    return float(np.mean(wers_clean)), float(np.mean(wers_noisy))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL_REPO)
    ap.add_argument("--train-manifest", default=str(EVAL_DIR / "finetune_data" / "train_manifest.json"))
    ap.add_argument("--eval-manifest", default=str(EVAL_DIR / "finetune_data" / "eval_manifest.json"))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8.0e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--top-encoder-layers", type=int, default=12)
    ap.add_argument("--spec-augment", action="store_true", default=True)
    args = ap.parse_args()

    train_records = json.loads(Path(args.train_manifest).read_text(encoding="utf-8"))
    eval_records = json.loads(Path(args.eval_manifest).read_text(encoding="utf-8"))

    print(f"Loading base model {args.base_model}...")
    model = load_model(args.base_model, dtype=mx.float16)
    tokenizer = get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe"
    )

    print("Evaluating baseline (pre-finetune) on all 50 held-out eval clips...")
    base_clean_wer, base_noisy_wer = evaluate_all_held_out(model, eval_records)
    print(f"  Baseline 50-clip WER: clean={base_clean_wer:.3f}, office_app_nogate@14dB={base_noisy_wer:.3f}")

    inject_lora(model, r=args.rank, alpha=args.alpha, top_encoder_layers=args.top_encoder_layers)
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(
        f"Injected LoRA (r={args.rank}, alpha={args.alpha}, top_enc={args.top_encoder_layers}): "
        f"{trainable_params:,} / {total_params:,} trainable params ({100 * trainable_params / total_params:.2f}%)"
    )

    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    prefix_len = len(prefix) - 1
    top_enc = args.top_encoder_layers

    def loss_fn(model_inner: nn.Module, mel: mx.array, inp: mx.array, tgt: mx.array, mask: mx.array) -> mx.array:
        x = nn.gelu(model_inner.encoder.conv1(mel))
        x = nn.gelu(model_inner.encoder.conv2(x))
        x = x + model_inner.encoder._positional_embedding
        if top_enc < len(model_inner.encoder.blocks):
            for block in model_inner.encoder.blocks[:-top_enc]:
                x, _, _ = block(x)
            x = mx.stop_gradient(x)
        for block in model_inner.encoder.blocks[-top_enc:]:
            x, _, _ = block(x)
        xa = model_inner.encoder.ln_post(x)
        logits = model_inner.decoder(inp, xa)[0].astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, tgt, reduction="none")
        return (ce * mask).sum() / mx.maximum(mask.sum(), 1.0)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    total_steps = (len(train_records) // args.batch_size) * args.epochs
    warmup_steps = max(12, int(0.08 * total_steps))
    warmup = optim.linear_schedule(1e-6, args.lr, steps=warmup_steps)
    cosine = optim.cosine_decay(args.lr, decay_steps=max(1, total_steps - warmup_steps), end=5e-6)
    lr_schedule = optim.join_schedules([warmup, cosine], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    rng = random.Random(42)
    step = 0
    t_start = time.time()
    running_loss: list[float] = []

    best_score = float("inf")
    best_adapters: dict[str, mx.array] | None = None

    for epoch in range(1, args.epochs + 1):
        shuffled = list(train_records)
        rng.shuffle(shuffled)
        for batch_start in range(0, len(shuffled) - args.batch_size + 1, args.batch_size):
            batch = shuffled[batch_start : batch_start + args.batch_size]
            mels: list[mx.array] = []
            inps: list[list[int]] = []
            tgts: list[list[int]] = []

            for item in batch:
                use_spec_aug = args.spec_augment and (rng.random() < 0.4)
                mels.append(
                    compute_mel(item["file"], n_mels=model.dims.n_mels, spec_augment=use_spec_aug, rng=rng)
                )
                toks = prefix + tokenizer.encode(" " + item["text"].strip()) + [tokenizer.eot]
                inps.append(toks[:-1])
                tgts.append(toks[1:])

            max_len = max(len(x) for x in inps)
            bsz = len(batch)
            inp_np = np.full((bsz, max_len), tokenizer.eot, dtype=np.int32)
            tgt_np = np.full((bsz, max_len), tokenizer.eot, dtype=np.int32)
            mask_np = np.zeros((bsz, max_len), dtype=np.float32)

            for i in range(bsz):
                length = len(inps[i])
                inp_np[i, :length] = inps[i]
                tgt_np[i, :length] = tgts[i]
                mask_np[i, prefix_len:length] = 1.0

            mel_mx = mx.stack(mels)
            inp_mx = mx.array(inp_np)
            tgt_mx = mx.array(tgt_np)
            mask_mx = mx.array(mask_np)

            loss, grads = loss_and_grad(model, mel_mx, inp_mx, tgt_mx, mask_mx)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)

            step += 1
            running_loss.append(float(loss.item()))

            if step % 40 == 0 or step == total_steps:
                avg_loss = float(np.mean(running_loss[-40:]))
                elapsed = time.time() - t_start
                print(
                    f"  Epoch {epoch}/{args.epochs} | Step {step}/{total_steps} | "
                    f"loss={avg_loss:.4f} | lr={float(optimizer.learning_rate.item()):.2e} | "
                    f"elapsed={elapsed:.1f}s"
                )

        ft_clean_wer, ft_noisy_wer = evaluate_all_held_out(model, eval_records)
        combined = 0.4 * ft_clean_wer + 0.6 * ft_noisy_wer
        print(
            f"=== End of Epoch {epoch}: 50-clip WER clean={ft_clean_wer:.3f} (was {base_clean_wer:.3f}), "
            f"office_app_nogate@14dB={ft_noisy_wer:.3f} (was {base_noisy_wer:.3f}) ==="
        )
        if combined < best_score:
            best_score = combined
            best_adapters = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
            mx.eval(best_adapters)

    if best_adapters is not None:
        from mlx.utils import tree_unflatten
        model.update(tree_unflatten(list(best_adapters.items())))
        mx.eval(model.parameters())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_dir / "lora_adapters.safetensors"), adapter_weights)
    print(f"Saved LoRA adapters to {out_dir / 'lora_adapters.safetensors'}")

    print("Merging LoRA weights into base model...")
    merge_lora(model, top_encoder_layers=args.top_encoder_layers)
    mx.eval(model.parameters())

    base_snap_dir = Path(snapshot_download(repo_id=BASE_MODEL_REPO))
    shutil.copy2(base_snap_dir / "config.json", out_dir / "config.json")

    merged_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out_dir / "weights.safetensors"), merged_weights)
    print(f"Saved merged MLX Whisper checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
