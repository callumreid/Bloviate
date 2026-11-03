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

Round 3 (600 packed same-condition windows + 50% timestamped targets) regressed:
  per-clip clean WER 0.050 -> 0.059, LPC-unvoiced 0.053 -> 0.063, and chunked long-form
  decoding loops ("Make sure Miley Miley Miley...") at 0.178 vs 0.068 clean. Attributed to
  packed windows concatenating many near-identical templated dictionary_carrier sentences,
  teaching the decoder that repetition is normal. Packing is OFF by default; the flags remain
  for experimentation.
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
    Path.home() / "Library" / "Application Support" / "Bloviate" / "models" / "whisper-v3-turbo-callum-mlx-r3"
)
SAMPLE_RATE = 16000
TIMESTAMP_QUANTUM_S = 0.02


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


def split_dev_set(
    train_records: list[dict], n_dev_base_ids: int, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    """Hold out ``n_dev_base_ids`` base_ids (all their conditions) as a dev set."""
    base_ids = sorted({r["base_id"] for r in train_records})
    rng.shuffle(base_ids)
    dev_ids = set(base_ids[:n_dev_base_ids])
    dev = [r for r in train_records if r["base_id"] in dev_ids]
    train = [r for r in train_records if r["base_id"] not in dev_ids]
    return train, dev


def build_dev_eval_records(dev_records: list[dict]) -> list[dict]:
    """Build one clean-source eval entry per dev base_id from raw_16k files.

    The train manifest's `file` points at already-noise-conditioned audio; using
    those for eval would double-apply noise. Instead we point at the clean
    `finetune_data/raw_16k/<base_id>.wav` source, mirroring how eval_manifest
    references raw_16k files. Asserts the clean files exist.
    """
    raw_dir = EVAL_DIR / "finetune_data" / "raw_16k"
    by_base: dict[str, dict] = {}
    for r in dev_records:
        by_base.setdefault(r["base_id"], r)
    out: list[dict] = []
    for base_id, rec in by_base.items():
        clean_path = raw_dir / f"{base_id}.wav"
        assert clean_path.is_file(), f"Missing clean dev source: {clean_path}"
        out.append({"file": str(clean_path), "text": rec["text"]})
    return out


def build_packed_window(
    records: list[dict],
    rng: random.Random,
    max_total_s: float = 29.0,
    distinct_categories: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int, str]]] | None:
    """Concatenate 2-6 same-condition records into one window <= ``max_total_s``.

    Returns (audio, segments) where segments is [(start_sample, end_sample, text), ...].
    Returns None if fewer than 2 records are available for the chosen condition.

    When ``distinct_categories`` is True, no two records in the window share a
    ``category`` and at most one ``dictionary_carrier`` record is allowed, so
    templated sentences cannot repeat inside a window.
    """
    by_condition: dict[str, list[dict]] = {}
    for r in records:
        by_condition.setdefault(r["condition"], []).append(r)
    conditions = [c for c, rs in by_condition.items() if len(rs) >= 2]
    if not conditions:
        return None
    condition = rng.choice(conditions)
    pool = list(by_condition[condition])
    rng.shuffle(pool)

    if distinct_categories:
        # Filter the pool so each category appears at most once, and at most one
        # dictionary_carrier record is kept.
        seen_cats: set[str] = set()
        dict_carrier_used = False
        filtered: list[dict] = []
        for rec in pool:
            cat = rec.get("category", "")
            if cat in seen_cats:
                continue
            if cat == "dictionary_carrier" and dict_carrier_used:
                continue
            seen_cats.add(cat)
            if cat == "dictionary_carrier":
                dict_carrier_used = True
            filtered.append(rec)
        pool = filtered

    parts: list[np.ndarray] = []
    segments: list[tuple[int, int, str]] = []
    cursor = 0
    leading_pad = rng.uniform(0.0, 0.5)
    if leading_pad > 0:
        parts.append(np.zeros(int(leading_pad * SAMPLE_RATE), dtype=np.float32))
        cursor = len(parts[0])
    count = 0
    for rec in pool:
        if count >= 6:
            break
        audio = wc.load_16k(rec["file"]).astype(np.float32)
        gap = rng.uniform(0.3, 0.9)
        gap_samples = int(gap * SAMPLE_RATE)
        start = cursor
        # Would adding this clip (plus a gap before it) exceed the budget?
        if count >= 2 and (cursor + (gap_samples if count > 0 else 0) + len(audio)) / SAMPLE_RATE > max_total_s:
            break
        if count > 0:
            parts.append(np.zeros(gap_samples, dtype=np.float32))
            cursor += gap_samples
            start = cursor
        parts.append(audio)
        cursor += len(audio)
        segments.append((start, cursor, rec["text"].strip()))
        count += 1
    if count < 2:
        return None
    return np.concatenate(parts), segments


def make_timestamped_target(
    segments: list[tuple[int, int, str]],
    tokenizer,
) -> list[int]:
    """Build a timestamped target token sequence for a packed window.

    prefix = tokenizer.sot_sequence (WITHOUT notimestamps), then per clip:
    <|start|> + encode(" " + text) + <|end|>, where start/end are the clip's
    audio boundaries (in seconds, quantized to 0.02s). Then eot.
    """
    toks = list(tokenizer.sot_sequence)
    for start_sample, end_sample, text in segments:
        start_s = start_sample / SAMPLE_RATE
        end_s = end_sample / SAMPLE_RATE
        start_tok = tokenizer.timestamp_begin + round(start_s / TIMESTAMP_QUANTUM_S)
        end_tok = tokenizer.timestamp_begin + round(end_s / TIMESTAMP_QUANTUM_S)
        toks.append(start_tok)
        toks.extend(tokenizer.encode(" " + text))
        toks.append(end_tok)
    toks.append(tokenizer.eot)
    return toks


def make_notimestamp_target(
    segments: list[tuple[int, int, str]],
    tokenizer,
) -> list[int]:
    """Notimestamps target: prefix includes notimestamps, texts joined by spaces."""
    toks = list(tokenizer.sot_sequence_including_notimestamps)
    joined = " ".join(text for _, _, text in segments)
    toks.extend(tokenizer.encode(" " + joined))
    toks.append(tokenizer.eot)
    return toks


def compute_mel_from_array(
    audio: np.ndarray,
    n_mels: int = 128,
    spec_augment: bool = False,
    rng: random.Random | None = None,
) -> mx.array:
    mel = pad_or_trim(log_mel_spectrogram(audio.astype(np.float32), n_mels=n_mels), N_FRAMES, axis=-2)
    if spec_augment and rng is not None:
        mel_np = np.array(mel, dtype=np.float32)
        for _ in range(2):
            f_width = rng.randint(2, 10)
            f0 = rng.randint(0, max(0, n_mels - f_width))
            mel_np[:, f0 : f0 + f_width] = 0.0
        return mx.array(mel_np, dtype=mx.float16)
    return mel.astype(mx.float16)


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
    # Round 3: long-form training. Off by default (round 3 regressed; see module docstring).
    ap.add_argument("--packed-windows", type=int, default=0,
                    help="Packed long-window examples per epoch (0 disables, default)")
    ap.add_argument("--timestamp-prob", type=float, default=0.0,
                    help="Probability a single or packed example uses timestamped targets (default 0)")
    ap.add_argument("--dev-base-ids", type=int, default=40,
                    help="Number of base_ids held out for dev best-checkpoint selection")
    ap.add_argument("--packed-distinct-categories", action="store_true", default=True,
                    help="When packing, never repeat a category or use two dictionary_carrier in one window")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="If >0, stop after this many optimizer steps (smoke run)")
    ap.add_argument("--print-sample-target", action="store_true",
                    help="Print one timestamped packed-window target token sequence and exit")
    args = ap.parse_args()

    train_records_all = json.loads(Path(args.train_manifest).read_text(encoding="utf-8"))
    eval_records = json.loads(Path(args.eval_manifest).read_text(encoding="utf-8"))

    # Assert eval ids never appear in train base_ids.
    eval_ids = {r["id"] for r in eval_records}
    train_base_ids = {r["base_id"] for r in train_records_all}
    overlap = eval_ids & train_base_ids
    assert not overlap, f"eval ids overlap train base_ids: {sorted(overlap)[:5]}"

    # Dev split for best-checkpoint selection.
    dev_rng = random.Random(1234)
    train_records, dev_records = split_dev_set(train_records_all, args.dev_base_ids, dev_rng)
    # Build clean-source dev eval records (one per base_id) from raw_16k so we
    # don't double-apply noise on already-conditioned train_wav files.
    dev_eval_records = build_dev_eval_records(dev_records)
    print(f"Dev split: {args.dev_base_ids} base_ids held out -> "
          f"{len(dev_eval_records)} dev eval records, {len(train_records)} train records")

    print(f"Loading base model {args.base_model}...")
    model = load_model(args.base_model, dtype=mx.float16)
    tokenizer = get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe"
    )

    print("Evaluating baseline (pre-finetune) on all 50 held-out eval clips...")
    base_clean_wer, base_noisy_wer = evaluate_all_held_out(model, eval_records)
    print(f"  Baseline 50-clip WER: clean={base_clean_wer:.3f}, office_app_nogate@14dB={base_noisy_wer:.3f}")

    # Optional: print one timestamped packed-window target and exit (sanity).
    if args.print_sample_target:
        sample_rng = random.Random(7)
        for _ in range(50):
            built = build_packed_window(
                train_records, sample_rng, distinct_categories=args.packed_distinct_categories
            )
            if built is not None:
                audio, segments = built
                toks = make_timestamped_target(segments, tokenizer)
                print("Sample timestamped packed-window target:")
                print(f"  segments: {[(round(s/SAMPLE_RATE,2), round(e/SAMPLE_RATE,2), t[:40]) for s,e,t in segments]}")
                print(f"  tokens ({len(toks)}): {toks}")
                # Decode a few tokens back for readability.
                decoded = [tokenizer.decode([t]) for t in toks[:12]]
                print(f"  decoded head: {decoded}")
                raise SystemExit(0)
        print("Could not build a packed window for sample target.")
        raise SystemExit(1)

    inject_lora(model, r=args.rank, alpha=args.alpha, top_encoder_layers=args.top_encoder_layers)
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(
        f"Injected LoRA (r={args.rank}, alpha={args.alpha}, top_enc={args.top_encoder_layers}): "
        f"{trainable_params:,} / {total_params:,} trainable params ({100 * trainable_params / total_params:.2f}%)"
    )

    # Round 3: prefixes differ per example (timestamped vs notimestamps), so the
    # loss mask must start at each example's own prefix length.
    prefix_notimestamps = list(tokenizer.sot_sequence_including_notimestamps)
    prefix_timestamps = list(tokenizer.sot_sequence)
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

    # Estimate total steps: single-clip batches + packed-window batches per epoch.
    single_batches = len(train_records) // args.batch_size
    packed_batches = (args.packed_windows + args.batch_size - 1) // args.batch_size if args.packed_windows > 0 else 0
    per_epoch = single_batches + packed_batches
    total_steps = per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
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

    def build_single_example(item: dict) -> tuple[mx.array, list[int], list[int], int]:
        use_spec_aug = args.spec_augment and (rng.random() < 0.4)
        mel = compute_mel(item["file"], n_mels=model.dims.n_mels, spec_augment=use_spec_aug, rng=rng)
        use_ts = rng.random() < args.timestamp_prob
        if use_ts:
            pre = prefix_timestamps
        else:
            pre = prefix_notimestamps
        toks = pre + tokenizer.encode(" " + item["text"].strip()) + [tokenizer.eot]
        return mel, toks[:-1], toks[1:], len(pre) - 1

    def build_packed_example() -> tuple[mx.array, list[int], list[int], int] | None:
        built = build_packed_window(
            train_records, rng, distinct_categories=args.packed_distinct_categories
        )
        if built is None:
            return None
        audio, segments = built
        use_spec_aug = args.spec_augment and (rng.random() < 0.4)
        mel = compute_mel_from_array(audio, n_mels=model.dims.n_mels, spec_augment=use_spec_aug, rng=rng)
        use_ts = rng.random() < args.timestamp_prob
        if use_ts:
            toks = make_timestamped_target(segments, tokenizer)
            pre_len = len(prefix_timestamps) - 1
        else:
            toks = make_notimestamp_target(segments, tokenizer)
            pre_len = len(prefix_notimestamps) - 1
        return mel, toks[:-1], toks[1:], pre_len

    def run_batch(examples: list[tuple[mx.array, list[int], list[int], int]]) -> None:
        nonlocal step
        mels = [e[0] for e in examples]
        inps = [e[1] for e in examples]
        tgts = [e[2] for e in examples]
        pre_lens = [e[3] for e in examples]
        max_len = max(len(x) for x in inps)
        bsz = len(examples)
        inp_np = np.full((bsz, max_len), tokenizer.eot, dtype=np.int32)
        tgt_np = np.full((bsz, max_len), tokenizer.eot, dtype=np.int32)
        mask_np = np.zeros((bsz, max_len), dtype=np.float32)
        for i in range(bsz):
            length = len(inps[i])
            inp_np[i, :length] = inps[i]
            tgt_np[i, :length] = tgts[i]
            mask_np[i, pre_lens[i]:length] = 1.0
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

    for epoch in range(1, args.epochs + 1):
        # --- Single-clip pass (every clip once, shuffled) ---
        shuffled = list(train_records)
        rng.shuffle(shuffled)
        for batch_start in range(0, len(shuffled) - args.batch_size + 1, args.batch_size):
            batch = shuffled[batch_start : batch_start + args.batch_size]
            examples = [build_single_example(item) for item in batch]
            run_batch(examples)
            if args.max_steps > 0 and step >= args.max_steps:
                break

        # --- Packed-window pass (regenerated each epoch) ---
        if args.packed_windows > 0:
            built = 0
            cur: list = []
            while built < args.packed_windows:
                ex = build_packed_example()
                if ex is None:
                    continue
                cur.append(ex)
                if len(cur) == args.batch_size:
                    run_batch(cur)
                    cur = []
                    built += args.batch_size
                    if args.max_steps > 0 and step >= args.max_steps:
                        break
            if cur and (args.max_steps == 0 or step < args.max_steps):
                run_batch(cur)
                built += len(cur)
        if args.max_steps > 0 and step >= args.max_steps:
            break

        # End-of-epoch eval: dev set (clean raw_16k sources) for selection, eval set for info.
        dev_clean_wer, dev_noisy_wer = evaluate_all_held_out(model, dev_eval_records)
        ft_clean_wer, ft_noisy_wer = evaluate_all_held_out(model, eval_records)
        combined = 0.4 * dev_clean_wer + 0.6 * dev_noisy_wer
        print(
            f"=== End of Epoch {epoch}: dev WER clean={dev_clean_wer:.3f}, "
            f"office_app_nogate@14dB={dev_noisy_wer:.3f} | "
            f"eval WER clean={ft_clean_wer:.3f} (was {base_clean_wer:.3f}), "
            f"office_app_nogate@14dB={ft_noisy_wer:.3f} (was {base_noisy_wer:.3f}) ==="
        )
        if combined < best_score:
            best_score = combined
            best_adapters = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
            mx.eval(best_adapters)
            print(f"  -> new best dev combined={combined:.4f}")

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
