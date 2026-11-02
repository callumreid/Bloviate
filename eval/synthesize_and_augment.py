#!/usr/bin/env python3
"""Synthesize Gradium whispered TTS clips and generate multi-condition training variants.

1. Synthesizes each item in eval/finetune_data/corpus.json using Gradium's TTS HTTP endpoint
   with Callum's Pro Voice Clone (voice_id="5YKZTb0mJeGayTbE", temp=0.0, cfg_coef=2.5),
   resampled to 16 kHz mono PCM_16 wav in eval/finetune_data/raw_16k/.
2. For the 450 train clips, creates multi-condition training variants in eval/finetune_data/train_wav/:
   - clean / quiet
   - office_app_nogate (SNR 10..18 dB + Bloviate prerecorded AGC)
   - office_app (SNR 10..18 dB + noisereduce gate + prerecorded AGC)
   - unvoiced_lpc (true 0-F0 LPC whisper vocoder, clean + noisy office_app_nogate)
3. Writes eval/finetune_data/train_manifest.json and eval/finetune_data/eval_manifest.json.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from scipy import signal as sig
import soundfile as sf

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import whisper_conditions as wc  # noqa: E402

DATA_DIR = EVAL_DIR / "finetune_data"
RAW_16K_DIR = DATA_DIR / "raw_16k"
TRAIN_WAV_DIR = DATA_DIR / "train_wav"
EVAL_UNVOICED_DIR = DATA_DIR / "eval_unvoiced_16k"

VOICE_ID = "5YKZTb0mJeGayTbE"
GRADIUM_TTS_URL = "https://api.gradium.ai/api/post/speech/tts"


def lpc_whisper_vocode(
    x: np.ndarray,
    sr: int = 16000,
    order: int = 20,
    frame_ms: int = 25,
    hop_ms: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Replace periodic glottal excitation with turbulent broadband noise while preserving LPC formants."""
    rng = np.random.default_rng(seed)
    x_pre = np.append(x[0], x[1:] - 0.97 * x[:-1])
    flen = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    win = np.hanning(flen)
    out = np.zeros(len(x) + flen)
    norm = np.zeros(len(x) + flen)

    for start in range(0, max(1, len(x_pre) - flen), hop):
        frame = x_pre[start : start + flen] * win
        energy = float(np.sum(frame ** 2))
        if energy < 1e-10:
            continue
        r = np.correlate(frame, frame, mode="full")[flen - 1 : flen + order]
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = float(r[0])
        for k in range(1, order + 1):
            lam = -float(np.dot(a[:k], r[k:0:-1])) / max(e, 1e-12)
            a_new = a.copy()
            for i in range(1, k):
                a_new[i] = a[i] + lam * a[k - i]
            a_new[k] = lam
            a = a_new
            e *= 1.0 - lam ** 2
            if e <= 1e-12:
                break
        gain = np.sqrt(max(e, 1e-12))
        bw_expand = 0.985 ** np.arange(order + 1)
        a_whisp = a * bw_expand
        exc = rng.standard_normal(flen) * gain
        synth = sig.lfilter([1.0], a_whisp, exc) * win
        out[start : start + flen] += synth
        norm[start : start + flen] += win ** 2

    out = out[: len(x)] / np.maximum(norm[: len(x)], 1e-6)
    out = sig.lfilter([1.0], [1.0, -0.85], out)
    orig_rms = float(np.sqrt(np.mean(x ** 2)))
    return wc._scale_rms(out, max(orig_rms, 0.012))


def synthesize_one(item: dict, api_key: str) -> tuple[str, bool, str]:
    clip_id = item["id"]
    out_path = RAW_16K_DIR / f"{clip_id}.wav"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return clip_id, True, "cached"

    payload = {
        "text": item["tts_text"],
        "voice_id": VOICE_ID,
        "output_format": "wav",
        "only_audio": True,
        "json_config": json.dumps({"temp": 0.0, "cfg_coef": 2.5}),
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
    }

    for attempt in range(4):
        try:
            req = urllib.request.Request(GRADIUM_TTS_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=45) as resp:
                wav_bytes = resp.read()
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float64")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != wc.TARGET_SR:
                g = np.gcd(sr, wc.TARGET_SR)
                audio = sig.resample_poly(audio, wc.TARGET_SR // g, sr // g)
            sf.write(str(out_path), np.clip(audio, -1, 1).astype("float32"), wc.TARGET_SR, subtype="PCM_16")
            return clip_id, True, "ok"
        except Exception as exc:
            if attempt == 3:
                return clip_id, False, str(exc)
            time.sleep(1.5 * (attempt + 1))
    return clip_id, False, "exhausted"


def build_augmented_splits(corpus: list[dict]) -> None:
    TRAIN_WAV_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_UNVOICED_DIR.mkdir(parents=True, exist_ok=True)

    train_records: list[dict] = []
    eval_records: list[dict] = []
    eval_unvoiced_records: list[dict] = []

    snr_choices = [10.0, 12.0, 14.0, 16.0, 18.0]

    for idx, item in enumerate(corpus):
        clip_id = item["id"]
        raw_path = RAW_16K_DIR / f"{clip_id}.wav"
        if not raw_path.exists():
            continue

        clean = wc.load_16k(str(raw_path))
        target_text = item["target_text"]

        if item["split"] == "eval":
            eval_records.append({
                "id": clip_id,
                "category": item["category"],
                "file": str(raw_path),
                "text": target_text,
                "tts_text": item["tts_text"],
            })
            unvoiced = lpc_whisper_vocode(clean, wc.TARGET_SR, seed=idx + 1000)
            uv_path = EVAL_UNVOICED_DIR / f"{clip_id}_unvoiced.wav"
            sf.write(str(uv_path), np.clip(unvoiced, -1, 1).astype("float32"), wc.TARGET_SR, subtype="PCM_16")
            eval_unvoiced_records.append({
                "id": f"{clip_id}_unvoiced",
                "category": item["category"],
                "file": str(uv_path),
                "text": target_text,
                "tts_text": item["tts_text"],
            })
            continue

        # For each training utterance, create 2 complementary acoustic variants so the model
        # learns both clean/quiet vocabulary and noisy runtime resilience + unvoiced LPC acoustics:
        # Variant A: cycling across clean, office_app_nogate, office_app at varying SNRs
        snr_a = snr_choices[idx % len(snr_choices)]
        cond_a = ["office_app_nogate", "office_app", "quiet"][idx % 3]
        arr_a = wc.make_condition(clean, cond_a, snr_db=snr_a, seed=idx + 1)
        path_a = TRAIN_WAV_DIR / f"{clip_id}_{cond_a}.wav"
        sf.write(str(path_a), np.clip(arr_a, -1, 1).astype("float32"), wc.TARGET_SR, subtype="PCM_16")
        train_records.append({
            "id": f"{clip_id}_{cond_a}",
            "base_id": clip_id,
            "condition": cond_a,
            "snr_db": snr_a if "office" in cond_a else None,
            "category": item["category"],
            "file": str(path_a),
            "text": target_text,
        })

        # Variant B: for half the clips, unvoiced LPC whisper (with or without office noise);
        # for the other half, a harder SNR (10-14 dB) office_app / office_app_nogate variant.
        if idx % 2 == 0:
            unvoiced = lpc_whisper_vocode(clean, wc.TARGET_SR, seed=idx + 500)
            cond_b = "unvoiced_office_nogate" if (idx % 4 == 0) else "unvoiced_quiet"
            snr_b = 14.0 if (idx % 8 == 0) else 16.0
            arr_b = (
                wc.make_condition(unvoiced, "office_app_nogate", snr_db=snr_b, seed=idx + 200)
                if cond_b == "unvoiced_office_nogate"
                else wc._scale_rms(unvoiced, wc.WHISPER_RMS)
            )
        else:
            cond_b = "office_app_nogate" if (idx % 3 != 0) else "office_app"
            snr_b = 12.0 if (idx % 5 == 0) else 14.0
            arr_b = wc.make_condition(clean, cond_b, snr_db=snr_b, seed=idx + 300)

        path_b = TRAIN_WAV_DIR / f"{clip_id}_{cond_b}.wav"
        sf.write(str(path_b), np.clip(arr_b, -1, 1).astype("float32"), wc.TARGET_SR, subtype="PCM_16")
        train_records.append({
            "id": f"{clip_id}_{cond_b}",
            "base_id": clip_id,
            "condition": cond_b,
            "snr_db": snr_b,
            "category": item["category"],
            "file": str(path_b),
            "text": target_text,
        })

    (DATA_DIR / "train_manifest.json").write_text(json.dumps(train_records, indent=2), encoding="utf-8")
    (DATA_DIR / "eval_manifest.json").write_text(json.dumps(eval_records, indent=2), encoding="utf-8")
    (DATA_DIR / "eval_unvoiced_manifest.json").write_text(
        json.dumps(eval_unvoiced_records, indent=2), encoding="utf-8"
    )
    print(
        f"Built manifests: train={len(train_records)} clips, "
        f"eval={len(eval_records)} clips, eval_unvoiced={len(eval_unvoiced_records)} clips"
    )


def main() -> None:
    api_key = os.environ.get("GRADIUM_API_KEY")
    if not api_key:
        raise SystemExit("GRADIUM_API_KEY is required.")

    RAW_16K_DIR.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))

    t0 = time.time()
    ok_count = 0
    fail_count = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(synthesize_one, item, api_key): item["id"] for item in corpus}
        for i, fut in enumerate(as_completed(futures), 1):
            clip_id, ok, status = fut.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                print(f"  FAIL {clip_id}: {status}")
            if i % 50 == 0 or i == len(corpus):
                elapsed = time.time() - t0
                print(f"  Synthesized {i}/{len(corpus)} (ok={ok_count}, fail={fail_count}) in {elapsed:.1f}s")

    print("Generating multi-condition training + evaluation variants...")
    build_augmented_splits(corpus)


if __name__ == "__main__":
    main()
