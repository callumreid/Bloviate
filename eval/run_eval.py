#!/usr/bin/env python3
"""
Transcription eval harness for Bloviate.

Runs recorded clips through the configured providers and reports word error
rate + latency per clip, so accuracy changes are measured instead of vibed.

Usage:
  venv/bin/python eval/run_eval.py                     # all clips, local whisper
  venv/bin/python eval/run_eval.py --providers whisper deepgram openai
  venv/bin/python eval/run_eval.py --manifest eval/manifest.yaml --tag builtin-mic

Record clips with:  venv/bin/bloviate --record-eval-clip <name>
"""

import argparse
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TARGET_RATE = 16000


def normalize_words(text: str) -> list[str]:
    text = re.sub(r"[^\w\s'-]", " ", str(text or "").lower())
    return [word for word in text.split() if word]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Word-level Levenshtein distance
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1] / len(ref)


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {width} in {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != TARGET_RATE:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(TARGET_RATE, rate)
        audio = resample_poly(audio, TARGET_RATE // g, rate // g).astype(np.float32)
    return audio


def main() -> int:
    parser = argparse.ArgumentParser(description="Bloviate transcription eval")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "eval" / "manifest.yaml"))
    parser.add_argument(
        "--providers", nargs="+", default=["whisper"],
        choices=["whisper", "deepgram", "openai"],
    )
    parser.add_argument("--tag", default=None, help="Only run clips carrying this tag")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"No manifest at {manifest_path}.")
        print("Record clips with `bloviate --record-eval-clip <name>`, then create the")
        print("manifest from eval/manifest.example.yaml.")
        return 1

    entries = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or []
    if args.tag:
        entries = [e for e in entries if args.tag in (e.get("tags") or [])]
    if not entries:
        print("Manifest has no matching clips.")
        return 1

    from settings_service import load_yaml_config
    from transcriber import Transcriber

    config, _ = load_yaml_config("config.yaml")
    transcriber = Transcriber(config)

    print(f"{len(entries)} clip(s), providers: {', '.join(args.providers)}\n")
    totals: dict[str, list[tuple[float, float]]] = {p: [] for p in args.providers}

    for entry in entries:
        clip_path = Path(entry["file"])
        if not clip_path.is_absolute():
            clip_path = manifest_path.parent / clip_path
        golden = str(entry.get("golden", "")).strip()
        if not clip_path.is_file() or not golden:
            print(f"skip {clip_path.name}: missing file or golden text")
            continue
        audio = load_wav(clip_path)
        print(f"— {clip_path.name} ({len(audio)/TARGET_RATE:.1f}s)")
        for provider in args.providers:
            start = time.monotonic()
            text = transcriber.transcribe_with_provider(provider, audio) or ""
            latency = time.monotonic() - start
            wer = word_error_rate(golden, text)
            totals[provider].append((wer, latency))
            print(f"    {provider:<9} WER {wer:6.1%}  {latency:5.2f}s  {text[:70]!r}")

    print("\n=== Summary ===")
    for provider, results in totals.items():
        if not results:
            print(f"{provider:<9} no results")
            continue
        avg_wer = sum(r[0] for r in results) / len(results)
        avg_latency = sum(r[1] for r in results) / len(results)
        print(f"{provider:<9} clips={len(results)}  avg WER {avg_wer:6.1%}  avg latency {avg_latency:5.2f}s")

    transcriber.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
