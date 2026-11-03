#!/usr/bin/env python3
"""Compare local MLX Whisper checkpoints on per-clip and long-form held-out audio.

Long-form passages are built by concatenating the held-out eval clips that are
NOT `dictionary_carrier` (those share one sentence template, and a passage of
repeated templates pushes decoding into high-temperature fallback, which makes
the numbers noisy and meaningless). Each passage is ~40-55 s with 0.35-0.8 s
gaps, so it crosses Whisper's 30 s window the way real long dictations do.

Decoding mirrors Bloviate's `_transcribe_whisper` (RMS normalisation, same
temperature fallback / thresholds, no prompt for local checkpoints).

  venv/bin/python eval/eval_longform.py r2=<model dir> r3=<model dir> [stock=mlx-community/whisper-large-v3-turbo]
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import sys

import mlx_whisper
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
from wer import _norm, wer  # noqa: E402
import whisper_conditions as wc  # noqa: E402

DATA = EVAL_DIR / "finetune_data"
KW = dict(language="en", temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.2,
          condition_on_previous_text=False, no_speech_threshold=0.5, verbose=None)


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    a = a * min(0.05 / (float(np.sqrt(np.mean(a ** 2))) + 1e-9), 10 ** (45 / 20))
    peak = float(np.abs(a).max())
    return a * (0.95 / peak) if peak > 0.95 else a


def _passages(clips: list[tuple[np.ndarray, str]], seed: int = 0) -> list[tuple[np.ndarray, str]]:
    rng = np.random.default_rng(seed)
    groups, cur, dur = [], [], 0.0
    for a, t in clips:
        cur.append((a, t))
        dur += len(a) / 16000
        if dur > 40:
            groups.append(cur)
            cur, dur = [], 0.0
    if cur and dur > 20:
        groups.append(cur)
    out = []
    for g in groups:
        parts = []
        for a, _ in g:
            parts += [a, np.zeros(int(16000 * rng.uniform(0.35, 0.8)), np.float32)]
        out.append((np.concatenate(parts), " ".join(t for _, t in g)))
    return out


def _noisy(items, snr_db=14.0):
    return [(wc.make_condition(a.astype(np.float64), "office_app_nogate", snr_db=snr_db, seed=i + 1).astype(np.float32), t)
            for i, (a, t) in enumerate(items)]


def _score(model: str, items, cats=None):
    temps: collections.Counter = collections.Counter()
    tot = [0.0, 0]
    by_cat: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
    for i, (a, ref) in enumerate(items):
        r = mlx_whisper.transcribe(_normalize(a), path_or_hf_repo=model, **KW)
        for s in r["segments"]:
            temps[s.get("temperature")] += 1
        n = len(_norm(ref))
        e = wer(ref, r["text"]) * n
        tot[0] += e
        tot[1] += n
        if cats:
            by_cat[cats[i]][0] += e
            by_cat[cats[i]][1] += n
    return (round(tot[0] / tot[1], 4), {k: round(v[0] / v[1], 3) for k, v in sorted(by_cat.items())},
            {str(k): v for k, v in temps.items()})


def main() -> None:
    models = dict(arg.split("=", 1) for arg in sys.argv[1:])
    if not models:
        sys.exit(__doc__)
    results = {}
    for cond, manifest in (("clean", "eval_manifest.json"), ("unvoiced", "eval_unvoiced_manifest.json")):
        man = json.loads((DATA / manifest).read_text())
        clips = [(wc.load_16k(m["file"]).astype(np.float32), m["text"]) for m in man]
        cats = [m["category"] for m in man]
        long_clean = _passages([c for c, m in zip(clips, man) if m["category"] != "dictionary_carrier"])
        sets = {
            "clip": (clips, cats),
            "clip_14dB": (_noisy(clips), cats),
            "long": (long_clean, None),
            "long_14dB": (_noisy(long_clean), None),
        }
        print(f"{cond}: {len(clips)} clips, long passages {[round(len(a) / 16000, 1) for a, _ in long_clean]}s", flush=True)
        for name, model in models.items():
            for sname, (items, c) in sets.items():
                w, per_cat, temps = _score(model, items, c)
                results[f"{cond}/{sname}/{name}"] = {"wer": w, "by_category": per_cat, "temperatures": temps}
                print(f"  {sname:10s} {name:8s} WER={w:.4f} {per_cat or ''} temps={temps}", flush=True)
    out = EVAL_DIR / "finetune_data" / "longform_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
