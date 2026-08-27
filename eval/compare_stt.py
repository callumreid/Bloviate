"""Compare STT models / conditions on your own clips.

Ties the pieces together: for each clip + text in a manifest, synthesize the
runtime conditions (whisper_conditions), transcribe with several providers
(providers), and score WER (wer). Prints a table.

Manifest is a JSON list of {file, text, [voice]} -- reuse eval/manifest.yaml's
golden texts, or point at whisper-clone clips. Example:

    export DEEPGRAM_API_KEY=... OPENROUTER_API_KEY=...
    venv/bin/python eval/compare_stt.py clips.json \\
        --conditions clean office office_app office_app_nogate --snr 15

Findings this harness produced (2026-08-26, on a Deepgram "whisper" voice
clone -- an optimistic proxy, see validate_acoustics.py):
  * the stationary spectral gate (office_app) is neutral-to-worse than
    dropping it (office_app_nogate) at every SNR tested -- the AGC is fine,
    the gate is the cost;
  * microsoft/mai-transcribe-1.5 held up far better under office noise than
    the other models, including the current final-pass qwen3-asr-flash.
Both are leads to confirm on *real* whispered recordings, not verdicts.
"""

import argparse
import json
import statistics
import time

import whisper_conditions as wc
import wer as werlib
import providers

DEFAULT_MODELS = [
    ("deepgram/nova-3", lambda f: providers.deepgram(f, keyterms=True)),
    ("qwen3-asr-flash", lambda f: providers.openrouter(f, "qwen/qwen3-asr-flash-2026-02-10")),
    ("mai-transcribe-1.5", lambda f: providers.openrouter(f, "microsoft/mai-transcribe-1.5")),
    ("gpt-4o-transcribe", lambda f: providers.openrouter(f, "openai/gpt-4o-transcribe")),
    ("parakeet-v3", lambda f: providers.openrouter(f, "nvidia/parakeet-tdt-0.6b-v3")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="JSON list of {file, text}")
    ap.add_argument("--conditions", nargs="+",
                    default=["clean", "office", "office_app", "office_app_nogate"])
    ap.add_argument("--snr", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workdir", default="/tmp/bloviate_stt_eval")
    args = ap.parse_args()

    import os
    os.makedirs(args.workdir, exist_ok=True)
    clips = json.load(open(args.manifest))

    # results[model][condition] -> list of WER
    results = {m: {c: [] for c in args.conditions} for m, _ in DEFAULT_MODELS}
    for clip in clips:
        clean = wc.load_16k(clip["file"])
        for cond in args.conditions:
            out = os.path.join(args.workdir, f"{os.path.basename(clip['file'])}.{cond}.wav")
            arr = wc.make_condition(clean, cond, snr_db=args.snr, seed=args.seed)
            import soundfile as sf
            import numpy as np
            sf.write(out, np.clip(arr, -1, 1).astype("float32"), wc.TARGET_SR, subtype="PCM_16")
            for mname, fn in DEFAULT_MODELS:
                try:
                    results[mname][cond].append(werlib.wer(clip["text"], fn(out)))
                except Exception:
                    pass
                time.sleep(0.15)

    def mean(xs):
        return statistics.fmean(xs) if xs else float("nan")

    header = f"{'model':22s} | " + "  ".join(f"{c:>16s}" for c in args.conditions)
    print(header)
    print("-" * len(header))
    ranked = sorted(DEFAULT_MODELS, key=lambda m: mean(results[m[0]][args.conditions[0]]))
    for mname, _ in ranked:
        cells = "  ".join(f"{mean(results[mname][c]):>16.3f}" for c in args.conditions)
        print(f"{mname:22s} | {cells}")


if __name__ == "__main__":
    main()
