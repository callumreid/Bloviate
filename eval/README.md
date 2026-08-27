# Transcription Eval

Measures word error rate and latency on your own recorded clips so accuracy
changes are verified, not vibed. Clips and the manifest are personal audio and
stay out of git.

## Record clips

```bash
venv/bin/bloviate --record-eval-clip builtin-quiet-normal
venv/bin/bloviate --record-eval-clip builtin-office-whisper --record-seconds 10
```

Cover the grid that matters: each mic you actually use x {quiet, office noise}
x {whisper, normal voice}. Ten to twenty clips is plenty to start.

## Build the manifest

Copy `manifest.example.yaml` to `manifest.yaml` and fill in `golden` with the
exact words you spoke for each clip.

## Run

```bash
venv/bin/python eval/run_eval.py                          # local whisper only
venv/bin/python eval/run_eval.py --providers whisper deepgram openai
venv/bin/python eval/run_eval.py --tag builtin-mic
```

Cloud providers need `DEEPGRAM_API_KEY` / `OPENAI_API_KEY` in the environment.
Run it before and after any change to gates, gain, models, or prompts.

## Whisper-clone / model comparison (added 2026-08-26)

Beyond WER on your own recordings, the eval can compare STT models across the
noisy conditions Bloviate actually runs in, and can vet a "whisper" voice clone
before you trust it.

- `validate_acoustics.py <clip.wav>` — is a clip true whisper (no pitch) or
  breathy phonation? Run it on any voice-clone output first. Needs
  `praat-parselmouth`.
- `whisper_conditions.py` — degrade a clean clip into runtime conditions
  (quiet, office noise, with/without Bloviate's spectral gate + AGC).
- `providers.py` — Deepgram / OpenRouter / local-MLX adapters (keys from env).
- `wer.py` — dictation-aware WER (spoken syntax folded to written form).
- `compare_stt.py clips.json --conditions clean office_app office_app_nogate` —
  the driver; prints a model × condition WER table.

See `STT_RESEARCH_2026-08-26.md` for findings and the two config changes to A/B.
