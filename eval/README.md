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
