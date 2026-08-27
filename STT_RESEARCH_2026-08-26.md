# STT Research For Bloviate — Whisper Voice Clone Eval

Date: August 26, 2026
Supersedes the model picks in `STT_RESEARCH_2026-03-27.md` for the *whispered*
use case. Harness added in this change: `eval/validate_acoustics.py`,
`eval/whisper_conditions.py`, `eval/providers.py`, `eval/wer.py`,
`eval/compare_stt.py`.

## Why this exists

The open question has always been which STT stack is best for **whispering in a
noisy office** — the condition Bloviate is built for and the one no public
leaderboard measures. Recording enough real whispers to A/B models is tedious,
so we made a Deepgram voice clone ("Callum Whispering") to generate whisper-ish
speech on demand. This is the eval built on that clone.

## Bottom line

1. **The clone is not true whisper — treat every number here as optimistic.**
   Praat analysis (`eval/validate_acoustics.py`) shows the clone's "whisper"
   is breathy, *high-pitched phonation*: mean F0 ≈ **410 Hz** with **36–46 %**
   of speech frames voiced. The normal-voice control (Harper) is F0 ≈ 191 Hz /
   74 % voiced. True whisper has **no F0 and near-zero voicing**. So the clone
   keeps a (very high) pitch and partial voicing that real whisper lacks; STT
   will do *better* on it than on real office whispering. Good for fast
   iteration on soft/breathy speech; **real recorded whispers remain the
   ground truth**, especially for final model selection.

2. **The `noisereduce` stationary spectral gate does not earn its place.**
   Reproducing the runtime path (`office_app` = gate + AGC vs
   `office_app_nogate` = AGC only), the gate is **neutral-to-worse at every
   SNR tested** and never a clear win — averaged across models it adds ~0.6 pp
   WER and is worse-or-equal in ~64 % of cells. The AGC-only path tracks raw
   noisy audio, so the AGC is fine; the **gate is the cost**. Deepgram and the
   OpenRouter models already have their own noise handling. On the clone the
   penalty is small (~1 pp); on true whisper — whose energy sits closer to the
   noise floor the gate subtracts — it is likely larger.
   → Recommend `noise_suppression.stationary_noise_reduction: 0.0` (or
   `enabled: false`), verified against real whispers before finalizing.

3. **Under office noise, `microsoft/mai-transcribe-1.5` is the standout.**
   The final pass runs through OpenRouter, so this is a one-line model swap.

## Model ranking

Mean WER over the two clone clips (fox / greeting), office babble.

Clean audio (upper bound — undiscriminating, everything is near-perfect):

| model                | WER   |
|----------------------|-------|
| qwen3-asr-flash      | 0.000 |
| parakeet-v3          | 0.000 |
| deepgram/nova-3      | 0.014 |
| gpt-4o-transcribe    | 0.041 |
| mai-transcribe-1.5   | 0.041 |

Office noise, SNR 15 dB, with the runtime `office_app` processing:

| model                | WER   |
|----------------------|-------|
| **mai-transcribe-1.5** | **0.108** |
| qwen3-asr-flash      | 0.294 |
| parakeet-v3          | 0.308 |
| deepgram/nova-3      | 0.349 |
| gpt-4o-transcribe    | 0.517 |

The current final-pass model, `qwen3-asr-flash`, **wins on clean audio but
falls apart under noise**; `mai-transcribe-1.5` is ~3× lower WER in noise. A
harder pooled run (SNR 15 + 12 dB, 2 seeds) reproduced the ordering:
mai 0.144, parakeet 0.423, deepgram 0.468, gpt-4o 0.538, qwen 0.551,
gpt-transcribe 0.565, whisper-large-v3-turbo 0.702. `x-ai/grok-stt-1.0`
returned empty and is unusable.

## Recommended changes (to A/B, not to assume)

1. `openai.model: microsoft/mai-transcribe-1.5` (from `qwen3-asr-flash-…`).
2. `noise_suppression.stationary_noise_reduction: 0.0`.

Both are one-line, instantly reversible, and independent — test them
separately. Confirm on **real** whispered recordings via `eval/run_eval.py`
before making either the permanent default.

## Caveats

- Two distinct clone texts only (see the extraction note below), so the
  ranking is a strong lead, not a final verdict; the mai-transcribe margin is
  large and consistent across SNRs and seeds, which is why it is worth acting
  on.
- Synthetic speech-shaped office babble, not a real room recording.
- Single-pass transcription; the live hybrid (Deepgram interim → OpenRouter
  final) is not modelled here.
- Clone caveat above — the single most important one.

### Note on generating more clone clips

The clone lives at studio.gradium.ai. Pulling audio out of an automated
browser session proved impossible (downloads and every byte-export path are
blocked), so the corpus here is small. To expand it, generate + download the
clips through a normal browser session, or — better for a trustworthy result —
record a handful of real whispers with `bloviate --record-eval-clip`.
