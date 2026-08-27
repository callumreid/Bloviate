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
so we made a Deepgram voice clone ("Callum Whispering") from Callum actually
whispering into his desk mic in his office — as close to the real use case as a
synthetic corpus gets. This is the eval built on that clone.

## Bottom line

1. **The clone carries residual pitch — treat absolute numbers as optimistic.**
   Praat (`eval/validate_acoustics.py`) measures the clone's "whisper" at mean
   F0 ≈ **410 Hz**, **36–46 %** of speech frames voiced; the normal-voice
   control (Harper) is F0 ≈ 191 Hz / 74 % voiced. A textbook whisper has no F0
   at all, so the clone is a *soft, breathy, high-pitched voice* rather than a
   silent whisper. Since it was generated from Callum's own desk-whispering, it
   is a faithful proxy for **his** whisper style — but STT still scores somewhat
   better on it than on true unvoiced whisper, so read the numbers as a ceiling.

2. **`microsoft/mai-transcribe-1.5` is the clear best model under noise** — and
   it is a one-line swap on the OpenRouter path Bloviate already uses.
   **Applied 2026-08-26** (live config + repo).

3. **The `noisereduce` spectral gate is roughly a wash** — model-dependent and
   small, not the clear liability an earlier shallower run suggested. It does
   modestly hurt the newly-chosen mai-transcribe at every SNR, so turning it off
   is a small free win *paired with mai* — but low-stakes. Not applied.

## Model ranking

Deeper run: 9 models × 2 clips (fox / greeting) × SNR {18, 14, 10} dB × 3 noise
seeds, gate on **and** off. "noisy mean" pools all noisy conditions.

| model                  | clean | quiet | 18 dB | 14 dB | 10 dB | noisy mean |
|------------------------|-------|-------|-------|-------|-------|------------|
| **mai-transcribe-1.5** | 0.041 | 0.041 | **0.016** | **0.109** | **0.489** | **0.205** |
| voxtral-small-24b      | 0.041 | 0.041 | 0.045 | 0.320 | 0.657 | 0.340 |
| parakeet-v3            | 0.000 | 0.041 | 0.094 | 0.370 | 0.734 | 0.399 |
| deepgram/nova-3        | 0.014 | 0.014 | 0.105 | 0.420 | 0.778 | 0.434 |
| qwen3-asr-flash (old)  | 0.000 | 0.000 | 0.196 | 0.488 | 0.910 | 0.531 |
| gpt-4o-transcribe      | 0.014 | 0.041 | 0.390 | 0.556 | 0.734 | 0.560 |
| gpt-transcribe         | 0.041 | 0.041 | 0.224 | 0.671 | 0.983 | 0.626 |
| whisper-large-v3-turbo | 0.041 | 0.081 | 0.514 | 0.820 | 0.936 | 0.757 |
| fish-transcribe-1      | 0.054 | 0.054 | 0.164 | 0.409 | >1.0  | hallucinates |

`mai-transcribe-1.5` wins at every SNR by a wide margin; `voxtral-small-24b` is
a credible runner-up. `fish-transcribe-1` hallucinates badly at 10 dB and
`x-ai/grok-stt-1.0` returned empty — both unusable.

### The clean-vs-noisy tradeoff (why the swap is right)

| condition   | mai-transcribe-1.5 | qwen3-asr-flash |
|-------------|--------------------|-----------------|
| clean       | 0.041 | **0.000** |
| quiet       | 0.041 | **0.000** |
| noise 18 dB | **0.016** | 0.196 |
| noise 14 dB | **0.109** | 0.488 |
| noise 10 dB | **0.489** | 0.910 |

The old default, `qwen3-asr-flash`, is *perfect* on clean/quiet audio but the
worst-but-two under any real noise. It is 4 pp better than mai on clean (a
single made-up word, "Gradium"); mai is **18–42 pp** better the moment there is
office noise. Whispering happens in an office, so the swap is clearly right.

## The noise suppressor, honestly

Splitting the runtime path into gate+AGC (`office_app`) vs AGC-only
(`office_app_nogate`) isolates the `noisereduce` stationary spectral gate.
Averaged across the 9 models (capping hallucinated WER at 1.0 so fish-transcribe
doesn't skew it) the gate is **near zero**: it helps gpt-4o / whisper-turbo /
parakeet a little and hurts mai / qwen / voxtral a little.

For **mai-transcribe-1.5 specifically**, gate-off is consistently a touch
better: 0.014 vs 0.018 @18 dB, 0.100 vs 0.117 @14 dB, 0.471 vs 0.508 @10 dB.
So `stationary_noise_reduction: 0.0` is a small (~1–4 pp) win paired with mai —
worth trying, but not urgent, and it also affects normal-voice dictation which
this eval did not cover. **Not applied.**

(An earlier, shallower Deepgram-only run reported the gate as "neutral-to-worse
everywhere" and recommended disabling it outright; the deeper run above shows
that was over-stated — hence this correction.)

## Changes

- **Applied:** `openai.model: microsoft/mai-transcribe-1.5` (from
  `qwen/qwen3-asr-flash-2026-02-10`) — live config + repo `config.yaml`.
  Verified end-to-end through the real `Transcriber._transcribe_openai` path on
  a whisper clip. Instantly reversible (live backup:
  `config.yaml.bak-20260826-mai`).
- **Optional, not applied:** `noise_suppression.stationary_noise_reduction: 0.0`
  — small win with mai; see above.

## Caveats

- **The clone is Callum's desk-whisper, but still voiced** (F0 ≈ 410 Hz), so
  numbers are a ceiling vs true unvoiced whisper.
- **Two texts only.** Every path for pulling more clip audio out of the
  automation browser session was blocked, so the corpus is small. The
  mai-transcribe margin is large and consistent across every SNR and seed, which
  is why it was safe to apply — widen it with more clips when convenient.
- **Synthetic speech-shaped office babble**, not a real room recording.
- **Single-pass** transcription; the live hybrid (Deepgram interim → OpenRouter
  final) is not modelled here.

### Generating more clips

The clone lives at studio.gradium.ai. To expand the corpus, generate + download
clips through a normal browser session, or — better — record real whispers with
`bloviate --record-eval-clip` and point `eval/compare_stt.py` at them.
