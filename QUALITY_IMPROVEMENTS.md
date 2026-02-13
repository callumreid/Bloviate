# Bloviate Quality Improvements (Research + Roadmap)

Date: 2026-02-06

## Goal

Bloviate is a dictation tool optimized for whispering in loud, multi-speaker environments. The core win is *speaker verification*: only audio that matches your enrolled whisper voice is allowed through to transcription.

The gap you called out is that in a perfectly quiet scenario, commercial dictation tools (for example WhisperFlow and Monologue) can feel higher quality. That gap is usually not about background rejection; it is about:

1. Stronger base ASR models (and/or higher compute).
2. Better post-processing (punctuation, cleanup, formatting, "rewrite for intent").
3. Better context injection (personal dictionary, snippets, app-aware modes, screen/selection context).
4. A frontend audio pipeline tuned for near-field low-volume speech without introducing artifacts.

This document:

1. Summarizes what is implemented in this repo today.
2. Summarizes what the paid tools advertise that is "better" (and why it matters).
3. Proposes concrete, code-level improvements and an evaluation loop so you can iterate.

## What You Have Today (Repo Audit)

### Core Pipeline

High-level flow (implemented primarily in `src/main.py`):

1. Audio capture (always on), buffers only while PTT is held.
2. Noise suppression (`src/noise_suppressor.py`):
   - High-pass filter (80 Hz cutoff)
   - Stationary noise reduction via `noisereduce.reduce_noise(...)`
3. Speaker verification (`src/voice_fingerprint.py`):
   - SpeechBrain ECAPA-TDNN (VoxCeleb) embeddings
   - Enrollment: store embeddings, reference is mean embedding
   - Verify: cosine similarity >= threshold (default `0.65`)
4. Transcription (`src/transcriber.py`):
   - Provider `deepgram` (default in `config.yaml`) with optional streaming
   - Offline fallback to local `openai-whisper` (models like `medium.en`)
   - Custom dictionary post-corrections from `custom_dictionary.yaml`
5. Output:
   - Clipboard + auto-paste (macOS via AppleScript), plus optional stdout
6. UI (`src/ui.py`):
   - Small Qt window + macOS menubar indicator + optional overlay

### Files That Matter Most

- `src/main.py`: orchestrates recording, streaming vs offline, voice verify, output
- `src/transcriber.py`: Deepgram live (websocket) + prerecorded, local Whisper fallback, custom dictionary
- `src/noise_suppressor.py`: high-pass + noisereduce (and a WebRTC VAD helper that currently is not used for segmentation)
- `src/voice_fingerprint.py`: SpeechBrain speaker embeddings and verification
- `config.yaml`: current "product defaults" (Deepgram streaming enabled, flux model)
- `custom_dictionary.yaml`: deterministic text substitutions after transcription

### Notable Implementation Gaps / Tech Debt (Quality-Relevant)

These are concrete observations from the current implementation that can affect accuracy or make future improvements harder:

- `noise_suppression.spectral_gate_threshold` is read from `config.yaml`, but not used anywhere in `src/noise_suppressor.py`.
- WebRTC VAD exists (`NoiseSuppressor.is_speech(...)`), and there is an `update_noise_profile(...)` method, but neither is used in the main pipeline today.
- Noise suppression is applied unconditionally (when enabled) even for clean audio, which risks introducing artifacts in quiet-room dictation.
- Speaker verification runs on *noise-suppressed* audio (`process_recording()` calls `noise_suppressor.process()` before `verify_speaker()`), which may reduce discriminative speaker cues in some cases.
- There is no gain normalization / compression stage. For whispering, this often matters more than denoising because consonants can be extremely low energy.

### What It's Already Great At

- Noisy-office "cross talk" problem: speaker verification is the big differentiator.
- Dictation ergonomics: PTT, auto-paste, quick feedback overlay.
- Low-latency mode: Deepgram live streaming with prebuffer to avoid clipping the first syllable.

## What The Paid Tools Advertise (And Why It Helps In Quiet Scenarios)

This is based on the public marketing pages for:

1. WhisperFlow: https://whisperflow.app/
2. Monologue: https://www.monologue.to/ (also seen referenced as monologue.so)

### Feature Gap Matrix (High Level)

This is intentionally focused on "quiet-room quality" contributors (accuracy + polished output), not on Bloviate's unique advantage (speaker verification).

| Capability | Bloviate (today) | WhisperFlow (advertised) | Monologue (advertised) |
| --- | --- | --- | --- |
| Speaker verification / voice fingerprinting | Yes (SpeechBrain ECAPA) | Not advertised | Not advertised |
| Base ASR | Deepgram (streaming) or local Whisper fallback | Not disclosed | "Built on open models" (not disclosed) |
| Auto-editing / cleanup | Minimal (custom dictionary substitutions only) | Yes (auto-edits + filler removal) | Yes ("Auto Editing") |
| Personal dictionary | Manual YAML + deterministic substitutions | Yes (learns words automatically) | Yes ("Auto dictionary") |
| Snippets / text shortcuts | No | Yes ("Snippet Library") | Not highlighted on homepage |
| App-aware modes | Only "command mode" for window management | Yes (pricing mentions "different tones for each app") | Yes ("Modes" per app) |
| Context-aware formatting | No | "Polished writing" implied | Explicit ("Context-aware text formatting"; deep context screenshots) |
| Offline mode | Local Whisper fallback exists | Not clear from homepage | Explicit ("Offline transcription support") |

### Common "Better Than Raw ASR" Features

Both products emphasize features that are not "just transcription":

- AI cleanup / auto-editing
  - Removing filler words, fixing casing, adding punctuation
  - This makes the output feel higher quality even when the underlying ASR accuracy is similar
- Personal dictionary / custom vocabulary
  - Not only post-substitution, but biasing decoding toward your vocabulary
- Snippets / templates
  - Often framed as productivity, but it also reduces perceived errors because you don't rely on ASR for repetitive phrases
- Modes
  - Email vs Slack vs notes vs coding: different formatting and correction rules

### Monologue-Specific Differentiator: Context

Monologue explicitly markets context-aware dictation (screenshots and "modes") so it can:

- Use what's on screen / in the current app to disambiguate what you meant.
- Produce formatting that matches the target (for example headings, bullets, code blocks).

This is hard to beat with pure ASR because the model has no idea what you are doing. In quiet scenarios, that "context layer" can be the difference between "accurate transcript" and "high quality writing".

### WhisperFlow Positioning

WhisperFlow's page emphasizes:

- Fast dictation "in every app"
- Auto-edits and filler word removal
- Personal dictionary + snippets
- Multi-device (desktop + iOS)

Even if you don't want cross-device or cloud, the "auto-edit + dictionary + snippets" combo is a large part of perceived quality.

## Likely Root Causes Of The Quiet-Scenario Quality Gap

These are the most plausible explanations given the current code:

1. **You are optimizing for noisy whispering**, which encourages aggressive preprocessing. In quiet conditions, that same preprocessing can reduce intelligibility.
   - Current noise reduction is always applied when enabled, even if the recording is clean.
2. **Deepgram Flux streaming is optimized for latency**, not maximum accuracy.
   - Your `config.yaml` uses `deepgram.model: flux-general-en` for streaming. Quiet scenarios usually tolerate higher latency, so you can spend accuracy budget.
3. **There is no LLM cleanup layer** (punctuation, rewriting, formatting).
   - Your post-processing is deterministic dictionary substitutions only.
4. **No context injection into ASR**.
   - Local Whisper supports `initial_prompt` but it is not used.
   - Deepgram supports keywords/hotwords style biasing; you have `keyterm` but it is not integrated with your full dictionary strategy.
5. **Whispering is an edge case for both ASR and VAD**.
   - Many VADs are tuned for normal speech energy.
   - Many ASR pipelines are trained primarily on normal voice levels; whispering benefits from careful gain/normalization/compression.

## Improvement Plan (Concrete And Prioritized)

### 1) Add A First-Class "Quiet Mode" Profile

Objective: don't let the "noisy whisper" choices degrade quality when you're in a clean room.

Proposed config-level modes:

- `mode: noisy_whisper`
  - Keep speaker verification on.
  - Keep some noise suppression.
  - Favor streaming latency.
- `mode: quiet_dictation`
  - Bypass speaker verification (or lower the threshold cost).
  - Disable noise suppression by default.
  - Favor accuracy-first ASR model/provider.
- `mode: coding`
  - Similar to quiet_dictation, but with formatting rules and vocabulary bias tuned for code and CLI.

Implementation sketch:

- Add `app.mode` in `config.yaml`.
- In `src/main.py` (on startup), derive effective settings for:
  - `noise_suppression.enabled`
  - `voice_fingerprint.mode` (talk vs whisper verify)
  - `transcription.provider` and model selection

This is the single biggest low-effort change that can improve "quiet-room quality" because it avoids unnecessary audio damage.

### 2) Make Noise Suppression Adaptive (Or At Least Optional Per Mode)

Objective: keep noise reduction when needed; avoid artifacts when not.

Immediate changes:

- For `quiet_dictation` mode: disable `NoiseSuppressor.process(...)`.
- Add a simple gate: only run noisereduce if the first N ms looks noisy.
  - Example heuristic: measure RMS in a "silence" window at the start; if it is below a threshold, skip reduction.

Medium upgrade (higher impact):

- Replace `noisereduce` with a modern speech denoiser (DeepFilterNet or RNNoise).
  - These can preserve consonants better than aggressive spectral gating, especially on whispers.

Where this lands in code:

- `src/noise_suppressor.py`

### 2.5) Add Gain Normalization + Light Compression (Whisper-First Front End)

Objective: whispers fail when key consonant energy is too low. Before denoising, make the speech "look like speech" to the ASR.

Low-risk baseline chain (configurable, and only enabled in whisper modes):

1. Peak normalization (or loudness normalization) to a target level.
2. Gentle high-pass (keep your 80 Hz filter).
3. Light dynamic range compression (so consonants lift without clipping vowels).

Where this lands:

- New helper functions in `src/noise_suppressor.py` or a new `src/audio_preprocess.py`
- Called from `Bloviate.process_recording(...)` before speaker verification/transcription

### 3) Add Whisper "initial_prompt" And Better Vocabulary Biasing

Objective: improve recognition of your domain words (tools, names, commands) without relying solely on after-the-fact substitutions.

Local Whisper path:

- Add `transcription.initial_prompt` to `config.yaml`.
- Thread it into `self.model.transcribe(..., initial_prompt=...)` in `src/transcriber.py`.
- Generate the prompt automatically from `custom_dictionary.yaml` (phrases only; keep it short).

Deepgram path:

- Expand `deepgram.keyterm` from the same source as your dictionary.
- Consider app/mode-specific keyterms (coding mode vs writing mode).

### 4) Add A Post-Processing "Cleanup And Format" Step (Optional)

Objective: match the "polished output" feel of commercial tools.

Add an optional stage after transcription (and before clipboard output):

- Remove filler words (configurable list).
- Normalize punctuation and casing.
- Lightweight grammar fixes.
- Mode-specific transformations:
  - coding mode: keep literal punctuation words, avoid rewriting identifiers
  - writing mode: add punctuation, fix capitalization, collapse repeated words

Start with deterministic rules (fast, predictable), then optionally add an LLM rewriter:

- Local LLM if you want privacy (higher setup cost).
- Remote LLM if you want best quality quickly.

Add mode and context support (this is the Monologue-style differentiator):

- Determine the active app (macOS: AppleScript `System Events` can report the frontmost app).
- Optional: capture selected text (what you are editing) and/or a small screenshot region.
- Feed that context into the cleanup/rewriter step so it can:
  - match tone (Slack vs email vs notes)
  - preserve code blocks and identifiers
  - keep formatting consistent with what is already on screen

Where this lands:

- New module like `src/post_processor.py`
- Called from `Transcriber.output_text(...)` or from `Bloviate.process_recording(...)`

### 5) Upgrade The ASR Provider Strategy For Accuracy

Objective: in quiet scenarios, pick the strongest model you can tolerate on latency/cost.

Concrete options:

- Deepgram:
  - Use your most accurate model for prerecorded dictation even if streaming uses Flux.
  - If Deepgram offers a higher-accuracy streaming model on your account, expose it as a selectable option.
- Whisper local:
  - Consider switching from `openai-whisper` to an optimized runtime (for example `faster-whisper`) for speed and to make running larger models practical.
  - If you have Apple Silicon, consider a Metal-accelerated backend (Whisper.cpp / WhisperKit style approach) rather than Python+torch CPU.
- Add OpenAI Speech-to-Text as an optional provider:
  - Paid tools often win here by using very strong hosted models plus a cleanup layer.

Where this lands:

- `src/transcriber.py`
- `config.yaml` additions

### 6) Segment-Level Processing (Optional, Helps Mixed Speech)

Objective: when you and a coworker overlap, treat it as multiple segments and only pass through the segments that match you.

Approach:

- Run VAD to segment the recording into speech chunks.
- For each chunk:
  - Verify speaker embedding
  - Keep only "matched" chunks
- Concatenate matched chunks and transcribe

This is more work, but it is a structural improvement (and aligns with how high-end diarization pipelines behave).

Where this lands:

- `src/main.py` and `src/noise_suppressor.py` (or a new `src/segmenter.py`)

## Quick Experiments To Run Before Big Refactors

These experiments can identify the main contributor to "quiet-room quality" issues quickly:

1. Compare transcription with noise suppression ON vs OFF (in a quiet room).
   - If OFF is better: your current denoising is harming signal quality.
2. Compare Deepgram Flux streaming transcript vs Deepgram prerecorded transcript (same audio).
   - This isolates "streaming model vs accuracy model" differences.
3. Compare Deepgram prerecorded vs local Whisper (`medium.en`) on the same clip.
   - This tells you whether it is primarily a model/provider issue.
4. Compare speaker similarity and transcription quality when verifying on raw vs denoised audio.
   - If raw verification is more stable, you can verify on raw and transcribe on denoised.

## Measurement: How To Iterate Without Guessing

If you want to close the quality gap quickly, add an evaluation loop:

1. Create a small "golden set" of whisper recordings:
   - quiet room
   - noisy office
   - mixed speech overlap
2. Write down exact reference transcripts (ground truth).
3. Compute WER (word error rate) for each change.

Implementation idea:

- Add a simple CLI script (future work): `python scripts/evaluate.py recordings/*.wav --refs refs.json`
- Use `jiwer` for WER.

Even 20 to 30 examples is enough to catch regressions and validate improvements.

## Suggested Next Step (To Use Your Example Audio)

When you send the "example audio of what whispers are being actively picked up":

1. Add a debug option that saves:
   - raw audio (WAV)
   - noise-suppressed audio (WAV)
   - speaker similarity score
   - final transcript (and interim stream transcript if Deepgram)
2. We can then pinpoint whether the quiet-quality gap is:
   - preprocessing artifacts
   - model choice (Flux vs Nova vs Whisper)
   - missing cleanup/formatting
