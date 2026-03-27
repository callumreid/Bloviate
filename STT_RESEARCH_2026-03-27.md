# STT Research For Bloviate

Date: March 27, 2026

## Bottom Line

If the question is "should Bloviate replace Deepgram entirely?", my answer is: not immediately.

If the question is "what is the strongest speech-to-text stack for this repo right now?", my answer is:

1. Keep **Deepgram Nova-3** for live interim transcription.
2. Make sure **OpenAI `gpt-4o-transcribe`** is actually active as the accuracy-first final pass.
3. If you want a serious live-streaming challenger to Deepgram, test **ElevenLabs Scribe v2 Realtime** next.
4. If you want a serious batch/final-pass challenger to OpenAI and Deepgram, test **AssemblyAI Universal-3 Pro** next.

There is not a credible single "absolute best" model I can defend from official vendor materials alone, because the official claims conflict:

- OpenAI says `gpt-4o-transcribe` sets a new state-of-the-art benchmark.
- AssemblyAI says Universal-3 Pro outperforms all ASR models on the market.
- Deepgram says Nova-3 is its highest-performing general-purpose ASR with large WER gains versus competitors.
- ElevenLabs says Scribe v2 Realtime is the most accurate real-time model with ~150 ms latency.

For Bloviate specifically, the best answer is a **two-stage system**, not a single model swap.

## What The Repo Is Doing Today

Current repo behavior:

- Primary provider is `deepgram` in [`config.yaml`](./config.yaml).
- Live streaming is enabled and currently uses `deepgram.model: "nova-3"`.
- Final pass mode is `hybrid`, so live text can be replaced by a higher-accuracy offline pass.
- Final-pass provider priority is already:
  - `openai`
  - `deepgram`
  - `whisper`
- OpenAI is already configured for `gpt-4o-transcribe`, but only works if `OPENAI_API_KEY` is actually present.

Relevant code paths:

- Provider config: [`config.yaml`](./config.yaml)
- Final-pass ordering: [`src/transcriber.py`](./src/transcriber.py)
- Hybrid final-pass selection: [`src/main.py`](./src/main.py)

Important implication: you may already have the best immediate upgrade path in the codebase, but not necessarily active in practice.

## What "Best" Means For Bloviate

Bloviate is not generic transcription. It is:

- short push-to-talk utterances
- whisper or near-whisper input
- noisy shared environments
- speaker verification in front of ASR
- low-latency interim text plus a higher-accuracy final pass
- coding and command vocabulary biasing

That makes these criteria more important than raw leaderboard claims:

1. Low-latency streaming with usable interim text
2. Strong short-utterance accuracy
3. Good domain-vocabulary biasing or prompting
4. Good behavior in noisy audio
5. Easy fit with 16 kHz mono PCM and the repo's current architecture

## Vendor Findings

### 1. Deepgram

#### Nova-3

Deepgram describes Nova-3 as its "highest-performing general-purpose ASR" and recommends it for noisy, far-field, multilingual, multi-speaker, batch, and streaming use cases.

Why it fits Bloviate well:

- already integrated
- supports streaming and prerecorded
- supports keyterm prompting
- explicitly positioned for noisy and multi-speaker audio

Relevant official claims:

- Deepgram docs say Nova-3 is recommended for "meetings, event captioning, multi-speaker, multilingual, noisy, or far-field audio in batch or streaming."
- Deepgram's Nova-3 launch notes claim 54.3% lower streaming WER and 47.4% lower batch WER versus competitors, while keeping latency comparable to Nova-2.
- Deepgram keyterm docs say keyterm prompting can improve recall for up to 100 important terms.

#### Flux

Flux is not a straight Nova-3 replacement. It is a different product shape.

Why Flux is interesting:

- optimized for low-latency conversational turn detection
- official docs claim Nova-3-level accuracy
- official docs claim about 260 ms end-of-turn latency

Why Flux is not an obvious upgrade for Bloviate:

- Flux is optimized for voice agents and turn-taking, but Bloviate already uses push-to-talk, so model-native turn detection is less valuable here.
- Deepgram itself positions Nova-3 as the better fit for meeting/event/general transcription and Flux as the fit for turn-based voice agents.
- Flux uses `/v2/listen` and Deepgram recommends 80 ms chunks; Bloviate currently uses 1024-sample chunks at 16 kHz, which is about 64 ms. That is close, but not ideal if you want to tune specifically for Flux.

Verdict:

- **Best Deepgram model for Bloviate today:** Nova-3
- **Best Deepgram model if you want to trade toward live latency and turn-taking:** Flux

### 2. OpenAI

#### `gpt-4o-transcribe`

OpenAI's current speech docs and launch post make this the strongest accuracy-first model already wired into the repo.

Why it fits Bloviate well:

- already integrated
- supports prompt-based transcription guidance
- official docs say it improves WER over Whisper
- official launch post says it performs especially well on accents, noisy environments, and varying speech speeds

Important repo-specific point:

- Bloviate already prefers OpenAI first for final-pass transcription in `hybrid` mode.
- If `OPENAI_API_KEY` is missing, that path is skipped and Deepgram stays your effective final-pass provider.

OpenAI-specific strengths relevant here:

- prompt support for `gpt-4o-transcribe`
- better than Whisper on official benchmarks
- Realtime API supports incremental transcript deltas, manual or VAD-based turn handling, and configurable near-field/far-field noise reduction

OpenAI-specific weakness for live replacement:

- the realtime transcription API is a bigger integration than the current file-transcription final pass
- the realtime docs specify 24 kHz mono PCM for `audio/pcm`, so it is not a drop-in replacement for Bloviate's current 16 kHz live path

Verdict:

- **Best immediate accuracy upgrade for this repo:** make sure `gpt-4o-transcribe` is actually active for the final pass
- **Best OpenAI live experiment:** add a separate realtime provider instead of replacing the current Deepgram path blindly

### 3. AssemblyAI

#### Universal-3 Pro

AssemblyAI is the strongest batch challenger on paper.

Why it is compelling:

- official docs describe it as the company's most powerful model
- official docs say it outperforms all ASR models on the market, especially on entities and rare words
- async mode supports strong prompting and up to 1,000 keyterms
- AssemblyAI publishes benchmark tables showing Universal-3 Pro improving over Universal-2

Why it is not an obvious live replacement for Bloviate:

- streaming U3 Pro is optimized for real-time utterances typically under 10 seconds
- official docs say partials are only produced during periods of silence, not as continuous word-by-word updates
- that is a worse fit for Bloviate's current live interim UX than Deepgram or OpenAI Realtime

Verdict:

- **Best reason to add AssemblyAI:** strongest experimental accuracy-first final pass for rare words, entities, and domain prompting
- **Not my first choice for Bloviate's live interim provider**

### 4. ElevenLabs

#### Scribe v2 Realtime

This is the most interesting live Deepgram challenger I found.

Why it is compelling:

- ElevenLabs claims Scribe v2 Realtime is the most accurate model for live transcription
- official launch materials claim under 150 ms latency
- official docs expose a WebSocket API with partial transcripts, committed transcripts, and manual commit support
- the realtime API example uses 16 kHz PCM, which fits Bloviate better than OpenAI Realtime's 24 kHz PCM requirement

Why I would still treat it as experimental:

- it is newer
- official performance claims are vendor-reported
- the docs across ElevenLabs pages are not perfectly consistent on some feature details, which raises some integration-risk questions
- current repo has no integration for it

Verdict:

- **Best live-streaming challenger to Deepgram for this repo**

### 5. Google Cloud

#### Chirp 3

Google's current STT offering is strong, but it is not the best fit for Bloviate's core failure mode.

Why it is good:

- supports streaming, recognize, and batch
- supports adaptation / phrase biasing
- offers built-in denoising and SNR filtering

Why it is weaker for Bloviate:

- Google's own docs say Chirp 3's denoiser cannot remove background human voices
- background human speech is exactly the thing Bloviate is built to reject

Verdict:

- a credible general STT option
- not my recommended next move for this repo

## Ranked Recommendation For Bloviate

### Best overall stack for this repo right now

1. **Live interim:** Deepgram Nova-3
2. **Final pass:** OpenAI `gpt-4o-transcribe`
3. **Fallback:** Deepgram Nova-3 prerecorded, then local Whisper

Why this wins:

- lowest integration effort
- strongest immediate quality upside
- preserves the repo's current architecture
- separates "fast live stream" from "best final text"

### Best single-provider option if you want to stay inside Deepgram

1. Keep `model: "nova-3"`
2. Keep `prerecorded_model: "nova-3"`
3. Keep `final_pass: "hybrid"`
4. Expand keyterm/prompt strategy

I would **not** switch blindly from Nova-3 to Flux unless your main complaint is end-of-turn latency or agent-style turn handling.

### Best external live challenger to test next

1. **ElevenLabs Scribe v2 Realtime**

Reason:

- strongest current official live-latency claim
- strong realtime API shape for this app
- 16 kHz PCM fit is better than OpenAI Realtime for this codebase

### Best external final-pass challenger to test next

1. **AssemblyAI Universal-3 Pro**

Reason:

- strongest official prompting/keyterm story
- strongest rare-word/entity positioning
- best fit as a prerecorded accuracy pass, not as the live interim stream

## Concrete Improvements You Could Make Here

### 1. Ensure OpenAI final pass is actually on

Today, the code already prefers OpenAI first for final pass, but only if the key exists.

Recommended change:

- Add an explicit startup warning when `final_pass_provider_priority` includes `openai` but `OPENAI_API_KEY` is missing.

Why:

- right now the repo may look like it is using OpenAI final pass on paper while silently falling back in practice

### 2. Start using the OpenAI prompt field

Current config has `openai.prompt: ""`.

Recommended change:

- auto-build a short prompt from:
  - `custom_dictionary.yaml`
  - command vocabulary
  - code/editor mode terms

Why:

- OpenAI explicitly supports prompt-guided transcription for `gpt-4o-transcribe`
- this is likely one of the highest-leverage quality improvements for command names, package names, people names, and coding jargon

### 3. Treat live and final-pass providers as separate product decisions

Recommended change:

- formalize config around:
  - `streaming_provider`
  - `streaming_model`
  - `final_pass_provider`
  - `final_pass_model`

Why:

- you are already implicitly doing this
- making it explicit will make A/B testing much easier

### 4. Add an experimental provider adapter for ElevenLabs Realtime

Recommended change:

- add a fourth provider for live streaming experiments

Why:

- it is the cleanest current challenger to Deepgram for the live path

### 5. Add an experimental provider adapter for AssemblyAI Universal-3 Pro async

Recommended change:

- add it as a final-pass-only provider at first

Why:

- that is where its prompting and rare-word strengths are most useful
- it avoids the awkwardness of U3 Pro Streaming's silence-based partial behavior

### 6. Build a real evaluation harness

This matters more than another round of vendor reading.

You do not currently have a local golden evaluation set in the repo.

Build a small benchmark set with:

- 30 clean whispers
- 30 noisy-office whispers
- 20 overlapping-speech cases
- 10 short command phrases
- 10 coding dictation snippets

For each clip, record:

- reference transcript
- speaker accepted/rejected
- live latency to first interim
- live latency to final
- final WER
- named-entity / command-term accuracy

Without this, "absolute best" stays speculative.

## Recommended Next Actions

### Low effort / high value

1. Verify whether `OPENAI_API_KEY` is set on the machine where you actually run Bloviate.
2. Add a non-empty `openai.prompt`.
3. Keep Deepgram Nova-3 as the live stream model for now.

### Medium effort / highest information gain

1. Build the benchmark set.
2. Run head-to-head comparisons:
   - Deepgram Nova-3 live + OpenAI final
   - Deepgram Nova-3 live + AssemblyAI final
   - ElevenLabs Realtime live + OpenAI final
   - Deepgram Flux live + OpenAI final

### High effort / only if the benchmark justifies it

1. Add a dedicated OpenAI Realtime live provider.
2. Add ElevenLabs Realtime.
3. Add AssemblyAI Universal-3 Pro async.

## My Actual Recommendation

As of March 27, 2026:

- **Do not replace Deepgram outright yet.**
- **Do make sure OpenAI `gpt-4o-transcribe` is your active final-pass model.**
- **If you want one serious live replacement experiment, test ElevenLabs Scribe v2 Realtime.**
- **If you want one serious accuracy-first final-pass experiment, test AssemblyAI Universal-3 Pro.**

For Bloviate, the most likely "best" outcome is:

- Deepgram Nova-3 or ElevenLabs Realtime for live interim
- OpenAI `gpt-4o-transcribe` or AssemblyAI Universal-3 Pro for final pass

Not one vendor everywhere.

## Sources

Deepgram:

- https://developers.deepgram.com/docs/models-languages-overview
- https://developers.deepgram.com/docs/flux/flux-nova-3-comparison
- https://developers.deepgram.com/docs/keyterm
- https://developers.deepgram.com/docs/flux/quickstart
- https://deepgram.com/changelog/introducing-nova-3

OpenAI:

- https://developers.openai.com/api/docs/guides/speech-to-text
- https://developers.openai.com/api/docs/guides/realtime-transcription
- https://openai.com/index/introducing-our-next-generation-audio-models/

AssemblyAI:

- https://www.assemblyai.com/docs/getting-started/universal-3-pro
- https://www.assemblyai.com/docs/pre-recorded-audio/universal-3-pro
- https://www.assemblyai.com/docs/streaming/universal-3-pro
- https://www.assemblyai.com/docs/benchmarks
- https://www.assemblyai.com/docs/pre-recorded-audio/keyterms-prompting

ElevenLabs:

- https://elevenlabs.io/docs/overview/capabilities/speech-to-text/
- https://elevenlabs.io/blog/introducing-scribe-v2-realtime
- https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime

Google Cloud:

- https://cloud.google.com/speech-to-text/docs/models/chirp-3
- https://docs.cloud.google.com/speech-to-text/docs/adaptation-model
