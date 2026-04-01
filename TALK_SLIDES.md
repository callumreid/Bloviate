# Slide Deck: "Your Computer Should Know Your Voice"

Each slide below has: what appears on screen, presenter notes, and transition cues.
Designed for ~11 slides. Keep slides minimal -- you're the content, slides are the backdrop.

---

## Slide 1: Title

```
YOUR COMPUTER SHOULD KNOW YOUR VOICE

Building voice-fingerprinting dictation for the real world

[Your Name] -- Coval
SF AI Meetup
```

**Presenter Notes:**
Walk up. Lean into mic. *Whisper:* "Can everyone hear me?"
Pause 2 beats. Let it land.
Normal voice: "Good. Because my computer can too. And unlike Siri, it knows it's *me* and not the person sitting next to me."
"I'm [name], I work at Coval where we build reliability infrastructure for voice AI, and tonight I want to talk about a personal tool I built that changed how I think about voice interfaces."

**Transition:** "Let me start with a question."

---

## Slide 2: The Problem

```
"Every dictation tool assumes
 you're alone in a quiet room."


  Whispr Flow    -->  transcribes your coworker
  Monologue      -->  transcribes the TV
  macOS Dictation --> transcribes... something
  Bloviate       -->  transcribes YOU
```

**Presenter Notes:**
"Raise your hand if you've ever tried dictating at a coffee shop or open office."
Wait for hands.
"What happens? You get the barista's latte order. You get your coworker's phone call."
"I tried every tool. Whispr Flow -- brilliant in a quiet room. Completely useless if there's a conversation behind you. They all have the same blind spot: zero concept of WHO is speaking. They transcribe the loudest voice. If you're whispering? Forget it."

**Transition:** "I should explain why I care about this so much."

---

## Slide 3: Why This Is Personal

```
RSI made me a voice-first engineer.

  2020-2024   Volley -- voice AI games (Alexa, smart TV)
  2024-now    Coval  -- voice AI evaluation & testing
  Always      typing hurts, whispering doesn't
```

**Presenter Notes:**
"I have tendonitis. Typing all day is not an option. Dictation isn't a nice-to-have for me -- it's how I write code, Slack messages, prompts for Claude. It's my primary input method."
"I spent four years at Volley building voice games for Alexa and smart TVs -- I know what it takes to get reliable voice input in imperfect environments."
"Now I'm at Coval. We help companies test voice AI agents. You build a voice agent, we simulate thousands of real conversations against it -- different personas, accents, background noise, interruption patterns. We measure latency, tone, accuracy, resolution. I think about voice quality all day. And I still couldn't find a dictation tool that worked at my desk."

**Transition:** "So I built one."

---

## Slide 4: Introducing Bloviate

```
BLOVIATE
/ˈblōvēˌāt/  (v.) to speak at length in a pompous way


Open-source voice-fingerprinting dictation
~1,500 lines of Python  |  11 modules

Core idea: verify it's YOU before transcribing anything.

github.com/[your-repo]
```

**Presenter Notes:**
"I called it Bloviate because irony is a feature."
"It's open source, about 1,500 lines of Python, and the idea is dead simple: before transcribing anything, verify the audio actually came from me. If it's not my voice, throw it away."
"I use a whisper mic setup -- I whisper into a close-talk mic so I don't disturb anyone. And because of speaker verification, even if three loud conversations are happening behind me, only my words get transcribed."

**Transition:** "Let me show you how it works under the hood."

---

## Slide 5: The Pipeline

```
                    THE PIPELINE

  Microphone (Scarlett 4i4, whisper mic)
       |
  Noise Suppression (high-pass + spectral)
       |
  Voice Activity Detection (WebRTC VAD)
       |
  ┌─────────────────────────────┐
  │   SPEAKER VERIFICATION      │  <-- the innovation
  │   SpeechBrain ECAPA-TDNN    │
  │   "Is this MY voice?"       │
  └─────────────────────────────┘
       |
  Transcription (Deepgram / OpenAI / Whisper)
       |
  Custom Dictionary (970 corrections)
       |
  Output (clipboard + auto-paste)
```

**Presenter Notes:**
"Most of this pipeline is standard. Audio capture, noise suppression, VAD, transcription -- you'd find these in any dictation tool. The thing that makes Bloviate different is this one box: speaker verification."
"Before any audio reaches the transcription engine, it has to pass a voice identity check. If it's not me, it gets rejected. No transcription, no output, nothing."

**Transition:** "Let me zoom in on that box."

---

## Slide 6: Speaker Verification

```
SPEAKER VERIFICATION

Model:      SpeechBrain ECAPA-TDNN
Training:   VoxCeleb (7,000+ speakers)
Output:     192-dimensional voice embedding
Compare:    Cosine similarity vs. enrolled profile

    ┌──────────────────────────────┐
    │                              │
    │  YOUR VOICE:    0.7 - 0.9   │  ✓ ACCEPTED
    │                              │
    │  BACKGROUND:    0.2 - 0.4   │  ✗ REJECTED
    │                              │
    │  Threshold:     0.70         │
    │                              │
    └──────────────────────────────┘

Enrollment: 8 whisper samples → averaged reference embedding
```

**Presenter Notes:**
"This model takes raw audio and produces a 192-dimensional embedding -- a voice fingerprint. During setup you record 8 whisper samples. The system averages those into a reference."
"At runtime, every recording gets compared via cosine similarity. And the separation is clean."
Drop the numbers: "My voice: 0.7 to 0.9. Background voices: 0.2 to 0.4. Massive gap. The threshold sits at 0.70."
For engineers: "Same tech used for voice biometrics in banking. Repurposed for dictation. The embeddings are invariant to background noise but sensitive to speaker identity."

**Transition:** "Enough theory. Let me prove it."

---

## Slide 7: Live Demo

```
LIVE DEMO
```

*Switch to screen share showing: Bloviate UI, terminal with logs, text editor*

**Presenter Notes:**

**Phase 1 -- Baseline (30s):**
"I have Bloviate running. You can see the indicator up here. Let me dictate a sentence."
Hold PTT, whisper: "Hello SF AI meetup, this is Bloviate."
Release. Text appears.
"See that score in the terminal? 0.82. That's me."

**Phase 2 -- The Noise Test (90s):**
"Now let's make it interesting."
Play YouTube podcast at medium volume. Let audience hear it 5 seconds.
"If I used macOS dictation right now, it would transcribe this podcast. Let's see what Bloviate does."
Hold PTT, whisper: "Bloviate only transcribes my voice, not the background noise."
Release. Correct text appears.
"0.79. Accepted. The podcast? Rejected before it ever reached Deepgram."

**Phase 3 -- Adversarial (60s):**
"What if I don't talk at all?"
Keep YouTube playing. Hold PTT 3 seconds without speaking.
Release. Show rejection.
"0.35. Rejected. That's the gap. My voice: 0.7 to 0.9. Everything else: 0.2 to 0.4."

**Phase 4 -- Voice Commands (30s):**
"One more thing. Bloviate isn't just dictation."
Switch to command hotkey. Whisper "window left" -- window snaps.
"Window right" -- snaps back.
"Direct voice control. Not Siri. Not a chatbot."

**If demo fails:** "The demo gods aren't with us, but I recorded this exact flow yesterday." Play backup video.

**Transition:** "So that's the core. But there's a layer of craft on top that I want to show you."

---

## Slide 8: The Dictionary

```
970 CUSTOM DICTIONARY ENTRIES

  "get push"       →   git push
  "glove eight"    →   Bloviate
  "pie torch"      →   PyTorch
  "cube control"   →   kubectl
  "co val"         →   Coval
  "wisper"         →   Whisper
  "does the linter pass" → 10 variation mappings

  + Deepgram keyword boosting (100 terms, 3x weight)
  + Auto-generated from Obsidian vault
```

**Presenter Notes:**
"Here's something that surprised me. Even great transcription models don't know YOUR vocabulary. They don't know 'get push' means git push. They don't know your coworker's names or your CLI commands."
"So I built a dictionary. 970 entries. YAML file mapping misheard phrases to corrections. It sounds tedious -- and it is -- but it's the difference between 80% accuracy and 98%."
"I also feed the top 100 terms to Deepgram as keyword boosts so the model is biased toward my vocabulary from the start. And I recently started auto-generating entries from my Obsidian vault, so new terms from my notes flow into the dictionary."

**Transition:** "And once you have reliable voice identity plus a tuned vocabulary, you start thinking bigger."

---

## Slide 9: Beyond Dictation

```
VOICE COMMANDS

  "window left"        →  snap to left half
  "window right"       →  snap to right half
  "window fullscreen"  →  maximize
  "window smaller"     →  shrink

  Separate hotkey: Cmd+Option = dictate, Ctrl+Cmd = command

  Not a chatbot. Deterministic. Instant.
  Speaker-verified: nobody else triggers your commands.
```

**Presenter Notes:**
"Bloviate has a command mode on a separate hotkey. I hold Ctrl+Cmd and whisper 'window left' and it snaps. It's not conversational AI. It's direct, deterministic control."
"The vision: anything I can do with a keyboard shortcut, I want to do with a whisper. Window management, app switching, eventually Raycast-level stuff. We're not all the way there yet, but the foundation is solid."
"And because everything goes through speaker verification, it's inherently secure. Nobody else can move my windows."

**Transition:** "So I want to zoom out and leave you with a bigger thought."

---

## Slide 10: The Bigger Picture

```
VOICE-FIRST COMPUTING IS BLOCKED BY
IDENTITY, NOT ACCURACY.


  Your phone knows your face.        FaceID
  Your laptop knows your fingerprint. TouchID
  Your computer should know your voice.


  Accuracy is solved.
  Identity is the missing layer.
```

**Presenter Notes:**
"We've had good speech-to-text for years. Accuracy is effectively solved for English. So why isn't everyone dictating?"
Pause.
"It's not an accuracy problem. It's an identity problem. Dictation tools don't know who's talking. Voice assistants don't verify you before acting."
"At Coval, I see this from the enterprise side. We test voice agents at scale, and one of the most common production issues is the agent responding to background noise, or a TV, or the wrong person in the room. The fix is the same: verify the speaker before processing the speech."
"Your phone knows your face. Your laptop knows your fingerprint. Your computer should know your voice. Once you add that identity layer, voice becomes a real interface. Not a party trick. Not an accommodation. A primary input method."

**Transition:** Pause. Let it sit. Then move to closing.

---

## Slide 11: Closing

```
BLOVIATE -- open source, looking for contributors
github.com/[your-repo]

COVAL -- reliability infrastructure for voice AI
coval.dev

[Your Name]
@[your-handle]
```

**Presenter Notes:**
"Bloviate is open source. Python, hackable, and I'd love contributors. If you're interested in voice AI, if you have RSI, or if you just think it's cool to whisper commands at your computer -- come find me after."
"And if you're building voice agents and need to test them -- realistic simulated conversations with personas, background noise, latency metrics, CI/CD integration -- that's what we do at Coval."
"Thanks."

---

## Appendix: Backup Slides (if demo fails)

### Backup Slide A: Demo Screenshot

```
DEMO: TERMINAL OUTPUT

  [14:23:01] PTT pressed
  [14:23:04] Voice similarity: 0.82 ✓ ACCEPTED
  [14:23:05] Transcription: "Hello SF AI meetup this is Bloviate"
  [14:23:05] Applied dictionary: no corrections needed
  [14:23:05] Pasted to clipboard

  [14:23:15] PTT pressed (YouTube playing in background)
  [14:23:18] Voice similarity: 0.79 ✓ ACCEPTED
  [14:23:19] Transcription: "Bloviate only transcribes my voice"

  [14:23:30] PTT pressed (only background audio, no speech)
  [14:23:33] Voice similarity: 0.35 ✗ REJECTED
  [14:23:33] Audio discarded -- speaker mismatch
```

### Backup Slide B: Architecture Detail

```
  Audio → 16kHz mono float32
       → High-pass 80Hz (Butterworth 4th order)
       → Spectral noise reduction (noisereduce)
       → WebRTC VAD (aggressiveness 3)
       → ECAPA-TDNN embedding (192-dim)
       → Cosine similarity ≥ 0.70?
           YES → Deepgram streaming + final pass
                 → Dictionary corrections (970 entries)
                 → Clipboard + auto-paste
           NO  → Discard. Silent. Nothing happens.
```
