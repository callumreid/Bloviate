# "Your Computer Should Know Your Voice"
## SF AI Meetup -- 20 Minute Talk

**One-sentence thesis:** Off-the-shelf dictation tools assume you're alone in a quiet room, so I built one that knows my voice -- and it changed how I think about what a computer interface should be.

---

## Timing Overview

| Section | Duration | Cumulative | Notes |
|---------|----------|------------|-------|
| Cold open | 0:30 | 0:30 | Whisper into mic |
| The problem | 2:00 | 2:30 | Audience pain point |
| Who I am | 2:00 | 4:30 | RSI, Volley, Coval |
| Introducing Bloviate | 1:30 | 6:00 | Name, concept, open source |
| The pipeline | 3:00 | 9:00 | Technical walkthrough |
| **LIVE DEMO** | 4:00 | 13:00 | Centerpiece |
| Custom dictionary | 1:30 | 14:30 | 970 entries, the craft |
| Voice commands | 1:30 | 16:00 | Beyond transcription |
| The bigger picture | 2:30 | 18:30 | Voice-first vision + Coval |
| Closing / CTA | 1:30 | 20:00 | Open source, Coval, thanks |

---

## Narrative Arc

### Act 1: The Itch (0:00 - 6:00)

**Goal:** Make the audience feel the problem. Establish credibility.

**Beat 1 -- Cold Open (0:30)**
Walk up. Lean into mic. *Whisper:* "Can everyone hear me?"
Pause. Let it land.
Normal voice: "Good. Because my computer can too. And unlike Siri, it knows it's *me* and not the person sitting next to me."

Transition: "I'm [name], I work at Coval where we build reliability infrastructure for voice AI, and tonight I want to talk about a tool I built for myself that changed how I think about voice interfaces."

**Beat 2 -- The Problem (2:00)**
"Raise your hand if you've ever tried dictating something at a coffee shop or an open office."

Wait for hands. "What happens? You get the barista's latte order. You get your coworker's phone call."

Run through the competitive landscape: "I tried everything. Whispr Flow -- great in a quiet room, completely useless if there's a conversation behind you. Monologue -- same problem. Apple's built-in dictation -- don't even get me started."

Land the insight: "They all have the same blind spot: **zero concept of who is speaking.** They just transcribe the loudest voice. If you're whispering -- because you're in a shared space, or because you don't want to be that person talking to their computer -- you're out of luck."

**Beat 3 -- Why This Is Personal (2:00)**
"I should explain why I care about this so much."

RSI story -- keep it real, don't milk it: "I have tendonitis. Typing all day is not an option for me. Dictation isn't a nice-to-have -- it's how I write code, write Slack messages, write prompts for Claude. It's my primary input method."

Volley background: "I spent four years at Volley building voice games -- Alexa games, voice-controlled smart TV games. So I know this problem space. I know what it takes to get reliable voice input from a user in an imperfect environment."

Coval context: "Now I'm at Coval. We help companies test and evaluate voice AI agents -- you build a voice agent, we simulate thousands of real conversations against it with different personas, accents, background noise, interruption patterns. We measure everything: latency, tone, accuracy, whether it actually resolves the customer's issue. I spend all day thinking about voice quality and edge cases. And I *still* couldn't find a dictation tool that worked in a real office."

Punch line: "So I built one."

**Beat 4 -- Introducing Bloviate (1:30)**
"I called it Bloviate -- which means to speak at length in a pompous way -- because irony is a feature."

Core pitch: "It's open source, about 1,500 lines of Python, and the core idea is dead simple: before transcribing anything, verify that the audio actually came from me. If it's not my voice, throw it away. Never transcribe it. Never show it to me."

"It uses a whisper mic setup -- I whisper into a close-talk mic so I'm not disturbing anyone around me. And because of speaker verification, even if three loud conversations are happening right behind me, the only text that appears is what I said."

---

### Act 2: The Build (6:00 - 16:00)

**Goal:** Show how it works, then prove it live. Reveal surprising depth.

**Beat 5 -- The Pipeline (3:00)**
Show the pipeline diagram. Point to Speaker Verification box.

"Most of this is standard stuff. Audio capture, noise suppression, VAD, transcription, clipboard. You'd find these stages in any dictation tool. The thing that makes Bloviate different is this box right here: **speaker verification**. Before any audio reaches the transcription engine, it has to pass a voice identity check."

Go deeper on verification: "The model is called ECAPA-TDNN, from the SpeechBrain library, trained on VoxCeleb -- that's a dataset of over 7,000 speakers. It takes raw audio and produces a 192-dimensional embedding -- think of it as a voice fingerprint."

"During setup, you record 8 samples of yourself whispering. The system averages those into a reference embedding. Then at runtime, every chunk of audio gets compared against your reference using cosine similarity."

Drop the numbers: "And here's the beautiful part: the separation is *clean*. My voice consistently scores 0.7 to 0.9. Background voices -- even loud ones, even people standing right next to me -- score 0.2 to 0.4. There's a massive gap. The threshold sits at 0.70, right in the middle."

For the engineers: "It's the same technology used for voice biometrics in banking -- speaker verification, not speaker identification. Repurposed for dictation."

**Beat 6 -- LIVE DEMO (4:00)**
"Alright, enough slides. Let me show you."

Switch to screen share. Walk through:

1. **Baseline** -- Whisper a sentence. Text appears. Point out similarity score in terminal: "0.82 -- that's me."

2. **The noise test** -- "Now let's make it interesting." Play YouTube podcast at medium volume. Let audience hear it for 5 seconds. "Right now, if I used macOS dictation, it would transcribe this podcast. Let's see what Bloviate does." Whisper a sentence. Correct text appears. "0.79 similarity. Accepted. The podcast? Never even made it to the transcription engine."

3. **Adversarial** -- "What if I don't talk at all?" Hold PTT for 3 seconds with just background audio. Show rejection: "0.35. Rejected. Not even close."

4. **Voice commands** -- "One more thing." Switch to command hotkey. Whisper "window left" -- window snaps. "Window right" -- snaps back. "Not Siri. Not a chatbot. Direct voice control."

*If demo fails:* "The demo gods are not with us today, but I recorded this exact flow yesterday -- same code, same mic." Play backup video. Totally fine for a meetup.

**Beat 7 -- The Dictionary (1:30)**
"Here's something that surprised me about building a personal dictation tool."

"Off-the-shelf transcription models -- even great ones like Deepgram or OpenAI's transcribe endpoint -- don't know your vocabulary. They don't know that 'get push' means `git push`. They don't know your coworker's names. They don't know `kubectl`."

Show the number: "Bloviate has a custom dictionary. **970 entries.** It's a YAML file where I map misheard phrases to what I actually meant. 'Glove eight' becomes 'Bloviate.' 'Pie torch' becomes 'PyTorch.' 'Cube control' becomes 'kubectl.'"

"It sounds tedious -- it is tedious -- but it's the difference between a tool that works 80% of the time and one that works 98% of the time. And I recently started auto-generating entries from my Obsidian vault, so it scales better than you'd think."

**Beat 8 -- Voice Commands (1:30)**
"Once you have a system that reliably knows your voice, you start thinking bigger."

"Bloviate has a command mode -- different hotkey from dictation. I hold a key and whisper 'window left,' and the focused window snaps. 'Fullscreen,' it maximizes. It's deterministic, it's instant, and because it goes through speaker verification, nobody else can accidentally trigger it."

"This is where it stops being a dictation tool and starts being a voice-controlled operating system. I want to get to the point where anything I can do with a keyboard shortcut, I can do with a whisper. We're not there yet -- but we're closer than you'd think."

---

### Act 3: The Vision (16:00 - 20:00)

**Goal:** Zoom out. Connect personal tool to industry insight. Leave them thinking.

**Beat 9 -- The Bigger Picture (2:30)**
"I want to leave you with a thought that's bigger than my RSI."

"We've had good speech-to-text for years. Whisper dropped in 2022. Deepgram and AssemblyAI keep shipping better models. Accuracy is effectively solved for English in clean audio."

"So why isn't everyone dictating? Why do we still default to keyboards?"

Pause. Let it hang.

"It's not an accuracy problem. **It's an identity problem.** Dictation tools don't know who's talking. Voice assistants don't verify you before acting. The entire voice AI stack -- from consumer dictation to enterprise voice agents -- is missing the identity layer."

Connect to Coval: "At Coval, I see this from the other side. We help companies test voice agents at scale, and one of the most common production issues is the agent responding to the wrong person, or to background noise, or to a TV playing in the room. The fix is the same thing Bloviate does: **verify the speaker before processing the speech.**"

The reframe: "Your phone knows your face. Your laptop knows your fingerprint. **Your computer should know your voice.** Once you add that identity layer, voice stops being an accessibility feature or a party trick. It becomes a primary interface."

**Beat 10 -- Closing (1:30)**
"Bloviate is open source. It's Python, it's hackable, and I'd love contributors. If you want better Linux support, streaming improvements, or you just want to add to the dictionary -- come find me after."

"And if you're building voice agents and you need to test them properly -- realistic simulated conversations, background noise, latency metrics, CI/CD integration, the works -- that's what we do at Coval. coval.dev."

"Thanks."

---

## Pre-Flight Checklist

- [ ] Scarlett 4i4 tested with venue AV setup (15 min before)
- [ ] Bloviate running with `show_main_window: true`
- [ ] Terminal visible with similarity score logs
- [ ] Text editor open as paste target
- [ ] YouTube podcast tab ready (medium volume, with clear speech)
- [ ] Backup demo video recorded on your machine
- [ ] Phone on silent
- [ ] Know the WiFi situation (Deepgram needs internet -- if flaky, pre-configure Whisper fallback)

---

## Q&A Cheat Sheet

| Question | Answer |
|----------|--------|
| "Why not a directional mic?" | Reduces volume, doesn't eliminate. Still picks up person across from you. And doesn't know WHO is speaking. |
| "What's the latency?" | 3-5s total. Streaming gives interim results while you're still talking. Fine for dictation, not for real-time convo. |
| "Why not fine-tune Whisper on your voice?" | Helps accuracy, doesn't solve identity. You'd transcribe background speakers more accurately. |
| "What about Apple Voice Isolation?" | Getting better but not exposed as API for third-party dictation. No enrollment-based verification. |
| "How much did this cost to build?" | Free. SpeechBrain is open source, Deepgram has a generous free tier, Whisper runs locally. Hardware cost is the Scarlett (~$230) and a decent mic. |
| "Does it work with other languages?" | Deepgram and Whisper both support multilingual. Speaker verification is language-agnostic (it's comparing voice characteristics, not words). Haven't tested extensively. |
