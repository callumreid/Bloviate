# Bloviate Production Readiness

## Recommended Scope

Bloviate is closest to a **macOS beta for technical users**, not a general-purpose consumer product yet.

That is the right story to tell in the short term:

- It has a real differentiated core: voice-gated whisper dictation in noisy environments
- It already has a credible internal workflow
- It still has platform, setup, and onboarding assumptions that are too sharp for broad release

## What Is Ready Now

- Core dictation loop exists and is usable
- Voice enrollment and verification are implemented
- Streaming plus final-pass transcription is in place
- Personal dictionary support exists
- The app now has a built-in preflight:
  - `python src/main.py --doctor`
  - `python src/main.py --list-devices`
- Safer tracked defaults are now in place:
  - no forced device preference
  - auto-paste disabled by default
  - window management disabled by default

## What Still Blocks Broader External Use

### 1. Installation Story

Right now the product is still repo-first:

- clone repo
- create venv
- install Python dependencies
- manage environment variables manually

That is acceptable for a beta audience, but not for a normal product launch.

### 2. User Configuration Still Lives In The Repo

Today the app assumes a repo checkout with tracked config files nearby.

For broader use, config and user data should move to per-user locations:

- `~/Library/Application Support/Bloviate/...`
- `~/Library/Preferences/...`

That includes:

- user config
- voice profile
- personal dictionary
- logs

### 3. macOS Permissions Are A Product Surface

The app depends on permissions that confuse users if they are not explained clearly:

- microphone
- accessibility / input monitoring for global hotkeys
- AppleScript-driven automation for some integrations

This needs a first-run onboarding flow, not just docs.

### 4. No Packaging / Distribution Artifact

There is no `.app`, installer, Homebrew formula, or signed release artifact yet.

For a serious external trial, the next step is a packaged macOS app with:

- a stable app icon/name
- bundled Python runtime
- signed and notarized distribution

### 5. No Automated Release Confidence

There is still no test suite or CI gate for:

- config parsing
- doctor command
- personal dictionary behavior
- command parsing
- transcription fallback logic

That means regressions are still too easy to introduce right before a talk or share-out.

## Best Near-Term Launch Plan

If the goal is "other people can try this soon," the best path is:

1. Position it as a **macOS technical beta**
2. Share a tight install guide plus `--doctor`
3. Ask testers to start in `--voice-mode talk`
4. Only then move them into voice enrollment / whisper verification
5. Collect the failures from first-run setup before doing packaging work

## Highest-Value Next Improvements

### Tier 1: Do These Before The Talk If Possible

- Keep the share scope to macOS
- Add a one-command install path or packaged app
- Add a first-run onboarding checklist to the README
- Keep defaults safe for strangers, not optimized for one personal setup
- Make it obvious how to recover from audio / permission failures

### Tier 2: Do These Next For A Real External Beta

- Move config and user data out of the repo
- Add structured logs and a user-visible log file location
- Add a tiny smoke-test suite and CI
- Add a release process for tagged builds
- Add a "Send diagnostic bundle" workflow for testers

### Tier 3: Product Improvements After Beta

- Better onboarding UI
- App-aware modes and formatting
- Cleaner packaging and updates
- Better cross-platform strategy, if that still matters

## Demo Framing

The strongest framing for the talk is:

"Bloviate is not trying to be a generic dictation app first. It is solving a harder problem: reliable whisper dictation when other people are talking nearby. The productization work now is about making that specialized capability easy for other people to install, permission, calibrate, and trust."

## Demo Checklist

Before the talk, verify these on the machine you will use:

1. `python src/main.py --doctor`
2. `python src/main.py --list-devices`
3. `python src/main.py --voice-mode talk`
4. `python src/main.py --enroll` if using whisper verification mode
5. API keys present for any hosted providers you plan to demo
6. macOS microphone and accessibility permissions already granted
7. A fallback path ready:
   - talk mode
   - local Whisper fallback
   - clipboard-only output if auto-paste is off
