# Bloviate

Bloviate is a macOS-first voice dictation app for whispering in noisy spaces. It combines push-to-talk dictation, speaker verification, streaming transcription, an accuracy-first final pass, personal vocabulary, local history, and bring-your-own API keys.

## What It Does

- Verifies your enrolled voice before transcribing in whisper mode
- Supports talk mode when you want normal dictation without speaker verification
- Supports hold-to-talk and toggle-to-talk; `Cmd+Option+Shift` toggles dictation by default
- Cycles cleanup mode with three quick `Command` taps and shows the selected mode in the bottom overlay
- Runs isolated voice commands like `screen left half`, `run command desktop right`, or `open Slack`
- Uses Deepgram for live interim text, OpenAI or Deepgram for final text, and local Whisper as fallback
- Lets you configure audio input, hotkeys, models, providers, API keys, dictionary, cleanup, history, startup behavior, and diagnostics from Settings
- Shows local usage insights in Settings: words dictated, speaking pace, cleanup fixes, app breakdown, and streak heatmap
- Stores user config/state under `~/Library/Application Support/Bloviate`
- Keeps transcript history locally in SQLite; raw audio is not stored by default

## Current Status

Bloviate is a technical macOS beta. It is ready to demo and share with technical users, but it is not yet a signed/notarized consumer `.app`.

Recommended first public install path:

```bash
brew install portaudio python@3.12 pipx
pipx ensurepath
pipx install git+https://github.com/callumreid/Bloviate.git
```

Homebrew HEAD install for beta E2E testing:

```bash
brew tap callumreid/bloviate https://github.com/callumreid/Bloviate
brew install --HEAD callumreid/bloviate/bloviate
bloviate --install-launcher
```

See [INSTALL.md](/Users/bronson/personal/bloviate/INSTALL.md) for beta install, permissions, upgrades, and the signed-app roadmap.

## First Run

```bash
bloviate --doctor
bloviate --show-paths
bloviate --list-devices
bloviate --voice-mode talk
```

Then open Settings from the window or menu bar icon and configure. After `bloviate --install-launcher`, you can start Bloviate from `~/Applications/Bloviate.app` instead of a terminal. On first launch, Settings shows a permissions checklist with buttons for microphone, Accessibility, Input Monitoring, and Automation.

- microphone
- push-to-talk hotkey
- toggle dictation hotkey
- OpenAI / Deepgram API keys
- provider/model priority
- dictionary terms and corrections
- cleanup mode
- transcript history preference

For whisper verification mode:

```bash
bloviate --enroll
bloviate
```

## API Keys

Bloviate is bring-your-own-key. Settings saves OpenAI and Deepgram keys to macOS Keychain through `keyring` when available. Environment variables still work:

```bash
export OPENAI_API_KEY="..."
export DEEPGRAM_API_KEY="..."
```

The launcher sources `~/.zshenv` and `~/.zprofile` so common shell-defined API keys are visible when starting from `Bloviate.app`. Keychain storage is still preferred for public installs.

Provider priority defaults to:

1. OpenAI final pass
2. Deepgram final pass
3. Local Whisper fallback

## Personal Dictionary

The dictionary lives at:

```bash
~/Library/Application Support/Bloviate/personal_dictionary.yaml
```

It contains:

- `preferred_terms`: words/names/tools used to bias transcription
- `corrections`: deterministic rewrites for recurring mistakes

Example:

```yaml
preferred_terms:
  - Raycast
  - kubectl
  - gpt-4o-transcribe

corrections:
  - phrase: "kubectl"
    variations:
      - "cube cuddle"
      - "cube control"
    match: "substring"
```

You can edit this in Settings or from the CLI:

```bash
bloviate --init-personal-dictionary
bloviate --add-term "Raycast" --add-term "kubectl"
bloviate --show-personal-dictionary
```

Older repo-local `custom_dictionary.yaml`, `personal_dictionary.yaml`, and `learned_terms.txt` files are imported into the App Support dictionary on launch.

## Cleanup Modes

- Verbatim: keeps the transcription as returned by the speech model.
- Clean prose: removes common filler and normalizes sentences.
- Coding: avoids prose rewrites that would damage commands, filenames, or identifiers.
- Message: formats dictated text like a concise message; it differs most when OpenAI cleanup is enabled.

Tap `Command` three times quickly to cycle these modes without opening Settings. The bottom waveform overlay briefly expands to show the new mode.

## Voice Commands

Voice commands only execute when the whole utterance is the command. If a command phrase appears inside a longer dictated paragraph, Bloviate treats it as text.

Window commands:

```text
window left half
window right half
window top half
window bottom half
window full screen
window exit full screen
window larger
window smaller
window top left quarter
window top right quarter
window bottom left quarter
window bottom right quarter
```

You can say `screen ...` instead of `window ...`, or prefix with `run command`, for example `run command screen left half`.

Desktop and app commands:

```text
desktop left
desktop right
run command desktop left
run command desktop right
open Slack
launch Chrome
start Calendar
```

## Local History

Transcript history is enabled by default and stored locally at:

```bash
~/Library/Application Support/Bloviate/history.sqlite
```

History stores text metadata such as timestamp, mode, provider, target app/window, voice score, and output action. It does not store raw audio by default. Whisper-mode voice rejections are still transcribed into history when possible, but they are not auto-pasted. You can search, copy, delete, clear, export, or disable history in Settings.

## Permissions

On macOS, grant permissions to the app host you use to run Bloviate:

- Microphone: required for audio capture
- Accessibility / Input Monitoring: required for global hotkeys and auto-paste
- Automation/System Events: required for window-management commands

If running through Terminal or iTerm, grant permissions to that terminal app. If launching `~/Applications/Bloviate.app`, grant permissions to Bloviate.

## Development

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
python -m unittest discover -s tests
```

Useful commands:

```bash
python src/main.py --doctor
python src/main.py --list-devices
python src/main.py --show-paths
python src/main.py --voice-mode talk
```

## Release Roadmap

Current beta distribution is Homebrew prerequisites plus `pipx`.

Signed `.app` distribution is planned after beta hardening and requires:

- PyInstaller or similar app bundling
- stable icon/bundle identifier
- Apple Developer Program membership
- Developer ID signing certificate
- hardened runtime and notarization
- release CI that builds, signs, notarizes, and attaches artifacts

See [PRODUCTION_READINESS.md](/Users/bronson/personal/bloviate/PRODUCTION_READINESS.md) for launch posture and remaining readiness work.
