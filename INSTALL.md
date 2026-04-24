# Installing Bloviate

Bloviate is currently distributed as a macOS technical beta. The recommended public path is Homebrew system prerequisites plus `pipx`.

## Beta Install With pipx

```bash
brew install portaudio python@3.12 pipx
pipx ensurepath
pipx install git+https://github.com/callumreid/Bloviate.git
```

Verify:

```bash
bloviate --help
bloviate --doctor
bloviate --show-paths
bloviate --list-devices
```

## Homebrew HEAD Install

For end-to-end Homebrew testing from the pushed `main` branch:

```bash
brew tap callumreid/bloviate https://github.com/callumreid/Bloviate
brew install --HEAD callumreid/bloviate/bloviate
```

Upgrade/reinstall after a new push:

```bash
brew update
brew reinstall --HEAD callumreid/bloviate/bloviate
```

Remove:

```bash
brew uninstall bloviate
brew untap callumreid/bloviate
```

This formula is intended for beta testing. It builds a Homebrew-managed Python virtualenv and installs Bloviate into it.

Smoke test without voice enrollment:

```bash
bloviate --voice-mode talk
```

Enroll for whisper verification:

```bash
bloviate --enroll
bloviate
```

## API Keys

Open Settings and paste API keys into the Models & Providers section. Bloviate saves them to macOS Keychain when `keyring` is available.

Environment variables are also supported:

```bash
export OPENAI_API_KEY="..."
export DEEPGRAM_API_KEY="..."
```

## User Data Locations

Default macOS location:

```bash
~/Library/Application Support/Bloviate
```

Important files:

- `config.yaml`
- `personal_dictionary.yaml`
- `history.sqlite`
- `models/voice_profile.pkl`
- `models/pretrained/...`
- `logs/`

Override the home directory:

```bash
export BLOVIATE_HOME="$HOME/Library/Application Support/Bloviate"
```

## macOS Permissions

Bloviate needs:

- Microphone permission for audio input
- Accessibility/Input Monitoring for global hotkeys and auto-paste
- Automation/System Events for window-management commands

If you run Bloviate from Terminal/iTerm, grant permissions to that terminal. If a future `.app` build is used, grant permissions to Bloviate.

## Upgrades

```bash
pipx upgrade bloviate
```

If installed directly from GitHub:

```bash
pipx reinstall git+https://github.com/callumreid/Bloviate.git
```

## Signed App Roadmap

A polished direct-download `.app` is a separate release track. It requires:

- app icon and bundle identifier
- PyInstaller app bundle
- Apple Developer Program membership
- Developer ID Application certificate
- hardened runtime
- notarization
- release automation

Until then, the Homebrew + `pipx` path is the intended public beta install.
