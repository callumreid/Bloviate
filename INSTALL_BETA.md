# Bloviate Beta Install

This is the fastest path for other macOS users to try Bloviate without a signed `.app`.

## Recommended Install Path

Use Homebrew for system prerequisites, then install Bloviate with `pipx` from GitHub.

```bash
brew install portaudio python@3.12 pipx
pipx ensurepath
pipx install git+https://github.com/callumreid/Bloviate.git
```

After that, the command should be available as:

```bash
bloviate --help
```

## First Run

```bash
bloviate --doctor
bloviate --show-paths
bloviate --list-devices
bloviate --voice-mode talk
```

If they want local vocabulary hints:

```bash
bloviate --init-personal-dictionary
```

If they want whisper verification mode:

```bash
bloviate --enroll
bloviate
```

## Where User Data Lives

On macOS, Bloviate now stores user data under:

```bash
~/Library/Application Support/Bloviate
```

That includes:

- `config.yaml`
- `personal_dictionary.yaml`
- `models/voice_profile.pkl`
- `models/pretrained/...`
- `logs/`

## Environment Variables

Hosted providers still expect API keys in the shell environment:

```bash
export DEEPGRAM_API_KEY="..."
export OPENAI_API_KEY="..."
```

Optional overrides:

```bash
export BLOVIATE_HOME="$HOME/Library/Application Support/Bloviate"
export BLOVIATE_ENV_FILE="$HOME/Library/Application Support/Bloviate/.env"
```

## Upgrades

```bash
pipx upgrade bloviate
```

If the package was installed directly from GitHub and you want the latest repo state:

```bash
pipx reinstall git+https://github.com/callumreid/Bloviate.git
```
