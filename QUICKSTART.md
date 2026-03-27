# Bloviate Quick Start Guide

## Installation

1. Run the setup script:
```bash
cd bloviate
./setup.sh
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Optional: create your personal dictionary file:
```bash
cp personal_dictionary.example.yaml personal_dictionary.yaml
```

`personal_dictionary.yaml` is gitignored, so collaborators do not get each other's entries when they clone the repo.

4. Optional: add preferred terms for names, tools, and commands you use a lot:
```bash
python src/main.py --add-term "Raycast" --add-term "kubectl" --add-term "gpt-4o-transcribe"
```

This writes to `personal_dictionary.yaml` locally and keeps it out of git.

## First Time Setup: Voice Enrollment

Before using Bloviate, you need to enroll your whisper voice so it can distinguish you from background speakers.

```bash
python src/main.py --enroll
```

You'll be prompted to whisper 5 phrases. Make sure to:
- Speak in the same whisper tone you'll use during dictation
- Use the same microphone (Tascam TM-95GN)
- Be in a quiet moment (less background noise is better for enrollment)

The system will create a voice profile at `models/voice_profile.pkl`.

## Running Bloviate

```bash
python src/main.py
```

A small window will appear showing:
- PTT status (active/inactive)
- Audio level meter
- Voice match indicator
- Last transcription

### Using Push-to-Talk

1. Press and hold `Cmd+Option`
2. Whisper into your microphone
3. Release the hotkey
4. Wait for transcription

The transcribed text will be:
- Copied to your clipboard (ready to paste)
- Displayed in the UI window

### Window Management (Voice Commands)

1. Press and hold `Ctrl+Cmd`
2. Say "left", "right", "top", or "bottom"
3. Release the hotkey to resize the focused window

The `Ctrl+Cmd+Arrow` shortcuts still work for instant resizing.

### Workflow Tips

1. **Keep the UI window visible** so you can see if your voice is being detected
2. **Watch the voice match indicator** - if it shows "Rejected", the system detected speech but determined it wasn't your voice (likely background)
3. **Audio level should be green/yellow** when whispering - if it's always gray, check your mic input
4. **Start small** - try short phrases first to verify it's working

## Configuration

Edit `config.yaml` to customize:

### Change Hotkey
```yaml
ptt:
  hotkey: "<cmd>+<option>"  # Primary hotkey
  secondary_hotkey: "<fn>"  # Optional secondary hotkey
  # Other options:
  # "<ctrl>+<shift>+<space>"
  # "<cmd>+<shift>+<space>"
```

### Window Management Hotkeys
```yaml
window_management:
  enabled: true
  hotkey_prefix: "<ctrl>+<cmd>"  # Base hotkey, combine with arrow keys
  command_hotkey: "<ctrl>+<cmd>"  # Hold to speak window commands
```

### Adjust Voice Matching Sensitivity
```yaml
voice_fingerprint:
  threshold: 0.65  # Lower = more lenient (0.5-0.8 recommended)
```

If you're getting too many false rejections, lower the threshold to 0.6 or 0.55.

### Talk Mode (Bypass Voice Matching)
Use talk mode when you want normal speech to pass without voice verification.

```yaml
voice_fingerprint:
  mode: "talk"  # whisper (verify) or talk (bypass)
```

Or run a one-off session:

```bash
python src/main.py --voice-mode talk
```

### Select Audio Device
```yaml
audio:
  device_name: "Scarlett"  # Change if using different interface
```

Run `python -c "import sounddevice as sd; print(sd.query_devices())"` to see all available devices.

### Change Whisper Model
```yaml
transcription:
  model: "base.en"  # Options: tiny.en, base.en, small.en, medium.en
```

- `tiny.en` - Fastest, less accurate
- `base.en` - Good balance (recommended)
- `small.en` - More accurate, slower
- `medium.en` - Most accurate, slowest

### Personal Dictionary
Bloviate looks for a local `personal_dictionary.yaml` in the repo root by default. It can hold both preferred terms and correction rules. To keep that file outside the repo, set either:

```yaml
transcription:
  personal_dictionary_path: "~/Library/Application Support/Bloviate/personal_dictionary.yaml"
```

or:

```bash
export BLOVIATE_PERSONAL_DICTIONARY_PATH="$HOME/Library/Application Support/Bloviate/personal_dictionary.yaml"
```

Example structure:

```yaml
preferred_terms:
  - Raycast
  - kubectl
  - Claude Code

corrections:
  - phrase: "kubectl"
    variations:
      - "cube cuddle"
      - "cube control"
    match: "substring"
```

Add preferred terms from the CLI:

```bash
python src/main.py --add-term "Raycast" --add-term "kubectl" --add-term "Claude Code"
python src/main.py --show-personal-dictionary
```

Use `preferred_terms` when you want the models to bias toward exact spellings. Use `corrections` when you want a known wrong output rewritten deterministically.

If you already have `custom_dictionary.yaml` or `learned_terms.txt`, Bloviate still reads them while you migrate.

### Quality-First Hybrid Dictation (Recommended)
Use streaming for live feedback and accuracy-first providers for final text.

```bash
export DEEPGRAM_API_KEY="your_key_here"
export OPENAI_API_KEY="your_openai_key_here"
```

```yaml
transcription:
  provider: "deepgram"
  final_pass: "hybrid"
  final_pass_provider_priority:
    - "openai"
    - "deepgram"
    - "whisper"

deepgram:
  streaming: true
  model: "nova-3"
  prerecorded_model: "nova-3"
  smart_format: true
  no_delay: true

openai:
  model: "gpt-4o-transcribe"
```

Notes:
- Streaming sends audio while you hold PTT for low-latency interim text.
- Final text runs through provider priority order, so OpenAI can win accuracy while Deepgram/Whisper remain fallback.
- OpenAI/Whisper prompting automatically includes personal dictionary preferred terms and built-in command phrases.
- Command mode retries the higher-accuracy final-pass providers if the live stream does not parse into a known command.

### Adjust Noise Suppression
```yaml
noise_suppression:
  stationary_noise_reduction: 0.8  # 0-1, higher = more aggressive
  vad_aggressiveness: 3  # 0-3, higher = stricter voice detection
```

## Re-enrolling Your Voice

If the voice matching isn't working well, you can re-enroll:

```bash
python src/main.py --clear-profile
python src/main.py --enroll
```

Tips for better enrollment:
- Record in your actual work environment
- Use your natural whisper volume
- Speak full sentences, not just words
- Keep consistent distance from the mic

## Troubleshooting

### "No audio captured"
- Check that your Scarlett is connected and powered on
- Verify the TM-95GN is plugged into the Scarlett
- Check phantom power is enabled on the Scarlett if needed
- Run the device list command to verify it's detected

### "Voice rejected" on every attempt
- Lower the `voice_fingerprint.threshold` in config.yaml
- Re-enroll your voice in the current environment
- Check that you're using the same whisper tone as enrollment

### Background voices still getting through
- Increase `voice_fingerprint.threshold` to 0.70 or 0.75
- Re-enroll with more samples (modify `min_enrollment_samples` in config)
- Increase `vad_aggressiveness` to filter out more background

### Transcription is inaccurate
- Use a larger Whisper model (small.en or medium.en)
- Speak more clearly/loudly
- Reduce background noise if possible
- Ensure you're holding PTT for the entire phrase

### Hotkey not working
- Check that the hotkey combination isn't used by another app
- On Mac, you may need to grant Accessibility permissions
- Try a different key combination in config.yaml
