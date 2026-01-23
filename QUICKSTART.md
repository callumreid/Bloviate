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
  hotkey: "<cmd>+<option>"  # Current default
  # Other options:
  # "<ctrl>+<shift>+<space>"
  # "<cmd>+<shift>+<space>"
```

### Adjust Voice Matching Sensitivity
```yaml
voice_fingerprint:
  threshold: 0.65  # Lower = more lenient (0.5-0.8 recommended)
```

If you're getting too many false rejections, lower the threshold to 0.6 or 0.55.

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
