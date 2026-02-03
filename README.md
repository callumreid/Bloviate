# Bloviate

A voice-fingerprinting dictation tool designed for whispering in noisy environments.

## Features

- **Voice Fingerprinting**: Only transcribes audio matching your voice signature
- **Noise Suppression**: Filters out background speech and noise
- **Push-to-Talk**: Global keyboard shortcut to activate listening
- **Real-time Feedback**: Visual display of audio levels and voice detection
- **Scarlett Integration**: Optimized for Scarlett 4i4 audio interface

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Enroll Your Voice
First, record samples of your whisper voice for fingerprinting:
```bash
python src/main.py --enroll
```
This will prompt you to whisper several phrases to build your voice profile.

### 2. Run the Dictation Tool
```bash
python src/main.py
```

- Press and hold `Cmd+Option` (configurable) to activate listening
- Whisper into your mic
- Release when done
- Only audio matching your voice will be transcribed
- Use talk mode to bypass voice matching:
  - `python src/main.py --voice-mode talk`
  - Or set `voice_fingerprint.mode: "talk"` in `config.yaml`

### 3. Configuration
Edit `config.yaml` to customize:
- Keyboard shortcuts
- Audio device selection
- Voice matching threshold
- Noise suppression levels

## How It Works

1. **Audio Capture**: Captures audio from your Scarlett interface
2. **Noise Suppression**: Applies spectral subtraction and adaptive filtering
3. **Voice Activity Detection**: Detects speech segments
4. **Speaker Verification**: Compares against your enrolled voice profile
5. **Transcription**: Sends verified audio to speech-to-text engine
