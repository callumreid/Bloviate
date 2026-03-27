# Bloviate

A voice-fingerprinting dictation tool designed for whispering in noisy environments.

## Features

- **Voice Fingerprinting**: Only transcribes audio matching your voice signature
- **Noise Suppression**: Filters out background speech and noise
- **Hybrid Final Pass**: Uses low-latency streaming plus accuracy-first final transcription
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

Create your local personal dictionary file:

```bash
cp personal_dictionary.example.yaml personal_dictionary.yaml
```

`personal_dictionary.yaml` is gitignored on purpose, so each user keeps their own dictionary locally.

It handles both parts of the vocabulary problem:

- `preferred_terms`: bias the models toward exact spellings before transcription
- `corrections`: deterministically rewrite known bad outputs after transcription

You can also add preferred terms from the CLI without editing YAML:

```bash
python src/main.py --add-term "Raycast" --add-term "kubectl" --add-term "gpt-4o-transcribe"
python src/main.py --show-personal-dictionary
```

Bloviate still reads legacy `custom_dictionary.yaml` and `learned_terms.txt` files if you already have them, but new writes go to `personal_dictionary.yaml`.

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
- Final-pass provider order (`openai` / `deepgram` / `whisper`)
- Optional personal dictionary path via `transcription.personal_dictionary_path`
- Prompt tuning via `transcription.initial_prompt`, `prompt_max_terms`, and `prompt_max_chars`

You can also set `BLOVIATE_PERSONAL_DICTIONARY_PATH` if you want the dictionary outside the repo entirely.
Legacy `BLOVIATE_CUSTOM_DICTIONARY_PATH` and `BLOVIATE_LEARNED_TERMS_PATH` still work for migration.

## How It Works

1. **Audio Capture**: Captures audio from your Scarlett interface
2. **Noise Suppression**: Applies spectral subtraction and adaptive filtering
3. **Voice Activity Detection**: Detects speech segments
4. **Speaker Verification**: Compares against your enrolled voice profile
5. **Transcription**: Sends verified audio to speech-to-text engine using live Deepgram plus an accuracy-first final pass
6. **Vocabulary Biasing**: Feeds personal dictionary preferred terms and command phrases into prompting/keyterms before post-correction runs
7. **Correction Rules**: Applies personal dictionary corrections after transcription for known recurring mistakes
