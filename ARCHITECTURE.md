# Bloviate Architecture

Technical documentation for understanding how Bloviate works.

## System Overview

Bloviate is a voice-fingerprinting dictation system designed to solve the problem of whisper-based speech-to-text in noisy, multi-speaker environments. It uses speaker verification to ensure only the enrolled user's voice is transcribed, filtering out background speech.

## Audio Pipeline

```
Microphone Input (Scarlett 4i4)
    ↓
Audio Capture (sounddevice)
    ↓
[PTT Active?] → No → Discard
    ↓ Yes
Noise Suppression
    ↓
Voice Activity Detection
    ↓
Speaker Verification
    ↓
[Voice Match?] → No → Reject
    ↓ Yes
Speech-to-Text (Whisper)
    ↓
Output (Clipboard)
```

## Core Components

### 1. Audio Capture (`audio_capture.py`)

**Purpose**: Capture audio from the Scarlett 4i4 interface in real-time.

**Key Technologies**:
- `sounddevice`: Cross-platform audio I/O
- Sample rate: 16kHz (optimal for speech recognition)
- Chunk size: 1024 samples (~64ms latency)

**Design Decisions**:
- Uses callback-based streaming for low latency
- Queue-based buffering to prevent data loss
- Device auto-detection by name matching

**Parameters**:
- `sample_rate`: 16000 Hz (Whisper's native rate)
- `channels`: 1 (mono)
- `dtype`: float32 (normalized -1.0 to 1.0)

### 2. Noise Suppression (`noise_suppressor.py`)

**Purpose**: Remove background noise while preserving whispered speech.

**Techniques Used**:

1. **High-pass filtering** (80Hz cutoff)
   - Removes HVAC rumble, desk vibrations
   - Butterworth 4th order filter for clean rolloff

2. **Spectral noise reduction** (noisereduce library)
   - Stationary noise profiling
   - Spectral gating to remove noise floor
   - Adaptive to changing noise environments

3. **Voice Activity Detection** (WebRTC VAD)
   - Aggressiveness level: 3 (strictest)
   - Distinguishes speech from silence/noise
   - Used for gating and noise profile updates

**Parameters to Tune**:
- `stationary_noise_reduction`: 0-1 (higher = more aggressive)
- `vad_aggressiveness`: 0-3 (higher = stricter speech detection)
- `spectral_gate_threshold`: 0.01-0.1 (noise floor)

### 3. Voice Fingerprinting (`voice_fingerprint.py`)

**Purpose**: Distinguish the enrolled user's voice from background speakers.

**Approach**: Speaker verification via embedding comparison

**Model**: SpeechBrain ECAPA-TDNN trained on VoxCeleb
- Input: Raw audio waveform
- Output: 192-dimensional speaker embedding
- Pre-trained on 7000+ speakers

**Enrollment Process**:
1. Capture 5+ samples of user's whisper voice
2. Extract embeddings for each sample
3. Compute reference embedding (mean of all samples)
4. Save profile to disk

**Verification Process**:
1. Extract embedding from input audio
2. Compute cosine similarity with reference embedding
3. Normalize similarity to [0, 1] range
4. Accept if similarity ≥ threshold (default 0.65)

**Why This Works**:
- Speaker embeddings are invariant to:
  - Background noise (mostly)
  - Room acoustics
  - Speaking style variations
- But sensitive to:
  - Speaker identity (what we want!)
  - Microphone changes (re-enroll if you change mics)

**Tuning the Threshold**:
- Too high (0.8+): May reject your own voice
- Too low (0.5-): May accept background speakers
- Sweet spot: 0.6-0.7 for whisper in noisy environments

### 4. Push-to-Talk (`ptt_handler.py`)

**Purpose**: Global keyboard shortcut handling for recording control.

**Implementation**:
- Uses `pynput` for cross-platform keyboard hooks
- Monitors key press/release events globally
- Handles modifier key combinations (Ctrl, Shift, etc.)

**Key Matching Logic**:
- Normalizes left/right modifier keys (Ctrl_L → Ctrl)
- Checks if all hotkey components are currently pressed
- Activates on full match, deactivates when any key released

**Design Trade-off**:
- Global hooks require OS permissions
- Mac: Accessibility permissions needed
- Linux: May require X11 (won't work on Wayland)
- Windows: Usually works without extra setup

### 5. Transcription (`transcriber.py`)

**Purpose**: Convert verified audio to text using OpenAI Whisper.

**Model Options**:
- `tiny.en`: 39M params, ~1GB RAM, fastest
- `base.en`: 74M params, ~1GB RAM, good balance
- `small.en`: 244M params, ~2GB RAM, more accurate
- `medium.en`: 769M params, ~5GB RAM, best accuracy

**Processing**:
1. Ensure audio is float32, 16kHz, mono
2. Pad to minimum 0.5s (Whisper requirement)
3. Run inference with language hint
4. Filter very short outputs (<2 chars)
5. Copy to system clipboard

**Performance Notes**:
- CPU inference: 2-5s for base.en model
- GPU inference: 0.5-1s (if CUDA available)
- Whisper is deterministic (same audio → same output)

### 6. UI (`ui.py`)

**Purpose**: Real-time feedback for monitoring system state.

**Framework**: PyQt6 (lightweight, cross-platform)

**Displays**:
- **PTT Status**: Visual confirmation of recording state
- **Audio Level**: RMS meter to verify mic input
- **Voice Match**: Similarity score and accept/reject status
- **Last Transcription**: What was just transcribed
- **Status**: Current operation (listening, processing, etc.)

**Threading Model**:
- UI runs in main thread (Qt requirement)
- Audio/processing in background threads
- Uses Qt signals for thread-safe UI updates

## Data Flow Example

Let's trace what happens when you use Bloviate:

1. **Idle State**
   - Audio stream is running (capturing to queue)
   - PTT handler is listening for hotkey
   - UI showing "Ready"

2. **You press Ctrl+Shift+Space**
   - PTT handler detects hotkey match
   - Calls `on_ptt_press()` callback
   - Sets `is_recording = True`
   - UI updates to "ACTIVE"
   - Audio chunks start buffering in `recorded_audio[]`

3. **You whisper "Hello world"**
   - Audio callback receives ~15 chunks/second
   - Each chunk appended to buffer
   - UI audio level meter animates
   - Total recording: ~2 seconds

4. **You release the hotkey**
   - PTT handler detects key release
   - Calls `on_ptt_release()` callback
   - Sets `is_recording = False`
   - Triggers `process_recording()`

5. **Processing Pipeline**
   ```python
   audio = concatenate(recorded_audio)  # ~32000 samples

   # Noise suppression
   audio = highpass_filter(audio, 80Hz)
   audio = spectral_reduce(audio)

   # Speaker verification
   embedding = extract_embedding(audio)  # 192-dim vector
   similarity = cosine_sim(embedding, reference)  # 0.78

   if similarity >= 0.65:  # PASS
       text = whisper.transcribe(audio)  # "Hello world"
       clipboard.copy(text)
       ui.show(text)
   ```

6. **Result**
   - UI shows "✓ Matched (0.78)"
   - "Hello world" appears in transcription area
   - Clipboard contains "Hello world"
   - Ready to paste anywhere

## Performance Characteristics

**Latency Breakdown**:
- Audio buffering: 0ms (real-time)
- Noise suppression: ~50ms
- Speaker verification: ~100-200ms
- Whisper transcription: 2-5s (CPU), 0.5-1s (GPU)
- **Total**: ~3-5 seconds from release to text

**Memory Usage**:
- Base system: ~500MB
- Whisper base.en: ~1GB
- Speaker verification model: ~200MB
- **Total**: ~1.7GB

**CPU Usage**:
- Idle: <1%
- During capture: ~5%
- During transcription: 100% (single core burst)

## Security & Privacy

**Data Storage**:
- Voice profile: `models/voice_profile.pkl` (192-dim embeddings)
- No raw audio saved (unless debugging)
- No cloud uploads (100% local processing)

**Permissions Required**:
- Microphone access
- Keyboard monitoring (for global hotkey)
- Accessibility (macOS only)

**Privacy Properties**:
- Your voice embeddings never leave your computer
- Transcriptions only in clipboard (you control paste)
- No telemetry or analytics

## Extending Bloviate

Given your background in voice AI evals, here are some extension ideas:

### 1. Multi-Speaker Profiles
Modify `voice_fingerprint.py` to support multiple enrolled speakers and label transcriptions by speaker.

### 2. Continuous Monitoring Mode
Remove PTT requirement, use pure VAD + speaker verification for automatic transcription.

### 3. Custom STT Models
Replace Whisper with your own models - the interface is in `transcriber.py`.

### 4. Quality Metrics
Add confidence scores, WER estimation, or audio quality metrics to the UI.

### 5. Active Learning
Log rejected audio samples for analysis - are they false rejections or correct background filtering?

### 6. Noise Profile Adaptation
Build adaptive noise profiles specific to your office environment.

## Technical Limitations

**Current Constraints**:
1. **Speaker verification accuracy**: ~90-95% in practice
   - Can fail with very similar voices nearby
   - Degraded by extreme noise

2. **Whisper limitations**:
   - Struggles with very quiet whispers (<30dB)
   - May hallucinate on pure noise
   - English-focused (other languages less accurate)

3. **VAD sensitivity**:
   - May miss very quiet speech
   - May trigger on non-speech sounds (keyboard, coughs)

4. **Latency**:
   - 3-5s is not real-time
   - Not suitable for live captioning

**Design Trade-offs**:
- Accuracy vs. latency (chose accuracy)
- Simplicity vs. features (chose simplicity)
- Privacy vs. cloud accuracy (chose privacy)

## Debugging

**Enable verbose logging**:
```python
# In main.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Save problematic audio**:
```python
# In process_recording(), add:
import wave
with wave.open(f'debug_{time.time()}.wav', 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.writeframes((audio * 32767).astype(np.int16))
```

**Inspect embeddings**:
```python
# In voice_fingerprint.py, print similarity scores:
print(f"Similarity: {similarity:.4f}, Threshold: {self.threshold}")
```

## Future Improvements

Potential enhancements:
1. GPU acceleration for Whisper
2. Streaming transcription (partial results)
3. Punctuation model
4. Voice activity-based auto-segmentation
5. Web interface for remote access
6. Mobile app integration
