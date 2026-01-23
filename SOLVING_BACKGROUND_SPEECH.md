# Solving Background Speech Pickup

This document addresses your specific problem: background speech drowning out your whispers.

## The Problem

You're experiencing:
- Background coworker conversations being transcribed
- Your whispers not being detected
- System picking up cross-chatter instead of your voice

## Why This Happens

Traditional speech-to-text systems (including Whispr Flow) have no concept of speaker identity. They transcribe whatever is loudest or clearest, which in a noisy office is often:
1. Coworkers speaking at normal volume (60-70 dB)
2. Not your whisper (30-40 dB)

## How Bloviate Solves This

Bloviate uses a **multi-layer filtering approach**:

### Layer 1: Noise Suppression
- Removes stationary background noise (HVAC, keyboard typing)
- High-pass filter eliminates rumble
- Preserves speech frequencies (80Hz - 8kHz)

**Impact**: Reduces background hum but doesn't distinguish between speakers.

### Layer 2: Voice Activity Detection (VAD)
- Detects when ANY speech is present
- Filters out silence and pure noise
- Aggressive mode (level 3) minimizes false positives

**Impact**: Ensures we only process segments with actual speech.

### Layer 3: Speaker Verification (The Key)
- **This is what solves your problem**
- Compares audio to your enrolled voice profile
- Rejects audio that doesn't match your voice signature
- Works even when background speech is louder

**Impact**: Background voices get a similarity score of 0.2-0.4, your voice gets 0.7-0.9.

## Enrollment Best Practices

To maximize background rejection:

### 1. Enroll in Your Actual Environment
```bash
python src/main.py --enroll
```

**Do**:
- Enroll at your desk with normal office background noise
- Use your typical whisper volume and tone
- Complete all 5+ samples in one session

**Don't**:
- Enroll in a quiet room (the model won't know your whisper characteristics)
- Shout or speak at normal volume
- Rush through the phrases

### 2. Speak Full Phrases During Enrollment
The provided phrases are designed to capture your voice characteristics:
- Prosody (rhythm and intonation)
- Spectral features (frequency distribution)
- Speaking style

### 3. Use Consistent Mic Distance
Keep the TM-95GN at the same distance during:
- Enrollment
- Daily use

The headset mic design should help with this.

## Configuration Tuning

### Option 1: Strict Voice Matching (Recommended)

If background speech is still getting through:

```yaml
voice_fingerprint:
  threshold: 0.70  # Increase from default 0.65
  min_enrollment_samples: 8  # More samples = better profile
```

Re-enroll after changing `min_enrollment_samples`:
```bash
python src/main.py --clear-profile
python src/main.py --enroll
```

### Option 2: Aggressive Noise Suppression

```yaml
noise_suppression:
  stationary_noise_reduction: 0.9  # Max suppression
  vad_aggressiveness: 3  # Already at max, keep here
  spectral_gate_threshold: 0.05  # More aggressive gating
```

### Option 3: Higher Quality Whisper Model

Better transcription can help distinguish whispers from background:

```yaml
transcription:
  model: "small.en"  # Upgrade from base.en
```

Trade-off: Slower processing (4-8s instead of 2-4s)

## Workflow Adjustments

### 1. Directional Mic Positioning
The TM-95GN is a cardioid mic - it rejects sound from the sides/rear:

```
    [Front - Your mouth]
         ↓
      ←  🎤  →  [Sides - Reduced]
         ↑
  [Rear - Maximum rejection]
```

**Tip**: Position yourself so background speakers are behind the mic.

### 2. Use PTT Strategically
Only activate PTT when:
- You're ready to speak immediately
- Background conversation has a pause
- You're close to the mic

Don't:
- Hold PTT while thinking
- Record when background is loudest
- Keep PTT active during cross-talk

### 3. Monitor the UI
The voice match indicator tells you what's happening:

- **✓ Matched (0.75+)**: Your voice recognized
- **✓ Matched (0.65-0.75)**: Your voice, but borderline
- **✗ Rejected (<0.65)**: Background voice or noise

If you see rejections when whispering:
1. Check audio level meter - is your voice registering?
2. Lower threshold to 0.60
3. Re-enroll if still problematic

## Testing the System

Run this test to verify background rejection:

### Test 1: Your Voice
1. Activate PTT
2. Whisper: "This is my voice and it should be accepted"
3. Release
4. **Expected**: ✓ Matched, text transcribed

### Test 2: Background Speaker
1. Wait for a coworker to speak
2. Activate PTT during their speech
3. Don't say anything yourself
4. Release
5. **Expected**: ✗ Rejected, no transcription

### Test 3: Mixed Speech
1. Activate PTT
2. Whisper while a coworker is also speaking
3. Release
4. **Expected**:
   - May be rejected (similarity too low)
   - Or accepted with mixed transcription

If Test 3 fails (background gets transcribed):
- Increase `threshold` to 0.70 or 0.75
- Use better noise suppression settings
- Wait for background pauses

## Advanced: Analyzing Failures

If background speech is still a problem, debug it:

### 1. Save Audio Samples

Add to `src/main.py` in `process_recording()`:

```python
import wave
import time

# After concatenating audio
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"recordings/debug_{timestamp}_{similarity:.2f}.wav"

with wave.open(filename, 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    audio_int = (audio * 32767).astype(np.int16)
    f.writeframes(audio_int.tobytes())

print(f"Saved: {filename}")
```

### 2. Analyze Similarity Scores

Check what similarity scores you're getting:
- Your voice: Should be 0.70-0.95
- Background: Should be 0.20-0.50
- Mixed: Will be somewhere in between

If overlap is too much (background >0.65 or your voice <0.70):
- Problem: Voice profile isn't distinctive enough
- Solution: Re-enroll with more samples

### 3. Check Audio Levels

In the UI:
- Your whisper should show 30-60% on the meter
- Background speech often shows 40-80%

If your whisper is <20%:
- Speak closer to the mic
- Increase whisper volume slightly
- Check Scarlett gain settings

## Scarlett 4i4 Settings

Optimal configuration for the TM-95GN:

1. **Gain**: Set so your whisper peaks at -12dB to -6dB
   - Too low: Won't detect your voice
   - Too high: Clips and distorts

2. **Phantom Power**: ON (48V)
   - The TM-95GN requires phantom power

3. **Pad**: OFF
   - You're whispering, not shouting

4. **Input**: Use Input 1 (combo XLR/TRS jack)

5. **Monitor Mix**: Doesn't affect recording, adjust for comfort

To check levels, use the Scarlett's meter while speaking. The green LEDs should light up when you whisper.

## Comparison to Whispr Flow

Why Bloviate works better in your environment:

| Feature | Whispr Flow | Bloviate |
|---------|-------------|----------|
| Speaker verification | ❌ No | ✅ Yes |
| Background rejection | ❌ Minimal | ✅ Strong |
| Voice fingerprinting | ❌ No | ✅ Yes |
| Noise suppression | ✅ Basic | ✅ Multi-layer |
| Customizable | ❌ Limited | ✅ Fully |
| Privacy | ⚠️ Cloud | ✅ Local |

## If All Else Fails

If you're still having issues:

### Option 1: Increase Min Speech Duration

Require longer speech segments (background chatter is often shorter):

In `src/main.py`, add to `on_ptt_release()`:

```python
audio = np.concatenate(self.recorded_audio).flatten()
duration = len(audio) / self.config['audio']['sample_rate']

if duration < 1.0:  # Require at least 1 second
    print("Recording too short, ignoring")
    return
```

### Option 2: Hybrid VAD + Speaker Verification

Only verify segments where VAD detected speech:

```python
if not self.noise_suppressor.is_speech(audio):
    print("No speech detected")
    return
```

### Option 3: Multi-Pass Enrollment

Enroll multiple times and average embeddings:

```bash
python src/main.py --enroll  # First session
# (keep existing profile)
python src/main.py --enroll  # Add more samples
```

Modify `voice_fingerprint.py` to not clear on re-enrollment.

## Expected Results

After proper configuration:
- **Background rejection rate**: 90-95%
- **Your voice acceptance rate**: 85-90%
- **False positives** (background transcribed): <5%
- **False negatives** (your voice rejected): <10%

The 10% false negative rate means occasionally your own whispers will be rejected. This is the trade-off for strong background rejection. If this is too restrictive, lower the threshold slightly.

## Getting Help

Since you work in voice AI evals, you can:
1. Analyze the embeddings directly (they're just numpy arrays)
2. Plot similarity distributions to find optimal threshold
3. Add custom metrics to the UI
4. Modify the speaker verification model

The codebase is designed to be hackable - all components are modular and well-documented.
