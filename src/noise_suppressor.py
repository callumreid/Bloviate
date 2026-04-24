"""
Noise suppression module for Bloviate.
Filters out background speech and stationary noise.
"""

import numpy as np
import noisereduce as nr
from scipy import signal
import webrtcvad
from typing import Optional


class NoiseSuppressor:
    """Applies noise reduction to audio signals."""

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config['audio']['sample_rate']
        self.enabled = config['noise_suppression']['enabled']
        self.stationary_reduction = config['noise_suppression']['stationary_noise_reduction']
        self.spectral_gate = config['noise_suppression']['spectral_gate_threshold']
        self.vad_aggressiveness = config['noise_suppression']['vad_aggressiveness']

        # WebRTC VAD for voice activity detection
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)

        # For adaptive noise profiling
        self.noise_profile: Optional[np.ndarray] = None
        self.noise_profile_samples = []
        self.max_noise_samples = 10

    def suppress(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise suppression to audio."""
        if not self.enabled:
            return audio

        # Ensure audio is in the right format
        audio = audio.squeeze()

        # Apply spectral gating with noisereduce
        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=self.stationary_reduction,
                thresh_n_mult_nonstationary=2,
            )
        except Exception as e:
            print(f"Noise reduction error: {e}")
            reduced = audio

        return reduced

    def is_speech(self, audio: np.ndarray, aggressive: bool = True) -> bool:
        """
        Detect if audio contains speech using WebRTC VAD.

        Args:
            audio: Audio data (must be 10, 20, or 30ms at 8000, 16000, or 32000 Hz)
            aggressive: If True, uses the configured aggressiveness level

        Returns:
            True if speech is detected
        """
        # WebRTC VAD requires specific frame sizes (10, 20, or 30 ms)
        # at specific sample rates (8000, 16000, or 32000 Hz)

        # Convert to 16-bit PCM
        audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)

        # Calculate appropriate frame size (20ms)
        frame_duration_ms = 20
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)

        # Pad or truncate to frame size
        if len(audio_int16) < frame_size:
            audio_int16 = np.pad(audio_int16, (0, frame_size - len(audio_int16)))
        else:
            audio_int16 = audio_int16[:frame_size]

        try:
            return self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
        except Exception as e:
            # Fallback to energy-based detection
            energy = np.sqrt(np.mean(audio ** 2))
            return energy > 0.01

    def speech_stats(self, audio: np.ndarray) -> dict:
        """Estimate whether a longer clip contains speech-like frames."""
        audio = audio.squeeze()
        if len(audio) == 0:
            return {
                "rms": 0.0,
                "frames": 0,
                "speech_frames": 0,
                "speech_ratio": 0.0,
            }

        rms = float(np.sqrt(np.mean(audio ** 2)))
        frame_duration_ms = 20
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        if frame_size <= 0:
            return {
                "rms": rms,
                "frames": 0,
                "speech_frames": 0,
                "speech_ratio": 0.0,
            }

        audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
        total_frames = 0
        speech_frames = 0

        for start in range(0, len(audio_int16) - frame_size + 1, frame_size):
            frame = audio_int16[start:start + frame_size]
            total_frames += 1
            try:
                if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                    speech_frames += 1
            except Exception:
                frame_audio = frame.astype(np.float32) / 32768.0
                frame_rms = float(np.sqrt(np.mean(frame_audio ** 2)))
                if frame_rms > 0.01:
                    speech_frames += 1

        speech_ratio = (speech_frames / total_frames) if total_frames else 0.0
        return {
            "rms": rms,
            "frames": total_frames,
            "speech_frames": speech_frames,
            "speech_ratio": speech_ratio,
        }

    def has_speech(self, audio: np.ndarray) -> bool:
        """Return True when a clip contains enough speech-like energy to transcribe."""
        stats = self.speech_stats(audio)
        min_rms = float(self.config['noise_suppression'].get('speech_min_rms', 0.003))
        min_speech_frames = int(self.config['noise_suppression'].get('speech_min_frames', 3))
        min_speech_ratio = float(self.config['noise_suppression'].get('speech_min_ratio', 0.12))

        if stats["rms"] < min_rms:
            return False
        if stats["speech_frames"] >= min_speech_frames:
            return True
        return stats["speech_ratio"] >= min_speech_ratio

    def update_noise_profile(self, audio: np.ndarray):
        """Update the noise profile with background audio."""
        if not self.is_speech(audio):
            self.noise_profile_samples.append(audio)
            if len(self.noise_profile_samples) > self.max_noise_samples:
                self.noise_profile_samples.pop(0)

            if len(self.noise_profile_samples) > 0:
                self.noise_profile = np.mean(
                    np.array(self.noise_profile_samples), axis=0
                )

    def apply_highpass_filter(self, audio: np.ndarray, cutoff: int = 80) -> np.ndarray:
        """
        Apply a high-pass filter to remove low-frequency noise.
        Useful for removing HVAC, rumble, etc.
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist

        # Design a 4th order Butterworth high-pass filter
        b, a = signal.butter(4, normalized_cutoff, btype='high')

        # Apply filter
        filtered = signal.filtfilt(b, a, audio)

        return filtered

    def process(self, audio: np.ndarray, apply_highpass: bool = True) -> np.ndarray:
        """
        Full processing pipeline: high-pass filter + noise reduction.
        """
        if not self.enabled:
            return audio

        processed = audio.copy()

        # Apply high-pass filter to remove low-frequency noise
        if apply_highpass:
            processed = self.apply_highpass_filter(processed)

        # Apply spectral noise reduction
        processed = self.suppress(processed)

        return processed
