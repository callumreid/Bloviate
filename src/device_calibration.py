"""
Per-device audio calibration for Bloviate.

Each microphone has a different noise floor and speech level; fixed global
gates tuned for one rig (the Scarlett gooseneck) are exactly what broke quiet
mics. This store learns floor/speech RMS per device from real dictations and
derives speech gates from them.
"""

import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def _frame_rms_values(audio: np.ndarray, sample_rate: int, frame_ms: int = 20) -> np.ndarray:
    """Per-frame RMS values for percentile-based floor/speech estimation."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    frame = int(sample_rate * frame_ms / 1000)
    if frame <= 0 or len(audio) < frame:
        return np.array([], dtype=np.float32)
    usable = len(audio) - (len(audio) % frame)
    frames = audio[:usable].reshape(-1, frame)
    return np.sqrt(np.mean(frames**2, axis=1))


class DeviceCalibrationStore:
    """Learns and persists per-device noise floor / speech RMS estimates."""

    MIN_CLIPS = 3
    EMA_ALPHA = 0.3

    def __init__(self, path: Path, sample_rate: int):
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self._lock = threading.Lock()
        self._profiles: dict = {}
        self._load()

    def _load(self):
        try:
            if self.path.is_file():
                data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    self._profiles = {
                        str(k): v for k, v in data.items() if isinstance(v, dict)
                    }
        except Exception as exc:
            print(f"[DeviceProfile] Could not load {self.path}: {exc}")
            self._profiles = {}

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(
                yaml.safe_dump(self._profiles, sort_keys=True), encoding="utf-8"
            )
            os.replace(tmp, self.path)
        except Exception as exc:
            print(f"[DeviceProfile] Could not save {self.path}: {exc}")

    def update_from_clip(self, device: str, audio: np.ndarray) -> Optional[dict]:
        """Fold one dictation clip into the device's floor/speech estimates."""
        device = str(device or "").strip()
        if not device:
            return None
        frame_rms = _frame_rms_values(audio, self.sample_rate)
        if frame_rms.size < 10:
            return None

        floor = float(np.percentile(frame_rms, 10))
        speech = float(np.percentile(frame_rms, 90))
        if speech <= 0 or speech <= floor * 1.5:
            # Clip has no usable dynamic range (likely silence); don't learn from it.
            return None

        with self._lock:
            profile = self._profiles.setdefault(
                device,
                {"noise_floor_rms": floor, "speech_rms": speech, "clips": 0},
            )
            alpha = self.EMA_ALPHA
            profile["noise_floor_rms"] = float(
                (1 - alpha) * float(profile.get("noise_floor_rms", floor)) + alpha * floor
            )
            profile["speech_rms"] = float(
                (1 - alpha) * float(profile.get("speech_rms", speech)) + alpha * speech
            )
            profile["clips"] = int(profile.get("clips", 0)) + 1
            profile["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._save()
            return dict(profile)

    def get(self, device: str) -> Optional[dict]:
        with self._lock:
            profile = self._profiles.get(str(device or "").strip())
            return dict(profile) if profile else None

    def gates(self, device: str, mic_sensitivity: float = 50.0) -> Optional[tuple[float, float]]:
        """Derive (speech_min_rms, energy_fallback_rms) for a calibrated device.

        Returns None until enough clips are learned, so config gates stay in
        charge for new devices. Sensitivity slider scales the gate: 50 -> 1x,
        100 -> 0.25x (more permissive), 0 -> 4x (stricter).
        """
        profile = self.get(device)
        if not profile or int(profile.get("clips", 0)) < self.MIN_CLIPS:
            return None
        floor = float(profile.get("noise_floor_rms", 0.0))
        speech = float(profile.get("speech_rms", 0.0))
        if speech <= 0:
            return None

        base_gate = max(floor * 3.0, speech * 0.08, 1e-5)
        base_gate = min(base_gate, speech * 0.5)
        try:
            sensitivity = float(mic_sensitivity)
        except (TypeError, ValueError):
            sensitivity = 50.0
        sensitivity = min(100.0, max(0.0, sensitivity))
        scale = 4.0 ** ((50.0 - sensitivity) / 50.0)
        gate = base_gate * scale
        return gate, gate * 0.6
