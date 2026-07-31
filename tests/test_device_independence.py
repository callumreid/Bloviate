import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio_capture import AudioCapture
from device_calibration import DeviceCalibrationStore
from noise_suppressor import NoiseSuppressor
from voice_fingerprint import VoiceFingerprint


SAMPLE_RATE = 16000


def _speech_like_clip(speech_rms: float, floor_rms: float, seconds: float = 2.0) -> np.ndarray:
    """Alternating loud/quiet 20ms frames so percentiles separate cleanly."""
    frame = int(SAMPLE_RATE * 0.02)
    frames = []
    total = int(seconds * 50)
    rng = np.random.default_rng(42)
    for i in range(total):
        level = speech_rms if i % 2 else floor_rms
        frames.append(rng.standard_normal(frame).astype(np.float32) * level)
    return np.concatenate(frames)


class DeviceCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = DeviceCalibrationStore(
            Path(self.tmp.name) / "device_profiles.yaml", SAMPLE_RATE
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_gates_require_min_clips(self):
        clip = _speech_like_clip(0.05, 0.001)
        self.assertIsNone(self.store.gates("Test Mic"))
        self.store.update_from_clip("Test Mic", clip)
        self.store.update_from_clip("Test Mic", clip)
        self.assertIsNone(self.store.gates("Test Mic"))
        self.store.update_from_clip("Test Mic", clip)
        self.assertIsNotNone(self.store.gates("Test Mic"))

    def test_gates_sit_between_floor_and_speech(self):
        clip = _speech_like_clip(0.05, 0.001)
        for _ in range(3):
            self.store.update_from_clip("Test Mic", clip)
        gate, fallback = self.store.gates("Test Mic")
        profile = self.store.get("Test Mic")
        self.assertGreater(gate, profile["noise_floor_rms"])
        self.assertLess(gate, profile["speech_rms"])
        self.assertLess(fallback, gate)

    def test_quiet_mic_learns_lower_gate_than_loud_mic(self):
        loud = _speech_like_clip(0.05, 0.001)
        quiet = _speech_like_clip(0.002, 0.00004)
        for _ in range(3):
            self.store.update_from_clip("Loud Mic", loud)
            self.store.update_from_clip("Quiet Mic", quiet)
        loud_gate, _ = self.store.gates("Loud Mic")
        quiet_gate, _ = self.store.gates("Quiet Mic")
        self.assertLess(quiet_gate, loud_gate)

    def test_silent_clips_are_not_learned(self):
        silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self.assertIsNone(self.store.update_from_clip("Test Mic", silent))
        self.assertIsNone(self.store.get("Test Mic"))

    def test_sensitivity_scales_gate(self):
        clip = _speech_like_clip(0.05, 0.001)
        for _ in range(3):
            self.store.update_from_clip("Test Mic", clip)
        default_gate, _ = self.store.gates("Test Mic", mic_sensitivity=50)
        permissive_gate, _ = self.store.gates("Test Mic", mic_sensitivity=100)
        strict_gate, _ = self.store.gates("Test Mic", mic_sensitivity=0)
        self.assertLess(permissive_gate, default_gate)
        self.assertGreater(strict_gate, default_gate)

    def test_profiles_persist_across_instances(self):
        clip = _speech_like_clip(0.05, 0.001)
        for _ in range(3):
            self.store.update_from_clip("Test Mic", clip)
        reloaded = DeviceCalibrationStore(
            Path(self.tmp.name) / "device_profiles.yaml", SAMPLE_RATE
        )
        self.assertIsNotNone(reloaded.gates("Test Mic"))


class DynamicGateTests(unittest.TestCase):
    def _suppressor(self):
        config = {
            "audio": {"sample_rate": SAMPLE_RATE},
            "noise_suppression": {
                "enabled": True,
                "stationary_noise_reduction": 0.5,
                "spectral_gate_threshold": 0.03,
                "vad_aggressiveness": 1,
                "speech_min_rms": 0.01,
                "speech_min_frames": 1,
                "speech_min_ratio": 0.01,
                "speech_energy_fallback_rms": 0.01,
                "mic_sensitivity": 50,
            },
        }
        return NoiseSuppressor(config)

    def test_dynamic_gates_override_config_gates(self):
        suppressor = self._suppressor()
        rng = np.random.default_rng(7)
        quiet_speech = rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.004

        with mock.patch.object(suppressor.vad, "is_speech", return_value=True):
            self.assertFalse(suppressor.has_speech(quiet_speech))
            suppressor.set_dynamic_gates(0.001, 0.0006)
            self.assertTrue(suppressor.has_speech(quiet_speech))
            suppressor.clear_dynamic_gates()
            self.assertFalse(suppressor.has_speech(quiet_speech))


class PrerollBufferTests(unittest.TestCase):
    def _capture(self):
        config = {
            "app": {},
            "audio": {
                "sample_rate": SAMPLE_RATE,
                "chunk_size": 1024,
                "channels": 1,
                "device_name": "",
                "queue_max_chunks": 8,
                "preroll_seconds": 1.0,
            },
        }
        with mock.patch("audio_capture.AudioCapture._find_device", return_value=None):
            return AudioCapture(config)

    def test_preroll_returns_recent_chunks(self):
        capture = self._capture()
        for value in range(30):
            chunk = np.full((1024, 1), float(value), dtype=np.float32)
            with capture._preroll_lock:
                capture._preroll.append(chunk)

        recent = capture.get_preroll(0.2)  # ~3 chunks at 64ms each
        self.assertEqual(len(recent), 3)
        self.assertEqual(float(recent[-1][0][0]), 29.0)

    def test_preroll_zero_seconds_is_empty(self):
        capture = self._capture()
        with capture._preroll_lock:
            capture._preroll.append(np.zeros((1024, 1), dtype=np.float32))
        self.assertEqual(capture.get_preroll(0.0), [])


class PerDeviceVoiceProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = {
            "app": {},
            "audio": {"sample_rate": SAMPLE_RATE},
            "voice_fingerprint": {
                "enabled": True,
                "threshold": 0.6,
                "embedding_model": "unused",
                "min_enrollment_samples": 2,
                "load_model_on_startup": False,
                "model_dir": self.tmp.name,
            },
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _fingerprint(self):
        with mock.patch(
            "voice_fingerprint.legacy_repo_voice_profile_path",
            return_value=Path(self.tmp.name) / "no_legacy" / "voice_profile.pkl",
        ):
            return VoiceFingerprint(self.config)

    def test_enroll_targets_active_device(self):
        fp = self._fingerprint()
        fp.set_active_device("MacBook Pro Microphone")
        embedding = np.ones(8, dtype=np.float32)
        with mock.patch.object(fp, "extract_embedding", return_value=embedding):
            fp.enroll_sample(np.zeros(SAMPLE_RATE, dtype=np.float32))
            fp.enroll_sample(np.zeros(SAMPLE_RATE, dtype=np.float32))
        summary = fp.profile_summary()
        self.assertTrue(summary["MacBook Pro Microphone"]["enrolled"])
        self.assertTrue(fp.is_enrolled())

    def test_verify_prefers_device_profile_over_default(self):
        fp = self._fingerprint()
        device_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        default_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        fp._profiles = {
            "MacBook Pro Microphone": {"embeddings": [device_vec], "reference": device_vec},
            fp.DEFAULT_PROFILE: {"embeddings": [default_vec], "reference": default_vec},
        }
        fp.set_active_device("MacBook Pro Microphone")
        with mock.patch.object(fp, "extract_embedding", return_value=device_vec):
            is_match, similarity = fp.verify_speaker(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.assertTrue(is_match)
        self.assertGreater(similarity, 0.99)

    def test_verify_falls_back_to_default_profile(self):
        fp = self._fingerprint()
        default_vec = np.array([1.0, 0.0], dtype=np.float32)
        fp._profiles = {
            fp.DEFAULT_PROFILE: {"embeddings": [default_vec], "reference": default_vec}
        }
        fp.set_active_device("Brand New Mic")
        with mock.patch.object(fp, "extract_embedding", return_value=default_vec):
            is_match, _ = fp.verify_speaker(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.assertTrue(is_match)

    def test_save_and_load_roundtrip_multi_device(self):
        fp = self._fingerprint()
        vec = np.ones(4, dtype=np.float32)
        fp.set_active_device("Mic A")
        with mock.patch.object(fp, "extract_embedding", return_value=vec):
            fp.enroll_sample(np.zeros(SAMPLE_RATE, dtype=np.float32))
        fp.save_profile()

        reloaded = self._fingerprint()
        self.assertIn("Mic A", reloaded.profile_summary())
        self.assertTrue(reloaded.multi_profile_path.exists())
        self.assertTrue(reloaded.profile_path.exists())  # legacy mirror

    def test_legacy_single_profile_migrates_to_default_slot(self):
        import pickle

        vec = np.ones(4, dtype=np.float32)
        legacy_path = Path(self.tmp.name) / "voice_profile.pkl"
        with open(legacy_path, "wb") as f:
            pickle.dump({"embeddings": [vec, vec], "reference": vec, "threshold": 0.6}, f)

        fp = self._fingerprint()
        summary = fp.profile_summary()
        self.assertIn(fp.DEFAULT_PROFILE, summary)
        self.assertTrue(fp.is_enrolled())

    def test_clear_profile_removes_everything(self):
        fp = self._fingerprint()
        vec = np.ones(4, dtype=np.float32)
        with mock.patch.object(fp, "extract_embedding", return_value=vec):
            fp.enroll_sample(np.zeros(SAMPLE_RATE, dtype=np.float32))
        fp.save_profile()
        fp.clear_profile()
        self.assertEqual(fp.profile_summary(), {})
        self.assertFalse(fp.multi_profile_path.exists())
        self.assertFalse(fp.profile_path.exists())


if __name__ == "__main__":
    unittest.main()
