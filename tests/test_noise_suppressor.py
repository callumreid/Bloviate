import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from noise_suppressor import NoiseSuppressor


class NoiseSuppressorSpeechGateTests(unittest.TestCase):
    def test_has_speech_rejects_low_rms(self):
        suppressor = NoiseSuppressor.__new__(NoiseSuppressor)
        suppressor.config = {
            "noise_suppression": {
                "speech_min_rms": 0.003,
                "speech_min_frames": 3,
                "speech_min_ratio": 0.12,
            }
        }

        suppressor.speech_stats = lambda audio: {
            "rms": 0.001,
            "frames": 10,
            "speech_frames": 10,
            "speech_ratio": 1.0,
        }

        self.assertFalse(NoiseSuppressor.has_speech(suppressor, []))

    def test_has_speech_accepts_enough_speech_frames(self):
        suppressor = NoiseSuppressor.__new__(NoiseSuppressor)
        suppressor.config = {
            "noise_suppression": {
                "speech_min_rms": 0.003,
                "speech_min_frames": 3,
                "speech_min_ratio": 0.12,
            }
        }

        suppressor.speech_stats = lambda audio: {
            "rms": 0.02,
            "frames": 10,
            "speech_frames": 3,
            "speech_ratio": 0.3,
        }

        self.assertTrue(NoiseSuppressor.has_speech(suppressor, []))


if __name__ == "__main__":
    unittest.main()
