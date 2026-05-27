import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber import Transcriber


class TranscriberAudioGainTests(unittest.TestCase):
    def test_openai_normalization_boosts_quiet_audio(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.openai_config = {
            "normalize_audio": True,
            "target_rms": 0.05,
            "noise_floor_rms": 0.0000005,
            "max_gain_db": 48.0,
            "min_gain_db": -8.0,
            "peak_ceiling": 0.95,
        }

        audio = np.full(16000, 0.0005, dtype=np.float32)
        boosted = Transcriber._normalize_openai_audio(transcriber, audio)

        self.assertGreater(Transcriber._audio_rms(boosted), 0.04)
        self.assertLessEqual(float(np.max(np.abs(boosted))), 0.95)

    def test_openai_normalization_leaves_near_silence_alone(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.openai_config = {
            "normalize_audio": True,
            "target_rms": 0.05,
            "noise_floor_rms": 0.0000005,
        }

        audio = np.full(16000, 0.0000001, dtype=np.float32)
        boosted = Transcriber._normalize_openai_audio(transcriber, audio)

        np.testing.assert_array_equal(boosted, audio)

    def test_openai_skips_near_silent_clips(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.openai_config = {
            "model": "gpt-4o-transcribe",
            "min_transcribe_rms": 0.00008,
        }
        transcriber.language = "en"
        transcriber._get_openai_api_key = lambda: "test-key"

        result = Transcriber._transcribe_openai(
            transcriber,
            np.full(16000, 0.00003, dtype=np.float32),
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
