import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ui import normalize_audio_level_for_meter


class AudioLevelMeterTests(unittest.TestCase):
    def test_normalization_keeps_silence_quiet(self):
        self.assertEqual(normalize_audio_level_for_meter(0.0), 0.0)
        self.assertEqual(normalize_audio_level_for_meter(0.0001), 0.0)

    def test_normalization_makes_airpods_whisper_visible(self):
        whisper = normalize_audio_level_for_meter(0.0006)
        normal_speech = normalize_audio_level_for_meter(0.0127)

        self.assertGreater(whisper, 0.15)
        self.assertLess(whisper, normal_speech)
        self.assertGreater(normal_speech, 0.75)

    def test_normalization_clamps_loud_input(self):
        self.assertEqual(normalize_audio_level_for_meter(0.26), 1.0)


if __name__ == "__main__":
    unittest.main()
