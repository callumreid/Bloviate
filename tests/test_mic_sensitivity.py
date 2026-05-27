import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ui import mic_sensitivity_from_config, mic_sensitivity_settings


class MicSensitivityTests(unittest.TestCase):
    def test_high_sensitivity_lowers_speech_gate(self):
        normal = mic_sensitivity_settings(50)
        quiet = mic_sensitivity_settings(95)

        self.assertGreater(
            normal["noise_suppression.speech_min_rms"],
            quiet["noise_suppression.speech_min_rms"],
        )
        self.assertEqual(quiet["noise_suppression.vad_aggressiveness"], 0)
        self.assertEqual(quiet["noise_suppression.speech_min_frames"], 1)
        self.assertLessEqual(quiet["noise_suppression.speech_energy_fallback_rms"], 0.0001)

    def test_sensitivity_clamps_to_valid_range(self):
        self.assertEqual(mic_sensitivity_settings(200)["noise_suppression.mic_sensitivity"], 100)
        self.assertEqual(mic_sensitivity_settings(-10)["noise_suppression.mic_sensitivity"], 0)

    def test_infers_sensitivity_from_existing_config(self):
        self.assertGreaterEqual(
            mic_sensitivity_from_config({"speech_min_rms": 0.00005}),
            90,
        )
        self.assertEqual(mic_sensitivity_from_config({"mic_sensitivity": 85}), 85)


if __name__ == "__main__":
    unittest.main()
