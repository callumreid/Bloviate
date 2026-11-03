import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber import Transcriber


class DegenerateRepetitionTests(unittest.TestCase):
    def test_fused_unit_repeated_hundreds_of_times_returns_empty(self):
        text = "ies" * 223  # 669 chars, observed fine-tuned-whisper hallucination

        self.assertEqual(Transcriber._collapse_degenerate_repetition(text), "")

    def test_trailing_repeated_unit_is_collapsed(self):
        text = "Hello world " + "ies " * 6 + "ies"

        result = Transcriber._collapse_degenerate_repetition(text)

        self.assertTrue(result.startswith("Hello world"))
        self.assertLessEqual(result.count("ies"), 1)

    def test_short_repeated_words_are_left_alone(self):
        self.assertEqual(
            Transcriber._collapse_degenerate_repetition("ha ha ha"), "ha ha ha"
        )

    def test_pure_digits_are_not_collapsed(self):
        self.assertEqual(
            Transcriber._collapse_degenerate_repetition("1000000"), "1000000"
        )

    def test_ellipsis_is_unchanged(self):
        self.assertEqual(
            Transcriber._collapse_degenerate_repetition("Wait..."), "Wait..."
        )

    def test_normal_text_is_unchanged(self):
        text = "The quick brown fox jumps over the lazy dog."
        self.assertEqual(Transcriber._collapse_degenerate_repetition(text), text)


if __name__ == "__main__":
    unittest.main()
