import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber import Transcriber


class TranscriberPromptTermsTests(unittest.TestCase):
    def test_prompt_terms_ignore_correction_phrases(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.transcription_config = {"prompt_terms": ["OpenAI"]}
        transcriber.deepgram_config = {"keyterm": ["Claude"]}
        transcriber.learned_terms = ["Bloviate"]
        transcriber.custom_dictionary = [
            {
                "phrase": "Does the linter pass?",
                "variations": ["does the winter pass"],
            },
            {
                "phrase": "git push",
                "variations": ["get push"],
            },
        ]

        terms = Transcriber._build_prompt_terms(transcriber)

        self.assertEqual(terms, ["OpenAI", "Claude", "Bloviate"])

    def test_deepgram_bias_terms_ignore_correction_phrases(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.deepgram_config = {
            "keyterm": ["Claude"],
            "include_dictionary_keyterms": True,
        }
        transcriber.learned_terms = ["Bloviate"]
        transcriber.custom_dictionary = [
            {
                "phrase": "Does the linter pass?",
                "variations": ["does the winter pass"],
            }
        ]
        transcriber._deepgram_max_keyterms = 20

        terms = Transcriber._build_deepgram_bias_terms(transcriber)

        self.assertEqual(terms, ["Claude", "Bloviate"])


if __name__ == "__main__":
    unittest.main()
