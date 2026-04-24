import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber import Transcriber


class TranscriberPromptTermsTests(unittest.TestCase):
    def test_prompt_terms_ignore_correction_phrases_and_provider_keyterms(self):
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

        self.assertEqual(terms, ["OpenAI", "Bloviate"])

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

    def test_reload_personal_dictionary_rebuilds_prompt_assets(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.config = {}
        transcriber.use_custom_dictionary = True
        transcriber.verbose_logs = False
        transcriber.provider = "openai"
        transcriber._auto_prompt_cache = {"dictation": "stale"}
        transcriber._build_prompt_terms = lambda: ["Acme"]
        transcriber._build_command_prompt_terms = lambda: ["window left"]
        transcriber._build_deepgram_bias_terms = lambda: ["Acme"]
        transcriber._build_deepgram_command_terms = lambda: ["window"]

        with mock.patch(
            "transcriber.load_personal_dictionary",
            return_value={
                "preferred_terms": ["Acme"],
                "corrections": [{"phrase": "Acme", "variations": ["ac me"], "match": "substring"}],
                "sources": ["/tmp/personal_dictionary.yaml"],
                "path": "/tmp/personal_dictionary.yaml",
            },
        ):
            stats = Transcriber.reload_personal_dictionary(transcriber)

        self.assertEqual(transcriber.learned_terms, ["Acme"])
        self.assertEqual(len(transcriber.custom_dictionary), 1)
        self.assertEqual(transcriber.personal_dictionary_sources, ["/tmp/personal_dictionary.yaml"])
        self.assertEqual(transcriber._prompt_terms, ["Acme"])
        self.assertEqual(transcriber._command_prompt_terms, ["window left"])
        self.assertEqual(transcriber._auto_prompt_cache, {})
        self.assertEqual(stats["preferred_terms"], 1)
        self.assertEqual(stats["corrections"], 1)


if __name__ == "__main__":
    unittest.main()
