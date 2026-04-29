import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from history_store import HistoryStore
from model_registry import ModelRegistry
from personal_dictionary import load_personal_dictionary, save_personal_dictionary
from post_processor import PostProcessor
from secret_store import SecretStore
from settings_service import SettingsService, load_yaml_config


class FakeKeyring:
    def __init__(self):
        self.values = {}

    def get_password(self, service, username):
        return self.values.get((service, username))

    def set_password(self, service, username, password):
        self.values[(service, username)] = password

    def delete_password(self, service, username):
        self.values.pop((service, username), None)


class ProductionServiceTests(unittest.TestCase):
    def test_settings_service_saves_without_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            config = {
                "__config_path__": str(config_path),
                "__config_dir__": tempdir,
                "ptt": {"hotkey": "<cmd>+<option>"},
            }
            service = SettingsService(config)
            service.set("ptt.hotkey", "<ctrl>+<space>")

            saved = config_path.read_text(encoding="utf-8")
            self.assertIn("hotkey: <ctrl>+<space>", saved)
            self.assertNotIn("__config_path__", saved)

    def test_load_config_merges_defaults_and_migrates_old_dark_theme(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            config_path.write_text("ui:\n  theme: dark\nptt: {}\n", encoding="utf-8")

            config, _ = load_yaml_config(config_path)

            self.assertEqual(config["ui"]["theme"], "graphite")
            self.assertEqual(config["ui"]["waveform"]["preset"], "theme")
            self.assertEqual(config["ptt"]["hotkey"], "<cmd>+<option>")
            self.assertTrue(config["history"]["enabled"])
            self.assertTrue(config["achievements"]["enabled"])
            self.assertFalse(config["achievements"]["ai_analysis_enabled"])

    def test_history_store_insert_search_delete_export(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "history.sqlite"
            export_path = Path(tempdir) / "history.csv"
            store = HistoryStore(db_path)

            record_id = store.add_transcript(
                text="Hello Bloviate.",
                original_text="hello bloviate",
                mode="dictation",
                provider="openai",
                target_app="TextEdit",
                output_action="clipboard",
            )

            records = store.recent(query="bloviate")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].id, record_id)
            self.assertEqual(records[0].target_app, "TextEdit")

            store.export_csv(export_path)
            self.assertIn("Hello Bloviate.", export_path.read_text(encoding="utf-8"))

            self.assertTrue(store.delete(record_id))
            self.assertEqual(store.recent(), [])

    def test_history_store_insights_aggregate_usage_and_streaks(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "history.sqlite"
            store = HistoryStore(db_path)

            store.add_transcript(
                text="hello bloviate world",
                original_text="hello bloviate world",
                duration_s=3.0,
                target_app="TextEdit",
                created_at="2026-04-26T12:00:00+00:00",
            )
            store.add_transcript(
                text="cleaned message output",
                original_text="um cleaned message output",
                duration_s=6.0,
                target_app="Messages",
                created_at="2026-04-27T12:00:00+00:00",
            )
            store.add_transcript(
                text="another dictated note",
                original_text="another dictated note",
                duration_s=6.0,
                target_app="TextEdit",
                created_at="2026-04-28T12:00:00+00:00",
            )

            insights = store.insights(today=date(2026, 4, 28))

            self.assertEqual(insights["total_transcripts"], 3)
            self.assertEqual(insights["total_words"], 9)
            self.assertEqual(insights["changed_outputs"], 1)
            self.assertEqual(insights["current_streak_days"], 3)
            self.assertEqual(insights["longest_streak_days"], 3)
            self.assertEqual(insights["app_usage"][0]["name"], "TextEdit")
            self.assertEqual(insights["app_usage"][0]["words"], 6)

    def test_secret_store_prefers_config_then_keychain_then_env(self):
        fake = FakeKeyring()
        store = SecretStore()
        store._keyring = fake
        config = {"openai": {"api_key_env": "OPENAI_API_KEY"}}

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            self.assertEqual(store.get_api_key("openai", config), "env-key")

            ok, _ = store.set_api_key("openai", "keychain-key")
            self.assertTrue(ok)
            self.assertEqual(store.get_api_key("openai", config), "keychain-key")
            self.assertEqual(store.status("openai", config).redacted_value, "keyc...-key")

            config["openai"]["api_key"] = "config-key"
            self.assertEqual(store.get_api_key("openai", config), "config-key")

    def test_model_registry_normalizes_provider_priority(self):
        registry = ModelRegistry()
        self.assertEqual(
            registry.normalize_provider_priority(["OpenAI", "local", "openai", "unknown"]),
            ["openai", "whisper"],
        )

    def test_personal_dictionary_save_round_trips_normalized_payload(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = {
                "__config_dir__": tempdir,
                "transcription": {"personal_dictionary_path": "personal_dictionary.yaml"},
            }
            save_personal_dictionary(
                config,
                [" Raycast ", "raycast", "kubectl"],
                [
                    {
                        "phrase": "kubectl",
                        "variations": [" cube cuddle ", "cube cuddle"],
                        "match": "whole_word",
                    }
                ],
            )

            payload = load_personal_dictionary(config)
            self.assertEqual(payload["preferred_terms"], ["Raycast", "kubectl"])
            self.assertEqual(payload["corrections"][0]["variations"], ["cube cuddle"])
            self.assertEqual(payload["corrections"][0]["match"], "whole_word")

    def test_post_processor_deterministic_modes_do_not_require_api_key(self):
        processor = PostProcessor({"post_processing": {"mode": "clean", "openai_enabled": True}})
        with mock.patch.object(processor.secret_store, "get_api_key", return_value=None):
            result = processor.process("um hello world", mode="clean")

        self.assertEqual(result.text, "Hello world.")
        self.assertEqual(result.provider, "deterministic")
        self.assertTrue(result.changed)


if __name__ == "__main__":
    unittest.main()
