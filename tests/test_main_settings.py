import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import main


class MainSettingsTests(unittest.TestCase):
    def _make_app(self, config_path: Path) -> main.Bloviate:
        app = main.Bloviate.__new__(main.Bloviate)
        app.verbose_logs = False
        app.ui_window = None
        app.is_recording = False
        app.is_command_recording = False
        app.config = {
            "__config_path__": str(config_path),
            "__config_dir__": str(config_path.parent),
            "audio": {"sample_rate": 16000},
            "app": {"startup_animation": False},
            "ui": {"show_main_window": True, "startup_splash": {"enabled": True}},
            "voice_fingerprint": {"mode": "whisper", "threshold": 0.6},
        }
        app.voice_mode = "whisper"
        app.talk_mode = False
        app.voice_fingerprint = SimpleNamespace(
            threshold=0.6,
            enrolled_embeddings=[],
            min_enrollment_samples=8,
            profile_path=config_path.parent / "voice_profile.pkl",
            is_enrolled=lambda: True,
            save_profile=lambda: None,
            clear_profile=lambda: app.voice_fingerprint.enrolled_embeddings.clear(),
        )
        app.transcriber = SimpleNamespace(
            reload_personal_dictionary=lambda: {"preferred_terms": 2, "corrections": 3}
        )
        return app

    def test_set_voice_threshold_clamps_and_persists(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            app = self._make_app(config_path)

            ok, message = app.set_voice_threshold(2.5)

            self.assertTrue(ok, message)
            self.assertEqual(app.voice_fingerprint.threshold, 1.0)
            self.assertEqual(app.config["voice_fingerprint"]["threshold"], 1.0)
            saved = config_path.read_text(encoding="utf-8")
            self.assertIn("threshold: 1.0", saved)

    def test_set_voice_mode_rejects_whisper_without_profile(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            app = self._make_app(config_path)
            app.voice_mode = "talk"
            app.talk_mode = True
            app.config["voice_fingerprint"]["mode"] = "talk"
            app.voice_fingerprint.is_enrolled = lambda: False

            ok, message = app.set_voice_mode("whisper")

            self.assertFalse(ok)
            self.assertIn("requires an enrolled voice profile", message)
            self.assertTrue(app.talk_mode)
            self.assertEqual(app.config["voice_fingerprint"]["mode"], "talk")

    def test_set_voice_mode_updates_runtime_and_config(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            app = self._make_app(config_path)

            ok, message = app.set_voice_mode("talk")

            self.assertTrue(ok, message)
            self.assertTrue(app.talk_mode)
            self.assertEqual(app.voice_mode, "talk")
            self.assertEqual(app.config["voice_fingerprint"]["mode"], "talk")
            saved = config_path.read_text(encoding="utf-8")
            self.assertIn("mode: talk", saved)

    def test_voice_profile_status_includes_next_enrollment_phrase(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            app = self._make_app(config_path)
            app.voice_fingerprint.enrolled_embeddings = [object(), object()]
            app.voice_fingerprint.is_enrolled = lambda: False

            status = app.get_voice_profile_status()

            self.assertEqual(status["next_sample_number"], 3)
            self.assertEqual(
                status["next_sample_phrase"],
                main.enrollment_phrase_for_sample(2),
            )

    def test_clear_voice_profile_switches_to_talk_mode(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            app = self._make_app(config_path)
            app.voice_fingerprint.enrolled_embeddings = [object()]

            ok, message = app.clear_voice_profile()

            self.assertTrue(ok, message)
            self.assertEqual(app.voice_fingerprint.enrolled_embeddings, [])
            self.assertTrue(app.talk_mode)
            self.assertEqual(app.voice_mode, "talk")
            self.assertEqual(app.config["voice_fingerprint"]["mode"], "talk")
            self.assertIn("Switched to talk mode", message)

    def test_reload_personal_dictionary_reports_counts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            app = self._make_app(config_path)

            ok, message = app.reload_personal_dictionary()

            self.assertTrue(ok, message)
            self.assertIn("2 terms", message)
            self.assertIn("3 rules", message)

    def test_subtle_achievements_do_not_emit_full_overlay(self):
        app = main.Bloviate.__new__(main.Bloviate)
        app.config = {"achievements": {"celebrations": "subtle"}}
        emitted = {"status": [], "overlay": []}
        app.ui_window = SimpleNamespace(
            signals=SimpleNamespace(
                update_status=SimpleNamespace(emit=lambda value: emitted["status"].append(value)),
                show_achievement_unlocks=SimpleNamespace(
                    emit=lambda value: emitted["overlay"].append(value)
                ),
            )
        )

        app._show_achievement_unlocks([{"title": "Recovered From The Void"}])

        self.assertEqual(emitted["status"], ["Recovered From The Void"])
        self.assertEqual(emitted["overlay"], [])


if __name__ == "__main__":
    unittest.main()
