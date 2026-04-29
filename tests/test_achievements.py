import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from achievement_badge import AchievementBadgeRenderer
from achievement_catalog import ACHIEVEMENTS
from achievement_service import AchievementService
from achievement_store import AchievementStore
from history_store import HistoryStore


class AchievementTests(unittest.TestCase):
    def test_catalog_has_large_unique_complete_achievement_set(self):
        self.assertGreaterEqual(len(ACHIEVEMENTS), 528)
        self.assertEqual(len({item.id for item in ACHIEVEMENTS}), len(ACHIEVEMENTS))
        self.assertGreaterEqual(sum(1 for item in ACHIEVEMENTS if not item.ai_required), 450)
        self.assertEqual(sum(1 for item in ACHIEVEMENTS if item.ai_required), 64)
        for item in ACHIEVEMENTS:
            self.assertTrue(item.title)
            self.assertTrue(item.description)
            self.assertTrue(item.category)
            self.assertTrue(item.metric)
            self.assertGreater(item.threshold, 0)
            self.assertTrue(item.badge_family)
            self.assertTrue(item.badge_motif)

    def test_store_unlocks_are_idempotent_and_resettable(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "history.sqlite"
            store = AchievementStore(db_path)
            definition = ACHIEVEMENTS[0]

            first = store.apply_evaluation([definition], {definition.metric: definition.threshold})
            second = store.apply_evaluation([definition], {definition.metric: definition.threshold + 1})

            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])
            self.assertIn(definition.id, store.unlock_map())
            self.assertEqual(store.reset(), 1)
            self.assertEqual(store.unlock_map(), {})

    def test_service_evaluates_history_dictionary_voice_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "history.sqlite"
            HistoryStore(db_path).add_transcript(
                text="HELLO kubectl https://example.com deploy_service.py supercalifragilisticexpialidocious?",
                original_text="um hello kubectl",
                duration_s=4.0,
                provider="deepgram_prerecorded",
                target_app="TextEdit",
                created_at="2026-04-28T08:00:00+00:00",
            )
            service = AchievementService(
                {"achievements": {"enabled": True}},
                store=AchievementStore(db_path),
                renderer=AchievementBadgeRenderer(Path(tempdir) / "badges"),
            )

            metrics = service.metric_values(
                dictionary_payload={
                    "preferred_terms": ["kubectl", "Bloviate", "Raycast"],
                    "corrections": [{"phrase": "kubectl", "variations": ["cube cuddle"]}],
                },
                voice_profile_status={"enrolled_samples": 8},
            )

            self.assertEqual(metrics["dictionary_terms"], 3)
            self.assertEqual(metrics["dictionary_corrections"], 1)
            self.assertGreaterEqual(metrics["longest_word_length"], 30)
            self.assertEqual(metrics["url_count"], 1)
            self.assertEqual(metrics["filename_count"], 1)
            self.assertGreater(metrics["words_per_minute"], 80)

            unlocks = service.evaluate(
                dictionary_payload={"preferred_terms": ["kubectl"], "corrections": []},
                voice_profile_status={"enrolled_samples": 8},
            )
            self.assertTrue(unlocks)
            summary = service.summary()
            self.assertEqual(summary["total"], len(ACHIEVEMENTS))
            self.assertGreater(summary["unlocked"], 0)

    def test_service_evaluates_easter_egg_metrics_from_config(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "history.sqlite"
            HistoryStore(db_path)
            service = AchievementService(
                {
                    "achievements": {"enabled": True},
                    "ui": {
                        "easter_eggs": {
                            "secret_themes_unlocked": True,
                            "cow_runs": 2,
                            "surprise_count": 1,
                            "about_opened": True,
                            "secret_theme_activations": 1,
                            "milestone_toasts": True,
                            "milestone_toasts_shown": 1,
                        }
                    },
                },
                store=AchievementStore(db_path),
                renderer=AchievementBadgeRenderer(Path(tempdir) / "badges"),
            )

            metrics = service.metric_values()

            self.assertEqual(metrics["easter_secret_themes"], 1)
            self.assertEqual(metrics["easter_cow_runs"], 2)
            self.assertEqual(metrics["easter_surprise_count"], 1)
            self.assertEqual(metrics["easter_about_opened"], 1)
            self.assertEqual(metrics["easter_secret_theme_activations"], 1)
            self.assertEqual(metrics["easter_milestone_toasts_shown"], 1)

    def test_ai_analysis_is_opt_in_and_stores_tags_only(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "history.sqlite"
            HistoryStore(db_path).add_transcript(
                text="Please summarize the meeting and create action items.",
                duration_s=3.0,
                target_app="Slack",
                created_at="2026-04-28T12:00:00+00:00",
            )
            service = AchievementService(
                {"achievements": {"enabled": True, "ai_analysis_enabled": False}},
                store=AchievementStore(db_path),
                renderer=AchievementBadgeRenderer(Path(tempdir) / "badges"),
            )

            ok, message, unlocks = service.analyze_history()
            self.assertFalse(ok)
            self.assertIn("disabled", message)
            self.assertEqual(unlocks, [])

            service.config["achievements"]["ai_analysis_enabled"] = True
            service.secret_store.get_api_key = lambda provider, config: "test-key"
            with mock.patch.object(
                service,
                "_classify_transcript",
                return_value={"genre_meeting": True, "content_action_items": True},
            ):
                ok, message, unlocks = service.analyze_history()

            self.assertTrue(ok, message)
            self.assertIn("Analyzed 1", message)
            self.assertTrue(unlocks)
            counts = service.store.ai_tag_counts()
            self.assertEqual(counts["ai_tag_genre_meeting"], 1)
            self.assertEqual(counts["ai_tag_content_action_items"], 1)

    def test_badge_renderer_creates_cached_png_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            renderer = AchievementBadgeRenderer(Path(tempdir))
            first = renderer.render_badge(ACHIEVEMENTS[0], unlocked=True)
            second = renderer.render_badge(ACHIEVEMENTS[0], unlocked=True)

            self.assertEqual(first, second)
            self.assertTrue(first.exists())
            self.assertGreater(first.stat().st_size, 0)

    def test_badge_renderer_handles_full_catalog_under_qt(self):
        try:
            from PyQt6.QtWidgets import QApplication
        except Exception:
            self.skipTest("PyQt6 is not installed")

        app = QApplication.instance() or QApplication([])
        self.assertIsNotNone(app)

        with tempfile.TemporaryDirectory() as tempdir:
            renderer = AchievementBadgeRenderer(Path(tempdir))
            for definition in ACHIEVEMENTS:
                path = renderer.render_badge(definition, unlocked=True, size=64)
                self.assertTrue(path.exists(), definition.id)
                self.assertGreater(path.stat().st_size, 0, definition.id)


if __name__ == "__main__":
    unittest.main()
