import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class AchievementUISmokeTests(unittest.TestCase):
    def test_permission_prompt_ignores_unknown_and_manual_states(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from ui import permission_status_requires_prompt
        except Exception as exc:
            self.skipTest(f"PyQt6 unavailable: {exc}")

        self.assertFalse(permission_status_requires_prompt({"state": "granted"}))
        self.assertFalse(permission_status_requires_prompt({"state": "manual"}))
        self.assertFalse(permission_status_requires_prompt({"state": "unknown"}))
        self.assertTrue(permission_status_requires_prompt({"state": "missing"}))
        self.assertTrue(permission_status_requires_prompt({"state": "denied"}))

    def test_streak_heatmap_uses_relative_word_buckets_and_tooltips(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from ui import InsightStreakHeatmap
        except Exception as exc:
            self.skipTest(f"PyQt6 unavailable: {exc}")

        self.assertEqual(InsightStreakHeatmap._bucket_for_words(0, 100), 0)
        self.assertEqual(InsightStreakHeatmap._bucket_for_words(1, 100), 1)
        self.assertEqual(InsightStreakHeatmap._bucket_for_words(50, 100), 2)
        self.assertEqual(InsightStreakHeatmap._bucket_for_words(75, 100), 3)
        self.assertEqual(InsightStreakHeatmap._bucket_for_words(100, 100), 4)

        tooltip = InsightStreakHeatmap._format_day_tooltip(
            {"date": "2026-05-01", "words": 1234, "transcripts": 2}
        )

        self.assertIn("Fri, May 1, 2026", tooltip)
        self.assertIn("1,234 words", tooltip)
        self.assertIn("2 entries", tooltip)

    def test_settings_achievement_grid_loads_from_summary_callback(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication
            from ui import BloviateUI
        except Exception as exc:
            self.skipTest(f"PyQt6 unavailable: {exc}")

        app = QApplication.instance() or QApplication([])
        config = {
            "app": {"startup_animation": False},
            "ui": {
                "window_size": [1180, 860],
                "show_menubar_indicator": False,
                "ptt_overlay": {"enabled": False},
                "startup_splash": {"enabled": False},
                "show_main_window": False,
            },
            "audio": {},
            "ptt": {},
            "window_management": {},
            "voice_fingerprint": {"mode": "talk", "threshold": 0.6},
            "noise_suppression": {},
            "transcription": {},
            "history": {"enabled": True, "max_ui_records": 100},
            "achievements": {"enabled": True, "ai_analysis_enabled": False},
            "post_processing": {},
        }

        def achievement_summary(query="", status_filter="all", limit=None):
            return {
                "enabled": True,
                "ai_analysis_enabled": False,
                "total": 534,
                "unlocked": 1,
                "recent": [],
                "achievements": [
                    {
                        "id": "test",
                        "title": "Keyboard Lease Canceled",
                        "description": "Dictate 100 words.",
                        "category": "Word volume",
                        "progress_label": "100 / 100 words",
                        "progress_ratio": 1.0,
                        "unlocked": True,
                        "rarity": "common",
                        "badge_path": "",
                        "ai_required": False,
                        "hidden": False,
                    }
                ],
            }

        window = BloviateUI(config, get_achievement_summary=achievement_summary)
        try:
            window._refresh_achievements()
            self.assertEqual(window.achievement_table.rowCount(), 1)
            self.assertIn("1 / 534", window.achievement_summary_label.text())
        finally:
            window._closing = True
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
