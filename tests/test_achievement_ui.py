import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class AchievementUISmokeTests(unittest.TestCase):
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

        def achievement_summary(query="", status_filter="all"):
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
