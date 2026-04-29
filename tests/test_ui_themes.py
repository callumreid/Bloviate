import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from ui_themes import (  # noqa: E402
    THEMES,
    WAVEFORM_PRESETS,
    is_hidden_theme,
    normalize_theme_id,
    theme_options,
    waveform_palette_for_config,
)


class UIThemeTests(unittest.TestCase):
    def test_theme_registry_has_multiple_named_themes(self):
        self.assertGreaterEqual(len(THEMES), 4)
        for theme_id, theme in THEMES.items():
            self.assertEqual(normalize_theme_id(theme_id), theme_id)
            self.assertIn("label", theme)
            self.assertIn("colors", theme)
            self.assertIn("waveform", theme)

    def test_dark_alias_maps_to_graphite(self):
        self.assertEqual(normalize_theme_id("dark"), "graphite")

    def test_hidden_themes_are_opt_in(self):
        visible_ids = {theme_id for theme_id, _label in theme_options()}
        all_ids = {theme_id for theme_id, _label in theme_options(include_hidden=True)}

        self.assertNotIn("lounge", visible_ids)
        self.assertIn("lounge", all_ids)
        self.assertTrue(is_hidden_theme("lounge"))

    def test_waveform_preset_and_custom_palette_resolution(self):
        ocean = waveform_palette_for_config({"ui": {"theme": "light", "waveform": {"preset": "ocean"}}})
        self.assertEqual(ocean["command"], WAVEFORM_PRESETS["ocean"]["command"])

        custom = waveform_palette_for_config(
            {
                "ui": {
                    "theme": "sunset",
                    "waveform": {
                        "preset": "custom",
                        "idle": "#111111",
                        "recording": "#222222",
                        "command": "#333333",
                        "accepted": "#444444",
                        "rejected": "#555555",
                        "processing": ["#123456", "not-a-color", "#ABCDEF"],
                    },
                }
            }
        )
        self.assertEqual(custom["idle"], "#111111")
        self.assertEqual(custom["processing"], ["#123456", "#ABCDEF"])


if __name__ == "__main__":
    unittest.main()
