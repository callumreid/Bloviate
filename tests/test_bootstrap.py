import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import app_paths
import main


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.env_patch = patch.dict(os.environ, {"BLOVIATE_HOME": self.tempdir.name}, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_default_config_is_created_in_user_home(self):
        config_path = app_paths.ensure_default_config()
        self.assertEqual(config_path, Path(self.tempdir.name) / "config.yaml")
        self.assertTrue(config_path.exists())

        config, resolved = main._load_config("config.yaml")
        self.assertEqual(resolved, config_path)
        self.assertEqual(config["__config_dir__"], str(Path(self.tempdir.name)))
        self.assertIn("transcription", config)

    def test_personal_dictionary_initializes_in_user_home(self):
        config, _ = main._load_config("config.yaml")
        result = main.init_personal_dictionary(config)
        self.assertEqual(result, 0)

        dictionary_path = app_paths.personal_dictionary_path()
        self.assertEqual(dictionary_path, Path(self.tempdir.name) / "personal_dictionary.yaml")
        self.assertTrue(dictionary_path.exists())
        self.assertIn("preferred_terms", dictionary_path.read_text(encoding="utf-8"))

    def test_save_config_strips_runtime_metadata(self):
        config, path = main._load_config("config.yaml")
        config["audio"]["device_name"] = "AirPods Pro"
        config["__temp"] = "ignore me"

        saved_path = main._save_config(config)

        self.assertEqual(saved_path, path)
        saved_text = path.read_text(encoding="utf-8")
        self.assertIn("device_name: AirPods Pro", saved_text)
        self.assertNotIn("__config_path__", saved_text)
        self.assertNotIn("__temp", saved_text)


if __name__ == "__main__":
    unittest.main()
