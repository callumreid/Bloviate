import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_help_command_exits_cleanly(self):
        result = subprocess.run(
            [sys.executable, "src/main.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--doctor", result.stdout)
        self.assertIn("--show-paths", result.stdout)


if __name__ == "__main__":
    unittest.main()
