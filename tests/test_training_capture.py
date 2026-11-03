import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training_capture import TrainingCapture


def _make_config(tmpdir: Path, **overrides) -> dict:
    cfg = {
        "audio": {"sample_rate": 16000},
        "training_capture": {
            "enabled": True,
            "max_hours": 20,
            "correction_hotkey": "<ctrl>+<option>+<cmd>",
            "correction_window_min": 30,
        },
    }
    cfg["training_capture"].update(overrides)
    os.environ["BLOVIATE_HOME"] = str(tmpdir)
    return cfg


class TrainingCaptureStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        os.environ.pop("BLOVIATE_HOME", None)

    def _store(self, **overrides) -> TrainingCapture:
        return TrainingCapture(_make_config(self.tmpdir, **overrides))

    def test_disabled_store_is_noop(self):
        store = self._store(enabled=False)
        store.record(
            raw_audio=np.zeros(16000, dtype=np.float32),
            history_id=1,
            audio_device="mic",
            raw_rms=0.01,
            provider="whisper",
            transcript="hi",
            output_text="hi",
            stream_text=None,
            target_app="Notes",
            outcome="pasted",
        )
        self.assertIsNone(store.root)
        self.assertIsNone(store.match_correction("hi"))

    def test_write_creates_audio_and_index_entry(self):
        store = self._store()
        audio = np.ones(16000, dtype=np.float32) * 0.1
        store.record(
            raw_audio=audio,
            history_id=42,
            audio_device="Scarlett",
            raw_rms=0.05,
            provider="whisper",
            transcript="hello world",
            output_text="Hello world.",
            stream_text=None,
            target_app="Notes",
            outcome="pasted",
        )
        # Allow background-free synchronous write (record is synchronous here).
        lines = store.captures_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["history_id"], 42)
        self.assertEqual(entry["output_text"], "Hello world.")
        self.assertEqual(entry["sample_rate"], 16000)
        rel = Path(entry["file"])
        self.assertTrue((store.root / rel).is_file())
        # WAV should be 1s of 16-bit mono.
        import wave

        with wave.open(str(store.root / rel), "rb") as wf:
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getnframes(), 16000)

    def test_prune_drops_oldest_beyond_max_hours(self):
        store = self._store(max_hours=0.0001)  # ~0.36s budget
        # Two 1s clips -> 2s total, exceeds budget; oldest pruned.
        for i in range(2):
            store.record(
                raw_audio=np.zeros(16000, dtype=np.float32),
                history_id=i,
                audio_device="mic",
                raw_rms=0.01,
                provider="whisper",
                transcript=f"clip {i}",
                output_text=f"clip {i}",
                stream_text=None,
                target_app="Notes",
                outcome="pasted",
            )
        lines = store.captures_path.read_text(encoding="utf-8").strip().splitlines()
        # At most one entry remains.
        self.assertLessEqual(len(lines), 1)
        if lines:
            self.assertEqual(json.loads(lines[0])["output_text"], "clip 1")


class CorrectionMatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        os.environ["BLOVIATE_HOME"] = str(self.tmpdir)

    def tearDown(self):
        self._tmp.cleanup()
        os.environ.pop("BLOVIATE_HOME", None)

    def _capture(self, output_text: str, age_minutes: float = 0.0) -> str:
        store = TrainingCapture(_make_config(self.tmpdir))
        store.record(
            raw_audio=np.zeros(1600, dtype=np.float32),
            history_id=1,
            audio_device="mic",
            raw_rms=0.01,
            provider="whisper",
            transcript=output_text,
            output_text=output_text,
            stream_text=None,
            target_app="Notes",
            outcome="pasted",
        )
        if age_minutes > 0:
            # Backdate the captures.jsonl entry timestamp.
            from datetime import datetime, timedelta, timezone

            lines = store.captures_path.read_text(encoding="utf-8").strip().splitlines()
            entry = json.loads(lines[-1])
            entry["created_at"] = (
                datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
            ).isoformat(timespec="seconds")
            lines[-1] = json.dumps(entry, ensure_ascii=False)
            store.captures_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return entry["capture_id"]
        return json.loads(store.captures_path.read_text().strip().splitlines()[-1])["capture_id"]

    def test_match_returns_capture_and_similarity(self):
        self._capture("hello world this is a test")
        store = TrainingCapture(_make_config(self.tmpdir))
        match = store.match_correction("hello world this is a test")
        self.assertIsNotNone(match)
        entry, sim = match
        self.assertGreaterEqual(sim, 0.5)

    def test_too_short_selection_does_not_match(self):
        self._capture("hello world this is a long dictation output")
        store = TrainingCapture(_make_config(self.tmpdir))
        self.assertIsNone(store.match_correction("hello"))

    def test_low_ratio_does_not_match(self):
        self._capture("completely different content here")
        store = TrainingCapture(_make_config(self.tmpdir))
        self.assertIsNone(store.match_correction("totally unrelated words now"))

    def test_outside_window_does_not_match(self):
        self._capture("hello world this is a test", age_minutes=120)
        store = TrainingCapture(_make_config(self.tmpdir, correction_window_min=30))
        self.assertIsNone(store.match_correction("hello world this is a test"))

    def test_save_correction_writes_jsonl(self):
        cap_id = self._capture("hello world this is a test")
        store = TrainingCapture(_make_config(self.tmpdir))
        store.save_correction(cap_id, "hello world this is a test", 1.0)
        lines = store.corrections_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["capture_id"], cap_id)
        self.assertEqual(entry["corrected_text"], "hello world this is a test")
        self.assertEqual(entry["similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
