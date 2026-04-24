import sys
import threading
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from transcriber import Transcriber


class _DummySession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _DummyThread:
    def __init__(self, alive=True):
        self._alive = alive
        self.join_called = False
        self.join_timeout = None

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.join_called = True
        self.join_timeout = timeout
        self._alive = False


class TranscriberShutdownTests(unittest.TestCase):
    def test_shutdown_joins_whisper_load_thread(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber._shutting_down = False
        transcriber._stream_lock = threading.Lock()
        transcriber._streams = {"dictation": _DummySession()}
        transcriber._pending_audio = {"dictation": [1, 2, 3]}
        transcriber._stream_ready_events = {"dictation": threading.Event()}
        transcriber._whisper_load_thread = _DummyThread(alive=True)

        transcriber.shutdown()

        self.assertTrue(transcriber._shutting_down)
        self.assertEqual(transcriber._streams, {})
        self.assertEqual(transcriber._pending_audio, {})
        self.assertEqual(transcriber._stream_ready_events, {})
        self.assertTrue(transcriber._whisper_load_thread is None)

    def test_shutdown_skips_join_for_absent_thread(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber._shutting_down = False
        transcriber._stream_lock = threading.Lock()
        transcriber._streams = {}
        transcriber._pending_audio = {}
        transcriber._stream_ready_events = {}
        transcriber._whisper_load_thread = None

        transcriber.shutdown()

        self.assertTrue(transcriber._shutting_down)
        self.assertIsNone(transcriber._whisper_load_thread)


if __name__ == "__main__":
    unittest.main()
