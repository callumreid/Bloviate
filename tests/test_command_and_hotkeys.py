import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pynput import keyboard


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import main
from post_processor import ProcessedTranscript
from ptt_handler import PTTHandler


class FakeListener:
    def stop(self):
        pass

    def join(self, timeout=None):
        pass


class FakeWindowManager:
    def __init__(self):
        self.resized = []
        self.desktops = []

    def resize_focused_window(self, command):
        self.resized.append(command)

    def switch_desktop(self, direction):
        self.desktops.append(direction)


class CommandAndHotkeyTests(unittest.TestCase):
    def _make_app(self):
        app = main.Bloviate.__new__(main.Bloviate)
        app.config = {
            "window_management": {
                "voice_command_prefixes": ["run command", "screen", "window", "desktop"]
            }
        }
        app.window_manager = FakeWindowManager()
        app.ui_window = None
        return app

    def test_inline_screen_command_executes_window_resize(self):
        app = self._make_app()

        handled = app._try_voice_command("screen left half")

        self.assertTrue(handled)
        self.assertEqual(app.window_manager.resized, ["left"])

    def test_run_command_desktop_executes_desktop_switch(self):
        app = self._make_app()

        handled = app._try_voice_command("run command desktop right")

        self.assertTrue(handled)
        self.assertEqual(app.window_manager.desktops, ["right"])

    def test_toggle_hotkey_does_not_trigger_shorter_ptt_prefix(self):
        handler = PTTHandler(
            {
                "ptt": {
                    "hotkey": "<cmd>+<option>",
                    "toggle_hotkey": "<cmd>+<option>+<shift>",
                    "press_delay_ms": 120,
                }
            }
        )
        events = []
        handler.listener = FakeListener()
        handler.on_press_callback = lambda: events.append("hold")
        handler.add_hotkey(
            "toggle_dictation",
            "<cmd>+<option>+<shift>",
            on_press=lambda: events.append("toggle"),
            match_exact=True,
            consume=True,
        )

        handler._on_press(keyboard.Key.cmd)
        handler._on_press(keyboard.Key.alt)
        handler._on_press(keyboard.Key.shift)
        time.sleep(0.16)

        self.assertEqual(events, ["toggle"])
        handler.stop()

    def test_rejected_voice_is_saved_to_history_without_output(self):
        app = main.Bloviate.__new__(main.Bloviate)
        app._shutdown_event = threading.Event()
        app.config = {
            "audio": {"sample_rate": 16000},
            "transcription": {"final_pass": "streaming", "output_format": "clipboard"},
            "voice_fingerprint": {"verify_on_raw_audio": True},
        }
        app.talk_mode = False
        app.ui_window = None
        app.voice_fingerprint = SimpleNamespace(verify_speaker=lambda _audio: (False, 0.58))
        app.noise_suppressor = SimpleNamespace(
            process=lambda audio: audio,
            speech_stats=lambda _audio: {"rms": 0.1, "speech_frames": 1, "frames": 1, "speech_ratio": 1.0},
            has_speech=lambda _audio: True,
        )
        output_calls = []
        app.transcriber = SimpleNamespace(
            supports_streaming=lambda: True,
            finish_stream=lambda _mode: "this should be recoverable",
            get_final_pass_provider_priority=lambda: [],
            transcribe_with_priority=lambda *_args, **_kwargs: (None, ""),
            output_text=lambda text: output_calls.append(text),
        )
        app.post_processor = SimpleNamespace(
            process=lambda text, target_app="": ProcessedTranscript(
                original_text=text,
                text=text,
                mode="verbatim",
                provider="deterministic",
                changed=False,
            )
        )
        history_calls = []
        app._active_target_context = lambda: {"target_app": "Tests", "target_window": "Unit"}
        app._record_history = lambda **kwargs: history_calls.append(kwargs)

        app.process_recording([np.ones((160,), dtype=np.float32)])

        self.assertEqual(output_calls, [])
        self.assertEqual(len(history_calls), 1)
        self.assertEqual(history_calls[0]["text"], "this should be recoverable")
        self.assertEqual(history_calls[0]["mode"], "dictation_rejected")
        self.assertEqual(history_calls[0]["output_action"], "voice_rejected_history_only")


if __name__ == "__main__":
    unittest.main()
