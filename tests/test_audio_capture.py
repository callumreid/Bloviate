import sys
import queue
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audio_capture import AudioCapture


class AudioCaptureDeviceSelectionTests(unittest.TestCase):
    def test_audio_queue_is_bounded_and_drops_oldest_chunk(self):
        capture = AudioCapture.__new__(AudioCapture)
        capture.audio_queue = queue.Queue(maxsize=2)

        first = np.array([[1]], dtype=np.float32)
        second = np.array([[2]], dtype=np.float32)
        third = np.array([[3]], dtype=np.float32)

        AudioCapture._put_audio_chunk(capture, first)
        AudioCapture._put_audio_chunk(capture, second)
        AudioCapture._put_audio_chunk(capture, third)

        self.assertEqual(capture.audio_queue.qsize(), 2)
        np.testing.assert_array_equal(capture.audio_queue.get_nowait(), second)
        np.testing.assert_array_equal(capture.audio_queue.get_nowait(), third)

    def test_queue_max_chunks_has_reasonable_floor(self):
        config = {
            "app": {},
            "audio": {
                "sample_rate": 16000,
                "chunk_size": 1024,
                "channels": 1,
                "device_name": "",
                "queue_max_chunks": 1,
            },
        }

        with mock.patch("audio_capture.AudioCapture._find_device", return_value=None):
            capture = AudioCapture(config)

        self.assertEqual(capture.audio_queue.maxsize, 8)

    def test_list_input_devices_marks_default(self):
        capture = AudioCapture.__new__(AudioCapture)

        devices = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "Scarlett 4i4 USB", "max_input_channels": 2},
            {"name": "Built-in Output", "max_input_channels": 0},
        ]

        with mock.patch("audio_capture.sd.default", SimpleNamespace(device=[1, 3])):
            with mock.patch("audio_capture.sd.query_devices", return_value=devices):
                result = AudioCapture.list_input_devices(capture)

        self.assertEqual(
            result,
            [
                {
                    "id": 0,
                    "name": "MacBook Pro Microphone",
                    "channels": 1,
                    "is_default": False,
                },
                {
                    "id": 1,
                    "name": "Scarlett 4i4 USB",
                    "channels": 2,
                    "is_default": True,
                },
            ],
        )

    def test_blank_device_name_uses_default_input_device(self):
        capture = AudioCapture.__new__(AudioCapture)
        capture.device_name = ""

        with mock.patch("audio_capture.sd.default", SimpleNamespace(device=[4, 7])):
            with mock.patch(
                "audio_capture.sd.query_devices",
                side_effect=lambda device_id=None: (
                    {"name": "MacBook Pro Microphone", "max_input_channels": 1}
                    if device_id == 4
                    else []
                ),
            ):
                device_id = AudioCapture._find_device(capture)

        self.assertEqual(device_id, 4)

    def test_named_device_prefers_matching_input(self):
        capture = AudioCapture.__new__(AudioCapture)
        capture.device_name = "Scarlett"

        devices = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "Scarlett 4i4 USB", "max_input_channels": 2},
        ]

        with mock.patch("audio_capture.sd.query_devices", return_value=devices):
            device_id = AudioCapture._find_device(capture)

        self.assertEqual(device_id, 1)


if __name__ == "__main__":
    unittest.main()
