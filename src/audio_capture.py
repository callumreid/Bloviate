"""
Audio capture module for Bloviate.
Handles audio input from Scarlett 4i4 interface.
"""

import math
import sounddevice as sd
import numpy as np
import queue
import threading
from scipy.signal import resample_poly
from typing import Optional, Callable


class AudioCapture:
    """Captures audio from the specified device."""

    def __init__(self, config: dict):
        self.config = config
        self.verbose_logs = bool(config.get("app", {}).get("verbose_logs", False))
        self.sample_rate = config['audio']['sample_rate']
        self.chunk_size = config['audio']['chunk_size']
        self.channels = config['audio']['channels']
        self.device_name = config['audio']['device_name']

        self.audio_queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.is_listening = False
        self.callbacks = []

        self.device_id = self._find_device()
        self._native_rate: Optional[int] = None
        self._needs_resample = False
        self._resample_up = 1
        self._resample_down = 1

    def _find_device(self) -> Optional[int]:
        """Find the configured audio device."""
        if not str(self.device_name or "").strip():
            default_input = getattr(sd, "default", None)
            device_pair = getattr(default_input, "device", None)
            if isinstance(device_pair, (list, tuple)) and len(device_pair) >= 1:
                try:
                    device_id = int(device_pair[0])
                except (TypeError, ValueError):
                    device_id = None
                if device_id is not None and device_id >= 0:
                    try:
                        device = sd.query_devices(device_id)
                        if device.get('max_input_channels', 0) > 0:
                            if getattr(self, "verbose_logs", False):
                                print(
                                    f"Using default input device: {device['name']} "
                                    f"(ID: {device_id})"
                                )
                            return device_id
                    except Exception:
                        pass

            if getattr(self, "verbose_logs", False):
                print("Using system default input device")
            return None

        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if self.device_name.lower() in device['name'].lower():
                if device['max_input_channels'] > 0:
                    if getattr(self, "verbose_logs", False):
                        print(f"Found audio device: {device['name']} (ID: {idx})")
                    return idx

        print(f"Warning: Could not find device matching '{self.device_name}'")
        print("Available input devices:")
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{idx}] {device['name']}")

        return None

    def list_input_devices(self) -> list[dict]:
        """Return available input devices with default-device metadata."""
        default_input_id = None
        default_input = getattr(sd, "default", None)
        device_pair = getattr(default_input, "device", None)
        if isinstance(device_pair, (list, tuple)) and len(device_pair) >= 1:
            try:
                default_input_id = int(device_pair[0])
            except (TypeError, ValueError):
                default_input_id = None

        devices = []
        for idx, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) <= 0:
                continue
            devices.append(
                {
                    "id": idx,
                    "name": str(device["name"]),
                    "channels": int(device.get("max_input_channels", 0)),
                    "is_default": idx == default_input_id,
                }
            )
        return devices

    def set_device(self, device_name: str):
        """Switch to a different configured input device."""
        was_running = self.stream is not None
        self.device_name = str(device_name or "").strip()
        self._native_rate = None
        self._needs_resample = False
        self._resample_up = 1
        self._resample_down = 1

        if was_running:
            self.stop()

        self.device_id = self._find_device()

        if was_running:
            self.start()

    def get_active_device_label(self) -> str:
        """Return a human-readable label for the active input selection."""
        if self.device_id is None:
            if str(self.device_name or "").strip():
                return str(self.device_name)
            return "System Default"

        try:
            info = sd.query_devices(self.device_id)
            return str(info.get("name") or f"Device {self.device_id}")
        except Exception:
            return f"Device {self.device_id}"

    def _compute_resample_ratio(self, native_rate: int, target_rate: int):
        """Set up integer resample ratio from native_rate to target_rate."""
        if native_rate == target_rate:
            self._needs_resample = False
            return
        g = math.gcd(target_rate, native_rate)
        self._resample_up = target_rate // g
        self._resample_down = native_rate // g
        self._needs_resample = True

    def start(self):
        """Start the audio stream."""
        if self.stream is not None:
            return

        # Determine the capture sample rate: try configured rate first,
        # fall back to device's native rate if PortAudio can't resample.
        capture_rate = self.sample_rate
        native_rate = None
        if self.device_id is not None:
            info = sd.query_devices(self.device_id)
            native_rate = int(info['default_samplerate'])

        def audio_callback(indata, frames, time, status):
            if status and getattr(self, "verbose_logs", False):
                print(f"Audio callback status: {status}")

            audio_data = indata.copy()

            if self._needs_resample:
                audio_data = resample_poly(
                    audio_data, self._resample_up, self._resample_down, axis=0
                ).astype(np.float32)

            self.audio_queue.put(audio_data)
            for callback in self.callbacks:
                callback(audio_data)

        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=capture_rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32
            )
            self.stream.start()
        except sd.PortAudioError:
            if native_rate and native_rate != capture_rate:
                if getattr(self, "verbose_logs", False):
                    print(f"Device does not support {capture_rate}Hz, "
                          f"capturing at native {native_rate}Hz and resampling")
                capture_rate = native_rate
                native_blocksize = int(self.chunk_size * native_rate / self.sample_rate)
                self._compute_resample_ratio(native_rate, self.sample_rate)
                self._native_rate = native_rate
                self.stream = sd.InputStream(
                    device=self.device_id,
                    channels=self.channels,
                    samplerate=capture_rate,
                    blocksize=native_blocksize,
                    callback=audio_callback,
                    dtype=np.float32
                )
                self.stream.start()
            else:
                raise

        if getattr(self, "verbose_logs", False):
            print(f"Audio stream started: capture={capture_rate}Hz, "
                  f"output={self.sample_rate}Hz, {self.channels} channel(s)")

    def stop(self):
        """Stop the audio stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            if getattr(self, "verbose_logs", False):
                print("Audio stream stopped")

    def register_callback(self, callback: Callable):
        """Register a callback to receive audio data."""
        self.callbacks.append(callback)

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the next audio chunk from the queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self):
        """Clear the audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def get_audio_level(self, audio_data: np.ndarray) -> float:
        """Calculate RMS audio level."""
        return float(np.sqrt(np.mean(audio_data ** 2)))
