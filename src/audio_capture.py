"""
Audio capture module for Bloviate.
Handles audio input from Scarlett 4i4 interface.
"""

import sounddevice as sd
import numpy as np
import queue
import threading
from typing import Optional, Callable


class AudioCapture:
    """Captures audio from the specified device."""

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config['audio']['sample_rate']
        self.chunk_size = config['audio']['chunk_size']
        self.channels = config['audio']['channels']
        self.device_name = config['audio']['device_name']

        self.audio_queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.is_listening = False
        self.callbacks = []

        self.device_id = self._find_device()

    def _find_device(self) -> Optional[int]:
        """Find the Scarlett audio device."""
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if self.device_name.lower() in device['name'].lower():
                if device['max_input_channels'] > 0:
                    print(f"Found audio device: {device['name']} (ID: {idx})")
                    return idx

        print(f"Warning: Could not find device matching '{self.device_name}'")
        print("Available input devices:")
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{idx}] {device['name']}")

        return None

    def start(self):
        """Start the audio stream."""
        if self.stream is not None:
            return

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")

            # Copy data to avoid issues with the buffer
            audio_data = indata.copy()

            # Put in queue for processing
            self.audio_queue.put(audio_data)

            # Call registered callbacks
            for callback in self.callbacks:
                callback(audio_data)

        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            print(f"Audio stream started: {self.sample_rate}Hz, {self.channels} channel(s)")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            raise

    def stop(self):
        """Stop the audio stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
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
