"""
Speech-to-text transcription module for Bloviate.
Handles audio transcription using Whisper.
"""

import whisper
import numpy as np
import subprocess
import sys
import time
import yaml
import re
from pathlib import Path
from typing import Optional
from pynput.keyboard import Controller, Key


class Transcriber:
    """Handles speech-to-text transcription."""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config['transcription']['model']
        self.language = config['transcription']['language']
        self.output_format = config['transcription']['output_format']
        self.sample_rate = config['audio']['sample_rate']
        self.auto_paste = config['transcription'].get('auto_paste', True)
        self.use_custom_dictionary = config['transcription'].get('use_custom_dictionary', True)

        # Keyboard controller for auto-paste
        self.keyboard = Controller()

        # Load custom dictionary
        self.custom_dictionary = []
        if self.use_custom_dictionary:
            self._load_custom_dictionary()

        # Load Whisper model
        print(f"Loading Whisper model: {self.model_name}")
        try:
            self.model = whisper.load_model(self.model_name)
            print("Whisper model loaded")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None

    def _load_custom_dictionary(self):
        """Load custom dictionary from YAML file."""
        dict_path = Path("custom_dictionary.yaml")

        if not dict_path.exists():
            print("Custom dictionary not found, skipping...")
            return

        try:
            with open(dict_path, 'r') as f:
                data = yaml.safe_load(f)

            if not data or 'entries' not in data:
                return

            # Build dictionary with variations sorted by length (longest first)
            for entry in data['entries']:
                phrase = entry.get('phrase', '')
                variations = entry.get('variations', [])

                if phrase and variations:
                    # Sort variations by length (longest first) to avoid partial replacements
                    sorted_variations = sorted(variations, key=len, reverse=True)
                    self.custom_dictionary.append({
                        'phrase': phrase,
                        'variations': sorted_variations
                    })

            print(f"Loaded {len(self.custom_dictionary)} custom dictionary entries")

        except Exception as e:
            print(f"Error loading custom dictionary: {e}")

    def _apply_custom_dictionary(self, text: str) -> str:
        """Apply custom dictionary corrections to transcribed text."""
        if not self.custom_dictionary:
            return text

        corrected = text

        # Try each dictionary entry
        for entry in self.custom_dictionary:
            phrase = entry['phrase']
            variations = entry['variations']

            # Try each variation (case-insensitive)
            for variation in variations:
                # Use case-insensitive regex with word boundaries
                pattern = re.compile(re.escape(variation), re.IGNORECASE)

                # Check if this variation exists in the text
                if pattern.search(corrected):
                    corrected = pattern.sub(phrase, corrected)
                    print(f"[Dictionary] Corrected '{variation}' → '{phrase}'")

        return corrected

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio signal as numpy array

        Returns:
            Transcribed text or None if transcription fails
        """
        if self.model is None:
            return None

        try:
            # Ensure audio is 1D
            if len(audio.shape) > 1:
                audio = audio.squeeze()

            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Pad or trim audio to at least 0.5 seconds
            min_samples = int(self.sample_rate * 0.5)
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)))

            # Transcribe
            result = self.model.transcribe(
                audio,
                language=self.language,
                fp16=False,  # Use FP32 for CPU compatibility
                verbose=False
            )

            text = result['text'].strip()

            # Filter out empty or very short transcriptions
            if len(text) < 2:
                return None

            # Apply custom dictionary corrections
            if self.use_custom_dictionary:
                text = self._apply_custom_dictionary(text)

            return text

        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def output_text(self, text: str):
        """
        Output transcribed text according to configured format.

        Args:
            text: Text to output
        """
        if not text:
            return

        if self.output_format in ['clipboard', 'both']:
            self._copy_to_clipboard(text)
            print(f"[Clipboard] {text}")

            # Auto-paste if enabled
            if self.auto_paste:
                self._auto_paste()

        if self.output_format in ['stdout', 'both']:
            print(f"[Transcription] {text}")

    def _copy_to_clipboard(self, text: str):
        """Copy text to system clipboard."""
        try:
            if sys.platform == 'darwin':  # macOS
                process = subprocess.Popen(
                    ['pbcopy'],
                    stdin=subprocess.PIPE,
                    close_fds=True
                )
                process.communicate(text.encode('utf-8'))

            elif sys.platform == 'linux':
                # Try xclip first, then xsel
                try:
                    process = subprocess.Popen(
                        ['xclip', '-selection', 'clipboard'],
                        stdin=subprocess.PIPE,
                        close_fds=True
                    )
                    process.communicate(text.encode('utf-8'))
                except FileNotFoundError:
                    process = subprocess.Popen(
                        ['xsel', '--clipboard'],
                        stdin=subprocess.PIPE,
                        close_fds=True
                    )
                    process.communicate(text.encode('utf-8'))

            elif sys.platform == 'win32':  # Windows
                import win32clipboard
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text)
                win32clipboard.CloseClipboard()

        except Exception as e:
            print(f"Error copying to clipboard: {e}")

    def _auto_paste(self):
        """Automatically paste clipboard contents by simulating Cmd+V."""
        try:
            # Small delay to ensure clipboard is ready
            time.sleep(0.15)

            if sys.platform == 'darwin':
                # Use osascript for more reliable keyboard simulation on macOS
                subprocess.run([
                    'osascript', '-e',
                    'tell application "System Events" to keystroke "v" using command down'
                ], check=True, capture_output=True)
                print("[Auto-paste] Pasted to active window")
            else:
                # Use pynput for other platforms
                with self.keyboard.pressed(Key.ctrl):
                    self.keyboard.press('v')
                    self.keyboard.release('v')

        except subprocess.CalledProcessError as e:
            print(f"Error auto-pasting (AppleScript): {e}")
            print("You may need to grant accessibility permissions to Terminal/iTerm")
        except Exception as e:
            print(f"Error auto-pasting: {e}")
