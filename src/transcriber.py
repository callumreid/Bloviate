"""
Speech-to-text transcription module for Bloviate.
Handles audio transcription using Whisper or Deepgram.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from pynput.keyboard import Controller, Key

try:
    from deepgram_stream import DeepgramLiveSession
except Exception:
    DeepgramLiveSession = None


class Transcriber:
    """Handles speech-to-text transcription."""

    def __init__(self, config: dict):
        self.config = config
        self.provider = str(config['transcription'].get('provider', 'whisper')).lower()
        self.model_name = config['transcription'].get('model', 'base.en')
        self.language = config['transcription']['language']
        self.output_format = config['transcription']['output_format']
        self.sample_rate = config['audio']['sample_rate']
        self.auto_paste = config['transcription'].get('auto_paste', True)
        self.use_custom_dictionary = config['transcription'].get('use_custom_dictionary', True)
        self.deepgram_config = config.get('deepgram', {})
        self.deepgram_streaming = bool(self.deepgram_config.get('streaming', True))
        self._pending_audio = {}
        self._prebuffer_chunks = int(self.deepgram_config.get("prebuffer_chunks", 12))

        # Keyboard controller for auto-paste
        self.keyboard = Controller()

        # Load custom dictionary
        self.custom_dictionary = []
        if self.use_custom_dictionary:
            self._load_custom_dictionary()

        # Track active Deepgram streams by mode name
        self._streams = {}

        if self.provider not in {"whisper", "deepgram"}:
            print(f"Unknown transcription provider '{self.provider}', defaulting to whisper")
            self.provider = "whisper"

        if self.provider == "deepgram" and DeepgramLiveSession is None:
            print("Deepgram live streaming unavailable (websocket-client not installed)")

        # Load Whisper model only when needed
        self.model = None
        if self.provider == "whisper":
            print(f"Loading Whisper model: {self.model_name}")
            try:
                import whisper
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
                match_mode = entry.get('match', 'substring')

                if phrase and variations:
                    # Sort variations by length (longest first) to avoid partial replacements
                    sorted_variations = sorted(variations, key=len, reverse=True)
                    self.custom_dictionary.append({
                        'phrase': phrase,
                        'variations': sorted_variations,
                        'match': match_mode
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
            match_mode = entry.get('match', 'substring')

            # Try each variation (case-insensitive)
            for variation in variations:
                escaped = re.escape(variation)
                if match_mode == 'whole_word':
                    pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
                else:
                    pattern = re.compile(escaped, re.IGNORECASE)

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
        if self.provider == "deepgram":
            return self._transcribe_deepgram_prerecorded(audio)

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

            # Transcribe with Whisper
            result = self.model.transcribe(
                audio,
                language=self.language,
                fp16=False,  # Use FP32 for CPU compatibility
                verbose=False,
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

    def supports_streaming(self) -> bool:
        """Return True if live streaming is available for the current provider."""
        return (
            self.provider == "deepgram"
            and self.deepgram_streaming
            and DeepgramLiveSession is not None
        )

    def start_stream(self, mode: str) -> bool:
        """Start a live streaming session for a given mode (e.g., dictation/command)."""
        if not self.supports_streaming():
            return False

        api_key = self._get_deepgram_api_key()
        if not api_key:
            print("Deepgram API key not set (DEEPGRAM_API_KEY or config)")
            return False

        url = self._build_deepgram_live_url()
        session = DeepgramLiveSession(
            api_key,
            url,
            finalize_wait_s=float(self.deepgram_config.get("finalize_wait_s", 0.6)),
            connect_timeout_s=float(self.deepgram_config.get("connect_timeout_s", 2.0)),
            log=print,
        )

        if not session.start():
            print("Deepgram live connection failed")
            return False

        self._streams[mode] = session
        pending = self._pending_audio.pop(mode, [])
        for chunk in pending:
            session.send_audio(chunk)
        return True

    def send_audio_chunk(self, mode: str, audio: np.ndarray):
        """Send a chunk of audio to an active live session."""
        session = self._streams.get(mode)
        if session:
            session.send_audio(audio)
            return

        # Buffer a small pre-roll so we don't lose the first syllable.
        if self.supports_streaming():
            buffer = self._pending_audio.setdefault(mode, [])
            buffer.append(audio.copy())
            if len(buffer) > self._prebuffer_chunks:
                buffer.pop(0)

    def finish_stream(self, mode: str) -> Optional[str]:
        """Finalize a live session and return the transcript."""
        session = self._streams.pop(mode, None)
        if not session:
            self._pending_audio.pop(mode, None)
            return None

        text = session.finish()
        if text and self.use_custom_dictionary:
            text = self._apply_custom_dictionary(text)
        return text

    def get_stream_interim(self, mode: str) -> Optional[str]:
        """Return interim text for an active live session, if any."""
        session = self._streams.get(mode)
        if not session:
            return None
        return session.get_interim_text()

    def _get_deepgram_api_key(self) -> Optional[str]:
        key = self.deepgram_config.get("api_key")
        if key:
            return key
        env_name = self.deepgram_config.get("api_key_env", "DEEPGRAM_API_KEY")
        return os.getenv(env_name)

    def _build_deepgram_live_url(self) -> str:
        api_version = self._deepgram_api_version(for_streaming=True)
        params = self._deepgram_query_params(
            for_streaming=True,
            api_version=api_version,
        )

        query = urllib.parse.urlencode(params, doseq=True)
        return f"wss://api.deepgram.com/{api_version}/listen?{query}"

    def _deepgram_api_version(self, for_streaming: bool) -> str:
        explicit = (
            self.deepgram_config.get("api_version")
            if for_streaming
            else self.deepgram_config.get("prerecorded_api_version")
        )
        if explicit:
            return str(explicit).strip().lower()

        model = str(self._deepgram_model_name(for_streaming=for_streaming) or "").lower()
        if model.startswith("flux"):
            return "v2"
        return "v1"

    def _deepgram_model_name(self, for_streaming: bool) -> Optional[str]:
        model = self.deepgram_config.get("model")
        if not model:
            return None

        model = str(model).strip()
        model_lower = model.lower()

        if model_lower.startswith("flux"):
            parts = model_lower.split("-")
            if len(parts) != 3:
                language = (self.language or "en").lower()
                model = f"flux-general-{language}"

        if not for_streaming:
            prerec = self.deepgram_config.get("prerecorded_model")
            if prerec:
                model = str(prerec).strip()
            elif model_lower.startswith("flux"):
                model = "nova-3"

        return model

    def _deepgram_query_params(self, for_streaming: bool, api_version: str) -> dict:
        params = {
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
        }

        model = self._deepgram_model_name(for_streaming=for_streaming)
        if model:
            params["model"] = model

        if api_version == "v2":
            # v2 Flux-compatible params only.
            endpointing = self.deepgram_config.get("endpointing")
            eot_timeout_ms = self.deepgram_config.get("eot_timeout_ms")
            if eot_timeout_ms is None and endpointing is not None:
                eot_timeout_ms = max(500, min(10000, int(endpointing)))
            if eot_timeout_ms is not None:
                params["eot_timeout_ms"] = int(eot_timeout_ms)

            eot_threshold = self.deepgram_config.get("eot_threshold")
            if eot_threshold is not None:
                params["eot_threshold"] = float(eot_threshold)

            eager_eot_threshold = self.deepgram_config.get("eager_eot_threshold")
            if eager_eot_threshold is not None:
                params["eager_eot_threshold"] = float(eager_eot_threshold)

            if "mip_opt_out" in self.deepgram_config:
                params["mip_opt_out"] = str(bool(self.deepgram_config.get("mip_opt_out"))).lower()

            keyterm = self.deepgram_config.get("keyterm")
            if keyterm:
                params["keyterm"] = keyterm

            tag = self.deepgram_config.get("tag")
            if tag:
                params["tag"] = tag
        else:
            # v1 parameters.
            params["channels"] = self.config["audio"]["channels"]
            params["language"] = self.language

            if "tier" in self.deepgram_config:
                params["tier"] = self.deepgram_config.get("tier")

            def _bool_param(name: str, default: Optional[bool] = None):
                if name in self.deepgram_config:
                    params[name] = str(bool(self.deepgram_config.get(name))).lower()
                elif default is not None:
                    params[name] = str(default).lower()

            _bool_param("punctuate", True)
            _bool_param("smart_format", False)
            if for_streaming:
                _bool_param("interim_results", True)

            if for_streaming:
                endpointing = self.deepgram_config.get("endpointing")
                if endpointing is not None:
                    params["endpointing"] = int(endpointing)

                no_delay = self.deepgram_config.get("no_delay")
                if no_delay is not None:
                    params["no_delay"] = str(bool(no_delay)).lower()

        extra = self.deepgram_config.get("extra_query_params", {})
        if isinstance(extra, dict):
            for key, value in extra.items():
                if value is not None:
                    params[str(key)] = value

        return params

    def _transcribe_deepgram_prerecorded(self, audio: np.ndarray) -> Optional[str]:
        api_key = self._get_deepgram_api_key()
        if not api_key:
            print("Deepgram API key not set (DEEPGRAM_API_KEY or config)")
            return None

        # Ensure audio is 1D
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        api_version = self._deepgram_api_version(for_streaming=False)
        params = self._deepgram_query_params(
            for_streaming=False,
            api_version=api_version,
        )
        query = urllib.parse.urlencode(params, doseq=True)
        url = f"https://api.deepgram.com/{api_version}/listen?{query}"

        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/raw;encoding=linear16;sample_rate="
            f"{self.sample_rate};channels={self.config['audio']['channels']}",
        }

        req = urllib.request.Request(url, data=audio_bytes, headers=headers, method="POST")

        try:
            timeout_s = float(self.deepgram_config.get("prerecorded_timeout_s", 30))
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))

            channel = payload.get("results", {}).get("channels", [])
            if not channel:
                return None
            alternatives = channel[0].get("alternatives", [])
            if not alternatives:
                return None
            text = alternatives[0].get("transcript", "").strip()
            if not text:
                return None
            if self.use_custom_dictionary:
                text = self._apply_custom_dictionary(text)
            return text
        except Exception as exc:
            print(f"Deepgram transcription error: {exc}")
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
