"""
Speech-to-text transcription module for Bloviate.
Handles audio transcription using Whisper, Deepgram, or OpenAI.
"""

import json
import io
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Optional, List, Tuple

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
        self.provider = self._normalize_provider_name(
            config['transcription'].get('provider', 'whisper')
        ) or "whisper"
        self.model_name = config['transcription'].get('model', 'base.en')
        self.language = config['transcription']['language']
        self.output_format = config['transcription']['output_format']
        self.sample_rate = config['audio']['sample_rate']
        self.auto_paste = config['transcription'].get('auto_paste', True)
        self.use_custom_dictionary = config['transcription'].get('use_custom_dictionary', True)
        self.deepgram_config = config.get('deepgram', {})
        self.openai_config = config.get("openai", {})
        self.deepgram_streaming = bool(self.deepgram_config.get('streaming', True))
        self._pending_audio = {}
        self._prebuffer_chunks = int(self.deepgram_config.get("prebuffer_chunks", 12))
        self._deepgram_max_keyterms = int(self.deepgram_config.get("max_keyterms", 80))
        self._openai_key_missing_warned = False

        # Keyboard controller for auto-paste
        self.keyboard = Controller()

        # Load custom dictionary
        self.custom_dictionary = []
        if self.use_custom_dictionary:
            self._load_custom_dictionary()

        self._deepgram_bias_terms = self._build_deepgram_bias_terms()
        if self.provider == "deepgram" and self._deepgram_bias_terms:
            print(f"[Deepgram] Loaded {len(self._deepgram_bias_terms)} bias terms")

        # Track active Deepgram streams by mode name
        self._streams = {}
        self._stream_ready_events = {}  # mode -> threading.Event

        if self.provider not in {"whisper", "deepgram", "openai"}:
            print(f"Unknown transcription provider '{self.provider}', defaulting to whisper")
            self.provider = "whisper"

        if self.provider == "deepgram" and DeepgramLiveSession is None:
            print("Deepgram live streaming unavailable (websocket-client not installed)")
        if self.provider == "openai" and not self._get_openai_api_key():
            env_name = self.openai_config.get("api_key_env", "OPENAI_API_KEY")
            print(f"[OpenAI] WARNING: No API key found. Set {env_name} to enable OpenAI STT.")

        # Load Whisper model. When Deepgram is primary, use a smaller
        # fallback model (base.en) and pre-load it in the background so
        # the first fallback doesn't stall on model loading.
        self.model = None
        self._whisper_load_thread = None
        if self.provider == "deepgram":
            self.model_name = config['transcription'].get('whisper_fallback_model', 'base.en')
            self._whisper_load_thread = threading.Thread(
                target=self._load_whisper_model, daemon=True
            )
            self._whisper_load_thread.start()
            self._validate_deepgram_key()
        elif self.provider == "openai":
            # Keep a local fallback model warm when OpenAI is primary.
            self.model_name = config['transcription'].get('whisper_fallback_model', 'base.en')
            self._whisper_load_thread = threading.Thread(
                target=self._load_whisper_model, daemon=True
            )
            self._whisper_load_thread.start()
        else:
            self._load_whisper_model()

    @staticmethod
    def _normalize_provider_name(provider: Optional[str]) -> Optional[str]:
        if provider is None:
            return None

        value = str(provider).strip().lower()
        aliases = {
            "local": "whisper",
            "local_whisper": "whisper",
            "openai-stt": "openai",
            "openai_transcribe": "openai",
        }
        return aliases.get(value, value)

    def get_final_pass_provider_priority(self) -> List[str]:
        """Resolve final-pass provider order from config with sensible defaults."""
        configured = self.config.get("transcription", {}).get("final_pass_provider_priority")
        providers: List[str] = []

        if isinstance(configured, str):
            configured = [item.strip() for item in configured.split(",")]

        if isinstance(configured, list):
            seen = set()
            for item in configured:
                normalized = self._normalize_provider_name(item)
                if not normalized or normalized in seen:
                    continue
                if normalized in {"whisper", "deepgram", "openai"}:
                    providers.append(normalized)
                    seen.add(normalized)

        if providers:
            return providers

        if self.provider == "openai":
            return ["openai", "deepgram", "whisper"]
        if self.provider == "deepgram":
            return ["deepgram", "whisper"]
        return ["whisper"]

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

        Falls back to local Whisper automatically when the configured provider fails.
        """
        if self.provider == "deepgram":
            result = self._transcribe_deepgram_prerecorded(audio)
            if result:
                return result
            print(f"[Fallback] Deepgram unavailable, using local Whisper ({self.model_name})")
            return self._transcribe_whisper(audio)
        if self.provider == "openai":
            result = self._transcribe_openai(audio)
            if result:
                return result
            print(f"[Fallback] OpenAI unavailable, using local Whisper ({self.model_name})")
            return self._transcribe_whisper(audio)

        return self._transcribe_whisper(audio)

    def transcribe_with_provider(self, provider: str, audio: np.ndarray) -> Optional[str]:
        """Transcribe with an explicit provider, without cross-provider fallback."""
        normalized = self._normalize_provider_name(provider)
        if normalized == "deepgram":
            return self._transcribe_deepgram_prerecorded(audio)
        if normalized == "openai":
            return self._transcribe_openai(audio)
        if normalized == "whisper":
            return self._transcribe_whisper(audio)
        return None

    def transcribe_with_priority(
        self, audio: np.ndarray, providers: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Try providers in order and return (text, provider_used)."""
        for provider in providers:
            normalized = self._normalize_provider_name(provider)
            if normalized not in {"whisper", "deepgram", "openai"}:
                continue
            if normalized == "openai" and not self._get_openai_api_key():
                if not self._openai_key_missing_warned:
                    env_name = self.openai_config.get("api_key_env", "OPENAI_API_KEY")
                    print(f"[OpenAI] Skipping provider (missing key: {env_name})")
                    self._openai_key_missing_warned = True
                continue
            text = self.transcribe_with_provider(normalized, audio)
            if text:
                return text, normalized
        return None, None

    def _transcribe_whisper(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe with local Whisper model, loading it lazily if needed."""
        if self.model is None and self._whisper_load_thread:
            self._whisper_load_thread.join(timeout=30)
        if self.model is None:
            self._load_whisper_model()
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

    def _load_whisper_model(self):
        """Lazily load the Whisper model on first use."""
        print(f"Loading Whisper model: {self.model_name}")
        try:
            import whisper
            self.model = whisper.load_model(self.model_name)
            print("Whisper model loaded")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None

    def supports_streaming(self) -> bool:
        """Return True if live streaming is available for the current provider."""
        return (
            self.provider == "deepgram"
            and self.deepgram_streaming
            and DeepgramLiveSession is not None
        )

    def start_stream(self, mode: str, _attempt: int = 0) -> bool:
        """Start a live streaming session for a given mode (e.g., dictation/command).

        Safe to call from any thread.  Sets a ready event so that
        finish_stream can wait for the connection attempt to complete.
        Retries once on connection failure before giving up.
        """
        max_retries = 1

        evt = threading.Event()
        self._stream_ready_events[mode] = evt

        if not self.supports_streaming():
            evt.set()
            return False

        api_key = self._get_deepgram_api_key()
        if not api_key:
            print("Deepgram API key not set (DEEPGRAM_API_KEY or config)")
            evt.set()
            return False

        url = self._build_deepgram_live_url()
        session = DeepgramLiveSession(
            api_key,
            url,
            finalize_wait_s=float(self.deepgram_config.get("finalize_wait_s", 0.6)),
            connect_timeout_s=float(self.deepgram_config.get("connect_timeout_s", 2.0)),
            stream_gain=self.deepgram_config.get("stream_gain", {}),
            log=print,
        )

        if not session.start():
            if _attempt < max_retries:
                print(f"[Deepgram] Connection failed, retrying ({_attempt + 1}/{max_retries})...")
                time.sleep(0.3)
                return self.start_stream(mode, _attempt=_attempt + 1)
            print("[Deepgram] Connection failed after retry")
            evt.set()
            return False

        self._streams[mode] = session
        pending = self._pending_audio.pop(mode, [])
        for chunk in pending:
            session.send_audio(chunk)
        evt.set()
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
        # Wait for the async connection attempt to finish (if any)
        evt = self._stream_ready_events.pop(mode, None)
        if evt:
            connect_timeout = float(self.deepgram_config.get("connect_timeout_s", 2.0))
            evt.wait(timeout=connect_timeout + 0.5)

        session = self._streams.pop(mode, None)
        if not session:
            self._pending_audio.pop(mode, None)
            print(f"[Deepgram] No active stream for '{mode}' (connection may have failed)")
            return None

        text = session.finish()

        if not text and session.error:
            error_type = session.error_type or "unknown"
            close_code = session.close_code
            detail = f"type={error_type}"
            if close_code:
                detail += f", close_code={close_code}"
            print(f"[Deepgram] Stream failed ({detail}), falling back to offline transcription")
        elif not text:
            print(f"[Deepgram] Stream returned empty transcript (no speech detected by Deepgram)")

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

    def _validate_deepgram_key(self):
        """Check the Deepgram API key at startup so failures are obvious."""
        api_key = self._get_deepgram_api_key()
        if not api_key:
            env_name = self.deepgram_config.get("api_key_env", "DEEPGRAM_API_KEY")
            print(
                f"[Deepgram] WARNING: No API key found. "
                f"Set {env_name} in your environment (e.g. ~/.zshrc)."
            )
            return

        url = "https://api.deepgram.com/v1/projects"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Token {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
            print("[Deepgram] API key verified")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                print(
                    f"[Deepgram] ERROR: API key rejected (HTTP {exc.code}). "
                    f"The key may be revoked or expired — generate a new one "
                    f"at https://console.deepgram.com and set DEEPGRAM_API_KEY."
                )
            else:
                print(f"[Deepgram] WARNING: Key validation returned HTTP {exc.code}")
        except Exception as exc:
            print(f"[Deepgram] WARNING: Could not validate API key ({exc})")

    def _build_deepgram_live_url(self) -> str:
        api_version = self._deepgram_api_version(for_streaming=True)
        params = self._deepgram_query_params(
            for_streaming=True,
            api_version=api_version,
        )

        query = urllib.parse.urlencode(params, doseq=True)
        return f"wss://api.deepgram.com/{api_version}/listen?{query}"

    def _build_deepgram_bias_terms(self) -> List[str]:
        """Build bias terms from config plus custom dictionary phrases."""
        terms: List[str] = []
        seen = set()

        def _add_term(value: str):
            term = str(value).strip()
            if not term:
                return
            if len(term) > 80:
                return
            key = term.lower()
            if key in seen:
                return
            seen.add(key)
            terms.append(term)

        configured = self.deepgram_config.get("keyterm", [])
        if isinstance(configured, str):
            configured = [configured]
        if isinstance(configured, list):
            for term in configured:
                _add_term(term)

        if self.deepgram_config.get("include_dictionary_keyterms", True):
            for entry in self.custom_dictionary:
                phrase = entry.get("phrase", "")
                _add_term(phrase)

        if len(terms) > self._deepgram_max_keyterms:
            terms = terms[:self._deepgram_max_keyterms]
            print(
                f"[Deepgram] Capped bias terms to {self._deepgram_max_keyterms} "
                f"entries for request-size safety"
            )

        return terms

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

            if self._deepgram_bias_terms:
                params["keyterm"] = self._deepgram_bias_terms

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

            model_name = str(model or "").lower()
            use_keyterm = model_name.startswith("nova-3")
            keywords = self.deepgram_config.get("keywords")
            if use_keyterm:
                if self._deepgram_bias_terms:
                    params["keyterm"] = self._deepgram_bias_terms
            elif keywords:
                params["keywords"] = keywords
            elif self._deepgram_bias_terms:
                single_word_terms = [term for term in self._deepgram_bias_terms if " " not in term]
                if single_word_terms:
                    boost = self.deepgram_config.get("keyword_boost")
                    if boost is None:
                        params["keywords"] = single_word_terms
                    else:
                        boost_value = float(boost)
                        params["keywords"] = [f"{term}:{boost_value:g}" for term in single_word_terms]

        extra = self.deepgram_config.get("extra_query_params", {})
        if isinstance(extra, dict):
            for key, value in extra.items():
                if value is not None:
                    params[str(key)] = value

        return params

    def _normalize_audio_for_int16(self, audio: np.ndarray) -> np.ndarray:
        """Apply capped RMS normalization before int16 conversion."""
        rms = float(np.sqrt(np.mean(audio ** 2)))
        noise_floor_rms = float(self.deepgram_config.get("prerecorded_noise_floor_rms", 5e-7))
        if rms <= noise_floor_rms:
            return audio

        target_rms = float(self.deepgram_config.get("prerecorded_target_rms", 0.05))
        max_gain_db = float(self.deepgram_config.get("prerecorded_max_gain_db", 45.0))
        min_gain_db = float(self.deepgram_config.get("prerecorded_min_gain_db", -8.0))
        max_gain = float(10 ** (max_gain_db / 20.0))
        min_gain = float(10 ** (min_gain_db / 20.0))

        gain = target_rms / max(rms, 1e-12)
        gain = min(max(gain, min_gain), max_gain)
        normalized = audio * gain

        peak_ceiling = float(self.deepgram_config.get("prerecorded_peak_ceiling", 0.95))
        peak = float(np.max(np.abs(normalized)))
        if peak > peak_ceiling > 0:
            normalized = normalized * (peak_ceiling / peak)

        return normalized

    def _get_openai_api_key(self) -> Optional[str]:
        key = self.openai_config.get("api_key")
        if key:
            return str(key)
        env_name = self.openai_config.get("api_key_env", "OPENAI_API_KEY")
        return os.getenv(env_name)

    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Serialize mono float audio to 16-bit PCM WAV bytes."""
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        return buffer.getvalue()

    @staticmethod
    def _build_multipart_form_data(
        fields: dict, file_field: str, file_name: str, file_content: bytes, file_mime: str
    ) -> Tuple[bytes, str]:
        boundary = f"----bloviate{int(time.time() * 1000)}"
        body = bytearray()

        for name, value in fields.items():
            if value is None:
                continue
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
            )
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'.encode(
                "utf-8"
            )
        )
        body.extend(f"Content-Type: {file_mime}\r\n\r\n".encode("utf-8"))
        body.extend(file_content)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        content_type = f"multipart/form-data; boundary={boundary}"
        return bytes(body), content_type

    def _transcribe_openai(self, audio: np.ndarray) -> Optional[str]:
        api_key = self._get_openai_api_key()
        if not api_key:
            if not self._openai_key_missing_warned:
                env_name = self.openai_config.get("api_key_env", "OPENAI_API_KEY")
                print(f"[OpenAI] API key not set ({env_name} or config)")
                self._openai_key_missing_warned = True
            return None

        model = str(self.openai_config.get("model", "gpt-4o-transcribe")).strip()
        if not model:
            model = "gpt-4o-transcribe"

        wav_bytes = self._audio_to_wav_bytes(audio)
        fields = {
            "model": model,
            "language": self.language,
        }

        prompt = self.openai_config.get("prompt")
        if prompt:
            fields["prompt"] = str(prompt)

        temperature = self.openai_config.get("temperature")
        if temperature is not None:
            fields["temperature"] = temperature

        response_format = self.openai_config.get("response_format")
        if response_format:
            fields["response_format"] = str(response_format)

        body, content_type = self._build_multipart_form_data(
            fields=fields,
            file_field="file",
            file_name="bloviate.wav",
            file_content=wav_bytes,
            file_mime="audio/wav",
        )

        base_url = str(self.openai_config.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        url = f"{base_url}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
        }

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        timeout_s = float(self.openai_config.get("timeout_s", 30))
        print(f"[OpenAI] Sending {len(wav_bytes)} bytes to model={model}")

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))

            text = str(payload.get("text", "")).strip()
            if not text:
                print("[OpenAI] Empty transcript")
                return None

            if self.use_custom_dictionary:
                text = self._apply_custom_dictionary(text)
            return text
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code in (401, 403):
                print(f"[OpenAI] Auth failed (HTTP {exc.code}). Check OPENAI_API_KEY.")
            elif exc.code == 429:
                print(f"[OpenAI] Rate limited (HTTP 429): {body}")
            else:
                print(f"[OpenAI] HTTP error {exc.code}: {body}")
            return None
        except Exception as exc:
            print(f"[OpenAI] Transcription error: {exc}")
            return None

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

        audio = self._normalize_audio_for_int16(audio)
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

        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"[Deepgram] Sending {len(audio_bytes)} bytes, {len(audio)/self.sample_rate:.2f}s, RMS={rms:.6f}")

        try:
            timeout_s = float(self.deepgram_config.get("prerecorded_timeout_s", 30))
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))

            metadata = payload.get("metadata", {})
            duration = metadata.get("duration", "?")
            model = metadata.get("model_info", {})
            model_name = next(iter(model.values()), {}).get("name", "?") if model else "?"
            print(f"[Deepgram] Response: duration={duration}s, model={model_name}")

            channel = payload.get("results", {}).get("channels", [])
            if not channel:
                print(f"[Deepgram] Prerecorded returned no channels: {json.dumps(payload)[:500]}")
                return None
            alternatives = channel[0].get("alternatives", [])
            if not alternatives:
                print("[Deepgram] Prerecorded returned no alternatives")
                return None
            text = alternatives[0].get("transcript", "").strip()
            if not text:
                print(f"[Deepgram] Prerecorded returned empty transcript (confidence={alternatives[0].get('confidence', '?')})")
                return None
            if self.use_custom_dictionary:
                text = self._apply_custom_dictionary(text)
            return text
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code in (401, 403):
                print(
                    f"[Deepgram] Auth failed (HTTP {exc.code}) — "
                    f"API key is likely revoked or expired. "
                    f"Generate a new key at https://console.deepgram.com"
                )
            elif exc.code == 429:
                print(f"[Deepgram] Rate limited (HTTP 429). {body}")
            else:
                print(f"[Deepgram] HTTP error {exc.code}: {body}")
            return None
        except Exception as exc:
            print(f"[Deepgram] Transcription error: {exc}")
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
