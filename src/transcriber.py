"""
Speech-to-text transcription module for Bloviate.
Handles audio transcription using Whisper, Deepgram, or OpenAI.
"""

import json
import io
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import List, Optional, Tuple

import numpy as np
from pynput.keyboard import Controller, Key
from command_vocabulary import get_command_prompt_phrases
from personal_dictionary import load_personal_dictionary
from secret_store import SecretStore

try:
    from deepgram_stream import DeepgramLiveSession
except Exception:
    DeepgramLiveSession = None


class Transcriber:
    """Handles speech-to-text transcription."""

    def __init__(self, config: dict):
        self.config = config
        self.secret_store = SecretStore()
        self.verbose_logs = bool(config.get("app", {}).get("verbose_logs", False))
        self.transcription_config = config.get("transcription", {})
        self.provider = self._normalize_provider_name(
            self.transcription_config.get('provider', 'whisper')
        ) or "whisper"
        self.model_name = self.transcription_config.get('model', 'base.en')
        self.language = self.transcription_config['language']
        self.output_format = self.transcription_config['output_format']
        self.sample_rate = config['audio']['sample_rate']
        self.auto_paste = self.transcription_config.get('auto_paste', True)
        self.use_custom_dictionary = self.transcription_config.get('use_custom_dictionary', True)
        self.deepgram_config = config.get('deepgram', {})
        self.openai_config = config.get("openai", {})
        self.deepgram_streaming = bool(self.deepgram_config.get('streaming', True))
        self._pending_audio = {}
        self._stream_lock = threading.Lock()
        self._shutting_down = False
        self._prebuffer_chunks = int(self.deepgram_config.get("prebuffer_chunks", 12))
        self._deepgram_max_keyterms = int(self.deepgram_config.get("max_keyterms", 80))
        self._openai_key_missing_warned = False
        self._base_initial_prompt = str(self.transcription_config.get("initial_prompt", "") or "").strip()
        self._prompt_max_terms = max(1, int(self.transcription_config.get("prompt_max_terms", 40)))
        self._prompt_max_chars = max(120, int(self.transcription_config.get("prompt_max_chars", 600)))
        self._include_command_vocabulary = bool(
            self.transcription_config.get("include_command_vocabulary", True)
        )
        self._auto_prompt_cache = {}

        # Keyboard controller for auto-paste
        self.keyboard = Controller()

        self.reload_personal_dictionary(log_on_success=False)

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

        self._log_transcription_plan()

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

    def reload_personal_dictionary(self, *, log_on_success: bool = True) -> dict:
        """Reload personal dictionary terms/corrections and rebuild prompt caches."""
        personal_dictionary = load_personal_dictionary(self.config)
        self.learned_terms = personal_dictionary.get("preferred_terms", [])
        self.custom_dictionary = (
            personal_dictionary.get("corrections", []) if self.use_custom_dictionary else []
        )
        self.personal_dictionary_sources = personal_dictionary.get("sources", [])

        self._prompt_terms = self._build_prompt_terms()
        self._command_prompt_terms = self._build_command_prompt_terms()
        self._deepgram_bias_terms = self._build_deepgram_bias_terms()
        self._deepgram_command_terms = self._build_deepgram_command_terms()
        self._auto_prompt_cache = {}

        if self.verbose_logs and self.provider == "deepgram" and self._deepgram_bias_terms:
            print(f"[Deepgram] Loaded {len(self._deepgram_bias_terms)} bias terms")
        if self.verbose_logs and self.provider == "deepgram" and self._deepgram_command_terms:
            print(f"[Deepgram] Loaded {len(self._deepgram_command_terms)} command keyterms")
        if self.verbose_logs and self.learned_terms:
            print(f"[Transcription] Loaded {len(self.learned_terms)} preferred terms")
        if self.verbose_logs and self.custom_dictionary:
            print(f"[Transcription] Loaded {len(self.custom_dictionary)} correction rules")
        if self.verbose_logs and self.personal_dictionary_sources:
            print(f"[Transcription] Personal dictionary sources: {', '.join(self.personal_dictionary_sources)}")
        if self.verbose_logs and log_on_success:
            print(
                "[Transcription] Reloaded personal dictionary "
                f"({len(self.learned_terms)} terms, {len(self.custom_dictionary)} rules)"
            )

        return {
            "preferred_terms": len(self.learned_terms),
            "corrections": len(self.custom_dictionary),
            "sources": list(self.personal_dictionary_sources),
            "path": personal_dictionary.get("path"),
        }

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

    def transcribe(self, audio: np.ndarray, mode: str = "dictation") -> Optional[str]:
        """
        Transcribe audio to text.

        Falls back to local Whisper automatically when the configured provider fails.
        """
        if self.provider == "deepgram":
            result = self._transcribe_deepgram_prerecorded(audio, mode=mode)
            if result:
                return result
            print(f"[Fallback] Deepgram unavailable, using local Whisper ({self.model_name})")
            return self._transcribe_whisper(audio, mode=mode)
        if self.provider == "openai":
            result = self._transcribe_openai(audio, mode=mode)
            if result:
                return result
            print(f"[Fallback] OpenAI unavailable, using local Whisper ({self.model_name})")
            return self._transcribe_whisper(audio, mode=mode)

        return self._transcribe_whisper(audio, mode=mode)

    def transcribe_with_provider(
        self, provider: str, audio: np.ndarray, mode: str = "dictation"
    ) -> Optional[str]:
        """Transcribe with an explicit provider, without cross-provider fallback."""
        normalized = self._normalize_provider_name(provider)
        if normalized == "deepgram":
            return self._transcribe_deepgram_prerecorded(audio, mode=mode)
        if normalized == "openai":
            return self._transcribe_openai(audio, mode=mode)
        if normalized == "whisper":
            return self._transcribe_whisper(audio, mode=mode)
        return None

    def transcribe_with_priority(
        self, audio: np.ndarray, providers: List[str], mode: str = "dictation"
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
            text = self.transcribe_with_provider(normalized, audio, mode=mode)
            if text:
                return text, normalized
        return None, None

    def _transcribe_whisper(self, audio: np.ndarray, mode: str = "dictation") -> Optional[str]:
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

            prompt = self._compose_prompt("whisper", mode=mode)

            # Transcribe with Whisper
            result = self.model.transcribe(
                audio,
                language=self.language,
                fp16=False,  # Use FP32 for CPU compatibility
                verbose=False,
                initial_prompt=prompt or None,
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
        if self.verbose_logs:
            print(f"Loading Whisper model: {self.model_name}")
        try:
            import whisper
            self.model = whisper.load_model(self.model_name)
            if self.verbose_logs:
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
        with self._stream_lock:
            self._stream_ready_events[mode] = evt

        if self._shutting_down:
            evt.set()
            return False

        if not self.supports_streaming():
            evt.set()
            return False

        api_key = self._get_deepgram_api_key()
        if not api_key:
            print("Deepgram API key not set (DEEPGRAM_API_KEY or config)")
            evt.set()
            return False

        url = self._build_deepgram_live_url(mode=mode)
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

        if self._shutting_down:
            session.close()
            evt.set()
            return False

        with self._stream_lock:
            self._streams[mode] = session
            pending = self._pending_audio.pop(mode, [])
        for chunk in pending:
            session.send_audio(chunk)
        evt.set()
        return True

    def send_audio_chunk(self, mode: str, audio: np.ndarray):
        """Send a chunk of audio to an active live session."""
        if self._shutting_down:
            return

        with self._stream_lock:
            session = self._streams.get(mode)
        if session:
            session.send_audio(audio)
            return

        # Buffer a small pre-roll so we don't lose the first syllable.
        if self.supports_streaming():
            with self._stream_lock:
                buffer = self._pending_audio.setdefault(mode, [])
                buffer.append(audio.copy())
                if len(buffer) > self._prebuffer_chunks:
                    buffer.pop(0)

    def finish_stream(self, mode: str) -> Optional[str]:
        """Finalize a live session and return the transcript."""
        if self._shutting_down:
            with self._stream_lock:
                session = self._streams.pop(mode, None)
                self._pending_audio.pop(mode, None)
                self._stream_ready_events.pop(mode, None)
            if session:
                session.close()
            return None

        # Wait for the async connection attempt to finish (if any)
        with self._stream_lock:
            evt = self._stream_ready_events.pop(mode, None)
        if evt:
            connect_timeout = float(self.deepgram_config.get("connect_timeout_s", 2.0))
            evt.wait(timeout=connect_timeout + 0.5)

        with self._stream_lock:
            session = self._streams.pop(mode, None)
        if not session:
            with self._stream_lock:
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
        with self._stream_lock:
            session = self._streams.get(mode)
        if not session:
            return None
        return session.get_interim_text()

    def shutdown(self):
        """Close any active streaming sessions and discard pending audio."""
        self._shutting_down = True

        with self._stream_lock:
            active_streams = list(self._streams.items())
            self._streams.clear()
            self._pending_audio.clear()
            ready_events = list(self._stream_ready_events.values())
            self._stream_ready_events.clear()

        for evt in ready_events:
            evt.set()

        for mode, session in active_streams:
            try:
                session.close()
            except Exception as e:
                print(f"[Deepgram] Error closing stream '{mode}': {e}")

        # Wait briefly for background Whisper preload to finish so Python
        # doesn't tear down torch/audio state while that thread is active.
        load_thread = self._whisper_load_thread
        if load_thread and load_thread.is_alive():
            load_thread.join(timeout=5.0)
        self._whisper_load_thread = None

    def _get_deepgram_api_key(self) -> Optional[str]:
        return self.secret_store.get_api_key("deepgram", self.config)

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

    def _build_deepgram_live_url(self, mode: Optional[str] = None) -> str:
        api_version = self._deepgram_api_version(for_streaming=True)
        params = self._deepgram_query_params(
            for_streaming=True,
            api_version=api_version,
            mode=mode,
        )

        query = urllib.parse.urlencode(params, doseq=True)
        return f"wss://api.deepgram.com/{api_version}/listen?{query}"

    @staticmethod
    def _append_unique_term(
        target: List[str], seen: set, value: str, *, max_length: Optional[int] = None
    ):
        term = str(value).strip()
        if not term:
            return
        if max_length and len(term) > max_length:
            return
        key = term.lower()
        if key in seen:
            return
        seen.add(key)
        target.append(term)

    @staticmethod
    def _coerce_string_list(value) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    def _build_prompt_terms(self) -> List[str]:
        """Build generic prompt terms from explicit config plus preferred vocabulary only."""
        terms: List[str] = []
        seen = set()

        for term in self._coerce_string_list(self.transcription_config.get("prompt_terms")):
            self._append_unique_term(terms, seen, term)

        for term in self.learned_terms:
            self._append_unique_term(terms, seen, term)

        return terms

    def _build_command_prompt_terms(self) -> List[str]:
        """Build command phrases for command-mode prompting."""
        if not self._include_command_vocabulary:
            return []

        terms: List[str] = []
        seen = set()
        for phrase in get_command_prompt_phrases():
            self._append_unique_term(terms, seen, phrase)
        return terms

    def _build_deepgram_bias_terms(self) -> List[str]:
        """Build base Deepgram bias terms from config plus preferred vocabulary."""
        terms: List[str] = []
        seen = set()

        for term in self._coerce_string_list(self.deepgram_config.get("keyterm")):
            self._append_unique_term(terms, seen, term, max_length=80)

        for term in self.learned_terms:
            self._append_unique_term(terms, seen, term, max_length=80)

        if len(terms) > self._deepgram_max_keyterms:
            terms = terms[:self._deepgram_max_keyterms]
            print(
                f"[Deepgram] Capped bias terms to {self._deepgram_max_keyterms} "
                f"entries for request-size safety"
            )

        return terms

    def _build_deepgram_command_terms(self) -> List[str]:
        """Build command-mode Deepgram keyterms from the shared command vocabulary."""
        terms: List[str] = []
        seen = set()
        for phrase in self._command_prompt_terms:
            self._append_unique_term(terms, seen, phrase, max_length=80)
        return terms

    def _deepgram_terms_for_mode(self, mode: Optional[str]) -> List[str]:
        """Return base terms plus command terms for command-mode requests."""
        mode_name = str(mode or "dictation").strip().lower()
        if mode_name != "command" or not self._deepgram_command_terms:
            return self._deepgram_bias_terms

        merged = list(self._deepgram_bias_terms)
        seen = {term.lower() for term in merged}
        for term in self._deepgram_command_terms:
            self._append_unique_term(merged, seen, term, max_length=80)
            if len(merged) >= self._deepgram_max_keyterms:
                break
        return merged

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

    def _deepgram_query_params(
        self, for_streaming: bool, api_version: str, mode: Optional[str] = None
    ) -> dict:
        params = {
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
        }

        model = self._deepgram_model_name(for_streaming=for_streaming)
        if model:
            params["model"] = model
        bias_terms = self._deepgram_terms_for_mode(mode)

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

            if bias_terms:
                params["keyterm"] = bias_terms

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
                if bias_terms:
                    params["keyterm"] = bias_terms
            elif keywords:
                params["keywords"] = keywords
            elif bias_terms:
                single_word_terms = [term for term in bias_terms if " " not in term]
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
        return self.secret_store.get_api_key("openai", self.config)

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

    def _build_auto_prompt(self, mode: str) -> str:
        """Build a concise prompt from repo-local vocabulary."""
        mode_name = str(mode or "dictation").strip().lower()
        if mode_name == "command":
            phrases = self._command_prompt_terms[: self._prompt_max_terms]
            if not phrases:
                return ""
            return (
                "This is a short macOS window-management voice command. "
                "Prefer exact command phrases such as: "
                + "; ".join(phrases)
                + "."
            )

        terms = self._prompt_terms[: self._prompt_max_terms]
        if not terms:
            return ""
        return (
            "Prefer exact spellings for technical terms, commands, names, and code phrases such as: "
            + ", ".join(terms)
            + "."
        )

    def _compose_prompt(self, provider: str, mode: str = "dictation") -> str:
        """Compose provider-specific prompt text with automatic vocabulary hints."""
        mode_name = str(mode or "dictation").strip().lower()
        cached = self._auto_prompt_cache.get(mode_name)
        if cached is None:
            cached = self._build_auto_prompt(mode_name)
            self._auto_prompt_cache[mode_name] = cached

        parts: List[str] = []
        if self._base_initial_prompt:
            parts.append(self._base_initial_prompt)
        if provider == "openai":
            manual_prompt = str(self.openai_config.get("prompt", "") or "").strip()
            if manual_prompt:
                parts.append(manual_prompt)
        if cached:
            parts.append(cached)

        prompt = " ".join(part for part in parts if part).strip()
        if len(prompt) > self._prompt_max_chars:
            prompt = prompt[: self._prompt_max_chars].rstrip(" ,;")
        return prompt

    def _provider_unavailable_reason(self, provider: str) -> Optional[str]:
        normalized = self._normalize_provider_name(provider)
        if normalized == "openai" and not self._get_openai_api_key():
            env_name = self.openai_config.get("api_key_env", "OPENAI_API_KEY")
            return f"missing API key ({env_name})"
        if normalized == "deepgram" and not self._get_deepgram_api_key():
            env_name = self.deepgram_config.get("api_key_env", "DEEPGRAM_API_KEY")
            return f"missing API key ({env_name})"
        return None

    def _log_transcription_plan(self):
        """Log which providers and prompts are active for this run."""
        if not self.verbose_logs:
            return
        final_providers = self.get_final_pass_provider_priority()
        if final_providers:
            print(f"[Transcription] Final-pass provider order: {', '.join(final_providers)}")

        available = []
        for provider in final_providers:
            reason = self._provider_unavailable_reason(provider)
            if reason:
                print(f"[Transcription] {provider} unavailable: {reason}")
            else:
                available.append(provider)
        if available:
            print(f"[Transcription] Active final-pass providers: {', '.join(available)}")

        if self._prompt_terms:
            print(
                f"[Transcription] Loaded {len(self._prompt_terms)} vocabulary terms for dictation prompting"
            )
        if self._command_prompt_terms:
            print(
                f"[Transcription] Loaded {len(self._command_prompt_terms)} command phrases for prompting"
            )

    def _transcribe_openai(self, audio: np.ndarray, mode: str = "dictation") -> Optional[str]:
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

        prompt = self._compose_prompt("openai", mode=mode)
        if prompt:
            fields["prompt"] = prompt

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
        if self.verbose_logs:
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

    def _transcribe_deepgram_prerecorded(
        self, audio: np.ndarray, mode: str = "dictation"
    ) -> Optional[str]:
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
            mode=mode,
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
        if self.verbose_logs:
            print(f"[Deepgram] Sending {len(audio_bytes)} bytes, {len(audio)/self.sample_rate:.2f}s, RMS={rms:.6f}")

        try:
            timeout_s = float(self.deepgram_config.get("prerecorded_timeout_s", 30))
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))

            metadata = payload.get("metadata", {})
            duration = metadata.get("duration", "?")
            model = metadata.get("model_info", {})
            model_name = next(iter(model.values()), {}).get("name", "?") if model else "?"
            if self.verbose_logs:
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
            print("Grant Accessibility/Automation permission to Bloviate or the terminal app that launched it.")
        except Exception as e:
            print(f"Error auto-pasting: {e}")
