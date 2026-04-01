#!/usr/bin/env python3
"""
Bloviate - Voice-fingerprinting dictation tool for whispering in noisy environments.

Main application entry point.
"""

import argparse
import importlib.util
import os
import platform
import shutil
import sys
import threading
import time
import re
import numpy as np
from pathlib import Path
from typing import Optional
import yaml


def _load_dotenv(path: Optional[Path] = None):
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    if path is not None:
        candidate_paths = [path]
    else:
        candidate_paths = []
        override = os.getenv("BLOVIATE_ENV_FILE")
        if override:
            candidate_paths.append(Path(override).expanduser())
        candidate_paths.append(default_user_config_path().parent / ".env")
        candidate_paths.append(project_root() / ".env")

    for candidate in candidate_paths:
        if not candidate.is_file():
            continue
        with open(candidate, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value

from command_vocabulary import (
    DESKTOP_PREFIX_SUFFIXES,
    WINDOW_COMMAND_ALIASES,
    WINDOW_PREFIX_SUFFIXES,
    sorted_aliases,
)
from app_paths import (
    config_path as default_user_config_path,
    describe_paths,
    ensure_default_config,
    models_dir as default_models_dir,
    project_root,
    read_resource_text,
)
from personal_dictionary import add_preferred_terms, load_personal_dictionary, resolve_personal_dictionary_path


def _load_config(config_path: str, *, allow_missing: bool = False) -> tuple[dict, Path]:
    """Load YAML config relative to the project root."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        default_path = default_user_config_path()
        if path == Path("config.yaml"):
            path = default_path
        else:
            path = Path.cwd() / path

    if not path.exists():
        if path == default_user_config_path():
            path = ensure_default_config()
        if allow_missing:
            return {}, path
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data["__config_path__"] = str(path)
    data["__config_dir__"] = str(path.parent)
    return data, path


def _module_available(name: str) -> bool:
    """Check whether a Python module can be resolved without importing it."""
    return importlib.util.find_spec(name) is not None


def _doctor_line(status: str, label: str, detail: str):
    print(f"[{status}] {label}: {detail}")


def _cli_invocation() -> str:
    executable = Path(sys.argv[0]).name
    if executable.startswith("bloviate"):
        return "bloviate"
    return "python src/main.py"


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


def _final_pass_providers(config: dict) -> list[str]:
    configured = config.get("transcription", {}).get("final_pass_provider_priority")
    providers: list[str] = []

    if isinstance(configured, str):
        configured = [item.strip() for item in configured.split(",")]

    if isinstance(configured, list):
        seen = set()
        for item in configured:
            normalized = _normalize_provider_name(item)
            if not normalized or normalized in seen:
                continue
            if normalized in {"whisper", "deepgram", "openai"}:
                providers.append(normalized)
                seen.add(normalized)

    primary = _normalize_provider_name(
        config.get("transcription", {}).get("provider", "whisper")
    )
    if primary in {"whisper", "deepgram", "openai"} and primary not in providers:
        providers.insert(0, primary)

    return providers


def list_audio_devices(config: Optional[dict] = None) -> int:
    """Print available audio input devices for configuration."""
    config = config or {}
    configured_name = str(config.get("audio", {}).get("device_name", "") or "").strip()

    if not _module_available("sounddevice"):
        print("sounddevice is not installed. Run `pip install -e .` first.")
        return 1

    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"Unable to import sounddevice: {exc}")
        return 1

    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"Unable to query audio devices: {exc}")
        return 1

    input_devices = []
    for idx, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            input_devices.append((idx, device["name"], device["max_input_channels"]))

    if not input_devices:
        print("No input devices detected.")
        return 1

    print("Available input devices:")
    for idx, name, channels in input_devices:
        match = ""
        if configured_name and configured_name.lower() in name.lower():
            match = "  <- config match"
        print(f"  [{idx}] {name} (inputs: {channels}){match}")

    if configured_name:
        matches = [name for _, name, _ in input_devices if configured_name.lower() in name.lower()]
        if not matches:
            print(
                f"\nConfigured device_name '{configured_name}' does not match any current input device."
            )

    return 0


def show_paths() -> int:
    """Print the current config/data locations."""
    print("Bloviate paths:")
    for name, path in describe_paths().items():
        print(f"  {name}: {path}")
    return 0


def init_personal_dictionary(config: Optional[dict] = None, *, force: bool = False) -> int:
    """Create a personal dictionary file from the packaged example."""
    config = config or {}
    path = resolve_personal_dictionary_path(config)
    if path.exists() and not force:
        print(f"Personal dictionary already exists: {path}")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(read_resource_text("personal_dictionary.example.yaml"), encoding="utf-8")
    print(f"Initialized personal dictionary: {path}")
    return 0


def run_doctor(config_path: str) -> int:
    """Run a lightweight environment and configuration preflight."""
    print("=== Bloviate Doctor ===")
    failures = 0
    warnings = 0

    _doctor_line(
        "OK",
        "Python",
        f"{platform.python_version()} ({platform.system()} {platform.release()})",
    )

    try:
        config, resolved_config_path = _load_config(config_path)
        _doctor_line("OK", "Config", str(resolved_config_path))
    except Exception as exc:
        _doctor_line("FAIL", "Config", str(exc))
        return 1

    path_info = describe_paths()
    _doctor_line("OK", "User Data", str(path_info["home"]))

    required_modules = {
        "yaml": "PyYAML",
        "numpy": "numpy",
        "sounddevice": "sounddevice",
        "pynput": "pynput",
        "PyQt6": "PyQt6",
        "torch": "torch",
        "torchaudio": "torchaudio",
        "speechbrain": "speechbrain",
        "whisper": "openai-whisper",
        "noisereduce": "noisereduce",
        "webrtcvad": "webrtcvad",
        "websocket": "websocket-client",
    }
    missing = [label for module_name, label in required_modules.items() if not _module_available(module_name)]
    if missing:
        failures += 1
        _doctor_line("FAIL", "Dependencies", "Missing modules: " + ", ".join(sorted(missing)))
    else:
        _doctor_line("OK", "Dependencies", "All required Python modules are installed")

    configured_device = str(config.get("audio", {}).get("device_name", "") or "").strip()
    if _module_available("sounddevice"):
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            input_devices = [
                device["name"]
                for device in devices
                if device.get("max_input_channels", 0) > 0
            ]
            if not input_devices:
                failures += 1
                _doctor_line("FAIL", "Audio Input", "No input devices detected")
            elif configured_device and any(
                configured_device.lower() in device_name.lower() for device_name in input_devices
            ):
                _doctor_line("OK", "Audio Input", f"Matched configured device '{configured_device}'")
            elif configured_device:
                warnings += 1
                preview = ", ".join(input_devices[:5])
                _doctor_line(
                    "WARN",
                    "Audio Input",
                    (
                        f"Configured device '{configured_device}' not found. "
                        f"Detected inputs: {preview}"
                    ),
                )
            else:
                _doctor_line("OK", "Audio Input", f"Detected {len(input_devices)} input device(s)")
        except Exception as exc:
            failures += 1
            _doctor_line("FAIL", "Audio Input", f"Unable to query devices: {exc}")
    else:
        failures += 1
        _doctor_line("FAIL", "Audio Input", "sounddevice is unavailable")

    providers = _final_pass_providers(config)
    deepgram_env = str(config.get("deepgram", {}).get("api_key_env", "DEEPGRAM_API_KEY"))
    openai_env = str(config.get("openai", {}).get("api_key_env", "OPENAI_API_KEY"))

    if "deepgram" in providers:
        if os.getenv(deepgram_env):
            _doctor_line("OK", "Deepgram", f"API key found in {deepgram_env}")
        else:
            warnings += 1
            _doctor_line("WARN", "Deepgram", f"API key missing ({deepgram_env})")

    if "openai" in providers:
        if os.getenv(openai_env):
            _doctor_line("OK", "OpenAI", f"API key found in {openai_env}")
        else:
            warnings += 1
            _doctor_line("WARN", "OpenAI", f"API key missing ({openai_env})")

    voice_cfg = config.get("voice_fingerprint", {})
    voice_enabled = bool(voice_cfg.get("enabled", False))
    voice_mode = str(voice_cfg.get("mode", "whisper") or "whisper").strip().lower()
    profile_path = default_models_dir() / "voice_profile.pkl"
    if voice_enabled and voice_mode != "talk":
        if profile_path.exists():
            _doctor_line("OK", "Voice Profile", f"Found enrolled profile at {profile_path}")
        else:
            warnings += 1
            _doctor_line(
                "WARN",
                "Voice Profile",
                f"No enrolled profile found. Run `{_cli_invocation()} --enroll` or use `--voice-mode talk`.",
            )
    else:
        _doctor_line("OK", "Voice Profile", f"Voice verification bypassed (mode={voice_mode})")

    personal_dictionary_path = resolve_personal_dictionary_path(config)
    if personal_dictionary_path.exists():
        _doctor_line("OK", "Personal Dictionary", str(personal_dictionary_path))
    else:
        warnings += 1
        _doctor_line(
            "WARN",
            "Personal Dictionary",
            (
                f"File not found at {personal_dictionary_path}. "
                f"Run `{_cli_invocation()} --init-personal-dictionary` if you want local vocabulary biasing."
            ),
        )

    output_format = str(config.get("transcription", {}).get("output_format", "clipboard"))
    auto_paste = bool(config.get("transcription", {}).get("auto_paste", False))
    window_management_enabled = bool(config.get("window_management", {}).get("enabled", False))
    is_macos = sys.platform == "darwin"

    if is_macos:
        missing_bins = []
        if output_format in {"clipboard", "both"} and shutil.which("pbcopy") is None:
            missing_bins.append("pbcopy")
        if (auto_paste or window_management_enabled) and shutil.which("osascript") is None:
            missing_bins.append("osascript")

        if missing_bins:
            failures += 1
            _doctor_line("FAIL", "macOS Tools", "Missing: " + ", ".join(missing_bins))
        else:
            _doctor_line("OK", "macOS Tools", "pbcopy/osascript available")

        warnings += 1
        _doctor_line(
            "WARN",
            "Permissions",
            "Microphone permission is required. Accessibility permission is also required for global hotkeys on macOS.",
        )
    else:
        if window_management_enabled:
            warnings += 1
            _doctor_line(
                "WARN",
                "Window Management",
                "window_management is enabled but current implementation is macOS-only.",
            )
        else:
            _doctor_line("OK", "Platform Features", "macOS-only integrations are disabled")

    if auto_paste:
        warnings += 1
        _doctor_line(
            "WARN",
            "Auto-paste",
            "Enabled by default. This is convenient for demos but risky for first-time external users.",
        )
    else:
        _doctor_line("OK", "Auto-paste", "Disabled")

    print("\nNext steps:")
    cli = _cli_invocation()
    print(f"  1. Run `{cli} --list-devices` if your microphone is not detected.")
    print(f"  2. Run `{cli} --enroll` before whisper mode.")
    print(f"  3. Use `{cli} --voice-mode talk` for first-run smoke tests.")

    if failures:
        print(f"\nDoctor finished with {failures} failure(s) and {warnings} warning(s).")
        return 1

    print(f"\nDoctor finished with 0 failures and {warnings} warning(s).")
    return 0


class Bloviate:
    """Main application class."""

    def __init__(self, config_path: str = "config.yaml", voice_mode_override: Optional[str] = None):
        # Load configuration
        self.config, _ = _load_config(config_path)

        self.voice_mode = self._resolve_voice_mode(voice_mode_override)
        self.talk_mode = self.voice_mode == "talk"

        # Initialize components
        from audio_capture import AudioCapture
        from noise_suppressor import NoiseSuppressor
        from ptt_handler import PTTHandler
        from transcriber import Transcriber
        from voice_fingerprint import VoiceFingerprint

        self.audio_capture = AudioCapture(self.config)
        self.noise_suppressor = NoiseSuppressor(self.config)
        self.voice_fingerprint = VoiceFingerprint(self.config)
        self.transcriber = Transcriber(self.config)
        self.ptt_handler = PTTHandler(self.config)

        # Window management
        self.window_manager = None
        if self.config.get('window_management', {}).get('enabled', False):
            from window_manager import WindowManager

            self.window_manager = WindowManager()

        # State
        self.is_recording = False
        self.recorded_audio = []
        self.is_command_recording = False
        self.recorded_command_audio = []
        self.ui_window = None
        self.ui_app = None
        self._last_interim_text = ""
        self._last_interim_update = 0.0
        self._shutdown_event = threading.Event()
        self._worker_threads = set()
        self._worker_threads_lock = threading.Lock()
        self._interim_update_interval_s = float(
            self.config.get("ui", {}).get("interim_update_interval_s", 0.15)
        )

    def _start_worker(self, target, *args):
        """Start a background worker and track it for shutdown."""
        if self._shutdown_event.is_set():
            return None

        def runner():
            try:
                target(*args)
            except Exception as e:
                print(f"Worker error in {getattr(target, '__name__', 'background task')}: {e}")
            finally:
                with self._worker_threads_lock:
                    self._worker_threads.discard(threading.current_thread())

        thread = threading.Thread(
            target=runner,
            daemon=True,
            name=f"bloviate-{getattr(target, '__name__', 'worker')}",
        )
        with self._worker_threads_lock:
            self._worker_threads.add(thread)
        thread.start()
        return thread

    def _join_workers(self, timeout: float = 3.0):
        """Wait briefly for background workers to finish."""
        deadline = time.time() + timeout
        while True:
            with self._worker_threads_lock:
                threads = [thread for thread in self._worker_threads if thread.is_alive()]
            if not threads:
                return
            remaining = deadline - time.time()
            if remaining <= 0:
                alive = ", ".join(thread.name for thread in threads)
                print(f"Shutdown timed out waiting for workers: {alive}")
                return
            for thread in threads:
                thread.join(timeout=min(0.25, remaining))

    def _resolve_voice_mode(self, override: Optional[str] = None) -> str:
        """Resolve voice mode from config/CLI override."""
        if override is not None:
            mode = override
        else:
            mode = self.config.get("voice_fingerprint", {}).get("mode", "whisper")

        mode = str(mode).strip().lower()
        if mode in {"talk", "open", "bypass"}:
            return "talk"
        if mode != "whisper":
            print(f"Unknown voice mode '{mode}', defaulting to whisper.")
            return "whisper"
        return mode

    def enroll_voice(self):
        """Run voice enrollment process."""
        n = self.voice_fingerprint.min_enrollment_samples
        print("\n=== Voice Enrollment ===")
        print(f"You'll record {n} samples. For each one:")
        print("  1. Read the phrase shown")
        print("  2. Press Enter — recording starts after a 1-second countdown")
        print("  3. Say the phrase clearly at your normal speaking volume\n")

        self.audio_capture.start()

        phrases = [
            "The quick brown fox jumps over the lazy dog",
            "Bloviate is my voice transcription tool",
            "I use this every day to capture my thoughts",
            "Voice recognition works best with consistent samples",
            "My voice has a unique signature that identifies me",
            "Speaking clearly helps the model learn my voice",
            "This is another sample to improve accuracy",
            "Final enrollment phrase for voice fingerprinting",
            "The weather today is perfect for working outside",
            "Artificial intelligence makes transcription effortless",
        ]

        i = 0
        while i < n:
            phrase = phrases[i % len(phrases)]
            print(f"\nSample {i+1}/{n} — say this phrase:")
            print(f"  \"{phrase}\"")
            input("Press Enter when ready...")

            # 1-second countdown so they're not caught off guard
            for countdown in range(1, 0, -1):
                print(f"  Starting in {countdown}...", end="\r")
                time.sleep(1)
            print("  Recording...          ", end="\r")

            # Record for 3 seconds with live countdown
            samples = []
            start_time = time.time()
            duration = 3.0
            while True:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                if remaining <= 0:
                    break
                print(f"  Recording... {remaining:.1f}s remaining  ", end="\r")
                chunk = self.audio_capture.get_audio_chunk(timeout=0.1)
                if chunk is not None:
                    samples.append(chunk)
            print("  Done.                          ")

            if len(samples) == 0:
                print("  No audio captured — check your microphone and try again.")
                continue

            audio = np.concatenate(samples).flatten()

            # Warn if audio is too quiet
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.01:
                print(f"  Audio level very low (RMS={rms:.4f}) — speak louder or move closer to the mic.")
                retry = input("  Retry this sample? [Y/n]: ").strip().lower()
                if retry != "n":
                    continue

            audio = self.noise_suppressor.process(audio)
            success = self.voice_fingerprint.enroll_sample(audio)

            if success:
                print(f"  ✓ Sample {i+1} enrolled (RMS={rms:.4f})")
                i += 1
            else:
                print(f"  ✗ Failed to enroll — retrying")

        self.audio_capture.stop()

        if self.voice_fingerprint.is_enrolled():
            self.voice_fingerprint.save_profile()
            print("\n✓ Voice enrollment complete! You can now run Bloviate normally.")
        else:
            print("\n✗ Voice enrollment failed. Please try again.")

    def on_ptt_press(self):
        """Called when PTT is activated."""
        if self._shutdown_event.is_set():
            return
        print("\n[PTT] Activated")

        if self.ui_window:
            self.ui_window.signals.update_ptt_status.emit(True)
            self.ui_window.signals.update_status.emit("Listening...")

        self.is_recording = True
        self.recorded_audio = []
        self.audio_capture.clear_queue()
        self._last_interim_text = ""
        self._last_interim_update = 0.0

        if self.transcriber.supports_streaming():
            # Connect asynchronously so PTT press returns immediately
            self._start_worker(self.transcriber.start_stream, "dictation")

    def on_ptt_release(self):
        """Called when PTT is released."""
        if self._shutdown_event.is_set():
            return
        print("[PTT] Released")

        if self.ui_window:
            self.ui_window.signals.update_ptt_status.emit(False)
            self.ui_window.signals.update_status.emit("Processing...")

        self.is_recording = False

        # Capture audio and process in background to keep PTT responsive
        recorded = self.recorded_audio
        self.recorded_audio = []
        if len(recorded) > 0:
            self._start_worker(self.process_recording, recorded)
        else:
            if self.transcriber.supports_streaming():
                self.transcriber.finish_stream("dictation")
            print("No audio recorded")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("No audio recorded")

    def on_command_press(self):
        """Called when command mode PTT is activated."""
        if self._shutdown_event.is_set():
            return
        print("\n[CMD] Activated")

        if self.ui_window:
            self.ui_window.signals.update_command_status.emit("CMD: Listening...", "listening")

        self.is_command_recording = True
        self.recorded_command_audio = []
        self.audio_capture.clear_queue()
        self._last_interim_text = ""
        self._last_interim_update = 0.0

        if self.transcriber.supports_streaming():
            self._start_worker(self.transcriber.start_stream, "command")

    def on_command_release(self):
        """Called when command mode PTT is released."""
        if self._shutdown_event.is_set():
            return
        print("[CMD] Released")

        if self.ui_window:
            self.ui_window.signals.update_command_status.emit("CMD: Processing...", "processing")

        self.is_command_recording = False

        # Capture audio and process in background
        recorded = self.recorded_command_audio
        self.recorded_command_audio = []
        if len(recorded) > 0:
            self._start_worker(self.process_command_recording, recorded)
        else:
            if self.transcriber.supports_streaming():
                self.transcriber.finish_stream("command")
            print("[CMD] No audio recorded")
            if self.ui_window:
                self.ui_window.signals.update_command_status.emit("CMD: No audio recorded", "unrecognized")

    def _normalize_command_text(self, text: str) -> str:
        """Normalize text for command matching."""
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        return f" {normalized} " if normalized else ""

    def _parse_window_command(self, text: str) -> Optional[str]:
        """Parse the transcribed text into a window command."""
        normalized = self._normalize_command_text(text)
        if not normalized:
            return None

        for phrase, position in sorted_aliases(WINDOW_COMMAND_ALIASES):
            if f" {phrase} " in normalized:
                return position

        return None

    def _try_voice_command(self, text: str) -> bool:
        """Check if dictated text contains a voice command (window/desktop).

        Looks for 'window <command>' or 'desktop <command>' patterns.
        Returns True if a command was found and executed.
        """
        if not self.window_manager:
            return False

        normalized = self._normalize_command_text(text)
        if not normalized:
            return False

        # Check for "window <command>"
        for suffix, position in sorted_aliases(WINDOW_PREFIX_SUFFIXES):
            if f" window {suffix} " in normalized:
                print(f"[VOICE CMD] Matched 'window {suffix}' → {position}")
                self._execute_voice_command(position, text)
                return True

        # Check for "desktop <command>"
        for suffix, position in sorted_aliases(DESKTOP_PREFIX_SUFFIXES):
            if f" desktop {suffix} " in normalized:
                print(f"[VOICE CMD] Matched 'desktop {suffix}' → {position}")
                self._execute_voice_command(position, text)
                return True

        return False

    def _execute_voice_command(self, command: str, original_text: str):
        """Execute a voice command and update UI."""
        if command.startswith("desktop_"):
            direction = command.replace("desktop_", "")
            self.window_manager.switch_desktop(direction)
        else:
            self.window_manager.resize_focused_window(command)

        if self.ui_window:
            self.ui_window.signals.update_command_status.emit(
                f"Voice: {command.replace('_', ' ').title()}",
                "recognized"
            )
            self.ui_window.signals.update_status.emit("Ready")

    def process_command_recording(self, recorded_chunks=None):
        """Process the recorded command audio."""
        if self._shutdown_event.is_set():
            return
        if recorded_chunks is None:
            recorded_chunks = self.recorded_command_audio
        # Concatenate all recorded chunks
        audio = np.concatenate(recorded_chunks).flatten()

        print(f"[CMD] Processing {len(audio)} samples ({len(audio)/self.config['audio']['sample_rate']:.2f}s)")

        # Prefer the live transcript for latency, but retry with the
        # higher-accuracy final-pass providers when the command does not parse.
        stream_text = None
        if self.transcriber.supports_streaming():
            stream_text = self.transcriber.finish_stream("command")
        if self._shutdown_event.is_set():
            return

        command = None
        text = stream_text
        if stream_text:
            command = self._parse_window_command(stream_text)
            if command:
                print(f"[CMD] Streaming recognized command: {command}")

        if not command:
            audio_for_transcription = self.noise_suppressor.process(audio)
            final_provider_order = self.transcriber.get_final_pass_provider_priority()
            final_text, provider_used = self.transcriber.transcribe_with_priority(
                audio_for_transcription,
                final_provider_order,
                mode="command",
            )
            if final_text:
                text = final_text
                command = self._parse_window_command(final_text)
                if provider_used:
                    print(f"[CMD] Final-pass provider: {provider_used}")
                if stream_text and final_text != stream_text:
                    print("[CMD] Using higher-accuracy final pass instead of streaming text")
        if self._shutdown_event.is_set():
            return

        if not text:
            print("[CMD] No transcription generated")
            if self.ui_window:
                self.ui_window.signals.update_command_status.emit("CMD: No speech detected", "unrecognized")
            return

        print(f"[CMD] Transcribed: {text}")

        if command:
            print(f"[CMD] Recognized command: {command}")
            if self.window_manager:
                if command.startswith("desktop_"):
                    direction = command.replace("desktop_", "")
                    self.window_manager.switch_desktop(direction)
                else:
                    self.window_manager.resize_focused_window(command)
            if self.ui_window:
                self.ui_window.signals.update_command_status.emit(
                    f"CMD: {command.replace('_', ' ').title()} (recognized)",
                    "recognized"
                )
        else:
            print(f"[CMD] Unrecognized command: {text}")
            if self.ui_window:
                self.ui_window.signals.update_command_status.emit(
                    f"CMD: Unrecognized ({text})",
                    "unrecognized"
                )

    def process_recording(self, recorded_chunks=None):
        """Process the recorded audio."""
        if self._shutdown_event.is_set():
            return
        if recorded_chunks is None:
            recorded_chunks = self.recorded_audio
        # Concatenate all recorded chunks
        raw_audio = np.concatenate(recorded_chunks).flatten()

        print(f"Processing {len(raw_audio)} samples ({len(raw_audio)/self.config['audio']['sample_rate']:.2f}s)")

        # Finalize streaming (if enabled) before running heavy processing
        stream_text = None
        if self.transcriber.supports_streaming():
            stream_text = self.transcriber.finish_stream("dictation")
        if self._shutdown_event.is_set():
            return

        # Keep a denoised path for transcription and an optional raw path for
        # speaker verification (noise suppression can blur speaker identity).
        audio_for_transcription = self.noise_suppressor.process(raw_audio)
        verify_on_raw = bool(
            self.config.get("voice_fingerprint", {}).get("verify_on_raw_audio", True)
        )
        audio_for_verification = raw_audio if verify_on_raw else audio_for_transcription
        if self._shutdown_event.is_set():
            return

        # Verify speaker (or bypass in talk mode)
        if self.talk_mode:
            is_match, similarity = True, -1.0
            print("Voice match: bypassed (talk mode)")
        else:
            is_match, similarity = self.voice_fingerprint.verify_speaker(audio_for_verification)
            print(f"Voice match: {is_match} (similarity: {similarity:.3f})")

        if self.ui_window:
            self.ui_window.signals.update_voice_match.emit(is_match, similarity)

        if not self.talk_mode and not is_match:
            print("✗ Voice rejected - does not match enrolled profile")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("Voice rejected")
            return

        # Transcribe (use streaming result if available)
        if self.ui_window:
            self.ui_window.signals.update_status.emit("Transcribing...")

        transcription_cfg = self.config.get("transcription", {})
        final_pass_mode = str(transcription_cfg.get("final_pass", "hybrid")).strip().lower()
        if final_pass_mode not in {"hybrid", "prerecorded", "streaming"}:
            print(f"Unknown transcription.final_pass '{final_pass_mode}', defaulting to hybrid")
            final_pass_mode = "hybrid"

        final_provider_order = self.transcriber.get_final_pass_provider_priority()
        text = None
        if final_pass_mode == "streaming":
            text = stream_text
            if not text:
                print("[Final] Streaming transcript unavailable, trying final-pass providers")
                text, provider_used = self.transcriber.transcribe_with_priority(
                    audio_for_transcription, final_provider_order, mode="dictation"
                )
                if provider_used:
                    print(f"[Final] Used provider: {provider_used}")
        else:
            final_text, provider_used = self.transcriber.transcribe_with_priority(
                audio_for_transcription, final_provider_order, mode="dictation"
            )
            if final_text:
                text = final_text
                if provider_used:
                    print(f"[Final] Used provider: {provider_used}")
                if final_pass_mode == "hybrid" and stream_text and stream_text != final_text:
                    print("[Final] Using higher-accuracy final pass instead of streaming text")
            elif final_pass_mode == "hybrid":
                text = stream_text

        if text:
            print(f"✓ Transcribed: {text}")
            if self._shutdown_event.is_set():
                return

            # Check if text contains a voice command
            if self._try_voice_command(text):
                return

            self.transcriber.output_text(text)

            if self.ui_window:
                self.ui_window.signals.update_transcription.emit(text)
                self.ui_window.signals.update_status.emit("Ready")
        else:
            print("✗ No transcription generated")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("No speech detected")

    def _setup_window_management_hotkeys(self):
        """Setup window management hotkeys."""
        prefix = self.config.get('window_management', {}).get('hotkey_prefix', '<ctrl>+<cmd>')
        command_hotkey = self.config.get('window_management', {}).get('command_hotkey', prefix)

        # Command mode hotkey (voice-driven window management)
        self.ptt_handler.add_hotkey(
            'window_command_mode',
            command_hotkey,
            on_press=self.on_command_press,
            on_release=self.on_command_release,
            match_exact=True
        )

        # Add hotkeys for each direction
        self.ptt_handler.add_hotkey(
            'window_left',
            f"{prefix}+<left>",
            on_press=lambda: self.window_manager.resize_focused_window('left')
        )
        self.ptt_handler.add_hotkey(
            'window_right',
            f"{prefix}+<right>",
            on_press=lambda: self.window_manager.resize_focused_window('right')
        )
        self.ptt_handler.add_hotkey(
            'window_top',
            f"{prefix}+<up>",
            on_press=lambda: self.window_manager.resize_focused_window('top')
        )
        self.ptt_handler.add_hotkey(
            'window_bottom',
            f"{prefix}+<down>",
            on_press=lambda: self.window_manager.resize_focused_window('bottom')
        )

        print(f"Window management enabled with hotkey prefix: {prefix}")
        print(f"Command mode hotkey: {command_hotkey}")
        print("  Ctrl+Cmd+← = Left half")
        print("  Ctrl+Cmd+→ = Right half")
        print("  Ctrl+Cmd+↑ = Top half")
        print("  Ctrl+Cmd+↓ = Bottom half")

    def audio_callback(self, audio_data: np.ndarray):
        """Called for each audio chunk."""
        if self._shutdown_event.is_set():
            return
        # Update UI with audio level
        if self.ui_window:
            level = self.audio_capture.get_audio_level(audio_data)
            self.ui_window.signals.update_audio_level.emit(level)

        # Record audio if PTT is active
        if self.is_recording:
            self.recorded_audio.append(audio_data.copy())
            if self.transcriber.supports_streaming():
                self.transcriber.send_audio_chunk("dictation", audio_data)
                self._emit_interim("dictation")

        # Record audio if command mode is active
        if self.is_command_recording:
            self.recorded_command_audio.append(audio_data.copy())
            if self.transcriber.supports_streaming():
                self.transcriber.send_audio_chunk("command", audio_data)

    def _emit_interim(self, mode: str):
        """Emit interim transcription updates with throttling."""
        if self._shutdown_event.is_set() or not self.ui_window:
            return

        now = time.time()
        if now - self._last_interim_update < self._interim_update_interval_s:
            return

        text = self.transcriber.get_stream_interim(mode)
        if not text or text == self._last_interim_text:
            return

        self._last_interim_text = text
        self._last_interim_update = now
        self.ui_window.signals.update_interim_transcription.emit(text)

    def run(self):
        """Run the main application."""
        # Check if voice is enrolled (unless talk mode)
        if not self.talk_mode and not self.voice_fingerprint.is_enrolled():
            print("Voice not enrolled. Please run with --enroll first.")
            return
        if not self.talk_mode and self.config.get("voice_fingerprint", {}).get("enabled", False) and not self.voice_fingerprint.enabled:
            print("Voice fingerprinting failed to initialize; aborting to avoid unverified dictation.")
            return
        if self.talk_mode and self.config.get("voice_fingerprint", {}).get("enabled", False) and not self.voice_fingerprint.enabled:
            print("Voice fingerprinting unavailable; talk mode bypasses verification.")

        print("\n=== Bloviate ===")
        if len(self.ptt_handler.hotkey_strs) > 1:
            print(f"Hotkeys: {', '.join(self.ptt_handler.hotkey_strs)}")
        else:
            print(f"Hotkey: {self.ptt_handler.hotkey_str}")
        print(f"Voice mode: {self.voice_mode}")
        print("Press and hold the hotkey to record, release to transcribe.")
        print("Press Ctrl+C to exit.\n")

        # Create UI
        from ui import create_ui

        self.ui_app, self.ui_window = create_ui(self.config)
        if self.talk_mode and self.ui_window:
            self.ui_window.signals.update_voice_match.emit(True, -1.0)

        # Start audio capture
        self.audio_capture.start()
        self.audio_capture.register_callback(self.audio_callback)

        # Start PTT handler
        self.ptt_handler.start(
            on_press=self.on_ptt_press,
            on_release=self.on_ptt_release
        )

        # Add window management hotkeys if enabled
        if self.window_manager:
            self._setup_window_management_hotkeys()

        exit_code = 0
        try:
            # Run UI event loop
            exit_code = self.ui_app.exec()

        except KeyboardInterrupt:
            print("\nShutting down...")

        finally:
            # Cleanup in proper order
            print("Cleaning up...")
            self._shutdown_event.set()
            self.is_recording = False
            self.is_command_recording = False

            ui_window = self.ui_window
            ui_app = self.ui_app
            self.ui_window = None
            self.ui_app = None

            # Stop PTT handler first
            try:
                self.ptt_handler.stop()
            except Exception as e:
                print(f"Error stopping PTT handler: {e}")

            # Stop audio capture
            try:
                self.audio_capture.stop()
            except Exception as e:
                print(f"Error stopping audio capture: {e}")

            # Close streaming/network resources before interpreter teardown
            try:
                self.transcriber.shutdown()
            except Exception as e:
                print(f"Error shutting down transcriber: {e}")

            # Let in-flight workers finish without touching torn-down UI objects
            self._join_workers()

            # Clean up UI
            try:
                if ui_window:
                    if hasattr(ui_window, 'menu_bar_indicator') and ui_window.menu_bar_indicator:
                        ui_window.menu_bar_indicator.close()
                    ui_window.close()
                if ui_app:
                    ui_app.quit()
            except Exception as e:
                print(f"Error cleaning up UI: {e}")

            # Small delay to let threads finish
            time.sleep(0.2)

            print("Shutdown complete")

        return exit_code


def main():
    """Main entry point."""
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Bloviate - Voice dictation with fingerprinting")
    parser.add_argument(
        '--enroll',
        action='store_true',
        help='Enroll your voice for fingerprinting'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(default_user_config_path()),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--clear-profile',
        action='store_true',
        help='Clear the existing voice profile'
    )
    parser.add_argument(
        '--voice-mode',
        choices=['whisper', 'talk'],
        help='Override voice mode (whisper=verify, talk=bypass)'
    )
    parser.add_argument(
        '--add-term',
        '--learn-term',
        action='append',
        dest='add_term',
        default=[],
        help='Add a preferred term to the local personal dictionary (repeatable)'
    )
    parser.add_argument(
        '--show-personal-dictionary',
        '--list-learned-terms',
        action='store_true',
        dest='show_personal_dictionary',
        help='Show the local personal dictionary and exit'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio input devices and exit'
    )
    parser.add_argument(
        '--doctor',
        action='store_true',
        help='Run environment and configuration checks and exit'
    )
    parser.add_argument(
        '--show-paths',
        action='store_true',
        help='Show the current config/data paths and exit'
    )
    parser.add_argument(
        '--init-personal-dictionary',
        action='store_true',
        help='Create a personal dictionary file from the packaged example and exit'
    )

    args = parser.parse_args()

    if args.show_paths:
        sys.exit(show_paths())

    if args.doctor:
        sys.exit(run_doctor(args.config))

    config, _ = _load_config(
        args.config,
        allow_missing=args.list_devices,
    )

    if args.list_devices:
        sys.exit(list_audio_devices(config))

    if args.init_personal_dictionary:
        sys.exit(init_personal_dictionary(config))

    if args.add_term:
        path, added = add_preferred_terms(config, args.add_term)
        if added:
            print(f"Added {len(added)} preferred term(s) to {path}:")
            for term in added:
                print(f"  - {term}")
        else:
            print(f"No new preferred terms added. File: {path}")
        sys.exit(0)

    if args.show_personal_dictionary:
        path = resolve_personal_dictionary_path(config)
        personal_dictionary = load_personal_dictionary(config)
        terms = personal_dictionary.get("preferred_terms", [])
        corrections = personal_dictionary.get("corrections", [])
        print(f"Personal dictionary file: {path}")
        print(f"Preferred terms: {len(terms)}")
        for term in terms:
            print(f"  - {term}")
        print(f"Corrections: {len(corrections)}")
        for entry in corrections:
            variations = ", ".join(entry.get("variations", []))
            print(f"  - {entry.get('phrase')}: {variations}")
        sys.exit(0)

    app = Bloviate(config_path=args.config, voice_mode_override=args.voice_mode)

    if args.clear_profile:
        app.voice_fingerprint.clear_profile()
        print("Voice profile cleared.")
        sys.exit(0)

    if args.enroll:
        app.enroll_voice()
        sys.exit(0)
    else:
        exit_code = app.run()
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
