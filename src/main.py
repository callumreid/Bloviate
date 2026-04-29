#!/usr/bin/env python3
"""
Bloviate - Voice-fingerprinting dictation tool for whispering in noisy environments.

Main application entry point.
"""

import argparse
import contextlib
import importlib.util
import io
import logging
import os
import platform
import subprocess
import shutil
import shlex
import sys
import threading
import time
import re
import warnings
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
        for legacy_dir in (
            Path.home() / "personal" / "bloviate",
            Path.home() / "dev" / "bloviate",
            Path.home() / "src" / "bloviate",
            Path.home() / "Projects" / "bloviate",
        ):
            candidate_paths.append(legacy_dir / ".env")

    seen_paths = set()
    for candidate in candidate_paths:
        try:
            candidate = candidate.expanduser().resolve()
        except Exception:
            candidate = candidate.expanduser()
        if candidate in seen_paths:
            continue
        seen_paths.add(candidate)
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
    models_dir as default_models_dir,
    project_root,
    read_resource_text,
)
from personal_dictionary import (
    add_preferred_terms,
    load_personal_dictionary,
    migrate_legacy_personal_dictionary,
    resolve_personal_dictionary_path,
    save_personal_dictionary,
)
from achievement_service import AchievementService
from history_store import HistoryStore
from macos_permissions import (
    accessibility_trusted,
    open_privacy_pane,
    request_accessibility,
)
from model_registry import ModelRegistry
from post_processor import PostProcessor
from secret_store import SecretStore
from settings_service import SettingsService, load_yaml_config, save_config


def _load_config(config_path: str, *, allow_missing: bool = False) -> tuple[dict, Path]:
    """Load YAML config relative to the project root."""
    return load_yaml_config(config_path, allow_missing=allow_missing)


def _save_config(config: dict) -> Path:
    """Persist the current config back to its resolved file."""
    return save_config(config)


def _is_verbose_logging_enabled(config: dict) -> bool:
    app_cfg = config.get("app", {})
    return bool(app_cfg.get("verbose_logs", False))


def _configure_runtime_output(config: dict):
    """Reduce third-party startup noise for end-user runs."""
    app_cfg = config.get("app", {})
    verbose_logs = bool(app_cfg.get("verbose_logs", False))
    suppress_third_party = bool(app_cfg.get("suppress_third_party_warnings", True))

    if verbose_logs or not suppress_third_party:
        return

    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.cuda\.amp\.custom_fwd\(args\.\.\.\) is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*You are using `torch\.load` with `weights_only=False`.*",
        category=FutureWarning,
    )

    for logger_name in ("speechbrain", "torchaudio", "torch", "matplotlib"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


@contextlib.contextmanager
def _suppress_startup_stdio(enabled: bool):
    """Optionally silence noisy third-party stdout/stderr during startup."""
    if not enabled:
        yield
        return

    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield


def _show_startup_animation(config: dict):
    """Render a short ASCII startup animation for interactive sessions."""
    app_cfg = config.get("app", {})
    if not bool(app_cfg.get("startup_animation", True)):
        return
    if not sys.stdout.isatty():
        return

    # Alternating leg/tail poses so the cow "runs" across the terminal.
    cow_frames = [
        [
            r"      (__)",
            r"      (oo)",
            r" /-----\/ ",
            r"/ |   ||  ",
            r"*  /\-\/\ ",
            r"   ~~  ~~ ",
        ],
        [
            r"      (__)",
            r"      (oo)",
            r" /-----\/ ",
            r"/ |   ||  ",
            r"*  /\/\-\ ",
            r"   ~~  ~~ ",
        ],
    ]
    moo_frames = ["Moo", "Mooo", "MOO", "moo"]

    loops = max(1, int(app_cfg.get("startup_animation_loops", 1)))
    frame_delay_s = float(app_cfg.get("startup_animation_frame_delay_s", 0.04))
    step = max(1, int(app_cfg.get("startup_animation_step", 2)))
    columns = max(40, shutil.get_terminal_size(fallback=(80, 20)).columns)
    cow_width = max(len(line) for frame in cow_frames for line in frame)

    def _place(text: str, x: int, width: int) -> str:
        if x < 0:
            offset = -x
            if offset >= len(text):
                return ""
            return text[offset:][:width]
        if x >= width:
            return ""
        return ((" " * x) + text)[:width]

    hide_cursor = "\033[?25l"
    show_cursor = "\033[?25h"
    clear = "\033[2J"
    home = "\033[H"

    try:
        sys.stdout.write(hide_cursor + clear)
        sys.stdout.flush()
        for loop_idx in range(loops):
            frame_idx = 0
            for x in range(-cow_width - 8, columns + 2, step):
                cow = cow_frames[frame_idx % len(cow_frames)]
                moo = f"< {moo_frames[frame_idx % len(moo_frames)]} >"
                bubble_x = x - len(moo) - 2
                lines = [
                    _place(moo, bubble_x, columns),
                    _place(r" " + ("_" * max(0, len(moo) - 2)), bubble_x + 1, columns),
                ]
                for line in cow:
                    lines.append(_place(line, x, columns))
                lines.append("=" * columns)
                lines.append(_place("Bloviate warming up...", max(0, columns - 26), columns))

                sys.stdout.write(home + "\n".join(lines) + "\n")
                sys.stdout.flush()
                time.sleep(frame_delay_s)
                frame_idx += 1
            if loop_idx < loops - 1:
                time.sleep(frame_delay_s * 2)
        sys.stdout.write(clear + home)
        sys.stdout.flush()
    finally:
        sys.stdout.write(show_cursor)
        sys.stdout.flush()


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
    if not force:
        migrated_path, terms, corrections, migrated = migrate_legacy_personal_dictionary(config)
        if migrated:
            print(
                f"Imported existing dictionary into {migrated_path} "
                f"({terms} term(s), {corrections} replacement(s))."
            )
            return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(read_resource_text("personal_dictionary.example.yaml"), encoding="utf-8")
    print(f"Initialized personal dictionary: {path}")
    return 0


def _open_macos_privacy_pane(kind: str) -> tuple[bool, str]:
    return open_privacy_pane(kind)


def _macos_accessibility_trusted() -> bool:
    return accessibility_trusted()


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
        "pkg_resources": "setuptools",
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

    secret_store = SecretStore()

    if "deepgram" in providers:
        deepgram_status = secret_store.status("deepgram", config)
        if deepgram_status.source != "missing":
            _doctor_line("OK", "Deepgram", f"API key found via {deepgram_status.source}")
        else:
            warnings += 1
            _doctor_line("WARN", "Deepgram", f"API key missing ({deepgram_env})")

    if "openai" in providers:
        openai_status = secret_store.status("openai", config)
        if openai_status.source != "missing":
            _doctor_line("OK", "OpenAI", f"API key found via {openai_status.source}")
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
        legacy_payload = load_personal_dictionary(config)
        legacy_terms = len(legacy_payload.get("preferred_terms", []))
        legacy_corrections = len(legacy_payload.get("corrections", []))
        if legacy_terms or legacy_corrections:
            _doctor_line(
                "OK",
                "Personal Dictionary",
                (
                    f"Found legacy dictionary source(s): {legacy_terms} term(s), "
                    f"{legacy_corrections} replacement(s). Bloviate will import them on launch."
                ),
            )
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
    auto_paste = bool(config.get("transcription", {}).get("auto_paste", True))
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

        if auto_paste or window_management_enabled:
            if accessibility_trusted():
                _doctor_line("OK", "Accessibility", "Current process is trusted for simulated paste/hotkeys")
            else:
                warnings += 1
                _doctor_line(
                    "WARN",
                    "Accessibility",
                    (
                        "Required for auto-paste and some hotkeys. Enable Bloviate in "
                        "Privacy & Security > Accessibility; if macOS lists Python instead, enable Python too."
                    ),
                )
        else:
            _doctor_line("OK", "Accessibility", "Not required by current config")

        warnings += 1
        _doctor_line("WARN", "Microphone Permission", "macOS may prompt on first audio capture")
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
        self.settings_service = SettingsService(self.config)
        self.secret_store = SecretStore()
        self.model_registry = ModelRegistry()
        self.history_store = HistoryStore()
        self.achievement_service = AchievementService(
            self.config,
            secret_store=self.secret_store,
        )
        self.post_processor = PostProcessor(self.config, secret_store=self.secret_store)
        self.verbose_logs = _is_verbose_logging_enabled(self.config)
        self._quiet_startup = not self.verbose_logs
        try:
            dictionary_path, terms, corrections, migrated = migrate_legacy_personal_dictionary(self.config)
            if migrated and self.verbose_logs:
                print(
                    f"[Dictionary] Imported {terms} term(s), {corrections} replacement(s) "
                    f"to {dictionary_path}"
                )
        except Exception as exc:
            print(f"[Dictionary] Legacy dictionary migration skipped: {exc}")

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
        with _suppress_startup_stdio(self._quiet_startup):
            self.voice_fingerprint = VoiceFingerprint(self.config)
        with _suppress_startup_stdio(self._quiet_startup):
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
        self.is_processing_recording = False
        self.is_processing_command = False
        self.ui_window = None
        self.ui_app = None
        self._last_interim_text = ""
        self._last_interim_update = 0.0
        self._last_audio_level_emit = 0.0
        self._last_audio_level_value = 0.0
        self._shutdown_event = threading.Event()
        self._worker_threads = set()
        self._worker_threads_lock = threading.Lock()
        self._enrollment_lock = threading.Lock()
        self._interim_update_interval_s = float(
            self.config.get("ui", {}).get("interim_update_interval_s", 0.15)
        )
        self._audio_level_update_interval_s = max(
            0.04,
            float(self.config.get("ui", {}).get("audio_level_update_interval_s", 0.10)),
        )
        self._audio_level_min_delta = max(
            0.0,
            float(self.config.get("ui", {}).get("audio_level_min_delta", 0.003)),
        )

    def list_audio_input_options(self) -> list[dict]:
        """Return current audio input options for the UI."""
        return self.audio_capture.list_input_devices()

    def set_audio_input_device(self, device_name: str) -> tuple[bool, str]:
        """Switch audio input device and persist the selection."""
        if self.is_recording or self.is_command_recording:
            return False, "Finish the current recording before switching input devices."

        selected = str(device_name or "").strip()
        previous = str(self.config.get("audio", {}).get("device_name", "") or "").strip()
        label = selected or "System Default"

        try:
            self.audio_capture.set_device(selected)
            self.config.setdefault("audio", {})["device_name"] = selected
            saved_path = _save_config(self.config)
            if self.verbose_logs:
                print(f"[Audio] Switched input device to {label}")
                print(f"[Config] Saved audio.device_name to {saved_path}")
            return True, f"Input device set to {label}"
        except Exception as exc:
            print(f"[Audio] Failed to switch input device to {label}: {exc}")
            try:
                self.audio_capture.set_device(previous)
            except Exception as restore_exc:
                print(f"[Audio] Failed to restore previous input device: {restore_exc}")
            return False, f"Could not switch input device: {exc}"

    def get_voice_profile_status(self) -> dict:
        """Return runtime status for voice-mode/profile settings."""
        enrolled = len(self.voice_fingerprint.enrolled_embeddings)
        minimum = int(self.voice_fingerprint.min_enrollment_samples)
        mode = "talk" if self.talk_mode else "whisper"
        return {
            "mode": mode,
            "threshold": float(self.voice_fingerprint.threshold),
            "enrolled_samples": enrolled,
            "min_samples": minimum,
            "is_enrolled": bool(self.voice_fingerprint.is_enrolled()),
            "profile_path": str(self.voice_fingerprint.profile_path),
        }

    def set_voice_mode(self, mode: str) -> tuple[bool, str]:
        """Set whisper/talk mode at runtime and persist config."""
        resolved = self._resolve_voice_mode(mode)
        if resolved == "whisper" and not self.voice_fingerprint.is_enrolled():
            return False, "Whisper mode requires an enrolled voice profile first."

        self.voice_mode = resolved
        self.talk_mode = resolved == "talk"
        self.config.setdefault("voice_fingerprint", {})["mode"] = resolved
        saved_path = _save_config(self.config)
        if self.verbose_logs:
            print(f"[Config] Saved voice_fingerprint.mode to {saved_path}")

        if self.ui_window:
            if self.talk_mode:
                self.ui_window.signals.update_voice_match.emit(True, -1.0)
            else:
                self.ui_window.signals.update_status.emit("Whisper mode enabled")

        label = "Talk mode (verification bypassed)" if self.talk_mode else "Whisper mode (verification enforced)"
        return True, f"Mode set to {label}"

    def set_voice_threshold(self, threshold: float) -> tuple[bool, str]:
        """Update speaker-match threshold and persist config/profile."""
        try:
            value = float(threshold)
        except (TypeError, ValueError):
            return False, "Threshold must be a number between 0.0 and 1.0."

        clamped = max(0.0, min(1.0, value))
        self.voice_fingerprint.threshold = clamped
        self.config.setdefault("voice_fingerprint", {})["threshold"] = clamped
        saved_path = _save_config(self.config)
        if self.verbose_logs:
            print(f"[Config] Saved voice_fingerprint.threshold to {saved_path}")

        if self.voice_fingerprint.enrolled_embeddings:
            self.voice_fingerprint.save_profile()
        return True, f"Voice threshold set to {clamped:.2f}"

    def capture_enrollment_sample(self, duration_s: float = 3.0) -> tuple[bool, str]:
        """Capture one sample from live audio and add it to the voice profile."""
        if self.is_recording or self.is_command_recording:
            return False, "Finish dictation before recording enrollment samples."
        if not self.voice_fingerprint.enabled:
            return False, "Voice fingerprinting is unavailable."
        if self._shutdown_event.is_set():
            return False, "App is shutting down."
        if not self._enrollment_lock.acquire(blocking=False):
            return False, "Enrollment capture already in progress."

        try:
            capture_seconds = max(1.0, float(duration_s))
            self.audio_capture.clear_queue()
            samples = []
            deadline = time.time() + capture_seconds
            while time.time() < deadline:
                chunk = self.audio_capture.get_audio_chunk(timeout=0.12)
                if chunk is not None:
                    samples.append(chunk)

            if not samples:
                return False, "No audio captured. Check your microphone and try again."

            audio = np.concatenate(samples).flatten()
            rms = float(np.sqrt(np.mean(audio ** 2)))
            processed_audio = self.noise_suppressor.process(audio)
            if not self.voice_fingerprint.enroll_sample(processed_audio):
                return False, "Could not extract a voice embedding from this sample."

            self.voice_fingerprint.save_profile()
            enrolled = len(self.voice_fingerprint.enrolled_embeddings)
            minimum = int(self.voice_fingerprint.min_enrollment_samples)
            self._evaluate_achievements()
            if self.voice_fingerprint.is_enrolled():
                return True, f"Captured sample {enrolled}/{minimum} (RMS={rms:.4f}). Profile is ready."
            return True, f"Captured sample {enrolled}/{minimum} (RMS={rms:.4f})."
        finally:
            self._enrollment_lock.release()

    def clear_voice_profile(self) -> tuple[bool, str]:
        """Remove enrolled voice profile samples."""
        if self.is_recording or self.is_command_recording:
            return False, "Finish dictation before clearing the profile."
        self.voice_fingerprint.clear_profile()
        return True, "Voice profile cleared."

    def get_personal_dictionary_path(self) -> str:
        """Return resolved personal dictionary file path."""
        return str(resolve_personal_dictionary_path(self.config))

    def ensure_personal_dictionary_exists(self) -> tuple[bool, str]:
        """Create personal dictionary if missing."""
        path = resolve_personal_dictionary_path(self.config)
        if path.exists():
            return True, f"Personal dictionary exists: {path}"
        result = init_personal_dictionary(self.config)
        if result == 0:
            return True, f"Created personal dictionary: {path}"
        return False, "Could not initialize personal dictionary."

    def open_personal_dictionary(self) -> tuple[bool, str]:
        """Open the personal dictionary in the system editor."""
        ok, message = self.ensure_personal_dictionary_exists()
        if not ok:
            return False, message

        path = resolve_personal_dictionary_path(self.config)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
            return True, f"Opened dictionary: {path}"
        except Exception as exc:
            return False, f"Could not open dictionary: {exc}"

    def reload_personal_dictionary(self) -> tuple[bool, str]:
        """Reload personal dictionary terms in the active transcriber."""
        try:
            stats = self.transcriber.reload_personal_dictionary()
            preferred = int(stats.get("preferred_terms", 0))
            corrections = int(stats.get("corrections", 0))
            return True, f"Reloaded dictionary ({preferred} terms, {corrections} rules)."
        except Exception as exc:
            return False, f"Could not reload dictionary: {exc}"

    def get_personal_dictionary_payload(self) -> dict:
        """Return loaded personal dictionary terms/corrections for editable UI."""
        return load_personal_dictionary(self.config)

    def save_personal_dictionary_payload(
        self, preferred_terms: list[str], corrections: list[dict]
    ) -> tuple[bool, str]:
        """Persist personal dictionary edits and reload the active transcriber."""
        try:
            path = save_personal_dictionary(self.config, preferred_terms, corrections)
            self.transcriber.reload_personal_dictionary()
            self._evaluate_achievements()
            return True, f"Saved dictionary: {path}"
        except Exception as exc:
            return False, f"Could not save dictionary: {exc}"

    def get_model_options(self) -> dict:
        """Return provider/model options for settings UI."""
        return {
            "providers": [provider.__dict__ for provider in self.model_registry.providers()],
            "whisper_models": [model.__dict__ for model in self.model_registry.models_for("whisper")],
            "deepgram_models": [model.__dict__ for model in self.model_registry.models_for("deepgram")],
            "openai_models": [model.__dict__ for model in self.model_registry.models_for("openai")],
            "cleanup_models": [
                model.__dict__
                for model in self.model_registry.models_for("openai", purpose="cleanup")
            ],
            "final_pass_modes": list(self.model_registry.FINAL_PASS_MODES),
            "post_processing_modes": list(self.model_registry.POST_PROCESSING_MODES),
            "output_formats": list(self.model_registry.OUTPUT_FORMATS),
        }

    def get_secret_statuses(self) -> dict:
        """Return API-key source status without exposing key values."""
        return {
            provider: self.secret_store.status(provider, self.config).__dict__
            for provider in ("openai", "deepgram")
        }

    def set_api_key(self, provider: str, value: str) -> tuple[bool, str]:
        """Store or clear an API key in Keychain."""
        ok, message = self.secret_store.set_api_key(provider, value)
        if ok and hasattr(self, "transcriber"):
            # No object rebuild needed: Transcriber resolves keys on demand.
            if self.verbose_logs:
                print(f"[Secrets] {message}")
        return ok, message

    def set_transcription_settings(self, updates: dict) -> tuple[bool, str]:
        """Persist model/provider/output settings and update runtime objects."""
        try:
            if "transcription.provider" in updates:
                updates["transcription.provider"] = self.model_registry.validate_provider(
                    updates["transcription.provider"]
                )
            if "transcription.final_pass_provider_priority" in updates:
                updates["transcription.final_pass_provider_priority"] = (
                    self.model_registry.normalize_provider_priority(
                        updates["transcription.final_pass_provider_priority"]
                    )
                )
            self.settings_service.update_many(updates)
            self._refresh_runtime_config_views()
            return True, "Transcription settings saved."
        except Exception as exc:
            return False, f"Could not save transcription settings: {exc}"

    def set_hotkey_settings(self, updates: dict) -> tuple[bool, str]:
        """Persist hotkeys and restart the global listener when possible."""
        if self.is_recording or self.is_command_recording:
            return False, "Finish the current recording before changing hotkeys."

        old_config = {
            "hotkey": self.config.get("ptt", {}).get("hotkey"),
            "secondary_hotkey": self.config.get("ptt", {}).get("secondary_hotkey"),
            "toggle_hotkey": self.config.get("ptt", {}).get("toggle_hotkey"),
            "mode_cycle_tap_key": self.config.get("ptt", {}).get("mode_cycle_tap_key"),
            "mode_cycle_tap_count": self.config.get("ptt", {}).get("mode_cycle_tap_count"),
            "mode_cycle_tap_window_ms": self.config.get("ptt", {}).get("mode_cycle_tap_window_ms"),
            "command_hotkey": self.config.get("window_management", {}).get("command_hotkey"),
            "hotkey_prefix": self.config.get("window_management", {}).get("hotkey_prefix"),
            "voice_command_prefixes": self.config.get("window_management", {}).get("voice_command_prefixes"),
        }

        try:
            self.settings_service.update_many(updates)
            from ptt_handler import PTTHandler

            candidate = PTTHandler(self.config)
        except Exception as exc:
            # Restore only the keys this method owns.
            self.config.setdefault("ptt", {})["hotkey"] = old_config["hotkey"]
            self.config.setdefault("ptt", {})["secondary_hotkey"] = old_config["secondary_hotkey"]
            self.config.setdefault("ptt", {})["toggle_hotkey"] = old_config["toggle_hotkey"]
            self.config.setdefault("ptt", {})["mode_cycle_tap_key"] = old_config["mode_cycle_tap_key"]
            self.config.setdefault("ptt", {})["mode_cycle_tap_count"] = old_config["mode_cycle_tap_count"]
            self.config.setdefault("ptt", {})["mode_cycle_tap_window_ms"] = old_config[
                "mode_cycle_tap_window_ms"
            ]
            self.config.setdefault("window_management", {})["command_hotkey"] = old_config["command_hotkey"]
            self.config.setdefault("window_management", {})["hotkey_prefix"] = old_config["hotkey_prefix"]
            self.config.setdefault("window_management", {})["voice_command_prefixes"] = old_config[
                "voice_command_prefixes"
            ]
            self.settings_service.save()
            return False, f"Invalid hotkey settings: {exc}"

        listener_was_running = bool(
            getattr(self.ptt_handler, "listener", None)
            or getattr(self.ptt_handler, "_is_started", False)
        )
        if listener_was_running:
            try:
                self.ptt_handler.stop()
            except Exception as exc:
                print(f"Error stopping old PTT handler: {exc}")
        self.ptt_handler = candidate
        self._setup_toggle_hotkey()
        self._setup_mode_cycle_tap()
        if listener_was_running:
            self.ptt_handler.start(
                on_press=self.on_ptt_press,
                on_release=self.on_ptt_release,
            )
            if self.window_manager:
                self._setup_window_management_hotkeys()
        return True, "Hotkeys saved."

    def set_general_settings(self, updates: dict) -> tuple[bool, str]:
        """Persist general UI/output settings."""
        try:
            self.settings_service.update_many(updates)
            self._refresh_runtime_config_views()
            if any(str(key).startswith("ui.easter_eggs.") for key in updates):
                self._evaluate_achievements()
            return True, "General settings saved."
        except Exception as exc:
            return False, f"Could not save general settings: {exc}"

    def _cleanup_mode_label(self, mode: str) -> str:
        labels = {
            "verbatim": "Verbatim",
            "clean": "Clean",
            "coding": "Coding",
            "message": "Message",
        }
        return labels.get(str(mode or "").strip().lower(), str(mode or "").title())

    def cycle_post_processing_mode(self) -> tuple[bool, str]:
        """Cycle output cleanup mode from the global command-key tap gesture."""
        modes = list(self.model_registry.POST_PROCESSING_MODES)
        current = str(
            self.config.get("post_processing", {}).get("mode", "verbatim") or "verbatim"
        ).strip().lower()
        try:
            index = modes.index(current)
        except ValueError:
            index = 0
        next_mode = modes[(index + 1) % len(modes)]
        self.config.setdefault("post_processing", {})["mode"] = next_mode
        saved_path = _save_config(self.config)
        self._refresh_runtime_config_views()

        label = self._cleanup_mode_label(next_mode)
        message = f"Cleanup mode: {label}"
        if self.verbose_logs:
            print(f"[Config] Saved post_processing.mode={next_mode} to {saved_path}")
        if self.ui_window:
            self.ui_window.signals.update_cleanup_mode.emit(next_mode, label)
            self.ui_window.signals.update_status.emit(message)
        return True, message

    def get_history_records(self, query: str = "", limit: int = 100) -> list[dict]:
        """Return recent transcript history as serializable dictionaries."""
        return [record.__dict__ for record in self.history_store.recent(query=query, limit=limit)]

    def get_history_insights(self) -> dict:
        """Return aggregate usage metrics and dictionary context for the settings UI."""
        insights = self.history_store.insights()
        try:
            payload = load_personal_dictionary(self.config)
        except Exception:
            payload = {}
        insights["dictionary_terms"] = len(payload.get("preferred_terms", []) or [])
        insights["dictionary_corrections"] = len(payload.get("corrections", []) or [])
        return insights

    def delete_history_record(self, record_id: int) -> tuple[bool, str]:
        ok = self.history_store.delete(record_id)
        return ok, "Deleted history item." if ok else "History item was not found."

    def clear_history(self) -> tuple[bool, str]:
        count = self.history_store.clear()
        return True, f"Cleared {count} history item(s)."

    def export_history(self, path: str) -> tuple[bool, str]:
        try:
            exported = self.history_store.export_csv(Path(path).expanduser())
            return True, f"Exported history to {exported}"
        except Exception as exc:
            return False, f"Could not export history: {exc}"

    def _achievement_context(self) -> tuple[dict, dict]:
        try:
            dictionary_payload = load_personal_dictionary(self.config)
        except Exception:
            dictionary_payload = {}
        try:
            voice_status = self.get_voice_profile_status()
        except Exception:
            voice_status = {}
        return dictionary_payload, voice_status

    def _evaluate_achievements(self, *, suppress_unlocks: bool = False, show: bool = True) -> list[dict]:
        if not hasattr(self, "achievement_service"):
            return []
        try:
            dictionary_payload, voice_status = self._achievement_context()
            unlocks = self.achievement_service.evaluate(
                dictionary_payload=dictionary_payload,
                voice_profile_status=voice_status,
                suppress_unlocks=suppress_unlocks,
            )
            if unlocks and show:
                self._show_achievement_unlocks(unlocks)
            return unlocks
        except Exception as exc:
            print(f"[Achievements] Evaluation failed: {exc}")
            return []

    def _show_achievement_unlocks(self, unlocks: list[dict]):
        if not unlocks or not self.ui_window:
            return
        celebrations = str(self.config.get("achievements", {}).get("celebrations", "full") or "full")
        if celebrations == "off":
            return
        if hasattr(self.ui_window.signals, "show_achievement_unlocks"):
            self.ui_window.signals.show_achievement_unlocks.emit(unlocks)

    def get_achievement_summary(self, query: str = "", status_filter: str = "all") -> dict:
        """Return achievement progress for Settings."""
        self._evaluate_achievements(show=False)
        dictionary_payload, voice_status = self._achievement_context()
        return self.achievement_service.summary(
            dictionary_payload=dictionary_payload,
            voice_profile_status=voice_status,
            query=query,
            status_filter=status_filter,
        )

    def reset_achievements(self) -> tuple[bool, str]:
        ok, message = self.achievement_service.reset()
        return ok, message

    def set_achievement_settings(self, updates: dict) -> tuple[bool, str]:
        try:
            dotted_updates = {f"achievements.{key}": value for key, value in updates.items()}
            self.settings_service.update_many(dotted_updates)
            self.achievement_service.set_settings(updates)
            return True, "Achievement settings saved."
        except Exception as exc:
            return False, f"Could not save achievement settings: {exc}"

    def analyze_achievement_history(self) -> tuple[bool, str]:
        dictionary_payload, voice_status = self._achievement_context()
        ok, message, unlocks = self.achievement_service.analyze_history(
            dictionary_payload=dictionary_payload,
            voice_profile_status=voice_status,
        )
        if unlocks:
            self._show_achievement_unlocks(unlocks)
        return ok, message

    def backfill_achievements(self):
        dictionary_payload, voice_status = self._achievement_context()
        try:
            unlocks = self.achievement_service.backfill_if_needed(
                dictionary_payload=dictionary_payload,
                voice_profile_status=voice_status,
            )
        except Exception as exc:
            print(f"[Achievements] Backfill failed: {exc}")
            return
        if not unlocks or not self.ui_window:
            return
        summary = dict(unlocks[0])
        summary.update(
            {
                "id": "achievements_backfill_summary",
                "title": "Achievement Shelf Stocked",
                "description": f"{len(unlocks)} achievement(s) unlocked from your existing history.",
                "progress_label": f"{len(unlocks)} unlocked",
                "category": "Achievements",
                "unlocked": True,
            }
        )
        self._show_achievement_unlocks([summary])

    def run_doctor_text(self) -> tuple[bool, str]:
        """Run doctor and capture text for the Advanced settings tab."""
        buffer = io.StringIO()
        config_path = self.config.get("__config_path__", str(default_user_config_path()))
        with contextlib.redirect_stdout(buffer):
            exit_code = run_doctor(str(config_path))
        return exit_code == 0, buffer.getvalue()

    def get_permission_statuses(self) -> dict:
        """Return macOS permission status for the Settings first-run checklist."""
        if sys.platform != "darwin":
            return {
                "platform": {
                    "label": "Platform permissions",
                    "state": "unsupported",
                    "detail": "macOS permission checks are not required here.",
                }
            }

        microphone_state = "granted" if getattr(self.audio_capture, "stream", None) is not None else "unknown"
        microphone_detail = (
            "Audio stream is running."
            if microphone_state == "granted"
            else "Click Request Microphone to trigger the macOS microphone prompt."
        )
        accessibility_granted = _macos_accessibility_trusted()
        auto_paste = bool(self.config.get("transcription", {}).get("auto_paste", True))

        return {
            "microphone": {
                "label": "Microphone",
                "state": microphone_state,
                "detail": microphone_detail,
            },
            "accessibility": {
                "label": "Accessibility",
                "state": "granted" if accessibility_granted else "missing",
                "detail": "Required for global hotkeys and simulated paste.",
            },
            "input_monitoring": {
                "label": "Input Monitoring",
                "state": "manual",
                "detail": "Required by macOS for reliable global hotkey capture.",
            },
            "automation": {
                "label": "Automation",
                "state": "manual" if auto_paste else "unsupported",
                "detail": "Required for AppleScript paste when auto-paste is enabled.",
            },
        }

    def request_permission(self, kind: str) -> tuple[bool, str]:
        """Trigger the relevant macOS permission prompt or open its Settings pane."""
        normalized = str(kind or "").strip().lower()
        if sys.platform != "darwin":
            return True, "No macOS permissions are required on this platform."

        if normalized == "microphone":
            try:
                self.audio_capture.start()
                return True, "Microphone permission is ready; audio capture started."
            except Exception as exc:
                _open_macos_privacy_pane("microphone")
                return False, f"Microphone access is not ready: {exc}"

        if normalized == "accessibility":
            if accessibility_trusted():
                return True, "Accessibility permission is already granted."
            if request_accessibility():
                return True, "Accessibility permission is ready."
            return True, (
                "Opened Accessibility settings. Enable Bloviate there. "
                "If macOS shows Python instead, enable Python too."
            )

        if normalized in {"input_monitoring", "automation"}:
            return _open_macos_privacy_pane(normalized)

        return _open_macos_privacy_pane("privacy")

    def reset_settings_to_defaults(self) -> tuple[bool, str]:
        """Reset YAML config to packaged defaults while preserving runtime metadata."""
        try:
            default_config = yaml.safe_load(read_resource_text("default_config.yaml")) or {}
            metadata = {
                "__config_path__": self.config.get("__config_path__"),
                "__config_dir__": self.config.get("__config_dir__"),
            }
            self.config.clear()
            self.config.update(default_config)
            for key, value in metadata.items():
                if value:
                    self.config[key] = value
            self.settings_service = SettingsService(self.config)
            self.settings_service.save()
            self._refresh_runtime_config_views()
            return True, "Settings reset to packaged defaults."
        except Exception as exc:
            return False, f"Could not reset settings: {exc}"

    def _refresh_runtime_config_views(self):
        """Apply changed config values to long-lived runtime objects."""
        if hasattr(self, "transcriber"):
            tx_cfg = self.config.get("transcription", {})
            self.transcriber.transcription_config = tx_cfg
            self.transcriber.provider = self.transcriber._normalize_provider_name(
                tx_cfg.get("provider", "whisper")
            ) or "whisper"
            self.transcriber.language = tx_cfg.get("language", self.transcriber.language)
            self.transcriber.output_format = tx_cfg.get("output_format", "clipboard")
            self.transcriber.auto_paste = bool(tx_cfg.get("auto_paste", True))
            self.transcriber.use_custom_dictionary = bool(tx_cfg.get("use_custom_dictionary", True))
            self.transcriber.deepgram_config = self.config.get("deepgram", {})
            self.transcriber.openai_config = self.config.get("openai", {})
            self.transcriber.deepgram_streaming = bool(
                self.transcriber.deepgram_config.get("streaming", True)
            )
            self.transcriber.model_name = tx_cfg.get("model", self.transcriber.model_name)
            if self.transcriber.provider in {"deepgram", "openai"}:
                self.transcriber.model_name = tx_cfg.get(
                    "whisper_fallback_model", self.transcriber.model_name
                )
            self.transcriber.reload_personal_dictionary(log_on_success=False)
        if hasattr(self, "noise_suppressor"):
            ns_cfg = self.config.get("noise_suppression", {})
            self.noise_suppressor.enabled = bool(ns_cfg.get("enabled", True))
            self.noise_suppressor.stationary_reduction = ns_cfg.get(
                "stationary_noise_reduction",
                self.noise_suppressor.stationary_reduction,
            )
            self.noise_suppressor.vad_aggressiveness = ns_cfg.get(
                "vad_aggressiveness",
                self.noise_suppressor.vad_aggressiveness,
            )
        if hasattr(self, "post_processor"):
            self.post_processor.config = self.config

    def set_show_main_window_on_startup(self, enabled: bool) -> tuple[bool, str]:
        """Persist whether the main window should be shown on startup."""
        value = bool(enabled)
        self.config.setdefault("ui", {})["show_main_window"] = value
        saved_path = _save_config(self.config)
        if self.verbose_logs:
            print(f"[Config] Saved ui.show_main_window to {saved_path}")
        return True, f"Startup window is now {'enabled' if value else 'hidden'}."

    def set_startup_splash_enabled(self, enabled: bool) -> tuple[bool, str]:
        """Persist startup splash preference."""
        value = bool(enabled)
        ui_cfg = self.config.setdefault("ui", {})
        splash_cfg = ui_cfg.setdefault("startup_splash", {})
        splash_cfg["enabled"] = value
        saved_path = _save_config(self.config)
        if self.verbose_logs:
            print(f"[Config] Saved ui.startup_splash.enabled to {saved_path}")
        return True, f"Startup splash {'enabled' if value else 'disabled'}."

    def set_terminal_startup_animation_enabled(self, enabled: bool) -> tuple[bool, str]:
        """Persist terminal startup animation preference."""
        value = bool(enabled)
        self.config.setdefault("app", {})["startup_animation"] = value
        saved_path = _save_config(self.config)
        if self.verbose_logs:
            print(f"[Config] Saved app.startup_animation to {saved_path}")
        return True, f"Terminal startup animation {'enabled' if value else 'disabled'}."

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
        if self.is_recording:
            return
        if self.is_processing_recording or self.is_processing_command:
            print("[PTT] Ignored: previous clip is still processing")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("Still processing previous clip...")
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
        if not self.is_recording:
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
            self.is_processing_recording = True
            self._start_worker(self._process_recording_worker, recorded)
        else:
            if self.transcriber.supports_streaming():
                self.transcriber.finish_stream("dictation")
            print("No audio recorded")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("No audio recorded")

    def toggle_ptt_recording(self):
        """Toggle dictation recording without requiring the PTT keys to be held."""
        if (
            self._shutdown_event.is_set()
            or self.is_command_recording
            or self.is_processing_recording
            or self.is_processing_command
        ):
            if self.ui_window and (self.is_processing_recording or self.is_processing_command):
                self.ui_window.signals.update_status.emit("Still processing previous clip...")
            return
        if self.is_recording:
            print("[PTT] Toggle off")
            self.on_ptt_release()
        else:
            print("[PTT] Toggle on")
            self.on_ptt_press()

    def _process_recording_worker(self, recorded_chunks):
        try:
            self.process_recording(recorded_chunks)
        finally:
            self.is_processing_recording = False

    def _setup_toggle_hotkey(self):
        """Register the optional toggle-dictation hotkey."""
        toggle_hotkey = str(
            self.config.get("ptt", {}).get("toggle_hotkey", "<cmd>+<option>+<shift>") or ""
        ).strip()
        if not toggle_hotkey:
            return
        self.ptt_handler.add_hotkey(
            "toggle_dictation",
            toggle_hotkey,
            on_press=self.toggle_ptt_recording,
            match_exact=True,
            consume=True,
        )

    def _setup_mode_cycle_tap(self):
        """Register the quick command-key tap gesture for cleanup-mode cycling."""
        ptt_cfg = self.config.get("ptt", {})
        tap_key = str(ptt_cfg.get("mode_cycle_tap_key", "<cmd>") or "").strip()
        if not tap_key:
            return
        try:
            tap_count = int(ptt_cfg.get("mode_cycle_tap_count", 3))
        except (TypeError, ValueError):
            tap_count = 3
        try:
            window_ms = int(ptt_cfg.get("mode_cycle_tap_window_ms", 650))
        except (TypeError, ValueError):
            window_ms = 650
        self.ptt_handler.add_tap_sequence(
            "cycle_cleanup_mode",
            tap_key,
            count=max(2, tap_count),
            max_interval_s=max(0.2, window_ms / 1000.0),
            callback=self.cycle_post_processing_mode,
        )

    def on_command_press(self):
        """Called when command mode PTT is activated."""
        if self._shutdown_event.is_set():
            return
        if self.is_command_recording:
            return
        if self.is_recording or self.is_processing_recording or self.is_processing_command:
            print("[CMD] Ignored: dictation is still active or processing")
            if self.ui_window:
                self.ui_window.signals.update_command_status.emit(
                    "CMD: Busy processing previous clip",
                    "processing",
                )
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
        if not self.is_command_recording:
            return
        print("[CMD] Released")

        if self.ui_window:
            self.ui_window.signals.update_command_status.emit("CMD: Processing...", "processing")

        self.is_command_recording = False

        # Capture audio and process in background
        recorded = self.recorded_command_audio
        self.recorded_command_audio = []
        if len(recorded) > 0:
            self.is_processing_command = True
            self._start_worker(self._process_command_recording_worker, recorded)
        else:
            if self.transcriber.supports_streaming():
                self.transcriber.finish_stream("command")
            print("[CMD] No audio recorded")
            if self.ui_window:
                self.ui_window.signals.update_command_status.emit("CMD: No audio recorded", "unrecognized")

    def _process_command_recording_worker(self, recorded_chunks):
        try:
            self.process_command_recording(recorded_chunks)
        finally:
            self.is_processing_command = False

    def _normalize_command_text(self, text: str) -> str:
        """Normalize text for command matching."""
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        return f" {normalized} " if normalized else ""

    def _command_key(self, text: str) -> str:
        """Normalize command text without padding."""
        return self._normalize_command_text(text).strip()

    def _parse_window_command(self, text: str) -> Optional[str]:
        """Parse the transcribed text into a window command."""
        normalized = self._command_key(text)
        if not normalized:
            return None

        for phrase, position in sorted_aliases(WINDOW_COMMAND_ALIASES):
            if phrase == normalized:
                return position

        return None

    def _app_command_aliases(self) -> dict[str, str]:
        """Return normalized app launch aliases."""
        aliases = {
            "arc": "Arc",
            "calendar": "Calendar",
            "chat gpt": "ChatGPT",
            "chatgpt": "ChatGPT",
            "chrome": "Google Chrome",
            "claude": "Claude",
            "code": "Visual Studio Code",
            "cursor": "Cursor",
            "discord": "Discord",
            "facetime": "FaceTime",
            "figma": "Figma",
            "finder": "Finder",
            "firefox": "Firefox",
            "google chrome": "Google Chrome",
            "i term": "iTerm",
            "iterm": "iTerm",
            "mail": "Mail",
            "messages": "Messages",
            "microsoft teams": "Microsoft Teams",
            "notes": "Notes",
            "notion": "Notion",
            "preview": "Preview",
            "reminders": "Reminders",
            "safari": "Safari",
            "slack": "Slack",
            "spotify": "Spotify",
            "teams": "Microsoft Teams",
            "terminal": "Terminal",
            "text edit": "TextEdit",
            "textedit": "TextEdit",
            "visual studio code": "Visual Studio Code",
            "vs code": "Visual Studio Code",
            "zoom": "zoom.us",
        }
        configured = self.config.get("window_management", {}).get("app_aliases", {})
        if isinstance(configured, dict):
            for alias, app_name in configured.items():
                normalized = self._command_key(str(alias))
                if normalized and app_name:
                    aliases[normalized] = str(app_name).strip()
        return aliases

    def _parse_app_command(self, text: str) -> tuple[Optional[str], str]:
        """Parse isolated app-launch commands such as 'open Slack'."""
        normalized = self._command_key(text)
        if not normalized:
            return None, ""

        prefixes = (
            "open the app",
            "launch the app",
            "start the app",
            "open app",
            "launch app",
            "start app",
            "open",
            "launch",
            "start",
        )
        for prefix in prefixes:
            prefix_with_space = f"{prefix} "
            if not normalized.startswith(prefix_with_space):
                continue
            requested = normalized[len(prefix_with_space):].strip()
            if not requested:
                return None, ""
            app_name = self._app_command_aliases().get(requested)
            if app_name:
                return f"open_app:{app_name}", f"{prefix} {requested}"
        return None, ""

    def _parse_easter_egg_command(self, text: str) -> tuple[Optional[str], str]:
        """Parse isolated non-output Easter egg commands."""
        normalized = self._command_key(text)
        if not normalized:
            return None, ""
        command_map = {
            "bloviate surprise me": "easter:surprise",
            "surprise me bloviate": "easter:surprise",
            "show the cows": "easter:cows",
            "run the cows": "easter:cows",
            "bloviate show the cows": "easter:cows",
            "bloviate run the cows": "easter:cows",
            "activate lounge mode": "easter:theme:lounge",
            "lounge mode": "easter:theme:lounge",
            "activate terminal cow": "easter:theme:terminal_cow",
            "terminal cow mode": "easter:theme:terminal_cow",
            "activate after dark": "easter:theme:after_dark",
            "dictation after dark": "easter:theme:after_dark",
            "activate studio radio": "easter:theme:studio_radio",
            "studio radio mode": "easter:theme:studio_radio",
            "open bloviate labs": "easter:about",
            "show bloviate labs": "easter:about",
        }
        return command_map.get(normalized), normalized

    def _voice_command_prefixes(self) -> list[str]:
        """Return normalized phrase prefixes that turn dictation into commands."""
        configured = self.config.get("window_management", {}).get(
            "voice_command_prefixes",
            ["run command", "screen", "window", "desktop"],
        )
        if isinstance(configured, str):
            configured = [configured]
        prefixes = []
        seen = set()
        for raw_prefix in configured or []:
            normalized = self._normalize_command_text(str(raw_prefix)).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            prefixes.append(normalized)
        return prefixes or ["run command", "screen", "window", "desktop"]

    def _parse_prefixed_voice_command(self, text: str) -> tuple[Optional[str], str]:
        """Parse isolated dictation commands such as 'screen left half'."""
        normalized = self._command_key(text)
        if not normalized:
            return None, ""

        prefixes = self._voice_command_prefixes()
        window_suffixes = sorted_aliases(WINDOW_PREFIX_SUFFIXES)
        desktop_suffixes = sorted_aliases(DESKTOP_PREFIX_SUFFIXES)

        def matches(phrase: str) -> bool:
            return phrase == normalized

        for prefix in prefixes:
            if prefix == "desktop":
                for suffix, command in desktop_suffixes:
                    phrase = f"desktop {suffix}"
                    if matches(phrase):
                        return command, phrase
                continue

            for suffix, command in window_suffixes:
                phrase = f"{prefix} {suffix}"
                if matches(phrase):
                    return command, phrase

            if prefix in {"run command", "command"}:
                for suffix, command in window_suffixes:
                    for scope in ("screen", "window"):
                        phrase = f"{prefix} {scope} {suffix}"
                        if matches(phrase):
                            return command, phrase
                for suffix, command in desktop_suffixes:
                    phrase = f"{prefix} desktop {suffix}"
                    if matches(phrase):
                        return command, phrase

        return None, ""

    def _try_voice_command(self, text: str) -> bool:
        """Check if dictated text contains a voice command (window/desktop).

        Looks for configured command prefixes such as 'run command',
        'screen', 'window', and 'desktop'.
        Returns True if a command was found and executed.
        """
        command, phrase = self._parse_easter_egg_command(text)
        if command:
            print(f"[VOICE CMD] Matched '{phrase}' → {command}")
            self._execute_voice_command(command, text, status_prefix="Voice")
            return True

        if not self.window_manager:
            return False

        command, phrase = self._parse_app_command(text)
        if not command:
            command, phrase = self._parse_prefixed_voice_command(text)
        if command:
            print(f"[VOICE CMD] Matched '{phrase}' → {command}")
            self._execute_voice_command(command, text, status_prefix="Voice")
            return True

        return False

    def _command_display_label(self, command: str) -> str:
        if command.startswith("easter:theme:"):
            return command.split(":", 2)[2].replace("_", " ").title()
        if command == "easter:cows":
            return "Run Cows"
        if command == "easter:surprise":
            return "Surprise"
        if command == "easter:about":
            return "Bloviate Labs"
        if command.startswith("open_app:"):
            return f"Open {command.split(':', 1)[1]}"
        return command.replace("_", " ").title()

    def _execute_voice_command(self, command: str, original_text: str, status_prefix: str = "Voice"):
        """Execute a voice command and update UI."""
        if command.startswith("easter:"):
            self._execute_easter_egg_command(command)
        elif command.startswith("open_app:"):
            self.window_manager.open_application(command.split(":", 1)[1])
        elif command.startswith("desktop_"):
            direction = command.replace("desktop_", "")
            self.window_manager.switch_desktop(direction)
        else:
            self.window_manager.resize_focused_window(command)

        if status_prefix and self.ui_window:
            self.ui_window.signals.update_command_status.emit(
                f"{status_prefix}: {self._command_display_label(command)}",
                "recognized"
            )
            self.ui_window.signals.update_status.emit("Ready")

    def _easter_config(self) -> dict:
        ui_config = self.config.setdefault("ui", {})
        easter_config = ui_config.setdefault("easter_eggs", {})
        if not isinstance(easter_config, dict):
            easter_config = {}
            ui_config["easter_eggs"] = easter_config
        return easter_config

    def _easter_enabled(self) -> bool:
        return bool(self._easter_config().get("enabled", True))

    def _set_easter_value(self, key: str, value):
        self._easter_config()[key] = value
        self.settings_service.update_many({f"ui.easter_eggs.{key}": value})
        self._refresh_runtime_config_views()
        self._evaluate_achievements()

    def _increment_easter_counter(self, key: str):
        current = int(self._easter_config().get(key, 0) or 0)
        self._set_easter_value(key, current + 1)

    def _execute_easter_egg_command(self, command: str):
        """Run a non-output Easter egg command."""
        if not self._easter_enabled():
            return
        if command == "easter:surprise":
            if self.ui_window and hasattr(self.ui_window, "surprise_waveform"):
                self.ui_window.surprise_waveform()
            else:
                self._increment_easter_counter("surprise_count")
        elif command == "easter:cows":
            if self.ui_window and hasattr(self.ui_window, "run_cow_runway"):
                self.ui_window.run_cow_runway()
            else:
                self._increment_easter_counter("cow_runs")
        elif command == "easter:about":
            if self.ui_window and hasattr(self.ui_window, "show_bloviate_labs"):
                self.ui_window.show_bloviate_labs()
            else:
                self._set_easter_value("about_opened", True)
        elif command.startswith("easter:theme:"):
            theme_id = command.split(":", 2)[2]
            if self.ui_window and hasattr(self.ui_window, "activate_easter_theme"):
                self.ui_window.activate_easter_theme(theme_id)
            else:
                self._set_easter_value("secret_themes_unlocked", True)
                self._increment_easter_counter("secret_theme_activations")
                self.settings_service.update_many({"ui.theme": theme_id})
                self._refresh_runtime_config_views()

    def _active_target_context(self) -> dict:
        """Best-effort active app/window metadata for local history."""
        if sys.platform != "darwin":
            return {"target_app": "", "target_window": ""}

        script = '''
        tell application "System Events"
            set appName to name of first application process whose frontmost is true
            set windowName to ""
            try
                set windowName to name of front window of first application process whose frontmost is true
            end try
            return appName & "\n" & windowName
        end tell
        '''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            if result.returncode != 0:
                return {"target_app": "", "target_window": ""}
            lines = result.stdout.splitlines()
            return {
                "target_app": lines[0].strip() if lines else "",
                "target_window": lines[1].strip() if len(lines) > 1 else "",
            }
        except Exception:
            return {"target_app": "", "target_window": ""}

    def _record_history(
        self,
        *,
        text: str,
        original_text: str = "",
        mode: str = "dictation",
        post_processing_mode: str = "verbatim",
        provider: str = "",
        voice_score: Optional[float] = None,
        duration_s: Optional[float] = None,
        output_action: str = "",
    ) -> Optional[int]:
        if not bool(self.config.get("history", {}).get("enabled", True)):
            return None
        try:
            context = self._active_target_context()
            record_id = self.history_store.add_transcript(
                text=text,
                original_text=original_text or text,
                mode=mode,
                post_processing_mode=post_processing_mode,
                provider=provider,
                voice_score=voice_score,
                duration_s=duration_s,
                audio_device=self.audio_capture.get_active_device_label(),
                target_app=context.get("target_app", ""),
                target_window=context.get("target_window", ""),
                output_action=output_action,
            )
            self._evaluate_achievements()
            return record_id
        except Exception as exc:
            print(f"[History] Could not record transcript: {exc}")
            return None

    def _transcribe_dictation_audio(self, audio_for_transcription: np.ndarray, stream_text: Optional[str]):
        """Return final dictation text/provider for accepted or rejected voice clips."""
        transcription_cfg = self.config.get("transcription", {})
        final_pass_mode = str(transcription_cfg.get("final_pass", "hybrid")).strip().lower()
        if final_pass_mode not in {"hybrid", "prerecorded", "streaming"}:
            print(f"Unknown transcription.final_pass '{final_pass_mode}', defaulting to hybrid")
            final_pass_mode = "hybrid"

        final_provider_order = self.transcriber.get_final_pass_provider_priority()
        provider_used = "deepgram_streaming" if stream_text else ""
        if not stream_text and not self.noise_suppressor.has_speech(audio_for_transcription):
            print("[Audio] Skipping final-pass transcription: clip does not contain enough speech")
            return None, provider_used

        if final_pass_mode == "streaming":
            text = stream_text
            if not text:
                print("[Final] Streaming transcript unavailable, trying final-pass providers")
                text, provider_used = self.transcriber.transcribe_with_priority(
                    audio_for_transcription, final_provider_order, mode="dictation"
                )
                if provider_used:
                    print(f"[Final] Used provider: {provider_used}")
            return text, provider_used

        text = None
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
        return text, provider_used

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
        provider_used = "deepgram_streaming" if stream_text else ""
        if stream_text:
            command, _ = self._parse_app_command(stream_text)
            if not command:
                command, _ = self._parse_prefixed_voice_command(stream_text)
            if not command:
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
                command, _ = self._parse_easter_egg_command(final_text)
                if not command:
                    command, _ = self._parse_app_command(final_text)
                if not command:
                    command, _ = self._parse_prefixed_voice_command(final_text)
                if not command:
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
        self._record_history(
            text=text,
            original_text=text,
            mode="command",
            post_processing_mode="verbatim",
            provider=provider_used,
            duration_s=len(audio) / self.config["audio"]["sample_rate"],
            output_action="command",
        )

        if command:
            print(f"[CMD] Recognized command: {command}")
            if command.startswith("easter:") or self.window_manager:
                self._execute_voice_command(command, text, status_prefix="")
            if self.ui_window:
                self.ui_window.signals.update_command_status.emit(
                    f"CMD: {self._command_display_label(command)} (recognized)",
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
        raw_speech = self.noise_suppressor.speech_stats(raw_audio)
        processed_speech = self.noise_suppressor.speech_stats(audio_for_transcription)
        print(
            "[Audio] "
            f"raw_rms={raw_speech['rms']:.5f}, "
            f"raw_speech={raw_speech['speech_frames']}/{raw_speech['frames']} "
            f"({raw_speech['speech_ratio']:.2%}), "
            f"processed_rms={processed_speech['rms']:.5f}, "
            f"processed_speech={processed_speech['speech_frames']}/{processed_speech['frames']} "
            f"({processed_speech['speech_ratio']:.2%})"
        )
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
                self.ui_window.signals.update_status.emit("Voice rejected; saving transcript to history...")
            text, provider_used = self._transcribe_dictation_audio(audio_for_transcription, stream_text)
            if self._shutdown_event.is_set():
                return
            if text:
                context = self._active_target_context()
                processed = self.post_processor.process(
                    text,
                    target_app=context.get("target_app", ""),
                )
                output_text = processed.text
                if processed.changed:
                    print(f"[Post-processing] rejected {processed.mode}/{processed.provider}: {output_text}")
                self._record_history(
                    text=output_text,
                    original_text=processed.original_text,
                    mode="dictation_rejected",
                    post_processing_mode=processed.mode,
                    provider=provider_used,
                    voice_score=similarity,
                    duration_s=len(raw_audio) / self.config["audio"]["sample_rate"],
                    output_action="voice_rejected_history_only",
                )
                if self.ui_window:
                    self.ui_window.signals.update_rejected_transcription.emit(output_text)
                    self.ui_window.signals.update_status.emit("Voice rejected; transcript saved to history")
            else:
                if self.ui_window:
                    self.ui_window.signals.update_status.emit("Voice rejected")
            return

        # Transcribe (use streaming result if available)
        if self.ui_window:
            self.ui_window.signals.update_status.emit("Transcribing...")

        text, provider_used = self._transcribe_dictation_audio(audio_for_transcription, stream_text)

        if text:
            print(f"✓ Transcribed: {text}")
            if self._shutdown_event.is_set():
                return

            # Check if text contains a voice command
            if self._try_voice_command(text):
                return

            context = self._active_target_context()
            processed = self.post_processor.process(
                text,
                target_app=context.get("target_app", ""),
            )
            output_text = processed.text
            if processed.changed:
                print(f"[Post-processing] {processed.mode}/{processed.provider}: {output_text}")

            self.transcriber.output_text(output_text)
            self._record_history(
                text=output_text,
                original_text=processed.original_text,
                mode="dictation",
                post_processing_mode=processed.mode,
                provider=provider_used,
                voice_score=None if similarity < 0 else similarity,
                duration_s=len(raw_audio) / self.config["audio"]["sample_rate"],
                output_action=self.config.get("transcription", {}).get("output_format", "clipboard"),
            )

            if self.ui_window:
                self.ui_window.signals.update_transcription.emit(output_text)
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
            now = time.monotonic()
            interval = 0.05 if (self.is_recording or self.is_command_recording) else self._audio_level_update_interval_s
            level_delta = abs(level - self._last_audio_level_value)
            if (
                now - self._last_audio_level_emit >= interval
                and (
                    self.is_recording
                    or self.is_command_recording
                    or level_delta >= self._audio_level_min_delta
                    or now - self._last_audio_level_emit >= 0.5
                )
            ):
                self._last_audio_level_emit = now
                self._last_audio_level_value = level
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

        _show_startup_animation(self.config)

        print("\n=== Bloviate ===")
        hotkey_label = (
            ", ".join(self.ptt_handler.hotkey_strs)
            if len(self.ptt_handler.hotkey_strs) > 1
            else self.ptt_handler.hotkey_str
        )
        input_label = self.audio_capture.get_active_device_label()
        mode_label = "Talk mode (verification off)" if self.talk_mode else "Whisper mode (verification on)"
        print(f"Ready • Hold {hotkey_label} to dictate")
        print(f"Input • {input_label}")
        print(f"Mode  • {mode_label}")
        print("Press Ctrl+C to exit.\n")

        # Create UI
        from ui import create_ui

        self.ui_app, self.ui_window = create_ui(
            self.config,
            get_audio_inputs=self.list_audio_input_options,
            set_audio_input=self.set_audio_input_device,
            get_voice_profile_status=self.get_voice_profile_status,
            set_voice_mode=self.set_voice_mode,
            set_voice_threshold=self.set_voice_threshold,
            capture_enrollment_sample=self.capture_enrollment_sample,
            clear_voice_profile=self.clear_voice_profile,
            get_personal_dictionary_path=self.get_personal_dictionary_path,
            ensure_personal_dictionary_exists=self.ensure_personal_dictionary_exists,
            open_personal_dictionary=self.open_personal_dictionary,
            reload_personal_dictionary=self.reload_personal_dictionary,
            get_personal_dictionary_payload=self.get_personal_dictionary_payload,
            save_personal_dictionary_payload=self.save_personal_dictionary_payload,
            get_model_options=self.get_model_options,
            get_secret_statuses=self.get_secret_statuses,
            set_api_key=self.set_api_key,
            set_transcription_settings=self.set_transcription_settings,
            set_hotkey_settings=self.set_hotkey_settings,
            set_general_settings=self.set_general_settings,
            toggle_dictation=self.toggle_ptt_recording,
            get_history_records=self.get_history_records,
            get_history_insights=self.get_history_insights,
            delete_history_record=self.delete_history_record,
            clear_history=self.clear_history,
            export_history=self.export_history,
            get_achievement_summary=self.get_achievement_summary,
            reset_achievements=self.reset_achievements,
            set_achievement_settings=self.set_achievement_settings,
            analyze_achievement_history=self.analyze_achievement_history,
            run_doctor_text=self.run_doctor_text,
            reset_settings_to_defaults=self.reset_settings_to_defaults,
            get_permission_statuses=self.get_permission_statuses,
            request_permission=self.request_permission,
            open_permission_settings=self.request_permission,
            set_show_main_window_on_startup=self.set_show_main_window_on_startup,
            set_startup_splash_enabled=self.set_startup_splash_enabled,
            set_terminal_startup_animation_enabled=self.set_terminal_startup_animation_enabled,
        )
        if self.talk_mode and self.ui_window:
            self.ui_window.signals.update_voice_match.emit(True, -1.0)
        self.backfill_achievements()

        self.audio_capture.register_callback(self.audio_callback)
        try:
            self.audio_capture.start()
        except Exception as exc:
            print(f"[Permissions] Audio capture could not start yet: {exc}")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("Microphone permission needed")

        # Start PTT handler
        self._setup_toggle_hotkey()
        self._setup_mode_cycle_tap()
        try:
            self.ptt_handler.start(
                on_press=self.on_ptt_press,
                on_release=self.on_ptt_release
            )
        except Exception as exc:
            print(f"[Permissions] Global hotkeys could not start yet: {exc}")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("Hotkey permission needed")

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


def install_macos_launcher() -> int:
    """Create a small .app wrapper around the current bloviate command."""
    if sys.platform != "darwin":
        print("--install-launcher is only supported on macOS.")
        return 1

    command = shutil.which("bloviate") or sys.argv[0]
    command_path = Path(command).expanduser()
    if not command_path.is_absolute():
        command_path = (Path.cwd() / command_path).resolve()

    app_dir = Path.home() / "Applications" / "Bloviate.app"
    contents_dir = app_dir / "Contents"
    macos_dir = contents_dir / "MacOS"
    macos_dir.mkdir(parents=True, exist_ok=True)

    executable = macos_dir / "Bloviate"
    launch_script = f"""#!/bin/zsh
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export PYTHONUNBUFFERED=1
log_dir="$HOME/Library/Application Support/Bloviate/logs"
mkdir -p "$log_dir"
exec >>"$log_dir/launcher.log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching Bloviate via {shlex.quote(str(command_path))}"
for rc in "$HOME/.zshenv" "$HOME/.zprofile"; do
  if [ -r "$rc" ]; then
    source "$rc" >/dev/null 2>&1
  fi
done
exec {shlex.quote(str(command_path))} "$@"
"""
    executable.write_text(launch_script, encoding="utf-8")
    executable.chmod(0o755)

    plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>Bloviate</string>
  <key>CFBundleIdentifier</key>
  <string>com.callumreid.bloviate</string>
  <key>CFBundleName</key>
  <string>Bloviate</string>
  <key>CFBundleDisplayName</key>
  <string>Bloviate</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.3.4</string>
  <key>CFBundleVersion</key>
  <string>0.3.4</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>Bloviate uses the microphone to capture push-to-talk dictation.</string>
  <key>NSAppleEventsUsageDescription</key>
  <string>Bloviate uses Apple Events to identify the active window and paste transcribed text.</string>
</dict>
</plist>
"""
    (contents_dir / "Info.plist").write_text(plist, encoding="utf-8")
    (contents_dir / "PkgInfo").write_text("APPL????", encoding="utf-8")

    print(f"Installed launcher: {app_dir}")
    print(f"Target command: {command_path}")
    print(f"Open it with: open {shlex.quote(str(app_dir))}")
    return 0


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
        '--verbose',
        action='store_true',
        help='Enable verbose startup/runtime logs (developer mode)'
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
    parser.add_argument(
        '--install-launcher',
        action='store_true',
        help='Create ~/Applications/Bloviate.app so Bloviate can launch without a terminal'
    )

    args = parser.parse_args()

    if args.install_launcher:
        sys.exit(install_macos_launcher())

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

    if args.verbose:
        config.setdefault("app", {})["verbose_logs"] = True

    _configure_runtime_output(config)

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
