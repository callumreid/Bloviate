#!/usr/bin/env python3
"""
Bloviate - Voice-fingerprinting dictation tool for whispering in noisy environments.

Main application entry point.
"""

import argparse
import yaml
import sys
import threading
import time
import re
import numpy as np
from pathlib import Path
from typing import Optional

from audio_capture import AudioCapture
from noise_suppressor import NoiseSuppressor
from voice_fingerprint import VoiceFingerprint
from ptt_handler import PTTHandler
from transcriber import Transcriber
from ui import create_ui
from window_manager import WindowManager


class Bloviate:
    """Main application class."""

    def __init__(self, config_path: str = "config.yaml", voice_mode_override: Optional[str] = None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.voice_mode = self._resolve_voice_mode(voice_mode_override)
        self.talk_mode = self.voice_mode == "talk"

        # Initialize components
        self.audio_capture = AudioCapture(self.config)
        self.noise_suppressor = NoiseSuppressor(self.config)
        self.voice_fingerprint = VoiceFingerprint(self.config)
        self.transcriber = Transcriber(self.config)
        self.ptt_handler = PTTHandler(self.config)

        # Window management
        self.window_manager = None
        if self.config.get('window_management', {}).get('enabled', False):
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
        self._interim_update_interval_s = float(
            self.config.get("ui", {}).get("interim_update_interval_s", 0.15)
        )

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
        print("\n=== Voice Enrollment ===")
        print(f"You need to record {self.voice_fingerprint.min_enrollment_samples} samples of your whisper voice.")
        print("Press Enter when ready to record each sample, then whisper a phrase.\n")

        self.audio_capture.start()

        phrases = [
            "This is my voice for enrollment",
            "I am enrolling my whisper voice",
            "Bloviate should recognize my voice",
            "Testing voice fingerprint enrollment",
            "Final sample for voice enrollment"
        ]

        for i in range(self.voice_fingerprint.min_enrollment_samples):
            phrase = phrases[i] if i < len(phrases) else "Another enrollment sample"

            input(f"\nSample {i+1}/{self.voice_fingerprint.min_enrollment_samples}")
            print(f"Please whisper: '{phrase}'")
            print("Recording for 3 seconds...")

            # Record for 3 seconds
            samples = []
            start_time = time.time()
            while time.time() - start_time < 3.0:
                chunk = self.audio_capture.get_audio_chunk(timeout=0.5)
                if chunk is not None:
                    samples.append(chunk)

            if len(samples) == 0:
                print("No audio captured. Please try again.")
                continue

            # Concatenate audio
            audio = np.concatenate(samples).flatten()

            # Apply noise suppression
            audio = self.noise_suppressor.process(audio)

            # Enroll sample
            success = self.voice_fingerprint.enroll_sample(audio)

            if success:
                print(f"✓ Sample {i+1} enrolled successfully")
            else:
                print(f"✗ Failed to enroll sample {i+1}")

        self.audio_capture.stop()

        # Save profile
        if self.voice_fingerprint.is_enrolled():
            self.voice_fingerprint.save_profile()
            print("\n✓ Voice enrollment complete!")
            print("You can now run Bloviate normally.")
        else:
            print("\n✗ Voice enrollment failed. Please try again.")

    def on_ptt_press(self):
        """Called when PTT is activated."""
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
            threading.Thread(
                target=self.transcriber.start_stream,
                args=("dictation",),
                daemon=True,
            ).start()

    def on_ptt_release(self):
        """Called when PTT is released."""
        print("[PTT] Released")

        if self.ui_window:
            self.ui_window.signals.update_ptt_status.emit(False)
            self.ui_window.signals.update_status.emit("Processing...")

        self.is_recording = False

        # Capture audio and process in background to keep PTT responsive
        recorded = self.recorded_audio
        self.recorded_audio = []
        if len(recorded) > 0:
            threading.Thread(
                target=self.process_recording,
                args=(recorded,),
                daemon=True,
            ).start()
        else:
            if self.transcriber.supports_streaming():
                self.transcriber.finish_stream("dictation")
            print("No audio recorded")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("No audio recorded")

    def on_command_press(self):
        """Called when command mode PTT is activated."""
        print("\n[CMD] Activated")

        if self.ui_window:
            self.ui_window.signals.update_command_status.emit("CMD: Listening...", "listening")

        self.is_command_recording = True
        self.recorded_command_audio = []
        self.audio_capture.clear_queue()
        self._last_interim_text = ""
        self._last_interim_update = 0.0

        if self.transcriber.supports_streaming():
            threading.Thread(
                target=self.transcriber.start_stream,
                args=("command",),
                daemon=True,
            ).start()

    def on_command_release(self):
        """Called when command mode PTT is released."""
        print("[CMD] Released")

        if self.ui_window:
            self.ui_window.signals.update_command_status.emit("CMD: Processing...", "processing")

        self.is_command_recording = False

        # Capture audio and process in background
        recorded = self.recorded_command_audio
        self.recorded_command_audio = []
        if len(recorded) > 0:
            threading.Thread(
                target=self.process_command_recording,
                args=(recorded,),
                daemon=True,
            ).start()
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

        command_phrases = [
            # Halves
            ("left", ["left", "left half", "left side", "left hand", "left-hand"]),
            ("right", ["right", "right half", "right side", "right hand", "right-hand"]),
            ("top", ["top", "top half", "upper", "upper half", "up"]),
            ("bottom", ["bottom", "bottom half", "lower", "lower half", "down"]),
            # Full screen
            ("fullscreen", ["full screen", "fullscreen", "maximize"]),
            ("exit_fullscreen", ["exit full screen", "exit fullscreen", "unmaximize", "restore"]),
            # Resize
            ("larger", ["larger", "bigger", "grow"]),
            ("smaller", ["smaller", "shrink"]),
            # Quarters
            ("top_left_quarter", ["top left quarter", "top left", "first quarter"]),
            ("top_right_quarter", ["top right quarter", "top right", "second quarter"]),
            ("bottom_left_quarter", ["bottom left quarter", "bottom left", "third quarter"]),
            ("bottom_right_quarter", ["bottom right quarter", "bottom right", "fourth quarter"]),
            # Desktop switching
            ("desktop_left", ["desktop left"]),
            ("desktop_right", ["desktop right"]),
        ]

        # Build flat list and sort by phrase length descending so longer
        # phrases match before shorter ones (e.g. "top left" before "top")
        all_phrases = []
        for position, phrases in command_phrases:
            for phrase in phrases:
                all_phrases.append((phrase, position))
        all_phrases.sort(key=lambda x: len(x[0]), reverse=True)

        for phrase, position in all_phrases:
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

        # Define the known command suffixes for each prefix
        window_suffixes = [
            ("left", ["left", "left half", "left side"]),
            ("right", ["right", "right half", "right side"]),
            ("top", ["top", "top half", "upper half"]),
            ("bottom", ["bottom", "bottom half", "lower half"]),
            ("fullscreen", ["full screen", "fullscreen", "maximize"]),
            ("exit_fullscreen", ["exit full screen", "exit fullscreen", "unmaximize", "restore"]),
            ("larger", ["larger", "bigger", "grow"]),
            ("smaller", ["smaller", "shrink"]),
            ("top_left_quarter", ["top left quarter", "top left"]),
            ("top_right_quarter", ["top right quarter", "top right"]),
            ("bottom_left_quarter", ["bottom left quarter", "bottom left"]),
            ("bottom_right_quarter", ["bottom right quarter", "bottom right"]),
        ]

        desktop_suffixes = [
            ("desktop_left", ["left"]),
            ("desktop_right", ["right"]),
        ]

        # Build sorted phrase lists (longest first)
        window_phrases = []
        for position, suffixes in window_suffixes:
            for suffix in suffixes:
                window_phrases.append((suffix, position))
        window_phrases.sort(key=lambda x: len(x[0]), reverse=True)

        desktop_phrases = []
        for position, suffixes in desktop_suffixes:
            for suffix in suffixes:
                desktop_phrases.append((suffix, position))
        desktop_phrases.sort(key=lambda x: len(x[0]), reverse=True)

        # Check for "window <command>"
        for suffix, position in window_phrases:
            if f" window {suffix} " in normalized:
                print(f"[VOICE CMD] Matched 'window {suffix}' → {position}")
                self._execute_voice_command(position, text)
                return True

        # Check for "desktop <command>"
        for suffix, position in desktop_phrases:
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
        if recorded_chunks is None:
            recorded_chunks = self.recorded_command_audio
        # Concatenate all recorded chunks
        audio = np.concatenate(recorded_chunks).flatten()

        print(f"[CMD] Processing {len(audio)} samples ({len(audio)/self.config['audio']['sample_rate']:.2f}s)")

        # Finalize streaming (if enabled) or fall back to offline transcription
        text = None
        if self.transcriber.supports_streaming():
            text = self.transcriber.finish_stream("command")

        if not text:
            # Apply noise suppression for offline transcription
            audio = self.noise_suppressor.process(audio)
            text = self.transcriber.transcribe(audio)

        if not text:
            print("[CMD] No transcription generated")
            if self.ui_window:
                self.ui_window.signals.update_command_status.emit("CMD: No speech detected", "unrecognized")
            return

        print(f"[CMD] Transcribed: {text}")

        command = self._parse_window_command(text)
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
        if recorded_chunks is None:
            recorded_chunks = self.recorded_audio
        # Concatenate all recorded chunks
        audio = np.concatenate(recorded_chunks).flatten()

        print(f"Processing {len(audio)} samples ({len(audio)/self.config['audio']['sample_rate']:.2f}s)")

        # Finalize streaming (if enabled) before running heavy processing
        stream_text = None
        if self.transcriber.supports_streaming():
            stream_text = self.transcriber.finish_stream("dictation")

        # Apply noise suppression for fingerprinting/transcription
        audio = self.noise_suppressor.process(audio)

        # Verify speaker (or bypass in talk mode)
        if self.talk_mode:
            is_match, similarity = True, -1.0
            print("Voice match: bypassed (talk mode)")
        else:
            is_match, similarity = self.voice_fingerprint.verify_speaker(audio)
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

        text = stream_text or self.transcriber.transcribe(audio)

        if text:
            print(f"✓ Transcribed: {text}")

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
        if not self.ui_window:
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

            # Clean up UI
            try:
                if self.ui_window:
                    if hasattr(self.ui_window, 'menu_bar_indicator') and self.ui_window.menu_bar_indicator:
                        self.ui_window.menu_bar_indicator.close()
                    self.ui_window.close()
                if self.ui_app:
                    self.ui_app.quit()
            except Exception as e:
                print(f"Error cleaning up UI: {e}")

            # Small delay to let threads finish
            time.sleep(0.2)

            print("Shutdown complete")

        return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bloviate - Voice dictation with fingerprinting")
    parser.add_argument(
        '--enroll',
        action='store_true',
        help='Enroll your voice for fingerprinting'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
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

    args = parser.parse_args()

    # Change to project directory
    project_dir = Path(__file__).parent.parent
    import os
    os.chdir(project_dir)

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
