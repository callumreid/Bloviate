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
import numpy as np
from pathlib import Path

from audio_capture import AudioCapture
from noise_suppressor import NoiseSuppressor
from voice_fingerprint import VoiceFingerprint
from ptt_handler import PTTHandler
from transcriber import Transcriber
from ui import create_ui
from window_manager import WindowManager


class Bloviate:
    """Main application class."""

    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

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
        self.ui_window = None
        self.ui_app = None

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

    def on_ptt_release(self):
        """Called when PTT is released."""
        print("[PTT] Released")

        if self.ui_window:
            self.ui_window.signals.update_ptt_status.emit(False)
            self.ui_window.signals.update_status.emit("Processing...")

        self.is_recording = False

        # Process recorded audio
        if len(self.recorded_audio) > 0:
            self.process_recording()
        else:
            print("No audio recorded")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("No audio recorded")

    def process_recording(self):
        """Process the recorded audio."""
        # Concatenate all recorded chunks
        audio = np.concatenate(self.recorded_audio).flatten()

        print(f"Processing {len(audio)} samples ({len(audio)/self.config['audio']['sample_rate']:.2f}s)")

        # Apply noise suppression
        audio = self.noise_suppressor.process(audio)

        # Verify speaker
        is_match, similarity = self.voice_fingerprint.verify_speaker(audio)

        print(f"Voice match: {is_match} (similarity: {similarity:.3f})")

        if self.ui_window:
            self.ui_window.signals.update_voice_match.emit(is_match, similarity)

        if not is_match:
            print("✗ Voice rejected - does not match enrolled profile")
            if self.ui_window:
                self.ui_window.signals.update_status.emit("Voice rejected")
            return

        # Transcribe
        if self.ui_window:
            self.ui_window.signals.update_status.emit("Transcribing...")

        text = self.transcriber.transcribe(audio)

        if text:
            print(f"✓ Transcribed: {text}")
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

    def run(self):
        """Run the main application."""
        # Check if voice is enrolled
        if not self.voice_fingerprint.is_enrolled():
            print("Voice not enrolled. Please run with --enroll first.")
            return

        print("\n=== Bloviate ===")
        print(f"Hotkey: {self.ptt_handler.hotkey_str}")
        print("Press and hold the hotkey to record, release to transcribe.")
        print("Press Ctrl+C to exit.\n")

        # Create UI
        self.ui_app, self.ui_window = create_ui(self.config)

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

    args = parser.parse_args()

    # Change to project directory
    project_dir = Path(__file__).parent.parent
    import os
    os.chdir(project_dir)

    app = Bloviate(config_path=args.config)

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
