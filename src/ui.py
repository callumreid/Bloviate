"""
Minimal real-time feedback UI for Bloviate.
Shows audio levels, voice detection status, and PTT state.
"""

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QProgressBar, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPalette, QColor, QFont, QIcon, QPixmap, QPainter
import sys
import numpy as np


class UISignals(QObject):
    """Signals for thread-safe UI updates."""
    update_audio_level = pyqtSignal(float)
    update_ptt_status = pyqtSignal(bool)
    update_voice_match = pyqtSignal(bool, float)
    update_transcription = pyqtSignal(str)
    update_interim_transcription = pyqtSignal(str)
    update_status = pyqtSignal(str)
    update_command_status = pyqtSignal(str, str)


class MenuBarIndicator:
    """Menu bar indicator showing status with emoji and audio level."""

    def __init__(self, parent=None):
        self.tray_icon = QSystemTrayIcon(parent)
        self.audio_level = 0
        self.current_state = "idle"  # idle, recording, processing, success, rejected, command_*
        self._pulse_phase = False
        self._pulse_timer = QTimer()
        self._pulse_timer.setInterval(320)
        self._pulse_timer.timeout.connect(self._toggle_pulse)

        # Create context menu
        menu = QMenu()
        menu.addAction("Show Window", self._show_main_window)
        menu.addAction("Quit", self._quit_app)
        self.tray_icon.setContextMenu(menu)

        # Set initial icon
        self._update_icon()
        self.tray_icon.show()

        self.parent = parent

    def _show_main_window(self):
        """Show the main window."""
        if self.parent:
            self.parent.show()
            self.parent.raise_()
            self.parent.activateWindow()

    def _quit_app(self):
        """Quit the application."""
        QApplication.instance().quit()

    def _create_icon(self, text: str, color: QColor = None) -> QIcon:
        """Create an icon with text."""
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        font_size = 32
        if len(text) >= 3:
            font_size = 22
        elif len(text) == 2:
            font_size = 26
        font = QFont("Arial", font_size)
        painter.setFont(font)

        if color:
            painter.setPen(color)

        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()

        return QIcon(pixmap)

    def _create_eq_icon(self, level: float, color: QColor) -> QIcon:
        """Create an equalizer-style icon based on audio level."""
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)

        level = max(0.0, min(level, 1.0))

        # Equalizer bars
        bar_count = 5
        gap = 4
        margin = 8
        usable_width = 64 - margin * 2
        usable_height = 64 - margin * 2
        bar_width = int((usable_width - gap * (bar_count - 1)) / bar_count)

        # Base profile gives a "meter" look; scale by level
        profile = [0.35, 0.6, 0.9, 0.6, 0.35]
        min_height = 0.12

        for idx, base in enumerate(profile):
            height_ratio = min_height + (base - min_height) * level
            height = int(usable_height * height_ratio)
            x = margin + idx * (bar_width + gap)
            y = margin + (usable_height - height)
            painter.drawRoundedRect(x, y, bar_width, height, 3, 3)

        painter.end()
        return QIcon(pixmap)

    def _pulse_color(self) -> QColor:
        """Return a pulsing purple color."""
        return QColor(156, 39, 176) if self._pulse_phase else QColor(123, 31, 162)

    def _toggle_pulse(self):
        self._pulse_phase = not self._pulse_phase
        if self.current_state in {"processing", "command_processing"}:
            self._update_icon()

    def _start_pulse(self):
        if not self._pulse_timer.isActive():
            self._pulse_timer.start()

    def _stop_pulse(self):
        if self._pulse_timer.isActive():
            self._pulse_timer.stop()

    def _level_pct(self) -> int:
        """Return clamped audio level as a percentage."""
        level = max(0.0, min(self.audio_level, 1.0))
        return min(int(level * 100), 99)

    def _update_icon(self):
        """Update the menu bar icon based on current state."""
        level_pct = self._level_pct()
        if self.current_state == "idle":
            icon = self._create_eq_icon(self.audio_level, QColor(160, 160, 160))
            self.tray_icon.setIcon(icon)
            if self.audio_level > 0.05:
                self.tray_icon.setToolTip(f"Bloviate - Audio: {level_pct}%")
            else:
                self.tray_icon.setToolTip("Bloviate - Ready")

        elif self.current_state == "recording":
            icon = self._create_eq_icon(self.audio_level, QColor(255, 193, 7))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip(f"Bloviate - PTT Active (Audio: {level_pct}%)")

        elif self.current_state == "processing":
            icon = self._create_eq_icon(self.audio_level, self._pulse_color())
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Processing...")

        elif self.current_state == "command_recording":
            icon = self._create_eq_icon(self.audio_level, QColor(33, 150, 243))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip(f"Bloviate - Command listening (Audio: {level_pct}%)")

        elif self.current_state == "command_processing":
            icon = self._create_eq_icon(self.audio_level, self._pulse_color())
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Command processing...")

        elif self.current_state == "command_success":
            icon = self._create_eq_icon(self.audio_level, QColor(76, 175, 80))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Command recognized")

        elif self.current_state == "command_unknown":
            icon = self._create_eq_icon(self.audio_level, QColor(244, 67, 54))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Command unrecognized")

        elif self.current_state == "accepted":
            icon = self._create_eq_icon(self.audio_level, QColor(76, 175, 80))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Voice accepted")

        elif self.current_state == "rejected":
            icon = self._create_eq_icon(self.audio_level, QColor(244, 67, 54))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Voice Rejected")

    def set_audio_level(self, level: float):
        """Update audio level display."""
        self.audio_level = level
        if self.current_state in {"idle", "recording", "command_recording"}:
            self._update_icon()

    def set_recording(self):
        """Set to recording state."""
        self.current_state = "recording"
        self._stop_pulse()
        self._update_icon()

    def set_processing(self):
        """Set to processing state."""
        self.current_state = "processing"
        self._start_pulse()
        self._update_icon()

    def set_command_recording(self):
        """Set to command recording state."""
        self.current_state = "command_recording"
        self._stop_pulse()
        self._update_icon()

    def set_command_processing(self):
        """Set to command processing state."""
        self.current_state = "command_processing"
        self._start_pulse()
        self._update_icon()

    def set_command_success(self):
        """Set to command success state."""
        self.current_state = "command_success"
        self._stop_pulse()
        self._update_icon()
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_command_unknown(self):
        """Set to command unrecognized state."""
        self.current_state = "command_unknown"
        self._stop_pulse()
        self._update_icon()
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_accepted(self):
        """Set to accepted state."""
        self.current_state = "accepted"
        self._stop_pulse()
        self._update_icon()
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_rejected(self):
        """Set to rejected state."""
        self.current_state = "rejected"
        self._stop_pulse()
        self._update_icon()
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_idle(self):
        """Set to idle state."""
        self.current_state = "idle"
        self._stop_pulse()
        self._update_icon()

    def hide(self):
        """Hide the tray icon."""
        self.tray_icon.hide()

    def close(self):
        """Close and cleanup."""
        self.tray_icon.hide()


class BloviateUI(QMainWindow):
    """Minimal UI showing real-time feedback."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.signals = UISignals()

        # Create menu bar indicator if enabled
        self.menu_bar_indicator = None
        if config['ui'].get('show_menubar_indicator', True):
            self.menu_bar_indicator = MenuBarIndicator(parent=self)

        # Connect signals
        self.signals.update_audio_level.connect(self._update_audio_level)
        self.signals.update_ptt_status.connect(self._update_ptt_status)
        self.signals.update_voice_match.connect(self._update_voice_match)
        self.signals.update_transcription.connect(self._update_transcription)
        self.signals.update_interim_transcription.connect(self._update_interim_transcription)
        self.signals.update_status.connect(self._update_status)
        self.signals.update_command_status.connect(self._update_command_status)

        self._last_final_text = ""
        self._transcription_style_final = ""
        self._transcription_style_interim = ""

        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Bloviate")

        # Set window size
        width, height = self.config['ui']['window_size']
        self.resize(width, height)

        # Apply dark theme
        if self.config['ui']['theme'] == 'dark':
            self.set_dark_theme()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # PTT Status
        self.ptt_label = QLabel("PTT: Inactive")
        self.ptt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ptt_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.ptt_label)

        # Command Mode Status
        self.command_label = QLabel("CMD: Inactive")
        self.command_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.command_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 6px;")
        layout.addWidget(self.command_label)

        # Audio Level
        level_layout = QHBoxLayout()
        level_label = QLabel("Audio Level:")
        self.audio_bar = QProgressBar()
        self.audio_bar.setMaximum(100)
        self.audio_bar.setValue(0)
        level_layout.addWidget(level_label)
        level_layout.addWidget(self.audio_bar)
        layout.addLayout(level_layout)

        # Voice Match Status
        match_layout = QHBoxLayout()
        match_label = QLabel("Voice Match:")
        self.match_status_label = QLabel("--")
        self.match_score_label = QLabel("")
        match_layout.addWidget(match_label)
        match_layout.addWidget(self.match_status_label)
        match_layout.addWidget(self.match_score_label)
        match_layout.addStretch()
        layout.addLayout(match_layout)

        # Status message
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(self.status_label)

        # Last transcription
        self.transcription_label = QLabel("")
        self.transcription_label.setWordWrap(True)
        self._transcription_style_final = (
            "padding: 10px; background-color: #2a2a2a; border-radius: 5px; min-height: 40px;"
        )
        self._transcription_style_interim = (
            "padding: 10px; background-color: #2a2a2a; border-radius: 5px; min-height: 40px; "
            "color: #9E9E9E; font-style: italic;"
        )
        self.transcription_label.setStyleSheet(self._transcription_style_final)
        layout.addWidget(self.transcription_label)

        layout.addStretch()

    def set_dark_theme(self):
        """Apply dark theme to the application."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

        self.setPalette(palette)

    def _update_audio_level(self, level: float):
        """Update the audio level bar."""
        # Convert to percentage (assume max level is 0.3 for speaking)
        percentage = min(int(level / 0.3 * 100), 100)
        self.audio_bar.setValue(percentage)

        # Update menu bar indicator
        if self.menu_bar_indicator:
            self.menu_bar_indicator.set_audio_level(level / 0.3)

        # Color based on level
        if percentage > 60:
            color = "#4CAF50"  # Green
        elif percentage > 20:
            color = "#FFC107"  # Yellow
        else:
            color = "#9E9E9E"  # Gray

        self.audio_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid gray;
                border-radius: 3px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

    def _update_ptt_status(self, is_active: bool):
        """Update PTT status indicator."""
        if is_active:
            self.ptt_label.setText("PTT: ACTIVE")
            self.ptt_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; padding: 10px; "
                "background-color: #4CAF50; color: white; border-radius: 5px;"
            )
            # Update menu bar indicator
            if self.menu_bar_indicator:
                self.menu_bar_indicator.set_recording()
        else:
            self.ptt_label.setText("PTT: Inactive")
            self.ptt_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; padding: 10px;"
            )

    def _update_command_status(self, message: str, state: str):
        """Update command mode indicator."""
        styles = {
            "inactive": "font-size: 14px; font-weight: bold; padding: 6px;",
            "listening": (
                "font-size: 14px; font-weight: bold; padding: 6px; "
                "background-color: #2196F3; color: white; border-radius: 5px;"
            ),
            "processing": (
                "font-size: 14px; font-weight: bold; padding: 6px; "
                "background-color: #FFC107; color: black; border-radius: 5px;"
            ),
            "recognized": (
                "font-size: 14px; font-weight: bold; padding: 6px; "
                "background-color: #4CAF50; color: white; border-radius: 5px;"
            ),
            "unrecognized": (
                "font-size: 14px; font-weight: bold; padding: 6px; "
                "background-color: #F44336; color: white; border-radius: 5px;"
            ),
        }

        self.command_label.setText(message)
        self.command_label.setStyleSheet(styles.get(state, styles["inactive"]))

        if self.menu_bar_indicator:
            if state == "listening":
                self.menu_bar_indicator.set_command_recording()
            elif state == "processing":
                self.menu_bar_indicator.set_command_processing()
            elif state == "recognized":
                self.menu_bar_indicator.set_command_success()
            elif state == "unrecognized":
                self.menu_bar_indicator.set_command_unknown()
            elif state == "inactive":
                self.menu_bar_indicator.set_idle()

    def _update_voice_match(self, is_match: bool, score: float):
        """Update voice match status."""
        if is_match:
            self.match_status_label.setText("✓ Matched")
            self.match_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            if self.menu_bar_indicator:
                self.menu_bar_indicator.set_accepted()
        else:
            self.match_status_label.setText("✗ Rejected")
            self.match_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            # Update menu bar indicator
            if self.menu_bar_indicator:
                self.menu_bar_indicator.set_rejected()

        self.match_score_label.setText(f"({score:.2f})")

    def _update_status(self, message: str):
        """Update status message."""
        self.status_label.setText(message)

        # Update menu bar indicator based on status
        if self.menu_bar_indicator:
            if message == "Processing..." or message == "Transcribing...":
                self.menu_bar_indicator.set_processing()
            elif message == "Ready":
                self.menu_bar_indicator.set_idle()
            elif message in ["Voice rejected", "No audio recorded", "No speech detected"]:
                if message == "Voice rejected":
                    self.menu_bar_indicator.set_rejected()
                else:
                    self.menu_bar_indicator.set_rejected()

    def _update_transcription(self, text: str):
        """Update the last transcription display."""
        self.transcription_label.setText(f"Last: {text}")
        self.transcription_label.setStyleSheet(self._transcription_style_final)
        self._last_final_text = text
        # Show success in menu bar
        if self.menu_bar_indicator:
            self.menu_bar_indicator.set_accepted()

    def _update_interim_transcription(self, text: str):
        """Update interim transcription display while recording."""
        if not text or not text.strip():
            if self._last_final_text:
                self.transcription_label.setText(f"Last: {self._last_final_text}")
                self.transcription_label.setStyleSheet(self._transcription_style_final)
            return

        self.transcription_label.setText(f"Live: {text}")
        self.transcription_label.setStyleSheet(self._transcription_style_interim)

    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up menu bar indicator
        if self.menu_bar_indicator:
            self.menu_bar_indicator.close()
        super().closeEvent(event)


def create_ui(config: dict) -> tuple[QApplication, BloviateUI]:
    """
    Create and return the UI application and window.

    Returns:
        Tuple of (app, window)
    """
    app = QApplication(sys.argv)
    window = BloviateUI(config)
    window.show()

    return app, window
