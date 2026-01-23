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
    update_status = pyqtSignal(str)


class MenuBarIndicator:
    """Menu bar indicator showing status with emoji and audio level."""

    def __init__(self, parent=None):
        self.tray_icon = QSystemTrayIcon(parent)
        self.audio_level = 0
        self.current_state = "idle"  # idle, recording, processing, success, rejected

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
        font = QFont("Arial", 32)
        painter.setFont(font)

        if color:
            painter.setPen(color)

        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()

        return QIcon(pixmap)

    def _update_icon(self):
        """Update the menu bar icon based on current state."""
        if self.current_state == "idle":
            # Show audio level or just a dot
            if self.audio_level > 0.1:
                level_pct = int(self.audio_level * 100)
                icon = self._create_icon(f"{level_pct}")
                self.tray_icon.setIcon(icon)
                self.tray_icon.setToolTip(f"Bloviate - Audio: {level_pct}%")
            else:
                icon = self._create_icon("·")
                self.tray_icon.setIcon(icon)
                self.tray_icon.setToolTip("Bloviate - Ready")

        elif self.current_state == "recording":
            icon = self._create_icon("🐄")
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Recording...")

        elif self.current_state == "processing":
            icon = self._create_icon("🐄")
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Processing...")

        elif self.current_state == "success":
            icon = self._create_icon("✓", QColor(76, 175, 80))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Success!")

        elif self.current_state == "rejected":
            icon = self._create_icon("✗", QColor(244, 67, 54))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Voice Rejected")

    def set_audio_level(self, level: float):
        """Update audio level display."""
        self.audio_level = level
        if self.current_state == "idle":
            self._update_icon()

    def set_recording(self):
        """Set to recording state."""
        self.current_state = "recording"
        self._update_icon()

    def set_processing(self):
        """Set to processing state."""
        self.current_state = "processing"
        self._update_icon()

    def set_success(self):
        """Set to success state."""
        self.current_state = "success"
        self._update_icon()
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_rejected(self):
        """Set to rejected state."""
        self.current_state = "rejected"
        self._update_icon()
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_idle(self):
        """Set to idle state."""
        self.current_state = "idle"
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
        self.signals.update_status.connect(self._update_status)

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
        self.transcription_label.setStyleSheet(
            "padding: 10px; background-color: #2a2a2a; border-radius: 5px; min-height: 40px;"
        )
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

    def _update_voice_match(self, is_match: bool, score: float):
        """Update voice match status."""
        if is_match:
            self.match_status_label.setText("✓ Matched")
            self.match_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
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
                # These are already handled by voice_match or stay in their state
                pass

    def _update_transcription(self, text: str):
        """Update the last transcription display."""
        self.transcription_label.setText(f"Last: {text}")
        # Show success in menu bar
        if self.menu_bar_indicator:
            self.menu_bar_indicator.set_success()

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
