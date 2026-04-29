"""
Minimal real-time feedback UI for Bloviate.
Shows audio levels, voice detection status, and PTT state.
"""

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QProgressBar, QSystemTrayIcon, QMenu,
    QComboBox, QPushButton, QFrame, QStackedWidget, QGroupBox,
    QSlider, QCheckBox, QMessageBox, QScrollArea, QLineEdit,
    QTextEdit, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFileDialog, QSizePolicy,
    QGridLayout, QColorDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPropertyAnimation, QSignalBlocker, QRectF, QSize
from PyQt6.QtGui import QPalette, QColor, QFont, QIcon, QPixmap, QPainter, QPen
import sys
import time
import numpy as np

from ui_themes import (
    get_theme,
    is_hidden_theme,
    normalize_hex,
    normalize_theme_id,
    normalize_waveform_preset_id,
    theme_options,
    waveform_palette_for_config,
    waveform_preset_options,
    waveform_values_for_preset,
)


class UISignals(QObject):
    """Signals for thread-safe UI updates."""
    update_audio_level = pyqtSignal(float)
    update_ptt_status = pyqtSignal(bool)
    update_voice_match = pyqtSignal(bool, float)
    update_transcription = pyqtSignal(str)
    update_rejected_transcription = pyqtSignal(str)
    update_interim_transcription = pyqtSignal(str)
    update_status = pyqtSignal(str)
    update_command_status = pyqtSignal(str, str)
    update_cleanup_mode = pyqtSignal(str, str)
    show_achievement_unlocks = pyqtSignal(object)


class MenuBarIndicator:
    """Menu bar indicator showing status with emoji and audio level."""

    def __init__(self, parent=None):
        self.parent = parent
        self.tray_icon = QSystemTrayIcon(parent)
        self.audio_level = 0
        self.current_state = "idle"  # idle, recording, processing, success, rejected, command_*
        self._pulse_phase = False
        self._closed = False
        self._last_icon_update = 0.0
        self._last_icon_signature = None
        self.waveform_palette = waveform_palette_for_config(parent.config if parent else {})
        self._pulse_timer = QTimer(self.tray_icon)
        self._pulse_timer.setInterval(320)
        self._pulse_timer.timeout.connect(self._toggle_pulse)

        # Create context menu
        menu = QMenu()
        menu.addAction("Show Window", self._show_main_window)
        menu.addAction("Open Settings", self._open_settings)
        self.audio_menu = menu.addMenu("Input Device")
        menu.addAction("Quit", self._quit_app)
        self.tray_icon.setContextMenu(menu)
        self.refresh_audio_inputs_menu()

        # Set initial icon
        self._update_icon()
        self.tray_icon.show()

    def refresh_audio_inputs_menu(self):
        """Rebuild the tray-menu audio input submenu."""
        self.audio_menu.clear()

        if not self.parent or not hasattr(self.parent, "get_audio_input_options"):
            unavailable = self.audio_menu.addAction("Unavailable")
            unavailable.setEnabled(False)
            return

        try:
            devices = self.parent.get_audio_input_options() or []
            current = self.parent.get_current_audio_input_name()
        except Exception as exc:
            failed = self.audio_menu.addAction(f"Error: {exc}")
            failed.setEnabled(False)
            return

        self._add_audio_input_action("System Default", "", checked=not current)

        if devices:
            self.audio_menu.addSeparator()
            for device in devices:
                label = str(device.get("name", "Unknown Input"))
                if device.get("is_default"):
                    label += " (Default)"
                channels = int(device.get("channels", 0))
                if channels:
                    label += f" [{channels}ch]"
                name = str(device.get("name", "") or "").strip()
                self._add_audio_input_action(label, name, checked=(name == current))

        self.audio_menu.addSeparator()
        self.audio_menu.addAction("Refresh Inputs", self.refresh_audio_inputs_menu)

    def _add_audio_input_action(self, label: str, device_name: str, *, checked: bool):
        """Add one audio-input choice to the tray menu."""
        action = self.audio_menu.addAction(label)
        action.setCheckable(True)
        action.setChecked(checked)
        action.triggered.connect(
            lambda _checked=False, selected=device_name: self._select_audio_input(selected)
        )

    def _select_audio_input(self, device_name: str):
        """Switch the current audio input from the tray menu."""
        if self.parent and hasattr(self.parent, "switch_audio_input"):
            self.parent.switch_audio_input(device_name)
        self.refresh_audio_inputs_menu()

    def _show_main_window(self):
        """Show the main window."""
        if self.parent:
            self.parent.show()
            self.parent.raise_()
            self.parent.activateWindow()

    def _open_settings(self):
        """Open settings tab in the main window."""
        if self.parent and hasattr(self.parent, "show_settings_tab"):
            self.parent.show_settings_tab()
            return
        self._show_main_window()

    def _quit_app(self):
        """Quit the application."""
        if self.parent and hasattr(self.parent, "request_quit"):
            self.parent.request_quit()
            return
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
        colors = self.waveform_palette.get("processing", ["#8E5CF7", "#2D6B6B"])
        if not colors:
            colors = ["#8E5CF7", "#2D6B6B"]
        index = 0 if self._pulse_phase else min(1, len(colors) - 1)
        return QColor(colors[index])

    def set_waveform_palette(self, palette: dict):
        self.waveform_palette = dict(palette or {})
        self._update_icon()

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

    def _update_icon(self, *, force: bool = False):
        """Update the menu bar icon based on current state."""
        if self._closed:
            return
        level_pct = self._level_pct()
        level_bucket = int(max(0.0, min(self.audio_level, 1.0)) * 20)
        signature = (self.current_state, level_bucket, self._pulse_phase)
        now = time.monotonic()
        if (
            not force
            and self.current_state == "idle"
            and signature == self._last_icon_signature
            and now - self._last_icon_update < 0.35
        ):
            return
        self._last_icon_update = now
        self._last_icon_signature = signature
        if self.current_state == "idle":
            icon = self._create_eq_icon(self.audio_level, QColor(self.waveform_palette.get("idle", "#A0A0A0")))
            self.tray_icon.setIcon(icon)
            if self.audio_level > 0.05:
                self.tray_icon.setToolTip(f"Bloviate - Audio: {level_pct}%")
            else:
                self.tray_icon.setToolTip("Bloviate - Ready")

        elif self.current_state == "recording":
            icon = self._create_eq_icon(self.audio_level, QColor(self.waveform_palette.get("recording", "#FFC107")))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip(f"Bloviate - PTT Active (Audio: {level_pct}%)")

        elif self.current_state == "processing":
            icon = self._create_eq_icon(self.audio_level, self._pulse_color())
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Processing...")

        elif self.current_state == "command_recording":
            icon = self._create_eq_icon(self.audio_level, QColor(self.waveform_palette.get("command", "#2196F3")))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip(f"Bloviate - Command listening (Audio: {level_pct}%)")

        elif self.current_state == "command_processing":
            icon = self._create_eq_icon(self.audio_level, self._pulse_color())
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Command processing...")

        elif self.current_state == "command_success":
            icon = self._create_eq_icon(self.audio_level, QColor(self.waveform_palette.get("accepted", "#4CAF50")))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Command recognized")

        elif self.current_state == "command_unknown":
            icon = self._create_eq_icon(self.audio_level, QColor(self.waveform_palette.get("rejected", "#F44336")))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Command unrecognized")

        elif self.current_state == "accepted":
            icon = self._create_eq_icon(self.audio_level, QColor(self.waveform_palette.get("accepted", "#4CAF50")))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Voice accepted")

        elif self.current_state == "rejected":
            icon = self._create_eq_icon(self.audio_level, QColor(self.waveform_palette.get("rejected", "#F44336")))
            self.tray_icon.setIcon(icon)
            self.tray_icon.setToolTip("Bloviate - Voice Rejected")

    def set_audio_level(self, level: float):
        """Update audio level display."""
        if self._closed:
            return
        if abs(float(level or 0.0) - self.audio_level) < 0.02 and self.current_state == "idle":
            return
        self.audio_level = level
        if self.current_state in {"idle", "recording", "command_recording"}:
            self._update_icon()

    def set_recording(self):
        """Set to recording state."""
        self.current_state = "recording"
        self._stop_pulse()
        self._update_icon(force=True)

    def set_processing(self):
        """Set to processing state."""
        self.current_state = "processing"
        self._start_pulse()
        self._update_icon(force=True)

    def set_command_recording(self):
        """Set to command recording state."""
        self.current_state = "command_recording"
        self._stop_pulse()
        self._update_icon(force=True)

    def set_command_processing(self):
        """Set to command processing state."""
        self.current_state = "command_processing"
        self._start_pulse()
        self._update_icon(force=True)

    def set_command_success(self):
        """Set to command success state."""
        self.current_state = "command_success"
        self._stop_pulse()
        self._update_icon(force=True)
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_command_unknown(self):
        """Set to command unrecognized state."""
        self.current_state = "command_unknown"
        self._stop_pulse()
        self._update_icon(force=True)
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_accepted(self):
        """Set to accepted state."""
        self.current_state = "accepted"
        self._stop_pulse()
        self._update_icon(force=True)
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_rejected(self):
        """Set to rejected state."""
        self.current_state = "rejected"
        self._stop_pulse()
        self._update_icon(force=True)
        # Auto-reset to idle after 2 seconds
        QTimer.singleShot(2000, self.set_idle)

    def set_idle(self):
        """Set to idle state."""
        self.current_state = "idle"
        self._stop_pulse()
        self._update_icon(force=True)

    def hide(self):
        """Hide the tray icon."""
        if self._closed:
            return
        self.tray_icon.hide()

    def close(self):
        """Close and cleanup."""
        if self._closed:
            return
        self._closed = True
        self._stop_pulse()
        self.tray_icon.hide()
        self.tray_icon.setContextMenu(None)
        self.tray_icon.deleteLater()


class BottomOverlayIndicator(QWidget):
    """Bottom-center overlay mirroring the menu bar equalizer icon."""

    _SIZE = 48  # Points — visible on screen while still compact
    _MESSAGE_WIDTH = 154
    _BAR_COUNT = 5
    _GAP = 3
    _MARGIN = 6
    _PROFILE = [0.35, 0.6, 0.9, 0.6, 0.35]
    _MIN_BAR_HEIGHT = 0.12
    _BAR_RADIUS = 2
    def __init__(self, config: dict):
        super().__init__(None)
        self.config = config
        self.waveform_palette = waveform_palette_for_config(config)
        self.audio_level = 0.0
        self.current_state = "idle"
        self._pulse_phase = False
        self._processing_phase = 0
        self._hold_state = None
        self._message_text = ""
        self._message_state = ""
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(85)
        self._pulse_timer.timeout.connect(self._toggle_pulse)
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._clear_hold_and_idle)
        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self._clear_message)

        overlay_cfg = self.config.get("ui", {}).get("ptt_overlay", {})
        self._screen_margin = int(overlay_cfg.get("margin", 20))

        self.setFixedSize(self._SIZE, self._SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        # Use a plain Window — not Tool (NSPanel), which macOS silently
        # hides when there is no visible parent window.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._closed = False
        self._objc = None  # cached ctypes handles
        self._visibility_timer = QTimer(self)
        self._visibility_timer.setInterval(2000)
        self._visibility_timer.timeout.connect(self._ensure_visible)
        # Defer show until the event loop is running
        QTimer.singleShot(0, self._initial_show)

    # ------------------------------------------------------------------
    # Cocoa helpers (macOS only)
    # ------------------------------------------------------------------

    def _get_objc(self):
        """Lazily load and cache the Objective-C runtime handles."""
        if self._objc is not None:
            return self._objc
        if sys.platform != 'darwin':
            return None
        try:
            import ctypes
            import ctypes.util
            lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
            lib.sel_registerName.restype = ctypes.c_void_p
            lib.sel_registerName.argtypes = [ctypes.c_char_p]
            self._objc = (lib, ctypes)
            return self._objc
        except Exception:
            return None

    def _get_ns_window(self):
        """Return the native NSWindow pointer, or None."""
        pair = self._get_objc()
        if pair is None:
            return None
        lib, ctypes = pair
        msg = lib.objc_msgSend
        msg.restype = ctypes.c_void_p
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        return msg(int(self.winId()),
                   lib.sel_registerName(b'window')) or None

    def _apply_macos_window_properties(self):
        """Configure the native NSWindow so the overlay is always visible,
        on every Space, alongside full-screen apps, and never steals focus."""
        pair = self._get_objc()
        if pair is None:
            return
        lib, ctypes = pair
        ns_window = self._get_ns_window()
        if not ns_window:
            return
        try:
            msg = lib.objc_msgSend
            sel = lib.sel_registerName

            # Window level: NSStatusWindowLevel (25)
            msg.restype = None
            msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
            msg(ns_window, sel(b'setLevel:'), 25)

            # Collection behavior:
            #   canJoinAllSpaces(1) | stationary(16) |
            #   ignoresCycle(64)    | fullScreenAuxiliary(256)
            msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
            msg(ns_window, sel(b'setCollectionBehavior:'),
                1 | 16 | 64 | 256)

            # Don't hide when the app loses focus
            msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            msg(ns_window, sel(b'setHidesOnDeactivate:'), False)

            # Ignore mouse events at the Cocoa level too
            msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            msg(ns_window, sel(b'setIgnoresMouseEvents:'), True)

            # Force the window on screen regardless of app activation
            msg.restype = None
            msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            msg(ns_window, sel(b'orderFrontRegardless'))
        except Exception as e:
            print(f"Warning: could not set macOS overlay properties: {e}")

    # ------------------------------------------------------------------
    # Show / visibility
    # ------------------------------------------------------------------

    def _initial_show(self):
        self._position_bottom_center()
        self.show()
        # Short delay so the native NSWindow is fully wired up before
        # we poke at it through the Objective-C runtime.
        QTimer.singleShot(50, self._apply_macos_window_properties)
        self._visibility_timer.start()

    def _ensure_visible(self):
        """Watchdog: re-show and reconfigure the overlay if it vanished."""
        if self._closed:
            return
        if not self.isVisible():
            self._position_bottom_center()
            self.show()
        # Always re-apply — catches cases where macOS hid the native
        # window without Qt knowing (e.g. Expose, space transitions).
        self._apply_macos_window_properties()

    def close(self):
        """Permanently close the overlay and stop all timers."""
        if self._closed:
            return
        self._closed = True
        self._visibility_timer.stop()
        self._pulse_timer.stop()
        self._hold_timer.stop()
        self._message_timer.stop()
        super().close()

    def _toggle_pulse(self):
        self._pulse_phase = not self._pulse_phase
        self._processing_phase = (self._processing_phase + 1) % 1000
        if self.current_state in {"processing", "command_processing"}:
            self.update()

    def _pulse_color(self) -> QColor:
        colors = self.waveform_palette.get("processing", ["#8E5CF7", "#2D6B6B"])
        if not colors:
            colors = ["#8E5CF7", "#2D6B6B"]
        index = 0 if self._pulse_phase else min(1, len(colors) - 1)
        return QColor(colors[index])

    def set_waveform_palette(self, palette: dict):
        self.waveform_palette = dict(palette or {})
        self.update()

    def _set_hold(self, state: str, hold_ms: int = 2000):
        self._hold_state = state
        self._hold_timer.start(hold_ms)

    def _clear_hold(self):
        self._hold_state = None
        if self._hold_timer.isActive():
            self._hold_timer.stop()

    def _clear_hold_and_idle(self):
        if self._hold_state and self.current_state == self._hold_state:
            self._hold_state = None
            self.set_idle()

    def _set_overlay_width(self, width: int):
        if self.width() == width:
            return
        self.setFixedSize(width, self._SIZE)
        self._position_bottom_center()

    def _clear_message(self):
        self._message_text = ""
        self._message_state = ""
        self._set_overlay_width(self._SIZE)
        self.update()

    def show_message(self, text: str, state: str = "mode", hold_ms: int = 1800):
        self._message_text = str(text or "").strip()
        self._message_state = str(state or "mode").strip()
        if self._message_text:
            self._set_overlay_width(self._MESSAGE_WIDTH)
            self._message_timer.start(max(500, int(hold_ms)))
        self.update()

    def _position_bottom_center(self):
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        x = geo.x() + int((geo.width() - self.width()) / 2)
        y = geo.y() + geo.height() - self._SIZE - self._screen_margin
        self.move(x, y)

    def _state_color(self) -> QColor:
        if self.current_state in {"processing", "command_processing"}:
            return self._pulse_color()
        if self.current_state == "recording":
            return QColor(self.waveform_palette.get("recording", "#FFC107"))
        if self.current_state == "command_recording":
            return QColor(self.waveform_palette.get("command", "#2196F3"))
        if self.current_state in {"command_success", "accepted"}:
            return QColor(self.waveform_palette.get("accepted", "#4CAF50"))
        if self.current_state in {"command_unknown", "rejected"}:
            return QColor(self.waveform_palette.get("rejected", "#F44336"))
        return QColor(self.waveform_palette.get("idle", "#A0A0A0"))

    def set_audio_level(self, level: float):
        next_level = max(0.0, min(level, 1.0))
        if abs(next_level - self.audio_level) < 0.02 and self.current_state == "idle":
            return
        self.audio_level = next_level
        self.update()

    def set_recording(self):
        self._clear_hold()
        self._clear_message()
        self.current_state = "recording"
        self._stop_pulse()
        self.update()

    def set_processing(self):
        self._clear_hold()
        self.current_state = "processing"
        self._processing_phase = 0
        self._start_pulse()
        self.update()

    def set_command_recording(self):
        self._clear_hold()
        self._clear_message()
        self.current_state = "command_recording"
        self._stop_pulse()
        self.update()

    def set_command_processing(self):
        self._clear_hold()
        self.current_state = "command_processing"
        self._processing_phase = 0
        self._start_pulse()
        self.update()

    def set_command_success(self):
        self._clear_hold()
        self.current_state = "command_success"
        self._stop_pulse()
        self._set_hold("command_success")
        self.update()

    def set_command_unknown(self):
        self._clear_hold()
        self.current_state = "command_unknown"
        self._stop_pulse()
        self._set_hold("command_unknown")
        self.update()

    def set_accepted(self):
        self._clear_hold()
        self.current_state = "accepted"
        self._stop_pulse()
        self._set_hold("accepted")
        self.update()

    def set_rejected(self):
        self._clear_hold()
        self.current_state = "rejected"
        self._stop_pulse()
        self._set_hold("rejected")
        self.update()

    def set_idle(self):
        if self._hold_state and self.current_state == self._hold_state:
            return
        self.current_state = "idle"
        self._stop_pulse()
        self._clear_hold()
        self.update()

    def _start_pulse(self):
        if not self._pulse_timer.isActive():
            self._pulse_timer.start()

    def _stop_pulse(self):
        if self._pulse_timer.isActive():
            self._pulse_timer.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        # Equalizer bars only — no background
        has_message = bool(self._message_text)
        if has_message:
            background = QColor(self.waveform_palette.get("background", "#FFFDF7"))
            background.setAlpha(236)
            painter.setBrush(background)
            painter.drawRoundedRect(0, 0, self.width(), self.height(), 12, 12)

        level = max(0.0, min(self.audio_level, 1.0))
        meter_w = self._SIZE
        usable_w = meter_w - self._MARGIN * 2
        usable_h = self._SIZE - self._MARGIN * 2
        bar_w = int((usable_w - self._GAP * (self._BAR_COUNT - 1)) / self._BAR_COUNT)

        for idx, base in enumerate(self._PROFILE):
            if self.current_state in {"processing", "command_processing"}:
                cycle_len = self._BAR_COUNT + 3
                head = self._processing_phase % cycle_len
                trail = head - idx
                wave = 1.0 - trail * 0.24 if 0 <= trail <= 3 else 0.10
                height_ratio = self._MIN_BAR_HEIGHT + (0.95 - self._MIN_BAR_HEIGHT) * wave
                colors = self.waveform_palette.get("processing", ["#E7C873", "#8E5CF7", "#2D6B6B"])
                if not colors:
                    colors = ["#E7C873", "#8E5CF7", "#2D6B6B"]
                color = QColor(colors[(self._processing_phase + idx) % len(colors)])
                painter.setBrush(color)
            else:
                height_ratio = self._MIN_BAR_HEIGHT + (base - self._MIN_BAR_HEIGHT) * level
                painter.setBrush(self._state_color())
            h = int(usable_h * height_ratio)
            x = self._MARGIN + idx * (bar_w + self._GAP)
            y = self._MARGIN + (usable_h - h)
            painter.drawRoundedRect(x, y, bar_w, h, self._BAR_RADIUS, self._BAR_RADIUS)

        if has_message:
            painter.setPen(QColor(self.waveform_palette.get("text", "#26211D")))
            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            text_rect = self.rect().adjusted(self._SIZE + 4, 5, -8, -5)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._message_text)

        painter.end()


class AudioLevelMeter(QWidget):
    """Smoothed equalizer-style audio meter for the status page."""

    _BAR_COUNT = 14
    _GAP = 5
    _MARGIN = 8
    _PROFILE = [0.25, 0.42, 0.66, 0.88, 0.58, 0.36, 0.74, 0.96, 0.72, 0.44, 0.62, 0.82, 0.48, 0.3]
    _MIN_BAR_HEIGHT = 0.10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveform_palette = waveform_palette_for_config({})
        self._target_level = 0.0
        self._display_level = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._animate)
        self.setMinimumHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_waveform_palette(self, palette: dict):
        self.waveform_palette = dict(palette or {})
        self.update()

    def set_audio_level(self, level: float):
        next_level = max(0.0, min(float(level or 0.0), 1.0))
        if not self.isVisible():
            self._target_level = next_level
            self._display_level = next_level
            self._timer.stop()
            return
        if abs(next_level - self._target_level) < 0.015 and self._timer.isActive():
            return
        self._target_level = next_level
        if not self._timer.isActive():
            self._timer.start()

    def _animate(self):
        delta = self._target_level - self._display_level
        if abs(delta) < 0.004:
            self._display_level = self._target_level
            if self._target_level <= 0.004:
                self._timer.stop()
        else:
            self._display_level += delta * 0.18
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        painter.setBrush(QColor(self.waveform_palette.get("background", "#FFFDF7")))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 8, 8)

        level = max(0.0, min(self._display_level, 1.0))
        usable_w = max(1, self.width() - self._MARGIN * 2)
        usable_h = max(1, self.height() - self._MARGIN * 2)
        bar_w = max(4, int((usable_w - self._GAP * (self._BAR_COUNT - 1)) / self._BAR_COUNT))
        total_w = bar_w * self._BAR_COUNT + self._GAP * (self._BAR_COUNT - 1)
        start_x = self._MARGIN + max(0, int((usable_w - total_w) / 2))

        base_color = QColor(self.waveform_palette.get("accepted", "#2D6B6B"))
        quiet_color = QColor(self.waveform_palette.get("quiet", "#BFB2A1"))
        for idx, base in enumerate(self._PROFILE):
            height_ratio = self._MIN_BAR_HEIGHT + (base - self._MIN_BAR_HEIGHT) * level
            h = max(4, int(usable_h * height_ratio))
            x = start_x + idx * (bar_w + self._GAP)
            y = self._MARGIN + int((usable_h - h) / 2)
            color = QColor(base_color if level > 0.04 else quiet_color)
            color.setAlpha(110 + int(120 * min(1.0, level + base * 0.25)))
            painter.setBrush(color)
            painter.drawRoundedRect(x, y, bar_w, h, 3, 3)

        painter.end()


class InsightGauge(QWidget):
    """Compact semicircle gauge used by the insights cards."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.center_text = "--"
        self.caption = ""
        self.theme_colors = get_theme("light")["colors"]
        self.setMinimumHeight(118)

    def set_theme_colors(self, colors: dict):
        self.theme_colors = dict(colors or self.theme_colors)
        self.update()

    def set_data(self, value: float, center_text: str, caption: str):
        self.value = max(0.0, min(float(value or 0.0), 1.0))
        self.center_text = str(center_text or "--")
        self.caption = str(caption or "")
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(18, 18, self.width() - 36, max(70, self.height() + 18))
        arc_rect = QRectF(rect.x(), rect.y(), rect.width(), rect.height())
        track_pen = QPen(QColor(self.theme_colors.get("border", "#D8D0C2")), 14)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(arc_rect, 200 * 16, -220 * 16)

        value_pen = QPen(QColor(self.theme_colors.get("primary", "#2D6B6B")), 14)
        value_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(value_pen)
        painter.drawArc(arc_rect, 200 * 16, int(-220 * self.value * 16))

        painter.setPen(QColor(self.theme_colors.get("text", "#26211D")))
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(0, 42, 0, -28), Qt.AlignmentFlag.AlignCenter, self.center_text)

        painter.setPen(QColor(self.theme_colors.get("muted", "#6F665E")))
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(0, 70, 0, -8), Qt.AlignmentFlag.AlignCenter, self.caption)
        painter.end()


class InsightStreakHeatmap(QWidget):
    """GitHub-style calendar heatmap for recent dictation days."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.days = []
        self.theme_colors = get_theme("light")["colors"]
        self.heatmap_colors = ["#F0E9DE", "#CFEDEA", "#72C9BE", "#2D8B7E", "#1F6F68"]
        self.setMinimumHeight(166)

    def set_theme_colors(self, colors: dict):
        self.theme_colors = dict(colors or self.theme_colors)
        self.heatmap_colors = [
            self.theme_colors.get("surface_alt", "#F0E9DE"),
            self.theme_colors.get("card_alt", "#CFEDEA"),
            self.theme_colors.get("command", "#72C9BE"),
            self.theme_colors.get("primary", "#2D8B7E"),
            self.theme_colors.get("success", "#1F6F68"),
        ]
        self.update()

    def set_days(self, days: list[dict]):
        self.days = list(days or [])
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        days = self.days[-84:]
        if not days:
            painter.setPen(QColor(self.theme_colors.get("muted", "#6F665E")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No dictation history yet")
            painter.end()
            return

        max_words = max([int(day.get("words", 0) or 0) for day in days] + [1])
        cell = min(14, max(8, int((self.width() - 34) / 14)))
        gap = 6
        start_x = 8
        start_y = 20

        labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        painter.setPen(QColor(self.theme_colors.get("muted", "#7C7166")))
        label_font = QFont()
        label_font.setPointSize(8)
        painter.setFont(label_font)
        for row, label in enumerate(labels):
            painter.drawText(0, start_y + row * (cell + gap) + cell - 2, label[:3])

        painter.setPen(Qt.PenStyle.NoPen)
        grid_x = start_x + 30
        for idx, day in enumerate(days):
            col = idx // 7
            row = idx % 7
            words = int(day.get("words", 0) or 0)
            if words <= 0:
                color_index = 0
            else:
                color_index = min(4, 1 + int((words / max_words) * 3.99))
            painter.setBrush(QColor(self.heatmap_colors[color_index]))
            painter.drawRoundedRect(
                grid_x + col * (cell + gap),
                start_y + row * (cell + gap),
                cell,
                cell,
                3,
                3,
            )

        legend_y = start_y + 7 * (cell + gap) + 10
        painter.setPen(QColor(self.theme_colors.get("muted", "#7C7166")))
        painter.drawText(grid_x, legend_y + cell - 1, "Less")
        painter.setPen(Qt.PenStyle.NoPen)
        for idx, color in enumerate(self.heatmap_colors):
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(grid_x + 42 + idx * (cell + 4), legend_y, cell, cell, 3, 3)
        painter.setPen(QColor(self.theme_colors.get("muted", "#7C7166")))
        painter.drawText(
            grid_x + 42 + len(self.heatmap_colors) * (cell + 4) + 6,
            legend_y + cell - 1,
            "More",
        )
        painter.end()


class InsightCard(QFrame):
    """Shared card chrome for settings insights."""

    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        self.setObjectName("InsightCard")
        self.setMinimumHeight(156)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(18, 16, 18, 16)
        self.layout.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("InsightTitle")
        self.layout.addWidget(self.title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("InsightSubtitle")
            self.layout.addWidget(subtitle_label)


class AchievementCelebrationOverlay(QFrame):
    """Nonblocking full-window achievement unlock celebration."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.unlocks = []
        self.index = 0
        self.setObjectName("AchievementOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(36, 36, 36, 36)
        outer.addStretch()

        self.card = QFrame(self)
        self.card.setObjectName("AchievementOverlayCard")
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(28, 28, 28, 24)
        card_layout.setSpacing(14)

        self.badge_label = QLabel()
        self.badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge_label.setMinimumHeight(142)
        card_layout.addWidget(self.badge_label)

        self.title_label = QLabel("Achievement Unlocked")
        self.title_label.setObjectName("AchievementOverlayTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setWordWrap(True)
        card_layout.addWidget(self.title_label)

        self.description_label = QLabel("")
        self.description_label.setObjectName("AchievementOverlayDescription")
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setWordWrap(True)
        card_layout.addWidget(self.description_label)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("AchievementOverlayProgress")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.progress_label)

        controls = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.dismiss_button = QPushButton("Dismiss")
        self.dismiss_button.setObjectName("PrimaryActionButton")
        controls.addStretch()
        controls.addWidget(self.previous_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.dismiss_button)
        controls.addStretch()
        card_layout.addLayout(controls)

        outer.addWidget(self.card, alignment=Qt.AlignmentFlag.AlignCenter)
        outer.addStretch()

        self.previous_button.clicked.connect(self._previous)
        self.next_button.clicked.connect(self._next)
        self.dismiss_button.clicked.connect(self.hide)

    def show_unlocks(self, unlocks):
        self.unlocks = list(unlocks or [])
        if not self.unlocks:
            return
        self.index = 0
        self.setGeometry(self.parentWidget().rect())
        self._render()
        self.show()
        self.raise_()

    def _render(self):
        item = self.unlocks[self.index]
        badge_path = str(item.get("badge_path", "") or "")
        pixmap = QPixmap(badge_path) if badge_path else QPixmap()
        if not pixmap.isNull():
            self.badge_label.setPixmap(
                pixmap.scaled(132, 132, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self.badge_label.setText("Achievement")

        prefix = "Achievement Unlocked"
        if len(self.unlocks) > 1:
            prefix = f"Achievement {self.index + 1} of {len(self.unlocks)}"
        self.title_label.setText(f"{prefix}\n{item.get('title', 'Achievement')}")
        self.description_label.setText(str(item.get("description", "")))
        self.progress_label.setText(str(item.get("progress_label", "")))
        self.previous_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.unlocks) - 1)

    def _previous(self):
        if self.index > 0:
            self.index -= 1
            self._render()

    def _next(self):
        if self.index < len(self.unlocks) - 1:
            self.index += 1
            self._render()


class CowRunwayOverlay(QFrame):
    """Short nonblocking cow runway animation for the dashboard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CowRunwayOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(85)
        self._timer.timeout.connect(self._advance)
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.hide)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.addStretch()
        self.cow_label = QLabel("")
        self.cow_label.setObjectName("CowRunwayText")
        self.cow_label.setFont(QFont("Menlo", 13))
        self.cow_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.cow_label.setMinimumHeight(72)
        layout.addWidget(self.cow_label)
        layout.addStretch()
        self.hide()

    def start(self, duration_ms: int = 3600):
        if self.parent():
            self.setGeometry(self.parent().rect())
        self._phase = 0
        self.cow_label.setText(self._runway_text())
        self.show()
        self.raise_()
        self._timer.start()
        self._close_timer.start(max(1200, int(duration_ms)))

    def hide(self):
        self._timer.stop()
        self._close_timer.stop()
        super().hide()

    def _advance(self):
        self._phase = (self._phase + 2) % 72
        self.cow_label.setText(self._runway_text())

    def _runway_text(self) -> str:
        width = 72
        lines = [" " * width, " " * width, " " * width]
        frame = ["(__)", "(oo)", "/--\\"]
        for offset in (0, 18, 36, 55):
            x = (self._phase + offset) % width
            for row, part in enumerate(frame):
                line = lines[row]
                if x + len(part) <= width:
                    lines[row] = line[:x] + part + line[x + len(part):]
                else:
                    first = width - x
                    lines[row] = part[first:] + line[len(part) - first:x] + part[:first]
        return "\n".join(lines)


class BloviateUI(QMainWindow):
    """Minimal UI showing real-time feedback."""

    def __init__(
        self,
        config: dict,
        get_audio_inputs=None,
        set_audio_input=None,
        get_voice_profile_status=None,
        set_voice_mode=None,
        set_voice_threshold=None,
        capture_enrollment_sample=None,
        clear_voice_profile=None,
        get_personal_dictionary_path=None,
        ensure_personal_dictionary_exists=None,
        open_personal_dictionary=None,
        reload_personal_dictionary=None,
        get_personal_dictionary_payload=None,
        save_personal_dictionary_payload=None,
        get_model_options=None,
        get_secret_statuses=None,
        set_api_key=None,
        set_transcription_settings=None,
        set_hotkey_settings=None,
        set_general_settings=None,
        toggle_dictation=None,
        get_history_records=None,
        get_history_insights=None,
        delete_history_record=None,
        clear_history=None,
        export_history=None,
        get_achievement_summary=None,
        reset_achievements=None,
        set_achievement_settings=None,
        analyze_achievement_history=None,
        run_doctor_text=None,
        reset_settings_to_defaults=None,
        get_permission_statuses=None,
        request_permission=None,
        open_permission_settings=None,
        set_show_main_window_on_startup=None,
        set_startup_splash_enabled=None,
        set_terminal_startup_animation_enabled=None,
    ):
        super().__init__()
        self.config = config
        self.get_audio_inputs = get_audio_inputs
        self.set_audio_input = set_audio_input
        self.get_voice_profile_status = get_voice_profile_status
        self.set_voice_mode = set_voice_mode
        self.set_voice_threshold = set_voice_threshold
        self.capture_enrollment_sample = capture_enrollment_sample
        self.clear_voice_profile = clear_voice_profile
        self.get_personal_dictionary_path = get_personal_dictionary_path
        self.ensure_personal_dictionary_exists = ensure_personal_dictionary_exists
        self.open_personal_dictionary = open_personal_dictionary
        self.reload_personal_dictionary = reload_personal_dictionary
        self.get_personal_dictionary_payload = get_personal_dictionary_payload
        self.save_personal_dictionary_payload = save_personal_dictionary_payload
        self.get_model_options = get_model_options
        self.get_secret_statuses = get_secret_statuses
        self.set_api_key = set_api_key
        self.set_transcription_settings = set_transcription_settings
        self.set_hotkey_settings = set_hotkey_settings
        self.set_general_settings = set_general_settings
        self.toggle_dictation = toggle_dictation
        self.get_history_records = get_history_records
        self.get_history_insights = get_history_insights
        self.delete_history_record = delete_history_record
        self.clear_history = clear_history
        self.export_history = export_history
        self.get_achievement_summary = get_achievement_summary
        self.reset_achievements = reset_achievements
        self.set_achievement_settings = set_achievement_settings
        self.analyze_achievement_history = analyze_achievement_history
        self.run_doctor_text = run_doctor_text
        self.reset_settings_to_defaults = reset_settings_to_defaults
        self.get_permission_statuses = get_permission_statuses
        self.request_permission = request_permission
        self.open_permission_settings = open_permission_settings
        self.set_show_main_window_on_startup = set_show_main_window_on_startup
        self.set_startup_splash_enabled = set_startup_splash_enabled
        self.set_terminal_startup_animation_enabled = set_terminal_startup_animation_enabled
        self.signals = UISignals()
        self._closing = False
        self._theme_id = normalize_theme_id(self.config.get("ui", {}).get("theme", "light"))
        self._theme = get_theme(self._theme_id)
        self.config.setdefault("ui", {})["theme"] = self._theme_id
        self._title_click_count = 0

        # Create menu bar indicator if enabled
        self.menu_bar_indicator = None
        if config['ui'].get('show_menubar_indicator', True):
            self.menu_bar_indicator = MenuBarIndicator(parent=self)

        # Create bottom overlay indicator if enabled
        self.ptt_overlay = None
        if config.get("ui", {}).get("ptt_overlay", {}).get("enabled", True):
            self.ptt_overlay = BottomOverlayIndicator(config)

        # Connect signals
        self.signals.update_audio_level.connect(self._update_audio_level)
        self.signals.update_ptt_status.connect(self._update_ptt_status)
        self.signals.update_voice_match.connect(self._update_voice_match)
        self.signals.update_transcription.connect(self._update_transcription)
        self.signals.update_rejected_transcription.connect(self._update_rejected_transcription)
        self.signals.update_interim_transcription.connect(self._update_interim_transcription)
        self.signals.update_status.connect(self._update_status)
        self.signals.update_command_status.connect(self._update_command_status)
        self.signals.update_cleanup_mode.connect(self._update_cleanup_mode_from_runtime)
        self.signals.show_achievement_unlocks.connect(self._show_achievement_unlocks)

        self._last_final_text = ""
        self._transcription_style_final = ""
        self._transcription_style_interim = ""
        self._audio_inputs_ready = bool(self.get_audio_inputs and self.set_audio_input)
        self._settings_status_default_style = "font-size: 12px; color: #6F665E;"
        self._settings_ok_style = "font-size: 12px; color: #2F7D4F; font-weight: 600;"
        self._settings_error_style = "font-size: 12px; color: #B23B35; font-weight: 600;"
        self._dictionary_terms = []
        self._dictionary_corrections = []
        self._achievement_rows = []
        self._achievement_refresh_timer = QTimer(self)
        self._achievement_refresh_timer.setSingleShot(True)
        self._achievement_refresh_timer.timeout.connect(self._refresh_achievements)
        self._permissions_prompt_shown = False

        self.init_ui()
        self.achievement_overlay = AchievementCelebrationOverlay(self)

    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Bloviate")

        # Set window size
        width, height = self.config['ui']['window_size']
        self.setMinimumSize(980, 700)
        self.resize(max(width, 1180), max(height, 860))

        self.set_light_theme()

        # Central widget
        central_widget = QWidget()
        central_widget.setObjectName("AppRoot")
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(layout)
        nav = QWidget()
        nav.setObjectName("TopNav")
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(28, 18, 28, 10)
        nav_layout.setSpacing(0)
        self.app_title_label = QLabel("Bloviate")
        self.app_title_label.setObjectName("AppTitle")
        self.app_title_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.app_title_label.mousePressEvent = self._handle_title_click
        nav_layout.addWidget(self.app_title_label)
        nav_layout.addStretch()
        self.status_nav_button = QPushButton("Status")
        self.settings_nav_button = QPushButton("Settings")
        for button in (self.status_nav_button, self.settings_nav_button):
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setMinimumWidth(140)
        nav_layout.addWidget(self.status_nav_button)
        nav_layout.addWidget(self.settings_nav_button)
        nav_layout.addStretch()
        layout.addWidget(nav)

        self.tabs = QStackedWidget()
        layout.addWidget(self.tabs)
        self.status_tab = QWidget()
        self.settings_tab = QWidget()
        self.tabs.addWidget(self.status_tab)
        self.tabs.addWidget(self.settings_tab)
        self.status_nav_button.clicked.connect(self.show_status_tab)
        self.settings_nav_button.clicked.connect(self.show_settings_tab)
        self._build_status_tab()
        self._build_settings_tab()
        self._select_main_page(self.status_tab)
        self._refresh_audio_inputs()
        self._refresh_voice_controls()
        self._refresh_dictionary_path()
        self._refresh_permissions()
        self._refresh_insights()
        QTimer.singleShot(900, self._maybe_show_permissions_prompt)

    def _build_status_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 26, 30, 30)
        layout.setSpacing(16)
        self.status_tab.setLayout(layout)

        # PTT Status
        self.ptt_label = QLabel("PTT: Inactive")
        self.ptt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ptt_label.setObjectName("StatusPill")
        colors = self._theme["colors"]
        self.ptt_label.setStyleSheet(self._status_pill_style(colors["nav"], colors["text"]))
        layout.addWidget(self.ptt_label)

        # Command Mode Status
        self.command_label = QLabel("CMD: Inactive")
        self.command_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.command_label.setObjectName("StatusPill")
        self.command_label.setStyleSheet(self._status_pill_style(colors["nav"], colors["text"]))
        layout.addWidget(self.command_label)

        controls_layout = QHBoxLayout()
        controls_layout.addStretch()
        self.toggle_dictation_button = QPushButton("Start Dictation")
        self.toggle_dictation_button.setObjectName("PrimaryActionButton")
        self.toggle_dictation_button.setMinimumWidth(180)
        controls_layout.addWidget(self.toggle_dictation_button)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        self.toggle_dictation_button.clicked.connect(self._toggle_dictation_from_ui)

        # Audio Level
        level_layout = QHBoxLayout()
        level_label = QLabel("Audio Level:")
        self.audio_bar = AudioLevelMeter()
        self.audio_bar.set_waveform_palette(self._waveform_palette())
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
        self.status_label.setStyleSheet(f"font-style: italic; color: {colors['muted']}; padding: 8px;")
        layout.addWidget(self.status_label)

        # Last transcription
        self.transcription_label = QLabel("")
        self.transcription_label.setWordWrap(True)
        self.transcription_label.setStyleSheet(self._transcription_style_final)
        layout.addWidget(self.transcription_label)
        layout.addStretch()

    def _build_settings_tab(self):
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_tab.setLayout(outer_layout)
        scroll = QScrollArea()
        scroll.setObjectName("SettingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.viewport().setStyleSheet(f"background-color: {self._theme['colors']['window']};")
        content = QWidget()
        content.setObjectName("SettingsContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 26, 28, 30)
        layout.setSpacing(16)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        # First-run permissions
        permissions_group = QGroupBox("Permissions")
        permissions_layout = QVBoxLayout(permissions_group)
        self.permissions_status_label = QLabel("")
        self.permissions_status_label.setWordWrap(True)
        self.permissions_status_label.setStyleSheet(self._settings_status_default_style)
        permissions_layout.addWidget(self.permissions_status_label)
        permissions_actions = QHBoxLayout()
        self.request_microphone_button = QPushButton("Request Microphone")
        self.open_accessibility_button = QPushButton("Open Accessibility")
        self.open_input_monitoring_button = QPushButton("Open Input Monitoring")
        self.open_automation_button = QPushButton("Open Automation")
        self.refresh_permissions_button = QPushButton("Refresh")
        permissions_actions.addWidget(self.request_microphone_button)
        permissions_actions.addWidget(self.open_accessibility_button)
        permissions_actions.addWidget(self.open_input_monitoring_button)
        permissions_actions.addWidget(self.open_automation_button)
        permissions_actions.addWidget(self.refresh_permissions_button)
        permissions_actions.addStretch()
        permissions_layout.addLayout(permissions_actions)
        layout.addWidget(permissions_group)
        self.request_microphone_button.clicked.connect(lambda: self._request_permission("microphone"))
        self.open_accessibility_button.clicked.connect(lambda: self._request_permission("accessibility"))
        self.open_input_monitoring_button.clicked.connect(lambda: self._request_permission("input_monitoring"))
        self.open_automation_button.clicked.connect(lambda: self._request_permission("automation"))
        self.refresh_permissions_button.clicked.connect(self._refresh_permissions)

        # Appearance
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)
        appearance_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.theme_combo = QComboBox()
        for theme_id, label in theme_options(include_hidden=self._secret_themes_unlocked()):
            self.theme_combo.addItem(label, theme_id)
        self._set_combo_data(self.theme_combo, self._theme_id)

        waveform_cfg = self.config.get("ui", {}).get("waveform", {})
        if not isinstance(waveform_cfg, dict):
            waveform_cfg = {}
        self.waveform_preset_combo = QComboBox()
        for preset_id, label in waveform_preset_options():
            self.waveform_preset_combo.addItem(label, preset_id)
        self._set_combo_data(
            self.waveform_preset_combo,
            normalize_waveform_preset_id(waveform_cfg.get("preset", "theme")),
        )
        waveform = self._waveform_palette()
        self.waveform_idle_edit = self._color_line_edit(waveform.get("idle", "#BFB2A1"))
        self.waveform_recording_edit = self._color_line_edit(waveform.get("recording", "#E7C873"))
        self.waveform_command_edit = self._color_line_edit(waveform.get("command", "#2D6B6B"))
        self.waveform_accepted_edit = self._color_line_edit(waveform.get("accepted", "#2F7D4F"))
        self.waveform_rejected_edit = self._color_line_edit(waveform.get("rejected", "#B23B35"))
        self.waveform_processing_edit = QLineEdit(", ".join(waveform.get("processing", [])))
        self.waveform_processing_edit.setPlaceholderText("#E7C873, #8E5CF7, #2D6B6B")
        easter_cfg = self._easter_config()
        self.milestone_toasts_checkbox = QCheckBox("Milestone achievement toasts")
        self.milestone_toasts_checkbox.setChecked(bool(easter_cfg.get("milestone_toasts", True)))

        appearance_layout.addRow("Theme:", self.theme_combo)
        appearance_layout.addRow("Waveform preset:", self.waveform_preset_combo)
        appearance_layout.addRow("Idle bars:", self._color_field_widget(self.waveform_idle_edit))
        appearance_layout.addRow("Listening bars:", self._color_field_widget(self.waveform_recording_edit))
        appearance_layout.addRow("Command bars:", self._color_field_widget(self.waveform_command_edit))
        appearance_layout.addRow("Accepted bars:", self._color_field_widget(self.waveform_accepted_edit))
        appearance_layout.addRow("Rejected bars:", self._color_field_widget(self.waveform_rejected_edit))
        appearance_layout.addRow("Processing colors:", self.waveform_processing_edit)
        appearance_layout.addRow("", self.milestone_toasts_checkbox)
        appearance_actions = QHBoxLayout()
        self.apply_appearance_button = QPushButton("Apply Appearance")
        self.run_cows_button = QPushButton("Run Cows")
        appearance_actions.addWidget(self.apply_appearance_button)
        appearance_actions.addWidget(self.run_cows_button)
        appearance_actions.addStretch()
        appearance_layout.addRow("", appearance_actions)
        self.appearance_status_label = QLabel("")
        self.appearance_status_label.setStyleSheet(self._settings_status_default_style)
        appearance_layout.addRow("", self.appearance_status_label)
        layout.addWidget(appearance_group)
        self.theme_combo.currentIndexChanged.connect(self._preview_theme_selection)
        self.waveform_preset_combo.currentIndexChanged.connect(self._preview_waveform_preset)
        self.apply_appearance_button.clicked.connect(self._apply_appearance_settings)
        self.run_cows_button.clicked.connect(self.run_cow_runway)

        # Insights
        insights_group = QGroupBox("Insights")
        insights_layout = QVBoxLayout(insights_group)
        insights_layout.setSpacing(16)
        insights_header = QHBoxLayout()
        insights_title = QLabel("Your Usage")
        insights_title.setObjectName("InsightSectionTitle")
        self.insights_status_label = QLabel("")
        self.insights_status_label.setStyleSheet(self._settings_status_default_style)
        self.refresh_insights_button = QPushButton("Refresh")
        insights_header.addWidget(insights_title)
        insights_header.addStretch()
        insights_header.addWidget(self.insights_status_label)
        insights_header.addWidget(self.refresh_insights_button)
        insights_layout.addLayout(insights_header)

        insights_grid = QGridLayout()
        insights_grid.setHorizontalSpacing(16)
        insights_grid.setVerticalSpacing(16)
        for column in range(3):
            insights_grid.setColumnStretch(column, 1)

        self.insight_wpm_card = InsightCard("Words per minute", "speaking pace")
        self.insight_wpm_value = QLabel("0")
        self.insight_wpm_value.setObjectName("InsightNumber")
        self.insight_wpm_gauge = InsightGauge()
        self.insight_wpm_gauge.set_theme_colors(self._theme["colors"])
        self.insight_wpm_card.layout.addWidget(self.insight_wpm_value)
        self.insight_wpm_card.layout.addWidget(self.insight_wpm_gauge)
        insights_grid.addWidget(self.insight_wpm_card, 0, 0)

        self.insight_fixes_card = InsightCard("Fixes made", "cleanup and dictionary")
        self.insight_fixes_value = QLabel("0")
        self.insight_fixes_value.setObjectName("InsightNumber")
        self.insight_changed_words_label = QLabel("0 words rewritten")
        self.insight_dictionary_rules_label = QLabel("0 dictionary rules active")
        self.insight_fixes_card.layout.addWidget(self.insight_fixes_value)
        self.insight_fixes_card.layout.addWidget(self.insight_changed_words_label)
        self.insight_fixes_card.layout.addWidget(self.insight_dictionary_rules_label)
        self.insight_fixes_card.layout.addStretch()
        insights_grid.addWidget(self.insight_fixes_card, 0, 1)

        self.insight_words_card = InsightCard("Total words dictated", "local history")
        self.insight_total_words_value = QLabel("0")
        self.insight_total_words_value.setObjectName("InsightNumber")
        self.insight_total_transcripts_label = QLabel("0 saved transcripts")
        self.insight_mode_bar = QProgressBar()
        self.insight_mode_bar.setRange(0, 100)
        self.insight_words_card.layout.addWidget(self.insight_total_words_value)
        self.insight_words_card.layout.addWidget(self.insight_total_transcripts_label)
        self.insight_words_card.layout.addWidget(self.insight_mode_bar)
        self.insight_words_card.layout.addStretch()
        insights_grid.addWidget(self.insight_words_card, 0, 2)

        self.insight_apps_card = InsightCard("Desktop usage", "top target apps")
        self.insight_apps_summary_label = QLabel("Apps used | 0")
        self.insight_apps_summary_label.setObjectName("InsightSubtitle")
        self.insight_app_rows_widget = QWidget()
        self.insight_app_rows_layout = QVBoxLayout(self.insight_app_rows_widget)
        self.insight_app_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.insight_app_rows_layout.setSpacing(8)
        self.insight_apps_card.layout.addWidget(self.insight_apps_summary_label)
        self.insight_apps_card.layout.addWidget(self.insight_app_rows_widget)
        self.insight_apps_card.layout.addStretch()
        insights_grid.addWidget(self.insight_apps_card, 1, 0, 1, 2)

        self.insight_streak_card = InsightCard("Dictation streak", "last 12 weeks")
        streak_header = QHBoxLayout()
        self.insight_current_streak_label = QLabel("0 day streak")
        self.insight_current_streak_label.setObjectName("InsightNumberSmall")
        self.insight_longest_streak_label = QLabel("Longest | 0 days")
        self.insight_longest_streak_label.setObjectName("InsightSubtitle")
        streak_header.addWidget(self.insight_current_streak_label)
        streak_header.addStretch()
        streak_header.addWidget(self.insight_longest_streak_label)
        self.insight_streak_heatmap = InsightStreakHeatmap()
        self.insight_streak_heatmap.set_theme_colors(self._theme["colors"])
        self.insight_streak_card.layout.addLayout(streak_header)
        self.insight_streak_card.layout.addWidget(self.insight_streak_heatmap)
        insights_grid.addWidget(self.insight_streak_card, 1, 2)

        insights_layout.addLayout(insights_grid)
        layout.addWidget(insights_group)
        self.refresh_insights_button.clicked.connect(self._refresh_insights)

        # Achievements
        achievements_group = QGroupBox("Achievements")
        achievements_layout = QVBoxLayout(achievements_group)
        achievements_layout.setSpacing(12)

        achievements_header = QHBoxLayout()
        self.achievement_summary_label = QLabel("0 unlocked")
        self.achievement_summary_label.setObjectName("InsightSectionTitle")
        self.achievement_status_label = QLabel("")
        self.achievement_status_label.setStyleSheet(self._settings_status_default_style)
        achievements_header.addWidget(self.achievement_summary_label)
        achievements_header.addStretch()
        achievements_header.addWidget(self.achievement_status_label)
        achievements_layout.addLayout(achievements_header)

        self.achievement_progress_bar = QProgressBar()
        self.achievement_progress_bar.setRange(0, 100)
        self.achievement_progress_bar.setFormat("0%")
        achievements_layout.addWidget(self.achievement_progress_bar)

        achievement_filters = QHBoxLayout()
        self.achievement_search_edit = QLineEdit()
        self.achievement_search_edit.setPlaceholderText("Search achievements")
        self.achievement_filter_combo = QComboBox()
        self.achievement_filter_combo.addItem("All", "all")
        self.achievement_filter_combo.addItem("Unlocked", "unlocked")
        self.achievement_filter_combo.addItem("Locked", "locked")
        self.achievement_filter_combo.addItem("AI-assisted", "ai")
        self.refresh_achievements_button = QPushButton("Refresh")
        achievement_filters.addWidget(self.achievement_search_edit, 1)
        achievement_filters.addWidget(self.achievement_filter_combo)
        achievement_filters.addWidget(self.refresh_achievements_button)
        achievements_layout.addLayout(achievement_filters)

        achievement_actions = QHBoxLayout()
        self.achievement_ai_checkbox = QCheckBox("Enable AI-assisted achievements")
        self.analyze_achievements_button = QPushButton("Analyze History")
        self.reset_achievements_button = QPushButton("Reset Achievements")
        achievement_actions.addWidget(self.achievement_ai_checkbox)
        achievement_actions.addWidget(self.analyze_achievements_button)
        achievement_actions.addWidget(self.reset_achievements_button)
        achievement_actions.addStretch()
        achievements_layout.addLayout(achievement_actions)

        self.achievement_recent_label = QLabel("Recent unlocks will appear here.")
        self.achievement_recent_label.setStyleSheet(self._settings_status_default_style)
        self.achievement_recent_label.setWordWrap(True)
        achievements_layout.addWidget(self.achievement_recent_label)

        self.achievement_table = QTableWidget(0, 5)
        self.achievement_table.setHorizontalHeaderLabels(["", "Achievement", "Category", "Progress", "Status"])
        self.achievement_table.setIconSize(QSize(42, 42))
        self.achievement_table.verticalHeader().setVisible(False)
        self.achievement_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.achievement_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.achievement_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.achievement_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.achievement_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.achievement_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.achievement_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.achievement_table.setMinimumHeight(260)
        achievements_layout.addWidget(self.achievement_table)

        self.achievement_detail = QTextEdit()
        self.achievement_detail.setReadOnly(True)
        self.achievement_detail.setMinimumHeight(92)
        achievements_layout.addWidget(self.achievement_detail)
        layout.addWidget(achievements_group)

        self.refresh_achievements_button.clicked.connect(self._refresh_achievements)
        self.achievement_search_edit.textChanged.connect(self._queue_achievement_refresh)
        self.achievement_filter_combo.currentIndexChanged.connect(self._queue_achievement_refresh)
        self.achievement_ai_checkbox.stateChanged.connect(self._toggle_achievement_ai)
        self.analyze_achievements_button.clicked.connect(self._analyze_achievements)
        self.reset_achievements_button.clicked.connect(self._reset_achievements)
        self.achievement_table.itemSelectionChanged.connect(self._show_selected_achievement_detail)

        # Audio settings
        audio_group = QGroupBox("Audio Input")
        audio_layout = QVBoxLayout(audio_group)
        device_layout = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setMinimumContentsLength(24)
        self.device_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.refresh_devices_button = QPushButton("Refresh")
        self.apply_device_button = QPushButton("Apply")
        device_layout.addWidget(self.device_combo, 1)
        device_layout.addWidget(self.refresh_devices_button)
        device_layout.addWidget(self.apply_device_button)
        audio_layout.addLayout(device_layout)
        self.device_status_label = QLabel("")
        self.device_status_label.setStyleSheet(self._settings_status_default_style)
        audio_layout.addWidget(self.device_status_label)
        layout.addWidget(audio_group)

        self.refresh_devices_button.clicked.connect(self._refresh_audio_inputs)
        self.apply_device_button.clicked.connect(self._apply_selected_audio_input)

        # Hotkeys
        hotkey_group = QGroupBox("Input && Hotkeys")
        hotkey_layout = QFormLayout(hotkey_group)
        hotkey_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.ptt_hotkey_edit = QLineEdit(self._config_text("ptt", "hotkey", "<cmd>+<option>"))
        self.ptt_secondary_hotkey_edit = QLineEdit(self._config_text("ptt", "secondary_hotkey", "<fn>"))
        self.ptt_toggle_hotkey_edit = QLineEdit(
            self._config_text("ptt", "toggle_hotkey", "<cmd>+<option>+<shift>")
        )
        self.mode_cycle_tap_key_edit = QLineEdit(self._config_text("ptt", "mode_cycle_tap_key", "<cmd>"))
        self.mode_cycle_tap_count_edit = QLineEdit(
            str(self.config.get("ptt", {}).get("mode_cycle_tap_count", 3))
        )
        self.mode_cycle_tap_window_edit = QLineEdit(
            str(self.config.get("ptt", {}).get("mode_cycle_tap_window_ms", 650))
        )
        self.command_hotkey_edit = QLineEdit(
            self._config_text("window_management", "command_hotkey", "<ctrl>+<cmd>")
        )
        self.window_prefix_hotkey_edit = QLineEdit(
            self._config_text("window_management", "hotkey_prefix", "<ctrl>+<cmd>")
        )
        voice_prefixes = self.config.get("window_management", {}).get(
            "voice_command_prefixes",
            ["run command", "screen", "window", "desktop"],
        )
        if isinstance(voice_prefixes, list):
            voice_prefixes = ", ".join(str(prefix) for prefix in voice_prefixes)
        self.voice_command_prefixes_edit = QLineEdit(str(voice_prefixes or "run command, screen, window, desktop"))
        hotkey_placeholders = {
            self.ptt_hotkey_edit: "<cmd>+<option>",
            self.ptt_secondary_hotkey_edit: "<fn>",
            self.ptt_toggle_hotkey_edit: "<cmd>+<option>+<shift>",
            self.mode_cycle_tap_key_edit: "<cmd>",
            self.mode_cycle_tap_count_edit: "3",
            self.mode_cycle_tap_window_edit: "650",
            self.command_hotkey_edit: "<ctrl>+<cmd>",
            self.window_prefix_hotkey_edit: "<ctrl>+<cmd>",
            self.voice_command_prefixes_edit: "run command, screen, window, desktop",
        }
        for edit, placeholder in hotkey_placeholders.items():
            edit.setPlaceholderText(placeholder)
            edit.setMinimumWidth(260)
        hotkey_layout.addRow("Primary PTT:", self.ptt_hotkey_edit)
        hotkey_layout.addRow("Secondary PTT:", self.ptt_secondary_hotkey_edit)
        hotkey_layout.addRow("Toggle PTT:", self.ptt_toggle_hotkey_edit)
        hotkey_layout.addRow("Mode tap key:", self.mode_cycle_tap_key_edit)
        hotkey_layout.addRow("Mode tap count:", self.mode_cycle_tap_count_edit)
        hotkey_layout.addRow("Mode tap window ms:", self.mode_cycle_tap_window_edit)
        hotkey_layout.addRow("Command PTT:", self.command_hotkey_edit)
        hotkey_layout.addRow("Window prefix:", self.window_prefix_hotkey_edit)
        hotkey_layout.addRow("Voice prefixes:", self.voice_command_prefixes_edit)
        hotkey_actions = QHBoxLayout()
        self.apply_hotkeys_button = QPushButton("Apply Hotkeys")
        hotkey_actions.addWidget(self.apply_hotkeys_button)
        hotkey_actions.addStretch()
        hotkey_layout.addRow("", hotkey_actions)
        self.hotkey_status_label = QLabel("")
        self.hotkey_status_label.setStyleSheet(self._settings_status_default_style)
        hotkey_layout.addRow("", self.hotkey_status_label)
        layout.addWidget(hotkey_group)
        self.apply_hotkeys_button.clicked.connect(self._apply_hotkey_settings)

        # Voice settings
        voice_group = QGroupBox("Voice Verification")
        voice_layout = QVBoxLayout(voice_group)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.voice_mode_combo = QComboBox()
        self.voice_mode_combo.addItem("Verified voice", "whisper")
        self.voice_mode_combo.addItem("Open talk", "talk")
        self.apply_voice_mode_button = QPushButton("Apply Mode")
        mode_layout.addWidget(self.voice_mode_combo, 1)
        mode_layout.addWidget(self.apply_voice_mode_button)
        voice_layout.addLayout(mode_layout)

        threshold_header = QHBoxLayout()
        threshold_header.addWidget(QLabel("Sensitivity:"))
        self.voice_threshold_value = QLabel("--")
        threshold_header.addStretch()
        threshold_header.addWidget(self.voice_threshold_value)
        voice_layout.addLayout(threshold_header)

        self.voice_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.voice_threshold_slider.setMinimum(30)
        self.voice_threshold_slider.setMaximum(95)
        self.voice_threshold_slider.setSingleStep(1)
        self.voice_threshold_slider.setPageStep(2)
        voice_layout.addWidget(self.voice_threshold_slider)

        threshold_actions = QHBoxLayout()
        self.apply_voice_threshold_button = QPushButton("Apply Sensitivity")
        threshold_actions.addWidget(self.apply_voice_threshold_button)
        threshold_actions.addStretch()
        voice_layout.addLayout(threshold_actions)

        self.voice_profile_label = QLabel("Profile: --")
        self.voice_profile_label.setStyleSheet(self._settings_status_default_style)
        voice_layout.addWidget(self.voice_profile_label)

        profile_actions = QHBoxLayout()
        self.capture_sample_button = QPushButton("Record 3s Sample")
        self.clear_profile_button = QPushButton("Reset Profile")
        profile_actions.addWidget(self.capture_sample_button)
        profile_actions.addWidget(self.clear_profile_button)
        voice_layout.addLayout(profile_actions)

        self.voice_settings_status_label = QLabel("")
        self.voice_settings_status_label.setStyleSheet(self._settings_status_default_style)
        voice_layout.addWidget(self.voice_settings_status_label)
        layout.addWidget(voice_group)

        self.apply_voice_mode_button.clicked.connect(self._apply_voice_mode)
        self.apply_voice_threshold_button.clicked.connect(self._apply_voice_threshold)
        self.capture_sample_button.clicked.connect(self._capture_voice_sample)
        self.clear_profile_button.clicked.connect(self._clear_voice_profile)
        self.voice_threshold_slider.valueChanged.connect(self._update_threshold_preview)

        # Dictation behavior
        dictation_group = QGroupBox("Dictation")
        dictation_layout = QFormLayout(dictation_group)
        tx_cfg = self.config.get("transcription", {})
        ns_cfg = self.config.get("noise_suppression", {})
        self.final_pass_combo = QComboBox()
        for value in self._model_options().get("final_pass_modes", ["hybrid", "prerecorded", "streaming"]):
            self.final_pass_combo.addItem(value.title(), value)
        self._set_combo_data(self.final_pass_combo, tx_cfg.get("final_pass", "hybrid"))
        self.output_format_combo = QComboBox()
        for value in self._model_options().get("output_formats", ["clipboard", "stdout", "both"]):
            self.output_format_combo.addItem(value.title(), value)
        self._set_combo_data(self.output_format_combo, tx_cfg.get("output_format", "clipboard"))
        self.auto_paste_checkbox = QCheckBox("Auto-paste after transcription")
        self.auto_paste_checkbox.setChecked(bool(tx_cfg.get("auto_paste", True)))
        self.use_dictionary_checkbox = QCheckBox("Apply dictionary corrections")
        self.use_dictionary_checkbox.setChecked(bool(tx_cfg.get("use_custom_dictionary", True)))
        self.noise_suppression_checkbox = QCheckBox("Enable noise suppression")
        self.noise_suppression_checkbox.setChecked(bool(ns_cfg.get("enabled", True)))
        self.history_enabled_checkbox = QCheckBox("Save local transcript history")
        self.history_enabled_checkbox.setChecked(bool(self.config.get("history", {}).get("enabled", True)))
        dictation_layout.addRow("Final pass:", self.final_pass_combo)
        dictation_layout.addRow("Output:", self.output_format_combo)
        dictation_layout.addRow("", self.auto_paste_checkbox)
        dictation_layout.addRow("", self.use_dictionary_checkbox)
        dictation_layout.addRow("", self.noise_suppression_checkbox)
        dictation_layout.addRow("", self.history_enabled_checkbox)
        dictation_actions = QHBoxLayout()
        self.apply_dictation_button = QPushButton("Apply Dictation Settings")
        dictation_actions.addWidget(self.apply_dictation_button)
        dictation_actions.addStretch()
        dictation_layout.addRow("", dictation_actions)
        self.dictation_status_label = QLabel("")
        self.dictation_status_label.setStyleSheet(self._settings_status_default_style)
        dictation_layout.addRow("", self.dictation_status_label)
        layout.addWidget(dictation_group)
        self.apply_dictation_button.clicked.connect(self._apply_dictation_settings)

        # Models and providers
        model_group = QGroupBox("Models && Providers")
        model_layout = QFormLayout(model_group)
        self.provider_combo = QComboBox()
        for provider in self._model_options().get("providers", []):
            self.provider_combo.addItem(provider.get("label", provider.get("value", "")), provider.get("value", ""))
        self._set_combo_data(self.provider_combo, tx_cfg.get("provider", "deepgram"))
        self.whisper_model_combo = self._model_combo("whisper_models", tx_cfg.get("model", "medium.en"))
        self.whisper_fallback_combo = self._model_combo(
            "whisper_models", tx_cfg.get("whisper_fallback_model", "medium.en")
        )
        self.deepgram_model_combo = self._model_combo(
            "deepgram_models", self.config.get("deepgram", {}).get("model", "nova-3")
        )
        self.deepgram_prerecorded_model_combo = self._model_combo(
            "deepgram_models", self.config.get("deepgram", {}).get("prerecorded_model", "nova-3")
        )
        self.openai_model_combo = self._model_combo(
            "openai_models", self.config.get("openai", {}).get("model", "gpt-4o-transcribe")
        )
        priority = tx_cfg.get("final_pass_provider_priority", ["openai", "deepgram", "whisper"])
        if isinstance(priority, list):
            priority = ", ".join(str(item) for item in priority)
        self.provider_priority_edit = QLineEdit(str(priority))
        self.openai_key_edit = QLineEdit()
        self.openai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key_edit.setPlaceholderText("Paste to replace")
        self.openai_key_source_label = QLabel("")
        self.openai_key_source_label.setStyleSheet(self._settings_status_default_style)
        self.deepgram_key_edit = QLineEdit()
        self.deepgram_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepgram_key_edit.setPlaceholderText("Paste to replace")
        self.deepgram_key_source_label = QLabel("")
        self.deepgram_key_source_label.setStyleSheet(self._settings_status_default_style)
        openai_key_widget = self._key_field_widget(self.openai_key_edit, self.openai_key_source_label)
        deepgram_key_widget = self._key_field_widget(self.deepgram_key_edit, self.deepgram_key_source_label)
        model_layout.addRow("Primary provider:", self.provider_combo)
        model_layout.addRow("Whisper model:", self.whisper_model_combo)
        model_layout.addRow("Whisper fallback:", self.whisper_fallback_combo)
        model_layout.addRow("Deepgram stream:", self.deepgram_model_combo)
        model_layout.addRow("Deepgram final:", self.deepgram_prerecorded_model_combo)
        model_layout.addRow("OpenAI STT:", self.openai_model_combo)
        model_layout.addRow("Final priority:", self.provider_priority_edit)
        model_layout.addRow("OpenAI key:", openai_key_widget)
        model_layout.addRow("Deepgram key:", deepgram_key_widget)
        model_actions = QHBoxLayout()
        self.apply_models_button = QPushButton("Apply Models")
        self.save_api_keys_button = QPushButton("Save API Keys")
        model_actions.addWidget(self.apply_models_button)
        model_actions.addWidget(self.save_api_keys_button)
        model_actions.addStretch()
        model_layout.addRow("", model_actions)
        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet(self._settings_status_default_style)
        model_layout.addRow("", self.model_status_label)
        layout.addWidget(model_group)
        self.apply_models_button.clicked.connect(self._apply_model_settings)
        self.save_api_keys_button.clicked.connect(self._save_api_keys)
        self._refresh_secret_status()

        # Post-processing
        cleanup_group = QGroupBox("Output Cleanup")
        cleanup_layout = QFormLayout(cleanup_group)
        pp_cfg = self.config.get("post_processing", {})
        self.post_processing_mode_combo = QComboBox()
        cleanup_labels = {
            "verbatim": "Verbatim",
            "clean": "Clean prose",
            "coding": "Coding",
            "message": "Message",
        }
        for value in self._model_options().get("post_processing_modes", ["verbatim", "clean", "coding", "message"]):
            self.post_processing_mode_combo.addItem(cleanup_labels.get(value, value.title()), value)
        self._set_combo_data(self.post_processing_mode_combo, pp_cfg.get("mode", "verbatim"))
        self.post_processing_mode_combo.setToolTip(
            "Verbatim keeps model output as-is. Clean removes filler and normalizes prose. "
            "Coding keeps code-like dictated text less rewritten. Message is concise prose, "
            "and is most useful with OpenAI cleanup enabled."
        )
        self.openai_cleanup_checkbox = QCheckBox("Use OpenAI cleanup when available")
        self.openai_cleanup_checkbox.setChecked(bool(pp_cfg.get("openai_enabled", True)))
        self.cleanup_model_combo = self._model_combo(
            "cleanup_models", pp_cfg.get("openai_model", "gpt-4o")
        )
        cleanup_layout.addRow("Mode:", self.post_processing_mode_combo)
        cleanup_layout.addRow("OpenAI model:", self.cleanup_model_combo)
        cleanup_layout.addRow("", self.openai_cleanup_checkbox)
        cleanup_actions = QHBoxLayout()
        self.apply_cleanup_button = QPushButton("Apply Cleanup")
        cleanup_actions.addWidget(self.apply_cleanup_button)
        cleanup_actions.addStretch()
        cleanup_layout.addRow("", cleanup_actions)
        self.cleanup_status_label = QLabel("")
        self.cleanup_status_label.setStyleSheet(self._settings_status_default_style)
        cleanup_layout.addRow("", self.cleanup_status_label)
        layout.addWidget(cleanup_group)
        self.apply_cleanup_button.clicked.connect(self._apply_cleanup_settings)

        # Dictionary settings
        dictionary_group = QGroupBox("Dictionary")
        dictionary_layout = QVBoxLayout(dictionary_group)
        dictionary_layout.setSpacing(12)
        self.dictionary_path_label = QLabel("")
        self.dictionary_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.dictionary_path_label.setStyleSheet(self._settings_status_default_style)
        dictionary_layout.addWidget(self.dictionary_path_label)

        dictionary_actions = QHBoxLayout()
        self.open_dictionary_button = QPushButton("Open")
        self.reload_dictionary_button = QPushButton("Reload")
        self.initialize_dictionary_button = QPushButton("Initialize")
        dictionary_actions.addWidget(self.open_dictionary_button)
        dictionary_actions.addWidget(self.reload_dictionary_button)
        dictionary_actions.addWidget(self.initialize_dictionary_button)
        dictionary_layout.addLayout(dictionary_actions)

        self.dictionary_status_label = QLabel("")
        self.dictionary_status_label.setStyleSheet(self._settings_status_default_style)
        dictionary_layout.addWidget(self.dictionary_status_label)

        preferred_layout = QVBoxLayout()
        preferred_layout.addWidget(QLabel("Preferred words and phrases"))
        add_term_layout = QHBoxLayout()
        self.dictionary_term_edit = QLineEdit()
        self.dictionary_term_edit.setPlaceholderText("Bloviate, kubectl, Callum Reid")
        self.add_dictionary_term_button = QPushButton("Add Word")
        add_term_layout.addWidget(self.dictionary_term_edit, 1)
        add_term_layout.addWidget(self.add_dictionary_term_button)
        preferred_layout.addLayout(add_term_layout)
        self.dictionary_terms_table = QTableWidget(0, 2)
        self.dictionary_terms_table.setHorizontalHeaderLabels(["Word or phrase", ""])
        self.dictionary_terms_table.verticalHeader().setVisible(False)
        self.dictionary_terms_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.dictionary_terms_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.dictionary_terms_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.dictionary_terms_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.dictionary_terms_table.setMinimumHeight(120)
        preferred_layout.addWidget(self.dictionary_terms_table)
        dictionary_layout.addLayout(preferred_layout)

        correction_layout = QVBoxLayout()
        correction_layout.addWidget(QLabel("Replacement rules"))
        add_correction_layout = QHBoxLayout()
        self.dictionary_wrong_edit = QLineEdit()
        self.dictionary_wrong_edit.setPlaceholderText("what gets transcribed")
        self.dictionary_right_edit = QLineEdit()
        self.dictionary_right_edit.setPlaceholderText("what it should say")
        self.dictionary_match_combo = QComboBox()
        self.dictionary_match_combo.addItem("Contains", "substring")
        self.dictionary_match_combo.addItem("Whole word", "whole_word")
        self.add_dictionary_correction_button = QPushButton("Add Replacement")
        add_correction_layout.addWidget(self.dictionary_wrong_edit, 1)
        add_correction_layout.addWidget(QLabel("->"))
        add_correction_layout.addWidget(self.dictionary_right_edit, 1)
        add_correction_layout.addWidget(self.dictionary_match_combo)
        add_correction_layout.addWidget(self.add_dictionary_correction_button)
        correction_layout.addLayout(add_correction_layout)
        self.dictionary_corrections_table = QTableWidget(0, 4)
        self.dictionary_corrections_table.setHorizontalHeaderLabels(["Replace", "With", "Match", ""])
        self.dictionary_corrections_table.verticalHeader().setVisible(False)
        self.dictionary_corrections_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.dictionary_corrections_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.dictionary_corrections_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.dictionary_corrections_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.dictionary_corrections_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.dictionary_corrections_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.dictionary_corrections_table.setMinimumHeight(150)
        correction_layout.addWidget(self.dictionary_corrections_table)
        dictionary_layout.addLayout(correction_layout)
        layout.addWidget(dictionary_group)

        self.open_dictionary_button.clicked.connect(self._open_dictionary)
        self.reload_dictionary_button.clicked.connect(self._reload_dictionary)
        self.initialize_dictionary_button.clicked.connect(self._initialize_dictionary)
        self.add_dictionary_term_button.clicked.connect(self._add_dictionary_term)
        self.dictionary_term_edit.returnPressed.connect(self._add_dictionary_term)
        self.add_dictionary_correction_button.clicked.connect(self._add_dictionary_correction)
        self.dictionary_right_edit.returnPressed.connect(self._add_dictionary_correction)
        self._load_dictionary_editor()

        # History
        history_group = QGroupBox("History")
        history_layout = QVBoxLayout(history_group)
        history_search_layout = QHBoxLayout()
        self.history_search_edit = QLineEdit()
        self.history_search_edit.setPlaceholderText("Search local transcript history")
        self.refresh_history_button = QPushButton("Refresh")
        history_search_layout.addWidget(self.history_search_edit, 1)
        history_search_layout.addWidget(self.refresh_history_button)
        history_layout.addLayout(history_search_layout)
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(["Date", "Mode", "Provider", "Target", "Text"])
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setMinimumHeight(180)
        history_layout.addWidget(self.history_table)
        history_actions = QHBoxLayout()
        self.copy_history_button = QPushButton("Copy")
        self.delete_history_button = QPushButton("Delete")
        self.clear_history_button = QPushButton("Clear All")
        self.export_history_button = QPushButton("Export CSV")
        history_actions.addWidget(self.copy_history_button)
        history_actions.addWidget(self.delete_history_button)
        history_actions.addWidget(self.clear_history_button)
        history_actions.addWidget(self.export_history_button)
        history_actions.addStretch()
        history_layout.addLayout(history_actions)
        self.history_status_label = QLabel("")
        self.history_status_label.setStyleSheet(self._settings_status_default_style)
        history_layout.addWidget(self.history_status_label)
        layout.addWidget(history_group)
        self.refresh_history_button.clicked.connect(self._refresh_history)
        self.history_search_edit.returnPressed.connect(self._refresh_history)
        self.history_search_edit.textChanged.connect(self._refresh_history)
        self.copy_history_button.clicked.connect(self._copy_history_selection)
        self.delete_history_button.clicked.connect(self._delete_history_selection)
        self.clear_history_button.clicked.connect(self._clear_history)
        self.export_history_button.clicked.connect(self._export_history)
        self._refresh_history()

        # Startup preferences
        startup_group = QGroupBox("Startup")
        startup_layout = QVBoxLayout(startup_group)
        self.show_window_checkbox = QCheckBox("Show dashboard window on startup")
        self.show_window_checkbox.setChecked(bool(self.config.get("ui", {}).get("show_main_window", True)))
        splash_cfg = self.config.get("ui", {}).get("startup_splash", {})
        self.show_splash_checkbox = QCheckBox("Show native splash on startup")
        self.show_splash_checkbox.setChecked(bool(splash_cfg.get("enabled", True)))
        self.show_terminal_animation_checkbox = QCheckBox("Show terminal cow animation on startup")
        self.show_terminal_animation_checkbox.setChecked(
            bool(self.config.get("app", {}).get("startup_animation", True))
        )
        startup_layout.addWidget(self.show_window_checkbox)
        startup_layout.addWidget(self.show_splash_checkbox)
        startup_layout.addWidget(self.show_terminal_animation_checkbox)
        self.startup_status_label = QLabel("")
        self.startup_status_label.setStyleSheet(self._settings_status_default_style)
        startup_layout.addWidget(self.startup_status_label)
        layout.addWidget(startup_group)

        self.show_window_checkbox.stateChanged.connect(self._toggle_show_window_on_startup)
        self.show_splash_checkbox.stateChanged.connect(self._toggle_splash_on_startup)
        self.show_terminal_animation_checkbox.stateChanged.connect(self._toggle_terminal_animation_on_startup)

        # Advanced
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QVBoxLayout(advanced_group)
        self.paths_label = QLabel(self._paths_text())
        self.paths_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.paths_label.setStyleSheet(self._settings_status_default_style)
        advanced_layout.addWidget(self.paths_label)
        advanced_actions = QHBoxLayout()
        self.run_doctor_button = QPushButton("Run Doctor")
        self.reset_defaults_button = QPushButton("Reset Defaults")
        advanced_actions.addWidget(self.run_doctor_button)
        advanced_actions.addWidget(self.reset_defaults_button)
        advanced_actions.addStretch()
        advanced_layout.addLayout(advanced_actions)
        self.doctor_output = QTextEdit()
        self.doctor_output.setReadOnly(True)
        self.doctor_output.setMinimumHeight(150)
        advanced_layout.addWidget(self.doctor_output)
        layout.addWidget(advanced_group)
        self.run_doctor_button.clicked.connect(self._run_doctor)
        self.reset_defaults_button.clicked.connect(self._reset_defaults)

        layout.addStretch()

    def _permission_statuses(self) -> dict:
        if not self.get_permission_statuses:
            return {}
        try:
            return self.get_permission_statuses() or {}
        except Exception as exc:
            return {
                "error": {
                    "label": "Permissions",
                    "state": "missing",
                    "detail": f"Could not check permissions: {exc}",
                }
            }

    def _refresh_permissions(self):
        if not hasattr(self, "permissions_status_label"):
            return
        statuses = self._permission_statuses()
        if not statuses:
            self.permissions_status_label.setText("Permission checks are unavailable on this platform.")
            return

        lines = []
        all_ready = True
        for key in ("microphone", "accessibility", "input_monitoring", "automation"):
            status = statuses.get(key)
            if not status:
                continue
            label = status.get("label", key.replace("_", " ").title())
            state = status.get("state", "unknown")
            detail = status.get("detail", "")
            if state == "granted":
                prefix = "OK"
            elif state == "manual":
                prefix = "Check"
                all_ready = False
            else:
                prefix = "Needs setup"
                all_ready = False
            lines.append(f"{prefix}: {label}" + (f" - {detail}" if detail else ""))

        if not lines:
            lines.append("Permission checks are unavailable on this platform.")
        self.permissions_status_label.setText("\n".join(lines))
        self.permissions_status_label.setStyleSheet(
            self._settings_ok_style if all_ready else self._settings_status_default_style
        )

    def _request_permission(self, kind: str):
        if self.request_permission:
            ok, message = self.request_permission(kind)
            self._refresh_permissions()
            self._set_settings_status(self.permissions_status_label, message, ok=ok)
            return
        if self.open_permission_settings:
            ok, message = self.open_permission_settings(kind)
            self._refresh_permissions()
            self._set_settings_status(self.permissions_status_label, message, ok=ok)
            return
        self._set_settings_status(self.permissions_status_label, "Permission setup is unavailable.", ok=False)

    def _toggle_dictation_from_ui(self):
        if not self.toggle_dictation:
            self.status_label.setText("Dictation control unavailable")
            return
        try:
            self.toggle_dictation()
        except Exception as exc:
            self.status_label.setText(f"Could not toggle dictation: {exc}")

    def _maybe_show_permissions_prompt(self):
        if self._permissions_prompt_shown:
            return
        prompt_cfg = self.config.get("ui", {}).get("permissions_prompt", {})
        if not bool(prompt_cfg.get("enabled", True)):
            return
        statuses = self._permission_statuses()
        missing = [
            status
            for status in statuses.values()
            if status.get("state") in {"missing", "unknown"}
        ]
        if not missing:
            return

        self._permissions_prompt_shown = True
        self.show_settings_tab()
        self._refresh_permissions()
        message = (
            "Bloviate needs microphone access for dictation, Accessibility/Input Monitoring "
            "for global hotkeys, and Automation/Accessibility for auto-paste. "
            "Use the Permissions section at the top of Settings to open each macOS prompt."
        )
        self._set_settings_status(self.permissions_status_label, message, ok=False)
        self.status_label.setText("Permissions needed")
        if self.ptt_overlay:
            self.ptt_overlay.show_message("PERMISSIONS", state="rejected", hold_ms=2600)
        QTimer.singleShot(0, self.raise_)
        QTimer.singleShot(0, self.activateWindow)

    def _refresh_audio_inputs(self):
        """Reload the list of available audio inputs."""
        self.device_combo.clear()
        self.device_combo.addItem("System Default", "")

        if not self._audio_inputs_ready:
            self.device_status_label.setText("Audio input switching unavailable.")
            self.device_combo.setEnabled(False)
            self.refresh_devices_button.setEnabled(False)
            self.apply_device_button.setEnabled(False)
            return

        try:
            devices = self.get_audio_inputs() or []
        except Exception as exc:
            self.device_status_label.setText(f"Could not load inputs: {exc}")
            return

        current = str(self.config.get("audio", {}).get("device_name", "") or "").strip()
        selected_index = 0

        for offset, device in enumerate(devices, start=1):
            label = str(device.get("name", "Unknown Input"))
            if device.get("is_default"):
                label += " (Default)"
            channels = int(device.get("channels", 0))
            if channels:
                label += f" [{channels}ch]"
            name = str(device.get("name", "") or "").strip()
            self.device_combo.addItem(label, name)
            if name == current:
                selected_index = offset

        self.device_combo.setCurrentIndex(selected_index)
        self.device_status_label.setText(
            f"{len(devices)} input device(s) available. Current: {current or 'System Default'}"
        )
        if self.menu_bar_indicator:
            self.menu_bar_indicator.refresh_audio_inputs_menu()

    def get_audio_input_options(self):
        """Return audio inputs for UI surfaces like the tray menu."""
        if not self.get_audio_inputs:
            return []
        return self.get_audio_inputs() or []

    def get_current_audio_input_name(self) -> str:
        """Return the configured current input name, or empty for system default."""
        return str(self.config.get("audio", {}).get("device_name", "") or "").strip()

    def switch_audio_input(self, selected: str) -> tuple[bool, str]:
        """Switch input device and refresh all UI affordances."""
        if not self._audio_inputs_ready:
            message = "Audio input switching unavailable."
            self.device_status_label.setText(message)
            return False, message

        selected_name = str(selected or "").strip()
        ok, message = self.set_audio_input(selected_name)
        self.device_status_label.setText(message)
        if ok:
            self.config.setdefault("audio", {})["device_name"] = selected_name
            self._refresh_audio_inputs()
        return ok, message

    def _apply_selected_audio_input(self):
        """Switch the active audio input to the selected device."""
        selected = str(self.device_combo.currentData() or "").strip()
        self.switch_audio_input(selected)

    def _model_options(self) -> dict:
        if self.get_model_options:
            try:
                return self.get_model_options() or {}
            except Exception:
                return {}
        return {}

    def _set_combo_data(self, combo: QComboBox, value):
        index = combo.findData(value)
        if index < 0:
            index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _config_text(self, section: str, key: str, default: str = "") -> str:
        value = self.config.get(section, {}).get(key)
        text = str(value or "").strip()
        return text or default

    def _easter_config(self) -> dict:
        ui_config = self.config.setdefault("ui", {})
        easter_config = ui_config.setdefault("easter_eggs", {})
        if not isinstance(easter_config, dict):
            easter_config = {}
            ui_config["easter_eggs"] = easter_config
        return easter_config

    def _easter_enabled(self) -> bool:
        return bool(self._easter_config().get("enabled", True))

    def _secret_themes_unlocked(self) -> bool:
        return bool(self._easter_config().get("secret_themes_unlocked", False))

    def _increment_easter_counter(self, key: str, *, value: int | None = None):
        easter_config = self._easter_config()
        current = int(easter_config.get(key, 0) or 0)
        easter_config[key] = int(value if value is not None else current + 1)
        if self.set_general_settings:
            self.set_general_settings({f"ui.easter_eggs.{key}": easter_config[key]})

    def _set_easter_flag(self, key: str, value: bool):
        easter_config = self._easter_config()
        easter_config[key] = bool(value)
        if self.set_general_settings:
            self.set_general_settings({f"ui.easter_eggs.{key}": bool(value)})

    def _key_field_widget(self, line_edit: QLineEdit, status_label: QLabel) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(line_edit)
        layout.addWidget(status_label)
        return widget

    def _color_line_edit(self, value: str) -> QLineEdit:
        edit = QLineEdit(normalize_hex(value, "#000000"))
        edit.setMaxLength(7)
        edit.setMinimumWidth(110)
        edit.setPlaceholderText("#2D6B6B")
        return edit

    def _color_field_widget(self, line_edit: QLineEdit) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        button = QPushButton("Pick")
        button.clicked.connect(lambda _checked=False, edit=line_edit: self._choose_color(edit))
        layout.addWidget(line_edit)
        layout.addWidget(button)
        layout.addStretch()
        return widget

    def _choose_color(self, line_edit: QLineEdit):
        initial = QColor(normalize_hex(line_edit.text(), "#2D6B6B"))
        color = QColorDialog.getColor(initial, self, "Choose waveform color")
        if color.isValid():
            line_edit.setText(color.name().upper())

    def _handle_title_click(self, event):
        if event and event.modifiers() & Qt.KeyboardModifier.AltModifier:
            self.show_bloviate_labs()
            return
        if not self._easter_enabled():
            return
        self._title_click_count += 1
        if self._title_click_count >= 5:
            self._title_click_count = 0
            self.unlock_secret_themes(source="title")

    def unlock_secret_themes(self, source: str = "manual") -> bool:
        if not self._easter_enabled():
            return False
        already_unlocked = self._secret_themes_unlocked()
        if not already_unlocked:
            self._set_easter_flag("secret_themes_unlocked", True)
            self._refresh_theme_combo(include_hidden=True)
        if self.ptt_overlay:
            self.ptt_overlay.show_message("SECRET THEMES", state="mode", hold_ms=2200)
        if hasattr(self, "appearance_status_label"):
            message = "Secret themes unlocked." if not already_unlocked else "Secret themes already unlocked."
            self._set_settings_status(self.appearance_status_label, message, ok=True)
        return True

    def _refresh_theme_combo(self, *, include_hidden: bool | None = None):
        if not hasattr(self, "theme_combo"):
            return
        include_hidden = self._secret_themes_unlocked() if include_hidden is None else include_hidden
        current = self.theme_combo.currentData() or self._theme_id
        with QSignalBlocker(self.theme_combo):
            self.theme_combo.clear()
            for theme_id, label in theme_options(include_hidden=bool(include_hidden)):
                self.theme_combo.addItem(label, theme_id)
            self._set_combo_data(self.theme_combo, current)

    def activate_easter_theme(self, theme_id: str) -> bool:
        theme_id = normalize_theme_id(theme_id)
        if not is_hidden_theme(theme_id):
            return False
        self.unlock_secret_themes(source="voice")
        if hasattr(self, "theme_combo"):
            self._set_combo_data(self.theme_combo, theme_id)
        self.apply_theme(theme_id)
        self.config.setdefault("ui", {})["theme"] = theme_id
        self._increment_easter_counter("secret_theme_activations")
        if self.set_general_settings:
            self.set_general_settings({"ui.theme": theme_id})
        if self.ptt_overlay:
            self.ptt_overlay.show_message(get_theme(theme_id)["label"].upper(), state="mode", hold_ms=2200)
        if hasattr(self, "appearance_status_label"):
            self._set_settings_status(self.appearance_status_label, f"Activated {get_theme(theme_id)['label']}.", ok=True)
        return True

    def show_bloviate_labs(self):
        self._set_easter_flag("about_opened", True)
        insights = {}
        if self.get_history_insights:
            try:
                insights = self.get_history_insights() or {}
            except Exception:
                insights = {}
        total_words = self._format_int(insights.get("total_words", 0))
        total_transcripts = self._format_int(insights.get("total_transcripts", 0))
        theme_label = self._theme.get("label", self._theme_id)
        text = (
            f"<b>Bloviate Labs</b><br><br>"
            f"Version: {self.config.get('app', {}).get('version', 'local')}<br>"
            f"Theme: {theme_label}<br>"
            f"Local words: {total_words}<br>"
            f"Saved clips: {total_transcripts}<br><br>"
            "Open source dictation, unusually willing to count things."
        )
        QMessageBox.information(self, "Bloviate Labs", text)

    def run_cow_runway(self):
        if not self._easter_enabled():
            return
        self._increment_easter_counter("cow_runs")
        if not hasattr(self, "cow_runway_overlay"):
            self.cow_runway_overlay = CowRunwayOverlay(self)
        self.cow_runway_overlay.start()
        if self.ptt_overlay:
            self.ptt_overlay.show_message("COWS", state="mode", hold_ms=1600)

    def surprise_waveform(self):
        if not self._easter_enabled():
            return
        self._increment_easter_counter("surprise_count")
        original_palette = self._waveform_palette()
        colors = {
            "idle": "#8E5CF7",
            "recording": "#E7C873",
            "command": "#62A8E5",
            "accepted": "#6AD49F",
            "rejected": "#EF6F78",
            "quiet": "#B9B0C9",
            "background": original_palette.get("background", "#FFFDF7"),
            "text": original_palette.get("text", "#26211D"),
            "processing": ["#E7C873", "#8E5CF7", "#62A8E5", "#6AD49F", "#EF6F78"],
        }
        self._apply_waveform_palette(colors)
        if self.ptt_overlay:
            self.ptt_overlay.show_message("SURPRISE", state="mode", hold_ms=1800)
        QTimer.singleShot(3500, lambda: self._apply_waveform_palette(self._waveform_palette()))

    def _waveform_palette(self) -> dict:
        return waveform_palette_for_config(self.config)

    def _current_waveform_values_from_fields(self) -> dict:
        theme_id = self.theme_combo.currentData() if hasattr(self, "theme_combo") else self._theme_id
        fallback = waveform_values_for_preset(theme_id, "theme")
        processing = [
            part.strip()
            for part in self.waveform_processing_edit.text().split(",")
            if part.strip()
        ]
        processing = [
            normalize_hex(part, "")
            for part in processing
            if normalize_hex(part, "")
        ]
        return {
            "idle": normalize_hex(self.waveform_idle_edit.text(), fallback["idle"]),
            "recording": normalize_hex(self.waveform_recording_edit.text(), fallback["recording"]),
            "command": normalize_hex(self.waveform_command_edit.text(), fallback["command"]),
            "accepted": normalize_hex(self.waveform_accepted_edit.text(), fallback["accepted"]),
            "rejected": normalize_hex(self.waveform_rejected_edit.text(), fallback["rejected"]),
            "quiet": normalize_hex(self.waveform_idle_edit.text(), fallback["quiet"]),
            "background": fallback["background"],
            "text": fallback["text"],
            "processing": processing or list(fallback["processing"]),
        }

    def _set_waveform_fields(self, values: dict):
        self.waveform_idle_edit.setText(str(values.get("idle", "#BFB2A1")).upper())
        self.waveform_recording_edit.setText(str(values.get("recording", "#E7C873")).upper())
        self.waveform_command_edit.setText(str(values.get("command", "#2D6B6B")).upper())
        self.waveform_accepted_edit.setText(str(values.get("accepted", "#2F7D4F")).upper())
        self.waveform_rejected_edit.setText(str(values.get("rejected", "#B23B35")).upper())
        self.waveform_processing_edit.setText(", ".join(values.get("processing", [])))

    def _preview_theme_selection(self):
        theme_id = normalize_theme_id(self.theme_combo.currentData())
        self.apply_theme(theme_id)
        if self.waveform_preset_combo.currentData() == "theme":
            self._set_waveform_fields(waveform_values_for_preset(theme_id, "theme"))
        self._apply_waveform_palette_preview()

    def _preview_waveform_preset(self):
        preset_id = normalize_waveform_preset_id(self.waveform_preset_combo.currentData())
        if preset_id != "custom":
            self._set_waveform_fields(
                waveform_values_for_preset(self.theme_combo.currentData(), preset_id)
            )
        self._apply_waveform_palette_preview()

    def _apply_waveform_palette_preview(self):
        original = self.config.setdefault("ui", {}).get("waveform", {})
        theme_id = normalize_theme_id(self.theme_combo.currentData() if hasattr(self, "theme_combo") else self._theme_id)
        preset_id = normalize_waveform_preset_id(
            self.waveform_preset_combo.currentData() if hasattr(self, "waveform_preset_combo") else "theme"
        )
        preview_config = {"ui": {"theme": theme_id, "waveform": {"preset": preset_id}}}
        if preset_id == "custom":
            preview_config["ui"]["waveform"].update(self._current_waveform_values_from_fields())
        palette = waveform_palette_for_config(preview_config)
        self._apply_waveform_palette(palette)
        self.config.setdefault("ui", {})["waveform"] = original

    def _apply_waveform_palette(self, palette: dict):
        if hasattr(self, "audio_bar"):
            self.audio_bar.set_waveform_palette(palette)
        if self.menu_bar_indicator:
            self.menu_bar_indicator.set_waveform_palette(palette)
        if self.ptt_overlay:
            self.ptt_overlay.set_waveform_palette(palette)

    def _model_combo(self, option_key: str, current_value: str) -> QComboBox:
        combo = QComboBox()
        current = str(current_value or "").strip()
        seen = set()
        for option in self._model_options().get(option_key, []):
            value = str(option.get("value", "")).strip()
            if not value:
                continue
            seen.add(value)
            label = option.get("label") or value
            combo.addItem(label, value)
        if current and current not in seen:
            combo.addItem(current, current)
        self._set_combo_data(combo, current)
        return combo

    def _apply_appearance_settings(self):
        theme_id = normalize_theme_id(self.theme_combo.currentData())
        preset_id = normalize_waveform_preset_id(self.waveform_preset_combo.currentData())
        updates = {
            "ui.theme": theme_id,
            "ui.waveform.preset": preset_id,
            "ui.easter_eggs.milestone_toasts": self.milestone_toasts_checkbox.isChecked(),
        }
        if preset_id == "custom":
            values = self._current_waveform_values_from_fields()
            for key in ("idle", "recording", "command", "accepted", "rejected", "quiet", "background", "text"):
                updates[f"ui.waveform.{key}"] = values[key]
            updates["ui.waveform.processing"] = values["processing"]
        callback = self.set_general_settings
        if not callback:
            self._set_settings_status(self.appearance_status_label, "Appearance updates unavailable.", ok=False)
            return
        ok, message = callback(updates)
        if ok:
            ui_config = self.config.setdefault("ui", {})
            ui_config["theme"] = theme_id
            ui_config.setdefault("easter_eggs", {})["milestone_toasts"] = self.milestone_toasts_checkbox.isChecked()
            waveform_config = ui_config.setdefault("waveform", {})
            waveform_config["preset"] = preset_id
            if preset_id == "custom":
                waveform_config.update(self._current_waveform_values_from_fields())
            self.apply_theme(theme_id)
            self._apply_waveform_palette(self._waveform_palette())
        self._set_settings_status(self.appearance_status_label, message, ok=ok)

    def _apply_hotkey_settings(self):
        voice_prefixes = [
            part.strip()
            for part in self.voice_command_prefixes_edit.text().split(",")
            if part.strip()
        ]
        try:
            mode_tap_count = max(2, int(self.mode_cycle_tap_count_edit.text().strip()))
            mode_tap_window_ms = max(200, int(self.mode_cycle_tap_window_edit.text().strip()))
        except ValueError:
            self._set_settings_status(
                self.hotkey_status_label,
                "Mode tap count and window must be whole numbers.",
                ok=False,
            )
            return
        updates = {
            "ptt.hotkey": self.ptt_hotkey_edit.text().strip(),
            "ptt.secondary_hotkey": self.ptt_secondary_hotkey_edit.text().strip(),
            "ptt.toggle_hotkey": self.ptt_toggle_hotkey_edit.text().strip(),
            "ptt.mode_cycle_tap_key": self.mode_cycle_tap_key_edit.text().strip(),
            "ptt.mode_cycle_tap_count": mode_tap_count,
            "ptt.mode_cycle_tap_window_ms": mode_tap_window_ms,
            "window_management.command_hotkey": self.command_hotkey_edit.text().strip(),
            "window_management.hotkey_prefix": self.window_prefix_hotkey_edit.text().strip(),
            "window_management.voice_command_prefixes": voice_prefixes,
        }
        if not self.set_hotkey_settings:
            self._set_settings_status(self.hotkey_status_label, "Hotkey updates unavailable.", ok=False)
            return
        ok, message = self.set_hotkey_settings(updates)
        if ok:
            self.config.setdefault("ptt", {})["hotkey"] = updates["ptt.hotkey"]
            self.config.setdefault("ptt", {})["secondary_hotkey"] = updates["ptt.secondary_hotkey"]
            self.config.setdefault("ptt", {})["toggle_hotkey"] = updates["ptt.toggle_hotkey"]
            self.config.setdefault("ptt", {})["mode_cycle_tap_key"] = updates["ptt.mode_cycle_tap_key"]
            self.config.setdefault("ptt", {})["mode_cycle_tap_count"] = updates["ptt.mode_cycle_tap_count"]
            self.config.setdefault("ptt", {})["mode_cycle_tap_window_ms"] = updates[
                "ptt.mode_cycle_tap_window_ms"
            ]
            self.config.setdefault("window_management", {})["command_hotkey"] = updates[
                "window_management.command_hotkey"
            ]
            self.config.setdefault("window_management", {})["hotkey_prefix"] = updates[
                "window_management.hotkey_prefix"
            ]
            self.config.setdefault("window_management", {})["voice_command_prefixes"] = updates[
                "window_management.voice_command_prefixes"
            ]
        self._set_settings_status(self.hotkey_status_label, message, ok=ok)

    def _apply_dictation_settings(self):
        updates = {
            "transcription.final_pass": self.final_pass_combo.currentData(),
            "transcription.output_format": self.output_format_combo.currentData(),
            "transcription.auto_paste": self.auto_paste_checkbox.isChecked(),
            "transcription.use_custom_dictionary": self.use_dictionary_checkbox.isChecked(),
            "noise_suppression.enabled": self.noise_suppression_checkbox.isChecked(),
            "history.enabled": self.history_enabled_checkbox.isChecked(),
        }
        callback = self.set_general_settings or self.set_transcription_settings
        if not callback:
            self._set_settings_status(self.dictation_status_label, "Dictation updates unavailable.", ok=False)
            return
        ok, message = callback(updates)
        if ok:
            self.config.setdefault("transcription", {})["final_pass"] = updates["transcription.final_pass"]
            self.config.setdefault("transcription", {})["output_format"] = updates["transcription.output_format"]
            self.config.setdefault("transcription", {})["auto_paste"] = updates["transcription.auto_paste"]
            self.config.setdefault("transcription", {})["use_custom_dictionary"] = updates[
                "transcription.use_custom_dictionary"
            ]
            self.config.setdefault("noise_suppression", {})["enabled"] = updates["noise_suppression.enabled"]
            self.config.setdefault("history", {})["enabled"] = updates["history.enabled"]
        self._set_settings_status(self.dictation_status_label, message, ok=ok)

    def _apply_model_settings(self):
        priority = [
            part.strip()
            for part in self.provider_priority_edit.text().split(",")
            if part.strip()
        ]
        updates = {
            "transcription.provider": self.provider_combo.currentData(),
            "transcription.model": self.whisper_model_combo.currentData(),
            "transcription.whisper_fallback_model": self.whisper_fallback_combo.currentData(),
            "deepgram.model": self.deepgram_model_combo.currentData(),
            "deepgram.prerecorded_model": self.deepgram_prerecorded_model_combo.currentData(),
            "openai.model": self.openai_model_combo.currentData(),
            "transcription.final_pass_provider_priority": priority,
        }
        if not self.set_transcription_settings:
            self._set_settings_status(self.model_status_label, "Model updates unavailable.", ok=False)
            return
        ok, message = self.set_transcription_settings(updates)
        if ok:
            self.config.setdefault("transcription", {})["provider"] = updates["transcription.provider"]
            self.config.setdefault("transcription", {})["model"] = updates["transcription.model"]
            self.config.setdefault("transcription", {})["whisper_fallback_model"] = updates[
                "transcription.whisper_fallback_model"
            ]
            self.config.setdefault("transcription", {})["final_pass_provider_priority"] = updates[
                "transcription.final_pass_provider_priority"
            ]
            self.config.setdefault("deepgram", {})["model"] = updates["deepgram.model"]
            self.config.setdefault("deepgram", {})["prerecorded_model"] = updates["deepgram.prerecorded_model"]
            self.config.setdefault("openai", {})["model"] = updates["openai.model"]
        self._set_settings_status(self.model_status_label, message, ok=ok)
        self._refresh_secret_status()

    def _save_api_keys(self):
        if not self.set_api_key:
            self._set_settings_status(self.model_status_label, "API key storage unavailable.", ok=False)
            return

        messages = []
        ok_all = True
        openai_key = self.openai_key_edit.text().strip()
        deepgram_key = self.deepgram_key_edit.text().strip()
        if openai_key:
            ok, message = self.set_api_key("openai", openai_key)
            ok_all = ok_all and ok
            messages.append(message)
        if deepgram_key:
            ok, message = self.set_api_key("deepgram", deepgram_key)
            ok_all = ok_all and ok
            messages.append(message)
        if not messages:
            messages.append("Paste a key before saving.")
            ok_all = False
        self.openai_key_edit.clear()
        self.deepgram_key_edit.clear()
        self._set_settings_status(self.model_status_label, " ".join(messages), ok=ok_all)
        self._refresh_secret_status()

    def _refresh_secret_status(self):
        if not self.get_secret_statuses or not hasattr(self, "model_status_label"):
            return
        try:
            statuses = self.get_secret_statuses() or {}
        except Exception:
            return
        parts = []
        for provider in ("openai", "deepgram"):
            status = statuses.get(provider, {})
            source = status.get("source", "missing")
            env_name = status.get("env_name", "")
            redacted = status.get("redacted_value", "")
            label = getattr(self, f"{provider}_key_source_label", None)
            edit = getattr(self, f"{provider}_key_edit", None)
            if source == "missing":
                parts.append(f"{provider.title()}: missing")
                if edit:
                    edit.setPlaceholderText(f"Paste {provider.title()} key to save in Keychain")
                if label:
                    label.setText(f"No key found. Environment: {env_name}")
                    label.setStyleSheet(self._settings_error_style)
            else:
                source_text = f"{source}: {redacted}" if redacted else source
                parts.append(f"{provider.title()}: {source_text}")
                if edit:
                    edit.setPlaceholderText(f"{redacted} via {source} - paste to replace")
                if label:
                    label.setText(f"Configured via {source}" + (f" ({env_name})" if env_name else ""))
                    label.setStyleSheet(self._settings_ok_style)
        self.model_status_label.setText(" • ".join(parts))
        self.model_status_label.setStyleSheet(self._settings_status_default_style)

    def _apply_cleanup_settings(self):
        updates = {
            "post_processing.mode": self.post_processing_mode_combo.currentData(),
            "post_processing.openai_enabled": self.openai_cleanup_checkbox.isChecked(),
            "post_processing.openai_model": self.cleanup_model_combo.currentData(),
        }
        callback = self.set_general_settings or self.set_transcription_settings
        if not callback:
            self._set_settings_status(self.cleanup_status_label, "Cleanup updates unavailable.", ok=False)
            return
        ok, message = callback(updates)
        if ok:
            pp_cfg = self.config.setdefault("post_processing", {})
            pp_cfg["mode"] = updates["post_processing.mode"]
            pp_cfg["openai_enabled"] = updates["post_processing.openai_enabled"]
            pp_cfg["openai_model"] = updates["post_processing.openai_model"]
        self._set_settings_status(self.cleanup_status_label, message, ok=ok)

    def _update_cleanup_mode_from_runtime(self, mode: str, label: str):
        """Reflect a cleanup-mode change triggered outside Settings."""
        mode = str(mode or "").strip()
        label = str(label or mode.title()).strip()
        self.config.setdefault("post_processing", {})["mode"] = mode
        if hasattr(self, "post_processing_mode_combo"):
            with QSignalBlocker(self.post_processing_mode_combo):
                self._set_combo_data(self.post_processing_mode_combo, mode)
        if hasattr(self, "cleanup_status_label"):
            self._set_settings_status(self.cleanup_status_label, f"Cleanup mode: {label}", ok=True)
        if self.ptt_overlay:
            self.ptt_overlay.show_message(label.upper(), state="mode")

    def _set_settings_status(self, label: QLabel, message: str, ok: bool = True):
        label.setText(message)
        label.setStyleSheet(self._settings_ok_style if ok else self._settings_error_style)

    def _update_threshold_preview(self, value: int):
        threshold = max(0.0, min(1.0, value / 100.0))
        percent = int((1.0 - threshold) * 100)
        self.voice_threshold_value.setText(f"{threshold:.2f} ({percent}% permissive)")

    def _refresh_voice_controls(self):
        if not self.get_voice_profile_status:
            self.voice_mode_combo.setEnabled(False)
            self.apply_voice_mode_button.setEnabled(False)
            self.voice_threshold_slider.setEnabled(False)
            self.apply_voice_threshold_button.setEnabled(False)
            self.capture_sample_button.setEnabled(False)
            self.clear_profile_button.setEnabled(False)
            self.voice_profile_label.setText("Profile: unavailable")
            self.voice_settings_status_label.setText("Voice settings unavailable.")
            self.voice_settings_status_label.setStyleSheet(self._settings_error_style)
            return

        try:
            status = self.get_voice_profile_status() or {}
        except Exception as exc:
            self.voice_profile_label.setText("Profile: error")
            self._set_settings_status(
                self.voice_settings_status_label,
                f"Could not load voice settings: {exc}",
                ok=False,
            )
            return

        mode = str(status.get("mode", "whisper"))
        mode_index = self.voice_mode_combo.findData(mode)
        if mode_index >= 0:
            with QSignalBlocker(self.voice_mode_combo):
                self.voice_mode_combo.setCurrentIndex(mode_index)

        threshold = float(status.get("threshold", 0.6))
        slider_value = int(max(30, min(95, round(threshold * 100))))
        with QSignalBlocker(self.voice_threshold_slider):
            self.voice_threshold_slider.setValue(slider_value)
        self._update_threshold_preview(slider_value)

        enrolled = int(status.get("enrolled_samples", 0))
        minimum = int(status.get("min_samples", 0))
        ready = bool(status.get("is_enrolled", False))
        profile_path = str(status.get("profile_path", "") or "")
        ready_text = "ready" if ready else "incomplete"
        self.voice_profile_label.setText(
            f"Profile: {enrolled}/{minimum} samples ({ready_text}) • {profile_path}"
        )

    def _apply_voice_mode(self):
        if not self.set_voice_mode:
            self._set_settings_status(
                self.voice_settings_status_label, "Mode update is unavailable.", ok=False
            )
            return
        selected_mode = str(self.voice_mode_combo.currentData() or "").strip()
        ok, message = self.set_voice_mode(selected_mode)
        if ok:
            self.config.setdefault("voice_fingerprint", {})["mode"] = selected_mode
            self._refresh_voice_controls()
        self._set_settings_status(self.voice_settings_status_label, message, ok=ok)

    def _apply_voice_threshold(self):
        if not self.set_voice_threshold:
            self._set_settings_status(
                self.voice_settings_status_label, "Threshold update is unavailable.", ok=False
            )
            return
        threshold = self.voice_threshold_slider.value() / 100.0
        ok, message = self.set_voice_threshold(threshold)
        if ok:
            self.config.setdefault("voice_fingerprint", {})["threshold"] = threshold
            self._refresh_voice_controls()
        self._set_settings_status(self.voice_settings_status_label, message, ok=ok)

    def _capture_voice_sample(self):
        if not self.capture_enrollment_sample:
            self._set_settings_status(
                self.voice_settings_status_label, "Enrollment capture is unavailable.", ok=False
            )
            return
        self.capture_sample_button.setEnabled(False)
        self._set_settings_status(self.voice_settings_status_label, "Recording sample...", ok=True)
        QApplication.processEvents()
        ok, message = self.capture_enrollment_sample(3.0)
        self.capture_sample_button.setEnabled(True)
        self._set_settings_status(self.voice_settings_status_label, message, ok=ok)
        self._refresh_voice_controls()

    def _clear_voice_profile(self):
        if not self.clear_voice_profile:
            self._set_settings_status(
                self.voice_settings_status_label, "Profile reset is unavailable.", ok=False
            )
            return

        confirmed = QMessageBox.question(
            self,
            "Reset Voice Profile",
            "Delete all enrolled voice samples? You will need to re-enroll before using whisper mode.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        ok, message = self.clear_voice_profile()
        self._set_settings_status(self.voice_settings_status_label, message, ok=ok)
        self._refresh_voice_controls()

    def _refresh_dictionary_path(self):
        path = str(self.config.get("transcription", {}).get("personal_dictionary_path", "") or "").strip()
        if self.get_personal_dictionary_path:
            try:
                path = self.get_personal_dictionary_path()
            except Exception:
                pass
        self.dictionary_path_label.setText(f"Path: {path or '(not configured)'}")

    def _initialize_dictionary(self):
        if not self.ensure_personal_dictionary_exists:
            self._set_settings_status(
                self.dictionary_status_label, "Dictionary initialization unavailable.", ok=False
            )
            return
        ok, message = self.ensure_personal_dictionary_exists()
        self._set_settings_status(self.dictionary_status_label, message, ok=ok)
        self._refresh_dictionary_path()
        if ok:
            self._load_dictionary_editor()

    def _open_dictionary(self):
        if not self.open_personal_dictionary:
            self._set_settings_status(
                self.dictionary_status_label, "Dictionary open action unavailable.", ok=False
            )
            return
        ok, message = self.open_personal_dictionary()
        self._set_settings_status(self.dictionary_status_label, message, ok=ok)
        self._refresh_dictionary_path()

    def _reload_dictionary(self):
        if not self.reload_personal_dictionary:
            self._set_settings_status(
                self.dictionary_status_label, "Dictionary reload action unavailable.", ok=False
            )
            return
        ok, message = self.reload_personal_dictionary()
        self._set_settings_status(self.dictionary_status_label, message, ok=ok)
        self._refresh_voice_controls()
        if ok:
            self._load_dictionary_editor()

    def _load_dictionary_editor(self):
        if not hasattr(self, "dictionary_terms_table"):
            return
        if not self.get_personal_dictionary_payload:
            self._dictionary_terms = []
            self._dictionary_corrections = []
            self._populate_dictionary_tables()
            return
        try:
            payload = self.get_personal_dictionary_payload() or {}
            self._dictionary_terms = [str(term) for term in payload.get("preferred_terms", []) if str(term).strip()]
            self._dictionary_corrections = [
                dict(correction)
                for correction in payload.get("corrections", [])
                if isinstance(correction, dict)
            ]
            self._populate_dictionary_tables()
            self._set_settings_status(
                self.dictionary_status_label,
                f"Loaded {len(self._dictionary_terms)} term(s), {len(self._dictionary_corrections)} replacement(s).",
                ok=True,
            )
        except Exception as exc:
            self._set_settings_status(self.dictionary_status_label, f"Could not load dictionary: {exc}", ok=False)

    def _save_dictionary_editor(self):
        if not self.save_personal_dictionary_payload:
            self._set_settings_status(self.dictionary_status_label, "Dictionary save unavailable.", ok=False)
            return

        ok, message = self.save_personal_dictionary_payload(
            self._dictionary_terms,
            self._dictionary_corrections,
        )
        self._set_settings_status(self.dictionary_status_label, message, ok=ok)
        if ok:
            self._load_dictionary_editor()

    def _populate_dictionary_tables(self):
        self.dictionary_terms_table.setRowCount(len(self._dictionary_terms))
        for row_idx, term in enumerate(self._dictionary_terms):
            self.dictionary_terms_table.setItem(row_idx, 0, QTableWidgetItem(str(term)))
            delete_button = QPushButton("Delete")
            delete_button.clicked.connect(lambda _checked=False, row=row_idx: self._delete_dictionary_term(row))
            self.dictionary_terms_table.setCellWidget(row_idx, 1, delete_button)

        self.dictionary_corrections_table.setRowCount(len(self._dictionary_corrections))
        for row_idx, correction in enumerate(self._dictionary_corrections):
            variations = correction.get("variations", []) or []
            if not isinstance(variations, list):
                variations = [variations]
            replace_text = "; ".join(str(item) for item in variations)
            with_text = str(correction.get("phrase", ""))
            match = str(correction.get("match", "substring") or "substring")
            match_label = "Whole word" if match == "whole_word" else "Contains"
            values = [replace_text, with_text, match_label]
            for col_idx, value in enumerate(values):
                self.dictionary_corrections_table.setItem(row_idx, col_idx, QTableWidgetItem(value))
            delete_button = QPushButton("Delete")
            delete_button.clicked.connect(lambda _checked=False, row=row_idx: self._delete_dictionary_correction(row))
            self.dictionary_corrections_table.setCellWidget(row_idx, 3, delete_button)

    def _add_dictionary_term(self):
        term = " ".join(self.dictionary_term_edit.text().strip().split())
        if not term:
            self._set_settings_status(self.dictionary_status_label, "Enter a word or phrase first.", ok=False)
            return
        seen = {existing.lower() for existing in self._dictionary_terms}
        if term.lower() not in seen:
            self._dictionary_terms.append(term)
        self.dictionary_term_edit.clear()
        self._save_dictionary_editor()

    def _delete_dictionary_term(self, row_idx: int):
        if 0 <= row_idx < len(self._dictionary_terms):
            self._dictionary_terms.pop(row_idx)
            self._save_dictionary_editor()

    def _add_dictionary_correction(self):
        wrong = " ".join(self.dictionary_wrong_edit.text().strip().split())
        right = " ".join(self.dictionary_right_edit.text().strip().split())
        if not wrong or not right:
            self._set_settings_status(
                self.dictionary_status_label,
                "Enter both the misheard text and the replacement.",
                ok=False,
            )
            return
        correction = {
            "phrase": right,
            "variations": [wrong],
            "match": str(self.dictionary_match_combo.currentData() or "substring"),
        }
        key = (
            correction["phrase"].lower(),
            tuple(item.lower() for item in correction["variations"]),
            correction["match"],
        )
        existing = {
            (
                str(item.get("phrase", "")).lower(),
                tuple(str(variation).lower() for variation in item.get("variations", []) or []),
                str(item.get("match", "substring") or "substring"),
            )
            for item in self._dictionary_corrections
            if isinstance(item, dict)
        }
        if key not in existing:
            self._dictionary_corrections.append(correction)
        self.dictionary_wrong_edit.clear()
        self.dictionary_right_edit.clear()
        self._save_dictionary_editor()

    def _delete_dictionary_correction(self, row_idx: int):
        if 0 <= row_idx < len(self._dictionary_corrections):
            self._dictionary_corrections.pop(row_idx)
            self._save_dictionary_editor()

    @staticmethod
    def _format_int(value) -> str:
        try:
            return f"{int(round(float(value or 0))):,}"
        except (TypeError, ValueError):
            return "0"

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            child_layout = item.layout()
            if child_layout:
                BloviateUI._clear_layout(child_layout)

    def _refresh_insights(self):
        if not hasattr(self, "insight_wpm_value"):
            return
        if not self.get_history_insights:
            self.insights_status_label.setText("Insights unavailable.")
            return

        try:
            insights = self.get_history_insights() or {}
        except Exception as exc:
            self.insights_status_label.setText(f"Could not load insights: {exc}")
            self.insights_status_label.setStyleSheet(self._settings_error_style)
            return

        total_words = int(insights.get("total_words", 0) or 0)
        total_transcripts = int(insights.get("total_transcripts", 0) or 0)
        wpm = int(insights.get("words_per_minute", 0) or 0)
        changed_outputs = int(insights.get("changed_outputs", 0) or 0)
        changed_words = int(insights.get("changed_words", 0) or 0)
        dictionary_corrections = int(insights.get("dictionary_corrections", 0) or 0)

        if wpm <= 0:
            pace_caption = "No timed clips yet"
        elif wpm < 90:
            pace_caption = "Measured"
        elif wpm < 140:
            pace_caption = "Steady"
        elif wpm < 190:
            pace_caption = "Fast"
        else:
            pace_caption = "Very fast"

        self.insight_wpm_value.setText(self._format_int(wpm))
        self.insight_wpm_gauge.set_data(min(wpm / 220.0, 1.0), pace_caption, "pace")

        self.insight_fixes_value.setText(self._format_int(changed_outputs))
        self.insight_changed_words_label.setText(f"{self._format_int(changed_words)} words rewritten")
        self.insight_dictionary_rules_label.setText(
            f"{self._format_int(dictionary_corrections)} dictionary rules active"
        )

        self.insight_total_words_value.setText(self._format_int(total_words))
        self.insight_total_transcripts_label.setText(f"{self._format_int(total_transcripts)} saved transcripts")

        mode_usage = insights.get("mode_usage", {}) or {}
        if mode_usage:
            dominant_mode, dominant_words = max(mode_usage.items(), key=lambda item: item[1])
            percent = int(round((float(dominant_words) / max(1, total_words)) * 100))
            self.insight_mode_bar.setValue(percent)
            self.insight_mode_bar.setFormat(f"{percent}% {str(dominant_mode).replace('_', ' ')}")
        else:
            self.insight_mode_bar.setValue(0)
            self.insight_mode_bar.setFormat("No dictation yet")

        apps = list(insights.get("app_usage", []) or [])
        apps_total_words = sum(int(app.get("words", 0) or 0) for app in apps) or 1
        self.insight_apps_summary_label.setText(
            f"Apps used | {self._format_int(insights.get('apps_used', 0))}"
        )
        self._clear_layout(self.insight_app_rows_layout)
        if apps:
            for app in apps[:6]:
                words = int(app.get("words", 0) or 0)
                percent = int(round((words / apps_total_words) * 100))
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(10)
                name_label = QLabel(str(app.get("name", "Unknown app"))[:28])
                name_label.setMinimumWidth(132)
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(percent)
                bar.setFormat(f"{percent}%")
                count_label = QLabel(f"{self._format_int(words)} words")
                count_label.setMinimumWidth(92)
                row_layout.addWidget(name_label)
                row_layout.addWidget(bar, 1)
                row_layout.addWidget(count_label)
                self.insight_app_rows_layout.addWidget(row)
        else:
            empty = QLabel("New dictations will appear here automatically.")
            empty.setStyleSheet(self._settings_status_default_style)
            self.insight_app_rows_layout.addWidget(empty)

        current_streak = int(insights.get("current_streak_days", 0) or 0)
        longest_streak = int(insights.get("longest_streak_days", 0) or 0)
        self.insight_current_streak_label.setText(
            f"{current_streak} day streak" if current_streak != 1 else "1 day streak"
        )
        self.insight_longest_streak_label.setText(f"Longest | {longest_streak} days")
        self.insight_streak_heatmap.set_days(insights.get("daily_usage", []) or [])

        self.insights_status_label.setText(f"{self._format_int(total_words)} words indexed")
        self.insights_status_label.setStyleSheet(self._settings_status_default_style)

    def _queue_achievement_refresh(self):
        if hasattr(self, "_achievement_refresh_timer"):
            self._achievement_refresh_timer.start(180)
        else:
            self._refresh_achievements()

    def _refresh_achievements(self):
        if not hasattr(self, "achievement_table"):
            return
        if not self.get_achievement_summary:
            self.achievement_table.setRowCount(0)
            self._set_settings_status(self.achievement_status_label, "Achievements unavailable.", ok=False)
            return

        query = self.achievement_search_edit.text().strip()
        status_filter = str(self.achievement_filter_combo.currentData() or "all")
        try:
            summary = self.get_achievement_summary(query, status_filter) or {}
        except Exception as exc:
            self._set_settings_status(self.achievement_status_label, f"Could not load achievements: {exc}", ok=False)
            return

        total = int(summary.get("total", 0) or 0)
        unlocked = int(summary.get("unlocked", 0) or 0)
        percent = int(round((unlocked / max(1, total)) * 100))
        self.achievement_summary_label.setText(f"{self._format_int(unlocked)} / {self._format_int(total)} unlocked")
        self.achievement_progress_bar.setValue(percent)
        self.achievement_progress_bar.setFormat(f"{percent}%")

        with QSignalBlocker(self.achievement_ai_checkbox):
            self.achievement_ai_checkbox.setChecked(bool(summary.get("ai_analysis_enabled", False)))

        recent = summary.get("recent", []) or []
        if recent:
            labels = [str(item.get("title", "Achievement")) for item in recent[:4]]
            self.achievement_recent_label.setText("Recent: " + " | ".join(labels))
        else:
            self.achievement_recent_label.setText("Recent unlocks will appear here.")

        achievements = list(summary.get("achievements", []) or [])
        self._achievement_rows = achievements
        self.achievement_table.setUpdatesEnabled(False)
        try:
            self.achievement_table.setRowCount(0)
            self.achievement_table.setRowCount(len(achievements))
            for row_idx, item in enumerate(achievements):
                icon_item = QTableWidgetItem()
                badge_path = str(item.get("badge_path", "") or "")
                if badge_path:
                    icon_item.setIcon(QIcon(badge_path))
                icon_item.setData(Qt.ItemDataRole.UserRole, item)
                self.achievement_table.setItem(row_idx, 0, icon_item)

                title = str(item.get("title", "Achievement"))
                if item.get("ai_required"):
                    title += " [AI]"
                title_item = QTableWidgetItem(title)
                title_item.setData(Qt.ItemDataRole.UserRole, item)
                title_item.setToolTip(str(item.get("description", "")))
                self.achievement_table.setItem(row_idx, 1, title_item)
                self.achievement_table.setItem(row_idx, 2, QTableWidgetItem(str(item.get("category", ""))))
                self.achievement_table.setItem(row_idx, 3, QTableWidgetItem(str(item.get("progress_label", ""))))
                status = "Unlocked" if item.get("unlocked") else "Locked"
                if item.get("hidden"):
                    status = "Secret"
                self.achievement_table.setItem(row_idx, 4, QTableWidgetItem(status))
                self.achievement_table.setRowHeight(row_idx, 54)
        finally:
            self.achievement_table.setUpdatesEnabled(True)

        self._set_settings_status(
            self.achievement_status_label,
            f"Showing {len(achievements)} achievement(s).",
            ok=True,
        )
        self._show_selected_achievement_detail()

    def _toggle_achievement_ai(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.config.setdefault("achievements", {})["ai_analysis_enabled"] = enabled
        if not self.set_achievement_settings:
            self._set_settings_status(self.achievement_status_label, "Achievement settings are local-only this session.", ok=False)
            return
        ok, message = self.set_achievement_settings({"ai_analysis_enabled": enabled})
        self._set_settings_status(self.achievement_status_label, message, ok=ok)

    def _analyze_achievements(self):
        if not self.analyze_achievement_history:
            self._set_settings_status(self.achievement_status_label, "AI analysis unavailable.", ok=False)
            return
        self._set_settings_status(self.achievement_status_label, "Analyzing transcript history...", ok=True)
        QApplication.processEvents()
        ok, message = self.analyze_achievement_history()
        self._set_settings_status(self.achievement_status_label, message, ok=ok)
        self._refresh_achievements()

    def _reset_achievements(self):
        if not self.reset_achievements:
            self._set_settings_status(self.achievement_status_label, "Achievement reset unavailable.", ok=False)
            return
        confirmed = QMessageBox.question(
            self,
            "Reset Achievements",
            "Delete all local achievement unlocks, progress, and AI achievement tags?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        ok, message = self.reset_achievements()
        self._set_settings_status(self.achievement_status_label, message, ok=ok)
        self._refresh_achievements()

    def _show_selected_achievement_detail(self):
        if not hasattr(self, "achievement_detail"):
            return
        rows = self.achievement_table.selectionModel().selectedRows() if self.achievement_table.selectionModel() else []
        item = None
        if rows:
            item = self.achievement_table.item(rows[0].row(), 1)
        if item is None and self._achievement_rows:
            item_data = self._achievement_rows[0]
        elif item is not None:
            item_data = item.data(Qt.ItemDataRole.UserRole) or {}
        else:
            self.achievement_detail.setPlainText("No achievements match this filter.")
            return
        lines = [
            str(item_data.get("title", "Achievement")),
            str(item_data.get("description", "")),
            "",
            f"Category: {item_data.get('category', '')}",
            f"Rarity: {item_data.get('rarity', '')}",
            f"Progress: {item_data.get('progress_label', '')}",
            f"Status: {'Unlocked' if item_data.get('unlocked') else 'Locked'}",
        ]
        if item_data.get("ai_required"):
            lines.append("AI-assisted: requires opt-in transcript analysis.")
        self.achievement_detail.setPlainText("\n".join(lines))

    def _refresh_history(self):
        if not hasattr(self, "history_table"):
            return
        self.history_table.setRowCount(0)
        if not self.get_history_records:
            self._set_settings_status(self.history_status_label, "History unavailable.", ok=False)
            return
        query = self.history_search_edit.text().strip()
        try:
            max_records = int(self.config.get("history", {}).get("max_ui_records", 100))
            records = self.get_history_records(query, max_records) or []
        except Exception as exc:
            self._set_settings_status(self.history_status_label, f"Could not load history: {exc}", ok=False)
            return
        self.history_table.setRowCount(len(records))
        for row_idx, record in enumerate(records):
            target = record.get("target_app", "")
            if record.get("target_window"):
                target = f"{target} - {record.get('target_window')}" if target else record.get("target_window")
            values = [
                record.get("created_at", ""),
                record.get("mode", ""),
                record.get("provider", ""),
                target,
                record.get("text", ""),
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, record.get("id"))
                if col_idx == 4:
                    item.setToolTip(
                        "Original: " + str(record.get("original_text", ""))
                        if record.get("original_text") != record.get("text")
                        else str(record.get("text", ""))
                    )
                self.history_table.setItem(row_idx, col_idx, item)
        if records:
            self._set_settings_status(self.history_status_label, f"Loaded {len(records)} history item(s).", ok=True)
        else:
            self._set_settings_status(
                self.history_status_label,
                "No saved transcripts yet. New dictations are recorded automatically.",
                ok=True,
            )
        self._refresh_insights()

    def _selected_history_id(self):
        rows = self.history_table.selectionModel().selectedRows() if self.history_table.selectionModel() else []
        if not rows:
            return None
        item = self.history_table.item(rows[0].row(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _copy_history_selection(self):
        rows = self.history_table.selectionModel().selectedRows() if self.history_table.selectionModel() else []
        if not rows:
            self._set_settings_status(self.history_status_label, "Select a history row first.", ok=False)
            return
        text_item = self.history_table.item(rows[0].row(), 4)
        if text_item:
            QApplication.clipboard().setText(text_item.text())
            self._set_settings_status(self.history_status_label, "Copied history text.", ok=True)

    def _delete_history_selection(self):
        record_id = self._selected_history_id()
        if record_id is None:
            self._set_settings_status(self.history_status_label, "Select a history row first.", ok=False)
            return
        if not self.delete_history_record:
            self._set_settings_status(self.history_status_label, "History delete unavailable.", ok=False)
            return
        ok, message = self.delete_history_record(int(record_id))
        self._set_settings_status(self.history_status_label, message, ok=ok)
        self._refresh_history()

    def _clear_history(self):
        if not self.clear_history:
            self._set_settings_status(self.history_status_label, "History clear unavailable.", ok=False)
            return
        confirmed = QMessageBox.question(
            self,
            "Clear History",
            "Delete all local transcript history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        ok, message = self.clear_history()
        self._set_settings_status(self.history_status_label, message, ok=ok)
        self._refresh_history()

    def _export_history(self):
        if not self.export_history:
            self._set_settings_status(self.history_status_label, "History export unavailable.", ok=False)
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export History",
            "bloviate-history.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        ok, message = self.export_history(path)
        self._set_settings_status(self.history_status_label, message, ok=ok)

    def _paths_text(self) -> str:
        config_path = str(self.config.get("__config_path__", ""))
        config_dir = str(self.config.get("__config_dir__", ""))
        dictionary = str(self.config.get("transcription", {}).get("personal_dictionary_path", "personal_dictionary.yaml"))
        return f"Config: {config_path}\nData directory: {config_dir}\nDictionary: {dictionary}"

    def _run_doctor(self):
        if not self.run_doctor_text:
            self.doctor_output.setPlainText("Doctor unavailable.")
            return
        ok, text = self.run_doctor_text()
        self.doctor_output.setPlainText(text)
        colors = self._theme["colors"]
        if ok:
            self.doctor_output.setStyleSheet(f"color: {colors['text']};")
        else:
            self.doctor_output.setStyleSheet(f"color: {colors['danger']};")

    def _reset_defaults(self):
        if not self.reset_settings_to_defaults:
            self.doctor_output.setPlainText("Reset unavailable.")
            return
        confirmed = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset Bloviate settings to packaged defaults? API keys and history are not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return
        ok, message = self.reset_settings_to_defaults()
        self.doctor_output.setPlainText(message)
        colors = self._theme["colors"]
        self.doctor_output.setStyleSheet(f"color: {colors['text' if ok else 'danger']};")

    def _toggle_show_window_on_startup(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.config.setdefault("ui", {})["show_main_window"] = enabled
        if not self.set_show_main_window_on_startup:
            self._set_settings_status(
                self.startup_status_label,
                "Startup window preference is local-only this session.",
                ok=False,
            )
            return
        ok, message = self.set_show_main_window_on_startup(enabled)
        self._set_settings_status(self.startup_status_label, message, ok=ok)

    def _toggle_splash_on_startup(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.config.setdefault("ui", {}).setdefault("startup_splash", {})["enabled"] = enabled
        if not self.set_startup_splash_enabled:
            self._set_settings_status(
                self.startup_status_label,
                "Splash preference is local-only this session.",
                ok=False,
            )
            return
        ok, message = self.set_startup_splash_enabled(enabled)
        self._set_settings_status(self.startup_status_label, message, ok=ok)

    def _toggle_terminal_animation_on_startup(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.config.setdefault("app", {})["startup_animation"] = enabled
        if not self.set_terminal_startup_animation_enabled:
            self._set_settings_status(
                self.startup_status_label,
                "Terminal animation preference is local-only this session.",
                ok=False,
            )
            return
        ok, message = self.set_terminal_startup_animation_enabled(enabled)
        self._set_settings_status(self.startup_status_label, message, ok=ok)

    def show_settings_tab(self):
        """Bring the settings tab into focus."""
        self.show()
        self.raise_()
        self.activateWindow()
        self._select_main_page(self.settings_tab)
        self._refresh_voice_controls()
        self._refresh_dictionary_path()
        self._load_dictionary_editor()
        self._refresh_history()
        self._refresh_achievements()

    def show_status_tab(self):
        """Bring the status tab into focus."""
        self.show()
        self.raise_()
        self.activateWindow()
        self._select_main_page(self.status_tab)

    def _select_main_page(self, page: QWidget):
        if hasattr(self, "tabs"):
            self.tabs.setCurrentWidget(page)
        if hasattr(self, "status_nav_button"):
            self.status_nav_button.setChecked(page is self.status_tab)
        if hasattr(self, "settings_nav_button"):
            self.settings_nav_button.setChecked(page is self.settings_tab)

    def request_quit(self):
        """Mark the window as closing and quit the app."""
        self._closing = True
        QApplication.instance().quit()

    def _status_pill_style(self, background: str, color: str) -> str:
        return (
            "font-size: 15px; font-weight: 700; padding: 10px 14px; "
            f"background-color: {background}; color: {color}; "
            f"border: 1px solid {self._theme['colors']['border']}; border-radius: 8px;"
        )

    def set_light_theme(self):
        """Apply the configured app theme."""
        self.apply_theme(self.config.get("ui", {}).get("theme", "light"))

    def set_dark_theme(self):
        """Compatibility wrapper for older callers."""
        self.apply_theme("graphite")

    def apply_theme(self, theme_id: str | None = None):
        """Apply a named theme to the dashboard and settings chrome."""
        self._theme_id = normalize_theme_id(theme_id or self._theme_id)
        self._theme = get_theme(self._theme_id)
        colors = self._theme["colors"]

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["surface_alt"]))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(colors["danger"]))
        palette.setColor(QPalette.ColorRole.Link, QColor(colors["primary"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["selection"]))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors["primary_text"]))
        self.setPalette(palette)
        if QApplication.instance():
            QApplication.instance().setPalette(palette)

        self._settings_status_default_style = f"font-size: 12px; color: {colors['muted']};"
        self._settings_ok_style = f"font-size: 12px; color: {colors['success']}; font-weight: 600;"
        self._settings_error_style = f"font-size: 12px; color: {colors['danger']}; font-weight: 600;"
        self._transcription_style_final = (
            f"padding: 14px; background-color: {colors['surface']}; color: {colors['text']}; "
            f"border: 1px solid {colors['border']}; border-radius: 8px; min-height: 52px;"
        )
        self._transcription_style_interim = (
            f"padding: 14px; background-color: {colors['surface_alt']}; color: {colors['warning']}; "
            f"border: 1px solid {colors['warning']}; border-radius: 8px; min-height: 52px; "
            "font-style: italic;"
        )
        self.setStyleSheet(self._style_sheet_for_theme(colors))
        self._apply_dynamic_theme_styles()

    def _style_sheet_for_theme(self, colors: dict) -> str:
        return f"""
            QMainWindow, QWidget#AppRoot, QWidget#SettingsContent, QStackedWidget {{
                background: {colors['window']};
                color: {colors['text']};
            }}
            QWidget#TopNav {{
                background: {colors['window']};
            }}
            QLabel#AppTitle {{
                color: {colors['text_soft']};
                font-size: 17px;
                font-weight: 800;
                padding: 10px 0;
            }}
            QPushButton#NavButton {{
                background: {colors['nav']};
                color: {colors['text_soft']};
                min-width: 140px;
                padding: 13px 24px;
                border: 1px solid {colors['border']};
                border-radius: 8px;
                font-size: 15px;
                font-weight: 700;
            }}
            QPushButton#NavButton:checked {{
                background: {colors['primary']};
                color: {colors['primary_text']};
                border-color: {colors['primary']};
            }}
            QPushButton#NavButton:hover {{
                background: {colors['nav_hover']};
            }}
            QPushButton#NavButton:checked:hover {{
                background: {colors['primary_hover']};
            }}
            QPushButton#PrimaryActionButton {{
                background: {colors['primary']};
                color: {colors['primary_text']};
                border: 1px solid {colors['primary']};
                border-radius: 8px;
                padding: 10px 18px;
                font-weight: 800;
            }}
            QPushButton#PrimaryActionButton:hover {{
                background: {colors['primary_hover']};
            }}
            QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {{
                background: {colors['window']};
                border: 0;
            }}
            QGroupBox {{
                background: {colors['card']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                margin-top: 22px;
                padding: 18px 18px 16px 18px;
                font-weight: 700;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                color: {colors['text']};
                background: {colors['window']};
            }}
            QLabel {{
                color: {colors['text']};
            }}
            QFrame#InsightCard {{
                background: {colors['card_alt']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
            }}
            QLabel#InsightSectionTitle {{
                font-size: 18px;
                font-weight: 800;
                color: {colors['text']};
            }}
            QLabel#InsightTitle {{
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 0;
                color: {colors['text_soft']};
                text-transform: uppercase;
            }}
            QLabel#InsightSubtitle {{
                font-size: 11px;
                font-weight: 700;
                color: {colors['muted']};
            }}
            QLabel#InsightNumber {{
                font-size: 28px;
                font-weight: 800;
                color: {colors['text']};
            }}
            QLabel#InsightNumberSmall {{
                font-size: 22px;
                font-weight: 800;
                color: {colors['text']};
            }}
            QLineEdit, QTextEdit, QComboBox, QTableWidget {{
                background: {colors['surface']};
                color: {colors['text']};
                border: 1px solid {colors['border_strong']};
                border-radius: 6px;
                padding: 7px 9px;
                selection-background-color: {colors['selection']};
                selection-color: {colors['primary_text']};
            }}
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
                border: 1px solid {colors['primary']};
            }}
            QLineEdit::placeholder {{
                color: {colors['placeholder']};
            }}
            QComboBox::drop-down {{
                border: 0;
                width: 26px;
            }}
            QPushButton {{
                background-color: {colors['surface']};
                color: {colors['text']};
                border: 1px solid {colors['border_strong']};
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {colors['surface_alt']};
            }}
            QPushButton:pressed {{
                background-color: {colors['nav']};
            }}
            QPushButton:disabled {{
                background-color: {colors['surface_alt']};
                color: {colors['muted']};
                border-color: {colors['border']};
            }}
            QCheckBox {{
                color: {colors['text']};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid {colors['border_strong']};
                background: {colors['surface']};
            }}
            QCheckBox::indicator:checked {{
                background: {colors['primary']};
                border-color: {colors['primary']};
            }}
            QTableWidget {{
                gridline-color: {colors['border']};
                alternate-background-color: {colors['card_alt']};
            }}
            QHeaderView::section {{
                background: {colors['nav']};
                color: {colors['text']};
                border: 0;
                border-right: 1px solid {colors['border']};
                padding: 7px 9px;
                font-weight: 700;
            }}
            QProgressBar {{
                background: {colors['surface']};
                border: 1px solid {colors['border_strong']};
                border-radius: 6px;
                color: {colors['text']};
                text-align: center;
                min-height: 24px;
            }}
            QProgressBar::chunk {{
                background-color: {colors['primary']};
                border-radius: 5px;
            }}
            QSlider::groove:horizontal {{
                background: {colors['surface_alt']};
                height: 7px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {colors['surface']};
                border: 1px solid {colors['border_strong']};
                width: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }}
            QSlider::sub-page:horizontal {{
                background: {colors['primary']};
                border-radius: 3px;
            }}
            QScrollBar:vertical {{
                background: {colors['surface_alt']};
                width: 12px;
                margin: 0;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {colors['border_strong']};
                min-height: 36px;
                border-radius: 6px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QFrame#AchievementOverlay {{
                background-color: {colors['overlay']};
            }}
            QFrame#AchievementOverlayCard {{
                background: {colors['surface']};
                border: 1px solid {colors['border']};
                border-radius: 16px;
                min-width: 440px;
                max-width: 560px;
            }}
            QLabel#AchievementOverlayTitle {{
                color: {colors['text']};
                font-size: 24px;
                font-weight: 800;
            }}
            QLabel#AchievementOverlayDescription {{
                color: {colors['text_soft']};
                font-size: 14px;
                font-weight: 600;
            }}
            QLabel#AchievementOverlayProgress {{
                color: {colors['primary']};
                font-size: 13px;
                font-weight: 800;
            }}
            QFrame#CowRunwayOverlay {{
                background-color: {colors['overlay']};
            }}
            QLabel#CowRunwayText {{
                color: {colors['primary']};
                background: {colors['surface']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
                padding: 12px;
            }}
        """

    def _apply_dynamic_theme_styles(self):
        colors = self._theme["colors"]
        if hasattr(self, "settings_tab") and self.settings_tab.layout():
            for scroll in self.settings_tab.findChildren(QScrollArea):
                scroll.viewport().setStyleSheet(f"background-color: {colors['window']};")
        if hasattr(self, "status_label"):
            self.status_label.setStyleSheet(f"font-style: italic; color: {colors['muted']}; padding: 8px;")
        if hasattr(self, "transcription_label"):
            self.transcription_label.setStyleSheet(self._transcription_style_final)
        if hasattr(self, "insight_wpm_gauge"):
            self.insight_wpm_gauge.set_theme_colors(colors)
        if hasattr(self, "insight_streak_heatmap"):
            self.insight_streak_heatmap.set_theme_colors(colors)
        if hasattr(self, "ptt_label") and "Inactive" in self.ptt_label.text():
            self.ptt_label.setStyleSheet(self._status_pill_style(colors["nav"], colors["text"]))
        if hasattr(self, "command_label") and "Inactive" in self.command_label.text():
            self.command_label.setStyleSheet(self._status_pill_style(colors["nav"], colors["text"]))
        for attr in (
            "permissions_status_label", "insights_status_label", "achievement_status_label",
            "achievement_recent_label", "device_status_label", "hotkey_status_label",
            "voice_profile_label", "voice_settings_status_label", "dictation_status_label",
            "openai_key_source_label", "deepgram_key_source_label", "model_status_label",
            "cleanup_status_label", "dictionary_path_label", "dictionary_status_label",
            "history_status_label", "startup_status_label", "paths_label",
            "appearance_status_label",
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setStyleSheet(self._settings_status_default_style)

    def _update_audio_level(self, level: float):
        """Update the audio level bar."""
        normalized_level = max(0.0, min(level / 0.26, 1.0))
        if (
            hasattr(self, "tabs")
            and hasattr(self, "status_tab")
            and self.tabs.currentWidget() is self.status_tab
            and self.isVisible()
        ):
            self.audio_bar.set_audio_level(normalized_level)

        # Update menu bar indicator
        if self.menu_bar_indicator:
            self.menu_bar_indicator.set_audio_level(normalized_level)

        if self.ptt_overlay:
            self.ptt_overlay.set_audio_level(normalized_level)

    def _update_ptt_status(self, is_active: bool):
        """Update PTT status indicator."""
        colors = self._theme["colors"]
        if is_active:
            self.ptt_label.setText("PTT: ACTIVE")
            self.ptt_label.setStyleSheet(self._status_pill_style(colors["success"], colors["primary_text"]))
            if hasattr(self, "toggle_dictation_button"):
                self.toggle_dictation_button.setText("Stop Dictation")
            # Update menu bar indicator
            if self.menu_bar_indicator:
                self.menu_bar_indicator.set_recording()
            if self.ptt_overlay:
                self.ptt_overlay.set_recording()
        else:
            self.ptt_label.setText("PTT: Inactive")
            self.ptt_label.setStyleSheet(self._status_pill_style(colors["nav"], colors["text"]))
            if hasattr(self, "toggle_dictation_button"):
                self.toggle_dictation_button.setText("Start Dictation")

    def _update_command_status(self, message: str, state: str):
        """Update command mode indicator."""
        colors = self._theme["colors"]
        styles = {
            "inactive": self._status_pill_style(colors["nav"], colors["text"]),
            "listening": self._status_pill_style(colors["command"], colors["primary_text"]),
            "processing": self._status_pill_style(colors["warning"], colors["text"]),
            "recognized": self._status_pill_style(colors["success"], colors["primary_text"]),
            "unrecognized": self._status_pill_style(colors["danger"], colors["primary_text"]),
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

        if self.ptt_overlay:
            if state == "listening":
                self.ptt_overlay.set_command_recording()
            elif state == "processing":
                self.ptt_overlay.set_command_processing()
            elif state == "recognized":
                self.ptt_overlay.set_command_success()
            elif state == "unrecognized":
                self.ptt_overlay.set_command_unknown()
            elif state == "inactive":
                self.ptt_overlay.set_idle()

    def _update_voice_match(self, is_match: bool, score: float):
        """Update voice match status."""
        colors = self._theme["colors"]
        if score < 0:
            self.match_status_label.setText("Talk mode")
            self.match_status_label.setStyleSheet(f"color: {colors['muted']}; font-weight: bold;")
            self.match_score_label.setText("")
            return

        if is_match:
            self.match_status_label.setText("Matched")
            self.match_status_label.setStyleSheet(f"color: {colors['success']}; font-weight: bold;")
            if self.menu_bar_indicator:
                self.menu_bar_indicator.set_accepted()
            if self.ptt_overlay:
                self.ptt_overlay.set_accepted()
        else:
            self.match_status_label.setText("Rejected")
            self.match_status_label.setStyleSheet(f"color: {colors['danger']}; font-weight: bold;")
            # Update menu bar indicator
            if self.menu_bar_indicator:
                self.menu_bar_indicator.set_rejected()
            if self.ptt_overlay:
                self.ptt_overlay.set_rejected()

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

        if self.ptt_overlay:
            if message == "Processing..." or message == "Transcribing...":
                self.ptt_overlay.set_processing()
            elif message == "Ready":
                self.ptt_overlay.set_idle()
            elif message in ["Voice rejected", "No audio recorded", "No speech detected"]:
                self.ptt_overlay.set_rejected()

    def _update_transcription(self, text: str):
        """Update the last transcription display."""
        self.transcription_label.setText(f"Last: {text}")
        self.transcription_label.setStyleSheet(self._transcription_style_final)
        self._last_final_text = text
        if hasattr(self, "history_table"):
            QTimer.singleShot(0, self._refresh_history)
        # Show success in menu bar
        if self.menu_bar_indicator:
            self.menu_bar_indicator.set_accepted()
        if self.ptt_overlay:
            self.ptt_overlay.set_accepted()

    def _update_rejected_transcription(self, text: str):
        """Show a rejected transcript without marking the output as accepted."""
        self.transcription_label.setText(f"Rejected (history only): {text}")
        self.transcription_label.setStyleSheet(
            "padding: 14px; background-color: #FFF9E9; color: #7A4D1C; "
            "border: 1px solid #E3C88A; border-radius: 8px; min-height: 52px;"
        )
        self._last_final_text = f"Rejected (history only): {text}"
        if hasattr(self, "history_table"):
            QTimer.singleShot(0, self._refresh_history)
        if self.menu_bar_indicator:
            self.menu_bar_indicator.set_rejected()
        if self.ptt_overlay:
            self.ptt_overlay.set_rejected()

    def _show_achievement_unlocks(self, unlocks):
        """Show batched achievement unlocks and refresh the Settings grid."""
        if hasattr(self, "achievement_overlay"):
            self.achievement_overlay.show_unlocks(unlocks)
        if (
            unlocks
            and self.ptt_overlay
            and bool(self._easter_config().get("milestone_toasts", True))
        ):
            self.ptt_overlay.show_message(self._milestone_toast(unlocks), state="mode", hold_ms=2400)
            self._increment_easter_counter("milestone_toasts_shown")
        if hasattr(self, "achievement_table"):
            QTimer.singleShot(0, self._refresh_achievements)

    def _milestone_toast(self, unlocks) -> str:
        titles = [str(item.get("title", "")) for item in list(unlocks or []) if isinstance(item, dict)]
        joined = " ".join(titles).lower()
        if "word" in joined or "keyboard" in joined:
            return "KEYBOARD ALARMED"
        if "dictionary" in joined:
            return "DICTIONARY LAWYER"
        if "command" in joined or "window" in joined:
            return "REMOTE CONTROL"
        if "achievement" in joined:
            return "TROPHY SHELF"
        return "ACHIEVEMENT"

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
        if not self._closing and self.menu_bar_indicator and not self.menu_bar_indicator._closed:
            self.hide()
            self.status_label.setText("Running in menu bar. Use tray icon to reopen settings.")
            event.ignore()
            return
        if self._closing:
            super().closeEvent(event)
            return
        self._closing = True
        # Clean up menu bar indicator
        if self.menu_bar_indicator:
            self.menu_bar_indicator.close()
        if self.ptt_overlay:
            self.ptt_overlay.close()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "achievement_overlay") and self.achievement_overlay.isVisible():
            self.achievement_overlay.setGeometry(self.rect())
        if hasattr(self, "cow_runway_overlay") and self.cow_runway_overlay.isVisible():
            self.cow_runway_overlay.setGeometry(self.rect())


class StartupSplash(QWidget):
    """Simple native startup splash shown while Bloviate boots."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        splash_cfg = config.get("ui", {}).get("startup_splash", {})
        self.duration_ms = int(splash_cfg.get("duration_ms", 1400))
        self._fade_ms = int(splash_cfg.get("fade_out_ms", 220))
        self._animation = None
        self._closed = False
        self._objc = None
        self._cow_phase = 0
        self._show_cows = bool(splash_cfg.get("show_cows", True))
        self._cow_timer = QTimer(self)
        self._cow_timer.setInterval(90)
        self._cow_timer.timeout.connect(self._advance_cows)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.resize(540, 250)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)

        card = QFrame()
        card.setObjectName("splashCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(8)

        title = QLabel("BLOVIATE")
        title.setObjectName("splashTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Voice dictation, tuned for humans.")
        subtitle.setObjectName("splashSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        status = QLabel("Starting audio, voice profile, and hotkeys...")
        status.setObjectName("splashStatus")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.cow_label = QLabel(self._cow_runway_text())
        self.cow_label.setObjectName("splashCows")
        self.cow_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.cow_label.setMinimumHeight(54)
        self.cow_label.setFont(QFont("Menlo", 11))

        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        if self._show_cows:
            card_layout.addSpacing(4)
            card_layout.addWidget(self.cow_label)
        card_layout.addSpacing(6)
        card_layout.addWidget(status)
        root.addWidget(card)

        self.setStyleSheet(
            """
            QFrame#splashCard {
                background-color: rgba(255, 253, 247, 246);
                border: 1px solid rgba(216, 204, 186, 210);
                border-radius: 16px;
            }
            QLabel#splashTitle {
                color: #26211D;
                font-size: 28px;
                font-weight: 800;
            }
            QLabel#splashSubtitle {
                color: #6F665E;
                font-size: 13px;
                font-weight: 500;
            }
            QLabel#splashCows {
                color: #2D6B6B;
                font-size: 11px;
                font-weight: 700;
                background: #F7F3EA;
                border: 1px solid #E4D9C8;
                border-radius: 8px;
                padding: 6px;
            }
            QLabel#splashStatus {
                color: #2F7D4F;
                font-size: 12px;
                font-weight: 500;
            }
            """
        )

    def _cow_runway_text(self) -> str:
        width = 52
        phase = self._cow_phase % width
        lines = [" " * width, " " * width, " " * width]
        cow = ["(__)", "(oo)", "/--\\"]
        for offset in (0, 14, 29, 43):
            x = (phase + offset) % width
            for row, part in enumerate(cow):
                line = lines[row]
                if x + len(part) <= width:
                    lines[row] = line[:x] + part + line[x + len(part):]
                else:
                    first = width - x
                    lines[row] = part[first:] + line[len(part) - first:x] + part[:first]
        return "\n".join(lines)

    def _advance_cows(self):
        self._cow_phase = (self._cow_phase + 2) % 52
        if self._show_cows and hasattr(self, "cow_label"):
            self.cow_label.setText(self._cow_runway_text())

    def _get_objc(self):
        """Lazily load Objective-C runtime handles (macOS only)."""
        if self._objc is not None:
            return self._objc
        if sys.platform != 'darwin':
            return None
        try:
            import ctypes
            import ctypes.util
            lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
            lib.sel_registerName.restype = ctypes.c_void_p
            lib.sel_registerName.argtypes = [ctypes.c_char_p]
            self._objc = (lib, ctypes)
            return self._objc
        except Exception:
            return None

    def _get_ns_window(self):
        pair = self._get_objc()
        if pair is None:
            return None
        lib, ctypes = pair
        msg = lib.objc_msgSend
        msg.restype = ctypes.c_void_p
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        return msg(int(self.winId()), lib.sel_registerName(b'window')) or None

    def _force_front_macos(self):
        """Force splash to front on macOS even when launched from Terminal."""
        pair = self._get_objc()
        if pair is None:
            return
        lib, ctypes = pair
        ns_window = self._get_ns_window()
        if not ns_window:
            return
        try:
            msg = lib.objc_msgSend
            sel = lib.sel_registerName
            msg.restype = None
            msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
            msg(ns_window, sel(b'setLevel:'), 25)  # NSStatusWindowLevel
            msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            msg(ns_window, sel(b'orderFrontRegardless'))
        except Exception:
            return

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        x = geo.x() + int((geo.width() - self.width()) / 2)
        y = geo.y() + int((geo.height() - self.height()) / 2)
        self.move(x, y)

    def start(self):
        self._center_on_screen()
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()
        self._force_front_macos()
        if self._show_cows:
            self._cow_timer.start()
        QTimer.singleShot(max(100, self.duration_ms), self.fade_out)

    def fade_out(self):
        if self._closed:
            return
        self._animation = QPropertyAnimation(self, b"windowOpacity")
        self._animation.setDuration(max(100, self._fade_ms))
        self._animation.setStartValue(1.0)
        self._animation.setEndValue(0.0)
        self._animation.finished.connect(self.close)
        self._animation.start()

    def closeEvent(self, event):
        self._closed = True
        self._cow_timer.stop()
        super().closeEvent(event)


def create_ui(
    config: dict,
    get_audio_inputs=None,
    set_audio_input=None,
    get_voice_profile_status=None,
    set_voice_mode=None,
    set_voice_threshold=None,
    capture_enrollment_sample=None,
    clear_voice_profile=None,
    get_personal_dictionary_path=None,
    ensure_personal_dictionary_exists=None,
    open_personal_dictionary=None,
    reload_personal_dictionary=None,
    get_personal_dictionary_payload=None,
    save_personal_dictionary_payload=None,
    get_model_options=None,
    get_secret_statuses=None,
    set_api_key=None,
    set_transcription_settings=None,
    set_hotkey_settings=None,
    set_general_settings=None,
    toggle_dictation=None,
    get_history_records=None,
    get_history_insights=None,
    delete_history_record=None,
    clear_history=None,
    export_history=None,
    get_achievement_summary=None,
    reset_achievements=None,
    set_achievement_settings=None,
    analyze_achievement_history=None,
    run_doctor_text=None,
    reset_settings_to_defaults=None,
    get_permission_statuses=None,
    request_permission=None,
    open_permission_settings=None,
    set_show_main_window_on_startup=None,
    set_startup_splash_enabled=None,
    set_terminal_startup_animation_enabled=None,
) -> tuple[QApplication, BloviateUI]:
    """
    Create and return the UI application and window.

    Returns:
        Tuple of (app, window)
    """
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = BloviateUI(
        config,
        get_audio_inputs=get_audio_inputs,
        set_audio_input=set_audio_input,
        get_voice_profile_status=get_voice_profile_status,
        set_voice_mode=set_voice_mode,
        set_voice_threshold=set_voice_threshold,
        capture_enrollment_sample=capture_enrollment_sample,
        clear_voice_profile=clear_voice_profile,
        get_personal_dictionary_path=get_personal_dictionary_path,
        ensure_personal_dictionary_exists=ensure_personal_dictionary_exists,
        open_personal_dictionary=open_personal_dictionary,
        reload_personal_dictionary=reload_personal_dictionary,
        get_personal_dictionary_payload=get_personal_dictionary_payload,
        save_personal_dictionary_payload=save_personal_dictionary_payload,
        get_model_options=get_model_options,
        get_secret_statuses=get_secret_statuses,
        set_api_key=set_api_key,
        set_transcription_settings=set_transcription_settings,
        set_hotkey_settings=set_hotkey_settings,
        set_general_settings=set_general_settings,
        toggle_dictation=toggle_dictation,
        get_history_records=get_history_records,
        get_history_insights=get_history_insights,
        delete_history_record=delete_history_record,
        clear_history=clear_history,
        export_history=export_history,
        get_achievement_summary=get_achievement_summary,
        reset_achievements=reset_achievements,
        set_achievement_settings=set_achievement_settings,
        analyze_achievement_history=analyze_achievement_history,
        run_doctor_text=run_doctor_text,
        reset_settings_to_defaults=reset_settings_to_defaults,
        get_permission_statuses=get_permission_statuses,
        request_permission=request_permission,
        open_permission_settings=open_permission_settings,
        set_show_main_window_on_startup=set_show_main_window_on_startup,
        set_startup_splash_enabled=set_startup_splash_enabled,
        set_terminal_startup_animation_enabled=set_terminal_startup_animation_enabled,
    )
    splash_cfg = config.get("ui", {}).get("startup_splash", {})
    show_splash = bool(splash_cfg.get("enabled", True))
    if show_splash:
        window.startup_splash = StartupSplash(config)
        # Defer splash show until the event loop starts so macOS actually paints it.
        QTimer.singleShot(0, window.startup_splash.start)

    if config.get("ui", {}).get("show_main_window", True):
        window.show()
    else:
        window.hide()

    return app, window
