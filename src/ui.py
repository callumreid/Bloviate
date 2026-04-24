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
    QHeaderView, QAbstractItemView, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPropertyAnimation, QSignalBlocker
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
        self.parent = parent
        self.tray_icon = QSystemTrayIcon(parent)
        self.audio_level = 0
        self.current_state = "idle"  # idle, recording, processing, success, rejected, command_*
        self._pulse_phase = False
        self._closed = False
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
        if self._closed:
            return
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
        if self._closed:
            return
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
    _BAR_COUNT = 5
    _GAP = 3
    _MARGIN = 6
    _PROFILE = [0.35, 0.6, 0.9, 0.6, 0.35]
    _MIN_BAR_HEIGHT = 0.12
    _BAR_RADIUS = 2

    def __init__(self, config: dict):
        super().__init__(None)
        self.config = config
        self.audio_level = 0.0
        self.current_state = "idle"
        self._pulse_phase = False
        self._hold_state = None
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(320)
        self._pulse_timer.timeout.connect(self._toggle_pulse)
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._clear_hold_and_idle)

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
        self._visibility_timer.setInterval(500)
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
        super().close()

    def _toggle_pulse(self):
        self._pulse_phase = not self._pulse_phase
        if self.current_state in {"processing", "command_processing"}:
            self.update()

    def _pulse_color(self) -> QColor:
        return QColor(156, 39, 176) if self._pulse_phase else QColor(123, 31, 162)

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

    def _position_bottom_center(self):
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        x = geo.x() + int((geo.width() - self._SIZE) / 2)
        y = geo.y() + geo.height() - self._SIZE - self._screen_margin
        self.move(x, y)

    def _state_color(self) -> QColor:
        if self.current_state in {"processing", "command_processing"}:
            return self._pulse_color()
        if self.current_state == "recording":
            return QColor(255, 193, 7)
        if self.current_state == "command_recording":
            return QColor(33, 150, 243)
        if self.current_state in {"command_success", "accepted"}:
            return QColor(76, 175, 80)
        if self.current_state in {"command_unknown", "rejected"}:
            return QColor(244, 67, 54)
        return QColor(160, 160, 160)

    def set_audio_level(self, level: float):
        self.audio_level = max(0.0, min(level, 1.0))
        self.update()

    def set_recording(self):
        self._clear_hold()
        self.current_state = "recording"
        self._stop_pulse()
        self.update()

    def set_processing(self):
        self._clear_hold()
        self.current_state = "processing"
        self._start_pulse()
        self.update()

    def set_command_recording(self):
        self._clear_hold()
        self.current_state = "command_recording"
        self._stop_pulse()
        self.update()

    def set_command_processing(self):
        self._clear_hold()
        self.current_state = "command_processing"
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
        painter.setBrush(self._state_color())
        level = max(0.0, min(self.audio_level, 1.0))
        usable_w = self._SIZE - self._MARGIN * 2
        usable_h = self._SIZE - self._MARGIN * 2
        bar_w = int((usable_w - self._GAP * (self._BAR_COUNT - 1)) / self._BAR_COUNT)

        for idx, base in enumerate(self._PROFILE):
            height_ratio = self._MIN_BAR_HEIGHT + (base - self._MIN_BAR_HEIGHT) * level
            h = int(usable_h * height_ratio)
            x = self._MARGIN + idx * (bar_w + self._GAP)
            y = self._MARGIN + (usable_h - h)
            painter.drawRoundedRect(x, y, bar_w, h, self._BAR_RADIUS, self._BAR_RADIUS)

        painter.end()


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
        get_history_records=None,
        delete_history_record=None,
        clear_history=None,
        export_history=None,
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
        self.get_history_records = get_history_records
        self.delete_history_record = delete_history_record
        self.clear_history = clear_history
        self.export_history = export_history
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
        self.signals.update_interim_transcription.connect(self._update_interim_transcription)
        self.signals.update_status.connect(self._update_status)
        self.signals.update_command_status.connect(self._update_command_status)

        self._last_final_text = ""
        self._transcription_style_final = ""
        self._transcription_style_interim = ""
        self._audio_inputs_ready = bool(self.get_audio_inputs and self.set_audio_input)
        self._settings_status_default_style = "font-size: 12px; color: #6F665E;"
        self._settings_ok_style = "font-size: 12px; color: #2F7D4F; font-weight: 600;"
        self._settings_error_style = "font-size: 12px; color: #B23B35; font-weight: 600;"
        self._dictionary_terms = []
        self._dictionary_corrections = []
        self._permissions_prompt_shown = False

        self.init_ui()

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
        nav_layout.setContentsMargins(0, 18, 0, 10)
        nav_layout.setSpacing(0)
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
        self.ptt_label.setStyleSheet(self._status_pill_style("#E9E1D3", "#24201C"))
        layout.addWidget(self.ptt_label)

        # Command Mode Status
        self.command_label = QLabel("CMD: Inactive")
        self.command_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.command_label.setObjectName("StatusPill")
        self.command_label.setStyleSheet(self._status_pill_style("#E9E1D3", "#24201C"))
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
        self.status_label.setStyleSheet("font-style: italic; color: #6F665E; padding: 8px;")
        layout.addWidget(self.status_label)

        # Last transcription
        self.transcription_label = QLabel("")
        self.transcription_label.setWordWrap(True)
        self._transcription_style_final = (
            "padding: 14px; background-color: #FFFDF7; color: #26211D; "
            "border: 1px solid #DDD2C1; border-radius: 8px; min-height: 52px;"
        )
        self._transcription_style_interim = (
            "padding: 14px; background-color: #FFF9E9; color: #7A5B2C; "
            "border: 1px solid #E3C88A; border-radius: 8px; min-height: 52px; "
            "font-style: italic;"
        )
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
        scroll.viewport().setStyleSheet("background-color: #F7F3EA;")
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
        self.command_hotkey_edit = QLineEdit(
            self._config_text("window_management", "command_hotkey", "<ctrl>+<cmd>")
        )
        self.window_prefix_hotkey_edit = QLineEdit(
            self._config_text("window_management", "hotkey_prefix", "<ctrl>+<cmd>")
        )
        hotkey_placeholders = {
            self.ptt_hotkey_edit: "<cmd>+<option>",
            self.ptt_secondary_hotkey_edit: "<fn>",
            self.command_hotkey_edit: "<ctrl>+<cmd>",
            self.window_prefix_hotkey_edit: "<ctrl>+<cmd>",
        }
        for edit, placeholder in hotkey_placeholders.items():
            edit.setPlaceholderText(placeholder)
            edit.setMinimumWidth(260)
        hotkey_layout.addRow("Primary PTT:", self.ptt_hotkey_edit)
        hotkey_layout.addRow("Secondary PTT:", self.ptt_secondary_hotkey_edit)
        hotkey_layout.addRow("Command PTT:", self.command_hotkey_edit)
        hotkey_layout.addRow("Window prefix:", self.window_prefix_hotkey_edit)
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
        QMessageBox.information(
            self,
            "Bloviate Permissions",
            (
                "Bloviate needs microphone access for dictation, Accessibility/Input Monitoring "
                "for global hotkeys, and Automation/Accessibility for auto-paste. "
                "Use the Permissions section at the top of Settings to open each macOS prompt."
            ),
        )

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

    def _key_field_widget(self, line_edit: QLineEdit, status_label: QLabel) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(line_edit)
        layout.addWidget(status_label)
        return widget

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

    def _apply_hotkey_settings(self):
        updates = {
            "ptt.hotkey": self.ptt_hotkey_edit.text().strip(),
            "ptt.secondary_hotkey": self.ptt_secondary_hotkey_edit.text().strip(),
            "window_management.command_hotkey": self.command_hotkey_edit.text().strip(),
            "window_management.hotkey_prefix": self.window_prefix_hotkey_edit.text().strip(),
        }
        if not self.set_hotkey_settings:
            self._set_settings_status(self.hotkey_status_label, "Hotkey updates unavailable.", ok=False)
            return
        ok, message = self.set_hotkey_settings(updates)
        if ok:
            self.config.setdefault("ptt", {})["hotkey"] = updates["ptt.hotkey"]
            self.config.setdefault("ptt", {})["secondary_hotkey"] = updates["ptt.secondary_hotkey"]
            self.config.setdefault("window_management", {})["command_hotkey"] = updates[
                "window_management.command_hotkey"
            ]
            self.config.setdefault("window_management", {})["hotkey_prefix"] = updates[
                "window_management.hotkey_prefix"
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
        if ok:
            self.doctor_output.setStyleSheet("color: #26211D;")
        else:
            self.doctor_output.setStyleSheet("color: #B23B35;")

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
        self.doctor_output.setStyleSheet("color: #26211D;" if ok else "color: #B23B35;")

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
            "border: 1px solid #D8CCBA; border-radius: 8px;"
        )

    def set_light_theme(self):
        """Apply the polished default light theme."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#F7F3EA"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#26211D"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#FFFDF7"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#F1E9DB"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#26211D"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#FFFDF7"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#26211D"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#FFFDF7"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#26211D"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#B23B35"))
        palette.setColor(QPalette.ColorRole.Link, QColor("#2D6B6B"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#2D6B6B"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
        self.setPalette(palette)
        QApplication.instance().setPalette(palette)
        self.setStyleSheet(
            """
            QMainWindow, QWidget#AppRoot, QWidget#SettingsContent, QStackedWidget {
                background: #F7F3EA;
                color: #26211D;
            }
            QWidget#TopNav {
                background: #F7F3EA;
            }
            QPushButton#NavButton {
                background: #E8DFD0;
                color: #37312B;
                min-width: 140px;
                padding: 13px 24px;
                border: 1px solid #D8CCBA;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton#NavButton:checked {
                background: #2D6B6B;
                color: #FFFFFF;
                border-color: #2D6B6B;
            }
            QPushButton#NavButton:hover {
                background: #DCD2C2;
            }
            QPushButton#NavButton:checked:hover {
                background: #2D6B6B;
            }
            QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {
                background: #F7F3EA;
                border: 0;
            }
            QGroupBox {
                background: #FFFDF7;
                border: 1px solid #DDD2C1;
                border-radius: 8px;
                margin-top: 22px;
                padding: 18px 18px 16px 18px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                color: #26211D;
                background: #F7F3EA;
            }
            QLabel {
                color: #26211D;
            }
            QLineEdit, QTextEdit, QComboBox, QTableWidget {
                background: #FFFFFF;
                color: #26211D;
                border: 1px solid #CFC3B2;
                border-radius: 6px;
                padding: 7px 9px;
                selection-background-color: #2D6B6B;
                selection-color: #FFFFFF;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 1px solid #2D6B6B;
            }
            QLineEdit::placeholder {
                color: #8E8376;
            }
            QComboBox::drop-down {
                border: 0;
                width: 26px;
            }
            QPushButton {
                background-color: #FFFDF7;
                color: #26211D;
                border: 1px solid #2E332F;
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #F1E9DB;
            }
            QPushButton:pressed {
                background-color: #E8DFD0;
            }
            QPushButton:disabled {
                background-color: #E3DBCE;
                color: #6F665E;
                border-color: #D8D0C2;
            }
            QCheckBox {
                color: #26211D;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 5px;
                border: 1px solid #BFB2A1;
                background: #FFFFFF;
            }
            QCheckBox::indicator:checked {
                background: #2D6B6B;
                border-color: #2D6B6B;
            }
            QTableWidget {
                gridline-color: #E5DBC9;
                alternate-background-color: #F7F1E7;
            }
            QHeaderView::section {
                background: #E9E1D3;
                color: #26211D;
                border: 0;
                border-right: 1px solid #D8CCBA;
                padding: 7px 9px;
                font-weight: 700;
            }
            QProgressBar {
                background: #FFFFFF;
                border: 1px solid #CFC3B2;
                border-radius: 6px;
                color: #26211D;
                text-align: center;
                min-height: 24px;
            }
            QProgressBar::chunk {
                background-color: #2D6B6B;
                border-radius: 5px;
            }
            QSlider::groove:horizontal {
                background: #E2D7C6;
                height: 7px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 1px solid #BFB2A1;
                width: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #2D6B6B;
                border-radius: 3px;
            }
            QScrollBar:vertical {
                background: #F1E9DB;
                width: 12px;
                margin: 0;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #BFB2A1;
                min-height: 36px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

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
        percentage = min(int(level / 0.26 * 100), 100)
        self.audio_bar.setValue(percentage)

        # Update menu bar indicator
        if self.menu_bar_indicator:
            self.menu_bar_indicator.set_audio_level(level / 0.26)

        if self.ptt_overlay:
            self.ptt_overlay.set_audio_level(level / 0.26)

        # Color based on level
        if percentage > 60:
            color = "#2F7D4F"
        elif percentage > 20:
            color = "#C58C2A"
        else:
            color = "#BFB2A1"

        self.audio_bar.setStyleSheet(f"""
            QProgressBar {{
                background: #FFFFFF;
                border: 1px solid #CFC3B2;
                border-radius: 6px;
                color: #26211D;
                text-align: center;
                min-height: 24px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 5px;
            }}
        """)

    def _update_ptt_status(self, is_active: bool):
        """Update PTT status indicator."""
        if is_active:
            self.ptt_label.setText("PTT: ACTIVE")
            self.ptt_label.setStyleSheet(self._status_pill_style("#2F7D4F", "#FFFFFF"))
            # Update menu bar indicator
            if self.menu_bar_indicator:
                self.menu_bar_indicator.set_recording()
            if self.ptt_overlay:
                self.ptt_overlay.set_recording()
        else:
            self.ptt_label.setText("PTT: Inactive")
            self.ptt_label.setStyleSheet(self._status_pill_style("#E9E1D3", "#24201C"))

    def _update_command_status(self, message: str, state: str):
        """Update command mode indicator."""
        styles = {
            "inactive": self._status_pill_style("#E9E1D3", "#24201C"),
            "listening": self._status_pill_style("#2D6B6B", "#FFFFFF"),
            "processing": self._status_pill_style("#E7C873", "#24201C"),
            "recognized": self._status_pill_style("#2F7D4F", "#FFFFFF"),
            "unrecognized": self._status_pill_style("#B23B35", "#FFFFFF"),
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
        if score < 0:
            self.match_status_label.setText("Talk mode")
            self.match_status_label.setStyleSheet("color: #6F665E; font-weight: bold;")
            self.match_score_label.setText("")
            return

        if is_match:
            self.match_status_label.setText("Matched")
            self.match_status_label.setStyleSheet("color: #2F7D4F; font-weight: bold;")
            if self.menu_bar_indicator:
                self.menu_bar_indicator.set_accepted()
            if self.ptt_overlay:
                self.ptt_overlay.set_accepted()
        else:
            self.match_status_label.setText("Rejected")
            self.match_status_label.setStyleSheet("color: #B23B35; font-weight: bold;")
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
    get_history_records=None,
    delete_history_record=None,
    clear_history=None,
    export_history=None,
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
        get_history_records=get_history_records,
        delete_history_record=delete_history_record,
        clear_history=clear_history,
        export_history=export_history,
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
