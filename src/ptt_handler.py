"""
Push-to-talk handler for Bloviate.
Manages global keyboard shortcuts for activating dictation.
"""

from pynput import keyboard
from typing import Callable, Optional, Dict
import ctypes
import ctypes.util
import sys
import threading
import time


class PTTHandler:
    """Handles push-to-talk functionality with global keyboard shortcuts."""

    def __init__(self, config: dict):
        self.config = config
        self.verbose_logs = bool(config.get("app", {}).get("verbose_logs", False))
        ptt_config = config.get('ptt', {})
        raw_hotkeys = self._resolve_hotkey_strs(ptt_config)

        self.is_active = False
        self.listener: Optional[keyboard.Listener] = None
        self._is_started = False
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._poll_interval_s = max(0.01, float(ptt_config.get("modifier_poll_interval_ms", 20)) / 1000.0)
        self._flag_reader = None
        self._press_timer: Optional[threading.Timer] = None
        self.press_delay_s = max(0.0, float(ptt_config.get("press_delay_ms", 90)) / 1000.0)

        # Callbacks for PTT hotkey
        self.on_press_callback: Optional[Callable] = None
        self.on_release_callback: Optional[Callable] = None

        # Additional hotkeys with their callbacks
        self.additional_hotkeys: Dict[str, dict] = {}
        self.active_hotkeys = set()
        self.tap_sequences: Dict[str, dict] = {}
        self._tap_candidate_key: Optional[str] = None
        self._tap_chord_detected = False

        # Parse PTT hotkeys
        self.hotkey_strs = []
        self.hotkeys = []
        for hotkey_str in raw_hotkeys:
            hotkey_set = self._parse_hotkey(hotkey_str)
            if not hotkey_set:
                if self.verbose_logs:
                    print(f"Warning: Hotkey '{hotkey_str}' has no valid keys; skipping.")
                continue
            self.hotkey_strs.append(hotkey_str)
            self.hotkeys.append(hotkey_set)
        if not self.hotkeys:
            raise ValueError("No valid PTT hotkeys configured.")

        # Keep primary hotkey string for display/backwards compatibility
        self.hotkey_str = self.hotkey_strs[0]
        self.hotkey = self.hotkeys[0]
        self.current_keys = set()
        self._listener_backend = self._resolve_listener_backend(ptt_config)

    def _resolve_listener_backend(self, ptt_config: dict) -> str:
        """Choose the global hotkey backend."""
        configured = str(ptt_config.get("listener_backend", "auto") or "auto").strip().lower()
        if configured in {"pynput", "modifier_poll"}:
            return configured
        if sys.platform == "darwin":
            return "modifier_poll"
        return "pynput"

    def _resolve_hotkey_strs(self, ptt_config: dict) -> list:
        """Resolve a list of PTT hotkey strings from config."""
        hotkeys = ptt_config.get('hotkeys')
        if hotkeys is None:
            hotkeys = []
            primary = ptt_config.get('hotkey')
            if primary:
                hotkeys.append(primary)
            secondary = ptt_config.get('secondary_hotkey')
            if secondary:
                hotkeys.append(secondary)

        if isinstance(hotkeys, str):
            hotkeys = [hotkeys]

        return [str(h).strip() for h in hotkeys if str(h).strip()]

    def _parse_hotkey(self, hotkey_str: str) -> set:
        """
        Parse hotkey string into a set of keys.

        Examples:
            "<ctrl>+<shift>+<space>" -> {Key.ctrl, Key.shift, Key.space}
            "<cmd>+<option>" -> {Key.cmd, Key.alt}
            "<cmd>+<shift>+<space>" -> {Key.cmd, Key.shift, Key.space}
            "<ctrl>+<cmd>+<left>" -> {Key.ctrl, Key.cmd, Key.left}
        """
        # Map of string representations to pynput keys
        key_map = {
            '<ctrl>': keyboard.Key.ctrl,
            '<shift>': keyboard.Key.shift,
            '<alt>': keyboard.Key.alt,
            '<option>': keyboard.Key.alt,  # Mac Option key (same as Alt)
            '<cmd>': keyboard.Key.cmd,
            '<space>': keyboard.Key.space,
            '<enter>': keyboard.Key.enter,
            '<tab>': keyboard.Key.tab,
            '<esc>': keyboard.Key.esc,
            '<left>': keyboard.Key.left,
            '<right>': keyboard.Key.right,
            '<up>': keyboard.Key.up,
            '<down>': keyboard.Key.down,
        }
        fn_key = getattr(keyboard.Key, 'fn', None)
        if fn_key is not None:
            key_map['<fn>'] = fn_key

        keys = set()
        parts = hotkey_str.lower().split('+')

        for part in parts:
            part = part.strip()
            if part in key_map:
                keys.add(key_map[part])
            elif len(part) == 1:
                # Single character key
                keys.add(keyboard.KeyCode.from_char(part))

        return keys

    @staticmethod
    def _modifier_keys() -> set:
        keys = {
            keyboard.Key.ctrl,
            keyboard.Key.shift,
            keyboard.Key.alt,
            keyboard.Key.cmd,
        }
        fn_key = getattr(keyboard.Key, 'fn', None)
        if fn_key is not None:
            keys.add(fn_key)
        return keys

    def _hotkey_is_modifier_only(self, hotkey_set: set) -> bool:
        return bool(hotkey_set) and hotkey_set.issubset(self._modifier_keys())

    def _normalize_key(self, key) -> Optional[str]:
        """Return a normalized key name for comparison."""
        if hasattr(key, 'name') and key.name:
            return key.name.replace('_l', '').replace('_r', '')
        if hasattr(key, 'char') and key.char:
            return key.char.lower()
        return None

    def _key_set(self, keys: set) -> set:
        """Normalize a set of keys to comparable names."""
        normalized = set()
        for key in keys:
            name = self._normalize_key(key)
            if name:
                normalized.add(name)
        return normalized

    def _matches_hotkey(self, hotkey_set: set, exact: bool = False) -> bool:
        """Check if currently pressed keys match the hotkey combination."""
        current_keys = self._key_set(self.current_keys)
        target_keys = self._key_set(hotkey_set)

        if not target_keys.issubset(current_keys):
            return False

        if exact and current_keys != target_keys:
            return False

        return True

    def _matches_any_ptt_hotkey(self) -> bool:
        """Check if any configured PTT hotkey is currently active."""
        return any(self._matches_hotkey(hotkey_set) for hotkey_set in self.hotkeys)

    def _current_key_names(self) -> set:
        return self._key_set(self.current_keys)

    def _has_pending_consuming_extension(self) -> bool:
        """Return True if current keys could still become a consuming hotkey."""
        current_names = self._current_key_names()
        if not current_names:
            return False
        for hotkey_info in self.additional_hotkeys.values():
            if not hotkey_info.get("consume", False):
                continue
            target_names = self._key_set(hotkey_info["hotkey"])
            if current_names < target_names:
                return True
        return False

    def _cancel_pending_ptt(self):
        timer = self._press_timer
        self._press_timer = None
        if timer:
            timer.cancel()

    def _activate_ptt_if_still_matching(self):
        self._press_timer = None
        if self._is_started and self._matches_any_ptt_hotkey() and not self.is_active:
            self.is_active = True
            if self.on_press_callback:
                self.on_press_callback()

    def add_hotkey(
        self,
        name: str,
        hotkey_str: str,
        on_press: Callable,
        on_release: Optional[Callable] = None,
        match_exact: bool = False,
        consume: bool = False
    ):
        """
        Add an additional hotkey.

        Args:
            name: Identifier for this hotkey
            hotkey_str: Hotkey string (e.g., "<ctrl>+<cmd>")
            on_press: Callback when hotkey is pressed
            on_release: Optional callback when hotkey is released
            match_exact: If True, require no extra keys beyond the hotkey
        """
        hotkey_set = self._parse_hotkey(hotkey_str)
        if not hotkey_set:
            if self.verbose_logs:
                print(f"Warning: Hotkey '{hotkey_str}' has no valid keys; ignoring.")
            return
        self.additional_hotkeys[name] = {
            'hotkey': hotkey_set,
            'on_press': on_press,
            'on_release': on_release,
            'is_active': False,
            'match_exact': match_exact,
            'consume': consume,
        }
        if self.verbose_logs:
            print(f"Added hotkey '{name}': {hotkey_str}")

    def add_tap_sequence(
        self,
        name: str,
        key_str: str,
        count: int,
        max_interval_s: float,
        callback: Callable,
    ):
        """Run a callback when one key is tapped repeatedly in quick succession."""
        key_set = self._parse_hotkey(key_str)
        key_names = self._key_set(key_set)
        if len(key_names) != 1:
            if self.verbose_logs:
                print(f"Warning: tap sequence '{name}' needs exactly one key: {key_str}")
            return
        self.tap_sequences[name] = {
            "key": next(iter(key_names)),
            "count": max(2, int(count)),
            "max_interval_s": max(0.1, float(max_interval_s)),
            "callback": callback,
            "current_count": 0,
            "last_tap": 0.0,
        }

    def _handle_tap_release(self, key):
        released_name = self._normalize_key(key)
        if (
            not released_name
            or self.current_keys
            or self._tap_chord_detected
            or self._tap_candidate_key != released_name
        ):
            return
        now = time.monotonic()
        for sequence in self.tap_sequences.values():
            if sequence["key"] != released_name:
                continue
            if now - sequence["last_tap"] <= sequence["max_interval_s"]:
                sequence["current_count"] += 1
            else:
                sequence["current_count"] = 1
            sequence["last_tap"] = now
            if sequence["current_count"] >= sequence["count"]:
                sequence["current_count"] = 0
                sequence["last_tap"] = 0.0
                callback = sequence.get("callback")
                if callback:
                    callback()

    def _on_press(self, key):
        """Handle key press events."""
        pressed_name = self._normalize_key(key)
        if not self.current_keys:
            self._tap_candidate_key = pressed_name
            self._tap_chord_detected = False
        elif pressed_name != self._tap_candidate_key:
            self._tap_chord_detected = True
        self.current_keys.add(key)

        consumed = False
        # Check additional hotkeys first so an exact toggle like
        # Cmd+Option+Shift does not also activate the shorter Cmd+Option PTT.
        for name, hotkey_info in self.additional_hotkeys.items():
            if self._matches_hotkey(
                hotkey_info['hotkey'],
                exact=hotkey_info.get('match_exact', False)
            ) and not hotkey_info['is_active']:
                hotkey_info['is_active'] = True
                self.active_hotkeys.add(name)
                if hotkey_info['on_press']:
                    hotkey_info['on_press']()
                consumed = consumed or bool(hotkey_info.get('consume', False))

        if consumed:
            self._cancel_pending_ptt()
            return

        # Check main PTT hotkey
        if self._matches_any_ptt_hotkey() and not self.is_active:
            if self.press_delay_s > 0 and self._has_pending_consuming_extension():
                if self._press_timer is None:
                    self._press_timer = threading.Timer(
                        self.press_delay_s,
                        self._activate_ptt_if_still_matching,
                    )
                    self._press_timer.daemon = True
                    self._press_timer.start()
            else:
                self._activate_ptt_if_still_matching()

    def _on_release(self, key):
        """Handle key release events."""
        try:
            self.current_keys.remove(key)
        except KeyError:
            pass

        if self._press_timer and not self._matches_any_ptt_hotkey():
            self._cancel_pending_ptt()

        # Check if main PTT hotkey is no longer active
        if self.is_active and not self._matches_any_ptt_hotkey():
            self.is_active = False
            if self.on_release_callback:
                self.on_release_callback()

        # Check additional hotkeys
        for name, hotkey_info in self.additional_hotkeys.items():
            if hotkey_info['is_active'] and not self._matches_hotkey(
                hotkey_info['hotkey'],
                exact=hotkey_info.get('match_exact', False)
            ):
                hotkey_info['is_active'] = False
                self.active_hotkeys.discard(name)
                if hotkey_info['on_release']:
                    hotkey_info['on_release']()

        self._handle_tap_release(key)
        if not self.current_keys:
            self._tap_candidate_key = None
            self._tap_chord_detected = False

    def _load_modifier_flag_reader(self):
        """Return CGEventSourceFlagsState for low-risk modifier polling on macOS."""
        if self._flag_reader is not None:
            return self._flag_reader
        if sys.platform != "darwin":
            return None
        path = ctypes.util.find_library("ApplicationServices")
        if not path:
            return None
        app_services = ctypes.cdll.LoadLibrary(path)
        reader = app_services.CGEventSourceFlagsState
        reader.argtypes = [ctypes.c_int]
        reader.restype = ctypes.c_ulonglong
        self._flag_reader = reader
        return reader

    def _keys_from_modifier_flags(self, flags: int) -> set:
        """Map CoreGraphics modifier flags to pynput Key objects."""
        # CGEventFlag masks:
        # shift 0x20000, control 0x40000, option 0x80000,
        # command 0x100000, secondary fn 0x800000.
        keys = set()
        if flags & 0x00020000:
            keys.add(keyboard.Key.shift)
        if flags & 0x00040000:
            keys.add(keyboard.Key.ctrl)
        if flags & 0x00080000:
            keys.add(keyboard.Key.alt)
        if flags & 0x00100000:
            keys.add(keyboard.Key.cmd)
        fn_key = getattr(keyboard.Key, 'fn', None)
        if fn_key is not None and flags & 0x00800000:
            keys.add(fn_key)
        return keys

    def _poll_modifier_hotkeys(self):
        reader = self._load_modifier_flag_reader()
        if reader is None:
            if self.verbose_logs:
                print("Modifier polling backend unavailable; falling back to no global hotkeys.")
            return
        previous_keys = set()
        while not self._poll_stop.is_set():
            try:
                # kCGEventSourceStateHIDSystemState = 1, system-wide hardware key state.
                current_keys = self._keys_from_modifier_flags(int(reader(1)))
                for key in current_keys - previous_keys:
                    self._on_press(key)
                for key in previous_keys - current_keys:
                    self._on_release(key)
                previous_keys = current_keys
            except Exception as exc:
                print(f"[Hotkeys] Modifier poll error: {exc}")
            self._poll_stop.wait(self._poll_interval_s)

        for key in list(previous_keys):
            self._on_release(key)

    def start(self, on_press: Callable, on_release: Callable):
        """
        Start listening for the PTT hotkey.

        Args:
            on_press: Callback when PTT is activated
            on_release: Callback when PTT is deactivated
        """
        self.on_press_callback = on_press
        self.on_release_callback = on_release
        self._is_started = True

        if self._listener_backend == "modifier_poll":
            unsupported = [
                hotkey
                for hotkey in self.hotkeys
                if not self._hotkey_is_modifier_only(hotkey)
            ]
            unsupported.extend(
                hotkey_info["hotkey"]
                for hotkey_info in self.additional_hotkeys.values()
                if not self._hotkey_is_modifier_only(hotkey_info["hotkey"])
            )
            if unsupported and self.verbose_logs:
                print("[Hotkeys] Modifier polling ignores non-modifier hotkeys.")
            self._poll_stop.clear()
            self.listener = None
            self._poll_thread = threading.Thread(
                target=self._poll_modifier_hotkeys,
                name="bloviate-modifier-hotkeys",
                daemon=True,
            )
            self._poll_thread.start()
            print(f"PTT handler started with modifier polling: {', '.join(self.hotkey_strs)}")
            return

        # Start keyboard listener in a separate thread
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

        if self.verbose_logs:
            if len(self.hotkey_strs) > 1:
                print(f"PTT handler started with hotkeys: {', '.join(self.hotkey_strs)}")
            else:
                print(f"PTT handler started with hotkey: {self.hotkey_str}")

    def stop(self, join_timeout: float = 1.0):
        """Stop listening for keyboard events."""
        listener = self.listener
        self.listener = None
        self._is_started = False
        if self._poll_thread:
            self._poll_stop.set()
            self._poll_thread.join(timeout=join_timeout)
            self._poll_thread = None

        self.is_active = False
        self._cancel_pending_ptt()
        self.current_keys.clear()
        self.active_hotkeys.clear()
        for hotkey_info in self.additional_hotkeys.values():
            hotkey_info['is_active'] = False
        for sequence in self.tap_sequences.values():
            sequence["current_count"] = 0
            sequence["last_tap"] = 0.0
        self._tap_candidate_key = None
        self._tap_chord_detected = False

        if listener:
            listener.stop()
            listener.join(timeout=join_timeout)
            if self.verbose_logs:
                print("PTT handler stopped")

    def wait(self):
        """Wait for the listener thread to finish."""
        if self.listener:
            self.listener.join()
        if self._poll_thread:
            self._poll_thread.join()
