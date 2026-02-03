"""
Push-to-talk handler for Bloviate.
Manages global keyboard shortcuts for activating dictation.
"""

from pynput import keyboard
from typing import Callable, Optional, Dict
import threading


class PTTHandler:
    """Handles push-to-talk functionality with global keyboard shortcuts."""

    def __init__(self, config: dict):
        self.config = config
        ptt_config = config.get('ptt', {})
        raw_hotkeys = self._resolve_hotkey_strs(ptt_config)

        self.is_active = False
        self.listener: Optional[keyboard.Listener] = None

        # Callbacks for PTT hotkey
        self.on_press_callback: Optional[Callable] = None
        self.on_release_callback: Optional[Callable] = None

        # Additional hotkeys with their callbacks
        self.additional_hotkeys: Dict[str, dict] = {}
        self.active_hotkeys = set()

        # Parse PTT hotkeys
        self.hotkey_strs = []
        self.hotkeys = []
        for hotkey_str in raw_hotkeys:
            hotkey_set = self._parse_hotkey(hotkey_str)
            if not hotkey_set:
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

    def add_hotkey(
        self,
        name: str,
        hotkey_str: str,
        on_press: Callable,
        on_release: Optional[Callable] = None,
        match_exact: bool = False
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
            print(f"Warning: Hotkey '{hotkey_str}' has no valid keys; ignoring.")
            return
        self.additional_hotkeys[name] = {
            'hotkey': hotkey_set,
            'on_press': on_press,
            'on_release': on_release,
            'is_active': False,
            'match_exact': match_exact
        }
        print(f"Added hotkey '{name}': {hotkey_str}")

    def _on_press(self, key):
        """Handle key press events."""
        self.current_keys.add(key)

        # Check main PTT hotkey
        if self._matches_any_ptt_hotkey() and not self.is_active:
            self.is_active = True
            if self.on_press_callback:
                self.on_press_callback()

        # Check additional hotkeys
        for name, hotkey_info in self.additional_hotkeys.items():
            if self._matches_hotkey(
                hotkey_info['hotkey'],
                exact=hotkey_info.get('match_exact', False)
            ) and not hotkey_info['is_active']:
                hotkey_info['is_active'] = True
                self.active_hotkeys.add(name)
                if hotkey_info['on_press']:
                    hotkey_info['on_press']()

    def _on_release(self, key):
        """Handle key release events."""
        try:
            self.current_keys.remove(key)
        except KeyError:
            pass

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

    def start(self, on_press: Callable, on_release: Callable):
        """
        Start listening for the PTT hotkey.

        Args:
            on_press: Callback when PTT is activated
            on_release: Callback when PTT is deactivated
        """
        self.on_press_callback = on_press
        self.on_release_callback = on_release

        # Start keyboard listener in a separate thread
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

        if len(self.hotkey_strs) > 1:
            print(f"PTT handler started with hotkeys: {', '.join(self.hotkey_strs)}")
        else:
            print(f"PTT handler started with hotkey: {self.hotkey_str}")

    def stop(self):
        """Stop listening for keyboard events."""
        if self.listener:
            self.listener.stop()
            self.listener = None
            print("PTT handler stopped")

    def wait(self):
        """Wait for the listener thread to finish."""
        if self.listener:
            self.listener.join()
