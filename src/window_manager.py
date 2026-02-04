"""
Window management for macOS.
Handles resizing and positioning the focused window.
"""

import subprocess
from typing import Literal

WindowPosition = Literal[
    "left", "right", "top", "bottom",
    "fullscreen", "exit_fullscreen",
    "larger", "smaller",
    "top_left_quarter", "top_right_quarter",
    "bottom_left_quarter", "bottom_right_quarter",
]


class WindowManager:
    """Manages window positioning on macOS using AppleScript."""

    MENU_BAR_HEIGHT = 25  # Approximate macOS menu bar height

    def __init__(self):
        self._get_screen_size()

    def _get_screen_size(self):
        """Get the main screen size in point-based coordinates."""
        # Use NSScreen for reliable point-based dimensions that match
        # the coordinate system System Events uses for window positioning.
        # The old Finder desktop-bounds approach can return wrong values
        # on Retina displays or when Finder isn't responsive.
        script = (
            'use framework "AppKit"\n'
            'set f to (current application\'s NSScreen\'s mainScreen()\'s frame())\n'
            'set w to current application\'s NSWidth(f)\n'
            'set h to current application\'s NSHeight(f)\n'
            'return ((w as integer) as text) & "," & ((h as integer) as text)'
        )
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                w, h = result.stdout.strip().split(',')
                self.screen_width = int(w)
                self.screen_height = int(h)
                print(f"Screen size: {self.screen_width}x{self.screen_height} (points)")
                return
        except Exception:
            pass

        # Fallback to common resolution
        self.screen_width = 1920
        self.screen_height = 1080
        print(f"Screen size: {self.screen_width}x{self.screen_height} (fallback)")

    def resize_focused_window(self, position: WindowPosition):
        """
        Resize the currently focused window to the specified position.

        Args:
            position: One of "left", "right", "top", "bottom"
        """
        # Calculate new bounds based on position (accounting for menu bar)
        menu_offset = self.MENU_BAR_HEIGHT
        usable_height = self.screen_height - menu_offset

        half_width = self.screen_width // 2
        half_height = usable_height // 2

        if position == "left":
            x, y = 0, menu_offset
            width, height = half_width, usable_height
        elif position == "right":
            x, y = half_width, menu_offset
            width, height = half_width, usable_height
        elif position == "top":
            x, y = 0, menu_offset
            width, height = self.screen_width, half_height
        elif position == "bottom":
            x, y = 0, menu_offset + half_height
            width, height = self.screen_width, half_height
        elif position == "fullscreen":
            x, y = 0, menu_offset
            width, height = self.screen_width, usable_height
        elif position == "exit_fullscreen":
            # Restore to centered ~75% size
            width = int(self.screen_width * 0.75)
            height = int(usable_height * 0.75)
            x = (self.screen_width - width) // 2
            y = menu_offset + (usable_height - height) // 2
        elif position == "larger":
            self._resize_relative(1.15)
            return
        elif position == "smaller":
            self._resize_relative(0.85)
            return
        elif position == "top_left_quarter":
            x, y = 0, menu_offset
            width, height = half_width, half_height
        elif position == "top_right_quarter":
            x, y = half_width, menu_offset
            width, height = half_width, half_height
        elif position == "bottom_left_quarter":
            x, y = 0, menu_offset + half_height
            width, height = half_width, half_height
        elif position == "bottom_right_quarter":
            x, y = half_width, menu_offset + half_height
            width, height = half_width, half_height
        else:
            print(f"Unknown position: {position}")
            return

        # AppleScript to resize the frontmost application's window.
        script = f'''
        tell application "System Events"
            set targetApp to missing value
            try
                set targetApp to first application process whose frontmost is true
            end try
            if targetApp is missing value then return "NO_WINDOW"
            if (count of windows of targetApp) is 0 then return "NO_WINDOW"

            set targetWindow to missing value
            try
                set targetWindow to first window of targetApp whose value of attribute "AXMain" is true
            end try
            if targetWindow is missing value then
                try
                    set targetWindow to front window of targetApp
                end try
            end if
            if targetWindow is missing value then return "NO_WINDOW"

            set position of targetWindow to {{{x}, {y}}}
            set size of targetWindow to {{{width}, {height}}}
        end tell
        return "OK"
        '''

        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode != 0:
                print(f"Window resize error: {result.stderr}")
            elif result.stdout.strip() == "NO_WINDOW":
                print("Window resize error: No visible window found to resize.")
        except Exception as e:
            print(f"Error resizing window: {e}")

    def _resize_relative(self, scale: float):
        """Resize the focused window by a scale factor, keeping it centered."""
        script = '''
        tell application "System Events"
            set targetApp to first application process whose frontmost is true
            if (count of windows of targetApp) is 0 then return "NO_WINDOW"
            set targetWindow to front window of targetApp
            set {curX, curY} to position of targetWindow
            set {curW, curH} to size of targetWindow
        end tell
        return (curX as text) & "," & (curY as text) & "," & (curW as text) & "," & (curH as text)
        '''
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode != 0 or result.stdout.strip() == "NO_WINDOW":
                print("Window resize error: No visible window found.")
                return

            parts = [int(x.strip()) for x in result.stdout.strip().split(',')]
            cur_x, cur_y, cur_w, cur_h = parts

            new_w = int(cur_w * scale)
            new_h = int(cur_h * scale)
            new_x = cur_x - (new_w - cur_w) // 2
            new_y = cur_y - (new_h - cur_h) // 2

            # Clamp to screen bounds
            new_x = max(0, min(new_x, self.screen_width - new_w))
            new_y = max(self.MENU_BAR_HEIGHT, min(new_y, self.screen_height - new_h))

            set_script = f'''
            tell application "System Events"
                set targetWindow to front window of (first application process whose frontmost is true)
                set position of targetWindow to {{{new_x}, {new_y}}}
                set size of targetWindow to {{{new_w}, {new_h}}}
            end tell
            '''
            subprocess.run(
                ['osascript', '-e', set_script],
                capture_output=True, text=True, timeout=1
            )
        except Exception as e:
            print(f"Error resizing window: {e}")

    def switch_desktop(self, direction: str):
        """Switch macOS desktop Space using Ctrl+Arrow key simulation."""
        if direction == "left":
            arrow = "123"  # left arrow key code
        elif direction == "right":
            arrow = "124"  # right arrow key code
        else:
            print(f"Unknown desktop direction: {direction}")
            return

        script = f'''
        tell application "System Events"
            key code {arrow} using {{control down}}
        end tell
        '''
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                print(f"Desktop switch error: {result.stderr}")
        except Exception as e:
            print(f"Error switching desktop: {e}")

    def show_position_menu(self):
        """Show a simple menu to select window position."""
        # For now, we'll cycle through positions or use a simple approach
        # This could be enhanced with a proper UI menu later
        pass
