"""
Window management for macOS.
Handles resizing and positioning the focused window.
"""

import subprocess
from typing import Literal

WindowPosition = Literal["left", "right", "top", "bottom"]


class WindowManager:
    """Manages window positioning on macOS using AppleScript."""

    MENU_BAR_HEIGHT = 25  # Approximate macOS menu bar height

    def __init__(self):
        self._get_screen_size()

    def _get_screen_size(self):
        """Get the main screen size using AppleScript."""
        script = '''
        tell application "Finder"
            get bounds of window of desktop
        end tell
        '''
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                # Parse bounds: "x, y, width, height"
                bounds = [int(x.strip()) for x in result.stdout.strip().split(',')]
                self.screen_width = bounds[2] - bounds[0]
                self.screen_height = bounds[3] - bounds[1]
            else:
                # Fallback to common resolution
                self.screen_width = 1920
                self.screen_height = 1080
        except Exception:
            self.screen_width = 1920
            self.screen_height = 1080

    def resize_focused_window(self, position: WindowPosition):
        """
        Resize the currently focused window to the specified position.

        Args:
            position: One of "left", "right", "top", "bottom"
        """
        # Calculate new bounds based on position (accounting for menu bar)
        menu_offset = self.MENU_BAR_HEIGHT
        usable_height = self.screen_height - menu_offset

        if position == "left":
            x, y = 0, menu_offset
            width = self.screen_width // 2
            height = usable_height
        elif position == "right":
            x = self.screen_width // 2
            y = menu_offset
            width = self.screen_width // 2
            height = usable_height
        elif position == "top":
            x, y = 0, menu_offset
            width = self.screen_width
            height = usable_height // 2
        elif position == "bottom":
            x = 0
            y = menu_offset + (usable_height // 2)
            width = self.screen_width
            height = usable_height // 2
        else:
            print(f"Unknown position: {position}")
            return

        # AppleScript to resize focused window
        script = f'''
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            tell frontApp
                tell window 1
                    set position to {{{x}, {y}}}
                    set size to {{{width}, {height}}}
                end tell
            end tell
        end tell
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
        except Exception as e:
            print(f"Error resizing window: {e}")

    def show_position_menu(self):
        """Show a simple menu to select window position."""
        # For now, we'll cycle through positions or use a simple approach
        # This could be enhanced with a proper UI menu later
        pass
