"""Small macOS permission helpers used by the UI and paste path."""

from __future__ import annotations

import ctypes
import ctypes.util
import subprocess
import sys


def open_privacy_pane(kind: str) -> tuple[bool, str]:
    """Open a relevant macOS Privacy & Security settings pane."""
    if sys.platform != "darwin":
        return False, "macOS permission panes are only available on macOS."

    urls = {
        "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
        "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        "input_monitoring": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
        "automation": "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
    }
    url = urls.get(kind, "x-apple.systempreferences:com.apple.preference.security?Privacy")
    try:
        subprocess.Popen(["open", url])
        return True, f"Opened macOS {kind.replace('_', ' ')} permission settings."
    except Exception as exc:
        return False, f"Could not open macOS permission settings: {exc}"


def accessibility_trusted(*, prompt: bool = False) -> bool:
    """Return whether the current process is trusted for Accessibility.

    When ``prompt`` is true, macOS may show the system prompt and/or add the
    process to the Accessibility list. The user still has to enable it there.
    """
    if sys.platform != "darwin":
        return True

    try:
        app_services_path = ctypes.util.find_library("ApplicationServices")
        if not app_services_path:
            return False

        app_services = ctypes.cdll.LoadLibrary(app_services_path)

        if not prompt:
            app_services.AXIsProcessTrusted.restype = ctypes.c_bool
            return bool(app_services.AXIsProcessTrusted())

        try:
            return _accessibility_trusted_with_prompt(app_services)
        except Exception:
            app_services.AXIsProcessTrusted.restype = ctypes.c_bool
            return bool(app_services.AXIsProcessTrusted())
    except Exception:
        return False


def request_accessibility() -> bool:
    """Ask macOS to surface Accessibility setup and open the settings pane."""
    trusted = accessibility_trusted(prompt=True)
    if not trusted:
        open_privacy_pane("accessibility")
    return trusted


def _accessibility_trusted_with_prompt(app_services) -> bool:
    core_foundation_path = ctypes.util.find_library("CoreFoundation")
    if not core_foundation_path:
        raise RuntimeError("CoreFoundation unavailable")

    core_foundation = ctypes.cdll.LoadLibrary(core_foundation_path)
    prompt_key = ctypes.c_void_p.in_dll(app_services, "kAXTrustedCheckOptionPrompt").value
    true_value = ctypes.c_void_p.in_dll(core_foundation, "kCFBooleanTrue").value
    if not prompt_key or not true_value:
        raise RuntimeError("Accessibility prompt constants unavailable")

    keys = (ctypes.c_void_p * 1)(prompt_key)
    values = (ctypes.c_void_p * 1)(true_value)

    create_dictionary = core_foundation.CFDictionaryCreate
    create_dictionary.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    create_dictionary.restype = ctypes.c_void_p

    release = core_foundation.CFRelease
    release.argtypes = [ctypes.c_void_p]
    release.restype = None

    options = create_dictionary(None, keys, values, 1, None, None)
    if not options:
        raise RuntimeError("Could not build Accessibility prompt options")

    try:
        trusted_with_options = app_services.AXIsProcessTrustedWithOptions
        trusted_with_options.argtypes = [ctypes.c_void_p]
        trusted_with_options.restype = ctypes.c_bool
        return bool(trusted_with_options(options))
    finally:
        release(options)
