"""
Default-input device watcher for Bloviate (macOS).

Polls CoreAudio for the system default input device so the app can follow
desk/mic changes live instead of staying bound to whatever PortAudio saw at
stream-open time.
"""

import ctypes
import ctypes.util
import sys
import threading
from typing import Callable, Optional

_K_AUDIO_OBJECT_SYSTEM_OBJECT = 1
_K_DEFAULT_INPUT_SELECTOR = 0x64496E20  # 'dIn '
_K_SCOPE_GLOBAL = 0x676C6F62  # 'glob'
_K_ELEMENT_MAIN = 0
_K_DEVICE_NAME_SELECTOR = 0x6C6E616D  # 'lnam'
_K_CF_STRING_ENCODING_UTF8 = 0x08000100


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


def _load_frameworks():
    core_audio_path = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
    core_foundation_path = ctypes.util.find_library("CoreFoundation")
    if not core_foundation_path:
        return None, None
    try:
        core_audio = ctypes.cdll.LoadLibrary(core_audio_path)
        core_foundation = ctypes.cdll.LoadLibrary(core_foundation_path)
    except OSError:
        return None, None
    return core_audio, core_foundation


def get_default_input_device() -> tuple[Optional[int], str]:
    """Return (coreaudio_device_id, device_name) for the system default input."""
    if sys.platform != "darwin":
        return None, ""
    core_audio, core_foundation = _load_frameworks()
    if core_audio is None:
        return None, ""

    get_data = core_audio.AudioObjectGetPropertyData
    get_data.restype = ctypes.c_int32
    get_data.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(_PropertyAddress),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]

    address = _PropertyAddress(_K_DEFAULT_INPUT_SELECTOR, _K_SCOPE_GLOBAL, _K_ELEMENT_MAIN)
    device_id = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(device_id))
    status = get_data(
        _K_AUDIO_OBJECT_SYSTEM_OBJECT,
        ctypes.byref(address),
        0,
        None,
        ctypes.byref(size),
        ctypes.byref(device_id),
    )
    if status != 0 or device_id.value == 0:
        return None, ""

    name_address = _PropertyAddress(_K_DEVICE_NAME_SELECTOR, _K_SCOPE_GLOBAL, _K_ELEMENT_MAIN)
    cf_string = ctypes.c_void_p(0)
    size = ctypes.c_uint32(ctypes.sizeof(cf_string))
    status = get_data(
        device_id.value,
        ctypes.byref(name_address),
        0,
        None,
        ctypes.byref(size),
        ctypes.byref(cf_string),
    )
    if status != 0 or not cf_string.value:
        return int(device_id.value), ""

    get_c_string = core_foundation.CFStringGetCString
    get_c_string.restype = ctypes.c_bool
    get_c_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    release = core_foundation.CFRelease
    release.argtypes = [ctypes.c_void_p]

    buffer = ctypes.create_string_buffer(256)
    name = ""
    try:
        if get_c_string(cf_string, buffer, len(buffer), _K_CF_STRING_ENCODING_UTF8):
            name = buffer.value.decode("utf-8", errors="replace")
    finally:
        release(cf_string)
    return int(device_id.value), name


class DefaultInputWatcher:
    """Polls the system default input device and fires on_change(name) on switches."""

    def __init__(self, on_change: Callable[[str], None], interval_s: float = 2.0):
        self.on_change = on_change
        self.interval_s = max(0.5, float(interval_s))
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_device_id: Optional[int] = None

    def start(self):
        if sys.platform != "darwin" or self._thread is not None:
            return
        self._last_device_id, _ = get_default_input_device()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="bloviate-device-watcher"
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=self.interval_s + 1.0)

    def _poll_loop(self):
        while not self._stop.wait(self.interval_s):
            try:
                device_id, name = get_default_input_device()
            except Exception:
                continue
            if device_id is None:
                continue
            if self._last_device_id is not None and device_id != self._last_device_id:
                self._last_device_id = device_id
                try:
                    self.on_change(name)
                except Exception as exc:
                    print(f"[Audio] Device-change handler error: {exc}")
            else:
                self._last_device_id = device_id
