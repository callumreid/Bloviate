"""
Deepgram live streaming helper for low-latency transcription.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Optional, Callable, List, Dict, Any

import numpy as np
import websocket


_WS_CLOSE_CODE_NAMES = {
    1000: "normal_closure",
    1001: "going_away",
    1002: "protocol_error",
    1003: "unsupported_data",
    1005: "no_status_received",
    1006: "abnormal_closure",
    1008: "policy_violation",
    1009: "message_too_big",
    1011: "internal_error",
    1012: "service_restart",
    1013: "try_again_later",
    1014: "bad_gateway",
}


class DeepgramLiveSession:
    """
    Manages a single Deepgram live transcription session.
    """

    def __init__(
        self,
        api_key: str,
        url: str,
        *,
        finalize_wait_s: float = 0.6,
        connect_timeout_s: float = 2.0,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.api_key = api_key
        self.url = url
        self.finalize_wait_s = finalize_wait_s
        self.connect_timeout_s = connect_timeout_s
        self.log = log or (lambda *_: None)

        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._sender_thread: Optional[threading.Thread] = None

        self._send_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._stop_event = threading.Event()
        self._connected = threading.Event()
        self._closed = threading.Event()

        self._final_parts: List[str] = []
        self._partial: str = ""
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._error_type: Optional[str] = None  # categorized: server_close, connect_timeout, send_error, network_error
        self._close_code: Optional[int] = None
        self._finalize_sent = False
        self._got_final_after_finalize = threading.Event()

    def _on_open(self, _ws):
        self._connected.set()

    def _on_message(self, _ws, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        channel = data.get("channel")
        if not channel:
            return

        alternatives = channel.get("alternatives") or []
        if not alternatives:
            return

        transcript = (alternatives[0].get("transcript") or "").strip()
        if not transcript:
            return

        is_final = bool(data.get("is_final") or data.get("speech_final"))

        with self._lock:
            if is_final:
                self._final_parts.append(transcript)
                if self._finalize_sent:
                    self._got_final_after_finalize.set()
            else:
                self._partial = transcript

    def _on_error(self, _ws, error):
        err_str = str(error)
        if "opcode=8" in err_str:
            # WebSocket close frame received — server dropped the connection
            self._error = "server_close"
            self._error_type = "server_close"
            if not self._finalize_sent:
                self.log("[Deepgram] Server closed connection mid-stream (possible rate limit or session timeout)")
            # else: expected close after finalize, don't log as error
        elif isinstance(error, ConnectionRefusedError):
            self._error = err_str
            self._error_type = "network_error"
            self.log(f"[Deepgram] Connection refused: {error}")
        elif isinstance(error, TimeoutError):
            self._error = err_str
            self._error_type = "network_error"
            self.log(f"[Deepgram] Connection timed out: {error}")
        else:
            self._error = err_str
            self._error_type = "network_error"
            self.log(f"[Deepgram] WebSocket error: {error}")
        self._got_final_after_finalize.set()

    def _on_close(self, _ws, status_code, message):
        self._closed.set()
        self._close_code = status_code
        self._got_final_after_finalize.set()
        if status_code and status_code != 1000:
            code_name = _WS_CLOSE_CODE_NAMES.get(status_code, "unknown")
            self.log(f"[Deepgram] Connection closed: {status_code}/{code_name} {message or ''}")
        elif not self._finalize_sent and not self._error:
            # Server closed without us asking and no prior error logged
            self.log("[Deepgram] Server closed connection unexpectedly (no close code)")

    def start(self) -> bool:
        headers = [f"Authorization: Token {self.api_key}"]

        self._ws = websocket.WebSocketApp(
            self.url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 15, "ping_timeout": 5},
            daemon=True,
        )
        self._ws_thread.start()

        if not self._connected.wait(self.connect_timeout_s):
            self._error = "connect_timeout"
            self.log("[Deepgram] Connection timeout")
            return False

        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
        )
        self._sender_thread.start()
        return True

    def _sender_loop(self):
        while not self._stop_event.is_set():
            try:
                payload = self._send_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if payload is None:
                break

            try:
                if self._ws:
                    self._ws.send(payload, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as exc:  # pragma: no cover - network error handling
                self._error = str(exc)
                self._error_type = "send_error"
                self.log(f"[Deepgram] Send error: {exc}")
                break

    def send_audio(self, audio: np.ndarray):
        if self._stop_event.is_set():
            return
        if not self._connected.is_set():
            return

        audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
        self._send_queue.put(audio_int16.tobytes())

    def finalize(self):
        if self._ws and self._connected.is_set():
            try:
                self._ws.send(json.dumps({"type": "Finalize"}))
            except Exception as exc:  # pragma: no cover - network error handling
                self._error = str(exc)
                self.log(f"[Deepgram] Finalize error: {exc}")

    def finish(self) -> Optional[str]:
        self._finalize_sent = True
        self.finalize()

        # Stop sending audio.
        self._stop_event.set()
        self._send_queue.put(None)

        if self._sender_thread:
            self._sender_thread.join(timeout=1.0)

        # Wait for Deepgram to flush the final transcript, but return
        # early if we already received it (or on error/close).
        self._got_final_after_finalize.wait(timeout=self.finalize_wait_s)

        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

        if self._ws_thread:
            self._ws_thread.join(timeout=1.0)

        return self.get_text()

    def get_text(self) -> Optional[str]:
        with self._lock:
            if self._final_parts:
                text = " ".join(self._final_parts)
            else:
                text = self._partial

        text = " ".join(text.split()).strip()
        return text or None

    def get_interim_text(self) -> Optional[str]:
        """Return the best-effort interim text (final parts + current partial)."""
        with self._lock:
            parts = list(self._final_parts)
            if self._partial:
                parts.append(self._partial)

        text = " ".join(parts)
        text = " ".join(text.split()).strip()
        return text or None

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def error_type(self) -> Optional[str]:
        return self._error_type

    @property
    def close_code(self) -> Optional[int]:
        return self._close_code
