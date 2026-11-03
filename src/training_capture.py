"""Opt-in capture of real dictations for fine-tuning data.

When enabled, every accepted dictation (one that produced text and was not a
voice command) is written to disk as lossless audio plus a JSONL index entry,
so the user can later build a fine-tuning dataset from his real audio. A
correction hotkey lets him fix a pasted dictation in the target app and save
the corrected text as a gold label. Everything stays local.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np

from app_paths import app_support_dir


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


class TrainingCapture:
    """Append-only audio + metadata store for fine-tuning data."""

    def __init__(self, config: dict):
        self.enabled = bool(config.get("training_capture", {}).get("enabled", False))
        cfg = config.get("training_capture", {})
        self.max_hours = float(cfg.get("max_hours", 20) or 0)
        self.correction_window_min = float(cfg.get("correction_window_min", 30) or 30)
        self.sample_rate = int(config.get("audio", {}).get("sample_rate", 16000))
        self._lock = threading.Lock()
        if not self.enabled:
            self.root = None
            self.captures_path = None
            self.corrections_path = None
            return
        self.root = app_support_dir() / "training_capture"
        (self.root / "audio").mkdir(parents=True, exist_ok=True)
        self.captures_path = self.root / "captures.jsonl"
        self.corrections_path = self.root / "corrections.jsonl"
        self.captures_path.touch(exist_ok=True)
        self.corrections_path.touch(exist_ok=True)

    # ------------------------------------------------------------------ audio
    def _write_audio(self, audio: np.ndarray) -> str:
        """Write mono float32 audio as 16-bit PCM WAV; return the relative path."""
        capture_id = uuid.uuid4().hex
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rel_dir = Path("audio") / day
        abs_dir = self.root / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        rel_path = rel_dir / f"{capture_id}.wav"
        abs_path = self.root / rel_path
        import wave

        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(abs_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return str(rel_path)

    # ------------------------------------------------------------------ index
    def _append_capture(self, entry: dict) -> None:
        with self._lock:
            with self.captures_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _total_duration_s(self) -> float:
        total = 0.0
        with self.captures_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    total += float(json.loads(line).get("duration_s", 0.0) or 0.0)
                except (json.JSONDecodeError, ValueError):
                    continue
        return total

    def _prune(self) -> None:
        if self.max_hours <= 0:
            return
        max_s = self.max_hours * 3600.0
        lines: list[dict] = []
        with self.captures_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        total = sum(float(e.get("duration_s", 0.0) or 0.0) for e in lines)
        if total <= max_s:
            return
        # Drop oldest until under budget.
        while lines and total > max_s:
            entry = lines.pop(0)
            total -= float(entry.get("duration_s", 0.0) or 0.0)
            rel = entry.get("file")
            if rel:
                try:
                    (self.root / rel).unlink(missing_ok=True)
                except OSError:
                    pass
        with self.captures_path.open("w", encoding="utf-8") as fh:
            for e in lines:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ public
    def record(
        self,
        *,
        raw_audio: np.ndarray,
        history_id: Optional[int],
        audio_device: str,
        raw_rms: float,
        provider: str,
        transcript: str,
        output_text: str,
        stream_text: Optional[str],
        target_app: str,
        outcome: str,
    ) -> None:
        """Persist one accepted dictation. Never raises; logs a warning on failure."""
        if not self.enabled or self.root is None:
            return
        try:
            duration_s = len(raw_audio) / self.sample_rate
            rel_path = self._write_audio(np.asarray(raw_audio, dtype=np.float32))
            entry = {
                "capture_id": Path(rel_path).stem,
                "created_at": _utc_iso(),
                "history_id": history_id,
                "file": rel_path,
                "duration_s": duration_s,
                "sample_rate": self.sample_rate,
                "audio_device": audio_device,
                "raw_rms": float(raw_rms),
                "provider": provider,
                "transcript": transcript,
                "output_text": output_text,
                "stream_text": stream_text,
                "target_app": target_app,
                "outcome": outcome,
            }
            self._append_capture(entry)
            self._prune()
        except Exception as exc:  # never break the paste path
            print(f"[TrainingCapture] Could not save capture: {exc}")

    # ------------------------------------------------------------- corrections
    def _recent_captures(self) -> list[dict]:
        if not self.enabled or self.captures_path is None:
            return []
        cutoff = time.time() - self.correction_window_min * 60.0
        recent: list[dict] = []
        with self.captures_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                created = entry.get("created_at", "")
                try:
                    ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
                if ts >= cutoff:
                    recent.append(entry)
        return recent

    def match_correction(self, selected_text: str) -> Optional[tuple[dict, float]]:
        """Return (capture_entry, similarity) for the best match, or None."""
        if not self.enabled or not selected_text.strip():
            return None
        selected_norm = _normalize(selected_text)
        if not selected_norm:
            return None
        best: Optional[tuple[dict, float]] = None
        for entry in self._recent_captures():
            output = entry.get("output_text") or ""
            if not output:
                continue
            output_norm = _normalize(output)
            if not output_norm:
                continue
            if len(selected_norm) < 0.6 * len(output_norm):
                continue
            ratio = SequenceMatcher(None, output_norm, selected_norm).ratio()
            if ratio < 0.5:
                continue
            if best is None or ratio > best[1]:
                best = (entry, ratio)
        return best

    def save_correction(self, capture_id: str, corrected_text: str, similarity: float) -> None:
        if not self.enabled or self.corrections_path is None:
            return
        try:
            with self.corrections_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "capture_id": capture_id,
                            "corrected_text": corrected_text,
                            "similarity": float(similarity),
                            "created_at": _utc_iso(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:
            print(f"[TrainingCapture] Could not save correction: {exc}")
