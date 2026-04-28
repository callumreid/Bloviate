"""
Local SQLite transcript history.

History is intentionally text-only by default: raw audio is never stored here.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from app_paths import history_db_path


SCHEMA_VERSION = 1
WORD_RE = re.compile(r"\b[\w'-]+\b")


@dataclass(frozen=True)
class TranscriptRecord:
    id: int
    created_at: str
    mode: str
    text: str
    original_text: str
    post_processing_mode: str
    provider: str
    voice_score: Optional[float]
    duration_s: Optional[float]
    audio_device: str
    target_app: str
    target_window: str
    output_action: str


class HistoryStore:
    """Persist and query transcript history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or history_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'dictation',
                    text TEXT NOT NULL,
                    original_text TEXT NOT NULL DEFAULT '',
                    post_processing_mode TEXT NOT NULL DEFAULT 'verbatim',
                    provider TEXT NOT NULL DEFAULT '',
                    voice_score REAL,
                    duration_s REAL,
                    audio_device TEXT NOT NULL DEFAULT '',
                    target_app TEXT NOT NULL DEFAULT '',
                    target_window TEXT NOT NULL DEFAULT '',
                    output_action TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def add_transcript(
        self,
        *,
        text: str,
        original_text: str = "",
        mode: str = "dictation",
        post_processing_mode: str = "verbatim",
        provider: str = "",
        voice_score: Optional[float] = None,
        duration_s: Optional[float] = None,
        audio_device: str = "",
        target_app: str = "",
        target_window: str = "",
        output_action: str = "",
        created_at: Optional[str] = None,
    ) -> int:
        timestamp = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO transcripts (
                    created_at, mode, text, original_text, post_processing_mode,
                    provider, voice_score, duration_s, audio_device, target_app,
                    target_window, output_action
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    mode,
                    text,
                    original_text or text,
                    post_processing_mode,
                    provider,
                    voice_score,
                    duration_s,
                    audio_device,
                    target_app,
                    target_window,
                    output_action,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, *, limit: int = 100, query: str = "") -> list[TranscriptRecord]:
        limit = max(1, min(int(limit), 1000))
        query = str(query or "").strip()
        sql = "SELECT * FROM transcripts"
        params: list[object] = []
        if query:
            sql += (
                " WHERE text LIKE ? OR original_text LIKE ? OR target_app LIKE ? "
                "OR target_window LIKE ? OR provider LIKE ?"
            )
            needle = f"%{query}%"
            params.extend([needle, needle, needle, needle, needle])
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._record_from_row(row) for row in rows]

    def delete(self, record_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM transcripts WHERE id = ?", (int(record_id),))
            return cursor.rowcount > 0

    def clear(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM transcripts")
            return cursor.rowcount

    def insights(self, *, today: Optional[date] = None) -> dict:
        """Return aggregate usage metrics for the Settings insights view."""
        today = today or datetime.now().date()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    created_at, mode, text, original_text, post_processing_mode,
                    provider, duration_s, target_app, output_action
                FROM transcripts
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()

        total_words = 0
        total_duration_s = 0.0
        total_transcripts = len(rows)
        changed_outputs = 0
        changed_words = 0
        app_usage: dict[str, dict[str, int]] = {}
        provider_usage: dict[str, int] = {}
        mode_usage: dict[str, int] = {}
        output_usage: dict[str, int] = {}
        day_usage: dict[str, dict[str, int]] = {}

        for row in rows:
            text = str(row["text"] or "")
            original_text = str(row["original_text"] or "")
            words = self._word_count(text)
            total_words += words

            try:
                duration = float(row["duration_s"] or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            if duration > 0:
                total_duration_s += duration

            if original_text.strip() and original_text.strip() != text.strip():
                changed_outputs += 1
                changed_words += max(words, self._word_count(original_text))

            target_app = str(row["target_app"] or "Unknown app").strip() or "Unknown app"
            app_entry = app_usage.setdefault(target_app, {"words": 0, "transcripts": 0})
            app_entry["words"] += words
            app_entry["transcripts"] += 1

            provider = str(row["provider"] or "unknown").strip() or "unknown"
            provider_usage[provider] = provider_usage.get(provider, 0) + words
            mode = str(row["mode"] or "dictation").strip() or "dictation"
            mode_usage[mode] = mode_usage.get(mode, 0) + words
            output = str(row["output_action"] or "unknown").strip() or "unknown"
            output_usage[output] = output_usage.get(output, 0) + words

            day = self._local_date(row["created_at"]).isoformat()
            day_entry = day_usage.setdefault(day, {"words": 0, "transcripts": 0})
            day_entry["words"] += words
            day_entry["transcripts"] += 1

        wpm = int(round(total_words / (total_duration_s / 60.0))) if total_duration_s > 0 else 0
        app_rows = [
            {"name": name, **values}
            for name, values in sorted(
                app_usage.items(),
                key=lambda item: (-item[1]["words"], item[0].lower()),
            )
        ]
        days = [
            {
                "date": (today - timedelta(days=offset)).isoformat(),
                "words": day_usage.get((today - timedelta(days=offset)).isoformat(), {}).get("words", 0),
                "transcripts": day_usage.get((today - timedelta(days=offset)).isoformat(), {}).get("transcripts", 0),
            }
            for offset in range(83, -1, -1)
        ]

        return {
            "total_words": total_words,
            "total_transcripts": total_transcripts,
            "total_duration_s": total_duration_s,
            "words_per_minute": wpm,
            "changed_outputs": changed_outputs,
            "changed_words": changed_words,
            "apps_used": len([name for name in app_usage if name != "Unknown app"]),
            "app_usage": app_rows,
            "provider_usage": provider_usage,
            "mode_usage": mode_usage,
            "output_usage": output_usage,
            "daily_usage": days,
            "current_streak_days": self._current_streak(day_usage, today),
            "longest_streak_days": self._longest_streak(day_usage),
        }

    def export_csv(self, path: Path, records: Optional[Iterable[TranscriptRecord]] = None) -> Path:
        rows = list(records) if records is not None else self.recent(limit=1000)
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "created_at",
                    "mode",
                    "provider",
                    "text",
                    "original_text",
                    "post_processing_mode",
                    "voice_score",
                    "duration_s",
                    "audio_device",
                    "target_app",
                    "target_window",
                    "output_action",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.id,
                        row.created_at,
                        row.mode,
                        row.provider,
                        row.text,
                        row.original_text,
                        row.post_processing_mode,
                        row.voice_score,
                        row.duration_s,
                        row.audio_device,
                        row.target_app,
                        row.target_window,
                        row.output_action,
                    ]
                )
        return path

    @staticmethod
    def _word_count(text: str) -> int:
        return len(WORD_RE.findall(str(text or "")))

    @staticmethod
    def _local_date(created_at: str) -> date:
        raw = str(created_at or "").strip()
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone()
            return parsed.date()
        except Exception:
            try:
                return date.fromisoformat(raw[:10])
            except Exception:
                return datetime.now().date()

    @staticmethod
    def _current_streak(day_usage: dict[str, dict[str, int]], today: date) -> int:
        streak = 0
        current = today
        while day_usage.get(current.isoformat(), {}).get("words", 0) > 0:
            streak += 1
            current -= timedelta(days=1)
        return streak

    @staticmethod
    def _longest_streak(day_usage: dict[str, dict[str, int]]) -> int:
        dates = sorted(
            date.fromisoformat(day)
            for day, usage in day_usage.items()
            if usage.get("words", 0) > 0
        )
        if not dates:
            return 0
        longest = current = 1
        previous = dates[0]
        for active_day in dates[1:]:
            if active_day == previous + timedelta(days=1):
                current += 1
            else:
                longest = max(longest, current)
                current = 1
            previous = active_day
        return max(longest, current)

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> TranscriptRecord:
        return TranscriptRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            mode=str(row["mode"]),
            text=str(row["text"]),
            original_text=str(row["original_text"]),
            post_processing_mode=str(row["post_processing_mode"]),
            provider=str(row["provider"]),
            voice_score=row["voice_score"],
            duration_s=row["duration_s"],
            audio_device=str(row["audio_device"]),
            target_app=str(row["target_app"]),
            target_window=str(row["target_window"]),
            output_action=str(row["output_action"]),
        )
