"""
Local SQLite transcript history.

History is intentionally text-only by default: raw audio is never stored here.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from app_paths import history_db_path


SCHEMA_VERSION = 1


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
