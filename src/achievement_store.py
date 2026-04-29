"""
SQLite storage for achievement progress, unlocks, and compact AI tags.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from app_paths import history_db_path
from achievement_catalog import AchievementDefinition


@dataclass(frozen=True)
class AchievementUnlock:
    id: str
    unlocked_at: str
    progress_value: float
    evidence: dict


class AchievementStore:
    """Persist achievement state in the same SQLite file as transcript history."""

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
                CREATE TABLE IF NOT EXISTS achievement_unlocks (
                    id TEXT PRIMARY KEY,
                    unlocked_at TEXT NOT NULL,
                    progress_value REAL NOT NULL DEFAULT 0,
                    evidence_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS achievement_progress (
                    id TEXT PRIMARY KEY,
                    current_value REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcript_ai_analysis (
                    transcript_id INTEGER PRIMARY KEY,
                    model TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '{}',
                    analyzed_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def get_metadata(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def set_metadata(self, key: str, value: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                (str(key), str(value)),
            )

    def transcript_rows(self) -> list[sqlite3.Row]:
        reset_at = self.get_metadata("achievements.reset_at", "")
        where = ""
        params: tuple[object, ...] = ()
        if reset_at:
            where = "WHERE created_at > ?"
            params = (reset_at,)
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT
                    id, created_at, mode, text, original_text, post_processing_mode,
                    provider, voice_score, duration_s, audio_device, target_app,
                    target_window, output_action
                FROM transcripts
                {where}
                ORDER BY created_at ASC, id ASC
                """.format(where=where),
                params,
            ).fetchall()

    def update_progress(self, values: dict[str, float]):
        timestamp = self._now()
        with self._connect() as conn:
            for achievement_id, value in values.items():
                conn.execute(
                    """
                    INSERT INTO achievement_progress(id, current_value, updated_at)
                    VALUES(?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        current_value = excluded.current_value,
                        updated_at = excluded.updated_at
                    """,
                    (achievement_id, float(value or 0), timestamp),
                )

    def progress_map(self) -> dict[str, float]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, current_value FROM achievement_progress").fetchall()
        return {str(row["id"]): float(row["current_value"] or 0) for row in rows}

    def unlock_map(self) -> dict[str, AchievementUnlock]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, unlocked_at, progress_value, evidence_json FROM achievement_unlocks"
            ).fetchall()
        return {str(row["id"]): self._unlock_from_row(row) for row in rows}

    def recent_unlocks(self, limit: int = 8) -> list[AchievementUnlock]:
        limit = max(1, min(int(limit or 8), 100))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, unlocked_at, progress_value, evidence_json
                FROM achievement_unlocks
                ORDER BY unlocked_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._unlock_from_row(row) for row in rows]

    def apply_evaluation(
        self,
        catalog: Iterable[AchievementDefinition],
        metric_values: dict[str, float],
        *,
        suppress_unlocks: bool = False,
    ) -> list[AchievementUnlock]:
        current_values: dict[str, float] = {}
        for definition in catalog:
            current_values[definition.id] = float(metric_values.get(definition.metric, 0) or 0)
        self.update_progress(current_values)

        if suppress_unlocks:
            return []

        timestamp = self._now()
        new_unlocks: list[AchievementUnlock] = []
        with self._connect() as conn:
            for definition in catalog:
                value = current_values.get(definition.id, 0.0)
                if value < definition.threshold:
                    continue
                evidence = {
                    "metric": definition.metric,
                    "value": value,
                    "threshold": definition.threshold,
                }
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO achievement_unlocks(
                        id, unlocked_at, progress_value, evidence_json
                    )
                    VALUES(?, ?, ?, ?)
                    """,
                    (definition.id, timestamp, value, json.dumps(evidence, sort_keys=True)),
                )
                if cursor.rowcount:
                    new_unlocks.append(
                        AchievementUnlock(
                            id=definition.id,
                            unlocked_at=timestamp,
                            progress_value=value,
                            evidence=evidence,
                        )
                    )
        return new_unlocks

    def reset(self) -> int:
        timestamp = self._now()
        with self._connect() as conn:
            unlock_count = conn.execute("SELECT COUNT(*) FROM achievement_unlocks").fetchone()[0]
            conn.execute("DELETE FROM achievement_unlocks")
            conn.execute("DELETE FROM achievement_progress")
            conn.execute("DELETE FROM transcript_ai_analysis")
            conn.execute("DELETE FROM metadata WHERE key LIKE 'achievements.%'")
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                ("achievements.reset_at", timestamp),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                ("achievements.backfill_completed", "1"),
            )
        return int(unlock_count or 0)

    def transcripts_without_ai_analysis(self, limit: int = 100) -> list[sqlite3.Row]:
        limit = max(1, min(int(limit or 100), 500))
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT id, created_at, text, target_app
                FROM transcripts
                WHERE text IS NOT NULL
                  AND TRIM(text) != ''
                  AND id NOT IN (SELECT transcript_id FROM transcript_ai_analysis)
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def upsert_ai_analysis(self, transcript_id: int, model: str, tags: dict):
        clean_tags = {
            str(key): bool(value)
            for key, value in (tags or {}).items()
            if str(key).strip()
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transcript_ai_analysis(transcript_id, model, tags_json, analyzed_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(transcript_id) DO UPDATE SET
                    model = excluded.model,
                    tags_json = excluded.tags_json,
                    analyzed_at = excluded.analyzed_at
                """,
                (int(transcript_id), str(model), json.dumps(clean_tags, sort_keys=True), self._now()),
            )

    def ai_tag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._connect() as conn:
            rows = conn.execute("SELECT tags_json FROM transcript_ai_analysis").fetchall()
        for row in rows:
            try:
                tags = json.loads(str(row["tags_json"] or "{}"))
            except json.JSONDecodeError:
                continue
            if isinstance(tags, dict):
                for key, value in tags.items():
                    if value:
                        metric = f"ai_tag_{key}"
                        counts[metric] = counts.get(metric, 0) + 1
        return counts

    @staticmethod
    def _unlock_from_row(row: sqlite3.Row) -> AchievementUnlock:
        try:
            evidence = json.loads(str(row["evidence_json"] or "{}"))
        except json.JSONDecodeError:
            evidence = {}
        if not isinstance(evidence, dict):
            evidence = {}
        return AchievementUnlock(
            id=str(row["id"]),
            unlocked_at=str(row["unlocked_at"]),
            progress_value=float(row["progress_value"] or 0),
            evidence=evidence,
        )
