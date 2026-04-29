"""
Achievement evaluation and optional AI tag analysis.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from achievement_badge import AchievementBadgeRenderer
from achievement_catalog import ACHIEVEMENTS, AchievementDefinition
from achievement_store import AchievementStore
from secret_store import SecretStore


WORD_RE = re.compile(r"\b[\w'-]+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
FILENAME_RE = re.compile(r"\b[\w-]+\.(?:py|js|ts|tsx|jsx|md|txt|yaml|yml|json|csv|sql|rb|go|rs|html|css)\b", re.IGNORECASE)
CODE_TOKEN_RE = re.compile(r"(--?\w+|[\w-]+/[\w./-]+|[A-Za-z_][A-Za-z0-9_]*\(\)|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)")
LIST_MARKER_RE = re.compile(r"(^|\n)\s*(?:[-*]|\d+[.)])\s+")

APP_CATEGORIES = {
    "browser": ["safari", "chrome", "arc", "firefox", "brave", "edge"],
    "chat": ["messages", "slack", "teams", "discord", "whatsapp", "signal"],
    "docs": ["docs", "word", "pages", "notion", "obsidian", "bear", "notes"],
    "editor": ["code", "cursor", "xcode", "pycharm", "terminal", "iterm", "textedit"],
    "email": ["mail", "outlook", "gmail", "superhuman", "hey"],
}

AI_TAGS = {
    "intent_message",
    "genre_todo",
    "genre_meeting",
    "genre_email",
    "genre_plan",
    "genre_bug_report",
    "genre_docs",
    "genre_idea",
    "tone_polished",
    "tone_urgent",
    "tone_funny",
    "content_code",
    "content_action_items",
    "content_question",
    "content_decision",
    "content_summary",
}


class AchievementService:
    """Evaluate, summarize, and analyze Bloviate achievements."""

    def __init__(
        self,
        config: dict,
        *,
        secret_store: Optional[SecretStore] = None,
        store: Optional[AchievementStore] = None,
        renderer: Optional[AchievementBadgeRenderer] = None,
    ):
        self.config = config
        self.secret_store = secret_store or SecretStore()
        self.store = store or AchievementStore()
        self.renderer = renderer or AchievementBadgeRenderer()
        self.catalog = list(ACHIEVEMENTS)
        self.definitions_by_id = {definition.id: definition for definition in self.catalog}

    def enabled(self) -> bool:
        return bool(self.config.get("achievements", {}).get("enabled", True))

    def evaluate(
        self,
        *,
        dictionary_payload: Optional[dict] = None,
        voice_profile_status: Optional[dict] = None,
        suppress_unlocks: bool = False,
    ) -> list[dict]:
        if not self.enabled():
            return []
        metrics = self.metric_values(
            dictionary_payload=dictionary_payload,
            voice_profile_status=voice_profile_status,
        )
        new_unlocks = self.store.apply_evaluation(
            self.catalog,
            metrics,
            suppress_unlocks=suppress_unlocks,
        )
        return [
            self._achievement_dict(self.definitions_by_id[unlock.id], metrics, unlock=unlock)
            for unlock in new_unlocks
            if unlock.id in self.definitions_by_id
        ]

    def summary(
        self,
        *,
        dictionary_payload: Optional[dict] = None,
        voice_profile_status: Optional[dict] = None,
        query: str = "",
        status_filter: str = "all",
    ) -> dict:
        metrics = self.metric_values(
            dictionary_payload=dictionary_payload,
            voice_profile_status=voice_profile_status,
        )
        unlocks = self.store.unlock_map()
        query = str(query or "").strip().lower()
        status_filter = str(status_filter or "all").strip().lower()

        achievements = []
        for definition in self.catalog:
            unlock = unlocks.get(definition.id)
            item = self._achievement_dict(definition, metrics, unlock=unlock)
            if query and query not in " ".join(
                [
                    item["title"],
                    item["description"],
                    item["category"],
                    item["id"],
                ]
            ).lower():
                continue
            if status_filter == "unlocked" and not item["unlocked"]:
                continue
            if status_filter == "locked" and item["unlocked"]:
                continue
            if status_filter == "ai" and not item["ai_required"]:
                continue
            achievements.append(item)

        achievements.sort(
            key=lambda item: (
                not item["unlocked"],
                -float(item.get("progress_ratio", 0)),
                item["category"],
                item["title"],
            )
        )
        recent = []
        for unlock in self.store.recent_unlocks(limit=8):
            definition = self.definitions_by_id.get(unlock.id)
            if definition:
                recent.append(self._achievement_dict(definition, metrics, unlock=unlock))

        return {
            "enabled": self.enabled(),
            "ai_analysis_enabled": bool(
                self.config.get("achievements", {}).get("ai_analysis_enabled", False)
            ),
            "total": len(self.catalog),
            "unlocked": len(unlocks),
            "locked": max(0, len(self.catalog) - len(unlocks)),
            "achievements": achievements,
            "recent": recent,
        }

    def reset(self) -> tuple[bool, str]:
        count = self.store.reset()
        return True, f"Reset {count} unlocked achievement(s)."

    def set_settings(self, updates: dict) -> tuple[bool, str]:
        cfg = self.config.setdefault("achievements", {})
        for key, value in updates.items():
            cfg[str(key)] = value
        return True, "Achievement settings saved."

    def backfill_if_needed(
        self,
        *,
        dictionary_payload: Optional[dict] = None,
        voice_profile_status: Optional[dict] = None,
    ) -> list[dict]:
        cfg = self.config.get("achievements", {})
        if not self.enabled() or not bool(cfg.get("backfill_on_startup", True)):
            return []
        if self.store.get_metadata("achievements.backfill_completed") == "1":
            return []
        unlocks = self.evaluate(
            dictionary_payload=dictionary_payload,
            voice_profile_status=voice_profile_status,
            suppress_unlocks=False,
        )
        self.store.set_metadata("achievements.backfill_completed", "1")
        return unlocks

    def analyze_history(
        self,
        *,
        dictionary_payload: Optional[dict] = None,
        voice_profile_status: Optional[dict] = None,
        limit: int = 100,
    ) -> tuple[bool, str, list[dict]]:
        cfg = self.config.get("achievements", {})
        if not bool(cfg.get("ai_analysis_enabled", False)):
            return False, "AI achievement analysis is disabled.", []
        api_key = self.secret_store.get_api_key("openai", self.config)
        if not api_key:
            return False, "OpenAI key is required for AI achievement analysis.", []

        rows = self.store.transcripts_without_ai_analysis(limit=limit)
        if not rows:
            unlocks = self.evaluate(
                dictionary_payload=dictionary_payload,
                voice_profile_status=voice_profile_status,
            )
            return True, "No unanalyzed transcripts found.", unlocks

        analyzed = 0
        model = str(cfg.get("ai_analysis_model") or self.config.get("post_processing", {}).get("openai_model", "gpt-4o"))
        for row in rows:
            tags = self._classify_transcript(
                text=str(row["text"] or ""),
                target_app=str(row["target_app"] or ""),
                api_key=api_key,
                model=model,
            )
            self.store.upsert_ai_analysis(int(row["id"]), model, tags)
            analyzed += 1

        unlocks = self.evaluate(
            dictionary_payload=dictionary_payload,
            voice_profile_status=voice_profile_status,
        )
        return True, f"Analyzed {analyzed} transcript(s) for AI achievements.", unlocks

    def metric_values(
        self,
        *,
        dictionary_payload: Optional[dict] = None,
        voice_profile_status: Optional[dict] = None,
    ) -> dict[str, float]:
        rows = self.store.transcript_rows()
        metrics: dict[str, float] = defaultdict(float)
        metrics["achievement_count"] = float(len(self.store.unlock_map()))

        day_words: dict[date, int] = defaultdict(int)
        week_words: dict[tuple[int, int], int] = defaultdict(int)
        month_words: dict[tuple[int, int], int] = defaultdict(int)
        active_days: set[date] = set()
        weekend_days: set[date] = set()
        unique_hours: set[int] = set()
        apps: set[str] = set()
        target_windows: set[str] = set()
        providers: set[str] = set()
        modes: set[str] = set()
        app_categories_used: set[str] = set()
        unique_words: set[str] = set()

        today = datetime.now().date()
        dictation_words = 0
        total_words = 0
        total_duration = 0.0
        max_session_wpm = 0.0
        max_transcript_words = 0
        max_daily_tuesday_words = 0

        for row in rows:
            text = str(row["text"] or "")
            original_text = str(row["original_text"] or "")
            words = WORD_RE.findall(text)
            word_count = len(words)
            total_words += word_count
            max_transcript_words = max(max_transcript_words, word_count)
            for word in words:
                normalized = word.strip("'").lower()
                if normalized:
                    unique_words.add(normalized)
                    metrics["longest_word_length"] = max(metrics["longest_word_length"], len(normalized))
                    if normalized.isupper() and len(normalized) > 1:
                        metrics["uppercase_word_count"] += 1

            duration = self._float(row["duration_s"])
            if duration > 0:
                total_duration += duration
                wpm = word_count / (duration / 60.0)
                max_session_wpm = max(max_session_wpm, wpm)
                if wpm >= 120:
                    metrics["fast_transcripts_120"] += 1
                if word_count >= 250 and wpm >= 160:
                    metrics["rare_big_fast"] = 1
            if duration and duration <= 10:
                metrics["short_transcripts_under_10s"] += 1

            if original_text.strip() and original_text.strip() != text.strip():
                metrics["changed_outputs"] += 1
                metrics["changed_words"] += max(word_count, len(WORD_RE.findall(original_text)))
            else:
                metrics["unchanged_outputs"] += 1

            provider = str(row["provider"] or "unknown").strip().lower() or "unknown"
            providers.add(provider)
            if "deepgram" in provider:
                metrics["deepgram_words"] += word_count
            if "openai" in provider:
                metrics["openai_words"] += word_count

            mode = str(row["mode"] or "dictation").strip().lower() or "dictation"
            modes.add(mode)
            if mode == "dictation":
                dictation_words += word_count
            if mode == "dictation_rejected":
                metrics["rejected_count"] += 1
            if mode == "command":
                metrics["command_count"] += 1
                lowered = text.lower()
                if "window" in lowered or "screen" in lowered:
                    metrics["window_command_count"] += 1
                if lowered.startswith(("open ", "launch ", "start ")):
                    metrics["app_command_count"] += 1

            voice_score = row["voice_score"]
            if mode in {"dictation", "dictation_rejected"}:
                if voice_score is None:
                    metrics["talk_words"] += word_count
                else:
                    metrics["whisper_words"] += word_count

            app = str(row["target_app"] or "").strip()
            if app and app.lower() != "unknown app":
                apps.add(app)
                for category, needles in APP_CATEGORIES.items():
                    if any(needle in app.lower() for needle in needles):
                        metrics[f"app_category_{category}_words"] += word_count
                        app_categories_used.add(category)
            target_window = str(row["target_window"] or "").strip()
            if target_window:
                target_windows.add(f"{app}:{target_window}")

            local_dt = self._parse_datetime(str(row["created_at"] or ""))
            day = local_dt.date()
            day_words[day] += word_count
            active_days.add(day)
            if day.weekday() >= 5 and word_count > 0:
                weekend_days.add(day)
            if day.weekday() == 1:
                max_daily_tuesday_words = max(max_daily_tuesday_words, day_words[day])
            iso = day.isocalendar()
            week_words[(iso.year, iso.week)] += word_count
            month_words[(day.year, day.month)] += word_count
            unique_hours.add(local_dt.hour)
            if local_dt.hour < 9:
                metrics["morning_transcripts"] += 1
            if local_dt.hour >= 22:
                metrics["night_transcripts"] += 1

            metrics["question_count"] += text.count("?")
            metrics["exclamation_count"] += text.count("!")
            metrics["url_count"] += len(URL_RE.findall(text))
            metrics["filename_count"] += len(FILENAME_RE.findall(text))
            metrics["code_token_count"] += len(CODE_TOKEN_RE.findall(text))
            metrics["number_count"] += len(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))
            metrics["parenthetical_count"] += text.count("(") + text.count("[")
            metrics["list_marker_count"] += len(LIST_MARKER_RE.findall(text))
            if row["post_processing_mode"] and str(row["post_processing_mode"]).lower() != "verbatim":
                if provider == "openai" or original_text.strip() != text.strip():
                    metrics["openai_cleanup_outputs"] += 1

        metrics["total_words"] = float(total_words)
        metrics["total_transcripts"] = float(len(rows))
        metrics["total_duration_minutes"] = total_duration / 60.0
        metrics["words_per_minute"] = total_words / (total_duration / 60.0) if total_duration > 0 else 0.0
        metrics["max_session_wpm"] = max_session_wpm
        metrics["max_transcript_words"] = float(max_transcript_words)
        metrics["average_transcript_words"] = total_words / len(rows) if rows else 0.0
        metrics["dictation_words"] = float(dictation_words)
        metrics["max_daily_words"] = float(max(day_words.values(), default=0))
        metrics["max_weekly_words"] = float(max(week_words.values(), default=0))
        metrics["max_monthly_words"] = float(max(month_words.values(), default=0))
        metrics["current_streak_days"] = float(self._current_streak(day_words, today))
        metrics["longest_streak_days"] = float(self._longest_streak(day_words))
        metrics["active_days"] = float(len([day for day, words in day_words.items() if words > 0]))
        metrics["weekend_active_days"] = float(len(weekend_days))
        metrics["unique_hours"] = float(len(unique_hours))
        metrics["apps_used"] = float(len(apps))
        metrics["target_windows_used"] = float(len(target_windows))
        metrics["provider_count"] = float(len([provider for provider in providers if provider != "unknown"]))
        metrics["unique_words"] = float(len(unique_words))

        dictionary_payload = dictionary_payload or {}
        terms = dictionary_payload.get("preferred_terms", []) or []
        corrections = dictionary_payload.get("corrections", []) or []
        metrics["dictionary_terms"] = float(len(terms))
        metrics["dictionary_corrections"] = float(len(corrections))
        metrics["dictionary_total_entries"] = float(len(terms) + len(corrections))

        voice_profile_status = voice_profile_status or {}
        metrics["voice_samples"] = float(voice_profile_status.get("enrolled_samples", 0) or 0)

        for key, value in self.store.ai_tag_counts().items():
            metrics[key] = float(value)

        ai_unlocks = sum(
            1
            for unlock_id in self.store.unlock_map()
            if self.definitions_by_id.get(unlock_id, None)
            and self.definitions_by_id[unlock_id].ai_required
        )
        metrics["rare_ai_sampler"] = 1 if ai_unlocks >= 8 else 0
        metrics["rare_provider_trio"] = 1 if metrics["provider_count"] >= 3 else 0
        metrics["rare_all_modes"] = 1 if {"dictation", "command", "dictation_rejected"}.issubset(modes) else 0
        metrics["rare_dictionary_power"] = 1 if metrics["dictionary_total_entries"] >= 25 and metrics["changed_outputs"] >= 25 else 0
        metrics["rare_app_chameleon"] = 1 if metrics["apps_used"] >= 10 and len(app_categories_used) >= 5 else 0
        metrics["rare_weekend_warrior"] = 1 if metrics["weekend_active_days"] >= 10 else 0
        metrics["rare_night_owl"] = 1 if metrics["night_transcripts"] >= 25 else 0
        metrics["rare_morning_lark"] = 1 if metrics["morning_transcripts"] >= 25 else 0
        metrics["rare_recovery_artist"] = 1 if metrics["rejected_count"] >= 10 else 0
        metrics["rare_window_mage"] = 1 if metrics["window_command_count"] >= 25 else 0
        metrics["rare_long_word_fast"] = 1 if metrics["longest_word_length"] >= 24 and metrics["max_session_wpm"] >= 140 else 0
        metrics["rare_clean_machine"] = 1 if metrics["unchanged_outputs"] >= 100 else 0
        metrics["rare_exact_tuesday"] = 1 if max_daily_tuesday_words >= 500 else 0
        metrics["rare_settings_archaeologist"] = 1 if (
            metrics["total_transcripts"] >= 1
            and metrics["dictionary_total_entries"] >= 1
            and metrics["changed_outputs"] >= 1
            and metrics["command_count"] >= 1
        ) else 0

        return dict(metrics)

    def _achievement_dict(
        self,
        definition: AchievementDefinition,
        metrics: dict[str, float],
        *,
        unlock=None,
    ) -> dict:
        value = float(metrics.get(definition.metric, 0) or 0)
        progress_ratio = min(1.0, value / definition.threshold) if definition.threshold > 0 else 0.0
        unlocked = unlock is not None
        badge_path = self.renderer.render_badge(definition, unlocked=unlocked)
        return {
            "id": definition.id,
            "title": definition.title,
            "description": definition.description,
            "category": definition.category,
            "metric": definition.metric,
            "current_value": value,
            "threshold": definition.threshold,
            "unit": definition.unit,
            "progress_ratio": progress_ratio,
            "progress_label": self._progress_label(value, definition.threshold, definition.unit),
            "unlocked": unlocked,
            "unlocked_at": getattr(unlock, "unlocked_at", ""),
            "rarity": definition.rarity,
            "badge_path": str(badge_path),
            "ai_required": definition.ai_required,
            "hidden": definition.hidden and not unlocked,
        }

    def _classify_transcript(self, *, text: str, target_app: str, api_key: str, model: str) -> dict[str, bool]:
        allowed_tags = sorted(AI_TAGS)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify a dictated transcript for local achievement tags. "
                        "Return compact JSON only: {\"tags\":[...]} using only allowed tags."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Allowed tags: {', '.join(allowed_tags)}\n"
                        f"Target app: {target_app or 'unknown'}\n"
                        "Transcript:\n"
                        f"{text[:4000]}"
                    ),
                },
            ],
            "temperature": 0,
        }
        base_url = str(self.config.get("openai", {}).get("base_url", "https://api.openai.com/v1")).rstrip("/")
        timeout = float(self.config.get("achievements", {}).get("ai_analysis_timeout_s", 12))
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = str(body["choices"][0]["message"]["content"]).strip()
            if content.startswith("```"):
                content = content.strip("`").strip()
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            parsed = json.loads(content)
            tags = parsed.get("tags", [])
        except (urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
            print(f"[Achievements] AI analysis failed: {exc}")
            tags = []
        except Exception as exc:
            print(f"[Achievements] AI analysis unavailable: {exc}")
            tags = []
        return {tag: tag in set(tags) for tag in allowed_tags}

    @staticmethod
    def _progress_label(value: float, threshold: float, unit: str) -> str:
        def fmt(number: float) -> str:
            if abs(number - int(number)) < 0.001:
                return f"{int(number):,}"
            return f"{number:,.1f}"

        suffix = f" {unit}" if unit else ""
        return f"{fmt(value)} / {fmt(threshold)}{suffix}"

    @staticmethod
    def _float(value) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _parse_datetime(raw: str) -> datetime:
        text = str(raw or "").strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.astimezone()
            return parsed
        except Exception:
            return datetime.now()

    @staticmethod
    def _current_streak(day_words: dict[date, int], today: date) -> int:
        streak = 0
        current = today
        while day_words.get(current, 0) > 0:
            streak += 1
            current -= timedelta(days=1)
        return streak

    @staticmethod
    def _longest_streak(day_words: dict[date, int]) -> int:
        days = sorted(day for day, words in day_words.items() if words > 0)
        if not days:
            return 0
        longest = current = 1
        previous = days[0]
        for day in days[1:]:
            if day == previous + timedelta(days=1):
                current += 1
            else:
                longest = max(longest, current)
                current = 1
            previous = day
        return max(longest, current)
