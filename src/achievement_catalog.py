"""
Achievement catalog for Bloviate.

The catalog is generated from fixed threshold ladders so v1 can ship hundreds of
achievements without maintaining hundreds of near-identical records by hand.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AchievementDefinition:
    id: str
    title: str
    description: str
    category: str
    metric: str
    threshold: float
    unit: str
    tier: int
    rarity: str
    badge_family: str
    badge_motif: str
    badge_seed: int
    ai_required: bool = False
    hidden: bool = False


WORD_THRESHOLDS = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
TRANSCRIPT_THRESHOLDS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
STREAK_THRESHOLDS = [2, 3, 5, 7, 14, 21, 30, 60, 100, 365]
WPM_THRESHOLDS = [80, 100, 120, 140, 160, 180, 200, 220]

SMALL_WORD_THRESHOLDS = [50, 100, 250, 500, 1000, 2500, 5000, 10000]
COUNT_5 = [1, 3, 5, 10, 25]
COUNT_6 = [1, 3, 5, 10, 25, 50]
COUNT_7 = [1, 3, 5, 10, 25, 50, 100]
COUNT_8 = [1, 3, 5, 10, 25, 50, 100, 250]
COUNT_10 = [1, 3, 5, 10, 25, 50, 100, 250, 500, 1000]
LONG_TEXT_THRESHOLDS = [10, 25, 50, 100, 150, 250, 500, 1000, 1500, 2500]
VOCAB_THRESHOLDS = [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000]
LONG_WORD_THRESHOLDS = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
MINUTE_THRESHOLDS = [1, 5, 10, 25, 50, 100, 250, 500]
AI_THRESHOLDS = [1, 5, 15, 50]

TIER_NAMES = [
    "Pocket Edition",
    "Desk Edition",
    "Reasonable Person",
    "Mildly Concerning",
    "Office Folklore",
    "Calendar Menace",
    "Keyboard Lease Canceled",
    "Accidental Novelist",
    "Monologue Industrial Complex",
    "Public Utility",
    "National Treasure",
    "Myth Department",
    "Weather System",
]

BADGE_FAMILIES = [
    "trophy",
    "medal",
    "ribbon",
    "waveform",
    "microphone",
    "dictionary",
    "window",
    "compass",
    "clock",
    "lightning",
    "stack",
    "spark",
]

MOTIFS = [
    "cowbell",
    "chair",
    "typebar",
    "spark",
    "wave",
    "book",
    "window",
    "cursor",
    "clock",
    "bolt",
    "stack",
    "orbit",
]


def _slug(value: object) -> str:
    text = str(value).lower().replace(".", "_")
    out = []
    for char in text:
        if char.isalnum():
            out.append(char)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")


def _stable_seed(value: str) -> int:
    return zlib.crc32(value.encode("utf-8")) % 100000


def _format_threshold(value: float, unit: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric >= 1000000 and numeric % 1000000 == 0:
        label = f"{int(numeric / 1000000)}M"
    elif numeric >= 1000 and numeric % 1000 == 0:
        label = f"{int(numeric / 1000)}k"
    elif numeric >= 1000:
        label = f"{numeric / 1000:g}k"
    else:
        label = f"{numeric:g}"
    return f"{label} {unit}".strip()


def _rarity(index: int, total: int) -> str:
    if total <= 1:
        return "common"
    ratio = index / max(1, total - 1)
    if ratio >= 0.92:
        return "mythic"
    if ratio >= 0.76:
        return "legendary"
    if ratio >= 0.56:
        return "epic"
    if ratio >= 0.32:
        return "rare"
    if ratio >= 0.12:
        return "uncommon"
    return "common"


def _tiered(
    *,
    category: str,
    prefix: str,
    metric: str,
    thresholds: Iterable[float],
    unit: str,
    title: str,
    description: str,
    badge_family: str,
    badge_motif: str,
    ai_required: bool = False,
) -> list[AchievementDefinition]:
    values = list(thresholds)
    definitions: list[AchievementDefinition] = []
    for index, threshold in enumerate(values):
        tier_name = TIER_NAMES[min(index, len(TIER_NAMES) - 1)]
        threshold_label = _format_threshold(threshold, unit)
        definitions.append(
            AchievementDefinition(
                id=f"{prefix}_{index + 1:02d}_{_slug(threshold_label)}",
                title=f"{title}: {tier_name}",
                description=description.format(threshold=threshold_label),
                category=category,
                metric=metric,
                threshold=float(threshold),
                unit=unit,
                tier=index + 1,
                rarity=_rarity(index, len(values)),
                badge_family=badge_family,
                badge_motif=badge_motif,
                badge_seed=_stable_seed(f"{prefix}:{metric}:{threshold}"),
                ai_required=ai_required,
            )
        )
    return definitions


def build_achievement_catalog() -> list[AchievementDefinition]:
    catalog: list[AchievementDefinition] = []

    # Word volume and sessions: 84
    catalog += _tiered(category="Word volume", prefix="words_total", metric="total_words", thresholds=WORD_THRESHOLDS, unit="words", title="Keyboard Lease Canceled", description="Dictate {threshold} total.", badge_family="trophy", badge_motif="typebar")
    catalog += _tiered(category="Word volume", prefix="transcripts_total", metric="total_transcripts", thresholds=TRANSCRIPT_THRESHOLDS, unit="clips", title="One More Thing", description="Save {threshold} transcript clips.", badge_family="stack", badge_motif="stack")
    catalog += _tiered(category="Word volume", prefix="longest_clip_words", metric="max_transcript_words", thresholds=LONG_TEXT_THRESHOLDS, unit="words", title="A Small Essay Happened", description="Dictate a single clip with {threshold}.", badge_family="ribbon", badge_motif="book")
    catalog += _tiered(category="Word volume", prefix="daily_words", metric="max_daily_words", thresholds=LONG_TEXT_THRESHOLDS, unit="words", title="Daily Monologue Permit", description="Dictate {threshold} in one day.", badge_family="clock", badge_motif="clock")
    catalog += _tiered(category="Word volume", prefix="weekly_words", metric="max_weekly_words", thresholds=[250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000], unit="words", title="Weeklong Typing Strike", description="Dictate {threshold} in one week.", badge_family="calendar", badge_motif="clock")
    catalog += _tiered(category="Word volume", prefix="monthly_words", metric="max_monthly_words", thresholds=[500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000], unit="words", title="Accidental Novelist", description="Dictate {threshold} in one month.", badge_family="trophy", badge_motif="book")
    catalog += _tiered(category="Word volume", prefix="dictation_words", metric="dictation_words", thresholds=[50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000], unit="words", title="Actual Dictation Happened", description="Dictate {threshold} accepted words.", badge_family="microphone", badge_motif="wave")
    catalog += _tiered(category="Word volume", prefix="avg_clip_words", metric="average_transcript_words", thresholds=[5, 10, 15, 25, 40, 60, 90, 120, 180, 240], unit="avg words", title="Paragraph Gravity", description="Average {threshold} per saved clip.", badge_family="stack", badge_motif="typebar")

    # Speed and flow: 54
    catalog += _tiered(category="Speed and flow", prefix="all_time_wpm", metric="words_per_minute", thresholds=WPM_THRESHOLDS, unit="WPM", title="Mouth Turbo", description="Reach an all-time average of {threshold}.", badge_family="lightning", badge_motif="bolt")
    catalog += _tiered(category="Speed and flow", prefix="session_wpm", metric="max_session_wpm", thresholds=WPM_THRESHOLDS, unit="WPM", title="Syllable Speedrun", description="Hit {threshold} in one clip.", badge_family="lightning", badge_motif="wave")
    catalog += _tiered(category="Speed and flow", prefix="fast_clip_count", metric="fast_transcripts_120", thresholds=COUNT_8, unit="fast clips", title="Cruise Control", description="Record {threshold} clips over 120 WPM.", badge_family="waveform", badge_motif="bolt")
    catalog += _tiered(category="Speed and flow", prefix="rewritten_count", metric="changed_outputs", thresholds=COUNT_8, unit="rewrites", title="No Filler, No Witnesses", description="Have cleanup rewrite {threshold}.", badge_family="spark", badge_motif="spark")
    catalog += _tiered(category="Speed and flow", prefix="clean_clip_count", metric="unchanged_outputs", thresholds=COUNT_8, unit="clean clips", title="Clean On Arrival", description="Save {threshold} clips that needed no rewrite.", badge_family="medal", badge_motif="chair")
    catalog += _tiered(category="Speed and flow", prefix="dictation_minutes", metric="total_duration_minutes", thresholds=MINUTE_THRESHOLDS, unit="minutes", title="Air Time", description="Record {threshold} of dictation time.", badge_family="clock", badge_motif="clock")
    catalog += _tiered(category="Speed and flow", prefix="short_clip_count", metric="short_transcripts_under_10s", thresholds=COUNT_6, unit="short clips", title="Tiny Broadcast", description="Save {threshold} clips under ten seconds.", badge_family="microphone", badge_motif="spark")

    # Streaks and time: 52
    catalog += _tiered(category="Streaks and time", prefix="current_streak", metric="current_streak_days", thresholds=STREAK_THRESHOLDS, unit="days", title="Calendar Negotiator", description="Build a current {threshold} streak.", badge_family="clock", badge_motif="clock")
    catalog += _tiered(category="Streaks and time", prefix="longest_streak", metric="longest_streak_days", thresholds=STREAK_THRESHOLDS, unit="days", title="Nine Lives of Momentum", description="Reach a longest streak of {threshold}.", badge_family="medal", badge_motif="clock")
    catalog += _tiered(category="Streaks and time", prefix="active_days", metric="active_days", thresholds=STREAK_THRESHOLDS, unit="active days", title="Attendance Optional", description="Dictate on {threshold}.", badge_family="stack", badge_motif="calendar")
    catalog += _tiered(category="Streaks and time", prefix="weekend_days", metric="weekend_active_days", thresholds=COUNT_6, unit="weekend days", title="Saturday Had Notes", description="Dictate on {threshold}.", badge_family="ribbon", badge_motif="clock")
    catalog += _tiered(category="Streaks and time", prefix="morning_clips", metric="morning_transcripts", thresholds=COUNT_6, unit="morning clips", title="Before Coffee Counsel", description="Save {threshold}.", badge_family="clock", badge_motif="spark")
    catalog += _tiered(category="Streaks and time", prefix="night_clips", metric="night_transcripts", thresholds=COUNT_6, unit="night clips", title="Midnight Dictator", description="Save {threshold}.", badge_family="clock", badge_motif="moon")
    catalog += _tiered(category="Streaks and time", prefix="unique_hours", metric="unique_hours", thresholds=[2, 4, 8, 12], unit="hours", title="Clock Roulette", description="Dictate during {threshold} different hours of the day.", badge_family="compass", badge_motif="clock")

    # Apps and workflow: 76
    catalog += _tiered(category="Apps and workflow", prefix="apps_used", metric="apps_used", thresholds=COUNT_10, unit="apps", title="App Sommelier", description="Dictate into {threshold}.", badge_family="window", badge_motif="window")
    for cat, title in [("browser", "Browser Bard"), ("chat", "Chatty Professional"), ("docs", "Document Whisperer"), ("editor", "Editor Grease"), ("email", "Inbox Monologue")]:
        catalog += _tiered(category="Apps and workflow", prefix=f"{cat}_words", metric=f"app_category_{cat}_words", thresholds=SMALL_WORD_THRESHOLDS, unit="words", title=title, description="Dictate {threshold} in this app family.", badge_family="window", badge_motif=cat)
    catalog += _tiered(category="Apps and workflow", prefix="command_count", metric="command_count", thresholds=COUNT_8, unit="commands", title="Remote Control Personality", description="Run {threshold}.", badge_family="compass", badge_motif="cursor")
    catalog += _tiered(category="Apps and workflow", prefix="window_commands", metric="window_command_count", thresholds=COUNT_8, unit="window commands", title="Window Tetris", description="Run {threshold}.", badge_family="window", badge_motif="window")
    catalog += _tiered(category="Apps and workflow", prefix="app_commands", metric="app_command_count", thresholds=COUNT_5, unit="app commands", title="Launch Sequence", description="Run {threshold}.", badge_family="compass", badge_motif="cursor")
    catalog += _tiered(category="Apps and workflow", prefix="target_windows", metric="target_windows_used", thresholds=COUNT_5, unit="windows", title="Window Tourist", description="Dictate into {threshold}.", badge_family="window", badge_motif="window")

    # Dictionary and fixes: 58
    catalog += _tiered(category="Dictionary and fixes", prefix="dictionary_terms", metric="dictionary_terms", thresholds=COUNT_10, unit="terms", title="Personal Canon", description="Add {threshold} preferred dictionary terms.", badge_family="dictionary", badge_motif="book")
    catalog += _tiered(category="Dictionary and fixes", prefix="dictionary_rules", metric="dictionary_corrections", thresholds=COUNT_10, unit="rules", title="Correctional Facility", description="Add {threshold} replacement rules.", badge_family="dictionary", badge_motif="book")
    catalog += _tiered(category="Dictionary and fixes", prefix="dictionary_total", metric="dictionary_total_entries", thresholds=COUNT_10, unit="entries", title="The Dictionary Has a Lawyer", description="Maintain {threshold}.", badge_family="dictionary", badge_motif="book")
    catalog += _tiered(category="Dictionary and fixes", prefix="cleanup_rewrites", metric="changed_outputs", thresholds=COUNT_10, unit="rewrites", title="Typo Tax Refund", description="Rewrite {threshold} clips.", badge_family="spark", badge_motif="spark")
    catalog += _tiered(category="Dictionary and fixes", prefix="changed_words", metric="changed_words", thresholds=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000], unit="words", title="Polish Department", description="Rewrite {threshold}.", badge_family="spark", badge_motif="typebar")
    catalog += _tiered(category="Dictionary and fixes", prefix="openai_cleanup", metric="openai_cleanup_outputs", thresholds=COUNT_8, unit="AI cleanups", title="Robot Copyeditor", description="Use OpenAI cleanup {threshold}.", badge_family="spark", badge_motif="orbit")

    # Voice, provider, mode: 44
    catalog += _tiered(category="Voice and providers", prefix="whisper_words", metric="whisper_words", thresholds=SMALL_WORD_THRESHOLDS, unit="words", title="Whisper License", description="Dictate {threshold} with voice verification.", badge_family="microphone", badge_motif="wave")
    catalog += _tiered(category="Voice and providers", prefix="talk_words", metric="talk_words", thresholds=SMALL_WORD_THRESHOLDS, unit="words", title="Outside Voice Permit", description="Dictate {threshold} in talk mode.", badge_family="microphone", badge_motif="wave")
    catalog += _tiered(category="Voice and providers", prefix="rejected_recoveries", metric="rejected_count", thresholds=COUNT_8, unit="recoveries", title="Recovered From The Void", description="Save {threshold} rejected clips to history.", badge_family="medal", badge_motif="spark")
    catalog += _tiered(category="Voice and providers", prefix="provider_count", metric="provider_count", thresholds=COUNT_5, unit="providers", title="Provider Tour", description="Use {threshold}.", badge_family="compass", badge_motif="orbit")
    catalog += _tiered(category="Voice and providers", prefix="deepgram_words", metric="deepgram_words", thresholds=[50, 250, 1000, 5000, 25000], unit="words", title="Deepgram Diver", description="Transcribe {threshold} with Deepgram.", badge_family="waveform", badge_motif="wave")
    catalog += _tiered(category="Voice and providers", prefix="openai_words", metric="openai_words", thresholds=[50, 250, 1000, 5000, 25000], unit="words", title="OpenAI Orator", description="Transcribe {threshold} with OpenAI.", badge_family="spark", badge_motif="orbit")
    catalog += _tiered(category="Voice and providers", prefix="voice_samples", metric="voice_samples", thresholds=[1, 3, 8, 16, 32], unit="samples", title="Profile Seasoned", description="Record {threshold} voice profile samples.", badge_family="microphone", badge_motif="chair")

    # Vocabulary and text shape: 76
    catalog += _tiered(category="Vocabulary", prefix="longest_word", metric="longest_word_length", thresholds=LONG_WORD_THRESHOLDS, unit="letters", title="Sesquipedalian Incident", description="Use a word with {threshold}.", badge_family="ribbon", badge_motif="typebar")
    catalog += _tiered(category="Vocabulary", prefix="unique_words", metric="unique_words", thresholds=VOCAB_THRESHOLDS, unit="unique words", title="Thesaurus With Shoes", description="Use {threshold}.", badge_family="dictionary", badge_motif="book")
    catalog += _tiered(category="Vocabulary", prefix="question_marks", metric="question_count", thresholds=COUNT_7, unit="questions", title="Questionable Choices", description="Dictate {threshold}.", badge_family="compass", badge_motif="cursor")
    catalog += _tiered(category="Vocabulary", prefix="exclamation_marks", metric="exclamation_count", thresholds=COUNT_7, unit="exclamations", title="Punctuation Situation", description="Dictate {threshold}.", badge_family="spark", badge_motif="spark")
    catalog += _tiered(category="Vocabulary", prefix="urls", metric="url_count", thresholds=COUNT_6, unit="URLs", title="Link Wrangler", description="Dictate {threshold}.", badge_family="compass", badge_motif="cursor")
    catalog += _tiered(category="Vocabulary", prefix="filenames", metric="filename_count", thresholds=COUNT_6, unit="filenames", title="Filename Sommelier", description="Dictate {threshold}.", badge_family="stack", badge_motif="typebar")
    catalog += _tiered(category="Vocabulary", prefix="code_tokens", metric="code_token_count", thresholds=COUNT_8, unit="code tokens", title="Slash Command Energy", description="Dictate {threshold}.", badge_family="lightning", badge_motif="cursor")
    catalog += _tiered(category="Vocabulary", prefix="numbers", metric="number_count", thresholds=COUNT_6, unit="numbers", title="Spreadsheet Soliloquy", description="Dictate {threshold}.", badge_family="stack", badge_motif="stack")
    catalog += _tiered(category="Vocabulary", prefix="uppercase_words", metric="uppercase_word_count", thresholds=COUNT_6, unit="uppercase words", title="Acronym Weather", description="Dictate {threshold}.", badge_family="ribbon", badge_motif="typebar")
    catalog += _tiered(category="Vocabulary", prefix="parentheticals", metric="parenthetical_count", thresholds=COUNT_5, unit="parentheticals", title="Parenthetical Citizen", description="Dictate {threshold}.", badge_family="ribbon", badge_motif="typebar")
    catalog += _tiered(category="Vocabulary", prefix="list_markers", metric="list_marker_count", thresholds=COUNT_5, unit="list markers", title="Bullet Point Diplomat", description="Dictate {threshold}.", badge_family="stack", badge_motif="stack")

    # AI-assisted content: 64
    ai_tags = [
        ("intent_message", "Message In A Bottle", "AI tags {threshold} message-like clips."),
        ("genre_todo", "Todo Rodeo", "AI tags {threshold} todo clips."),
        ("genre_meeting", "Meeting That Could Have Been Text", "AI tags {threshold} meeting clips."),
        ("genre_email", "Email Wearing A Hat", "AI tags {threshold} email clips."),
        ("genre_plan", "Plan With Side Quests", "AI tags {threshold} planning clips."),
        ("genre_bug_report", "Bug Report With Feelings", "AI tags {threshold} bug reports."),
        ("genre_docs", "Documentation Costume", "AI tags {threshold} documentation clips."),
        ("genre_idea", "Idea Popcorn", "AI tags {threshold} idea clips."),
        ("tone_polished", "Executive Summary Costume", "AI tags {threshold} polished clips."),
        ("tone_urgent", "Gentle Panic Button", "AI tags {threshold} urgent clips."),
        ("tone_funny", "Joke With A Clipboard", "AI tags {threshold} funny clips."),
        ("content_code", "Code Adjacent Behavior", "AI tags {threshold} code clips."),
        ("content_action_items", "Action Item Goblet", "AI tags {threshold} action-item clips."),
        ("content_question", "Question Factory", "AI tags {threshold} question-heavy clips."),
        ("content_decision", "Decision Receipt", "AI tags {threshold} decision clips."),
        ("content_summary", "Summary Soup", "AI tags {threshold} summary clips."),
    ]
    for tag, title, description in ai_tags:
        catalog += _tiered(category="AI-assisted", prefix=f"ai_{tag}", metric=f"ai_tag_{tag}", thresholds=AI_THRESHOLDS, unit="clips", title=title, description=description, badge_family="spark", badge_motif="orbit", ai_required=True)

    # Rare and secret: 20
    rare_specs = [
        ("rare_provider_trio", "Three Provider Hat Trick", "Use at least three transcription providers.", "rare_provider_trio"),
        ("rare_all_modes", "Every Button Got A Turn", "Use dictation, command mode, and rejected-history recovery.", "rare_all_modes"),
        ("rare_big_fast", "Large And In Charge", "Dictate a 250-word clip at 160 WPM or faster.", "rare_big_fast"),
        ("rare_dictionary_power", "Dictionary Supreme Court", "Maintain 25 dictionary entries and 25 cleanup rewrites.", "rare_dictionary_power"),
        ("rare_app_chameleon", "App Chameleon", "Use 10 apps and at least five app categories.", "rare_app_chameleon"),
        ("rare_weekend_warrior", "Weekend Department Open", "Dictate on 10 weekend days.", "rare_weekend_warrior"),
        ("rare_night_owl", "Night Shift Narrator", "Save 25 clips after 10 PM.", "rare_night_owl"),
        ("rare_morning_lark", "Sunrise Speechwriter", "Save 25 clips before 9 AM.", "rare_morning_lark"),
        ("rare_recovery_artist", "Near Miss Archivist", "Recover 10 rejected voice clips.", "rare_recovery_artist"),
        ("rare_window_mage", "Window Wizardry", "Run 25 window commands.", "rare_window_mage"),
        ("rare_long_word_fast", "Dictionary On Roller Skates", "Use a 24-letter word and hit 140 WPM.", "rare_long_word_fast"),
        ("rare_clean_machine", "No Cleanup Needed Apparently", "Save 100 clips that needed no rewrite.", "rare_clean_machine"),
        ("rare_ai_sampler", "AI Taste Test", "Unlock at least eight AI-assisted achievements.", "rare_ai_sampler"),
        ("rare_exact_tuesday", "Extremely Specific Tuesday", "Dictate 500 words on a Tuesday.", "rare_exact_tuesday"),
        ("rare_settings_archaeologist", "Settings Archaeologist", "Use history, dictionary, cleanup, and commands.", "rare_settings_archaeologist"),
        ("rare_achievement_25", "Achievement For Achieving Achievements", "Unlock 25 achievements.", "achievement_count"),
        ("rare_achievement_50", "Achievement Shelf Needs Shelves", "Unlock 50 achievements.", "achievement_count"),
        ("rare_achievement_100", "Tiny Trophy Warehouse", "Unlock 100 achievements.", "achievement_count"),
        ("rare_achievement_250", "The Trophy Room Has Zoning Issues", "Unlock 250 achievements.", "achievement_count"),
        ("rare_achievement_500", "This Was Probably Too Robust", "Unlock 500 achievements.", "achievement_count"),
    ]
    achievement_thresholds = {
        "rare_achievement_25": 25,
        "rare_achievement_50": 50,
        "rare_achievement_100": 100,
        "rare_achievement_250": 250,
        "rare_achievement_500": 500,
    }
    for index, (achievement_id, title, description, metric) in enumerate(rare_specs):
        threshold = float(achievement_thresholds.get(achievement_id, 1))
        catalog.append(
            AchievementDefinition(
                id=achievement_id,
                title=title,
                description=description,
                category="Rare and secret",
                metric=metric,
                threshold=threshold,
                unit="",
                tier=index + 1,
                rarity="secret" if index < 15 else "mythic",
                badge_family=BADGE_FAMILIES[index % len(BADGE_FAMILIES)],
                badge_motif=MOTIFS[index % len(MOTIFS)],
                badge_seed=70000 + index,
                hidden=True,
            )
        )

    if len(catalog) != 528:
        raise RuntimeError(f"Achievement catalog must contain 528 definitions, got {len(catalog)}")
    if len({item.id for item in catalog}) != len(catalog):
        raise RuntimeError("Achievement catalog contains duplicate ids")
    return catalog


ACHIEVEMENTS = build_achievement_catalog()
