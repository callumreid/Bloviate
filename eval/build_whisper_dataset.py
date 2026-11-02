#!/usr/bin/env python3
"""Build a personalized whispered-speech training + evaluation corpus for Bloviate.

Mines 500 utterances across four categories:
  1. dictionary_carrier: every term & phrase in personal_dictionary.yaml placed in
     multiple natural dictation carrier sentences (train + held-out eval split).
  2. syntax_dictation: spoken CLI/git/coding syntax where tts_text spells out
     flags/slashes/hyphens ("git checkout dash b feat slash...") and target_text
     is the exact formatted string ("git checkout -b feat/...").
  3. history_dictation: real sentences mined from Bloviate's local history.sqlite.
  4. obsidian_prose: clean sentences mined from Bronniopollis daily/work notes.

Outputs:
  eval/finetune_data/corpus.json  (500 items: 450 train, 50 eval)
"""

from __future__ import annotations

import json
import os
import random
import re
import sqlite3
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "eval" / "finetune_data"
LIVE_DICT_PATH = Path.home() / "Library" / "Application Support" / "Bloviate" / "personal_dictionary.yaml"
HISTORY_DB_PATH = Path.home() / "Library" / "Application Support" / "Bloviate" / "history.sqlite"
VAULT_PATH = Path.home() / "personal" / "Bronniopollis"

TRAIN_CARRIERS = [
    "Can you check on {term} and let me know if that looks right?",
    "I was just talking to the team about {term} during standup.",
    "Let's make sure {term} is properly configured before we merge this.",
    "We saw an issue with {term} in the logs earlier this morning.",
    "Please update the documentation for {term} and open a pull request.",
]

EVAL_CARRIERS = [
    "Hey, quick question about {term}, is that live in production right now?",
    "Make sure {term} passes all the checks before we ship it today.",
]


def load_dictionary_terms() -> list[str]:
    if not LIVE_DICT_PATH.exists():
        return []
    data = yaml.safe_load(LIVE_DICT_PATH.read_text(encoding="utf-8")) or {}
    terms: list[str] = []
    seen: set[str] = set()
    for term in data.get("preferred_terms") or []:
        t = str(term).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)
    for entry in data.get("corrections") or []:
        phrase = str(entry.get("phrase") or "").strip()
        if not phrase or phrase in {"feat-", "fix-", "fucking", "still getting", "persona in"}:
            continue
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            terms.append(phrase)
    return terms


def build_syntax_pairs() -> list[dict]:
    """Pairs where spoken words (dash, slash, dot) map to literal CLI syntax."""
    branches = [
        ("feat slash whisper eval", "feat/whisper-eval"),
        ("fix slash prod simulation", "fix/prod-simulation"),
        ("feat slash add voice metrics", "feat/add-voice-metrics"),
        ("fix slash rds replica peg", "fix/rds-replica-peg"),
        ("feat slash local llm shadow", "feat/local-llm-shadow"),
        ("chore slash bump version", "chore/bump-version"),
        ("fix slash livekit redaction", "fix/livekit-redaction"),
        ("feat slash priority queues", "feat/priority-queues"),
        ("fix slash stuck pipeline", "fix/stuck-pipeline"),
        ("feat slash gradium voices", "feat/gradium-voices"),
        ("fix slash deepgram streaming", "fix/deepgram-streaming"),
        ("feat slash cowcode speed", "feat/cowcode-speed"),
    ]
    pairs: list[dict] = []
    for i, (spoken_branch, written_branch) in enumerate(branches):
        is_eval = i >= len(branches) - 2
        pairs.append({
            "category": "syntax_dictation",
            "split": "eval" if is_eval else "train",
            "tts_text": f"Now run git checkout dash b {spoken_branch} and push it to origin.",
            "target_text": f"Now run git checkout -b {written_branch} and push it to origin.",
        })
        pairs.append({
            "category": "syntax_dictation",
            "split": "eval" if is_eval else "train",
            "tts_text": f"Let's create branch {spoken_branch} from origin slash main.",
            "target_text": f"Let's create branch {written_branch} from origin/main.",
        })

    cli_commands = [
        (
            "Run pnpm install dash dash frozen dash lockfile in the frontend directory.",
            "Run pnpm install --frozen-lockfile in the frontend directory.",
        ),
        (
            "Check git diff origin slash main dot dot dot HEAD before opening the PR.",
            "Check git diff origin/main...HEAD before opening the PR.",
        ),
        (
            "Run just lint dash changed dash dash changed dash from origin slash main.",
            "Run just lint-changed --changed-from origin/main.",
        ),
        (
            "We need to test Coval and Bloviate with Deepgram nova dash three and OpenRouter.",
            "We need to test Coval and Bloviate with Deepgram nova-3 and OpenRouter.",
        ),
        (
            "Run pytest dash q on the metric directory tests in the backend repo.",
            "Run pytest -q on the metric directory tests in the backend repo.",
        ),
        (
            "Use gh pr create without dash dash draft so the pull request is ready for review.",
            "Use gh pr create without --draft so the pull request is ready for review.",
        ),
        (
            "Check the prod dash simulator dash high dash queue metrics in CloudWatch.",
            "Check the prod-simulator-high-queue metrics in CloudWatch.",
        ),
        (
            "Check the prod dash simulation dash metrics dash medium dash queue backlog.",
            "Check the prod-simulation-metrics-medium-queue backlog.",
        ),
        (
            "Let's run pnpm vitest run on the changed components in Coval frontend.",
            "Let's run pnpm vitest run on the changed components in Coval frontend.",
        ),
        (
            "Compare Qwen three ASR flash against Microsoft MAI transcribe one point five.",
            "Compare Qwen3-ASR-Flash against microsoft/mai-transcribe-1.5.",
        ),
    ]
    for i, (tts_text, target_text) in enumerate(cli_commands):
        pairs.append({
            "category": "syntax_dictation",
            "split": "eval" if i >= len(cli_commands) - 2 else "train",
            "tts_text": tts_text,
            "target_text": target_text,
        })
    return pairs


def clean_sentence_for_tts(s: str) -> str | None:
    s = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"\b(uh|um|er)\b[,.]?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or len(s) < 35 or len(s) > 165:
        return None
    words = s.split()
    if len(words) < 6 or len(words) > 28:
        return None
    if any(ch in s for ch in ["http://", "https://", "{", "}", "|", "<", ">", "#", "==", "→", "⚠", "⛔", "✅"]):
        return None
    if not s[0].isupper():
        return None
    if s[-1] not in ".?!":
        s = s + "."
    return s


def mine_history_sentences(max_count: int = 120) -> list[str]:
    if not HISTORY_DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT text FROM transcripts WHERE text IS NOT NULL AND length(trim(text)) > 25 ORDER BY id DESC"
    ).fetchall()
    conn.close()

    sentences: list[str] = []
    seen: set[str] = set()
    for (raw_text,) in rows:
        for part in re.split(r"(?<=[.?!])\s+", str(raw_text)):
            cleaned = clean_sentence_for_tts(part)
            if not cleaned:
                continue
            # Fix known historical STT mishearings so we never train on old errors
            cleaned = re.sub(r"\bSophia\b", "Sofia", cleaned)
            cleaned = re.sub(r"\bSelfieVal\b", "self-eval", cleaned)
            cleaned = re.sub(r"\bCove\s*Al\b|\bCovey\s*Al\b|\bKoval\b", "Coval", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\bBloviat\b|\bBlow\s*V\s*eight\b", "Bloviate", cleaned, flags=re.IGNORECASE)
            if cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                sentences.append(cleaned)
                if len(sentences) >= max_count:
                    return sentences
    return sentences


def mine_obsidian_sentences(max_count: int = 120) -> list[str]:
    if not VAULT_PATH.exists():
        return []
    sentences: list[str] = []
    seen: set[str] = set()
    md_files = sorted(VAULT_PATH.glob("2026/**/*.md"), reverse=True)
    for md in md_files[:80]:
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # Strip YAML frontmatter
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
        for line in text.splitlines():
            line = line.strip().lstrip("-*•>0123456789. ")
            for part in re.split(r"(?<=[.?!])\s+", line):
                cleaned = clean_sentence_for_tts(part)
                if cleaned and cleaned.lower() not in seen:
                    seen.add(cleaned.lower())
                    sentences.append(cleaned)
                    if len(sentences) >= max_count:
                        return sentences
    return sentences


def main() -> None:
    rng = random.Random(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    terms = load_dictionary_terms()
    print(f"Loaded {len(terms)} terms from {LIVE_DICT_PATH}")

    corpus: list[dict] = []

    # 1. Dictionary carriers (~300 items: every term in train, plus 25 key terms in eval)
    priority_eval_terms = [
        "Coval", "Bloviate", "Callum", "Kappi", "Kobi Hudson", "Pipecat", "LiveKit",
        "Vapi", "Deepgram", "ElevenLabs", "Telnyx", "Supabase", "Vercel", "Obsidian",
        "Strava", "Datadog", "ServiceNow", "Hippocratic AI", "Dana Dzik", "Kevin Shen",
        "Ben Dyer", "Cole Amick", "Jake Levi", "Sausalito", "San Francisco",
    ]
    for i, term in enumerate(terms):
        carrier = TRAIN_CARRIERS[i % len(TRAIN_CARRIERS)]
        text = carrier.format(term=term)
        corpus.append({
            "category": "dictionary_carrier",
            "split": "train",
            "term": term,
            "tts_text": text,
            "target_text": text,
        })
    # Add second carrier for the first 90 terms so high-priority terms get multiple contexts
    for i, term in enumerate(terms[:90]):
        carrier = TRAIN_CARRIERS[(i + 2) % len(TRAIN_CARRIERS)]
        text = carrier.format(term=term)
        corpus.append({
            "category": "dictionary_carrier",
            "split": "train",
            "term": term,
            "tts_text": text,
            "target_text": text,
        })

    for i, term in enumerate(priority_eval_terms):
        carrier = EVAL_CARRIERS[i % len(EVAL_CARRIERS)]
        text = carrier.format(term=term)
        corpus.append({
            "category": "dictionary_carrier",
            "split": "eval",
            "term": term,
            "tts_text": text,
            "target_text": text,
        })

    # 2. Spoken syntax pairs (~34 items)
    syntax_items = build_syntax_pairs()
    corpus.extend(syntax_items)

    # 3. Real Bloviate history sentences (~100 items)
    history_sents = mine_history_sentences(110)
    for i, sent in enumerate(history_sents):
        corpus.append({
            "category": "history_dictation",
            "split": "eval" if i < 12 else "train",
            "tts_text": sent,
            "target_text": sent,
        })

    # 4. Obsidian prose sentences (fill up to 500 total)
    needed = max(0, 500 - len(corpus))
    obsidian_sents = mine_obsidian_sentences(needed + 20)
    for i, sent in enumerate(obsidian_sents[:needed]):
        corpus.append({
            "category": "obsidian_prose",
            "split": "eval" if i < 7 else "train",
            "tts_text": sent,
            "target_text": sent,
        })

    # Assign stable IDs
    for idx, item in enumerate(corpus):
        item["id"] = f"clip_{idx:04d}"

    train_items = [x for x in corpus if x["split"] == "train"]
    eval_items = [x for x in corpus if x["split"] == "eval"]
    print(f"Total corpus: {len(corpus)} (train={len(train_items)}, eval={len(eval_items)})")
    by_cat: dict[str, int] = {}
    for x in corpus:
        by_cat[x["category"]] = by_cat.get(x["category"], 0) + 1
    print("By category:", by_cat)

    out_path = OUT_DIR / "corpus.json"
    out_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
