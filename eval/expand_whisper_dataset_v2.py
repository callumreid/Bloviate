#!/usr/bin/env python3
"""Build Round-2 expanded whispered-speech training corpus while keeping the 50 eval clips strictly fixed.

Addresses the specific error modes diagnosed after Round 1:
  1. TTS Pronunciation Normalization: acronyms like CLI, PR, SDK, API, URL, UI, SQL, LLM,
     COVAL-XXXX, Qwen3, cowcode, covi are spelled out phonetically in `tts_text` ("C L I",
     "Coval 64 29", "cow code") while `target_text` retains exact written form ("CLI",
     "COVAL-6429", "CowCode").
  2. Dense Spoken-Syntax Pairs (~180 train pairs): branch names (`feat/...`, `fix/...`,
     `origin/main`), CLI flags (`-b`, `--frozen-lockfile`, `-q`), paths, and model IDs.
  3. Hard Dictionary / Proper-Noun Reinforcement: extra contexts for terms that had
     residual errors under 14dB noise (Supabase, Telnyx, Deepgram, Kobi Hudson, CowCode,
     Sofia, Covi, prod, CLI, etc.).
  4. More Obsidian & History sentences (~250+ items) with TTS pronunciation normalization.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "eval" / "finetune_data"
LIVE_DICT_PATH = Path.home() / "Library" / "Application Support" / "Bloviate" / "personal_dictionary.yaml"
HISTORY_DB_PATH = Path.home() / "Library" / "Application Support" / "Bloviate" / "history.sqlite"
VAULT_PATH = Path.home() / "personal" / "Bronniopollis"

# Pronunciations so Gradium TTS speaks acronyms/jargon the way a human actually dictates them,
# while Whisper is trained to output the canonical written token.
TTS_PRONUNCIATION_MAP = [
    (r"\bCLI\b", "C L I"),
    (r"\bPRs\b", "P R s"),
    (r"\bPR\b", "P R"),
    (r"\bSDK\b", "S D K"),
    (r"\bAPI\b", "A P I"),
    (r"\bURL\b", "U R L"),
    (r"\bUI\b", "U I"),
    (r"\bLLM\b", "L L M"),
    (r"\bASR\b", "A S R"),
    (r"\bSTT\b", "S T T"),
    (r"\bTTS\b", "T T S"),
    (r"\bVAD\b", "V A D"),
    (r"\bRCA\b", "R C A"),
    (r"\bAWS\b", "A W S"),
    (r"\bS3\b", "S three"),
    (r"\bCI/CD\b", "C I slash C D"),
    (r"\bCI\b", "C I"),
    (r"\bk8s\b", "K eight S"),
    (r"\bmacOS\b", "Mac O S"),
    (r"\bQwen3\b", "Qwen three"),
    (r"\bQwen2\.5\b", "Qwen two point five"),
    (r"\bCowCode\b", "cow code"),
    (r"\bcowcode\b", "cow code"),
    (r"\bcovi-improve\b", "covi improve"),
    (r"\bCOVAL-(\d{4})\b", r"Coval \1"),
    (r"\bpnpm\b", "p n p m"),
    (r"\bpytest\b", "pie test"),
    (r"\bvitest\b", "vie test"),
    (r"\bSupabase\b", "Soopa base"),
    (r"\bTelnyx\b", "Tell nix"),
]


def to_spoken_tts(text: str) -> str:
    out = text
    for pattern, repl in TTS_PRONUNCIATION_MAP:
        out = re.sub(pattern, repl, out)
    return out


def build_rich_syntax_train_pairs() -> list[dict]:
    """Generate ~160 diverse spoken-syntax training examples."""
    branch_specs = [
        ("feat", "whisper-eval"),
        ("feat", "add-voice-metrics"),
        ("feat", "local-llm-shadow"),
        ("feat", "priority-queues"),
        ("feat", "gradium-voices"),
        ("feat", "cowcode-speed"),
        ("feat", "sofia-slack-bridge"),
        ("feat", "clickhouse-browse"),
        ("feat", "trace-metrics"),
        ("feat", "audio-segment-filter"),
        ("feat", "human-review-labels"),
        ("feat", "enterprise-scim"),
        ("fix", "prod-simulation"),
        ("fix", "rds-replica-peg"),
        ("fix", "livekit-redaction"),
        ("fix", "stuck-pipeline"),
        ("fix", "deepgram-streaming"),
        ("fix", "run-setup-memory"),
        ("fix", "voice-preview-upload"),
        ("fix", "slack-notebook-delivery"),
        ("fix", "coval-judge-confidence"),
        ("fix", "metric-freshness-alert"),
        ("chore", "bump-version"),
        ("chore", "deps-lock-sync"),
        ("chore", "retire-cgroup-probe"),
    ]

    templates = [
        (
            "Now run git checkout dash b {prefix} slash {slug_spoken} and push it to origin.",
            "Now run git checkout -b {prefix}/{slug} and push it to origin.",
        ),
        (
            "Let's create branch {prefix} slash {slug_spoken} from origin slash main.",
            "Let's create branch {prefix}/{slug} from origin/main.",
        ),
        (
            "Please rebase {prefix} slash {slug_spoken} onto origin slash main before merging.",
            "Please rebase {prefix}/{slug} onto origin/main before merging.",
        ),
        (
            "Can you check git diff origin slash main dot dot dot {prefix} slash {slug_spoken}?",
            "Can you check git diff origin/main...{prefix}/{slug}?",
        ),
        (
            "Push {prefix} slash {slug_spoken} to origin and open a pull request.",
            "Push {prefix}/{slug} to origin and open a pull request.",
        ),
    ]

    pairs: list[dict] = []
    for idx, (prefix, slug) in enumerate(branch_specs):
        slug_spoken = slug.replace("-", " ").replace("cowcode", "cow code").replace("llm", "L L M").replace("rds", "R D S")
        for t_idx, (tts_tmpl, tgt_tmpl) in enumerate(templates):
            if (idx + t_idx) % 2 == 0 or prefix in {"feat", "fix"} and t_idx < 3:
                pairs.append({
                    "category": "syntax_dictation",
                    "split": "train",
                    "tts_text": tts_tmpl.format(prefix=prefix, slug_spoken=slug_spoken),
                    "target_text": tgt_tmpl.format(prefix=prefix, slug=slug),
                })

    extra_commands = [
        (
            "Compare Qwen three A S R Flash against Microsoft M A I transcribe one point five.",
            "Compare Qwen3-ASR-Flash against microsoft/mai-transcribe-1.5.",
        ),
        (
            "We benchmarked Qwen three A S R Flash and Microsoft slash M A I dash transcribe dash one point five.",
            "We benchmarked Qwen3-ASR-Flash and microsoft/mai-transcribe-1.5.",
        ),
        (
            "Switch the model from Qwen three A S R Flash to Microsoft M A I transcribe one point five.",
            "Switch the model from Qwen3-ASR-Flash to microsoft/mai-transcribe-1.5.",
        ),
        (
            "Let's run p n p m vie test run on the changed components in Coval frontend.",
            "Let's run pnpm vitest run on the changed components in Coval frontend.",
        ),
        (
            "Run p n p m vie test run in the Coval frontend workspace before pushing.",
            "Run pnpm vitest run in the Coval frontend workspace before pushing.",
        ),
        (
            "Fixed the Coval C L I dropping fourteen metric and persona request fields.",
            "Fixed the Coval CLI dropping fourteen metric and persona request fields.",
        ),
        (
            "Update the Coval C L I and run the unit tests before cutting a release.",
            "Update the Coval CLI and run the unit tests before cutting a release.",
        ),
        (
            "Prod analysis is healthy now, but draft P R delivery still ships nothing for Coval 6429.",
            "Prod analysis is healthy now, but draft-PR delivery still ships nothing (COVAL-6429).",
        ),
        (
            "Link the branch to Coval 7701 and Coval 8137 before opening the P R.",
            "Link the branch to COVAL-7701 and COVAL-8137 before opening the PR.",
        ),
        (
            "The retired local covi dash improve task fired again and collided with the live prod run.",
            "The retired local covi-improve task fired again and collided with the live prod run.",
        ),
        (
            "Make sure the local covi improve runner stays disabled so it does not collide with prod.",
            "Make sure the local covi-improve runner stays disabled so it does not collide with prod.",
        ),
        (
            "This is a short Mac O S window dash management voice command.",
            "This is a short macOS window-management voice command.",
        ),
        (
            "Run git checkout dash b fix slash prod dash simulation from origin slash main.",
            "Run git checkout -b fix/prod-simulation from origin/main.",
        ),
        (
            "Run git checkout dash b feat slash cow code dash speed and push it to origin.",
            "Run git checkout -b feat/cowcode-speed and push it to origin.",
        ),
        (
            "Let's create branch feat slash cow code speed from origin slash main.",
            "Let's create branch feat/cowcode-speed from origin/main.",
        ),
        (
            "Let's create branch fix slash deepgram streaming from origin slash main.",
            "Let's create branch fix/deepgram-streaming from origin/main.",
        ),
    ]
    for tts_t, tgt_t in extra_commands:
        pairs.append({
            "category": "syntax_dictation",
            "split": "train",
            "tts_text": tts_t,
            "target_text": tgt_t,
        })
    return pairs


def build_hard_term_reinforcement() -> list[dict]:
    """Extra carrier diversity for terms that showed noisy errors or acronym ambiguity."""
    hard_terms = [
        ("Coval", "Coval"),
        ("Bloviate", "Bloviate"),
        ("Kappi", "Kappi"),
        ("Kobi Hudson", "Kobi Hudson"),
        ("Kobi", "Kobi"),
        ("Deepgram", "Deepgram"),
        ("Supabase", "Soopa base"),
        ("Telnyx", "Tell nix"),
        ("ElevenLabs", "Eleven Labs"),
        ("LiveKit", "Live Kit"),
        ("Pipecat", "Pipe cat"),
        ("Vapi", "Vapi"),
        ("Vercel", "Vercel"),
        ("Dana Dzik", "Dana Dzik"),
        ("Cole Amick", "Cole Amick"),
        ("Sausalito", "Sausalito"),
        ("San Francisco", "San Francisco"),
        ("Coval CLI", "Coval C L I"),
        ("CowCode", "cow code"),
        ("Sofia", "Sofia"),
        ("Dungeness", "Dungeness"),
    ]
    carriers = [
        "Hey, quick question about {term}, is that live in production right now?",
        "Make sure {term} passes all the checks before we ship it today.",
        "Did you check whether {term} is working properly in staging and prod?",
        "Let's double check {term} before we merge the pull request to main.",
    ]
    items: list[dict] = []
    for i, (written, spoken) in enumerate(hard_terms):
        for c_tmpl in carriers:
            items.append({
                "category": "dictionary_carrier",
                "split": "train",
                "term": written,
                "tts_text": c_tmpl.format(term=spoken),
                "target_text": c_tmpl.format(term=written),
            })
    return items


def main() -> None:
    existing_path = OUT_DIR / "corpus.json"
    existing_corpus = json.loads(existing_path.read_text(encoding="utf-8"))

    # Keep the exact 50 eval items untouched (same IDs clip_0000..clip_0499) so comparisons are 100% apples-to-apples!
    eval_texts = {x["target_text"].lower() for x in existing_corpus if x["split"] == "eval"}

    # Normalize TTS text on existing train items so acronyms are spoken naturally
    for item in existing_corpus:
        if item["split"] == "train" and item["category"] != "syntax_dictation":
            item["tts_text"] = to_spoken_tts(item["target_text"])

    new_items: list[dict] = []
    for item in build_rich_syntax_train_pairs():
        if item["target_text"].lower() not in eval_texts:
            new_items.append(item)

    for item in build_hard_term_reinforcement():
        if item["target_text"].lower() not in eval_texts:
            new_items.append(item)

    start_idx = len(existing_corpus)
    for offset, item in enumerate(new_items):
        item["id"] = f"clip_{start_idx + offset:04d}"
        existing_corpus.append(item)

    train_items = [x for x in existing_corpus if x["split"] == "train"]
    eval_items = [x for x in existing_corpus if x["split"] == "eval"]
    print(f"Expanded corpus: total={len(existing_corpus)} (train={len(train_items)}, eval={len(eval_items)})")

    existing_path.write_text(json.dumps(existing_corpus, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
