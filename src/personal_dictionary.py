"""
Helpers for managing the local personal dictionary.

This is the user-facing dictionary surface:
- preferred terms bias transcription before the model decodes
- corrections rewrite known bad outputs after transcription

Legacy files (`custom_dictionary.yaml` and `learned_terms.txt`) are still loaded
for backward compatibility, but new writes go to `personal_dictionary.yaml`.
"""

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml
from app_paths import (
    app_support_dir,
    config_base_dir,
    custom_dictionary_path,
    learned_terms_path,
    legacy_repo_custom_dictionary_path,
    legacy_repo_learned_terms_path,
    legacy_repo_personal_dictionary_path,
    personal_dictionary_path,
    resolve_path,
)


DEFAULT_PERSONAL_DICTIONARY_FILE = "personal_dictionary.yaml"
DEFAULT_LEARNED_TERMS_FILE = "learned_terms.txt"
DEFAULT_CUSTOM_DICTIONARY_FILE = "custom_dictionary.yaml"

def _resolve_config_path(config: dict, config_key: str, env_name: str, default_file: str) -> Path:
    raw_path = config.get("transcription", {}).get(config_key) or os.getenv(env_name)
    if raw_path:
        return resolve_path(str(raw_path), base_dir=config_base_dir(config))
    if default_file == DEFAULT_PERSONAL_DICTIONARY_FILE:
        return personal_dictionary_path()
    if default_file == DEFAULT_LEARNED_TERMS_FILE:
        return learned_terms_path()
    if default_file == DEFAULT_CUSTOM_DICTIONARY_FILE:
        return custom_dictionary_path()
    return app_support_dir() / default_file


def normalize_term(value: str) -> str:
    """Normalize a preferred term while preserving readable casing."""
    return " ".join(str(value).strip().split())


def _normalize_correction(entry: dict) -> dict:
    phrase = normalize_term(entry.get("phrase", ""))
    variations = []
    seen = set()
    for raw_variation in entry.get("variations", []) or []:
        variation = normalize_term(raw_variation)
        if not variation:
            continue
        key = variation.lower()
        if key in seen:
            continue
        seen.add(key)
        variations.append(variation)

    match_mode = str(entry.get("match", "substring") or "substring").strip().lower()
    if match_mode not in {"substring", "whole_word"}:
        match_mode = "substring"

    if not phrase or not variations:
        return {}

    return {
        "phrase": phrase,
        "variations": variations,
        "match": match_mode,
    }


def resolve_personal_dictionary_path(config: dict) -> Path:
    """Resolve the primary personal dictionary path."""
    return _resolve_config_path(
        config,
        "personal_dictionary_path",
        "BLOVIATE_PERSONAL_DICTIONARY_PATH",
        DEFAULT_PERSONAL_DICTIONARY_FILE,
    )


def resolve_legacy_learned_terms_path(config: dict) -> Path:
    """Resolve the legacy learned-terms file path."""
    return _resolve_config_path(
        config,
        "learned_terms_path",
        "BLOVIATE_LEARNED_TERMS_PATH",
        DEFAULT_LEARNED_TERMS_FILE,
    )


def resolve_legacy_custom_dictionary_path(config: dict) -> Path:
    """Resolve the legacy custom dictionary path."""
    return _resolve_config_path(
        config,
        "custom_dictionary_path",
        "BLOVIATE_CUSTOM_DICTIONARY_PATH",
        DEFAULT_CUSTOM_DICTIONARY_FILE,
    )


def _load_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception:
        return {}


def _load_terms_txt(path: Path) -> List[str]:
    if not path.exists():
        return []

    terms: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            term = normalize_term(line)
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
    return terms


def _extract_terms(data: dict) -> List[str]:
    terms: List[str] = []
    seen = set()
    for raw_term in data.get("preferred_terms", []) or []:
        term = normalize_term(raw_term)
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _extract_corrections(data: dict) -> List[dict]:
    corrections: List[dict] = []
    seen = set()
    raw_entries = data.get("corrections")
    if raw_entries is None:
        raw_entries = data.get("entries", [])

    for entry in raw_entries or []:
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_correction(entry)
        if not normalized:
            continue
        key = (
            normalized["phrase"].lower(),
            tuple(variation.lower() for variation in normalized["variations"]),
            normalized["match"],
        )
        if key in seen:
            continue
        seen.add(key)
        corrections.append(normalized)
    return corrections


def load_personal_dictionary(config: dict) -> Dict[str, List]:
    """Load the primary personal dictionary plus any legacy local files."""
    preferred_terms: List[str] = []
    corrections: List[dict] = []
    preferred_seen = set()
    correction_seen = set()
    sources: List[str] = []

    def add_term(term: str):
        key = term.lower()
        if key in preferred_seen:
            return
        preferred_seen.add(key)
        preferred_terms.append(term)

    def add_correction(entry: dict):
        key = (
            entry["phrase"].lower(),
            tuple(variation.lower() for variation in entry["variations"]),
            entry["match"],
        )
        if key in correction_seen:
            return
        correction_seen.add(key)
        corrections.append(entry)

    primary_path = resolve_personal_dictionary_path(config)
    if primary_path.exists():
        data = _load_yaml(primary_path)
        for term in _extract_terms(data):
            add_term(term)
        for correction in _extract_corrections(data):
            add_correction(correction)
        sources.append(str(primary_path))

    legacy_custom_path = resolve_legacy_custom_dictionary_path(config)
    if legacy_custom_path.exists() and legacy_custom_path != primary_path:
        data = _load_yaml(legacy_custom_path)
        before = len(corrections)
        for correction in _extract_corrections(data):
            add_correction(correction)
        if len(corrections) > before:
            sources.append(str(legacy_custom_path))

    legacy_learned_path = resolve_legacy_learned_terms_path(config)
    if legacy_learned_path.exists() and legacy_learned_path != primary_path:
        before = len(preferred_terms)
        for term in _load_terms_txt(legacy_learned_path):
            add_term(term)
        if len(preferred_terms) > before:
            sources.append(str(legacy_learned_path))

    repo_personal_path = legacy_repo_personal_dictionary_path()
    if repo_personal_path.exists() and repo_personal_path != primary_path:
        data = _load_yaml(repo_personal_path)
        before_terms = len(preferred_terms)
        before_corrections = len(corrections)
        for term in _extract_terms(data):
            add_term(term)
        for correction in _extract_corrections(data):
            add_correction(correction)
        if len(preferred_terms) > before_terms or len(corrections) > before_corrections:
            sources.append(str(repo_personal_path))

    repo_custom_path = legacy_repo_custom_dictionary_path()
    if repo_custom_path.exists() and repo_custom_path != legacy_custom_path:
        before = len(corrections)
        data = _load_yaml(repo_custom_path)
        for correction in _extract_corrections(data):
            add_correction(correction)
        if len(corrections) > before:
            sources.append(str(repo_custom_path))

    repo_learned_path = legacy_repo_learned_terms_path()
    if repo_learned_path.exists() and repo_learned_path != legacy_learned_path:
        before = len(preferred_terms)
        for term in _load_terms_txt(repo_learned_path):
            add_term(term)
        if len(preferred_terms) > before:
            sources.append(str(repo_learned_path))

    return {
        "preferred_terms": preferred_terms,
        "corrections": corrections,
        "sources": sources,
        "path": str(primary_path),
    }


def add_preferred_terms(config: dict, terms: Iterable[str]) -> Tuple[Path, List[str]]:
    """Add preferred terms to the primary personal dictionary file."""
    path = resolve_personal_dictionary_path(config)
    data = _load_yaml(path) if path.exists() else {}
    preferred_terms = _extract_terms(data)
    seen = {term.lower() for term in preferred_terms}
    added: List[str] = []

    for raw_term in terms:
        term = normalize_term(raw_term)
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        preferred_terms.append(term)
        added.append(term)

    corrections = _extract_corrections(data)
    payload = {
        "preferred_terms": preferred_terms,
        "corrections": corrections,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)

    return path, added
