"""
Path helpers for Bloviate user data, config bootstrap, and packaged resources.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import os
import sys
from typing import Dict


APP_NAME = "Bloviate"
APP_ENV_HOME = "BLOVIATE_HOME"
RESOURCE_PACKAGE = "bloviate_resources"
RESOURCE_FALLBACKS = {
    "default_config.yaml": "config.yaml",
    "personal_dictionary.example.yaml": "personal_dictionary.example.yaml",
    "custom_dictionary.example.yaml": "custom_dictionary.example.yaml",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def app_support_dir() -> Path:
    override = os.getenv(APP_ENV_HOME)
    if override:
        return Path(override).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME

    xdg_config_home = os.getenv("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home).expanduser() / APP_NAME.lower()

    return Path.home() / ".config" / APP_NAME.lower()


def config_path() -> Path:
    return app_support_dir() / "config.yaml"


def logs_dir() -> Path:
    return app_support_dir() / "logs"


def history_db_path() -> Path:
    return app_support_dir() / "history.sqlite"


def achievements_dir() -> Path:
    return app_support_dir() / "achievements"


def achievement_badges_dir() -> Path:
    return achievements_dir() / "badges"


def models_dir() -> Path:
    return app_support_dir() / "models"


def personal_dictionary_path() -> Path:
    return app_support_dir() / "personal_dictionary.yaml"


def learned_terms_path() -> Path:
    return app_support_dir() / "learned_terms.txt"


def custom_dictionary_path() -> Path:
    return app_support_dir() / "custom_dictionary.yaml"


def legacy_repo_personal_dictionary_path() -> Path:
    return project_root() / "personal_dictionary.yaml"


def legacy_repo_learned_terms_path() -> Path:
    return project_root() / "learned_terms.txt"


def legacy_repo_custom_dictionary_path() -> Path:
    return project_root() / "custom_dictionary.yaml"


def legacy_repo_voice_profile_path() -> Path:
    return project_root() / "models" / "voice_profile.pkl"


def config_base_dir(config: dict | None) -> Path:
    if config:
        raw = config.get("__config_dir__")
        if raw:
            return Path(str(raw)).expanduser()
    return app_support_dir()


def resolve_path(raw_path: str, *, base_dir: Path | None = None) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir or app_support_dir()) / path


def ensure_support_dirs() -> Dict[str, Path]:
    paths = {
        "home": app_support_dir(),
        "logs": logs_dir(),
        "models": models_dir(),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_resource_text(name: str) -> str:
    try:
        resource = resources.files(RESOURCE_PACKAGE).joinpath(name)
        return resource.read_text(encoding="utf-8")
    except Exception:
        fallback_name = RESOURCE_FALLBACKS.get(name)
        if not fallback_name:
            raise
        fallback_path = project_root() / fallback_name
        return fallback_path.read_text(encoding="utf-8")


def write_resource_if_missing(name: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(read_resource_text(name), encoding="utf-8")
    return target


def ensure_default_config() -> Path:
    ensure_support_dirs()
    return write_resource_if_missing("default_config.yaml", config_path())


def describe_paths() -> Dict[str, Path]:
    return {
        "home": app_support_dir(),
        "config": config_path(),
        "history": history_db_path(),
        "achievements": achievements_dir(),
        "logs": logs_dir(),
        "models": models_dir(),
        "personal_dictionary": personal_dictionary_path(),
        "legacy_repo_config": project_root() / "config.yaml",
        "legacy_repo_personal_dictionary": legacy_repo_personal_dictionary_path(),
        "legacy_repo_voice_profile": legacy_repo_voice_profile_path(),
    }
