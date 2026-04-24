"""
Settings persistence helpers for Bloviate.

The runtime still uses a YAML config file for portability, but all UI/CLI writes
should pass through this service so saves stay consistent and runtime metadata is
not written back to disk.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

from app_paths import config_path as default_user_config_path, ensure_default_config


RUNTIME_KEY_PREFIX = "__"


def serialize_config_for_save(value: Any) -> Any:
    """Recursively strip runtime-only metadata before writing YAML."""
    if isinstance(value, dict):
        serialized = {}
        for key, item in value.items():
            if isinstance(key, str) and key.startswith(RUNTIME_KEY_PREFIX):
                continue
            serialized[key] = serialize_config_for_save(item)
        return serialized
    if isinstance(value, list):
        return [serialize_config_for_save(item) for item in value]
    return value


class SettingsService:
    """Small wrapper around the YAML config file with dotted-path updates."""

    def __init__(self, config: dict):
        self.config = config

    @property
    def path(self) -> Path:
        raw_path = self.config.get("__config_path__")
        return Path(raw_path).expanduser() if raw_path else default_user_config_path()

    def save(self) -> Path:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = serialize_config_for_save(self.config)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(serialized, f, sort_keys=False)
        return path

    def get(self, dotted_path: str, default: Any = None) -> Any:
        node: Any = self.config
        for part in self._parts(dotted_path):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_path: str, value: Any, *, save: bool = True) -> Path | None:
        node = self.config
        parts = self._parts(dotted_path)
        if not parts:
            raise ValueError("dotted_path cannot be empty")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
        return self.save() if save else None

    def update_many(self, updates: dict[str, Any], *, save: bool = True) -> Path | None:
        for dotted_path, value in updates.items():
            self.set(dotted_path, value, save=False)
        return self.save() if save else None

    def reset_section(self, section: str, default_config: dict, *, save: bool = True) -> Path | None:
        if section not in default_config:
            raise KeyError(f"Unknown config section: {section}")
        self.config[section] = deepcopy(default_config[section])
        return self.save() if save else None

    @staticmethod
    def _parts(dotted_path: str) -> list[str]:
        return [part for part in str(dotted_path).split(".") if part]


def save_config(config: dict) -> Path:
    """Compatibility helper for older call sites."""
    return SettingsService(config).save()


def load_yaml_config(path: str | Path, *, allow_missing: bool = False) -> tuple[dict, Path]:
    """Load a YAML config and attach runtime metadata."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        if resolved == Path("config.yaml"):
            resolved = default_user_config_path()
        else:
            resolved = Path.cwd() / resolved

    if not resolved.exists():
        if resolved == default_user_config_path():
            resolved = ensure_default_config()
        if allow_missing:
            return {}, resolved
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data["__config_path__"] = str(resolved)
    data["__config_dir__"] = str(resolved.parent)
    return data, resolved


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def coerce_csv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]
