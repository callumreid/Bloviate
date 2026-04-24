"""
API-key storage and lookup.

Bloviate prefers macOS Keychain through the optional `keyring` package, but keeps
environment variables and legacy inline config keys working for beta users.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


SERVICE_NAME = "Bloviate"


PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
}


def _load_keyring():
    try:
        import keyring  # type: ignore

        return keyring
    except Exception:
        return None


@dataclass(frozen=True)
class SecretStatus:
    provider: str
    env_name: str
    has_config_key: bool
    has_keychain_key: bool
    has_env_key: bool
    source: str
    redacted_value: str = ""


class SecretStore:
    """Resolve and store provider API keys."""

    def __init__(self, service_name: str = SERVICE_NAME):
        self.service_name = service_name
        self._keyring = _load_keyring()

    def env_name_for(self, provider: str, config: Optional[dict] = None) -> str:
        normalized = self._normalize_provider(provider)
        section = (config or {}).get(normalized, {}) if isinstance(config, dict) else {}
        return str(section.get("api_key_env") or PROVIDER_ENV.get(normalized, "")).strip()

    def get_api_key(self, provider: str, config: Optional[dict] = None) -> Optional[str]:
        normalized = self._normalize_provider(provider)
        section = (config or {}).get(normalized, {}) if isinstance(config, dict) else {}

        inline = section.get("api_key")
        if inline:
            return str(inline)

        keychain_value = self.get_keychain_key(normalized)
        if keychain_value:
            return keychain_value

        env_name = self.env_name_for(normalized, config)
        return os.getenv(env_name) if env_name else None

    def set_api_key(self, provider: str, value: str) -> tuple[bool, str]:
        normalized = self._normalize_provider(provider)
        value = str(value or "").strip()
        if not value:
            return self.delete_api_key(normalized)
        if self._keyring is None:
            return False, "Keychain support is unavailable. Install keyring or use environment variables."
        try:
            self._keyring.set_password(self.service_name, normalized, value)
            return True, f"{normalized.title()} API key saved to Keychain."
        except Exception as exc:
            return False, f"Could not save {normalized} API key: {exc}"

    def delete_api_key(self, provider: str) -> tuple[bool, str]:
        normalized = self._normalize_provider(provider)
        if self._keyring is None:
            return False, "Keychain support is unavailable."
        try:
            existing = self._keyring.get_password(self.service_name, normalized)
            if existing is None:
                return True, f"No {normalized} API key was stored in Keychain."
            self._keyring.delete_password(self.service_name, normalized)
            return True, f"{normalized.title()} API key removed from Keychain."
        except Exception as exc:
            return False, f"Could not remove {normalized} API key: {exc}"

    def get_keychain_key(self, provider: str) -> Optional[str]:
        normalized = self._normalize_provider(provider)
        if self._keyring is None:
            return None
        try:
            value = self._keyring.get_password(self.service_name, normalized)
        except Exception:
            return None
        return str(value) if value else None

    def status(self, provider: str, config: Optional[dict] = None) -> SecretStatus:
        normalized = self._normalize_provider(provider)
        section = (config or {}).get(normalized, {}) if isinstance(config, dict) else {}
        env_name = self.env_name_for(normalized, config)
        has_config_key = bool(section.get("api_key"))
        has_keychain_key = bool(self.get_keychain_key(normalized))
        has_env_key = bool(os.getenv(env_name)) if env_name else False
        source = "missing"
        if has_config_key:
            source = "config"
        elif has_keychain_key:
            source = "keychain"
        elif has_env_key:
            source = "environment"
        redacted_value = self.redacted_api_key(normalized, config) if source != "missing" else ""
        return SecretStatus(
            provider=normalized,
            env_name=env_name,
            has_config_key=has_config_key,
            has_keychain_key=has_keychain_key,
            has_env_key=has_env_key,
            source=source,
            redacted_value=redacted_value,
        )

    def redacted_api_key(self, provider: str, config: Optional[dict] = None) -> str:
        """Return a display-safe representation of the resolved key."""
        value = self.get_api_key(provider, config)
        if not value:
            return ""
        return self._redact(value)

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        value = str(provider or "").strip().lower()
        aliases = {"openai-stt": "openai", "dg": "deepgram"}
        return aliases.get(value, value)

    @staticmethod
    def _redact(value: str) -> str:
        value = str(value or "").strip()
        if not value:
            return ""
        if len(value) <= 8:
            return "configured"
        return f"{value[:4]}...{value[-4:]}"
