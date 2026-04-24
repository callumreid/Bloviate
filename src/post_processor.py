"""
Transcript cleanup and formatting.

Deterministic cleanup is always available. Optional OpenAI cleanup is used only
when enabled and an API key can be resolved.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from secret_store import SecretStore


FILLER_PATTERN = re.compile(
    r"\b(?:um+|uh+|er+|ah+|like|you know|sort of|kind of)\b[, ]*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProcessedTranscript:
    original_text: str
    text: str
    mode: str
    provider: str
    changed: bool


class PostProcessor:
    """Apply deterministic and optional model-based transcript cleanup."""

    def __init__(self, config: dict, secret_store: Optional[SecretStore] = None):
        self.config = config
        self.secret_store = secret_store or SecretStore()

    def process(self, text: str, *, target_app: str = "", mode: Optional[str] = None) -> ProcessedTranscript:
        original = str(text or "").strip()
        cfg = self.config.get("post_processing", {})
        configured_mode = str(mode or cfg.get("mode", "verbatim") or "verbatim").strip().lower()
        if configured_mode not in {"verbatim", "clean", "coding", "message"}:
            configured_mode = "verbatim"

        deterministic = self._deterministic_cleanup(original, configured_mode)
        provider = "deterministic"

        if bool(cfg.get("openai_enabled", False)) and configured_mode != "verbatim":
            remote = self._openai_cleanup(deterministic, configured_mode, target_app=target_app)
            if remote:
                deterministic = remote
                provider = "openai"

        return ProcessedTranscript(
            original_text=original,
            text=deterministic,
            mode=configured_mode,
            provider=provider,
            changed=deterministic != original,
        )

    def _deterministic_cleanup(self, text: str, mode: str) -> str:
        if mode == "verbatim":
            return text

        cleaned = " ".join(text.split())
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)

        if mode in {"clean", "message"}:
            cleaned = FILLER_PATTERN.sub("", cleaned)
            cleaned = " ".join(cleaned.split())

        if mode != "coding":
            cleaned = self._capitalize_sentence(cleaned)
            if cleaned and cleaned[-1] not in ".!?":
                cleaned += "."

        return cleaned.strip()

    @staticmethod
    def _capitalize_sentence(text: str) -> str:
        if not text:
            return text
        for index, char in enumerate(text):
            if char.isalpha():
                return text[:index] + char.upper() + text[index + 1 :]
        return text

    def _openai_cleanup(self, text: str, mode: str, *, target_app: str = "") -> Optional[str]:
        api_key = self.secret_store.get_api_key("openai", self.config)
        if not api_key:
            return None

        cfg = self.config.get("post_processing", {})
        model = str(cfg.get("openai_model", "gpt-4o") or "gpt-4o").strip()
        base_url = str(self.config.get("openai", {}).get("base_url", "https://api.openai.com/v1")).rstrip("/")
        timeout_s = float(cfg.get("timeout_s", 12))
        system = (
            "You clean up dictated text. Preserve meaning, names, identifiers, commands, "
            "and code-like tokens. Return only the final text."
        )
        user = (
            f"Mode: {mode}\n"
            f"Target app: {target_app or 'unknown'}\n"
            "Clean this transcript without adding new ideas:\n"
            f"{text}"
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(cfg.get("temperature", 0.1)),
        }
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
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content", "")
            cleaned = str(content or "").strip()
            return cleaned or None
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            print(f"[Post-processing] OpenAI cleanup failed (HTTP {exc.code}): {detail}")
            return None
        except Exception as exc:
            print(f"[Post-processing] OpenAI cleanup unavailable: {exc}")
            return None
