"""
Provider and model metadata used by the settings UI and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ModelOption:
    value: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class ProviderOption:
    value: str
    label: str
    requires_key: bool


class ModelRegistry:
    PROVIDERS = [
        ProviderOption("deepgram", "Deepgram", True),
        ProviderOption("openai", "OpenAI", True),
        ProviderOption("whisper", "Local Whisper", False),
    ]

    WHISPER_MODELS = [
        ModelOption("tiny.en", "tiny.en", "Fastest local English model"),
        ModelOption("base.en", "base.en", "Fast local English model"),
        ModelOption("small.en", "small.en", "Better local accuracy"),
        ModelOption("medium.en", "medium.en", "Highest default local accuracy"),
        ModelOption("large", "large", "Multilingual local Whisper model"),
        ModelOption("large-v3", "large-v3", "Latest common Whisper large model"),
    ]

    DEEPGRAM_MODELS = [
        ModelOption("nova-3", "nova-3", "Best general-purpose Deepgram model"),
        ModelOption("nova-3-general", "nova-3-general", "Explicit Nova-3 general model"),
        ModelOption("flux-general-en", "flux-general-en", "Low-latency turn-aware English streaming"),
        ModelOption("flux-general-multi", "flux-general-multi", "Low-latency multilingual Flux"),
        ModelOption("nova-2", "nova-2", "Legacy fallback where Nova-3 is unavailable"),
    ]

    OPENAI_TRANSCRIBE_MODELS = [
        ModelOption("gpt-4o-transcribe", "gpt-4o-transcribe", "Accuracy-first OpenAI STT"),
        ModelOption("gpt-4o-mini-transcribe", "gpt-4o-mini-transcribe", "Lower-cost OpenAI STT"),
        ModelOption("gpt-transcribe", "gpt-transcribe", "Newer OpenAI STT; weaker on spoken syntax"),
        ModelOption("whisper-1", "whisper-1", "OpenAI-hosted Whisper"),
    ]

    # OpenRouter exposes many vendors behind one OpenAI-compatible key. Slugs are
    # vendor-prefixed and must be sent verbatim. Note: OpenRouter accepts but
    # ignores `prompt`, so dictionary bias does not reach these models.
    OPENROUTER_TRANSCRIBE_MODELS = [
        ModelOption(
            "qwen/qwen3-asr-flash-2026-02-10",
            "qwen3-asr-flash",
            "Best spoken-syntax rendering (dash b -> -b, slash -> /)",
        ),
        ModelOption("openai/gpt-4o-transcribe", "gpt-4o-transcribe", "Strong syntax rendering, pricier"),
        ModelOption("microsoft/mai-transcribe-1.5", "mai-transcribe-1.5", "Good accuracy, mid latency"),
        ModelOption("openai/whisper-large-v3-turbo", "whisper-large-v3-turbo", "Cheapest usable; literal dash/slash"),
        ModelOption("mistralai/voxtral-mini-transcribe", "voxtral-mini-transcribe", "Fast, literal dash/slash"),
        ModelOption("deepgram/nova-3", "nova-3", "Lowest latency; weaker on technical terms"),
        ModelOption("nvidia/parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3", "Very fast, mangles technical terms"),
        ModelOption("openai/gpt-transcribe", "gpt-transcribe", "Newer OpenAI STT; weaker on spoken syntax"),
        ModelOption("x-ai/grok-stt-1.0", "grok-stt-1.0", "Mid accuracy"),
        ModelOption("google/chirp-3", "chirp-3", "Weakest on technical terms; most expensive"),
    ]

    OPENAI_CLEANUP_MODELS = [
        ModelOption("gpt-4o", "gpt-4o", "Stable high-quality cleanup model"),
        ModelOption("gpt-4o-mini", "gpt-4o-mini", "Lower-cost cleanup model"),
    ]

    OPENROUTER_CLEANUP_MODELS = [
        ModelOption("openai/gpt-4o", "gpt-4o", "Stable high-quality cleanup model"),
        ModelOption("openai/gpt-4o-mini", "gpt-4o-mini", "Lower-cost cleanup model"),
    ]

    FINAL_PASS_MODES = ["hybrid", "prerecorded", "streaming"]
    POST_PROCESSING_MODES = ["verbatim", "tidy", "clean", "coding", "message"]
    OUTPUT_FORMATS = ["clipboard", "stdout", "both"]

    def providers(self) -> list[ProviderOption]:
        return list(self.PROVIDERS)

    def provider_values(self) -> list[str]:
        return [provider.value for provider in self.PROVIDERS]

    def models_for(
        self,
        provider: str,
        *,
        purpose: str = "transcription",
        base_url: str = "",
    ) -> list[ModelOption]:
        normalized = self.normalize_provider(provider)
        if normalized == "deepgram":
            return list(self.DEEPGRAM_MODELS)
        if normalized == "openai" and self.is_openrouter(base_url):
            if purpose == "cleanup":
                return list(self.OPENROUTER_CLEANUP_MODELS)
            return list(self.OPENROUTER_TRANSCRIBE_MODELS)
        if normalized == "openai" and purpose == "cleanup":
            return list(self.OPENAI_CLEANUP_MODELS)
        if normalized == "openai":
            return list(self.OPENAI_TRANSCRIBE_MODELS)
        if normalized == "whisper":
            return list(self.WHISPER_MODELS)
        return []

    def validate_provider(self, provider: str) -> str:
        normalized = self.normalize_provider(provider)
        if normalized not in self.provider_values():
            raise ValueError(f"Unsupported provider: {provider}")
        return normalized

    def validate_model(self, provider: str, model: str, *, purpose: str = "transcription") -> str:
        value = str(model or "").strip()
        if not value:
            raise ValueError("Model cannot be blank")
        # Allow custom provider model strings, but keep known models discoverable in UI.
        return value

    def normalize_provider_priority(self, providers: Iterable[str]) -> list[str]:
        priority = []
        seen = set()
        for provider in providers:
            try:
                normalized = self.validate_provider(provider)
            except ValueError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            priority.append(normalized)
        return priority or ["openai", "deepgram", "whisper"]

    @staticmethod
    def is_openrouter(base_url: str) -> bool:
        """OpenRouter is OpenAI-compatible, so it rides the openai provider path."""
        return "openrouter.ai" in str(base_url or "").lower()

    @staticmethod
    def normalize_provider(provider: str) -> str:
        value = str(provider or "").strip().lower()
        aliases = {
            "local": "whisper",
            "local_whisper": "whisper",
            "openai-stt": "openai",
            "openai_transcribe": "openai",
        }
        return aliases.get(value, value)
