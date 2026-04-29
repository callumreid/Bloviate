"""
Theme definitions and waveform color helpers for the Bloviate PyQt UI.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


DEFAULT_THEME_ID = "light"
DEFAULT_WAVEFORM_PRESET = "theme"

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "label": "Eames Light",
        "description": "Warm ivory, walnut lines, teal focus.",
        "colors": {
            "window": "#F7F3EA",
            "surface": "#FFFDF7",
            "surface_alt": "#F1E9DB",
            "card": "#FFFDF7",
            "card_alt": "#F8F3EA",
            "nav": "#E8DFD0",
            "nav_hover": "#DCD2C2",
            "border": "#DDD2C1",
            "border_strong": "#CFC3B2",
            "text": "#26211D",
            "text_soft": "#514941",
            "muted": "#6F665E",
            "placeholder": "#8E8376",
            "primary": "#2D6B6B",
            "primary_hover": "#245C5C",
            "primary_text": "#FFFFFF",
            "success": "#2F7D4F",
            "danger": "#B23B35",
            "warning": "#C58C2A",
            "command": "#2D6B6B",
            "selection": "#2D6B6B",
            "overlay": "rgba(38, 33, 29, 160)",
        },
        "waveform": {
            "idle": "#BFB2A1",
            "recording": "#E7C873",
            "command": "#2D6B6B",
            "accepted": "#2F7D4F",
            "rejected": "#B23B35",
            "quiet": "#BFB2A1",
            "background": "#FFFDF7",
            "text": "#26211D",
            "processing": ["#E7C873", "#8E5CF7", "#2D6B6B", "#2F7D4F", "#C58C2A"],
        },
    },
    "meadow": {
        "label": "Meadow Studio",
        "description": "Soft green, cream, and cobalt accents.",
        "colors": {
            "window": "#F4F7EE",
            "surface": "#FFFFFA",
            "surface_alt": "#E6EDDC",
            "card": "#FFFFFA",
            "card_alt": "#EEF4E6",
            "nav": "#DCE8CF",
            "nav_hover": "#CEDFBE",
            "border": "#CAD9BC",
            "border_strong": "#AFC59E",
            "text": "#1F2A22",
            "text_soft": "#405244",
            "muted": "#657366",
            "placeholder": "#7E8C7F",
            "primary": "#317A58",
            "primary_hover": "#286648",
            "primary_text": "#FFFFFF",
            "success": "#317A58",
            "danger": "#B3433F",
            "warning": "#BC842E",
            "command": "#315F9D",
            "selection": "#317A58",
            "overlay": "rgba(31, 42, 34, 154)",
        },
        "waveform": {
            "idle": "#AAB99F",
            "recording": "#D0A43A",
            "command": "#315F9D",
            "accepted": "#317A58",
            "rejected": "#B3433F",
            "quiet": "#B9C6AF",
            "background": "#FFFFFA",
            "text": "#1F2A22",
            "processing": ["#D0A43A", "#6F86D6", "#317A58", "#6AAE9D", "#BC842E"],
        },
    },
    "sunset": {
        "label": "Sunset Drafting",
        "description": "Paper, brick, teal, and gold.",
        "colors": {
            "window": "#FAF1E8",
            "surface": "#FFFDF9",
            "surface_alt": "#F2DDD0",
            "card": "#FFFDF9",
            "card_alt": "#F8E9DE",
            "nav": "#ECD6C6",
            "nav_hover": "#E3C8B6",
            "border": "#DEC1AE",
            "border_strong": "#CDA48C",
            "text": "#2E241F",
            "text_soft": "#5B4538",
            "muted": "#776255",
            "placeholder": "#947D70",
            "primary": "#A1493D",
            "primary_hover": "#883B31",
            "primary_text": "#FFFFFF",
            "success": "#2E7A62",
            "danger": "#B33D4B",
            "warning": "#C27B2B",
            "command": "#286C79",
            "selection": "#A1493D",
            "overlay": "rgba(46, 36, 31, 154)",
        },
        "waveform": {
            "idle": "#C2A99A",
            "recording": "#D69635",
            "command": "#286C79",
            "accepted": "#2E7A62",
            "rejected": "#B33D4B",
            "quiet": "#CDB8AA",
            "background": "#FFFDF9",
            "text": "#2E241F",
            "processing": ["#D69635", "#A1493D", "#286C79", "#2E7A62", "#C27B2B"],
        },
    },
    "graphite": {
        "label": "Graphite Desk",
        "description": "Dark neutral with brass and teal.",
        "colors": {
            "window": "#222524",
            "surface": "#2D312F",
            "surface_alt": "#383B36",
            "card": "#2D312F",
            "card_alt": "#343833",
            "nav": "#373A36",
            "nav_hover": "#424741",
            "border": "#4E564E",
            "border_strong": "#667166",
            "text": "#F5EFE5",
            "text_soft": "#D7CBBF",
            "muted": "#B4A99D",
            "placeholder": "#948A80",
            "primary": "#70B8AE",
            "primary_hover": "#88CCC3",
            "primary_text": "#16201E",
            "success": "#88C28D",
            "danger": "#EE8279",
            "warning": "#E1B75D",
            "command": "#7EA7D8",
            "selection": "#70B8AE",
            "overlay": "rgba(12, 14, 13, 170)",
        },
        "waveform": {
            "idle": "#807A73",
            "recording": "#E1B75D",
            "command": "#7EA7D8",
            "accepted": "#88C28D",
            "rejected": "#EE8279",
            "quiet": "#6F6861",
            "background": "#2D312F",
            "text": "#F5EFE5",
            "processing": ["#E1B75D", "#A68EF2", "#70B8AE", "#88C28D", "#E08B5C"],
        },
    },
    "ink": {
        "label": "Ink Blue",
        "description": "Focused blue-black with copper and mint.",
        "colors": {
            "window": "#182026",
            "surface": "#202B32",
            "surface_alt": "#2B3840",
            "card": "#202B32",
            "card_alt": "#26323A",
            "nav": "#2B3840",
            "nav_hover": "#34444D",
            "border": "#465963",
            "border_strong": "#5D747F",
            "text": "#F2F6F4",
            "text_soft": "#C8D2CE",
            "muted": "#A8B3AF",
            "placeholder": "#879591",
            "primary": "#6CB7D4",
            "primary_hover": "#84C8E0",
            "primary_text": "#132027",
            "success": "#78C6A3",
            "danger": "#E36C6C",
            "warning": "#D9A94E",
            "command": "#9B8FE8",
            "selection": "#6CB7D4",
            "overlay": "rgba(10, 15, 18, 176)",
        },
        "waveform": {
            "idle": "#728089",
            "recording": "#D9A94E",
            "command": "#9B8FE8",
            "accepted": "#78C6A3",
            "rejected": "#E36C6C",
            "quiet": "#65727A",
            "background": "#202B32",
            "text": "#F2F6F4",
            "processing": ["#D9A94E", "#9B8FE8", "#6CB7D4", "#78C6A3", "#D17B62"],
        },
    },
    "lounge": {
        "label": "Lounge Chair",
        "description": "Walnut, leather, and calm teal.",
        "hidden": True,
        "colors": {
            "window": "#F3E8D8",
            "surface": "#FFF7EC",
            "surface_alt": "#E4D1BC",
            "card": "#FFF7EC",
            "card_alt": "#ECDCC9",
            "nav": "#D8BFA7",
            "nav_hover": "#CBAF94",
            "border": "#C6A98D",
            "border_strong": "#9E765B",
            "text": "#2D2119",
            "text_soft": "#5B4132",
            "muted": "#786150",
            "placeholder": "#947A66",
            "primary": "#2F706B",
            "primary_hover": "#275D58",
            "primary_text": "#FFF7EC",
            "success": "#34785C",
            "danger": "#A74337",
            "warning": "#AD782E",
            "command": "#7059A8",
            "selection": "#2F706B",
            "overlay": "rgba(45, 33, 25, 160)",
        },
        "waveform": {
            "idle": "#A98D72",
            "recording": "#D4A03E",
            "command": "#7059A8",
            "accepted": "#34785C",
            "rejected": "#A74337",
            "quiet": "#BCA48D",
            "background": "#FFF7EC",
            "text": "#2D2119",
            "processing": ["#D4A03E", "#7059A8", "#2F706B", "#34785C", "#A66A3D"],
        },
    },
    "terminal_cow": {
        "label": "Terminal Cow",
        "description": "Pasture green phosphor on terminal black.",
        "hidden": True,
        "colors": {
            "window": "#101711",
            "surface": "#172117",
            "surface_alt": "#213021",
            "card": "#172117",
            "card_alt": "#1E2A1E",
            "nav": "#263626",
            "nav_hover": "#324832",
            "border": "#3D583D",
            "border_strong": "#5C7A5C",
            "text": "#E8F7E3",
            "text_soft": "#C7E2BF",
            "muted": "#9CB493",
            "placeholder": "#7C9275",
            "primary": "#84D96B",
            "primary_hover": "#A1EF87",
            "primary_text": "#101711",
            "success": "#84D96B",
            "danger": "#F07B68",
            "warning": "#E5C45E",
            "command": "#77D7B8",
            "selection": "#84D96B",
            "overlay": "rgba(7, 11, 7, 182)",
        },
        "waveform": {
            "idle": "#5F7A59",
            "recording": "#E5C45E",
            "command": "#77D7B8",
            "accepted": "#84D96B",
            "rejected": "#F07B68",
            "quiet": "#465A43",
            "background": "#172117",
            "text": "#E8F7E3",
            "processing": ["#84D96B", "#77D7B8", "#E5C45E", "#A1EF87", "#F07B68"],
        },
    },
    "after_dark": {
        "label": "Dictation After Dark",
        "description": "Late-night violet, blue, and ember.",
        "hidden": True,
        "colors": {
            "window": "#16151F",
            "surface": "#211F2E",
            "surface_alt": "#2D2A3D",
            "card": "#211F2E",
            "card_alt": "#292638",
            "nav": "#302C42",
            "nav_hover": "#3A3550",
            "border": "#4B4565",
            "border_strong": "#655D86",
            "text": "#F6F1FF",
            "text_soft": "#D8CFF0",
            "muted": "#B6A9CF",
            "placeholder": "#9487AC",
            "primary": "#A78BFA",
            "primary_hover": "#C4B5FD",
            "primary_text": "#15131F",
            "success": "#7DD3A8",
            "danger": "#FB7185",
            "warning": "#FBBF6A",
            "command": "#67D7F0",
            "selection": "#A78BFA",
            "overlay": "rgba(12, 10, 20, 184)",
        },
        "waveform": {
            "idle": "#746A91",
            "recording": "#FBBF6A",
            "command": "#67D7F0",
            "accepted": "#7DD3A8",
            "rejected": "#FB7185",
            "quiet": "#5B536F",
            "background": "#211F2E",
            "text": "#F6F1FF",
            "processing": ["#FBBF6A", "#A78BFA", "#67D7F0", "#7DD3A8", "#FB7185"],
        },
    },
    "studio_radio": {
        "label": "Studio Radio",
        "description": "Broadcast red, brushed graphite, and signal green.",
        "hidden": True,
        "colors": {
            "window": "#252524",
            "surface": "#30302E",
            "surface_alt": "#3C3A36",
            "card": "#30302E",
            "card_alt": "#373632",
            "nav": "#444139",
            "nav_hover": "#504C43",
            "border": "#5E584C",
            "border_strong": "#777064",
            "text": "#F7F2E8",
            "text_soft": "#DAD2C2",
            "muted": "#B7AD9D",
            "placeholder": "#998F80",
            "primary": "#D35D4B",
            "primary_hover": "#ED725F",
            "primary_text": "#FFF7EC",
            "success": "#74C476",
            "danger": "#F06D6D",
            "warning": "#E0B354",
            "command": "#67A5D8",
            "selection": "#D35D4B",
            "overlay": "rgba(19, 18, 17, 178)",
        },
        "waveform": {
            "idle": "#847D70",
            "recording": "#E0B354",
            "command": "#67A5D8",
            "accepted": "#74C476",
            "rejected": "#F06D6D",
            "quiet": "#676158",
            "background": "#30302E",
            "text": "#F7F2E8",
            "processing": ["#E0B354", "#D35D4B", "#67A5D8", "#74C476", "#A78BFA"],
        },
    },
}


WAVEFORM_PRESETS: dict[str, dict[str, Any]] = {
    "theme": {"label": "Follow app theme"},
    "aurora": {
        "label": "Aurora",
        "idle": "#8F95A3",
        "recording": "#E7C873",
        "command": "#62A8E5",
        "accepted": "#6AD49F",
        "rejected": "#EF6F78",
        "quiet": "#A9B0BA",
        "background": "#FFFDF7",
        "text": "#26211D",
        "processing": ["#E7C873", "#8E5CF7", "#62A8E5", "#6AD49F", "#EF8D58"],
    },
    "citrus": {
        "label": "Citrus",
        "idle": "#B8AD8F",
        "recording": "#F0B73E",
        "command": "#3C8E87",
        "accepted": "#74A94A",
        "rejected": "#CF5148",
        "quiet": "#C6BDA7",
        "background": "#FFFDF4",
        "text": "#2B261E",
        "processing": ["#F0B73E", "#E58A38", "#3C8E87", "#74A94A", "#B77EE0"],
    },
    "ocean": {
        "label": "Ocean",
        "idle": "#8395A2",
        "recording": "#D4B55A",
        "command": "#427FD0",
        "accepted": "#4EB6A5",
        "rejected": "#D86576",
        "quiet": "#A5B4BE",
        "background": "#F6FBFC",
        "text": "#16242B",
        "processing": ["#D4B55A", "#427FD0", "#4EB6A5", "#7E66C7", "#D86576"],
    },
    "mono": {
        "label": "Monochrome",
        "idle": "#9A9A9A",
        "recording": "#555555",
        "command": "#333333",
        "accepted": "#222222",
        "rejected": "#666666",
        "quiet": "#B9B9B9",
        "background": "#FFFFFF",
        "text": "#1F1F1F",
        "processing": ["#444444", "#666666", "#888888", "#666666", "#444444"],
    },
    "custom": {"label": "Custom"},
}


def is_hidden_theme(theme_id: Any) -> bool:
    return bool(THEMES.get(normalize_theme_id(theme_id), {}).get("hidden", False))


def theme_options(*, include_hidden: bool = False) -> list[tuple[str, str]]:
    return [
        (theme_id, theme["label"])
        for theme_id, theme in THEMES.items()
        if include_hidden or not theme.get("hidden", False)
    ]


def waveform_preset_options() -> list[tuple[str, str]]:
    return [(preset_id, preset["label"]) for preset_id, preset in WAVEFORM_PRESETS.items()]


def normalize_theme_id(value: Any) -> str:
    theme_id = str(value or DEFAULT_THEME_ID).strip().lower()
    if theme_id == "dark":
        return "graphite"
    return theme_id if theme_id in THEMES else DEFAULT_THEME_ID


def normalize_waveform_preset_id(value: Any) -> str:
    preset_id = str(value or DEFAULT_WAVEFORM_PRESET).strip().lower()
    return preset_id if preset_id in WAVEFORM_PRESETS else DEFAULT_WAVEFORM_PRESET


def get_theme(theme_id: Any) -> dict[str, Any]:
    return deepcopy(THEMES[normalize_theme_id(theme_id)])


def normalize_hex(value: Any, fallback: str) -> str:
    candidate = str(value or "").strip()
    return candidate.upper() if HEX_COLOR_RE.match(candidate) else fallback


def _coerce_processing_colors(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    colors = [
        normalize_hex(item, "")
        for item in raw_items
        if normalize_hex(item, "")
    ]
    return colors[:8] if colors else list(fallback)


def waveform_values_for_preset(theme_id: Any, preset_id: Any) -> dict[str, Any]:
    theme_waveform = get_theme(theme_id)["waveform"]
    preset = normalize_waveform_preset_id(preset_id)
    if preset == "theme":
        return deepcopy(theme_waveform)
    if preset == "custom":
        return deepcopy(theme_waveform)
    values = deepcopy(theme_waveform)
    values.update({k: deepcopy(v) for k, v in WAVEFORM_PRESETS[preset].items() if k != "label"})
    return values


def waveform_palette_for_config(config: dict) -> dict[str, Any]:
    ui_config = config.get("ui", {}) if isinstance(config, dict) else {}
    theme_id = normalize_theme_id(ui_config.get("theme", DEFAULT_THEME_ID))
    waveform_config = ui_config.get("waveform", {})
    if not isinstance(waveform_config, dict):
        waveform_config = {}
    preset_id = normalize_waveform_preset_id(waveform_config.get("preset", DEFAULT_WAVEFORM_PRESET))
    palette = waveform_values_for_preset(theme_id, preset_id)
    if preset_id == "custom":
        for key in ("idle", "recording", "command", "accepted", "rejected", "quiet", "background", "text"):
            palette[key] = normalize_hex(waveform_config.get(key), palette[key])
        palette["processing"] = _coerce_processing_colors(
            waveform_config.get("processing"),
            list(palette["processing"]),
        )
    return palette


def normalize_waveform_config(config: Any, theme_id: Any = DEFAULT_THEME_ID) -> dict[str, Any]:
    if not isinstance(config, dict):
        config = {}
    preset_id = normalize_waveform_preset_id(config.get("preset", DEFAULT_WAVEFORM_PRESET))
    values = waveform_values_for_preset(theme_id, preset_id)
    normalized = {"preset": preset_id}
    if preset_id == "custom":
        for key in ("idle", "recording", "command", "accepted", "rejected", "quiet", "background", "text"):
            normalized[key] = normalize_hex(config.get(key), values[key])
        normalized["processing"] = _coerce_processing_colors(
            config.get("processing"),
            list(values["processing"]),
        )
    return normalized
