"""
Shared voice-command vocabulary for parsing and ASR prompting.
"""

from typing import List, Sequence, Tuple


AliasGroups = Sequence[Tuple[str, Sequence[str]]]


WINDOW_COMMAND_ALIASES: AliasGroups = [
    ("left", ["left", "left half", "left side", "left hand", "left-hand"]),
    ("right", ["right", "right half", "right side", "right hand", "right-hand"]),
    ("top", ["top", "top half", "upper", "upper half", "up"]),
    ("bottom", ["bottom", "bottom half", "lower", "lower half", "down"]),
    ("fullscreen", ["full screen", "fullscreen", "maximize"]),
    ("exit_fullscreen", ["exit full screen", "exit fullscreen", "unmaximize", "restore"]),
    ("larger", ["larger", "bigger", "grow"]),
    ("smaller", ["smaller", "shrink"]),
    ("top_left_quarter", ["top left quarter", "top left", "first quarter"]),
    ("top_right_quarter", ["top right quarter", "top right", "second quarter"]),
    ("bottom_left_quarter", ["bottom left quarter", "bottom left", "third quarter"]),
    ("bottom_right_quarter", ["bottom right quarter", "bottom right", "fourth quarter"]),
    ("desktop_left", ["desktop left"]),
    ("desktop_right", ["desktop right"]),
]


WINDOW_PREFIX_SUFFIXES: AliasGroups = [
    ("left", ["left", "left half", "left side"]),
    ("right", ["right", "right half", "right side"]),
    ("top", ["top", "top half", "upper half"]),
    ("bottom", ["bottom", "bottom half", "lower half"]),
    ("fullscreen", ["full screen", "fullscreen", "maximize"]),
    ("exit_fullscreen", ["exit full screen", "exit fullscreen", "unmaximize", "restore"]),
    ("larger", ["larger", "bigger", "grow"]),
    ("smaller", ["smaller", "shrink"]),
    ("top_left_quarter", ["top left quarter", "top left"]),
    ("top_right_quarter", ["top right quarter", "top right"]),
    ("bottom_left_quarter", ["bottom left quarter", "bottom left"]),
    ("bottom_right_quarter", ["bottom right quarter", "bottom right"]),
]


DESKTOP_PREFIX_SUFFIXES: AliasGroups = [
    ("desktop_left", ["left"]),
    ("desktop_right", ["right"]),
]


def sorted_aliases(alias_groups: AliasGroups) -> List[Tuple[str, str]]:
    """Flatten command aliases and sort longest phrases first."""
    flattened: List[Tuple[str, str]] = []
    for command, phrases in alias_groups:
        for phrase in phrases:
            flattened.append((phrase, command))
    flattened.sort(key=lambda item: len(item[0]), reverse=True)
    return flattened


def get_command_prompt_phrases() -> List[str]:
    """Return distinct command phrases for ASR prompting and keyterms."""
    phrases: List[str] = []
    seen = set()

    def add_phrase(value: str):
        phrase = str(value).strip()
        if not phrase:
            return
        key = phrase.lower()
        if key in seen:
            return
        seen.add(key)
        phrases.append(phrase)

    for _, aliases in WINDOW_COMMAND_ALIASES:
        for phrase in aliases:
            add_phrase(phrase)

    for _, suffixes in WINDOW_PREFIX_SUFFIXES:
        for suffix in suffixes:
            add_phrase(f"window {suffix}")
            add_phrase(f"screen {suffix}")
            add_phrase(f"run command {suffix}")
            add_phrase(f"run command screen {suffix}")

    for _, suffixes in DESKTOP_PREFIX_SUFFIXES:
        for suffix in suffixes:
            add_phrase(f"desktop {suffix}")
            add_phrase(f"run command desktop {suffix}")

    return phrases
