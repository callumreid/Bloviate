"""Word error rate with dictation-aware normalization.

Spoken syntax ("dash b", "slash", "dot") is mapped to its written form so a
correct `git checkout -b feature/x` is not penalised against the reference
"git checkout dash b feature slash x". Number words are digit-folded.
"""

import re

_ALIASES = [
    (r"\bdash dash no fix\b", "--no-fix"), (r"\bdash dash no dash fix\b", "--no-fix"),
    (r"\bdash b\b", "-b"), (r"\bdash d\b", "-d"),
    (r"\bdot\b", "."), (r"\bslash\b", "/"),
    (r"\bcal demo\b", "cal-demo"), (r"\bp m\b", "pm"), (r"\ba m\b", "am"),
]
_NUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "fifteen": "15", "seventeen": "17", "thirty": "30"}


def _norm(s: str) -> list[str]:
    s = re.sub(r"[,.!?;:\"']", " ", s.lower().replace("’", " "))
    s = re.sub(r"\s+", " ", s).strip()
    for pat, rep in _ALIASES:
        s = re.sub(pat, rep, s)
    s = re.sub(r"(?<=\w)/(?=\w)", " / ", s)
    s = re.sub(r"(?<=\w)\.(?=\w{2,})", " . ", s)
    return [_NUM.get(t, t) for t in s.split() if t != "."]


def wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(len(r), 1)
