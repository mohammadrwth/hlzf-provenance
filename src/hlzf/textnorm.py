"""Deterministic string helpers: matching keys, time parsing, label mapping.

Nothing in here calls a model. Grounding and normalization deliberately stay
deterministic, so an LLM is never asked to judge its own output.
"""

from __future__ import annotations

import re
import unicodedata

from .models import Season

_DASHES = "‐‑‒–—―−⁃﹘﹣－"
_TIME_RE = re.compile(r"(?<!\d)([01]?\d|2[0-4])\s*[:.]\s*([0-5]\d)(?!\d)")
_RANGE_RE = re.compile(
    r"(?<!\d)([01]?\d|2[0-4])\s*[:.]\s*([0-5]\d)\s*(?:uhr)?\s*(?:-|bis)\s*"
    r"([01]?\d|2[0-4])\s*[:.]\s*([0-5]\d)(?!\d)",
    re.IGNORECASE,
)


def match_key(text: str) -> str:
    """Normalize text for fuzzy matching: NFKC, unify dashes, drop 'Uhr', collapse space."""
    t = unicodedata.normalize("NFKC", text)
    for d in _DASHES:
        t = t.replace(d, "-")
    t = t.lower().replace("─", " ")  # table ruling lines of the layout text
    t = re.sub(r"\buhr\b", " ", t)
    t = re.sub(r"(\d)\s*\.\s*(\d\d)(?!\d)", r"\1:\2", t)  # 07.45 -> 07:45
    t = re.sub(r"\s*-\s*", "-", t)
    t = re.sub(r"\s*:\s*", ":", t)
    t = re.sub(r"(?<!\d)(\d):(\d\d)", r"0\1:\2", t)  # 7:45 -> 07:45
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def parse_time(text: str) -> int | None:
    """'07:45' -> 465 minutes. Accepts 7:45, 07.45 and 24:00. None if not a clock time."""
    m = _TIME_RE.fullmatch(text.strip().replace("Uhr", "").strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h == 24 and mi != 0:
        return None
    return h * 60 + mi


def times_in(text: str) -> list[int]:
    """All clock times printed in a text, in minutes, in order of appearance."""
    out = []
    for m in _TIME_RE.finditer(unicodedata.normalize("NFKC", text)):
        h, mi = int(m.group(1)), int(m.group(2))
        if h < 24 or (h == 24 and mi == 0):
            out.append(h * 60 + mi)
    return out


def time_ranges_in(text: str) -> list[tuple[int, int]]:
    """All 'HH:MM - HH:MM' ranges printed in a quote, in minutes."""
    out: list[tuple[int, int]] = []
    for m in _RANGE_RE.finditer(unicodedata.normalize("NFKC", text).translate(
            {ord(d): "-" for d in _DASHES})):
        h1, m1, h2, m2 = (int(g) for g in m.groups())
        out.append((h1 * 60 + m1, h2 * 60 + m2))
    return out


# --- grid levels -------------------------------------------------------------------------
# Order matters: transformation levels (Umspannung) are checked before single voltages,
# and "Höchstspannung" before "Hochspannung".
_LEVEL_PATTERNS: list[tuple[str, str]] = [
    (r"netzebene\s*([1-7])\b|\bne\s*([1-7])\b|\bebene\s*([1-7])\b|\blevel\s*([1-7])\b", "NUM"),
    # explicit pairs first ("Hoch-/Mittelspannung", "HS/MS", ...)
    (r"\bh(ö|oe)s\s*/\s*hs\b|h(ö|oe)chst\w*[-\s]*/\s*hoch", "NE2"),
    (r"\bhs\s*/\s*ms\b|hoch\w*[-\s]*/\s*mittel", "NE4"),
    (r"\bms\s*/\s*ns\b|mittel\w*[-\s]*/\s*nieder", "NE6"),
    # "Umspannung in (die) Mittelspannung" names the target voltage of the transformation
    (r"umspannung\s+(in|auf|zur)\s+(die\s+)?hoch", "NE2"),
    (r"umspannung\s+(in|auf|zur)\s+(die\s+)?mittel", "NE4"),
    (r"umspannung\s+(in|auf|zur)\s+(die\s+)?nieder", "NE6"),
    (r"h(ö|oe)chstspannung|\bh(ö|oe)s\b", "NE1"),
    (r"hochspannung|\bhs\b", "NE3"),
    (r"mittelspannung|\bms\b", "NE5"),
    (r"niederspannung|\bns\b", "NE7"),
]


def map_level(label: str) -> str | None:
    """Map a DSO's printed grid-level label to NE1-NE7, or None if it is not recognised."""
    t = unicodedata.normalize("NFKC", label).lower()
    t = t.replace("\n", " ")
    for pattern, target in _LEVEL_PATTERNS:
        m = re.search(pattern, t)
        if not m:
            continue
        if target == "NUM":
            digit = next(g for g in m.groups() if g)
            return f"NE{digit}"
        return target
    return None


_SEASONS: list[tuple[str, Season]] = [
    (r"winter|januar|dezember|\bdez\b|\bdec\b|\bjan\b|\bfeb", Season.winter),
    (r"fr(ü|ue)hling|fr(ü|ue)hjahr|spring|m(ä|ae)rz", Season.spring),
    (r"sommer|summer|juni|jun", Season.summer),
    (r"herbst|autumn|fall|september|sep", Season.autumn),
]


def map_season(label: str) -> Season | None:
    t = unicodedata.normalize("NFKC", label).lower()
    for pattern, season in _SEASONS:
        if re.search(pattern, t):
            return season
    return None
