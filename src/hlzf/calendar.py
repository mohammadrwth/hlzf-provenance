"""Which days are off-peak for a given DSO publication.

Each publication states its own rule, and the rule is taken from the document: N-ERGIE Netz
(network in Bavaria and Baden-Württemberg, per its document) counts only holidays valid in its
whole network area and names Mariä Himmelfahrt as an exception. In Bavaria, Mariä Himmelfahrt
(15.08.) is a holiday only in some municipalities, so without the customer's municipality that
day is reported as uncertain instead of guessed. Defaults for a silent document (Christmas
period, bridge days) follow the ruling most publications cite, BNetzA BK4-13-739, section 2.e.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

import holidays

GUIDELINE_CHRISTMAS = ("12-24", "01-01")  # BK4-13-739, 2.e: 24. Dezember - 1. Januar


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


@dataclass
class DayRules:
    states: list[str] = field(default_factory=list)
    holiday_mode: str = "union"  # union | intersection
    holiday_exclude: list[str] = field(default_factory=list)
    holiday_source: str = "registry"  # document | registry
    bridge: str = "unstated"  # working_day | off_peak | off_peak_max_one[_per_week] | unstated
    christmas: tuple[str, str] | None = GUIDELINE_CHRISTMAS
    christmas_source: str = "ruling default"  # document | document (no dates) | ...
    workdays_only: bool = True


@dataclass
class DayInfo:
    kind: str  # workday | weekend | holiday | christmas | bridge
    off_peak: bool | None  # None = cannot be decided from the document
    reason: str


@lru_cache(maxsize=256)
def _state_holidays(state: str, year: int, catholic: bool) -> dict[date, str]:
    cats = ("public", "catholic") if catholic else ("public",)
    try:
        h = holidays.Germany(subdiv=state, years=year, categories=cats)
    except (NotImplementedError, KeyError, ValueError):
        h = holidays.Germany(subdiv=state, years=year)
    return dict(h.items())


def holiday_map(rules: DayRules, year: int) -> tuple[dict[date, str], dict[date, str]]:
    """Return (certain holidays, municipality-dependent holidays) for the rule set."""
    states = rules.states or ["DE"]
    maps = []
    for st in states:
        if st == "DE":  # nationwide holidays only ("bundeseinheitliche Feiertage")
            maps.append(dict(holidays.Germany(years=year).items()))
        else:
            maps.append(_state_holidays(st, year, catholic=False))
    if rules.holiday_mode == "intersection":
        keys = set(maps[0])
        for m in maps[1:]:
            keys &= set(m)
    else:
        keys = set().union(*maps)
    merged = {d: next(m[d] for m in maps if d in m) for d in keys}

    uncertain: dict[date, str] = {}
    if "BY" in rules.states and rules.holiday_mode != "intersection":
        by_cath = _state_holidays("BY", year, catholic=True)
        for d, name in by_cath.items():
            if d not in merged:
                uncertain[d] = f"{name} (depends on the customer's municipality in Bavaria)"

    excl = [_fold(x) for x in rules.holiday_exclude]

    def excluded(name: str) -> bool:
        n = _fold(name)
        return any(x and (x in n or n in x) for x in excl)

    merged = {d: n for d, n in merged.items() if not excluded(n)}
    uncertain = {d: n for d, n in uncertain.items() if not excluded(n) and d not in merged}
    return merged, uncertain


def _in_christmas(d: date, period: tuple[str, str]) -> bool:
    start = date(d.year, int(period[0][:2]), int(period[0][3:]))
    end = date(d.year, int(period[1][:2]), int(period[1][3:]))
    if start <= end:
        return start <= d <= end
    return d >= start or d <= end  # wraps over New Year


def day_info(d: date, rules: DayRules) -> DayInfo:
    if d.weekday() >= 5:
        return DayInfo("weekend", True, "weekend (windows apply Monday-Friday only)")
    hol, maybe = holiday_map(rules, d.year)
    if d in hol:
        return DayInfo("holiday", True, f"public holiday: {hol[d]} ({rules.holiday_source})")

    if rules.christmas is not None:
        if _in_christmas(d, rules.christmas):
            return DayInfo("christmas", True,
                           f"Christmas period {rules.christmas[0]}..{rules.christmas[1]} "
                           f"({rules.christmas_source})")
    else:
        # "zwischen Weihnachten und Neujahr" without dates: 27.-30.12. are certainly inside;
        # 24.12. and 31.12. are not decidable from the text.
        if d.month == 12 and 27 <= d.day <= 30:
            return DayInfo("christmas", True, "between Christmas and New Year (document)")
        if d.month == 12 and d.day in (24, 31):
            return DayInfo("christmas", None,
                           "document says 'between Christmas and New Year' without dates; "
                           f"{d.day}.12. is not decidable")

    if d in maybe:
        return DayInfo("holiday", None, maybe[d])

    prev_hol = (d - timedelta(days=1)) in hol
    next_hol = (d + timedelta(days=1)) in hol
    is_bridge = (d.weekday() == 0 and next_hol) or (d.weekday() == 4 and prev_hol)
    if is_bridge:
        if rules.bridge == "working_day":
            return DayInfo("bridge", False, "bridge day, counted as working day (document)")
        if rules.bridge == "off_peak":
            return DayInfo("bridge", True, "bridge day, off-peak (document)")
        if rules.bridge in ("off_peak_max_one", "off_peak_max_one_per_week"):
            return DayInfo("bridge", None,
                           "bridge day; the document allows at most one bridge day as "
                           "off-peak without naming it")
        return DayInfo("bridge", None,
                       "bridge day; the document does not say how bridge days count "
                       "(BK4-13-739 allows at most one per week as off-peak)")
    return DayInfo("workday", False, "working day")
