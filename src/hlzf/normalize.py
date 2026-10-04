"""S3 normalize: raw model output -> canonical values (NE1-NE7, seasons, minutes, convention,
typed rules), each with its grounding evidence. Deterministic, no model calls."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .grounding import Grounding, ground_quote, ground_window
from .models import (
    Convention,
    Evidence,
    Issue,
    PageText,
    RawExtraction,
    RawWindow,
    Rule,
    RuleKind,
    Season,
    Severity,
    Stage,
    Window,
    fmt_min,
)
from .textnorm import map_level, map_season, match_key, parse_time

BRIDGE_VALUES = {"working_day", "off_peak", "off_peak_max_one", "off_peak_max_one_per_week"}


@dataclass
class NormWindow:
    window: Window
    raw: RawWindow
    grounding: Grounding


@dataclass
class NormRule:
    rule: Rule
    grounding: Grounding


@dataclass
class Normalized:
    dso: str
    year: int
    year_grounding: Grounding
    publication_date: str | None
    version_label: str | None
    correction_note: str | None
    correction_quote: str | None
    convention: Convention
    convention_page: int | None
    convention_quote: str | None
    levels_listed: list[tuple[str, str | None]]
    windows: list[NormWindow]
    empties: list[tuple[str, Season, str, int]]
    rules: list[NormRule]
    anomalies: list[str]
    issues: list[Issue] = field(default_factory=list)

    def window_keys(self) -> set[tuple[str, str, int, int]]:
        return {nw.window.key for nw in self.windows}


def map_convention(raw: RawExtraction) -> Convention:
    conv = raw.convention
    if conv.meter_label == "start":
        return Convention.interval_start
    if conv.meter_label == "end":
        ex = conv.worked_example
        if ex:
            a, ts = parse_time(ex.listed_start), parse_time(ex.first_timestamp)
            if a is not None and ts is not None and ts == a + 15:
                return Convention.interval_end_physical
            if a is not None and ts is not None and ts == a:
                return Convention.interval_end_labels
        return Convention.interval_end_ambiguous
    return Convention.assumed


# A worked example must map a listed WINDOW to meter timestamps (Ratingen: "ein
# Hochlastzeitfenster von 08:00 bis 11:30 [entspricht] den Messwerten mit den Zeitstempeln
# von 08:15 bis 11:30"). A sentence that only defines what one timestamp covers (Bayreuth:
# "Der Zeitstempel 09:15 Uhr definiert den Zeitraum von 09:00 Uhr bis 09:15 Uhr") says
# nothing about where a listed window starts; read as an example it flips the convention.
_T = r"(\d\d:\d\d)"
_TS_DEFINITION = re.compile(
    rf"zeitstempel {_T} (?:definiert|bezeichnet|entspricht|steht f[üu]r|umfasst) "
    rf"(?:den |die |das )?(?:zeitraum|viertelstunde|intervall|messperiode) "
    rf"(?:von )?{_T}(?: bis |-){_T}")
_LISTED_ARE_TIMESTAMPS = re.compile(
    r"(?:angegebenen|aufgef[üu]hrten|genannten|dargestellten|ver[öo]ffentlichten) "
    r"(?:zeiten|viertelstunden|uhrzeiten|zeitpunkte) sind (?:die )?zeitstempel")


def _example_maps_a_window(text: str, listed: int | None, first_ts: int | None) -> bool:
    """Does the text map a window range starting at `listed` onto a timestamp range starting
    at `first_ts` ("08:00 bis 11:30 ... Zeitstempeln von 08:15 bis 11:30")? The timestamp
    definition "Zeitstempel 09:15 definiert den Zeitraum von 09:00 bis 09:15" has the order
    the other way round and does not match."""
    if listed is None or first_ts is None:
        return False
    a, t = fmt_min(listed), fmt_min(first_ts)
    return re.search(rf"{a}(?: bis |-)\d\d:\d\d.{{0,80}}?zeitstempel\w* (?:von )?{t}"
                     rf"(?: bis |-)\d\d:\d\d", text) is not None


def _sentence(text: str, key_fragment: str) -> str:
    """The sentence of `text` (original spelling) that contains a match_key fragment."""
    for s in re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])", text):
        if key_fragment in match_key(s):
            return " ".join(s.split())
    return key_fragment


def resolve_convention(raw: RawExtraction, pages: dict[int, PageText]
                       ) -> tuple[Convention, int | None, str | None, Issue | None]:
    """The model's convention reading, checked against the page text.

    Returns (convention, page, quote, issue). The issue says why the reading was changed."""
    conv = map_convention(raw)
    page, quote = raw.convention.page, raw.convention.quote
    order = sorted(pages, key=lambda p: (p != page, p))
    texts = {p: match_key(pages[p].text) for p in order}
    definition = listed_ts = None
    for p in order:
        d = _TS_DEFINITION.search(texts[p])
        if d and parse_time(d.group(3)) == parse_time(d.group(1)) == (
                (parse_time(d.group(2)) or 0) + 15):
            definition, def_page = d, p
            listed_ts = _LISTED_ARE_TIMESTAMPS.search(texts[p])
            break
    explicit_labels = definition is not None and listed_ts is not None

    ex = raw.convention.worked_example
    if conv in (Convention.interval_end_physical, Convention.interval_end_labels) and ex:
        a, ts = parse_time(ex.listed_start), parse_time(ex.first_timestamp)
        if any(_example_maps_a_window(t, a, ts) for t in texts.values()):
            return conv, page, quote, None
        new = Convention.interval_end_labels if explicit_labels else \
            Convention.interval_end_ambiguous
        if new is conv:
            return conv, page, quote, None
        why = (f"The worked example {ex.listed_start} -> {ex.first_timestamp} is not a window "
               f"mapped to meter timestamps on the page")
        if explicit_labels:
            q = (f"{_sentence(pages[def_page].text, listed_ts.group(0))} "
                 f"{_sentence(pages[def_page].text, definition.group(0))}")
            msg = (f"{why}; it defines one timestamp. The document also says the listed "
                   f"quarter-hours ARE meter timestamps, so a window a-b covers [a-15, b): "
                   f"{conv.value} -> {new.value}.")
            return new, def_page, q, Issue(code="CONVENTION_CORRECTED", severity=Severity.warning,
                                           suspected_stage=Stage.extract, message=msg)
        return new, page, quote, Issue(
            code="CONVENTION_CORRECTED", severity=Severity.warning,
            suspected_stage=Stage.extract,
            message=f"{why}: {conv.value} -> {new.value}.")
    if explicit_labels and conv in (Convention.assumed, Convention.interval_end_ambiguous):
        q = (f"{_sentence(pages[def_page].text, listed_ts.group(0))} "
                 f"{_sentence(pages[def_page].text, definition.group(0))}")
        return Convention.interval_end_labels, def_page, q, Issue(
            code="CONVENTION_CORRECTED", severity=Severity.info, suspected_stage=Stage.extract,
            message=f"The document says the listed quarter-hours are meter timestamps and "
                    f"defines a timestamp as the quarter-hour it ends: {conv.value} -> "
                    f"interval_end_labels.")
    return conv, page, quote, None


def _evidence(g: Grounding, page: int, quote: str) -> Evidence:
    return Evidence(page=page, quote=quote, bbox=g.bbox, match_ratio=g.ratio,
                    cell_aligned=g.cell_aligned)


def _rule_value(kind: RuleKind, value: Any) -> tuple[Any, bool, str | None]:
    """Coerce a rule value to its canonical form. Returns (value, resolved, problem)."""
    try:
        if kind is RuleKind.workdays_only:
            return bool(value) if value is not None else True, True, None
        if kind is RuleKind.bridge_days:
            v = str(value)
            if v not in BRIDGE_VALUES:
                return v, False, f"unknown bridge_days value {v!r}"
            return v, v in {"working_day", "off_peak"}, None
        if kind is RuleKind.holiday_handling:
            v = dict(value or {})
            states = [str(s).upper() for s in v.get("states", [])]
            mode = v.get("mode", "unspecified")
            exclude = [str(x) for x in v.get("exclude", [])]
            # with a single state (or "DE": nationwide holidays only) union and
            # intersection coincide, so an unspecified mode does not matter
            return ({"states": states, "mode": mode, "exclude": exclude},
                    len(states) == 1 or (bool(states) and mode in {"union", "intersection"}),
                    None)
        if kind is RuleKind.christmas_period:
            v = dict(value or {})
            out = {"from": v.get("from"), "to": v.get("to")}
            return out, bool(out["from"] and out["to"]), None
        if kind in (RuleKind.significance_threshold, RuleKind.min_kw_diff,
                    RuleKind.de_minimis_eur):
            if isinstance(value, str):
                value = value.replace("%", "").replace("€", "").replace(",", ".").strip()
            return float(value), True, None
    except (TypeError, ValueError) as err:
        return value, False, f"cannot read value {value!r}: {err}"
    return value, True, None


def normalize(raw: RawExtraction, pages: dict[int, PageText]) -> Normalized:
    issues: list[Issue] = []
    windows: list[NormWindow] = []
    for rw in raw.windows:
        level = map_level(rw.level_label)
        season = map_season(rw.season_label)
        start, end = parse_time(rw.start), parse_time(rw.end)
        g = ground_window(pages, rw.page, rw.quote, rw.start, rw.end,
                          rw.level_quote, rw.season_quote, rw.quote_end)
        where = f"{rw.level_label} / {rw.season_label} {rw.start}-{rw.end}"
        if level is None:
            issues.append(Issue(code="LEVEL_UNMAPPED", severity=Severity.error,
                                suspected_stage=Stage.normalize,
                                message=f"Grid level label {rw.level_label!r} does not map to "
                                        f"NE1-NE7 ({where})."))
            continue
        if season is None:
            issues.append(Issue(code="SEASON_UNMAPPED", severity=Severity.error,
                                suspected_stage=Stage.normalize,
                                message=f"Season label {rw.season_label!r} not recognised "
                                        f"({where})."))
            continue
        if start is None or end is None:
            issues.append(Issue(code="TIME_UNPARSEABLE", severity=Severity.error,
                                suspected_stage=Stage.extract,
                                message=f"Window times are not clock times ({where})."))
            continue
        quote = f"{rw.quote} … {rw.quote_end}" if rw.quote_end else rw.quote
        w = Window(grid_level=level, season=season, start_min=start, end_min=end,
                   raw_level_label=rw.level_label, evidence=_evidence(g, rw.page, quote))
        windows.append(NormWindow(window=w, raw=rw, grounding=g))

    empties = []
    for ec in raw.empty_cells:
        level, season = map_level(ec.level_label), map_season(ec.season_label)
        if level and season:
            empties.append((level, season, ec.level_label, ec.page))
        else:
            issues.append(Issue(code="EMPTY_CELL_UNMAPPED", severity=Severity.warning,
                                suspected_stage=Stage.normalize,
                                message=f"Empty cell {ec.level_label!r} / {ec.season_label!r} "
                                        "could not be mapped."))

    rules: list[NormRule] = []
    for rr in raw.rules:
        try:
            kind = RuleKind(rr.kind)
        except ValueError:
            kind = RuleKind.other
        value, resolved, problem = _rule_value(kind, rr.value)
        g = ground_quote(pages, rr.page, rr.quote)
        level = map_level(rr.level_label) if rr.level_label else None
        rules.append(NormRule(rule=Rule(kind=kind, value=value, grid_level=level,
                                        evidence=_evidence(g, rr.page, rr.quote),
                                        resolved=resolved), grounding=g))
        if problem:
            issues.append(Issue(code="RULE_VALUE_INVALID", severity=Severity.warning,
                                suspected_stage=Stage.extract,
                                message=f"Rule {kind.value}: {problem}"))

    conv, conv_page, conv_quote, conv_issue = resolve_convention(raw, pages)
    if conv_issue:
        issues.append(conv_issue)
    year_g = ground_quote(pages, raw.validity_year.page, raw.validity_year.quote)
    return Normalized(
        dso=raw.dso_name.value,
        year=raw.validity_year.value,
        year_grounding=year_g,
        publication_date=raw.publication_date.value if raw.publication_date else None,
        version_label=raw.version_label.value if raw.version_label else None,
        correction_note=raw.correction_note.value if raw.correction_note else None,
        correction_quote=raw.correction_note.quote if raw.correction_note else None,
        convention=conv,
        convention_page=conv_page,
        convention_quote=conv_quote,
        levels_listed=[(lv.label, map_level(lv.label)) for lv in raw.levels_listed],
        windows=windows,
        empties=empties,
        rules=rules,
        anomalies=list(raw.anomalies),
        issues=issues,
    )
