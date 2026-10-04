"""S6 query: is a quarter-hour inside an HLZF for DSO X / grid level L?

Semantics, stated once because they are the whole point:

* The unit is the quarter-hour interval: the publications define windows in quarter-hours.
  `ts` names one quarter-hour; `ts_label` says whether `ts` is its start or its end (several
  publications say their times mark quarter-hour ENDS). `ts_label="dso"` uses the DSO's own
  convention from its document.
* A quarter-hour [s, s+15) is inside a window if it lies fully within the window's physical
  span on a day that counts as a working day under the document's rules.
* The answer is three-valued: true, false or "uncertain". Uncertain means the document does
  not determine the answer (convention ambiguity at a window's first quarter-hour, an unnamed
  bridge day, a municipality-dependent holiday).
* Wall-clock time is Europe/Berlin, DST-aware: the spring-forward day has 92 quarter-hours,
  the fall-back day 100.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .calendar import GUIDELINE_CHRISTMAS, DayRules, day_info
from .models import SEASON_MONTHS, Convention, Season, fmt_min
from .prov import Prov
from .store import loads, rows

BERLIN = ZoneInfo("Europe/Berlin")
QH = timedelta(minutes=15)


class QueryError(ValueError):
    pass


def season_of(month: int) -> Season:
    for s, months in SEASON_MONTHS.items():
        if month in months:
            return s
    raise ValueError(month)


@dataclass
class Context:
    document: dict[str, Any]
    level: str
    convention: Convention
    rules: DayRules
    windows: dict[Season, list[dict[str, Any]]] = field(default_factory=dict)
    empty: set[Season] = field(default_factory=set)
    level_known: bool = True
    rule_evidence: list[dict[str, Any]] = field(default_factory=list)


def find_documents(conn: sqlite3.Connection, dso: str) -> list[dict[str, Any]]:
    q = dso.strip().lower()
    docs = rows(conn, "SELECT * FROM documents WHERE processed_at IS NOT NULL")
    exact = [d for d in docs if d["id"].lower() == q or d["dso"].lower() == q]
    if exact:
        same = {(d["dso"], d["synthetic"]) for d in exact}
        return [d for d in docs if (d["dso"], d["synthetic"]) in same]
    return [d for d in docs if q in d["dso"].lower() or d["id"].lower().startswith(q)]


def pick_document(conn: sqlite3.Connection, dso: str, year: int) -> dict[str, Any]:
    docs = [d for d in find_documents(conn, dso) if d["year"] == year]
    if not docs:
        names = sorted({d["dso"] for d in find_documents(conn, dso)})
        raise QueryError(f"No processed HLZF document for DSO {dso!r} and year {year}"
                         + (f" (known: {', '.join(names)})" if names else "") + ".")
    docs = [d for d in docs if not d["synthetic"]] or docs  # a real publication wins
    if len({d["dso"] for d in docs}) > 1:
        raise QueryError(f"DSO {dso!r} is ambiguous: {sorted({d['dso'] for d in docs})}.")
    # A corrected re-publication supersedes the original: latest publication date wins.
    docs.sort(key=lambda d: (d["publication_date"] or "", d["id"]), reverse=True)
    return docs[0]


def day_rules(conn: sqlite3.Connection, doc: dict[str, Any]) -> tuple[DayRules, list[dict]]:
    r = DayRules(states=loads(doc["states"], []))
    evidence = []
    for rule in rows(conn, "SELECT * FROM rules WHERE document_id=? AND active=1 "
                           "AND status != 'rejected'", (doc["id"],)):
        value = loads(rule["value"])
        kind = rule["kind"]
        ev = {"kind": kind, "value": value, "quote": rule["quote"], "page": rule["page"],
              "entity": rule["entity_id"]}
        if kind == "holiday_handling" and isinstance(value, dict):
            if value.get("states"):
                r.states = value["states"]
            r.holiday_mode = value.get("mode") if value.get("mode") in (
                "union", "intersection") else "union"
            r.holiday_exclude = value.get("exclude", [])
            r.holiday_source = "document"
            evidence.append(ev)
        elif kind == "bridge_days":
            r.bridge = str(value)
            evidence.append(ev)
        elif kind == "christmas_period" and isinstance(value, dict):
            if value.get("from") and value.get("to"):
                r.christmas = (value["from"], value["to"])
                r.christmas_source = "document"
            else:
                r.christmas = None
                r.christmas_source = "document (no dates)"
            evidence.append(ev)
        elif kind == "workdays_only":
            r.workdays_only = bool(value)
            evidence.append(ev)
    if r.christmas == GUIDELINE_CHRISTMAS and r.christmas_source == "ruling default":
        r.christmas_source = "document silent; default 24.12.-01.01. from BK4-13-739"
    if r.holiday_source == "registry":
        r.holiday_source = f"registry states {r.states} (document silent)"
    return r, evidence


def load_context(conn: sqlite3.Connection, dso: str, level: str, year: int) -> Context:
    level = level.upper().replace(" ", "")
    if not level.startswith("NE"):
        level = f"NE{level}"
    doc = pick_document(conn, dso, year)
    rules, evidence = day_rules(conn, doc)
    ctx = Context(document=doc, level=level,
                  convention=Convention(doc["convention"] or "assumed"),
                  rules=rules, rule_evidence=evidence)
    for w in rows(conn, "SELECT * FROM windows WHERE document_id=? AND grid_level=? AND "
                        "active=1 AND status != 'rejected' ORDER BY start_min",
                  (doc["id"], level)):
        ctx.windows.setdefault(Season(w["season"]), []).append(w)
    for e in rows(conn, "SELECT season FROM empty_cells WHERE document_id=? AND grid_level=?",
                  (doc["id"], level)):
        ctx.empty.add(Season(e["season"]))
    listed = {ne for _, ne in (tuple(x) for x in loads(doc["levels_listed"], []))}
    ctx.level_known = level in listed or bool(ctx.windows) or bool(ctx.empty)
    return ctx


def physical_span(w: dict[str, Any], conv: Convention) -> tuple[int, int, int]:
    """Return (certain_start, end, uncertain_start) in wall-clock minutes."""
    a, b = int(w["start_min"]), int(w["end_min"])
    if conv is Convention.interval_end_labels:
        return a - 15, b, a - 15
    if conv is Convention.interval_end_ambiguous:
        return a, b, a - 15
    return a, b, a


def interval_start(ts: datetime, label: str, conv: Convention) -> tuple[datetime, list[str]]:
    notes = []
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=BERLIN)
        notes.append("timestamp had no offset; interpreted as Europe/Berlin local time")
    eff = label
    if label == "dso":
        eff = "end" if conv in (Convention.interval_end_physical,
                                Convention.interval_end_ambiguous,
                                Convention.interval_end_labels) else "start"
        if conv is Convention.assumed:
            notes.append("DSO does not state its meter-timestamp convention; 'start' assumed")
    start = ts - QH if eff == "end" else ts
    utc = start.astimezone(ZoneInfo("UTC"))
    if utc.minute % 15 or utc.second or utc.microsecond:
        floored = utc.replace(minute=utc.minute - utc.minute % 15, second=0, microsecond=0)
        notes.append(f"timestamp not on the quarter-hour grid; using the quarter-hour "
                     f"starting {floored.astimezone(BERLIN).isoformat()}")
        utc = floored
    return utc, notes


def evaluate(ctx: Context, start_utc: datetime) -> dict[str, Any]:
    local = start_utc.astimezone(BERLIN)
    minute = local.hour * 60 + local.minute
    season = season_of(local.month)
    day = day_info(local.date(), ctx.rules)
    result: dict[str, Any] = {
        "interval_local": [local.isoformat(), (start_utc + QH).astimezone(BERLIN).isoformat()],
        "interval_utc": [start_utc.isoformat(), (start_utc + QH).isoformat()],
        "season": season.value,
        "day_type": day.kind,
        "day_reason": day.reason,
        "matching_window": None,
        "notes": [],
    }
    if not ctx.level_known:
        result["in_hlzf"] = None
        result["notes"].append(f"{ctx.level} is not covered by this document")
        return result

    hit, boundary = None, None
    for w in ctx.windows.get(season, []):
        certain, end, uncertain = physical_span(w, ctx.convention)
        if certain <= minute and minute + 15 <= end:
            hit = w
            break
        if uncertain <= minute < certain:
            boundary = w
    if hit or boundary:
        w = hit or boundary
        assert w is not None
        result["matching_window"] = {
            "window_id": w["wid"], "entity": w["entity_id"],
            "start": fmt_min(w["start_min"]), "end": fmt_min(w["end_min"]),
            "status": w["status"], "page": w["page"], "quote": w["quote"]}
    if day.off_peak is True:
        result["in_hlzf"] = False
        return result
    if hit and day.off_peak is False:
        result["in_hlzf"] = True
    elif hit or boundary:
        result["in_hlzf"] = "uncertain"
        if boundary and not hit:
            result["notes"].append(
                "first quarter-hour of the window: the document's convention leaves it open "
                "whether this quarter-hour is inside")
        if day.off_peak is None:
            result["notes"].append(day.reason)
    else:
        result["in_hlzf"] = False
        if season in ctx.empty:
            result["notes"].append(f"no HLZF for {ctx.level} in {season.value}")
    if result["matching_window"] and result["matching_window"]["status"] in (
            "needs-review",):
        result["notes"].append("matching window is not yet human-verified")
    return result


def check(conn: sqlite3.Connection, dso: str, level: str, ts: datetime,
          ts_label: str = "start", explain: bool = False) -> dict[str, Any]:
    if ts_label not in ("start", "end", "dso"):
        raise QueryError("ts_label must be start, end or dso")
    probe = ts if ts.tzinfo else ts.replace(tzinfo=BERLIN)
    year = (probe - (QH if ts_label == "end" else timedelta())).astimezone(BERLIN).year
    ctx = load_context(conn, dso, level, year)
    start_utc, notes = interval_start(ts, ts_label, ctx.convention)
    local_year = start_utc.astimezone(BERLIN).year
    if local_year != year:
        ctx = load_context(conn, dso, level, local_year)
    out = evaluate(ctx, start_utc)
    out["notes"] = notes + out["notes"]
    doc = ctx.document
    out.update({
        "dso": doc["dso"], "grid_level": ctx.level, "document": doc["id"],
        "document_status": doc["status"], "convention": ctx.convention.value,
        "ts_label": ts_label,
        "rules_applied": [{"kind": e["kind"], "value": e["value"], "quote": e["quote"],
                           "page": e["page"]} for e in ctx.rule_evidence],
    })
    order = ["in_hlzf", "dso", "grid_level", "interval_local", "interval_utc", "season",
             "day_type", "day_reason", "matching_window", "convention", "ts_label",
             "rules_applied", "document", "document_status", "notes"]
    out = {k: out[k] for k in order if k in out}
    if explain:
        target = (out["matching_window"] or {}).get("entity") or doc["entity_id"]
        out["provenance"] = Prov(conn).chain(target)
    return out
