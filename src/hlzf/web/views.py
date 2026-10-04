"""View models for the review UI: plain dicts, computed from SQLite, rendered by Jinja."""

from __future__ import annotations

import sqlite3
from typing import Any

from ..models import GRID_LEVEL_NAMES, GRID_LEVELS, Convention, Season, fmt_min
from ..prov import Prov
from ..store import loads, row, rows

SEASON_LABELS = {"winter": "Winter", "spring": "Spring", "summer": "Summer",
                 "autumn": "Autumn"}
SEASON_MONTHS_TXT = {"winter": "Jan, Feb, Dec", "spring": "Mar–May", "summer": "Jun–Aug",
                     "autumn": "Sep–Nov"}
STATUS_LABELS = {"auto-ok": "Auto-OK", "caveats": "Auto-OK, caveats",
                 "auto-corrected": "Auto-corrected",
                 "needs-review": "Needs review", "verified": "Verified",
                 "edited": "Edited", "rejected": "Rejected", "not fetched": "Not fetched",
                 "not extracted": "Not extracted"}
STAGE_LABELS = {"source_document": "Source document", "parse": "Parse", "extract": "Extract",
                "normalize": "Normalize", "vision_misread": "Vision check misread",
                "inconclusive": "Inconclusive"}
CONVENTION_TEXT = {
    Convention.interval_start.value: (
        "Interval start", "Times label the start of each quarter-hour; a window a–b covers "
                          "the quarter-hours from a up to b."),
    Convention.interval_end_physical.value: (
        "Interval end, physical span", "Meter timestamps label quarter-hour ends, and the "
        "document's worked example confirms a window a–b is the physical span a to b."),
    Convention.interval_end_labels.value: (
        "Interval end labels", "Listed times are meter end-stamps: a window a–b starts one "
                               "quarter-hour before a."),
    Convention.interval_end_ambiguous.value: (
        "Interval end, ambiguous", "The document says times are quarter-hour ends but gives "
        "no example, so the first quarter-hour of every window is uncertain."),
    Convention.assumed.value: (
        "Not stated", "The document does not say; the physical span a to b is assumed."),
}
CODE_TEXT = {
    "GROUNDING_NOT_FOUND": "Quote not found on page",
    "GROUNDING_VALUE_MISMATCH": "Value disagrees with its quote",
    "CELL_MISALIGNED": "Quote in a different table cell",
    "CROSS_CHECK_DISAGREE": "Text and vision readings differ",
    "COVERAGE_GAP": "Table cell missing",
    "TIME_INVALID": "Start not before end",
    "OFF_GRID": "Not on the quarter-hour grid",
    "OVERLAP": "Overlapping windows",
    "DUPLICATE_WINDOW": "Window listed twice",
    "DAILY_CAP_EXCEEDED": "Above the 10 h daily cap",
    "YEAR_MISMATCH": "Different validity year",
    "CORRECTION_NOTED": "Corrected publication",
    "CONVENTION_ASSUMED": "Timestamp convention not stated",
    "CONVENTION_AMBIGUOUS": "Timestamp convention ambiguous",
    "RULE_AMBIGUOUS": "Rule not decidable per date",
    "RULE_UNGROUNDED": "Rule quote not found",
    "RULE_VALUE_INVALID": "Rule value unreadable",
    "MODEL_ANOMALY": "Extractor note",
    "PAGE_UNREADABLE": "Page has no readable text",
    "CROSS_CHECK_UNAVAILABLE": "Vision cross-check not run",
    "YOY_DRIFT": "Changed since previous year",
    "SUPERSEDES": "Replaces an earlier version",
    "LEVEL_UNMAPPED": "Unknown grid level",
    "SEASON_UNMAPPED": "Unknown season",
    "LEVEL_NOT_LISTED": "Level missing from table list",
    "YEAR_UNGROUNDED": "Year quote not found",
    "TIME_UNPARSEABLE": "Times unreadable",
    "EMPTY_CELL_UNMAPPED": "Empty cell unmapped",
    "DSO_MISMATCH": "Different grid operator than labelled",
    "NO_HLZF_TABLE": "No HLZF table found",
    "DSO_NAME_FROM_PAGE": "Operator name taken from the page",
    "DSO_NAME_UNGROUNDED": "Operator name not found on the page",
}


def _num(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:g}"


def rule_text(kind: str, value: Any) -> str:
    """Plain-language rendering of a canonical rule value."""
    if kind == "workdays_only":
        return "Monday to Friday only" if value else "every day"
    if kind == "bridge_days":
        return {"working_day": "count as working days",
                "off_peak": "are off-peak",
                "off_peak_max_one": "at most one is off-peak; the document does not say which",
                "off_peak_max_one_per_week": "at most one per week is off-peak; not named",
                }.get(str(value), str(value))
    if kind == "holiday_handling" and isinstance(value, dict):
        states = value.get("states") or []
        joined = " and ".join(states) if states else "unspecified states"
        if value.get("mode") == "intersection" and len(states) > 1:
            text = f"holidays valid in all of {joined}"
        else:
            text = f"public holidays in {joined}"
        if value.get("exclude"):
            text += ", except " + ", ".join(value["exclude"])
        return text
    if kind == "christmas_period" and isinstance(value, dict):
        a, b = value.get("from"), value.get("to")
        if a and b:
            return f"{a[3:]}.{a[:2]}. to {b[3:]}.{b[:2]}. off-peak"
        return "between Christmas and New Year, dates not stated"
    if kind == "significance_threshold":
        return f"{_num(value)} %"
    if kind == "min_kw_diff":
        return f"{_num(value)} kW"
    if kind == "de_minimis_eur":
        return f"{_num(value)} €"
    return str(value)


RULE_NAMES = {"workdays_only": "Valid days", "bridge_days": "Bridge days",
              "holiday_handling": "Holidays", "christmas_period": "Christmas period",
              "significance_threshold": "Significance threshold",
              "min_kw_diff": "Minimum load shift", "de_minimis_eur": "Minimum saving",
              "other": "Other"}


def pct(minutes: int) -> float:
    return round(100 * minutes / 1440, 4)


STATES = [("BW", "Baden-Württemberg"), ("BY", "Bayern"), ("BE", "Berlin"),
          ("BB", "Brandenburg"), ("HB", "Bremen"), ("HH", "Hamburg"), ("HE", "Hessen"),
          ("MV", "Mecklenburg-Vorpommern"), ("NI", "Niedersachsen"),
          ("NW", "Nordrhein-Westfalen"), ("RP", "Rheinland-Pfalz"), ("SL", "Saarland"),
          ("SN", "Sachsen"), ("ST", "Sachsen-Anhalt"), ("SH", "Schleswig-Holstein"),
          ("TH", "Thüringen")]


def display_name(d: dict[str, Any]) -> str:
    """DSO name, or the file name of an upload whose DSO is not known (yet)."""
    return d.get("dso") or d.get("filename") or d.get("id") or "–"


def cost_estimate(conn: sqlite3.Connection) -> float | None:
    """Average live model cost per real document so far (None before the first live run)."""
    r = row(conn, "SELECT SUM(c.cost_usd) cost, COUNT(DISTINCT c.document_id) n FROM llm_calls c "
                  "JOIN documents d ON d.id = c.document_id WHERE c.source='live' AND "
                  "d.synthetic=0")
    return round(r["cost"] / r["n"], 3) if r and r["n"] else None


def documents(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    from ..pipeline import doc_summary

    out = []
    for d in doc_summary(conn):
        d = dict(d)
        d["ribbon"] = thumbnail(conn, d["id"]) if d["processed_at"] else []
        d["status_label"] = STATUS_LABELS.get(d["status"] or "", d["status"] or "–")
        d["convention_label"] = CONVENTION_TEXT.get(d["convention"] or "", ("–", ""))[0]
        out.append(d)
    return out


def thumbnail(conn: sqlite3.Connection, doc_id: str) -> list[dict[str, Any]]:
    """Rows for the small per-document fingerprint: level x season, bars in percent."""
    d = row(conn, "SELECT levels_listed FROM documents WHERE id=?", (doc_id,))
    levels = _levels(conn, doc_id, d)
    wins = rows(conn, "SELECT * FROM windows WHERE document_id=? AND active=1", (doc_id,))
    out = []
    for lv in levels:
        for s in Season:
            bars = [{"left": pct(w["start_min"]), "width": pct(max(15, w["end_min"]
                                                                   - w["start_min"])),
                     "status": w["status"]}
                    for w in wins if w["grid_level"] == lv and w["season"] == s.value
                    and w["end_min"] > w["start_min"]]
            out.append({"level": lv, "season": s.value, "bars": bars})
    return out


def _levels(conn: sqlite3.Connection, doc_id: str, d: dict | None) -> list[str]:
    listed = [ne for _, ne in loads(d["levels_listed"], [])] if d else []
    have = [r["grid_level"] for r in rows(
        conn, "SELECT DISTINCT grid_level FROM windows WHERE document_id=?", (doc_id,))]
    have += [r["grid_level"] for r in rows(
        conn, "SELECT DISTINCT grid_level FROM empty_cells WHERE document_id=?", (doc_id,))]
    s = {x for x in listed + have if x}
    return [lv for lv in GRID_LEVELS if lv in s]


def ribbon(conn: sqlite3.Connection, doc: dict[str, Any], selected: int | None
           ) -> list[dict[str, Any]]:
    """Big ribbon: one track per level x season, windows as bars on a 24 h axis."""
    conv = doc["convention"] or "assumed"
    shift = conv in (Convention.interval_end_ambiguous.value,
                     Convention.interval_end_labels.value)
    wins = rows(conn, "SELECT * FROM windows WHERE document_id=? AND active=1 "
                      "ORDER BY start_min", (doc["id"],))
    empties = {(e["grid_level"], e["season"]) for e in rows(
        conn, "SELECT * FROM empty_cells WHERE document_id=?", (doc["id"],))}
    flagged = {r["target"] for r in rows(
        conn, "SELECT target FROM issues WHERE document_id=? AND resolved=0 AND severity IN "
              "('error','warning')", (doc["id"],))}
    out = []
    for lv in _levels(conn, doc["id"], doc):
        seasons = []
        for s in Season:
            bars = []
            cell_flag = f"cell:{doc['id']}/{lv}/{s.value}" in flagged
            for w in wins:
                if w["grid_level"] != lv or w["season"] != s.value:
                    continue
                a, b = w["start_min"], w["end_min"]
                invalid = b <= a
                bars.append({
                    "wid": w["wid"], "status": w["status"], "invalid": invalid,
                    "left": pct(min(a, b)), "width": pct(max(15, abs(b - a))),
                    "boundary": pct(15) if (shift and not invalid) else 0,
                    "label": f"{fmt_min(a)}–{fmt_min(b)}",
                    "selected": w["wid"] == selected,
                    "flagged": w["entity_id"] in flagged or cell_flag,
                })
            hours = sum(max(0, x["width"]) for x in bars) * 24 / 100
            seasons.append({"season": s.value, "label": SEASON_LABELS[s.value],
                            "bars": bars, "empty": (lv, s.value) in empties,
                            "missing": not bars and (lv, s.value) not in empties,
                            "hours": round(hours, 2), "flagged": cell_flag})
        out.append({"level": lv, "name": GRID_LEVEL_NAMES[lv], "seasons": seasons})
    return out


def document_detail(conn: sqlite3.Connection, doc_id: str, selected: int | None
                    ) -> dict[str, Any] | None:
    d = row(conn, "SELECT * FROM documents WHERE id=?", (doc_id,))
    if not d:
        return None
    d["states"] = loads(d["states"], [])
    d["status_label"] = STATUS_LABELS.get(d["status"] or "", d["status"] or "–")
    d["convention_label"], d["convention_text"] = CONVENTION_TEXT.get(
        d["convention"] or "", ("–", ""))
    pages = rows(conn, "SELECT page_no, width, height, text_layer, source FROM pages WHERE "
                       "document_id=? ORDER BY page_no", (doc_id,))
    windows = rows(conn, "SELECT * FROM windows WHERE document_id=? AND active=1 "
                         "ORDER BY grid_level, CASE season WHEN 'winter' THEN 0 WHEN "
                         "'spring' THEN 1 WHEN 'summer' THEN 2 ELSE 3 END, start_min",
                   (doc_id,))
    for w in windows:
        w["label"] = f"{fmt_min(w['start_min'])}–{fmt_min(w['end_min'])}"
        w["season_label"] = SEASON_LABELS[w["season"]]
        w["status_label"] = _status_label(w)
        w["bbox"] = loads(w["bbox"])
    sel = next((w for w in windows if w["wid"] == selected), None)
    issues = rows(conn, "SELECT * FROM issues WHERE document_id=? ORDER BY resolved, CASE "
                        "severity WHEN 'error' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, iid",
                  (doc_id,))
    by_entity = {w["entity_id"]: w for w in windows}
    for i in issues:
        i["attribution"] = loads(i["attribution"])
        i["title"] = CODE_TEXT.get(i["code"], i["code"].replace("_", " ").capitalize())
        i["stage_label"] = STAGE_LABELS.get(i["suspected_stage"], i["suspected_stage"])
        if i["attribution"]:
            i["verdict_label"] = STAGE_LABELS.get(i["attribution"].get("stage", ""),
                                                  i["attribution"].get("stage"))
        tgt = i["target"] or ""
        i["window"] = by_entity.get(tgt)
        lv = se = None
        if tgt.startswith("cell:"):
            _, lv, se = tgt.split("/")[-3:]
            i["cell"] = f"{lv} {SEASON_LABELS.get(se, se)}"
            i["cell_windows"] = [w for w in windows if w["grid_level"] == lv
                                 and w["season"] == se and w["status"] != "rejected"]
        elif i["window"]:
            lv, se = i["window"]["grid_level"], i["window"]["season"]
        proposal = (i["attribution"] or {}).get("suggested_cell")
        auto = (i["attribution"] or {}).get("auto_correction") or {}
        if lv and i["resolved"] == 2 and auto.get("applied") and "before" in auto:
            i["undo"] = {"level": lv, "season": se, "windows": ",".join(auto["before"]),
                         "label": ", ".join(p.replace("-", "–") for p in auto["before"])
                         or "empty", "cell": f"{lv} {SEASON_LABELS.get(se, se)}"}
        if lv and proposal is not None and not i["resolved"]:
            i["apply"] = {"level": lv, "season": se, "windows": ",".join(proposal),
                          "label": ", ".join(p.replace("-", "–") for p in proposal) or "empty",
                          "cell": f"{lv} {SEASON_LABELS.get(se, se)}"}
    if sel is not None:
        sel["suggestion"] = suggestion_for(sel, windows, issues)
        mine = sel["label"].replace("–", "-")
        for i in issues:
            ap = i.get("apply")
            if ap and (ap["level"], ap["season"]) == (sel["grid_level"], sel["season"]):
                sel["cell_proposal"] = {**ap, "excludes_me": mine not in ap["windows"].split(",")}
                break
        if sel["origin"] == "auto":
            sel["auto_note"] = next(
                (i for i in issues if i["resolved"] == 2
                 and ((i["attribution"] or {}).get("auto_correction") or {}).get("applied")
                 and (i["target"] or "").endswith(f"/{sel['grid_level']}/{sel['season']}")),
                None)
    rules = rows(conn, "SELECT * FROM rules WHERE document_id=? AND active=1 ORDER BY rid",
                 (doc_id,))
    for r in rules:
        r["value"] = loads(r["value"])
        r["text"] = rule_text(r["kind"], r["value"])
        r["name"] = RULE_NAMES.get(r["kind"], r["kind"])
    siblings = rows(conn, "SELECT id, year, version_label, publication_date FROM documents "
                          "WHERE dso=? AND synthetic=? AND id != ? AND processed_at IS NOT NULL "
                          "ORDER BY year", (d["dso"], d["synthetic"], doc_id))
    # cost of live calls only: a cache hit is logged with the cost of the call it replays
    calls = row(conn, "SELECT COUNT(*) n, COALESCE(SUM(CASE WHEN source='live' THEN cost_usd "
                      "END),0) c, GROUP_CONCAT(DISTINCT "
                      "source) s, GROUP_CONCAT(DISTINCT model) m FROM llm_calls WHERE "
                      "document_id=?", (doc_id,))
    return {"doc": d, "pages": pages, "windows": windows, "selected": sel, "issues": issues,
            "rules": rules, "ribbon": ribbon(conn, d, selected), "siblings": siblings,
            "calls": calls,
            "open_issues": sum(1 for i in issues if not i["resolved"]
                               and i["severity"] in ("error", "warning"))}


def _status_label(w: dict[str, Any]) -> str:
    if w["status"] == "rejected" and w.get("origin") == "auto":
        return "Removed (consensus)"
    return STATUS_LABELS.get(w["status"], w["status"])


def _related(candidate: str, current: str) -> bool:
    """Same start, same end, or overlapping ("HH:MM-HH:MM" strings compare correctly)."""
    x, y = candidate.split("-")
    a, b = current.split("-")
    return x == a or y == b or (x < b and a < y)


def suggestion_for(sel: dict[str, Any], windows: list[dict[str, Any]],
                   issues: list[dict[str, Any]]) -> dict[str, Any] | None:
    """If an intervention proposed a reading of the selected window's cell, return the one
    window of that reading that replaces the selected value (for a one-click edit)."""
    cell = [w for w in windows if w["grid_level"] == sel["grid_level"]
            and w["season"] == sel["season"]]
    current = {w["label"].replace("–", "-") for w in cell}
    mine = sel["label"].replace("–", "-")
    for i in issues:
        att = i.get("attribution") or {}
        proposal = att.get("suggested_cell")
        if i.get("resolved") or not proposal:
            continue
        hits_cell = i.get("window") is sel or (
            (i.get("target") or "").endswith(f"/{sel['grid_level']}/{sel['season']}"))
        if not hits_cell or mine in proposal:
            continue
        new = [p for p in proposal if p not in current]
        cands = [p for p in new if _related(p, mine)]
        if len(cands) == 1:
            start, end = cands[0].split("-")
            source = ("OCR reading" if att.get("stage") == "parse"
                      else "majority of resamples")
            return {"start": start, "end": end, "source": source,
                    "label": f"{start}–{end}"}
    return None


def provenance(conn: sqlite3.Connection, entity: str) -> list[dict[str, Any]]:
    steps = Prov(conn).chain(entity)
    for s in steps:
        attrs = s.get("activity_attrs", {})
        s["facts"] = [(k, v) for k, v in attrs.items()
                      if k in ("channel", "sample", "promptVersion", "promptHash",
                               "responseSource", "costUsd", "latencyMs", "action", "comment",
                               "before", "after", "sha256", "url", "library", "reason",
                               "filename")
                      and v not in ("", None)
                      and not (k in ("costUsd", "latencyMs") and not v)]
    return steps


def window_history(conn: sqlite3.Connection, wid: int) -> list[dict[str, Any]]:
    w = row(conn, "SELECT * FROM windows WHERE wid=?", (wid,))
    if not w:
        return []
    base = w["entity_id"].split("@")[0]
    hist = rows(conn, "SELECT * FROM windows WHERE entity_id = ? OR entity_id LIKE ? "
                      "ORDER BY revision", (base, base + "@%"))
    for h in hist:
        h["label"] = f"{fmt_min(h['start_min'])}–{fmt_min(h['end_min'])}"
        h["status_label"] = _status_label(h)
    return hist
