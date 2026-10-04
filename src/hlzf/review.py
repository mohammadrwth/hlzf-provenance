"""S5 review: human decisions as PROV activities with a prov:Person agent.

Every decision creates a new revision entity of the value (`wasRevisionOf` the previous one);
nothing is updated in place, so the review history stays in the provenance graph.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from .grounding import ground_window
from .models import Line, PageText, Status, fmt_min
from .prov import PERSON, Prov, now_iso
from .store import dumps, row, rows
from .textnorm import parse_time

# issues.resolved: 0 open, 1 decided by a person, 2 settled by the consensus correction
RESOLVED_BY_PERSON = 1
RESOLVED_BY_CONSENSUS = 2
SEASON_HEADER = {"winter": "Winter", "spring": "Frühling", "summer": "Sommer",
                 "autumn": "Herbst"}
UNDECIDED = ("auto-ok", "needs-review", "auto-corrected")


class ReviewError(ValueError):
    pass


def person_agent(prov: Prov, name: str) -> str:
    """The PROV agent of a person (reviewer, uploader), one per name."""
    rid = "".join(ch if ch.isalnum() else "-" for ch in name.strip().lower()) or "reviewer"
    return prov.agent(f"hlzf:agent/person/{rid}", PERSON, name.strip() or "reviewer")


_person = person_agent


def _active_window(conn: sqlite3.Connection, wid: int) -> dict[str, Any]:
    w = row(conn, "SELECT * FROM windows WHERE wid=?", (wid,))
    if not w:
        raise ReviewError(f"window {wid} does not exist")
    if not w["active"]:
        raise ReviewError(f"window {wid} has been superseded by a later revision")
    return w


def review_window(conn: sqlite3.Connection, wid: int, action: str, reviewer: str,
                  comment: str = "", start: str | None = None, end: str | None = None
                  ) -> dict[str, Any]:
    """action: approve | edit | reject. Edit and reject require a comment."""
    if action not in ("approve", "edit", "reject"):
        raise ReviewError("action must be approve, edit or reject")
    if action in ("edit", "reject") and not comment.strip():
        raise ReviewError(f"{action} requires a comment")
    old = _active_window(conn, wid)
    new_start, new_end = old["start_min"], old["end_min"]
    if action == "edit":
        s = parse_time(start or "") if start else old["start_min"]
        e = parse_time(end or "") if end else old["end_min"]
        if s is None or e is None:
            raise ReviewError("start/end must be HH:MM")
        if not (0 <= s < e <= 1440) or s % 15 or e % 15:
            raise ReviewError("window must satisfy start < end on the 15-minute grid")
        if (s, e) == (old["start_min"], old["end_min"]):
            raise ReviewError("edit does not change the window; use approve")
        new_start, new_end = s, e
    status = {"approve": Status.verified, "edit": Status.edited,
              "reject": Status.rejected}[action].value

    prov = Prov(conn)
    rev = old["revision"] + 1
    base = old["entity_id"].split("@")[0]
    new_ent = f"{base}@r{rev}"
    ts = now_iso()
    act = prov.activity(f"hlzf:act/review/{old['document_id']}/{old['wid']}-r{rev}",
                        "hlzf:Review", action, started_at=ts, ended_at=ts, action=action,
                        comment=comment.strip(),
                        before=f"{fmt_min(old['start_min'])}-{fmt_min(old['end_min'])}",
                        after=f"{fmt_min(new_start)}-{fmt_min(new_end)}")
    prov.associated(act, _person(prov, reviewer), "reviewer")
    prov.used(act, old["entity_id"], "reviewed")
    prov.entity(new_ent, "hlzf:Window",
                f"{old['grid_level']} {old['season']} {fmt_min(new_start)}-{fmt_min(new_end)}"
                f" [{status}]", status=status, page=old["page"], quote=old["quote"])
    prov.generated(new_ent, act)
    prov.revision(new_ent, old["entity_id"], act)

    conn.execute("UPDATE windows SET active=0 WHERE wid=?", (wid,))
    cur = conn.execute(
        "INSERT INTO windows (entity_id, document_id, grid_level, season, start_min, end_min, "
        "raw_level_label, page, quote, bbox, match_ratio, cell_aligned, status, active, "
        "revision, origin) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?, 'review')",
        (new_ent, old["document_id"], old["grid_level"], old["season"], new_start, new_end,
         old["raw_level_label"], old["page"], old["quote"], old["bbox"], old["match_ratio"],
         old["cell_aligned"], status, rev))
    cell = f"cell:{old['document_id']}/{old['grid_level']}/{old['season']}"
    conn.execute("UPDATE issues SET target=? WHERE target=?", (new_ent, old["entity_id"]))
    # A decision on a window settles the issues raised about it or its cell, once every
    # window of that cell has been decided.
    conn.execute("UPDATE issues SET resolved=1 WHERE target=?", (new_ent,))
    undecided = row(conn, "SELECT COUNT(*) n FROM windows WHERE document_id=? AND grid_level=? "
                          "AND season=? AND active=1 AND status IN (?,?,?)",
                    (old["document_id"], old["grid_level"], old["season"], *UNDECIDED))["n"]
    if not undecided:
        conn.execute("UPDATE issues SET resolved=1 WHERE target=?", (cell,))
    refresh_document_status(conn, old["document_id"])
    conn.commit()
    return {"wid": cur.lastrowid, "entity": new_ent, "status": status}


def acknowledge_issue(conn: sqlite3.Connection, iid: int, reviewer: str, comment: str
                      ) -> None:
    """Mark a document-level issue as reviewed (e.g. an ambiguous convention confirmed with
    the DSO). Recorded as a PROV activity; the issue stays visible as resolved."""
    if not comment.strip():
        raise ReviewError("acknowledging an issue requires a comment")
    issue = row(conn, "SELECT * FROM issues WHERE iid=?", (iid,))
    if not issue:
        raise ReviewError(f"issue {iid} does not exist")
    prov = Prov(conn)
    act = prov.activity(f"hlzf:act/ack/{issue['document_id']}/{iid}-{now_iso()}",
                        "hlzf:Review", "acknowledge issue", comment=comment.strip(),
                        issue=issue["code"])
    prov.associated(act, _person(prov, reviewer), "reviewer")
    prov.used(act, issue["entity_id"], "issue")
    conn.execute("UPDATE issues SET resolved=1 WHERE iid=?", (iid,))
    refresh_document_status(conn, issue["document_id"])
    conn.commit()


def refresh_document_status(conn: sqlite3.Connection, doc_id: str) -> str:
    """needs-review: a value or the extraction is in doubt. caveats: only the document itself
    leaves something open (document-level warnings blamed on the source)."""
    open_issues = rows(conn, "SELECT severity, suspected_stage, target FROM issues WHERE "
                             "document_id=? AND resolved=0 AND severity IN ('error','warning')",
                       (doc_id,))
    blocking = [i for i in open_issues if i["severity"] == "error" or i["target"]
                or i["suspected_stage"] != "source_document"]
    statuses = [r["status"] for r in rows(
        conn, "SELECT status FROM windows WHERE document_id=? AND active=1", (doc_id,))]
    if blocking or any(s == Status.needs_review.value for s in statuses):
        status = Status.needs_review.value
    elif open_issues:
        status = Status.caveats.value
    elif statuses and all(s in ("verified", "edited", "rejected") for s in statuses):
        status = Status.verified.value
    else:
        status = Status.auto_ok.value
    conn.execute("UPDATE documents SET status=? WHERE id=?", (status, doc_id))
    return status


# --- setting a whole table cell -------------------------------------------------------------

@dataclass
class CellChange:
    """Who sets a table cell to a reading, and how the result is marked."""
    agent: str            # PROV agent id
    role: str             # "reviewer" | "corrector"
    activity_type: str    # "hlzf:Review" | "hlzf:AutoCorrect"
    label: str
    status: str           # status of added or changed values
    origin: str           # windows.origin: "review" | "auto"
    resolved: int         # RESOLVED_BY_PERSON | RESOLVED_BY_CONSENSUS
    keep_status: str | None = None  # new status for values the reading keeps (None: as is)
    comment: str = ""


def person_change(conn: sqlite3.Connection, reviewer: str, comment: str) -> CellChange:
    return CellChange(agent=_person(Prov(conn), reviewer), role="reviewer",
                      activity_type="hlzf:Review", label="set cell",
                      status=Status.edited.value, origin="review",
                      resolved=RESOLVED_BY_PERSON, keep_status=Status.verified.value,
                      comment=comment.strip())


def _db_pages(conn: sqlite3.Connection, doc_id: str) -> dict[int, PageText]:
    return {r["page_no"]: PageText(page_no=r["page_no"], text=r["text"], width=r["width"],
                                   height=r["height"], text_layer=r["text_layer"],
                                   source=r["source"],
                                   lines=[Line(**ln) for ln in json.loads(r["lines"])])
            for r in rows(conn, "SELECT * FROM pages WHERE document_id=? ORDER BY page_no",
                          (doc_id,))}


def ground_in_cell(pages: dict[int, PageText], level_label: str, season: str, a: int, b: int
                   ) -> dict[str, Any] | None:
    """Find the printed range a-b in the cell (level row x season column) on any page."""
    for no, pg in pages.items():
        g = ground_window(pages, no, f"{fmt_min(a)} - {fmt_min(b)}", fmt_min(a), fmt_min(b),
                          level_label, SEASON_HEADER.get(season, season))
        if g.found and g.cell_aligned is not False:
            box = g.bbox or (0.0, 0.0, 0.0, 0.0)
            quote = " ".join(ln.text for ln in pg.lines
                             if ln.bbox[0] >= box[0] - 1 and ln.bbox[1] >= box[1] - 1
                             and ln.bbox[2] <= box[2] + 1 and ln.bbox[3] <= box[3] + 1)
            return {"page": no, "quote": quote or f"{fmt_min(a)}-{fmt_min(b)}", "bbox": g.bbox,
                    "match_ratio": g.ratio, "cell_aligned": g.cell_aligned}
    return None


def _entity_for(conn: sqlite3.Connection, base: str) -> tuple[str, int, str | None]:
    """(entity id, revision, previous entity) for a value newly added under `base`."""
    prev = row(conn, "SELECT entity_id, revision FROM windows WHERE entity_id=? OR entity_id "
                     "LIKE ? ORDER BY revision DESC LIMIT 1", (base, base + "@%"))
    if not prev:
        return base, 0, None
    rev = prev["revision"] + 1
    return f"{base}@r{rev}", rev, prev["entity_id"]


def apply_cell(conn: sqlite3.Connection, doc_id: str, level: str, season: str,
               reading: list[tuple[int, int]], change: CellChange,
               evidence: dict[tuple[int, int], dict[str, Any]] | None = None,
               used: list[tuple[str, str]] | tuple = (), derived_from: list[str] | tuple = ()
               ) -> dict[str, list[str]]:
    """Set the cell (level x season) to `reading`. Values the reading drops are rejected,
    values it adds are created, and a dropped value that overlaps or shares a bound with an
    added one becomes its revision. Every change is a new revision with PROV, nothing is
    edited in place. Does not commit."""
    prov = Prov(conn)
    cur = rows(conn, "SELECT * FROM windows WHERE document_id=? AND grid_level=? AND season=? "
                     "AND active=1 AND status != 'rejected' ORDER BY start_min",
               (doc_id, level, season))
    target = sorted(set(reading))
    have = {(w["start_min"], w["end_min"]): w for w in cur}
    keep = [have[k] for k in target if k in have]
    dropped = [w for k, w in have.items() if k not in target]
    added = [k for k in target if k not in have]
    pairs = []
    for w in list(dropped):
        a, b = w["start_min"], w["end_min"]
        m = next((k for k in added if k[0] == a or k[1] == b or (k[0] < b and a < k[1])), None)
        if m:
            pairs.append((w, m))
            dropped.remove(w)
            added.remove(m)

    def fmt(ks) -> str:
        return ", ".join(f"{fmt_min(a)}-{fmt_min(b)}" for a, b in sorted(ks)) or "(empty)"

    ts = now_iso()
    kind = "auto-cell" if change.origin == "auto" else "review-cell"
    act = prov.activity(f"hlzf:act/{kind}/{doc_id}/{level}/{season}-{ts}", change.activity_type,
                        change.label, started_at=ts, ended_at=ts, action="set cell",
                        comment=change.comment, before=fmt(have), after=fmt(target))
    prov.associated(act, change.agent, change.role)
    for ent, role in used:
        prov.used(act, ent, role)
    label_row = row(conn, "SELECT raw_level_label FROM windows WHERE document_id=? AND "
                          "grid_level=? AND raw_level_label IS NOT NULL LIMIT 1", (doc_id, level))
    level_label = label_row["raw_level_label"] if label_row else level
    pages: dict[int, PageText] | None = None

    def ev(k: tuple[int, int]) -> dict[str, Any]:
        nonlocal pages
        if evidence and k in evidence:
            return evidence[k]
        if pages is None:
            pages = _db_pages(conn, doc_id)
        return ground_in_cell(pages, level_label, season, *k) or {
            "page": None, "quote": "", "bbox": None, "match_ratio": 0.0, "cell_aligned": None}

    out: dict[str, list[str]] = {"before": [f"{fmt_min(a)}-{fmt_min(b)}" for a, b in sorted(have)],
                                 "changed": [], "added": [], "removed": [], "kept": []}
    new_entities: list[str] = []

    def insert(ent: str, rev: int, k: tuple[int, int], status: str, e: dict[str, Any],
               label: str | None) -> None:
        conn.execute(
            "INSERT INTO windows (entity_id, document_id, grid_level, season, start_min, "
            "end_min, raw_level_label, page, quote, bbox, match_ratio, cell_aligned, status, "
            "active, revision, origin) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)",
            (ent, doc_id, level, season, k[0], k[1], e.get("raw_level_label") or label,
             e.get("page"), e.get("quote", ""),
             dumps(e["bbox"]) if e.get("bbox") else None, e.get("match_ratio"),
             None if e.get("cell_aligned") is None else int(e["cell_aligned"]), status, rev,
             change.origin))
        new_entities.append(ent)

    def revise(old: dict[str, Any], k: tuple[int, int], status: str,
               e: dict[str, Any] | None) -> str:
        rev = old["revision"] + 1
        new_ent = f"{old['entity_id'].split('@')[0]}@r{rev}"
        e = e or {"page": old["page"], "quote": old["quote"], "bbox": loads_bbox(old["bbox"]),
                  "match_ratio": old["match_ratio"], "cell_aligned": old["cell_aligned"]}
        prov.used(act, old["entity_id"], "revised")
        prov.entity(new_ent, "hlzf:Window", f"{level} {season} {fmt_min(k[0])}-"
                    f"{fmt_min(k[1])} [{status}]", status=status, page=e.get("page"),
                    quote=e.get("quote", ""))
        prov.generated(new_ent, act)
        prov.revision(new_ent, old["entity_id"], act)
        if status != Status.rejected.value:
            for src in derived_from:
                prov.derived(new_ent, src, act)
        conn.execute("UPDATE windows SET active=0 WHERE wid=?", (old["wid"],))
        insert(new_ent, rev, k, status, e, old["raw_level_label"])
        conn.execute("UPDATE issues SET target=? WHERE target=?", (new_ent, old["entity_id"]))
        return new_ent

    for old, k in pairs:
        revise(old, k, change.status, ev(k))
        out["changed"].append(f"{fmt_min(old['start_min'])}-{fmt_min(old['end_min'])} -> "
                              f"{fmt_min(k[0])}-{fmt_min(k[1])}")
    for old in dropped:
        revise(old, (old["start_min"], old["end_min"]), Status.rejected.value, None)
        out["removed"].append(f"{fmt_min(old['start_min'])}-{fmt_min(old['end_min'])}")
    for k in added:
        hhmm = f"{fmt_min(k[0])}-{fmt_min(k[1])}".replace(":", "")
        ent, rev, prev = _entity_for(conn, f"hlzf:window/{doc_id}/{level}/{season}/{hhmm}")
        e = ev(k)
        prov.entity(ent, "hlzf:Window", f"{level} {season} {fmt_min(k[0])}-{fmt_min(k[1])} "
                    f"[{change.status}]", status=change.status, page=e.get("page"),
                    quote=e.get("quote", ""))
        prov.generated(ent, act)
        if prev:
            prov.revision(ent, prev, act)
            conn.execute("UPDATE windows SET active=0 WHERE entity_id=?", (prev,))
        for src in derived_from:
            prov.derived(ent, src, act)
        insert(ent, rev, k, change.status, e, level_label)
        out["added"].append(f"{fmt_min(k[0])}-{fmt_min(k[1])}")
    for w in keep:
        out["kept"].append(f"{fmt_min(w['start_min'])}-{fmt_min(w['end_min'])}")
        if change.keep_status:
            new_entities.append(revise(w, (w["start_min"], w["end_min"]), change.keep_status,
                                       None))
    if not target:
        conn.execute("INSERT OR IGNORE INTO empty_cells VALUES (?,?,?,?,?)",
                     (doc_id, level, season, level_label, None))
    # a person's decision also overrides an earlier consensus settlement
    settle = [f"cell:{doc_id}/{level}/{season}", *new_entities]
    open_states = (0, RESOLVED_BY_CONSENSUS) if change.resolved == RESOLVED_BY_PERSON else (0,)
    conn.execute(f"UPDATE issues SET resolved=? WHERE document_id=? AND resolved IN "
                 f"({','.join('?' * len(open_states))}) AND "
                 f"target IN ({','.join('?' * len(settle))})",
                 (change.resolved, doc_id, *open_states, *settle))
    return out


def loads_bbox(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) else v


def review_cell(conn: sqlite3.Connection, doc_id: str, level: str, season: str,
                windows: list[str], reviewer: str, comment: str) -> dict[str, list[str]]:
    """A person sets a whole table cell (e.g. to the reading an intervention suggested):
    the only way to add a window the extraction missed."""
    if not comment.strip():
        raise ReviewError("setting a cell requires a comment")
    reading = []
    for w in windows:
        try:
            a, b = (parse_time(x) for x in w.replace("–", "-").split("-"))
        except ValueError:
            raise ReviewError(f"{w!r} is not HH:MM-HH:MM") from None
        if a is None or b is None or not (0 <= a < b <= 1440) or a % 15 or b % 15:
            raise ReviewError(f"{w!r} is not a window on the 15-minute grid")
        reading.append((a, b))
    if not row(conn, "SELECT 1 FROM documents WHERE id=?", (doc_id,)):
        raise ReviewError(f"document {doc_id} does not exist")
    res = apply_cell(conn, doc_id, level, season, reading,
                     person_change(conn, reviewer, comment))
    refresh_document_status(conn, doc_id)
    conn.commit()
    return res


def reflag_windows(conn: sqlite3.Connection, doc_id: str) -> None:
    """Undecided values are needs-review exactly when an open warning or error targets them
    or their cell; consensus-corrected values keep their mark otherwise."""
    open_targets = {r["target"] for r in rows(
        conn, "SELECT target FROM issues WHERE document_id=? AND resolved=0 AND target IS NOT "
              "NULL AND severity IN ('error','warning')", (doc_id,))}
    for w in rows(conn, "SELECT * FROM windows WHERE document_id=? AND active=1 AND status IN "
                        "(?,?,?)", (doc_id, *UNDECIDED)):
        flagged = (w["entity_id"] in open_targets
                   or f"cell:{doc_id}/{w['grid_level']}/{w['season']}" in open_targets)
        if flagged:
            status = Status.needs_review.value
        elif w["origin"] == "auto":
            status = Status.auto_corrected.value
        else:
            status = Status.auto_ok.value
        if status != w["status"]:
            conn.execute("UPDATE windows SET status=? WHERE wid=?", (status, w["wid"]))
