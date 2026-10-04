"""Diff two publications of a DSO (two years, or original vs corrected version)."""

from __future__ import annotations

import sqlite3
from typing import Any

from .models import GRID_LEVELS, Season, fmt_min
from .store import loads, row, rows


def _cells(conn: sqlite3.Connection, doc_id: str) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for w in rows(conn, "SELECT * FROM windows WHERE document_id=? AND active=1 AND "
                        "status != 'rejected' ORDER BY start_min", (doc_id,)):
        out.setdefault((w["grid_level"], w["season"]), []).append(
            f"{fmt_min(w['start_min'])}-{fmt_min(w['end_min'])}")
    for e in rows(conn, "SELECT * FROM empty_cells WHERE document_id=?", (doc_id,)):
        out.setdefault((e["grid_level"], e["season"]), [])
    return out


def diff_documents(conn: sqlite3.Connection, a: str, b: str) -> dict[str, Any]:
    da = row(conn, "SELECT * FROM documents WHERE id=?", (a,))
    db = row(conn, "SELECT * FROM documents WHERE id=?", (b,))
    if not da or not db:
        raise KeyError(f"unknown document {a if not da else b}")
    ca, cb = _cells(conn, a), _cells(conn, b)
    cells = []
    order = {lv: i for i, lv in enumerate(GRID_LEVELS)}
    season_order = {s.value: i for i, s in enumerate(Season)}
    for key in sorted(set(ca) | set(cb), key=lambda k: (order.get(k[0], 9),
                                                        season_order.get(k[1], 9))):
        before, after = ca.get(key), cb.get(key)
        if before is None:
            change = "added"
        elif after is None:
            change = "removed"
        elif before == after:
            change = "unchanged"
        else:
            change = "changed"
        cells.append({"grid_level": key[0], "season": key[1], "change": change,
                      "before": ", ".join(before) if before else ("–" if before == [] else
                                                                   "not listed"),
                      "after": ", ".join(after) if after else ("–" if after == [] else
                                                               "not listed")})
    rules_a = {(r["kind"], r["grid_level"]): loads(r["value"]) for r in rows(
        conn, "SELECT * FROM rules WHERE document_id=? AND active=1", (a,))}
    rules_b = {(r["kind"], r["grid_level"]): loads(r["value"]) for r in rows(
        conn, "SELECT * FROM rules WHERE document_id=? AND active=1", (b,))}
    rule_changes = [{"kind": k[0], "grid_level": k[1], "before": rules_a.get(k),
                     "after": rules_b.get(k)}
                    for k in sorted(set(rules_a) | set(rules_b), key=str)
                    if rules_a.get(k) != rules_b.get(k)]
    return {
        "a": {"id": a, "dso": da["dso"], "year": da["year"], "published": da["publication_date"],
              "version": da["version_label"]},
        "b": {"id": b, "dso": db["dso"], "year": db["year"], "published": db["publication_date"],
              "version": db["version_label"], "correction_note": db["correction_note"]},
        "convention": {"before": da["convention"], "after": db["convention"]},
        "cells": cells,
        "rules": rule_changes,
    }
