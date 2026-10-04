"""Exports an optimizer can consume.

`mask`: one row per quarter-hour of the year (UTC-indexed, DST-correct), with the three-valued
HLZF flag. 2026 has 35,040 quarter-hours; in local time 29.03. has 92 and 25.10. has 100.
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .query import BERLIN, QH, evaluate, load_context

UTC = ZoneInfo("UTC")


def quarter_hours(year: int):
    t = datetime(year, 1, 1, tzinfo=BERLIN).astimezone(UTC)
    end = datetime(year + 1, 1, 1, tzinfo=BERLIN).astimezone(UTC)
    while t < end:
        yield t
        t += QH


def mask_rows(conn: sqlite3.Connection, dso: str, level: str, year: int
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ctx = load_context(conn, dso, level, year)
    out = []
    counts = {"true": 0, "false": 0, "uncertain": 0}
    for t in quarter_hours(year):
        r = evaluate(ctx, t)
        flag = r["in_hlzf"]
        val = "uncertain" if flag == "uncertain" else ("1" if flag is True else "0")
        counts["true" if val == "1" else "false" if val == "0" else "uncertain"] += 1
        mw = r["matching_window"] or {}
        out.append({
            "interval_start_utc": t.isoformat().replace("+00:00", "Z"),
            "interval_start_local": t.astimezone(BERLIN).isoformat(),
            "in_hlzf": val,
            "season": r["season"],
            "day_type": r["day_type"],
            "window_id": mw.get("window_id", ""),
            "window_status": mw.get("status", ""),
        })
    meta = {"dso": ctx.document["dso"], "document": ctx.document["id"], "grid_level": ctx.level,
            "year": year, "rows": len(out), "counts": counts,
            "convention": ctx.convention.value}
    return out, meta


def write_mask(conn: sqlite3.Connection, dso: str, level: str, year: int, out: Path
               ) -> dict[str, Any]:
    data, meta = mask_rows(conn, dso, level, year)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
    return {**meta, "path": str(out)}
