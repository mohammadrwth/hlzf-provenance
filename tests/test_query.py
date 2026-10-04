import csv
from collections import Counter
from datetime import datetime

import pytest

from hlzf.export import mask_rows, write_mask
from hlzf.query import QueryError, check


def q(conn, dso, level, ts, label="start", **kw):
    return check(conn, dso, level, datetime.fromisoformat(ts), label, **kw)


def test_inside_and_outside_a_window(conn):
    assert q(conn, "alpenland", "NE5", "2026-01-15T08:30+01:00")["in_hlzf"] is True
    assert q(conn, "alpenland", "NE5", "2026-01-15T09:00+01:00")["in_hlzf"] is False
    assert q(conn, "alpenland", "NE5", "2026-01-15T07:30+01:00")["in_hlzf"] is False


def test_meter_timestamps_label_the_end(conn):
    # a meter value stamped 07:45 covers 07:30-07:45: before the 07:45 window start
    assert q(conn, "alpenland", "NE5", "2026-01-15T07:45+01:00", "end")["in_hlzf"] is False
    assert q(conn, "alpenland", "NE5", "2026-01-15T08:00+01:00", "end")["in_hlzf"] is True
    # alpenland states the end convention, so "dso" behaves like "end"
    assert q(conn, "alpenland", "NE5", "2026-01-15T08:00+01:00", "dso")["ts_label"] == "dso"
    assert q(conn, "alpenland", "NE5", "2026-01-15T07:45+01:00", "dso")["in_hlzf"] is False


def test_dst_spring_forward(conn):
    # Monday after the switch to CEST: 14:45Z is 16:45 local, start of NE5 spring window
    r = q(conn, "alpenland", "NE5", "2026-03-30T14:45+00:00")
    assert r["interval_local"][0] == "2026-03-30T16:45:00+02:00" and r["in_hlzf"] is True
    # the Friday before, still CET: 15:45Z is 16:45 local
    assert q(conn, "alpenland", "NE5", "2026-03-27T15:45+00:00")["in_hlzf"] is True
    assert q(conn, "alpenland", "NE5", "2026-03-27T14:45+00:00")["in_hlzf"] is False


def test_dst_fall_back(conn):
    # musterstadt NE5 autumn 17:00-19:00; after the switch back to CET 15:45Z is 16:45
    assert q(conn, "musterstadt", "NE5", "2026-10-26T15:45+00:00")["in_hlzf"] is False
    assert q(conn, "musterstadt", "NE5", "2026-10-26T16:00+00:00")["in_hlzf"] is True
    # before the switch (CEST) 15:00Z is already 17:00 local
    assert q(conn, "musterstadt", "NE5", "2026-10-23T15:00+00:00")["in_hlzf"] is True


def test_off_grid_and_naive_timestamps(conn):
    r = q(conn, "alpenland", "NE5", "2026-01-15T08:37+01:00")
    assert r["interval_local"][0].startswith("2026-01-15T08:30") and r["in_hlzf"] is True
    assert any("quarter-hour grid" in n for n in r["notes"])
    r = q(conn, "alpenland", "NE5", "2026-01-15T08:30")
    assert any("no offset" in n for n in r["notes"])


def test_ambiguous_convention_makes_first_quarter_hour_uncertain(conn):
    assert q(conn, "beispielstadt", "NE5", "2026-01-14T08:15+01:00")["in_hlzf"] == "uncertain"
    assert q(conn, "beispielstadt", "NE5", "2026-01-14T08:30+01:00")["in_hlzf"] is True


def test_corrected_publication_supersedes_original(conn):
    r = q(conn, "beispielstadt", "NE5", "2026-01-14T13:45+01:00")
    assert r["document"] == "beispielstadt-2026-korr" and r["in_hlzf"] is False


def test_calendar_rules_from_the_document(conn):
    r = q(conn, "alpenland", "NE3", "2026-01-06T08:00+01:00")
    assert r["in_hlzf"] is False and r["day_type"] == "holiday"
    r = q(conn, "musterstadt", "NE5", "2026-12-28T17:00+01:00")
    assert r["in_hlzf"] is False and r["day_type"] == "christmas"
    r = q(conn, "beispielstadt", "NE5", "2026-01-05T09:00+01:00")  # bridge, max. one
    assert r["in_hlzf"] == "uncertain"
    assert q(conn, "alpenland", "NE5", "2026-01-17T08:30+01:00")["day_type"] == "weekend"


def test_errors_and_uncovered_levels(conn):
    with pytest.raises(QueryError):
        q(conn, "no such dso", "NE5", "2026-01-15T08:30+01:00")
    with pytest.raises(QueryError):
        q(conn, "alpenland", "NE5", "2031-01-15T08:30+01:00")
    assert q(conn, "alpenland", "NE1", "2026-01-15T08:30+01:00")["in_hlzf"] is None


def test_explain_walks_back_to_the_document(conn):
    r = q(conn, "alpenland", "NE5", "2026-01-15T08:30+01:00", explain=True)
    chain = r["provenance"]
    assert chain[0]["entity"].startswith("hlzf:window/alpenland-2026/NE5/winter/0745-0900")
    assert chain[-1]["entity"] == "hlzf:doc/alpenland-2026"
    kinds = [s.get("activity_type") for s in chain]
    assert kinds[:4] == ["hlzf:Normalize", "hlzf:Extract", "hlzf:Parse", "hlzf:Acquire"]


def test_year_mask_is_dst_correct(conn, tmp_path):
    rows, meta = mask_rows(conn, "alpenland", "NE5", 2026)
    assert len(rows) == 35040 == meta["rows"]
    per_day = Counter(r["interval_start_local"][:10] for r in rows)
    assert per_day["2026-03-29"] == 92 and per_day["2026-10-25"] == 100
    assert per_day["2026-01-15"] == 96
    assert rows[0]["interval_start_local"] == "2026-01-01T00:00:00+01:00"
    out = write_mask(conn, "alpenland", "NE5", 2026, tmp_path / "m.csv")
    with open(out["path"], newline="") as fh:
        assert sum(1 for _ in csv.DictReader(fh)) == 35040
