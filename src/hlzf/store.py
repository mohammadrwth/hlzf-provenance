"""SQLite persistence. One file (`data/hlzf.db`), plain `sqlite3`, explicit schema."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id               TEXT PRIMARY KEY,
    entity_id        TEXT NOT NULL,
    dso              TEXT NOT NULL,
    year             INTEGER NOT NULL,
    url              TEXT,
    sha256           TEXT,
    fetched_at       TEXT,
    path             TEXT,
    states           TEXT NOT NULL DEFAULT '[]',
    synthetic        INTEGER NOT NULL DEFAULT 0,
    version_label    TEXT,
    publication_date TEXT,
    correction_note  TEXT,
    extracted_dso    TEXT,
    extracted_year   INTEGER,
    convention       TEXT,
    convention_page  INTEGER,
    convention_quote TEXT,
    levels_listed    TEXT NOT NULL DEFAULT '[]',
    anomalies        TEXT NOT NULL DEFAULT '[]',
    run_key          TEXT,
    processed_at     TEXT,
    status           TEXT,
    source           TEXT NOT NULL DEFAULT 'corpus',
    filename         TEXT,
    uploaded_by      TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    lines       TEXT NOT NULL,
    width       REAL NOT NULL,
    height      REAL NOT NULL,
    text_layer  TEXT NOT NULL,
    source      TEXT NOT NULL,
    PRIMARY KEY (document_id, page_no)
);

CREATE TABLE IF NOT EXISTS windows (
    wid             INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id       TEXT NOT NULL UNIQUE,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    grid_level      TEXT NOT NULL,
    season          TEXT NOT NULL,
    start_min       INTEGER NOT NULL,
    end_min         INTEGER NOT NULL,
    raw_level_label TEXT,
    page            INTEGER,
    quote           TEXT,
    bbox            TEXT,
    match_ratio     REAL,
    cell_aligned    INTEGER,
    status          TEXT NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1,
    revision        INTEGER NOT NULL DEFAULT 0,
    origin          TEXT NOT NULL DEFAULT 'extract'
);

CREATE TABLE IF NOT EXISTS empty_cells (
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    grid_level      TEXT NOT NULL,
    season          TEXT NOT NULL,
    raw_level_label TEXT,
    page            INTEGER,
    PRIMARY KEY (document_id, grid_level, season)
);

CREATE TABLE IF NOT EXISTS rules (
    rid          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id    TEXT NOT NULL UNIQUE,
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,
    value        TEXT,
    grid_level   TEXT,
    page         INTEGER,
    quote        TEXT,
    bbox         TEXT,
    match_ratio  REAL,
    resolved     INTEGER NOT NULL DEFAULT 1,
    status       TEXT NOT NULL,
    active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS issues (
    iid             INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id       TEXT NOT NULL UNIQUE,
    document_id     TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target          TEXT,
    code            TEXT NOT NULL,
    message         TEXT NOT NULL,
    severity        TEXT NOT NULL,
    suspected_stage TEXT NOT NULL,
    attribution     TEXT,
    resolved        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS extractions (
    entity_id   TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    channel     TEXT NOT NULL,
    sample      INTEGER NOT NULL,
    model       TEXT NOT NULL,
    source      TEXT NOT NULL,
    payload     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    document_id   TEXT,
    purpose       TEXT NOT NULL,
    model         TEXT NOT NULL,
    cache_key     TEXT NOT NULL,
    prompt_hash   TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0,
    latency_ms    INTEGER NOT NULL DEFAULT 0,
    source        TEXT NOT NULL
);

-- W3C PROV (PROV-DM core). Ids are qualified names in the `hlzf:` namespace.
CREATE TABLE IF NOT EXISTS entity (
    id    TEXT PRIMARY KEY,
    type  TEXT NOT NULL,
    label TEXT,
    attrs TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS activity (
    id         TEXT PRIMARY KEY,
    type       TEXT NOT NULL,
    label      TEXT,
    started_at TEXT,
    ended_at   TEXT,
    attrs      TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS agent (
    id    TEXT PRIMARY KEY,
    type  TEXT NOT NULL,
    label TEXT,
    attrs TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS used (
    activity_id TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    role        TEXT,
    PRIMARY KEY (activity_id, entity_id)
);
CREATE TABLE IF NOT EXISTS was_generated_by (
    entity_id   TEXT NOT NULL,
    activity_id TEXT NOT NULL,
    PRIMARY KEY (entity_id, activity_id)
);
CREATE TABLE IF NOT EXISTS was_derived_from (
    generated_id TEXT NOT NULL,
    used_id      TEXT NOT NULL,
    activity_id  TEXT,
    PRIMARY KEY (generated_id, used_id)
);
CREATE TABLE IF NOT EXISTS was_associated_with (
    activity_id TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    role        TEXT,
    PRIMARY KEY (activity_id, agent_id)
);
-- Background runs of uploaded PDFs (web UI). Live progress is held in memory by the job
-- runner; this table keeps the outcome, so it survives a server restart.
CREATE TABLE IF NOT EXISTS jobs (
    jid          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  TEXT NOT NULL,
    filename     TEXT NOT NULL,
    uploaded_by  TEXT,
    status       TEXT NOT NULL,          -- queued | running | done | failed | interrupted
    stage        TEXT,
    log          TEXT NOT NULL DEFAULT '[]',
    error        TEXT,
    result       TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS was_revision_of (
    new_id      TEXT NOT NULL,
    old_id      TEXT NOT NULL,
    activity_id TEXT,
    PRIMARY KEY (new_id, old_id)
);
"""


# Columns added after the first release; `connect` adds them to an existing database.
ADDED_COLUMNS = {
    "documents": {"source": "TEXT NOT NULL DEFAULT 'corpus'", "filename": "TEXT",
                  "uploaded_by": "TEXT"},
}


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    # The web UI runs uploads on a background thread while requests read and write: WAL lets
    # readers proceed during a write, and the timeout lets a writer wait for the other one.
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    for table, cols in ADDED_COLUMNS.items():
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if have:
            for name, decl in cols.items():
                if name not in have:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.executescript(SCHEMA)
    return conn


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def rows(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


def row(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    r = conn.execute(sql, tuple(params)).fetchone()
    return dict(r) if r else None


def delete_document_data(conn: sqlite3.Connection, document_id: str) -> None:
    """Drop derived rows for a document before it is reprocessed.

    PROV records are append-only and kept: a reprocessed document gets new activities (their
    ids carry the run key), and the old ones stay in the graph as history.
    """
    for table in ("pages", "windows", "empty_cells", "rules", "issues", "extractions"):
        conn.execute(f"DELETE FROM {table} WHERE document_id = ?", (document_id,))
