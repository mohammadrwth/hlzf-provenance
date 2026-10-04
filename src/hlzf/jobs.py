"""Background runs of uploaded PDFs for the web UI.

One worker thread processes uploads one after another (SQLite has one writer, and the LLM
budget guard counts per process). Live progress (stage, log lines) is kept in memory for the
polling UI; the outcome is written to the `jobs` table. Jobs that were queued or running when
the server stopped are marked `interrupted` on the next start and can be retried.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .config import Settings
from .models import CorpusEntry
from .pipeline import Runtime, cross_document_checks, open_runtime, process_document
from .store import connect, dumps, loads, row, rows

STAGES = [("stored", "Stored"), ("parse", "Pages read"),
          ("extract", "Text and vision extraction"), ("check", "Checks and interventions"),
          ("correct", "Consensus correction"), ("done", "Done")]
STAGE_INDEX = {k: i for i, (k, _) in enumerate(STAGES)}
TERMINAL = {"done", "failed", "interrupted", "duplicate", "rejected"}
RuntimeFactory = Callable[[Callable[[str], None]], Runtime]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Live:
    stage: str = "stored"
    log: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)


def explain_failure(reason: str) -> str:
    """Model-side failures in words a person can act on."""
    if "OfflineCacheMiss" in reason or reason.startswith(("No cached", "No live")):
        return ("No live model access (ZAI_API_KEY missing or HLZF_OFFLINE=1), and this "
                "file's answers are not in the response cache. Set ZAI_API_KEY in .env, "
                "restart `hlzf serve`, then retry.")
    if "budget" in reason.lower():
        return f"{reason} Raise LLM_BUDGET_USD in .env, restart, then retry."
    return reason


class JobRunner:
    def __init__(self, settings: Settings, runtime_factory: RuntimeFactory | None = None):
        self.s = settings
        self.factory: RuntimeFactory = runtime_factory or (
            lambda log: open_runtime(settings, log=log))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hlzf-upload")
        self.live: dict[int, Live] = {}
        self.futures: dict[int, Future] = {}
        self._lock = threading.Lock()
        conn = connect(settings.db_path)
        try:
            conn.execute("UPDATE jobs SET status='interrupted', finished_at=?, error=? WHERE "
                         "status IN ('queued', 'running')",
                         (now_iso(), "The server stopped while this upload was processed."))
            conn.commit()
        finally:
            conn.close()

    # --- queue ---------------------------------------------------------------------------
    def record(self, conn: sqlite3.Connection, document_id: str, filename: str,
               uploaded_by: str, status: str, error: str | None = None,
               result: dict[str, Any] | None = None) -> int:
        """A job row for an upload attempt that is not processed (duplicate, rejected)."""
        cur = conn.execute(
            "INSERT INTO jobs (document_id, filename, uploaded_by, status, stage, error, "
            "result, created_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (document_id, filename, uploaded_by, status, None, error,
             dumps(result) if result else None, now_iso(), now_iso()))
        conn.commit()
        return int(cur.lastrowid or 0)

    def submit(self, conn: sqlite3.Connection, entry: CorpusEntry, force: bool = False,
               extra: dict[str, Any] | None = None) -> int:
        cur = conn.execute(
            "INSERT INTO jobs (document_id, filename, uploaded_by, status, stage, result, "
            "created_at) VALUES (?,?,?,?,?,?,?)",
            (entry.id, entry.filename or entry.id, entry.uploaded_by, "queued", "stored",
             dumps(extra) if extra else None, now_iso()))
        conn.commit()
        jid = int(cur.lastrowid or 0)
        with self._lock:
            self.live[jid] = Live()
            self.futures[jid] = self.pool.submit(self._run, jid, entry, force, extra or {})
        return jid

    def wait(self, timeout: float = 120) -> None:
        """Block until every submitted job has finished (tests, CLI)."""
        for f in list(self.futures.values()):
            f.result(timeout=timeout)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    # --- worker --------------------------------------------------------------------------
    def _run(self, jid: int, entry: CorpusEntry, force: bool, extra: dict[str, Any]) -> None:
        live = self.live[jid]

        def log(msg: str) -> None:
            msg = msg.strip()
            if msg:
                live.log.append(msg)
                del live.log[:-200]

        def step(stage: str) -> None:
            live.stage = stage

        rt: Runtime | None = None
        started = now_iso()
        try:
            rt = self.factory(log)
            rt.step = step
            rt.conn.execute("UPDATE jobs SET status='running', started_at=? WHERE jid=?",
                            (started, jid))
            rt.conn.commit()
            res = process_document(rt, entry, force=force)
            if res.status not in ("not extracted", "not fetched"):
                cross_document_checks(rt)
            failed = res.status in ("not extracted", "not fetched")
            result = {**extra, **summarize(rt.conn, entry.id, started),
                      "seconds": round(time.monotonic() - live.started, 1)}
            live.stage = "done" if not failed else live.stage
            rt.conn.execute(
                "UPDATE jobs SET status=?, stage=?, log=?, error=?, result=?, finished_at=? "
                "WHERE jid=?",
                ("failed" if failed else "done", live.stage, dumps(live.log),
                 explain_failure(res.skipped or "") if failed else None, dumps(result),
                 now_iso(), jid))
            rt.conn.commit()
        except Exception as err:  # noqa: BLE001 - the job must end in a recorded state
            log(f"{type(err).__name__}: {err}")
            conn = rt.conn if rt else connect(self.s.db_path)
            try:
                conn.rollback()
                conn.execute("UPDATE jobs SET status='failed', log=?, error=?, finished_at=? "
                             "WHERE jid=?", (dumps(live.log), explain_failure(
                                 f"{type(err).__name__}: {err}"), now_iso(), jid))
                conn.commit()
            finally:
                if rt is None:
                    conn.close()
        finally:
            if rt is not None:
                rt.conn.close()

    # --- read ----------------------------------------------------------------------------
    def view(self, conn: sqlite3.Connection, jid: int) -> dict[str, Any] | None:
        j = row(conn, "SELECT * FROM jobs WHERE jid=?", (jid,))
        if not j:
            return None
        return job_view(j, self.live.get(jid))


def summarize(conn: sqlite3.Connection, doc_id: str, since: str) -> dict[str, Any]:
    d = row(conn, "SELECT dso, year, status, filename FROM documents WHERE id=?", (doc_id,)) or {}
    w = row(conn, "SELECT COUNT(*) n, SUM(status='needs-review') r, SUM(origin='auto') a "
                  "FROM windows WHERE document_id=? AND active=1 AND status != 'rejected'",
            (doc_id,)) or {}
    iss = {r["severity"]: r["n"] for r in rows(
        conn, "SELECT severity, COUNT(*) n FROM issues WHERE document_id=? AND resolved=0 "
              "GROUP BY severity", (doc_id,))}
    calls = row(conn, "SELECT COUNT(*) n, SUM(source='live') live, "
                      "COALESCE(SUM(CASE WHEN source='live' THEN cost_usd END), 0) cost "
                      "FROM llm_calls WHERE document_id=? AND ts >= ?", (doc_id, since)) or {}
    return {"dso": d.get("dso") or "", "year": d.get("year") or 0,
            "doc_status": d.get("status"), "windows": w.get("n") or 0,
            "review": w.get("r") or 0, "auto": w.get("a") or 0,
            "errors": iss.get("error", 0), "warnings": iss.get("warning", 0),
            "calls": calls.get("n") or 0, "live_calls": calls.get("live") or 0,
            "cost": round(float(calls.get("cost") or 0), 4)}


def job_view(j: dict[str, Any], live: Live | None) -> dict[str, Any]:
    j = dict(j)
    j["result"] = loads(j["result"], {}) or {}
    j["log"] = list(live.log) if live and j["status"] not in TERMINAL else loads(j["log"], [])
    stage = live.stage if live and j["status"] not in TERMINAL else (j["stage"] or "stored")
    j["stage"] = stage
    cur = STAGE_INDEX.get(stage, 0)
    terminal = j["status"] in TERMINAL
    j["terminal"] = terminal
    j["stages"] = [{"key": k, "label": label,
                    "state": ("done" if i < cur or (terminal and j["status"] == "done")
                              else "failed" if i == cur and j["status"] in ("failed",
                                                                            "interrupted")
                              else "active" if i == cur and not terminal else "todo")}
                   for i, (k, label) in enumerate(STAGES)]
    j["stage_no"], j["stage_label"] = cur + 1, STAGES[cur][1]
    if live and not terminal:
        j["elapsed"] = round(time.monotonic() - live.started)
    else:
        j["elapsed"] = j["result"].get("seconds")
    return j
