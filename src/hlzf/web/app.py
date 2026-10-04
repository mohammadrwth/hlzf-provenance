"""Review UI and API: FastAPI + Jinja2 + HTMX, no JS build step."""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__
from .. import uploads as uploads_mod
from ..config import Settings, load_settings
from ..diff import diff_documents
from ..export import mask_rows
from ..jobs import JobRunner, RuntimeFactory, job_view
from ..models import GRID_LEVELS
from ..parse import render_page
from ..prov import Prov
from ..query import QueryError, check
from ..review import ReviewError, acknowledge_issue, review_cell, review_window
from ..store import connect, loads, row, rows
from ..uploads import UploadError
from . import views

HERE = Path(__file__).parent


def create_app(settings: Settings | None = None,
               runtime_factory: RuntimeFactory | None = None) -> FastAPI:
    s = settings or load_settings()
    s.ensure_dirs()
    runner = JobRunner(s, runtime_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        runner.shutdown()

    app = FastAPI(title="HLZF provenance", version=__version__, lifespan=lifespan)
    app.state.jobs = runner
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(version=__version__, reviewer=s.reviewer,
                                 levels=GRID_LEVELS, season_labels=views.SEASON_LABELS,
                                 stage_labels=views.STAGE_LABELS,
                                 status_labels=views.STATUS_LABELS,
                                 display_name=views.display_name)

    def db() -> Iterator[Any]:
        conn = connect(s.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx)

    # --- pages -----------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, conn=Depends(db)):
        docs = views.documents(conn)
        return page(request, "documents.html",
                    uploaded=[d for d in docs if d["source"] == "upload"],
                    docs=[d for d in docs if not d["synthetic"] and d["source"] != "upload"],
                    synthetic=[d for d in docs if d["synthetic"]], nav="documents")

    @app.get("/doc/{doc_id}", response_class=HTMLResponse)
    def document(request: Request, doc_id: str, w: int | None = None, conn=Depends(db)):
        detail = views.document_detail(conn, doc_id, w)
        if detail is None:
            raise HTTPException(404, f"unknown document {doc_id}")
        prov = views.provenance(conn, detail["selected"]["entity_id"]) \
            if detail["selected"] else []
        history = views.window_history(conn, w) if detail["selected"] else []
        return page(request, "document.html", nav="documents", provenance=prov,
                    history=history, **detail)

    @app.get("/doc/{doc_id}/page/{page_no}.png")
    def page_png(doc_id: str, page_no: int, conn=Depends(db)):
        d = row(conn, "SELECT path FROM documents WHERE id=?", (doc_id,))
        if not d or not d["path"] or not Path(d["path"]).exists():
            raise HTTPException(404, "PDF not available locally")
        return FileResponse(render_page(Path(d["path"]), page_no, s.pages_dir / doc_id),
                            media_type="image/png")

    @app.get("/doc/{doc_id}/pdf")
    def original_pdf(doc_id: str, conn=Depends(db)):
        d = row(conn, "SELECT path, filename FROM documents WHERE id=?", (doc_id,))
        if not d or not d["path"] or not Path(d["path"]).exists():
            raise HTTPException(404, "PDF not available locally")
        name = d["filename"] or f"{doc_id}.pdf"
        return FileResponse(d["path"], media_type="application/pdf", headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(name)}"})

    # --- uploads ---------------------------------------------------------------------
    @app.get("/uploads", response_class=HTMLResponse)
    def uploads_page(request: Request, jobs: str = "", conn=Depends(db)):
        ids = [int(x) for x in jobs.split(",") if x.strip().isdigit()]
        batch = [v for v in (runner.view(conn, i) for i in ids) if v]
        history = [job_view(j, runner.live.get(j["jid"])) for j in rows(
            conn, "SELECT * FROM jobs ORDER BY jid DESC LIMIT 25")]
        docs = [d for d in views.documents(conn) if d["source"] == "upload"]
        return page(request, "uploads.html", nav="uploads", batch=batch,
                    auto=len(batch) == 1, history=history, docs=docs, live=s.live,
                    estimate=views.cost_estimate(conn), states=views.STATES,
                    max_mb=s.upload_max_mb, max_pages=s.upload_max_pages)

    @app.post("/upload")
    async def post_upload(files: list[UploadFile] = File(...), dso: str = Form(""),
                          year: str = Form(""), states: list[str] = Form([]),
                          url: str = Form(""), uploaded_by: str = Form("")):
        who = uploaded_by.strip() or s.reviewer
        y: int | None = None
        if year.strip():
            if not year.strip().isdigit():
                return RedirectResponse("/uploads?error=Year must be a number, e.g. 2026",
                                        status_code=303)
            y = int(year.strip())
        limit = int(s.upload_max_mb * 1024 * 1024) + 1
        jids: list[int] = []
        conn = connect(s.db_path)
        try:
            for f in files:
                name = Path(f.filename or "upload.pdf").name
                body = await f.read(limit)
                if not body and not f.filename:
                    continue
                try:
                    reg = uploads_mod.register(s, conn, body, name, who, dso=dso, year=y,
                                               states=states, url=url)
                except UploadError as err:
                    jids.append(runner.record(conn, "", name, who, "rejected", error=str(err)))
                    continue
                if reg.entry is None:
                    jids.append(runner.record(
                        conn, reg.duplicate_of or "", name, who, "duplicate",
                        result={"processed": reg.duplicate_processed}))
                    continue
                insp = reg.inspection
                jids.append(runner.submit(conn, reg.entry, extra={
                    "pages": insp.pages, "bytes": insp.bytes, "sha256": insp.sha256,
                    "looks_like_hlzf": insp.looks_like_hlzf} if insp else None))
        finally:
            conn.close()
        if not jids:
            return RedirectResponse("/uploads?error=Choose at least one PDF", status_code=303)
        return RedirectResponse(f"/uploads?jobs={','.join(map(str, jids))}#batch",
                                status_code=303)

    @app.get("/upload/job/{jid}", response_class=HTMLResponse)
    def job_fragment(request: Request, jid: int, auto: bool = False, conn=Depends(db)):
        j = runner.view(conn, jid)
        if j is None:
            raise HTTPException(404)
        resp = page(request, "_job.html", j=j, auto=auto)
        if auto and j["status"] == "done":
            resp.headers["HX-Redirect"] = f"/doc/{j['document_id']}"
        return resp

    @app.post("/upload/job/{jid}/retry")
    def retry_job(jid: int, conn=Depends(db)):
        j = row(conn, "SELECT * FROM jobs WHERE jid=?", (jid,))
        entry = uploads_mod.entry_for(s, j["document_id"]) if j else None
        if entry is None:
            raise HTTPException(404, "upload not found")
        extra = {k: v for k, v in (loads(j["result"], {}) or {}).items()
                 if k in ("pages", "bytes", "sha256", "looks_like_hlzf")}
        new = runner.submit(conn, entry, force=True, extra=extra)
        return RedirectResponse(f"/uploads?jobs={new}#batch", status_code=303)

    @app.post("/doc/{doc_id}/remove")
    def remove_upload(doc_id: str, reviewer: str = Form(""), conn=Depends(db)):
        try:
            uploads_mod.remove(s, conn, doc_id, reviewer or s.reviewer)
        except UploadError as err:
            return RedirectResponse(f"/doc/{doc_id}?error={err}", status_code=303)
        return RedirectResponse("/uploads", status_code=303)

    @app.post("/window/{wid}/review")
    def post_review(wid: int, action: str = Form(...), reviewer: str = Form(""),
                    comment: str = Form(""), start: str = Form(""), end: str = Form(""),
                    conn=Depends(db)):
        w = row(conn, "SELECT document_id FROM windows WHERE wid=?", (wid,))
        if not w:
            raise HTTPException(404)
        try:
            res = review_window(conn, wid, action, reviewer or s.reviewer, comment,
                                start or None, end or None)
        except ReviewError as err:
            return RedirectResponse(f"/doc/{w['document_id']}?w={wid}&error={err}#review",
                                    status_code=303)
        return RedirectResponse(f"/doc/{w['document_id']}?w={res['wid']}#review",
                                status_code=303)

    @app.post("/doc/{doc_id}/cell")
    def post_cell(doc_id: str, level: str = Form(...), season: str = Form(...),
                  windows: str = Form(""), reviewer: str = Form(""), comment: str = Form(""),
                  conn=Depends(db)):
        try:
            review_cell(conn, doc_id, level, season,
                        [w for w in windows.split(",") if w.strip()], reviewer or s.reviewer,
                        comment)
        except ReviewError as err:
            return RedirectResponse(f"/doc/{doc_id}?error={err}#issues", status_code=303)
        w = row(conn, "SELECT wid FROM windows WHERE document_id=? AND grid_level=? AND "
                      "season=? AND active=1 ORDER BY start_min LIMIT 1",
                (doc_id, level, season))
        return RedirectResponse(f"/doc/{doc_id}" + (f"?w={w['wid']}#review" if w else "#issues"),
                                status_code=303)

    @app.post("/issue/{iid}/ack")
    def post_ack(iid: int, reviewer: str = Form(""), comment: str = Form(""),
                 conn=Depends(db)):
        i = row(conn, "SELECT document_id FROM issues WHERE iid=?", (iid,))
        if not i:
            raise HTTPException(404)
        try:
            acknowledge_issue(conn, iid, reviewer or s.reviewer, comment)
        except ReviewError as err:
            return RedirectResponse(f"/doc/{i['document_id']}?error={err}#issues",
                                    status_code=303)
        return RedirectResponse(f"/doc/{i['document_id']}#issues", status_code=303)

    @app.get("/query", response_class=HTMLResponse)
    def query_page(request: Request, dso: str = "", level: str = "", ts: str = "",
                   ts_label: str = "start", conn=Depends(db)):
        dsos = rows(conn, "SELECT DISTINCT dso, synthetic FROM documents WHERE processed_at "
                          "IS NOT NULL ORDER BY synthetic, dso")
        result, error = None, None
        if dso and level and ts:
            result, error = _run_query(conn, dso, level, ts, ts_label)
        return page(request, "query.html", nav="query", dsos=dsos, result=result,
                    error=error, form={"dso": dso, "level": level or "NE5",
                                       "ts": ts or "2026-01-15T08:30+01:00",
                                       "ts_label": ts_label})

    @app.get("/query/result", response_class=HTMLResponse)
    def query_fragment(request: Request, dso: str, level: str, ts: str,
                       ts_label: str = "start", conn=Depends(db)):
        result, error = _run_query(conn, dso, level, ts, ts_label)
        return page(request, "_query_result.html", result=result, error=error)

    @app.get("/diff", response_class=HTMLResponse)
    def diff_page(request: Request, a: str = "", b: str = "", conn=Depends(db)):
        docs = rows(conn, "SELECT id, dso, year, version_label, publication_date, synthetic "
                          "FROM documents WHERE processed_at IS NOT NULL ORDER BY synthetic, "
                          "dso, year, publication_date")
        pairs = []
        by_dso: dict[tuple[str, int], list] = {}
        for d in docs:
            by_dso.setdefault((d["dso"], d["synthetic"]), []).append(d)
        for lst in by_dso.values():
            for x, y in zip(lst, lst[1:], strict=False):
                pairs.append((x, y))
        result, error = None, None
        if a and b:
            try:
                result = diff_documents(conn, a, b)
                result["ribbon_a"] = views.ribbon(conn, row(
                    conn, "SELECT * FROM documents WHERE id=?", (a,)), None)
                result["ribbon_b"] = views.ribbon(conn, row(
                    conn, "SELECT * FROM documents WHERE id=?", (b,)), None)
            except KeyError as err:
                error = str(err)
        return page(request, "diff.html", nav="diff", docs=docs, pairs=pairs, result=result,
                    error=error, a=a, b=b)

    # --- API -------------------------------------------------------------------------
    @app.get("/api/hlzf/check")
    def api_check(dso: str, level: str, ts: str, ts_label: str = "start",
                  explain: bool = False, conn=Depends(db)):
        try:
            when = datetime.fromisoformat(ts.replace(" ", "+"))
            return JSONResponse(json.loads(json.dumps(
                check(conn, dso, level, when, ts_label, explain), default=str)))
        except (QueryError, ValueError) as err:
            return JSONResponse({"error": str(err)}, status_code=400)

    @app.get("/api/documents")
    def api_documents(conn=Depends(db)):
        return [{k: d[k] for k in ("id", "dso", "year", "status", "convention", "n_windows",
                                   "n_review", "n_auto", "errors", "warnings", "synthetic",
                                   "source")}
                for d in views.documents(conn)]

    @app.get("/api/windows")
    def api_windows(document: str, conn=Depends(db)):
        return rows(conn, "SELECT wid, entity_id, grid_level, season, start_min, end_min, "
                          "status, page, quote FROM windows WHERE document_id=? AND active=1",
                    (document,))

    @app.get("/api/prov")
    def api_prov(entity: str | None = None, window_id: int | None = None, conn=Depends(db)):
        if window_id is not None:
            w = row(conn, "SELECT entity_id FROM windows WHERE wid=?", (window_id,))
            if not w:
                raise HTTPException(404)
            entity = w["entity_id"]
        if not entity:
            raise HTTPException(400, "give entity or window_id")
        return JSONResponse(Prov(conn).to_prov_json(entity))

    @app.get("/api/mask")
    def api_mask(dso: str, level: str, year: int, conn=Depends(db)):
        try:
            data, meta = mask_rows(conn, dso, level, year)
        except QueryError as err:
            raise HTTPException(400, str(err)) from err
        buf = io.StringIO()
        import csv

        wr = csv.DictWriter(buf, fieldnames=list(data[0].keys()))
        wr.writeheader()
        wr.writerows(data)
        name = f"hlzf-mask-{meta['document']}-{level}-{year}.csv"
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/healthz")
    def healthz() -> Response:
        return Response("ok")

    return app


def _run_query(conn, dso: str, level: str, ts: str, ts_label: str):
    try:
        when = datetime.fromisoformat(ts.strip().replace(" ", "+"))
        res = check(conn, dso, level, when, ts_label, explain=True)
        res["provenance"] = views.provenance(
            conn, (res["matching_window"] or {}).get("entity")
            or row(conn, "SELECT entity_id FROM documents WHERE id=?",
                   (res["document"],))["entity_id"])
        res["day_ribbon"] = _day_ribbon(conn, res)
        return res, None
    except (QueryError, ValueError) as err:
        return None, str(err)


def _day_ribbon(conn, res: dict[str, Any]) -> dict[str, Any]:
    local = datetime.fromisoformat(res["interval_local"][0])
    minute = local.hour * 60 + local.minute
    d = row(conn, "SELECT * FROM documents WHERE id=?", (res["document"],))
    tracks = views.ribbon(conn, d, (res["matching_window"] or {}).get("window_id"))
    track = next((t for t in tracks if t["level"] == res["grid_level"]), None)
    season = next((x for x in track["seasons"] if x["season"] == res["season"]), None) \
        if track else None
    return {"marker_left": views.pct(minute), "marker_width": views.pct(15),
            "season": season, "date": local.strftime("%A, %d %B %Y")}
