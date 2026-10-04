"""`hlzf` command-line interface."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import corpus as corpus_mod
from . import fixtures
from . import uploads as uploads_mod
from .config import load_settings
from .diff import diff_documents
from .evaluate import blank_template, markdown_report, run_eval, write_readme_block
from .export import write_mask
from .pipeline import cross_document_checks, doc_summary, open_runtime, process_document, run_corpus
from .prov import Prov
from .query import QueryError, check
from .review import ReviewError, review_window
from .store import row

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Grounded, provenance-tracked HLZF extraction from DSO publications.")
console = Console()


def _rt(offline: bool | None = None):
    overrides = {} if offline is None else {"offline": offline}
    s = load_settings(**overrides)
    return open_runtime(s, log=lambda m: console.print(m, style="dim", markup=False,
                                                     highlight=False))


@app.command()
def fetch(only: list[str] = typer.Option(None, "--only", help="Document id(s)."),
          accept_changes: bool = typer.Option(False, help="Adopt changed DSO files.")) -> None:
    """Download the registered PDFs; detects files a DSO replaced since the last fetch."""
    s = load_settings()
    results = corpus_mod.fetch(s, only=only, accept_changes=accept_changes)
    t = Table("document", "result", "sha256", "detail")
    for r in results:
        style = {"failed": "red", "changed": "yellow"}.get(r.status, "")
        t.add_row(r.id, f"[{style}]{r.status}[/]" if style else r.status,
                  (r.sha256 or "")[:12], r.detail)
    console.print(t)
    failed = [r.id for r in results if r.status == "failed"]
    if failed:
        console.print(f"[red]{len(failed)} download(s) failed.[/] Download them in a browser "
                      "and register each with `hlzf import <id> <file.pdf>`.")


@app.command("import")
def import_pdf(doc_id: str, path: Path) -> None:
    """Register a manually downloaded PDF for a corpus.yaml entry."""
    r = corpus_mod.import_file(load_settings(), doc_id, path)
    console.print(f"{r.id}: {r.status} {r.detail} {(r.sha256 or '')[:12]}")


@app.command()
def add(files: list[Path] = typer.Argument(..., help="PDF file(s) to add."),
        dso: str = typer.Option("", help="Grid operator (default: read from the document)."),
        year: int = typer.Option(None, help="Validity year (default: read from the document)."),
        state: list[str] = typer.Option(None, "--state", help="Federal state code, e.g. BY."),
        url: str = typer.Option("", help="Where the PDF was published."),
        by: str = typer.Option(None, help="Uploader name (default: HLZF_REVIEWER)."),
        offline: bool | None = typer.Option(None, "--offline/--live")) -> None:
    """Add PDFs that are not in corpus.yaml and process them (same as the web upload)."""
    rt = _rt(offline)
    s = rt.settings
    who = by or s.reviewer
    added = []
    for f in files:
        try:
            reg = uploads_mod.register(s, rt.conn, f.read_bytes(), f.name, who, dso=dso,
                                       year=year, states=state or [], url=url)
        except (OSError, uploads_mod.UploadError) as err:
            console.print(f"[red]{f.name}[/]: {err}")
            continue
        if reg.entry is None:
            console.print(f"{f.name}: already known as [bold]{reg.duplicate_of}[/], skipped")
            continue
        insp = reg.inspection
        note = "" if not insp or insp.looks_like_hlzf is not False else \
            " [yellow](text never mentions Hochlast / § 19: maybe not an HLZF PDF)[/]"
        console.print(f"{f.name}: stored as [bold]{reg.entry.id}[/]{note}")
        added.append(reg.entry)
    for entry in added:
        res = process_document(rt, entry)
        if res.skipped:
            console.print(f"[yellow]{entry.id}[/]: {res.status} – {res.skipped}")
    if added:
        cross_document_checks(rt)
        status()


@app.command("remove-upload")
def remove_upload(doc_id: str, by: str = typer.Option(None)) -> None:
    """Withdraw an uploaded PDF (file and derived data; the PROV record stays)."""
    rt = _rt()
    try:
        uploads_mod.remove(rt.settings, rt.conn, doc_id, by or rt.settings.reviewer)
    except uploads_mod.UploadError as err:
        console.print(f"[red]{err}[/]")
        raise typer.Exit(1) from err
    console.print(f"{doc_id} removed")


@app.command("fixtures")
def build_fixtures() -> None:
    """(Re)generate the synthetic test PDFs."""
    s = load_settings()
    s.ensure_dirs()
    for doc_id, p in fixtures.build_all(s.synthetic_pdf_dir).items():
        console.print(f"{doc_id}: {p}")


@app.command()
def run(only: list[str] = typer.Option(None, "--only"),
        real: bool = typer.Option(True, help="Include real DSO documents."),
        synthetic: bool = typer.Option(True, help="Include the synthetic test corpus."),
        force: bool = typer.Option(False, help="Reprocess even if inputs are unchanged."),
        offline: bool | None = typer.Option(None, "--offline/--live",
                                            help="Force cache-only or allow API calls.")) -> None:
    """Run the pipeline (acquire, parse, extract, cross-check, normalize, validate, attribute)."""
    rt = _rt(offline)
    mode = "live" if rt.settings.live else "offline (cache only)"
    console.print(f"LLM mode: [bold]{mode}[/], budget ${rt.settings.budget_usd:.2f}")
    results = run_corpus(rt, include_synthetic=synthetic, include_real=real, only=only,
                         force=force)
    skipped = [r for r in results if r.skipped and r.skipped != "unchanged"]
    for r in skipped:
        console.print(f"[yellow]{r.id}[/]: {r.status} – {r.skipped}")
    status()


@app.command()
def status() -> None:
    """Summary table of processed documents."""
    rt = _rt()
    t = Table("document", "DSO", "year", "windows", "auto-fixed", "review", "err/warn",
              "convention", "suspected stages", "status")
    for d in doc_summary(rt.conn):
        if not d["processed_at"]:
            continue
        t.add_row(d["id"] + (" [dim](synthetic)[/]" if d["synthetic"] else ""), d["dso"],
                  str(d["year"]), str(d["n_windows"]), str(d["n_auto"]), str(d["n_review"]),
                  f"{d['errors']}/{d['warnings']}", d["convention"] or "–",
                  ", ".join(d["stages"]), d["status"])
    console.print(t)
    spent = rt.conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM llm_calls WHERE "
                            "source='live'").fetchone()[0]
    console.print(f"Live LLM spend recorded in this database: ${spent:.4f}")


@app.command()
def query(dso: str = typer.Option(..., help="DSO name or document id prefix."),
          level: str = typer.Option(..., help="NE1-NE7."),
          ts: str = typer.Option(..., help="ISO timestamp, e.g. 2026-01-15T08:30+01:00."),
          ts_label: str = typer.Option("start", help="start | end | dso (DSO's convention)."),
          explain: bool = typer.Option(False, help="Print the provenance chain."),
          as_json: bool = typer.Option(False, "--json")) -> None:
    """Is this quarter-hour inside an HLZF? Three-valued: true / false / uncertain."""
    rt = _rt()
    try:
        res = check(rt.conn, dso, level, datetime.fromisoformat(ts), ts_label, explain)
    except QueryError as err:
        console.print(f"[red]{err}[/]")
        raise typer.Exit(1) from err
    if as_json:
        console.print_json(json.dumps(res, default=str))
        return
    flag = res["in_hlzf"]
    color = {True: "red", False: "green", "uncertain": "yellow"}.get(flag, "white")
    console.print(f"in_hlzf: [bold {color}]{flag}[/]   {res['dso']} {res['grid_level']}  "
                  f"{res['interval_local'][0]} → {res['interval_local'][1]}")
    console.print(f"season {res['season']}, day: {res['day_reason']}")
    if res["matching_window"]:
        mw = res["matching_window"]
        console.print(f"window {mw['start']}-{mw['end']} (id {mw['window_id']}, {mw['status']})"
                      f", page {mw['page']}: “{mw['quote']}”")
    console.print(f"convention: {res['convention']} (ts_label={res['ts_label']})")
    for r in res["rules_applied"]:
        console.print(f"  rule {r['kind']} = {r['value']}  “{r['quote']}”", style="dim")
    for n in res["notes"]:
        console.print(f"  note: {n}", style="yellow")
    if explain:
        console.print("\nprovenance (value → source):", style="bold")
        for step in res["provenance"]:
            agents = ", ".join(a.get("label") or a["id"] for a in step.get("agents", []))
            act = step.get("activity_type", "–")
            console.print(f"  {step['entity']}  ← {act} [{agents}]")


@app.command()
def issues(doc_id: str = typer.Argument(..., help="Document id, e.g. n-ergie-2026."),
           all_: bool = typer.Option(False, "--all", help="Include info and resolved issues.")
           ) -> None:
    """Issues of one document: what was flagged, where, and which stage caused it."""
    from .store import loads, rows

    rt = _rt()
    if not row(rt.conn, "SELECT 1 FROM documents WHERE id=?", (doc_id,)):
        console.print(f"[red]unknown document {doc_id}[/]")
        raise typer.Exit(1)
    items = rows(rt.conn, "SELECT * FROM issues WHERE document_id=? ORDER BY CASE severity "
                          "WHEN 'error' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, iid", (doc_id,))
    shown = [i for i in items if all_ or (i["severity"] != "info" and not i["resolved"])]
    counts: dict[tuple[str, str], int] = {}
    for i in items:
        counts[(i["severity"], i["code"])] = counts.get((i["severity"], i["code"]), 0) + 1
    console.print(", ".join(f"{c}× {code} ({sev})" for (sev, code), c in counts.items())
                  or "no issues")
    t = Table(show_lines=True, expand=True)
    t.add_column("severity", no_wrap=True)
    t.add_column("issue", no_wrap=True)
    t.add_column("stage", no_wrap=True)
    t.add_column("verdict", ratio=2)
    t.add_column("message", ratio=3)
    for i in shown:
        att = loads(i["attribution"]) or {}
        verdict = ""
        if att:
            verdict = f"{att.get('stage')} ({att.get('confidence')}, {att.get('method')})"
            if att.get("suggestion"):
                verdict += f"\n{att['suggestion']}"
        style = {"error": "red", "warning": "yellow"}.get(i["severity"], "dim")
        t.add_row(f"[{style}]{i['severity']}[/]", i["code"], i["suspected_stage"], verdict,
                  i["message"])
    if shown:
        console.print(t)
    hidden = len(items) - len(shown)
    if hidden:
        console.print(f"{hidden} info or resolved issue(s) hidden; --all shows them", style="dim")


@app.command()
def prov(window_id: int = typer.Option(None, "--window-id"),
         entity: str = typer.Option(None, help="Any entity id."),
         out: Path = typer.Option(None, help="Write to file instead of stdout.")) -> None:
    """Export the provenance of one value as W3C PROV-JSON."""
    rt = _rt()
    if window_id is not None:
        w = row(rt.conn, "SELECT entity_id FROM windows WHERE wid=?", (window_id,))
        if not w:
            console.print(f"[red]no window {window_id}[/]")
            raise typer.Exit(1)
        entity = w["entity_id"]
    if not entity:
        console.print("[red]give --window-id or --entity[/]")
        raise typer.Exit(1)
    doc = Prov(rt.conn).to_prov_json(entity)
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    if out:
        out.write_text(text, encoding="utf-8")
        console.print(f"wrote {out}")
    else:
        print(text)


@app.command()
def diff(a: str, b: str) -> None:
    """Compare two publications (two years, or original vs corrected)."""
    rt = _rt()
    d = diff_documents(rt.conn, a, b)
    t = Table("level", "season", "change", d["a"]["id"], d["b"]["id"])
    for c in d["cells"]:
        if c["change"] == "unchanged":
            continue
        t.add_row(c["grid_level"], c["season"], c["change"], c["before"], c["after"])
    console.print(t)
    if d["b"].get("correction_note"):
        console.print(f"correction note in {b}: {d['b']['correction_note']}")
    for r in d["rules"]:
        console.print(f"rule {r['kind']} {r['grid_level'] or ''}: {r['before']} → {r['after']}")


@app.command("export-mask")
def export_mask(dso: str = typer.Option(...), level: str = typer.Option(...),
                year: int = typer.Option(...), out: Path = typer.Option(None)) -> None:
    """Write a 15-minute HLZF mask for a whole year (CSV, DST-correct)."""
    rt = _rt()
    out = out or rt.settings.exports_dir / f"hlzf-mask-{dso.split()[0].lower()}-{level}-{year}.csv"
    meta = write_mask(rt.conn, dso, level, year, out)
    console.print(meta)


@app.command("golden-template")
def golden_template(doc_id: str) -> None:
    """Write a blank labeling template to golden/<doc_id>.yaml."""
    rt = _rt()
    p = rt.settings.root / "golden" / f"{doc_id}.yaml"
    if p.exists():
        console.print(f"[yellow]{p} exists, not overwriting[/]")
        raise typer.Exit(1)
    p.parent.mkdir(exist_ok=True)
    p.write_text(blank_template(rt.conn, doc_id), encoding="utf-8")
    console.print(f"wrote {p}")


@app.command("eval")
def evaluate(write_readme: bool = typer.Option(False, help="Update README results block.")
             ) -> None:
    """Accuracy vs. golden labels, review share, cost; and attribution on injected faults."""
    rt = _rt()
    result = run_eval(rt.conn, rt.settings.root)
    report = markdown_report(result, doc_summary(rt.conn))
    console.print(report)
    if write_readme:
        write_readme_block(rt.settings.root / "README.md", report)
        console.print("README results block updated.")


@app.command()
def review(window_id: int, action: str = typer.Argument(..., help="approve | edit | reject"),
           comment: str = typer.Option(""), start: str = typer.Option(None),
           end: str = typer.Option(None), reviewer: str = typer.Option(None)) -> None:
    """Record a review decision (same effect as the UI buttons)."""
    rt = _rt()
    try:
        res = review_window(rt.conn, window_id, action, reviewer or rt.settings.reviewer,
                            comment, start, end)
    except ReviewError as err:
        console.print(f"[red]{err}[/]")
        raise typer.Exit(1) from err
    console.print(res)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the review UI and API."""
    import uvicorn

    from .web.app import create_app

    console.print(f"Review UI on http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command()
def reset(yes: bool = typer.Option(False, "--yes")) -> None:
    """Delete the SQLite database (the response cache and raw PDFs are kept)."""
    s = load_settings()
    if not yes:
        typer.confirm(f"Delete {s.db_path}?", abort=True)
    s.db_path.unlink(missing_ok=True)
    console.print("database removed")


@app.command()
def demo(serve_ui: bool = typer.Option(True, "--serve/--no-serve"),
         port: int = 8000) -> None:
    """One command: rebuild, run everything available, show results, start the UI."""
    s = load_settings()
    s.ensure_dirs()
    s.db_path.unlink(missing_ok=True)
    shutil.rmtree(s.synthetic_pdf_dir, ignore_errors=True)
    rt = _rt()
    have_real = any((s.raw_dir / f"{e.id}.pdf").exists()
                    for e in corpus_mod.load_registry(s))
    console.rule("HLZF provenance pipeline")
    console.print(f"LLM: {'live' if s.live else 'offline, replaying the committed cache'}; "
                  f"real PDFs present: {'yes' if have_real else 'no (run `hlzf fetch`)'}")
    run_corpus(rt)
    status()
    result = run_eval(rt.conn, s.root)
    console.print(markdown_report(result, doc_summary(rt.conn)))
    probe = check(rt.conn, "alpenland-2026", "NE5",
                  datetime.fromisoformat("2026-01-15T08:30+01:00"), "start")
    console.print(f"\nexample query alpenland NE5 2026-01-15 08:30 → in_hlzf="
                  f"{probe['in_hlzf']} ({probe['matching_window']})")
    if serve_ui:
        serve(port=port)


if __name__ == "__main__":
    app()
