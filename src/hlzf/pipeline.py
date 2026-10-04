"""Pipeline orchestration: S0 acquire -> S1 parse -> S2 extract (+ vision check) ->
S3 normalize -> S4 validate -> attribution. Every stage is a PROV activity with an agent.

A document is reprocessed only when its run key changes (PDF hash, prompt version, models,
code version), so `hlzf run` is idempotent and cheap to repeat.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf

from . import __version__, fixtures
from .attribution import (
    Verdict,
    cell_reading,
    channel_agreement,
    discriminate,
    discriminate_cross_check,
)
from .config import Settings
from .corpus import load_corpus
from .extract import ExtractionRun, Gateway
from .grounding import ground_window
from .llm import BudgetExceeded, InvalidModelOutput, ModelCallFailed, OfflineCacheMiss
from .models import CorpusEntry, PageText, Season, Severity, Stage, Status, fmt_min
from .normalize import Normalized, normalize
from .parse import parse_pdf
from .prompts import PROMPT_VERSION
from .prov import SOFTWARE, Prov
from .review import (
    RESOLVED_BY_CONSENSUS,
    CellChange,
    apply_cell,
    person_agent,
    reflag_windows,
    refresh_document_status,
)
from .store import connect, delete_document_data, dumps, loads, row, rows
from .uploads import canonical_dso, printed_name
from .uploads import pdf_path as upload_pdf_path
from .validate import Finding, cell_windows, validate

# Model-side failures degrade a single step (reported as an issue) instead of aborting the run.
SOFT_FAILURES = (OfflineCacheMiss, InvalidModelOutput, ModelCallFailed, BudgetExceeded)

INTERVENE_CODES = {"GROUNDING_NOT_FOUND", "GROUNDING_VALUE_MISMATCH", "CROSS_CHECK_DISAGREE",
                   "COVERAGE_GAP"}
AGREEMENT_CODES = {"TIME_INVALID", "OFF_GRID", "OVERLAP", "DAILY_CAP_EXCEEDED"}


@dataclass
class Runtime:
    settings: Settings
    conn: sqlite3.Connection
    prov: Prov
    gateway: Gateway
    log: Callable[[str], None] = print
    # progress for the web UI's upload jobs: parse | extract | check | correct
    step: Callable[[str], None] = lambda stage: None


def open_runtime(settings: Settings, log: Callable[[str], None] = print,
                 gateway: Gateway | None = None) -> Runtime:
    settings.ensure_dirs()
    conn = connect(settings.db_path)
    prov = Prov(conn)
    return Runtime(settings, conn, prov, gateway or Gateway(settings, conn, prov), log)


@dataclass
class DocResult:
    id: str
    status: str
    windows: int = 0
    issues: dict[str, int] = field(default_factory=dict)
    skipped: str | None = None


def pdf_path_for(rt: Runtime, entry: CorpusEntry) -> Path:
    if entry.source == "upload":
        return upload_pdf_path(rt.settings, entry.id)
    if entry.synthetic:
        p = rt.settings.synthetic_pdf_dir / f"{entry.id}.pdf"
        if not p.exists():
            fixtures.build_pdf(fixtures.spec_by_id(entry.id), p)
        return p
    return rt.settings.raw_dir / f"{entry.id}.pdf"


def run_key(sha: str, settings: Settings) -> str:
    return hashlib.sha256(json.dumps(
        [sha, PROMPT_VERSION, settings.extract_model, settings.check_model, settings.ocr_model,
         settings.resamples, settings.autocorrect, __version__]).encode()).hexdigest()[:16]


def window_entity(doc_id: str, key: tuple[str, str, int, int]) -> str:
    level, season, a, b = key
    hhmm = f"{fmt_min(a)}-{fmt_min(b)}".replace(":", "")
    return f"hlzf:window/{doc_id}/{level}/{season}/{hhmm}"


class _Lazy:
    """Resamples and the OCR swap are computed at most once per document, on demand."""

    def __init__(self, rt: Runtime, entry: CorpusEntry, pages: list[PageText], pdf: Path,
                 page_entities: list[str]):
        self.rt, self.entry, self.pages, self.pdf = rt, entry, pages, pdf
        self.page_entities = page_entities
        self.run = run_key(entry.sha256 or "", rt.settings)
        self._resamples: list[Normalized | None] | None = None
        self._swap: tuple[Normalized | None, dict[str, Any]] | bool = False
        self.vision: Normalized | None = None
        self.runs: list[ExtractionRun] = []
        self.ocr_page_entities: list[str] = []

    def resamples(self) -> list[Normalized | None]:
        if self._resamples is None:
            out: list[Normalized | None] = []
            pmap = {p.page_no: p for p in self.pages}
            n = self.rt.settings.resamples
            t0 = time.monotonic()
            # All resamples and the OCR of each page (for the parse swap) run in parallel;
            # the loop below then reads them from the cache.
            self.rt.conn.commit()  # do not hold the write lock while the models work
            self.rt.gateway.prefetch(
                [(self.entry, "text", self.pages, self.pdf, i) for i in range(1, n + 1)],
                ocr=(self.entry, self.pdf, [p.page_no for p in self.pages]))
            if not self.entry.synthetic:
                self.rt.log(f"  intervention: {n} resamples + OCR in parallel "
                            f"({time.monotonic() - t0:.0f} s)")
            for i in range(1, self.rt.settings.resamples + 1):
                try:
                    run = self.rt.gateway.extract(self.entry, "text", self.pages, self.pdf,
                                                  sample=i, page_entities=self.page_entities)
                    self.runs.append(run)
                    out.append(normalize(run.raw, pmap))
                except SOFT_FAILURES as err:
                    self.rt.log(f"  resample {i} unavailable: {err}")
                    out.append(None)
            self._resamples = out
        return self._resamples

    def swap(self) -> tuple[Normalized | None, dict[str, Any]]:
        """do(parse := GLM-OCR) with model and prompt held fixed. If OCR is unavailable, fall
        back to the vision reading and say that the intervention is confounded (the model
        changes too)."""
        if self._swap is False:
            meta: dict[str, Any] = {"intervention": "do(parse := glm-ocr)"}
            self.rt.conn.commit()
            try:
                ocr_pages = []
                act = self.rt.prov.activity(
                    f"hlzf:act/parse-ocr/{self.entry.id}/{self.run}",
                    "hlzf:Parse", "parse via GLM-OCR (intervention)")
                self.rt.prov.associated(act, self.rt.gateway.model_agent(
                    self.rt.settings.ocr_model, self.entry.synthetic), "parser")
                self.rt.prov.used(act, f"hlzf:doc/{self.entry.id}", "input")
                for p in self.pages:
                    op = self.rt.gateway.ocr_page(self.entry, self.pdf, p.page_no)
                    ocr_pages.append(op)
                    ent = self.rt.prov.entity(
                        f"hlzf:page-ocr/{self.entry.id}/{p.page_no}", "hlzf:Page",
                        f"page {p.page_no} (OCR)", source="ocr")
                    self.rt.prov.generated(ent, act)
                    self.ocr_page_entities.append(ent)
                run = self.rt.gateway.extract(self.entry, "ocr", ocr_pages, self.pdf, sample=0,
                                              page_entities=self.ocr_page_entities)
                self.runs.append(run)
                self._swap = (normalize(run.raw, {p.page_no: p for p in ocr_pages}), meta)
            except SOFT_FAILURES as err:
                self.rt.log(f"  OCR parse swap unavailable: {err}")
                if self.vision is not None:
                    meta = {"intervention": "do(parse := page image read by the vision model)",
                            "confounded": "OCR was unavailable; the extractor model changed "
                                          "together with the parse stage"}
                    self._swap = (self.vision, meta)
                else:
                    self._swap = (None, meta)
        return self._swap  # type: ignore[return-value]


def _symptom_for(f: Finding, text: Normalized):
    if f.target and f.target[0] == "window":
        key = f.target[1]
        return (lambda n: key in n.window_keys()), cell_reading(key[0], Season(key[1]))
    if f.target and f.target[0] == "cell":
        level, season = f.target[1]  # type: ignore[misc]
        reading = cell_reading(level, season)
        if f.issue.code == "COVERAGE_GAP":
            def listed_empty(n: Normalized) -> bool:
                return (level, season) in {(lv, s) for lv, s, _, _ in n.empties}

            def gap(n: Normalized) -> bool:
                return not reading(n) and not listed_empty(n)

            def cell_state(n: Normalized) -> str:
                if reading(n):
                    return "windows " + ", ".join(f"{fmt_min(a)}-{fmt_min(b)}"
                                                  for a, b in sorted(reading(n)))
                return "an explicit empty cell" if listed_empty(n) else "nothing (missing)"
            return gap, cell_state
        observed = reading(text)
        return (lambda n: reading(n) == observed), reading
    return None, None


def attribute(rt: Runtime, f: Finding, text: Normalized, vision: Normalized | None,
              lazy: _Lazy) -> Verdict | None:
    code = f.issue.code
    if code in AGREEMENT_CODES and f.target:
        level, season = (f.target[1][0], Season(f.target[1][1])) if f.target[0] == "window" \
            else f.target[1]  # type: ignore[misc]
        t = cell_windows(text, level, season)
        v = cell_windows(vision, level, season) if vision else None
        if f.issue.suspected_stage is Stage.source_document:
            return channel_agreement(t, v)
        return None
    if code not in INTERVENE_CODES:
        return None
    symptom, reading = _symptom_for(f, text)
    if symptom is None:
        return None
    if code == "CROSS_CHECK_DISAGREE" and vision is not None and reading is not None:
        return discriminate_cross_check(reading, text, vision, lazy.resamples(), lazy.swap)
    return discriminate(symptom, lazy.resamples(), lazy.swap, reading)


def process_document(rt: Runtime, entry: CorpusEntry, force: bool = False) -> DocResult:
    s, conn, prov = rt.settings, rt.conn, rt.prov
    pdf = pdf_path_for(rt, entry)
    if not pdf.exists():
        return DocResult(id=entry.id, status="not fetched",
                         skipped="PDF missing; run `hlzf fetch` (needs network)")
    data = pdf.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    entry = entry.model_copy(update={"sha256": sha})
    rkey = run_key(sha, s)
    existing = row(conn, "SELECT run_key, status FROM documents WHERE id=?", (entry.id,))
    if existing and existing["run_key"] == rkey and not force:
        return DocResult(id=entry.id, status=existing["status"], skipped="unchanged")

    rt.log(f"[{entry.id}] processing")
    started = time.monotonic()
    delete_document_data(conn, entry.id)
    tool = prov.agent(f"hlzf:agent/hlzf-provenance-{__version__}", SOFTWARE,
                      f"hlzf-provenance {__version__}", version=__version__)

    # S0 acquire (download from the DSO, or a file a person uploaded) ------------------------
    upload = entry.source == "upload"
    label = f"{entry.dso} {entry.year}" if entry.dso and entry.year else \
        (entry.filename or entry.id)
    doc_ent = prov.entity(f"hlzf:doc/{entry.id}", "hlzf:Document", label,
                          sha256=sha, url=entry.url or "", synthetic=entry.synthetic,
                          dso=entry.dso, year=entry.year, source=entry.source,
                          **({"filename": entry.filename or "",
                              "uploadedBy": entry.uploaded_by or ""} if upload else {}))
    fetched = row(conn, "SELECT fetched_at FROM documents WHERE id=?", (entry.id,))
    if upload:
        act = prov.activity(f"hlzf:act/upload/{entry.id}/{rkey}", "hlzf:Upload", "upload",
                            started_at=entry.uploaded_at, ended_at=entry.uploaded_at,
                            sha256=sha, filename=entry.filename or "")
        prov.associated(act, person_agent(prov, entry.uploaded_by or "unknown"), "uploader")
        prov.associated(act, tool, "receiver")
    else:
        act = prov.activity(f"hlzf:act/acquire/{entry.id}/{rkey}", "hlzf:Acquire",
                            "acquire", sha256=sha, url=entry.url or "synthetic fixture")
        prov.associated(act, tool, "fetcher")
    prov.generated(doc_ent, act)
    conn.execute(
        "INSERT INTO documents (id, entity_id, dso, year, url, sha256, fetched_at, path, "
        "states, synthetic, status, source, filename, uploaded_by) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET entity_id=excluded.entity_id, dso=excluded.dso, "
        "year=excluded.year, url=excluded.url, sha256=excluded.sha256, path=excluded.path, "
        "states=excluded.states, synthetic=excluded.synthetic, source=excluded.source, "
        "filename=excluded.filename, uploaded_by=excluded.uploaded_by, "
        "fetched_at=excluded.fetched_at",
        (entry.id, doc_ent, entry.dso, entry.year, entry.url, sha,
         entry.uploaded_at if upload else (fetched["fetched_at"] if fetched else None),
         str(pdf), dumps(entry.states), int(entry.synthetic), "processing", entry.source,
         entry.filename, entry.uploaded_by))

    # S1 parse --------------------------------------------------------------------------
    rt.step("parse")
    pages = parse_pdf(pdf)
    act = prov.activity(f"hlzf:act/parse/{entry.id}/{rkey}", "hlzf:Parse", "parse",
                        library=f"PyMuPDF {pymupdf.VersionBind}, pdfplumber "
                                f"{pdfplumber.__version__} (layout mode)")
    prov.associated(act, prov.agent(f"hlzf:agent/pymupdf-{pymupdf.VersionBind}", SOFTWARE,
                                    f"PyMuPDF {pymupdf.VersionBind}"), "parser")
    prov.used(act, doc_ent, "input")
    page_entities = []
    for i, p in enumerate(pages):
        if p.text_layer != "ok":
            try:
                ocr = rt.gateway.ocr_page(entry, pdf, p.page_no)
                pages[i] = ocr
                ocr_act = prov.activity(f"hlzf:act/ocr/{entry.id}/{p.page_no}/{rkey}",
                                        "hlzf:Parse", f"OCR page {p.page_no}",
                                        reason=f"text layer {p.text_layer}")
                prov.associated(ocr_act, rt.gateway.model_agent(s.ocr_model, entry.synthetic),
                                "parser")
                prov.used(ocr_act, doc_ent, "input")
                ent = prov.entity(f"hlzf:page/{entry.id}/{p.page_no}", "hlzf:Page",
                                  f"page {p.page_no}", source="ocr",
                                  textLayer=p.text_layer)
                prov.generated(ent, ocr_act)
                page_entities.append(ent)
                continue
            except SOFT_FAILURES as err:
                rt.log(f"  OCR unavailable for page {p.page_no}: {err}")
        ent = prov.entity(f"hlzf:page/{entry.id}/{p.page_no}", "hlzf:Page",
                          f"page {p.page_no}", source="text_layer", textLayer=p.text_layer)
        prov.generated(ent, act)
        page_entities.append(ent)
    for p in pages:
        conn.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?)",
                     (entry.id, p.page_no, p.text, dumps([ln.model_dump() for ln in p.lines]),
                      p.width, p.height, p.text_layer, p.source))
    pmap = {p.page_no: p for p in pages}

    # S2 extract (text) + vision cross-check ----------------------------------------------
    # Commit before waiting on the models, so the database is not locked for other writers
    # (review UI) meanwhile. A crash after this point leaves run_key NULL: reprocessed next run.
    conn.commit()
    rt.step("extract")
    t0 = time.monotonic()
    rt.gateway.prefetch([(entry, "text", pages, pdf, 0), (entry, "vision", pages, pdf, 0)])
    if not entry.synthetic:
        rt.log(f"  text extraction + vision check in parallel ({time.monotonic() - t0:.0f} s)")
    try:
        text_run = rt.gateway.extract(entry, "text", pages, pdf, page_entities=page_entities)
    except SOFT_FAILURES as err:
        rt.log(f"  extraction unavailable: {err}")
        conn.execute("UPDATE documents SET status=?, processed_at=NULL, run_key=NULL "
                     "WHERE id=?", ("not extracted", entry.id))
        conn.commit()
        return DocResult(id=entry.id, status="not extracted", skipped=str(err)[:160])
    text_norm = normalize(text_run.raw, pmap)
    identity_note: Finding | None = None
    if upload:
        entry, identity_note = _adopt_identity(conn, entry, text_norm, pmap)

    vision_norm: Normalized | None = None
    vision_note = None
    try:
        vision_run = rt.gateway.extract(entry, "vision", pages, pdf,
                                        page_entities=page_entities)
        vision_norm = normalize(vision_run.raw, pmap)
    except SOFT_FAILURES as err:
        vision_note = str(err)[:200]

    # S3 normalize ------------------------------------------------------------------------
    norm_act = prov.activity(f"hlzf:act/normalize/{entry.id}/{rkey}", "hlzf:Normalize",
                             "normalize", version=__version__)
    prov.associated(norm_act, tool, "normalizer")
    prov.used(norm_act, text_run.entity_id, "input")
    window_ids: dict[tuple[str, str, int, int], str] = {}
    for nw in text_norm.windows:
        w = nw.window
        ent = window_entity(entry.id, w.key)
        if w.key in window_ids:
            continue
        window_ids[w.key] = ent
        prov.entity(ent, "hlzf:Window",
                    f"{w.grid_level} {w.season.value} {fmt_min(w.start_min)}-"
                    f"{fmt_min(w.end_min)}", page=w.evidence.page, quote=w.evidence.quote)
        prov.generated(ent, norm_act)
        prov.derived(ent, text_run.entity_id, norm_act)
        conn.execute(
            "INSERT INTO windows (entity_id, document_id, grid_level, season, start_min, "
            "end_min, raw_level_label, page, quote, bbox, match_ratio, cell_aligned, status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ent, entry.id, w.grid_level, w.season.value, w.start_min, w.end_min,
             w.raw_level_label, w.evidence.page, w.evidence.quote,
             dumps(w.evidence.bbox) if w.evidence.bbox else None, w.evidence.match_ratio,
             None if w.evidence.cell_aligned is None else int(w.evidence.cell_aligned),
             Status.auto_ok.value))
    for level, season, label, page in text_norm.empties:
        conn.execute("INSERT OR IGNORE INTO empty_cells VALUES (?,?,?,?,?)",
                     (entry.id, level, season.value, label, page))
    for idx, nr in enumerate(text_norm.rules):
        r = nr.rule
        ent = f"hlzf:rule/{entry.id}/{idx}-{r.kind.value}"
        prov.entity(ent, "hlzf:Rule", f"{r.kind.value} = {r.value}", quote=r.evidence.quote)
        prov.generated(ent, norm_act)
        prov.derived(ent, text_run.entity_id, norm_act)
        conn.execute(
            "INSERT INTO rules (entity_id, document_id, kind, value, grid_level, page, quote, "
            "bbox, match_ratio, resolved, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ent, entry.id, r.kind.value, dumps(r.value), r.grid_level, r.evidence.page,
             r.evidence.quote, dumps(r.evidence.bbox) if r.evidence.bbox else None,
             r.evidence.match_ratio, int(r.resolved), Status.auto_ok.value))

    # S4 validate + attribution -------------------------------------------------------------
    conn.commit()
    rt.step("check")
    findings = validate(text_norm, entry, pmap, vision_norm)
    if identity_note:
        findings.append(identity_note)
    if vision_note:
        findings.append(Finding(issue=_info("CROSS_CHECK_UNAVAILABLE", Stage.extract,
                                            f"Vision cross-check not run: {vision_note}"),
                                target=None))
    val_act = prov.activity(f"hlzf:act/validate/{entry.id}/{rkey}", "hlzf:Validate",
                            "validate", version=__version__)
    prov.associated(val_act, tool, "validator")
    lazy = _Lazy(rt, entry, pages, pdf, page_entities)
    lazy.vision = vision_norm
    attributed: list[tuple[Finding, str, str, Verdict]] = []
    for n, f in enumerate(findings):
        verdict = attribute(rt, f, text_norm, vision_norm, lazy)
        issue = f.issue
        target = _target_id(entry.id, f, window_ids)
        iss_ent = prov.entity(f"hlzf:issue/{entry.id}/{n}-{issue.code}", "hlzf:Issue",
                              issue.code, severity=issue.severity.value,
                              suspectedStage=issue.suspected_stage.value)
        prov.generated(iss_ent, val_act)
        if target and target.startswith("hlzf:window"):
            prov.used(val_act, target, "checked")
            prov.derived(iss_ent, target, val_act)
        if verdict is not None:
            issue = _apply_verdict(issue, verdict)
            att_act = prov.activity(f"hlzf:act/attribute/{entry.id}/{n}/{rkey}",
                                    "hlzf:Intervention", verdict.method,
                                    **{k: v for k, v in verdict.to_dict().items()
                                       if k in ("method", "n", "valid", "persisted", "p_hat",
                                                "confidence")})
            prov.associated(att_act, tool, "attributor")
            prov.used(att_act, iss_ent, "issue")
            for run in lazy.runs:
                prov.used(att_act, run.entity_id, f"{run.channel}-sample-{run.sample}")
            att_ent = prov.entity(f"hlzf:attribution/{entry.id}/{n}", "hlzf:Attribution",
                                  f"{verdict.stage} ({verdict.confidence})",
                                  stage=verdict.stage, confidence=verdict.confidence)
            prov.generated(att_ent, att_act)
            prov.derived(att_ent, iss_ent, att_act)
            attributed.append((f, iss_ent, att_ent, verdict))
        conn.execute(
            "INSERT INTO issues (entity_id, document_id, target, code, message, severity, "
            "suspected_stage, attribution) VALUES (?,?,?,?,?,?,?,?)",
            (iss_ent, entry.id, target, issue.code, issue.message, issue.severity.value,
             issue.suspected_stage.value, dumps(issue.attribution) if issue.attribution
             else None))

    # S5 consensus correction ----------------------------------------------------------------
    rt.step("correct")
    if s.autocorrect and vision_norm is not None:
        sources = [vision_run.entity_id] + [r.entity_id for r in lazy.runs if r.channel == "ocr"]
        for f, iss_ent, att_ent, verdict in attributed:
            consensus_correct(rt, entry.id, f, iss_ent, att_ent, verdict, vision_norm, lazy,
                              pmap, tool, sources)
    reflag_windows(conn, entry.id)

    conn.execute(
        "UPDATE documents SET version_label=?, publication_date=?, correction_note=?, "
        "extracted_dso=?, extracted_year=?, convention=?, convention_page=?, "
        "convention_quote=?, levels_listed=?, anomalies=?, run_key=?, processed_at=?, "
        "status=? WHERE id=?",
        (text_norm.version_label or entry.version_label, text_norm.publication_date,
         text_norm.correction_note, text_norm.dso, text_norm.year, text_norm.convention.value,
         text_norm.convention_page, text_norm.convention_quote,
         dumps(text_norm.levels_listed), dumps(text_norm.anomalies), rkey,
         datetime.now(UTC).isoformat(timespec="seconds"), Status.needs_review.value,
         entry.id))
    status = refresh_document_status(conn, entry.id)
    conn.commit()
    if not entry.synthetic:
        rt.log(f"  done: {len(window_ids)} windows, {status} "
               f"({time.monotonic() - started:.0f} s)")
    counts: dict[str, int] = {}
    for r in rows(conn, "SELECT severity, COUNT(*) AS n FROM issues WHERE document_id=? "
                        "GROUP BY severity", (entry.id,)):
        counts[r["severity"]] = r["n"]
    return DocResult(id=entry.id, status=status, windows=len(window_ids), issues=counts)


DAILY_CAP_MIN = 600


def _adopt_identity(conn: sqlite3.Connection, entry: CorpusEntry, text: Normalized,
                    pages: dict[int, PageText]) -> tuple[CorpusEntry, Finding | None]:
    """An upload without DSO/year hints takes both from the document. The name must be printed
    on a page (a model can abbreviate it); it is then mapped onto an already known operator,
    so the upload joins its other versions."""
    note: Finding | None = None
    dso = entry.dso
    if not dso:
        raw = (text.dso or "").strip()
        name = printed_name(pages, raw)
        if name is None:
            name = raw
            if raw:
                from .models import Issue

                note = Finding(issue=Issue(
                    code="DSO_NAME_UNGROUNDED", severity=Severity.warning,
                    suspected_stage=Stage.extract,
                    message=f"The operator name read from the document ({raw!r}) is not "
                            "printed on its pages; check it."), target=None)
        elif name != raw:
            note = Finding(issue=_info(
                "DSO_NAME_FROM_PAGE", Stage.extract,
                f"The extractor returned {raw!r}; the operator name is taken from the printed "
                f"line {name!r}."), target=None)
        dso = canonical_dso(conn, name, exclude=entry.id)
    year = entry.year or (text.year or 0)
    conn.execute("UPDATE documents SET dso=?, year=? WHERE id=?", (dso, year, entry.id))
    return entry.model_copy(update={"dso": dso, "year": year}), note


def _sane(reading: list[tuple[int, int]]) -> bool:
    """The agreed reading breaks no time rule (else a person should look at the page)."""
    if any(not (0 <= a < b <= 1440) or a % 15 or b % 15 for a, b in reading):
        return False
    if any(x[1] > y[0] for x, y in zip(reading, reading[1:], strict=False)):
        return False
    return sum(b - a for a, b in reading) <= DAILY_CAP_MIN


def consensus_evidence(level: str, season: Season, reading: list[tuple[int, int]],
                       vision: Normalized, swapped: Normalized | None,
                       pages: dict[int, PageText], parse_fault: bool
                       ) -> tuple[dict[tuple[int, int], dict[str, Any]] | None, str]:
    """Where each agreed value is printed. A value counts only if its quote is on the page
    with exactly these times and in this table cell; the OCR page stands in for a text layer
    the intervention showed to be wrong (parse fault)."""
    if not _sane(reading):
        return None, "the agreed reading breaks a time rule"
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for a, b in reading:
        key = (level, season.value, a, b)
        vw = next((x for x in vision.windows if x.window.key == key), None)
        if vw is not None:
            r = vw.raw
            g = ground_window(pages, r.page, r.quote, r.start, r.end, r.level_quote,
                              r.season_quote, r.quote_end)
            if g.found and g.value_consistent is not False and g.cell_aligned is not False:
                out[(a, b)] = {"page": r.page, "bbox": g.bbox, "match_ratio": g.ratio,
                               "cell_aligned": g.cell_aligned, "raw_level_label": r.level_label,
                               "quote": f"{r.quote} … {r.quote_end}" if r.quote_end else r.quote}
                continue
        sw = next((x for x in swapped.windows if x.window.key == key), None) \
            if swapped is not None and parse_fault else None
        if sw is not None and sw.grounding.found and sw.grounding.value_consistent is not False:
            out[(a, b)] = {"page": sw.raw.page, "bbox": None, "match_ratio": sw.grounding.ratio,
                           "cell_aligned": None, "raw_level_label": sw.raw.level_label,
                           "quote": sw.raw.quote}
            continue
        return None, (f"{fmt_min(a)}-{fmt_min(b)} is not printed with these times in the "
                      f"{level} {season.value} cell of the text layer")
    return out, ""


def consensus_correct(rt: Runtime, doc_id: str, f: Finding, iss_ent: str, att_ent: str,
                      verdict: Verdict, vision: Normalized, lazy: _Lazy,
                      pages: dict[int, PageText], tool: str, sources: list[str]) -> None:
    """Majority of independent readings: when GLM-OCR text (through the same extractor) and
    the vision model agree on a disputed cell against the text channel, and every agreed
    value is printed in that cell, the cell is set to it. The change is a PROV revision by
    the software agent; the issue is settled (resolved = 2) and stays visible with the reason.
    Otherwise the suggestion waits for a person."""
    if f.issue.code != "CROSS_CHECK_DISAGREE" or not verdict.consensus or not f.target \
            or f.target[0] != "cell":
        return
    level, season = f.target[1]  # type: ignore[misc]
    reading = sorted(cell_windows(vision, level, season))
    swapped = lazy.swap()[0]
    evidence, why = consensus_evidence(level, season, reading, vision, swapped, pages,
                                       verdict.stage == Stage.parse.value)
    conn = rt.conn
    issue = row(conn, "SELECT iid, message, attribution FROM issues WHERE entity_id=?",
                (iss_ent,))
    att = loads(issue["attribution"], {}) or {}
    if evidence is None:
        att["auto_correction"] = {"applied": False, "reason": why}
        conn.execute("UPDATE issues SET attribution=? WHERE iid=?", (dumps(att), issue["iid"]))
        return
    cell = ", ".join(f"{fmt_min(a)}-{fmt_min(b)}" for a, b in reading) or "(empty)"
    change = CellChange(
        agent=tool, role="corrector", activity_type="hlzf:AutoCorrect",
        label="consensus correction", status=Status.auto_corrected.value, origin="auto",
        resolved=RESOLVED_BY_CONSENSUS,
        comment=(f"GLM-OCR and the vision model agree on {cell} against the text reading; "
                 f"every value is printed in this cell (verdict: {verdict.stage})."))
    res = apply_cell(conn, doc_id, level, season.value, reading, change, evidence=evidence,
                     used=[(iss_ent, "issue"), (att_ent, "attribution"),
                           *[(s, "reading") for s in sources]],
                     derived_from=sources)
    att["auto_correction"] = {"applied": True, "cell": cell, **res}
    conn.execute("UPDATE issues SET attribution=?, message=? WHERE iid=?",
                 (dumps(att), issue["message"] + f" Auto-corrected to {cell}: OCR and vision "
                  "agree, and every value is printed in this cell.", issue["iid"]))


def _info(code: str, stage: Stage, msg: str):
    from .models import Issue

    return Issue(code=code, severity=Severity.info, suspected_stage=stage, message=msg)


def _apply_verdict(issue, verdict: Verdict):
    att = verdict.to_dict()
    update: dict[str, Any] = {"attribution": att}
    if verdict.stage in {s.value for s in Stage}:
        update["suspected_stage"] = Stage(verdict.stage)
    if verdict.stage == "vision_misread":
        update["severity"] = Severity.info
        update["message"] = issue.message + " Resolved: vision channel misread (see attribution)."
    return issue.model_copy(update=update)


def _target_id(doc_id: str, f: Finding, window_ids: dict) -> str | None:
    if not f.target:
        return None
    kind, val = f.target
    if kind == "window":
        return window_ids.get(val)
    if kind == "cell":
        level, season = val  # type: ignore[misc]
        return f"cell:{doc_id}/{level}/{season.value}"
    return None


# --------------------------------------------------------------------------------------
# Cross-document checks
# --------------------------------------------------------------------------------------

def cross_document_checks(rt: Runtime) -> None:
    from .diff import diff_documents

    conn = rt.conn
    conn.execute("DELETE FROM issues WHERE code IN ('YOY_DRIFT', 'SUPERSEDES')")
    docs = rows(conn, "SELECT * FROM documents WHERE processed_at IS NOT NULL")
    # versions of one DSO; fictional fixtures never pair with real publications
    by_dso: dict[tuple[str, int], list[dict]] = {}
    for d in docs:
        by_dso.setdefault((d["dso"], d["synthetic"]), []).append(d)
    for dso_docs in by_dso.values():
        dso_docs.sort(key=lambda d: (d["year"], d["publication_date"] or ""))
        for a, b in zip(dso_docs, dso_docs[1:], strict=False):
            diff = diff_documents(conn, a["id"], b["id"])
            changed = [c for c in diff["cells"] if c["change"] != "unchanged"]
            if a["year"] == b["year"]:
                code = "SUPERSEDES"
                msg = (f"Supersedes {a['id']} (published {a['publication_date']}): "
                       f"{len(changed)} cell(s) changed: "
                       + "; ".join(f"{c['grid_level']} {c['season']}: {c['before']} -> "
                                   f"{c['after']}" for c in changed[:6]))
            else:
                code = "YOY_DRIFT"
                msg = (f"{len(changed)} of {len(diff['cells'])} cells changed vs. "
                       f"{a['year']} ({a['id']}).")
            conn.execute(
                "INSERT INTO issues (entity_id, document_id, target, code, message, severity, "
                "suspected_stage) VALUES (?,?,?,?,?,?,?)",
                (f"hlzf:issue/{b['id']}/{code}-{a['id']}", b["id"], None, code, msg,
                 Severity.info.value, Stage.source_document.value))
    conn.commit()


def run_corpus(rt: Runtime, include_synthetic: bool = True, include_real: bool = True,
               only: list[str] | None = None, force: bool = False) -> list[DocResult]:
    entries = load_corpus(rt.settings, include_synthetic=include_synthetic,
                          include_real=include_real)
    results = []
    for e in entries:
        if only and e.id not in only:
            continue
        try:
            results.append(process_document(rt, e, force=force))
        except BudgetExceeded as err:
            rt.log(f"[{e.id}] stopped: {err}")
            results.append(DocResult(id=e.id, status="budget", skipped=str(err)))
            break
    cross_document_checks(rt)
    return results


def doc_summary(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    out = []
    for d in rows(conn, "SELECT * FROM documents ORDER BY synthetic, dso, year, "
                        "publication_date"):
        # open issues only: settled ones (by a person or the consensus) stay in the record
        counts = {r["severity"]: r["n"] for r in rows(
            conn, "SELECT severity, COUNT(*) n FROM issues WHERE document_id=? AND resolved=0 "
                  "GROUP BY severity", (d["id"],))}
        w = row(conn, "SELECT SUM(status != 'rejected') n, SUM(status='needs-review') r, "
                      "SUM(origin='auto') a FROM windows WHERE document_id=? AND active=1",
                (d["id"],))
        stages = [r["suspected_stage"] for r in rows(
            conn, "SELECT suspected_stage FROM issues WHERE document_id=? AND severity IN "
                  "('error','warning')", (d["id"],))]
        out.append({**d, "errors": counts.get("error", 0), "warnings": counts.get("warning", 0),
                    "infos": counts.get("info", 0), "n_windows": w["n"] or 0,
                    "n_review": w["r"] or 0, "n_auto": w["a"] or 0,
                    "stages": sorted(set(stages)),
                    "levels_listed": loads(d["levels_listed"], [])})
    return out
