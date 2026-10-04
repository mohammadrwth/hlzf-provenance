"""PDFs added by a person (web UI or `hlzf add`) instead of the curated corpus.yaml.

* `data/uploads/<id>.pdf`        the bytes as received, never modified.
* `data/uploads/registry.json`   who added which file when, plus optional hints (DSO, year,
                                 federal states, source URL). A file, not a database table,
                                 so `hlzf reset` / `hlzf demo` rebuild uploads like the corpus.

DSO and year are read from the document when no hint is given; a hint that contradicts the
document is reported as an issue rather than silently trusted. An uploaded file whose bytes
are already known (corpus or earlier upload) is not processed twice.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf
from rapidfuzz import fuzz

from . import fixtures
from .config import Settings
from .models import CorpusEntry, PageText
from .store import row, rows
from .textnorm import match_key

REGISTRY = "registry.json"
_LOCK = threading.Lock()
# words that mark a page as an HLZF publication (text layer only; scans give no signal)
HLZF_MARKERS = ("hochlast", "atypisch", "hlzf", "§ 19 abs. 2", "§19 abs. 2")
LEGAL_FORMS = {"gmbh", "ag", "kg", "co", "mbh", "se", "eg", "ug", "und", "kgaa"}
# where a model abbreviates: "SWW WUNSI..." or "Talwerk Energ… (Talwerk Energ…)"
_ABBREVIATION = re.compile(r"\.{2,}|…|\(")


class UploadError(ValueError):
    """The file cannot be accepted (shown to the person as is)."""


@dataclass
class Inspection:
    pages: int
    bytes: int
    sha256: str
    looks_like_hlzf: bool | None  # None: no text layer to judge by
    text_layer: bool


@dataclass
class Registered:
    entry: CorpusEntry | None  # None for a duplicate
    duplicate_of: str | None = None
    duplicate_processed: bool = False
    inspection: Inspection | None = None


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def inspect_pdf(body: bytes, settings: Settings) -> Inspection:
    limit = int(settings.upload_max_mb * 1024 * 1024)
    if len(body) > limit:
        raise UploadError(f"file is {len(body) / 1048576:.1f} MB; the limit is "
                          f"{settings.upload_max_mb:g} MB (HLZF_UPLOAD_MAX_MB)")
    if b"%PDF" not in body[:1024]:
        raise UploadError("not a PDF (no %PDF header)")
    try:
        doc = pymupdf.open(stream=body, filetype="pdf")
    except Exception as err:  # noqa: BLE001 - any parser failure means the same to the user
        raise UploadError(f"the PDF cannot be opened: {err}") from err
    with doc:
        if doc.needs_pass:
            raise UploadError("the PDF is password-protected")
        n = doc.page_count
        if n == 0:
            raise UploadError("the PDF has no pages")
        if n > settings.upload_max_pages:
            raise UploadError(f"{n} pages; the limit is {settings.upload_max_pages} "
                              "(HLZF publications have 1-3 pages; HLZF_UPLOAD_MAX_PAGES)")
        text = " ".join(page.get_text() for page in doc).lower()
    has_text = len(text.strip()) > 40
    looks = any(m in text for m in HLZF_MARKERS) if has_text else None
    return Inspection(pages=n, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                      looks_like_hlzf=looks, text_layer=has_text)


# --- registry ----------------------------------------------------------------------------

def _registry_path(settings: Settings) -> Path:
    return settings.uploads_dir / REGISTRY


def load_registry(settings: Settings) -> dict[str, dict[str, Any]]:
    p = _registry_path(settings)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_registry(settings: Settings, reg: dict[str, dict[str, Any]]) -> None:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    p = _registry_path(settings)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(p)


def pdf_path(settings: Settings, doc_id: str) -> Path:
    return settings.uploads_dir / f"{doc_id}.pdf"


def to_entry(doc_id: str, meta: dict[str, Any]) -> CorpusEntry:
    return CorpusEntry(
        id=doc_id, dso=meta.get("dso") or "", year=int(meta.get("year") or 0),
        url=meta.get("url") or None, states=list(meta.get("states") or []),
        notes=meta.get("note") or "", sha256=meta.get("sha256"), source="upload",
        filename=meta.get("filename"), uploaded_by=meta.get("uploaded_by"),
        uploaded_at=meta.get("uploaded_at"))


def load_uploads(settings: Settings) -> list[CorpusEntry]:
    return [to_entry(k, v) for k, v in sorted(
        load_registry(settings).items(), key=lambda kv: kv[1].get("uploaded_at", ""))]


def entry_for(settings: Settings, doc_id: str) -> CorpusEntry | None:
    meta = load_registry(settings).get(doc_id)
    return to_entry(doc_id, meta) if meta else None


# --- ids, names --------------------------------------------------------------------------

def slug(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        t = t.replace(a, b)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    words = [w for w in re.split(r"[^a-z0-9]+", t) if w and w not in LEGAL_FORMS]
    return "-".join(words)[:48].strip("-")


def _taken_ids(settings: Settings, conn: sqlite3.Connection | None) -> set[str]:
    from .corpus import load_registry as corpus_registry

    taken = set(load_registry(settings)) | {e.id for e in corpus_registry(settings)}
    taken |= {e.id for e in fixtures.corpus_entries()}
    if conn is not None:
        taken |= {r["id"] for r in rows(conn, "SELECT id FROM documents")}
    return taken


def make_id(dso: str, year: int | None, sha: str, taken: set[str]) -> str:
    base = slug(dso) if dso else ""
    if base and year:
        base = f"{base}-{year}"
    if not base:
        return f"upload-{sha[:10]}"
    return base if base not in taken else f"{base}-{sha[:6]}"


def printed_name(pages: dict[int, PageText], name: str) -> str | None:
    """The operator name as the document prints it: the extracted name if it is on a page;
    else the shortest printed line that starts with what the model wrote before an ellipsis
    (seen live: "SWW WUNSI... (SWW WUNSI...)" for the printed "SWW WUNSIEDEL GMBH"); else
    None. An upload's name comes from the document, so it must be grounded like any value."""
    name = (name or "").strip()
    if not name:
        return None
    ordered = [pages[k] for k in sorted(pages)]
    key = match_key(name)
    if any(key in match_key(pg.text) for pg in ordered):
        return name
    core = match_key(_ABBREVIATION.split(name)[0])
    if len(core) < 4:
        return None
    lines = [ln.text.strip() for pg in ordered for ln in pg.lines
             if len(ln.text.strip()) <= 80 and match_key(ln.text).startswith(core)]
    return min(lines, key=len) if lines else None


def _dso_key(name: str) -> str:
    return " ".join(w for w in slug(name).split("-") if w not in {"netz", "netze"})


def same_dso(a: str, b: str) -> bool:
    """Two spellings of one grid operator ("N-ERGIE Netz GmbH" / "N-ERGIE Netz")."""
    ka, kb = _dso_key(a), _dso_key(b)
    return bool(ka and kb) and fuzz.token_set_ratio(ka, kb) >= 90


def canonical_dso(conn: sqlite3.Connection, name: str, exclude: str = "") -> str:
    """The name an already known DSO is stored under, so an upload joins its other versions
    (year-over-year diff, supersession, the query's DSO lookup); else the name as read."""
    if not name:
        return ""
    known = rows(conn, "SELECT DISTINCT dso FROM documents WHERE synthetic=0 AND id != ? AND "
                       "dso != ''", (exclude,))
    exact = [k["dso"] for k in known if k["dso"].strip().lower() == name.strip().lower()]
    if exact:
        return exact[0]
    close = [k["dso"] for k in known if same_dso(k["dso"], name)]
    return close[0] if len(close) == 1 else name.strip()


# --- add / remove ------------------------------------------------------------------------

def _known_sha(settings: Settings, conn: sqlite3.Connection | None, sha: str
               ) -> tuple[str | None, bool]:
    if conn is not None:
        d = row(conn, "SELECT id, processed_at FROM documents WHERE sha256=? ORDER BY "
                      "processed_at IS NULL LIMIT 1", (sha,))
        if d:
            return d["id"], d["processed_at"] is not None
    for doc_id, meta in load_registry(settings).items():
        if meta.get("sha256") == sha:
            return doc_id, False
    from .corpus import load_lock

    for doc_id, meta in load_lock(settings).items():
        if meta.get("sha256") == sha:
            return doc_id, False
    return None, False


def register(settings: Settings, conn: sqlite3.Connection | None, body: bytes, filename: str,
             uploaded_by: str, dso: str = "", year: int | None = None,
             states: list[str] | None = None, url: str = "", note: str = "") -> Registered:
    """Validate, deduplicate and store an uploaded PDF. Raises UploadError."""
    insp = inspect_pdf(body, settings)
    with _LOCK:
        dup, processed = _known_sha(settings, conn, insp.sha256)
        if dup:
            return Registered(entry=None, duplicate_of=dup, duplicate_processed=processed,
                              inspection=insp)
        if year is not None and not 2000 <= year <= 2100:
            raise UploadError(f"year {year} is not plausible")
        doc_id = make_id(dso.strip(), year, insp.sha256, _taken_ids(settings, conn))
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        pdf_path(settings, doc_id).write_bytes(body)
        meta = {"filename": Path(filename or "upload.pdf").name[:200], "sha256": insp.sha256,
                "bytes": insp.bytes, "pages": insp.pages, "uploaded_at": now_iso(),
                "uploaded_by": uploaded_by.strip() or "unknown", "dso": dso.strip(),
                "year": year, "states": sorted({s.strip().upper() for s in states or []
                                                if s.strip()}),
                "url": url.strip(), "note": note.strip(),
                "looks_like_hlzf": insp.looks_like_hlzf}
        reg = load_registry(settings)
        reg[doc_id] = meta
        _save_registry(settings, reg)
    return Registered(entry=to_entry(doc_id, meta), inspection=insp)


def remove(settings: Settings, conn: sqlite3.Connection, doc_id: str, by: str) -> None:
    """Withdraw an upload: derived rows and the file go; the PROV graph keeps the record
    (append-only) plus a Withdraw activity naming who removed it."""
    from .prov import Prov
    from .review import person_agent
    from .store import delete_document_data

    with _LOCK:
        reg = load_registry(settings)
        if doc_id not in reg:
            raise UploadError(f"{doc_id} is not an uploaded document")
        if row(conn, "SELECT 1 FROM jobs WHERE document_id=? AND status IN ('queued', "
                     "'running')", (doc_id,)):
            raise UploadError(f"{doc_id} is still being processed; remove it when that is done")
        meta = reg.pop(doc_id)
        prov = Prov(conn)
        act = prov.activity(f"hlzf:act/withdraw/{doc_id}/{now_iso()}", "hlzf:Withdraw",
                            "withdraw upload", filename=meta.get("filename"))
        prov.associated(act, person_agent(prov, by), "remover")
        prov.used(act, f"hlzf:doc/{doc_id}", "withdrawn")
        delete_document_data(conn, doc_id)
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.execute("DELETE FROM jobs WHERE document_id=?", (doc_id,))
        conn.commit()
        pdf_path(settings, doc_id).unlink(missing_ok=True)
        _save_registry(settings, reg)
