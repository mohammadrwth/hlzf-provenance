"""S0 acquire: corpus registry, immutable downloads, change detection.

* `corpus.yaml`       what to fetch (url, dso, year, states, notes). Edited by hand.
* `corpus.lock.json`  what was fetched (sha256, size, HTTP validators, time). Committed, so a
                      later fetch can tell whether a DSO silently replaced a file.
* `data/raw/<id>.pdf` the bytes, never overwritten. A changed file is stored next to the old
                      one as `<id>.<sha12>.pdf` and reported, so corrections stay auditable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from . import fixtures
from .config import Settings
from .models import CorpusEntry

USER_AGENT = "hlzf-provenance/0.1 (research prototype; contact via GitHub)"
KEEP_HEADERS = ("content-type", "content-length", "last-modified", "etag", "date")


def load_registry(settings: Settings) -> list[CorpusEntry]:
    if not settings.corpus_file.exists():
        return []
    data = yaml.safe_load(settings.corpus_file.read_text(encoding="utf-8")) or {}
    return [CorpusEntry(**e) for e in data.get("documents", [])]


def load_lock(settings: Settings) -> dict[str, Any]:
    p = settings.root / "corpus.lock.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_lock(settings: Settings, lock: dict[str, Any]) -> None:
    p = settings.root / "corpus.lock.json"
    p.write_text(json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                 encoding="utf-8")


def load_corpus(settings: Settings, include_synthetic: bool = True,
                include_real: bool = True, include_uploads: bool | None = None
                ) -> list[CorpusEntry]:
    """corpus.yaml entries, then uploaded PDFs (default: with the real ones), then the
    synthetic fixtures."""
    from .uploads import load_uploads

    out: list[CorpusEntry] = []
    if include_real:
        lock = load_lock(settings)
        for e in load_registry(settings):
            pinned = lock.get(e.id, {}).get("sha256")
            out.append(e.model_copy(update={"sha256": pinned}) if pinned else e)
    if include_real if include_uploads is None else include_uploads:
        out.extend(load_uploads(settings))
    if include_synthetic:
        out.extend(fixtures.corpus_entries())
    return out


@dataclass
class FetchResult:
    id: str
    status: str  # new | unchanged | changed | failed | no-url
    detail: str = ""
    sha256: str | None = None


def fetch(settings: Settings, only: list[str] | None = None, accept_changes: bool = False,
          client: httpx.Client | None = None) -> list[FetchResult]:
    settings.ensure_dirs()
    lock = load_lock(settings)
    http = client or httpx.Client(timeout=60, follow_redirects=True,
                                  headers={"User-Agent": USER_AGENT})
    results: list[FetchResult] = []
    for e in load_registry(settings):
        if only and e.id not in only:
            continue
        if not e.url:
            results.append(FetchResult(e.id, "no-url", "no URL registered"))
            continue
        try:
            resp = http.get(e.url)
            resp.raise_for_status()
        except httpx.HTTPError as err:
            results.append(FetchResult(e.id, "failed", f"{type(err).__name__}: {err}"[:200]))
            continue
        body = resp.content
        if not body.startswith(b"%PDF"):
            results.append(FetchResult(e.id, "failed",
                                       f"not a PDF (content-type "
                                       f"{resp.headers.get('content-type')})"))
            continue
        sha = hashlib.sha256(body).hexdigest()
        target = settings.raw_dir / f"{e.id}.pdf"
        prev = lock.get(e.id, {}).get("sha256")
        meta = {"sha256": sha, "url": e.url, "fetched_at": datetime.now(UTC).isoformat(
            timespec="seconds"), "bytes": len(body),
            "headers": {k: resp.headers.get(k) for k in KEEP_HEADERS if resp.headers.get(k)}}
        if prev and prev != sha:
            alt = settings.raw_dir / f"{e.id}.{sha[:12]}.pdf"
            alt.write_bytes(body)
            if accept_changes:
                target.write_bytes(body)
                lock[e.id] = {**meta, "previous_sha256": prev}
                results.append(FetchResult(e.id, "changed", f"accepted; was {prev[:12]}", sha))
            else:
                results.append(FetchResult(
                    e.id, "changed", f"DSO file changed ({prev[:12]} -> {sha[:12]}); kept the "
                    f"old version, new bytes at {alt.name}. Re-run with --accept-changes.", sha))
            continue
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha:
            results.append(FetchResult(e.id, "unchanged", "", sha))
        else:
            target.write_bytes(body)
            results.append(FetchResult(e.id, "new", f"{len(body)} bytes", sha))
        lock[e.id] = {**lock.get(e.id, {}), **meta}
    save_lock(settings, lock)
    return results


def import_file(settings: Settings, doc_id: str, path: Path) -> FetchResult:
    """Register a manually downloaded PDF (for URLs that block automated downloads)."""
    reg = {e.id for e in load_registry(settings)}
    if doc_id not in reg:
        return FetchResult(doc_id, "failed", "id not in corpus.yaml")
    body = path.read_bytes()
    if not body.startswith(b"%PDF"):
        return FetchResult(doc_id, "failed", "not a PDF")
    sha = hashlib.sha256(body).hexdigest()
    settings.ensure_dirs()
    (settings.raw_dir / f"{doc_id}.pdf").write_bytes(body)
    lock = load_lock(settings)
    lock[doc_id] = {"sha256": sha, "imported_from": path.name, "bytes": len(body),
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
    save_lock(settings, lock)
    return FetchResult(doc_id, "new", "imported manually", sha)
