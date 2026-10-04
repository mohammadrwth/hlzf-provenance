"""S2 extract: one entry point for every model call the pipeline makes.

Channels
* text   GLM_EXTRACT_MODEL (glm-5.3, text-only) reads the parsed text layer.
* vision GLM_CHECK_MODEL (glm-5.3-flash, multimodal) reads rendered page images. This is the
         independent cross-check: a different evidence channel, not just a second sample.
* ocr    GLM_EXTRACT_MODEL on GLM-OCR output. Used by the parse-swap intervention, so only
         the parse stage changes while model and prompt stay fixed.

Synthetic test documents never reach the API; they are answered by `fixture-replay`, which
returns the scripted responses from `fixtures.py`.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__, fixtures
from .config import Settings
from .llm import CallRecord, LLMClient, sha256_text
from .models import CorpusEntry, Line, PageText, RawExtraction
from .parse import render_page
from .prompts import (
    EXTRACT_SYSTEM,
    PROMPT_VERSION,
    VISION_SYSTEM,
    text_user_message,
    vision_user_message,
)
from .prov import SOFTWARE, Prov

FIXTURE_MODEL = "fixture-replay"
MAX_VISION_PAGES = 4  # HLZF publications are one or two pages; cap image tokens

# (document, channel, pages, pdf path, sample)
Job = tuple[CorpusEntry, str, list[PageText], Path, int]


@dataclass
class ExtractionRun:
    raw: RawExtraction
    record: CallRecord
    channel: str
    sample: int
    model: str
    entity_id: str
    activity_id: str


class Gateway:
    def __init__(self, settings: Settings, conn: sqlite3.Connection, prov: Prov,
                 llm: LLMClient | None = None):
        self.s = settings
        self.conn = conn
        self.prov = prov
        self.llm = llm or LLMClient(settings, conn)

    # --- agents ----------------------------------------------------------------------
    def model_agent(self, model: str, synthetic: bool) -> str:
        if synthetic:
            return self.prov.agent(
                f"hlzf:agent/{FIXTURE_MODEL}", SOFTWARE, FIXTURE_MODEL,
                note="scripted responses for synthetic test documents (fixtures.py)",
                version=__version__)
        return self.prov.agent(f"hlzf:agent/{model}", SOFTWARE, model, provider="Z.ai",
                               api="openai-compatible chat completions, JSON mode")

    # --- extraction ------------------------------------------------------------------
    def extract(self, doc: CorpusEntry, channel: str, pages: list[PageText],
                pdf_path: Path, sample: int = 0, page_entities: list[str] | None = None,
                purpose_suffix: str = "") -> ExtractionRun:
        purpose = f"extract:{channel}"
        if doc.synthetic:
            payload = fixtures.scripted_response(doc.id, purpose, sample)
            raw = RawExtraction.model_validate(payload)
            model = FIXTURE_MODEL
            rec = CallRecord(model=model, purpose=purpose,
                             cache_key=sha256_text(f"{doc.id}:{purpose}:{sample}")[:32],
                             prompt_hash="", source="fixture")
            self.conn.execute(
                "INSERT INTO llm_calls (ts, document_id, purpose, model, cache_key, "
                "prompt_hash, source) VALUES (datetime('now'), ?, ?, ?, ?, '', 'fixture')",
                (doc.id, purpose, model, rec.cache_key))
        else:
            req = self._request(doc, channel, pages, pdf_path, sample)
            model = req["model"]
            raw, rec = self.llm.chat_json(**req)
        assert isinstance(raw, RawExtraction)

        act = self.prov.activity(
            f"hlzf:act/extract/{doc.id}/{channel}/s{sample}{purpose_suffix}/{rec.cache_key[:12]}",
            "hlzf:Extract", f"extract ({channel}, sample {sample})",
            channel=channel, sample=sample, promptVersion=PROMPT_VERSION,
            promptHash=rec.prompt_hash, responseSource=rec.source,
            inputTokens=rec.input_tokens, outputTokens=rec.output_tokens,
            costUsd=rec.cost_usd, latencyMs=rec.latency_ms, repaired=rec.repaired)
        self.prov.associated(act, self.model_agent(model, doc.synthetic), "extractor")
        for pe in page_entities or []:
            self.prov.used(act, pe, "input")
        ent = self.prov.entity(
            f"hlzf:raw/{doc.id}/{channel}/s{sample}{purpose_suffix}/{rec.cache_key[:12]}",
            "hlzf:RawExtraction", f"raw extraction ({channel}, sample {sample})",
            model=model, channel=channel, sample=sample)
        self.prov.generated(ent, act)
        self.conn.execute(
            "INSERT OR REPLACE INTO extractions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ent, doc.id, channel + purpose_suffix, sample, model, rec.source,
             raw.model_dump_json()))
        return ExtractionRun(raw=raw, record=rec, channel=channel, sample=sample, model=model,
                             entity_id=ent, activity_id=act)

    def _request(self, doc: CorpusEntry, channel: str, pages: list[PageText], pdf_path: Path,
                 sample: int) -> dict[str, Any]:
        """The exact chat_json arguments for one call. Shared by `extract` and `prefetch`, so a
        prefetched answer lands under the cache key the sequential pass looks up."""
        purpose = f"extract:{channel}"
        if channel == "vision":
            shown = pages[:MAX_VISION_PAGES]
            pngs = [render_page(pdf_path, p.page_no, self.s.pages_dir / doc.id) for p in shown]
            return dict(model=self.s.check_model, system=VISION_SYSTEM,
                        user=vision_user_message(pngs), schema=RawExtraction, purpose=purpose,
                        prompt_version=PROMPT_VERSION, sample=sample, document_id=doc.id,
                        input_key=f"{doc.sha256}:pages={[p.page_no for p in shown]}:zoom=2")
        return dict(model=self.s.extract_model, system=EXTRACT_SYSTEM,
                    user=text_user_message(pages), schema=RawExtraction, purpose=purpose,
                    prompt_version=PROMPT_VERSION, sample=sample, document_id=doc.id)

    def prefetch(self, jobs: list[Job], ocr: tuple[CorpusEntry, Path, list[int]] | None = None
                 ) -> list[str]:
        """Run independent model calls in parallel so their answers are cached.

        The pipeline then walks the same calls sequentially and hits the cache, which keeps
        all PROV and table writes on one thread. Failures are only collected here; the
        sequential pass meets them again and handles them like any other failed call.
        """
        tasks = [lambda r=self._request(d, c, p, f, n): self.llm.chat_json(**r)
                 for d, c, p, f, n in jobs if not d.synthetic]
        if ocr and not ocr[0].synthetic:
            doc, pdf_path, page_nos = ocr
            data = pdf_path.read_bytes()
            tasks += [lambda n=n: self.llm.layout_parse(pdf_bytes=data, page_no=n,
                                                        document_id=doc.id)
                      for n in page_nos]
        if not tasks:
            return []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=min(self.s.parallel, len(tasks))) as pool:
            for fut in [pool.submit(t) for t in tasks]:
                try:
                    fut.result()
                except Exception as err:  # reported again by the sequential pass
                    errors.append(f"{type(err).__name__}: {err}"[:200])
        return errors

    # --- OCR -------------------------------------------------------------------------
    def ocr_page(self, doc: CorpusEntry, pdf_path: Path, page_no: int) -> PageText:
        if doc.synthetic:
            data = fixtures.scripted_ocr(doc.id, pdf_path, page_no)
            self.conn.execute(
                "INSERT INTO llm_calls (ts, document_id, purpose, model, cache_key, "
                "prompt_hash, source) VALUES (datetime('now'), ?, 'ocr', ?, ?, '', 'fixture')",
                (doc.id, FIXTURE_MODEL, f"{doc.id}:ocr:{page_no}"))
        else:
            data, _ = self.llm.layout_parse(pdf_bytes=pdf_path.read_bytes(), page_no=page_no,
                                            document_id=doc.id)
        return ocr_to_page(data, page_no)


def ocr_to_page(data: dict, page_no: int) -> PageText:
    """Turn a layout_parsing response into a PageText (bbox_2d are normalized 0-1)."""
    info = (data.get("data_info") or {}).get("pages") or [{}]
    width = float(info[0].get("width") or 595.0)
    height = float(info[0].get("height") or 842.0)
    details = data.get("layout_details") or [[]]
    elems = details[0] if details and isinstance(details[0], list) else details
    lines: list[Line] = []
    for el in elems:
        content = str(el.get("content") or "").strip()
        box = el.get("bbox_2d") or [0, 0, 0, 0]
        if not content:
            continue
        scale = (width, height, width, height) if max(box) <= 1.0 else (1, 1, 1, 1)
        bbox = tuple(round(float(v) * s, 2) for v, s in zip(box, scale, strict=True))
        for k, part in enumerate(content.splitlines()):
            if part.strip():
                lines.append(Line(text=part.strip(), bbox=bbox if k == 0 else bbox))
    text = data.get("md_results") or "\n".join(ln.text for ln in lines)
    return PageText(page_no=page_no, text=text, lines=lines, width=width, height=height,
                    text_layer="ok", source="ocr")
