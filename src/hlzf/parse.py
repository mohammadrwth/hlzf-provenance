"""S1 parse: PDF -> pages with text, line bounding boxes, layout text and a text-layer
health check.

Two renderings of the same text layer:
* `text` and `lines` (PyMuPDF): content-stream order plus a bounding box per line. Used for
  grounding and evidence boxes.
* `layout` (pdfplumber, layout mode): the characters placed on a character grid by their
  position, so table cells stay in their columns. This is what the extractor reads. The first
  live run showed why: in content-stream order Wunsiedel's table cells come out scrambled, and
  N-ERGIE's blank cells vanish, so the model cannot tell which column a window belongs to.

Also renders pages to PNG (for the review UI and the vision cross-check) and locates
verbatim quotes on a page (for grounding and evidence boxes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pymupdf
from rapidfuzz import fuzz

from .models import Line, PageText
from .textnorm import match_key

RENDER_ZOOM = 2.0
LAYOUT_X_DENSITY = 4.5  # points per character column; small enough to keep columns apart
LAYOUT_Y_DENSITY = 13.0
GROUNDING_THRESHOLD = 90.0  # rapidfuzz partial ratio, 0-100 (spec: ratio >= 0.9)


def assess_text_layer(text: str, has_images: bool) -> str:
    """Classify a page's text layer as ok / empty / garbled.

    Note what this can NOT see: an invisible OCR layer that reads cleanly but disagrees with
    the printed pixels. Only the vision cross-check catches that case.
    """
    stripped = text.strip()
    if len(stripped) < 15:
        return "empty" if has_images or not stripped else "ok"
    total = len(stripped)
    bad = stripped.count("�") + len(re.findall(r"\(cid:\d+\)", stripped)) * 6
    bad += sum(1 for ch in stripped if 0xE000 <= ord(ch) <= 0xF8FF)
    printable = sum(1 for ch in stripped if ch.isprintable() or ch in "\n\t")
    if bad / total > 0.05 or printable / total < 0.9:
        return "garbled"
    return "ok"


def parse_pdf(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            lines: list[Line] = []
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                for ln in block.get("lines", []):
                    t = "".join(span.get("text", "") for span in ln.get("spans", [])).strip()
                    if t:
                        lines.append(Line(text=t, bbox=tuple(round(v, 2) for v in ln["bbox"])))
            has_images = bool(page.get_images(full=False))
            pages.append(PageText(
                page_no=i, text=text, lines=lines,
                width=page.rect.width, height=page.rect.height,
                text_layer=assess_text_layer(text, has_images),  # type: ignore[arg-type]
            ))
    for page, layout in zip(pages, layout_texts(path), strict=False):
        page.layout = layout
    return pages


def layout_texts(path: Path) -> list[str]:
    """Layout-mode text per page (pdfplumber). Empty strings if the PDF cannot be read."""
    out: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                raw = _with_rules(page).extract_text(
                    layout=True, x_density=LAYOUT_X_DENSITY, y_density=LAYOUT_Y_DENSITY) or ""
                out.append(_tidy_layout(raw))
    except Exception:  # a parse problem here must not stop the pipeline; text still works
        return []
    return out


RULE_CHAR = "─"
RULE_MIN_WIDTH = 40.0  # points; shorter strokes are underlines or glyph decoration
RULE_MERGE = 2.0  # strokes closer than this vertically are one ruling line
RULE_MIN_COUNT = 3  # fewer ruling lines than this is no table grid


def horizontal_rules(page) -> list[tuple[float, float, float]]:  # type: ignore[no-untyped-def]
    """(y, x0, x1) of the page's horizontal ruling lines: strokes and the edges of cell
    rectangles, merged where they lie on top of each other."""
    edges = sorted((e for e in page.edges if e["orientation"] == "h"
                    and e["x1"] - e["x0"] >= RULE_MIN_WIDTH), key=lambda e: e["top"])
    merged: list[list[float]] = []
    for e in edges:
        if merged and e["top"] - merged[-1][3] <= RULE_MERGE and \
                e["x0"] <= merged[-1][2] + 1 and e["x1"] >= merged[-1][1] - 1:
            m = merged[-1]
            m[1], m[2], m[3] = min(m[1], e["x0"]), max(m[2], e["x1"]), e["top"]
        else:
            merged.append([e["top"], e["x0"], e["x1"], e["top"]])
    return [(m[0], m[1], m[2]) for m in merged]


def _with_rules(page):  # type: ignore[no-untyped-def]
    """The page with each horizontal ruling line added as a row of RULE_CHAR characters, so
    the layout text carries which text sits between which rules (row membership). Borderless
    pages are returned unchanged."""
    rules = horizontal_rules(page)
    if len(rules) < RULE_MIN_COUNT or not page.chars:
        return page
    tpl = page.chars[0]
    h = 4.0
    extra = []
    for y, x0, x1 in rules:
        x = x0
        while x < x1 - 1:
            w = min(LAYOUT_X_DENSITY, x1 - x)
            extra.append({**tpl, "text": RULE_CHAR, "x0": x, "x1": x + w, "top": y - h / 2,
                          "bottom": y + h / 2, "doctop": page.initial_doctop + y - h / 2,
                          "width": w, "height": h, "size": h, "upright": True,
                          "adv": w})
            x += w
    clone = page.__class__(page.pdf, page.page_obj, page_number=page.page_number,
                           initial_doctop=page.initial_doctop)
    clone._objects = {**page.objects, "char": list(page.chars) + extra}
    return clone


def _tidy_layout(raw: str) -> str:
    lines = [ln.rstrip() for ln in raw.splitlines()]
    # common left margin
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    cut = min(indents) if indents else 0
    out: list[str] = []
    blank = 0
    for ln in lines:
        blank = blank + 1 if not ln.strip() else 0
        if blank <= 1:
            out.append(ln[cut:])
    return "\n".join(out).strip("\n")


def render_page(pdf_path: Path, page_no: int, out_dir: Path) -> Path:
    """Render one page to PNG at 2x (cached on disk)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"p{page_no}.png"
    if out.exists() and out.stat().st_mtime >= pdf_path.stat().st_mtime:
        return out
    with pymupdf.open(pdf_path) as doc:
        pix = doc[page_no - 1].get_pixmap(matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM))
        pix.save(out)
    return out


@dataclass
class Located:
    bbox: tuple[float, float, float, float] | None
    ratio: float
    multiline: bool = False


def _union(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def locate_all(page: PageText, quote: str, min_ratio: float = GROUNDING_THRESHOLD
               ) -> list[Located]:
    """All places on the page where `quote` appears (fuzzy), best first.

    Matches single lines and pairs of consecutive lines, because table cells often wrap.
    """
    q = match_key(quote)
    if not q:
        return []
    cands: list[Located] = []
    lines = page.lines
    for i, ln in enumerate(lines):
        options = [(ln.text, ln.bbox, False)]
        # A wrapped cell continues on the next line *below* (overlapping in x), not in the
        # neighbouring cell to the right.
        for nxt in lines[i + 1:i + 6]:
            # tops may overlap the line above by a point or two (N-ERGIE 2026: 1.2 pt)
            mid = (ln.bbox[1] + ln.bbox[3]) / 2
            below = mid < nxt.bbox[1] <= ln.bbox[3] + 6
            if below and min(ln.bbox[2], nxt.bbox[2]) - max(ln.bbox[0], nxt.bbox[0]) > 0:
                options.append((ln.text + " " + nxt.text, _union(ln.bbox, nxt.bbox), True))
                break
        for text, bbox, is_pair in options:
            t = match_key(text)
            if not t:
                continue
            # partial_ratio scores the best-aligned substring; require the line to be at least
            # roughly as long as the quote so a short line cannot "contain" a long quote.
            if len(t) + 2 < len(q):
                ratio = fuzz.ratio(q, t) if len(t) > 0.8 * len(q) else 0.0
            else:
                ratio = fuzz.partial_ratio(q, t)
            if ratio >= min_ratio:
                cands.append(Located(bbox=bbox, ratio=round(ratio, 1), multiline=is_pair))
    # De-duplicate (a single-line hit also appears inside the two-line option); prefer the
    # tighter single-line box at equal score.
    cands.sort(key=lambda c: (-c.ratio, c.multiline,
                              (c.bbox[3] - c.bbox[1]) if c.bbox else 0))
    seen: list[tuple[float, float, float, float]] = []
    out: list[Located] = []
    for c in cands:
        if c.bbox and any(_contains(s, c.bbox) or _contains(c.bbox, s) for s in seen):
            continue
        if c.bbox:
            seen.append(c.bbox)
        out.append(c)
    return out


def _contains(outer: tuple[float, ...], inner: tuple[float, ...], tol: float = 1.0) -> bool:
    return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1]
            and outer[2] + tol >= inner[2] and outer[3] + tol >= inner[3])


def page_ratio(page: PageText, quote: str) -> float:
    """Best fuzzy score of `quote` anywhere in the page text (0-100)."""
    q = match_key(quote)
    if not q:
        return 0.0
    hits = locate_all(page, quote, min_ratio=0)
    best = hits[0].ratio if hits else 0.0
    whole = fuzz.partial_ratio(q, match_key(page.text)) if len(q) >= 12 else 0.0
    return max(best, whole)
