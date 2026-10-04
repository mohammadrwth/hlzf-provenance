"""Grounding: does the cited quote exist on the cited page, does the value agree with its own
quote, and does the quote sit in the table cell the value claims (level x season)?

All checks are deterministic (fuzzy string match + regex + bbox geometry). The model's answer
is never re-checked by another model call here.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .models import PageText
from .parse import GROUNDING_THRESHOLD, Located, locate_all
from .textnorm import map_level, match_key, parse_time, time_ranges_in, times_in

BBox = tuple[float, float, float, float]


@dataclass
class Grounding:
    found: bool
    ratio: float
    bbox: BBox | None = None
    value_consistent: bool | None = None  # None: quote carries no comparable value
    cell_aligned: bool | None = None  # None: alignment could not be determined
    candidates: list[Located] = field(default_factory=list)
    note: str = ""


def ground_quote(pages: dict[int, PageText], page: int, quote: str) -> Grounding:
    pg = pages.get(page)
    if pg is None:
        return Grounding(found=False, ratio=0.0, note=f"page {page} does not exist")
    q = match_key(quote)
    # Fuzzy matching tolerates typos, not other numbers: "HS/MS 30%" scores 93 against a
    # printed "HS/MS 20%". Every number in the quote must be printed where it matched.
    cands = [c for c in locate_all(pg, quote) if c.bbox is None or _numbers_in(q, match_key(
        " ".join(ln.text for ln in pg.lines if _inside(ln.bbox, c.bbox))))]
    if cands:
        return Grounding(found=True, ratio=cands[0].ratio, bbox=cands[0].bbox, candidates=cands)
    # Quotes may run over a line break: fall back to the whole page text (no bbox). Long
    # quotes match fuzzily; short ones ("MS/NS 30%") only exactly, because a fuzzy match of a
    # few characters proves nothing.
    best = 0.0
    # The extractor reads the layout text, so a quote may span the cells of one table row
    # ("HS/MS   20%   500 €"); that row exists only in the layout text, not in any one line.
    for source, text in (("page text", pg.text), ("layout text", pg.layout)):
        page_key = match_key(text)
        if not page_key:
            continue
        note = "found across a line break" if source == "page text" else \
            "found in a table row of the layout text"
        if len(q) >= 12:
            al = fuzz.partial_ratio_alignment(q, page_key)
            whole = al.score if al else 0.0
            best = max(best, whole)
            region = page_key[max(0, al.dest_start - 8):al.dest_end + 8] if al else ""
            if whole >= GROUNDING_THRESHOLD and _numbers_in(q, region):
                return Grounding(found=True, ratio=round(whole, 1), note=note)
        elif q and re.search(r"(?<![0-9a-zäöüß/])" + re.escape(q) + r"(?![0-9])", page_key):
            return Grounding(found=True, ratio=100.0, note=note)
    return Grounding(found=False, ratio=round(best, 1))


def _numbers_in(quote_key: str, text_key: str) -> bool:
    need = Counter(re.findall(r"\d+", quote_key))
    return not need - Counter(re.findall(r"\d+", text_key))


def _ranges(text: str) -> set[tuple[int, int]]:
    return set(time_ranges_in(text))


def ground_window(pages: dict[int, PageText], page: int, quote: str, start: str, end: str,
                  level_quote: str = "", season_quote: str = "",
                  quote_end: str | None = None) -> Grounding:
    s, e = parse_time(start), parse_time(end)
    if quote_end:
        return _ground_split(pages, page, quote, quote_end, s, e, level_quote, season_quote)
    g = ground_quote(pages, page, quote)
    quote_ranges = _ranges(quote)
    if s is not None and e is not None and quote_ranges:
        g.value_consistent = (s, e) in quote_ranges
    elif s is not None and e is not None and len(times_in(quote)) == 2:
        g.value_consistent = tuple(times_in(quote)) == (s, e)
    if not g.found or not quote_ranges:
        return g

    # A fuzzy hit is not enough for times: "07:45-19:00" vs printed "07:45-09:00" scores 90+.
    # Keep only candidates whose own line prints exactly one of the quote's ranges.
    pg = pages[page]
    exact = []
    for c in g.candidates:
        if c.bbox is None:
            continue
        line_text = " ".join(ln.text for ln in pg.lines if _inside(ln.bbox, c.bbox))
        if _ranges(line_text) & quote_ranges:
            exact.append(c)
    if not exact:
        g.found = False
        g.note = "printed time range not found verbatim on page"
        g.bbox = None
        return g
    g.candidates = exact
    g.bbox = exact[0].bbox

    if level_quote and season_quote:
        aligned = [c for c in exact if _aligned(pg, c.bbox, level_quote, season_quote)]
        if aligned:
            g.cell_aligned = True
            g.bbox = aligned[0].bbox
        elif all(_contradicted(pg, c.bbox, level_quote, season_quote) for c in exact):
            # the same range may be printed in several cells; misaligned only if none of
            # its places can be the claimed one
            g.cell_aligned = False
    return g


def _ground_split(pages: dict[int, PageText], page: int, quote: str, quote_end: str,
                  s: int | None, e: int | None, level_quote: str, season_quote: str
                  ) -> Grounding:
    """Start and end printed in separate von / bis cells: both cells must be on the page with
    exactly the extracted times, and the start cell must sit in the claimed table cell."""
    pg = pages.get(page)
    if pg is None:
        return Grounding(found=False, ratio=0.0, note=f"page {page} does not exist")
    t_start, t_end = times_in(quote), times_in(quote_end)
    g = Grounding(found=False, ratio=0.0)
    if s is not None and e is not None and t_start and t_end:
        g.value_consistent = s in t_start and e in t_end

    def lines_with(minutes: list[int]) -> list[BBox]:
        return [ln.bbox for ln in pg.lines if set(times_in(ln.text)) & set(minutes)]

    starts, ends = lines_with(t_start), lines_with(t_end)
    if not starts or not ends:
        g.note = "start or end cell not found verbatim on page"
        return g
    # the end cell sits on the same row as a start cell (von / bis side by side)
    pairs = [a for a in starts if any(_same_row(a, b) and b[0] > a[0] for b in ends)]
    if not pairs:
        g.note = "start and end cells are not side by side on the page"
        return g
    g.found, g.ratio, g.bbox = True, 100.0, pairs[0]
    g.candidates = [Located(bbox=b, ratio=100.0) for b in pairs]
    if level_quote and season_quote:
        aligned = [b for b in pairs if _aligned(pg, b, level_quote, season_quote)]
        if aligned:
            g.cell_aligned, g.bbox = True, aligned[0]
        elif all(_contradicted(pg, b, level_quote, season_quote) for b in pairs):
            g.cell_aligned = False
    return g


def _inside(inner: BBox, outer: BBox, tol: float = 1.0) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _label_hits(pg: PageText, label: str) -> list[BBox]:
    """Where a row/column label is printed (single line, or wrapped onto the next line)."""
    key = match_key(label)
    hits = []
    for ln in pg.lines:
        t = match_key(ln.text)
        if t and (fuzz.ratio(key, t) >= 92 or (len(key) >= 4 and key in t)):
            hits.append(ln.bbox)
    for c in locate_all(pg, label, min_ratio=92):
        if c.multiline and c.bbox and fuzz.ratio(key, match_key(" ".join(
                ln.text for ln in pg.lines if _inside(ln.bbox, c.bbox)))) >= 92:
            hits.append(c.bbox)
    return hits


def _x_overlap(a: BBox, b: BBox) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 2


def _same_row(a: BBox, b: BBox) -> bool:
    return abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) < 4


def _nearest_above(cands: list[BBox], target: BBox) -> BBox | None:
    above = [c for c in cands if c[1] <= target[1] + 2]
    return max(above, key=lambda c: c[1]) if above else None


def _row_level(pg: PageText, cell: BBox, level_hits: list[BBox], claimed: str | None
               ) -> str | None:
    """Grid level of the row label closest to the cell (by vertical centre), among the labels
    printed left of it in the claimed label's column and not entirely below it. None if no
    other recognisable level label is in that column."""
    cands: list[tuple[BBox, str | None]] = [(h, claimed) for h in level_hits]
    for ln in pg.lines:
        if ln.bbox[2] > cell[0] + 1 or not any(_x_overlap(ln.bbox, h) for h in level_hits):
            continue
        lv = map_level(ln.text)
        if lv:
            cands.append((ln.bbox, lv))
    cands = [(b, lv) for b, lv in cands if b[1] <= cell[3] and b[2] <= cell[0] + 1]
    if not cands:
        return None
    mid = (cell[1] + cell[3]) / 2
    dist, level = min((abs((b[1] + b[3]) / 2 - mid), lv) for b, lv in cands)
    if all(lv == claimed for _, lv in cands):
        # no other level label in reach: only trust a label printed close to the cell
        return claimed if dist < 20 else None
    return level


def _aligned(pg: PageText, cell: BBox, level_label: str, season_label: str) -> bool:
    levels, seasons = _label_hits(pg, level_label), _label_hits(pg, season_label)
    if not levels or not seasons:
        return False
    claimed = map_level(level_label)
    row = _row_level(pg, cell, levels, claimed) if claimed else None
    if row is not None:
        # a closer label of another level owns this row (N-ERGIE 2026: the NE7 window
        # 11:00-12:30 read as NE6 sits nearer "Niederspannung NS" than "... MS/NS")
        level_ok = row == claimed
    else:
        level_ok = any(_same_row(lv, cell) for lv in levels) or (
            _nearest_above(levels, cell) is not None
            and cell[1] - _nearest_above(levels, cell)[1] < 60)  # type: ignore[index]
    season_ok = any(_same_row(sv, cell) for sv in seasons) or any(
        _x_overlap(sv, cell) and sv[1] < cell[1] for sv in seasons)
    return level_ok and season_ok


def _contradicted(pg: PageText, cell: BBox, level_label: str, season_label: str) -> bool:
    """True if the cell clearly sits under/next to a *different* season label."""
    seasons = _label_hits(pg, season_label)
    season_names = {"winter", "frühling", "fruehling", "sommer", "herbst"}
    claimed = match_key(season_label)
    # rows layout: a season name printed in the same row, left of the cell
    same_row = [ln for ln in pg.lines if _same_row(ln.bbox, cell) and ln.bbox[2] <= cell[0]
                and match_key(ln.text) in season_names]
    if same_row:
        return all(match_key(ln.text) != claimed for ln in same_row)
    headers_above = [ln.bbox for ln in pg.lines if ln.bbox[1] < cell[1] and _x_overlap(
        ln.bbox, cell) and match_key(ln.text) in {
            "winter", "frühling", "fruehling", "sommer", "herbst"}]
    if not headers_above or not seasons:
        return False
    nearest = max(headers_above, key=lambda b: b[1])
    return not any(abs(nearest[0] - s[0]) < 2 and abs(nearest[1] - s[1]) < 2 for s in seasons)
