"""S4 validate: canonical values -> issues, each with severity and a suspected stage.

`suspected_stage` here is the cheap first guess (heuristics from the spec). Issues whose
cause is genuinely unclear are then handed to `attribution.py`, which replaces the guess
with an interventional verdict.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .models import (
    Convention,
    CorpusEntry,
    Issue,
    PageText,
    Season,
    Severity,
    Stage,
    fmt_min,
)
from .normalize import Normalized, NormWindow

DAILY_CAP_MIN = 600
# Bundesnetzagentur ruling BK4-13-739, section 2.e (cited by most of the publications; copy at
# regulierung.niedersachsen.de/download/145339): windows are capped at 10 h per day and season.
# Used only as a plausibility warning.
DAILY_CAP_SOURCE = "plausibility check; the cap is from BNetzA ruling BK4-13-739, section 2.e"

Target = tuple[str, object] | None  # ("window", key) | ("cell", (level, season)) | None


@dataclass
class Finding:
    issue: Issue
    target: Target


def cell_windows(n: Normalized, level: str, season: Season) -> set[tuple[int, int]]:
    return {(nw.window.start_min, nw.window.end_min) for nw in n.windows
            if nw.window.grid_level == level and nw.window.season == season}


def _fmt_set(s: set[tuple[int, int]]) -> str:
    return ", ".join(f"{fmt_min(a)}-{fmt_min(b)}" for a, b in sorted(s)) or "(empty)"


def validate(n: Normalized, entry: CorpusEntry, pages: dict[int, PageText],
             vision: Normalized | None = None) -> list[Finding]:
    out: list[Finding] = [Finding(i, None) for i in n.issues]

    def add(code: str, sev: Severity, stage: Stage, msg: str, target: Target = None) -> None:
        out.append(Finding(Issue(code=code, severity=sev, suspected_stage=stage,
                                 message=msg), target))

    # 1. grounding ---------------------------------------------------------------------
    for nw in n.windows:
        w, g = nw.window, nw.grounding
        key = ("window", w.key)
        label = f"{w.grid_level} {w.season.value} {fmt_min(w.start_min)}-{fmt_min(w.end_min)}"
        page = pages.get(nw.raw.page)
        if not g.found:
            bad_layer = page is None or page.text_layer != "ok" or page.source == "ocr"
            add("GROUNDING_NOT_FOUND", Severity.error,
                Stage.parse if bad_layer else Stage.extract,
                f"{label}: quote {nw.raw.quote!r} not found on page {nw.raw.page}"
                + (f" ({g.note})" if g.note else "") + ".", key)
        if g.value_consistent is False:
            add("GROUNDING_VALUE_MISMATCH", Severity.error, Stage.extract,
                f"{label}: value disagrees with its own quote {nw.raw.quote!r}.", key)
        if g.cell_aligned is False:
            add("CELL_MISALIGNED", Severity.warning, Stage.extract,
                f"{label}: quote sits under a different season column than claimed.", key)
    for nr in n.rules:
        if not nr.grounding.found:
            add("RULE_UNGROUNDED", Severity.warning, Stage.extract,
                f"Rule {nr.rule.kind.value}: quote {nr.rule.evidence.quote!r} not found.")
    if not n.year_grounding.found:
        add("YEAR_UNGROUNDED", Severity.warning, Stage.extract,
            "Validity-year quote not found on the cited page.")

    # 2. time sanity ---------------------------------------------------------------------
    def stage_for(nw: NormWindow) -> Stage:
        # A grounded, self-consistent value in the right table cell that breaks a rule was
        # printed that way. A value under the wrong season or level (the N-ERGIE live run:
        # autumn windows read as winter, which then "overlap") is the extractor's doing.
        g = nw.grounding
        consistent = g.found and g.value_consistent is not False and g.cell_aligned is not False
        return Stage.source_document if consistent else Stage.extract

    for nw in n.windows:
        w = nw.window
        key = ("window", w.key)
        label = f"{w.grid_level} {w.season.value} {fmt_min(w.start_min)}-{fmt_min(w.end_min)}"
        if not (0 <= w.start_min < w.end_min <= 1440):
            add("TIME_INVALID", Severity.error, stage_for(nw),
                f"{label}: start must be before end within 00:00-24:00.", key)
        if w.start_min % 15 or w.end_min % 15:
            add("OFF_GRID", Severity.error, stage_for(nw),
                f"{label}: not on the quarter-hour grid.", key)

    # 3. overlap / duplicates ------------------------------------------------------------
    by_cell: dict[tuple[str, Season], list[NormWindow]] = defaultdict(list)
    for nw in n.windows:
        by_cell[(nw.window.grid_level, nw.window.season)].append(nw)
    for (level, season), lst in by_cell.items():
        lst = sorted(lst, key=lambda x: (x.window.start_min, x.window.end_min))
        for a, b in zip(lst, lst[1:], strict=False):
            if a.window.key == b.window.key:
                add("DUPLICATE_WINDOW", Severity.warning, Stage.extract,
                    f"{level} {season.value}: window listed twice.", ("window", b.window.key))
            elif a.window.end_min > b.window.start_min and a.window.start_min < a.window.end_min:
                stage = (Stage.source_document
                         if stage_for(a) is Stage.source_document
                         and stage_for(b) is Stage.source_document else Stage.extract)
                add("OVERLAP", Severity.error, stage,
                    f"{level} {season.value}: {fmt_min(a.window.start_min)}-"
                    f"{fmt_min(a.window.end_min)} overlaps {fmt_min(b.window.start_min)}-"
                    f"{fmt_min(b.window.end_min)}.", ("window", b.window.key))

    # 4. coverage --------------------------------------------------------------------------
    listed = {ne for _, ne in n.levels_listed if ne}
    for label, ne in n.levels_listed:
        if ne is None:
            add("LEVEL_UNMAPPED", Severity.error, Stage.normalize,
                f"Listed grid level {label!r} does not map to NE1-NE7.")
    empty = {(lv, s) for lv, s, _, _ in n.empties}
    if vision is not None:
        # The text channel may skip a blank cell instead of listing it as empty. When the
        # independent vision reading has no window there either, the cell is empty.
        empty |= {(lv, s) for lv, s, _, _ in vision.empties}
        empty |= {(lv, s) for lv in listed for s in Season
                  if (lv, s) not in by_cell and not cell_windows(vision, lv, s)}
    for level in sorted(listed):
        for season in Season:
            if (level, season) not in by_cell and (level, season) not in empty:
                add("COVERAGE_GAP", Severity.warning, Stage.extract,
                    f"{level} {season.value}: neither a window nor an explicit empty cell.",
                    ("cell", (level, season)))
    for level in sorted({nw.window.grid_level for nw in n.windows} - listed):
        add("LEVEL_NOT_LISTED", Severity.info, Stage.extract,
            f"{level} has windows but is not in levels_listed.")

    # 5. daily cap -------------------------------------------------------------------------
    for (level, season), lst in by_cell.items():
        total = sum(max(0, nw.window.end_min - nw.window.start_min) for nw in lst)
        if total > DAILY_CAP_MIN:
            stage = (Stage.source_document
                     if all(stage_for(x) is Stage.source_document for x in lst)
                     else Stage.extract)
            add("DAILY_CAP_EXCEEDED", Severity.warning, stage,
                f"{level} {season.value}: windows total {total / 60:.2f} h per day, above "
                f"10 h ({DAILY_CAP_SOURCE}).", ("cell", (level, season)))

    # 6. year / correction -----------------------------------------------------------------
    if n.year != entry.year:
        expected = "the upload was labelled" if entry.source == "upload" else \
            "registry expects"
        add("YEAR_MISMATCH", Severity.error,
            Stage.source_document if n.year_grounding.found else Stage.extract,
            f"Document says {n.year}, {expected} {entry.year}.")
    if entry.source == "upload":
        from .uploads import same_dso

        if entry.dso and n.dso and not same_dso(entry.dso, n.dso):
            add("DSO_MISMATCH", Severity.warning, Stage.source_document,
                f"Uploaded as {entry.dso!r}, but the document names {n.dso!r}.")
        if not n.windows and not n.empties:
            add("NO_HLZF_TABLE", Severity.error, Stage.source_document,
                "No high-load window table found. Is this an HLZF publication?")
    if n.correction_note:
        add("CORRECTION_NOTED", Severity.info, Stage.source_document,
            f"Document is a corrected version: {n.correction_note}")

    # 7. convention --------------------------------------------------------------------------
    if n.convention is Convention.assumed:
        add("CONVENTION_ASSUMED", Severity.warning, Stage.source_document,
            "Document does not say whether times label quarter-hour starts or ends; "
            "physical span [start, end) assumed.")
    elif n.convention is Convention.interval_end_ambiguous:
        add("CONVENTION_AMBIGUOUS", Severity.warning, Stage.source_document,
            "Document says times are quarter-hour ENDS but gives no example: the first "
            "quarter-hour of every window is uncertain (query answers 'uncertain' there).")

    # 8. unresolved rules ---------------------------------------------------------------------
    for nr in n.rules:
        if not nr.rule.resolved:
            add("RULE_AMBIGUOUS", Severity.warning, Stage.source_document,
                f"Rule {nr.rule.kind.value} = {nr.rule.value!r} cannot be applied to a "
                f"specific date from the document alone (quote: {nr.rule.evidence.quote!r}).")
    for a in n.anomalies:
        add("MODEL_ANOMALY", Severity.info, Stage.extract, f"Extractor noted: {a}")

    # 9. pages without readable text ---------------------------------------------------------
    for p in pages.values():
        if p.text_layer != "ok" and p.source != "ocr":
            add("PAGE_UNREADABLE", Severity.error, Stage.parse,
                f"Page {p.page_no}: text layer {p.text_layer} and no OCR available.")

    # 10. cross-check against the vision channel ---------------------------------------------
    if vision is not None:
        cells = {(nw.window.grid_level, nw.window.season) for nw in n.windows} | {
            (nw.window.grid_level, nw.window.season) for nw in vision.windows}
        for level, season in sorted(cells):
            t, v = cell_windows(n, level, season), cell_windows(vision, level, season)
            if t != v:
                add("CROSS_CHECK_DISAGREE", Severity.warning, Stage.extract,
                    f"{level} {season.value}: text channel reads {_fmt_set(t)}, vision channel "
                    f"reads {_fmt_set(v)}.", ("cell", (level, season)))
    return out
