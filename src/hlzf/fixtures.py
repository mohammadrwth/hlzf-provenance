"""Synthetic test corpus: fictional DSOs, generated PDFs, scripted model responses.

Why this exists: the pipeline's validation and fault attribution can only be tested if we know
the truth and know which stage broke. Each synthetic document is generated from a ground-truth
spec below, and its "model" responses are scripted from that same spec with *documented,
injected* faults. Nothing here is real DSO data, and nothing here pretends to be GLM output:
the PROV agent for these responses is `fixture-replay`.

| document              | what is injected                                     | expected verdict |
|-----------------------|------------------------------------------------------|------------------|
| musterstadt-2026      | sample 0 drops the window NE6 spring 18:00-19:15     | extract          |
| alpenland-2026        | sample 0 reads 19:15 where the page prints 19:45     | extract          |
| talwerk-2026          | invisible text layer says 01:45, page shows 07:45    | parse            |
| quellbach-2026        | page itself prints 16:30-16:00 and a 10.5 h day      | source_document  |
| nordheim-2026         | scanned page, no text layer                          | OCR path         |
| beispielstadt-2026(+korr) | corrected re-publication changes NE5 winter      | correction/diff  |
| alpenland-2025        | previous year of alpenland                           | year-over-year   |
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from .models import CorpusEntry

PAGE_W, PAGE_H = 595.0, 842.0
X0 = 50.0
FONT, BOLD = "helv", "hebo"
_FONTS: dict[str, pymupdf.Font] = {}


def _font(name: str) -> pymupdf.Font:
    # TextWriter + Font objects encode full Unicode (en dash, umlauts, euro sign);
    # page.insert_text with base-14 names would fall back to Latin-1.
    if name not in _FONTS:
        _FONTS[name] = pymupdf.Font(name)
    return _FONTS[name]

SEASON_HEADERS = {
    "winter": ("Winter", "Jan., Feb., Dez."),
    "spring": ("Frühling", "März bis Mai"),
    "summer": ("Sommer", "Juni bis August"),
    "autumn": ("Herbst", "Sept. bis Nov."),
}
SEASON_ORDER = ("winter", "spring", "summer", "autumn")


@dataclass
class RuleSpec:
    sentence: str  # printed, one line
    kind: str
    value: Any
    level_label: str | None = None


@dataclass
class DocSpec:
    id: str
    dso: str
    year: int
    states: list[str]
    stand: str  # printed date, DD.MM.YYYY
    levels: list[tuple[str, str]]  # (printed label, NE code)
    table: dict[tuple[str, str], list[tuple[str, str]]]  # (NE, season) -> [(start, end)]
    rules: list[RuleSpec] = field(default_factory=list)
    convention_sentence: str | None = None
    meter_label: str = "unstated"
    worked_example: tuple[str, str] | None = None
    version_label: str | None = None
    correction_sentence: str | None = None
    correction_summary: str | None = None
    layout: str = "grid"  # grid | rows
    render: str = "text"  # text | scanned | bad_text_layer
    text_layer_subs: dict[str, str] = field(default_factory=dict)
    title_year: int | None = None
    notes: str = ""

    @property
    def printed_year(self) -> int:
        return self.title_year or self.year


def _r(*pairs: str) -> list[tuple[str, str]]:
    it = iter(pairs)
    return list(zip(it, it, strict=True))


BY_RULES = [
    RuleSpec("Die Hochlastzeitfenster gelten ausschließlich an Werktagen (Montag bis Freitag).",
             "workdays_only", True),
]


def specs() -> list[DocSpec]:
    alpen_levels = [("Hochspannung (HS)", "NE3"), ("Umspannung HS/MS", "NE4"),
                    ("Mittelspannung (MS)", "NE5"), ("Umspannung MS/NS", "NE6"),
                    ("Niederspannung (NS)", "NE7")]
    alpen_rules = [
        *BY_RULES,
        RuleSpec("Brückentage werden als Werktage betrachtet.", "bridge_days", "working_day"),
        RuleSpec("Feiertage gelten nur, wenn sie in Bayern und Baden-Württemberg gelten "
                 "(nicht Mariä Himmelfahrt).", "holiday_handling",
                 {"states": ["BY", "BW"], "mode": "intersection",
                  "exclude": ["Mariä Himmelfahrt"]}),
        RuleSpec("Die Zeit vom 24.12. bis einschließlich 01.01. gilt als Nebenzeit.",
                 "christmas_period", {"from": "12-24", "to": "01-01"}),
        RuleSpec("Erheblichkeitsschwelle Hochspannung (HS): 10 %", "significance_threshold",
                 10, "Hochspannung (HS)"),
        RuleSpec("Erheblichkeitsschwelle Mittelspannung (MS): 20 %", "significance_threshold",
                 20, "Mittelspannung (MS)"),
        RuleSpec("Erheblichkeitsschwelle Niederspannung (NS): 30 %", "significance_threshold",
                 30, "Niederspannung (NS)"),
        RuleSpec("Die Lastverlagerung muss mindestens 100 kW betragen.", "min_kw_diff", 100),
        RuleSpec("Die zu erwartende Entgeltreduzierung muss mindestens 500 € betragen.",
                 "de_minimis_eur", 500),
    ]
    alpen_2026 = {
        ("NE3", "winter"): _r("07:45", "09:00", "15:45", "19:15"),
        ("NE3", "spring"): _r("16:00", "16:15"),
        ("NE3", "summer"): _r("17:00", "17:15"),
        ("NE3", "autumn"): [],
        ("NE4", "winter"): _r("07:30", "09:00", "16:00", "19:15"),
        ("NE4", "spring"): _r("16:30", "16:45"),
        ("NE4", "summer"): [],
        ("NE4", "autumn"): [],
        ("NE5", "winter"): _r("07:45", "09:00", "16:30", "19:45"),
        ("NE5", "spring"): _r("16:45", "18:45"),
        ("NE5", "summer"): [],
        ("NE5", "autumn"): [],
        ("NE6", "winter"): [],
        ("NE6", "spring"): _r("17:00", "19:15"),
        ("NE6", "summer"): _r("17:30", "18:15"),
        ("NE6", "autumn"): [],
        ("NE7", "winter"): _r("11:00", "12:30", "17:30", "18:15"),
        ("NE7", "spring"): [],
        ("NE7", "summer"): [],
        ("NE7", "autumn"): [],
    }
    alpen_2025 = copy.deepcopy(alpen_2026)
    alpen_2025[("NE3", "winter")] = _r("08:45", "09:30", "12:30", "19:15")
    alpen_2025[("NE5", "winter")] = _r("08:00", "09:00", "16:30", "19:30")
    alpen_2025[("NE6", "winter")] = _r("17:00", "19:00")

    muster_levels = [("Mittelspannung", "NE5"), ("Umspannung Mittel-/Niederspannung", "NE6"),
                     ("Niederspannung", "NE7")]
    muster_table = {
        ("NE5", "winter"): _r("16:30", "19:30"), ("NE5", "spring"): [],
        ("NE5", "summer"): [], ("NE5", "autumn"): _r("17:00", "19:00"),
        ("NE6", "winter"): _r("16:45", "19:45"), ("NE6", "spring"): _r("18:00", "19:15"),
        ("NE6", "summer"): [], ("NE6", "autumn"): _r("17:15", "19:15"),
        ("NE7", "winter"): _r("16:30", "19:30"), ("NE7", "spring"): [],
        ("NE7", "summer"): [], ("NE7", "autumn"): _r("17:15", "19:30"),
    }
    beispiel_levels = [("Netzebene 4 (HS/MS)", "NE4"), ("Netzebene 5 (MS)", "NE5"),
                       ("Netzebene 6 (MS/NS)", "NE6"), ("Netzebene 7 (NS)", "NE7")]
    beispiel_orig = {
        ("NE4", "winter"): [], ("NE4", "spring"): [], ("NE4", "summer"): [],
        ("NE4", "autumn"): _r("10:15", "14:15", "16:30", "18:30"),
        ("NE5", "winter"): _r("08:30", "14:00", "14:45", "19:00"),
        ("NE5", "spring"): [], ("NE5", "summer"): [],
        ("NE5", "autumn"): _r("11:15", "12:30", "17:00", "18:45"),
        ("NE6", "winter"): [], ("NE6", "spring"): _r("17:00", "19:15"),
        ("NE6", "summer"): [], ("NE6", "autumn"): _r("17:15", "18:45"),
        ("NE7", "winter"): [], ("NE7", "spring"): _r("17:00", "19:15"),
        ("NE7", "summer"): [], ("NE7", "autumn"): _r("17:15", "18:30"),
    }
    beispiel_korr = copy.deepcopy(beispiel_orig)
    beispiel_korr[("NE5", "winter")] = _r("08:30", "13:45", "14:45", "19:00")
    beispiel_rules = [
        *BY_RULES,
        RuleSpec("Wochenenden, Feiertage und maximal ein Brückentag gelten als Nebenzeiten.",
                 "bridge_days", "off_peak_max_one"),
        RuleSpec("Die Zeit zwischen Weihnachten und Neujahr gilt als Nebenzeit.",
                 "christmas_period", {"from": None, "to": None}),
        RuleSpec("Die zu erwartende Entgeltreduzierung muss mindestens 500,00 € betragen.",
                 "de_minimis_eur", 500),
    ]
    beispiel_conv = ("Bei den Zeitangaben handelt es sich jeweils um das Ende einer "
                     "Viertelstunde.")

    quell_levels = [("Umspannung HS/MS", "NE4"), ("Mittelspannung", "NE5"),
                    ("Umspannung MS/NS", "NE6"), ("Niederspannung", "NE7")]
    quell_table = {
        ("NE4", "winter"): _r("09:15", "10:30", "11:45", "17:30"),
        ("NE4", "autumn"): _r("12:15", "13:15", "15:00", "18:15"),
        ("NE4", "spring"): [], ("NE4", "summer"): [],
        ("NE5", "winter"): _r("12:15", "15:00", "15:45", "18:30"),
        ("NE5", "autumn"): _r("12:45", "13:15", "15:00", "17:30"),
        ("NE5", "spring"): [], ("NE5", "summer"): [],
        ("NE6", "winter"): _r("12:45", "13:30", "15:45", "19:15"),
        ("NE6", "autumn"): _r("16:30", "16:00"),  # printed as-is: end before start
        ("NE6", "spring"): [], ("NE6", "summer"): [],
        ("NE7", "winter"): _r("06:00", "12:00", "13:00", "17:30"),  # 10.5 h per day
        ("NE7", "autumn"): _r("16:15", "19:15"),
        ("NE7", "spring"): [], ("NE7", "summer"): [],
    }

    tal_levels = [("Mittelspannung (MS)", "NE5"), ("Niederspannung (NS)", "NE7")]
    tal_table = {
        ("NE5", "winter"): _r("07:45", "09:00", "17:00", "19:15"),
        ("NE5", "spring"): [], ("NE5", "summer"): [], ("NE5", "autumn"): _r("17:00", "18:30"),
        ("NE7", "winter"): _r("17:15", "19:30"), ("NE7", "spring"): [],
        ("NE7", "summer"): [], ("NE7", "autumn"): _r("17:30", "19:00"),
    }
    nord_levels = [("Mittelspannung", "NE5"), ("Niederspannung", "NE7")]
    nord_table = {
        ("NE5", "winter"): _r("16:45", "19:00"), ("NE5", "spring"): [],
        ("NE5", "summer"): [], ("NE5", "autumn"): _r("17:00", "18:45"),
        ("NE7", "winter"): _r("16:30", "19:15"), ("NE7", "spring"): [],
        ("NE7", "summer"): [], ("NE7", "autumn"): [],
    }

    return [
        DocSpec(
            id="musterstadt-2026", dso="Stadtwerke Musterstadt Netz GmbH", year=2026,
            states=["BY"], stand="31.10.2025", levels=muster_levels, table=muster_table,
            rules=[*BY_RULES,
                   RuleSpec("Wochenenden und gesetzliche Feiertage in Bayern gelten als "
                            "Nebenzeiten.", "holiday_handling",
                            {"states": ["BY"], "mode": "union", "exclude": []}),
                   RuleSpec("Die Lastverlagerung muss mindestens 100 kW betragen.",
                            "min_kw_diff", 100)],
            notes="Clean document; convention not stated.",
        ),
        DocSpec(
            id="alpenland-2025", dso="Alpenland Verteilnetz AG", year=2025, states=["BY", "BW"],
            stand="30.10.2024", levels=alpen_levels, table=alpen_2025, rules=alpen_rules,
            convention_sentence=("Angegeben ist jeweils das Ende des 1/4-Stunden-Intervalls "
                                 "(08:00 bis 11:30 entspricht Zeitstempeln 08:15 bis 11:30)."),
            meter_label="end", worked_example=("08:00", "08:15"),
            notes="Previous year of alpenland (year-over-year drift).",
        ),
        DocSpec(
            id="alpenland-2026", dso="Alpenland Verteilnetz AG", year=2026, states=["BY", "BW"],
            stand="31.10.2025", levels=alpen_levels, table=alpen_2026, rules=alpen_rules,
            convention_sentence=("Angegeben ist jeweils das Ende des 1/4-Stunden-Intervalls "
                                 "(08:00 bis 11:30 entspricht Zeitstempeln 08:15 bis 11:30)."),
            meter_label="end", worked_example=("08:00", "08:15"),
            notes="Multiple windows per season; network spans two states.",
        ),
        DocSpec(
            id="beispielstadt-2026", dso="Stadtwerke Beispielstadt GmbH", year=2026,
            states=["BY"], stand="31.10.2025", levels=beispiel_levels, table=beispiel_orig,
            rules=beispiel_rules, convention_sentence=beispiel_conv, meter_label="end",
            notes="Original publication, later corrected.",
        ),
        DocSpec(
            id="beispielstadt-2026-korr", dso="Stadtwerke Beispielstadt GmbH", year=2026,
            states=["BY"], stand="16.12.2025", levels=beispiel_levels, table=beispiel_korr,
            rules=beispiel_rules, convention_sentence=beispiel_conv, meter_label="end",
            version_label="Korrigierte Fassung",
            correction_sentence=("Korrektur: Das Winterfenster der Netzebene 5 (MS) wurde von "
                                 "08:30 – 14:00 Uhr auf 08:30 – 13:45 Uhr geändert."),
            correction_summary="NE5 winter window changed from 08:30-14:00 to 08:30-13:45.",
            notes="Corrected re-publication.",
        ),
        DocSpec(
            id="quellbach-2026", dso="Quellbach Netze GmbH", year=2026, states=["NW"],
            stand="29.10.2025", levels=quell_levels, table=quell_table, layout="rows",
            rules=[RuleSpec("Die Fenster gelten nur an Werktagen (Montag bis Freitag).",
                            "workdays_only", True),
                   RuleSpec("Die Fenster gelten an Werktagen; NRW-Feiertage sind Nebenzeiten.",
                            "holiday_handling", {"states": ["NW"], "mode": "union",
                                                 "exclude": []}),
                   RuleSpec("Die Lastverlagerung muss mindestens 100 kW betragen.",
                            "min_kw_diff", 100)],
            convention_sentence="Die Zeitangaben bezeichnen jeweils den Beginn der Viertelstunde.",
            meter_label="start",
            notes="Source errors printed in the document itself.",
        ),
        DocSpec(
            id="talwerk-2026", dso="Talwerk Energienetze GmbH", year=2026, states=["BY"],
            stand="27.10.2025", levels=tal_levels, table=tal_table, rules=list(BY_RULES),
            render="bad_text_layer", text_layer_subs={"07:45 – 09:00": "01:45 – 09:00"},
            notes="Scan with an invisible OCR text layer that misreads one digit.",
        ),
        DocSpec(
            id="nordheim-2026", dso="Netzgesellschaft Nordheim mbH", year=2026, states=["NI"],
            stand="31.10.2025", levels=nord_levels, table=nord_table, rules=list(BY_RULES),
            render="scanned", notes="Image-only scan; needs OCR.",
        ),
    ]


def spec_by_id(doc_id: str) -> DocSpec:
    for s in specs():
        if s.id == doc_id:
            return s
    raise KeyError(doc_id)


def corpus_entries() -> list[CorpusEntry]:
    return [CorpusEntry(id=s.id, dso=s.dso, year=s.year, states=s.states, synthetic=True,
                        version_label=s.version_label, notes=s.notes) for s in specs()]


# --------------------------------------------------------------------------------------
# PDF rendering
# --------------------------------------------------------------------------------------

def fmt_window(start: str, end: str) -> str:
    return f"{start} – {end} Uhr"


def _wrap(text: str, width: float, size: float, font: str = FONT) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if _font(font).text_length(trial, fontsize=size) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class _Canvas:
    """Draws text and records each printed line with its bbox (used for OCR fixtures)."""

    def __init__(self, page: pymupdf.Page, subs: dict[str, str] | None = None,
                 render_mode: int = 0, draw_lines: bool = True):
        self.page = page
        self.subs = subs or {}
        self.mode = render_mode
        self.draw_lines = draw_lines
        self.printed: list[tuple[str, tuple[float, float, float, float]]] = []

    def text(self, x: float, y: float, s: str, size: float = 8.5, font: str = FONT) -> None:
        for a, b in self.subs.items():
            s = s.replace(a, b)
        tw = pymupdf.TextWriter(self.page.rect)
        tw.append((x, y), s, font=_font(font), fontsize=size)
        tw.write_text(self.page, render_mode=self.mode)
        w = _font(font).text_length(s, fontsize=size)
        self.printed.append((s, (x, y - size * 0.8, x + w, y + size * 0.25)))

    def rect(self, r: tuple[float, float, float, float]) -> None:
        if self.draw_lines:
            self.page.draw_rect(pymupdf.Rect(*r), color=(0.35, 0.35, 0.35), width=0.6)


def _draw(spec: DocSpec, canvas: _Canvas) -> None:
    c = canvas
    y = 62.0
    c.text(X0, y, spec.dso, size=13, font=BOLD)
    c.text(420, y, f"Stand: {spec.stand}", size=9)
    y += 34
    c.text(X0, y, "Hochlastzeitfenster für atypische Netznutzung", size=11.5, font=BOLD)
    y += 15
    c.text(X0, y, f"nach § 19 Abs. 2 Satz 1 StromNEV für das Jahr {spec.printed_year}",
           size=11.5, font=BOLD)
    y += 16
    if spec.version_label:
        c.text(X0, y, spec.version_label, size=10, font=BOLD)
        y += 14
    if spec.correction_sentence:
        for ln in _wrap(spec.correction_sentence, PAGE_W - 2 * X0, 8.5):
            c.text(X0, y, ln)
            y += 11
        y += 4
    intro = ("Nach der Festlegung BK4-13-739 der Bundesnetzagentur veröffentlichen wir "
             "folgende Hochlastzeitfenster:")
    for ln in _wrap(intro, PAGE_W - 2 * X0, 8.5):
        c.text(X0, y, ln)
        y += 11
    y += 8
    y = _draw_grid(spec, c, y) if spec.layout == "grid" else _draw_rows(spec, c, y)
    y += 18
    sentences = [r.sentence for r in spec.rules]
    if spec.convention_sentence:
        sentences.insert(0, spec.convention_sentence)
    for s in sentences:
        for ln in _wrap(s, PAGE_W - 2 * X0, 8.5):
            c.text(X0, y, ln)
            y += 11
        y += 3


def _draw_grid(spec: DocSpec, c: _Canvas, y: float) -> float:
    col0, colw = 132.0, (PAGE_W - 2 * X0 - 132.0) / 4
    xs = [X0, X0 + col0] + [X0 + col0 + colw * (i + 1) for i in range(4)]
    head_h = 30.0
    c.rect((xs[0], y, xs[-1], y + head_h))
    c.text(xs[0] + 4, y + 13, "Netzebene", font=BOLD)
    for i, s in enumerate(SEASON_ORDER):
        top, sub = SEASON_HEADERS[s]
        c.rect((xs[i + 1], y, xs[i + 2], y + head_h))
        c.text(xs[i + 1] + 4, y + 12, top, font=BOLD)
        c.text(xs[i + 1] + 4, y + 23, sub, size=7.5)
    y += head_h
    for label, ne in spec.levels:
        cells = [spec.table.get((ne, s), []) for s in SEASON_ORDER]
        label_lines = _wrap(label, col0 - 8, 8.5)
        n = max([len(label_lines)] + [max(1, len(w)) for w in cells])
        h = 12.0 * n + 8
        c.rect((xs[0], y, xs[1], y + h))
        for k, ln in enumerate(label_lines):
            c.text(xs[0] + 4, y + 13 + 12 * k, ln)
        for i, wins in enumerate(cells):
            c.rect((xs[i + 1], y, xs[i + 2], y + h))
            if not wins:
                c.text(xs[i + 1] + 4, y + 13, "–")
            for k, (a, b) in enumerate(wins):
                c.text(xs[i + 1] + 4, y + 13 + 12 * k, fmt_window(a, b))
        y += h
    return y


def _draw_rows(spec: DocSpec, c: _Canvas, y: float) -> float:
    xs = [X0, X0 + 150, X0 + 250, PAGE_W - X0]
    c.rect((xs[0], y, xs[-1], y + 18))
    for x, t in zip(xs, ("Spannungsebene", "Jahreszeit", "Hochlastzeitfenster (1/4 h)"),
                    strict=False):
        c.text(x + 4, y + 12, t, font=BOLD)
    y += 18
    for label, ne in spec.levels:
        for s in ("winter", "autumn", "spring", "summer"):
            wins = spec.table.get((ne, s), [])
            c.rect((xs[0], y, xs[1], y + 16))
            c.rect((xs[1], y, xs[2], y + 16))
            c.rect((xs[2], y, xs[3], y + 16))
            c.text(xs[0] + 4, y + 11, label)
            c.text(xs[1] + 4, y + 11, SEASON_HEADERS[s][0])
            c.text(xs[2] + 4, y + 11,
                   "; ".join(fmt_window(a, b) for a, b in wins) if wins else "–")
            y += 16
    return y


def build_pdf(spec: DocSpec, out: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Write the synthetic PDF; returns the printed lines (text, bbox) of page 1."""
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    if spec.render == "text":
        canvas = _Canvas(page)
        _draw(spec, canvas)
        printed = canvas.printed
    else:
        # Draw the visible page on a scratch document and paste it in as an image.
        scratch = pymupdf.open()
        sp = scratch.new_page(width=PAGE_W, height=PAGE_H)
        visible = _Canvas(sp)
        _draw(spec, visible)
        pix = sp.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        page.insert_image(page.rect, stream=pix.tobytes("png"))
        printed = visible.printed
        if spec.render == "bad_text_layer":
            # Invisible OCR-style text layer that misreads part of the page.
            _draw(spec, _Canvas(page, subs=spec.text_layer_subs, render_mode=3,
                                draw_lines=False))
        scratch.close()
    doc.set_metadata({"title": f"{spec.dso} HLZF {spec.year} (SYNTHETIC TEST DOCUMENT)",
                      "producer": "hlzf-provenance fixtures", "creationDate": "",
                      "modDate": ""})
    doc.save(out, garbage=4, deflate=True, no_new_id=True)
    doc.close()
    return printed


def build_all(out_dir: Path) -> dict[str, Path]:
    paths = {}
    for s in specs():
        p = out_dir / f"{s.id}.pdf"
        build_pdf(s, p)
        paths[s.id] = p
    return paths


# --------------------------------------------------------------------------------------
# Scripted responses ("fixture-replay" model)
# --------------------------------------------------------------------------------------

def _season_label(season: str) -> str:
    return SEASON_HEADERS[season][0]


def truth_extraction(spec: DocSpec, visible: bool = True) -> dict[str, Any]:
    """The RawExtraction a perfect reader would return for this document.

    `visible=False` reads the (possibly corrupted) text layer instead of the printed page.
    """
    subs = {} if visible else spec.text_layer_subs

    def s(text: str) -> str:
        for a, b in subs.items():
            text = text.replace(a, b)
        return text

    windows, empties = [], []
    for label, ne in spec.levels:
        for season in SEASON_ORDER:
            wins = spec.table.get((ne, season), [])
            if not wins:
                empties.append({"level_label": label, "season_label": _season_label(season),
                                "page": 1})
            for a, b in wins:
                q = s(fmt_window(a, b))
                start, end = q.split(" – ")[0], q.split(" – ")[1].replace(" Uhr", "")
                windows.append({
                    "level_label": label, "season_label": _season_label(season),
                    "start": start, "end": end, "page": 1, "quote": q,
                    "level_quote": label, "season_quote": _season_label(season)})
    d, m, y = spec.stand.split(".")
    out: dict[str, Any] = {
        "dso_name": {"value": spec.dso, "page": 1, "quote": spec.dso},
        "validity_year": {"value": spec.printed_year, "page": 1,
                          "quote": f"für das Jahr {spec.printed_year}"},
        "publication_date": {"value": f"{y}-{m}-{d}", "page": 1,
                             "quote": f"Stand: {spec.stand}"},
        "version_label": ({"value": spec.version_label, "page": 1,
                           "quote": spec.version_label} if spec.version_label else None),
        "correction_note": ({"value": spec.correction_summary, "page": 1,
                             "quote": spec.correction_sentence[:110]}
                            if spec.correction_sentence else None),
        "convention": {"meter_label": spec.meter_label,
                       "page": 1 if spec.convention_sentence else None,
                       "quote": spec.convention_sentence[:110]
                       if spec.convention_sentence else None,
                       "worked_example": ({"listed_start": spec.worked_example[0],
                                           "first_timestamp": spec.worked_example[1]}
                                          if spec.worked_example else None)},
        "levels_listed": [{"label": lbl, "page": 1, "quote": lbl} for lbl, _ in spec.levels],
        "windows": windows,
        "empty_cells": empties,
        "rules": [{"kind": r.kind, "value": r.value, "level_label": r.level_label, "page": 1,
                   "quote": r.sentence[:110]} for r in spec.rules],
        "anomalies": [],
    }
    return out


def scripted_response(doc_id: str, purpose: str, sample: int) -> dict[str, Any]:
    """Return the fixture-replay answer for one model call.

    purpose: "extract:text" | "extract:vision" | "extract:ocr" (text extraction run on the
    OCR'd page, used by the parse-swap intervention).
    """
    spec = spec_by_id(doc_id)
    if purpose == "extract:text":
        # The text channel reads the text layer; for bad_text_layer docs that is wrong.
        resp = truth_extraction(spec, visible=spec.render != "bad_text_layer")
    else:
        resp = truth_extraction(spec, visible=True)

    if purpose == "extract:text" and sample == 0:
        if doc_id == "musterstadt-2026":
            # Drops the only window of a cell -> coverage gap + cross-check disagreement
            # (an extract fault). Dropping an *empty* cell is harmless: when the vision
            # channel sees no window there either, the cell counts as empty.
            resp["windows"] = [w for w in resp["windows"]
                               if not (w["season_label"] == "Frühling"
                                       and w["start"] == "18:00")]
        if doc_id == "alpenland-2026":
            # Value disagrees with its own quote: 19:15 read for a printed 19:45.
            for w in resp["windows"]:
                if w["level_label"] == "Mittelspannung (MS)" and w["season_label"] == "Winter" \
                        and w["end"] == "19:45":
                    w["end"] = "19:15"
    return resp


def scripted_ocr(doc_id: str, pdf_path: Path, page_no: int) -> dict[str, Any]:
    """GLM-OCR layout_parsing-shaped response for a synthetic page (bbox_2d normalized)."""
    spec = spec_by_id(doc_id)
    scratch = pymupdf.open()
    sp = scratch.new_page(width=PAGE_W, height=PAGE_H)
    canvas = _Canvas(sp)
    _draw(spec, canvas)
    scratch.close()
    details = [{"index": i, "label": "text",
                "bbox_2d": [round(b[0] / PAGE_W, 5), round(b[1] / PAGE_H, 5),
                            round(b[2] / PAGE_W, 5), round(b[3] / PAGE_H, 5)],
                "content": t, "width": PAGE_W, "height": PAGE_H}
               for i, (t, b) in enumerate(canvas.printed)]
    return {
        "id": "fixture-" + hashlib.sha256(f"{doc_id}:{page_no}".encode()).hexdigest()[:12],
        "model": "fixture-replay",
        "md_results": "\n".join(t for t, _ in canvas.printed),
        "layout_details": [details],
        "data_info": {"num_pages": 1, "pages": [{"width": PAGE_W, "height": PAGE_H}]},
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def golden(doc_id: str) -> dict[str, Any]:
    """Ground truth for a synthetic document, in the golden-set format (see evaluate.py)."""
    spec = spec_by_id(doc_id)
    convention = {"start": "interval_start", "unstated": "assumed"}.get(spec.meter_label)
    if spec.meter_label == "end":
        convention = ("interval_end_physical" if spec.worked_example
                      else "interval_end_ambiguous")
    cells = [{"grid_level": ne, "season": season,
              "windows": [f"{a}-{b}" for a, b in spec.table.get((ne, season), [])]}
             for _, ne in spec.levels for season in SEASON_ORDER]
    return {
        "document": doc_id,
        "labeled_by": "construction (synthetic document)",
        "verified": True,
        "convention": convention,
        "cells": cells,
        "rules": [{"kind": r.kind, "value": r.value} for r in spec.rules],
    }
