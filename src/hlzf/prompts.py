"""Versioned prompts. Bump PROMPT_VERSION whenever a prompt changes: it is part of every
cache key, so stale cached answers are never replayed against a new prompt."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .models import PageText

PROMPT_VERSION = "extract-v3"

_SCHEMA = """{
  "dso_name": {"value": "<operator name as printed>", "page": 1, "quote": "<verbatim>"},
  "validity_year": {"value": 2026, "page": 1, "quote": "<verbatim text naming the year>"},
  "publication_date": {"value": "YYYY-MM-DD", "page": 1, "quote": "<verbatim>"} | null,
  "version_label": {"value": "<e.g. 'korrigierte Fassung'>", "page": 1, "quote": "..."} | null,
  "correction_note": {"value": "<English summary of what changed>", "page": 1,
                      "quote": "<verbatim correction passage>"} | null,
  "convention": {"meter_label": "start" | "end" | "unstated", "page": 1 | null,
                 "quote": "<verbatim>" | null,
                 "worked_example": {"listed_start": "HH:MM", "first_timestamp": "HH:MM"} | null},
  "levels_listed": [{"label": "<grid level as printed>", "page": 1, "quote": "<verbatim>"}],
  "windows": [{"level_label": "...", "season_label": "...", "start": "HH:MM", "end": "HH:MM",
               "page": 1, "quote": "<the printed time range, or the start cell>",
               "quote_end": "<the end cell, only for separate von/bis cells>" | null,
               "level_quote": "<printed level label>", "season_quote": "<printed season label>"}],
  "empty_cells": [{"level_label": "...", "season_label": "...", "page": 1}],
  "rules": [{"kind": "<see list>", "value": <see formats>, "level_label": "..." | null,
             "page": 1, "quote": "<verbatim>"}],
  "anomalies": ["<one English sentence per thing that does not fit this schema>"]
}"""

_RULES = """Rule kinds and value formats:
- workdays_only: true
- bridge_days: "working_day" (bridge days count as working days) | "off_peak" (all bridge
  days are off-peak) | "off_peak_max_one" ("maximal ein Brückentag") |
  "off_peak_max_one_per_week"
- holiday_handling: {"states": ["BY", ...] (German state codes the document names; ["DE"]
  for nationwide holidays only, e.g. "bundeseinheitliche Feiertage"; [] if it names none),
  "mode": "union" | "intersection" | "unspecified", "exclude": ["<holiday names the document
  explicitly excludes>"]}. "Feiertage, die im gesamten Netzgebiet gleichzeitig gelten" means
  mode "intersection".
- christmas_period: {"from": "MM-DD" | null, "to": "MM-DD" | null} — null where the document
  only says e.g. "zwischen Weihnachten und Neujahr" without dates.
- significance_threshold: number in percent, with level_label of the grid level
- min_kw_diff: number in kW
- de_minimis_eur: number in EUR
- other: short English string"""

EXTRACT_SYSTEM = f"""You extract Hochlastzeitfenster (HLZF, high-load time windows for atypical
grid usage under § 19 Abs. 2 Satz 1 StromNEV) from one German grid operator publication.
The pages are marked "=== PAGE n (...) ===". A page is given either as LAYOUT TEXT, where
every character sits at its position on the page so table columns line up vertically, or as
OCR TEXT, where tables are HTML (<table><tr><td>).

Reading tables in layout text: assign each value to the season whose header stands above it
in the same column, and to the grid level whose row it is in. Tables can be transposed
(seasons as rows, levels as columns). A level label can wrap over several lines, and the
values printed between those lines belong to that row. A blank position under a header is an
empty cell. Start and end times may sit in separate "von" / "bis" columns.
Where the table has ruling lines they appear as rows of "─" characters. The text between two
such rows is ONE table row: a value printed there belongs to the level label in the same
band, never to a label above or below a rule, even if it sits closer to that label.

Return exactly ONE JSON object with this shape and nothing else:
{_SCHEMA}

Hard rules:
1. Evidence. Every object carries "page" and "quote". A quote is a short substring copied
   character for character from that page (max 120 characters); runs of spaces may be
   shortened to one. Never paraphrase, translate, reformat or normalize inside a quote, and
   never add words that are not next to each other on the page (no season or level names in a
   window's quote).
2. Windows. One object per contiguous time range. A cell "07:45 – 09:00 Uhr 15:45 – 19:15 Uhr"
   gives two windows. Copy start and end as printed (24h clock, "24:00" allowed). Do not round,
   shift or merge ranges. Only take windows from the HLZF table of the year the document is for.
   If the range is printed as one text ("07:45 – 09:00 Uhr"), quote it whole. If start and end
   are in separate von / bis cells, quote the start cell in "quote" and the end cell in
   "quote_end".
3. Empty cells. List every grid-level x season combination that the table shows as empty
   ("–", "-", "keine", blank). Never invent a window for an empty cell.
4. levels_listed: every grid level that has a row or column in the HLZF table.
5. Dates. "Datenbasis" or reference periods (e.g. "01.12.2023 – 28.02.2024") are NOT the
   validity period. validity_year is the year the windows apply to.
6. Convention. meter_label is "end" if the document says times or meter timestamps denote the
   END of a quarter hour (e.g. "Zeitstempel 09:15 definiert 09:00 bis 09:15", "Endzeit einer
   1/4-Stunden-Periode"), "start" if it says they denote the start, else "unstated". If the
   document gives a worked example mapping a window to meter timestamps, fill worked_example
   with the window start it names and the first meter timestamp it names.
7. Rules. Only rules the document states. {_RULES}
8. correction_note: only if the document says it corrects or replaces an earlier version.
9. If something does not fit this schema, describe it in "anomalies" instead of forcing it.
"""

VISION_SYSTEM = EXTRACT_SYSTEM.split("The pages are marked")[0] + (
    "You see the pages as images, in order; page numbers start at 1. Quotes must be text as "
    "printed on the page image.\n\nRead tables by their columns: assign each value to the "
    "season whose header is above it and the grid level of its row. A blank cell is an empty "
    "cell.\n" + EXTRACT_SYSTEM.split("empty cell. Start and end times may sit in separate")[1]
    .split("\n", 1)[1])


def page_input(p: PageText) -> tuple[str, str]:
    """(label, text) the extractor sees for one page."""
    if p.source == "ocr":
        return "OCR TEXT", p.text.strip()
    if p.layout.strip():
        return "LAYOUT TEXT", p.layout
    return "TEXT", p.text.strip()


def text_user_message(pages: list[PageText]) -> str:
    parts = []
    for p in pages:
        label, body = page_input(p)
        parts.append(f"=== PAGE {p.page_no} ({label}) ===\n{body}")
    return "\n\n".join(parts)


def vision_user_message(png_paths: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Extract the HLZF publication shown in these page images."}]
    for p in png_paths:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return content
