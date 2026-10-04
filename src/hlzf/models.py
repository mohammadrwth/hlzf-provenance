"""Typed data model.

Two layers:

* `Raw*` models are what the LLM must return (JSON mode + pydantic validation). Every value
  carries the page and a verbatim quote it was read from, so it can be grounded.
* Canonical models (`Window`, `Rule`, `Issue`, ...) are what the rest of the pipeline uses
  after normalization. They are persisted in SQLite and each one is a W3C PROV entity.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Season(StrEnum):
    winter = "winter"  # Jan, Feb, Dec
    spring = "spring"  # Mar-May
    summer = "summer"  # Jun-Aug
    autumn = "autumn"  # Sep-Nov


SEASON_MONTHS: dict[Season, tuple[int, ...]] = {
    Season.winter: (1, 2, 12),
    Season.spring: (3, 4, 5),
    Season.summer: (6, 7, 8),
    Season.autumn: (9, 10, 11),
}

GRID_LEVELS = ("NE1", "NE2", "NE3", "NE4", "NE5", "NE6", "NE7")
GRID_LEVEL_NAMES = {
    "NE1": "HöS",
    "NE2": "HöS/HS",
    "NE3": "HS",
    "NE4": "HS/MS",
    "NE5": "MS",
    "NE6": "MS/NS",
    "NE7": "NS",
}


class Convention(StrEnum):
    """How listed window times relate to physical quarter-hours and meter timestamps.

    interval_start          times label quarter-hour starts; window a-b covers [a, b).
    interval_end_physical   meter timestamps label quarter-hour ends, and the document's
                            worked example confirms the window a-b is the physical span
                            [a, b) (meter values stamped a+15 ... b). Ratingen-style.
    interval_end_labels     the worked example shows listed times ARE meter end-stamps:
                            window a-b covers quarter-hours ending a ... b, i.e. [a-15, b).
    interval_end_ambiguous  the document says times are quarter-hour ENDS but gives no
                            example. Read literally, window a-b would start at a-15; read
                            as a physical span it starts at a. The first quarter-hour is
                            therefore uncertain and the query reports it as such.
    assumed                 the document says nothing; we assume physical span [a, b).
    """

    interval_start = "interval_start"
    interval_end_physical = "interval_end_physical"
    interval_end_labels = "interval_end_labels"
    interval_end_ambiguous = "interval_end_ambiguous"
    assumed = "assumed"


class Stage(StrEnum):
    source_document = "source_document"
    parse = "parse"
    extract = "extract"
    normalize = "normalize"


class Severity(StrEnum):
    error = "error"
    warning = "warning"
    info = "info"


class Status(StrEnum):
    auto_ok = "auto-ok"
    # documents only: every value is fine, but the document itself leaves something open
    # (convention unstated, holidays without dates); a person acknowledges that once
    caveats = "caveats"
    # a value set by the consensus correction (OCR and vision agree against the text reading)
    auto_corrected = "auto-corrected"
    needs_review = "needs-review"
    verified = "verified"
    edited = "edited"
    rejected = "rejected"


class RuleKind(StrEnum):
    workdays_only = "workdays_only"
    bridge_days = "bridge_days"
    holiday_handling = "holiday_handling"
    christmas_period = "christmas_period"
    significance_threshold = "significance_threshold"
    min_kw_diff = "min_kw_diff"
    de_minimis_eur = "de_minimis_eur"
    other = "other"


# --------------------------------------------------------------------------------------
# Raw LLM output
# --------------------------------------------------------------------------------------

class _Raw(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Cited(_Raw):
    page: int = Field(ge=1)
    quote: str


class CitedStr(Cited):
    value: str


class CitedInt(Cited):
    value: int


class RawWindow(_Raw):
    level_label: str
    season_label: str
    start: str
    end: str
    page: int = Field(ge=1)
    quote: str
    # When start and end are printed in separate cells (von / bis columns), `quote` is the
    # start cell and `quote_end` the end cell.
    quote_end: str | None = None
    level_quote: str = ""
    season_quote: str = ""


class RawEmptyCell(_Raw):
    level_label: str
    season_label: str
    page: int = Field(ge=1)


class RawLevel(_Raw):
    label: str
    page: int = Field(ge=1)
    quote: str


class WorkedExample(_Raw):
    listed_start: str
    first_timestamp: str


class RawConvention(_Raw):
    meter_label: Literal["start", "end", "unstated"] = "unstated"
    page: int | None = None
    quote: str | None = None
    worked_example: WorkedExample | None = None


class RawRule(_Raw):
    kind: str
    value: Any = None
    level_label: str | None = None
    page: int = Field(ge=1)
    quote: str


class RawExtraction(_Raw):
    dso_name: CitedStr
    validity_year: CitedInt
    publication_date: CitedStr | None = None
    version_label: CitedStr | None = None
    correction_note: CitedStr | None = None
    convention: RawConvention = Field(default_factory=RawConvention)
    levels_listed: list[RawLevel] = Field(default_factory=list)
    windows: list[RawWindow] = Field(default_factory=list)
    empty_cells: list[RawEmptyCell] = Field(default_factory=list)
    rules: list[RawRule] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Corpus + parsed pages
# --------------------------------------------------------------------------------------

class CorpusEntry(BaseModel):
    id: str
    dso: str
    year: int
    url: str | None = None
    states: list[str] = Field(default_factory=list)
    version_label: str | None = None
    notes: str = ""
    sha256: str | None = None
    synthetic: bool = False
    wayback: str | None = None
    # "corpus" (corpus.yaml), "upload" (added through the UI or `hlzf add`), "synthetic"
    source: str = "corpus"
    filename: str | None = None
    uploaded_by: str | None = None
    uploaded_at: str | None = None


class Line(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]


class PageText(BaseModel):
    page_no: int
    text: str
    lines: list[Line]
    width: float
    height: float
    text_layer: Literal["ok", "empty", "garbled"]
    source: Literal["text_layer", "ocr"] = "text_layer"
    layout: str = ""  # layout-preserving text for the extractor (see parse.py)


# --------------------------------------------------------------------------------------
# Canonical values
# --------------------------------------------------------------------------------------

class Evidence(BaseModel):
    page: int
    quote: str
    bbox: tuple[float, float, float, float] | None = None
    match_ratio: float = 0.0
    cell_aligned: bool | None = None


class Window(BaseModel):
    grid_level: str
    season: Season
    start_min: int
    end_min: int
    raw_level_label: str
    evidence: Evidence

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.grid_level, self.season.value, self.start_min, self.end_min)


class Rule(BaseModel):
    kind: RuleKind
    value: Any
    grid_level: str | None = None
    evidence: Evidence
    resolved: bool = True


class Issue(BaseModel):
    code: str
    message: str
    severity: Severity
    suspected_stage: Stage
    target: str | None = None  # entity id of the affected value, if any
    attribution: dict[str, Any] | None = None


def fmt_min(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"
