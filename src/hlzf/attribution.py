"""Interventional fault attribution (the Htrace method, applied to a document pipeline).

Htrace localises a fault by intervening and measuring whether the symptom survives, never by
asking a model whose fault it was. Here the same two interventions decide which pipeline stage
produced a suspicious value:

1. do(resample extract | parse fixed)
   Re-run the extractor N times on the *same* parsed pages. If the symptom reappears in most
   samples (p_hat >= 0.6) the input entails it, so the fault is upstream of extract. If it
   mostly disappears (p_hat <= 0.4) the original sample was at fault: an extract fault.
   p_hat gets a Wilson 95% interval; fewer than N/2 evaluable samples caps confidence at low.
   Thresholds, interval and quorum rule are the ones Htrace uses.

2. do(parse := GLM-OCR)   (only when step 1 points upstream)
   Replace the parse stage (PDF text layer) by OCR of the rendered page and run the *same*
   extractor and prompt. Symptom gone -> parse fault. Symptom still there -> the printed page
   says it: source_document. For cross-check disagreements, "still there" means two
   independent readings agree against the vision channel, so the vision check misread.

Rule violations on values that are grounded, self-consistent and read identically by the
vision channel need no resampling: two independent channels read the same print, so the
verdict is source_document by channel agreement.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .models import Season, Stage, fmt_min
from .normalize import Normalized

INSTRUCTION_THRESHOLD = 0.6  # p_hat >= 0.6: input reliably yields the symptom
IMPLEMENTATION_THRESHOLD = 0.4  # p_hat <= 0.4: the sampled extraction was at fault

Symptom = Callable[[Normalized], bool]
CellReading = Callable[[Normalized], Any]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class Verdict:
    stage: str  # a Stage value, "vision_misread" or "inconclusive"
    method: str
    confidence: str  # high | medium | low
    n: int = 0
    valid: int = 0
    persisted: int = 0
    p_hat: float | None = None
    ci: tuple[float, float] | None = None
    swap: dict[str, Any] | None = None
    suggestion: str | None = None
    suggested_cell: list[str] | None = None  # machine-readable form of the suggestion
    explanation: str = ""
    # True when two independent readings of the page image (GLM-OCR text through the same
    # extractor, and the vision model) agree on `suggested_cell` against the text channel
    consensus: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def fmt_cell(cell: Any) -> str:
    if isinstance(cell, (set, frozenset)):
        return ", ".join(f"{fmt_min(a)}-{fmt_min(b)}" for a, b in sorted(cell)) or "(empty)"
    return str(cell)


def cell_strings(cell: Any) -> list[str] | None:
    if isinstance(cell, (set, frozenset)):
        return [f"{fmt_min(a)}-{fmt_min(b)}" for a, b in sorted(cell)]
    return None


def cell_reading(level: str, season: Season) -> CellReading:
    def read(n: Normalized) -> frozenset[tuple[int, int]]:
        return frozenset((nw.window.start_min, nw.window.end_min) for nw in n.windows
                         if nw.window.grid_level == level and nw.window.season == season)
    return read


def discriminate(symptom: Symptom, resamples: list[Normalized | None],
                 swap_run: Callable[[], tuple[Normalized | None, dict[str, Any]]] | None,
                 reading: CellReading | None = None,
                 cross_check: bool = False) -> Verdict:
    n = len(resamples)
    valid = [r for r in resamples if r is not None]
    persisted = sum(1 for r in valid if symptom(r))
    p_hat = persisted / len(valid) if valid else None
    ci = wilson_interval(persisted, len(valid)) if valid else None
    quorum = len(valid) * 2 >= n
    suggestion, majority = None, None
    if reading is not None and valid:
        common = Counter(reading(r) for r in valid).most_common(1)[0]
        majority = common[0]
        suggestion = f"resamples read {fmt_cell(common[0])} ({common[1]}/{len(valid)})"

    base = dict(n=n, valid=len(valid), persisted=persisted, p_hat=p_hat, ci=ci)
    if p_hat is None:
        return Verdict(stage="inconclusive", method="resample", confidence="low",
                       explanation="no evaluable resamples", **base)
    excludes_half = ci is not None and (ci[0] > 0.5 or ci[1] < 0.5)

    if p_hat <= IMPLEMENTATION_THRESHOLD:
        conf = "high" if (excludes_half and quorum) else ("medium" if quorum else "low")
        return Verdict(
            stage=Stage.extract.value, method="resample", confidence=conf,
            suggestion=suggestion, suggested_cell=cell_strings(majority),
            explanation=(f"Re-extracting from the same parsed pages reproduced the symptom in "
                         f"{persisted}/{len(valid)} samples: the original extraction was at "
                         f"fault, not its input."), **base)
    if p_hat < INSTRUCTION_THRESHOLD:
        return Verdict(stage="inconclusive", method="resample", confidence="low",
                       suggestion=suggestion,
                       explanation=f"Symptom reproduced in {persisted}/{len(valid)} samples, "
                                   "between the thresholds.", **base)

    # Upstream of extract: swap the parse stage and hold model + prompt fixed.
    swapped, swap_meta = swap_run() if swap_run else (None, {})
    if swapped is None:
        return Verdict(stage="inconclusive", method="resample", confidence="low",
                       explanation="Symptom is stable under resampling (upstream of extract) "
                                   "but the parse swap (OCR) was not available.", **base)
    gone = not symptom(swapped)
    swap = {"intervention": "do(parse := glm-ocr)", **swap_meta, "symptom_after_swap": not gone}
    swap_cell = reading(swapped) if reading is not None else None
    if reading is not None:
        swap["swap_reading"] = fmt_cell(swap_cell)
    conf = "high" if (excludes_half and quorum) else ("medium" if quorum else "low")
    if swap_meta.get("confounded") and conf == "high":
        conf = "medium"
    if gone:
        return Verdict(
            stage=Stage.parse.value, method="resample+swap", confidence=conf, swap=swap,
            suggestion=(f"OCR reading: {swap['swap_reading']}" if "swap_reading" in swap
                        else None),
            suggested_cell=cell_strings(swap_cell),
            explanation=(f"Symptom survived {persisted}/{len(valid)} resamples, so the parsed "
                         "input entails it; replacing the text layer by OCR of the page image "
                         "removed it. The text layer, as parsed, does not carry what the page "
                         "shows."),
            **base)
    if cross_check:
        return Verdict(
            stage="vision_misread", method="resample+swap", confidence=conf, swap=swap,
            explanation=("Text layer and OCR both read the value the vision channel disputes; "
                         "two independent readings agree, so the vision check misread."),
            **base)
    return Verdict(
        stage=Stage.source_document.value, method="resample+swap", confidence=conf, swap=swap,
        explanation=("Symptom survived resampling and the parse swap: the printed document "
                     "itself says this."), **base)


def discriminate_cross_check(reading: CellReading, text: Normalized, vision: Normalized,
                             resamples: list[Normalized | None],
                             swap_run: Callable[[], tuple[Normalized | None, dict[str, Any]]]
                             | None) -> Verdict:
    """Text and vision channel read a table cell differently: who is right, and which stage
    of the text channel failed?

    Four readings of the same cell decide it: the original text extraction (t), the
    resamples on the same parsed text, the OCR swap (o, same model and prompt on GLM-OCR text
    of the page image) and the vision channel (v).

    * o == v != t: two readings of the page image agree against the text channel.
      - resamples escape to v (p_hat <= 0.4, a stable majority on v): the parsed text was
        enough, the original sample was wrong -> extract;
      - resamples keep t (p_hat >= 0.6) or scatter without a stable majority: the parsed text
        does not carry what the page shows -> parse. Scatter is the case resampling alone
        would misread as "extract": the first live run (Wunsiedel, content-stream order
        scrambled) produced it in most cells.
    * o == t != v: text layer and OCR agree against the vision model -> vision misread.
    * otherwise: three different readings -> inconclusive.
    """
    t, v = reading(text), reading(vision)
    n = len(resamples)
    valid = [r for r in resamples if r is not None]
    persisted = sum(1 for r in valid if reading(r) == t)
    p_hat = persisted / len(valid) if valid else None
    ci = wilson_interval(persisted, len(valid)) if valid else None
    quorum = bool(valid) and len(valid) * 2 >= n
    modal, modal_n = Counter(reading(r) for r in valid).most_common(1)[0] if valid else (None, 0)
    stable = bool(valid) and modal_n / len(valid) >= INSTRUCTION_THRESHOLD
    base = dict(n=n, valid=len(valid), persisted=persisted, p_hat=p_hat, ci=ci)

    swapped, meta = swap_run() if swap_run else (None, {})
    if swapped is None or meta.get("confounded"):
        # Without an independent OCR reading the decision falls back to resampling alone.
        return discriminate(lambda r: reading(r) == t, resamples, swap_run, reading,
                            cross_check=True)
    o = reading(swapped)
    swap = {"intervention": "do(parse := glm-ocr)", **meta, "swap_reading": fmt_cell(o),
            "symptom_after_swap": o == t}
    resample_txt = (f"{persisted}/{len(valid)} resamples on the same parsed text repeated "
                    f"the text reading" if valid else "no resamples were evaluable")
    if o == v and o != t:
        if p_hat is not None and p_hat <= IMPLEMENTATION_THRESHOLD and stable and modal == v:
            excludes_half = ci is not None and ci[1] < 0.5
            return Verdict(
                stage=Stage.extract.value, method="resample+swap",
                confidence="high" if (quorum and excludes_half) else "medium", swap=swap,
                suggestion=f"OCR and vision read {fmt_cell(v)}", suggested_cell=cell_strings(v),
                consensus=True,
                explanation=(f"OCR of the page and the vision model agree on {fmt_cell(v)}, "
                             f"and the resamples read the same ({modal_n}/{len(valid)}): the "
                             "parsed text was enough, the original extraction was wrong."),
                **base)
        scatter = not stable
        why = (f"the resamples scatter (most common reading only {modal_n}/{len(valid)}), so "
               "the parsed text does not determine this cell" if scatter else
               f"{resample_txt}, so the parsed text entails the wrong reading")
        return Verdict(
            stage=Stage.parse.value, method="resample+swap",
            confidence="high" if quorum and (scatter or (p_hat or 0) >= INSTRUCTION_THRESHOLD)
            else "medium", swap=swap,
            suggestion=f"OCR and vision read {fmt_cell(v)}", suggested_cell=cell_strings(v),
            consensus=True,
            explanation=(f"OCR of the page image and the vision model agree on {fmt_cell(v)}; "
                         f"{why}. The text layer, as parsed, does not carry what the page "
                         "shows (wrong characters, or a table whose structure got lost)."),
            **base)
    if o == t and o != v:
        return Verdict(
            stage="vision_misread", method="resample+swap",
            confidence="high" if (p_hat or 0) >= INSTRUCTION_THRESHOLD and quorum else "medium",
            swap=swap,
            explanation=(f"The text layer and OCR of the page both read {fmt_cell(t)}; two "
                         "independent readings agree, so the vision check misread."), **base)
    return Verdict(
        stage="inconclusive", method="resample+swap", confidence="low", swap=swap,
        explanation=(f"Text ({fmt_cell(t)}), OCR ({fmt_cell(o)}) and vision ({fmt_cell(v)}) "
                     "read this cell three different ways."), **base)


def channel_agreement(text_reading: Any, vision_reading: Any) -> Verdict | None:
    """Deterministic verdict for rule violations on grounded values."""
    if vision_reading is None:
        return None
    if text_reading == vision_reading:
        return Verdict(stage=Stage.source_document.value, method="channel_agreement",
                       confidence="high",
                       explanation=("The value is grounded verbatim in the text layer and the "
                                    "vision channel reads the same print; the document itself "
                                    "states it."))
    return None
