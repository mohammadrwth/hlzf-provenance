import pytest

from hlzf.attribution import channel_agreement, discriminate, wilson_interval


def test_wilson_interval_matches_reference_values():
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.4345, abs=1e-3)
    lo, hi = wilson_interval(5, 5)
    assert lo == pytest.approx(0.5655, abs=1e-3) and hi == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


# discriminate() only passes samples to the symptom/reading callables, so plain strings
# stand in for normalized extractions here.
def symptom(sample) -> bool:
    return sample == "bad"


def test_escape_under_resampling_is_extract_fault():
    v = discriminate(symptom, ["good"] * 5, swap_run=None, reading=lambda s: s)
    assert v.stage == "extract" and v.confidence == "high"
    assert v.suggestion == "resamples read good (5/5)"


def test_persistence_then_swap_gone_is_parse_fault():
    v = discriminate(symptom, ["bad"] * 5, swap_run=lambda: ("good", {}))
    assert v.stage == "parse" and v.method == "resample+swap" and v.confidence == "high"
    assert v.swap["symptom_after_swap"] is False


def test_persistence_through_swap_is_source_document():
    v = discriminate(symptom, ["bad"] * 5, swap_run=lambda: ("bad", {}))
    assert v.stage == "source_document"


def test_cross_check_where_vision_is_the_outlier():
    v = discriminate(symptom, ["bad"] * 5, swap_run=lambda: ("bad", {}), cross_check=True)
    assert v.stage == "vision_misread"


def test_thresholds_follow_htrace():
    # 2/5 = 0.4 is still "implementation" (<= 0.4), but the interval does not exclude 0.5
    v = discriminate(symptom, ["bad", "good", "bad", "good", "good"], swap_run=None)
    assert v.stage == "extract" and v.confidence == "medium"
    # 2/4 = 0.5 sits between the thresholds
    v = discriminate(symptom, ["bad", "good", "bad", "good"], swap_run=None)
    assert v.stage == "inconclusive" and v.p_hat == pytest.approx(0.5)
    # 3/5 = 0.6 points upstream and triggers the parse swap
    v = discriminate(symptom, ["bad", "good", "bad", "good", "bad"],
                     swap_run=lambda: ("good", {}))
    assert v.stage == "parse" and v.confidence == "medium"


def test_errored_resamples_cap_confidence():
    v = discriminate(symptom, ["good", None, None, None, None], swap_run=None)
    assert v.stage == "extract" and v.confidence == "low" and v.valid == 1


def test_no_swap_available_is_inconclusive():
    v = discriminate(symptom, ["bad"] * 5, swap_run=lambda: (None, {}))
    assert v.stage == "inconclusive"


def test_confounded_swap_lowers_confidence():
    v = discriminate(symptom, ["bad"] * 5,
                     swap_run=lambda: ("good", {"confounded": "model changed too"}))
    assert v.stage == "parse" and v.confidence == "medium"


def test_channel_agreement():
    assert channel_agreement({(1, 2)}, {(1, 2)}).stage == "source_document"
    assert channel_agreement({(1, 2)}, {(1, 3)}) is None
    assert channel_agreement({(1, 2)}, None) is None


# --- cross-check disagreements: four readings ------------------------------------------
from hlzf.attribution import discriminate_cross_check  # noqa: E402


def ident(x):
    return x


def xc(text, vision, resamples, ocr, meta=None):
    return discriminate_cross_check(ident, text, vision, resamples,
                                    (lambda: (ocr, meta or {})) if ocr is not ...
                                    else None)


def test_resamples_escape_to_the_page_reading_is_extract():
    v = xc("T", "V", ["V"] * 5, "V")
    assert v.stage == "extract" and v.confidence == "high"


def test_resamples_repeat_the_text_reading_is_parse():
    v = xc("T", "V", ["T"] * 5, "V")
    assert v.stage == "parse" and v.confidence == "high"
    assert v.swap["symptom_after_swap"] is False


def test_scattering_resamples_point_at_parse_not_extract():
    # Wunsiedel: resamples on scrambled text read five different things
    v = xc("T", "V", ["A", "B", "T", "C", "V"], "V")
    assert v.stage == "parse" and "scatter" in v.explanation


def test_text_and_ocr_against_vision_is_a_vision_misread():
    assert xc("T", "V", ["T"] * 5, "T").stage == "vision_misread"


def test_three_readings_are_inconclusive():
    assert xc("T", "V", ["T"] * 5, "O").stage == "inconclusive"


def test_without_independent_ocr_it_falls_back_to_resampling():
    v = xc("T", "V", ["V"] * 5, None)
    assert v.stage == "extract" and v.method == "resample"
    v = xc("T", "V", ["T"] * 5, "V", {"confounded": "vision used instead of OCR"})
    assert v.method == "resample+swap" and v.confidence == "medium"
