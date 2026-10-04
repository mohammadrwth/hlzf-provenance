from pathlib import Path

import pytest

from hlzf import fixtures
from hlzf.grounding import ground_window
from hlzf.models import Convention, RawExtraction, RuleKind
from hlzf.normalize import _rule_value, map_convention, normalize
from hlzf.parse import parse_pdf


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    out = tmp_path_factory.mktemp("pdf")
    built = {}
    for doc_id in ("alpenland-2026", "quellbach-2026", "talwerk-2026", "nordheim-2026"):
        p = out / f"{doc_id}.pdf"
        fixtures.build_pdf(fixtures.spec_by_id(doc_id), p)
        built[doc_id] = {pg.page_no: pg for pg in parse_pdf(p)}
    return built


def raw(doc_id: str, **patch) -> RawExtraction:
    data = fixtures.truth_extraction(fixtures.spec_by_id(doc_id))
    data.update(patch)
    return RawExtraction.model_validate(data)


def test_convention_mapping():
    base = fixtures.truth_extraction(fixtures.spec_by_id("musterstadt-2026"))

    def conv(meter, example=None):
        d = dict(base)
        d["convention"] = {"meter_label": meter, "worked_example": example}
        return map_convention(RawExtraction.model_validate(d))

    assert conv("unstated") is Convention.assumed
    assert conv("start") is Convention.interval_start
    assert conv("end") is Convention.interval_end_ambiguous
    # Ratingen: window 08:00-11:30 = meter stamps 08:15-11:30 -> physical span
    assert conv("end", {"listed_start": "08:00", "first_timestamp": "08:15"}) \
        is Convention.interval_end_physical
    assert conv("end", {"listed_start": "08:00", "first_timestamp": "08:00"}) \
        is Convention.interval_end_labels


def test_rule_values_are_coerced():
    assert _rule_value(RuleKind.significance_threshold, "20 %")[0] == 20.0
    assert _rule_value(RuleKind.de_minimis_eur, "500,00 €")[0] == 500.0
    assert _rule_value(RuleKind.bridge_days, "off_peak_max_one") == ("off_peak_max_one",
                                                                      False, None)
    v, resolved, _ = _rule_value(RuleKind.christmas_period, {"from": None, "to": None})
    assert not resolved
    v, resolved, problem = _rule_value(RuleKind.bridge_days, "sometimes")
    assert problem and not resolved


def test_normalize_maps_levels_and_minutes(pages):
    n = normalize(raw("alpenland-2026"), pages["alpenland-2026"])
    keys = n.window_keys()
    assert ("NE5", "winter", 465, 540) in keys
    assert ("NE4", "spring", 990, 1005) in keys
    assert n.convention is Convention.interval_end_physical
    assert {lv for _, lv in n.levels_listed} == {"NE3", "NE4", "NE5", "NE6", "NE7"}
    assert all(nw.grounding.found for nw in n.windows)
    assert all(nw.grounding.cell_aligned for nw in n.windows)


def test_unmapped_level_becomes_normalize_issue(pages):
    data = fixtures.truth_extraction(fixtures.spec_by_id("alpenland-2026"))
    data["windows"][0]["level_label"] = "Fernwärme"
    n = normalize(RawExtraction.model_validate(data), pages["alpenland-2026"])
    assert any(i.code == "LEVEL_UNMAPPED" and i.suspected_stage.value == "normalize"
               for i in n.issues)


def test_grounding_rejects_near_miss_times(pages):
    pg = pages["alpenland-2026"]
    ok = ground_window(pg, 1, "07:45 – 09:00 Uhr", "07:45", "09:00",
                       "Mittelspannung (MS)", "Winter")
    assert ok.found and ok.value_consistent and ok.cell_aligned
    # one digit off scores >= 90 in fuzzy matching but is not printed on the page
    near = ground_window(pg, 1, "07:45 – 19:00 Uhr", "07:45", "19:00")
    assert not near.found


def test_value_quote_mismatch(pages):
    g = ground_window(pages["alpenland-2026"], 1, "16:30 – 19:45 Uhr", "16:30", "19:15")
    assert g.found and g.value_consistent is False


def test_cell_alignment_detects_wrong_column(pages):
    g = ground_window(pages["alpenland-2026"], 1, "16:45 – 18:45 Uhr", "16:45", "18:45",
                      "Mittelspannung (MS)", "Winter")  # printed under Frühling
    assert g.cell_aligned is False


def test_cell_alignment_rows_layout(pages):
    pg = pages["quellbach-2026"]
    right = ground_window(pg, 1, "15:00 – 18:15 Uhr", "15:00", "18:15", "Umspannung HS/MS",
                          "Herbst")
    wrong = ground_window(pg, 1, "15:00 – 18:15 Uhr", "15:00", "18:15", "Umspannung HS/MS",
                          "Winter")
    assert right.cell_aligned is True
    assert wrong.cell_aligned is False


def test_bad_text_layer_reads_cleanly_but_wrong(pages):
    pg = pages["talwerk-2026"][1]
    assert pg.text_layer == "ok"  # invisible to heuristics: only the vision check sees it
    assert "01:45 – 09:00" in pg.text and "07:45 – 09:00" not in pg.text


def test_scanned_page_has_no_text_layer(pages):
    assert pages["nordheim-2026"][1].text_layer == "empty"


def test_ocr_fixture_has_page_geometry(tmp_path):
    from hlzf.extract import ocr_to_page

    pdf = tmp_path / "n.pdf"
    fixtures.build_pdf(fixtures.spec_by_id("nordheim-2026"), pdf)
    page = ocr_to_page(fixtures.scripted_ocr("nordheim-2026", pdf, 1), 1)
    assert page.source == "ocr"
    assert any("16:45 – 19:00 Uhr" in ln.text for ln in page.lines)
    assert all(0 <= ln.bbox[0] <= page.width for ln in page.lines)
    assert Path(pdf).exists()


# --- layout text, split von/bis cells, quotes over line breaks ---------------------------

def _table_pdf(path, cells, scrambled=False, header=("Winter", "Herbst")):
    """A small bordered table; `scrambled` writes the cells in shuffled content order, the
    way the Wunsiedel PDF stores them."""
    import random

    import pymupdf

    from hlzf.fixtures import _font

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    xs = [50, 200, 330, 460]
    items = [(xs[1] + 4, 120, header[0]), (xs[2] + 4, 120, header[1])]
    for r, (label, a, b) in enumerate(cells):
        y = 150 + 30 * r
        items += [(xs[0] + 4, y, label), (xs[1] + 4, y, a), (xs[2] + 4, y, b)]
    if scrambled:
        random.Random(7).shuffle(items)
    for x, y, t in items:
        tw = pymupdf.TextWriter(page.rect)
        tw.append((x, y), t, font=_font("helv"), fontsize=9)
        tw.write_text(page)
    for r in range(len(cells) + 1):
        for c in range(3):
            page.draw_rect(pymupdf.Rect(xs[c], 105 + 30 * r, xs[c + 1], 135 + 30 * r))
    doc.save(path)
    return path


def test_layout_text_keeps_columns_when_content_order_is_scrambled(tmp_path):
    p = _table_pdf(tmp_path / "t.pdf", [("NETZEBENE 5", "-", "16:00-19:00"),
                                        ("NETZEBENE 6", "17:00-20:00", "-")], scrambled=True)
    page = parse_pdf(p)[0]
    # content order is scrambled ...
    assert "NETZEBENE 5\n-\n16:00-19:00" not in page.text
    # ... but the layout text puts each value under its header
    import re

    lines = page.layout.splitlines()
    header = next(ln for ln in lines if "Winter" in ln)
    row5 = next(ln for ln in lines if re.search(r"NETZEBENE\s+5", ln))
    row6 = next(ln for ln in lines if re.search(r"NETZEBENE\s+6", ln))
    assert abs(row5.index("16:00-19:00") - header.index("Herbst")) <= 2
    assert abs(row6.index("17:00-20:00") - header.index("Winter")) <= 2


def test_extractor_input_uses_layout_text(tmp_path):
    from hlzf.prompts import text_user_message

    p = _table_pdf(tmp_path / "t.pdf", [("NETZEBENE 5", "-", "16:00-19:00")], scrambled=True)
    msg = text_user_message(parse_pdf(p))
    assert msg.startswith("=== PAGE 1 (LAYOUT TEXT) ===")


def test_split_von_bis_cells_are_grounded_together(tmp_path):
    p = _table_pdf(tmp_path / "t.pdf", [("MS", "12:15", "13:15"), ("MS", "15:00", "17:30")],
                   header=("von", "bis"))
    pg = {1: parse_pdf(p)[0]}
    ok = ground_window(pg, 1, "12:15", "12:15", "13:15", quote_end="13:15")
    assert ok.found and ok.value_consistent
    # 12:15 and 17:30 exist, but not side by side
    assert not ground_window(pg, 1, "12:15", "12:15", "17:30", quote_end="17:30").found
    wrong_value = ground_window(pg, 1, "12:15", "12:15", "13:30", quote_end="13:15")
    assert wrong_value.value_consistent is False


def test_short_quote_across_a_line_break():
    from hlzf.grounding import ground_quote
    from hlzf.models import Line, PageText

    text = "Lastverlagerung: HS 10%, MS 20%, MS/NS\n30%, NS 30% und"
    page = PageText(page_no=1, text=text, width=595, height=842, text_layer="ok",
                    lines=[Line(text="Lastverlagerung: HS 10%, MS 20%, MS/NS",
                                bbox=(50, 100, 400, 110)),
                           Line(text="30%, NS 30% und", bbox=(50, 130, 200, 140))])
    assert ground_quote({1: page}, 1, "MS/NS 30%").found
    assert not ground_quote({1: page}, 1, "MS/NS 31%").found


def test_misplaced_window_is_never_blamed_on_the_source(pages):
    from hlzf.models import CorpusEntry
    from hlzf.validate import validate

    data = fixtures.truth_extraction(fixtures.spec_by_id("alpenland-2026"))
    for w in data["windows"]:  # read the spring window into the winter column
        if w["level_label"] == "Mittelspannung (MS)" and w["season_label"] == "Frühling":
            w["season_label"] = w["season_quote"] = "Winter"
    n = normalize(RawExtraction.model_validate(data), pages["alpenland-2026"])
    entry = CorpusEntry(id="alpenland-2026", dso="x", year=2026)
    overlap = [f.issue for f in validate(n, entry, pages["alpenland-2026"])
               if f.issue.code == "OVERLAP"]
    assert overlap and overlap[0].suspected_stage.value == "extract"


# --- fixes after the second live run ------------------------------------------------------

def _text_page(text: str, layout: str = ""):
    from hlzf.models import PageText

    return {1: PageText(page_no=1, text=text, layout=layout, width=595, height=842,
                        text_layer="ok", lines=[])}


def _conv_raw(listed: str, first: str) -> RawExtraction:
    data = fixtures.truth_extraction(fixtures.spec_by_id("musterstadt-2026"))
    data["convention"] = {"meter_label": "end", "page": 1, "quote": "x",
                          "worked_example": {"listed_start": listed, "first_timestamp": first}}
    return RawExtraction.model_validate(data)


BAYREUTH = ("Die angegebenen Viertelstunden sind Zeitstempel aus den Lastgängen. \n"
            "Der Zeitstempel 09:15 Uhr definiert den Zeitraum von 09:00 Uhr bis 09:15 Uhr.")
RATINGEN = ("Bei den Zeiten ist jeweils das Ende des 1/4-Stunden-Intervalls angegeben (z.B. "
            "entspricht ein \nHochlastzeitfenster von 08:00 bis 11:30 den Messwerten mit den "
            "Zeitstempeln von 08:15 bis \n11:30).")


def test_timestamp_definition_is_not_a_window_example():
    from hlzf.normalize import resolve_convention

    # Bayreuth: the model read "timestamp 09:15 = 09:00-09:15" as window 09:00 -> stamp 09:15
    conv, _, quote, issue = resolve_convention(_conv_raw("09:00", "09:15"), _text_page(BAYREUTH))
    assert conv is Convention.interval_end_labels
    assert issue.code == "CONVENTION_CORRECTED" and "Zeitstempel" in quote
    # the same definition without "the listed quarter-hours are timestamps" stays open
    conv, *_ = resolve_convention(_conv_raw("09:00", "09:15"),
                                  _text_page(BAYREUTH.split("\n")[1]))
    assert conv is Convention.interval_end_ambiguous
    # Ratingen maps a window onto timestamps: the example stands
    conv, _, _, issue = resolve_convention(_conv_raw("08:00", "08:15"), _text_page(RATINGEN))
    assert conv is Convention.interval_end_physical and issue is None


def _nergie_page():
    """Geometry of the N-ERGIE 2026 table (PyMuPDF lines, rounded): wrapped level labels whose
    lines overlap by a point, and 17:30-18:15 printed twice (NE6 autumn, NE7 winter)."""
    from hlzf.models import Line, PageText

    spec = [((210, 255, 244, 269), "Winter"), ((479, 255, 514, 269), "Herbst"),
            ((62, 393.8, 138, 407.4), "Umspannung in"),
            ((62, 406.2, 176, 419.8), "Niederspannung MS/NS"),
            ((185, 400, 267, 414), "17:00 – 19:15 Uhr"),
            ((455, 400, 538, 414), "17:30 – 18:15 Uhr"),
            ((62, 425.5, 158, 439.1), "Niederspannung NS"),
            ((184, 419, 267, 433), "11:00 – 12:30 Uhr"),
            ((184, 432, 267, 445), "17:30 – 18:15 Uhr")]
    lines = [Line(text=t, bbox=b) for b, t in spec]
    return {1: PageText(page_no=1, text="\n".join(t for _, t in spec), width=595, height=842,
                        text_layer="ok", lines=lines)}


def test_repeated_range_under_wrapped_label_is_aligned():
    pg = _nergie_page()
    ne6 = "Umspannung    in\n  Niederspannung   MS/NS"
    # printed under Herbst in the NE6 row, and again under Winter in the NE7 row
    assert ground_window(pg, 1, "17:30 – 18:15 Uhr", "17:30", "18:15", ne6, "Herbst"
                         ).cell_aligned is True
    assert ground_window(pg, 1, "17:30 – 18:15 Uhr", "17:30", "18:15", "Niederspannung NS",
                         "Winter").cell_aligned is True
    assert ground_window(pg, 1, "17:00 – 19:15 Uhr", "17:00", "19:15", ne6, "Winter"
                         ).cell_aligned is True
    # the live run's row shift: NE7's 11:00-12:30 read as NE6 is not confirmed as aligned
    assert ground_window(pg, 1, "11:00 – 12:30 Uhr", "11:00", "12:30", ne6, "Winter"
                         ).cell_aligned is not True


def test_rule_quote_from_a_layout_row():
    from hlzf.grounding import ground_quote

    text = "HS/MS\nMS\n20%\n20%\n500 €\n500 €"
    layout = "HS/MS        20%        500 €\nMS           20%        500 €"
    pages = _text_page(text, layout)
    g = ground_quote(pages, 1, "HS/MS                    20%                     500 €")
    assert g.found and "layout" in g.note
    assert not ground_quote(pages, 1, "HS/MS 30% 500 €").found
    assert not ground_quote(_text_page(text), 1, "HS/MS 20% 500 €").found


def test_single_state_holidays_are_resolved():
    hol = RuleKind.holiday_handling
    assert _rule_value(hol, {"states": ["DE"], "mode": "unspecified"})[1]
    assert _rule_value(hol, {"states": ["BY"], "mode": "unspecified"})[1]
    assert not _rule_value(hol, {"states": ["BY", "BW"], "mode": "unspecified"})[1]
    assert not _rule_value(hol, {"states": [], "mode": "unspecified"})[1]


def test_blank_cell_seen_blank_by_both_channels_is_no_gap(pages):
    from hlzf.models import CorpusEntry
    from hlzf.validate import validate

    entry = CorpusEntry(id="alpenland-2026", dso="x", year=2026)
    truth = fixtures.truth_extraction(fixtures.spec_by_id("alpenland-2026"))
    vision = normalize(RawExtraction.model_validate(truth), pages["alpenland-2026"])
    skipped = dict(truth, empty_cells=truth["empty_cells"][1:])  # blank cell not listed
    n = normalize(RawExtraction.model_validate(skipped), pages["alpenland-2026"])
    codes = [f.issue.code for f in validate(n, entry, pages["alpenland-2026"], vision)]
    assert "COVERAGE_GAP" not in codes
    # without the second reading the same omission is a gap
    codes = [f.issue.code for f in validate(n, entry, pages["alpenland-2026"])]
    assert "COVERAGE_GAP" in codes


def _ruled_pdf(path, ruled: bool):
    """N-ERGIE style: the value of a row sits between two level labels, nearer the upper one."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Entnahmenetzebene", fontsize=9)
    page.insert_text((250, 60), "Winter", fontsize=9)
    rows = [("Umspannung in", "Niederspannung MS/NS", "17:00 - 19:15 Uhr"),
            ("Niederspannung NS", None, "11:00 - 12:30 Uhr")]
    for i, (a, b, v) in enumerate(rows):
        y = 100 + i * 50
        page.insert_text((50, y), a, fontsize=9)
        if b:
            page.insert_text((50, y + 22), b, fontsize=9)
        page.insert_text((250, y + 11), v, fontsize=9)
    if ruled:
        for y in (80, 95, 145, 195):
            page.draw_line((40, y), (400, y))
    doc.save(path)
    return path


def test_ruling_lines_enter_the_layout_text(tmp_path):
    from hlzf.parse import RULE_CHAR

    lay = parse_pdf(_ruled_pdf(tmp_path / "r.pdf", True))[0].layout.splitlines()
    rules = [i for i, ln in enumerate(lay) if ln.strip().startswith(RULE_CHAR)]
    assert len(rules) >= 3
    first = next(i for i, ln in enumerate(lay) if "17:00" in ln)
    second = next(i for i, ln in enumerate(lay) if "11:00" in ln)
    # each value has a rule above and below, and a different pair of rules from the other row
    def between(i):
        return max(r for r in rules if r < i), min(r for r in rules if r > i)

    assert between(first) != between(second)
    # the level label of the row lies in the same band as its value
    lv = next(i for i, ln in enumerate(lay) if "Niederspannung" in ln and "MS/NS" in ln)
    assert between(lv) == between(first)


def test_borderless_page_has_no_rules(tmp_path):
    from hlzf.parse import RULE_CHAR

    lay = parse_pdf(_ruled_pdf(tmp_path / "b.pdf", False))[0].layout
    assert RULE_CHAR not in lay
