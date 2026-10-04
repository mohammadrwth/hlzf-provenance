import json

import pytest

from hlzf.review import (
    ReviewError,
    acknowledge_issue,
    refresh_document_status,
    review_cell,
    review_window,
)
from hlzf.store import row, rows


def win(conn, doc, level, season, start):
    return row(conn, "SELECT * FROM windows WHERE document_id=? AND grid_level=? AND season=? "
                     "AND start_min=? AND active=1", (doc, level, season, start))


def test_edit_requires_comment_and_valid_times(conn):
    w = win(conn, "alpenland-2025", "NE5", "winter", 480)
    with pytest.raises(ReviewError):
        review_window(conn, w["wid"], "edit", "Mo", "", "07:45", "09:00")
    with pytest.raises(ReviewError):
        review_window(conn, w["wid"], "edit", "Mo", "typo", "07:50", "09:00")  # off grid
    with pytest.raises(ReviewError):
        review_window(conn, w["wid"], "reject", "Mo", "")


def test_edit_creates_revision_and_supersedes(conn):
    w = win(conn, "alpenland-2025", "NE5", "winter", 480)
    res = review_window(conn, w["wid"], "edit", "Mo", "page shows 08:15", "08:15", "09:00")
    old = row(conn, "SELECT * FROM windows WHERE wid=?", (w["wid"],))
    new = row(conn, "SELECT * FROM windows WHERE wid=?", (res["wid"],))
    assert old["active"] == 0 and new["active"] == 1
    assert (new["start_min"], new["status"], new["revision"]) == (495, "edited", 1)
    assert row(conn, "SELECT * FROM was_revision_of WHERE new_id=?", (new["entity_id"],))
    act = row(conn, "SELECT a.* FROM was_generated_by g JOIN activity a ON a.id=g.activity_id "
                    "WHERE g.entity_id=?", (new["entity_id"],))
    assert act["type"] == "hlzf:Review" and "page shows 08:15" in act["attrs"]
    with pytest.raises(ReviewError):
        review_window(conn, w["wid"], "approve", "Mo")  # superseded revision


def test_cell_issue_resolves_once_the_cell_is_decided(conn):
    a = win(conn, "quellbach-2026", "NE7", "winter", 360)
    b = win(conn, "quellbach-2026", "NE7", "winter", 780)
    review_window(conn, a["wid"], "approve", "Mo")
    issue = row(conn, "SELECT * FROM issues WHERE document_id='quellbach-2026' AND "
                      "code='DAILY_CAP_EXCEEDED'")
    assert issue["resolved"] == 0
    review_window(conn, b["wid"], "approve", "Mo")
    issue = row(conn, "SELECT * FROM issues WHERE iid=?", (issue["iid"],))
    assert issue["resolved"] == 1


def test_document_becomes_verified(conn):
    doc = "alpenland-2025"
    for w in rows(conn, "SELECT wid FROM windows WHERE document_id=? AND active=1", (doc,)):
        review_window(conn, w["wid"], "approve", "Mo")
    for i in rows(conn, "SELECT iid FROM issues WHERE document_id=? AND severity IN "
                        "('error','warning')", (doc,)):
        acknowledge_issue(conn, i["iid"], "Mo", "checked")
    assert refresh_document_status(conn, doc) == "verified"
    assert row(conn, "SELECT status FROM documents WHERE id=?", (doc,))["status"] == "verified"


def test_acknowledge_requires_comment(conn):
    i = row(conn, "SELECT iid FROM issues WHERE code='CONVENTION_ASSUMED' LIMIT 1")
    with pytest.raises(ReviewError):
        acknowledge_issue(conn, i["iid"], "Mo", " ")


def test_consensus_correction_fixes_the_lying_text_layer(conn):
    """talwerk: the text layer says 01:45, the page shows 07:45. OCR and vision agree on
    07:45 and the value is printed in the OCR page's cell: the pipeline sets it, as a PROV
    revision by the software agent, and settles the issue (resolved = 2)."""
    old = row(conn, "SELECT * FROM windows WHERE document_id='talwerk-2026' AND start_min=105")
    new = win(conn, "talwerk-2026", "NE5", "winter", 465)
    assert old["active"] == 0 and old["origin"] == "extract"
    assert (new["status"], new["origin"], new["revision"]) == ("auto-corrected", "auto", 1)
    assert row(conn, "SELECT * FROM was_revision_of WHERE new_id=? AND old_id=?",
               (new["entity_id"], old["entity_id"]))
    act = row(conn, "SELECT a.* FROM was_generated_by g JOIN activity a ON a.id=g.activity_id "
                    "WHERE g.entity_id=?", (new["entity_id"],))
    assert act["type"] == "hlzf:AutoCorrect"
    agent = row(conn, "SELECT ag.* FROM was_associated_with w JOIN agent ag ON "
                      "ag.id=w.agent_id WHERE w.activity_id=?", (act["id"],))
    assert agent["type"] == "prov:SoftwareAgent"
    issue = row(conn, "SELECT * FROM issues WHERE document_id='talwerk-2026' AND "
                      "code='CROSS_CHECK_DISAGREE'")
    att = json.loads(issue["attribution"])
    assert issue["resolved"] == 2 and att["auto_correction"]["applied"]
    assert att["auto_correction"]["changed"] == ["01:45-09:00 -> 07:45-09:00"]
    assert row(conn, "SELECT status FROM documents WHERE id='talwerk-2026'")["status"] \
        != "needs-review"


def test_consensus_adds_a_dropped_window(conn):
    w = win(conn, "musterstadt-2026", "NE6", "spring", 1080)  # 18:00-19:15, dropped by sample 0
    assert w and (w["status"], w["origin"]) == ("auto-corrected", "auto")
    assert w["quote"] and w["cell_aligned"] == 1
    gap = row(conn, "SELECT * FROM issues WHERE document_id='musterstadt-2026' AND "
                    "code='COVERAGE_GAP'")
    assert gap["resolved"] == 2


def test_person_sets_a_cell(conn):
    with pytest.raises(ReviewError):
        review_cell(conn, "quellbach-2026", "NE7", "winter", ["06:00-12:00"], "Mo", "")
    with pytest.raises(ReviewError):
        review_cell(conn, "quellbach-2026", "NE7", "winter", ["06:00-12:10"], "Mo", "x")
    res = review_cell(conn, "quellbach-2026", "NE7", "winter", ["06:00-12:00", "16:00-17:00"],
                      "Mo", "checked the page")
    assert res["kept"] == ["06:00-12:00"]
    assert res["changed"] == ["13:00-17:30 -> 16:00-17:00"]
    cell = rows(conn, "SELECT * FROM windows WHERE document_id='quellbach-2026' AND "
                      "grid_level='NE7' AND season='winter' AND active=1 ORDER BY start_min")
    assert [(w["start_min"], w["status"], w["origin"]) for w in cell] == [
        (360, "verified", "review"), (960, "edited", "review")]
    cap = row(conn, "SELECT * FROM issues WHERE document_id='quellbach-2026' AND "
                    "code='DAILY_CAP_EXCEEDED'")
    assert cap["resolved"] == 1
    # removing every value marks the cell as empty
    review_cell(conn, "quellbach-2026", "NE7", "winter", [], "Mo", "cell is blank")
    assert row(conn, "SELECT * FROM empty_cells WHERE document_id='quellbach-2026' AND "
                     "grid_level='NE7' AND season='winter'")
    assert not row(conn, "SELECT * FROM windows WHERE document_id='quellbach-2026' AND "
                         "grid_level='NE7' AND season='winter' AND active=1 AND "
                         "status != 'rejected'")


def test_person_can_undo_a_consensus_correction(conn):
    issue = row(conn, "SELECT * FROM issues WHERE document_id='talwerk-2026' AND "
                      "code='CROSS_CHECK_DISAGREE'")
    before = json.loads(issue["attribution"])["auto_correction"]["before"]
    assert before == ["01:45-09:00", "17:00-19:15"]
    res = review_cell(conn, "talwerk-2026", "NE5", "winter", before, "Mo", "text layer is right")
    assert res["changed"] == ["07:45-09:00 -> 01:45-09:00"]
    w = win(conn, "talwerk-2026", "NE5", "winter", 105)
    assert (w["status"], w["origin"], w["revision"]) == ("edited", "review", 2)
    assert row(conn, "SELECT resolved FROM issues WHERE iid=?", (issue["iid"],))["resolved"] == 1
