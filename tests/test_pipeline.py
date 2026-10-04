from hlzf.evaluate import EXPECTED_VERDICTS, attribution_check, markdown_report, run_eval
from hlzf.pipeline import doc_summary, run_corpus
from hlzf.store import loads, row, rows


def test_injected_faults_are_attributed_to_the_right_stage(conn):
    results = attribution_check(conn)
    assert len(results) == len(EXPECTED_VERDICTS)
    for r in results:
        assert r["ok"], r


def test_parse_fault_comes_with_ocr_suggestion(conn):
    i = row(conn, "SELECT * FROM issues WHERE document_id='talwerk-2026' AND "
                  "code='CROSS_CHECK_DISAGREE'")
    att = loads(i["attribution"])
    assert att["stage"] == "parse" and att["method"] == "resample+swap"
    assert att["persisted"] == 5 and att["swap"]["symptom_after_swap"] is False
    assert att["suggested_cell"] == ["07:45-09:00", "17:00-19:15"]


def test_extract_fault_suggests_majority_reading(conn):
    i = row(conn, "SELECT * FROM issues WHERE document_id='alpenland-2026' AND "
                  "code='GROUNDING_VALUE_MISMATCH'")
    att = loads(i["attribution"])
    assert att["stage"] == "extract" and att["p_hat"] == 0
    assert "16:30-19:45" in att["suggested_cell"]


def test_wrong_values_never_pass_silently(conn):
    result = run_eval(conn, conn_root())
    for r in result["synthetic"]:
        assert r["wrong_flagged"] == r["wrong"], r
        assert r["missed_flagged"] == r["missed"], r
    muster = next(r for r in result["synthetic"] if r["document"] == "musterstadt-2026")
    # sample 0 dropped a window; the consensus correction put it back
    assert muster["raw_recall"] < 1 and muster["recall"] == 1 and muster["auto_corrected"] == 1


def conn_root():
    from hlzf.config import PROJECT_ROOT

    return PROJECT_ROOT


def test_scanned_page_went_through_ocr(conn):
    page = row(conn, "SELECT * FROM pages WHERE document_id='nordheim-2026'")
    assert page["source"] == "ocr"
    assert row(conn, "SELECT COUNT(*) n FROM windows WHERE document_id='nordheim-2026'")["n"] == 3


def test_cross_document_findings(conn):
    codes = {(r["document_id"], r["code"]) for r in rows(conn, "SELECT * FROM issues")}
    assert ("alpenland-2026", "YOY_DRIFT") in codes
    assert ("beispielstadt-2026-korr", "SUPERSEDES") in codes
    assert ("beispielstadt-2026-korr", "CORRECTION_NOTED") in codes


def test_clean_document_has_no_review_load(conn):
    d = {x["id"]: x for x in doc_summary(conn)}
    assert d["alpenland-2025"]["n_review"] == 0 and d["alpenland-2025"]["errors"] == 0


def test_rerun_is_idempotent(rt):
    before = rt.conn.execute("SELECT COUNT(*) FROM activity").fetchone()[0]
    results = run_corpus(rt, include_real=False)
    assert all(r.skipped == "unchanged" for r in results)
    assert rt.conn.execute("SELECT COUNT(*) FROM activity").fetchone()[0] == before


def test_report_renders(conn):
    report = markdown_report(run_eval(conn, conn_root()), doc_summary(conn))
    assert "Fault attribution on injected faults" in report and "✗" not in report


def test_every_value_has_a_generating_activity_and_agent(conn):
    orphans = rows(conn, "SELECT w.entity_id FROM windows w LEFT JOIN was_generated_by g ON "
                         "g.entity_id = w.entity_id WHERE g.entity_id IS NULL")
    assert orphans == []
    unassociated = rows(conn, "SELECT a.id FROM activity a LEFT JOIN was_associated_with w ON "
                              "w.activity_id = a.id WHERE w.activity_id IS NULL")
    assert unassociated == []


def test_golden_status_explains_uncounted_labels(conn, tmp_path):
    from hlzf.evaluate import golden_status

    gdir = tmp_path / "golden"
    gdir.mkdir()
    cells = "cells:\n- {grid_level: NE5, season: winter, windows: [16:00-19:00]}\n"
    (gdir / "a.yaml").write_text("document: alpenland-2026\nverified: false\n" + cells)
    (gdir / "b.yaml").write_text("document: alpenland-2026\nverified: true\n" + cells)
    (gdir / "c.yaml").write_text("document: not-processed\nverified: true\n" + cells)
    (gdir / "d.yaml").write_text("document: x\nverified: [unclosed\n")
    st = {g["file"]: g for g in golden_status(conn, tmp_path)}
    assert "verified: true" in st["a.yaml"]["hint"]
    assert st["b.yaml"]["state"] == "counted"
    assert "not been processed" in st["c.yaml"]["hint"]
    assert st["d.yaml"]["state"] == "unreadable"


def test_issues_command_lists_verdicts(rt, monkeypatch):
    from typer.testing import CliRunner

    from hlzf.cli import app

    monkeypatch.setenv("HLZF_DATA_DIR", str(rt.settings.data_dir))
    monkeypatch.setenv("HLZF_OFFLINE", "1")
    monkeypatch.setenv("COLUMNS", "200")
    out = CliRunner().invoke(app, ["issues", "talwerk-2026", "--all"])
    assert out.exit_code == 0, out.output
    assert "CROSS_CHECK_DISAGREE" in out.output and "parse (high" in out.output
    assert CliRunner().invoke(app, ["issues", "nope"]).exit_code == 1
