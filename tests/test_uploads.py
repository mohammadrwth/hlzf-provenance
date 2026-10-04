"""PDFs uploaded through the web UI (or `hlzf add`): validation, deduplication, background
processing with progress, identity read from the document, PROV, removal."""

import sqlite3

import pymupdf
import pytest
from fastapi.testclient import TestClient

from conftest import make_settings
from hlzf import fixtures, uploads
from hlzf.corpus import load_corpus
from hlzf.extract import Gateway
from hlzf.llm import LLMClient
from hlzf.pipeline import Runtime, open_runtime, process_document
from hlzf.prov import Prov
from hlzf.store import connect, loads, row, rows
from hlzf.web.app import create_app
from test_live_path import fake_api


def _pdf(tmp_path, doc_id="talwerk-2026"):
    p = tmp_path / f"{doc_id}.pdf"
    fixtures.build_pdf(fixtures.spec_by_id(doc_id), p)
    return p.read_bytes()


def _live_app(tmp_path, doc_id="talwerk-2026"):
    s = make_settings(tmp_path / "data", offline=False, api_key="test-key", reviewer="Mo")
    s.ensure_dirs()
    chat, http, calls = fake_api(doc_id)

    def factory(log):
        conn = connect(s.db_path)
        prov = Prov(conn)
        llm = LLMClient(s, conn, chat_fn=chat, http=http)
        return Runtime(s, conn, prov, Gateway(s, conn, prov, llm), log)

    app = create_app(s, runtime_factory=factory)
    return s, app, TestClient(app), calls


def _upload(client, body, name="HLZF_2026.pdf", **form):
    return client.post("/upload", files=[("files", (name, body, "application/pdf"))],
                       data=form, follow_redirects=False)


# --- validation and registry ---------------------------------------------------------------

def test_inspect_rejects_what_is_not_a_usable_pdf(tmp_path):
    s = make_settings(tmp_path / "d", upload_max_pages=2)
    with pytest.raises(uploads.UploadError, match="not a PDF"):
        uploads.inspect_pdf(b"hello", s)
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    with pytest.raises(uploads.UploadError, match="3 pages; the limit is 2"):
        uploads.inspect_pdf(doc.tobytes(), s)
    locked = pymupdf.open()
    locked.new_page().insert_text((50, 50), "x")
    enc = locked.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    with pytest.raises(uploads.UploadError, match="password"):
        uploads.inspect_pdf(enc, s)
    ok = uploads.inspect_pdf(_pdf(tmp_path), s)
    assert ok.pages == 1 and ok.looks_like_hlzf is True


def test_ids_and_dso_names():
    assert uploads.slug("Stadtwerke Fürth Netz GmbH & Co. KG") == "stadtwerke-fuerth-netz"
    assert uploads.make_id("N-ERGIE Netz GmbH", 2025, "ab" * 32, set()) == "n-ergie-netz-2025"
    assert uploads.make_id("N-ERGIE Netz GmbH", 2025, "ab" * 32,
                           {"n-ergie-netz-2025"}) == "n-ergie-netz-2025-ababab"
    assert uploads.make_id("", None, "cd" * 32, set()) == "upload-cdcdcdcdcd"
    assert uploads.same_dso("N-ERGIE Netz GmbH", "N-ERGIE Netz")
    assert uploads.same_dso("Stadtwerke Passau GmbH", "Stadtwerke Passau Netz GmbH")
    assert not uploads.same_dso("Stadtwerke Passau GmbH", "Stadtwerke Bayreuth GmbH")


def test_register_deduplicates_by_content(tmp_path):
    s = make_settings(tmp_path / "d")
    s.ensure_dirs()
    body = _pdf(tmp_path)
    first = uploads.register(s, None, body, "a.pdf", "Mo")
    assert first.entry and first.entry.id.startswith("upload-")
    assert uploads.pdf_path(s, first.entry.id).read_bytes() == body
    again = uploads.register(s, None, body, "renamed.pdf", "Mo")
    assert again.entry is None and again.duplicate_of == first.entry.id
    # uploads join the corpus that `hlzf run` processes, after corpus.yaml
    ids = [e.id for e in load_corpus(s, include_synthetic=False)]
    assert ids[-1] == first.entry.id
    assert load_corpus(s, include_synthetic=False, include_uploads=False) == [
        e for e in load_corpus(s, include_synthetic=False) if e.source != "upload"]


def test_old_database_gains_the_new_columns(tmp_path):
    db = tmp_path / "old.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, entity_id TEXT NOT NULL, "
              "dso TEXT NOT NULL, year INTEGER NOT NULL, status TEXT)")
    c.execute("INSERT INTO documents VALUES ('x', 'e', 'D', 2026, 'verified')")
    c.commit()
    c.close()
    conn = connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
    assert {"source", "filename", "uploaded_by"} <= cols
    assert row(conn, "SELECT source FROM documents WHERE id='x'")["source"] == "corpus"
    assert row(conn, "SELECT COUNT(*) n FROM jobs")["n"] == 0


# --- web: the whole flow -------------------------------------------------------------------

def test_upload_is_processed_in_the_background_and_lands_on_its_page(tmp_path):
    s, app, client, calls = _live_app(tmp_path)
    body = _pdf(tmp_path)
    r = _upload(client, body)
    assert r.status_code == 303 and r.headers["location"].startswith("/uploads?jobs=")
    jid = int(r.headers["location"].split("=")[1].split("#")[0])
    app.state.jobs.wait()

    job = row(connect(s.db_path), "SELECT * FROM jobs WHERE jid=?", (jid,))
    assert job["status"] == "done", job
    res = loads(job["result"])
    # identity read from the document, since the upload carried no hints
    assert res["dso"] == "Talwerk Energienetze GmbH" and res["year"] == 2026
    assert res["doc_status"] == "caveats" and res["live_calls"] == 9 and res["cost"] > 0
    assert calls == {"chat": 8, "ocr": 1}
    doc_id = job["document_id"]

    conn = connect(s.db_path)
    d = row(conn, "SELECT * FROM documents WHERE id=?", (doc_id,))
    assert (d["source"], d["filename"], d["uploaded_by"], d["dso"], d["year"]) == (
        "upload", "HLZF_2026.pdf", "Mo", "Talwerk Energienetze GmbH", 2026)
    # PROV: the document was generated by an Upload, associated with the person who uploaded
    up = row(conn, "SELECT a.* FROM was_generated_by g JOIN activity a ON a.id=g.activity_id "
                   "WHERE g.entity_id=?", (f"hlzf:doc/{doc_id}",))
    assert up["type"] == "hlzf:Upload"
    agents = {(r["agent_id"], r["role"]) for r in rows(
        conn, "SELECT agent_id, role FROM was_associated_with WHERE activity_id=?", (up["id"],))}
    assert ("hlzf:agent/person/mo", "uploader") in agents

    # progress card: terminal, links the document; ?auto=1 redirects there via HTMX
    frag = client.get(f"/upload/job/{jid}")
    assert "Open the document" in frag.text and "hx-trigger" not in frag.text
    assert client.get(f"/upload/job/{jid}?auto=1").headers["HX-Redirect"] == f"/doc/{doc_id}"
    page = client.get(f"/doc/{doc_id}").text
    assert "Uploaded by Mo" in page and "Remove this upload" in page
    assert client.get(f"/doc/{doc_id}/pdf").content == body
    home = client.get("/").text
    assert "Uploaded publications" in home and ">uploaded</span>" in home

    # the same bytes again: recorded, not processed twice
    r2 = _upload(client, body, name="copy.pdf")
    jid2 = int(r2.headers["location"].split("=")[1].split("#")[0])
    assert row(conn, "SELECT status FROM jobs WHERE jid=?", (jid2,))["status"] == "duplicate"
    assert "already in the corpus as" in client.get(f"/upload/job/{jid2}").text
    assert calls == {"chat": 8, "ocr": 1}

    # removal: file and derived rows go, the PROV graph records who withdrew it
    r3 = client.post(f"/doc/{doc_id}/remove", data={"reviewer": "Mo"}, follow_redirects=False)
    assert r3.status_code == 303
    conn = connect(s.db_path)
    assert row(conn, "SELECT 1 FROM documents WHERE id=?", (doc_id,)) is None
    assert not uploads.pdf_path(s, doc_id).exists() and uploads.load_registry(s) == {}
    assert row(conn, "SELECT 1 FROM activity WHERE type='hlzf:Withdraw'") is not None
    assert row(conn, "SELECT 1 FROM entity WHERE id=?", (f"hlzf:doc/{doc_id}",)) is not None


def test_hints_name_the_upload_and_a_wrong_hint_is_an_issue(tmp_path):
    s, app, client, _ = _live_app(tmp_path)
    r = _upload(client, _pdf(tmp_path), dso="Ganz Andere Stadtwerke GmbH", year="2026",
                states=["BY"])
    app.state.jobs.wait()
    conn = connect(s.db_path)
    d = row(conn, "SELECT * FROM documents WHERE source='upload'")
    assert d["id"] == "ganz-andere-stadtwerke-2026" and loads(d["states"]) == ["BY"]
    assert row(conn, "SELECT 1 FROM issues WHERE document_id=? AND code='DSO_MISMATCH'",
               (d["id"],)) is not None
    assert r.status_code == 303


def test_offline_upload_fails_with_a_reason_and_can_be_retried(tmp_path):
    s = make_settings(tmp_path / "data", reviewer="Mo")  # offline, no key
    app = create_app(s)
    client = TestClient(app)
    assert "Offline mode" in client.get("/uploads").text
    r = _upload(client, _pdf(tmp_path))
    jid = int(r.headers["location"].split("=")[1].split("#")[0])
    app.state.jobs.wait()
    conn = connect(s.db_path)
    job = row(conn, "SELECT * FROM jobs WHERE jid=?", (jid,))
    assert job["status"] == "failed" and "ZAI_API_KEY" in job["error"]
    card = client.get(f"/upload/job/{jid}").text
    assert "Retry" in card and 'class="failed"' in card
    r2 = client.post(f"/upload/job/{jid}/retry", follow_redirects=False)
    assert r2.status_code == 303
    app.state.jobs.wait()
    assert row(conn, "SELECT COUNT(*) n FROM jobs")["n"] == 2


def test_rejected_files_and_bad_year(tmp_path):
    s = make_settings(tmp_path / "data")
    client = TestClient(create_app(s))
    r = _upload(client, b"not a pdf at all", name="notes.txt")
    jid = int(r.headers["location"].split("=")[1].split("#")[0])
    card = client.get(f"/upload/job/{jid}").text
    assert "Not accepted" in card and "not a PDF" in card
    r = _upload(client, _pdf(tmp_path), year="soon")
    assert "error=Year" in r.headers["location"]


def test_an_upload_in_progress_cannot_be_removed(tmp_path):
    s = make_settings(tmp_path / "data")
    s.ensure_dirs()
    conn = connect(s.db_path)
    reg = uploads.register(s, conn, _pdf(tmp_path), "a.pdf", "Mo")
    conn.execute("INSERT INTO jobs (document_id, filename, status, created_at) "
                 "VALUES (?, 'a.pdf', 'running', '2026-10-04T06:00:00+00:00')", (reg.entry.id,))
    with pytest.raises(uploads.UploadError, match="still being processed"):
        uploads.remove(s, conn, reg.entry.id, "Mo")


def test_interrupted_jobs_are_marked_on_restart(tmp_path):
    s = make_settings(tmp_path / "data")
    s.ensure_dirs()
    conn = connect(s.db_path)
    conn.execute("INSERT INTO jobs (document_id, filename, status, created_at) "
                 "VALUES ('x', 'x.pdf', 'running', '2026-10-04T06:00:00+00:00')")
    conn.commit()
    create_app(s)
    assert row(conn, "SELECT status FROM jobs")["status"] == "interrupted"


def test_cli_path_reuses_the_cache(tmp_path):
    """`hlzf add` and a later `hlzf run` see the same entry; an unchanged upload is skipped."""
    s, app, client, _ = _live_app(tmp_path)
    _upload(client, _pdf(tmp_path))
    app.state.jobs.wait()
    rt = open_runtime(make_settings(tmp_path / "data"), log=lambda m: None)  # offline now
    [entry] = [e for e in load_corpus(rt.settings, include_synthetic=False)
               if e.source == "upload"]
    assert process_document(rt, entry).skipped == "unchanged"
    # forced re-run offline: replayed from the cache, same identity and outcome
    res = process_document(rt, entry, force=True)
    assert res.status == "caveats"
    assert row(rt.conn, "SELECT dso FROM documents WHERE id=?",
               (entry.id,))["dso"] == "Talwerk Energienetze GmbH"


def test_upload_never_pairs_with_a_fictional_fixture_of_the_same_name(rt, tmp_path):
    """The synthetic talwerk-2026 and an upload naming the same (fictional) DSO stay apart:
    no supersession/drift issue, no sibling link, and the fixture id still finds the fixture."""
    from datetime import datetime

    from hlzf.query import check

    s = make_settings(rt.settings.data_dir, offline=False, api_key="k", reviewer="Mo")
    chat, http, _ = fake_api("talwerk-2026")

    def factory(log):
        conn = connect(s.db_path)
        prov = Prov(conn)
        return Runtime(s, conn, prov, Gateway(s, conn, prov, LLMClient(
            s, conn, chat_fn=chat, http=http)), log)

    app = create_app(s, runtime_factory=factory)
    doc = pymupdf.open(stream=_pdf(tmp_path), filetype="pdf")
    doc.set_metadata({"title": "other bytes, same content"})
    _upload(TestClient(app), doc.tobytes())
    app.state.jobs.wait()
    conn = connect(s.db_path)
    up = row(conn, "SELECT * FROM documents WHERE source='upload'")
    assert up["dso"] == "Talwerk Energienetze GmbH" and up["synthetic"] == 0
    assert row(conn, "SELECT 1 FROM issues WHERE code IN ('SUPERSEDES', 'YOY_DRIFT') AND "
                     "(document_id=? OR message LIKE ?)", (up["id"], f"%{up['id']}%")) is None
    probe = check(conn, "talwerk-2026", "NE5", datetime.fromisoformat("2026-01-15T08:00+01:00"))
    assert probe["document"] == "talwerk-2026"
    probe = check(conn, "Talwerk Energienetze GmbH", "NE5",
                  datetime.fromisoformat("2026-01-15T08:00+01:00"))
    assert probe["document"] == up["id"]  # by name, the real publication wins


def test_cli_add_and_remove(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from hlzf.cli import app as cli

    monkeypatch.setenv("HLZF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HLZF_OFFLINE", "1")
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    f = tmp_path / "Netz_HLZF.pdf"
    f.write_bytes(_pdf(tmp_path))
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"nope")
    res = CliRunner().invoke(cli, ["add", str(f), str(junk), "--by", "Mo"])
    assert res.exit_code == 0, res.output
    assert "stored as upload-" in res.output and "not a PDF" in res.output
    assert "not extracted" in res.output  # offline, nothing cached for it
    again = CliRunner().invoke(cli, ["add", str(f)])
    assert "already known as upload-" in again.output
    doc_id = next(iter(uploads.load_registry(make_settings(tmp_path / "data"))))
    rm = CliRunner().invoke(cli, ["remove-upload", doc_id, "--by", "Mo"])
    assert rm.exit_code == 0 and f"{doc_id} removed" in rm.output


def test_printed_name_repairs_an_abbreviated_operator_name():
    """Seen in a live run: GLM-5.3 returned "SWW WUNSI... (SWW WUNSI...)" as the operator."""
    from hlzf.models import Line, PageText

    page = PageText(page_no=1, text="SWW WUNSIEDEL GMBH\nIndividuelle Netzentgelte nach ...\n"
                    "SWW WUNSIEDEL GMBH veröffentlicht für 2024 folgende Fenster:",
                    lines=[Line(text="SWW WUNSIEDEL GMBH", bbox=(60, 43, 300, 61)),
                           Line(text="Individuelle Netzentgelte nach ...",
                                bbox=(60, 70, 400, 80)),
                           Line(text="SWW WUNSIEDEL GMBH veröffentlicht für 2024 folgende "
                                     "Fenster:", bbox=(60, 90, 500, 100))],
                    width=595, height=842, text_layer="ok")
    pages = {1: page}
    assert uploads.printed_name(pages, "SWW WUNSI... (SWW WUNSI...)") == "SWW WUNSIEDEL GMBH"
    assert uploads.printed_name(pages, "SWW Wunsiedel GmbH") == "SWW Wunsiedel GmbH"
    assert uploads.printed_name(pages, "Stadtwerke Irgendwo GmbH") is None
    assert uploads.printed_name(pages, "SW…") is None  # too little to go on


def test_upload_takes_the_printed_name_when_the_model_abbreviates_it(tmp_path):
    import json as _json

    s = make_settings(tmp_path / "data", offline=False, api_key="test-key", reviewer="Mo")
    s.ensure_dirs()
    chat0, http, _ = fake_api("talwerk-2026")

    def chat(**kw):
        resp = chat0(**kw)
        body = _json.loads(resp.choices[0].message.content)
        body["dso_name"]["value"] = "Talwerk Energ... (Talwerk Energ...)"
        resp.choices[0].message.content = _json.dumps(body)
        return resp

    def factory(log):
        conn = connect(s.db_path)
        prov = Prov(conn)
        return Runtime(s, conn, prov, Gateway(s, conn, prov, LLMClient(
            s, conn, chat_fn=chat, http=http)), log)

    app = create_app(s, runtime_factory=factory)
    _upload(TestClient(app), _pdf(tmp_path))
    app.state.jobs.wait()
    conn = connect(s.db_path)
    d = row(conn, "SELECT * FROM documents WHERE source='upload'")
    assert d["dso"] == "Talwerk Energienetze GmbH"
    note = row(conn, "SELECT * FROM issues WHERE document_id=? AND code='DSO_NAME_FROM_PAGE'",
               (d["id"],))
    assert note and note["severity"] == "info" and "Talwerk Energ..." in note["message"]
