"""The live path end to end, with the Z.ai API replaced by fakes.

A synthetic PDF is registered as if it were a real DSO document, so the pipeline goes through
LLMClient (JSON mode, cache, cost log), renders pages for the vision call, OCRs via the
layout_parsing endpoint and resamples. Then the cache alone must reproduce the run offline:
that is what lets someone without a key run the demo.
"""

import json
from types import SimpleNamespace

import httpx
import yaml

from conftest import make_settings
from hlzf import fixtures
from hlzf.extract import Gateway
from hlzf.llm import LLMClient
from hlzf.pipeline import Runtime, run_corpus
from hlzf.prov import Prov
from hlzf.store import connect, loads, row, rows


def fake_api(doc_id: str):
    spec = fixtures.spec_by_id(doc_id)
    calls = {"chat": 0, "ocr": 0}

    def chat(**kw):
        calls["chat"] += 1
        user = kw["messages"][1]["content"]
        is_image = isinstance(user, list)
        # The text channel sees the corrupted text layer; images and OCR see the print.
        visible = is_image or "01:45" not in str(user)
        body = fixtures.truth_extraction(spec, visible=visible)
        usage = SimpleNamespace(prompt_tokens=3000, completion_tokens=1500,
                                prompt_tokens_details=None)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(body)))])

    def ocr(request: httpx.Request):
        calls["ocr"] += 1
        assert request.url.path.endswith("/layout_parsing")
        return httpx.Response(200, json=fixtures.scripted_ocr(doc_id, None, 1))

    return chat, httpx.Client(transport=httpx.MockTransport(ocr)), calls


def runtime(settings, chat=None, http=None):
    settings.ensure_dirs()
    conn = connect(settings.db_path)
    prov = Prov(conn)
    llm = LLMClient(settings, conn, chat_fn=chat, http=http)
    return Runtime(settings, conn, prov, Gateway(settings, conn, prov, llm), lambda m: None)


def test_live_run_then_offline_replay(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "corpus.yaml").write_text(yaml.safe_dump({"documents": [{
        "id": "talwerk-real", "dso": "Talwerk Energienetze GmbH", "year": 2026,
        "url": "https://example.invalid/hlzf.pdf", "states": ["BY"]}]}))
    data = tmp_path / "data"
    live = make_settings(data, root=root, offline=False, api_key="test-key")
    live.ensure_dirs()
    fixtures.build_pdf(fixtures.spec_by_id("talwerk-2026"), live.raw_dir / "talwerk-real.pdf")

    chat, http, calls = fake_api("talwerk-2026")
    rt = runtime(live, chat, http)
    [res] = run_corpus(rt, include_synthetic=False)
    # the lying text layer is outvoted by OCR and vision: only document caveats remain
    assert res.status == "caveats", res
    issue = row(rt.conn, "SELECT * FROM issues WHERE code='CROSS_CHECK_DISAGREE'")
    att = loads(issue["attribution"])
    assert att["stage"] == "parse" and att["swap"]["symptom_after_swap"] is False
    assert att["auto_correction"]["applied"] and issue["resolved"] == 2
    fixed = row(rt.conn, "SELECT * FROM windows WHERE active=1 AND grid_level='NE5' AND "
                         "season='winter' AND start_min=465")
    assert fixed["status"] == "auto-corrected" and fixed["origin"] == "auto"
    # 1 text + 1 vision + 5 resamples + 1 extraction on OCR text; 1 OCR call
    assert calls == {"chat": 8, "ocr": 1}
    spent = rt.conn.execute("SELECT SUM(cost_usd) FROM llm_calls WHERE source='live'"
                            ).fetchone()[0]
    assert spent > 0
    cache_files = list(live.cache_dir.rglob("*.json"))
    assert len(cache_files) == 9
    windows = {(w["grid_level"], w["season"], w["start_min"], w["end_min"]) for w in rows(
        rt.conn, "SELECT * FROM windows WHERE active=1")}
    rt.conn.close()

    # Offline: new database, no key, no API, only the cache.
    live.db_path.unlink()
    offline = make_settings(data, root=root)

    def no_api(**kw):
        raise AssertionError("offline replay must not call the API")

    rt2 = runtime(offline, no_api, httpx.Client(transport=httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(AssertionError("no OCR offline")))))
    [res2] = run_corpus(rt2, include_synthetic=False)
    assert res2.status == "caveats"
    replay = {(w["grid_level"], w["season"], w["start_min"], w["end_min"]) for w in rows(
        rt2.conn, "SELECT * FROM windows WHERE active=1")}
    assert replay == windows
    att2 = loads(row(rt2.conn, "SELECT * FROM issues WHERE code='CROSS_CHECK_DISAGREE'")
                 ["attribution"])
    assert att2["stage"] == "parse"
    sources = {r["source"] for r in rows(rt2.conn, "SELECT source FROM llm_calls")}
    assert sources == {"cache"}
