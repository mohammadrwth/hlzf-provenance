import json
from types import SimpleNamespace

import httpx
import pytest

from conftest import make_settings
from hlzf import fixtures
from hlzf.llm import (
    BudgetExceeded,
    LLMClient,
    ModelCallFailed,
    OfflineCacheMiss,
    cost_usd,
    extract_json,
)
from hlzf.models import RawExtraction
from hlzf.store import connect

GOOD = json.dumps(fixtures.truth_extraction(fixtures.spec_by_id("musterstadt-2026")))


def response(text, prompt=1000, completion=500, cached=0):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=usage)


class FakeChat:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return response(self.texts.pop(0))


def client(tmp_path, chat, **kw):
    s = make_settings(tmp_path / "data", **{"offline": False, "api_key": "test", **kw})
    s.ensure_dirs()
    return LLMClient(s, connect(s.db_path), chat_fn=chat)


def ask(c, user="page text", sample=0):
    return c.chat_json(model="glm-5.3", system="sys", user=user, schema=RawExtraction,
                       purpose="extract:text", prompt_version="t1", sample=sample,
                       document_id="d")


def test_live_call_is_cached_and_costed(tmp_path):
    chat = FakeChat(GOOD, GOOD)
    c = client(tmp_path, chat)
    raw, rec = ask(c)
    assert rec.source == "live" and rec.cost_usd == cost_usd("glm-5.3", 1000, 0, 500)
    kw = chat.calls[0]
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["extra_body"]["thinking"] == {"type": "enabled"}
    raw2, rec2 = ask(c)  # same request -> cache
    assert rec2.source == "cache" and len(chat.calls) == 1
    assert raw2 == raw
    _, rec3 = ask(c, sample=1)  # a resample is a different cache entry
    assert rec3.source == "live" and len(chat.calls) == 2
    assert c.spent_usd() == pytest.approx(2 * rec.cost_usd)


def test_repair_round(tmp_path):
    chat = FakeChat("```json\n{\"dso_name\": 1}\n```", GOOD)
    raw, rec = ask(client(tmp_path, chat))
    assert rec.repaired and len(chat.calls) == 2
    assert chat.calls[1]["messages"][-1]["content"].startswith("Your previous answer failed")


def test_offline_miss(tmp_path):
    c = client(tmp_path, FakeChat(GOOD), offline=True)
    with pytest.raises(OfflineCacheMiss):
        ask(c)


def test_budget_guard_blocks_before_calling(tmp_path):
    chat = FakeChat(GOOD)
    c = client(tmp_path, chat, budget_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        ask(c)
    assert chat.calls == []


def test_api_errors_are_wrapped(tmp_path):
    def boom(**kwargs):
        raise RuntimeError("401 Unauthorized")
    with pytest.raises(ModelCallFailed):
        ask(client(tmp_path, boom))


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert extract_json("Here you go: {\"a\": 2} thanks") == {"a": 2}


def test_layout_parse_retries_with_data_uri(tmp_path):
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append(body["file"][:30])
        if not body["file"].startswith("data:"):
            return httpx.Response(400, json={"error": "bad file"})
        return httpx.Response(200, json={"md_results": "x", "layout_details": [[]],
                                          "usage": {"prompt_tokens": 10,
                                                    "completion_tokens": 5}})
    s = make_settings(tmp_path / "data", offline=False, api_key="k")
    s.ensure_dirs()
    c = LLMClient(s, connect(s.db_path), http=httpx.Client(transport=httpx.MockTransport(
        handler)))
    data, rec = c.layout_parse(pdf_bytes=b"%PDF-1.7 test", page_no=1)
    assert data["md_results"] == "x" and len(seen) == 2 and seen[1].startswith("data:")
    again, rec2 = c.layout_parse(pdf_bytes=b"%PDF-1.7 test", page_no=1)
    assert rec2.source == "cache" and len(seen) == 2


# --- retries, parallel prefetch and budget reservation --------------------------------

def test_rate_limits_are_retried_with_backoff(tmp_path):
    waits = []
    answers = [RuntimeError("Error code: 429 - {'error': {'code': '1302'}}"), GOOD]

    def chat(**kwargs):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return response(a)

    s = make_settings(tmp_path / "data", offline=False, api_key="test")
    s.ensure_dirs()
    c = LLMClient(s, connect(s.db_path), chat_fn=chat, sleep=waits.append)
    _, rec = ask(c)
    assert rec.source == "live" and rec.retries == 1 and waits == [2.0]


def test_auth_errors_are_not_retried(tmp_path):
    calls = []

    def chat(**kwargs):
        calls.append(1)
        raise RuntimeError("Error code: 401 - token expired or incorrect")

    s = make_settings(tmp_path / "data", offline=False, api_key="test")
    s.ensure_dirs()
    c = LLMClient(s, connect(s.db_path), chat_fn=chat, sleep=lambda _: None)
    with pytest.raises(ModelCallFailed):
        ask(c)
    assert len(calls) == 1


def _gateway(tmp_path, chat, **kw):
    from hlzf.extract import Gateway
    from hlzf.prov import Prov

    s = make_settings(tmp_path / "data", offline=False, api_key="test", **kw)
    s.ensure_dirs()
    conn = connect(s.db_path)
    llm = LLMClient(s, conn, chat_fn=chat, sleep=lambda _: None)
    return s, Gateway(s, conn, Prov(conn), llm)


def _real_entry(tmp_path):
    from hlzf.models import CorpusEntry

    pdf = tmp_path / "doc.pdf"
    fixtures.build_pdf(fixtures.spec_by_id("musterstadt-2026"), pdf)
    from hlzf.parse import parse_pdf

    entry = CorpusEntry(id="doc", dso="DSO", year=2026, sha256="abc", url="https://x/y.pdf")
    return entry, pdf, parse_pdf(pdf)


def test_prefetch_runs_calls_in_parallel_and_fills_the_cache(tmp_path):
    import threading
    import time as _time

    lock, active, peak = threading.Lock(), [0], [0]

    def slow_chat(**kwargs):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _time.sleep(0.15)
        with lock:
            active[0] -= 1
        return response(GOOD)

    s, gw = _gateway(tmp_path, slow_chat, parallel=4)
    entry, pdf, pages = _real_entry(tmp_path)
    gw.conn.execute("INSERT INTO documents (id, entity_id, dso, year) "
                    "VALUES ('doc', 'hlzf:doc/doc', 'DSO', 2026)")
    jobs = [(entry, "text", pages, pdf, i) for i in range(4)]
    t0 = _time.monotonic()
    assert gw.prefetch(jobs) == []
    assert peak[0] == 4 and _time.monotonic() - t0 < 0.5
    # the sequential pass now reads every answer from the cache
    runs = [gw.extract(entry, "text", pages, pdf, sample=i) for i in range(4)]
    assert {r.record.source for r in runs} == {"cache"}


def test_parallel_calls_cannot_overshoot_the_budget(tmp_path):
    calls = []

    def chat(**kwargs):
        calls.append(1)
        return response(GOOD)

    # one glm-5.3 call reserves about $0.04 (8k output tokens); $0.09 allows two at once
    s, gw = _gateway(tmp_path, chat, parallel=6, budget_usd=0.09)
    entry, pdf, pages = _real_entry(tmp_path)
    errors = gw.prefetch([(entry, "text", pages, pdf, i) for i in range(6)])
    assert len(calls) + len(errors) == 6
    assert all("BudgetExceeded" in e for e in errors)
    assert gw.llm.spent_usd() <= 0.09
