import json

import httpx
import yaml

from conftest import make_settings
from hlzf.corpus import fetch, import_file, load_corpus, load_lock

PDF_V1 = b"%PDF-1.7\n% version one\n"
PDF_V2 = b"%PDF-1.7\n% corrected version\n"


def setup_root(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "corpus.yaml").write_text(yaml.safe_dump({"documents": [
        {"id": "dso-a-2026", "dso": "DSO A", "year": 2026, "url": "https://a.example/h.pdf",
         "states": ["BY"]},
        {"id": "dso-b-2026", "dso": "DSO B", "year": 2026, "url": "https://b.example/h.pdf"},
    ]}))
    return make_settings(root / "data", root=root)


def transport(content_for):
    def handler(request: httpx.Request):
        body = content_for(request.url.host)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, content=body, headers={"content-type": "application/pdf",
                                                          "etag": '"abc"'})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_new_unchanged_changed(tmp_path):
    s = setup_root(tmp_path)
    served = {"a.example": PDF_V1, "b.example": b"<html>not a pdf</html>"}
    res = {r.id: r for r in fetch(s, client=transport(served.get))}
    assert res["dso-a-2026"].status == "new"
    assert res["dso-b-2026"].status == "failed" and "not a PDF" in res["dso-b-2026"].detail
    lock = load_lock(s)
    assert lock["dso-a-2026"]["headers"]["etag"] == '"abc"'

    res = {r.id: r for r in fetch(s, only=["dso-a-2026"], client=transport(served.get))}
    assert res["dso-a-2026"].status == "unchanged"

    # The DSO silently replaces the file: keep the old bytes, store the new ones beside them.
    served["a.example"] = PDF_V2
    res = {r.id: r for r in fetch(s, only=["dso-a-2026"], client=transport(served.get))}
    assert res["dso-a-2026"].status == "changed"
    assert (s.raw_dir / "dso-a-2026.pdf").read_bytes() == PDF_V1
    assert any(p.name.startswith("dso-a-2026.") and p.name != "dso-a-2026.pdf"
               for p in s.raw_dir.iterdir())
    res = {r.id: r for r in fetch(s, only=["dso-a-2026"], accept_changes=True,
                                  client=transport(served.get))}
    assert res["dso-a-2026"].status == "changed"
    assert (s.raw_dir / "dso-a-2026.pdf").read_bytes() == PDF_V2
    assert "previous_sha256" in load_lock(s)["dso-a-2026"]


def test_http_errors_are_reported(tmp_path):
    s = setup_root(tmp_path)
    res = {r.id: r for r in fetch(s, client=transport(lambda host: None))}
    assert res["dso-a-2026"].status == "failed"


def test_import_and_pinning(tmp_path):
    s = setup_root(tmp_path)
    f = tmp_path / "manual.pdf"
    f.write_bytes(PDF_V1)
    assert import_file(s, "dso-a-2026", f).status == "new"
    assert import_file(s, "unknown", f).status == "failed"
    entries = {e.id: e for e in load_corpus(s, include_synthetic=False)}
    assert entries["dso-a-2026"].sha256 == json.loads(
        (s.root / "corpus.lock.json").read_text())["dso-a-2026"]["sha256"]


def test_repository_corpus_registry_is_well_formed():
    from hlzf.config import PROJECT_ROOT, load_settings

    entries = load_corpus(load_settings(root=PROJECT_ROOT), include_synthetic=False)
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids)) >= 13
    assert all(e.url and e.url.startswith("https://") and e.url.lower().endswith(".pdf")
               for e in entries)
    bavarian = [e for e in entries if "BY" in e.states]
    assert len(bavarian) >= 8
