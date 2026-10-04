import csv
import io

import pytest
from fastapi.testclient import TestClient

from hlzf.store import row
from hlzf.web.app import create_app


@pytest.fixture()
def client(rt):
    return TestClient(create_app(rt.settings))


def test_pages_render(client, conn):
    assert "Synthetic test corpus" in client.get("/").text
    w = row(conn, "SELECT wid FROM windows WHERE document_id='talwerk-2026' AND start_min=465 "
                  "AND active=1")
    page = client.get(f"/doc/talwerk-2026?w={w['wid']}").text
    assert "Fault located at: Parse" in page
    assert "Auto-corrected by consensus" in page and "settled by consensus" in page
    assert "Provenance, from this value back to the source" in page
    assert client.get("/doc/talwerk-2026/page/1.png").headers["content-type"] == "image/png"
    assert client.get("/doc/nope").status_code == 404
    assert "1 of 16 cells changed" in client.get(
        "/diff?a=beispielstadt-2026&b=beispielstadt-2026-korr").text


def test_query_page_and_fragment(client):
    page = client.get("/query", params={"dso": "Stadtwerke Beispielstadt GmbH", "level": "NE5",
                                        "ts": "2026-01-14T08:15+01:00"}).text
    assert "Uncertain" in page
    frag = client.get("/query/result", params={"dso": "Alpenland Verteilnetz AG",
                                               "level": "NE5",
                                               "ts": "2026-01-15T08:30+01:00"}).text
    assert "In HLZF" in frag and "<html" not in frag


def test_check_api(client):
    r = client.get("/api/hlzf/check", params={"dso": "alpenland", "level": "NE5",
                                              "ts": "2026-01-15T08:30+01:00",
                                              "explain": True}).json()
    assert r["in_hlzf"] is True and r["provenance"][-1]["entity"] == "hlzf:doc/alpenland-2026"
    bad = client.get("/api/hlzf/check", params={"dso": "x", "level": "NE5",
                                                "ts": "2026-01-15T08:30+01:00"})
    assert bad.status_code == 400


def test_prov_and_mask_downloads(client, conn):
    w = row(conn, "SELECT wid FROM windows WHERE document_id='alpenland-2026' LIMIT 1")
    doc = client.get("/api/prov", params={"window_id": w["wid"]}).json()
    assert "wasGeneratedBy" in doc and "prefix" in doc
    mask = client.get("/api/mask", params={"dso": "alpenland", "level": "NE5",
                                           "year": 2026})
    assert sum(1 for _ in csv.DictReader(io.StringIO(mask.text))) == 35040


def test_review_through_the_ui(client, conn):
    w = row(conn, "SELECT wid FROM windows WHERE document_id='alpenland-2025' AND start_min=480 "
                  "AND grid_level='NE5' AND season='winter' AND active=1")
    r = client.post(f"/window/{w['wid']}/review", data={
        "action": "edit", "start": "08:15", "end": "09:00", "reviewer": "Mo",
        "comment": "page shows 08:15"}, follow_redirects=False)
    assert r.status_code == 303 and "/doc/alpenland-2025?w=" in r.headers["location"]
    new = row(conn, "SELECT * FROM windows WHERE document_id='alpenland-2025' AND active=1 AND "
                    "start_min=495 AND grid_level='NE5' AND season='winter'")
    assert new and new["status"] == "edited"
    r = client.post(f"/window/{w['wid']}/review", data={"action": "approve"},
                    follow_redirects=False)
    assert "error=" in r.headers["location"]  # superseded revision


def test_set_cell_through_the_ui(client, conn):
    w = row(conn, "SELECT wid FROM windows WHERE document_id='quellbach-2026' AND "
                  "grid_level='NE7' AND season='winter' AND start_min=360")
    page = client.get(f"/doc/quellbach-2026?w={w['wid']}").text
    assert "Approve" in page
    r = client.post("/doc/quellbach-2026/cell", data={
        "level": "NE7", "season": "winter", "windows": "06:00-12:00", "reviewer": "Mo",
        "comment": "13:00-17:30 is not on the page"}, follow_redirects=False)
    assert r.status_code == 303 and "error" not in r.headers["location"]
    r = client.post("/doc/quellbach-2026/cell", data={
        "level": "NE7", "season": "winter", "windows": "06:00-12:00"}, follow_redirects=False)
    assert "error=" in r.headers["location"]  # a comment is required
