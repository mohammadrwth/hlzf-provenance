import json

from prov.model import ProvDocument

from hlzf.prov import Prov
from hlzf.review import review_window
from hlzf.store import row


def window(conn, doc="alpenland-2026", level="NE5", season="winter", start=465):
    return row(conn, "SELECT * FROM windows WHERE document_id=? AND grid_level=? AND season=? "
                     "AND start_min=? AND active=1", (doc, level, season, start))


def test_prov_json_is_valid_w3c(conn):
    w = window(conn)
    doc = Prov(conn).to_prov_json(w["entity_id"])
    parsed = ProvDocument.deserialize(content=json.dumps(doc), format="json")
    records = {type(r).__name__ for r in parsed.get_records()}
    assert {"ProvEntity", "ProvActivity", "ProvAgent", "ProvUsage", "ProvGeneration",
            "ProvAssociation", "ProvDerivation"} <= records
    assert "hlzf:doc/alpenland-2026" in doc["entity"]
    agents = {a["prov:label"] for a in doc["agent"].values()}
    assert "fixture-replay" in agents and any(a.startswith("PyMuPDF") for a in agents)


def test_review_adds_revision_and_person(conn):
    w = window(conn)
    res = review_window(conn, w["wid"], "approve", "Mo")
    doc = Prov(conn).to_prov_json(res["entity"])
    revisions = [d for d in doc["wasDerivedFrom"].values()
                 if d.get("prov:type", {}).get("$") == "prov:Revision"]
    assert revisions and revisions[0]["prov:usedEntity"] == w["entity_id"]
    assert any(a["prov:type"]["$"] == "prov:Person" for a in doc["agent"].values())
    ProvDocument.deserialize(content=json.dumps(doc), format="json")


def test_attribution_is_recorded_as_intervention(conn):
    issue = row(conn, "SELECT * FROM issues WHERE document_id='talwerk-2026' AND "
                      "code='CROSS_CHECK_DISAGREE'")
    acts = [r[0] for r in conn.execute(
        "SELECT a.type FROM used u JOIN activity a ON a.id=u.activity_id WHERE u.entity_id=?",
        (issue["entity_id"],))]
    assert "hlzf:Intervention" in acts
    roles = {r[0] for r in conn.execute(
        "SELECT u.role FROM used u JOIN activity a ON a.id=u.activity_id "
        "WHERE a.type='hlzf:Intervention' AND a.id LIKE 'hlzf:act/attribute/talwerk-2026/%'")}
    assert {"text-sample-1", "text-sample-5", "ocr-sample-0"} <= roles
