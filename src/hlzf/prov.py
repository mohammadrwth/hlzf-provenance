"""W3C PROV recorder and PROV-JSON export.

The model is PROV-DM core: Entity, Activity, Agent and the relations used,
wasGeneratedBy, wasDerivedFrom, wasAssociatedWith and wasRevisionOf. Records are
append-only (`INSERT OR IGNORE`), so re-running an idempotent stage never rewrites history.

Htrace (the author's bachelor thesis) records an equivalent graph with its own typed node and
edge vocabulary; `docs/htrace-prov-mapping.md` maps those types onto the PROV terms used here.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

from .store import dumps, loads, row, rows

NAMESPACE = "https://github.com/mohammadrwth/hlzf-provenance/ns#"

PERSON = "prov:Person"
SOFTWARE = "prov:SoftwareAgent"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Prov:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- nodes -----------------------------------------------------------------------
    def entity(self, id: str, type: str, label: str | None = None, **attrs: Any) -> str:
        self.conn.execute(
            "INSERT OR IGNORE INTO entity (id, type, label, attrs) VALUES (?, ?, ?, ?)",
            (id, type, label, dumps(attrs)),
        )
        return id

    def activity(self, id: str, type: str, label: str | None = None,
                 started_at: str | None = None, ended_at: str | None = None,
                 **attrs: Any) -> str:
        self.conn.execute(
            "INSERT OR IGNORE INTO activity (id, type, label, started_at, ended_at, attrs) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (id, type, label, started_at or now_iso(), ended_at or now_iso(), dumps(attrs)),
        )
        return id

    def agent(self, id: str, type: str, label: str | None = None, **attrs: Any) -> str:
        self.conn.execute(
            "INSERT OR IGNORE INTO agent (id, type, label, attrs) VALUES (?, ?, ?, ?)",
            (id, type, label, dumps(attrs)),
        )
        return id

    # --- relations -------------------------------------------------------------------
    def used(self, activity: str, entity: str, role: str | None = None) -> None:
        self.conn.execute("INSERT OR IGNORE INTO used VALUES (?, ?, ?)", (activity, entity, role))

    def generated(self, entity: str, activity: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO was_generated_by VALUES (?, ?)", (entity, activity))

    def derived(self, generated: str, used: str, activity: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO was_derived_from VALUES (?, ?, ?)",
            (generated, used, activity))

    def associated(self, activity: str, agent: str, role: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO was_associated_with VALUES (?, ?, ?)",
            (activity, agent, role))

    def revision(self, new: str, old: str, activity: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO was_revision_of VALUES (?, ?, ?)", (new, old, activity))
        self.derived(new, old, activity)

    # --- queries ---------------------------------------------------------------------
    def ancestors(self, entity_id: str, max_depth: int = 50) -> dict[str, set[str]]:
        """Collect the provenance closure of an entity: all upstream entities, the
        activities that generated them and the agents associated with those activities."""
        entities: set[str] = set()
        activities: set[str] = set()
        agents: set[str] = set()
        frontier = [entity_id]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            nxt: list[str] = []
            for ent in frontier:
                if ent in entities:
                    continue
                entities.add(ent)
                for r in rows(self.conn,
                              "SELECT activity_id FROM was_generated_by WHERE entity_id=?",
                              (ent,)):
                    act = r["activity_id"]
                    if act in activities:
                        continue
                    activities.add(act)
                    for a in rows(self.conn, "SELECT agent_id FROM was_associated_with "
                                             "WHERE activity_id=?", (act,)):
                        agents.add(a["agent_id"])
                    for u in rows(self.conn,
                                  "SELECT entity_id FROM used WHERE activity_id=?", (act,)):
                        nxt.append(u["entity_id"])
                for d in rows(self.conn,
                              "SELECT used_id FROM was_derived_from WHERE generated_id=?",
                              (ent,)):
                    nxt.append(d["used_id"])
            frontier = nxt
        return {"entities": entities, "activities": activities, "agents": agents}

    def chain(self, entity_id: str) -> list[dict[str, Any]]:
        """Linear, human-readable chain from the value back to its source document, used by
        `--explain` and the UI provenance panel. Follows the first generating activity."""
        steps: list[dict[str, Any]] = []
        seen: set[str] = set()
        current: str | None = entity_id
        while current and current not in seen:
            seen.add(current)
            ent = row(self.conn, "SELECT * FROM entity WHERE id=?", (current,))
            gen = row(self.conn,
                      "SELECT a.* FROM was_generated_by g JOIN activity a ON a.id=g.activity_id "
                      "WHERE g.entity_id=? ORDER BY a.started_at DESC LIMIT 1", (current,))
            step: dict[str, Any] = {
                "entity": current,
                "entity_type": ent["type"] if ent else None,
                "label": ent["label"] if ent else None,
                "entity_attrs": loads(ent["attrs"], {}) if ent else {},
            }
            nxt: str | None = None
            if gen:
                step["activity"] = gen["id"]
                step["activity_type"] = gen["type"]
                step["activity_attrs"] = loads(gen["attrs"], {})
                step["started_at"] = gen["started_at"]
                step["agents"] = [
                    {"id": a["id"], "type": a["type"], "label": a["label"],
                     **loads(a["attrs"], {})}
                    for a in rows(self.conn,
                                  "SELECT ag.* FROM was_associated_with w JOIN agent ag "
                                  "ON ag.id=w.agent_id WHERE w.activity_id=? "
                                  "ORDER BY ag.type, ag.id", (gen["id"],))  # persons first
                ]
                used_rows = rows(self.conn,
                                 "SELECT u.entity_id, u.role FROM used u WHERE u.activity_id=? "
                                 "ORDER BY u.role", (gen["id"],))
                step["used"] = [u["entity_id"] for u in used_rows]
                # Prefer a revision predecessor, then what the value was derived from, then
                # the first input, so the chain walks value -> extraction -> pages -> document
                # (and a consensus-corrected value -> the OCR / vision reading it came from).
                rev = row(self.conn, "SELECT old_id FROM was_revision_of WHERE new_id=?",
                          (current,))
                der = row(self.conn, "SELECT used_id FROM was_derived_from WHERE "
                                     "generated_id=? AND activity_id=? ORDER BY used_id",
                          (current, gen["id"]))
                if rev:
                    nxt = rev["old_id"]
                elif der:
                    nxt = der["used_id"]
                elif used_rows:
                    nxt = used_rows[0]["entity_id"]
            else:
                rev = row(self.conn, "SELECT old_id FROM was_revision_of WHERE new_id=?",
                          (current,))
                nxt = rev["old_id"] if rev else None
            steps.append(step)
            current = nxt
        return steps

    def to_prov_json(self, entity_id: str) -> dict[str, Any]:
        """Serialize the provenance closure of `entity_id` as W3C PROV-JSON.

        wasRevisionOf is expressed the PROV-JSON way: a wasDerivedFrom record with
        `prov:type = prov:Revision`.
        """
        closure = self.ancestors(entity_id)
        ents, acts, agts = closure["entities"], closure["activities"], closure["agents"]
        doc: dict[str, Any] = {
            "prefix": {"hlzf": NAMESPACE, "xsd": "http://www.w3.org/2001/XMLSchema#"},
            "entity": {}, "activity": {}, "agent": {},
            "used": {}, "wasGeneratedBy": {}, "wasDerivedFrom": {}, "wasAssociatedWith": {},
        }

        def attrs(kind_type: str, label: str | None, raw: str) -> dict[str, Any]:
            out: dict[str, Any] = {"prov:type": {"$": kind_type, "type": "prov:QUALIFIED_NAME"}}
            if label:
                out["prov:label"] = label
            for k, v in loads(raw, {}).items():
                key = k if ":" in k else f"hlzf:{k}"
                out[key] = v if isinstance(v, (str, int, float, bool)) else dumps(v)
            return out

        for e in ents:
            r = row(self.conn, "SELECT * FROM entity WHERE id=?", (e,))
            if r:
                doc["entity"][e] = attrs(r["type"], r["label"], r["attrs"])
        for a in acts:
            r = row(self.conn, "SELECT * FROM activity WHERE id=?", (a,))
            if r:
                rec = attrs(r["type"], r["label"], r["attrs"])
                if r["started_at"]:
                    rec["prov:startTime"] = r["started_at"]
                if r["ended_at"]:
                    rec["prov:endTime"] = r["ended_at"]
                doc["activity"][a] = rec
        for g in agts:
            r = row(self.conn, "SELECT * FROM agent WHERE id=?", (g,))
            if r:
                doc["agent"][g] = attrs(r["type"], r["label"], r["attrs"])

        n = 0

        def bid() -> str:
            nonlocal n
            n += 1
            return f"_:r{n}"

        for a in sorted(acts):
            for u in rows(self.conn, "SELECT * FROM used WHERE activity_id=?", (a,)):
                if u["entity_id"] in ents:
                    rec = {"prov:activity": a, "prov:entity": u["entity_id"]}
                    if u["role"]:
                        rec["prov:role"] = u["role"]
                    doc["used"][bid()] = rec
            for w in rows(self.conn, "SELECT * FROM was_associated_with WHERE activity_id=?",
                          (a,)):
                rec = {"prov:activity": a, "prov:agent": w["agent_id"]}
                if w["role"]:
                    rec["prov:role"] = w["role"]
                doc["wasAssociatedWith"][bid()] = rec
        revisions = {(r["new_id"], r["old_id"]) for r in rows(
            self.conn, "SELECT new_id, old_id FROM was_revision_of")}
        for e in sorted(ents):
            for g in rows(self.conn, "SELECT * FROM was_generated_by WHERE entity_id=?", (e,)):
                if g["activity_id"] in acts:
                    doc["wasGeneratedBy"][bid()] = {
                        "prov:entity": e, "prov:activity": g["activity_id"]}
            for d in rows(self.conn, "SELECT * FROM was_derived_from WHERE generated_id=?",
                          (e,)):
                if d["used_id"] not in ents:
                    continue
                rec = {"prov:generatedEntity": e, "prov:usedEntity": d["used_id"]}
                if d["activity_id"] and d["activity_id"] in acts:
                    rec["prov:activity"] = d["activity_id"]
                if (e, d["used_id"]) in revisions:
                    rec["prov:type"] = {"$": "prov:Revision", "type": "prov:QUALIFIED_NAME"}
                doc["wasDerivedFrom"][bid()] = rec
        return {k: v for k, v in doc.items() if v or k == "prefix"}
