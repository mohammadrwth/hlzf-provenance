"""Evaluation against a golden set.

Golden files live in `golden/<doc_id>.yaml` and are labeled by a human from the PDF alone
(`hlzf golden-template <doc_id>` writes a blank grid, deliberately without the model's
answers, so labels are not anchored on them). Only files with `verified: true` count.
Synthetic documents have ground truth by construction (`fixtures.golden`).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from . import fixtures
from .models import Season, fmt_min
from .store import loads, row, rows
from .textnorm import parse_time

# Expected attribution for the injected faults of the synthetic corpus.
EXPECTED_VERDICTS = {
    ("musterstadt-2026", "COVERAGE_GAP"): "extract",
    ("musterstadt-2026", "CROSS_CHECK_DISAGREE"): "extract",
    ("alpenland-2026", "GROUNDING_VALUE_MISMATCH"): "extract",
    ("talwerk-2026", "CROSS_CHECK_DISAGREE"): "parse",
    ("quellbach-2026", "TIME_INVALID"): "source_document",
    ("quellbach-2026", "DAILY_CAP_EXCEEDED"): "source_document",
}

TEMPLATE_HEADER = """# Golden label for {doc_id} ({dso} {year})
# Label from the PDF only, without looking at the pipeline's output. For every grid level x
# season write the printed windows as "HH:MM-HH:MM" strings, or [] if the cell is printed
# empty. Delete cells the table does not have; add levels it has. Then set verified: true and
# your name in labeled_by.
#
# convention: interval_start | interval_end_physical | interval_end_labels |
#             interval_end_ambiguous | assumed    (definitions: src/hlzf/models.py, Convention)
# rules: kinds and value formats as in src/hlzf/prompts.py, e.g.
#   {{kind: bridge_days, value: working_day}}, {{kind: min_kw_diff, value: 100}},
#   {{kind: holiday_handling, value: {{states: [BY], mode: union, exclude: []}}}}
"""


def blank_template(conn: sqlite3.Connection, doc_id: str) -> str:
    d = row(conn, "SELECT * FROM documents WHERE id=?", (doc_id,))
    if not d:
        raise KeyError(f"unknown document {doc_id}; run the pipeline first")
    levels = sorted({ne for _, ne in loads(d["levels_listed"], []) if ne}) or [
        "NE3", "NE4", "NE5", "NE6", "NE7"]
    body = {
        "document": doc_id, "labeled_by": "", "verified": False, "convention": "",
        "cells": [{"grid_level": lv, "season": s.value, "windows": []}
                  for lv in levels for s in Season],
        "rules": [{"kind": "workdays_only", "value": True}],
    }
    return (TEMPLATE_HEADER.format(doc_id=doc_id, dso=d["dso"], year=d["year"])
            + yaml.safe_dump(body, sort_keys=False, allow_unicode=True))


def load_goldens(root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    gdir = root / "golden"
    if gdir.is_dir():
        for p in sorted(gdir.glob("*.yaml")):
            g = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if g.get("verified") and g.get("document"):
                out[g["document"]] = g
    return out


def golden_status(conn: sqlite3.Connection, root: Path) -> list[dict[str, str]]:
    """Every golden file and whether it counts, so an unverified label never disappears
    silently from the report."""
    out = []
    gdir = root / "golden"
    for p in sorted(gdir.glob("*.yaml")) if gdir.is_dir() else []:
        try:
            g = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as err:
            out.append({"file": p.name, "state": "unreadable",
                        "hint": f"YAML error: {str(err).splitlines()[0]}"})
            continue
        doc = g.get("document", "")
        wins, _ = golden_windows(g) if g.get("cells") else (set(), set())
        processed = bool(doc) and row(conn, "SELECT 1 FROM documents WHERE id=? AND "
                                            "processed_at IS NOT NULL", (doc,)) is not None
        if not g.get("verified"):
            state, hint = "not counted", "set `verified: true` once the labels are checked"
        elif not wins:
            state, hint = "not counted", "no windows labeled"
        elif not processed:
            state, hint = "not counted", f"`{doc}` has not been processed; run `hlzf run`"
        else:
            state, hint = "counted", f"{len(wins)} windows, labeled by {g.get('labeled_by') or '?'}"
        out.append({"file": p.name, "state": state, "hint": hint})
    return out


def golden_windows(g: dict[str, Any]) -> tuple[set[tuple], set[tuple]]:
    wins, cells = set(), set()
    for c in g.get("cells", []):
        cells.add((c["grid_level"], c["season"]))
        for w in c.get("windows") or []:
            a, b = str(w).replace("–", "-").split("-")
            wins.add((c["grid_level"], c["season"], parse_time(a), parse_time(b)))
    return wins, cells


def _rule_set(items) -> set[tuple[str, str]]:
    out = set()
    for r in items:
        v = r["value"]
        if isinstance(v, dict):
            v = {k: (sorted(x) if isinstance(x, list) else x) for k, x in v.items()}
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
        out.add((r["kind"], json.dumps(v, sort_keys=True, ensure_ascii=False)))
    return out


def evaluate_document(conn: sqlite3.Connection, doc_id: str, g: dict[str, Any]
                      ) -> dict[str, Any]:
    gw, gcells = golden_windows(g)

    def key(w: dict[str, Any]) -> tuple:
        return (w["grid_level"], w["season"], w["start_min"], w["end_min"])
    # What the pipeline delivers before any person looks: the extraction, after the consensus
    # correction (origin 'auto'). The raw extraction is reported alongside.
    pipeline_rows = rows(conn, "SELECT * FROM windows WHERE document_id=? AND origin IN "
                               "('extract','auto') ORDER BY revision", (doc_id,))
    raw = {key(w) for w in pipeline_rows if w["origin"] == "extract"}
    latest: dict[str, dict[str, Any]] = {}
    for w in pipeline_rows:
        latest[w["entity_id"].split("@")[0]] = w
    delivered = [w for w in latest.values() if w["status"] != "rejected"]
    pred = {key(w) for w in delivered}
    flagged = {key(w) for w in delivered if w["status"] == "needs-review"}
    wrong = pred - gw
    missed = gw - pred
    # a missing window is caught when its cell carries a warning or error that the consensus
    # correction did not settle
    targets = {r["target"] for r in rows(
        conn, "SELECT target FROM issues WHERE document_id=? AND target IS NOT NULL AND "
              "severity IN ('error','warning') AND resolved != 2", (doc_id,))}

    def cell_flagged(level: str, season: str) -> bool:
        return f"cell:{doc_id}/{level}/{season}" in targets or any(
            t.startswith(f"hlzf:window/{doc_id}/{level}/{season}/") for t in targets)
    def pr(found: set) -> tuple[float, float]:
        tp = len(gw & found)
        return (tp / len(found) if found else 0.0,
                tp / len(gw) if gw else (1.0 if not found else 0.0))
    precision, recall = pr(pred)
    raw_precision, raw_recall = pr(raw)
    # exact per-cell agreement over the golden cells
    def cell(s, lv, se):
        return frozenset((a, b) for (l2, s2, a, b) in s if (l2, s2) == (lv, se))
    cell_ok = sum(1 for lv, se in gcells if cell(gw, lv, se) == cell(pred, lv, se))
    d = row(conn, "SELECT * FROM documents WHERE id=?", (doc_id,))
    pred_rules = _rule_set([{"kind": r["kind"], "value": loads(r["value"])} for r in rows(
        conn, "SELECT * FROM rules WHERE document_id=?", (doc_id,))])
    gold_rules = _rule_set(g.get("rules", []))
    rules_recall = (len(gold_rules & pred_rules) / len(gold_rules)) if gold_rules else None
    # One row per distinct request: reruns and cache replays log the same call again.
    cost = row(conn, "SELECT COALESCE(SUM(c),0) c, COALESCE(SUM(l),0) l, COUNT(*) n, "
                     "GROUP_CONCAT(DISTINCT s) s FROM (SELECT cache_key, MAX(cost_usd) c, "
                     "MAX(latency_ms) l, MIN(source) s FROM llm_calls WHERE document_id=? "
                     "GROUP BY cache_key)", (doc_id,))
    return {
        "document": doc_id, "labeled_by": g.get("labeled_by", ""),
        "windows_gold": len(gw), "windows_pred": len(pred), "precision": precision,
        "recall": recall, "cells_exact": f"{cell_ok}/{len(gcells)}",
        "convention_ok": (d["convention"] == g.get("convention")) if g.get("convention")
        else None,
        "rules_recall": rules_recall,
        "review_share": len(flagged) / len(pred) if pred else 0.0,
        "raw_precision": raw_precision, "raw_recall": raw_recall,
        "auto_corrected": sum(1 for w in latest.values() if w["origin"] == "auto"),
        # The safety metric: of the windows that are wrong, how many were sent to review
        # (instead of silently reaching an optimizer)?
        "wrong": len(wrong), "wrong_flagged": len(wrong & flagged),
        "missed": len(missed),
        "missed_flagged": sum(1 for lv, se, _, _ in missed if cell_flagged(lv, se)),
        "llm_calls": cost["n"], "cost_usd": cost["c"], "latency_s": cost["l"] / 1000,
        "response_source": cost["s"] or "",
    }


def attribution_check(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    out = []
    for (doc_id, code), expected in EXPECTED_VERDICTS.items():
        issues = rows(conn, "SELECT * FROM issues WHERE document_id=? AND code=?",
                      (doc_id, code))
        got = [(i["suspected_stage"], loads(i["attribution"], {}) or {}) for i in issues]
        stage = got[0][0] if got else None
        att = got[0][1] if got else {}
        out.append({"document": doc_id, "issue": code, "expected": expected,
                    "got": att.get("stage", stage), "method": att.get("method", "heuristic"),
                    "confidence": att.get("confidence", ""),
                    "p_hat": att.get("p_hat"), "ok": att.get("stage", stage) == expected,
                    "auto": (att.get("auto_correction") or {}).get("applied")})
    return out


def run_eval(conn: sqlite3.Connection, root: Path) -> dict[str, Any]:
    real = []
    for doc_id, g in load_goldens(root).items():
        if row(conn, "SELECT 1 FROM documents WHERE id=? AND processed_at IS NOT NULL",
               (doc_id,)):
            real.append(evaluate_document(conn, doc_id, g))
    synthetic = []
    for spec in fixtures.specs():
        if row(conn, "SELECT 1 FROM documents WHERE id=? AND processed_at IS NOT NULL",
               (spec.id,)):
            synthetic.append(evaluate_document(conn, spec.id, fixtures.golden(spec.id)))
    return {"real": real, "synthetic": synthetic, "attribution": attribution_check(conn),
            "golden": golden_status(conn, root)}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f} %"


def _pr(r: dict[str, Any], prefix: str = "") -> str:
    return f"{_pct(r[prefix + 'precision'])} / {_pct(r[prefix + 'recall'])}"


def _caught(r: dict[str, Any]) -> str:
    parts = []
    if r["wrong"]:
        parts.append(f"{r['wrong_flagged']} of {r['wrong']} wrong")
    if r.get("missed"):
        parts.append(f"{r['missed_flagged']} of {r['missed']} missing")
    return ", ".join(parts) or "none wrong or missing"


def markdown_report(result: dict[str, Any], summary: list[dict[str, Any]]) -> str:
    lines = []
    real_docs = [d for d in summary if not d["synthetic"]]
    lines.append("**Real DSO publications** (live GLM run; human-labeled where a golden "
                 "file exists)\n")
    if real_docs:
        lines.append("| Document | DSO | Year | Windows | Auto-corrected | Sent to review "
                     "| Errors / warnings | Convention | Status |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for d in real_docs:
            lines.append(f"| `{d['id']}` | {d['dso']} | {d['year']} | {d['n_windows']} | "
                         f"{d['n_auto']} | {d['n_review']} | {d['errors']} / {d['warnings']} | "
                         f"{d['convention'] or '–'} | {d['status']} |")
        lines.append("\n_Auto-corrected: values set by the consensus correction (GLM-OCR and the "
                     "vision model agree against the text reading, every value printed in that "
                     "cell). Status `caveats`: every value checked out, but the document itself "
                     "leaves something open (no quarter-hour convention, a holiday period "
                     "without dates); a person acknowledges that once._")
    else:
        lines.append("_Not run yet in this checkout: `hlzf fetch` + `hlzf run --live` "
                     "need network access and a Z.ai key; then `hlzf eval --write-readme`._")
    pending = [g for g in result.get("golden", []) if g["state"] != "counted"]
    if pending:
        lines.append("\n_Golden files not counted yet: "
                     + "; ".join(f"`{g['file']}` ({g['hint']})" for g in pending) + "._")
    if result["real"]:
        lines.append("\nAccuracy against human-verified golden files:\n")
        lines.append("| Document | Labeled by | Raw extraction P / R | Delivered P / R "
                     "| Cells exact | Auto-corrected | Wrong / missing flagged | Convention "
                     "| Rules recall | Review share | Cost | Latency |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in result["real"]:
            conv = "–" if r["convention_ok"] is None else ("✓" if r["convention_ok"] else "✗")
            lines.append(f"| `{r['document']}` | {r['labeled_by']} | {_pr(r, 'raw_')} | "
                         f"{_pr(r)} | {r['cells_exact']} | {r['auto_corrected']} | "
                         f"{_caught(r)} | {conv} | {_pct(r['rules_recall'])} | "
                         f"{_pct(r['review_share'])} | ${r['cost_usd']:.3f} | "
                         f"{r['latency_s']:.0f} s |")
    lines.append("\n**Synthetic test corpus** (fictional DSOs, ground truth by construction, "
                 "scripted responses with injected faults; tests the checks, not the model)\n")
    lines.append("| Document | Raw extraction P / R | Delivered P / R | Cells exact "
                 "| Auto-corrected | Wrong / missing flagged | Convention | Rules recall "
                 "| Review share |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in result["synthetic"]:
        conv = "–" if r["convention_ok"] is None else ("✓" if r["convention_ok"] else "✗")
        lines.append(f"| `{r['document']}` | {_pr(r, 'raw_')} | {_pr(r)} | {r['cells_exact']} | "
                     f"{r['auto_corrected']} | {_caught(r)} | {conv} | "
                     f"{_pct(r['rules_recall'])} | {_pct(r['review_share'])} |")
    lines.append("\n**Fault attribution on injected faults**\n")
    lines.append("| Document | Issue | Injected at | Verdict | Method | p̂ (persisted) "
                 "| Confidence | Auto-corrected |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a in result["attribution"]:
        mark = "✓" if a["ok"] else "✗"
        p = "–" if a["p_hat"] is None else f"{a['p_hat']:.2f}"
        lines.append(f"| `{a['document']}` | {a['issue']} | {a['expected']} | {a['got']} "
                     f"{mark} | {a['method']} | {p} | {a['confidence']} | "
                     f"{'yes' if a.get('auto') else '–'} |")
    return "\n".join(lines)


def write_readme_block(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    if start not in text or end not in text:
        raise ValueError("README has no results markers")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{block}\n{end}{tail}", encoding="utf-8")


__all__ = ["blank_template", "run_eval", "markdown_report", "write_readme_block", "fmt_min"]
