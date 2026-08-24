#!/usr/bin/env python3
"""Offline audit/replay of self-improvement artifacts from a completed run."""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.resume import atomic_write_json, stable_hash
from binderloop.skills.self_improvement import validate_skill_document


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit self-improvement skill evolution")
    parser.add_argument("--out", required=True, help="Completed harness output directory")
    parser.add_argument("--report", help="Optional report path; defaults inside --out")
    args = parser.parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    if not out_dir.exists():
        raise SystemExit("output directory not found: %s" % out_dir)
    report = build_report(out_dir)
    report_path = Path(args.report).expanduser() if args.report else out_dir / "self_improvement_replay_report.json"
    atomic_write_json(report_path, report)
    print("Self-improvement replay report: %s" % report_path)
    return 0 if not report["validation_errors"] else 1


def build_report(out_dir: Path) -> dict:
    rounds = sorted(
        [path for path in out_dir.glob("round_*") if path.is_dir()],
        key=lambda path: path.name,
    )
    rows = []
    validation_errors = []
    prior_exposure = {}
    for round_dir in rounds:
        round_id = _round_id(round_dir)
        snapshot_path = round_dir / "self_improvement_skill_snapshot.yaml"
        update_path = round_dir / "self_improvement_update.json"
        outcome_path = round_dir / "rollback_decision.json"
        exposure_path = round_dir / "next_strategy_exposure.json"
        merge_path = round_dir / "next_round_config_merge_report.json"
        jobs_path = round_dir / "next_jobs.json"
        row = {
            "round_id": round_id,
            "prior_exposure_id": prior_exposure.get("exposure_id"),
            "prior_cited_rule_ids": list(prior_exposure.get("cited_rule_ids") or []),
            "operation_count": 0,
            "semantic_relation_count": 0,
            "active_rule_count": 0,
            "open_conflict_count": 0,
            "reward": None,
            "rollback_action": None,
        }
        if snapshot_path.exists():
            try:
                document = validate_skill_document(
                    yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
                )
                row["active_rule_count"] = sum(
                    1
                    for section in document["modules"].values()
                    for rule in section["rules"].values()
                    if rule.get("status") in {"seed_active", "active"}
                )
                row["open_conflict_count"] = sum(
                    1
                    for item in document.get("conflict_sets", {}).values()
                    if item.get("status", "open") == "open"
                )
            except Exception as exc:
                validation_errors.append({
                    "round_id": round_id,
                    "artifact": str(snapshot_path),
                    "error": str(exc),
                })
        if update_path.exists():
            update = _read_json(update_path)
            row["operation_count"] = len(update.get("operations") or [])
            row["semantic_relation_count"] = len(update.get("semantic_relations") or [])
            row["llm_used"] = bool(update.get("llm_used"))
        if outcome_path.exists():
            outcome_payload = _read_json(outcome_path)
            outcome = dict(outcome_payload.get("outcome") or {})
            decision = dict(outcome_payload.get("decision") or {})
            row["reward"] = outcome.get("reward")
            row["rollback_action"] = decision.get("action")
        if exposure_path.exists():
            prior_exposure = _read_json(exposure_path)
            row["next_exposure_id"] = prior_exposure.get("exposure_id")
            row["next_cited_rule_ids"] = list(prior_exposure.get("cited_rule_ids") or [])
        elif merge_path.exists() or jobs_path.exists():
            merge = _read_json(merge_path)
            try:
                jobs = json.loads(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else []
            except Exception:
                jobs = []
            prior_exposure = {
                "schema_version": "legacy-reconstructed-1.0",
                "origin_round_id": round_id,
                "execution_round_id": round_id + 1,
                "applied_update": dict(merge.get("applied_update") or {}),
                "applied_sources": dict(merge.get("applied_sources") or {}),
                "next_arms": sorted({
                    str((job.get("params") or {}).get("exploration_arm"))
                    for job in jobs
                    if isinstance(job, dict) and (job.get("params") or {}).get("exploration_arm")
                }),
                "cited_rule_ids": [],
            }
            prior_exposure["exposure_id"] = stable_hash(prior_exposure)[:24]
            row["next_exposure_id"] = prior_exposure["exposure_id"]
            row["legacy_exposure_reconstructed"] = True
        rows.append(row)
    cited_outcomes = [
        {
            "round_id": row["round_id"],
            "cited_rule_ids": row.get("prior_cited_rule_ids") or [],
            "reward": row.get("reward"),
            "rollback_action": row.get("rollback_action"),
        }
        for row in rows
        if row.get("prior_cited_rule_ids")
    ]
    active_rule_transitions = [
        {
            "round_id": rows[index]["round_id"],
            "before": rows[index - 1]["active_rule_count"],
            "after": rows[index]["active_rule_count"],
            "delta": rows[index]["active_rule_count"] - rows[index - 1]["active_rule_count"],
        }
        for index in range(1, len(rows))
        if rows[index]["active_rule_count"] != rows[index - 1]["active_rule_count"]
    ]
    return {
        "schema_version": "1.0",
        "run_dir": str(out_dir),
        "round_count": len(rows),
        "rounds": rows,
        "validation_errors": validation_errors,
        "summary": {
            "llm_update_rounds": sum(1 for row in rows if row.get("llm_used")),
            "total_operations": sum(int(row["operation_count"]) for row in rows),
            "total_semantic_relations": sum(int(row["semantic_relation_count"]) for row in rows),
            "rounds_with_rule_citations": sum(1 for row in rows if row.get("next_cited_rule_ids")),
            "rollback_rounds": sum(1 for row in rows if row.get("rollback_action") not in {None, "advance"}),
            "cited_outcome_count": len(cited_outcomes),
            "active_rule_transition_count": len(active_rule_transitions),
        },
        "cited_outcomes": cited_outcomes,
        "active_rule_transitions": active_rule_transitions,
        "report_digest": stable_hash(rows),
    }


def _round_id(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return -1


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else {}
    except Exception:
        return {}


if __name__ == "__main__":
    raise SystemExit(main())

