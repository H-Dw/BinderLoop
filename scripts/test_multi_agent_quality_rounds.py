#!/usr/bin/env python3
"""Token-bounded multi-agent quality analysis for SC2RBD rounds 04 and 05."""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.binder_quality_analysis_agent import (
    BinderQualityAnalysisAgent,
)
from binderloop.agents.config_parameter_contract import (
    supported_config_changes,
    unsupported_config_keys,
)
from binderloop.agents.context_compaction import (
    compact_context_for_quality,
    compact_structure_summary,
    fact_check_text_against_metric_facts,
)
from binderloop.memory import ExperimentMemoryStore, parameter_diff
from scripts.probe_llm_limits import build_prompt_corpus
from scripts.test_round04_quality_strategies import (
    _call_once,
    _configured_client,
    _json_bytes,
)


DEFAULT_RUN_DIR = "outputs/sc2rbd_closed_loop_llm_np_160s_8r_v18"

POSITIVE_SYSTEM = """You are SuccessMechanismAgent, a protein-interface
specialist. Analyze only evidence in the positive packet. Separate whole-binder
strict-positive success from boundary near misses and reusable local fragment
quality. A provisional_reference keeps label=near_miss and is never success. Return JSON only:
{
  "claims":[{"claim_id":"P1","claim":"...","scope":"local_fragment|whole_binder|population","evidence_ids":["..."],"counterevidence_ids":["..."],"confidence":0-1}],
  "reusable_mechanisms":[{"mechanism":"...","evidence_ids":["..."],"preserve":"...","risk":"..."}],
  "uncertainties":[{"question":"...","needed_evidence":["..."]}]
}
At most 6 claims, 5 mechanisms, and 4 uncertainties. Do not infer affinity or
causality from confidence metrics or two observational rounds."""

NEGATIVE_SYSTEM = """You are FailureMechanismAgent, a protein-interface
failure specialist. Analyze only other_negative_examples in the negative packet. Distinguish
foldability, interface localization, hotspot coverage, clash/geometry, and
filtering failures. Return JSON only:
{
  "claims":[{"claim_id":"N1","claim":"...","scope":"candidate|cluster|population","evidence_ids":["..."],"counterevidence_ids":["..."],"confidence":0-1}],
  "failure_clusters":[{"cluster":"...","mechanism":"...","evidence_ids":["..."],"repair_hypothesis":"...","risk":"..."}],
  "uncertainties":[{"question":"...","needed_evidence":["..."]}]
}
At most 6 claims, 5 clusters, and 4 uncertainties. Parameter associations are
not causal effects unless the packet explicitly provides controlled evidence."""

TRAJECTORY_SYSTEM = """You are TrajectoryMemoryAgent. Compare rounds 04 and 05
with indexed historical memory. Detect inheritance, regression, contradiction,
and current-vs-prior fact confusion. Return JSON only:
{
  "claims":[{"claim_id":"T1","claim":"...","evidence_ids":["..."],"status":"inherited|contradicted|uncertain","confidence":0-1}],
  "conflicts":[{"conflict_id":"C1","view_a":"...","view_b":"...","evidence_ids":["..."],"resolution_needed":"..."}],
  "parameter_lessons":[{"parameter":"...","observed_move":"...","outcome":"...","evidence_ids":["..."],"causal_strength":"none|weak|moderate"}],
  "uncertainties":[{"question":"...","needed_evidence":["..."]}]
}
At most 6 claims, 5 conflicts, 6 parameter lessons, and 4 uncertainties. Always
name the source round when quoting a metric."""

MANAGER_SYSTEM = """You are PhysicsDebateManager. You manage three evidence
specialists. Check every claim against immutable facts and physical constraints.
Prefer exact current-round metrics over memories and whole-binder outcomes over
local fragment quality when they conflict. Return JSON only:
{
  "claim_verdicts":[{"claim_id":"...","verdict":"accept|reject|revise","reason":"...","trusted_evidence_ids":["..."]}],
  "conflict_resolutions":[{"conflict":"...","resolution":"...","trusted_view":"positive|negative|trajectory|mixed","evidence_ids":["..."]}],
  "revision_requests":[{"agent":"positive|negative|trajectory","question":"...","needed_evidence_ids":["..."],"reason":"..."}],
  "provisional_strategy":[{"action":"...","evidence_ids":["..."],"expected_signal":"...","risk":"..."}],
  "manager_uncertainties":["..."]
}
Request revision only when a material decision cannot be made from supplied
evidence. At most two revision requests and six strategy actions."""

REVISION_SYSTEM = """You are revising a prior specialist analysis after a
physics manager identified a specific evidence gap. Answer only that request.
Return JSON only:
{
  "revised_claims":[{"claim_id":"...","claim":"...","evidence_ids":["..."],"confidence":0-1,"change_reason":"..."}],
  "request_answer":"...",
  "remaining_uncertainty":"..."
}
Do not add claims unrelated to the request."""

FINAL_SYSTEM = """You are PhysicsDebateManager producing the final binder
quality decision after specialist debate. Return JSON only:
{
  "overall_assessment":"...",
  "current_round_facts":{"round_id":5,"best_iptm":0.0,"success_count":0,"reward":0.0},
  "high_quality_modules":[{"module_id":"...","evidence":["..."],"evidence_ids":["..."],"likely_causes":["..."],"reuse_guidance":"...","confidence":0-1}],
  "low_quality_modules":[{"module_id":"...","evidence":["..."],"evidence_ids":["..."],"likely_causes":["..."],"repair_guidance":"...","confidence":0-1}],
  "causal_factors":[{"factor":"...","evidence":["..."],"evidence_ids":["..."],"impact":"positive|negative|mixed","confidence":0-1}],
  "next_round_guidance":[{"action":"...","evidence_ids":["..."],"parameter_or_constraint_change":"...","config_parameter_changes":{},"expected_signal":"...","risk":"..."}],
  "debate_audit":{"accepted_claim_ids":[],"rejected_claim_ids":[],"revised_claim_ids":[],"resolved_conflicts":[]},
  "uncertainties":["..."]
}
Use at most 5 items in each module/factor/guidance list. Every recommendation
must cite evidence IDs. Preserve current-vs-history labels. Use only executable
config keys supplied in the packet. Local fragment quality is not whole-binder
success. Avoid claiming controlled causality from observational rounds."""


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _round_context(round_dir: Path) -> Dict[str, Any]:
    quality = next(
        item
        for item in build_prompt_corpus(round_dir)
        if item["kind"] == "quality.round_analysis"
    )
    return copy.deepcopy(dict(quality["user"]["context"]))


def _candidate_examples(round_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    active = dict(_read_json(round_dir / "active_learning_examples.json", {}) or {})
    current = dict(active.get("current_round") or {})
    positives = [dict(item) for item in (
        current.get("strict_positive_examples") or current.get("positive_examples") or []
    )]
    near_misses = [dict(item) for item in current.get("near_miss_examples") or []]
    negatives = [dict(item) for item in (
        current.get("other_negative_examples") or current.get("hard_negative_examples") or []
    )]
    return positives, near_misses, negatives


def _trim_structure(item: Mapping[str, Any], evidence_id: str) -> Dict[str, Any]:
    row = compact_structure_summary(item)
    row["evidence_id"] = evidence_id
    source = str(row.get("structure_file") or "")
    row["structure_file"] = Path(source).name if source else ""
    row["high_quality_fragments"] = sorted(
        list(row.get("high_quality_fragments") or []),
        key=lambda value: float((value or {}).get("quality_score") or 0.0),
        reverse=True,
    )[:2]
    row["low_quality_fragments"] = sorted(
        list(row.get("low_quality_fragments") or []),
        key=lambda value: float((value or {}).get("quality_score") or 0.0),
    )[:2]
    row["target_contact_residues"] = list(
        row.get("target_contact_residues") or []
    )[:6]
    return row


def _structures_for_candidates(
    round_dir: Path,
    candidate_ids: Sequence[str],
    *,
    prefix: str,
    per_candidate: int = 2,
) -> List[Dict[str, Any]]:
    structural = dict(_read_json(round_dir / "structure_evaluation.json", {}) or {})
    summaries = list(structural.get("summaries") or [])
    selected: List[Dict[str, Any]] = []
    seen = set()
    for candidate_id in candidate_ids:
        matches = [
            item
            for item in summaries
            if candidate_id and candidate_id in str((item or {}).get("structure_file") or "")
        ]
        matches.sort(
            key=lambda item: (
                float((item or {}).get("reliability_score") or 0.0),
                max(
                    (
                        float((fragment or {}).get("quality_score") or 0.0)
                        for fragment in (
                            list((item or {}).get("high_quality_fragments") or [])
                            + list((item or {}).get("low_quality_fragments") or [])
                        )
                    ),
                    default=0.0,
                ),
            ),
            reverse=True,
        )
        for index, item in enumerate(matches[:per_candidate]):
            source = str((item or {}).get("structure_file") or "")
            if source in seen:
                continue
            seen.add(source)
            selected.append(
                _trim_structure(
                    item,
                    f"{prefix}:STRUCT:{candidate_id}:{index + 1}",
                )
            )
    return selected


def _metric_evidence(round_id: int, round_dir: Path) -> Dict[str, Any]:
    evaluation = dict(_read_json(round_dir / "evaluation_summary.json", {}) or {})
    rollback = dict(_read_json(round_dir / "rollback_decision.json", {}) or {})
    context = _round_context(round_dir)
    return {
        "evidence_id": f"R{round_id}:METRICS",
        "round_id": round_id,
        "candidate_filtering": evaluation.get("candidate_filtering"),
        "tag_counts": evaluation.get("tag_counts"),
        "metric_facts": (context.get("evaluation") or {}).get("metric_facts"),
        "outcome": rollback.get("outcome"),
        "decision": rollback.get("decision"),
    }


def _config_for_round(round_dir: Path) -> Dict[str, Any]:
    checkpoint = dict(_read_json(round_dir / "round_checkpoint.json", {}) or {})
    jobs = list(checkpoint.get("current_jobs") or [])
    params = dict((jobs[0] if jobs else {}).get("params") or {})
    return supported_config_changes(params, include_internal=True)


def _historical_memory(run_dir: Path) -> List[Dict[str, Any]]:
    store = ExperimentMemoryStore(run_dir / "memory")
    memory = store.load()
    metrics = {
        int(item.get("round_id", -1)): dict(item)
        for item in memory.round_metrics or []
    }
    result: List[Dict[str, Any]] = []
    for record in sorted(memory.rounds, key=lambda item: item.round_id):
        evaluation = dict(record.evaluation or {})
        metric = metrics.get(record.round_id, {})
        result.append({
            "evidence_id": f"MEM:R{record.round_id}",
            "round_id": record.round_id,
            "reward": metric.get("reward", record.reward),
            "best_iptm": metric.get("best_iptm"),
            "median_iptm": metric.get("median_iptm"),
            "core_objective": metric.get("core_objective"),
            "success_count": metric.get("success_count"),
            "arm": metric.get("arm_signature"),
            "failure_tags": [
                key
                for key, value in dict(evaluation.get("tag_counts") or {}).items()
                if value and not str(key).startswith("pass_")
            ],
            "config": supported_config_changes(
                dict(record.config_snapshot or {}),
                include_internal=True,
            ),
        })
    return result


def _build_packets(run_dir: Path) -> Dict[str, Any]:
    round4 = run_dir / "round_04"
    round5 = run_dir / "round_05"
    r4_pos, r4_near, r4_neg = _candidate_examples(round4)
    r5_pos, r5_near, r5_neg = _candidate_examples(round5)
    r4_metric = _metric_evidence(4, round4)
    r5_metric = _metric_evidence(5, round5)

    positive_ids = [str(item.get("candidate_id") or "") for item in r5_pos]
    # Round 4 has no strict positives; near misses remain explicitly provisional.
    promising_r4 = r4_near[:2]
    promising_ids = [str(item.get("candidate_id") or "") for item in promising_r4]
    positive_packet = {
        "definitions": {
            "whole_binder_positive": "iPTM>=0.50, PAE<=10A, pTM>=0.70, refold_RMSD<=2.5A",
            "local_fragment_warning": (
                "A high-quality fragment or near miss is boundary/local evidence, "
                "not a positive binder."
            ),
        },
        "round_metrics": [r4_metric, r5_metric],
        "strict_positive_examples": [
            {"evidence_id": f"R5:POS:{item.get('candidate_id')}", **item}
            for item in r5_pos[:4]
        ],
        "near_miss_boundary_examples": [
            {"evidence_id": f"R5:NEAR:{item.get('candidate_id')}", **item}
            for item in r5_near[:4]
        ],
        "positive_structures": _structures_for_candidates(
            round5,
            positive_ids,
            prefix="R5:POS",
            per_candidate=2,
        ),
        "provisional_reference": [
            {"evidence_id": f"R4:NEAR:{item.get('candidate_id')}", **item,
             "label": "near_miss", "success_counted": False,
             "evidence_role": "provisional_reference"}
            for item in promising_r4
        ],
        "round4_counterexample_structures": _structures_for_candidates(
            round4,
            promising_ids,
            prefix="R4:LOCAL",
            per_candidate=1,
        ),
    }

    negative_examples = r4_neg[:5] + r5_neg[:6]
    negative_ids_r4 = [str(item.get("candidate_id") or "") for item in r4_neg[:3]]
    negative_ids_r5 = [str(item.get("candidate_id") or "") for item in r5_neg[:4]]
    negative_packet = {
        "round_metrics": [r4_metric, r5_metric],
        "other_negative_examples": [
            {
                "evidence_id": f"R{item.get('round_id')}:NEG:{item.get('candidate_id')}",
                **item,
            }
            for item in negative_examples
        ],
        "negative_structures": (
            _structures_for_candidates(
                round4,
                negative_ids_r4,
                prefix="R4:NEG",
                per_candidate=1,
            )
            + _structures_for_candidates(
                round5,
                negative_ids_r5,
                prefix="R5:NEG",
                per_candidate=1,
            )
        ),
        "aggregate_structure_tags": [
            {
                "evidence_id": "R4:STRUCT_AGG",
                "round_id": 4,
                "aggregate_tags": (
                    _read_json(round4 / "structure_evaluation.json", {}) or {}
                ).get("aggregate_tags"),
                "reliable_seed_fraction": (
                    _read_json(round4 / "structure_evaluation.json", {}) or {}
                ).get("reliable_seed_fraction"),
            },
            {
                "evidence_id": "R5:STRUCT_AGG",
                "round_id": 5,
                "aggregate_tags": (
                    _read_json(round5 / "structure_evaluation.json", {}) or {}
                ).get("aggregate_tags"),
                "reliable_seed_fraction": (
                    _read_json(round5 / "structure_evaluation.json", {}) or {}
                ).get("reliable_seed_fraction"),
            },
        ],
    }

    config4, config5 = _config_for_round(round4), _config_for_round(round5)
    trajectory_packet = {
        "round_metrics": [r4_metric, r5_metric],
        "round4_config": {"evidence_id": "R4:CONFIG", **config4},
        "round5_config": {"evidence_id": "R5:CONFIG", **config5},
        "round4_to_round5_parameter_diff": {
            "evidence_id": "R4-R5:PARAM_DIFF",
            "changes": parameter_diff(
                config4,
                config5,
                allowed_keys=set(config4) | set(config5),
            ),
        },
        "historical_memory": _historical_memory(run_dir),
        "strict_positive_populations": {
            "round_04": {
                "strict_positive_count": len(r4_pos),
                "representative_samples": r4_pos[:3],
            },
            "round_05": {
                "strict_positive_count": len(r5_pos),
                "representative_samples": r5_pos[:3],
            },
        },
    }

    physics_packet = {
        "immutable_current_round": r5_metric,
        "comparison_round": r4_metric,
        "rules": [
            {
                "evidence_id": "PHYS:LOCAL_GLOBAL",
                "rule": "Local fragment quality cannot establish whole-binder success.",
            },
            {
                "evidence_id": "PHYS:IPTM",
                "rule": "Higher iPTM supports interface confidence but does not prove affinity.",
            },
            {
                "evidence_id": "PHYS:PAE",
                "rule": "Lower interface PAE supports better localized relative geometry.",
            },
            {
                "evidence_id": "PHYS:RMSD",
                "rule": "Low refold RMSD supports consistency, not binding by itself.",
            },
            {
                "evidence_id": "PHYS:CAUSALITY",
                "rule": "Two observational rounds cannot identify controlled parameter causality.",
            },
            {
                "evidence_id": "PHYS:CHAIN",
                "rule": "Output chain relabeling alone is not a chain-mismatch failure.",
            },
        ],
        "executable_config_keys": sorted(
            set(config4) | set(config5)
        ),
    }
    return {
        "positive": positive_packet,
        "negative": negative_packet,
        "trajectory": trajectory_packet,
        "physics": physics_packet,
    }


def _collect_evidence_ids(value: Any) -> set:
    result = set()
    if isinstance(value, Mapping):
        evidence_id = value.get("evidence_id")
        if evidence_id:
            result.add(str(evidence_id))
        for child in value.values():
            result.update(_collect_evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_collect_evidence_ids(child))
    return result


def _referenced_evidence_ids(value: Any) -> set:
    result = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"evidence_ids", "counterevidence_ids", "trusted_evidence_ids"}:
                result.update(str(item) for item in child or [])
            else:
                result.update(_referenced_evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_referenced_evidence_ids(child))
    return result


def _find_evidence_items(value: Any, wanted_ids: set) -> List[Dict[str, Any]]:
    """Return only evidence records explicitly requested by the manager."""
    result: List[Dict[str, Any]] = []
    seen = set()
    if isinstance(value, Mapping):
        evidence_id = str(value.get("evidence_id") or "")
        if evidence_id and evidence_id in wanted_ids and evidence_id not in seen:
            result.append(copy.deepcopy(dict(value)))
            seen.add(evidence_id)
        for child in value.values():
            for item in _find_evidence_items(child, wanted_ids):
                item_id = str(item.get("evidence_id") or "")
                if item_id and item_id not in seen:
                    result.append(item)
                    seen.add(item_id)
    elif isinstance(value, list):
        for child in value:
            for item in _find_evidence_items(child, wanted_ids):
                item_id = str(item.get("evidence_id") or "")
                if item_id and item_id not in seen:
                    result.append(item)
                    seen.add(item_id)
    return result


def _call_agent(
    client,
    *,
    agent: str,
    system: str,
    packet: Mapping[str, Any],
    max_tokens: int,
    attempts: List[Dict[str, Any]],
    state: Dict[str, Any],
    output: Path,
) -> Optional[Dict[str, Any]]:
    record, parsed = _call_once(
        client,
        strategy="multi_agent_quality",
        stage=agent,
        system=system,
        user=packet,
        max_tokens=max_tokens,
        attempt=1,
    )
    attempts.append(record)
    state["attempts"] = attempts
    if parsed is not None:
        state.setdefault("agent_outputs", {})[agent] = parsed
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    time.sleep(15.0)
    return parsed


def _validate_final(
    final: Optional[Mapping[str, Any]],
    packets: Mapping[str, Any],
) -> Dict[str, Any]:
    if not final:
        return {"valid": False, "reason": "missing_final"}
    required = {
        "overall_assessment",
        "current_round_facts",
        "high_quality_modules",
        "low_quality_modules",
        "causal_factors",
        "next_round_guidance",
        "debate_audit",
        "uncertainties",
    }
    missing = sorted(required - set(final))
    facts = dict(final.get("current_round_facts") or {})
    expected = dict(packets["physics"]["immutable_current_round"]["outcome"] or {})
    fact_errors = []
    for key in ("round_id", "best_iptm", "success_count", "reward"):
        actual, wanted = facts.get(key), expected.get(key)
        if actual != wanted:
            fact_errors.append(f"{key}: {actual!r} != {wanted!r}")
    known_ids = _collect_evidence_ids(packets)
    referenced = _referenced_evidence_ids(final)
    unknown_ids = sorted(referenced - known_ids)
    unsupported = []
    for index, row in enumerate(final.get("next_round_guidance") or []):
        changes = dict((row or {}).get("config_parameter_changes") or {})
        for key in unsupported_config_keys(changes):
            unsupported.append({"guidance_index": index, "key": key})
    tags = dict(
        packets["physics"]["immutable_current_round"].get("tag_counts") or {}
    )
    dominant_tags = [
        key
        for key, value in sorted(
            tags.items(),
            key=lambda pair: float(pair[1] or 0),
            reverse=True,
        )
        if value and not str(key).startswith("pass_")
    ][:3]
    text = json.dumps(final, ensure_ascii=False).lower()
    missing_failure_tags = [
        tag for tag in dominant_tags if str(tag).lower() not in text
    ]
    return {
        "valid": not missing and not fact_errors and not unknown_ids and not unsupported,
        "missing_keys": missing,
        "current_fact_errors": fact_errors,
        "unknown_evidence_ids": unknown_ids,
        "unsupported_config_changes": unsupported,
        "dominant_failure_tags": dominant_tags,
        "missing_dominant_failure_tags": missing_failure_tags,
        "known_evidence_id_count": len(known_ids),
        "referenced_evidence_id_count": len(referenced),
        "evidence_reference_coverage": round(
            len(referenced & known_ids) / max(1, len(known_ids)),
            6,
        ),
    }


def _validate_baseline(
    result: Optional[Mapping[str, Any]],
    packets: Mapping[str, Any],
) -> Dict[str, Any]:
    if not result:
        return {"valid": False, "reason": "missing_baseline"}
    required = {
        "overall_assessment",
        "high_quality_modules",
        "low_quality_modules",
        "causal_factors",
        "next_round_guidance",
    }
    missing = sorted(required - set(result))
    metric_facts = dict(
        packets["physics"]["immutable_current_round"].get("metric_facts") or {}
    )
    fact_issues = fact_check_text_against_metric_facts(
        json.dumps(result, ensure_ascii=False),
        metric_facts,
    )
    unsupported = []
    for index, row in enumerate(result.get("next_round_guidance") or []):
        for key in unsupported_config_keys(
            dict((row or {}).get("config_parameter_changes") or {})
        ):
            unsupported.append({"guidance_index": index, "key": key})
    tags = dict(
        packets["physics"]["immutable_current_round"].get("tag_counts") or {}
    )
    dominant_tags = [
        key
        for key, value in sorted(
            tags.items(),
            key=lambda pair: float(pair[1] or 0),
            reverse=True,
        )
        if value and not str(key).startswith("pass_")
    ][:3]
    text = json.dumps(result, ensure_ascii=False).lower()
    missing_tags = [
        tag for tag in dominant_tags if str(tag).lower() not in text
    ]
    return {
        "valid": not missing and not fact_issues and not unsupported,
        "missing_keys": missing,
        "fact_check_issues": fact_issues,
        "unsupported_config_changes": unsupported,
        "dominant_failure_tags": dominant_tags,
        "missing_dominant_failure_tags": missing_tags,
        "has_evidence_provenance": bool(_referenced_evidence_ids(result)),
    }


def _sanitize_final_guidance(
    final: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if final is None:
        return None
    result = copy.deepcopy(dict(final))
    sanitized = []
    for row in result.get("next_round_guidance") or []:
        item = dict(row or {})
        proposed = dict(item.get("config_parameter_changes") or {})
        ignored = unsupported_config_keys(proposed)
        changes = supported_config_changes(proposed)
        if "config_overrides" in changes and not isinstance(
            changes["config_overrides"],
            list,
        ):
            ignored.append("config_overrides:invalid_shape")
            changes.pop("config_overrides", None)
        item["config_parameter_changes"] = changes
        if ignored:
            item["ignored_config_parameter_changes"] = sorted(set(ignored))
        sanitized.append(item)
    result["next_round_guidance"] = sanitized
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--llm-config", default="configs/llm_endpoints.gpt.json")
    parser.add_argument("--llm-model", default="gpt-5.5")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    run_dir = resolve(args.run_dir)
    packets = _build_packets(run_dir)
    default_name = (
        "multi_agent_quality_live.json"
        if args.live
        else "multi_agent_quality_packets.json"
    )
    output = resolve(args.out) if args.out else (
        root / "outputs/gpt55_limit_probe_sc2rbd_round45" / default_name
    )
    state: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "live": bool(args.live),
        "protocol": {
            "logical_agents": [
                "SuccessMechanismAgent",
                "FailureMechanismAgent",
                "TrajectoryMemoryAgent",
                "PhysicsDebateManager",
            ],
            "max_outstanding_requests": 1,
            "reasoning": "low",
            "max_revisions": max(0, int(args.max_revisions)),
        },
        "packet_bytes": {
            key: _json_bytes(value) for key, value in packets.items()
        },
        "packet_evidence_counts": {
            key: len(_collect_evidence_ids(value))
            for key, value in packets.items()
        },
        "attempts": [],
        "agent_outputs": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.live:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    client = _configured_client(
        resolve(args.llm_config),
        args.llm_model,
        reasoning="low",
    )
    attempts: List[Dict[str, Any]] = []

    if not args.skip_baseline:
        round5_context = _round_context(run_dir / "round_05")
        baseline_packet = {"round_id": 5, "context": round5_context}
        baseline = _call_agent(
            client,
            agent="single_agent_baseline",
            system=BinderQualityAnalysisAgent.SYSTEM,
            packet=baseline_packet,
            max_tokens=2400,
            attempts=attempts,
            state=state,
            output=output,
        )
        state["baseline_validation"] = _validate_baseline(
            baseline,
            packets,
        )
        if args.baseline_only:
            state["summary"] = {
                "request_count": len(attempts),
                "successful_request_count": sum(
                    1 for item in attempts if item.get("ok")
                ),
                "error_types": sorted({
                    str(item.get("error_type"))
                    for item in attempts
                    if item.get("error_type")
                }),
                "elapsed_seconds_total": round(
                    sum(
                        float(item.get("elapsed_seconds") or 0.0)
                        for item in attempts
                    ),
                    3,
                ),
            }
            output.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return

    outputs: Dict[str, Optional[Dict[str, Any]]] = {}
    for name, system in (
        ("positive", POSITIVE_SYSTEM),
        ("negative", NEGATIVE_SYSTEM),
        ("trajectory", TRAJECTORY_SYSTEM),
    ):
        outputs[name] = _call_agent(
            client,
            agent=name,
            system=system,
            packet=packets[name],
            max_tokens=1400,
            attempts=attempts,
            state=state,
            output=output,
        )

    manager_packet = {
        "immutable_physics_packet": packets["physics"],
        "specialist_outputs": outputs,
        "three_class_evidence": {
            "strict_positive_examples": packets["positive"].get("strict_positive_examples"),
            "near_miss_examples": packets["positive"].get("near_miss_boundary_examples"),
            "other_negative_examples": packets["negative"].get("other_negative_examples"),
        },
    }
    manager = _call_agent(
        client,
        agent="manager_deliberation",
        system=MANAGER_SYSTEM,
        packet=manager_packet,
        max_tokens=1600,
        attempts=attempts,
        state=state,
        output=output,
    )

    revisions: Dict[str, Any] = {}
    requests = list((manager or {}).get("revision_requests") or [])
    for index, request in enumerate(
        requests[: max(0, int(args.max_revisions))]
    ):
        agent = str((request or {}).get("agent") or "")
        if agent not in outputs:
            continue
        revision_packet = {
            "agent": agent,
            "manager_request": request,
            "requested_evidence": _find_evidence_items(
                packets,
                {
                    str(item)
                    for item in ((request or {}).get("needed_evidence_ids") or [])
                },
            ),
            "original_output": outputs[agent],
            "immutable_physics_packet": packets["physics"],
        }
        revisions[agent] = _call_agent(
            client,
            agent=f"revision_{index + 1}_{agent}",
            system=REVISION_SYSTEM,
            packet=revision_packet,
            max_tokens=1000,
            attempts=attempts,
            state=state,
            output=output,
        )

    final_packet = {
        "immutable_physics_packet": packets["physics"],
        "specialist_outputs": outputs,
        "manager_deliberation": manager,
        "targeted_revisions": revisions,
        "executable_config_keys": packets["physics"]["executable_config_keys"],
    }
    final_raw = _call_agent(
        client,
        agent="manager_final",
        system=FINAL_SYSTEM,
        packet=final_packet,
        max_tokens=2400,
        attempts=attempts,
        state=state,
        output=output,
    )
    final = _sanitize_final_guidance(final_raw)
    if final is not None:
        state["agent_outputs"]["manager_final_sanitized"] = final
    state["final_validation"] = _validate_final(final, packets)
    state["summary"] = {
        "request_count": len(attempts),
        "successful_request_count": sum(1 for item in attempts if item.get("ok")),
        "error_types": sorted({
            str(item.get("error_type"))
            for item in attempts
            if item.get("error_type")
        }),
        "elapsed_seconds_total": round(
            sum(float(item.get("elapsed_seconds") or 0.0) for item in attempts),
            3,
        ),
        "prompt_tokens_total": sum(
            int((item.get("usage") or {}).get("prompt_tokens") or 0)
            for item in attempts
        ),
        "completion_tokens_total": sum(
            int((item.get("usage") or {}).get("completion_tokens") or 0)
            for item in attempts
        ),
        "revision_count": len(revisions),
    }
    output.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
