"""Executable harness-side candidate selection policies."""
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def apply_candidate_selection(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policies_by_job: Mapping[str, Mapping[str, Any]],
    structure_summaries: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summaries = list(structure_summaries or [])
    by_path: Dict[str, Mapping[str, Any]] = {}
    for summary in summaries:
        source = str(summary.get("structure_file") or "").strip()
        if source:
            by_path[_normalized_path(source)] = summary
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    not_evaluable: List[str] = []
    for raw in candidates or []:
        row = dict(raw)
        job_id = str(row.get("job_id") or "")
        policy = dict(policies_by_job.get(job_id) or {})
        if policy.get("type") != "cross_chain_heavy_atom_clash":
            accepted.append(row)
            continue
        summary = _summary_for_candidate(row, by_path)
        if summary is None:
            not_evaluable.append(str(row.get("candidate_id") or row.get("design") or row.get("name") or ""))
            # An unbound candidate has aggregate structural evidence only. It is
            # not safe to reject or rank that candidate from another row's clash.
            accepted.append(row)
            continue
        if policy.get("gate", True) and not bool(summary.get("clash_gate_pass", True)):
            rejected.append({"candidate": row, "reason": "heavy_atom_clash_gate", "clash_count": summary.get("heavy_atom_clash_count"), "clash_density": summary.get("heavy_atom_clash_density")})
            continue
        row["harness_clash_rank"] = list(summary.get("clash_rank") or [])
        accepted.append(row)
    accepted.sort(key=lambda row: tuple(row.get("harness_clash_rank") or [1.0, 0.0, 0.0]), reverse=True)
    return accepted, {"policy": "cross_chain_heavy_atom_clash", "input": len(candidates or []), "accepted": len(accepted), "rejected": rejected, "not_evaluable": not_evaluable}


def _normalized_path(value: Any) -> str:
    return str(Path(str(value)).expanduser())


def _summary_for_candidate(
    row: Mapping[str, Any],
    by_path: Mapping[str, Mapping[str, Any]],
):
    structure_file = str(row.get("structure_file") or "").strip()
    if not structure_file:
        return None
    return by_path.get(_normalized_path(structure_file))
