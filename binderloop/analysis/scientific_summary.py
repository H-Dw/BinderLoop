"""Deterministic, versioned summaries of scientific candidate metrics."""

import math
from typing import Any, Dict, Mapping, Optional, Sequence

SCIENTIFIC_SUMMARY_SCHEMA_VERSION = "1.0"

_METRICS = {
    "iptm": ("design_to_target_iptm", "iptm", "interface_confidence"),
    "design_ptm": ("design_ptm", "binder_plddt", "plddt"),
    "min_pae": ("min_design_to_target_pae", "min_interaction_pae"),
    "refold_rmsd": ("designfolding-filter_rmsd", "designfolding_filter_rmsd", "filter_rmsd_design", "bb_rmsd"),
}

def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def _value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for source in (row, _mapping(row.get("raw")), _mapping(row.get("raw_metrics")), _mapping(row.get("metrics"))):
        for key in keys:
            if key in source and source.get(key) not in (None, ""):
                return source.get(key)
    return None

def _metric_summary(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], *, lower_is_better: bool = False) -> Dict[str, Any]:
    values = []
    invalid = 0
    missing = 0
    for row in rows:
        raw = _value(row, keys)
        if raw is None:
            missing += 1
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not math.isfinite(value):
            invalid += 1
            continue
        values.append(value)
    result: Dict[str, Any] = {"valid_count": len(values), "missing_count": missing, "invalid_count": invalid}
    if values:
        result.update({"min": round(min(values), 6), "max": round(max(values), 6), "mean": round(sum(values) / len(values), 6), "best": round(min(values) if lower_is_better else max(values), 6)})
    return result

def build_scientific_summary(candidates: Optional[Sequence[Mapping[str, Any]]], *, evaluation: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return immutable facts without coercing absent/bad measurements to zero."""
    rows = list(candidates or [])
    data = _mapping(evaluation)
    filtering = _mapping(data.get("candidate_filtering"))
    metrics = {name: _metric_summary(rows, keys, lower_is_better=name in {"min_pae", "refold_rmsd"}) for name, keys in _METRICS.items()}
    iptm_values = [_value(row, _METRICS["iptm"]) for row in rows]
    finite_iptm = []
    for raw in iptm_values:
        try:
            value = float(raw)
            if math.isfinite(value): finite_iptm.append(value)
        except (TypeError, ValueError):
            pass
    gates: Dict[str, Any] = {
        "harness_compute_gate": {"pass_count": data.get("success_count"), "fail_count": data.get("failure_count"), "source": "evaluation"},
    }
    per_filter = list(filtering.get("per_filter") or [])
    if per_filter:
        gates["additional_filters"] = per_filter
    elif finite_iptm:
        gates["iptm_gt_0_35_derived"] = {"pass_count": sum(v > .35 for v in finite_iptm), "fail_count": sum(v <= .35 for v in finite_iptm), "source": "metric_derivation"}
    pass_filters = [_value(row, ("pass_filters",)) for row in rows]
    known = [v for v in pass_filters if isinstance(v, bool)]
    if known:
        gates["boltzgen_pass_filters"] = {"pass_count": sum(known), "fail_count": len(known)-sum(known), "source": "candidate_column"}
    return {
        "schema": "binder_harness.scientific_summary", "schema_version": SCIENTIFIC_SUMMARY_SCHEMA_VERSION,
        "scope": {"name": filtering.get("analysis_scope") or "provided_candidates", "candidate_count": len(rows), "input_candidate_count": filtering.get("input_candidate_count"), "analysis_candidate_count": filtering.get("analysis_candidate_count")},
        "metrics": metrics, "gates": gates,
    }
