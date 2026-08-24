"""Canonical lexicographic ranking for whole-binder quality.

``core_objective`` remains a monitoring/legacy scalar. New decisions must use
``core_rank_key`` so a strong metric cannot compensate for a failed gate.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from binderloop.analysis.quality_thresholds import success_thresholds


CORE_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "iptm": 0.35,
    "pae": 0.25,
    "ptm": 0.25,
    "rmsd": 0.15,
}

SECONDARY_TIE_BREAKER_WEIGHT = 0.04
CoreRankKey = Tuple[int, float, float, float, float]
RoundRankKey = Tuple[int, float, float, float, float]


def _float_first(row: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    nested = []
    for nested_key in ("raw", "raw_metrics", "metrics"):
        value = row.get(nested_key)
        if isinstance(value, Mapping):
            nested.append(value)
    for source in (row, *nested):
        for key in keys:
            value = source.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float(default)


def pae_confidence(pae: float) -> float:
    """Normalize design-to-target PAE so lower PAE means higher confidence."""
    return min(1.0, max(0.0, 1.0 - (float(pae) - 2.0) / 18.0))


def rmsd_consistency(rmsd: float) -> float:
    """Normalize refold RMSD so lower RMSD means higher consistency."""
    return min(1.0, max(0.0, 1.0 - float(rmsd) / 5.0))


def secondary_tie_breaker(row: Mapping[str, Any]) -> float:
    hbond_raw = _float_first(row, "hotspot_contact_raw", "hotspot_contact", "bindingsite_contact", "design_residue_iptm", "plip_hbonds_refolded")
    sasa_raw = _float_first(row, "delta_sasa_refolded", "buried_sasa", "interface_sasa", default=0.0)
    hbond = min(1.0, max(0.0, hbond_raw / 12.0))
    sasa = min(1.0, max(0.0, sasa_raw / 1200.0))
    return 0.65 * hbond + 0.35 * sasa


def core_metrics_from_row(row: Mapping[str, Any]) -> Dict[str, float]:
    iptm = _float_first(row, "design_to_target_iptm", "iptm", "interface_confidence")
    pae = _float_first(row, "min_design_to_target_pae", "min_interaction_pae", default=100000.0)
    ptm = _float_first(row, "design_ptm", "binder_plddt", "plddt")
    rmsd = _float_first(row, "designfolding-filter_rmsd", "designfolding_filter_rmsd", "filter_rmsd_design", "bb_rmsd", default=100000.0)
    pae_score = pae_confidence(pae)
    rmsd_score = rmsd_consistency(rmsd)
    core = (
        CORE_OBJECTIVE_WEIGHTS["iptm"] * min(1.0, max(0.0, iptm))
        + CORE_OBJECTIVE_WEIGHTS["pae"] * pae_score
        + CORE_OBJECTIVE_WEIGHTS["ptm"] * min(1.0, max(0.0, ptm))
        + CORE_OBJECTIVE_WEIGHTS["rmsd"] * rmsd_score
    )
    return {
        "design_to_target_iptm": iptm,
        "min_design_to_target_pae": pae,
        "design_ptm": ptm,
        "designfolding_filter_rmsd": rmsd,
        "pae_confidence": pae_score,
        "refold_consistency": rmsd_score,
        "core_objective": core,
    }


def normalized_core_margins(row: Mapping[str, Any]) -> Dict[str, float]:
    """Signed normalized margins; zero is the canonical success boundary."""
    metrics = core_metrics_from_row(row)
    thresholds = success_thresholds()
    return {
        "iptm": (metrics["design_to_target_iptm"] - thresholds["design_to_target_iptm"]) / 0.20,
        "pae": (thresholds["min_design_to_target_pae"] - metrics["min_design_to_target_pae"]) / 8.0,
        "ptm": (metrics["design_ptm"] - thresholds["design_ptm"]) / 0.30,
        "rmsd": (thresholds["designfolding_filter_rmsd"] - metrics["designfolding_filter_rmsd"]) / 2.5,
    }


def core_rank_key(row: Mapping[str, Any]) -> CoreRankKey:
    """Return the sole new-decision candidate ordering key (larger is better)."""
    metrics = core_metrics_from_row(row)
    margins = normalized_core_margins(metrics)
    gate_pass = int(min(margins.values()) >= 0.0)
    return (
        gate_pass,
        min(margins.values()),
        metrics["design_to_target_iptm"],
        -metrics["min_design_to_target_pae"],
        -metrics["designfolding_filter_rmsd"],
    )


def candidate_core_score(row: Mapping[str, Any], *, include_secondary_tiebreaker: bool = False) -> float:
    core = float(core_metrics_from_row(row).get("core_objective", 0.0))
    if include_secondary_tiebreaker:
        core += SECONDARY_TIE_BREAKER_WEIGHT * secondary_tie_breaker(row)
    return core


def rank_by_core_objective(candidates: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Compatibility name for canonical CoreRankKey ordering."""
    return sorted(list(candidates or []), key=core_rank_key, reverse=True)


def median(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def round_core_objective(candidates: Sequence[Mapping[str, Any]], *, top_k: int = 5) -> float:
    ranked = rank_by_core_objective(candidates)
    top = [candidate_core_score(row) for row in ranked[: max(1, int(top_k or 1))]]
    return median(top)


def round_rank_key(candidates: Sequence[Mapping[str, Any]], *, top_k: int = 5) -> RoundRankKey:
    """Rank rounds by strict-positive count then robust top-k quality.

    The primary component is deliberately an absolute count over the complete
    evaluable population.  Analysis filters (for example ``iPTM > 0.35``) must
    never change the denominator or the historical-best decision.
    """
    rows = list(candidates or [])
    if not rows:
        return (0, -1_000_000.0, 0.0, -100_000.0, -100_000.0)
    ranked = rank_by_core_objective(rows)[:max(1, int(top_k or 1))]
    keys = [core_rank_key(row) for row in ranked]
    metrics = [core_metrics_from_row(row) for row in ranked]
    strict_count = sum(key[0] for key in (core_rank_key(row) for row in rows))
    return (
        strict_count,
        median([key[1] for key in keys]),
        median([item["design_to_target_iptm"] for item in metrics]),
        -median([item["min_design_to_target_pae"] for item in metrics]),
        -median([item["designfolding_filter_rmsd"] for item in metrics]),
    )


def monitoring_scalar_from_round_rank(key: Sequence[float]) -> float:
    """Simple scalar for collaboration/trends; never use it for selection."""
    values = list(key or [])
    if len(values) < 2:
        return 0.0
    worst_margin = max(-10.0, min(10.0, float(values[1])))
    return float(values[0]) + 0.01 * worst_margin


def core_metric_stats(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    metrics = [core_metrics_from_row(row) for row in candidates or []]
    if not metrics:
        return {
            "best_iptm": 0.0,
            "mean_iptm": 0.0,
            "best_min_pae": 100000.0,
            "mean_min_pae": 0.0,
            "best_design_ptm": 0.0,
            "mean_design_ptm": 0.0,
            "best_refold_rmsd": 100000.0,
            "mean_refold_rmsd": 0.0,
            "best_core_objective": 0.0,
            "mean_core_objective": 0.0,
        }

    def values(key: str) -> List[float]:
        return [float(item.get(key, 0.0)) for item in metrics]

    def mean(items: Sequence[float]) -> float:
        return sum(items) / len(items) if items else 0.0

    iptm = values("design_to_target_iptm")
    pae = values("min_design_to_target_pae")
    ptm = values("design_ptm")
    rmsd = values("designfolding_filter_rmsd")
    core = values("core_objective")
    return {
        "best_iptm": max(iptm),
        "mean_iptm": mean(iptm),
        "best_min_pae": min(pae),
        "mean_min_pae": mean(pae),
        "best_design_ptm": max(ptm),
        "mean_design_ptm": mean(ptm),
        "best_refold_rmsd": min(rmsd),
        "mean_refold_rmsd": mean(rmsd),
        "best_core_objective": max(core),
        "mean_core_objective": mean(core),
    }
