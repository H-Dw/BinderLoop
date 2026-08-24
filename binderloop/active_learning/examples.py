"""Contrastive example mining for closed-loop active learning.

The harness keeps user-owned ``additional_filters`` as provenance while these
helpers classify every evaluable candidate into strict positives, boundary
near misses, or other negatives for LLM reasoning and strategy selection.
"""

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from binderloop.analysis.core_objective import core_metrics_from_row, core_rank_key
from binderloop.analysis.quality_thresholds import (
    SUCCESS_IPTM_MIN,
    SUCCESS_PAE_MAX_ANGSTROM,
    SUCCESS_PTM_MIN,
    SUCCESS_RMSD_MAX_ANGSTROM,
)


POSITIVE_IPTM_MIN = SUCCESS_IPTM_MIN
DEFAULT_HARD_NEGATIVE_IPTM_MIN = 0.0
POSITIVE_PAE_MAX = SUCCESS_PAE_MAX_ANGSTROM
POSITIVE_PTM_MIN = SUCCESS_PTM_MIN
POSITIVE_REFOLD_RMSD_MAX = SUCCESS_RMSD_MAX_ANGSTROM
DEFAULT_MAX_CURRENT_POSITIVES = 6
DEFAULT_MAX_CURRENT_HARD_NEGATIVES = 10
DEFAULT_NEAR_MISS_TOP_K = 4
DEFAULT_NEAR_MISS_MIN_CONFIDENCE = 0.30
DEFAULT_NEAR_MISS_WEIGHT = 0.25
DEFAULT_MAX_PRIOR_POSITIVES = 8
DEFAULT_MAX_PRIOR_HARD_NEGATIVES = 12


def build_active_learning_examples(
    *,
    round_id: int,
    current_candidates: Sequence[Mapping[str, Any]],
    prior_rounds: Optional[Sequence[Mapping[str, Any]]] = None,
    additional_filters: Optional[Sequence[Any]] = None,
    max_current_positives: int = DEFAULT_MAX_CURRENT_POSITIVES,
    max_current_hard_negatives: int = DEFAULT_MAX_CURRENT_HARD_NEGATIVES,
    max_prior_positives: int = DEFAULT_MAX_PRIOR_POSITIVES,
    max_prior_hard_negatives: int = DEFAULT_MAX_PRIOR_HARD_NEGATIVES,
    prior_positive_decay_after_zero_rounds: int = 2,
    near_miss_top_k: int = DEFAULT_NEAR_MISS_TOP_K,
    near_miss_min_confidence: float = DEFAULT_NEAR_MISS_MIN_CONFIDENCE,
    near_miss_weight: float = DEFAULT_NEAR_MISS_WEIGHT,
    reward: Optional[float] = None,
    rollback: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return current-vs-prior contrastive examples for active learning."""

    hard_negative_iptm_min, hard_negative_threshold_source = hard_negative_iptm_min_from_additional_filters(additional_filters)
    current = extract_round_examples(
        current_candidates,
        round_id=round_id,
        hard_negative_iptm_min=hard_negative_iptm_min,
        max_positive=max_current_positives,
        max_hard_negative=max_current_hard_negatives,
        near_miss_top_k=near_miss_top_k,
        near_miss_min_confidence=near_miss_min_confidence,
        near_miss_weight=near_miss_weight,
    )
    prior = summarize_prior_round_examples(
        prior_rounds or [],
        max_positive=max_prior_positives,
        max_hard_negative=max_prior_hard_negatives,
    )
    cumulative_positive_count = int(current["counts"]["strict_positive"]) + int(prior["counts"]["strict_positive"])
    cumulative_near_miss_count = int(current["counts"]["near_miss"]) + int(prior["counts"]["near_miss"])
    cumulative_other_negative_count = int(current["counts"]["other_negative"]) + int(prior["counts"]["other_negative"])
    consecutive_zero_positives = consecutive_zero_current_positive_rounds(prior.get("by_round") or [], current)
    decay_after = max(1, int(prior_positive_decay_after_zero_rounds or 1))
    prior_positive_weight = 0.0 if consecutive_zero_positives >= decay_after else 1.0
    signal = _learning_signal(current, reward=reward, rollback=rollback)
    return {
        "schema_version": "3.0",
        "purpose": (
            "Contrast strict positives against boundary near misses and other negatives. "
            "Use current_round for immediate diagnosis and prior_rounds for accumulated evidence."
        ),
        "thresholds": threshold_spec(
            hard_negative_iptm_min=hard_negative_iptm_min,
            hard_negative_threshold_source=hard_negative_threshold_source,
        ),
        "current_round": current,
        "learning_signal": signal,
        "prior_rounds": prior,
        "cumulative": {
            "strict_positive_count": cumulative_positive_count,
            "near_miss_count": cumulative_near_miss_count,
            "other_negative_count": cumulative_other_negative_count,
            "contrastive_example_count": cumulative_positive_count + cumulative_near_miss_count + cumulative_other_negative_count,
            "current_strict_positive_count": int(current["counts"]["strict_positive"]),
            "consecutive_zero_current_positive_rounds": consecutive_zero_positives,
            "prior_positive_weight": prior_positive_weight,
            "prior_positive_decay_after_zero_rounds": decay_after,
        },
    }


def extract_round_examples(
    candidates: Sequence[Mapping[str, Any]],
    *,
    round_id: int,
    hard_negative_iptm_min: float = DEFAULT_HARD_NEGATIVE_IPTM_MIN,
    max_positive: int = DEFAULT_MAX_CURRENT_POSITIVES,
    max_hard_negative: int = DEFAULT_MAX_CURRENT_HARD_NEGATIVES,
    near_miss_top_k: int = DEFAULT_NEAR_MISS_TOP_K,
    near_miss_min_confidence: float = DEFAULT_NEAR_MISS_MIN_CONFIDENCE,
    near_miss_weight: float = DEFAULT_NEAR_MISS_WEIGHT,
) -> Dict[str, Any]:
    """Split one round into strict positives, boundary near misses, and others."""

    positives: List[Dict[str, Any]] = []
    non_positives: List[Dict[str, Any]] = []
    iptm_positive_metric_only = 0
    evaluable = 0
    for index, row in enumerate(candidates or []):
        example = compact_contrastive_candidate(row, round_id=round_id, row_index=index)
        if not example:
            continue
        evaluable += 1
        metrics = dict(example.get("metrics") or {})
        margins = confidence_margins(metrics)
        confidence = continuous_confidence(margins)
        item = dict(example, confidence=round(confidence, 6), margins=margins)
        iptm = float(metrics.get("design_to_target_iptm") or 0.0)
        if iptm >= POSITIVE_IPTM_MIN:
            iptm_positive_metric_only += 1
        if min(margins.values()) >= 0.0:
            item.update(label="strict_positive", weight=round(confidence, 6), label_reason="all iPTM/PAE/RMSD/pTM margins pass")
            positives.append(item)
        else:
            non_positives.append(item)

    positives.sort(key=_positive_sort_key, reverse=True)
    non_positives.sort(key=_hard_negative_sort_key, reverse=True)
    guarded = [item for item in non_positives if float(item.get("confidence") or 0.0) >= float(near_miss_min_confidence)]
    near_misses = guarded[:max(0, int(near_miss_top_k))]
    near_ids = {item["_sample_ordinal"] for item in near_misses}
    other_negatives = [item for item in non_positives if item["_sample_ordinal"] not in near_ids]
    for item in near_misses:
        item.update(label="near_miss", weight=round(float(near_miss_weight) * float(item["confidence"]), 6), label_reason="top-k continuous-confidence near miss above guardrail")
    for item in other_negatives:
        item.update(label="other_negative", weight=round(max(0.05, 1.0 - float(item["confidence"])), 6), label_reason="failed strict margins outside guarded near-miss top-k")
    weighted_yield = sum(float(item["weight"]) for item in positives + near_misses) / max(1, evaluable)
    return {
        "round_id": round_id,
        "candidate_count": len(candidates or []),
        "evaluable_candidate_count": evaluable,
        "counts": {
            "strict_positive": len(positives),
            "near_miss": len(near_misses),
            "other_negative": len(other_negatives),
            "hard_negative_iptm_min": float(hard_negative_iptm_min),
            "iptm_ge_0_5_before_quality_gates": iptm_positive_metric_only,
            "iptm_ge_0_5_rejected_by_quality_gates": max(0, iptm_positive_metric_only - len(positives)),
        },
        "margin_weighted_yield": round(weighted_yield, 6),
        "near_miss_top_k": max(0, int(near_miss_top_k)),
        "near_miss_min_confidence": float(near_miss_min_confidence),
        "strict_positive_examples": [_anonymous_example(item) for item in positives[:max(0, int(max_positive))]],
        "near_miss_examples": [_anonymous_example(item) for item in near_misses],
        "other_negative_examples": [_anonymous_example(item) for item in other_negatives[:max(0, int(max_hard_negative))]],
    }




def _anonymous_example(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in dict(item).items() if key not in {"_sample_ordinal", "candidate_id", "source"}}

def summarize_prior_round_examples(
    prior_rounds: Sequence[Mapping[str, Any]],
    *,
    max_positive: int = DEFAULT_MAX_PRIOR_POSITIVES,
    max_hard_negative: int = DEFAULT_MAX_PRIOR_HARD_NEGATIVES,
) -> Dict[str, Any]:
    """Merge already-mined examples from previous rounds without losing round IDs."""

    by_round: List[Dict[str, Any]] = []
    positives: List[Dict[str, Any]] = []
    near_misses: List[Dict[str, Any]] = []
    other_negatives: List[Dict[str, Any]] = []
    positive_count = 0
    near_miss_count = 0
    other_negative_count = 0
    for payload in prior_rounds or []:
        round_payload = _round_payload(payload)
        if not round_payload:
            continue
        counts = dict(round_payload.get("counts") or {})
        strict_items = list(round_payload.get("strict_positive_examples") or round_payload.get("positive_examples") or [])
        near_items = list(round_payload.get("near_miss_examples") or [])
        other_items = list(round_payload.get("other_negative_examples") or round_payload.get("hard_negative_examples") or [])
        pos_count = int(counts.get("strict_positive", counts.get("positive", len(strict_items))) or 0)
        near_count = int(counts.get("near_miss", len(near_items)) or 0)
        neg_count = int(counts.get("other_negative", counts.get("hard_negative", len(other_items))) or 0)
        positive_count += pos_count
        near_miss_count += near_count
        other_negative_count += neg_count
        round_id = round_payload.get("round_id")
        by_round.append({
            "round_id": round_id,
            "strict_positive_count": pos_count,
            "near_miss_count": near_count,
            "other_negative_count": neg_count,
        })
        positives.extend(dict(item, source_round_id=round_id, label="strict_positive") for item in strict_items)
        near_misses.extend(dict(item, source_round_id=round_id, label="near_miss") for item in near_items)
        other_negatives.extend(dict(item, source_round_id=round_id, label="other_negative") for item in other_items)

    positives.sort(key=_positive_sort_key, reverse=True)
    near_misses.sort(key=_hard_negative_sort_key, reverse=True)
    other_negatives.sort(key=_hard_negative_sort_key, reverse=True)
    return {
        "round_count": len(by_round),
        "counts": {
            "strict_positive": positive_count,
            "near_miss": near_miss_count,
            "other_negative": other_negative_count,
        },
        "by_round": by_round,
        "strict_positive_examples": positives[: max(0, int(max_positive))],
        "near_miss_examples": near_misses[: max(0, int(max_hard_negative))],
        "other_negative_examples": other_negatives[: max(0, int(max_hard_negative))],
    }


def consecutive_zero_current_positive_rounds(
    prior_by_round: Sequence[Mapping[str, Any]],
    current_round: Mapping[str, Any],
) -> int:
    """Count trailing rounds whose current-round positive count is zero."""

    counts: List[int] = []
    for item in prior_by_round or []:
        try:
            counts.append(int((item or {}).get("strict_positive_count", (item or {}).get("positive_count", 0)) or 0))
        except (TypeError, ValueError):
            counts.append(0)
    current_counts = dict((current_round or {}).get("counts") or {})
    try:
        counts.append(int(current_counts.get("strict_positive", current_counts.get("positive", 0)) or 0))
    except (TypeError, ValueError):
        counts.append(0)
    streak = 0
    for count in reversed(counts):
        if int(count or 0) > 0:
            break
        streak += 1
    return streak



def confidence_margins(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Signed, normalized margins; zero is the physical quality boundary."""
    iptm = float(metrics.get("design_to_target_iptm") or 0.0)
    pae = float(metrics.get("min_design_to_target_pae") or 100000.0)
    rmsd = float(metrics.get("designfolding_filter_rmsd") or 100000.0)
    ptm = float(metrics.get("design_ptm") or 0.0)
    return {"iptm": round((iptm-POSITIVE_IPTM_MIN)/0.20, 6), "pae": round((POSITIVE_PAE_MAX-pae)/8.0, 6), "rmsd": round((POSITIVE_REFOLD_RMSD_MAX-rmsd)/2.5, 6), "ptm": round((ptm-POSITIVE_PTM_MIN)/0.3, 6)}


def continuous_confidence(margins: Mapping[str, float]) -> float:
    """Smooth confidence avoids a discontinuous 0.5 iPTM label flip."""
    weighted = 0.40*float(margins["iptm"]) + 0.25*float(margins["pae"]) + 0.20*float(margins["rmsd"]) + 0.15*float(margins["ptm"])
    return 1.0 / (1.0 + pow(2.718281828459045, -2.0*weighted))


def _learning_signal(current: Mapping[str, Any], *, reward: Optional[float], rollback: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    action = str((rollback or {}).get("action") or "none")
    factor = 0.5 if action in {"replay_best", "branch_from_best", "stop"} else 1.0
    base = float(current.get("margin_weighted_yield") or 0.0)
    return {"reward": reward, "rollback_action": action, "rollback_factor": factor, "margin_weighted_yield": round(base*factor, 6), "top_k_confidence": [item.get("confidence") for item in current.get("near_miss_examples") or []]}

def compact_contrastive_candidate(row: Mapping[str, Any], *, round_id: int, row_index: int = 0) -> Dict[str, Any]:
    data = dict(row or {})
    if _float_from_keys(data, "design_to_target_iptm", "iptm", "interface_confidence") is None:
        return {}
    metrics = core_metrics_from_row(data)
    raw_metrics = {
        key: data.get(key)
        for key in (
            "id",
            "final_rank",
            "file_name",
            "num_design",
            "pass_iptm_filter",
            "pass_filters",
            "plip_hbonds_refolded",
            "delta_sasa_refolded",
            "ALA_fraction",
            "GLY_fraction",
            "GLU_fraction",
            "LEU_fraction",
            "VAL_fraction",
        )
        if key in data
    }
    return {
        "round_id": round_id,
        "_sample_ordinal": int(row_index),
        "metrics": {
            "design_to_target_iptm": round(float(metrics.get("design_to_target_iptm") or 0.0), 6),
            "min_design_to_target_pae": round(float(metrics.get("min_design_to_target_pae") or 100000.0), 6),
            "design_ptm": round(float(metrics.get("design_ptm") or 0.0), 6),
            "designfolding_filter_rmsd": round(float(metrics.get("designfolding_filter_rmsd") or 100000.0), 6),
            "pae_confidence": round(float(metrics.get("pae_confidence") or 0.0), 6),
            "refold_consistency": round(float(metrics.get("refold_consistency") or 0.0), 6),
            "core_objective": round(float(metrics.get("core_objective") or 0.0), 6),
            "core_rank_key": list(core_rank_key(data)),
        },
        "raw_metrics": raw_metrics,
    }


def prior_examples_from_memory(memory: Any, *, before_round_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Collect active-learning examples from durable memory records."""

    out: List[Dict[str, Any]] = []
    for record in sorted(getattr(memory, "rounds", []) or [], key=lambda item: getattr(item, "round_id", 0)):
        round_id = getattr(record, "round_id", None)
        if before_round_id is not None and round_id is not None and int(round_id) >= int(before_round_id):
            continue
        payload = getattr(record, "active_learning_examples", None)
        if not payload:
            evaluation = getattr(record, "evaluation", None)
            if isinstance(evaluation, Mapping):
                payload = evaluation.get("active_learning_examples")
        if payload:
            out.append(dict(payload))
    return out


def threshold_spec(
    *,
    hard_negative_iptm_min: float = DEFAULT_HARD_NEGATIVE_IPTM_MIN,
    hard_negative_threshold_source: str = "default_no_iptm_additional_filter",
) -> Dict[str, Any]:
    return {
        "strict_positive_examples": {
            "design_to_target_iptm_min": POSITIVE_IPTM_MIN,
            "min_design_to_target_pae_max": POSITIVE_PAE_MAX,
            "design_ptm_min": POSITIVE_PTM_MIN,
            "designfolding_filter_rmsd_max": POSITIVE_REFOLD_RMSD_MAX,
            "rationale": (
                "Unified success requires iPTM>=0.50, PAE<=10A, pTM>=0.70, "
                "and refold RMSD<=2.5A."
            ),
        },
        "near_miss_examples": {
            "design_to_target_iptm_min_exclusive": float(hard_negative_iptm_min),
            "design_to_target_iptm_max_exclusive": POSITIVE_IPTM_MIN,
            "threshold_source": hard_negative_threshold_source,
            "rationale": (
                "Boundary examples are selected by continuous confidence from every non-strict candidate. "
                "The legacy lower iPTM bound is retained as provenance but does not suppress classification."
            ),
        },
        "other_negative_examples": {
            "definition": "All non-strict candidates not selected into the guarded near-miss top-k."
        },
    }


def hard_negative_iptm_min_from_additional_filters(additional_filters: Optional[Sequence[Any]]) -> Tuple[float, str]:
    """Derive the hard-negative lower iPTM bound from user additional_filters.

    Only lower-bound iPTM filters (``iptm>...``, ``iptm>=...`` or dict filters
    with ``lower_is_better=false``) define the near-miss band.  Non-iPTM filters
    and upper-bound iPTM filters intentionally fall back to 0.0.
    """

    thresholds: List[Tuple[float, str]] = []
    for item in additional_filters or []:
        parsed = _parse_iptm_lower_filter(item)
        if parsed is not None:
            thresholds.append(parsed)
    if not thresholds:
        return DEFAULT_HARD_NEGATIVE_IPTM_MIN, "default_no_iptm_additional_filter"
    value, source = max(thresholds, key=lambda pair: pair[0])
    return float(value), source


def _parse_iptm_lower_filter(item: Any) -> Optional[Tuple[float, str]]:
    if isinstance(item, Mapping):
        feature = str(item.get("feature") or "").strip()
        if not _is_iptm_filter_feature(feature):
            return None
        if bool(item.get("lower_is_better")):
            return None
        try:
            return float(item.get("threshold")), f"additional_filters:{feature}>={item.get('threshold')}"
        except (TypeError, ValueError):
            return None
    token = str(item or "").strip()
    match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*(>=|>)\s*([-+]?\d+(?:\.\d+)?)\s*$", token)
    if not match:
        return None
    feature, _op, raw_threshold = match.groups()
    if not _is_iptm_filter_feature(feature):
        return None
    try:
        return float(raw_threshold), f"additional_filters:{token}"
    except ValueError:
        return None


def _is_iptm_filter_feature(feature: str) -> bool:
    normalized = str(feature or "").strip().lower().replace("-", "_")
    return normalized in {"iptm", "design_to_target_iptm"}


def _round_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(payload or {})
    if data.get("current_round"):
        return dict(data.get("current_round") or {})
    return data



def _float_from_keys(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _positive_sort_key(example: Mapping[str, Any]) -> tuple:
    metrics = dict(example.get("metrics") or {})
    return core_rank_key(metrics)


def _hard_negative_sort_key(example: Mapping[str, Any]) -> tuple:
    metrics = dict(example.get("metrics") or {})
    return core_rank_key(metrics)
