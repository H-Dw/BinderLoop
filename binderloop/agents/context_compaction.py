"""Shared prompt-context compaction utilities.

The closed-loop orchestrator assembles a very large ``context`` dict every
round (full ``evaluation``, ``structural_analysis``, ``fragment_templates``,
per-structure ``ca_coordinates``, raw ``metrics``, full ``messages`` history,
etc.).  Passing that whole object verbatim into every LLM agent caused the
request to balloon to several million tokens, tripping the provider's
``HTTP 400`` context-length limit and silently forcing a deterministic
fallback (``llm_used=false``).

This module centralises the logic for trimming that context down to *only the
fields each agent actually needs for its job*.  Every agent imports the
helpers it needs so the trimming rules live in exactly one place and stay
consistent across agents.

Design goals
------------
* **Drop unbounded payloads** – ``ca_coordinates`` (per-residue 3-D coords),
  ``binder_sequence`` blobs inside templates, raw per-candidate ``metrics``
  arrays, and similar large fields are removed entirely; the agents reason
  from scalar summaries, tags and scores, not raw coordinates.
* **Cap list lengths** – summaries / candidates / messages / fragments are
  truncated to small, fixed limits.
* **Keep only task-relevant slices** – each ``compact_context_for_*`` helper
  returns the minimal projection a given agent needs.
"""

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from binderloop.analysis.candidate_clusters import compact_cluster_cards
from binderloop.analysis.core_objective import core_metric_stats, rank_by_core_objective
from binderloop.analysis.scientific_summary import build_scientific_summary
from binderloop.execution_error_summary import sanitize_error_text

# ---------------------------------------------------------------------------
# Tunable caps.  Kept conservative so even a busy round stays well under the
# provider context window.
# ---------------------------------------------------------------------------
MAX_STRUCTURE_SUMMARIES = 15
MAX_TOP_CANDIDATES = 10
MAX_FAILED_EXAMPLES = 8
MAX_ACTIVE_LEARNING_POSITIVES = 6
MAX_ACTIVE_LEARNING_NEAR_MISSES = 4
MAX_ACTIVE_LEARNING_HARD_NEGATIVES = 10
MAX_PRIOR_ACTIVE_LEARNING_POSITIVES = 8
MAX_PRIOR_ACTIVE_LEARNING_NEAR_MISSES = 8
MAX_PRIOR_ACTIVE_LEARNING_HARD_NEGATIVES = 12
MAX_FRAGMENTS_PER_STRUCTURE = 4
MAX_MESSAGES = 30
MAX_HYPOTHESES = 6
MAX_GUIDANCE = 6
MAX_CORRECTIVE_ACTIONS = 6
MAX_RECENT_ROUNDS = 5
MAX_RECALLED_MEMORY_ITEMS = 8

# Quality-agent caps are intentionally tighter than the shared defaults. Live
# round_04 testing showed that the old quality projection spent most of its
# budget on duplicated execution messages and 15 detailed structure summaries.
# A six-structure diverse projection preserved a fact-valid answer while
# reducing the serialized user payload from 451 KB to 85 KB.
QUALITY_MAX_STRUCTURE_SUMMARIES = 6
QUALITY_MAX_FRAGMENTS_PER_CLASS = 2
QUALITY_MAX_TARGET_CONTACTS = 6
QUALITY_MAX_TOP_CANDIDATES = 5
QUALITY_MAX_FAILED_EXAMPLES = 4

# ---------------------------------------------------------------------------
# Hard byte budget for any LLM prompt payload.  The provider enforces a
# context-length limit; the per-agent compactors above usually keep us well
# under it, but a pathological round (huge tag dicts, very long observation
# strings, deeply nested fallbacks) could still blow past it.
# ``enforce_byte_budget`` is the last-resort guard that *guarantees* the
# serialised user payload stays below this cap before it ever reaches the
# HTTP client.
# ---------------------------------------------------------------------------
MAX_PROMPT_BYTES = 1_000_000  # 1 MB hard ceiling for the serialised user payload

# Per-string truncation length used by the progressive compactor.
_STRING_TRUNCATE_LEN = 600
# Hard cap on any list length used by the progressive compactor.
_LIST_TRUNCATE_LEN = 8
# Marker appended to truncated strings so downstream/debugging is obvious.
_TRUNCATE_MARK = "...[truncated]"

# Fields that must never be forwarded to an LLM prompt because they are
# unbounded / per-residue payloads.
_HEAVY_KEYS = frozenset(
    {
        "ca_coordinates",
        "coordinates",
        "coords",
        "atom_coordinates",
        "all_atom_coordinates",
        "binder_sequence",
        "sequence",
        "raw_metrics",
        "per_residue_plddt",
        "pae",
        "pae_matrix",
        "contact_map",
        "distance_matrix",
    }
)


def _strip_heavy(value: Any) -> Any:
    """Recursively drop heavy/unbounded keys from nested mappings/lists."""
    if isinstance(value, Mapping):
        return {k: _strip_heavy(v) for k, v in value.items() if k not in _HEAVY_KEYS}
    if isinstance(value, (list, tuple)):
        return [_strip_heavy(item) for item in value]
    return value


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}




def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_iptm(candidate: Mapping[str, Any]) -> Optional[float]:
    data = _as_dict(candidate)
    raw = _as_dict(data.get("raw"))
    metrics = _as_dict(data.get("metrics"))
    raw_metrics = _as_dict(data.get("raw_metrics"))
    for source in (raw, raw_metrics, metrics, data):
        for key in ("design_to_target_iptm", "iptm", "interface_confidence"):
            value = _float_or_none(source.get(key))
            if value is not None:
                return value
    return None


def _candidate_pass_bool(candidate: Mapping[str, Any], key: str) -> Optional[bool]:
    data = _as_dict(candidate)
    raw = _as_dict(data.get("raw"))
    raw_metrics = _as_dict(data.get("raw_metrics"))
    value = raw.get(key, raw_metrics.get(key, data.get(key)))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def build_metric_facts(
    evaluation: Optional[Mapping[str, Any]],
    *,
    candidates: Optional[Sequence[Mapping[str, Any]]] = None,
    additional_filter_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Build immutable scalar facts for LLM prompts and post-hoc checks.

    The facts intentionally separate three gates that are easy to conflate:
    additional_filter_pass is the user BoltzGen filter such as iptm>0.35;
    boltzgen_pass_filters is BoltzGen's aggregate pass_filters column; and
    harness_success_count is the harness compute-gate success count.
    """
    data = _as_dict(evaluation)
    rows = list(candidates or [])
    if not rows:
        rows = list(data.get("top_candidates") or []) + list(data.get("failed_examples") or [])

    iptm_values = [v for v in (_candidate_iptm(row) for row in rows) if v is not None]
    pass_iptm_values = [_candidate_pass_bool(row, "pass_iptm_filter") for row in rows]
    pass_filter_values = [_candidate_pass_bool(row, "pass_filters") for row in rows]

    filtering = _as_dict(data.get("candidate_filtering"))
    total_candidates = int(data.get("total_candidates") or len(rows) or 0)
    success_count = int(data.get("success_count") or 0)
    facts: Dict[str, Any] = {
        "total_candidates": total_candidates,
        "success_count": success_count,
        "failure_count": data.get("failure_count"),
        "harness_success_count": success_count,
        "gate_definitions": {
            "additional_filter_pass": "candidate-level user BoltzGen additional_filters result, e.g. pass_iptm_filter for iptm>0.35",
            "boltzgen_pass_filters": "candidate-level BoltzGen aggregate pass_filters column",
            "harness_success_count": "EvaluationAgent compute-gate pass count; not equivalent to either BoltzGen filter column",
        },
    }
    if filtering:
        facts["candidate_scope"] = filtering.get("analysis_scope") or "filtered_candidates"
        for key in (
            "filters",
            "input_candidate_count",
            "analysis_candidate_count",
            "rejected_candidate_count",
            "filtering_applied",
            "filtering_reason",
        ):
            if key in filtering:
                facts[key] = filtering.get(key)
    if iptm_values:
        facts.update({
            "best_iptm": round(max(iptm_values), 6),
            "mean_iptm": round(sum(iptm_values) / len(iptm_values), 6),
            "count_iptm_gt_035": sum(1 for v in iptm_values if v > additional_filter_threshold),
            "count_iptm_le_035": sum(1 for v in iptm_values if v <= additional_filter_threshold),
            "iptm_count_source": len(iptm_values),
        })
    scientific = build_scientific_summary(rows, evaluation=data)
    facts["scientific_summary"] = scientific
    metric_map = scientific["metrics"]
    for metric_name, best_key, mean_key in (("min_pae", "best_min_pae", "mean_min_pae"), ("design_ptm", "best_design_ptm", "mean_design_ptm"), ("refold_rmsd", "best_refold_rmsd", "mean_refold_rmsd")):
        summary = metric_map[metric_name]
        if summary["valid_count"]:
            facts[best_key] = summary["best"]
            facts[mean_key] = summary["mean"]
    facts["core_rank_definition"] = (
        "Lexicographic: primary gate pass, worst normalized margin, iPTM descending, "
        "PAE ascending, RMSD ascending. No compensation or secondary tie-break."
    )
    facts["core_objective_definition"] = "Legacy monitoring/display scalar only; not used for new decisions."
    filtering_stats = list(filtering.get("per_filter") or [])
    iptm_filter_stats = None
    for item in filtering_stats:
        row = _as_dict(item)
        if str(row.get("metric") or "").lower() in {"iptm", "design_to_target_iptm"}:
            iptm_filter_stats = row
            break
    if iptm_filter_stats:
        facts["additional_filter_pass"] = {
            "filter": iptm_filter_stats.get("filter"),
            "pass_count": iptm_filter_stats.get("pass_count"),
            "fail_count": iptm_filter_stats.get("fail_count"),
        }
    elif any(v is not None for v in pass_iptm_values):
        facts["additional_filter_pass"] = {
            "filter": f"iptm>{additional_filter_threshold}",
            "pass_count": sum(1 for v in pass_iptm_values if v is True),
            "fail_count": sum(1 for v in pass_iptm_values if v is False),
        }
    elif iptm_values:
        facts["additional_filter_pass"] = {
            "filter": f"iptm>{additional_filter_threshold}",
            "pass_count": sum(1 for v in iptm_values if v > additional_filter_threshold),
            "fail_count": sum(1 for v in iptm_values if v <= additional_filter_threshold),
            "derived_from_metric": True,
        }
    if any(v is not None for v in pass_filter_values):
        facts["boltzgen_pass_filters"] = {
            "pass_count": sum(1 for v in pass_filter_values if v is True),
            "fail_count": sum(1 for v in pass_filter_values if v is False),
        }
    # Preserve gate denominators explicitly. Missing booleans must not silently
    # become failures, so every gate records its observed denominator.
    facts["gate_denominators"] = {
        "harness_compute_gate": total_candidates,
        "additional_filter_observed": sum(v is not None for v in pass_iptm_values) or len(iptm_values),
        "boltzgen_pass_filters_observed": sum(v is not None for v in pass_filter_values),
        "metric_rows_observed": len(rows),
    }
    facts["diagnostic_signals"] = {
        "pose_failure_count": int(_as_dict(data.get("tag_counts")).get("binding_pose_failure") or 0),
        "high_pae_count": int(_as_dict(data.get("tag_counts")).get("primary_gate_high_pae") or _as_dict(data.get("tag_counts")).get("high_pae") or 0),
        "hotspot_miss_count": int(_as_dict(data.get("tag_counts")).get("hotspot_miss") or 0),
        "folding_failure_count": int(_as_dict(data.get("tag_counts")).get("folding_failure") or 0),
    }
    return facts


def compact_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Anonymous representative metric row for aggregate round reasoning."""
    item = _as_dict(_strip_heavy(candidate))
    raw = _as_dict(item.get("raw")) or item
    metrics = _as_dict(item.get("metrics"))
    compact: Dict[str, Any] = {}
    if "total" in item:
        compact["score"] = item.get("total")
    if metrics:
        compact["metrics"] = metrics
    if item.get("tags"):
        compact["tags"] = item.get("tags")
    raw_metrics = {
        key: raw.get(key)
        for key in (
            "design_to_target_iptm", "iptm", "pass_iptm_filter", "pass_filters",
            "min_design_to_target_pae", "design_ptm", "filter_rmsd",
            "designfolding-filter_rmsd", "plip_hbonds_refolded",
            "delta_sasa_refolded", "num_design",
        )
        if key in raw
    }
    if raw_metrics:
        compact["raw_metrics"] = raw_metrics
    return compact


def top_candidates_by_iptm(candidates: Sequence[Mapping[str, Any]], *, limit: int = MAX_TOP_CANDIDATES) -> List[Dict[str, Any]]:
    rows = [row for row in candidates or [] if _candidate_iptm(row) is not None]
    rows.sort(key=lambda row: _candidate_iptm(row) or -1.0, reverse=True)
    return [compact_candidate(row) for row in rows[:limit]]


def top_candidates_by_core(candidates: Sequence[Mapping[str, Any]], *, limit: int = MAX_TOP_CANDIDATES) -> List[Dict[str, Any]]:
    return [compact_candidate(row) for row in rank_by_core_objective(list(candidates or []))[:limit]]


def context_digest(payload: Any) -> str:
    encoded = json.dumps(_strip_heavy(payload), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def fact_check_text_against_metric_facts(text: str, metric_facts: Optional[Mapping[str, Any]]) -> List[str]:
    facts = _as_dict(metric_facts)
    if not text or not facts:
        return []
    lowered = text.lower()
    issues: List[str] = []
    pass_count = int(_as_dict(facts.get("additional_filter_pass")).get("pass_count") or facts.get("count_iptm_gt_035") or 0)
    if pass_count > 0:
        elimination_patterns = (
            "filter eliminated all",
            "eliminated every design",
            "zero passing iptm",
            "zero designs passed the iptm",
            "zero candidates passed the iptm",
            "zero passing iptm",
            "all failing the iptm",
            "all fail the iptm",
            "all candidates fail the iptm",
        )
        if any(pattern in lowered for pattern in elimination_patterns):
            issues.append(f"LLM claimed the iptm additional filter eliminated all candidates, but metric_facts show {pass_count} candidate(s) passed it.")
    best = _float_or_none(facts.get("best_iptm"))
    if best is not None:
        for match in re.finditer(r"(?:best|max(?:imum)?)\s+(?:observed\s+)?(?:design[-_ ]to[-_ ]target\s+)?i?ptm[^0-9]{0,24}([0-9]+(?:\.[0-9]+)?)", lowered):
            claimed = _float_or_none(match.group(1))
            if claimed is not None and abs(claimed - best) > 0.05:
                issues.append(f"LLM claimed best/max iPTM {claimed:.3f}, but metric_facts best_iptm is {best:.3f}.")
                break
    return issues

# ---------------------------------------------------------------------------
# Per-component compactors
# ---------------------------------------------------------------------------
def compact_evaluation(evaluation: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep scalar facts + representative candidates with explicit ranking semantics."""
    data = _as_dict(evaluation)
    top_by_score = [compact_candidate(c) for c in (data.get("top_by_score") or data.get("top_candidates") or [])[:MAX_TOP_CANDIDATES]]
    top_by_core = [compact_candidate(c) for c in (data.get("top_by_core") or [])[:MAX_TOP_CANDIDATES]]
    top_by_iptm = [compact_candidate(c) for c in (data.get("top_by_iptm") or [])[:MAX_TOP_CANDIDATES]]
    failed = [compact_candidate(c) for c in (data.get("failed_examples") or [])[:MAX_FAILED_EXAMPLES]]
    out = {
        "total_candidates": data.get("total_candidates"),
        "success_count": data.get("success_count"),
        "failure_count": data.get("failure_count"),
        "tag_counts": data.get("tag_counts"),
        "observations": data.get("observations"),
        "candidate_filtering": data.get("candidate_filtering"),
        "metric_facts": data.get("metric_facts") or build_metric_facts(data),
        "core_metric_trends": data.get("core_metric_trends"),
        "core_metric_stats": data.get("core_metric_stats"),
        "pressure_conflict": data.get("pressure_conflict"),
        "top_by_score": top_by_score,
        "top_by_core": top_by_core,
        "top_by_iptm": top_by_iptm,
        "top_candidates": top_by_core or top_by_iptm or top_by_score,
        "failed_examples": failed,
    }
    active_examples = compact_active_learning_examples(data.get("active_learning_examples"))
    if active_examples:
        out["active_learning_examples"] = active_examples
    return out


def compact_active_learning_examples(examples: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep all three evidence classes, while reading legacy two-class payloads."""
    data = _as_dict(examples)
    if not data:
        return {}
    current = _as_dict(data.get("current_round"))
    prior = _as_dict(data.get("prior_rounds"))
    out: Dict[str, Any] = {
        "schema_version": data.get("schema_version"),
        "thresholds": data.get("thresholds"),
        "cumulative": _three_class_cumulative(data.get("cumulative")),
    }
    if current:
        strict = current.get("strict_positive_examples") or current.get("positive_examples") or []
        near = current.get("near_miss_examples") or []
        other = current.get("other_negative_examples") or current.get("hard_negative_examples") or []
        out["current_round"] = {
            "round_id": current.get("round_id"),
            "candidate_count": current.get("candidate_count"),
            "evaluable_candidate_count": current.get("evaluable_candidate_count"),
            "counts": _three_class_counts(current, strict, near, other),
            "strict_positive_examples": [
                _compact_contrastive_example(item, label="strict_positive")
                for item in strict[:MAX_ACTIVE_LEARNING_POSITIVES]
            ],
            "near_miss_examples": [
                _compact_contrastive_example(item, label="near_miss")
                for item in near[:MAX_ACTIVE_LEARNING_NEAR_MISSES]
            ],
            "other_negative_examples": [
                _compact_contrastive_example(item, label="other_negative")
                for item in other[:MAX_ACTIVE_LEARNING_HARD_NEGATIVES]
            ],
        }
    if prior:
        strict = prior.get("strict_positive_examples") or prior.get("positive_examples") or []
        near = prior.get("near_miss_examples") or []
        other = prior.get("other_negative_examples") or prior.get("hard_negative_examples") or []
        out["prior_rounds"] = {
            "round_count": prior.get("round_count"),
            "counts": _three_class_counts(prior, strict, near, other),
            "by_round": list(prior.get("by_round") or [])[-MAX_RECENT_ROUNDS:],
            "strict_positive_examples": [
                _compact_contrastive_example(item, label="strict_positive")
                for item in strict[:MAX_PRIOR_ACTIVE_LEARNING_POSITIVES]
            ],
            "near_miss_examples": [
                _compact_contrastive_example(item, label="near_miss")
                for item in near[:MAX_PRIOR_ACTIVE_LEARNING_NEAR_MISSES]
            ],
            "other_negative_examples": [
                _compact_contrastive_example(item, label="other_negative")
                for item in other[:MAX_PRIOR_ACTIVE_LEARNING_HARD_NEGATIVES]
            ],
        }
    return out


def _three_class_counts(
    payload: Mapping[str, Any],
    strict: Sequence[Any],
    near: Sequence[Any],
    other: Sequence[Any],
) -> Dict[str, Any]:
    counts = _as_dict(payload.get("counts"))
    preserved = {
        key: value
        for key, value in counts.items()
        if key not in {"positive", "hard_negative", "strict_positive", "near_miss", "other_negative"}
    }
    return {
        "strict_positive": int(counts.get("strict_positive", counts.get("positive", len(strict))) or 0),
        "near_miss": int(counts.get("near_miss", len(near)) or 0),
        "other_negative": int(counts.get("other_negative", counts.get("hard_negative", len(other))) or 0),
        **preserved,
    }


def _three_class_cumulative(value: Any) -> Dict[str, Any]:
    data = _as_dict(value)
    if not data:
        return {}
    out = {
        key: item
        for key, item in data.items()
        if key not in {
            "positive_count",
            "hard_negative_count",
            "current_positive_count",
        }
    }
    out.setdefault("strict_positive_count", int(data.get("positive_count") or 0))
    out.setdefault("near_miss_count", 0)
    out.setdefault("other_negative_count", int(data.get("hard_negative_count") or 0))
    out.setdefault("current_strict_positive_count", int(data.get("current_positive_count") or 0))
    return out


def _compact_contrastive_example(
    example: Mapping[str, Any],
    *,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    item = _as_dict(example)
    raw = _as_dict(item.get("raw_metrics"))
    return {
        key: value
        for key, value in {
            "round_id": item.get("round_id"),
            "source_round_id": item.get("source_round_id"),
            "label": label or item.get("label"),
            "label_reason": item.get("label_reason"),
            "metrics": item.get("metrics"),
            "raw_metrics": {
                k: raw.get(k)
                for k in (
                    "id",
                    "final_rank",
                    "num_design",
                    "pass_iptm_filter",
                    "pass_filters",
                    "plip_hbonds_refolded",
                    "delta_sasa_refolded",
                )
                if k in raw
            },
        }.items()
        if value not in (None, {}, [])
    }


def compact_structure_summary(summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Project a single structural summary to scalar tags/scores (no coords)."""
    item = _as_dict(summary)
    return {
        "candidate_id": item.get("candidate_id") or item.get("id"),
        "binder_chain": item.get("binder_chain"),
        "target_chains": item.get("target_chains"),
        "chain_detection_note": item.get("chain_detection_note"),
        "reliability_score": item.get("reliability_score"),
        "reliability_tags": item.get("reliability_tags"),
        "interface_contact_count": item.get("interface_contact_count"),
        "interface_residue_count": item.get("interface_residue_count"),
        "hotspot_contacts": item.get("hotspot_contacts"),
        "clash_density": item.get("clash_density"),
        "interface_hydrophobic_fraction": item.get("interface_hydrophobic_fraction"),
        "interface_polar_fraction": item.get("interface_polar_fraction"),
        "high_quality_fragments": [
            _strip_heavy(f) for f in (item.get("high_quality_fragments") or [])[:MAX_FRAGMENTS_PER_STRUCTURE]
        ],
        "low_quality_fragments": [
            _strip_heavy(f) for f in (item.get("low_quality_fragments") or [])[:MAX_FRAGMENTS_PER_STRUCTURE]
        ],
        "target_contact_residues": [
            {
                "target_residue": _as_dict(c).get("target_residue"),
                "min_distance": _as_dict(c).get("min_distance"),
                "contact_type": _as_dict(c).get("contact_type"),
            }
            for c in (item.get("contacts_preview") or [])[:12]
        ],
    }


def compact_structural_aggregate_from_object(value: Any) -> Dict[str, Any]:
    """Shallow aggregate projection for mappings or dataclass-like objects.

    This intentionally never reads ``summaries`` and never calls ``asdict``.
    """
    fields = (
        "total_structures", "aggregate_tags", "reliable_seed_fraction",
        "observations", "interface_data_quality",
    )
    if isinstance(value, Mapping):
        getter = value.get
    else:
        getter = lambda key, default=None: getattr(value, key, default)
    return {key: _strip_heavy(getter(key)) for key in fields if getter(key) is not None}


def compact_structural_analysis(
    structural: Optional[Mapping[str, Any]],
    *,
    include_summaries: bool = True,
    max_summaries: int = MAX_STRUCTURE_SUMMARIES,
) -> Dict[str, Any]:
    """Aggregate-level structural analysis, optionally with capped summaries."""
    data = _as_dict(structural)
    summaries_for_mapping = list(data.get("summaries") or [])
    first_summary = _as_dict(summaries_for_mapping[0]) if summaries_for_mapping else {}
    out: Dict[str, Any] = {
        "total_structures": data.get("total_structures"),
        "aggregate_tags": data.get("aggregate_tags"),
        "reliable_seed_fraction": data.get("reliable_seed_fraction"),
        "observations": data.get("observations"),
        "interface_data_quality": data.get("interface_data_quality"),
    }
    if first_summary:
        out["output_chain_mapping"] = {
            "binder_chain": first_summary.get("binder_chain"),
            "target_chains": first_summary.get("target_chains"),
            "chain_detection_note": first_summary.get("chain_detection_note"),
            "namespace_note": (
                "BoltzGen output chains are entity-order labels, commonly A=binder and B=target; "
                "configured target/hotspot chain IDs refer to the input design spec."
            ),
        }
    if include_summaries:
        summaries = summaries_for_mapping[:max_summaries]
        out["summaries"] = [compact_structure_summary(s) for s in summaries]
    return out


def compact_fragment_templates(templates: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Drop the heavy ``ca_coordinates``/``binder_sequence`` blobs from templates.

    The LLM only needs to know that templates exist, their counts and quality,
    and the structure-redesign source path / metadata — not the raw geometry.
    """
    data = _as_dict(templates)
    if not data:
        return {}
    out: Dict[str, Any] = {}
    for key in (
        "total_templates",
        "high_quality_count",
        "low_quality_count",
        "mean_quality",
        "observations",
    ):
        if key in data:
            out[key] = data.get(key)
    redesign = _as_dict(data.get("structure_redesign") or data.get("binder_template"))
    if redesign:
        out["structure_redesign"] = {
            "source_structure_file": redesign.get("source_structure_file"),
            "quality_score": redesign.get("quality_score"),
            "quality_rank": redesign.get("quality_rank"),
            "fragment_id": redesign.get("fragment_id"),
            "within_proximity": redesign.get("within_proximity"),
            "residue_span": redesign.get("residue_span") or redesign.get("residues"),
        }
    return _strip_heavy(out)


def compact_messages(messages: Optional[Sequence[Any]], *, limit: int = MAX_MESSAGES) -> List[Any]:
    return [_strip_heavy(m) for m in list(messages or [])[-limit:]]


def compact_memory(memory: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep indexed recalled evidence first, with recent-round compatibility.

    When runtime.extend_memory is enabled, ExperimentMemoryStore annotates each
    job with a small params_summary.  Preserve only that tiny allowlist so the
    LLM can relate length/hotspot/template settings to outcomes without pulling
    full per-job params into the prompt.
    """
    data = _as_dict(memory)
    if not data:
        return {}
    extend_memory = bool(data.get("extend_memory"))
    out = {
        key: value
        for key, value in data.items()
        if key not in {"recent_rounds", "recalled_items", "recent_messages", "retrieval"}
    }
    recalled = [
        _compact_memory_item(item)
        for item in list(data.get("recalled_items") or [])[:MAX_RECALLED_MEMORY_ITEMS]
    ]
    recalled = [item for item in recalled if item]
    if recalled:
        out["recalled_items"] = recalled
    retrieval = _as_dict(data.get("retrieval"))
    if retrieval:
        out["retrieval"] = {
            "structured_candidate_count": retrieval.get("structured_candidate_count"),
            "selected_count": retrieval.get("selected_count"),
            "semantic_rerank_used": retrieval.get("semantic_rerank_used"),
            "retrieval_mode": retrieval.get("retrieval_mode") or ("semantic_opt_in" if retrieval.get("semantic_rerank_used") else "deterministic_structured_mmr"),
            "cache_hit": retrieval.get("cache_hit"),
            "query": _strip_heavy(retrieval.get("query")),
            "selected_scores": _strip_heavy(retrieval.get("selected_scores")),
        }
    # Recalled cards are the bounded primary path. Raw recent rounds remain the
    # compatibility fallback only when retrieval produced nothing.
    recent = [] if recalled else list(data.get("recent_rounds") or [])[-MAX_RECENT_ROUNDS:]
    compact_recent: List[Dict[str, Any]] = []
    for record in recent:
        rec = _as_dict(record)
        evaluation = _as_dict(rec.get("evaluation"))
        active_example_counts = (
            _as_dict(_as_dict(evaluation.get("active_learning_examples")).get("current_round")).get("counts")
            or _as_dict(_as_dict(rec.get("active_learning_examples")).get("current_round")).get("counts")
        )
        compact_recent.append(
            {
                "round_id": rec.get("round_id"),
                "evaluation": {
                    "total_candidates": evaluation.get("total_candidates"),
                    "success_count": evaluation.get("success_count"),
                    "tag_counts": evaluation.get("tag_counts"),
                    "metric_facts": evaluation.get("metric_facts") or build_metric_facts(evaluation),
                    "hotspot_coverage": evaluation.get("hotspot_coverage"),
                    "foldability": evaluation.get("foldability"),
                    "active_learning_example_counts": active_example_counts,
                },
                "execution_parameters": compact_config(rec.get("config_snapshot") or rec.get("execution_parameters")),
                "merge_overrides": _strip_heavy(rec.get("config_merge_report") or rec.get("merge_overrides")),
                "rollback_lineage": _strip_heavy(rec.get("rollback_decision") or rec.get("rollback")),
                "jobs": [
                    _compact_memory_job(j, include_params_summary=extend_memory)
                    for j in (rec.get("jobs") or [])
                ],
            }
        )
    if compact_recent:
        out["recent_rounds"] = compact_recent
    return _strip_heavy(out)


def _compact_memory_item(value: Any) -> Dict[str, Any]:
    item = _as_dict(value)
    if not item or item.get("archived"):
        return {}
    performance = _as_dict(item.get("performance"))
    return {
        key: candidate
        for key, candidate in {
            "item_id": item.get("item_id"),
            "round_id": item.get("round_id"),
            "item_type": item.get("item_type"),
            "target_key": item.get("target_key"),
            "failure_tags": list(item.get("failure_tags") or [])[:12],
            "parameter_diff": _strip_heavy(item.get("parameter_diff")),
            "reward": item.get("reward"),
            "reward_delta": item.get("reward_delta"),
            "performance": {
                key: performance.get(key)
                for key in (
                    "best_iptm",
                    "median_iptm",
                    "core_objective",
                    "success_count",
                    "reward_min",
                    "reward_max",
                    "source_count",
                    "execution_failure_reason",
                )
                if performance.get(key) is not None
            },
            "summary": _truncate_str(item.get("summary"), limit=1200),
            "source_round_ids": list(item.get("source_round_ids") or [])[:20],
            "compression_level": item.get("compression_level"),
        }.items()
        if candidate not in (None, "", [], {})
    }



def _compact_memory_job(job: Any, *, include_params_summary: bool = False) -> Dict[str, Any]:
    item = _as_dict(job)
    params = _as_dict(item.get("params"))
    out = {
        "binder_length": item.get("binder_length"),
        "status": item.get("status"),
    }
    if include_params_summary:
        summary = _as_dict(item.get("params_summary")) or _params_summary_from_params(params)
        if summary:
            out["params_summary"] = summary
    return out


def _params_summary_from_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "binder_lengths",
        "hotspot_weight",
        "prioritize_hotspots",
        "auxiliary_hotspots",
        "template_conditioned",
        "template_free_exploration",
        "diffusion_batch_size",
        "alpha",
        "noise_scale",
        "step_scale",
        "num_designs",
    )
    return {key: params.get(key) for key in keys if key in params and params.get(key) is not None}

def compact_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep only the tunable knobs relevant to configuration reasoning."""
    data = _as_dict(config)
    keys = (
        "binder_lengths",
        "binder_length_range",
        "hotspots",
        "hotspot_weight",
        "auxiliary_hotspots",
        "prioritize_hotspots",
        "num_designs",
        "num_designs_per_round",
        "max_binders_per_round",
        "budget",
        "alpha",
        "noise_scale",
        "step_scale",
        "diffusion_batch_size",
        "inverse_fold_num_sequences",
        "refolding_rmsd_threshold",
        "exploration_ratio",
        "additional_filters",
        "run_filtering",
        "config_overrides",
        "epitope_crop_mode",
        "allow_agent_epitope_crop",
        "target_include",
        "target_binding_types",
        "original_target_include",
        "original_target_binding_types",
        "protocol",
        "binder_chain",
        "top_k",
        "max_rounds",
    )
    return {k: data.get(k) for k in keys if k in data}


def compact_target_profile(profile: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep current-task target facts without embedding target knowledge in SYSTEM prompts."""
    data = _as_dict(profile)
    keys = (
        "target_name",
        "structure_path",
        "primary_chain_id",
        "target_chains",
        "target_include",
        "target_binding_types",
        "hotspots",
        "notes",
        "profile",
        "structure_groups",
        "source",
    )
    return {k: _strip_heavy(data.get(k)) for k in keys if k in data and data.get(k) is not None}


def compact_hypotheses(hypotheses: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for hyp in list(hypotheses or [])[:MAX_HYPOTHESES]:
        item = _as_dict(hyp)
        out.append(
            {
                "name": item.get("name"),
                "confidence": item.get("confidence"),
                "intervention": item.get("intervention"),
                "config_parameter_changes": item.get("config_parameter_changes"),
                "expected_signal_next_round": item.get("expected_signal_next_round"),
            }
        )
    return out


def compact_quality_analysis(quality: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    data = _as_dict(quality)
    if not data:
        return {}
    return {
        "overall_assessment": data.get("overall_assessment"),
        "causal_factors": list(data.get("causal_factors") or [])[:MAX_GUIDANCE],
        "next_round_guidance": list(data.get("next_round_guidance") or [])[:MAX_GUIDANCE],
    }


def compact_diagnostic_report(report: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    data = _as_dict(report)
    if not data:
        return {}
    return {
        "status_diagnosis": data.get("status_diagnosis"),
        "metric_interpretation": data.get("metric_interpretation"),
        "corrective_actions": list(data.get("corrective_actions") or [])[:MAX_CORRECTIVE_ACTIONS],
        "pipeline_health": data.get("pipeline_health"),
    }


def _quality_compact_evaluation(evaluation: Any) -> Dict[str, Any]:
    """Remove duplicated rankings/examples from the quality evaluation slice."""
    result = compact_evaluation(_as_dict(evaluation))
    # The same contrastive examples are already supplied as a top-level quality
    # section. Keeping them here as well added ~26 KB in round_04.
    result.pop("active_learning_examples", None)
    preferred = (
        result.get("top_candidates")
        or result.get("top_by_core")
        or result.get("top_by_iptm")
        or result.get("top_by_score")
        or []
    )
    for key in ("top_by_score", "top_by_core", "top_by_iptm"):
        result.pop(key, None)
    result["top_candidates"] = list(preferred)[:QUALITY_MAX_TOP_CANDIDATES]
    result["failed_examples"] = list(
        result.get("failed_examples") or []
    )[:QUALITY_MAX_FAILED_EXAMPLES]
    return result


def _quality_compact_active_examples(examples: Any) -> Dict[str, Any]:
    result = compact_active_learning_examples(_as_dict(examples))
    current = _as_dict(result.get("current_round"))
    if current:
        current["strict_positive_examples"] = list(
            current.get("strict_positive_examples") or []
        )[:4]
        current["near_miss_examples"] = list(
            current.get("near_miss_examples") or []
        )[:4]
        current["other_negative_examples"] = list(
            current.get("other_negative_examples") or []
        )[:5]
        result["current_round"] = current
    prior = _as_dict(result.get("prior_rounds"))
    if prior:
        prior["by_round"] = list(prior.get("by_round") or [])[-3:]
        prior["strict_positive_examples"] = list(
            prior.get("strict_positive_examples") or []
        )[:4]
        prior["near_miss_examples"] = list(
            prior.get("near_miss_examples") or []
        )[:4]
        prior["other_negative_examples"] = list(
            prior.get("other_negative_examples") or []
        )[:4]
        result["prior_rounds"] = prior
    return result


def _quality_structure_tokens(item: Mapping[str, Any]) -> set:
    tokens = {str(value) for value in (item.get("reliability_tags") or [])}
    tokens.update(
        str(key)
        for key, value in _as_dict(item.get("hotspot_contacts")).items()
        if value
    )
    for key in ("high_quality_fragments", "low_quality_fragments"):
        for fragment in item.get(key) or []:
            tokens.update(
                str(reason) for reason in (_as_dict(fragment).get("reasons") or [])
            )
    return tokens


def _quality_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def _quality_structure_score(item: Mapping[str, Any]) -> float:
    reliability = float(item.get("reliability_score") or 0.0)
    fragments = list(item.get("high_quality_fragments") or [])
    fragments += list(item.get("low_quality_fragments") or [])
    best_fragment = max(
        (
            float(_as_dict(fragment).get("quality_score") or 0.0)
            for fragment in fragments
        ),
        default=0.0,
    )
    contacts = min(
        1.0,
        float(item.get("interface_contact_count") or 0.0) / 30.0,
    )
    return 0.45 * reliability + 0.35 * best_fragment + 0.2 * contacts


def _fragment_rank_key(value: Any) -> tuple:
    item = _as_dict(value)
    rank = item.get("quality_rank")
    if isinstance(rank, (list, tuple)) and rank:
        try:
            return tuple(float(part) for part in rank)
        except (TypeError, ValueError):
            pass
    return (1.0 if item.get("quality_label") == "high" else 0.0, float(item.get("quality_score") or 0.0))


def _quality_diverse_structures(
    summaries: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Select strong but non-redundant structures with deterministic MMR."""
    remaining = [dict(item) for item in summaries]
    selected: List[Dict[str, Any]] = []
    while remaining and len(selected) < QUALITY_MAX_STRUCTURE_SUMMARIES:
        def rank(item: Mapping[str, Any]):
            relevance = _quality_structure_score(item)
            redundancy = max(
                (
                    _quality_jaccard(
                        _quality_structure_tokens(item),
                        _quality_structure_tokens(chosen),
                    )
                    for chosen in selected
                ),
                default=0.0,
            )
            return 0.7 * relevance - 0.3 * redundancy, relevance

        chosen = max(remaining, key=rank)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _quality_prune_structure(item: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(item)
    # File paths/names are transport metadata, not LLM evidence.
    row.pop("structure_file", None)
    row.pop("source_structure_file", None)
    row.pop("filename", None)
    row.pop("path", None)
    row["high_quality_fragments"] = sorted(
        list(row.get("high_quality_fragments") or []),
        key=_fragment_rank_key,
        reverse=True,
    )[:QUALITY_MAX_FRAGMENTS_PER_CLASS]
    row["low_quality_fragments"] = sorted(
        list(row.get("low_quality_fragments") or []),
        key=_fragment_rank_key,
    )[:QUALITY_MAX_FRAGMENTS_PER_CLASS]
    row["target_contact_residues"] = list(
        row.get("target_contact_residues") or []
    )[:QUALITY_MAX_TARGET_CONTACTS]
    return row


def _quality_compact_structural_analysis(structural: Any) -> Dict[str, Any]:
    result = compact_structural_analysis(
        _as_dict(structural),
        include_summaries=True,
        max_summaries=MAX_STRUCTURE_SUMMARIES,
    )
    summaries = list(result.get("summaries") or [])
    # Guarantee role-layer coverage before filling remaining slots with MMR.
    # This prevents a numerically strong cluster from crowding out successes or
    # a decision-relevant failure mode.
    selected: List[Dict[str, Any]] = []
    layers = ("strict_positive", "harness_success", "high_pae", "hotspot",
              "foldability", "clash", "filtering", "pose")
    for layer in layers:
        candidates = [dict(item) for item in summaries
                      if layer in " ".join(map(str, [item.get("reliability_tags"),
                                                    item.get("failure_tags"),
                                                    item.get("classification")])).lower()]
        if candidates and len(selected) < QUALITY_MAX_STRUCTURE_SUMMARIES:
            chosen = max(candidates, key=_quality_structure_score)
            if chosen not in selected:
                selected.append(chosen)
    remainder = [item for item in summaries if dict(item) not in selected]
    if len(selected) < QUALITY_MAX_STRUCTURE_SUMMARIES:
        selected.extend(_quality_diverse_structures(remainder)[
            :QUALITY_MAX_STRUCTURE_SUMMARIES-len(selected)])
    result["summaries"] = [_quality_prune_structure(item) for item in selected]
    return result


def _quality_compact_messages(messages: Any) -> List[Dict[str, Any]]:
    """Keep status facts, not duplicated execution records or submit specs."""
    result: List[Dict[str, Any]] = []
    for raw in list(messages or [])[-MAX_MESSAGES:]:
        message = _as_dict(raw)
        content = _as_dict(message.get("content"))
        result.append({
            "sender": message.get("sender"),
            "message_type": message.get("message_type"),
            "round_id": message.get("round_id"),
            "content": {
                key: _strip_heavy(content.get(key))
                for key in (
                    "event",
                    "status",
                    "attempts",
                    "error",
                    "action",
                    "best_round",
                    "best_reward",
                    "current_reward",
                    "relative_drop",
                )
                if content.get(key) is not None
            },
        })
    return result


def _quality_compact_skills(skills: Any) -> List[Dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "type": item.get("type"),
            "priority": item.get("priority"),
            "origin": item.get("origin"),
            "trigger_reason": item.get("trigger_reason"),
            "guidance": list(item.get("guidance") or [])[:3],
            "role_metadata": _strip_heavy(item.get("role_metadata")),
            "learned_rules": list(item.get("learned_rules") or [])[:6],
            "allowed_config_keys": list(item.get("allowed_config_keys") or []),
            "required_inputs": list(item.get("required_inputs") or [])[:6],
            "deterministic_controls": _strip_heavy(item.get("deterministic_controls")),
        }
        for item in (_as_dict(value) for value in list(skills or []))
    ]


def _prompt_cluster_cards(context: Mapping[str, Any]) -> Dict[str, Any]:
    cards = compact_cluster_cards(
        context.get("candidate_clusters") or context.get("candidates.clusters")
    )
    return cards if cards.get("clusters") else {}


# ---------------------------------------------------------------------------
# Per-agent context builders.  Each returns the minimal projection the agent
# needs for its specific task.
# ---------------------------------------------------------------------------
def compact_context_for_hypothesis(context: Mapping[str, Any]) -> Dict[str, Any]:
    """HypothesisAgent: needs failure evidence + structural tags to explain *why*.

    It does NOT need raw coordinates, fragment templates, or full message logs.
    Quality/manager conclusions are intentionally omitted: Hypothesis runs in the
    same dependency wave as quality specialists and must not wait on them.
    Uses the Quality evaluation/structure projection so ranking lists and raw
    structure summaries are not duplicated.
    """
    data = _as_dict(context)
    active_examples = (
        data.get("active_learning_examples")
        or _as_dict(data.get("evaluation")).get("active_learning_examples")
    )
    out: Dict[str, Any] = {
        "round_id": data.get("round_id"),
        "evaluation": _quality_compact_evaluation(data.get("evaluation")),
        "active_learning_examples": _quality_compact_active_examples(active_examples),
        "structural_analysis": _quality_compact_structural_analysis(
            data.get("structural_analysis")
        ),
        "current_config": compact_config(data.get("current_config") or data.get("config")),
        "active_skills": _quality_compact_skills(data.get("active_skills")),
    }
    clusters = _prompt_cluster_cards(data)
    if clusters:
        out["candidate_clusters"] = clusters
        evaluation = dict(out["evaluation"])
        evaluation.pop("top_candidates", None)
        evaluation.pop("failed_examples", None)
        out["evaluation"] = evaluation
        examples = dict(out.get("active_learning_examples") or {})
        current = dict(examples.get("current_round") or {})
        if current:
            examples["current_round"] = {
                key: current.get(key)
                for key in ("counts", "thresholds", "near_miss_definition")
                if current.get(key) not in (None, {}, [])
            }
        out["active_learning_examples"] = examples
    return out


def compact_context_for_quality(context: Mapping[str, Any]) -> Dict[str, Any]:
    """BinderQualityAnalysisAgent: needs per-fragment quality detail + metrics.

    The projection deliberately removes duplicated active examples/rankings and
    verbose execution records, then selects diverse structural evidence. This
    keeps the request below the provider's practical processing-time ceiling,
    not merely its nominal context window.
    """
    data = _as_dict(context)
    active_examples = (
        data.get("active_learning_examples")
        or _as_dict(data.get("evaluation")).get("active_learning_examples")
    )
    out = {
        "evaluation": _quality_compact_evaluation(data.get("evaluation")),
        "active_learning_examples": _quality_compact_active_examples(active_examples),
        "structural_analysis": _quality_compact_structural_analysis(
            data.get("structural_analysis")
        ),
        "target_analysis": _strip_heavy(data.get("target_analysis")),
        "current_config": compact_config(data.get("current_config") or data.get("config")),
        "constraints": _strip_heavy(data.get("constraints")),
        "active_skills": _quality_compact_skills(data.get("active_skills")),
    }
    clusters = _prompt_cluster_cards(data)
    if clusters:
        out["candidate_clusters"] = clusters
    return out


def compact_context_for_diagnostic(
    *,
    round_id: int,
    monitor_snapshot: Optional[Mapping[str, Any]],
    metrics_summary: Optional[Mapping[str, Any]],
    evaluation_summary: Optional[Mapping[str, Any]],
    structural_analysis: Optional[Mapping[str, Any]],
    job_history: Optional[Sequence[Mapping[str, Any]]],
    config: Optional[Mapping[str, Any]],
    active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
    candidate_clusters: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """DiagnosticCoachAgent: execution status + aggregate metrics, no fragments.

    Quality conclusions are not an input; this agent shares Wave A with quality
    specialists and must not wait on the manager.
    """
    ctx: Dict[str, Any] = {"round_id": round_id}
    monitor = _as_dict(monitor_snapshot)
    if monitor:
        ctx["monitor"] = {
            "schema_version": monitor.get("schema_version"),
            "execution_error_summary": monitor.get("execution_error_summary"),
            "state": monitor.get("state"),
            "is_terminal": monitor.get("is_terminal"),
            "is_success": monitor.get("is_success"),
            "status_counts": monitor.get("status_counts"),
            "retried_jobs": monitor.get("retried_jobs"),
            "failed_jobs": monitor.get("failed_jobs"),
            "failure_hints": monitor.get("failure_hints"),
            "missing_outputs": monitor.get("missing_outputs"),
            "execution_parameters": _strip_heavy(monitor.get("execution_parameters") or monitor.get("submit_parameters")),
            "merge_overrides": _strip_heavy(monitor.get("merge_overrides") or monitor.get("config_merge_report")),
            "rollback_lineage": _strip_heavy(monitor.get("rollback_lineage") or monitor.get("rollback")),
        }
    if metrics_summary:
        ctx["metrics_summary"] = _strip_heavy(metrics_summary)
    if evaluation_summary:
        eval_data = _as_dict(evaluation_summary)
        ctx["evaluation"] = {
            "total_candidates": eval_data.get("total_candidates"),
            "success_count": eval_data.get("success_count"),
            "failure_count": eval_data.get("failure_count"),
            "tag_counts": eval_data.get("tag_counts"),
            "observations": eval_data.get("observations"),
            "candidate_filtering": eval_data.get("candidate_filtering"),
            "metric_facts": eval_data.get("metric_facts") or build_metric_facts(eval_data),
            "core_metric_trends": eval_data.get("core_metric_trends"),
            "core_metric_stats": eval_data.get("core_metric_stats"),
            "pressure_conflict": eval_data.get("pressure_conflict"),
            "hotspot_coverage": eval_data.get("hotspot_coverage"),
            "foldability": eval_data.get("foldability"),
            "gate_denominators": eval_data.get("gate_denominators") or _as_dict(eval_data.get("metric_facts")).get("gate_denominators"),
            "top_by_core": [compact_candidate(c) for c in (eval_data.get("top_by_core") or [])[:5]],
            "top_by_score": [compact_candidate(c) for c in (eval_data.get("top_by_score") or eval_data.get("top_candidates") or [])[:5]],
            "top_by_iptm": [compact_candidate(c) for c in (eval_data.get("top_by_iptm") or [])[:5]],
        }
        active_examples = compact_active_learning_examples(eval_data.get("active_learning_examples"))
        if active_examples:
            ctx["active_learning_examples"] = active_examples
    if structural_analysis:
        ctx["structural_analysis"] = compact_structural_analysis(
            structural_analysis, include_summaries=False
        )
    if job_history:
        ctx["job_history"] = [
            {
                "round_id": _as_dict(j).get("round_id"),
                "state": _as_dict(j).get("state") or _as_dict(j).get("status"),
                "execution_parameters": compact_config(_as_dict(j).get("config_snapshot") or _as_dict(j).get("execution_parameters")),
                "merge_overrides": _strip_heavy(_as_dict(j).get("config_merge_report") or _as_dict(j).get("merge_overrides")),
                "rollback_lineage": _strip_heavy(_as_dict(j).get("rollback_decision") or _as_dict(j).get("rollback")),
            }
            for j in list(job_history)[-MAX_RECENT_ROUNDS:]
        ]
    if config:
        ctx["config"] = compact_config(config)
    clusters = compact_cluster_cards(
        candidate_clusters
        or _as_dict(evaluation_summary).get("candidate_clusters")
    )
    if clusters.get("clusters"):
        ctx["candidate_clusters"] = clusters
    return ctx


def compact_context_for_input_config(
    *,
    target_name: str,
    current_config: Mapping[str, Any],
    diagnostic_report: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
    round_id: int,
    target_profile: Optional[Mapping[str, Any]] = None,
    structural_analysis: Optional[Mapping[str, Any]] = None,
    quality_analysis: Optional[Mapping[str, Any]] = None,
    hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
    memory_summary: Optional[Mapping[str, Any]] = None,
    constraints: Optional[Mapping[str, Any]] = None,
    tuning_feedback: Optional[Mapping[str, Any]] = None,
    active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """InputConfigurationAgent: the previously-uncompressed token bomb.

    It only needs *distilled decisions* (diagnostic corrective actions, quality
    guidance, hypotheses' config changes) plus the current knobs and a scalar
    evaluation summary — never raw structure summaries or coordinates.
    """
    ctx: Dict[str, Any] = {
        "target_name": target_name,
        "round_id": round_id,
        "task": "configure_next_round_based_on_diagnostics",
        "current_config": compact_config(current_config),
        "evaluation_summary": {
            "total_candidates": _as_dict(evaluation_summary).get("total_candidates"),
            "success_count": _as_dict(evaluation_summary).get("success_count"),
            "failure_count": _as_dict(evaluation_summary).get("failure_count"),
            "tag_counts": _as_dict(evaluation_summary).get("tag_counts"),
            "observations": _as_dict(evaluation_summary).get("observations"),
            "candidate_filtering": _as_dict(evaluation_summary).get("candidate_filtering"),
            "metric_facts": _as_dict(evaluation_summary).get("metric_facts") or build_metric_facts(evaluation_summary),
            "core_metric_trends": _as_dict(evaluation_summary).get("core_metric_trends"),
            "core_metric_stats": _as_dict(evaluation_summary).get("core_metric_stats"),
            "pressure_conflict": _as_dict(evaluation_summary).get("pressure_conflict"),
            "active_learning_examples": compact_active_learning_examples(_as_dict(evaluation_summary).get("active_learning_examples")),
        },
        "diagnostic_report": compact_diagnostic_report(diagnostic_report),
    }
    if target_profile:
        ctx["target_profile"] = compact_target_profile(target_profile)
    if structural_analysis:
        # Only aggregate tags matter for tuning decisions, no per-structure detail.
        ctx["structural_analysis"] = compact_structural_analysis(
            structural_analysis, include_summaries=False
        )
    if quality_analysis:
        ctx["quality_analysis"] = compact_quality_analysis(quality_analysis)
    if hypotheses:
        ctx["hypotheses"] = compact_hypotheses(hypotheses)
    if memory_summary:
        # compact_memory prefers recalled_items from MemoryRetrievalAgent and
        # only falls back to recent_rounds when retrieval produced nothing.
        ctx["memory_summary"] = compact_memory(memory_summary)
    if constraints:
        ctx["constraints"] = _strip_heavy(constraints)
    if tuning_feedback:
        ctx["tuning_feedback"] = _strip_heavy(tuning_feedback)
    clusters = compact_cluster_cards(
        _as_dict(evaluation_summary).get("candidate_clusters")
        or _as_dict(memory_summary).get("candidate_clusters")
    )
    if clusters.get("clusters"):
        ctx["candidate_clusters"] = clusters
    return ctx


def compact_context_for_config_validation(
    *,
    target_model: str,
    activation: str,
    config: Mapping[str, Any],
    deterministic_prefilter: Mapping[str, Any],
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """ConfigValidationAgent: only needs the *executable config itself*, the
    deterministic pre-filter verdict, and any execution error text.

    It must NOT receive structural summaries, coordinates, candidate metrics,
    message logs or memory — none of those affect whether a config is
    *submittable*.  Only the config shape and the error that triggered a repair
    are relevant.
    """
    prefilter = _as_dict(deterministic_prefilter)
    ctx: Dict[str, Any] = {
        "target_model": target_model,
        "activation": activation,
        # The config is the object under validation; strip heavy/unbounded keys
        # but keep every executable knob.
        "config": _strip_heavy(config),
        "deterministic_prefilter": {
            "is_valid": prefilter.get("is_valid"),
            "corrected_config": _strip_heavy(prefilter.get("corrected_config")),
            "issues": list(prefilter.get("issues") or [])[:MAX_CORRECTIVE_ACTIONS],
            "recommendations": list(prefilter.get("recommendations") or [])[:MAX_GUIDANCE],
        },
    }
    data = _as_dict(context)
    # Only the error text (for repair) is task-relevant; keep nothing else.
    error_context = data.get("error_context")
    if error_context is not None:
        err = _as_dict(error_context)
        ctx["error_context"] = {
            k: sanitize_error_text(err.get(k))
            for k in ("error_type", "message", "stderr_tail", "stdout_tail", "exit_code", "step")
            if k in err
        } or _truncate_str(error_context)
    return ctx


# Blocked-arm review uses an especially narrow evidence projection.  These
# allowlists are intentionally local to the prompt boundary; persisted memory
# and summary schemas remain unchanged.
_BLOCKED_ARM_STATE_FIELDS = (
    "arm_id", "source_round_id", "reason", "cooldown_until_round",
    "intervention_digest", "status",
)
_BLOCKED_ARM_EVIDENCE_FIELDS = (
    "evidence_id", "evidence_ids", "arm_id", "status", "requested_budget",
    "completed_budget", "trials", "successes", "endpoints", "confounders",
    "positive_features", "negative_features", "config_digest",
    "intervention_digest", "branch_id", "is_baseline",
)
_BLOCKED_ARM_FEATURE_LIMIT = 8


def _compact_small_value(value: Any, *, depth: int = 0) -> Any:
    """Keep scalars and bounded scalar mappings/lists only."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_str(value, limit=240)
    if depth >= 2:
        return None
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in list(value.items())[:16]:
            compact = _compact_small_value(item, depth=depth + 1)
            if compact is not None:
                out[str(key)] = compact
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in list(value)[:8]:
            compact = _compact_small_value(item, depth=depth + 1)
            if compact is not None:
                out.append(compact)
        return out
    return None


def _compact_blocked_arm_state(value: Any) -> Dict[str, Any]:
    item = value if isinstance(value, Mapping) else {}
    return {key: item.get(key) for key in _BLOCKED_ARM_STATE_FIELDS if item.get(key) is not None}


def _compact_blocked_arm_evidence(value: Any) -> Dict[str, Any]:
    item = value if isinstance(value, Mapping) else {}
    out: Dict[str, Any] = {}
    for key in _BLOCKED_ARM_EVIDENCE_FIELDS:
        if key not in item or item.get(key) is None:
            continue
        raw = item.get(key)
        if key in {"positive_features", "negative_features", "confounders", "evidence_ids"}:
            out[key] = [_compact_small_value(v) for v in list(raw or [])[:_BLOCKED_ARM_FEATURE_LIMIT]]
        elif key == "endpoints":
            out[key] = _compact_small_value(raw) or {}
        else:
            out[key] = raw
    return out


def _compact_blocked_ledger_summary(kind: str, value: Any) -> Any:
    if kind == "reward":
        return _compact_small_value(value)
    item = value if isinstance(value, Mapping) else {}
    fields = {
        "comparison": ("status", "winner_arm_id", "selected_arm_id", "closed_arm_ids", "evidence_ids", "reason"),
        "rollback": ("is_regression", "best_round", "best_reward", "current_reward", "relative_drop", "action", "reason"),
    }.get(kind, ())
    return {key: _compact_small_value(item.get(key)) for key in fields if item.get(key) is not None}


def _compact_blocked_ledger_round(value: Any, blocked_ids: set[str]) -> Dict[str, Any]:
    row = _as_dict(value)
    outcome = _as_dict(row.get("outcome"))
    arm_rows = list(row.get("per_arm_outcomes") or outcome.get("per_arm_outcomes") or [])
    arm_rows = [_compact_blocked_arm_evidence(item) for item in arm_rows
                if str(_as_dict(item).get("arm_id") or "") in blocked_ids]
    cards = _as_dict(row.get("arm_evidence_cards") or outcome.get("arm_evidence_cards"))
    card_rows = [_compact_blocked_arm_evidence(item) for item in cards.get("arms", []) or []
                 if str(_as_dict(item).get("arm_id") or "") in blocked_ids]
    compact: Dict[str, Any] = {"round_id": row.get("round_id")}
    if arm_rows:
        compact["arm_outcomes"] = arm_rows
    if card_rows:
        compact["arm_evidence_cards"] = {"arms": card_rows}
    for target, candidates in (
        ("comparison", (row.get("arm_comparison"), outcome.get("arm_comparison"), outcome.get("comparison"))),
        ("rollback", (row.get("rollback"), row.get("rollback_decision"), outcome.get("rollback"))),
        ("reward", (row.get("reward"), outcome.get("reward"))),
    ):
        source = next((candidate for candidate in candidates if candidate is not None), None)
        if source is not None:
            compact[target] = _compact_blocked_ledger_summary(target, source)
    return compact


def blocked_arm_ledger_view(
    ledger: Any,
    blocked_arm_ids: Iterable[str],
    *,
    max_rounds: int = MAX_RECENT_ROUNDS,
) -> Dict[str, Any]:
    """Read a tiny blocked-arm ledger view without dataclass deep-copying.

    ``LedgerRound`` may contain enormous policy/evaluation payloads. This helper
    deliberately accesses only named scalar metadata, ``per_arm_outcomes``, and
    a few optional decision summaries. It never calls ``asdict`` and never
    visits ``policy_snapshot`` or unrelated outcome keys.
    """
    blocked_ids = {str(value) for value in blocked_arm_ids if str(value)}
    out: Dict[str, Any] = {}
    for key in ("schema_version", "best_round_id", "best_reward", "best_round_rank_key"):
        value = getattr(ledger, key, None)
        if value is not None:
            out[key] = _compact_small_value(value)
    rounds = list(getattr(ledger, "rounds", None) or [])[-max(1, int(max_rounds)):]
    recent: List[Dict[str, Any]] = []
    for row in rounds:
        outcome = getattr(row, "outcome", None)
        outcome = outcome if isinstance(outcome, Mapping) else {}
        arm_rows = getattr(row, "per_arm_outcomes", None) or outcome.get("per_arm_outcomes") or []
        compact: Dict[str, Any] = {"round_id": getattr(row, "round_id", None)}
        related = [
            _compact_blocked_arm_evidence(item)
            for item in arm_rows
            if isinstance(item, Mapping) and str(item.get("arm_id") or "") in blocked_ids
        ]
        if related:
            compact["arm_outcomes"] = related
        cards = outcome.get("arm_evidence_cards")
        cards = cards if isinstance(cards, Mapping) else {}
        related_cards = [
            _compact_blocked_arm_evidence(item)
            for item in cards.get("arms", []) or []
            if isinstance(item, Mapping) and str(item.get("arm_id") or "") in blocked_ids
        ]
        if related_cards:
            compact["arm_evidence_cards"] = {"arms": related_cards}
        for target, keys in (
            ("comparison", ("arm_comparison", "comparison")),
            ("rollback", ("rollback", "rollback_decision")),
            ("reward", ("reward",)),
        ):
            source = next((outcome.get(key) for key in keys if outcome.get(key) is not None), None)
            if source is not None:
                compact[target] = _compact_blocked_ledger_summary(target, source)
        recent.append(compact)
    out["recent_rounds"] = recent
    return out


def compact_context_for_blocked_arm_review(
    *,
    round_id: int,
    blocked_arms: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    context: Optional[Mapping[str, Any]] = None,
    max_bytes: int = MAX_PROMPT_BYTES,
) -> Dict[str, Any]:
    """Build the fail-closed blocked-arm review prompt projection."""
    data = _as_dict(context)
    states = [_compact_blocked_arm_state(item) for item in blocked_arms]
    blocked_ids = {str(item.get("arm_id") or "") for item in states if item.get("arm_id")}
    ledger = _as_dict(data.get("ledger_history"))
    recent = [
        _compact_blocked_ledger_round(item, blocked_ids)
        for item in list(ledger.get("recent_rounds") or [])[-MAX_RECENT_ROUNDS:]
    ]
    analysis: Dict[str, Any] = {
        "selection_context": _compact_small_value(data.get("selection_context")) or {},
        "hypotheses": compact_hypotheses(data.get("hypotheses")),
        "quality_analysis": compact_quality_analysis(data.get("quality_analysis") or data.get("quality")),
        "structural_summary": compact_structural_analysis(
            data.get("structural_summary") or data.get("structural_analysis"),
            include_summaries=False,
        ),
        "ledger_history": {
            key: _compact_small_value(ledger.get(key))
            for key in ("schema_version", "best_round_id", "best_reward", "best_round_rank_key")
            if ledger.get(key) is not None
        },
    }
    analysis["ledger_history"]["recent_rounds"] = recent
    payload = {
        "round_id": int(round_id),
        "blocked_arms": states,
        "evidence": [_compact_blocked_arm_evidence(item) for item in evidence],
        "analysis_context": analysis,
    }
    return enforce_byte_budget(payload, max_bytes=max_bytes)


def compact_context_for_target_config(context: Mapping[str, Any]) -> Dict[str, Any]:
    """InputConfigurationAgent.configure (initial target-only path).

    This path runs before any round results exist, so the only meaningful
    inputs are the target description, prior results (if resuming) and the
    constraints.  Strip everything heavy and cap the projection.
    """
    data = _as_dict(context)
    ctx: Dict[str, Any] = {
        "target_name": data.get("target_name"),
        "target_info": _strip_heavy(data.get("target_info")),
    }
    if data.get("target_profile") is not None:
        ctx["target_profile"] = compact_target_profile(data.get("target_profile"))
    if data.get("previous_results") is not None:
        ctx["previous_results"] = compact_evaluation(data.get("previous_results"))
    if data.get("constraints") is not None:
        ctx["constraints"] = _strip_heavy(data.get("constraints"))
    return ctx


# ---------------------------------------------------------------------------
# Byte-budget guard (last-resort progressive compactor).
# ---------------------------------------------------------------------------
def _truncate_str(value: Any, *, limit: int = _STRING_TRUNCATE_LEN) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + _TRUNCATE_MARK
    return value


def _payload_bytes(payload: Any) -> int:
    """Serialised size of a payload exactly as the LLM client will send it.

    The client serialises compact JSON; measure the same representation so the
    byte budget describes the actual wire payload.
    """
    try:
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(payload).encode("utf-8"))


def _shrink_once(value: Any, *, str_limit: int, list_limit: int) -> Any:
    """One progressive-shrink pass: truncate long strings and cap list lengths."""
    if isinstance(value, str):
        return _truncate_str(value, limit=str_limit)
    if isinstance(value, Mapping):
        return {k: _shrink_once(v, str_limit=str_limit, list_limit=list_limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        capped = list(value)[:list_limit]
        shrunk = [_shrink_once(item, str_limit=str_limit, list_limit=list_limit) for item in capped]
        if len(value) > list_limit:
            shrunk.append({"_omitted_items": len(value) - list_limit})
        return shrunk
    return value


def enforce_byte_budget(
    payload: Mapping[str, Any],
    *,
    max_bytes: int = MAX_PROMPT_BYTES,
) -> Dict[str, Any]:
    """Guarantee the serialised payload is below ``max_bytes``.

    Strategy (applied only when the per-agent compaction was not enough):
    1. Strip heavy/unbounded keys (idempotent if already done upstream).
    2. Progressively shrink: each pass truncates long strings harder and caps
       list lengths smaller, until the payload fits or we hit the floor.
    3. As an absolute fallback, replace the payload with a minimal stub that
       preserves the most decision-relevant scalar fields plus a notice.

    Returns a new dict; never mutates the input.  Safe to call on an already
    compact payload (it returns it unchanged if already under budget).
    """
    current: Any = _strip_heavy(dict(payload))
    if _payload_bytes(current) <= max_bytes:
        return current if isinstance(current, dict) else dict(payload)

    # Progressive shrink passes with shrinking limits.
    for str_limit, list_limit in ((400, 6), (200, 4), (120, 3), (60, 2)):
        current = _shrink_once(current, str_limit=str_limit, list_limit=list_limit)
        if _payload_bytes(current) <= max_bytes:
            if isinstance(current, dict):
                current.setdefault("_context_compaction", {})
                if isinstance(current["_context_compaction"], dict):
                    audit = current["_context_compaction"]
                    audit.update({
                        "byte_budget_applied": True,
                        "policy": "deterministic_progressive_truncation",
                        "max_bytes": max_bytes,
                        "original_bytes": _payload_bytes(_strip_heavy(dict(payload))),
                    })
                    audit["final_bytes"] = 0
                    for _ in range(3):
                        audit["final_bytes"] = _payload_bytes(current)
            if _payload_bytes(current) <= max_bytes:
                return current if isinstance(current, dict) else {"payload": current}

    # Absolute fallback: keep only top-level scalars + a notice.
    src = dict(payload)
    stub: Dict[str, Any] = {
        "_context_compaction": {
            "byte_budget_applied": True,
            "fallback_stub": True,
            "original_bytes": _payload_bytes(_strip_heavy(dict(payload))),
            "max_bytes": max_bytes,
        }
    }
    for key in ("round_id", "target_name", "target_model", "activation", "task"):
        if key in src:
            stub[key] = src[key]
    return stub
