"""Deterministic phenotype clustering for prompt injection.

Leaves stay in artifacts. Prompts receive cluster cards plus 1–2 representatives
per cluster, partitioned by active-learning label so strict / near-miss / other
never merge.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "candidate-clusters-v1"
AL_LABELS = ("strict_positive", "near_miss", "other_negative")
_LEN_RE = re.compile(r"(?:^|[_-])len(\d+)(?:[_-]|$)", re.IGNORECASE)


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _iptm(item: Mapping[str, Any]) -> Optional[float]:
    metrics = _as_dict(item.get("metrics"))
    raw = _as_dict(item.get("raw_metrics") or item.get("raw"))
    for source in (metrics, raw, item):
        for key in ("design_to_target_iptm", "iptm", "interface_confidence"):
            number = _float(source.get(key))
            if number is not None:
                return number
    return None


def _metric(item: Mapping[str, Any], *keys: str) -> Optional[float]:
    metrics = _as_dict(item.get("metrics"))
    raw = _as_dict(item.get("raw_metrics") or item.get("raw"))
    for source in (metrics, raw, item):
        for key in keys:
            number = _float(source.get(key))
            if number is not None:
                return number
    return None


def _al_label(item: Mapping[str, Any]) -> str:
    label = str(item.get("label") or "").strip()
    if label in AL_LABELS:
        return label
    if label in {"positive"}:
        return "strict_positive"
    if label in {"hard_negative"}:
        return "other_negative"
    tags = [str(tag) for tag in (item.get("tags") or [])]
    if tags == ["pass_compute_gate"] or item.get("primary_gate_pass") is True:
        return "strict_positive"
    return "other_negative"


def _length_bucket(item: Mapping[str, Any]) -> str:
    raw = _as_dict(item.get("raw_metrics") or item.get("raw"))
    for source in (raw, item, _as_dict(item.get("metrics"))):
        for key in ("num_design", "binder_length", "length"):
            number = _float(source.get(key))
            if number is not None and number > 0:
                return "len%d" % (int(number) // 10 * 10)
    for key in ("id", "file_name", "filename", "candidate_id"):
        match = _LEN_RE.search(str(item.get(key) or raw.get(key) or ""))
        if match:
            return "len%d" % (int(match.group(1)) // 10 * 10)
    return "len_unknown"


def _tag_signature(item: Mapping[str, Any]) -> Tuple[str, ...]:
    tags = [str(tag) for tag in (item.get("tags") or []) if str(tag)]
    failures = [str(tag) for tag in (item.get("primary_gate_failures") or []) if str(tag)]
    merged = [tag for tag in tags + failures if tag != "pass_compute_gate"]
    unique = tuple(sorted(set(merged)))
    return unique or ("untagged",)


def _hotspot_signature(item: Mapping[str, Any]) -> Tuple[str, ...]:
    contacts = item.get("hotspot_contacts")
    if isinstance(contacts, Mapping):
        keys = tuple(sorted(str(key) for key, value in contacts.items() if value))
        return keys or ("no_hotspot",)
    structure = _as_dict(item.get("structural") or item.get("structure"))
    nested = structure.get("hotspot_contacts")
    if isinstance(nested, Mapping):
        keys = tuple(sorted(str(key) for key, value in nested.items() if value))
        return keys or ("no_hotspot",)
    return ("no_hotspot",)


def _cluster_key(item: Mapping[str, Any], *, al_label: str) -> Tuple[Any, ...]:
    return (al_label, _tag_signature(item), _length_bucket(item), _hotspot_signature(item))


def _short_hash(parts: Sequence[Any]) -> str:
    encoded = "|".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _core_sort_key(item: Mapping[str, Any]) -> Tuple[float, float, float]:
    iptm = _iptm(item)
    pae = _metric(item, "min_design_to_target_pae", "min_pae")
    rmsd = _metric(item, "designfolding-filter_rmsd", "designfolding_filter_rmsd", "refold_rmsd")
    return (
        iptm if iptm is not None else -1.0,
        -(pae if pae is not None else 99.0),
        -(rmsd if rmsd is not None else 99.0),
    )


def _evidence_id(round_id: int, al_label: str, index: int) -> str:
    prefix = {
        "strict_positive": "STRICT_POS",
        "near_miss": "NEAR_MISS",
        "other_negative": "OTHER_NEG",
    }.get(al_label, "LEAF")
    return "R%s:%s:%s" % (int(round_id), prefix, index)


def _range(values: Sequence[Optional[float]]) -> Optional[List[float]]:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return [round(min(numbers), 6), round(max(numbers), 6)]


def _collect_leaves(
    *,
    evaluation: Mapping[str, Any],
    active_learning_examples: Mapping[str, Any],
    structural_analysis: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    current = _as_dict(active_learning_examples.get("current_round"))
    labeled: List[Dict[str, Any]] = []
    for key, label in (
        ("strict_positive_examples", "strict_positive"),
        ("positive_examples", "strict_positive"),
        ("near_miss_examples", "near_miss"),
        ("other_negative_examples", "other_negative"),
        ("hard_negative_examples", "other_negative"),
    ):
        for item in list(current.get(key) or []):
            row = dict(item)
            row["label"] = label
            labeled.append(row)
    if labeled:
        return labeled
    rows = list(evaluation.get("top_candidates") or []) + list(evaluation.get("failed_examples") or [])
    summaries = {_as_dict(item).get("candidate_id"): item for item in list(structural_analysis.get("summaries") or [])}
    out = []
    for item in rows:
        row = dict(item)
        cid = str(row.get("candidate_id") or row.get("id") or "")
        if cid and cid in summaries:
            row["hotspot_contacts"] = row.get("hotspot_contacts") or _as_dict(summaries[cid]).get("hotspot_contacts")
            row["reliability_tags"] = _as_dict(summaries[cid]).get("reliability_tags")
        out.append(row)
    return out


def aggregate_candidate_phenotypes(
    *,
    round_id: int,
    evaluation: Optional[Mapping[str, Any]] = None,
    active_learning_examples: Optional[Mapping[str, Any]] = None,
    structural_analysis: Optional[Mapping[str, Any]] = None,
    max_representatives: int = 2,
) -> Dict[str, Any]:
    """Return cluster cards. Leaves are stored under ``leaves`` for artifacts only."""
    evaluation = _as_dict(evaluation)
    examples = _as_dict(active_learning_examples)
    structural = _as_dict(structural_analysis)
    leaves = _collect_leaves(
        evaluation=evaluation,
        active_learning_examples=examples,
        structural_analysis=structural,
    )
    buckets: Dict[Tuple[Any, ...], List[Tuple[int, Dict[str, Any]]]] = {}
    indexed_leaves: List[Dict[str, Any]] = []
    for index, item in enumerate(leaves, start=1):
        label = _al_label(item)
        key = _cluster_key(item, al_label=label)
        evidence_id = str(item.get("evidence_id") or _evidence_id(round_id, label, index))
        leaf = dict(item)
        leaf["evidence_id"] = evidence_id
        leaf["label"] = label
        indexed_leaves.append(leaf)
        buckets.setdefault(key, []).append((index, leaf))

    clusters: List[Dict[str, Any]] = []
    representatives: List[Dict[str, Any]] = []
    for key, members in sorted(buckets.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        al_label, tags, length_bucket, hotspot = key
        ordered = [leaf for _, leaf in sorted(members, key=lambda pair: _core_sort_key(pair[1]), reverse=True)]
        reps = ordered[:1]
        if max_representatives > 1 and len(ordered) > 1:
            reps.append(ordered[-1])
        cluster_id = "C:%s:%s:%s" % (al_label, length_bucket, _short_hash(key))
        card = {
            "cluster_id": cluster_id,
            "size": len(ordered),
            "al_label": al_label,
            "shared_tags": list(tags),
            "length_bucket": length_bucket,
            "shared_hotspot_contacts": [item for item in hotspot if item != "no_hotspot"],
            "metric_range": {
                key_name: _range(values)
                for key_name, values in (
                    ("iptm", [_iptm(item) for item in ordered]),
                    ("pae", [_metric(item, "min_design_to_target_pae", "min_pae") for item in ordered]),
                    ("ptm", [_metric(item, "design_ptm", "ptm") for item in ordered]),
                    ("rmsd", [_metric(item, "designfolding-filter_rmsd", "designfolding_filter_rmsd") for item in ordered]),
                )
                if _range(values) is not None
            },
            "representatives": [str(item.get("evidence_id")) for item in reps],
            "evidence_ids": [str(item.get("evidence_id")) for item in ordered],
        }
        clusters.append(card)
        for item in reps:
            representatives.append({
                "cluster_id": cluster_id,
                "evidence_id": item.get("evidence_id"),
                "al_label": al_label,
                "metrics": _as_dict(item.get("metrics")) or {
                    key_name: value
                    for key_name, value in {
                        "design_to_target_iptm": _iptm(item),
                        "min_design_to_target_pae": _metric(item, "min_design_to_target_pae", "min_pae"),
                        "design_ptm": _metric(item, "design_ptm", "ptm"),
                    }.items()
                    if value is not None
                },
                "tags": list(item.get("tags") or tags),
            })

    structure_clusters = _structure_phenotype_clusters(structural, round_id=round_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "round_id": int(round_id),
        "cluster_count": len(clusters),
        "leaf_count": len(indexed_leaves),
        "clusters": clusters,
        "representatives": representatives,
        "structure_clusters": structure_clusters,
        "leaves": indexed_leaves,
    }


def compact_cluster_cards(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Prompt-safe view: cards + representatives, no leaves."""
    data = _as_dict(payload)
    if not data:
        return {}
    return {
        "schema_version": data.get("schema_version") or SCHEMA_VERSION,
        "round_id": data.get("round_id"),
        "cluster_count": data.get("cluster_count"),
        "leaf_count": data.get("leaf_count"),
        "clusters": [
            {
                key: item.get(key)
                for key in (
                    "cluster_id", "size", "al_label", "shared_tags", "length_bucket",
                    "shared_hotspot_contacts", "metric_range", "representatives",
                )
                if item.get(key) not in (None, [], {})
            }
            for item in list(data.get("clusters") or [])
        ],
        "representatives": list(data.get("representatives") or []),
        "structure_clusters": list(data.get("structure_clusters") or []),
    }


def _structure_tokens(item: Mapping[str, Any]) -> Tuple[str, ...]:
    tokens = {str(value) for value in (item.get("reliability_tags") or [])}
    contacts = item.get("hotspot_contacts")
    if isinstance(contacts, Mapping):
        tokens.update(str(key) for key, value in contacts.items() if value)
    return tuple(sorted(tokens)) or ("no_structure_tokens",)


def _structure_phenotype_clusters(structural: Mapping[str, Any], *, round_id: int) -> List[Dict[str, Any]]:
    summaries = [_as_dict(item) for item in list(structural.get("summaries") or [])]
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for item in summaries:
        buckets.setdefault(_structure_tokens(item), []).append(item)
    cards = []
    for tokens, members in sorted(buckets.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        if len(members) < 2 and len(summaries) <= 6:
            continue
        cards.append({
            "cluster_id": "S:%s:%s" % (round_id, _short_hash(tokens)),
            "size": len(members),
            "shared_tokens": list(tokens)[:12],
            "mean_reliability": round(
                sum(float(item.get("reliability_score") or 0.0) for item in members) / max(1, len(members)),
                4,
            ),
        })
    return cards
