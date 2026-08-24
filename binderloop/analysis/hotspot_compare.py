"""Hotspot-set comparison metrics used only by post-run verification scripts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from binderloop.analysis.hotspot_descriptors import parse_hotspot_token


def normalize_hotspot_set(values: Optional[Iterable[Any]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in values or []:
        chain, number = parse_hotspot_token(item)
        if number is None:
            continue
        token = f"{chain}:{number}" if chain else str(number)
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def hotspot_numbers(values: Optional[Iterable[Any]]) -> List[int]:
    numbers: List[int] = []
    seen = set()
    for item in values or []:
        _chain, number = parse_hotspot_token(item)
        if number is None or number in seen:
            continue
        seen.add(number)
        numbers.append(number)
    return sorted(numbers)


def jaccard_index(left: Sequence[str], right: Sequence[str]) -> float:
    a = set(normalize_hotspot_set(left))
    b = set(normalize_hotspot_set(right))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def number_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a = set(hotspot_numbers(left))
    b = set(hotspot_numbers(right))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def sequence_hausdorff(left: Sequence[str], right: Sequence[str]) -> Optional[float]:
    a = hotspot_numbers(left)
    b = hotspot_numbers(right)
    if not a or not b:
        return None
    def directed(src: Sequence[int], dst: Sequence[int]) -> float:
        return max(min(abs(x - y) for y in dst) for x in src)
    return float(max(directed(a, b), directed(b, a)))


def compare_hotspot_sets(
    predicted: Sequence[str],
    prior: Sequence[str],
    *,
    label: str = "predicted_vs_prior",
) -> Dict[str, Any]:
    pred = normalize_hotspot_set(predicted)
    ref = normalize_hotspot_set(prior)
    pred_numbers = set(hotspot_numbers(pred))
    ref_numbers = set(hotspot_numbers(ref))
    overlap_tokens = sorted(set(pred) & set(ref))
    overlap_numbers = sorted(pred_numbers & ref_numbers)
    return {
        "label": label,
        "predicted": pred,
        "prior": ref,
        "predicted_count": len(pred),
        "prior_count": len(ref),
        "overlap_tokens": overlap_tokens,
        "overlap_residue_numbers": overlap_numbers,
        "jaccard_tokens": round(jaccard_index(pred, ref), 6),
        "jaccard_residue_numbers": round(number_jaccard(pred, ref), 6),
        "sequence_hausdorff": sequence_hausdorff(pred, ref),
        "exact_set_match": pred == ref,
    }


def load_prior_hotspots(path_or_mapping: Any) -> List[str]:
    """Load user-provided prior hotspots. Never used by the closed-loop harness."""
    import json
    from pathlib import Path

    import yaml

    if isinstance(path_or_mapping, Mapping):
        data = dict(path_or_mapping)
    else:
        path = Path(path_or_mapping)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text) or {}
    if isinstance(data, list):
        return normalize_hotspot_set(data)
    for key in ("hotspots", "prior_hotspots", "common_hotspots", "literature_hotspots"):
        if key in data:
            return normalize_hotspot_set(data.get(key))
    task = dict(data.get("task") or data.get("owner") or {})
    hard = dict(task.get("task_hard_constraints") or task)
    if hard.get("hotspots"):
        return normalize_hotspot_set(hard.get("hotspots"))
    return []


def rank_tuple(values: Any) -> Tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        return (float("-inf"),)
    out: List[float] = []
    for item in values:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(float("-inf"))
    return tuple(out) if out else (float("-inf"),)


def choose_best_round(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    rows = [dict(item) for item in records or [] if isinstance(item, Mapping)]
    if not rows:
        return None
    with_rank = [item for item in rows if item.get("round_rank_key")]
    if with_rank:
        return max(with_rank, key=lambda item: rank_tuple(item.get("round_rank_key")))
    return max(rows, key=lambda item: (float(item.get("success_rate") or 0.0), int(item.get("success_count") or 0)))


def load_round_hotspot_record(round_id: int, round_dir: Any) -> Dict[str, Any]:
    from pathlib import Path
    import json

    path = Path(round_dir)
    used = _read_mapping(path / "llm_hotspot_selection.json")
    bundle = _read_mapping(path / "round_analysis_bundle.json")
    evaluation = dict(bundle.get("evaluation_summary") or {})
    metrics = dict(used.get("round_metrics") or {})
    success_count = int(metrics.get("success_count") or evaluation.get("success_count") or 0)
    total = int(metrics.get("total_candidates") or evaluation.get("total_candidates") or 0)
    success_rate = metrics.get("success_rate")
    if success_rate is None:
        success_rate = evaluation.get("success_rate")
    if success_rate is None:
        success_rate = (success_count / total) if total else 0.0
    rank_key = metrics.get("round_rank_key") or evaluation.get("round_rank_key") or []
    hotspots = list(used.get("hotspots") or [])
    if not hotspots:
        hotspots = list((bundle.get("llm_hotspot_selection") or {}).get("hotspots") or [])
    return {
        "round_id": int(round_id),
        "hotspots": normalize_hotspot_set(hotspots),
        "success_count": success_count,
        "total_candidates": total,
        "success_rate": float(success_rate or 0.0),
        "round_rank_key": list(rank_key or []),
        "source": used.get("source"),
        "allow_web_search": used.get("allow_web_search"),
        "identity_hidden": used.get("identity_hidden"),
        "path": str(path / "llm_hotspot_selection.json"),
    }


def collect_run_hotspot_records(run_dir: Any) -> List[Dict[str, Any]]:
    from pathlib import Path

    root = Path(run_dir)
    records: List[Dict[str, Any]] = []
    for path in sorted(root.glob("round_*")):
        if not path.is_dir():
            continue
        try:
            round_id = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        records.append(load_round_hotspot_record(round_id, path))
    return records


def compare_run_to_prior(run_dir: Any, prior: Sequence[str]) -> Dict[str, Any]:
    records = collect_run_hotspot_records(run_dir)
    best = choose_best_round(records)
    return {
        "rounds": records,
        "best_round": best,
        "hotspot_comparison": compare_hotspot_sets(
            (best or {}).get("hotspots") or [],
            prior,
            label="best_round_llm_vs_prior",
        ),
    }


def _read_mapping(path: Any) -> Dict[str, Any]:
    from pathlib import Path
    import json

    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, Mapping) else {}
