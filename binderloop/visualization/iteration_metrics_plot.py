
import json
import math
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Optional, Union, List, Dict, Tuple

import numpy as np

from binderloop.resume import atomic_write_json

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional at import time, required for plotting
    plt = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_better: bool
    aliases: Tuple[str, ...] = ()


STRUCTURAL_METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("design_to_target_iptm", "Design-to-target iPTM", True, ("iptm", "interface_confidence")),
    MetricSpec("min_design_to_target_pae", "Min design-to-target PAE", False, ("min_interaction_pae",)),
    MetricSpec("design_ptm", "Design PTM", True, ("binder_plddt", "sequence_designability")),
    MetricSpec("filter_rmsd", "Filter RMSD (Å)", False, ("bb_rmsd",)),
    MetricSpec(
        "designfolding-filter_rmsd",
        "Design folding RMSD (Å)",
        False,
        (
            "filter_rmsd_design",
            "designfolding-bb_rmsd_design",
            "designfolding-bb_rmsd",
            "bb_rmsd_design",
        ),
    ),
    MetricSpec("plip_hbonds_refolded", "PLIP H-bonds (refolded)", True, ("hotspot_contact",)),
    MetricSpec("delta_sasa_refolded", "ΔSASA refolded (Å²)", True, ()),
)

ROUND_ANALYSIS_BUNDLE_FILENAME = "round_analysis_bundle.json"
ROUND_ANALYSIS_BUNDLE_SCHEMA = "binder_harness.round_analysis_bundle"
ROUND_ANALYSIS_BUNDLE_VERSION = 2


class IterationMetricsPlotError(RuntimeError):
    """Base error for iteration-metrics plotting."""


class IterationMetricsNoDataError(IterationMetricsPlotError):
    """Raised when plotting inputs are valid but contain no metric data."""


class IterationMetricsInputError(IterationMetricsPlotError):
    """Raised when plotting input or options are invalid."""


@dataclass
class RoundMetricStats:
    round_id: int
    metric_key: str
    metric_label: str
    n: int
    best: float
    mean: float
    std: float
    scope: str = "all_valid"
    median: float = 0.0
    q25: float = 0.0
    q75: float = 0.0
    top_k: int = 0
    top_k_value: float = 0.0


@dataclass
class RoundQualitySummary:
    round_id: int
    total_candidates: int
    success_count: int
    failure_count: int
    success_rate: float
    best_total_score: Optional[float]
    mean_top_total_score: Optional[float]
    best_iptm: Optional[float]
    mean_iptm: Optional[float]
    best_design_ptm: Optional[float]
    mean_design_ptm: Optional[float]
    min_pae: Optional[float]
    min_filter_rmsd: Optional[float]
    total_structures: int
    reliable_seed_fraction: float
    high_quality_fragment_count: int
    low_quality_fragment_count: int
    preserve_template_count: int
    avoid_template_count: int
    dominant_failure_tags: Dict[str, int]
    dominant_structure_tags: Dict[str, int]


@dataclass(frozen=True)
class _RoundAnalysis:
    candidates: List[Dict[str, Any]]
    evaluation: Dict[str, Any]
    structure: Dict[str, Any]
    templates: Dict[str, Any]


def _float_value(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if value in (None, ""):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


METRIC_SCOPES = ("all_valid", "additional_filter_passed", "boltzgen_passed", "harness_passed")


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool): return value
    if isinstance(value, (int, float)): return bool(value)
    token = str(value or "").strip().lower()
    if token in {"true", "1", "yes", "pass", "passed"}: return True
    if token in {"false", "0", "no", "fail", "failed"}: return False
    return None


def candidates_for_scope(candidates: Sequence[Mapping[str, Any]], scope: str) -> List[Mapping[str, Any]]:
    if scope not in METRIC_SCOPES: raise IterationMetricsInputError(f"Unknown metric scope: {scope}")
    if scope == "all_valid": return list(candidates)
    keys = {"additional_filter_passed": ("additional_filter_passed", "pass_iptm_filter"), "boltzgen_passed": ("boltzgen_passed", "pass_filters"), "harness_passed": ("harness_passed", "success", "passed")}[scope]
    return [row for row in candidates if any(_as_bool(row.get(key)) is True for key in keys)]


def extract_metric_values(candidates: Sequence[Mapping[str, Any]], spec: MetricSpec) -> List[float]:
    keys = (spec.key, *spec.aliases)
    values: List[float] = []
    for row in candidates:
        value = _float_value(row, *keys)
        if value is not None:
            values.append(value)
    return values


def aggregate_round_stats(round_id: int, candidates: Sequence[Mapping[str, Any]], specs: Optional[Sequence[MetricSpec]] = None, *, scope: str = "all_valid", top_k: int = 5) -> List[RoundMetricStats]:
    specs = list(specs or STRUCTURAL_METRICS)
    stats: List[RoundMetricStats] = []
    scoped = candidates_for_scope(candidates, scope)
    for spec in specs:
        values = extract_metric_values(scoped, spec)
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        best = float(np.max(arr) if spec.higher_better else np.min(arr))
        stats.append(
            RoundMetricStats(
                round_id=round_id,
                metric_key=spec.key,
                metric_label=spec.label,
                n=int(arr.size),
                best=best,
                mean=float(np.mean(arr)),
                std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                scope=scope, median=float(np.median(arr)), q25=float(np.quantile(arr, 0.25)), q75=float(np.quantile(arr, 0.75)),
                top_k=min(max(1, int(top_k)), int(arr.size)), top_k_value=float(np.mean(sorted(arr, reverse=spec.higher_better)[:max(1, int(top_k))])),
            )
        )
    return stats


def discover_round_dirs(out_dir: Union[str, Path]) -> List[Tuple[int, Path]]:
    root = Path(out_dir)
    rounds: List[Tuple[int, Path]] = []
    for path in sorted(root.glob("round_*")):
        if not path.is_dir():
            continue
        try:
            round_id = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        rounds.append((round_id, path))
    return sorted(rounds, key=lambda item: item[0])


def load_round_candidates(round_dir: Path) -> List[Dict[str, Any]]:
    round_dir = Path(round_dir)
    bundle = _analysis_from_bundle(
        _read_json(round_dir / ROUND_ANALYSIS_BUNDLE_FILENAME, default=None)
    )
    if bundle is not None:
        return bundle.candidates
    return _load_legacy_candidates(round_dir)


def _select_final_stage_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in candidates]
    if not rows:
        return []
    final_design_rows = [row for row in rows if _is_final_design_metrics(row.get("_metrics_file"))]
    if final_design_rows:
        return final_design_rows
    final_ranked_rows = [row for row in rows if _is_final_ranked_metrics(row.get("_metrics_file"))]
    if final_ranked_rows:
        return final_ranked_rows
    return rows


def _is_final_design_metrics(path: Any) -> bool:
    text = str(path or "").replace("\\", "/")
    return "/final_ranked_designs/" in text and Path(text).name.startswith("final_designs_metrics")


def _is_final_ranked_metrics(path: Any) -> bool:
    text = str(path or "").replace("\\", "/")
    name = Path(text).name
    return ("/final_ranked_designs/" in text and name in {"all_designs_metrics.csv", "final_designs_metrics.csv"}) or _is_final_design_metrics(text)


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _json_source(source: Any, *, default: Any) -> Any:
    if source is None:
        return default
    if isinstance(source, (str, Path)):
        return _read_json(Path(source), default=default)
    return source


def _evaluation_candidates(evaluation: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in evaluation.get("top_candidates") or []:
        row = dict(item.get("raw") or {})
        row.setdefault("candidate_id", item.get("candidate_id"))
        if item.get("source"):
            row.setdefault("_metrics_file", item.get("source"))
        candidates.append(row)
    for item in evaluation.get("failed_examples") or []:
        row = dict(item.get("raw") or {})
        row.setdefault("candidate_id", item.get("candidate_id"))
        if item.get("source"):
            row.setdefault("_metrics_file", item.get("source"))
        candidates.append(row)
    return _select_final_stage_candidates(candidates)


def _candidate_rows(source: Any) -> List[Dict[str, Any]]:
    payload = _json_source(source, default=[])
    if isinstance(payload, Mapping):
        if "top_candidates" in payload or "failed_examples" in payload:
            return _evaluation_candidates(payload)
        nested = payload.get("candidates")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            return _select_final_stage_candidates([dict(row) for row in nested if isinstance(row, Mapping)])
        return [dict(payload)]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return []

    rows: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        nested = item.get("candidates")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            rows.extend(dict(row) for row in nested if isinstance(row, Mapping))
        else:
            rows.append(dict(item))
    return _select_final_stage_candidates(rows)


def _load_legacy_candidates(
    round_dir: Path,
    *,
    evaluation: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    ingestion_path = round_dir / "ingestions.json"
    if ingestion_path.exists():
        candidates = _candidate_rows(ingestion_path)
        if candidates:
            return candidates
    if evaluation is None:
        evaluation = _read_json(round_dir / "evaluation_summary.json", default={})
    return _evaluation_candidates(evaluation) if isinstance(evaluation, Mapping) else []


def _compact_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for candidate in candidates:
        row: Dict[str, Any] = {}
        for spec in STRUCTURAL_METRICS:
            value = _float_value(candidate, spec.key, *spec.aliases)
            if value is not None:
                row[spec.key] = value
        for key in ("additional_filter_passed", "pass_iptm_filter", "boltzgen_passed", "pass_filters", "harness_passed", "success", "passed"):
            if key in candidate:
                row[key] = candidate[key]
        compact.append(row)
    return compact


def _round_id_from_dir(round_dir: Optional[Path]) -> Optional[int]:
    if round_dir is None:
        return None
    try:
        return int(round_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def build_round_analysis_bundle(
    *,
    round_id: Optional[int] = None,
    round_dir: Union[str, Path, None] = None,
    candidates: Any = None,
    population_candidates: Any = None,
    evaluation: Any = None,
    structure: Any = None,
    templates: Any = None,
) -> Dict[str, Any]:
    """Build the lightweight, versioned per-round input consumed by plotting."""
    source_dir = Path(round_dir) if round_dir is not None else None
    evaluation_payload = _json_source(
        evaluation if evaluation is not None else (source_dir / "evaluation_summary.json" if source_dir else None),
        default={},
    )
    structure_payload = _json_source(
        structure if structure is not None else (source_dir / "structure_evaluation.json" if source_dir else None),
        default={},
    )
    template_payload = _json_source(
        templates if templates is not None else (source_dir / "fragment_templates.json" if source_dir else None),
        default={},
    )
    evaluation_data = dict(evaluation_payload) if isinstance(evaluation_payload, Mapping) else {}
    structure_data = dict(structure_payload) if isinstance(structure_payload, Mapping) else {}
    template_data = dict(template_payload) if isinstance(template_payload, Mapping) else {}

    if population_candidates is not None:
        candidate_rows = _candidate_rows(population_candidates)
    elif candidates is None:
        candidate_rows = (
            _load_legacy_candidates(source_dir, evaluation=evaluation_data)
            if source_dir is not None
            else _evaluation_candidates(evaluation_data)
        )
    else:
        candidate_rows = _candidate_rows(candidates)
    analysis_rows = _candidate_rows(candidates) if candidates is not None else list(candidate_rows)

    top_scores = [
        value
        for value in (
            _float_value(item, "total")
            for item in evaluation_data.get("top_candidates") or []
            if isinstance(item, Mapping)
        )
        if value is not None
    ]
    if not top_scores:
        top_scores = [
            value
            for value in (
                _float_value({"value": item}, "value")
                for item in evaluation_data.get("top_total_scores") or []
            )
            if value is not None
        ]

    structure_summaries = [
        item for item in structure_data.get("summaries") or [] if isinstance(item, Mapping)
    ]
    high_fragment_count = structure_data.get("high_quality_fragment_count")
    if high_fragment_count is None:
        high_fragment_count = sum(len(item.get("high_quality_fragments") or []) for item in structure_summaries)
    low_fragment_count = structure_data.get("low_quality_fragment_count")
    if low_fragment_count is None:
        low_fragment_count = sum(len(item.get("low_quality_fragments") or []) for item in structure_summaries)

    template_items = [
        item for item in template_data.get("templates") or [] if isinstance(item, Mapping)
    ]
    preserve_count = template_data.get("preserve_count")
    if preserve_count is None:
        preserve_count = sum(1 for item in template_items if item.get("reuse_mode") == "preserve")
    avoid_count = template_data.get("avoid_count")
    if avoid_count is None:
        avoid_count = sum(1 for item in template_items if item.get("reuse_mode") == "avoid")

    # In schema v2 the plotting population is authoritative for denominators.
    total_candidates = len(candidate_rows)
    success_count = int(evaluation_data.get("success_count") or 0)
    failure_count = max(0, total_candidates - success_count)
    candidate_filtering = dict(evaluation_data.get("candidate_filtering") or {})
    analysis_candidate_count = len(analysis_rows)
    if candidate_filtering.get("analysis_candidate_count") is not None:
        try:
            analysis_candidate_count = int(candidate_filtering["analysis_candidate_count"])
        except (TypeError, ValueError):
            pass

    return {
        "schema": ROUND_ANALYSIS_BUNDLE_SCHEMA,
        "schema_version": ROUND_ANALYSIS_BUNDLE_VERSION,
        "round_id": round_id if round_id is not None else _round_id_from_dir(source_dir),
        "candidates": _compact_candidates(candidate_rows),
        "evaluation_summary": {
            "total_candidates": int(total_candidates),
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "success_rate": success_count / max(1, total_candidates),
            "analysis_candidate_count": int(analysis_candidate_count),
            "candidate_filtering": candidate_filtering,
            "top_total_scores": top_scores,
            "tag_counts": dict(evaluation_data.get("tag_counts") or {}),
        },
        "structure_summary": {
            "total_structures": int(structure_data.get("total_structures") or len(structure_summaries)),
            "reliable_seed_fraction": float(structure_data.get("reliable_seed_fraction") or 0.0),
            "high_quality_fragment_count": int(high_fragment_count),
            "low_quality_fragment_count": int(low_fragment_count),
            "aggregate_tags": dict(structure_data.get("aggregate_tags") or {}),
        },
        "template_summary": {
            "preserve_count": int(preserve_count),
            "avoid_count": int(avoid_count),
        },
    }


def write_round_analysis_bundle(
    path: Union[str, Path],
    *,
    round_id: Optional[int] = None,
    round_dir: Union[str, Path, None] = None,
    candidates: Any = None,
    population_candidates: Any = None,
    evaluation: Any = None,
    structure: Any = None,
    templates: Any = None,
) -> Path:
    """Build and write a round bundle; ``path`` may be a JSON path or round directory."""
    path = Path(path)
    output_path = path if path.suffix.lower() == ".json" else path / ROUND_ANALYSIS_BUNDLE_FILENAME
    source_dir = round_dir if round_dir is not None else output_path.parent
    bundle = build_round_analysis_bundle(
        round_id=round_id,
        round_dir=source_dir,
        candidates=candidates,
        population_candidates=population_candidates,
        evaluation=evaluation,
        structure=structure,
        templates=templates,
    )
    return atomic_write_json(output_path, bundle)


def _analysis_from_bundle(bundle: Any) -> Optional[_RoundAnalysis]:
    if not isinstance(bundle, Mapping):
        return None
    if bundle.get("schema") != ROUND_ANALYSIS_BUNDLE_SCHEMA:
        return None
    if bundle.get("schema_version") != ROUND_ANALYSIS_BUNDLE_VERSION:
        return None
    candidates = bundle.get("candidates")
    evaluation = bundle.get("evaluation_summary")
    structure = bundle.get("structure_summary")
    templates = bundle.get("template_summary")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return None
    if not all(isinstance(item, Mapping) for item in candidates):
        return None
    if not all(isinstance(item, Mapping) for item in (evaluation, structure, templates)):
        return None
    return _RoundAnalysis(
        candidates=[dict(item) for item in candidates],
        evaluation=dict(evaluation),
        structure=dict(structure),
        templates=dict(templates),
    )


def _load_round_analysis(round_dir: Path) -> _RoundAnalysis:
    bundle = _analysis_from_bundle(
        _read_json(round_dir / ROUND_ANALYSIS_BUNDLE_FILENAME, default=None)
    )
    if bundle is not None:
        return bundle

    evaluation = _read_json(round_dir / "evaluation_summary.json", default={})
    structure = _read_json(round_dir / "structure_evaluation.json", default={})
    templates = _read_json(round_dir / "fragment_templates.json", default={})
    evaluation = dict(evaluation) if isinstance(evaluation, Mapping) else {}
    structure = dict(structure) if isinstance(structure, Mapping) else {}
    templates = dict(templates) if isinstance(templates, Mapping) else {}
    return _RoundAnalysis(
        candidates=_load_legacy_candidates(round_dir, evaluation=evaluation),
        evaluation=evaluation,
        structure=structure,
        templates=templates,
    )


class IterationMetricsRoundCache:
    """Incrementally load round analysis once, unless a round is invalidated."""

    def __init__(self) -> None:
        self._roots: Dict[Path, Dict[int, _RoundAnalysis]] = {}

    @staticmethod
    def _root(out_dir: Union[str, Path]) -> Path:
        return Path(out_dir).resolve()

    def invalidate(
        self,
        round_ids: Optional[Union[int, Iterable[int]]] = None,
        *,
        out_dir: Union[str, Path, None] = None,
    ) -> None:
        roots = [self._root(out_dir)] if out_dir is not None else list(self._roots)
        normalized_ids = (
            [round_ids]
            if isinstance(round_ids, int)
            else list(round_ids) if round_ids is not None else None
        )
        for root in roots:
            if round_ids is None:
                self._roots.pop(root, None)
                continue
            cached = self._roots.get(root)
            if cached is None:
                continue
            assert normalized_ids is not None
            for round_id in normalized_ids:
                cached.pop(int(round_id), None)

    def add_bundle(
        self,
        out_dir: Union[str, Path],
        round_id: int,
        bundle: Mapping[str, Any],
    ) -> None:
        analysis = _analysis_from_bundle(bundle)
        if analysis is None:
            raise IterationMetricsInputError("Invalid round analysis bundle")
        self._roots.setdefault(self._root(out_dir), {})[int(round_id)] = analysis

    def refresh(
        self,
        out_dir: Union[str, Path],
        *,
        invalidate_rounds: Optional[Union[int, Iterable[int]]] = None,
    ) -> List[Tuple[int, _RoundAnalysis]]:
        root = self._root(out_dir)
        if invalidate_rounds is not None:
            self.invalidate(invalidate_rounds, out_dir=root)
        cached = self._roots.setdefault(root, {})
        discovered = discover_round_dirs(root)
        for round_id, round_dir in discovered:
            if round_id not in cached:
                cached[round_id] = _load_round_analysis(round_dir)
        return [(round_id, cached[round_id]) for round_id, _ in discovered]


def _top_counts(counts: Mapping[str, Any], *, limit: int = 5) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for key, value in counts.items():
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True)[:limit])


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _best_summary(summaries: Sequence[RoundQualitySummary], key: str, *, higher: bool) -> Optional[RoundQualitySummary]:
    valid = [item for item in summaries if getattr(item, key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda item: getattr(item, key)) if higher else min(valid, key=lambda item: getattr(item, key))


def _build_iteration_stats_from_rounds(
    rounds: Sequence[Tuple[int, _RoundAnalysis]],
    specs: Optional[Sequence[MetricSpec]] = None,
    *, scope: str = "all_valid", top_k: int = 5,
) -> List[RoundMetricStats]:
    stats: List[RoundMetricStats] = []
    for round_id, analysis in rounds:
        stats.extend(aggregate_round_stats(round_id, analysis.candidates, specs=specs, scope=scope, top_k=top_k))
    return stats


def build_iteration_stats(
    out_dir: Union[str, Path],
    specs: Optional[Sequence[MetricSpec]] = None,
    *,
    cache: Optional[IterationMetricsRoundCache] = None,
    scope: str = "all_valid", top_k: int = 5,
) -> List[RoundMetricStats]:
    round_cache = cache or IterationMetricsRoundCache()
    return _build_iteration_stats_from_rounds(round_cache.refresh(out_dir), specs=specs, scope=scope, top_k=top_k)


def _build_round_quality_summary_from_rounds(
    rounds: Sequence[Tuple[int, _RoundAnalysis]],
) -> List[RoundQualitySummary]:
    summaries: List[RoundQualitySummary] = []
    for round_id, analysis in rounds:
        evaluation = analysis.evaluation
        structural = analysis.structure
        templates = analysis.templates
        candidates = analysis.candidates

        total_candidates = int(evaluation.get("total_candidates") or len(candidates))
        success_count = int(evaluation.get("success_count") or 0)
        failure_count = int(evaluation.get("failure_count") or max(0, total_candidates - success_count))
        top_candidates = list(evaluation.get("top_candidates") or [])
        top_scores = [_float_value(item, "total") for item in top_candidates]
        top_scores = [value for value in top_scores if value is not None]
        if not top_scores:
            top_scores = [
                value
                for value in (
                    _float_value({"value": item}, "value")
                    for item in evaluation.get("top_total_scores") or []
                )
                if value is not None
            ]
        iptm_values = extract_metric_values(candidates, STRUCTURAL_METRICS[0])
        design_ptm_values = extract_metric_values(candidates, STRUCTURAL_METRICS[2])
        pae_values = extract_metric_values(candidates, STRUCTURAL_METRICS[1])
        rmsd_values = extract_metric_values(candidates, STRUCTURAL_METRICS[3])

        structure_summaries = list(structural.get("summaries") or [])
        high_fragments = structural.get("high_quality_fragment_count")
        if high_fragments is None:
            high_fragments = sum(len(item.get("high_quality_fragments") or []) for item in structure_summaries)
        low_fragments = structural.get("low_quality_fragment_count")
        if low_fragments is None:
            low_fragments = sum(len(item.get("low_quality_fragments") or []) for item in structure_summaries)
        template_items = list(templates.get("templates") or [])
        preserve_count = templates.get("preserve_count")
        if preserve_count is None:
            preserve_count = sum(1 for item in template_items if item.get("reuse_mode") == "preserve")
        avoid_count = templates.get("avoid_count")
        if avoid_count is None:
            avoid_count = sum(1 for item in template_items if item.get("reuse_mode") == "avoid")
        summaries.append(
            RoundQualitySummary(
                round_id=round_id,
                total_candidates=total_candidates,
                success_count=success_count,
                failure_count=failure_count,
                success_rate=success_count / max(1, total_candidates),
                best_total_score=max(top_scores) if top_scores else None,
                mean_top_total_score=float(np.mean(top_scores)) if top_scores else None,
                best_iptm=max(iptm_values) if iptm_values else None,
                mean_iptm=float(np.mean(iptm_values)) if iptm_values else None,
                best_design_ptm=max(design_ptm_values) if design_ptm_values else None,
                mean_design_ptm=float(np.mean(design_ptm_values)) if design_ptm_values else None,
                min_pae=min(pae_values) if pae_values else None,
                min_filter_rmsd=min(rmsd_values) if rmsd_values else None,
                total_structures=int(structural.get("total_structures") or len(structure_summaries)),
                reliable_seed_fraction=float(structural.get("reliable_seed_fraction") or 0.0),
                high_quality_fragment_count=int(high_fragments),
                low_quality_fragment_count=int(low_fragments),
                preserve_template_count=int(preserve_count),
                avoid_template_count=int(avoid_count),
                dominant_failure_tags=_top_counts(dict(evaluation.get("tag_counts") or {})),
                dominant_structure_tags=_top_counts(dict(structural.get("aggregate_tags") or {})),
            )
        )
    return summaries


def build_round_quality_summary(
    out_dir: Union[str, Path],
    *,
    cache: Optional[IterationMetricsRoundCache] = None,
) -> List[RoundQualitySummary]:
    round_cache = cache or IterationMetricsRoundCache()
    return _build_round_quality_summary_from_rounds(round_cache.refresh(out_dir))


def write_iteration_stats_json(stats: Sequence[RoundMetricStats], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in stats]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_round_quality_summary_json(summaries: Sequence[RoundQualitySummary], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in summaries], ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_round_quality_summary_csv(summaries: Sequence[RoundQualitySummary], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(summaries[0]).keys()) if summaries else list(RoundQualitySummary.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            row = asdict(item)
            row["dominant_failure_tags"] = json.dumps(row["dominant_failure_tags"], ensure_ascii=False, sort_keys=True)
            row["dominant_structure_tags"] = json.dumps(row["dominant_structure_tags"], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
    return path


def write_round_quality_report(summaries: Sequence[RoundQualitySummary], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Closed-loop Binder Quality Summary", ""]
    if not summaries:
        lines.append("No round summaries were found.")
    else:
        latest = summaries[-1]
        best_iptm_round = _best_summary(summaries, "best_iptm", higher=True)
        best_success_round = _best_summary(summaries, "success_rate", higher=True)
        lines.extend([
            f"- Rounds analyzed: {len(summaries)}",
            f"- Latest round: {latest.round_id}, candidates={latest.total_candidates}, success_rate={latest.success_rate:.3f}",
            f"- Best iPTM round: {best_iptm_round.round_id if best_iptm_round else 'n/a'} ({_fmt(best_iptm_round.best_iptm if best_iptm_round else None)})",
            f"- Best success-rate round: {best_success_round.round_id if best_success_round else 'n/a'} ({_fmt(best_success_round.success_rate if best_success_round else None)})",
            "",
            "## Per-round Summary",
            "",
            "| round | candidates | success | success_rate | best_iPTM | mean_iPTM | best_design_ptm | reliable_structures | high_frag | low_frag | dominant_failures |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for item in summaries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.round_id),
                        str(item.total_candidates),
                        str(item.success_count),
                        _fmt(item.success_rate),
                        _fmt(item.best_iptm),
                        _fmt(item.mean_iptm),
                        _fmt(item.best_design_ptm),
                        _fmt(item.reliable_seed_fraction),
                        str(item.high_quality_fragment_count),
                        str(item.low_quality_fragment_count),
                        _format_counts(item.dominant_failure_tags),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class IterationMetricsPlotter:
    """Build per-round best/mean/std line charts for binder structural metrics."""

    def __init__(self, specs: Optional[Sequence[MetricSpec]] = None):
        self.specs = list(specs or STRUCTURAL_METRICS)

    def plot(
        self,
        out_dir: Union[str, Path],
        *,
        output_path: Union[str, Optional[Path]] = None,
        stats_json_path: Union[str, Optional[Path]] = None,
        cache: Optional[IterationMetricsRoundCache] = None,
        scope: str = "all_valid", top_k: int = 5,
    ) -> Dict[str, Path]:
        if plt is None:
            raise RuntimeError("matplotlib is required for iteration metric plots; install with: pip install matplotlib")

        out_dir = Path(out_dir)
        round_cache = cache or IterationMetricsRoundCache()
        rounds = round_cache.refresh(out_dir)
        all_round_ids = [round_id for round_id, _ in rounds]
        stats = _build_iteration_stats_from_rounds(rounds, specs=self.specs, scope=scope, top_k=top_k)
        if not stats:
            raise IterationMetricsNoDataError(f"No round metrics found under {out_dir}")
        quality_summaries = _build_round_quality_summary_from_rounds(rounds)

        stats_json = Path(stats_json_path or out_dir / "iteration_metrics_stats.json")
        write_iteration_stats_json(stats, stats_json)

        plot_path = Path(output_path or out_dir / "iteration_metrics_trends.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        self._render(stats, plot_path, all_round_ids=all_round_ids, quality_summaries=quality_summaries)
        return {"stats_json": stats_json, "plot_png": plot_path}

    def _render(
        self,
        stats: Sequence[RoundMetricStats],
        plot_path: Path,
        *,
        all_round_ids: Optional[Sequence[int]] = None,
        quality_summaries: Optional[Sequence[RoundQualitySummary]] = None,
    ) -> None:
        assert plt is not None
        by_metric: Dict[str, List[RoundMetricStats]] = {}
        for item in stats:
            by_metric.setdefault(item.metric_key, []).append(item)
        for rows in by_metric.values():
            rows.sort(key=lambda row: row.round_id)
        x_ticks = sorted(set(all_round_ids or [item.round_id for item in stats]))

        metric_order = [spec.key for spec in self.specs if spec.key in by_metric]
        n_metrics = len(metric_order)
        n_plots = n_metrics + (1 if quality_summaries else 0)
        n_cols = 2
        n_rows = int(math.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.6 * n_rows), squeeze=False)
        fig.suptitle(f"Closed-loop iteration: scope={stats[0].scope}; n is per metric/round", fontsize=14, y=0.995)

        for index, metric_key in enumerate(metric_order):
            ax = axes[index // n_cols][index % n_cols]
            rows = by_metric[metric_key]
            row_by_round = {row.round_id: row for row in rows}
            rounds = x_ticks
            best = [_nan(row_by_round[round_id].best) if round_id in row_by_round else float("nan") for round_id in rounds]
            mean = [_nan(row_by_round[round_id].mean) if round_id in row_by_round else float("nan") for round_id in rounds]
            std = [_nan(row_by_round[round_id].std) if round_id in row_by_round else float("nan") for round_id in rounds]
            label = rows[0].metric_label

            ax.plot(rounds, best, marker="o", linewidth=2.0, label="Best")
            ax.plot(rounds, mean, marker="s", linewidth=1.8, linestyle="--", label="Mean")
            bridge_label_added = False
            for values in (best, mean):
                for x0, y0, x1, y1 in _nan_bridge_segments(rounds, values):
                    ax.plot(
                        [x0, x1],
                        [y0, y1],
                        color="0.55",
                        linewidth=1.3,
                        linestyle=":",
                        alpha=0.9,
                        label="Missing-round bridge" if not bridge_label_added else None,
                    )
                    bridge_label_added = True
            lower = [m - s for m, s in zip(mean, std)]
            upper = [m + s for m, s in zip(mean, std)]
            ax.fill_between(rounds, lower, upper, alpha=0.18, label="Mean ± 1 SD")
            ax.set_title(label)
            ax.set_xlabel("Round")
            ax.set_ylabel(label)
            ax.set_xticks(x_ticks)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        next_index = n_metrics
        if quality_summaries:
            ax = axes[next_index // n_cols][next_index % n_cols]
            self._render_success_count_plot(ax, quality_summaries, x_ticks)
            next_index += 1

        for index in range(next_index, n_rows * n_cols):
            axes[index // n_cols][index % n_cols].axis("off")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _render_success_count_plot(ax: Any, summaries: Sequence[RoundQualitySummary], rounds: Sequence[int]) -> None:
        summary_by_round = {item.round_id: item for item in summaries}
        success_counts = [
            float(summary_by_round[round_id].success_count) if round_id in summary_by_round else float("nan")
            for round_id in rounds
        ]
        ax.plot(rounds, success_counts, marker="o", linewidth=2.0, label="Success binders")
        for x0, y0, x1, y1 in _nan_bridge_segments(rounds, success_counts):
            ax.plot([x0, x1], [y0, y1], color="0.55", linewidth=1.3, linestyle=":", alpha=0.9)
        ax.set_title("Success binders per round")
        ax.set_xlabel("Round")
        ax.set_ylabel("Success binder count")
        ax.set_xticks(rounds)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)


class IterationQualityAnalyzer:
    """Write per-round quality summaries and plots for closed-loop runs."""

    def analyze(
        self,
        out_dir: Union[str, Path],
        *,
        summary_json_path: Union[str, Optional[Path]] = None,
        summary_csv_path: Union[str, Optional[Path]] = None,
        report_md_path: Union[str, Optional[Path]] = None,
        plot_path: Union[str, Optional[Path]] = None,
        write_plot: bool = True,
    ) -> Dict[str, Path]:
        out_dir = Path(out_dir)
        summaries = build_round_quality_summary(out_dir)
        if not summaries:
            raise IterationMetricsNoDataError(f"No round quality summaries found under {out_dir}")

        artifacts: Dict[str, Path] = {}
        summary_json = Path(summary_json_path or out_dir / "iteration_quality_summary.json")
        summary_csv = Path(summary_csv_path or out_dir / "iteration_quality_summary.csv")
        report_md = Path(report_md_path or out_dir / "iteration_quality_report.md")
        artifacts["summary_json"] = write_round_quality_summary_json(summaries, summary_json)
        artifacts["summary_csv"] = write_round_quality_summary_csv(summaries, summary_csv)
        artifacts["report_md"] = write_round_quality_report(summaries, report_md)
        if write_plot:
            if plt is None:
                raise RuntimeError("matplotlib is required for quality plots; install with: pip install matplotlib")
            quality_plot = Path(plot_path or out_dir / "iteration_quality_trends.png")
            self._render_quality_plot(summaries, quality_plot)
            artifacts["plot_png"] = quality_plot
        return artifacts

    @staticmethod
    def _render_quality_plot(summaries: Sequence[RoundQualitySummary], plot_path: Path) -> None:
        assert plt is not None
        rounds = [item.round_id for item in summaries]
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), squeeze=False)
        fig.suptitle("Closed-loop iteration: generation quality summary", fontsize=14, y=0.995)

        ax = axes[0][0]
        ax.plot(rounds, [item.total_candidates for item in summaries], marker="o", label="Total candidates")
        ax.plot(rounds, [item.success_count for item in summaries], marker="s", label="Success binders")
        ax.plot(rounds, [item.failure_count for item in summaries], marker="^", label="Failed candidates")
        ax.set_title("Candidate counts")
        ax.set_xlabel("Round")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        ax = axes[0][1]
        ax.plot(rounds, [item.success_rate for item in summaries], marker="o")
        ax.set_title("Success rate")
        ax.set_xlabel("Round")
        ax.set_ylabel("Success fraction")
        ax.set_ylim(bottom=0)

        ax = axes[1][0]
        ax.plot(rounds, [_nan(item.best_iptm) for item in summaries], marker="o", label="Best iPTM")
        ax.plot(rounds, [_nan(item.mean_iptm) for item in summaries], marker="s", linestyle="--", label="Mean iPTM")
        ax.set_title("Interface confidence")
        ax.set_xlabel("Round")
        ax.set_ylabel("iPTM")
        ax.legend(fontsize=8)

        ax = axes[1][1]
        ax.plot(rounds, [_nan(item.best_design_ptm) for item in summaries], marker="o", label="Best design PTM")
        ax.plot(rounds, [_nan(item.min_filter_rmsd) for item in summaries], marker="s", linestyle="--", label="Min filter RMSD")
        ax.set_title("Fold/refold quality")
        ax.set_xlabel("Round")
        ax.set_ylabel("Metric value")
        ax.legend(fontsize=8)

        ax = axes[2][0]
        ax.plot(rounds, [item.reliable_seed_fraction for item in summaries], marker="o")
        ax.set_title("Structure reliability")
        ax.set_xlabel("Round")
        ax.set_ylabel("Reliable structure fraction")
        ax.set_ylim(bottom=0)

        ax = axes[2][1]
        ax.plot(rounds, [item.high_quality_fragment_count for item in summaries], marker="o", label="High-quality fragments")
        ax.plot(rounds, [item.low_quality_fragment_count for item in summaries], marker="s", label="Low-quality fragments")
        ax.plot(rounds, [item.preserve_template_count for item in summaries], marker="^", linestyle="--", label="Preserve templates")
        ax.plot(rounds, [item.avoid_template_count for item in summaries], marker="v", linestyle="--", label="Avoid templates")
        ax.set_title("Fragment/template quality")
        ax.set_xlabel("Round")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        for row in axes:
            for ax in row:
                ax.set_xticks(rounds)
                ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def _nan(value: Optional[float]) -> float:
    return float("nan") if value is None else float(value)


def _nan_bridge_segments(rounds: Sequence[int], values: Sequence[float]) -> List[Tuple[int, float, int, float]]:
    """Return line segments that visually bridge missing rounds without inventing data."""
    segments: List[Tuple[int, float, int, float]] = []
    previous_index: Optional[int] = None
    previous_value: Optional[float] = None
    for index, (round_id, value) in enumerate(zip(rounds, values)):
        if not math.isfinite(float(value)):
            continue
        if previous_index is not None and previous_value is not None and index - previous_index > 1:
            segments.append((int(rounds[previous_index]), previous_value, int(round_id), float(value)))
        previous_index = index
        previous_value = float(value)
    return segments


def analyze_iteration_quality(
    out_dir: Union[str, Path],
    *,
    summary_json_path: Union[str, Optional[Path]] = None,
    summary_csv_path: Union[str, Optional[Path]] = None,
    report_md_path: Union[str, Optional[Path]] = None,
    plot_path: Union[str, Optional[Path]] = None,
    write_plot: bool = True,
) -> Dict[str, Path]:
    return IterationQualityAnalyzer().analyze(
        out_dir,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        report_md_path=report_md_path,
        plot_path=plot_path,
        write_plot=write_plot,
    )


def plot_iteration_metrics(
    out_dir: Union[str, Path],
    *,
    output_path: Union[str, Optional[Path]] = None,
    stats_json_path: Union[str, Optional[Path]] = None,
    cache: Optional[IterationMetricsRoundCache] = None,
    scope: str = "all_valid",
    top_k: int = 5,
) -> Dict[str, Path]:
    return IterationMetricsPlotter().plot(
        out_dir,
        output_path=output_path,
        stats_json_path=stats_json_path,
        cache=cache, scope=scope, top_k=top_k,
    )
