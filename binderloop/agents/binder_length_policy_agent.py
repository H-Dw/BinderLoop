
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from binderloop.agents.config_parameter_contract import supported_config_changes
from binderloop.resume import atomic_write_json

# Global physical envelope for designed binder lengths (residues). Even when the
# user did not pin an explicit ``binder_length_range`` the policy never proposes
# lengths outside this range. These mirror the clamp used by the strategy learner
# when it mutates per-parent lengths.
GLOBAL_MIN_LENGTH = 30
GLOBAL_MAX_LENGTH = 180


@dataclass
class BinderLengthRecommendation:
    """Structure-quality-driven binder length range chosen for the next round."""

    enabled: bool
    direction: str  # shorter | longer | focus | hold | fixed | disabled | no_structures
    current_lengths: List[int]
    recommended_lengths: List[int]
    recommended_range: List[int]
    allowed_range: List[int]
    rationale: List[str] = field(default_factory=list)
    per_length_quality: List[Dict[str, Any]] = field(default_factory=list)
    recommended_config: Dict[str, Any] = field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


class BinderLengthPolicyAgent:
    """Choose the next round's binder length range from previous-round structures.

    The agent reads the coordinate-level structure evaluation of the previous
    round (interface size, foldability/chain breaks, clashes, reliability and,
    when available, inter-chain PAE), buckets structures by their realized
    binder length, and proposes a small set of discrete ``binder_lengths`` for
    the next round. The decision is intentionally heuristic and auditable:

    * Frequent foldability problems (chain breaks / low reliability) -> shift the
      range *shorter* (shorter binders fold more reliably).
    * Weak/tiny interfaces while folding is fine -> shift the range *longer*
      (more residues to reach and bury interface area).
    * One length bucket is clearly best -> *focus* the range around it.
    * Otherwise *hold* the current range.

    The proposal is always clamped to the user's allowed ``binder_length_range``
    (when set) so a frozen range is respected: the agent only redistributes the
    discrete lengths *within* the user envelope, it never widens it. The result
    is emitted as an executable ``binder_lengths`` config change that the
    orchestrator merges and writes into the next BoltzGen generation config.
    """

    def recommend(
        self,
        structural_analysis: Any,
        *,
        current_lengths: Sequence[int],
        allowed_min: Optional[int] = None,
        allowed_max: Optional[int] = None,
        step: int = 10,
        interchain_pae_by_structure: Optional[Mapping[str, float]] = None,
        enabled: bool = True,
        max_lengths: int = 4,
        min_support_fraction: float = 0.15,
    ) -> BinderLengthRecommendation:
        current = sorted({int(x) for x in (current_lengths or []) if int(x) > 0})
        step = max(1, int(step or 10))
        pae_map = dict(interchain_pae_by_structure or {})

        lo, hi = self._allowed_bounds(current, allowed_min, allowed_max, step)
        grid = self._length_grid(lo, hi, step)

        if not enabled:
            return BinderLengthRecommendation(
                enabled=False, direction="disabled", current_lengths=current,
                recommended_lengths=current, recommended_range=[lo, hi], allowed_range=[lo, hi],
                rationale=["Dynamic binder-length selection is disabled (auto_binder_length=false)."],
            )

        structural = _as_mapping(structural_analysis)
        summaries = [_as_mapping(s) for s in list(structural.get("summaries") or [])]
        total = len(summaries)
        if total == 0:
            return BinderLengthRecommendation(
                enabled=True, direction="no_structures", current_lengths=current,
                recommended_lengths=current, recommended_range=[lo, hi], allowed_range=[lo, hi],
                rationale=["No structures available; keep the current binder length set."],
            )

        if lo == hi:
            return BinderLengthRecommendation(
                enabled=True, direction="fixed", current_lengths=current,
                recommended_lengths=[lo], recommended_range=[lo, hi], allowed_range=[lo, hi],
                rationale=[f"Binder length range is fixed to a single value ({lo}); no dynamic selection possible."],
            )

        buckets = self._bucketize(summaries, grid, pae_map)
        per_length_quality = [self._bucket_report(length, stats) for length, stats in sorted(buckets.items())]

        weak_frac, foldfail_frac, clash_frac = self._global_fractions(summaries)
        best_length, best_score, score_margin = self._best_bucket(buckets, total, min_support_fraction)

        center, direction, rationale = self._decide(
            current=current, grid=grid, weak_frac=weak_frac, foldfail_frac=foldfail_frac,
            best_length=best_length, score_margin=score_margin, step=step,
        )
        recommended = self._build_window(center, grid, max_lengths=max_lengths, direction=direction)
        recommended = self._drop_failing_lengths(recommended, buckets, grid)

        rationale.append(
            f"Evidence: weak_interface_fraction={weak_frac:.2f}, foldability_fail_fraction={foldfail_frac:.2f}, "
            f"clash_fraction={clash_frac:.2f}, best_length={best_length} (score={best_score:.2f})."
        )
        rationale.append(
            f"Selected next-round binder_lengths={recommended} within allowed range [{lo}, {hi}] (direction={direction})."
        )

        recommended_config: Dict[str, Any] = {}
        if recommended and recommended != current:
            recommended_config = supported_config_changes({"binder_lengths": recommended}, include_internal=True)

        return BinderLengthRecommendation(
            enabled=True,
            direction=direction,
            current_lengths=current,
            recommended_lengths=recommended,
            recommended_range=[min(recommended), max(recommended)] if recommended else [lo, hi],
            allowed_range=[lo, hi],
            rationale=rationale,
            per_length_quality=per_length_quality,
            recommended_config=recommended_config,
            analysis_metadata={
                "weak_interface_fraction": round(weak_frac, 4),
                "foldability_fail_fraction": round(foldfail_frac, 4),
                "clash_fraction": round(clash_frac, 4),
                "best_length": best_length,
                "best_score": round(best_score, 4),
                "center_length": center,
                "interchain_pae_used": bool(pae_map),
                "total_structures": total,
            },
        )

    def write_recommendation(self, recommendation: BinderLengthRecommendation, path: Union[str, Path]) -> Path:
        return atomic_write_json(path, asdict(recommendation))

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _allowed_bounds(current: List[int], allowed_min: Optional[int], allowed_max: Optional[int], step: int) -> tuple:
        if allowed_min is not None and allowed_max is not None:
            lo, hi = int(allowed_min), int(allowed_max)
        elif current:
            # No explicit user range: allow a +/- step exploration envelope around
            # the current lengths, clamped to the global physical envelope.
            lo = max(GLOBAL_MIN_LENGTH, min(current) - step)
            hi = min(GLOBAL_MAX_LENGTH, max(current) + step)
        else:
            lo, hi = GLOBAL_MIN_LENGTH, GLOBAL_MAX_LENGTH
        if lo > hi:
            lo, hi = hi, lo
        lo = max(GLOBAL_MIN_LENGTH, lo)
        hi = min(GLOBAL_MAX_LENGTH, hi)
        return lo, hi

    @staticmethod
    def _length_grid(lo: int, hi: int, step: int) -> List[int]:
        if lo >= hi:
            return [lo]
        grid = list(range(lo, hi + 1, step))
        if grid[-1] != hi:
            grid.append(hi)
        return sorted({int(x) for x in grid})

    @staticmethod
    def _nearest(value: int, grid: Sequence[int]) -> int:
        return min(grid, key=lambda g: (abs(g - value), g))

    def _bucketize(self, summaries: List[Mapping[str, Any]], grid: List[int], pae_map: Mapping[str, float]) -> Dict[int, Dict[str, List[float]]]:
        buckets: Dict[int, Dict[str, List[float]]] = {}
        for summary in summaries:
            length = int(summary.get("binder_residue_count") or 0)
            if length <= 0:
                continue
            bucket_length = self._nearest(length, grid)
            stats = buckets.setdefault(bucket_length, {"reliability": [], "chain_break": [], "interface": [], "clash": [], "pae": []})
            stats["reliability"].append(float(summary.get("reliability_score") or 0.0))
            stats["chain_break"].append(1.0 if int(summary.get("chain_break_count") or 0) > 0 else 0.0)
            stats["interface"].append(float(summary.get("interface_residue_count") or 0.0))
            stats["clash"].append(float(summary.get("clash_density") or 0.0))
            pae = _lookup_pae(pae_map, str(summary.get("structure_file") or ""))
            if pae is not None:
                stats["pae"].append(pae)
        return buckets

    @staticmethod
    def _bucket_score(stats: Mapping[str, List[float]]) -> float:
        """Legacy display scalar; length decisions use ``_bucket_rank_key``."""
        terms: List[float] = []
        if stats.get("reliability"):
            terms.append(_clip01(_fmean(stats["reliability"])))
        if stats.get("chain_break"):
            terms.append(_clip01(1.0 - _fmean(stats["chain_break"])))
        if stats.get("interface"):
            terms.append(_clip01(_fmean(stats["interface"]) / 12.0))
        if stats.get("clash"):
            terms.append(_clip01(1.0 - _fmean(stats["clash"]) / 0.3))
        if stats.get("pae"):
            # PAE in [~5, ~20]; 5 -> 1.0, 20 -> 0.0.
            terms.append(_clip01(1.0 - (_fmean(stats["pae"]) - 5.0) / 15.0))
        return float(_fmean(terms)) if terms else 0.0

    @staticmethod
    def _bucket_gate_failures(stats: Mapping[str, List[float]]) -> List[str]:
        failures: List[str] = []
        reliability = _fmean(stats.get("reliability") or [])
        chain_break = _fmean(stats.get("chain_break") or [])
        if reliability < 0.5:
            failures.append("low_foldability_reliability")
        if chain_break >= 0.5:
            failures.append("frequent_chain_break")
        return failures

    @classmethod
    def _bucket_rank_key(cls, stats: Mapping[str, List[float]]) -> tuple:
        count = len(stats.get("reliability") or [])
        failures = cls._bucket_gate_failures(stats)
        pae_values = list(stats.get("pae") or [])
        return (
            count,
            int(not failures),
            _fmean(stats.get("reliability") or []),
            -_fmean(stats.get("chain_break") or []),
            int(bool(pae_values)),
            -_fmean(pae_values) if pae_values else float("-inf"),
            _fmean(stats.get("interface") or []),
            -_fmean(stats.get("clash") or []),
        )

    def _bucket_report(self, length: int, stats: Mapping[str, List[float]]) -> Dict[str, Any]:
        return {
            "length": int(length),
            "count": len(stats.get("reliability", [])),
            "mean_reliability": round(_fmean(stats["reliability"]), 4) if stats.get("reliability") else None,
            "chain_break_fraction": round(_fmean(stats["chain_break"]), 4) if stats.get("chain_break") else None,
            "mean_interface_residues": round(_fmean(stats["interface"]), 4) if stats.get("interface") else None,
            "mean_clash_density": round(_fmean(stats["clash"]), 4) if stats.get("clash") else None,
            "mean_interchain_pae": round(_fmean(stats["pae"]), 4) if stats.get("pae") else None,
            "score": round(self._bucket_score(stats), 4),
            "length_rank_key": list(self._bucket_rank_key(stats)),
            "gate_failures": self._bucket_gate_failures(stats),
        }

    @staticmethod
    def _global_fractions(summaries: List[Mapping[str, Any]]) -> tuple:
        total = max(1, len(summaries))
        weak = fold = clash = 0
        for s in summaries:
            tags = set(s.get("reliability_tags") or [])
            if "weak_or_tiny_interface" in tags or int(s.get("interface_residue_count") or 0) < 6:
                weak += 1
            if "binder_chain_break" in tags or int(s.get("chain_break_count") or 0) > 0 or float(s.get("reliability_score") or 1.0) < 0.5:
                fold += 1
            if "interface_clash_risk" in tags or float(s.get("clash_density") or 0.0) > 0.15:
                clash += 1
        return weak / total, fold / total, clash / total

    def _best_bucket(self, buckets: Dict[int, Dict[str, List[float]]], total: int, min_support_fraction: float) -> tuple:
        if not buckets:
            return None, 0.0, 0.0
        min_support = max(1, int(math.ceil(min_support_fraction * total)))
        scored = sorted(
            ((length, self._bucket_score(stats), len(stats.get("reliability", [])), self._bucket_rank_key(stats)) for length, stats in buckets.items()),
            key=lambda item: item[3],
            reverse=True,
        )
        supported = [item for item in scored if item[2] >= min_support] or scored
        best_length, best_score, _, best_rank = supported[0]
        second_rank = supported[1][3] if len(supported) > 1 else None
        rank_advantage = 1.0 if second_rank is None or best_rank > second_rank else 0.0
        return int(best_length), float(best_score), rank_advantage

    def _decide(self, *, current: List[int], grid: List[int], weak_frac: float, foldfail_frac: float, best_length: Optional[int], score_margin: float, step: int) -> tuple:
        median_current = int(statistics.median(current)) if current else (grid[len(grid) // 2])
        center = self._nearest(median_current, grid)
        rationale: List[str] = []
        if foldfail_frac >= 0.4 and foldfail_frac >= weak_frac:
            center = self._nearest(center - step, grid)
            direction = "shorter"
            rationale.append("Foldability failures dominate (chain breaks / low reliability); shifting the binder length range shorter for more reliable folds.")
        elif weak_frac >= 0.4 and foldfail_frac < 0.4:
            center = self._nearest(center + step, grid)
            direction = "longer"
            rationale.append("Interfaces are weak/tiny while folding is acceptable; shifting the binder length range longer to increase interface reach and buried area.")
        elif best_length is not None and score_margin >= 0.05 and best_length != center:
            center = self._nearest(best_length, grid)
            direction = "focus"
            rationale.append(f"Length {best_length} shows clearly higher structural quality; focusing the range around it.")
        else:
            direction = "hold"
            rationale.append("No dominant structural failure mode; holding the current binder length range.")
        return center, direction, rationale

    def _build_window(self, center: int, grid: List[int], *, max_lengths: int, direction: str = "hold") -> List[int]:
        center = self._nearest(center, grid)
        idx = grid.index(center)
        n = len(grid)
        cap = max(1, max_lengths)
        ordered = [center]
        below = [grid[idx - off] for off in range(1, n) if idx - off >= 0]
        above = [grid[idx + off] for off in range(1, n) if idx + off < n]
        if direction == "shorter":
            sequence = below + above  # bias the window toward shorter lengths
        elif direction == "longer":
            sequence = above + below  # bias the window toward longer lengths
        else:
            sequence = []
            for left, right in zip(below, above):
                sequence.extend([left, right])
            sequence.extend(below[len(above):])
            sequence.extend(above[len(below):])
        for length in sequence:
            if len(ordered) >= cap:
                break
            ordered.append(length)
        return sorted({int(x) for x in ordered})

    def _drop_failing_lengths(self, recommended: List[int], buckets: Dict[int, Dict[str, List[float]]], grid: List[int]) -> List[int]:
        """Remove observed lengths that fail the foldability hard gate."""
        kept = []
        for length in recommended:
            stats = buckets.get(length)
            if stats is not None and self._bucket_gate_failures(stats):
                continue
            kept.append(length)
        if kept:
            return kept
        unobserved = [length for length in grid if length not in buckets]
        if unobserved:
            center = recommended[len(recommended) // 2] if recommended else grid[len(grid) // 2]
            return [min(unobserved, key=lambda length: (abs(length - center), length))]
        # Every allowed length has failing evidence. Keep the least-bad observed
        # fallback so the executable recommendation is non-empty.
        return [max(recommended or grid, key=lambda length: self._bucket_rank_key(buckets.get(length, {})))]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _fmean(values: Sequence[float]) -> float:
    """``statistics.fmean`` is only available on Python 3.8+; fall back safely."""
    _fast = getattr(statistics, "fmean", None)
    data = list(values)
    if not data:
        return 0.0
    if _fast is not None:
        return _fast(data)
    return float(sum(data)) / len(data)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lookup_pae(pae_map: Mapping[str, float], source: str) -> Optional[float]:
    if not source or not pae_map:
        return None
    if source in pae_map:
        return _coerce_pae(pae_map[source])
    name = Path(source).name
    stem = Path(source).stem
    for key in (name, stem):
        if key in pae_map:
            return _coerce_pae(pae_map[key])
    # Boundary-safe suffix match so ``_3_1`` never matches ``_3_10``.
    for key, value in pae_map.items():
        key_stem = Path(str(key)).stem
        if not key_stem:
            continue
        if stem.endswith(key_stem) or key_stem.endswith(stem):
            return _coerce_pae(value)
    return None


def _coerce_pae(value: Any) -> Optional[float]:
    try:
        pae = float(value)
    except (TypeError, ValueError):
        return None
    if pae <= 0.0 or pae >= 1000.0:
        return None
    return pae
