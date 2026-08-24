
import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union
import hashlib
import json

from binderloop.analysis.core_objective import (
    candidate_core_score,
    core_rank_key,
    core_metrics_from_row,
)
from binderloop.analysis.failure_taxonomy import classify_failures
from binderloop.analysis.quality_thresholds import success_thresholds
from binderloop.resume import atomic_write_json


@dataclass
class CandidateEvaluation:
    candidate_id: str
    total: float
    metrics: Dict[str, float]
    tags: List[str]
    source: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    primary_gate_pass: bool = False
    primary_gate_failures: List[str] = field(default_factory=list)
    core_rank_key: List[float] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    total_candidates: int
    success_count: int
    failure_count: int
    tag_counts: Dict[str, int]
    top_candidates: List[CandidateEvaluation]
    failed_examples: List[CandidateEvaluation]
    observations: List[str]
    requested_backbone_designs: int = 0
    expected_candidate_upper_bound: int = 0
    candidate_count_semantics: str = "total_candidates counts downstream metric rows; requested_backbone_designs counts BoltzGen --num_designs backbones."
    candidate_filtering: Dict[str, Any] = field(default_factory=dict)


class EvaluationAgent:
    """Score BoltzGen candidates and separate successful/failed design factors."""

    # Primary metrics define whether a candidate is credible. H-bonds/SASA are
    # only secondary exploitation terms once these gates pass.
    PRIMARY_GATE_THRESHOLDS = success_thresholds()

    PRIMARY_WEIGHTS = {
        "interface_confidence": 0.35,
        "pae_confidence": 0.25,
        "design_ptm": 0.25,
        "refold_consistency": 0.15,
    }
    SECONDARY_WEIGHTS = {
        "hotspot_contact": 0.65,
        "buried_sasa": 0.35,
    }
    DEFAULT_WEIGHTS = {**PRIMARY_WEIGHTS, **SECONDARY_WEIGHTS}

    def evaluate_candidates(
        self,
        candidates: List[Mapping[str, Any]],
        *,
        weights: Optional[Mapping[str, float]] = None,
        thresholds: Optional[Mapping[str, float]] = None,
    ) -> EvaluationSummary:
        # Candidate ordering and gate labels share the canonical physical
        # thresholds. ``thresholds`` remains available to failure taxonomy for
        # legacy diagnostic customization, but cannot change J_core decisions.
        primary_thresholds = dict(self.PRIMARY_GATE_THRESHOLDS)
        evaluated: List[CandidateEvaluation] = []
        for i, row in enumerate(candidates):
            metrics = self._map_boltzgen_metrics(row)
            primary_pass, primary_failures = self._primary_gate(metrics, primary_thresholds)
            tags = classify_failures(metrics, dict(thresholds or {}))
            if not primary_pass:
                tags = [tag for tag in tags if tag != "pass_compute_gate"]
                for failure in primary_failures:
                    if failure not in tags:
                        tags.append(failure)
            elif tags != ["pass_compute_gate"]:
                # Keep legacy diagnostic tags, but do not let secondary contact
                # terms turn a primary-pass candidate into a failure.
                tags = ["pass_compute_gate"]
            rank_key = core_rank_key(metrics)
            total = self._monitoring_score(metrics)
            raw = dict(row)
            native_id = next((str(row.get(key)) for key in ("candidate_id", "global_candidate_id", "id", "file_name", "filename") if row.get(key) not in (None, "")), "")
            provenance = {key: row.get(key) for key in ("_metrics_relative_path", "_metrics_row_ordinal", "job_id", "arm_id", "logical_branch_id", "execution_job_id") if row.get(key) not in (None, "")}
            if native_id:
                cid = native_id
            elif provenance:
                cid = "candidate_" + hashlib.sha256(json.dumps(provenance, sort_keys=True, default=str).encode()).hexdigest()[:16]
            else:
                cid = f"sample_{i + 1}"
            evaluated.append(CandidateEvaluation(
                candidate_id=cid,
                total=total,
                metrics=metrics,
                tags=tags,
                source=str(row.get("_metrics_file") or row.get("_metrics_relative_path") or ""),
                raw=raw,
                primary_gate_pass=primary_pass,
                primary_gate_failures=primary_failures,
                core_rank_key=list(rank_key),
            ))

        ranked = sorted(evaluated, key=lambda x: tuple(x.core_rank_key), reverse=True)
        successes = [c for c in ranked if c.tags == ["pass_compute_gate"]]
        failures = [c for c in ranked if c.tags != ["pass_compute_gate"]]
        tag_counts: Dict[str, int] = {}
        for c in evaluated:
            for tag in c.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        observations = self._observations(successes, failures, tag_counts, len(evaluated))
        return EvaluationSummary(
            total_candidates=len(evaluated),
            success_count=len(successes),
            failure_count=len(failures),
            tag_counts=tag_counts,
            top_candidates=successes[:10] if successes else ranked[:10],
            failed_examples=failures[:10],
            observations=observations,
        )

    def write_summary(self, summary: EvaluationSummary, path: Union[str, Path]) -> Path:
        return atomic_write_json(path, asdict(summary))

    def write_scores_csv(self, summary: EvaluationSummary, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = summary.top_candidates + summary.failed_examples
        fieldnames = [
            "candidate_id", "total", "tags", "core_objective", "interface_confidence",
            "min_design_to_target_pae", "design_ptm", "designfolding_filter_rmsd",
            "hotspot_contact", "buried_sasa", "binder_plddt", "clash_penalty",
            "diversity", "sequence_designability", "source",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for c in rows:
                writer.writerow({
                    "candidate_id": c.candidate_id,
                    "total": c.total,
                    "tags": ";".join(c.tags),
                    "source": c.source,
                    **c.metrics,
                })
        return path

    @staticmethod
    def _float(row: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            if key in row and row[key] not in (None, ""):
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    continue
        return default

    def _map_boltzgen_metrics(self, row: Mapping[str, Any]) -> Dict[str, float]:
        core = core_metrics_from_row(row)
        interface = core["design_to_target_iptm"]
        pae = core["min_design_to_target_pae"]
        binder_plddt = self._float(row, "plddt", "design_plddt", "binder_plddt", "design_ptm")
        design_ptm = core["design_ptm"]
        refold_rmsd = core["designfolding_filter_rmsd"]
        hotspot_raw = self._float(row, "hotspot_contact", "bindingsite_contact", "design_residue_iptm", "plip_hbonds_refolded")
        sasa_raw = self._float(row, "delta_sasa_refolded", "buried_sasa", "interface_sasa", default=0.0)
        hotspot = min(1.0, max(0.0, hotspot_raw / 12.0))
        buried_sasa = min(1.0, max(0.0, sasa_raw / 1200.0))
        pae_confidence = core["pae_confidence"]
        refold_consistency = core["refold_consistency"]
        clash = self._float(row, "clash", "clash_penalty", default=0.0)
        clash_penalty = max(clash, min(1.0, refold_rmsd / 10.0))
        diversity = self._float(row, "diversity", "vendi", "sequence_diversity", default=0.5)
        seq = self._float(row, "sequence_designability", "design_ptm", "ptm", default=binder_plddt)
        return {
            "interface_confidence": interface,
            "min_design_to_target_pae": pae,
            "pae_confidence": pae_confidence,
            "binder_plddt": binder_plddt,
            "design_ptm": design_ptm,
            "designfolding_filter_rmsd": refold_rmsd,
            "refold_consistency": refold_consistency,
            "core_objective": core["core_objective"],
            "hotspot_contact": hotspot,
            "hotspot_contact_raw": hotspot_raw,
            "buried_sasa": buried_sasa,
            "delta_sasa_refolded": sasa_raw,
            "clash_penalty": clash_penalty,
            "diversity": diversity,
            "sequence_designability": seq,
        }

    @classmethod
    def _primary_gate(cls, metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> tuple:
        failures: List[str] = []
        if float(metrics.get("interface_confidence", 0.0)) < float(thresholds["design_to_target_iptm"]):
            failures.append("primary_gate_low_iptm")
        if float(metrics.get("min_design_to_target_pae", 100000.0)) > float(thresholds["min_design_to_target_pae"]):
            failures.append("primary_gate_high_pae")
        if float(metrics.get("design_ptm", 0.0)) < float(thresholds["design_ptm"]):
            failures.append("primary_gate_low_design_ptm")
        if float(metrics.get("designfolding_filter_rmsd", 100000.0)) > float(thresholds["designfolding_filter_rmsd"]):
            failures.append("primary_gate_high_refold_rmsd")
        return not failures, failures

    @classmethod
    def _monitoring_score(cls, metrics: Mapping[str, float]) -> float:
        """Legacy display scalar; candidate decisions use ``core_rank_key``."""
        return 100.0 * candidate_core_score(metrics)

    @staticmethod
    def _observations(successes: List[CandidateEvaluation], failures: List[CandidateEvaluation], tag_counts: Dict[str, int], total: int) -> List[str]:
        obs: List[str] = []
        if total == 0:
            return ["No candidate metrics were collected; inspect run logs and output paths before parameter iteration."]
        if successes:
            obs.append(f"{len(successes)}/{total} candidates passed current compute gates; inspect their interface/hotspot metrics for exploitation.")
        else:
            obs.append("No candidate passed all compute gates; next round should relax or repair the dominant failure modes rather than only increasing sample count.")
        if tag_counts:
            dominant = sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            obs.append("Dominant tags: " + ", ".join(f"{k}={v}" for k, v in dominant))
        if tag_counts.get("hotspot_miss", 0) > total * 0.3:
            obs.append("Hotspot miss is frequent: increase hotspot conditioning, expand/soften patch, or sample hotspot subsets.")
        if tag_counts.get("folding_failure", 0) > total * 0.3:
            obs.append("Folding failure is frequent: shorten binder length, add scaffold/topology bias, or reduce exploration noise.")
        if tag_counts.get("binding_pose_failure", 0) > total * 0.3:
            obs.append("Binding pose failure is frequent: strengthen interface constraints or switch target conformer/patch definition.")
        if tag_counts.get("diversity_collapse", 0) > total * 0.3:
            obs.append("Diversity collapse is frequent: increase diversity quota, lower per-strategy exploitation, or broaden length/topology search.")
        return obs
