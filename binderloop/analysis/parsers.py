
from pathlib import Path
from typing import Dict, Iterable, List, Union
import csv
import hashlib

from .failure_taxonomy import classify_failures
from .scoring import CandidateScore, weighted_score, write_scores


def _stable_unit_float(text: str, salt: str) -> float:
    h = hashlib.sha256((salt + text).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def mock_metrics(candidate_id: str) -> Dict[str, float]:
    """Deterministic fake metrics for CPU-only tests and replay plumbing."""
    return {
        "interface_confidence": 0.35 + 0.60 * _stable_unit_float(candidate_id, "iface"),
        "hotspot_contact": 0.25 + 0.70 * _stable_unit_float(candidate_id, "hotspot"),
        "binder_plddt": 0.45 + 0.50 * _stable_unit_float(candidate_id, "plddt"),
        "clash_penalty": 0.05 + 0.45 * _stable_unit_float(candidate_id, "clash"),
        "diversity": 0.10 + 0.85 * _stable_unit_float(candidate_id, "div"),
        "sequence_designability": 0.30 + 0.65 * _stable_unit_float(candidate_id, "seq"),
    }


def parse_boltzgen_scores(output_dir: Union[str, Path], weights: Dict[str, float]) -> List[CandidateScore]:
    """Parse BoltzGen metrics when available; otherwise return empty.

    Expected upstream files include final_ranked_designs/all_designs_metrics.csv or
    final_designs_metrics_<budget>.csv. Column names differ across versions, so we
    opportunistically map known metrics and leave missing values as zero.
    """
    root = Path(output_dir)
    files = list(root.glob("final_ranked_designs/*metrics*.csv")) + list(root.glob("**/all_designs_metrics.csv"))
    scores: List[CandidateScore] = []
    for f in files:
        with f.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cid = row.get("design") or row.get("name") or row.get("candidate_id") or f"{f.stem}_{len(scores)}"
                metrics = {
                    "interface_confidence": float(row.get("iptm", row.get("interface_confidence", 0)) or 0),
                    "hotspot_contact": float(row.get("hotspot_contact", row.get("plip_hbonds_refolded", 0)) or 0),
                    "binder_plddt": float(row.get("plddt", row.get("binder_plddt", 0)) or 0),
                    "clash_penalty": float(row.get("clash", row.get("clash_penalty", 0)) or 0),
                    "diversity": float(row.get("diversity", 0) or 0),
                    "sequence_designability": float(row.get("sequence_designability", 0) or 0),
                }
                scores.append(CandidateScore(candidate_id=cid, model="boltzgen", path=str(f), total=weighted_score(metrics, weights), **metrics))
    return scores


def parse_rfd3_scores(output_dir: Union[str, Path], weights: Dict[str, float]) -> List[CandidateScore]:
    """Parse Foundry RF3 confidences or the harness bridge metrics CSV."""
    root = Path(output_dir)
    scores: List[CandidateScore] = []
    csv_files = list(root.glob("final_designs_metrics.csv")) + list(root.glob("**/final_designs_metrics.csv"))
    seen = set()
    for path in csv_files:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cid = row.get("design") or row.get("name") or path.stem
                metrics = {
                    "interface_confidence": float(row.get("iptm", row.get("interface_confidence", 0)) or 0),
                    "hotspot_contact": float(row.get("hotspot_contact", 0) or 0),
                    "binder_plddt": float(row.get("plddt", row.get("binder_plddt", 0)) or 0),
                    "clash_penalty": float(row.get("clash_penalty", row.get("clash", 0)) or 0),
                    "diversity": float(row.get("diversity", 0) or 0),
                    "sequence_designability": float(row.get("ranking_score", row.get("sequence_designability", 0)) or 0),
                }
                scores.append(CandidateScore(candidate_id=cid, model="rfd3", path=str(row.get("path") or path), total=weighted_score(metrics, weights), **metrics))
    if scores:
        return scores
    from binderloop.models.rfd3_step_bridge import discover_fold_confidences, parse_rf3_confidence
    for conf in discover_fold_confidences(root):
        row = parse_rf3_confidence(conf)
        metrics = {
            "interface_confidence": float(row.get("iptm") or 0),
            "hotspot_contact": 0.0,
            "binder_plddt": float(row.get("plddt") or 0),
            "clash_penalty": 0.0,
            "diversity": 0.0,
            "sequence_designability": float(row.get("ranking_score") or row.get("ptm") or 0),
        }
        scores.append(CandidateScore(candidate_id=row.get("design") or conf.stem, model="rfd3", path=str(conf), total=weighted_score(metrics, weights), **metrics))
    return scores


def parse_odesign_scores(output_dir: Union[str, Path], weights: Dict[str, float]) -> List[CandidateScore]:
    """Collect ODesign prediction CIFs and assign mock metrics until real scoring is added."""
    root = Path(output_dir)
    scores: List[CandidateScore] = []
    for cif in root.glob("**/predictions/*.cif"):
        metrics = mock_metrics(str(cif))
        scores.append(CandidateScore(candidate_id=cif.stem, model="odesign", path=str(cif), total=weighted_score(metrics, weights), **metrics))
    return scores


def write_failure_tags(scores: Iterable[CandidateScore], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in scores:
            metrics = s.__dict__.copy()
            tags = classify_failures(metrics)
            f.write({"candidate_id": s.candidate_id, "model": s.model, "tags": tags}.__repr__() + "\n")


def write_mock_scores(candidate_ids: List[str], path: Union[str, Path], weights: Dict[str, float]) -> List[CandidateScore]:
    scores = []
    for cid in candidate_ids:
        metrics = mock_metrics(cid)
        scores.append(CandidateScore(candidate_id=cid, model="mock", path="", total=weighted_score(metrics, weights), **metrics))
    write_scores(scores, path)
    return scores
