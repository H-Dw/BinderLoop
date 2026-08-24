
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Union
import csv


@dataclass
class CandidateScore:
    candidate_id: str
    model: str
    path: str
    interface_confidence: float = 0.0
    hotspot_contact: float = 0.0
    binder_plddt: float = 0.0
    clash_penalty: float = 0.0
    diversity: float = 0.0
    sequence_designability: float = 0.0
    total: float = 0.0


def weighted_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    score = 0.0
    for k, w in weights.items():
        v = float(metrics.get(k, 0.0))
        if k.endswith("penalty"):
            score -= w * v
        else:
            score += w * v
    return score


def rank_candidates(candidates: Iterable[CandidateScore]) -> List[CandidateScore]:
    return sorted(candidates, key=lambda c: c.total, reverse=True)


def write_scores(candidates: List[CandidateScore], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(candidates[0]).keys()) if candidates else ["candidate_id"])
        writer.writeheader()
        for c in candidates:
            writer.writerow(asdict(c))
