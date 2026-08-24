#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.analysis.parsers import write_mock_scores, write_failure_tags
from binderloop.analysis.scoring import rank_candidates

weights = {
    "interface_confidence": 0.30,
    "hotspot_contact": 0.25,
    "binder_plddt": 0.15,
    "clash_penalty": 0.15,
    "diversity": 0.10,
    "sequence_designability": 0.05,
}
out = Path("outputs/mock_test")
scores = write_mock_scores([f"candidate_{i}" for i in range(10)], out / "scores.csv", weights)
ranked = rank_candidates(scores)
assert ranked[0].total >= ranked[-1].total
write_failure_tags(ranked, out / "failure_tags.jsonl")
print(f"OK: wrote {out / 'scores.csv'} and {out / 'failure_tags.jsonl'}")
