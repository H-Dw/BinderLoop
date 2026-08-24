#!/usr/bin/env python3
"""Deterministic phenotype clustering for prompt cards."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.analysis.candidate_clusters import (
    aggregate_candidate_phenotypes,
    compact_cluster_cards,
)


def _al_examples():
    return {
        "current_round": {
            "strict_positive_examples": [
                {"id": "p1", "label": "strict_positive", "metrics": {"design_to_target_iptm": 0.55, "min_design_to_target_pae": 8.0, "design_ptm": 0.8}, "tags": ["pass_compute_gate"], "raw_metrics": {"num_design": 70}},
            ],
            "near_miss_examples": [
                {"id": "n1", "label": "near_miss", "metrics": {"design_to_target_iptm": 0.487, "min_design_to_target_pae": 9.5, "design_ptm": 0.74}, "tags": ["hotspot_miss"], "raw_metrics": {"num_design": 70}},
                {"id": "n2", "label": "near_miss", "metrics": {"design_to_target_iptm": 0.500, "min_design_to_target_pae": 9.8, "design_ptm": 0.73}, "tags": ["hotspot_miss"], "raw_metrics": {"num_design": 70}},
                {"id": "n3", "label": "near_miss", "metrics": {"design_to_target_iptm": 0.492, "min_design_to_target_pae": 9.1, "design_ptm": 0.72}, "tags": ["hotspot_miss"], "raw_metrics": {"num_design": 70}},
            ],
            "other_negative_examples": [
                {"id": "o1", "label": "other_negative", "metrics": {"design_to_target_iptm": 0.21, "min_design_to_target_pae": 14.0}, "tags": ["hotspot_miss", "binding_pose_failure"], "raw_metrics": {"num_design": 70}},
                {"id": "o2", "label": "other_negative", "metrics": {"design_to_target_iptm": 0.22, "min_design_to_target_pae": 13.5}, "tags": ["hotspot_miss", "binding_pose_failure"], "raw_metrics": {"num_design": 70}},
                {"id": "o3", "label": "other_negative", "metrics": {"design_to_target_iptm": 0.18, "min_design_to_target_pae": 16.0}, "tags": ["folding_failure"], "raw_metrics": {"num_design": 90}},
            ],
        }
    }


class CandidateClusterTests(unittest.TestCase):
    def test_al_partitions_never_merge(self):
        payload = aggregate_candidate_phenotypes(
            round_id=0,
            active_learning_examples=_al_examples(),
        )
        labels = {item["al_label"] for item in payload["clusters"]}
        self.assertEqual(labels, {"strict_positive", "near_miss", "other_negative"})
        near = [item for item in payload["clusters"] if item["al_label"] == "near_miss"]
        self.assertEqual(len(near), 1)
        self.assertEqual(near[0]["size"], 3)
        self.assertLessEqual(len(near[0]["representatives"]), 2)

    def test_compact_cards_drop_leaves(self):
        payload = aggregate_candidate_phenotypes(round_id=1, active_learning_examples=_al_examples())
        self.assertIn("leaves", payload)
        cards = compact_cluster_cards(payload)
        self.assertNotIn("leaves", cards)
        rendered = str(cards)
        self.assertIn("cluster_id", rendered)
        self.assertEqual(cards["leaf_count"], payload["leaf_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
