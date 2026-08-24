import json
import tempfile
import unittest
from pathlib import Path

from binderloop.strategy_governance import (
    BindingSiteResolution,
    LengthPolicyState,
    compare_matched_hotspot_outcome,
    retract_unbeneficial_expanded_hotspots,
)
from binderloop.memory import ExperimentMemoryStore


class DurableLengthPolicyTest(unittest.TestCase):
    def test_rollback_persists_across_memory_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentMemoryStore(Path(tmp))
            memory = store.load()
            memory.length_policy_state = LengthPolicyState.initialize([60, 70], [50, 100])
            memory.length_policy_state.record_round(0, [60, 70])
            memory.length_policy_state.record_round(1, [80, 90])
            selected = memory.length_policy_state.select_next(
                round_id=1, recommended_lengths=[90, 100], branch_action="replay_best",
                branch_from_round=0, recommendation={"direction": "longer"},
            )
            self.assertEqual(selected, [60, 70])
            store.save(memory)
            reloaded = store.load()
            self.assertEqual(reloaded.length_policy_state.current_lengths, [60, 70])
            self.assertEqual(reloaded.length_policy_state.round_lengths["0"], [60, 70])
            self.assertEqual(reloaded.length_policy_state.last_transition["reason"], "restored_branch_baseline")


class BindingSiteGovernanceTest(unittest.TestCase):
    def setUp(self):
        self.original = [
            {"chain": {"id": "E", "binding": "153,154"}},
            {"chain": {"id": "E", "not_binding": "120,121"}},
        ]

    def test_resolution_has_immutable_primary_negative_and_digest(self):
        resolved = BindingSiteResolution.rebuild(
            primary_residues=["E:153"], original_binding_types=self.original,
            expanded_residues=["E:154", "E:160"],
        )
        self.assertEqual(resolved.primary, ["E:153", "E:154"])
        self.assertEqual(resolved.expanded, ["E:160"])
        self.assertEqual(resolved.negative, ["E:120", "E:121"])
        self.assertTrue(resolved.semantic_digest)
        self.assertTrue(any(
            item.get("chain", {}).get("not_binding") == "120,121"
            for item in resolved.effective_binding_types
        ))

    def test_matched_control_detects_credible_benefit(self):
        control = [{
            "match_id": "L80", "design_to_target_iptm": .40,
            "min_design_to_target_pae": 11.0, "primary_coverage": .45,
            "binding_pose_score": .40, "strict_positive": False,
        }]
        expanded = [{
            "match_id": "L80", "design_to_target_iptm": .46,
            "min_design_to_target_pae": 8.0, "primary_coverage": .50,
            "binding_pose_score": .46, "strict_positive": True,
        }]
        outcome = compare_matched_hotspot_outcome(control, expanded)
        self.assertTrue(outcome.credible_benefit)
        self.assertEqual(outcome.matched_pairs, 1)
        self.assertAlmostEqual(outcome.deltas["interface_pae"], 3.0)

    def test_retracts_only_new_expanded_binding_without_benefit(self):
        previous = BindingSiteResolution.rebuild(
            primary_residues=["E:153"], original_binding_types=self.original,
            expanded_residues=["E:160"],
        )
        current = BindingSiteResolution.rebuild(
            primary_residues=["E:153"], original_binding_types=self.original,
            expanded_residues=["E:160", "E:165"],
        )
        control = [{"match_id": "x", "design_to_target_iptm": .45, "min_design_to_target_pae": 8,
                    "primary_coverage": .6, "binding_pose_score": .5, "strict_positive": True}]
        expanded = [{"match_id": "x", "design_to_target_iptm": .44, "min_design_to_target_pae": 9,
                     "primary_coverage": .58, "binding_pose_score": .48, "strict_positive": True}]
        outcome = compare_matched_hotspot_outcome(control, expanded)
        resolved, retracted = retract_unbeneficial_expanded_hotspots(previous, current, outcome)
        self.assertEqual(retracted, ["E:165"])
        self.assertEqual(resolved.expanded, ["E:160"])
        self.assertEqual(resolved.retracted_expanded, ["E:165"])
        self.assertEqual(resolved.primary, previous.primary)
        self.assertEqual(resolved.negative, ["E:120", "E:121"])
        self.assertTrue(any(
            item.get("chain", {}).get("not_binding") == "120,121"
            for item in resolved.effective_binding_types
        ))


if __name__ == "__main__":
    unittest.main()
