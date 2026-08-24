#!/usr/bin/env python3
"""Tests for finite-catalog probability parameter decisions."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.config import load_config
from binderloop.parameter_decision import (
    HOLD_CURRENT,
    DecisionThresholds,
    ParameterCandidate,
    ParameterDecisionSpec,
    decide_parameters,
    map_proposed_to_final,
    normalize_probabilities,
    remove_invalid_and_renormalize,
    decide_parameter_distribution,
)


class ParameterDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = ParameterCandidate(0.001, 0.6, 0.6)
        self.b = ParameterCandidate(0.003, 0.7, 0.8)
        self.catalog = (self.a, self.b)

    def test_default_catalog_and_thresholds(self) -> None:
        spec = ParameterDecisionSpec()
        self.assertEqual(len(spec.catalog), 5 * 4 * 3)
        self.assertIn(ParameterCandidate(0.05, 0.9, 1.0), spec.catalog)
        self.assertEqual(spec.thresholds, DecisionThresholds(0.65, 0.20, 0.75))

    def test_probability_normalization(self) -> None:
        result = normalize_probabilities({self.a: 2.0, self.b: 1.0, HOLD_CURRENT: 1.0})
        self.assertAlmostEqual(sum(result.values()), 1.0)
        self.assertEqual(result[self.a], 0.5)
        with self.assertRaises(ValueError):
            normalize_probabilities({self.a: -1.0})

    def test_invalid_candidates_are_removed_then_renormalized(self) -> None:
        invalid = ParameterCandidate(0.002, 0.65, 0.7)
        result = remove_invalid_and_renormalize(
            {self.a: 2.0, invalid: 100.0, HOLD_CURRENT: 2.0}, self.catalog
        )
        self.assertEqual(result, {self.a: 0.5, HOLD_CURRENT: 0.5})

    def test_conservative_gates_hold_uncertain_proposal(self) -> None:
        result = decide_parameters({self.a: 0.60, self.b: 0.40}, self.catalog)
        self.assertEqual(result.proposed, self.a)
        self.assertEqual(result.final, HOLD_CURRENT)
        self.assertTrue(result.held)

    def test_confident_proposal_maps_to_exact_catalog_member(self) -> None:
        equal_but_distinct = ParameterCandidate(0.001, 0.6, 0.6)
        result = decide_parameters({equal_but_distinct: 0.85, self.b: 0.15}, self.catalog)
        self.assertIs(result.final, self.a)
        self.assertIn(result.final, self.catalog)
        self.assertFalse(result.held)

    def test_non_catalog_proposal_is_rejected_not_averaged(self) -> None:
        midpoint = ParameterCandidate(0.002, 0.65, 0.7)
        with self.assertRaisesRegex(ValueError, "exact catalog member"):
            map_proposed_to_final(midpoint, self.catalog)

    def test_hold_is_an_implicit_probability_state(self) -> None:
        result = decide_parameters({HOLD_CURRENT: 0.8, self.a: 0.2}, self.catalog)
        self.assertEqual(result.proposed, HOLD_CURRENT)
        self.assertEqual(result.final, HOLD_CURRENT)

    def test_scalar_candidates_filter_bounds_and_inertia_without_clamp(self) -> None:
        result = decide_parameter_distribution(
            {"LOW": 0.8, "HIGH": 0.2}, labels_to_values={"LOW": 0.6, "HIGH": 0.9},
            candidates=[0.6, 0.7, 0.8, 0.9], current=0.8,
            bounds={"min": 0.6, "max": 0.9, "max_step_abs": 0.15},
        )
        self.assertEqual(result["final"], 0.9)
        self.assertNotIn(0.6, result["eligible_candidates"])
        self.assertNotEqual(result["final"], 0.65)

    def test_scalar_capability_auto_holds_required_fails(self) -> None:
        auto = decide_parameter_distribution({}, labels_to_values={}, candidates=[0.001], current=0.001, capability_status="indeterminate", capability_mode="auto")
        self.assertEqual(auto["final"], HOLD_CURRENT)
        with self.assertRaisesRegex(RuntimeError, "required logprobs"):
            decide_parameter_distribution({}, labels_to_values={}, candidates=[0.001], current=0.001, capability_status="unsupported", capability_mode="required")

    def test_legacy_yaml_gets_defaults(self) -> None:
        text = """schema_version: 1
owner:
  task_hard_constraints:
    target_structure_path: target.cif
    binder_length_range: [60, 80]
    num_designs: 4
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.yaml"
            path.write_text(text)
            cfg = load_config(path)
        self.assertEqual(cfg.owner.parameter_decision.alpha_candidates, (0.001, 0.003, 0.009, 0.027, 0.05))
        self.assertEqual(cfg.owner.parameter_decision.thresholds.top_probability, 0.65)

    def test_formal_structured_tasks_declare_conservative_defaults(self) -> None:
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "configs").glob("*structured_task*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                spec = load_config(path).owner.parameter_decision
                self.assertEqual(spec.alpha_candidates, (0.001, 0.003, 0.009, 0.027, 0.05))
                self.assertEqual(spec.noise_scale_candidates, (0.6, 0.7, 0.8, 0.9))
                self.assertEqual(spec.step_scale_candidates, (0.6, 0.8, 1.0))
                self.assertEqual(spec.thresholds, DecisionThresholds(0.65, 0.20, 0.75))


if __name__ == "__main__":
    unittest.main()
