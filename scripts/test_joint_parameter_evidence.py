import random
import unittest

from binderloop.parameter_decision import (
    JointParameterEvidence,
    JointSelectionPolicy,
    ParameterCandidate,
    ParameterDecisionSpec,
    deterministic_sampler_states,
    joint_candidate_scores,
    joint_parameter_evidence_from_rounds,
)


def candidate(alpha: float, noise: float = 10.0) -> ParameterCandidate:
    return ParameterCandidate(alpha=alpha, noise_scale=noise)


class JointParameterEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ParameterDecisionSpec(
            alpha_candidates=(1.0, 2.0, 3.0),
            noise_scale_candidates=(10.0,),
            sampler_axes=("alpha", "noise_scale"),
        )

    def test_no_evidence_preserves_seeded_shuffle_order(self) -> None:
        eligible = [item for item in self.spec.catalog if item != candidate(1.0)]
        expected = list(eligible)
        random.Random(17).shuffle(expected)

        actual = deterministic_sampler_states(
            self.spec,
            current={"alpha": 1.0, "noise_scale": 10.0},
            count=10,
            seed=17,
        )

        self.assertEqual(tuple(expected), actual)

    def test_joint_policy_activation_mode_is_validated(self) -> None:
        self.assertEqual("off", self.spec.joint_evidence_fallback_mode)
        normalized = ParameterDecisionSpec(
            alpha_candidates=(1.0,),
            noise_scale_candidates=(10.0,),
            sampler_axes=("alpha", "noise_scale"),
            joint_evidence_fallback_mode=" SHADOW ",
        )
        self.assertEqual("shadow", normalized.joint_evidence_fallback_mode)
        with self.assertRaisesRegex(ValueError, "off, shadow, active"):
            ParameterDecisionSpec(
                alpha_candidates=(1.0,),
                noise_scale_candidates=(10.0,),
                sampler_axes=("alpha", "noise_scale"),
                joint_evidence_fallback_mode="automatic",
            )

    def test_exploitation_requires_repeated_matched_controls(self) -> None:
        control = candidate(1.0)
        supported = candidate(2.0)
        unmatched = candidate(3.0)
        evidence = []
        for index in range(2):
            group = f"matched:{index}"
            evidence.extend((
                JointParameterEvidence(control, 1, 10, f"control:{index}", group, True, 1.0),
                JointParameterEvidence(supported, 8, 10, f"supported:{index}", group, False, 1.0),
                JointParameterEvidence(unmatched, 10, 10, f"unmatched:{index}", "", False, 1.0),
            ))
        policy = JointSelectionPolicy(
            exploitation_fraction=1.0,
            uncertainty_weight=0.0,
            novelty_weight=0.0,
            diversity_weight=0.0,
            cost_weight=0.0,
        )

        scores = joint_candidate_scores((supported, unmatched), evidence, policy=policy)
        selected = deterministic_sampler_states(
            self.spec,
            current=control.as_dict(),
            count=1,
            seed=7,
            evidence=evidence,
            policy=policy,
        )

        self.assertTrue(scores[supported].supported)
        self.assertFalse(scores[unmatched].supported)
        self.assertEqual((supported,), selected)

    def test_unmatched_rows_do_not_change_exploitation_posterior(self) -> None:
        challenger = candidate(2.0)
        matched = []
        for index in range(2):
            group = f"matched:{index}"
            matched.extend((
                JointParameterEvidence(candidate(1.0), 0, 10, f"control:{index}", group, True),
                JointParameterEvidence(challenger, 9, 10, f"challenger:{index}", group, False),
            ))
        with_unmatched = matched + [
            JointParameterEvidence(challenger, 0, 100, "unmatched", "", False),
        ]

        matched_score = joint_candidate_scores((challenger,), matched)[challenger]
        combined_score = joint_candidate_scores((challenger,), with_unmatched)[challenger]

        self.assertTrue(combined_score.supported)
        self.assertEqual(matched_score.matched_posterior_mean, combined_score.matched_posterior_mean)
        self.assertEqual(
            matched_score.matched_posterior_uncertainty,
            combined_score.matched_posterior_uncertainty,
        )
        self.assertNotEqual(matched_score.posterior_mean, combined_score.posterior_mean)

    def test_unmatched_rows_cannot_satisfy_support_replicates(self) -> None:
        challenger = candidate(2.0)
        evidence = (
            JointParameterEvidence(candidate(1.0), 0, 4, "control", "matched", True),
            JointParameterEvidence(challenger, 4, 4, "matched", "matched", False),
            JointParameterEvidence(challenger, 4, 4, "unmatched", "", False),
        )
        policy = JointSelectionPolicy(
            minimum_matched_controls=1,
            minimum_conservative_effect=-1.0,
        )

        score = joint_candidate_scores((challenger,), evidence, policy=policy)[challenger]

        self.assertEqual(2, score.replicates)
        self.assertEqual(1, score.matched_replicates)
        self.assertFalse(score.supported)

    def test_uncertainty_bonus_can_be_disabled_for_greedy_ranking(self) -> None:
        observed = candidate(2.0)
        unseen = candidate(3.0)
        evidence = (JointParameterEvidence(observed, 9, 10, "observed", "", False, 1.0),)
        greedy_policy = JointSelectionPolicy(
            exploitation_fraction=0.0,
            uncertainty_weight=0.0,
            novelty_weight=0.0,
            diversity_weight=0.0,
            cost_weight=0.0,
        )

        selected = deterministic_sampler_states(
            self.spec,
            current=candidate(1.0).as_dict(),
            count=1,
            seed=9,
            evidence=evidence,
            policy=greedy_policy,
        )

        self.assertEqual((observed,), selected)
        self.assertNotEqual((unseen,), selected)

    def test_replicate_identity_cannot_be_reused_for_another_candidate(self) -> None:
        evidence = (
            JointParameterEvidence(candidate(2.0), 3, 4, "duplicate", "round:1", False),
            JointParameterEvidence(candidate(3.0), 4, 4, "duplicate", "round:1", False),
        )

        with self.assertRaisesRegex(ValueError, "conflicting joint evidence"):
            joint_candidate_scores((candidate(2.0), candidate(3.0)), evidence)

    def test_cost_budget_and_axis_bounds_filter_before_selection(self) -> None:
        expensive = candidate(2.0)
        affordable = candidate(3.0)
        evidence = (
            JointParameterEvidence(expensive, 9, 10, "expensive", "", False, 5.0),
            JointParameterEvidence(affordable, 5, 10, "affordable", "", False, 1.0),
        )
        policy = JointSelectionPolicy(
            exploitation_fraction=0.0,
            max_total_cost=1.5,
            uncertainty_weight=0.0,
            novelty_weight=0.0,
            diversity_weight=0.0,
            cost_weight=0.0,
        )

        selected = deterministic_sampler_states(
            self.spec,
            current=candidate(1.0).as_dict(),
            count=3,
            seed=1,
            bounds={"alpha": {"max": 3.0}},
            evidence=evidence,
            policy=policy,
        )

        self.assertEqual((affordable,), selected)

    def test_unaffordable_exploitation_degrades_to_affordable_exploration(self) -> None:
        supported = candidate(2.0)
        exploratory = candidate(3.0)
        evidence = []
        for index in range(2):
            group = f"matched:{index}"
            evidence.extend((
                JointParameterEvidence(candidate(1.0), 0, 10, f"control:{index}", group, True, 1.0),
                JointParameterEvidence(supported, 9, 10, f"supported:{index}", group, False, 5.0),
            ))
        evidence.append(JointParameterEvidence(exploratory, 1, 2, "explore", "", False, 1.0))
        policy = JointSelectionPolicy(
            exploitation_fraction=1.0,
            max_total_cost=1.5,
            uncertainty_weight=0.0,
            novelty_weight=0.0,
            diversity_weight=0.0,
            cost_weight=0.0,
        )

        selected = deterministic_sampler_states(
            self.spec,
            current=candidate(1.0).as_dict(),
            count=1,
            seed=11,
            evidence=evidence,
            policy=policy,
        )

        self.assertEqual((exploratory,), selected)

    def test_batch_diversity_uses_joint_catalog_distance(self) -> None:
        anchor = candidate(1.0)
        evidence = (JointParameterEvidence(anchor, 1, 2, "anchor", "", True, 1.0),)
        diversity_only = JointSelectionPolicy(
            exploitation_fraction=0.0,
            uncertainty_weight=0.0,
            novelty_weight=0.0,
            diversity_weight=1.0,
            cost_weight=0.0,
        )

        selected = deterministic_sampler_states(
            self.spec,
            count=1,
            seed=3,
            evidence=evidence,
            policy=diversity_only,
            selected=(anchor,),
        )

        self.assertEqual((candidate(3.0),), selected)

    def test_previously_selected_state_is_not_emitted_again(self) -> None:
        prior = candidate(2.0)
        evidence = (JointParameterEvidence(prior, 9, 10, "prior", "", False, 1.0),)

        selected = deterministic_sampler_states(
            self.spec,
            current=candidate(1.0).as_dict(),
            count=2,
            seed=5,
            evidence=evidence,
            selected=(prior,),
        )

        self.assertNotIn(prior, selected)
        self.assertEqual((candidate(3.0),), selected)

    def test_round_extractor_keeps_only_complete_sampler_and_control_vectors(self) -> None:
        rounds = [{
            "round_id": 4,
            "jobs": [
                {"params": {"arm_id": "baseline_hold", "alpha": 1.0, "noise_scale": 10.0}},
                {"params": {"arm_id": "sampler_explore_fallback_00", "final_parameter_state": {"alpha": 2.0, "noise_scale": 10.0}, "random_sampler_fallback": True}},
                {"params": {"arm_id": "sampler_explore_fallback_01", "final_parameter_state": {"alpha": 3.0, "noise_scale": 10.0}, "random_sampler_fallback": True}},
                {"params": {"arm_id": "unrelated", "alpha": 3.0, "noise_scale": 10.0}},
            ],
            "arm_outcomes": [
                {"arm_id": "baseline_hold", "status": "closed", "requested_budget": 4, "completed_budget": 4, "trials": 4, "successes": 1, "is_baseline": True},
                {"arm_id": "sampler_explore_fallback_00", "status": "closed", "requested_budget": 4, "completed_budget": 4, "trials": 4, "successes": 3},
                {"arm_id": "sampler_explore_fallback_01", "status": "incomplete", "requested_budget": 4, "completed_budget": 2, "trials": 2, "successes": 2},
                {"arm_id": "unrelated", "status": "closed", "requested_budget": 4, "completed_budget": 4, "trials": 4, "successes": 4},
            ],
        }]

        extracted = joint_parameter_evidence_from_rounds(rounds, spec=self.spec)

        self.assertEqual(2, len(extracted))
        self.assertEqual({candidate(1.0), candidate(2.0)}, {item.candidate for item in extracted})
        self.assertEqual({"round:4"}, {item.comparison_group for item in extracted})
        self.assertEqual(1, sum(item.is_control for item in extracted))

    def test_round_extractor_ignores_control_only_round(self) -> None:
        rounds = [{
            "round_id": 5,
            "jobs": [{"params": {
                "arm_id": "baseline_hold",
                "alpha": 1.0,
                "noise_scale": 10.0,
            }}],
            "arm_outcomes": [{
                "arm_id": "baseline_hold",
                "status": "closed",
                "requested_budget": 4,
                "completed_budget": 4,
                "trials": 4,
                "successes": 2,
                "is_baseline": True,
            }],
        }]

        self.assertEqual((), joint_parameter_evidence_from_rounds(rounds, spec=self.spec))

    def test_round_extractor_does_not_reuse_arm_aggregate_across_branches(self) -> None:
        rounds = [{
            "round_id": 6,
            "jobs": [
                {"params": {"arm_id": "baseline_hold", "alpha": 1.0, "noise_scale": 10.0}},
                {"params": {
                    "arm_id": "sampler_shared",
                    "logical_branch_id": "branch-a",
                    "final_parameter_state": {"alpha": 2.0, "noise_scale": 10.0},
                    "random_sampler_fallback": True,
                }},
                {"params": {
                    "arm_id": "sampler_shared",
                    "logical_branch_id": "branch-b",
                    "final_parameter_state": {"alpha": 3.0, "noise_scale": 10.0},
                    "random_sampler_fallback": True,
                }},
            ],
            "arm_outcomes": [
                {"arm_id": "baseline_hold", "status": "closed", "requested_budget": 4, "completed_budget": 4, "trials": 4, "successes": 1, "is_baseline": True},
                {"arm_id": "sampler_shared", "status": "closed", "requested_budget": 4, "completed_budget": 4, "trials": 4, "successes": 4},
            ],
        }]

        self.assertEqual((), joint_parameter_evidence_from_rounds(rounds, spec=self.spec))

    def test_round_extractor_does_not_reuse_branchless_arm_aggregate(self) -> None:
        rounds = [{
            "round_id": 8,
            "jobs": [
                {"params": {"arm_id": "baseline_hold", "alpha": 1.0, "noise_scale": 10.0}},
                {"params": {
                    "arm_id": "sampler_shared",
                    "final_parameter_state": {"alpha": 2.0, "noise_scale": 10.0},
                    "random_sampler_fallback": True,
                }},
                {"params": {
                    "arm_id": "sampler_shared",
                    "final_parameter_state": {"alpha": 3.0, "noise_scale": 10.0},
                    "random_sampler_fallback": True,
                }},
            ],
            "arm_outcomes": [
                {"arm_id": "baseline_hold", "status": "closed", "requested_budget": 4, "completed_budget": 4, "trials": 4, "successes": 1, "is_baseline": True},
                {"arm_id": "sampler_shared", "status": "closed", "requested_budget": 4, "completed_budget": 4, "trials": 4, "successes": 4},
            ],
        }]

        self.assertEqual((), joint_parameter_evidence_from_rounds(rounds, spec=self.spec))

    def test_round_extractor_rejects_nonterminal_rows_without_status_or_budget(self) -> None:
        rounds = [{
            "round_id": 7,
            "jobs": [
                {"params": {"arm_id": "baseline_hold", "alpha": 1.0, "noise_scale": 10.0}},
                {"params": {
                    "arm_id": "sampler_explore_fallback_00",
                    "final_parameter_state": {"alpha": 2.0, "noise_scale": 10.0},
                    "random_sampler_fallback": True,
                }},
            ],
            "arm_outcomes": [
                {"arm_id": "baseline_hold", "trials": 4, "successes": 1, "is_baseline": True},
                {"arm_id": "sampler_explore_fallback_00", "trials": 4, "successes": 4},
            ],
        }]

        self.assertEqual((), joint_parameter_evidence_from_rounds(rounds, spec=self.spec))


if __name__ == "__main__":
    unittest.main()
