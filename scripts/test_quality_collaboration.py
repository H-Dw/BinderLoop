#!/usr/bin/env python3
"""Regression tests for performance-triggered quality collaboration."""

import json
import sys
import unittest
from unittest import mock
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.binder_quality_collaboration_agent import (
    BinderQualityCollaborationAgent,
    QualityCollaborationController,
)
from binderloop.config import QualityCollaborationSpec
from binderloop.memory import ExperimentMemory, RoundRecord


class FakeCollaborationLLM:
    def __init__(self):
        self.calls = []

    def available(self):
        return True

    def create_chat_completion(self, *, messages, **kwargs):
        system = messages[0]["content"]
        self.calls.append({"system": system, **kwargs})
        if "SuccessMechanismAgent" in system:
            payload = {"findings": [{"finding_id": "P1", "statement": "positive evidence exists", "scope": "whole_binder", "signal": "reusable", "evidence_ids": ["R2:METRICS"], "counterevidence_ids": [], "confidence": 0.8}]}
        elif "FailureMechanismAgent" in system:
            payload = {"findings": [{"finding_id": "N1", "statement": "high PAE is dominant", "scope": "population", "failure_type": "pae", "repair_family": "target_context", "evidence_ids": ["R2:METRICS"], "counterevidence_ids": [], "confidence": 0.8}]}
        elif "TrajectoryMemoryAgent" in system:
            payload = {"findings": []}
        elif "PhysicsDebateManager" in system:
            payload = {"accepted_finding_ids": ["P1", "N1"], "rejected_finding_ids": [], "strategy_intents": [], "uncertainties": []}
        elif "Revise only" in system:
            payload = {
                "revised_claims": [],
                "request_answer": "verified",
                "remaining_uncertainty": "",
            }
        else:
            payload = {
                "overall_assessment": "Current round quality assessed.",
                "current_round_facts": {
                    "round_id": 2,
                    "best_iptm": 0.4,
                    "success_count": 1,
                    "reward": 0.4,
                },
                "high_quality_modules": [],
                "low_quality_modules": [],
                "causal_factors": [],
                "next_round_guidance": [{
                    "action": "keep executable settings",
                    "evidence_ids": ["R2:METRICS"],
                    "parameter_or_constraint_change": "none",
                    "config_parameter_changes": {
                        "alpha": 0.001,
                        "additional_filters": ["iptm>0.4"],
                    },
                    "expected_signal": "stable reward",
                    "risk": "none",
                }],
                "debate_audit": {
                    "accepted_claim_ids": ["P1", "N1"],
                    "rejected_claim_ids": [],
                    "revised_claim_ids": [],
                    "resolved_conflicts": [],
                },
                "uncertainties": [],
            }
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": json.dumps(payload)},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }


class QualityModeControllerTest(unittest.TestCase):
    def setUp(self):
        self.spec = QualityCollaborationSpec(enabled=True, max_revisions=1)
        self.memory = ExperimentMemory(
            experiment_id="test",
            target={},
            round_metrics=[
                {"round_id": 0, "reward": 0.5, "execution_failed": False},
            ],
        )

    def test_regression_stays_multi_until_historical_best_is_reached(self):
        regressed = QualityCollaborationController.decide(
            self.memory,
            {"round_id": 1, "reward": 0.4, "execution_failed": False},
            self.spec,
        )
        self.assertEqual(regressed.mode, "multi")
        self.assertTrue(regressed.active)
        self.assertAlmostEqual(regressed.recovery_target_reward, 0.485)

        self.memory.round_metrics.append({
            "round_id": 1,
            "reward": 0.4,
            "execution_failed": False,
        })
        partial = QualityCollaborationController.decide(
            self.memory,
            {"round_id": 2, "reward": 0.484, "execution_failed": False},
            self.spec,
        )
        self.assertEqual(partial.mode, "multi")
        self.assertTrue(partial.active)

        self.memory.round_metrics.append({
            "round_id": 2,
            "reward": 0.49,
            "execution_failed": False,
        })
        recovered = QualityCollaborationController.decide(
            self.memory,
            {"round_id": 3, "reward": 0.5, "execution_failed": False},
            self.spec,
        )
        self.assertEqual(recovered.mode, "single")
        self.assertFalse(recovered.active)


    def test_zero_filter_pass_with_unfiltered_evidence_forces_multi(self):
        decision = QualityCollaborationController.decide(
            self.memory,
            {"round_id": 1, "reward": 0.0, "execution_failed": False},
            self.spec,
            signals={"zero_filter_pass_with_unfiltered_evidence": True},
        )
        self.assertEqual(decision.mode, "multi")
        self.assertTrue(decision.active)
        self.assertTrue(any(item["code"] == "zero_filter_pass_review" for item in decision.trigger_reasons))

    def test_execution_failure_preserves_active_state(self):
        QualityCollaborationController.decide(
            self.memory,
            {"round_id": 1, "reward": 0.4, "execution_failed": False},
            self.spec,
        )
        failed = QualityCollaborationController.decide(
            self.memory,
            {
                "round_id": 2,
                "reward": 0.0,
                "execution_failed": True,
            },
            self.spec,
        )
        self.assertEqual(failed.mode, "multi")
        self.assertTrue(self.memory.quality_collaboration_state["active"])


class CollaborationAgentTest(unittest.TestCase):
    def test_deterministic_assembly_avoids_fixed_revision_calls(self):
        llm = FakeCollaborationLLM()
        agent = BinderQualityCollaborationAgent(
            llm,
            max_revisions=1,
            max_api_calls=6,
        )
        memory = ExperimentMemory(
            experiment_id="test",
            target={},
            round_metrics=[
                {"round_id": 1, "reward": 0.5, "best_iptm": 0.5},
            ],
        )
        context = {
            "round_id": 2,
            "evaluation": {
                "total_candidates": 2,
                "success_count": 1,
                "failure_count": 1,
                "tag_counts": {
                    "pass_compute_gate": 1,
                    "primary_gate_high_pae": 1,
                },
                "metric_facts": {
                    "best_iptm": 0.4,
                    "harness_success_count": 1,
                },
                "top_candidates": [],
                "failed_examples": [],
            },
            "active_learning_examples": {
                "current_round": {
                    "positive_examples": [],
                    "hard_negative_examples": [],
                },
            },
            "structural_analysis": {"summaries": []},
            "current_config": {"alpha": 0.001},
            "reward": {
                "round_id": 2,
                "reward": 0.4,
                "best_iptm": 0.4,
                "success_count": 1,
                "execution_failed": False,
            },
        }
        result = agent.analyze(
            round_id=2,
            context=context,
            memory=memory,
            mode_decision={
                "mode": "multi",
                "current_reward": 0.4,
                "previous_reward": 0.5,
                "recovery_target_reward": 0.485,
            },
        )
        self.assertTrue(result.llm_used)
        self.assertEqual(result.raw["source"], "deterministic_collaboration_assembler")
        collaboration = result.raw["collaboration"]
        self.assertNotIn("targeted_revision", collaboration)
        self.assertTrue(collaboration["availability"]["manager"])
        self.assertEqual(len(collaboration["telemetry"]), 4)
        guidance = result.next_round_guidance[0]
        self.assertEqual(guidance["config_parameter_changes"], {})
        self.assertEqual(guidance["selection_policy"]["kind"], "cross_chain_heavy_atom_clash")
        self.assertTrue(all(
            call.get("thinking") == "low" for call in llm.calls
        ))

    def test_delivery_is_assembled_and_manager_gets_cited_registry_slice(self):
        llm = FakeCollaborationLLM()
        with tempfile.TemporaryDirectory() as tmp:
            agent = BinderQualityCollaborationAgent(llm, max_revisions=0, cache_dir=Path(tmp))
            memory = ExperimentMemory(experiment_id="delivery", target={})
            context = {
                "evaluation": {"metric_facts": {"best_iptm": 0.4}, "tag_counts": {}, "failed_examples": []},
                "active_learning_examples": {"current_round": {"positive_examples": [], "hard_negative_examples": []}},
                "structural_analysis": {"summaries": []}, "current_config": {"alpha": 0.001},
                "reward": {"round_id": 2, "reward": 0.4, "best_iptm": 0.4, "success_count": 1},
            }
            result = agent.analyze(round_id=2, context=context, memory=memory, mode_decision={"mode": "multi"})
            collab = result.raw["collaboration"]
            self.assertEqual(result.raw["source"], "deterministic_collaboration_assembler")
            self.assertEqual(collab["collaboration_grade"], "full")
            self.assertEqual(collab["authority"]["formal_analysis"], "deterministic_assembler")
            self.assertEqual(collab["manager_evidence_ids"], ["R2:METRICS"])
            first_calls = len(llm.calls)
            agent.analyze(round_id=2, context=context, memory=memory, mode_decision={"mode": "multi"})
            self.assertEqual(len(llm.calls), first_calls)
            self.assertTrue(any(x.get("cache_hit") for x in result.raw["collaboration"]["telemetry"]) or first_calls > 0)

    def test_learned_skill_routes_through_manager_without_biasing_specialists(self):
        llm = FakeCollaborationLLM()
        agent = BinderQualityCollaborationAgent(llm, max_revisions=0)
        memory = ExperimentMemory(experiment_id="learned-manager", target={})
        context = {
            "evaluation": {"metric_facts": {"best_iptm": 0.4}, "tag_counts": {}, "failed_examples": []},
            "active_learning_examples": {"current_round": {"positive_examples": [], "hard_negative_examples": []}},
            "structural_analysis": {"summaries": []},
            "current_config": {"alpha": 0.001},
            "reward": {"round_id": 2, "reward": 0.4, "best_iptm": 0.4, "success_count": 1},
            "active_skills": [{
                "id": "run-local-self-improvement",
                "type": "llm_reasoning",
                "priority": 900,
                "origin": "run_local_self_improvement",
                "guidance": ["[rule_1] preserve foldability while repairing interface"],
                "learned_rules": [{"rule_id": "rule_1", "strategy": "bounded repair"}],
            }],
        }
        result = agent.analyze(
            round_id=2,
            context=context,
            memory=memory,
            mode_decision={"mode": "multi"},
        )
        collaboration = result.raw["collaboration"]
        self.assertTrue(collaboration["availability"]["manager"])
        self.assertEqual(
            [item["role"] for item in collaboration["telemetry"][:3]],
            ["positive", "negative", "trajectory"],
        )
        self.assertEqual(len(collaboration["telemetry"]), 4)

    def test_split_specialist_wave_matches_analyze(self):
        memory = ExperimentMemory(experiment_id="split-wave", target={})
        context = {
            "round_id": 2,
            "evaluation": {
                "total_candidates": 2, "success_count": 1, "failure_count": 1,
                "tag_counts": {"pass_compute_gate": 1, "primary_gate_high_pae": 1},
                "metric_facts": {"best_iptm": 0.4, "harness_success_count": 1},
                "top_candidates": [], "failed_examples": [],
            },
            "active_learning_examples": {"current_round": {"positive_examples": [], "hard_negative_examples": []}},
            "structural_analysis": {"summaries": []},
            "current_config": {"alpha": 0.001},
            "reward": {"round_id": 2, "reward": 0.4, "best_iptm": 0.4, "success_count": 1, "execution_failed": False},
        }
        mode = {"mode": "multi", "current_reward": 0.4, "previous_reward": 0.5, "recovery_target_reward": 0.485}
        via_analyze = BinderQualityCollaborationAgent(FakeCollaborationLLM(), max_revisions=1, max_api_calls=6).analyze(
            round_id=2, context=context, memory=memory, mode_decision=mode,
        )
        agent = BinderQualityCollaborationAgent(FakeCollaborationLLM(), max_revisions=1, max_api_calls=6)
        batch = agent.prepare_specialists(round_id=2, context=context, memory=memory, mode_decision=mode)
        specialist_results = {}
        for role in batch.roles:
            _role, normalized, telemetry = agent.run_specialist(batch, role)
            specialist_results[role] = (normalized, telemetry)
        agent.absorb_specialist_results(batch, specialist_results)
        via_split = agent.assemble_with_manager(batch)
        self.assertEqual(via_analyze.raw["source"], via_split.raw["source"])
        self.assertEqual(via_analyze.overall_assessment, via_split.overall_assessment)
        self.assertEqual(len(via_analyze.raw["collaboration"]["telemetry"]), 4)
        self.assertEqual(len(via_split.raw["collaboration"]["telemetry"]), 4)
        self.assertEqual(
            [item["role"] for item in via_split.raw["collaboration"]["telemetry"][:3]],
            ["positive", "negative", "trajectory"],
        )

    def test_independent_specialists_overlap_and_manager_runs_after_join(self):
        memory = ExperimentMemory(experiment_id="parallel-specialists", target={})
        agent = BinderQualityCollaborationAgent(llm=FakeCollaborationLLM(), max_api_calls=6)
        barrier = threading.Barrier(3)
        completed = []
        def fake_call(*, role, telemetry, **kwargs):
            if role != "manager_deliberation":
                barrier.wait(timeout=2)
                time.sleep(0.02)
                completed.append(role)
                telemetry.append({"role": role, "ok": True})
                return {"findings": []}
            self.assertEqual(set(completed), {"positive", "negative", "trajectory"})
            telemetry.append({"role": role, "ok": True})
            return {"accepted_finding_ids": [], "rejected_finding_ids": [], "strategy_intents": [], "uncertainties": []}
        with mock.patch.object(agent, "_call", side_effect=fake_call):
            result = agent.analyze(round_id=2, context={"evaluation": {}, "active_learning_examples": {"current_round": {}}}, memory=memory, mode_decision={"mode": "multi"})
        roles = [item["role"] for item in result.raw["collaboration"]["telemetry"]]
        self.assertEqual(roles[:3], ["positive", "negative", "trajectory"])

    def test_packets_separate_strict_success_failures_and_trajectory_views(self):
        agent = BinderQualityCollaborationAgent(None)
        memory = ExperimentMemory(experiment_id="packets", target={})
        compact = {
            "evaluation": {"top_candidates": [{"candidate_id": "h1", "tags": ["pass_compute_gate"]}]},
            "active_learning_examples": {"current_round": {
                "strict_positive_examples": [{"candidate_id": "p1", "label": "strict_positive"}],
                "near_miss_examples": [{"candidate_id": "n1", "label": "near_miss", "label_reason": "high PAE"}],
                "other_negative_examples": [{"candidate_id": "o1", "label": "other_negative", "label_reason": "high PAE"}]}},
            "structural_analysis": {"summaries": []}, "current_config": {"alpha": 0.1}, "memory": {},
        }
        packets = agent._build_packets(round_id=3, compact=compact, memory=memory,
                                       mode_decision={}, round_outcome={})
        self.assertEqual(packets["positive"]["strict_metric_positives"][0]["evidence_id"], "R3:STRICT_POS:1")
        self.assertEqual(packets["positive"]["near_miss_boundary_examples"][0]["evidence_id"], "R3:NEAR_MISS:1")
        self.assertEqual(packets["positive"]["provisional_reference"], [])
        self.assertEqual(packets["positive"]["harness_successes"][0]["count"], 0)
        self.assertEqual(packets["negative"]["other_negative_examples"][0]["evidence_id"], "R3:OTHER_NEG:1")
        self.assertNotIn("candidate_id", json.dumps(packets))
        self.assertEqual(packets["physics"]["evidence_taxonomy"]["near_miss_count"], 1)
        self.assertEqual(set(packets["trajectory"]["cards"]), {"current", "previous", "historical_best", "same_config"})

    def test_near_miss_becomes_provisional_reference_without_success(self):
        agent = BinderQualityCollaborationAgent(None)
        memory = ExperimentMemory(experiment_id="provisional", target={})
        memory.rounds = [
            RoundRecord(
                round_id=1,
                jobs=[],
                evaluation={
                    "active_learning_examples": {
                        "current_round": {
                            "strict_positive_examples": [{
                                "candidate_id": "old_p",
                                "label": "strict_positive",
                                "metrics": {"design_to_target_iptm": 0.55},
                            }]
                        }
                    }
                },
                reward=0.6,
            )
        ]
        memory.round_metrics = [{"round_id": 1, "reward": 0.6}]
        compact = {
            "evaluation": {"top_candidates": []},
            "active_learning_examples": {"current_round": {
                "strict_positive_examples": [],
                "near_miss_examples": [{"candidate_id": "n1", "label": "near_miss"}],
                "other_negative_examples": [{"candidate_id": "o1", "label": "other_negative"}],
            }},
            "structural_analysis": {"summaries": []},
            "current_config": {},
            "memory": {},
        }
        packets = agent._build_packets(
            round_id=2, compact=compact, memory=memory,
            mode_decision={}, round_outcome={},
        )
        provisional = packets["positive"]["provisional_reference"][0]
        self.assertEqual(provisional["label"], "near_miss")
        self.assertEqual(provisional["evidence_role"], "provisional_reference")
        self.assertFalse(provisional["success_counted"])
        self.assertEqual(packets["positive"]["strict_metric_positives"], [])
        self.assertEqual(packets["specialist_activation_audit"]["provisional_reference_count"], 1)
        cards = packets["trajectory"]["cards"]
        self.assertEqual(cards["current"]["strict_positive_population"]["strict_positive_count"], 0)
        self.assertEqual(cards["previous"]["strict_positive_population"]["strict_positive_count"], 1)

    def test_deterministic_arbitration_reports_no_conflict(self):
        from binderloop.agents.binder_quality_collaboration_agent import _deterministic_arbitrate
        packets = {"physics": {"rules": [{"evidence_id": "PHYS:X", "rule": "x"}]},
                   "positive": {"fact": {"evidence_id": "E1"}}}
        outputs = {"positive": {"claims": [{"claim_id": "P1", "claim": "favorable local fragment",
                                                "scope": "local_fragment", "evidence_ids": ["E1"]}]}}
        result = _deterministic_arbitrate(outputs, packets)
        self.assertEqual(result["conflicts"], [])
        self.assertEqual(result["arbitration_mode"], "deterministic")

    def test_claim_validation_rejects_local_evidence_for_global_scope(self):
        from binderloop.agents.binder_quality_collaboration_agent import _validate_claims
        registry = {"R2:STRUCT:1": {"evidence_id": "R2:STRUCT:1"}}
        rows = _validate_claims({"positive": {"claims": [{"claim_id": "P1", "scope": "whole_binder", "evidence_ids": ["R2:STRUCT:1"]}]}}, registry, {})
        self.assertFalse(rows[0]["valid"])
        self.assertIn("local_evidence_for_whole_binder_claim", rows[0]["issues"])


class NewCollaborationContractTest(unittest.TestCase):
    def test_specialist_and_manager_outputs_are_strictly_bounded(self):
        from binderloop.agents.binder_quality_collaboration_agent import _normalize_manager_output, _normalize_specialist_output
        specialist = _normalize_specialist_output({"findings": [{"finding_id": str(i), "statement": "bounded", "scope": "whole_binder", "signal": "reusable", "evidence_ids": ["E1"], "counterevidence_ids": [], "confidence": .5} for i in range(4)], "extra": "reject"}, role="positive")
        self.assertEqual(specialist["claims"], [])
        self.assertTrue(specialist["validation_audit"])
        manager = _normalize_manager_output({"accepted_finding_ids": ["P1"], "rejected_finding_ids": [], "strategy_intents": [{"intent_id": "I1", "kind": "hold", "evidence_ids": ["E1"], "parameter_changes": {"alpha": 1}}], "uncertainties": []}, {"E1": {}}, ["P1"])
        self.assertEqual(manager["strategy_intents"], [])
        self.assertTrue(manager["validation_audit"])

    def test_quality_projection_drops_messages_and_memory(self):
        from binderloop.agents.context_compaction import compact_context_for_quality
        compact = compact_context_for_quality({"messages": [{"content": {"status": "done"}}], "memory": {"recent_rounds": [{"round_id": 1}]}})
        self.assertNotIn("messages", compact)
        self.assertNotIn("memory", compact)


if __name__ == "__main__":
    unittest.main(verbosity=2)
