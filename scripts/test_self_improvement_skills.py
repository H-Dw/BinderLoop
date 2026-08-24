#!/usr/bin/env python3
"""Regression tests for run-local self-improving Binder skills."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.self_improvement_skill_agent import (
    SelfImprovementSkillAgent,
    deidentify_experience,
)
from binderloop.agents.strategy_conflict_resolution_agent import (
    StrategyConflictResolution,
    StrategyConflictResolutionAgent,
    detect_strategy_conflicts,
)
from binderloop.config import HarnessConfig, SelfImprovementSpec, TargetSpec
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.models.base import DesignJob
from binderloop.skills import compose_agent_system
from binderloop.skills.self_improvement import (
    SelfImprovementSkillError,
    SelfImprovementSkillStore,
    SkillDocumentEditor,
    active_prompt_rules,
    apply_lifecycle,
    apply_semantic_relations,
    default_skill_document,
    semantic_candidates,
    validate_skill_document,
)
from scripts.run_closed_loop_orchestrator import _apply_self_improvement_args


def _rule(rule_id, *, family="hotspot_pressure", direction="increase", status="candidate"):
    return {
        "rule_id": rule_id,
        "title": "Adjust interface pressure conditionally",
        "condition": "Interface confidence is weak while foldability remains acceptable.",
        "strategy": "Adjust interface pressure by one bounded step and watch foldability.",
        "expected_signals": ["design_to_target_iptm"],
        "watch_signals": ["design_ptm"],
        "contraindications": ["foldability regression"],
        "status": status,
        "support_count": 0,
        "contradiction_count": 0,
        "utility": 0.0,
        "canonical_signature": {
            "experience_type": "parameter_effects",
            "parameter_families": [family],
            "action_directions": {family: direction},
            "trigger_phenotypes": ["weak_interface"],
            "expected_signals": ["design_to_target_iptm"],
            "watch_signals": ["design_ptm"],
            "contraindications": ["foldability regression"],
        },
    }


class FakeUpdateLLM:
    def __init__(self):
        self.calls = []

    def available(self):
        return True

    def chat_json(self, *, system, user, **kwargs):
        self.calls.append({"system": system, "user": user})
        if "comparisons" in user:
            row = user["comparisons"][0]
            return {
                "relations": [{
                    "left_rule_id": row["left_rule_id"],
                    "right_rule_id": row["right_rule_id"],
                    "relation": "equivalent",
                    "confidence": 0.95,
                    "evidence": ["same phenotype, family and bounded action"],
                }]
            }
        return {
            "summary": "Reinforce a bounded interface-pressure lesson.",
            "operations": [{
                "op": "UPSERT",
                "module": "parameter_effects",
                "rule_id": "proposed_equivalent",
                "rule": _rule("proposed_equivalent"),
            }],
        }


class UnavailableLLM:
    def available(self):
        return False


class FakeConflictLLM:
    def available(self):
        return True

    def chat_json(self, *, system, user, **kwargs):
        return {
            "summary": "Prefer the physically safer bounded option.",
            "decisions": [{
                "parameter_family": "sampling_exploration",
                "action": "choose",
                "selected_rule_ids": [],
                "suspended_rule_ids": [],
                "evidence_ids": ["round_2"],
                "physical_rationale": "Lower pressure preserved foldability and improved the core objective.",
                "parameter_changes": {"alpha": 0.001},
                "expected_signals": ["design_to_target_iptm"],
                "watch_signals": ["design_ptm"],
                "confidence": 0.8,
            }],
            "controlled_comparisons": [],
        }


class FakeUnsafeUpdateLLM:
    def __init__(self, operation):
        self.operation = operation

    def available(self):
        return True

    def chat_json(self, *, system, user, **kwargs):
        return {"summary": "unsafe proposal", "operations": [self.operation]}


class SelfImprovementConfigTest(unittest.TestCase):
    def test_spec_is_opt_in_and_validated(self):
        spec = SelfImprovementSpec()
        self.assertFalse(spec.enabled)
        self.assertIsNone(spec.skill_path)
        with self.assertRaises(Exception):
            SelfImprovementSpec(max_active_rules=10, max_rules=2)

    def test_cli_override_enable_path_and_disable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "configs" / "task.yaml"
            config_path.parent.mkdir()
            config_path.write_text("task: {}\n")
            cfg = HarnessConfig(target=TargetSpec(structure_path="missing.cif"))
            _apply_self_improvement_args(
                cfg,
                SimpleNamespace(
                    self_improvement_enabled=True,
                    self_improvement_skill=None,
                ),
                root=root,
                config_path=config_path,
            )
            self.assertTrue(cfg.self_improvement.enabled)
            _apply_self_improvement_args(
                cfg,
                SimpleNamespace(
                    self_improvement_enabled=None,
                    self_improvement_skill="seed.yaml",
                ),
                root=root,
                config_path=config_path,
            )
            self.assertEqual(cfg.self_improvement.skill_path, str((root / "seed.yaml").resolve()))
            _apply_self_improvement_args(
                cfg,
                SimpleNamespace(
                    self_improvement_enabled=False,
                    self_improvement_skill=None,
                ),
                root=root,
                config_path=config_path,
            )
            self.assertFalse(cfg.self_improvement.enabled)
            self.assertIsNone(cfg.self_improvement.skill_path)


class SkillStoreTest(unittest.TestCase):
    def test_unique_new_copy_on_write_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = SelfImprovementSkillStore.prepare(
                enabled=True, source_path=None, out_dir=root / "run1"
            )
            self.assertIsNotNone(first)
            original_path = first.path
            original_text = original_path.read_text()
            resumed = SelfImprovementSkillStore.prepare(
                enabled=True, source_path=None, out_dir=root / "run1"
            )
            self.assertEqual(resumed.path, original_path)

            reused = SelfImprovementSkillStore.prepare(
                enabled=True,
                source_path=str(original_path),
                out_dir=root / "run2",
            )
            self.assertNotEqual(reused.path.name, original_path.name)
            reused.apply_operations([{
                "operation_id": "add-one",
                "op": "UPSERT",
                "module": "parameter_effects",
                "rule_id": "r1",
                "rule": _rule("r1"),
            }])
            self.assertEqual(original_path.read_text(), original_text)
            self.assertIn("r1", reused.load()["modules"]["parameter_effects"]["rules"])

            other = SelfImprovementSkillStore.prepare(
                enabled=True, source_path=None, out_dir=root / "run3"
            )
            self.assertNotEqual(other.path.name, original_path.name)


class StructuredDocumentTest(unittest.TestCase):
    def setUp(self):
        self.document = default_skill_document(generation_id="test")

    def test_typed_module_update_and_idempotence(self):
        operation = {
            "operation_id": "op1",
            "op": "UPSERT",
            "module": "parameter_effects",
            "rule_id": "r1",
            "rule": _rule("r1"),
        }
        once = SkillDocumentEditor(self.document).apply([operation])
        twice = SkillDocumentEditor(once).apply([operation])
        self.assertEqual(
            set(twice["modules"]["parameter_effects"]["rules"]),
            {"r1"},
        )
        self.assertEqual(twice["provenance"]["applied_operation_ids"], ["op1"])

    def test_target_specific_rule_is_rejected(self):
        bad = _rule("bad")
        bad["strategy"] = "Increase pressure at E:153 using target.cif."
        with self.assertRaises(SelfImprovementSkillError):
            SkillDocumentEditor(self.document).apply([{
                "op": "UPSERT",
                "module": "parameter_effects",
                "rule_id": "bad",
                "rule": bad,
            }])

    def test_semantic_shortlist_relations_and_lifecycle(self):
        document = SkillDocumentEditor(self.document).apply([{
            "op": "UPSERT",
            "module": "parameter_effects",
            "rule_id": "old",
            "rule": _rule("old", status="active"),
        }])
        candidates = semantic_candidates(
            document, _rule("new"), module="parameter_effects"
        )
        self.assertEqual(candidates[0]["rule_id"], "old")
        document["modules"]["parameter_effects"]["rules"]["new"] = _rule("new")
        document = validate_skill_document(document)
        related = apply_semantic_relations(document, [{
            "left_rule_id": "new",
            "right_rule_id": "old",
            "relation": "contradictory",
            "confidence": 0.9,
        }])
        self.assertEqual(
            related["modules"]["parameter_effects"]["rules"]["old"]["status"],
            "contested",
        )
        promoted = SkillDocumentEditor(self.document).apply([{
            "op": "UPSERT", "module": "parameter_effects", "rule_id": "p",
            "rule": {**_rule("p"), "support_count": 2, "utility": 1.0},
        }])
        promoted = apply_lifecycle(
            promoted, promotion_min_support=2, retirement_contradictions=2, max_rules=10
        )
        self.assertEqual(
            promoted["modules"]["parameter_effects"]["rules"]["p"]["status"],
            "active",
        )


class LearningAndCompositionTest(unittest.TestCase):
    def test_llm_semantic_match_revises_stable_rule(self):
        document = SkillDocumentEditor(default_skill_document(generation_id="test")).apply([{
            "op": "UPSERT",
            "module": "parameter_effects",
            "rule_id": "existing",
            "rule": _rule("existing", status="active"),
        }])
        llm = FakeUpdateLLM()
        update = SelfImprovementSkillAgent(
            llm, semantic_confidence_threshold=0.7
        ).propose_update(
            round_id=2,
            document=document,
            evidence={"outcome": {"reward": 0.5}, "evaluation": {}},
            governance_skills=[],
        )
        self.assertEqual(update.operations[0]["op"], "REVISE")
        self.assertEqual(update.operations[0]["rule_id"], "existing")
        revised = SkillDocumentEditor(document).apply(update.operations)
        self.assertEqual(
            revised["modules"]["parameter_effects"]["rules"]["existing"]["status"],
            "active",
        )
        self.assertEqual(len(llm.calls), 2)

    def test_semantic_pair_digest_cache_avoids_repeat_match_call(self):
        document = SkillDocumentEditor(default_skill_document(generation_id="test")).apply([{
            "op": "UPSERT",
            "module": "parameter_effects",
            "rule_id": "existing",
            "rule": _rule("existing", status="active"),
        }])
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeUpdateLLM()
            agent = SelfImprovementSkillAgent(
                llm,
                semantic_confidence_threshold=0.7,
                cache_dir=Path(tmp),
            )
            for _ in range(2):
                agent.propose_update(
                    round_id=2,
                    document=document,
                    evidence={"outcome": {"reward": 0.5}, "evaluation": {}},
                    governance_skills=[],
                )
            self.assertEqual(len(llm.calls), 3)

    def test_deidentification_and_priority_rendering(self):
        clean = deidentify_experience({
            "target_name": "example",
            "hotspots": ["E:153"],
            "binder_lengths": [60, 80],
            "binder_length_range": [50, 100],
            "chain_id": "E",
            "target_chains": ["E", "F"],
            "fragment_id": "frag_17",
            "note": "candidate_17 came from /tmp/target.cif near E:153",
        })
        text = json.dumps(clean)
        self.assertNotIn("E:153", text)
        self.assertNotIn("candidate_17", text)
        self.assertNotIn('"chain_id"', text)
        self.assertNotIn("frag_17", text)
        system = compose_agent_system(
            "BASE CONTRACT",
            active_skills=[{
                "id": "learned", "type": "llm_reasoning", "priority": 900,
                "origin": "run_local_self_improvement", "guidance": ["[r1] use bounded change"],
                "learned_rules": [_rule("r1", status="active")],
            }],
        )
        self.assertIn("highest advisory priority", system)
        self.assertIn("learned_rule_ids", system)

    def test_deterministic_evidence_gate_blocks_uncited_votes_and_forces_candidate(self):
        document = default_skill_document(generation_id="test")
        active_attempt = _rule("unsafe", status="active")
        update = SelfImprovementSkillAgent(
            FakeUnsafeUpdateLLM({
                "op": "UPSERT",
                "module": "parameter_effects",
                "rule_id": "unsafe",
                "rule": active_attempt,
            })
        ).propose_update(
            round_id=1,
            document=document,
            evidence={"outcome": {"reward": 0.5}, "evaluation": {}, "recent_rounds": []},
        )
        self.assertEqual(update.operations[0]["rule"]["status"], "candidate")
        self.assertEqual(update.operations[0]["rule"]["support_count"], 0)

        strong = SelfImprovementSkillAgent(
            FakeUnsafeUpdateLLM({
                "op": "UPSERT",
                "module": "parameter_effects",
                "rule_id": "strong",
                "rule": _rule("strong", family="sampling_exploration"),
            }),
            strong_improvement_threshold=0.05,
        ).propose_update(
            round_id=2,
            document=document,
            evidence={
                "outcome": {"reward": 0.8},
                "evaluation": {},
                "recent_rounds": [{"reward": 0.7}],
                "strategy_exposure": {
                    "exposure_id": "exp1",
                    "applied_update": {"alpha": 0.001},
                },
            },
        )
        self.assertTrue(strong.operations[0]["rule"].get("strong_evidence", strong.operations[0].get("strong_evidence", True)))
        strong_document = SkillDocumentEditor(document).apply(strong.operations)
        strong_document = apply_lifecycle(
            strong_document,
            promotion_min_support=2,
            retirement_contradictions=2,
            max_rules=10,
        )
        strong_rule = next(
            iter(strong_document["modules"]["parameter_effects"]["rules"].values())
        )
        self.assertEqual(strong_rule["status"], "active")

        document = SkillDocumentEditor(document).apply([{
            "op": "UPSERT", "module": "parameter_effects", "rule_id": "existing",
            "rule": _rule("existing", status="active"),
        }])
        vote = SelfImprovementSkillAgent(
            FakeUnsafeUpdateLLM({
                "op": "UPVOTE",
                "module": "parameter_effects",
                "rule_id": "existing",
            })
        ).propose_update(
            round_id=2,
            document=document,
            evidence={
                "outcome": {"reward": 0.7},
                "evaluation": {},
                "recent_rounds": [{"reward": 0.5}],
                "strategy_exposure": {"cited_rule_ids": []},
            },
        )
        self.assertEqual(vote.operations, [])
        self.assertEqual(vote.rejected_operations[0]["reason"], "rule_was_not_cited_as_used")

        skipped_llm = FakeUpdateLLM()
        skipped = SelfImprovementSkillAgent(skipped_llm).propose_update(
            round_id=3,
            document=document,
            evidence={"outcome": {"execution_failed": True, "reward": 0.0}, "evaluation": {}},
        )
        self.assertFalse(skipped.llm_used)
        self.assertEqual(skipped_llm.calls, [])

    def test_llm_outputs_require_rule_citation_or_explicit_nonuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = HarnessConfig(target=TargetSpec(structure_path="missing.cif"))
            cfg.self_improvement.enabled = True
            orchestrator = BinderDesignOrchestrator(
                cfg, out_dir=Path(tmp), max_rounds=1
            )
            active_id = active_prompt_rules(
                orchestrator.self_improvement_document,
                limit=1,
            )[0]["rule_id"]
            with self.assertRaises(Exception):
                orchestrator._validate_learned_skill_usage(
                    {"llm_used": True, "raw": {}},
                    module_name="test",
                )
            orchestrator._validate_learned_skill_usage(
                {
                    "llm_used": True,
                    "raw": {"learned_rule_ids": [active_id]},
                },
                module_name="test",
            )
            orchestrator._validate_learned_skill_usage(
                {
                    "llm_used": True,
                    "raw": {"learned_skill_nonuse_reason": "No rule matched the current phenotype."},
                },
                module_name="test",
            )


class ConflictResolutionTest(unittest.TestCase):
    def test_detector_and_safe_hold_fallback(self):
        conflicts = detect_strategy_conflicts(
            merge_report={
                "ownership_conflicts": [{
                    "key": "alpha",
                    "kept_value": 0.003,
                    "rejected_value": 0.009,
                }]
            },
            proposed_update={"alpha": 0.003},
            tuning_feedback={"penalized_moves": [{
                "parameter": "alpha",
                "previous_move": "increased 0.001 -> 0.003",
            }]},
            pressure_conflict={"active": True},
            learned_document=None,
        )
        self.assertEqual(len(conflicts), 1)
        resolution = StrategyConflictResolutionAgent(UnavailableLLM()).resolve(
            round_id=3,
            conflicts=conflicts,
            context={"current_config": {"alpha": 0.001}},
        )
        self.assertFalse(resolution.llm_used)
        self.assertEqual(resolution.decisions[0]["action"], "hold")
        self.assertEqual(resolution.params_update, {})
        self.assertEqual(resolution.decisions[0]["probabilistic_sampler_veto"], ["alpha"])
        reverted = StrategyConflictResolutionAgent(UnavailableLLM()).resolve(
            round_id=3,
            conflicts=conflicts,
            context={
                "current_config": {"alpha": 0.001},
                "historical_best": {"config": {"alpha": 0.001}},
            },
        )
        self.assertEqual(reverted.decisions[0]["action"], "revert_to_best")
        self.assertEqual(reverted.params_update, {})
        self.assertEqual(reverted.decisions[0]["probabilistic_sampler_veto"], ["alpha"])

    def test_preview_merge_does_not_mutate_and_resolution_can_override_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = HarnessConfig(target=TargetSpec(structure_path="missing.cif"))
            cfg.search_space.boltzgen["alpha"] = 0.001
            orchestrator = BinderDesignOrchestrator(
                cfg, out_dir=Path(tmp), max_rounds=1
            )
            preview, report = orchestrator._merge_next_round_updates(
                ("input_configuration", {"alpha": 0.003}),
                ("policy_proposal", {"alpha": 0.009}),
                apply=False,
            )
            self.assertNotIn("alpha", preview)
            self.assertEqual(cfg.search_space.boltzgen["alpha"], 0.001)
            self.assertEqual(len(report["ownership_conflicts"]), 0)
            applied, _ = orchestrator._merge_next_round_updates(
                ("input_configuration", {"alpha": 0.003}),
                ("strategy_conflict_resolution", {"alpha": 0.001}),
                apply=True,
            )
            self.assertNotIn("alpha", applied)
            self.assertEqual(cfg.search_space.boltzgen["alpha"], 0.001)

    def test_detector_finds_opposite_active_learned_directions(self):
        document = default_skill_document(generation_id="conflict")
        document = SkillDocumentEditor(document).apply([
            {
                "op": "UPSERT", "module": "parameter_effects", "rule_id": "up",
                "rule": _rule("up", direction="increase", status="active"),
            },
            {
                "op": "UPSERT", "module": "parameter_effects", "rule_id": "down",
                "rule": _rule("down", direction="decrease", status="active"),
            },
        ])
        conflicts = detect_strategy_conflicts(
            merge_report={"ownership_conflicts": []},
            proposed_update={},
            tuning_feedback={},
            pressure_conflict={},
            learned_document=document,
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(set(conflicts[0]["rule_ids"]), {"up", "down"})

    def test_llm_physical_resolution_and_controlled_job_materialization(self):
        conflict = {
            "parameter_family": "sampling_exploration",
            "keys": ["alpha"],
            "rule_ids": [],
            "evidence_ids": ["round_2"],
            "sources": [],
        }
        llm_resolution = StrategyConflictResolutionAgent(FakeConflictLLM()).resolve(
            round_id=3,
            conflicts=[conflict],
            context={"current_config": {"alpha": 0.003}},
        )
        self.assertTrue(llm_resolution.llm_used)
        self.assertEqual(llm_resolution.params_update, {})
        self.assertEqual(llm_resolution.decisions[0]["action"], "hold")
        self.assertEqual(llm_resolution.decisions[0]["probabilistic_sampler_veto"], ["alpha"])

        # Sampler comparisons are now owned by the finite-catalog parameter
        # decision layer, so soft-conflict output cannot materialize free-form jobs.
        self.assertFalse(hasattr(llm_resolution, "controlled_comparisons"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

