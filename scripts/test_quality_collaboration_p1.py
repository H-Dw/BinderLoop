#!/usr/bin/env python3
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binderloop.agents.binder_quality_collaboration_agent import _normalize_specialist_output, _normalize_manager_output
from binderloop.agents.context_compaction import compact_context_for_quality
from binderloop.skills.composer import compose_agent_system

class P1Contracts(unittest.TestCase):
    def test_invalid_types_are_audited_not_raised(self):
        out = _normalize_specialist_output({"findings": [7]}, role="positive")
        self.assertEqual(out["claims"], [])
        self.assertTrue(out["validation_audit"])

    def test_duplicate_ids_and_bad_source_round_are_rejected(self):
        item = {"finding_id":"T1","statement":"x","scope":"trajectory","source_round_ids":[3],"family":"sampler","outcome":"mixed","causal_strength":"weak","evidence_ids":["R2:METRICS"],"counterevidence_ids":[],"confidence":.5}
        out = _normalize_specialist_output({"findings":[item, dict(item)]}, role="trajectory")
        self.assertEqual(out["claims"], [])
        self.assertTrue(out["validation_audit"])

    def test_manager_rejects_unknown_finding(self):
        out = _normalize_manager_output({"accepted_finding_ids":["X"],"rejected_finding_ids":[],"strategy_intents":[],"uncertainties":[]}, {}, ["P1"])
        self.assertEqual(out["accepted_finding_ids"], [])
        self.assertTrue(out["validation_audit"])

    def test_quality_projection_has_candidate_without_paths(self):
        out = compact_context_for_quality({"structural_analysis":{"summaries":[{"candidate_id":"c1","structure_file":"/secret/a.cif","filename":"a.cif","path":"/secret"}]}})
        text = str(out)
        self.assertIn("c1", text); self.assertNotIn("/secret", text); self.assertNotIn("a.cif", text)

    def test_composer_limits_complete_role_directives(self):
        skills=[{"id":"s","priority":10,"role_metadata":{"roles":["positive"]},"guidance":["one","two","three","four"]}]
        out=compose_agent_system("base", active_skills=skills, role="positive", max_directives=3)
        self.assertIn("one", out); self.assertIn("three", out); self.assertNotIn('"directive": "four"', out)

    def test_typed_proposal_conflict_is_preserved(self):
        from binderloop.agents.active_learning_policy_agent import ActiveLearningPolicyAgent
        from binderloop.agents.evaluation_agent import EvaluationSummary
        summary = EvaluationSummary(total_candidates=1, success_count=0, failure_count=1, tag_counts={}, top_candidates=[], failed_examples=[], observations=[])
        proposal = ActiveLearningPolicyAgent().propose_next_boltzgen_params(summary, {"diffusion_batch_size": 1}, diagnostic_report={"corrective_actions":[{"action":"d","parameter_changes":{"filter_biased": True}}]}, quality_analysis={"next_round_guidance":[{"action":"q","config_parameter_changes":{"filter_biased": False}}]})
        self.assertNotIn("filter_biased", proposal.params_update)
        self.assertEqual(proposal.analysis_metadata["proposal_conflicts"][0]["key"], "filter_biased")

    def test_input_configuration_rejects_round_strategy(self):
        from binderloop.agents.input_configuration_agent import InputConfigurationAgent
        class LLM:
            def available(self): return True
            def chat_json(self, **kwargs): return {"reasoning":"x","parameter_delta":{},"iteration_strategy":{"round_1_focus":"forbidden"},"evidence_finding_ids":["P1"],"hold_reasons":["weak evidence"],"expected_signals":["stable reward"]}
            def chat_label_distribution(self, **kwargs): return {}
        out = InputConfigurationAgent(LLM()).configure(target_name="t", target_info={})
        self.assertEqual(out.iteration_strategy, {})
        self.assertTrue(out.raw["ignored_iteration_strategy"])
        self.assertEqual(out.evidence_finding_ids, ["P1"])

if __name__ == "__main__": unittest.main(verbosity=2)
