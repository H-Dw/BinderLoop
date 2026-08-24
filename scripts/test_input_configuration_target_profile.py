#!/usr/bin/env python3
"""Regression tests for target-specific InputConfigurationAgent context."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.context_compaction import compact_context_for_input_config, compact_context_for_target_config
from binderloop.agents.input_configuration_agent import InputConfigurationAgent
from binderloop.config import load_config
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator


def _payload_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class InputConfigurationTargetProfileTest(unittest.TestCase):
    def test_system_prompt_has_no_static_il17a_target_facts(self):
        system = InputConfigurationAgent.SYSTEM
        forbidden = ("IL-17A", "A+B", "A:67", "A:89", "B:49", "homodimer")
        for token in forbidden:
            self.assertNotIn(token, system)

    def test_target_profile_is_dynamic_next_round_context(self):
        compact = compact_context_for_input_config(
            target_name="PD-L1_len50_120_iptm035",
            current_config={"hotspots": ["A:40", "A:99", "A:107"]},
            diagnostic_report={},
            evaluation_summary={},
            round_id=2,
            target_profile={
                "target_name": "PD-L1_len50_120_iptm035",
                "primary_chain_id": "A",
                "target_chains": ["A"],
                "hotspots": ["A:40", "A:99", "A:107"],
                "notes": "PD-L1 single-chain target profile for this task.",
                "source": "current_task_config",
            },
        )

        self.assertEqual(compact["target_profile"]["target_chains"], ["A"])
        self.assertEqual(compact["target_profile"]["hotspots"], ["A:40", "A:99", "A:107"])
        self.assertNotIn("IL-17A", _payload_text(compact))

    def test_target_profile_is_dynamic_initial_context(self):
        compact = compact_context_for_target_config({
            "target_name": "IL-17A_test",
            "target_info": {"hotspots": ["A:67", "A:89", "B:49"]},
            "target_profile": {
                "target_name": "IL-17A_test",
                "target_chains": ["A", "B"],
                "profile": {"oligomer_state": "homodimer"},
                "source": "current_task_config",
            },
        })

        self.assertEqual(compact["target_profile"]["target_chains"], ["A", "B"])
        self.assertEqual(compact["target_profile"]["profile"]["oligomer_state"], "homodimer")

    def test_deterministic_fallback_does_not_invent_il17a_hotspots(self):
        config = InputConfigurationAgent().configure(
            target_name="unknown_target",
            target_info={},
            target_profile={"target_name": "unknown_target", "source": "current_task_config"},
        )

        self.assertFalse(config.llm_used)
        self.assertNotIn("A:67", _payload_text(config.raw))
        self.assertTrue(
            any(item["risk"] == "No target hotspots were provided" for item in config.risk_assessment)
        )

    def test_orchestrator_builds_current_task_target_profile(self):
        repo = Path(__file__).resolve().parents[1]
        cfg = load_config(repo / "configs" / "pdl1_structured_task_extend_iptm035.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            orch = BinderDesignOrchestrator(cfg, out_dir=str(Path(tmp) / "pdl1"), max_rounds=1)

        profile = orch._target_profile_context()
        self.assertEqual(profile["target_chains"], ["A"])
        self.assertEqual(profile["hotspots"], ["A:40", "A:99", "A:107"])
        self.assertEqual(profile["source"], "current_task_config")
        self.assertNotIn("IL-17A", _payload_text(profile))


class AgentParameterProposalTest(unittest.TestCase):
    class LegacyLLM:
        def available(self): return True
        def chat_json(self, **_kwargs):
            return {"reasoning": "analysis", "recommended_config": {"alpha": 0.7, "noise_scale": 0.9, "step_scale": 1.0, "diffusion_batch_size": 1}}

    class DistributionLLM(LegacyLLM):
        def __init__(self): self.calls = []
        def chat_label_distribution(self, **kwargs):
            self.calls.append(kwargs)
            label = kwargs["labels"][1]
            return {"label": label, "distribution": {label: 0.8, "HOLD_CURRENT": 0.2}, "evidence": {"mock": True}}

    def test_legacy_mock_holds_without_distribution_method(self):
        result = InputConfigurationAgent(AgentParameterProposalTest.LegacyLLM()).configure(target_name="t", target_info={})
        self.assertEqual(result.recommended_config, {"diffusion_batch_size": 1})
        self.assertEqual(set(result.raw["ignored_recommended_config_keys"]), {"alpha", "noise_scale", "step_scale"})
        self.assertTrue(all(item["status"] == "unavailable" and item["proposed_value"] == "HOLD_CURRENT" for item in result.parameter_proposals.values()))

    def test_initial_config_collects_three_closed_label_distributions(self):
        llm = self.DistributionLLM()
        result = InputConfigurationAgent(llm, parameter_candidates={"alpha": [0.001], "noise_scale": [0.7], "step_scale": [0.8]}).configure(target_name="t", target_info={"alpha": 0.001})
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(set(result.parameter_proposals), {"alpha", "noise_scale", "step_scale"})
        self.assertTrue(all(call["labels"][0] == "HOLD_CURRENT" for call in llm.calls))
        self.assertTrue(all(item["execute"] is False for item in result.parameter_proposals.values()))

    def test_next_round_uses_per_call_candidate_override(self):
        llm = self.DistributionLLM()
        result = InputConfigurationAgent(llm).configure_next_round(target_name="t", current_config={"noise_scale": 0.75}, diagnostic_report={}, evaluation_summary={}, round_id=2, parameter_candidates={"noise_scale_candidates": [0.6, 0.9]})
        labels = result.parameter_proposals["noise_scale"]["labels"]
        self.assertIn(0.75, labels.values())

    def test_other_agent_sanitizers_strip_sampler_only(self):
        from binderloop.agents.hypothesis_agent import HypothesisAgent
        from binderloop.agents.binder_quality_analysis_agent import BinderQualityAnalysisAgent
        from binderloop.agents.diagnostic_coach_agent import DiagnosticCoachAgent
        payload = {"alpha": 0.01, "noise_scale": 0.8, "step_scale": 0.9, "diffusion_batch_size": 1}
        hypothesis = HypothesisAgent._sanitize_hypotheses([{"config_parameter_changes": payload}])[0]
        quality = BinderQualityAnalysisAgent._sanitize_guidance([{"config_parameter_changes": payload}])[0]
        diagnostic = DiagnosticCoachAgent._sanitize_corrective_actions([{"parameter_changes": payload}])[0]
        for item, key, ignored_key in ((hypothesis, "config_parameter_changes", "ignored_config_parameter_changes"), (quality, "config_parameter_changes", "ignored_config_parameter_changes"), (diagnostic, "parameter_changes", "ignored_parameter_changes")):
            self.assertEqual(item[key], {"diffusion_batch_size": 1})
            self.assertEqual(set(item[ignored_key]), {"alpha", "noise_scale", "step_scale"})


class SparseDeltaContractTest(unittest.TestCase):
    def test_parameter_delta_is_primary_with_legacy_alias(self):
        result = InputConfigurationAgent(AgentParameterProposalTest.LegacyLLM()).configure(target_name="t", target_info={})
        self.assertEqual(result.parameter_delta, {"diffusion_batch_size": 1})
        self.assertEqual(result.recommended_config, result.parameter_delta)
        self.assertEqual(result.raw["parameter_delta"], result.parameter_delta)
        self.assertEqual(result.raw["recommended_config"], result.parameter_delta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
