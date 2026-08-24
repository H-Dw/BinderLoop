#!/usr/bin/env python3
"""Isolation, search-profile, and RFD3 closed-loop arm tests (no GPU)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.active_learning.strategy import CANONICAL_STRATEGY_ARM_CATALOG, StrategyLevelActiveLearner
from binderloop.agents.active_learning_policy_agent import ActiveLearningPolicyAgent
from binderloop.agents.config_parameter_contract import PARAM_BOUNDS, clamp_config_with_inertia
from binderloop.agents.config_validation_agent import ConfigValidationAgent
from binderloop.agents.evaluation_agent import EvaluationSummary
from binderloop.agents.input_configuration_agent import InputConfigurationAgent
from binderloop.agents.result_ingestion_agent import ResultIngestionAgent
from binderloop.config import load_config
from binderloop.models.base import DesignJob
from binderloop.models.search_profile import (
    SearchProfileError,
    get_model_search_profile,
    isolate_model_params,
)
from binderloop.models.sequence import get_sequence_tool
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.parameter_decision import ParameterCandidate, ParameterDecisionSpec
from binderloop.strategy_governance import SUPPORTED_BUNDLES


ROOT = Path(__file__).resolve().parents[1]


class IsolationAndProfileTests(unittest.TestCase):
    def test_boltzgen_filter_strips_rfd3_keys(self):
        profile = get_model_search_profile("boltzgen")
        result = profile.filter_params({
            "alpha": 0.001, "step_scale": 0.8, "gamma_0": 0.2, "is_non_loopy": True,
            "select_hotspots": {"A56": "ALL"}, "num_designs": 8,
        })
        self.assertEqual(result.params["alpha"], 0.001)
        self.assertEqual(result.params["step_scale"], 0.8)
        self.assertNotIn("gamma_0", result.params)
        self.assertNotIn("is_non_loopy", result.params)
        self.assertNotIn("select_hotspots", result.params)
        self.assertEqual(result.params["sequence_tool"], "boltz_ifold")
        self.assertEqual(result.params["refold_tool"], "boltz2")

    def test_rfd3_filter_strips_boltzgen_keys(self):
        profile = get_model_search_profile("rfd3")
        result = profile.filter_params({
            "alpha": 0.001, "filter_biased": "true", "protocol": "protein-anything",
            "step_scale": 3.0, "gamma_0": 0.2, "num_designs": 8,
        })
        self.assertEqual(result.params["step_scale"], 3.0)
        self.assertEqual(result.params["gamma_0"], 0.2)
        self.assertNotIn("alpha", result.params)
        self.assertNotIn("filter_biased", result.params)
        self.assertNotIn("protocol", result.params)
        self.assertEqual(result.params["sequence_tool"], "protein_mpnn")
        self.assertEqual(result.params["refold_tool"], "rf3")

    def test_rfd3_step_scale_not_clamped_to_boltzgen(self):
        profile = get_model_search_profile("rfd3")
        clamped, notes = clamp_config_with_inertia({"step_scale": 3.0}, bounds=profile.param_bounds)
        self.assertEqual(clamped["step_scale"], 3.0)
        boltzgen_clamped, _ = clamp_config_with_inertia({"step_scale": 3.0}, bounds=PARAM_BOUNDS)
        self.assertLessEqual(boltzgen_clamped["step_scale"], 1.0)

    def test_sequence_tool_allowlists(self):
        ifold = get_sequence_tool("boltz_ifold")
        mpnn = get_sequence_tool("protein_mpnn")
        self.assertIn("filter_biased", ifold.allowed_keys)
        self.assertIn("temperature", ifold.forbidden_keys)
        self.assertIn("temperature", mpnn.allowed_keys)
        self.assertIn("filter_biased", mpnn.forbidden_keys)
        self.assertNotIn("filter_biased", mpnn.materialize("repair", {}))
        self.assertIn("temperature", mpnn.materialize("repair", {"temperature": 0.1}))

    def test_incompatible_tool_binding_rejected(self):
        cfg = load_config(ROOT / "configs" / "example_rfd3_binder_task.yaml")
        cfg.sequence.tool = "boltz_ifold"
        with self.assertRaises(SearchProfileError):
            get_model_search_profile("rfd3", cfg=cfg)

    def test_example_rfd3_yaml_isolation_and_catalog(self):
        cfg = load_config(ROOT / "configs" / "example_rfd3_binder_task.yaml")
        self.assertEqual(cfg.search_space.model_order, ["rfd3"])
        self.assertEqual(cfg.sequence.tool, "protein_mpnn")
        self.assertEqual(cfg.refolding.tool, "rf3")
        spec = cfg.owner.parameter_decision
        self.assertEqual(spec.active_sampler_keys(), ("step_scale", "gamma_0"))
        self.assertIn(3.0, spec.step_scale_candidates)
        self.assertNotIn("alpha", spec.active_axes())
        mixed = isolate_model_params(cfg, "rfd3", {"alpha": 0.05, "filter_biased": "true", "step_scale": 3.0, "gamma_0": 0.2})
        self.assertNotIn("alpha", mixed)
        self.assertNotIn("filter_biased", mixed)
        self.assertEqual(mixed["step_scale"], 3.0)
        boltzgen_job = isolate_model_params(cfg, "boltzgen", {"step_scale": 1.0, "gamma_0": 0.2})
        self.assertNotIn("gamma_0", boltzgen_job)

    def test_rfd3_catalog_member_is_generic(self):
        spec = ParameterDecisionSpec(
            sampler_axes=("step_scale", "gamma_0"),
            step_scale_candidates=(1.5, 3.0),
            gamma_0_candidates=(0.2, 0.4),
        )
        self.assertEqual(len(spec.catalog), 4)
        self.assertIn(ParameterCandidate(step_scale=3.0, gamma_0=0.2), spec.catalog)
        self.assertTrue(all("alpha" not in item.as_dict() for item in spec.catalog))

    def test_sequence_repair_arm_and_bundle(self):
        self.assertIn("sequence_repair", CANONICAL_STRATEGY_ARM_CATALOG)
        self.assertIn(frozenset({"sequence"}), SUPPORTED_BUNDLES)
        learner = StrategyLevelActiveLearner()
        arms = learner.candidate_arms(
            structural_summary=None,
            hypotheses=[{"failure_modes": ["folding_failure"]}],
            selection_context={"failure_tag_counts": {"folding_failure": 3}, "strict_positive_count": 0},
        )
        names = [arm["name"] for arm in arms]
        self.assertIn("sequence_repair", names)
        quiet = learner.candidate_arms(
            structural_summary=None,
            hypotheses=[{"failure_modes": ["hotspot_miss"]}],
            selection_context={"failure_tag_counts": {"hotspot_miss": 1}, "strict_positive_count": 1},
        )
        self.assertNotIn("sequence_repair", [arm["name"] for arm in quiet])

    def test_rfd3_orchestrator_base_params_and_restore(self):
        cfg = load_config(ROOT / "configs" / "example_rfd3_binder_task.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            orch = BinderDesignOrchestrator(cfg, out_dir=Path(tmp) / "out", max_rounds=1)
            params = orch._base_params()
            self.assertEqual(params["step_scale"], 3.0)
            self.assertEqual(params["gamma_0"], 0.2)
            self.assertNotIn("alpha", params)
            self.assertNotIn("filter_biased", params)
            self.assertEqual(params["sequence_tool"], "protein_mpnn")
            self.assertEqual(params["refold_tool"], "rf3")
            job = DesignJob("j", cfg.target.structure_path, "A", ["A:56"], 50, params={**params, "sampler_policy": "explore", "final_parameter_state": {"step_scale": 3.0, "gamma_0": 0.2}}, output_dir=str(Path(tmp) / "job"))
            materialized = orch._materialize_sampler_and_context_intents([job])[0]
            self.assertEqual(materialized.params["final_parameter_state"], {"step_scale": 3.0, "gamma_0": 0.2})
            self.assertNotIn("alpha", materialized.params)
            orch._apply_next_round_update({"step_scale": 3.5, "gamma_0": 0.4})
            self.assertEqual(cfg.search_space.rfd3["step_scale"], 3.5)
            self.assertEqual(cfg.search_space.rfd3["gamma_0"], 0.4)
            snapshot = orch._current_config_snapshot()
            self.assertIn("rfd3_config", snapshot)
            self.assertEqual(snapshot["search_profile_model"], "rfd3")

    def test_rfd3_ingest_sets_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "final_designs_metrics.csv"
            csv_path.write_text("design,iptm,plddt,ranking_score,path\ncand1,0.5,80,0.6,cand1.cif\n", encoding="utf-8")
            run = ResultIngestionAgent().ingest_rfd3_output(root, identity_context={"job_id": "j", "arm_id": "baseline_hold"})
            self.assertEqual(run.candidates[0]["model"], "rfd3")
            profile = get_model_search_profile("rfd3")
            ingested = profile.ingest(root, identity_context={"job_id": "j"})
            self.assertEqual(ingested.candidates[0]["model"], "rfd3")

    def test_sequence_repair_materialize_on_rfd3(self):
        cfg = load_config(ROOT / "configs" / "example_rfd3_binder_task.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            orch = BinderDesignOrchestrator(cfg, out_dir=Path(tmp) / "out", max_rounds=1)
            params = orch._base_params()
            params["sequence_policy"] = "repair"
            job = DesignJob("j", cfg.target.structure_path, "A", ["A:56"], 50, params=params, output_dir=str(Path(tmp) / "job"))
            result = orch._materialize_sampler_and_context_intents([job])[0]
            self.assertGreater(float(result.params["temperature"]), 0.1)
            self.assertNotIn("filter_biased", result.params)

    def test_boltzgen_regression_catalog_unchanged(self):
        spec = ParameterDecisionSpec()
        self.assertEqual(len(spec.catalog), 5 * 4 * 3)
        self.assertIn(ParameterCandidate(0.05, 0.9, 1.0), spec.catalog)

    def test_rfd3_validation_strips_boltzgen_keys_and_keeps_ppi_step_scale(self):
        result = ConfigValidationAgent().validate_for_submission(
            {"alpha": 0.05, "filter_biased": "true", "step_scale": 3.0, "gamma_0": 0.2, "num_designs": 2},
            target_model="rfd3",
        )
        self.assertNotIn("alpha", result.corrected_config)
        self.assertNotIn("filter_biased", result.corrected_config)
        self.assertEqual(result.corrected_config["step_scale"], 3.0)
        self.assertEqual(result.corrected_config["gamma_0"], 0.2)

    def test_agent_delta_still_strips_user_owned_filters(self):
        result = ConfigValidationAgent().validate_agent_delta(
            {
                "hotspot_weight": 2.0,
                "additional_filters": "iptm>0.35",
                "run_filtering": True,
                "target_include": [{"chain": {"id": "E", "res_index": "1..194"}}],
                "binder_lengths": [80, 90],
            },
            target_model="boltzgen",
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.corrected_config["binder_lengths"], [80, 90])
        self.assertNotIn("additional_filters", result.corrected_config)
        self.assertNotIn("run_filtering", result.corrected_config)
        self.assertNotIn("target_include", result.corrected_config)

    def test_rfd3_policy_holds_sampler_and_strips_boltzgen_keys(self):
        summary = EvaluationSummary(
            total_candidates=10, success_count=0, failure_count=10,
            tag_counts={"diversity_collapse": 8}, top_candidates=[], failed_examples=[], observations=[],
        )
        proposal = ActiveLearningPolicyAgent().propose_next_params(
            summary, {"diffusion_batch_size": 8, "step_scale": 3.0, "gamma_0": 0.2}, model="rfd3",
        )
        self.assertNotIn("alpha", proposal.params_update)
        self.assertNotIn("filter_biased", proposal.params_update)
        self.assertNotIn("config_overrides", proposal.params_update)
        self.assertEqual(proposal.params_update.get("diffusion_batch_size"), 1)
        self.assertEqual((proposal.analysis_metadata.get("probabilistic_sampler_directions") or {}).get("gamma_0"), "increase")

    def test_rfd3_input_prompt_uses_profile_axes(self):
        profile = get_model_search_profile("rfd3")
        agent = InputConfigurationAgent(
            parameter_candidates={"step_scale": (1.5, 3.0), "gamma_0": (0.1, 0.2)},
            adjustable_parameters=profile.adjustable_parameters,
            param_bounds=profile.param_bounds,
            sampler_axes=profile.sampler_axes,
        )
        prompt = agent._system_prompt()
        self.assertIn("gamma_0", prompt)
        self.assertIn("step_scale", prompt)
        self.assertNotIn("never above 0.05", prompt)

    def test_sequence_repair_triggers_on_low_sequence_designability(self):
        learner = StrategyLevelActiveLearner()
        arms = learner.candidate_arms(
            structural_summary=None,
            hypotheses=[{"failure_modes": ["hotspot_miss"]}],
            selection_context={"failure_tag_counts": {"hotspot_miss": 1}, "strict_positive_count": 1, "core_metric_stats": {"sequence_designability": {"mean": 0.2}}},
        )
        self.assertIn("sequence_repair", [arm["name"] for arm in arms])

    def test_diversity_collapse_sets_rfd3_batch_size_one(self):
        cfg = load_config(ROOT / "configs" / "example_rfd3_binder_task.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            orch = BinderDesignOrchestrator(cfg, out_dir=Path(tmp) / "out", max_rounds=1)
            params = orch._base_params()
            params["sampler_policy"] = "explore"
            params["diversity_collapse"] = True
            params["final_parameter_state"] = {"step_scale": 3.0, "gamma_0": 0.2}
            job = DesignJob("j", cfg.target.structure_path, "A", ["A:56"], 50, params=params, output_dir=str(Path(tmp) / "job"))
            result = orch._materialize_sampler_and_context_intents([job])[0]
            self.assertEqual(int(result.params["diffusion_batch_size"]), 1)
            self.assertEqual(result.params["final_parameter_state"], {"step_scale": 3.0, "gamma_0": 0.2})

    def test_template_exploit_unsupported_for_rfd3(self):
        self.assertNotIn("template_exploit", get_model_search_profile("rfd3").supported_arms)
        cfg = load_config(ROOT / "configs" / "example_rfd3_binder_task.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            orch = BinderDesignOrchestrator(cfg, out_dir=Path(tmp) / "out", max_rounds=1)
            params = orch._base_params()
            params["arm_id"] = "template_exploit"
            params["exploration_arm"] = "template_exploit"
            job = DesignJob("j", cfg.target.structure_path, "A", ["A:56"], 50, params=params, output_dir=str(Path(tmp) / "job"))
            orch._govern_exploration_jobs([job], current_jobs=[], next_round_id=1, strict_positive_count=2)
            self.assertEqual(job.params.get("strategy_applicability"), "unsupported")
            self.assertEqual(job.params.get("strategy_applicability_reason"), "unsupported_arm_for_search_profile")

    def test_site_expanded_adds_rfd3_hotspots_without_replacing_primary(self):
        cfg = load_config(ROOT / "configs" / "example_rfd3_binder_task.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            orch = BinderDesignOrchestrator(cfg, out_dir=Path(tmp) / "out", max_rounds=1)
            params = orch._base_params()
            params["binding_site_policy"] = "primary_expanded"
            params["auxiliary_hotspots"] = ["A:60"]
            params["select_hotspots"] = {"A56": "CG,OH"}
            job = DesignJob("j", cfg.target.structure_path, "A", ["A:56"], 50, params=params, output_dir=str(Path(tmp) / "job"))
            result = orch._materialize_sampler_and_context_intents([job])[0]
            hotspots = result.params["select_hotspots"]
            self.assertEqual(hotspots["A56"], "CG,OH")
            self.assertGreaterEqual(len(hotspots), 2)
            self.assertNotIn("filter_biased", result.params)
            self.assertNotIn("alpha", result.params)


if __name__ == "__main__":
    unittest.main()
