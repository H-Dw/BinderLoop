#!/usr/bin/env python3
"""Prompt catalog / assembler tests that do not require live round outputs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.context_compaction import compact_context_for_hypothesis
from binderloop.agents.hypothesis_agent import HypothesisAgent
from binderloop.agents.prompt_assembler import assemble, build_store, load_round_artifacts
from binderloop.agents.prompt_catalog import (
    AGENT_PROMPT_SPECS,
    CONTEXT_TAGS,
    compose_system,
    spec_for,
)
from binderloop.agents.role import LLMStructuredAgent
from binderloop.tools import TOOLS
from scripts.dump_agent_prompts import dump_round_prompts


def _round_payload() -> dict:
    return {
        "round_id": 0,
        "evaluation": {
            "total_candidates": 4,
            "success_count": 0,
            "failure_count": 4,
            "tag_counts": {"hotspot_miss": 3},
            "metric_facts": {"best_iptm": 0.41, "gate_denominators": {"harness_compute_gate": 4}},
            "top_by_score": [{"candidate_id": "c1", "metrics": {"design_to_target_iptm": 0.41}}],
            "top_by_core": [{"candidate_id": "c1", "metrics": {"design_to_target_iptm": 0.41}}],
            "top_by_iptm": [{"candidate_id": "c1", "metrics": {"design_to_target_iptm": 0.41}}],
            "failed_examples": [{"candidate_id": "c2", "tags": ["hotspot_miss"]}],
        },
        "active_learning_examples": {
            "current_round": {
                "counts": {"strict_positive": 0, "near_miss": 2, "other_negative": 2},
                "strict_positive_examples": [],
                "near_miss_examples": [
                    {"id": "n1", "label": "near_miss", "metrics": {"design_to_target_iptm": 0.49}, "tags": ["hotspot_miss"]},
                    {"id": "n2", "label": "near_miss", "metrics": {"design_to_target_iptm": 0.48}, "tags": ["hotspot_miss"]},
                ],
                "other_negative_examples": [
                    {"id": "o1", "label": "other_negative", "metrics": {"design_to_target_iptm": 0.2}, "tags": ["hotspot_miss"]},
                    {"id": "o2", "label": "other_negative", "metrics": {"design_to_target_iptm": 0.21}, "tags": ["binding_pose_failure"]},
                ],
            }
        },
        "structural_analysis": {"total_structures": 1, "aggregate_tags": {"hotspot_not_covered": 1}, "summaries": []},
        "current_config": {"alpha": 0.001, "binder_lengths": [70]},
        "constraints": {"epitope_crop_disabled_hard_constraint": True},
        "quality_analysis": {"overall_assessment": "must not leak into hypothesis compact"},
    }


class PromptCatalogTests(unittest.TestCase):
    def test_tools_registry_exposes_orchestrator_callables(self):
        names = set(TOOLS.names())
        for name in (
            "ingest_results", "evaluate_candidates", "analyze_structures",
            "validate_config", "apply_config_contract", "fact_check_metric_facts",
            "assemble_prompt", "cluster_candidates",
        ):
            self.assertIn(name, names)

    def test_required_tags_are_declared(self):
        self.assertIn("candidates.clusters", CONTEXT_TAGS)
        self.assertIn("HypothesisAgent", AGENT_PROMPT_SPECS)
        spec = spec_for("HypothesisAgent")
        self.assertIn("candidates.clusters", spec.required_tags)
        self.assertNotIn("candidates.leaves", spec.required_tags)

    def test_compose_system_includes_shared_knowledge(self):
        text = compose_system("knowledge.success_gate", "knowledge.al_three_class", "contract.config")
        self.assertIn("iPTM>=0.50", text)
        self.assertIn("near_miss", text)
        self.assertIn("executable", text.lower())

    def test_hypothesis_system_is_composed(self):
        self.assertIn("failure_modes", HypothesisAgent.SYSTEM)
        self.assertIn("iPTM>=0.50", HypothesisAgent.SYSTEM)
        self.assertTrue(issubclass(HypothesisAgent, LLMStructuredAgent))

    def test_hypothesis_projection_omits_quality_and_triple_rankings(self):
        compact = compact_context_for_hypothesis(_round_payload())
        self.assertNotIn("quality_analysis", compact)
        evaluation = compact.get("evaluation") or {}
        self.assertNotIn("top_by_score", evaluation)
        self.assertNotIn("top_by_core", evaluation)
        self.assertNotIn("top_by_iptm", evaluation)

    def test_assembler_tagged_user_excludes_leaves_and_skills(self):
        store = build_store(_round_payload())
        packet = assemble("HypothesisAgent", store, tagged=True)
        user = packet["user"]
        self.assertEqual(user["prompt_version"], "round-context-v1")
        self.assertEqual(user["role"], "HypothesisAgent")
        self.assertIn("candidates.clusters", user)
        self.assertNotIn("candidates.leaves", user)
        self.assertNotIn("active_skills", user)
        self.assertNotIn("quality_analysis", user)

    def test_dump_script_writes_tagged_json_without_live_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round_00"
            round_dir.mkdir()
            payload = _round_payload()
            (round_dir / "evaluation_summary.json").write_text(json.dumps(payload["evaluation"]), encoding="utf-8")
            (round_dir / "structure_evaluation.json").write_text(json.dumps(payload["structural_analysis"]), encoding="utf-8")
            (round_dir / "active_learning_examples.json").write_text(json.dumps(payload["active_learning_examples"]), encoding="utf-8")
            (round_dir / "round_checkpoint.json").write_text(json.dumps({"round_id": 0, "current_jobs": [{"params": payload["current_config"]}]}), encoding="utf-8")
            out_dir = Path(tmp) / "prompt_audit"
            packets = dump_round_prompts(round_dir, roles=["HypothesisAgent"], out_dir=out_dir)
            self.assertEqual(len(packets), 1)
            self.assertTrue((out_dir / "HypothesisAgent.json").exists())
            dumped = json.loads((out_dir / "HypothesisAgent.json").read_text(encoding="utf-8"))
            self.assertEqual(dumped["role"], "HypothesisAgent")
            self.assertIn("candidates.clusters", dumped["user"])
            self.assertNotIn("active_skills", dumped["user"])
            artifacts = load_round_artifacts(round_dir)
            self.assertEqual(artifacts["round_id"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
