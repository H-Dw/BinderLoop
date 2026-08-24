#!/usr/bin/env python3
"""Verify that every LLM-agent prompt context stays under the 1 MB hard cap.

This test builds a deliberately *pathological* round context — huge
``ca_coordinates`` arrays, thousands of candidates, very long observation
strings, deep message logs — and feeds it through every per-agent compactor
plus the final ``enforce_byte_budget`` guard.  It asserts that:

1. Each per-agent compactor already strips the heavy/unbounded payloads.
2. The final guard *guarantees* the serialised payload (as the LLM client
   sends it, i.e. ``indent=2``) is below ``MAX_PROMPT_BYTES`` even for the
   worst case, including a degenerate input that is already > 1 MB of scalars.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.context_compaction import (
    MAX_PROMPT_BYTES,
    compact_context_for_blocked_arm_review,
    compact_context_for_config_validation,
    compact_context_for_diagnostic,
    compact_context_for_hypothesis,
    compact_context_for_input_config,
    compact_context_for_quality,
    compact_context_for_target_config,
    enforce_byte_budget,
)
from binderloop.skills import compose_agent_system


def _wire_bytes(payload) -> int:
    """Bytes exactly as ``chat_json`` serialises the user payload."""
    return len(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def _huge_structure_summary(idx: int) -> dict:
    return {
        "structure_file": f"/tmp/struct_{idx}.cif",
        "reliability_score": 0.5,
        "reliability_tags": ["reliable"] * 20,
        "interface_contact_count": 30,
        "hotspot_contacts": 3,
        # Heavy / unbounded payloads that MUST be stripped:
        "ca_coordinates": [[float(i), float(i + 1), float(i + 2)] for i in range(5000)],
        "pae_matrix": [[0.1] * 200 for _ in range(200)],
        "high_quality_fragments": [
            {
                "fragment_id": f"frag_{idx}_{j}",
                "quality_score": 0.8,
                "binder_sequence": "ACDEFGHIKL" * 50,
                "ca_coordinates": [[1.0, 2.0, 3.0]] * 1000,
            }
            for j in range(30)
        ],
    }


def _pathological_context() -> dict:
    return {
        "round_id": 3,
        "evaluation": {
            "total_candidates": 5000,
            "success_count": 0,
            "failure_count": 5000,
            "tag_counts": {f"tag_{i}": i for i in range(200)},
            "observations": "X" * 100_000,
            "top_candidates": [
                {"candidate_id": f"c{i}", "iptm": 0.1, "raw_metrics": [0.0] * 500}
                for i in range(2000)
            ],
            "failed_examples": [
                {"candidate_id": f"f{i}", "pae_matrix": [[0.1] * 100 for _ in range(100)]}
                for i in range(500)
            ],
        },
        "structural_analysis": {
            "total_structures": 200,
            "aggregate_tags": {f"st_{i}": i for i in range(100)},
            "reliable_seed_fraction": 0.4,
            "observations": "Y" * 50_000,
            "summaries": [_huge_structure_summary(i) for i in range(200)],
        },
        "fragment_templates": {
            "total_templates": 300,
            "high_quality_count": 50,
            "ca_coordinates": [[1.0, 2.0, 3.0]] * 50000,
            "structure_redesign": {
                "source_structure_file": "/tmp/best.cif",
                "quality_score": 0.9,
                "binder_sequence": "ACDEFG" * 1000,
                "ca_coordinates": [[1.0, 2.0, 3.0]] * 10000,
            },
        },
        "target_analysis": {"chain": "E", "coords": [[1, 2, 3]] * 100000},
        "current_config": {"alpha": 0.001, "binder_lengths": [80, 90], "hotspots": ["E:153"]},
        "constraints": {"max_binders_per_round": 32},
        "memory": {
            "recent_rounds": [
                {"round_id": r, "evaluation": {"total_candidates": 100}, "jobs": [{"binder_length": 80}] * 50}
                for r in range(50)
            ],
        },
        "messages": [{"role": "agent", "content": "Z" * 5000} for _ in range(500)],
        "quality_analysis": {
            "overall_assessment": "W" * 20000,
            "causal_factors": [{"factor": f"cf_{i}"} for i in range(100)],
            "next_round_guidance": [{"action": f"g_{i}"} for i in range(100)],
        },
        "diagnostic_report": {
            "status_diagnosis": "D" * 20000,
            "corrective_actions": [{"action": f"a_{i}"} for i in range(100)],
            "pipeline_health": {"execution_ok": True},
        },
        "hypotheses": [{"name": f"h_{i}", "confidence": 0.5} for i in range(50)],
    }


class PromptByteBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = _pathological_context()
        # Sanity: the raw context is well over the budget so the test is meaningful.
        self.assertGreater(_wire_bytes(self.ctx), MAX_PROMPT_BYTES)

    def _assert_under_budget(self, label: str, payload) -> None:
        guarded = enforce_byte_budget(payload)
        size = _wire_bytes(guarded)
        self.assertLessEqual(
            size, MAX_PROMPT_BYTES,
            f"{label}: {size} bytes exceeds MAX_PROMPT_BYTES={MAX_PROMPT_BYTES}",
        )
        print(f"  {label:38s} -> {size:>9,d} bytes (<= {MAX_PROMPT_BYTES:,d})")

    def test_hypothesis_under_budget(self) -> None:
        self._assert_under_budget("hypothesis", {"context": compact_context_for_hypothesis(self.ctx)})

    def test_quality_under_budget(self) -> None:
        self._assert_under_budget(
            "quality", {"round_id": 3, "context": compact_context_for_quality(self.ctx)}
        )

    def test_diagnostic_under_budget(self) -> None:
        compact = compact_context_for_diagnostic(
            round_id=3,
            monitor_snapshot={"state": "done", "status_counts": {"ok": 0, "fail": 5000}},
            metrics_summary={"iptm": {"mean": 0.1}},
            evaluation_summary=self.ctx["evaluation"],
            structural_analysis=self.ctx["structural_analysis"],
            job_history=self.ctx["memory"]["recent_rounds"],
            config=self.ctx["current_config"],
        )
        self._assert_under_budget("diagnostic", {"round_id": 3, "pipeline_state": compact})

    def test_input_config_next_round_under_budget(self) -> None:
        compact = compact_context_for_input_config(
            target_name="sc2rbd",
            current_config=self.ctx["current_config"],
            diagnostic_report=self.ctx["diagnostic_report"],
            evaluation_summary=self.ctx["evaluation"],
            round_id=4,
            structural_analysis=self.ctx["structural_analysis"],
            quality_analysis=self.ctx["quality_analysis"],
            hypotheses=self.ctx["hypotheses"],
            memory_summary=self.ctx["memory"],
            constraints=self.ctx["constraints"],
        )
        self._assert_under_budget("input_config.next_round", compact)

    def test_input_config_target_under_budget(self) -> None:
        target_ctx = {
            "target_name": "sc2rbd",
            "target_info": self.ctx["target_analysis"],
            "previous_results": self.ctx["evaluation"],
            "constraints": self.ctx["constraints"],
        }
        compact = compact_context_for_target_config(target_ctx)
        self._assert_under_budget("input_config.target", compact)

    def test_config_validation_under_budget(self) -> None:
        compact = compact_context_for_config_validation(
            target_model="boltzgen",
            activation="taiji_failure",
            config={"alpha": 0.001, "config_overrides": [["filtering", "x=1"]], "junk": "Q" * 50000},
            deterministic_prefilter={"is_valid": False, "corrected_config": {"alpha": 0.001}, "issues": [{"x": i} for i in range(50)]},
            context={"error_context": {"message": "E" * 80000, "exit_code": 1, "stderr_tail": "S" * 80000}},
        )
        self._assert_under_budget("config_validation", compact)

    def test_blocked_arm_review_specialized_compactor(self) -> None:
        payload = compact_context_for_blocked_arm_review(
            round_id=4,
            blocked_arms=[{"arm_id":"sampler_explore","reason":"regressed","status":"soft_blocked","junk":"X"*100000}],
            evidence=[{"evidence_id":"E1","arm_id":"sampler_explore","status":"closed","requested_budget":4,"completed_budget":4,"trials":4,"successes":1,"positive_features":[str(i) for i in range(100)],"ca_coordinates":[[1,2,3]]*10000,"unknown":"bad"}],
            context={"selection_context":{"score":1,"nested":{"ok":True},"blob":"Y"*100000},"hypotheses":self.ctx["hypotheses"],"quality_analysis":self.ctx["quality_analysis"],"structural_summary":self.ctx["structural_analysis"],"ledger_history":{"recent_rounds":[{"round_id":1,"per_arm_outcomes":[{"arm_id":"sampler_explore","evidence_id":"OLD"},{"arm_id":"other","evidence_id":"DROP"}],"policy_snapshot":{"huge":"Z"*100000}}]}})
        self._assert_under_budget("blocked_arm_review", payload)
        self.assertEqual(payload["blocked_arms"][0]["arm_id"],"sampler_explore")
        self.assertEqual(payload["evidence"][0]["evidence_id"],"E1")
        self.assertLessEqual(len(payload["evidence"][0]["positive_features"]),8)
        rendered=json.dumps(payload)
        self.assertNotIn("ca_coordinates",rendered); self.assertNotIn("policy_snapshot",rendered)
        self.assertNotIn("DROP",rendered); self.assertIn("OLD",rendered)

    def test_degenerate_scalar_blob_still_capped(self) -> None:
        # A payload that is > 1 MB of pure scalar string (no heavy keys to strip)
        # must still be forced under budget by the progressive shrinker / stub.
        blob = {"round_id": 1, "task": "x", "note": "N" * 2_000_000}
        self._assert_under_budget("degenerate_scalar_blob", blob)

    def test_learned_skill_system_block_has_independent_budget(self) -> None:
        skills = [{
            "id": "run-local-self-improvement",
            "type": "llm_reasoning",
            "priority": 900,
            "origin": "run_local_self_improvement",
            "guidance": ["rule guidance " + ("X" * 20_000)] * 20,
            "learned_rules": [{"rule_id": "r%d" % index, "strategy": "Y" * 20_000}
                              for index in range(20)],
        }]
        rendered = compose_agent_system(
            "BASE",
            active_skills=skills,
            max_skill_bytes=12_000,
        )
        self.assertLessEqual(len(rendered.encode("utf-8")), 12_100)
        self.assertIn("truncated", rendered)


if __name__ == "__main__":
    print(f"MAX_PROMPT_BYTES = {MAX_PROMPT_BYTES:,d}")
    unittest.main(verbosity=2)
