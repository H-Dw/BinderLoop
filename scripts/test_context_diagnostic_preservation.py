#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.context_compaction import build_metric_facts, compact_context_for_diagnostic, compact_memory, enforce_byte_budget
from binderloop.agents.diagnostic_coach_agent import DiagnosticCoachAgent
from binderloop.agents.memory_retrieval_agent import MemoryRetrievalAgent, MemoryRetrievalQuery
from binderloop.memory import MemoryItem


class ContextDiagnosticPreservationTest(unittest.TestCase):
    def test_structured_diagnostics_and_lineage_survive(self):
        evaluation = {"total_candidates": 4, "success_count": 1, "tag_counts": {"binding_pose_failure": 2, "primary_gate_high_pae": 1, "hotspot_miss": 3, "folding_failure": 1}}
        facts = build_metric_facts(evaluation, candidates=[])
        self.assertEqual(facts["gate_denominators"]["harness_compute_gate"], 4)
        self.assertEqual(facts["diagnostic_signals"]["high_pae_count"], 1)
        ctx = compact_context_for_diagnostic(round_id=2, monitor_snapshot={"state": "done", "execution_parameters": {"alpha": .1}, "merge_overrides": {"alpha": {"before": .2, "after": .1}}, "rollback_lineage": {"branch_from_round": 0}}, metrics_summary={}, evaluation_summary={**evaluation, "metric_facts": facts, "hotspot_coverage": .25, "foldability": {"pass": 2}}, structural_analysis=None, job_history=[], config={})
        self.assertEqual(ctx["evaluation"]["hotspot_coverage"], .25)
        self.assertEqual(ctx["monitor"]["rollback_lineage"]["branch_from_round"], 0)

    def test_compact_memory_keeps_execution_merge_and_rollback(self):
        payload = compact_memory({"recent_rounds": [{"round_id": 1, "evaluation": {"total_candidates": 2, "tag_counts": {"hotspot_miss": 1}}, "config_snapshot": {"alpha": .1}, "config_merge_report": {"overrides": ["alpha"]}, "rollback_decision": {"branch_from_round": 0}}]})
        row = payload["recent_rounds"][0]
        self.assertEqual(row["execution_parameters"]["alpha"], .1)
        self.assertEqual(row["rollback_lineage"]["branch_from_round"], 0)

    def test_budget_records_audit(self):
        guarded = enforce_byte_budget({"round_id": 1, "note": "x" * 10000}, max_bytes=500)
        audit = guarded["_context_compaction"]
        self.assertEqual(audit["policy"], "deterministic_progressive_truncation")
        self.assertLessEqual(audit["final_bytes"], 500)

    def test_field_level_repair_preserves_independent_advice(self):
        context = compact_context_for_diagnostic(round_id=1, monitor_snapshot={}, metrics_summary={}, evaluation_summary={"total_candidates": 2, "success_count": 0, "metric_facts": {"best_iptm": .55, "additional_filter_pass": {"pass_count": 1}}}, structural_analysis=None, job_history=[], config={})
        result = {"status_diagnosis": "Filter eliminated all candidates", "root_causes": [{"cause": "sampling is narrow"}], "metric_interpretation": {}, "corrective_actions": [{"action": "preserve diversity", "parameter_changes": {"alpha": .01}}], "monitoring_recommendations": [], "pipeline_health": {}}
        repaired, audit = DiagnosticCoachAgent._repair_fact_invalid_fields(round_id=1, context=context, result=result, facts=context["evaluation"]["metric_facts"])
        self.assertIn("status_diagnosis", audit["repaired_fields"])
        self.assertEqual(repaired["root_causes"], result["root_causes"])
        self.assertEqual(repaired["corrective_actions"], result["corrective_actions"])
        self.assertTrue(audit["rejected_claims"])

    def test_retrieval_is_stable_without_semantic_rerank(self):
        rows = [MemoryItem(item_id=x, round_id=1, failure_tags=["pose"], summary="same") for x in ("b", "a")]
        agent = MemoryRetrievalAgent(top_k=2)
        query = MemoryRetrievalQuery(failure_tags=["pose"])
        first = [x.item_id for x in agent.retrieve(rows, query).items]
        second = [x.item_id for x in agent.retrieve(list(reversed(rows)), query).items]
        self.assertEqual(first, second)
        self.assertFalse(agent.retrieve(rows, query).semantic_rerank_used)


if __name__ == "__main__":
    unittest.main(verbosity=2)
