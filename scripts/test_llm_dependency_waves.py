#!/usr/bin/env python3
"""LLM agents fan out by dependency wave, not by agent roster order."""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.binder_quality_analysis_agent import BinderQualityAnalysis
from binderloop.agents.context_compaction import compact_context_for_hypothesis
from binderloop.agents.diagnostic_coach_agent import DiagnosticReport
from binderloop.agents.hypothesis_agent import HypothesisSet
from binderloop.config import HarnessConfig, TargetSpec
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator


def _orchestrator(tmp: str) -> BinderDesignOrchestrator:
    cfg = HarnessConfig(target=TargetSpec(structure_path="target.cif"))
    cfg.resource.max_parallel_jobs = 1
    return BinderDesignOrchestrator(cfg, out_dir=Path(tmp) / "out", max_parallel=1)


class FrozenRoundContextTests(unittest.TestCase):
    def test_wave_a_context_drops_later_conclusions(self):
        frozen = BinderDesignOrchestrator._frozen_round_reasoning_context({
            "round_id": 3,
            "evaluation": {"success_count": 1},
            "quality_analysis": {"overall_assessment": "should not leak"},
            "hypotheses": [{"name": "h"}],
            "diagnostic_report": {"status_diagnosis": "later"},
            "arm_comparison": {"winner_arm_id": "a"},
            "arm_history_resolution": {"selected_arm_id": "a"},
            "final_strategy_decision": {"selected_arm_id": "a"},
            "input_configuration": {"recommended_config": {}},
        })
        self.assertEqual(frozen["round_id"], 3)
        self.assertEqual(frozen["evaluation"], {"success_count": 1})
        for key in (
            "quality_analysis", "hypotheses", "diagnostic_report",
            "arm_comparison", "arm_history_resolution", "final_strategy_decision",
            "input_configuration",
        ):
            self.assertNotIn(key, frozen)

    def test_hypothesis_projection_omits_quality(self):
        compact = compact_context_for_hypothesis({
            "round_id": 2,
            "evaluation": {"metric_facts": {"best_iptm": 0.4}, "top_candidates": [], "failed_examples": []},
            "quality_analysis": {"overall_assessment": "fatter prompt only if physically dependent"},
            "current_config": {"alpha": 0.001},
        })
        self.assertNotIn("quality_analysis", compact)
        self.assertNotIn("arm_comparison", compact)


class LlmWaveRunnerTests(unittest.TestCase):
    def test_wave_runs_independent_tasks_concurrently(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = _orchestrator(tmp)
            barrier = threading.Barrier(3)
            started = []

            def task(name):
                started.append(name)
                barrier.wait(timeout=2)
                time.sleep(0.01)
                return name

            result = orchestrator._run_llm_wave(
                wave_name="A",
                round_id=1,
                tasks={
                    "hypotheses": lambda: task("hypotheses"),
                    "diagnostic": lambda: task("diagnostic"),
                    "quality_positive": lambda: task("quality_positive"),
                },
            )
            self.assertEqual(result["hypotheses"], "hypotheses")
            self.assertEqual(set(started), {"hypotheses", "diagnostic", "quality_positive"})

    def test_wave_failure_does_not_return_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = _orchestrator(tmp)

            def boom():
                raise ValueError("specialist failed")

            with self.assertRaises(RuntimeError) as ctx:
                orchestrator._run_llm_wave(
                    wave_name="A",
                    round_id=1,
                    tasks={"ok": lambda: "ok", "bad": boom},
                )
            self.assertIn("llm wave A failed", str(ctx.exception))
            self.assertIn("bad", str(ctx.exception))

    def test_empty_wave_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = _orchestrator(tmp)
            self.assertEqual(orchestrator._run_llm_wave(wave_name="A", round_id=1, tasks={}), {})


class WaveArtifactResumeTests(unittest.TestCase):
    def test_artifacts_ready_requires_all_wave_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "r0"
            round_dir.mkdir()
            self.assertFalse(BinderDesignOrchestrator._round_llm_wave_artifacts_ready(round_dir))
            names = [
                "binder_quality_analysis.json",
                "hypotheses.json",
                "diagnostic_report.json",
                "arm_comparison.json",
                "arm_history_resolution.json",
                "final_strategy_decision.json",
            ]
            for name in names[:-1]:
                (round_dir / name).write_text("{}", encoding="utf-8")
            self.assertFalse(BinderDesignOrchestrator._round_llm_wave_artifacts_ready(round_dir))
            (round_dir / names[-1]).write_text("{}", encoding="utf-8")
            self.assertTrue(BinderDesignOrchestrator._round_llm_wave_artifacts_ready(round_dir))

    def test_resume_holder_rehydrates_wave_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = _orchestrator(tmp)
            round_dir = Path(tmp) / "r1"
            round_dir.mkdir()
            quality = BinderQualityAnalysis(round_id=1, llm_used=False, overall_assessment="ok")
            hypotheses = HypothesisSet(hypotheses=[{"name": "h1"}], llm_used=False)
            diagnostic = DiagnosticReport(round_id=1, llm_used=False, status_diagnosis="stable")
            (round_dir / "binder_quality_analysis.json").write_text(json.dumps({
                "round_id": 1, "llm_used": False, "overall_assessment": "ok",
                "extra_ignored": True,
            }), encoding="utf-8")
            (round_dir / "hypotheses.json").write_text(json.dumps({
                "hypotheses": [{"name": "h1"}], "llm_used": False,
            }), encoding="utf-8")
            (round_dir / "diagnostic_report.json").write_text(json.dumps({
                "round_id": 1, "llm_used": False, "status_diagnosis": "stable",
            }), encoding="utf-8")
            (round_dir / "arm_comparison.json").write_text(json.dumps({"winner_arm_id": "baseline_hold"}), encoding="utf-8")
            (round_dir / "arm_history_resolution.json").write_text(json.dumps({"selected_arm_id": "baseline_hold"}), encoding="utf-8")
            (round_dir / "final_strategy_decision.json").write_text(json.dumps({"selected_arm_id": "baseline_hold"}), encoding="utf-8")
            holder = orchestrator._load_llm_wave_holder(round_dir)
            self.assertEqual(holder["quality_analysis"].overall_assessment, quality.overall_assessment)
            self.assertEqual(holder["hypotheses"].hypotheses, hypotheses.hypotheses)
            self.assertEqual(holder["diagnostic"].status_diagnosis, diagnostic.status_diagnosis)
            self.assertEqual(holder["final_strategy_decision"]["selected_arm_id"], "baseline_hold")


if __name__ == "__main__":
    unittest.main(verbosity=2)
