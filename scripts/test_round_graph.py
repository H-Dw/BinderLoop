#!/usr/bin/env python3
"""RoundGraph wave fan-out and declared Wave A/B/C tags."""

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.orchestration.round_graph import (
    WAVE_A_NODES,
    WAVE_B_NODES,
    WAVE_C_NODES,
    RoundGraph,
    nodes_for_wave,
)


class RoundGraphTests(unittest.TestCase):
    def test_declared_waves_cover_llm_nodes(self):
        self.assertEqual(
            [node.name for node in WAVE_A_NODES],
            ["hypotheses", "diagnostic", "arm_comparison", "quality"],
        )
        self.assertEqual([node.name for node in WAVE_B_NODES], ["quality_manager", "arm_history", "final_strategy"])
        self.assertEqual([node.name for node in WAVE_C_NODES], ["input_config", "policy"])
        hypothesis = nodes_for_wave("A")[0]
        self.assertIn("facts.metric", hypothesis.reads)
        self.assertEqual(hypothesis.writes, ("upstream.hypotheses",))
        self.assertNotIn("upstream.quality", hypothesis.reads)

    def test_run_wave_is_concurrent(self):
        graph = RoundGraph()
        barrier = threading.Barrier(3)
        started = []

        def task(name):
            started.append(name)
            barrier.wait(timeout=2)
            time.sleep(0.01)
            return name

        result = graph.run_wave("A", {
            "hypotheses": lambda: task("hypotheses"),
            "diagnostic": lambda: task("diagnostic"),
            "quality": lambda: task("quality"),
        })
        self.assertFalse(result.errors)
        self.assertEqual(set(result.results), {"hypotheses", "diagnostic", "quality"})
        self.assertEqual(set(started), {"hypotheses", "diagnostic", "quality"})

    def test_run_wave_collects_errors(self):
        graph = RoundGraph()

        def boom():
            raise ValueError("specialist failed")

        result = graph.run_wave("A", {"ok": lambda: "ok", "bad": boom})
        self.assertEqual(result.results["ok"], "ok")
        self.assertIn("bad", result.errors)

    def test_optional_recorder_failure_does_not_mask_node_outcome(self):
        class FailingRecorder:
            def record(self, event_type, payload):
                raise RuntimeError("journal unavailable")

        def boom():
            raise ValueError("node failed")

        result = RoundGraph(event_recorder=FailingRecorder()).run_wave(
            "A", {"ok": lambda: "completed", "bad": boom},
        )

        self.assertEqual("completed", result.results["ok"])
        self.assertIsInstance(result.errors["bad"], ValueError)
        self.assertIn("ok:started", result.telemetry_errors)
        self.assertIn("ok:succeeded", result.telemetry_errors)
        self.assertIn("bad:failed", result.telemetry_errors)

    def test_merge_writes_uses_declared_tags(self):
        graph = RoundGraph()
        state = {}
        graph.merge_writes(state, "A", {"hypotheses": {"n": 1}, "diagnostic": {"ok": True}})
        self.assertEqual(state["upstream.hypotheses"], {"n": 1})
        self.assertEqual(state["upstream.diagnostic"], {"ok": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
