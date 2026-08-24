#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.visualization.iteration_metrics_plot import (
    STRUCTURAL_METRICS,
    _nan_bridge_segments,
    aggregate_round_stats,
    build_iteration_stats,
    build_round_quality_summary,
    discover_round_dirs,
    load_round_candidates,
    plot_iteration_metrics,
)


class IterationMetricsPlotTest(unittest.TestCase):
    OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "sc2rbd_closed_loop_llm_5r"

    def test_discover_rounds_on_current_output(self) -> None:
        if not self.OUT_DIR.exists():
            self.skipTest(f"output directory not found: {self.OUT_DIR}")
        rounds = discover_round_dirs(self.OUT_DIR)
        self.assertGreaterEqual(len(rounds), 5)
        self.assertEqual(rounds[0][0], 0)

    def test_build_stats_has_all_rounds_and_metrics(self) -> None:
        if not self.OUT_DIR.exists():
            self.skipTest(f"output directory not found: {self.OUT_DIR}")
        stats = build_iteration_stats(self.OUT_DIR)
        self.assertGreater(len(stats), 0)
        round_ids = {item.round_id for item in stats}
        self.assertEqual(round_ids, {0, 1, 2, 3, 4})
        metric_keys = {item.metric_key for item in stats}
        expected = {spec.key for spec in STRUCTURAL_METRICS}
        self.assertTrue(expected.issubset(metric_keys))

    def test_aggregate_round_stats_best_direction(self) -> None:
        candidates = [
            {"design_to_target_iptm": "0.1", "min_design_to_target_pae": "20", "design_ptm": "0.7"},
            {"design_to_target_iptm": "0.5", "min_design_to_target_pae": "5", "design_ptm": "0.8"},
        ]
        stats = {item.metric_key: item for item in aggregate_round_stats(0, candidates)}
        self.assertAlmostEqual(stats["design_to_target_iptm"].best, 0.5)
        self.assertAlmostEqual(stats["min_design_to_target_pae"].best, 5.0)
        self.assertAlmostEqual(stats["design_ptm"].mean, 0.75)

    def test_design_folding_rmsd_uses_backbone_design_alias(self) -> None:
        candidates = [
            {"bb_rmsd_design": "1.4"},
            {"designfolding-bb_rmsd_design": "0.9"},
            {"designfolding-filter_rmsd": "0.5", "bb_rmsd_design": "9.9"},
        ]
        stats = {item.metric_key: item for item in aggregate_round_stats(0, candidates)}
        design_folding = stats["designfolding-filter_rmsd"]
        self.assertEqual(design_folding.n, 3)
        self.assertAlmostEqual(design_folding.best, 0.5)
        self.assertAlmostEqual(design_folding.mean, (1.4 + 0.9 + 0.5) / 3)

    def test_load_round_candidates_prefers_final_metrics_from_mixed_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round_00"
            round_dir.mkdir()
            (round_dir / "ingestions.json").write_text(
                json.dumps(
                    [
                        {
                            "candidates": [
                                {
                                    "id": "final_a",
                                    "design_to_target_iptm": "0.2",
                                    "_metrics_file": "/run/gpu_0/final_ranked_designs/final_designs_metrics_6.csv",
                                },
                                {
                                    "id": "all_a",
                                    "design_to_target_iptm": "0.3",
                                    "_metrics_file": "/run/gpu_0/final_ranked_designs/all_designs_metrics.csv",
                                },
                                {
                                    "id": "intermediate_a",
                                    "design_to_target_iptm": "0.9",
                                    "_metrics_file": "/run/gpu_0/intermediate_designs_inverse_folded/aggregate_metrics_analyze.csv",
                                },
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )

            candidates = load_round_candidates(round_dir)

            self.assertEqual([row["id"] for row in candidates], ["final_a"])

    def test_nan_bridge_segments_only_span_missing_rounds(self) -> None:
        segments = _nan_bridge_segments([0, 1, 2, 3, 4, 5], [1.0, float("nan"), 3.0, 4.0, float("nan"), 6.0])
        self.assertEqual(segments, [(0, 1.0, 2, 3.0), (3, 4.0, 5, 6.0)])

    def test_plot_writes_png_and_json(self) -> None:
        if not self.OUT_DIR.exists():
            self.skipTest(f"output directory not found: {self.OUT_DIR}")
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifacts = plot_iteration_metrics(
                self.OUT_DIR,
                output_path=tmp_path / "plot.png",
                stats_json_path=tmp_path / "stats.json",
            )
            self.assertTrue(artifacts["plot_png"].exists())
            self.assertTrue(artifacts["stats_json"].exists())
            payload = json.loads(artifacts["stats_json"].read_text(encoding="utf-8"))
            self.assertGreater(len(payload), 0)

    def test_plot_reads_success_binder_counts_per_round(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for round_id, success_count in ((0, 1), (1, 3)):
                round_dir = out_dir / f"round_{round_id:02d}"
                round_dir.mkdir()
                (round_dir / "evaluation_summary.json").write_text(
                    json.dumps({"total_candidates": 5, "success_count": success_count, "failure_count": 5 - success_count}),
                    encoding="utf-8",
                )
                (round_dir / "ingestions.json").write_text(
                    json.dumps(
                        [
                            {
                                "candidates": [
                                    {
                                        "design_to_target_iptm": str(0.2 + round_id),
                                        "min_design_to_target_pae": "10",
                                        "design_ptm": "0.8",
                                    }
                                ]
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

            summaries = build_round_quality_summary(out_dir)
            self.assertEqual([item.success_count for item in summaries], [1, 3])

            artifacts = plot_iteration_metrics(out_dir)
            self.assertTrue(artifacts["plot_png"].exists())


if __name__ == "__main__":
    raise SystemExit(unittest.main())
