#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import binderloop.visualization.iteration_metrics_plot as metrics_plot
import binderloop.orchestration.orchestrator as orchestrator_module
from binderloop.config import HarnessConfig, TargetSpec
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.visualization.iteration_metrics_plot import (
    IterationMetricsRoundCache,
    build_iteration_stats,
    aggregate_round_stats,
    STRUCTURAL_METRICS,
    build_round_analysis_bundle,
    build_round_quality_summary,
    write_round_analysis_bundle,
    IterationMetricsInputError,
    IterationMetricsNoDataError,
    plot_iteration_metrics,
)


class IterationMetricsIncrementalTest(unittest.TestCase):
    @staticmethod
    def _write_legacy_round(
        out_dir: Path,
        round_id: int,
        *,
        iptm: float,
        success_count: int,
    ) -> Path:
        round_dir = out_dir / f"round_{round_id:02d}"
        round_dir.mkdir()
        (round_dir / "ingestions.json").write_text(
            json.dumps(
                [
                    {
                        "candidates": [
                            {
                                "design_to_target_iptm": iptm,
                                "min_design_to_target_pae": 8.0,
                                "design_ptm": 0.75,
                                "unused_large_field": "not needed by plotting",
                            }
                        ]
                    }
                ]
            ),
            encoding="utf-8",
        )
        (round_dir / "evaluation_summary.json").write_text(
            json.dumps(
                {
                    "total_candidates": 5,
                    "success_count": success_count,
                    "failure_count": 5 - success_count,
                    "top_candidates": [{"total": iptm * 10}],
                    "tag_counts": {"ok": success_count},
                }
            ),
            encoding="utf-8",
        )
        (round_dir / "structure_evaluation.json").write_text(
            json.dumps(
                {
                    "total_structures": 2,
                    "reliable_seed_fraction": 0.5,
                    "summaries": [
                        {
                            "high_quality_fragments": [{"id": "high"}],
                            "low_quality_fragments": [{"id": "low"}],
                        }
                    ],
                    "aggregate_tags": {"stable": 1},
                }
            ),
            encoding="utf-8",
        )
        (round_dir / "fragment_templates.json").write_text(
            json.dumps(
                {
                    "templates": [
                        {"reuse_mode": "preserve"},
                        {"reuse_mode": "avoid"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return round_dir

    def test_bundle_is_lightweight_and_preferred_over_legacy_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = self._write_legacy_round(out_dir, 0, iptm=0.8, success_count=3)

            bundle_path = write_round_analysis_bundle(round_dir)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["candidates"], [{"design_to_target_iptm": 0.8, "min_design_to_target_pae": 8.0, "design_ptm": 0.75}])
            self.assertNotIn("unused_large_field", bundle["candidates"][0])

            self._write_conflicting_legacy_payloads(round_dir)
            stats = build_iteration_stats(out_dir)
            summaries = build_round_quality_summary(out_dir)

            iptm_stat = next(item for item in stats if item.metric_key == "design_to_target_iptm")
            self.assertAlmostEqual(iptm_stat.best, 0.8)
            self.assertEqual(summaries[0].success_count, 3)
            self.assertEqual(summaries[0].high_quality_fragment_count, 1)
            self.assertEqual(summaries[0].preserve_template_count, 1)

    @staticmethod
    def _write_conflicting_legacy_payloads(round_dir: Path) -> None:
        (round_dir / "ingestions.json").write_text(
            json.dumps([{"candidates": [{"design_to_target_iptm": 0.1}]}]),
            encoding="utf-8",
        )
        (round_dir / "evaluation_summary.json").write_text(
            json.dumps({"total_candidates": 5, "success_count": 0, "failure_count": 5}),
            encoding="utf-8",
        )

    def test_legacy_outputs_remain_a_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            self._write_legacy_round(out_dir, 0, iptm=0.65, success_count=2)

            stats = build_iteration_stats(out_dir)
            summaries = build_round_quality_summary(out_dir)

            iptm_stat = next(item for item in stats if item.metric_key == "design_to_target_iptm")
            self.assertAlmostEqual(iptm_stat.best, 0.65)
            self.assertEqual(summaries[0].success_count, 2)
            self.assertEqual(summaries[0].low_quality_fragment_count, 1)
            self.assertEqual(summaries[0].avoid_template_count, 1)

    def test_cache_does_not_reread_historical_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            old_round = self._write_legacy_round(out_dir, 0, iptm=0.6, success_count=1)
            cache = IterationMetricsRoundCache()
            read_counts = Counter()
            original_read_json = metrics_plot._read_json

            def tracking_read_json(path: Path, *, default: object) -> object:
                if path.exists():
                    read_counts[path] += 1
                return original_read_json(path, default=default)

            with patch.object(metrics_plot, "_read_json", side_effect=tracking_read_json):
                build_iteration_stats(out_dir, cache=cache)
                first_counts = dict(read_counts)
                build_round_quality_summary(out_dir, cache=cache)
                self.assertEqual(dict(read_counts), first_counts)

                new_round = self._write_legacy_round(out_dir, 1, iptm=0.7, success_count=2)
                build_iteration_stats(out_dir, cache=cache)

                for path, count in first_counts.items():
                    if path.parent == old_round:
                        self.assertEqual(read_counts[path], count)
                self.assertEqual(read_counts[new_round / "ingestions.json"], 1)

                cache.invalidate([0], out_dir=out_dir)
                build_iteration_stats(out_dir, cache=cache)
                self.assertEqual(
                    read_counts[old_round / "ingestions.json"],
                    first_counts[old_round / "ingestions.json"] + 1,
                )

    def test_bundle_helper_accepts_in_memory_data(self) -> None:
        bundle = build_round_analysis_bundle(
            round_id=4,
            candidates=[{"iptm": "0.91", "large_payload": list(range(100))}],
            evaluation={"total_candidates": 1, "success_count": 1},
            structure={"high_quality_fragment_count": 2},
            templates={"preserve_count": 1},
        )

        self.assertEqual(bundle["round_id"], 4)
        self.assertEqual(bundle["candidates"], [{"design_to_target_iptm": 0.91}])
        self.assertEqual(bundle["structure_summary"]["high_quality_fragment_count"], 2)
        self.assertEqual(bundle["template_summary"]["preserve_count"], 1)


    def test_scopes_and_distribution_statistics_are_explicit(self) -> None:
        rows = [
            {"design_to_target_iptm": 0.4, "pass_iptm_filter": False, "pass_filters": False},
            {"design_to_target_iptm": 0.6, "pass_iptm_filter": True, "pass_filters": False},
            {"design_to_target_iptm": 0.8, "pass_iptm_filter": True, "pass_filters": True, "harness_passed": True},
        ]
        all_stat = aggregate_round_stats(0, rows, [STRUCTURAL_METRICS[0]], top_k=2)[0]
        harness_stat = aggregate_round_stats(0, rows, [STRUCTURAL_METRICS[0]], scope="harness_passed")[0]
        self.assertEqual((all_stat.scope, all_stat.n, all_stat.median, all_stat.q25, all_stat.q75), ("all_valid", 3, 0.6, 0.5, 0.7))
        self.assertAlmostEqual(all_stat.top_k_value, 0.7)
        self.assertEqual((harness_stat.scope, harness_stat.n, harness_stat.best), ("harness_passed", 1, 0.8))

    def test_population_candidates_drive_v2_metrics_and_denominators(self) -> None:
        all_candidates = [
            {"design_to_target_iptm": 0.4},
            {"design_to_target_iptm": 0.6},
            {"design_to_target_iptm": 0.8},
        ]
        analysis_candidates = [all_candidates[-1]]
        bundle = build_round_analysis_bundle(
            round_id=2,
            candidates=analysis_candidates,
            population_candidates=all_candidates,
            evaluation={
                "total_candidates": 1,
                "success_count": 2,
                "failure_count": 0,
                "candidate_filtering": {"filtering_applied": True},
            },
        )
        cache = IterationMetricsRoundCache()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "round_02").mkdir()
            cache.add_bundle(out_dir, 2, bundle)
            stat = next(item for item in build_iteration_stats(out_dir, cache=cache) if item.metric_key == "design_to_target_iptm")
        self.assertEqual(bundle["schema_version"], 2)
        self.assertEqual(stat.n, 3)
        self.assertEqual(bundle["evaluation_summary"]["total_candidates"], 3)
        self.assertEqual(bundle["evaluation_summary"]["failure_count"], 1)
        self.assertEqual(bundle["evaluation_summary"]["analysis_candidate_count"], 1)
        self.assertEqual(bundle["evaluation_summary"]["success_count"], 2)

    def test_v1_bundle_falls_back_to_canonical_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = self._write_legacy_round(out_dir, 0, iptm=0.65, success_count=2)
            (round_dir / "round_analysis_bundle.json").write_text(json.dumps({
                "schema": metrics_plot.ROUND_ANALYSIS_BUNDLE_SCHEMA,
                "schema_version": 1,
                "candidates": [{"design_to_target_iptm": 0.1}],
                "evaluation_summary": {}, "structure_summary": {}, "template_summary": {},
            }), encoding="utf-8")
            stat = next(item for item in build_iteration_stats(out_dir) if item.metric_key == "design_to_target_iptm")
            self.assertAlmostEqual(stat.best, 0.65)


    def test_orchestrator_plot_failure_records_status_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            orchestrator = BinderDesignOrchestrator(
                HarnessConfig(target=TargetSpec("target.cif")), out_dir=out_dir, max_rounds=1
            )
            artifacts = {"plot_png": out_dir / "plot.png", "stats_json": out_dir / "stats.json"}
            with patch.object(orchestrator_module, "plot_iteration_metrics", side_effect=[IterationMetricsInputError("bad bundle"), artifacts]) as mocked:
                orchestrator._write_summary({"rounds": [{"round_id": 0}]})
                self.assertEqual(orchestrator._last_plotted_round_count, -1)
                summary = json.loads((out_dir / "orchestrator_summary.json").read_text())
                self.assertEqual(summary["iteration_metrics_plot_status"], {
                    "status": "failed", "event": "iteration_plot_input_failed", "error": "bad bundle"
                })
                orchestrator._write_summary({"rounds": [{"round_id": 0}]})
                self.assertEqual(orchestrator._last_plotted_round_count, 1)
                summary = json.loads((out_dir / "orchestrator_summary.json").read_text())
                self.assertEqual(summary["iteration_metrics_plot_status"]["status"], "completed")
                self.assertEqual(summary["iteration_metrics_plot_status"]["artifacts"]["plot_png"], str(out_dir / "plot.png"))
                self.assertEqual(mocked.call_count, 2)

    def test_orchestrator_plot_no_data_is_stable_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            orchestrator = BinderDesignOrchestrator(
                HarnessConfig(target=TargetSpec("target.cif")), out_dir=out_dir, max_rounds=1
            )
            with patch.object(orchestrator_module, "plot_iteration_metrics", side_effect=IterationMetricsNoDataError("empty")) as mocked:
                orchestrator._write_summary({"rounds": []})
                self.assertEqual(orchestrator._last_plotted_round_count, 0)
                orchestrator._write_summary({"rounds": []})
                self.assertEqual(mocked.call_count, 1)
            summary = json.loads((out_dir / "orchestrator_summary.json").read_text())
            self.assertEqual(summary["iteration_metrics_plot_status"], {
                "status": "no_data", "event": "iteration_plot_no_data", "error": "empty"
            })

    def test_plot_errors_distinguish_no_data_and_bad_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IterationMetricsNoDataError):
                plot_iteration_metrics(tmp)
        with self.assertRaises(IterationMetricsInputError):
            metrics_plot.candidates_for_scope([], "unknown")


if __name__ == "__main__":
    raise SystemExit(unittest.main())
