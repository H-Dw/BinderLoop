#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.config import HarnessConfig, TargetSpec
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator


def _cfg(structure_path: str = "target.cif") -> HarnessConfig:
    cfg = HarnessConfig(target=TargetSpec(structure_path=structure_path))
    cfg.resource.max_parallel_jobs = 1
    return cfg


class OrchestratorIoOptimizationTests(unittest.TestCase):
    def test_target_analysis_is_cached_for_unchanged_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = BinderDesignOrchestrator(
                _cfg(),
                out_dir=Path(tmp) / "out",
                max_parallel=1,
            )
            summary = mock.Mock()
            summary.to_dict.return_value = {"structure_file": "target.cif", "chain_id": "A"}
            with mock.patch(
                "binderloop.orchestration.orchestrator.analyze_target_structure",
                return_value=summary,
            ) as analyze:
                first = orchestrator._target_analysis()
                second = orchestrator._target_analysis()

            self.assertEqual(first, second)
            analyze.assert_called_once()

    def test_pressure_conflict_uses_active_memory_without_disk_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = BinderDesignOrchestrator(
                _cfg(),
                out_dir=Path(tmp) / "out",
                max_parallel=1,
            )
            orchestrator._active_memory = object()
            orchestrator._latest_pressure_conflict = {}
            with mock.patch.object(
                orchestrator,
                "_build_tuning_feedback",
                return_value={"pressure_conflict": {}},
            ) as feedback, mock.patch.object(
                orchestrator.memory_store,
                "load",
                side_effect=AssertionError("memory must not be reloaded"),
            ):
                merged, notes = orchestrator._resolve_pressure_conflicts({}, {})

            self.assertEqual(merged, {})
            self.assertEqual(notes, [])
            feedback.assert_called_once()

    def test_summary_only_replots_when_round_count_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = BinderDesignOrchestrator(
                _cfg(),
                out_dir=Path(tmp) / "out",
                max_parallel=1,
            )
            with mock.patch.object(orchestrator, "_write_iteration_metric_plots") as plot:
                orchestrator._write_summary({"rounds": []})
                orchestrator._write_summary({"rounds": []})
                orchestrator._write_summary({"rounds": [{"round_id": 0}]})
                orchestrator._write_summary({"rounds": [{"round_id": 0}]})

            self.assertEqual(plot.call_count, 2)

    def test_checkpoint_artifact_hash_is_reused_until_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = BinderDesignOrchestrator(
                _cfg(),
                out_dir=Path(tmp) / "out",
                max_parallel=1,
            )
            path = orchestrator._write_json(Path(tmp) / "artifact.json", {"version": 1})
            checkpoint = {"artifacts": [str(path)]}
            from binderloop import resume

            original_sha256 = resume.file_sha256
            with mock.patch(
                "binderloop.resume.file_sha256",
                wraps=original_sha256,
            ) as sha256:
                first = orchestrator._checkpoint_artifact_records(checkpoint)
                second = orchestrator._checkpoint_artifact_records(checkpoint)
                # The digest was computed from serialized bytes during the write,
                # so artifact validation performs no Ceph reread.
                self.assertEqual(sha256.call_count, 0)

                orchestrator._write_json(path, {"version": 2})
                third = orchestrator._checkpoint_artifact_records(checkpoint)

            self.assertEqual(first, second)
            self.assertNotEqual(first[0]["sha256"], third[0]["sha256"])

    def test_json_write_populates_digest_cache_without_reread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp) / "out", max_parallel=1)
            from binderloop import resume
            with mock.patch("binderloop.resume.file_sha256", side_effect=AssertionError("write-time digest must be reused")):
                path = orchestrator._write_json(Path(tmp) / "large.json", {"rows": list(range(100))})
                record = orchestrator._artifact_digest_cache.record(path)
            self.assertTrue(record["exists"])
            self.assertIn("sha256", record)

    def test_checkpoint_uses_refs_instead_of_duplicate_large_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out, max_parallel=1)
            round_dir = out / "round_00"; round_dir.mkdir(parents=True)
            (round_dir / "execution_state.json").write_text("{}")
            orchestrator._write_checkpoint(round_dir, 0, "evaluated", "running", {
                "execution_state": {"large": "x" * 10000}, "execution_state_path": str(round_dir / "execution_state.json"),
                "next_jobs": [{"large": "x" * 10000}], "next_jobs_path": str(round_dir / "next_jobs.json"),
            })
            import json
            payload = json.loads((round_dir / "round_checkpoint.json").read_text())
            self.assertNotIn("execution_state", payload)
            self.assertNotIn("next_jobs", payload)



if __name__ == "__main__":
    raise SystemExit(unittest.main())
