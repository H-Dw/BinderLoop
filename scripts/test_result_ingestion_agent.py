#!/usr/bin/env python3
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.evaluation_agent import EvaluationAgent
from binderloop.agents.result_ingestion_agent import ResultIngestionAgent
from binderloop.agents.structure_evaluation_agent import StructureEvaluationAgent
from binderloop.config import HarnessConfig, TargetSpec
from binderloop.models.base import DesignJob
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator


class AggregateResultIngestionTest(unittest.TestCase):
    def _metrics(self, path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row}) or ["id"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)

    def test_collects_only_native_final_structures_without_candidate_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            ranked = root / "gpu_00/shard_000/final_ranked_designs"
            structures = [
                root / "gpu_00/shard_000/intermediate_designs/native_0.cif",
                root / "gpu_00/shard_000/intermediate_designs_inverse_folded/refold_cif/native_0_0.cif",
                ranked / "final_1_designs/before_refolding/rank0_native_0_0.cif",
                ranked / "final_1_designs/rank0_native_0_0.cif",
            ]
            metrics = ranked / "final_designs_metrics_1.csv"
            self._metrics(metrics, [{"id": "native_0_0", "pass_filters": "True", "design_to_target_iptm": ".5"}])
            for index, path in enumerate(structures):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(f"data_{index}")
            run = ResultIngestionAgent().ingest_boltzgen_output(root, log_file=root / "missing.log")
            self.assertEqual(run.structure_files, [str(structures[-1])])
            self.assertEqual(run.structure_file_count, 1)
            self.assertFalse(any("stage" in row for row in run.candidates))
            self.assertEqual(run.collection_mode, "round_aggregate")
            self.assertFalse(run.population_metadata["candidate_structure_attribution"])

    def test_direct_gpu0_final_ranked_designs_are_ingested_as_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            ranked = root / "gpu_0/final_ranked_designs"
            metrics = ranked / "final_designs_metrics_1.csv"
            structure = ranked / "final_1_designs/rank0_native.cif"
            self._metrics(metrics, [{"id": "native_0", "pass_filters": "True", "design_to_target_iptm": ".5"}])
            structure.parent.mkdir(parents=True, exist_ok=True)
            structure.write_text("data_direct")
            run = ResultIngestionAgent().ingest_boltzgen_output(root)
            self.assertEqual(run.structure_files, [str(structure)])
            self.assertEqual(run.metrics_files, [str(metrics)])
            self.assertGreaterEqual(len(run.candidates), 1)
            self.assertEqual(run.candidates[0]["native_gpu"], "gpu_0")

    def test_manifest_inventory_preserves_paths_and_ignores_unlisted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            structure = root / "host_00/gpu_00/shard_000/final_ranked_designs/final_1_designs/native.cif"
            metrics = root / "host_00/gpu_00/shard_000/final_ranked_designs/final_designs_metrics_1.csv"
            self._metrics(metrics, [{"id": "x"}]); structure.parent.mkdir(parents=True, exist_ok=True); structure.write_text("x")
            unlisted = root / "host_00/unlisted.cif"; unlisted.write_text("x")
            root.mkdir(parents=True, exist_ok=True)
            (root / "result_manifest.json").write_text(json.dumps({"schema_version": 4, "files": [str(metrics.relative_to(root)), str(structure.relative_to(root))]}))
            run = ResultIngestionAgent().ingest_boltzgen_output(root)
            self.assertEqual(run.structure_files, [str(structure)])
            self.assertNotIn(str(unlisted), run.structure_files)


    def test_aggregate_expands_one_level_shards_and_validates_entities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            metrics = root / "host_00/gpu_01/shard_002/final_ranked_designs/final_designs_metrics_1.csv"
            structure = root / "host_00/gpu_01/shard_002/final_ranked_designs/final_1_designs/native.cif"
            self._metrics(metrics, [{"id": "native", "pass_filters": "True"}])
            structure.parent.mkdir(parents=True, exist_ok=True); structure.write_text("data")
            shard_ref = root / "host_00/shard_result_manifest.json"
            shard_ref.write_text(json.dumps({
                "files": [str(metrics.relative_to(root)), str(structure.relative_to(root)), "host_00/missing.csv"],
                "shard_manifests": ["host_00/nested.json"],
            }))
            (root / "result_manifest.json").write_text(json.dumps({
                "schema_version": 6, "collection_mode": "round_aggregate",
                "files": [str(shard_ref.relative_to(root))],
                "shard_manifests": [str(shard_ref.relative_to(root))],
            }))
            run = ResultIngestionAgent().ingest_boltzgen_output(root)
            self.assertEqual(run.metrics_files, [str(metrics)])
            self.assertEqual(run.structure_files, [str(structure)])
            self.assertIn("result_manifest_entity_missing:host_00/missing.csv", run.run_level_issues)
            self.assertIn("nested_shard_manifests_ignored:host_00/shard_result_manifest.json", run.run_level_issues)
            row = run.candidates[0]
            self.assertEqual((row["native_host"], row["native_gpu"], row["native_shard"]), ("host_00", "gpu_01", "shard_002"))
            shard = run.native_inventory["hosts"][0]["gpus"][0]["shards"][0]
            self.assertEqual(shard["shard"], "shard_002")
            self.assertEqual(shard["metrics"], [str(metrics)])

    def test_native_multihost_final_pdb_flows_into_structure_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            ranked = root / "host_00/gpu_01/shard_002/final_ranked_designs"
            metrics = ranked / "final_designs_metrics_1.csv"
            structure = ranked / "final_1_designs/final_native_design.pdb"
            self._metrics(metrics, [{"id": "native", "pass_filters": "True"}])
            structure.parent.mkdir(parents=True, exist_ok=True)
            structure.write_text(self._minimal_complex_pdb(), encoding="utf-8")

            shard_manifest = root / "host_00/gpu_01/shard_002/shard_result_manifest.json"
            shard_manifest.write_text(json.dumps({
                "files": [str(metrics.relative_to(root)), str(structure.relative_to(root))],
            }), encoding="utf-8")
            (root / "result_manifest.json").write_text(json.dumps({
                "schema_version": 6,
                "collection_mode": "round_aggregate",
                "files": [str(shard_manifest.relative_to(root))],
                "shard_manifests": [str(shard_manifest.relative_to(root))],
            }), encoding="utf-8")

            ingested = ResultIngestionAgent().ingest_boltzgen_output(root)
            evaluated = StructureEvaluationAgent().analyze_structures(
                ingested.structure_files,
                binder_chain="B",
                target_chains=["A"],
                auto_detect_chains=False,
            )

            self.assertEqual(ingested.structure_files, [str(structure)])
            self.assertGreater(evaluated.total_structures, 0)
            self.assertTrue(evaluated.summaries)
            self.assertGreater(evaluated.summaries[0]["atom_count"], 0)
            self.assertGreater(evaluated.summaries[0]["binder_residue_count"], 0)

    @staticmethod
    def _minimal_complex_pdb():
        atoms = [
            (1, "ALA", "A", 1, 0.0, 0.0, 0.0),
            (2, "GLU", "A", 2, 3.8, 0.0, 0.0),
            (3, "SER", "A", 3, 7.6, 0.0, 0.0),
            (4, "LEU", "B", 1, 0.0, 4.0, 0.0),
            (5, "TYR", "B", 2, 3.8, 4.0, 0.0),
        ]
        return "\n".join(
            f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            for serial, resname, chain, resseq, x, y, z in atoms
        ) + "\nEND\n"

    def test_selected_and_unfiltered_same_metrics_file_is_opened_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "final_ranked_designs/all_designs_metrics.csv"
            self._metrics(metrics, [{"id": "one", "pass_filters": "True"}])
            (root / "result_manifest.json").write_text(json.dumps({"files": [str(metrics.relative_to(root))]}))
            original = Path.open
            opened = []
            def tracked(path, *args, **kwargs):
                if Path(path) == metrics:
                    opened.append(str(path))
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", tracked):
                run = ResultIngestionAgent().ingest_boltzgen_output(root, log_file=root / "missing.log")
            self.assertEqual(run.metrics_rows_read, 1)
            self.assertEqual(opened.count(str(metrics)), 1)

    def test_log_tail_uses_bounded_binary_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.log"
            path.write_bytes(b"x" * 100000 + b"THE_END")
            tail = ResultIngestionAgent._read_tail(path, max_chars=32)
            self.assertTrue(tail.endswith("THE_END"))
            self.assertLessEqual(len(tail), 32)

    def test_manifest_unsafe_path_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"; root.mkdir()
            (root / "result_manifest.json").write_text(json.dumps({"files": ["../escape.cif"]}))
            run = ResultIngestionAgent().ingest_boltzgen_output(root)
            self.assertIn("result_manifest_unsafe_path:../escape.cif", run.run_level_issues)
            self.assertEqual(run.structure_files, [])

    def test_selected_and_true_filter_pass_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"; ranked = root / "final_ranked_designs"
            self._metrics(ranked / "final_designs_metrics_2.csv", [
                {"id": "a", "pass_filters": "True"}, {"id": "b", "pass_filters": "False"},
            ])
            run = ResultIngestionAgent().ingest_boltzgen_output(root)
            self.assertEqual((run.selected_metric_count, run.filter_pass_count, run.selected_failed_filter_count), (2, 1, 1))
            self.assertEqual(run.candidate_scope, "selected_ranked")

    def test_zero_pass_recovers_unfiltered_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"; ranked = root / "final_ranked_designs"
            self._metrics(ranked / "final_designs_metrics_1.csv", [{"id": "selected", "pass_filters": "False"}])
            self._metrics(ranked / "all_designs_metrics.csv", [{"id": "recovery", "pass_filters": "False"}])
            run = ResultIngestionAgent().ingest_boltzgen_output(root)
            self.assertEqual(run.candidate_scope, "unfiltered_zero_pass_recovery")
            self.assertEqual([row["id"] for row in run.candidates], ["recovery"])
            self.assertEqual(run.filter_pass_count, 0)

    def test_rows_preserve_native_and_identity_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"; metrics = root / "final_ranked_designs/final_designs_metrics_1.csv"
            self._metrics(metrics, [{"id": "native", "global_candidate_id": "untrusted", "design_to_target_iptm": ".4"}])
            row = ResultIngestionAgent().ingest_boltzgen_output(root, identity_context={"job_id": "job", "arm_id": "arm", "exploration_arm": "arm", "logical_branch_id": "r1_arm", "arm_root": str(root), "output_root": str(root)}).candidates[0]
            self.assertEqual(row["id"], "native")
            self.assertEqual(row["job_id"], "job"); self.assertEqual(row["arm_id"], "arm")
            self.assertIn("_metrics_file", row); self.assertEqual(row["_metrics_row_ordinal"], 0)

    def test_metrics_truncation_is_separate_from_structure_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"; ranked = root / "final_ranked_designs"
            self._metrics(ranked / "final_designs_metrics_3.csv", [{"id": value} for value in "abc"])
            for value in "abcd":
                path = ranked / f"final_4_designs/{value}.cif"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value)
            with self.assertWarnsRegex(RuntimeWarning, "above the 2-row LLM evidence budget"):
                run = ResultIngestionAgent().ingest_boltzgen_output(root, max_rows=2)
            self.assertFalse(run.metrics_rows_truncated)
            self.assertEqual(run.metrics_rows_read, 3); self.assertEqual(run.structure_file_count, 4)
            self.assertTrue(run.metrics_rows_over_limit)
            self.assertEqual(run.population_metadata["metrics_selection_policy"], "full_ingestion_then_skill_guided_compaction")
            self.assertTrue(run.population_metadata["populations_are_not_row_aligned"])


    def test_orchestrator_failure_skips_explicit_symlink_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            arm_root = base / "arm"
            job_output = arm_root / "job"
            job_output.mkdir(parents=True)
            outside = base / "remote_shared_output"
            outside.mkdir()
            explicit_output = job_output / "synced_output"
            explicit_output.symlink_to(outside, target_is_directory=True)
            log_file = base / "remote.log"
            job = DesignJob(
                job_id="failed-job", target_structure="missing.cif", chain_id="A",
                hotspots=[], binder_length=50, seed=0,
                params={"arm_root": str(arm_root), "arm_id": "arm-a"},
                output_dir=str(job_output),
            )
            cfg = HarnessConfig(target=TargetSpec(structure_path="missing.cif"))
            cfg.resource.backend = "local"
            orchestrator = BinderDesignOrchestrator(cfg, out_dir=base / "run", max_retries=0)
            record = {
                "job_id": job.job_id, "status": "failed",
                "local_output_dir": str(explicit_output), "log_file": str(log_file),
                "error": "remote execution failed",
                "monitor": {"failure_hints": ["missing_ceph_mount_secret"]},
            }

            ingested = orchestrator._ingest_execution_outputs([job], [record])[0]

            self.assertEqual(ingested["output_dir"], str(explicit_output))
            self.assertEqual(ingested["log_file"], str(log_file))
            self.assertIn("execution_failed_output_not_ingested", ingested["run_level_issues"])
            self.assertIn("remote execution failed", ingested["run_level_issues"])
            self.assertIn("missing_ceph_mount_secret", ingested["run_level_issues"])

    def test_orchestrator_success_rejects_explicit_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            arm_root = base / "arm"
            job_output = arm_root / "job"
            job_output.mkdir(parents=True)
            outside = base / "remote_shared_output"
            outside.mkdir()
            explicit_output = job_output / "synced_output"
            explicit_output.symlink_to(outside, target_is_directory=True)
            job = DesignJob(
                job_id="successful-job", target_structure="missing.cif", chain_id="A",
                hotspots=[], binder_length=50, seed=0,
                params={"arm_root": str(arm_root), "arm_id": "arm-a"},
                output_dir=str(job_output),
            )
            cfg = HarnessConfig(target=TargetSpec(structure_path="missing.cif"))
            cfg.resource.backend = "local"
            orchestrator = BinderDesignOrchestrator(cfg, out_dir=base / "run", max_retries=0)

            with self.assertRaisesRegex(ValueError, "outside declared arm_root"):
                orchestrator._ingest_execution_outputs([job], [{
                    "job_id": job.job_id, "status": "completed",
                    "local_output_dir": str(explicit_output),
                }])


    def _trusted_transport(self, base: Path, *, job_id: str = "job", attempt: int = 1, task_flag: str = "task", package_name: str = "project_package"):
        arm_root = base / "arm"
        local_package = arm_root / "jobs" / job_id / f"attempt_{attempt}" / package_name
        remote_package = base / "remote" / task_flag / package_name
        remote_output = remote_package / "outputs" / "boltzgen_output"
        remote_logs = remote_package / "logs"
        remote_output.mkdir(parents=True)
        remote_logs.mkdir()
        local_package.mkdir(parents=True)
        link_text = str(Path(__import__("os").path.relpath(remote_package / "outputs", local_package)))
        logs_link_text = str(Path(__import__("os").path.relpath(remote_logs, local_package)))
        (local_package / "outputs").symlink_to(link_text, target_is_directory=True)
        (local_package / "logs").symlink_to(logs_link_text, target_is_directory=True)
        local_alias = local_package / "outputs" / "boltzgen_output"
        binding = {
            "schema_version": 1, "mode": "symlink",
            "local_package_dir": str(local_package), "local_output_alias": str(local_alias),
            "remote_package_dir": str(remote_package), "remote_output_root": str(remote_output),
            "local_logs_alias": str(local_package / "logs"), "remote_logs_root": str(remote_logs),
            "link_text": link_text, "logs_link_text": logs_link_text,
            "job_id": job_id, "attempt": attempt, "task_flag": task_flag,
            "attempt_root": str(local_package.parent),
        }
        context = {
            "job_id": job_id, "attempt": attempt, "attempt_root": str(local_package.parent), "task_flag": task_flag,
            "arm_id": "arm-a", "arm_root": str(arm_root), "output_root": str(local_alias),
            "transport_binding": binding,
        }
        return local_alias, remote_output, binding, context

    def test_trusted_relative_transport_symlink_ingests_with_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_alias, remote_output, binding, context = self._trusted_transport(Path(tmp))
            metrics = remote_output / "final_ranked_designs/final_designs_metrics_1.csv"
            self._metrics(metrics, [{"id": "trusted", "pass_filters": "True"}])
            (remote_output / "result_manifest.json").write_text(json.dumps({"files": [str(metrics.relative_to(remote_output))]}))

            run = ResultIngestionAgent().ingest_boltzgen_output(local_alias, identity_context=context)

            expected_alias = local_alias / metrics.relative_to(remote_output)
            self.assertEqual(run.metrics_files, [str(expected_alias)])
            self.assertEqual(run.candidates[0]["_metrics_file"], str(expected_alias))
            self.assertFalse(Path(binding["link_text"]).is_absolute())

    def test_legacy_package_name_transport_still_ingests(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_alias, remote_output, binding, context = self._trusted_transport(
                Path(tmp), package_name="taiji_project_package",
            )
            metrics = remote_output / "final_ranked_designs/final_designs_metrics_1.csv"
            self._metrics(metrics, [{"id": "legacy", "pass_filters": "True"}])
            (remote_output / "result_manifest.json").write_text(
                json.dumps({"files": [str(metrics.relative_to(remote_output))]})
            )
            run = ResultIngestionAgent().ingest_boltzgen_output(local_alias, identity_context=context)
            expected_alias = local_alias / metrics.relative_to(remote_output)
            self.assertEqual(run.metrics_files, [str(expected_alias)])
            self.assertTrue(str(binding["remote_package_dir"]).endswith("taiji_project_package"))

    def test_same_transport_symlink_without_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_alias, _, _, context = self._trusted_transport(Path(tmp))
            context.pop("transport_binding")
            with self.assertRaisesRegex(ValueError, "outside declared arm_root"):
                ResultIngestionAgent().ingest_boltzgen_output(local_alias, identity_context=context)

    def test_binding_to_sibling_remote_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_alias, remote_output, binding, context = self._trusted_transport(Path(tmp))
            sibling = remote_output.parents[2] / "sibling_task" / "project_package"
            (sibling / "outputs" / "boltzgen_output").mkdir(parents=True)
            binding["remote_package_dir"] = str(sibling)
            binding["remote_output_root"] = str(sibling / "outputs" / "boltzgen_output")
            with self.assertRaisesRegex(ValueError, "transport binding mismatch"):
                ResultIngestionAgent().ingest_boltzgen_output(local_alias, identity_context=context)

    def test_trusted_transport_performs_no_per_file_realpath_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local_alias, remote_output, _, context = self._trusted_transport(base)
            metrics = remote_output / "host_00/gpu_00/shard_000/final_ranked_designs/final_designs_metrics_1.csv"
            self._metrics(metrics, [{"id": "trusted", "pass_filters": "True"}])
            structure = remote_output / "host_00/gpu_00/shard_000/final_ranked_designs/final_1_designs/trusted.cif"
            structure.parent.mkdir(parents=True); structure.write_text("data")
            (remote_output / "result_manifest.json").write_text(json.dumps({
                "files": [str(metrics.relative_to(remote_output)), str(structure.relative_to(remote_output))],
            }))

            original_contained = ResultIngestionAgent._contained
            with patch.object(ResultIngestionAgent, "_contained", wraps=original_contained) as contained:
                run = ResultIngestionAgent().ingest_boltzgen_output(local_alias, identity_context=context)

            self.assertEqual(run.metrics_files, [str(local_alias / metrics.relative_to(remote_output))])
            self.assertEqual(run.structure_files, [str(local_alias / structure.relative_to(remote_output))])
            self.assertEqual(contained.call_count, 2)
            self.assertEqual(
                [(call.args[0], call.args[1]) for call in contained.call_args_list],
                [(Path(context["transport_binding"]["remote_output_root"]), Path(context["transport_binding"]["remote_package_dir"])),
                 (Path(context["transport_binding"]["remote_logs_root"]), Path(context["transport_binding"]["remote_package_dir"]))],
            )

    def test_trusted_transport_rejects_absolute_and_parent_manifest_paths(self):
        for unsafe in ("/tmp/escape.csv", "../escape.csv", "nested/../../escape.csv"):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as tmp:
                local_alias, remote_output, _, context = self._trusted_transport(Path(tmp))
                (remote_output / "result_manifest.json").write_text(json.dumps({"files": [unsafe]}))
                with self.assertRaisesRegex(ValueError, "result manifest path outside declared output root"):
                    ResultIngestionAgent().ingest_boltzgen_output(local_alias, identity_context=context)

    def test_stale_transport_symlink_is_rejected_by_binding_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local_alias, _, _, context = self._trusted_transport(base)
            outputs_link = local_alias.parents[1] / "outputs"
            stale_outputs = base / "remote" / "stale_task" / "project_package" / "outputs"
            (stale_outputs / "boltzgen_output").mkdir(parents=True)
            outputs_link.unlink()
            outputs_link.symlink_to(stale_outputs, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "transport binding mismatch"):
                ResultIngestionAgent().ingest_boltzgen_output(local_alias, identity_context=context)


    def test_trusted_transport_rejects_log_outside_current_remote_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local_alias, remote_output, _, context = self._trusted_transport(base)
            (remote_output / "result_manifest.json").write_text(json.dumps({"files": []}))
            outside_log = base / "outside.log"
            outside_log.write_text("secret")
            with self.assertRaisesRegex(ValueError, "log lexical path outside local logs alias"):
                ResultIngestionAgent().ingest_boltzgen_output(local_alias, log_file=outside_log, identity_context=context)

    def test_legacy_execution_record_migrates_only_exact_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local_alias, remote_output, _, context = self._trusted_transport(base, job_id="legacy", task_flag="legacy_task")
            local_package = local_alias.parents[1]
            remote_package = remote_output.parents[1]
            metrics = remote_output / "final_ranked_designs/final_designs_metrics_1.csv"
            self._metrics(metrics, [{"id": "legacy", "pass_filters": "True"}])
            (remote_output / "result_manifest.json").write_text(json.dumps({"files": [str(metrics.relative_to(remote_output))]}))
            remote_log = remote_package / "logs" / "boltzgen_full.log"
            remote_log.write_text("legacy log")
            job = DesignJob(
                job_id="legacy", target_structure="missing.cif", chain_id="A", hotspots=[], binder_length=50, seed=0,
                params={"arm_root": context["arm_root"], "arm_id": "arm-a"}, output_dir=str(local_package.parent),
            )
            linked = []
            for name in ("outputs", "logs"):
                source, target = remote_package / name, local_package / name
                linked.append({"source": str(source), "target": str(target), "link": str(target.readlink())})
            record = {
                "job_id": "legacy", "status": "completed", "attempt": 1, "task_flag": "legacy_task",
                "attempt_root": str(local_package.parent), "local_package_dir": str(local_package),
                "remote_package_dir": str(remote_package), "local_output_dir": str(local_alias),
                "remote_output_dir": str(remote_output), "local_log_file": str(local_package / "logs" / remote_log.name),
                "result_sync": {"mode": "symlink", "local_package_dir": str(local_package),
                                "remote_package_dir": str(remote_package), "linked": linked},
            }
            cfg = HarnessConfig(target=TargetSpec(structure_path="missing.cif"))
            orchestrator = BinderDesignOrchestrator(cfg, out_dir=base / "run", max_retries=0)
            ingested = orchestrator._ingest_execution_outputs([job], [record])[0]
            self.assertEqual(ingested["metrics_files"], [str(local_alias / metrics.relative_to(remote_output))])
            self.assertEqual(ingested["log_tail"], "legacy log")

            forged = json.loads(json.dumps(record))
            forged["result_sync"]["linked"][0]["link"] = "../forged_outputs"
            with self.assertRaisesRegex(ValueError, "outside declared arm_root"):
                orchestrator._ingest_execution_outputs([job], [forged])

    def test_transport_containment_errors_are_non_retryable(self):
        from binderloop.agents.result_ingestion_agent import TransportBindingError
        self.assertFalse(BinderDesignOrchestrator._ingestion_error_retryable(TransportBindingError("transport binding mismatch")))
        self.assertFalse(BinderDesignOrchestrator._ingestion_error_retryable(ValueError("output root is outside declared arm_root")))

    def test_evaluation_preserves_stable_native_provenance(self):
        summary = EvaluationAgent().evaluate_candidates([{
            "id": "native", "global_candidate_id": "global", "file_name": "rank0_native.cif",
            "design_to_target_iptm": .7, "min_design_to_target_pae": 5,
            "design_ptm": .8, "designfolding_filter_rmsd": 1,
        }])
        candidate = summary.top_candidates[0]
        self.assertEqual(candidate.candidate_id, "global")
        self.assertEqual(candidate.raw["id"], "native")
        self.assertEqual(candidate.raw["global_candidate_id"], "global")


if __name__ == "__main__":
    raise SystemExit(unittest.main())
