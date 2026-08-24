#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.config import HarnessConfig, TargetSpec, load_config
from binderloop.models.base import DesignJob
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.resume import (
    CORRECTION_PATCH_SCHEMA_VERSION,
    PAYLOAD_CONTRACT_VERSION,
    VALIDATION_CONTRACT_VERSION,
    ResumeMismatchError,
    atomic_write_json,
    build_run_manifest,
    extract_target_identity,
    validate_or_write_run_manifest,
)


def _cfg() -> HarnessConfig:
    cfg = HarnessConfig(target=TargetSpec(structure_path="target.cif"))
    cfg.resource.backend = "local"
    cfg.resource.max_parallel_jobs = 1
    return cfg


def _job(out_dir: Path) -> DesignJob:
    return DesignJob(
        job_id="resume_job",
        target_structure="target.cif",
        chain_id="A",
        hotspots=[],
        binder_length=60,
        params={},
        output_dir=str(out_dir / "job"),
    )


def _write_minimal_task(root: Path, *, structure_name: str = "target.cif", task_name: str = "task_a", max_binders: int = 8, length_range=None) -> Path:
    structure = root / structure_name
    if not structure.exists():
        structure.write_text(f"structure:{structure_name}\n", encoding="utf-8")
    length_range = length_range or [60, 80]
    config_path = root / "task.yaml"
    config_path.write_text(
        "\n".join([
            "task:",
            f"  task_name: {task_name}",
            f"  target_structure_path: {structure_name}",
            "  target_chain_id: A",
            "  hotspots: ['A:1']",
            f"  binder_length_range: {length_range}",
            "  binder_length_step: 10",
            f"  max_binders_per_round: {max_binders}",
            "active_learning:",
            "  max_rounds: 2",
            "resource:",
            "  backend: dry_run",
            "  max_parallel_jobs: 1",
            "search_space:",
            "  model_order: [boltzgen]",
            "  boltzgen: {}",
            "",
        ]),
        encoding="utf-8",
    )
    return config_path


def _upgrade_test_config_to_owner(path: Path) -> Path:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "owner" in data:
        return path
    task = dict(data.get("task") or {})
    search = dict(data.get("search_space") or {})
    boltzgen = dict(search.get("boltzgen") or {})
    num_designs = int(task.pop("max_binders_per_round", 4))
    filtering_budget = {
        "budget": int(boltzgen.pop("budget", 10)),
        "run_filtering": True,
        "keep_unfiltered_for_failure_analysis": bool(boltzgen.pop("keep_unfiltered_for_failure_analysis", True)),
        "additional_filters": list(boltzgen.pop("additional_filters", []) or []),
    }
    owner = {
        "task_hard_constraints": {**task, "num_designs": num_designs},
        "boltzgen_design_native": {k: boltzgen.pop(k) for k in list(boltzgen) if k in {"protocol", "diffusion_batch_size", "steps"}},
        "boltzgen_inverse_fold_and_validation": {k: boltzgen.pop(k) for k in list(boltzgen) if k in {"inverse_fold_num_sequences", "inverse_fold_avoid"}},
        "boltzgen_filtering_ranking": boltzgen,
        "filtering_budget": filtering_budget,
        "harness_search_space": {"model_order": list(search.get("model_order") or ["boltzgen"])},
        "active_learning_and_rollback": dict(data.get("active_learning") or {}),
        "runtime_resources": {"runtime": dict(data.get("runtime") or {}), "resource": dict(data.get("resource") or {})},
        "llm_context_learning": {},
    }
    path.write_text(yaml.safe_dump({"schema_version": 1, "owner": owner}, sort_keys=False), encoding="utf-8")
    return path


class ResumeSupportTests(unittest.TestCase):
    def test_import_origin_is_top_level_repository_package(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        code = """
import json
from pathlib import Path
import binderloop
import binderloop.resume
print(json.dumps({
    "package_paths": [str(Path(value).resolve()) for value in binderloop.__path__],
    "resume_file": str(Path(binderloop.resume.__file__).resolve()),
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        origin = json.loads(completed.stdout)
        expected_package = (repo_root / "binderloop").resolve()
        self.assertEqual(origin["package_paths"], [str(expected_package)])
        self.assertEqual(Path(origin["resume_file"]).parent, expected_package)

    def test_new_manifest_records_execution_contract_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_minimal_task(root)
            cfg = load_config(_upgrade_test_config_to_owner(config_path))
            manifest = build_run_manifest(config_path=config_path, config=cfg, cli_identity={})
            self.assertEqual(manifest["payload_contract_version"], PAYLOAD_CONTRACT_VERSION)
            self.assertEqual(manifest["validation_contract_version"], VALIDATION_CONTRACT_VERSION)
            self.assertEqual(manifest["correction_patch_schema_version"], CORRECTION_PATCH_SCHEMA_VERSION)
            out_dir = root / "fresh-output"
            validate_or_write_run_manifest(out_dir, manifest)
            self.assertEqual(
                json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8")),
                manifest,
            )

    def test_legacy_manifest_upgrade_preserves_completed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = _write_minimal_task(root)
            cfg = load_config(_upgrade_test_config_to_owner(config_path))
            out_dir = root / "out"
            legacy = build_run_manifest(config_path=config_path, config=cfg, cli_identity={"max_rounds": 1})
            for key in (
                "payload_contract_version",
                "validation_contract_version",
                "correction_patch_schema_version",
            ):
                legacy.pop(key)
            legacy["schema_version"] = 3
            atomic_write_json(out_dir / "run_manifest.json", legacy)

            checkpoint_path = out_dir / "round_00" / "round_checkpoint.json"
            ledger_path = out_dir / "round_00" / "execution_attempts.json"
            atomic_write_json(checkpoint_path, {"round_id": 0, "status": "completed", "summary_round": {"round_id": 0}})
            atomic_write_json(ledger_path, {
                "round_id": 0,
                "jobs": {"old-job": {"terminal_record": {"job_id": "old-job", "status": "completed"}}},
            })
            checkpoint_before = checkpoint_path.read_bytes()
            ledger_before = ledger_path.read_bytes()

            current = build_run_manifest(config_path=config_path, config=cfg, cli_identity={"max_rounds": 3})
            validate_or_write_run_manifest(out_dir, current)

            stored = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["payload_contract_version"], PAYLOAD_CONTRACT_VERSION)
            self.assertEqual(stored["validation_contract_version"], VALIDATION_CONTRACT_VERSION)
            self.assertEqual(stored["correction_patch_schema_version"], CORRECTION_PATCH_SCHEMA_VERSION)
            self.assertEqual(stored["identity"]["cli_identity"]["max_rounds"], 3)
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_before)
            self.assertEqual(ledger_path.read_bytes(), ledger_before)

    def test_run_manifest_rejects_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = root / "target.cif"
            structure.write_text("cif", encoding="utf-8")
            config_path = root / "task.yaml"
            config_path.write_text(
                "\n".join([
                    "task:",
                    "  task_name: task_a",
                    "  target_structure_path: target.cif",
                    "  target_chain_id: A",
                    "  hotspots: ['A:1']",
                    "  binder_length_range: [60, 80]",
                    "  max_binders_per_round: 8",
                    "resource:",
                    "  backend: dry_run",
                    "search_space:",
                    "  boltzgen: {}",
                    "",
                ]),
                encoding="utf-8",
            )
            cfg = load_config(_upgrade_test_config_to_owner(config_path))
            out_dir = root / "out"
            manifest = build_run_manifest(config_path=config_path, config=cfg, cli_identity={"backend": "local"})
            validate_or_write_run_manifest(out_dir, manifest)

            cfg.task_name = "different_task"
            changed = build_run_manifest(config_path=config_path, config=cfg, cli_identity={"backend": "local"})
            with self.assertRaises(ResumeMismatchError):
                validate_or_write_run_manifest(out_dir, changed)

    def test_run_manifest_allows_higher_max_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = root / "target.cif"
            structure.write_text("cif", encoding="utf-8")
            config_path = root / "task.yaml"
            config_path.write_text(
                "\n".join([
                    "task:",
                    "  task_name: task_a",
                    "  target_structure_path: target.cif",
                    "  target_chain_id: A",
                    "  hotspots: ['A:1']",
                    "  binder_length_range: [60, 80]",
                    "  max_binders_per_round: 8",
                    "active_learning:",
                    "  max_rounds: 2",
                    "resource:",
                    "  backend: dry_run",
                    "search_space:",
                    "  boltzgen: {}",
                    "",
                ]),
                encoding="utf-8",
            )
            cfg = load_config(_upgrade_test_config_to_owner(config_path))
            out_dir = root / "out"
            first = build_run_manifest(config_path=config_path, config=cfg, cli_identity={"max_rounds": 2})
            validate_or_write_run_manifest(out_dir, first)
            cfg.active_learning.max_rounds = 5
            second = build_run_manifest(config_path=config_path, config=cfg, cli_identity={"max_rounds": 5})
            validate_or_write_run_manifest(out_dir, second)
            stored = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["identity"]["cli_identity"]["max_rounds"], 5)

    def test_run_manifest_rejects_binder_budget_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = root / "target.cif"
            structure.write_text("cif", encoding="utf-8")
            config_path = root / "task.yaml"
            config_path.write_text(
                "\n".join([
                    "task:",
                    "  task_name: t",
                    "  target_structure_path: target.cif",
                    "  binder_length_range: [60, 80]",
                    "  max_binders_per_round: 8",
                    "resource:",
                    "  backend: dry_run",
                    "search_space:",
                    "  boltzgen: {}",
                    "",
                ]),
                encoding="utf-8",
            )
            cfg = load_config(_upgrade_test_config_to_owner(config_path))
            out_dir = root / "out"
            first = build_run_manifest(config_path=config_path, config=cfg, cli_identity={})
            validate_or_write_run_manifest(out_dir, first)
            cfg.owner.task_hard_constraints.num_designs = 16
            cfg.task.max_binders_per_round = 16
            cfg.search_space.max_binders_per_round = 16
            changed = build_run_manifest(config_path=config_path, config=cfg, cli_identity={})
            with self.assertRaises(ResumeMismatchError):
                validate_or_write_run_manifest(out_dir, changed)

    def test_target_identity_includes_structure_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = root / "target.cif"
            structure.write_text("version-a", encoding="utf-8")
            config_path = root / "task.yaml"
            config_path.write_text(
                "\n".join([
                    "task:",
                    "  target_structure_path: target.cif",
                    "  target_chain_id: A",
                    "  binder_length_range: [60, 80]",
                    "  max_binders_per_round: 8",
                    "resource:",
                    "  backend: dry_run",
                    "search_space:",
                    "  boltzgen: {}",
                    "",
                ]),
                encoding="utf-8",
            )
            cfg = load_config(_upgrade_test_config_to_owner(config_path))
            identity = extract_target_identity(cfg, config_path=config_path)
            self.assertTrue(identity["structure_sha256"])
            structure.write_text("version-b", encoding="utf-8")
            changed = extract_target_identity(cfg, config_path=config_path)
            self.assertNotEqual(identity["structure_sha256"], changed["structure_sha256"])

    def test_completed_module_is_loaded_without_rerunning_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = out_dir / "round_00"
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
            checkpoint = {"round_id": 0, "artifacts": [], "modules": {}}
            calls = {"count": 0}

            def action():
                calls["count"] += 1
                path = orchestrator._write_json(round_dir / "module.json", {"ok": True})
                return {"payload": {"ok": True}, "path": str(path)}

            def validator(result):
                payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
                if payload != {"ok": True}:
                    raise AssertionError(payload)

            first = orchestrator._run_validated_module(
                module_name="unit_module",
                round_id=0,
                round_dir=round_dir,
                checkpoint=checkpoint,
                action=action,
                validator=validator,
                loader=lambda record: {"payload": {"ok": True}, "path": str(round_dir / "module.json")},
            )
            self.assertEqual(first["payload"], {"ok": True})
            self.assertEqual(calls["count"], 1)

            resumed_checkpoint = json.loads((round_dir / "round_checkpoint.json").read_text(encoding="utf-8"))
            resumed = orchestrator._run_validated_module(
                module_name="unit_module",
                round_id=0,
                round_dir=round_dir,
                checkpoint=resumed_checkpoint,
                action=lambda: (_ for _ in ()).throw(AssertionError("action should not rerun")),
                validator=validator,
                loader=lambda record: {"payload": {"ok": True}, "path": str(round_dir / "module.json")},
            )
            self.assertEqual(resumed["payload"], {"ok": True})
            self.assertEqual(calls["count"], 1)

    def test_checkpoint_records_module_digests_phase_and_publish_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = out_dir / "round_00"
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
            checkpoint = {"round_id": 0, "artifacts": [], "modules": {}}
            orchestrator._run_validated_module(
                module_name="unit_module", round_id=0, round_dir=round_dir, checkpoint=checkpoint,
                action=lambda: {"path": str(orchestrator._write_json(round_dir / "module.json", {"ok": True}))},
                validator=lambda result: json.loads(Path(result["path"]).read_text(encoding="utf-8")),
            )
            stored = json.loads((round_dir / "round_checkpoint.json").read_text(encoding="utf-8"))
            module = stored["modules"]["unit_module"]
            self.assertEqual(module["phase"], "output_validated")
            self.assertEqual(module["publish_status"], "not_published")
            self.assertEqual(len(module["input_digest"]), 64)
            self.assertEqual(len(module["output_digest"]), 64)

    def test_corrupt_completed_module_artifact_is_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = out_dir / "round_00"
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
            checkpoint = {"round_id": 0, "artifacts": [], "modules": {}}

            def validator(result):
                json.loads(Path(result["path"]).read_text(encoding="utf-8"))

            orchestrator._run_validated_module(
                module_name="unit_module",
                round_id=0,
                round_dir=round_dir,
                checkpoint=checkpoint,
                action=lambda: {"path": str(orchestrator._write_json(round_dir / "module.json", {"version": 1}))},
                validator=validator,
                loader=lambda record: {"path": str(round_dir / "module.json")},
            )

            (round_dir / "module.json").write_text("{not-json", encoding="utf-8")
            resumed_checkpoint = json.loads((round_dir / "round_checkpoint.json").read_text(encoding="utf-8"))
            calls = {"count": 0}

            def repair_action():
                calls["count"] += 1
                return {"path": str(orchestrator._write_json(round_dir / "module.json", {"version": 2}))}

            repaired = orchestrator._run_validated_module(
                module_name="unit_module",
                round_id=0,
                round_dir=round_dir,
                checkpoint=resumed_checkpoint,
                action=repair_action,
                validator=validator,
                loader=lambda record: {"path": str(round_dir / "module.json")},
            )
            self.assertEqual(json.loads(Path(repaired["path"]).read_text(encoding="utf-8")), {"version": 2})
            self.assertEqual(calls["count"], 1)

    def test_incomplete_attempt_reconciles_from_execution_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            job = _job(out_dir)
            attempts_path = out_dir / "round_00" / "execution_attempts.json"
            atomic_write_json(
                attempts_path,
                {
                    "round_id": 0,
                    "max_attempts_per_job": 3,
                    "jobs": {
                        job.job_id: {
                            "job": {
                                "job_id": job.job_id,
                                "target_structure": job.target_structure,
                                "chain_id": job.chain_id,
                                "hotspots": job.hotspots,
                                "binder_length": job.binder_length,
                                "seed": job.seed,
                                "params": job.params,
                                "output_dir": job.output_dir,
                            },
                            "attempts": [{"attempt": 1, "status": "started"}],
                        }
                    },
                },
            )
            atomic_write_json(
                Path(job.output_dir) / "execution_record.json",
                {"job_id": job.job_id, "attempt": 1, "status": "completed", "output_dir": job.output_dir},
            )
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=3)

            records = orchestrator._run_jobs(
                [job],
                0,
                lambda _job, _attempt: (_ for _ in ()).throw(AssertionError("executor should not rerun")),
                attempts_path=attempts_path,
            )
            self.assertEqual(records[0]["status"], "completed")
            self.assertEqual(records[0]["attempts"], 1)


    def test_legacy_duplicate_ids_use_ordered_readonly_association(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp); orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_retries=1)
            jobs = [_job(out_dir), _job(out_dir)]
            for index, job in enumerate(jobs):
                job.job_id = "legacy"; job.params["branch_id"] = "legacy"; job.output_dir = str(out_dir / f"legacy_{index}")
            records = [
                {"job_id": "legacy", "status": "completed", "output_dir": jobs[0].output_dir},
                {"job_id": "legacy", "status": "failed", "output_dir": jobs[1].output_dir},
            ]
            ordered = orchestrator._records_for_jobs(jobs, records, label="legacy", allow_legacy_order=True)
            self.assertEqual([row["status"] for row in ordered], ["completed", "failed"])
            with self.assertRaisesRegex(ValueError, "order_mismatch"):
                orchestrator._records_for_jobs(jobs, [{**records[0], "job_id": "other"}, records[1]], label="legacy", allow_legacy_order=True)

    def test_legacy_empty_next_jobs_are_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = out_dir / "round_00"
            round_dir.mkdir(parents=True)
            cfg = _cfg()
            cfg.search_space.binder_lengths = [60, 80]
            cfg.search_space.binder_length_range = [60, 80]
            cfg.active_learning.max_rounds = 2
            orchestrator = BinderDesignOrchestrator(cfg, out_dir=out_dir, max_rounds=2, max_parallel=1, max_retries=1)
            parent = _job(out_dir)
            parent.params = {"binder_lengths": [60, 80], "num_designs": 1}
            checkpoint = {
                "round_id": 0,
                "status": "completed",
                "current_jobs": [
                    {
                        "job_id": parent.job_id,
                        "target_structure": parent.target_structure,
                        "chain_id": parent.chain_id,
                        "hotspots": parent.hotspots,
                        "binder_length": parent.binder_length,
                        "params": parent.params,
                        "output_dir": parent.output_dir,
                    }
                ],
                "next_jobs": [],
                "applied_params_update": {},
                "summary_round": {
                    "round_id": 0,
                    "evaluation": {"top_candidates": []},
                    "proposal": {"params_update": {}},
                    "structural_analysis": {},
                    "hypotheses": {"hypotheses": []},
                    "quality_analysis": {},
                },
            }
            atomic_write_json(round_dir / "round_checkpoint.json", checkpoint)
            start_round, current_jobs, recovered = orchestrator._recover_completed_rounds([parent])
            self.assertEqual(start_round, 1)
            self.assertEqual(len(recovered), 1)
            self.assertGreater(len(current_jobs), 0)
            rebuilt = json.loads((round_dir / "round_checkpoint.json").read_text(encoding="utf-8"))
            self.assertTrue(rebuilt.get("continuation_rebuilt_from_legacy"))
            self.assertGreater(len(rebuilt.get("next_jobs") or []), 0)

    def test_completed_checkpoint_uses_compact_round_summary_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = out_dir / "round_00"
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_rounds=1, max_parallel=1, max_retries=1)
            checkpoint = {"round_id": 0, "current_jobs": [], "next_jobs": [], "artifacts": [], "modules": {}}
            summary_round = {
                "round_id": 0,
                "structural_analysis": {"full_payload": "x" * 20000},
                "final_strategy_decision": {"selected_arm_id": "arm-compact"},
            }

            summary_ref = orchestrator._store_completed_round_summary(
                checkpoint=checkpoint, round_dir=round_dir, round_id=0, summary_round=summary_round,
            )
            orchestrator._write_checkpoint(round_dir, 0, "round_completed", "completed", checkpoint)

            stored = json.loads((round_dir / "round_checkpoint.json").read_text(encoding="utf-8"))
            self.assertNotIn("summary_round", stored)
            self.assertNotIn("structural_analysis", json.dumps(stored))
            self.assertEqual(stored["round_summary_ref"], summary_ref)
            self.assertEqual(stored["preferred_arm_id"], "arm-compact")
            self.assertEqual(len(summary_ref["sha256"]), 64)
            self.assertEqual(summary_ref["size_bytes"], (round_dir / "round_summary.json").stat().st_size)
            compact = json.loads((round_dir / "round_summary.json").read_text(encoding="utf-8"))
            self.assertNotIn("structural_analysis", compact)
            details_ref = compact["round_details_ref"]
            self.assertEqual(json.loads(Path(details_ref["path"]).read_text(encoding="utf-8")), summary_round)
            self.assertLess((round_dir / "round_checkpoint.json").stat().st_size, details_ref["size_bytes"] // 10)

    def test_compact_completed_round_recovery_loads_summary_and_preferred_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = out_dir / "round_00"
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_rounds=1, max_parallel=1, max_retries=1)
            summary_round = {
                "round_id": 0,
                "evaluation": {"top_candidates": []},
                "structural_analysis": {"large": [1, 2, 3]},
                "final_strategy_decision": {"selected_arm_id": "arm-recovered"},
            }
            checkpoint = {"round_id": 0, "current_jobs": [], "next_jobs": [], "rollback_action": "stop"}
            orchestrator._store_completed_round_summary(
                checkpoint=checkpoint, round_dir=round_dir, round_id=0, summary_round=summary_round,
            )
            orchestrator._write_checkpoint(round_dir, 0, "round_completed", "completed", checkpoint)

            start_round, current_jobs, recovered = orchestrator._recover_completed_rounds([])
            self.assertEqual(start_round, 1)
            self.assertEqual(current_jobs, [])
            self.assertEqual(recovered[0]["round_id"], 0)
            self.assertEqual(recovered[0]["final_strategy_decision"], {"selected_arm_id": "arm-recovered"})
            self.assertNotIn("structural_analysis", recovered[0])
            self.assertEqual(orchestrator._preferred_arm_id, "arm-recovered")

    def test_legacy_embedded_completed_round_summary_still_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = out_dir / "round_00"
            summary_round = {
                "round_id": 0,
                "structural_analysis": {"legacy": True},
                "final_strategy_decision": {"selected_arm_id": "arm-legacy"},
            }
            atomic_write_json(round_dir / "round_checkpoint.json", {
                "round_id": 0, "status": "completed", "next_jobs": [], "rollback_action": "stop",
                "summary_round": summary_round,
            })
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_rounds=1, max_parallel=1, max_retries=1)

            start_round, _, recovered = orchestrator._recover_completed_rounds([])
            self.assertEqual(start_round, 1)
            self.assertEqual(recovered, [summary_round])
            self.assertEqual(orchestrator._preferred_arm_id, "arm-legacy")

    def test_compact_round_summary_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = out_dir / "round_00"
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_rounds=1, max_parallel=1, max_retries=1)
            checkpoint = {"round_id": 0, "next_jobs": [], "rollback_action": "stop"}
            orchestrator._store_completed_round_summary(
                checkpoint=checkpoint,
                round_dir=round_dir,
                round_id=0,
                summary_round={"round_id": 0, "structural_analysis": {"before": True}},
            )
            orchestrator._write_checkpoint(round_dir, 0, "round_completed", "completed", checkpoint)
            atomic_write_json(round_dir / "round_summary.json", {
                "round_id": 0, "structural_analysis": {"tampered": True},
            })

            with self.assertRaisesRegex(RuntimeError, "round summary digest mismatch"):
                orchestrator._recover_completed_rounds([])

    def test_resume_reads_pre_submit_summary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            round_dir = out_dir / "round_00"
            round_dir.mkdir(parents=True)
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_rounds=1, max_parallel=1, max_retries=1)
            parent = _job(out_dir)
            pre_submit = {"schema_version": 1, "round_id": 0, "job_count": 1, "jobs": [{"job_id": parent.job_id}]}
            atomic_write_json(round_dir / "pre_submit_summary.json", pre_submit)
            atomic_write_json(round_dir / "round_checkpoint.json", {
                "round_id": 0, "status": "completed",
                "current_jobs": [{"job_id": parent.job_id, "target_structure": parent.target_structure,
                    "chain_id": parent.chain_id, "hotspots": parent.hotspots, "binder_length": parent.binder_length,
                    "params": parent.params, "output_dir": parent.output_dir}],
                "next_jobs": [], "rollback_action": "stop",
                "pre_submit_summary_path": str(round_dir / "pre_submit_summary.json"),
                "summary_round": {"round_id": 0, "evaluation": {}, "proposal": {}, "structural_analysis": {},
                                  "hypotheses": {}, "quality_analysis": {}},
            })
            start_round, _, recovered = orchestrator._recover_completed_rounds([parent])
            self.assertEqual(start_round, 1)
            self.assertEqual(recovered[0]["pre_submit_summary"], pre_submit)
            summary = orchestrator.run()
            self.assertEqual(summary["rounds"][0]["pre_submit_summary"], pre_submit)
            stored = json.loads((out_dir / "orchestrator_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["rounds"][0]["pre_submit_summary"], pre_submit)

    def test_resumed_complete_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cfg = _cfg()
            cfg.resource.backend = "dry_run"
            cfg.search_space.binder_lengths = [60]
            cfg.active_learning.max_rounds = 1
            orchestrator = BinderDesignOrchestrator(cfg, out_dir=out_dir, max_rounds=1, max_parallel=1, max_retries=1)
            parent = _job(out_dir)
            parent.params = {"binder_lengths": [60], "num_designs": 1}
            round_dir = out_dir / "round_00"
            round_dir.mkdir(parents=True)
            atomic_write_json(
                round_dir / "round_checkpoint.json",
                {
                    "round_id": 0,
                    "status": "completed",
                    "current_jobs": [
                        {
                            "job_id": parent.job_id,
                            "target_structure": parent.target_structure,
                            "chain_id": parent.chain_id,
                            "hotspots": parent.hotspots,
                            "binder_length": parent.binder_length,
                            "params": parent.params,
                            "output_dir": parent.output_dir,
                        }
                    ],
                    "next_jobs": [
                        {
                            "job_id": "r1_round",
                            "target_structure": parent.target_structure,
                            "chain_id": parent.chain_id,
                            "hotspots": parent.hotspots,
                            "binder_length": parent.binder_length,
                            "params": parent.params,
                            "output_dir": str(out_dir / "r1" / "round"),
                        }
                    ],
                    "applied_params_update": {},
                    "summary_round": {"round_id": 0, "evaluation": {"top_candidates": []}},
                },
            )
            summary = orchestrator.run(execute_job=None)
            self.assertTrue(summary.get("resumed_complete"))
            self.assertEqual(summary.get("completed_rounds"), 1)
            self.assertEqual(len(summary.get("rounds") or []), 1)

    def test_resume_old_failed_ledger_does_not_conflict_with_refinalized_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            attempts_path = out_dir / "round_03" / "execution_attempts.json"
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
            old = orchestrator._finalize_semantic_job_identities([_job(out_dir)], round_id=3)[0]
            old = orchestrator._enforce_round_cap([old], round_id=3)[0]
            atomic_write_json(attempts_path, {
                "round_id": 3, "max_attempts_per_job": 1, "jobs": {old.job_id: {
                    "job": old.__dict__,
                    "job_identity_digest": orchestrator._job_identity_digest(old),
                    "attempts": [{"attempt": 1, "status": "failed"}],
                    "terminal_record": {"job_id": old.job_id, "status": "failed", "attempts": 1, "retryable": False},
                }},
            })
            changed = DesignJob(**{**old.__dict__, "params": {**old.params, "alpha": 0.003}})
            fresh = orchestrator._finalize_semantic_job_identities([changed], round_id=3)[0]
            fresh = orchestrator._enforce_round_cap([fresh], round_id=3)[0]
            self.assertNotEqual(old.job_id, fresh.job_id)
            calls = []
            records = orchestrator._run_jobs(
                [fresh], 3,
                lambda job, attempt: calls.append((job.job_id, attempt)) or {"job_id": job.job_id, "status": "completed", "output_dir": job.output_dir},
                attempts_path=attempts_path,
            )
            self.assertEqual(calls, [(fresh.job_id, 1)])
            self.assertEqual(records[0]["status"], "completed")
            stored = json.loads(attempts_path.read_text(encoding="utf-8"))
            self.assertEqual(set(stored["jobs"]), {old.job_id, fresh.job_id})


    def test_incomplete_attempt_prefers_attempt_scoped_execution_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp); job = _job(out_dir)
            attempts_path = out_dir / "round_00" / "execution_attempts.json"
            atomic_write_json(attempts_path, {"round_id": 0, "max_attempts_per_job": 3, "jobs": {job.job_id: {"job": job.__dict__, "attempts": [{"attempt": 1, "status": "started"}]}}})
            scoped = Path(job.output_dir) / "attempts" / "attempt_01" / "execution_record.json"
            atomic_write_json(scoped, {"job_id": job.job_id, "attempt": 1, "status": "completed", "output_dir": str(scoped.parent)})
            atomic_write_json(Path(job.output_dir) / "execution_record.json", {"job_id": job.job_id, "attempt": 1, "status": "failed"})
            orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=3)
            records = orchestrator._run_jobs([job], 0, lambda *_: (_ for _ in ()).throw(AssertionError("must reconcile")), attempts_path=attempts_path)
            self.assertEqual(records[0]["status"], "completed")
            self.assertEqual(records[0]["artifact_locators"]["execution_record"], str(scoped))


if __name__ == "__main__":
    unittest.main()
