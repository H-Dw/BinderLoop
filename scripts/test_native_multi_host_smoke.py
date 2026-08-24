#!/usr/bin/env python3

"""Run and validate a lightweight native Taiji multi-host closed loop."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.design_spec_agent import DesignSpecAgent
from binderloop.agents.result_ingestion_agent import IngestedBoltzGenRun
from binderloop.analysis.post_ingest_parity import validate_post_ingest_parity
from binderloop.models.base import DesignJob


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/sc2rbd_native_multi_host_smoke_2r_8d_2h4g.yaml"
DEFAULT_OUT = ROOT / "outputs/sc2rbd_native_multi_host_smoke_2r_8d_2h4g"
REQUIRED_ROUND_FILES = {
    "round_checkpoint.json",
    "execution_records.json",
    "ingestions.json",
    "evaluation_summary.json",
    "structure_evaluation.json",
}
STANDARD_SHARD_DIRS = {
    "intermediate_designs",
    "intermediate_designs_inverse_folded",
    "final_ranked_designs",
}



def test_local_two_host_manifest_aggregation() -> None:
    """Exercise two concurrent host exits without contacting Taiji or BoltzGen."""
    with tempfile.TemporaryDirectory() as temporary:
        temp_root = Path(temporary)
        missing_target = temp_root / "missing_target.cif"
        job = DesignJob(
            job_id="local_native_manifest_2host",
            target_structure=str(missing_target),
            chain_id="A",
            hotspots=["A:1"],
            binder_length=20,
            seed=0,
            params={},
            output_dir=str(temp_root / "run"),
        )
        params = {
            "devices": 1,
            "host_count": 2,
            "taiji_submit_host_num": 2,
            "taiji_multi_host_mode": "native",
            "num_designs": 2,
            "max_binders_per_round": 2,
            "budget": 1,
            "analysis_location": "taiji",
            "cache": str(temp_root),
            "moldir": str(temp_root),
        }
        spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(job, params=params)
        base_env = {
            **os.environ,
            "WORLD_SIZE": "2",
            "HARNESS_RUN_TOKEN": "local_manifest_2host",
            "HARNESS_CLUSTER_BARRIER_TIMEOUT": "10",
            "HARNESS_MANIFEST_BARRIER_TIMEOUT": "10",
        }
        processes = [
            subprocess.Popen(
                ["bash", spec.run_script_path],
                cwd=ROOT,
                env={**base_env, "RANK": str(rank)},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for rank in (0, 1)
        ]
        outputs = []
        for process in processes:
            output, _ = process.communicate(timeout=20)
            outputs.append(output)
            assert process.returncode == 11, output

        output_root = Path(spec.output_dir)
        aggregate_path = output_root / "result_manifest.json"
        assert aggregate_path.is_file(), "rank 0 did not publish aggregate manifest"
        aggregate = _load_json(aggregate_path)
        assert aggregate["schema_version"] == 6
        assert aggregate["contract"] == {"name": "binder_harness_result_manifest", "version": 1}
        assert "required_artifacts" not in aggregate  # this synthetic run fails before steps publication
        assert aggregate["candidate_attribution"] is False
        assert aggregate["attribution_scope"] == "job"
        assert aggregate["shard_manifests"] == [
            "host_00/shard_result_manifest.json",
            "host_01/shard_result_manifest.json",
        ]
        assert set(aggregate["shard_manifests"]).issubset(set(aggregate["files"]))
        for rank, reference in enumerate(aggregate["shard_manifests"]):
            referenced = output_root / reference
            assert referenced.is_file(), reference
            host_manifest = _load_json(referenced)
            assert host_manifest["collection_mode"] == "host_shard"
            assert host_manifest["host_rank"] == rank
            assert host_manifest["candidate_attribution"] is False
            assert host_manifest["attribution_scope"] == "job"

        parity = validate_post_ingest_parity(output_root, aggregate, [])
        assert parity.evaluable, parity.failures
        assert parity.checks["attribution_scope"] == "job"

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run two rounds with 8 designs/round on one native Taiji task "
            "(2 hosts x 4 GPUs), then validate the output contract."
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--submit", action="store_true", help="Submit the real Taiji smoke run, wait, then validate.")
    action.add_argument("--validate-only", action="store_true", help="Validate an already completed smoke output.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--llm-config", help="Optional LLM endpoint config; deterministic fallback is used when omitted.")
    parser.add_argument("--secret-config", help="Ignored local JSON containing CEPH_SECRET without enabling LLM calls.")
    parser.add_argument("--single-host-reference", help="Optional completed 1-host output whose ingestion schema is compared.")
    parser.add_argument("--report", help="Validation report path. Defaults to <out>/native_multi_host_smoke_report.json.")
    parser.add_argument("--resume", action="store_true", help="Allow --submit to resume a non-empty output directory.")
    args = parser.parse_args()

    config = _resolve(args.config)
    out_dir = _resolve(args.out)
    if args.submit:
        if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
            parser.error(f"output directory is not empty: {out_dir}; use a new --out or pass --resume")
        _run_smoke(
            config=config,
            out_dir=out_dir,
            llm_config=args.llm_config,
            secret_config=args.secret_config,
        )

    reference = _resolve(args.single_host_reference) if args.single_host_reference else None
    report = validate_smoke_output(out_dir, single_host_reference=reference)
    report_path = _resolve(args.report) if args.report else out_dir / "native_multi_host_smoke_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Validation report: {report_path}")
    return 0


def _run_smoke(
    *,
    config: Path,
    out_dir: Path,
    llm_config: Optional[str],
    secret_config: Optional[str],
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_closed_loop_orchestrator.py"),
        "--config",
        str(config),
        "--out",
        str(out_dir),
        "--max-rounds",
        "2",
        "--submit",
        "--boltzgen-heartbeat-seconds",
        "360",
        "--taiji-poll-seconds",
        "60",
        "--taiji-wait-timeout",
        "7200",
    ]
    if llm_config:
        command.extend(["--llm-config", str(_resolve(llm_config))])
    if secret_config:
        command.extend(["--secret-config", str(_resolve(secret_config))])
    subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, check=True)


def validate_smoke_output(
    out_dir: Path,
    *,
    single_host_reference: Optional[Path] = None,
) -> Dict[str, Any]:
    if not out_dir.is_dir():
        raise AssertionError(f"smoke output directory does not exist: {out_dir}")
    summary_path = out_dir / "orchestrator_summary.json"
    if not summary_path.is_file():
        raise AssertionError(f"missing orchestrator summary: {summary_path}")

    expected_ingestion_fields = {item.name for item in fields(IngestedBoltzGenRun)}
    reference_fields = _reference_ingestion_fields(single_host_reference) if single_host_reference else None
    round_reports: List[Dict[str, Any]] = []
    for round_id in range(2):
        round_dir = out_dir / f"round_{round_id:02d}"
        missing = sorted(name for name in REQUIRED_ROUND_FILES if not (round_dir / name).is_file())
        if missing:
            raise AssertionError(f"round {round_id} is missing standard closed-loop files: {missing}")

        checkpoint = _load_json(round_dir / "round_checkpoint.json")
        if checkpoint.get("status") != "completed":
            raise AssertionError(f"round {round_id} checkpoint is not completed: {checkpoint.get('status')}")

        execution_records = _load_json(round_dir / "execution_records.json")
        if not isinstance(execution_records, list) or len(execution_records) != 1:
            raise AssertionError(f"round {round_id} must contain exactly one native Taiji execution record")
        record = execution_records[0]
        if str(record.get("status") or "").lower() != "completed":
            raise AssertionError(f"round {round_id} Taiji task did not complete: {record.get('status')}")

        submit_config_path = Path(record["submit_spec"]["simple_config_path"])
        submit_config = _load_json(submit_config_path)
        if submit_config.get("host_num") != 2 or submit_config.get("host_gpu_num") != 4:
            raise AssertionError(
                f"round {round_id} requested unexpected resources: "
                f"host_num={submit_config.get('host_num')} host_gpu_num={submit_config.get('host_gpu_num')}"
            )
        if submit_config.get("exec_start_in_all_mpi_pods") is not True:
            raise AssertionError(f"round {round_id} does not start the script in every MPI pod")

        package_dir = Path(record["local_package_dir"])
        output_root = package_dir / "outputs/boltzgen_output"
        plan = _load_json(package_dir / "configs/cluster_shard_plan.json")
        result_manifest = _load_json(output_root / "result_manifest.json")
        steps = yaml.safe_load((output_root / "steps.yaml").read_text(encoding="utf-8")) or {}
        distribution = steps.get("gpu_distribution") or {}

        _validate_cluster_plan(round_id, plan)
        if result_manifest.get("host_count") != 2 or result_manifest.get("gpus_per_host") != 4:
            raise AssertionError(f"round {round_id} result manifest has the wrong cluster shape")
        if int((result_manifest.get("status") or {}).get("code", -1)) != 0:
            raise AssertionError(f"round {round_id} cluster result manifest reports failure")
        if result_manifest.get("schema_version") != 6 or result_manifest.get("contract") != {"name": "binder_harness_result_manifest", "version": 1}:
            raise AssertionError(f"round {round_id} result manifest contract is not versioned")
        if result_manifest.get("required_artifacts") != ["steps.yaml"] or result_manifest.get("authoritative") != {"inventory": "files", "entities": "artifacts"}:
            raise AssertionError(f"round {round_id} successful result manifest lacks authoritative steps contract")
        if "steps.yaml" not in (result_manifest.get("files") or []):
            raise AssertionError(f"round {round_id} aggregate inventory omits root steps.yaml")
        if steps.get("schema_version") != 1 or steps.get("contract") != "binder_harness_steps_manifest":
            raise AssertionError(f"round {round_id} steps manifest contract is not versioned")
        if distribution.get("mode") != "native_multi_host" or distribution.get("status") != 0:
            raise AssertionError(f"round {round_id} steps.yaml is not a successful native multi-host manifest")

        shard_roots = _expected_shard_roots(output_root, plan)
        for shard_root in shard_roots:
            if not shard_root.is_dir():
                raise AssertionError(f"round {round_id} missing shard output: {shard_root}")
            missing_dirs = sorted(name for name in STANDARD_SHARD_DIRS if not (shard_root / name).is_dir())
            if missing_dirs:
                raise AssertionError(
                    f"round {round_id} shard {shard_root.name} does not preserve standard "
                    f"single-host BoltzGen directories: {missing_dirs}"
                )

        ingestions = _load_json(round_dir / "ingestions.json")
        if not isinstance(ingestions, list) or len(ingestions) != 1:
            raise AssertionError(f"round {round_id} must ingest the unified output root exactly once")
        ingestion = ingestions[0]
        ingestion_fields = set(ingestion)
        if ingestion_fields != expected_ingestion_fields:
            raise AssertionError(
                f"round {round_id} ingestion schema differs from the standard 1-host contract: "
                f"missing={sorted(expected_ingestion_fields - ingestion_fields)} "
                f"extra={sorted(ingestion_fields - expected_ingestion_fields)}"
            )
        if reference_fields is not None and ingestion_fields != reference_fields:
            raise AssertionError(f"round {round_id} ingestion schema differs from the supplied 1-host reference")
        if Path(ingestion["output_dir"]).resolve() != output_root.resolve():
            raise AssertionError(f"round {round_id} ingestion did not use the unified output root")

        metrics_files = [Path(path) for path in ingestion.get("metrics_files") or []]
        if not metrics_files:
            raise AssertionError(f"round {round_id} produced no ingestible metrics files")
        if any(not _is_relative_to(path, output_root) for path in metrics_files):
            raise AssertionError(f"round {round_id} metrics escaped the unified output root")

        round_reports.append({
            "round_id": round_id,
            "taiji_tasks": 1,
            "host_num": 2,
            "gpus_per_host": 4,
            "planned_designs": plan["total_designs"],
            "shards": len(plan["shards"]),
            "metrics_files": len(metrics_files),
            "candidates": len(ingestion.get("candidates") or []),
            "standard_output_contract": True,
        })

    summary = _load_json(summary_path)
    if len(summary.get("rounds") or []) < 2:
        raise AssertionError("orchestrator summary does not contain two completed rounds")
    return {
        "status": "passed",
        "config": "2 rounds × 8 designs × 2 hosts × 4 GPUs",
        "output_dir": str(out_dir),
        "single_taiji_task_per_round": True,
        "single_ingestion_per_round": True,
        "single_host_output_schema_compatible": True,
        "rounds": round_reports,
    }


def _validate_cluster_plan(round_id: int, plan: Mapping[str, Any]) -> None:
    if plan.get("mode") != "native" or plan.get("host_count") != 2 or plan.get("gpus_per_host") != 4:
        raise AssertionError(f"round {round_id} has an unexpected native cluster plan")
    if plan.get("worker_count") != 8 or plan.get("total_designs") != 8:
        raise AssertionError(f"round {round_id} does not use 8 workers for exactly 8 designs")
    shards = plan.get("shards") or []
    if sum(int(shard["num_designs"]) for shard in shards) != 8:
        raise AssertionError(f"round {round_id} shard budget is not conserved")
    locations = {
        (int(shard["host"]), int(shard["gpu"]), int(shard["shard_index"]))
        for shard in shards
    }
    if len(locations) != len(shards):
        raise AssertionError(f"round {round_id} contains overlapping host/GPU/shard assignments")
    if {int(shard["host"]) for shard in shards} != {0, 1}:
        raise AssertionError(f"round {round_id} did not use both hosts")
    if any(not 0 <= int(shard["gpu"]) < 4 for shard in shards):
        raise AssertionError(f"round {round_id} contains an invalid local GPU id")


def _expected_shard_roots(output_root: Path, plan: Mapping[str, Any]) -> List[Path]:
    roots = []
    for shard in plan.get("shards") or []:
        roots.append(
            output_root
            / f"host_{int(shard['host']):02d}"
            / f"gpu_{int(shard['gpu']):02d}"
            / f"shard_{int(shard['shard_index']):03d}_len{int(shard['length'])}"
        )
    return roots


def _reference_ingestion_fields(reference: Path) -> set[str]:
    path = reference / "round_00/ingestions.json"
    ingestions = _load_json(path)
    if not isinstance(ingestions, list) or not ingestions:
        raise AssertionError(f"invalid single-host reference ingestion: {path}")
    return set(ingestions[0])


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise AssertionError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
