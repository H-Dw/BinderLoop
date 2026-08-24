#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents import (
    ActiveLearningPolicyAgent,
    DesignParameterAgent,
    DesignSpecAgent,
    EvaluationAgent,
    ResultIngestionAgent,
    RunMonitorAgent,
    TaijiExecutionAgent,
)
from binderloop.config import load_config
from binderloop.models.base import DesignJob
from binderloop.package_layout import PROJECT_PACKAGE_DIRNAME
from binderloop.secrets import SecretStore, redact_sensitive

ROOT = Path(__file__).resolve().parents[1]
BOLTZGEN_CHECKPOINT_DIR = Path("/aceph/daweihuang/program/boltzgen/checkpoints")
BOLTZGEN_CACHE_DIR = Path("/aceph/daweihuang/program/boltzgen/cache")
BOLTZGEN_MOLDIR = (
    BOLTZGEN_CACHE_DIR
    / "datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"
)
TAIJI_REMOTE_RUN_ROOT = Path("/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests")


REQUIRED_AGENT_FILES = {
    "DesignParameterAgent": ROOT / "binderloop/agents/design_parameter_agent.py",
    "DesignSpecAgent": ROOT / "binderloop/agents/design_spec_agent.py",
    "TaijiExecutionAgent": ROOT / "binderloop/agents/taiji_execution_agent.py",
    "RunMonitorAgent": ROOT / "binderloop/agents/run_monitor_agent.py",
    "ResultIngestionAgent": ROOT / "binderloop/agents/result_ingestion_agent.py",
    "EvaluationAgent": ROOT / "binderloop/agents/evaluation_agent.py",
    "ActiveLearningPolicyAgent": ROOT / "binderloop/agents/active_learning_policy_agent.py",
}


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def check_module_coverage(out_dir: Path) -> dict:
    coverage = {name: {"path": str(path), "exists": path.exists()} for name, path in REQUIRED_AGENT_FILES.items()}
    missing = [name for name, item in coverage.items() if not item["exists"]]
    report = {
        "docs_checked": [
            str(ROOT / "docs/research_plan.md"),
            str(ROOT / "docs/boltzgen_taiji_agents.md"),
            str(ROOT.parent / "Binder-Harness.pdf"),
        ],
        "required_agents": coverage,
        "missing_agents": missing,
        "status": "pass" if not missing else "missing_agents",
        "notes": [
            "research_plan requires StrategyPlanner/ModelExecutor/ResultParser/ScientificCritic/ActiveLearner separation.",
            "boltzgen_taiji_agents requires parameter/spec/taiji/monitor separation.",
            "Binder-Harness requires failure capture, evaluation, and next-round strategy update.",
        ],
    }
    write_json(out_dir / "00_module_coverage.json", report)
    return report


def maybe_create_mock_boltzgen_results(output_dir: Path) -> Path:
    final_dir = output_dir / "final_ranked_designs"
    final_dir.mkdir(parents=True, exist_ok=True)
    metrics = final_dir / "all_designs_metrics.csv"
    rows = [
        {
            "design": "success_high_interface",
            "iptm": "0.76",
            "hotspot_contact": "0.82",
            "plddt": "0.88",
            "designfolding-filter_rmsd": "1.2",
            "diversity": "0.61",
            "sequence_designability": "0.84",
        },
        {
            "design": "failure_hotspot_miss",
            "iptm": "0.69",
            "hotspot_contact": "0.18",
            "plddt": "0.81",
            "designfolding-filter_rmsd": "1.4",
            "diversity": "0.57",
            "sequence_designability": "0.78",
        },
        {
            "design": "failure_folding_unstable",
            "iptm": "0.55",
            "hotspot_contact": "0.62",
            "plddt": "0.42",
            "designfolding-filter_rmsd": "5.8",
            "diversity": "0.49",
            "sequence_designability": "0.38",
        },
        {
            "design": "failure_pose_low_confidence",
            "iptm": "0.31",
            "hotspot_contact": "0.54",
            "plddt": "0.79",
            "designfolding-filter_rmsd": "1.9",
            "diversity": "0.52",
            "sequence_designability": "0.72",
        },
    ]
    with metrics.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "intermediate_designs").mkdir(exist_ok=True)
    (output_dir / "intermediate_designs_inverse_folded").mkdir(exist_ok=True)
    return metrics


def resolve_taiji_template(template_json: Optional[str], allow_template_secrets: bool) -> Optional[Path]:
    if template_json:
        path = Path(template_json).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Taiji template JSON not found: {path}")
        return path

    if not allow_template_secrets:
        return None

    # Prefer an operator-provided real secret template when present.  Do not
    # silently load *.template.json here: it contains placeholder/legacy fields
    # intended for reference, while the v2 path below can submit without it.
    real_template = ROOT / "examples/bg_example/boltzgen_test_v100.json"
    if real_template.exists():
        return real_template

    return None


def sync_package_to_remote_run_dir(package_dir: Union[str, Path], task_flag: str) -> Path:
    package_dir = Path(package_dir)
    remote_package_dir = TAIJI_REMOTE_RUN_ROOT / task_flag / PROJECT_PACKAGE_DIRNAME
    remote_package_dir.parent.mkdir(parents=True, exist_ok=True)
    if remote_package_dir.exists():
        shutil.rmtree(remote_package_dir)
    shutil.copytree(package_dir, remote_package_dir, symlinks=True)
    return remote_package_dir


def point_run_spec_to_package(run_spec, package_dir: Path) -> None:
    output_dir = package_dir / "outputs" / "boltzgen_output"
    log_file = package_dir / "logs" / "boltzgen_full.log"
    previous_expected = dict(run_spec.expected_outputs)
    run_spec.package_dir = str(package_dir)
    run_spec.design_spec_path = str(package_dir / "configs" / "boltzgen_design_spec.yaml")
    run_spec.run_script_path = str(package_dir / "scripts" / "run_boltzgen_full.sh")
    run_spec.output_dir = str(output_dir)
    run_spec.log_file = str(log_file)
    expected_outputs = {
        "package_dir": str(package_dir),
        "target_file": str(package_dir / "inputs" / "IL-17A.cif"),
        "boltzgen_output_dir": str(output_dir),
        "steps_manifest": str(output_dir / "steps.yaml"),
        "log_file": str(log_file),
    }
    optional_paths = {
        "intermediate_designs": str(output_dir / "intermediate_designs"),
        "inverse_folded_designs": str(output_dir / "intermediate_designs_inverse_folded"),
        "final_ranked_designs": str(output_dir / "final_ranked_designs"),
        "analysis_metrics_candidates": str(output_dir / "**" / "*.csv"),
    }
    for key, value in optional_paths.items():
        if key in previous_expected:
            expected_outputs[key] = value
    run_spec.expected_outputs = expected_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Small end-to-end BoltzGen/Taiji harness path test")
    parser.add_argument("--submit", action="store_true", help="Actually run taiji_client start -scfg. Without this, generate dry-run artifacts only.")
    parser.add_argument("--allow-template-secrets", action="store_true", help="Allow using a Taiji template JSON. Placeholder secrets are dropped automatically.")
    parser.add_argument("--template-json", help="Optional Taiji template JSON path. Relative paths are resolved from the repository root.")
    parser.add_argument("--secret-config", default="configs/llm_endpoints.local.json", help="Ignored local JSON that may contain secrets.CEPH_SECRET.")
    parser.add_argument("--analysis-on-taiji", action="store_true", help="Run BoltzGen analysis/filtering inside Taiji. Default is GPU generation on Taiji and analysis locally.")
    parser.add_argument("--run-local-analysis", action="store_true", help="Run the generated local BoltzGen analysis/filtering script before result ingestion.")
    parser.add_argument("--no-mock-results", action="store_true", help="Do not write mock metrics when real analysis metrics are not available yet.")
    parser.add_argument("--task-flag", help="Optional Taiji task_flag. Defaults to a unique timestamped value for real submissions.")
    parser.add_argument("--out", default="outputs/boltzgen_complete_path_test")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    secret_config_path = Path(args.secret_config).expanduser()
    if not secret_config_path.is_absolute():
        secret_config_path = ROOT / secret_config_path
    secret_store = SecretStore.from_json(secret_config_path)

    report = {"out_dir": str(out_dir), "steps": []}
    coverage = check_module_coverage(out_dir)
    report["module_coverage"] = coverage
    if coverage["missing_agents"]:
        write_json(out_dir / "path_test_report.json", report)
        print(f"Missing agents: {coverage['missing_agents']}")
        return 2

    cfg = load_config(ROOT / "configs/example_binder_task.yaml")
    params = DesignParameterAgent().choose_boltzgen_parameters(cfg)
    params.update(
        {
            "task_id": "complete_path_len50",
            "target_include": [
                {"chain": {"id": "A", "res_index": "1..104"}},
                {"chain": {"id": "B", "res_index": "1..109"}},
            ],
            "target_binding_types": [
                {"chain": {"id": "A", "binding": "67,89"}},
                {"chain": {"id": "B", "binding": "49"}},
            ],
            "structure_groups": "all",
            "devices": 1,
            "num_designs": 4,
            "budget": 2,
            "diffusion_batch_size": 1,
            "analysis_location": "taiji" if args.analysis_on_taiji else "local",
            "cache": str(BOLTZGEN_CACHE_DIR),
            "moldir": str(BOLTZGEN_MOLDIR),
            "design_checkpoints": [
                str(BOLTZGEN_CHECKPOINT_DIR / "boltzgen1_diverse.ckpt"),
                str(BOLTZGEN_CHECKPOINT_DIR / "boltzgen1_adherence.ckpt"),
            ],
            "inverse_fold_checkpoint": str(BOLTZGEN_CHECKPOINT_DIR / "boltzgen1_ifold.ckpt"),
            "run_filtering": True,
            "keep_unfiltered_for_failure_analysis": True,
        }
    )
    DesignParameterAgent().write_parameter_plan(params, out_dir / "01_design_parameter_plan.yaml")
    report["steps"].append("design_parameter_plan_written")

    job = DesignJob(
        job_id="complete_path_len50_seed0",
        target_structure=str(ROOT / "examples/bg_example/IL-17A.cif"),
        chain_id="A",
        hotspots=["A:67", "A:89", "B:49"],
        binder_length=50,
        seed=0,
        params=params,
        output_dir=str(out_dir / "round0_len50_seed0"),
    )
    run_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(job, params=params)
    report["steps"].append("boltzgen_full_run_script_written")

    task_flag = args.task_flag or (
        f"binder_boltzgen_complete_path_len50_{int(time.time())}"
        if args.submit
        else "binder_boltzgen_complete_path_len50"
    )
    remote_package_dir = sync_package_to_remote_run_dir(run_spec.package_dir or Path(run_spec.run_script_path).parents[1], task_flag)
    point_run_spec_to_package(run_spec, remote_package_dir)
    report["run_spec"] = asdict(run_spec)
    report["remote_package_dir"] = str(remote_package_dir)
    report["steps"].append("project_package_synced_to_ceph")

    template = resolve_taiji_template(args.template_json, args.allow_template_secrets)
    taiji_options = {
        "business_flag": "pathology_gpu_chongqing",
        "project_id": 192631,
        "image_full_name": "mirrors.tencent.com/davedwhuang/boltzgen:cu118",
        "GPUName": "V100",
        "host_gpu_num": 1,
        "host_num": 1,
        "cuda_version": "11.0",
        "priority_level": "HIGH",
        "quota_type": "public",
        "location": "cq",
        "version": "v2.0",
        "dataset_id": "8b1d82389dfc7401019dfd3046540076",
        "model_id": "8b1d81e89dfc747f019dfd304ccf0080",
        "remote_project_dir": str(remote_package_dir),
    }
    ceph_secret = secret_store.ceph_secret()
    if ceph_secret:
        taiji_options["envs"] = {"CEPH_SECRET": ceph_secret}
    submit_spec = TaijiExecutionAgent(dry_run=not args.submit).create_boltzgen_taiji_spec(
        run_spec,
        template_json=template,
        output_json=out_dir / "02_taiji_simple_config.json",
        task_flag=task_flag,
        taiji_options=taiji_options,
    )
    report["taiji_submit_spec"] = {k: v for k, v in asdict(submit_spec).items() if k != "simple_config"}
    report["steps"].append("taiji_simple_config_written")

    submission = TaijiExecutionAgent(dry_run=not args.submit).submit(submit_spec)
    report["submission"] = redact_sensitive(asdict(submission))
    report["steps"].append("taiji_submit_attempted" if args.submit else "taiji_submit_dry_run")
    submit_failed = bool(args.submit and (submission.returncode not in (0, None) or not submission.taiji_job_id))

    monitor_snapshot = None
    if submission.taiji_job_id:
        monitor_snapshot = RunMonitorAgent().check_once(
            task_flag=submit_spec.task_flag,
            instance_id=submission.taiji_job_id,
            expected_outputs=run_spec.expected_outputs,
            simple_config_path=submit_spec.simple_config_path,
            config_path=submit_spec.full_config_path,
        )
        RunMonitorAgent().write_snapshot(monitor_snapshot, out_dir / "03_run_monitor_snapshot.json")
        report["monitor"] = asdict(monitor_snapshot)
    else:
        report["monitor"] = {
            "state": "not_started_or_no_instance_id",
            "reason": "Taiji submission did not return an instance/job id; inspect submission stdout/stderr.",
        }
    report["steps"].append("run_monitor_checked")

    local_analysis_script = Path(run_spec.package_dir or Path(run_spec.run_script_path).parents[1]) / "scripts/run_boltzgen_analysis_local.sh"
    if args.run_local_analysis:
        if not local_analysis_script.exists():
            report["local_analysis"] = {"status": "missing_script", "path": str(local_analysis_script)}
        else:
            proc = subprocess.run(["bash", str(local_analysis_script)], text=True, capture_output=True, check=False)
            report["local_analysis"] = {
                "status": "completed" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "script": str(local_analysis_script),
            }
        report["steps"].append("local_analysis_attempted")

    real_ingested = ResultIngestionAgent().ingest_boltzgen_output(run_spec.output_dir, log_file=run_spec.log_file)
    ResultIngestionAgent().write_manifest(real_ingested, out_dir / "04_real_result_ingestion.json")

    if not real_ingested.candidates and not args.no_mock_results and (not args.submit or args.analysis_on_taiji):
        mock_metrics = maybe_create_mock_boltzgen_results(Path(run_spec.output_dir))
        report["result_collection_note"] = f"No real BoltzGen metrics collected; wrote mock metrics for CPU-only evaluation-path test: {mock_metrics}"
    elif not real_ingested.candidates:
        report["result_collection_note"] = (
            "No real BoltzGen metrics collected yet. Taiji analysis is disabled by default; "
            "run scripts/run_boltzgen_analysis_local.sh in the synced package after GPU generation finishes, "
            "or rerun with --analysis-on-taiji."
        )
    ingested = ResultIngestionAgent().ingest_boltzgen_output(run_spec.output_dir, log_file=run_spec.log_file)
    ResultIngestionAgent().write_manifest(ingested, out_dir / "05_result_ingestion_for_evaluation.json")
    report["steps"].append("results_collected")

    evaluator = EvaluationAgent()
    summary = evaluator.evaluate_candidates(ingested.candidates)
    evaluator.write_summary(summary, out_dir / "06_evaluation_summary.json")
    evaluator.write_scores_csv(summary, out_dir / "06_scores_preview.csv")
    report["evaluation"] = asdict(summary)
    report["steps"].append("evaluation_completed")

    proposal = ActiveLearningPolicyAgent().propose_next_boltzgen_params(summary, params, round_id=1)
    ActiveLearningPolicyAgent().write_proposal(proposal, out_dir / "07_next_round_parameter_proposal.json")
    report["next_round_proposal"] = asdict(proposal)
    report["steps"].append("next_round_parameters_proposed")

    write_json(out_dir / "path_test_report.json", report)
    print(f"Path test report: {out_dir / 'path_test_report.json'}")
    print(f"Submission dry_run={submission.dry_run} returncode={submission.returncode}")
    if submission.stderr:
        print("Submission stderr tail:", submission.stderr[-500:])
    if submit_failed:
        print("Submission did not produce a Taiji instance id; inspect taiji_submission_record.json")
    print("Evaluation observations:")
    for obs in summary.observations:
        print("-", obs)
    print("Next round rationale:")
    for item in proposal.rationale:
        print("-", item)
    return 3 if submit_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
