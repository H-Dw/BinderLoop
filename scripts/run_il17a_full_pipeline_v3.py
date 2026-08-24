#!/usr/bin/env python3
"""IL-17A Full Pipeline Test v3: Use Ceph-mounted code path.

Strategy: Submit with model_id/dataset_id (fast, no upload), but start_cmd
mounts Ceph and runs the script from the actual Ceph path where our package lives.
"""

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents import (
    DesignParameterAgent,
    DesignSpecAgent,
    TaijiExecutionAgent,
    RunMonitorAgent,
    ResultIngestionAgent,
    EvaluationAgent,
    ActiveLearningPolicyAgent,
    StructureEvaluationAgent,
)
from binderloop.agents.hypothesis_agent import HypothesisAgent
from binderloop.agents.binder_quality_analysis_agent import BinderQualityAnalysisAgent
from binderloop.config import load_config
from binderloop.llm import OpenAICompatibleClient
from binderloop.models.base import DesignJob
from binderloop.secrets import SecretStore

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/il17a_full_pipeline_test_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOLTZGEN_CHECKPOINT_DIR = Path("/aceph/daweihuang/program/boltzgen/checkpoints")
BOLTZGEN_CACHE_DIR = Path("/aceph/daweihuang/program/boltzgen/cache")
BOLTZGEN_MOLDIR = (
    BOLTZGEN_CACHE_DIR
    / "datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"
)


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    print("=" * 70)
    print("IL-17A Full Pipeline Test v3 (Ceph-mounted code path)")
    print("=" * 70)

    cfg = load_config(ROOT / "configs/il17a_full_pipeline_test.yaml")
    secrets = SecretStore.from_json(ROOT / "configs/llm_endpoints.local.json")
    ceph_secret = secrets.ceph_secret()
    llm = OpenAICompatibleClient.from_json(ROOT / "configs/llm_endpoints.local.json")
    print(f"  LLM available: {llm.available() if llm else False}")
    print(f"  CEPH_SECRET: {'set' if ceph_secret else 'MISSING'}")

    # ─── Step 1: Parameter Design ───────────────────────────────────────
    print("\n[Step 1] DesignParameterAgent...")
    param_agent = DesignParameterAgent()
    params = param_agent.choose_boltzgen_parameters(cfg)
    params.update({
        "task_id": "il17a_len70_seed0_v3",
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
        "num_designs": 10,
        "budget": 5,
        "diffusion_batch_size": 1,
        "cache": str(BOLTZGEN_CACHE_DIR),
        "moldir": str(BOLTZGEN_MOLDIR),
        "design_checkpoints": [
            str(BOLTZGEN_CHECKPOINT_DIR / "boltzgen1_diverse.ckpt"),
            str(BOLTZGEN_CHECKPOINT_DIR / "boltzgen1_adherence.ckpt"),
        ],
        "inverse_fold_checkpoint": str(BOLTZGEN_CHECKPOINT_DIR / "boltzgen1_ifold.ckpt"),
        "run_filtering": True,
        "keep_unfiltered_for_failure_analysis": True,
    })
    param_agent.write_parameter_plan(params, OUT_DIR / "01_design_parameter_plan.yaml")
    print(f"  OK: protocol={params.get('protocol')}, designs={params.get('num_designs')}")

    # ─── Step 2: Design Spec ────────────────────────────────────────────
    print("\n[Step 2] DesignSpecAgent: creating run spec (length=70)...")
    job = DesignJob(
        job_id="il17a_len70_seed0_v3",
        target_structure=str(ROOT / "examples/bg_example/IL-17A.cif"),
        chain_id="A",
        hotspots=["A:67", "A:89", "B:49"],
        binder_length=70,
        seed=0,
        params=params,
        output_dir=str(OUT_DIR / "round0_len70_seed0"),
    )
    spec_agent = DesignSpecAgent(ROOT.parent / "boltzgen")
    run_spec = spec_agent.create_boltzgen_run_spec(job, params=params)
    print(f"  Package: {run_spec.package_dir}")
    print(f"  Script: {run_spec.run_script_path}")

    # ─── Step 3: Direct Taiji Submission (Ceph code path) ───────────────
    print("\n[Step 3] TaijiExecutionAgent: submitting with Ceph-path start_cmd...")
    task_flag = f"il17a_binder_v3_{int(time.time())}"

    # Build a start_cmd that mounts Ceph first, then cd to the package dir on Ceph
    package_dir = run_spec.package_dir  # e.g. /aceph/daweihuang/program/binder-harness/outputs/.../taiji_project_package
    run_script_rel = "scripts/run_boltzgen_full.sh"

    start_cmd = f"""bash -lc '
set -euo pipefail
mkdir -p /aceph/daweihuang
if ! mountpoint -q /aceph/daweihuang 2>/dev/null; then
  mount -t ceph 11.18.83.17:6789,11.18.83.31:6789,11.18.83.32:6789:/fandiwu/buddy1/daweihuang /aceph/daweihuang -o name=fandiwubuddy1,secret="${{CEPH_SECRET}}"
fi
cd {package_dir}
echo "[HARNESS] workspace=$(pwd)"
bash {run_script_rel}
'"""

    # Build Taiji simple config manually
    taiji_config = {
        "business_flag": "pathology_gpu_chongqing",
        "project_id": 192631,
        "task_flag": task_flag,
        "readable_name": task_flag,
        "image_full_name": "mirrors.tencent.com/davedwhuang/boltzgen:cu118",
        "GPUName": "V100",
        "host_gpu_num": 1,
        "host_num": 1,
        "cuda_version": "11.0",
        "priority_level": "HIGH",
        "quota_type": "public",
        "location": "cq",
        "is_resource_waiting": True,
        "is_elasticity": False,
        "enable_evicted_end_task": True,
        "exec_start_in_all_mpi_pods": True,
        "version": "v2.0",
        "dataset_id": "8b1d82389dfc7401019dfd3046540076",
        "model_id": "8b1d81e89dfc747f019dfd304ccf0080",
        "start_cmd": start_cmd,
        "envs": {
            "CEPH_SECRET": ceph_secret,
            "NUM_DESIGNS": "10",
            "TAIJI_TIMEOUT": "3600",
        },
    }

    config_path = OUT_DIR / "02_taiji_simple_config.json"
    config_path.write_text(json.dumps(taiji_config, ensure_ascii=False, indent=4), encoding="utf-8")

    # Submit using taiji_client directly
    import subprocess
    submit_cmd = ["taiji_client", "start", "-scfg", str(config_path)]
    print(f"  Command: {' '.join(submit_cmd)}")
    proc = subprocess.run(submit_cmd, text=True, capture_output=True, check=False)
    print(f"  returncode: {proc.returncode}")
    print(f"  stdout: {proc.stdout.strip()}")
    if proc.stderr:
        print(f"  stderr: {proc.stderr.strip()[-300:]}")

    # Extract job ID
    import re
    taiji_job_id = None
    for pattern in [r"instance[_ -]?id[:=\s]+([A-Za-z0-9_.:-]+)", r"job[_ -]?id[:=\s]+([A-Za-z0-9_.:-]+)"]:
        m = re.search(pattern, proc.stdout + proc.stderr, flags=re.IGNORECASE)
        if m:
            taiji_job_id = m.group(1)
            break

    submission_record = {
        "task_flag": task_flag,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "taiji_job_id": taiji_job_id,
        "submit_command": " ".join(submit_cmd),
        "config_path": str(config_path),
    }
    write_json(OUT_DIR / "03_submission_record.json", submission_record)
    print(f"  Taiji job ID: {taiji_job_id}")

    if proc.returncode != 0:
        print("  ERROR: Submission failed!")
        # Continue anyway for demonstration

    # ─── Step 4: Monitor ────────────────────────────────────────────────
    print("\n[Step 4] RunMonitorAgent: monitoring...")
    monitor = RunMonitorAgent()
    max_checks = 60
    check_interval = 30
    final_snapshot = None

    for check_num in range(1, max_checks + 1):
        snapshot = monitor.check_once(
            task_flag=task_flag,
            instance_id=taiji_job_id,
            expected_outputs=run_spec.expected_outputs,
            simple_config_path=str(config_path),
        )
        state = snapshot.state
        ts = time.strftime('%H:%M:%S')
        print(f"  [{ts}] #{check_num}: state={state} terminal={snapshot.is_terminal}")

        if snapshot.log_tail and (check_num % 5 == 0 or snapshot.is_terminal):
            tail = snapshot.log_tail.strip()[-300:]
            print(f"    Log: {tail}")

        final_snapshot = snapshot
        if snapshot.is_terminal:
            break
        if not snapshot.needs_followup and state not in ("unknown",):
            break
        time.sleep(check_interval)

    if final_snapshot:
        monitor.write_snapshot(final_snapshot, OUT_DIR / "04_monitor_final.json")
        print(f"\n  Final: state={final_snapshot.state} success={final_snapshot.is_success}")
        if final_snapshot.failure_hints:
            print(f"  Hints: {final_snapshot.failure_hints}")

    # ─── Step 5-10: Result collection & analysis ────────────────────────
    print("\n[Step 5] Collecting results...")
    ingestor = ResultIngestionAgent()
    ingested = ingestor.ingest_boltzgen_output(run_spec.output_dir, log_file=run_spec.log_file)
    ingestor.write_manifest(ingested, OUT_DIR / "05_result_ingestion.json")
    print(f"  Metrics: {len(ingested.metrics_files)}, Candidates: {len(ingested.candidates)}")

    print("\n[Step 6] Evaluation...")
    evaluator = EvaluationAgent()
    if ingested.candidates:
        evaluation = evaluator.evaluate_candidates(ingested.candidates)
    else:
        from binderloop.agents.evaluation_agent import EvaluationSummary
        evaluation = EvaluationSummary(0, 0, 0, {}, [], [], ["No candidates collected."])
    evaluator.write_summary(evaluation, OUT_DIR / "06_evaluation_summary.json")
    evaluator.write_scores_csv(evaluation, OUT_DIR / "06_scores.csv")
    print(f"  Total={evaluation.total_candidates} Pass={evaluation.success_count} Fail={evaluation.failure_count}")
    print(f"  Tags: {evaluation.tag_counts}")
    for obs in evaluation.observations:
        print(f"    - {obs}")

    print("\n[Step 7] Structure evaluation...")
    struct_agent = StructureEvaluationAgent()
    structure_files = [p for p in ingested.final_design_files if p.lower().endswith((".pdb", ".cif"))]
    struct_eval = struct_agent.analyze_structures(structure_files, binder_chain="D", target_chains=["A", "B"], hotspots=["A:67", "A:89", "B:49"])
    struct_agent.write_batch(struct_eval, OUT_DIR / "07_structure_eval.json")
    print(f"  Structures: {struct_eval.total_structures}, Reliable: {struct_eval.reliable_seed_fraction:.2f}")

    print("\n[Step 8] Quality analysis (LLM)...")
    quality_agent = BinderQualityAnalysisAgent(llm=llm)
    context = {"round_id": 0, "evaluation": asdict(evaluation), "structural_analysis": asdict(struct_eval), "memory": {}, "messages": []}
    quality = quality_agent.analyze(round_id=0, context=context)
    quality_agent.write_analysis(quality, OUT_DIR / "08_quality_analysis.json")
    print(f"  LLM={quality.llm_used}: {quality.overall_assessment[:150]}")

    print("\n[Step 9] Hypothesis generation...")
    hyp_agent = HypothesisAgent(llm=llm)
    context["quality_analysis"] = asdict(quality)
    hypotheses = hyp_agent.propose(context)
    write_json(OUT_DIR / "09_hypotheses.json", asdict(hypotheses))
    print(f"  LLM={hypotheses.llm_used}, Count={len(hypotheses.hypotheses)}")
    for h in hypotheses.hypotheses[:3]:
        print(f"    - {h.get('name')}: conf={h.get('confidence')}")

    print("\n[Step 10] Next round proposal...")
    policy = ActiveLearningPolicyAgent()
    proposal = policy.propose_next_boltzgen_params(evaluation, params, round_id=1, structural_summary=struct_eval, hypotheses=hypotheses.hypotheses, quality_analysis=asdict(quality))
    policy.write_proposal(proposal, OUT_DIR / "10_next_round_proposal.json")
    for r in proposal.rationale:
        print(f"    - {r}")

    # ─── Summary ────────────────────────────────────────────────────────
    report = {
        "pipeline": "IL-17A Full Pipeline Test v3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "IL-17A homodimer (A+B)",
        "design_length": 70,
        "task_flag": task_flag,
        "taiji_job_id": taiji_job_id,
        "final_state": final_snapshot.state if final_snapshot else "unknown",
        "final_success": final_snapshot.is_success if final_snapshot else False,
        "candidates": len(ingested.candidates),
        "pass_count": evaluation.success_count,
        "fail_count": evaluation.failure_count,
        "structures": struct_eval.total_structures,
        "llm_quality": quality.llm_used,
        "hypotheses": len(hypotheses.hypotheses),
        "proposal_rationale": proposal.rationale,
        "instance_url": f"http://taiji.woa.com/#/project-list/jizhi/task-inst-detail/{taiji_job_id}" if taiji_job_id else None,
    }
    write_json(OUT_DIR / "pipeline_report.json", report)
    print("\n" + "=" * 70)
    print(f"Done! Report: {OUT_DIR}/pipeline_report.json")
    if taiji_job_id:
        print(f"Taiji: http://taiji.woa.com/#/project-list/jizhi/task-inst-detail/{taiji_job_id}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
