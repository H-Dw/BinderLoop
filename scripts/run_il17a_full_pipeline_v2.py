#!/usr/bin/env python3
"""IL-17A Full Pipeline Test v2: Fix Taiji code path by using local file upload.

The v2.0 model_id/dataset_id approach uses a pre-registered model package. Since
our generated run scripts are new, we must use model_local_file_path to upload the
project package that contains our scripts, configs, and inputs.
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
OUT_DIR = ROOT / "outputs/il17a_full_pipeline_test_v2"
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
    print("IL-17A Full Pipeline Test v2 (Local Upload)")
    print("=" * 70)

    # Load config
    cfg = load_config(ROOT / "configs/il17a_full_pipeline_test.yaml")

    # Load secrets for CEPH_SECRET
    secrets = SecretStore.from_json(ROOT / "configs/llm_endpoints.local.json")
    ceph_secret = secrets.ceph_secret()
    if not ceph_secret:
        print("ERROR: CEPH_SECRET not found in configs/llm_endpoints.local.json")
        return 1

    # Load LLM client
    llm = OpenAICompatibleClient.from_json(ROOT / "configs/llm_endpoints.local.json")
    print(f"  LLM available: {llm.available() if llm else False}")

    # ─── Step 1: Design Parameter Agent ─────────────────────────────────
    print("\n[Step 1] DesignParameterAgent: choosing parameters...")
    param_agent = DesignParameterAgent()
    params = param_agent.choose_boltzgen_parameters(cfg)
    params.update({
        "task_id": "il17a_len70_seed0_v2",
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
    print(f"  Protocol: {params.get('protocol')}, Designs: {params.get('num_designs')}")

    # ─── Step 2: Design Spec Agent ──────────────────────────────────────
    print("\n[Step 2] DesignSpecAgent: creating BoltzGen run spec (length=70)...")
    job = DesignJob(
        job_id="il17a_len70_seed0_v2",
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
    print(f"  Package dir: {run_spec.package_dir}")
    print(f"  Run script: {run_spec.run_script_path}")

    # ─── Step 3: Taiji Execution Agent (Local Upload) ───────────────────
    print("\n[Step 3] TaijiExecutionAgent: creating Taiji config (LOCAL UPLOAD)...")
    task_flag = f"il17a_binder_len70_v2_{int(time.time())}"

    # KEY FIX: Use model_local_file_path instead of model_id/dataset_id
    # This uploads the project package to Taiji so scripts are available
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
        # Use local file upload - NOT model_id/dataset_id
        "model_local_file_path": run_spec.package_dir,
        "envs": {"CEPH_SECRET": ceph_secret},
    }

    taiji_agent = TaijiExecutionAgent(dry_run=False)
    submit_spec = taiji_agent.create_boltzgen_taiji_spec(
        run_spec,
        template_json=None,
        output_json=OUT_DIR / "02_taiji_simple_config.json",
        task_flag=task_flag,
        taiji_options=taiji_options,
    )
    print(f"  Task flag: {task_flag}")
    print(f"  Submit cmd: {submit_spec.submit_command}")

    # Verify the config has model_local_file_path set
    cfg_data = json.loads(Path(submit_spec.simple_config_path).read_text())
    print(f"  model_local_file_path: {cfg_data.get('model_local_file_path', 'NOT SET')}")
    print(f"  model_id: {cfg_data.get('model_id', 'NOT SET')}")

    # Submit!
    submission = taiji_agent.submit(submit_spec)
    write_json(OUT_DIR / "03_taiji_submission_record.json", asdict(submission))
    print(f"\n  Submitted! returncode={submission.returncode}")
    print(f"  Taiji job ID: {submission.taiji_job_id}")
    if submission.stdout:
        print(f"  stdout: {submission.stdout.strip()}")
    if submission.stderr:
        print(f"  stderr: {submission.stderr.strip()[-300:]}")

    if submission.returncode != 0 or not submission.taiji_job_id:
        print("\n  WARNING: Submission may have failed. Continuing with monitoring anyway...")

    # ─── Step 4: Run Monitor Agent ──────────────────────────────────────
    print("\n[Step 4] RunMonitorAgent: monitoring job status...")
    monitor = RunMonitorAgent()
    max_checks = 60  # Up to ~30 minutes
    check_interval = 30

    final_snapshot = None
    for check_num in range(1, max_checks + 1):
        snapshot = monitor.check_once(
            task_flag=task_flag,
            instance_id=submission.taiji_job_id,
            expected_outputs=run_spec.expected_outputs,
            simple_config_path=submit_spec.simple_config_path,
            config_path=submit_spec.full_config_path,
        )
        state = snapshot.state
        is_running = state in ("training_running", "running", "ready", "starting")
        print(f"  [{time.strftime('%H:%M:%S')}] Check #{check_num}: state={state} terminal={snapshot.is_terminal}")

        if snapshot.log_tail and check_num % 3 == 0:
            # Print log periodically
            tail = snapshot.log_tail.replace('\n', ' ')[-200:]
            print(f"    Log: ...{tail}")

        final_snapshot = snapshot
        if snapshot.is_terminal:
            break
        if not snapshot.needs_followup and state != "unknown":
            break
        time.sleep(check_interval)

    if final_snapshot:
        monitor.write_snapshot(final_snapshot, OUT_DIR / "04_run_monitor_final.json")
        print(f"\n  Final state: {final_snapshot.state}")
        print(f"  Success: {final_snapshot.is_success}")
        if final_snapshot.failure_hints:
            print(f"  Hints: {final_snapshot.failure_hints}")

    # ─── Step 5: Result Ingestion ───────────────────────────────────────
    print("\n[Step 5] ResultIngestionAgent: collecting results...")
    ingestor = ResultIngestionAgent()
    ingested = ingestor.ingest_boltzgen_output(run_spec.output_dir, log_file=run_spec.log_file)
    ingestor.write_manifest(ingested, OUT_DIR / "05_result_ingestion.json")
    print(f"  Metrics files: {len(ingested.metrics_files)}")
    print(f"  Candidates: {len(ingested.candidates)}")
    print(f"  Issues: {ingested.run_level_issues}")

    # ─── Step 6: Evaluation Agent ───────────────────────────────────────
    print("\n[Step 6] EvaluationAgent: scoring candidates...")
    evaluator = EvaluationAgent()
    if ingested.candidates:
        evaluation = evaluator.evaluate_candidates(ingested.candidates)
    else:
        from binderloop.agents.evaluation_agent import EvaluationSummary
        evaluation = EvaluationSummary(0, 0, 0, {}, [], [], ["No candidates collected from this run."])
    evaluator.write_summary(evaluation, OUT_DIR / "06_evaluation_summary.json")
    evaluator.write_scores_csv(evaluation, OUT_DIR / "06_scores_preview.csv")
    print(f"  Total: {evaluation.total_candidates}, Pass: {evaluation.success_count}, Fail: {evaluation.failure_count}")
    print(f"  Tags: {evaluation.tag_counts}")
    for obs in evaluation.observations:
        print(f"    - {obs}")

    # ─── Step 7: Structure Evaluation ───────────────────────────────────
    print("\n[Step 7] StructureEvaluationAgent: analyzing structures...")
    struct_agent = StructureEvaluationAgent()
    structure_files = [p for p in ingested.final_design_files if p.lower().endswith((".pdb", ".cif"))]
    struct_eval = struct_agent.analyze_structures(
        structure_files, binder_chain="D",
        target_chains=["A", "B"], hotspots=["A:67", "A:89", "B:49"]
    )
    struct_agent.write_batch(struct_eval, OUT_DIR / "07_structure_evaluation.json")
    print(f"  Structures: {struct_eval.total_structures}")

    # ─── Step 8: Quality Analysis (LLM) ────────────────────────────────
    print("\n[Step 8] BinderQualityAnalysisAgent: LLM-assisted analysis...")
    quality_agent = BinderQualityAnalysisAgent(llm=llm)
    context = {
        "round_id": 0,
        "evaluation": asdict(evaluation),
        "structural_analysis": asdict(struct_eval),
        "memory": {},
        "messages": [],
    }
    quality_analysis = quality_agent.analyze(round_id=0, context=context)
    quality_agent.write_analysis(quality_analysis, OUT_DIR / "08_binder_quality_analysis.json")
    print(f"  LLM used: {quality_analysis.llm_used}")
    print(f"  Assessment: {quality_analysis.overall_assessment[:200]}")

    # ─── Step 9: Hypothesis Generation ──────────────────────────────────
    print("\n[Step 9] HypothesisAgent: generating hypotheses...")
    hyp_agent = HypothesisAgent(llm=llm)
    context["quality_analysis"] = asdict(quality_analysis)
    hypotheses = hyp_agent.propose(context)
    write_json(OUT_DIR / "09_hypotheses.json", asdict(hypotheses))
    print(f"  LLM used: {hypotheses.llm_used}")
    for h in hypotheses.hypotheses[:3]:
        print(f"    - {h.get('name')}: conf={h.get('confidence')}")

    # ─── Step 10: Next Round Proposal ───────────────────────────────────
    print("\n[Step 10] ActiveLearningPolicyAgent: next round...")
    policy_agent = ActiveLearningPolicyAgent()
    proposal = policy_agent.propose_next_boltzgen_params(
        evaluation, params, round_id=1,
        structural_summary=struct_eval,
        hypotheses=hypotheses.hypotheses,
        quality_analysis=asdict(quality_analysis),
    )
    policy_agent.write_proposal(proposal, OUT_DIR / "10_next_round_proposal.json")
    for r in proposal.rationale:
        print(f"    - {r}")

    # ─── Final Report ───────────────────────────────────────────────────
    report = {
        "pipeline": "IL-17A Full Pipeline Test v2 (Local Upload)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "IL-17A (chains A+B)",
        "binder_length": 70,
        "seed": 0,
        "task_flag": task_flag,
        "taiji_job_id": submission.taiji_job_id,
        "submission_returncode": submission.returncode,
        "final_state": final_snapshot.state if final_snapshot else "unknown",
        "final_success": final_snapshot.is_success if final_snapshot else False,
        "candidates_collected": len(ingested.candidates),
        "evaluation_success": evaluation.success_count,
        "evaluation_failure": evaluation.failure_count,
        "structures_analyzed": struct_eval.total_structures,
        "llm_quality_used": quality_analysis.llm_used,
        "hypotheses_count": len(hypotheses.hypotheses),
        "next_round_rationale": proposal.rationale,
    }
    write_json(OUT_DIR / "pipeline_report.json", report)
    print("\n" + "=" * 70)
    print(f"Pipeline complete! Report: {OUT_DIR / 'pipeline_report.json'}")
    print(f"Taiji instance URL: http://taiji.woa.com/#/project-list/jizhi/task-inst-detail/{submission.taiji_job_id}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
