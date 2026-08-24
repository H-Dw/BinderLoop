#!/usr/bin/env python3
"""IL-17A Full Pipeline Test: Generate spec, submit to Taiji, monitor, evaluate.

This script runs a single BoltzGen job (length 70, seed 0) to Taiji as the first
validation step, then collects results for evaluation.
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
OUT_DIR = ROOT / "outputs/il17a_full_pipeline_test"
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
    print("IL-17A Full Pipeline Test")
    print("=" * 70)

    # Load config
    cfg = load_config(ROOT / "configs/il17a_full_pipeline_test.yaml")

    # Load secrets for CEPH_SECRET
    secrets = SecretStore.from_json(ROOT / "configs/llm_endpoints.local.json")
    ceph_secret = secrets.ceph_secret()

    # Load LLM client
    llm = OpenAICompatibleClient.from_json(ROOT / "configs/llm_endpoints.local.json")

    # ─── Step 1: Design Parameter Agent ─────────────────────────────────
    print("\n[Step 1] DesignParameterAgent: choosing parameters...")
    param_agent = DesignParameterAgent()
    params = param_agent.choose_boltzgen_parameters(cfg)
    # Override with pipeline-specific settings
    params.update({
        "task_id": "il17a_len70_seed0",
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
    print(f"  Written: {OUT_DIR / '01_design_parameter_plan.yaml'}")
    print(f"  Protocol: {params.get('protocol')}")
    print(f"  Num designs: {params.get('num_designs')}")

    # ─── Step 2: Design Spec Agent ──────────────────────────────────────
    print("\n[Step 2] DesignSpecAgent: creating BoltzGen run spec...")
    # We pick length=70 seed=0 as our first submission
    job = DesignJob(
        job_id="il17a_len70_seed0",
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
    print(f"  Run script: {run_spec.run_script_path}")
    print(f"  Output dir: {run_spec.output_dir}")
    print(f"  Command: {run_spec.command_string[:120]}...")

    # ─── Step 3: Taiji Execution Agent ──────────────────────────────────
    print("\n[Step 3] TaijiExecutionAgent: creating Taiji config and submitting...")
    task_flag = f"il17a_binder_len70_{int(time.time())}"
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
    }
    if ceph_secret:
        taiji_options["envs"] = {"CEPH_SECRET": ceph_secret}

    taiji_agent = TaijiExecutionAgent(dry_run=False)
    submit_spec = taiji_agent.create_boltzgen_taiji_spec(
        run_spec,
        template_json=None,
        output_json=OUT_DIR / "02_taiji_simple_config.json",
        task_flag=task_flag,
        taiji_options=taiji_options,
    )
    print(f"  Task flag: {task_flag}")
    print(f"  Submit command: {submit_spec.submit_command}")

    # Submit!
    submission = taiji_agent.submit(submit_spec)
    write_json(OUT_DIR / "03_taiji_submission_record.json", asdict(submission))
    print(f"  Submitted! dry_run={submission.dry_run}, returncode={submission.returncode}")
    print(f"  Taiji job ID: {submission.taiji_job_id}")
    if submission.stderr:
        print(f"  stderr (tail): {submission.stderr[-300:]}")
    if submission.stdout:
        print(f"  stdout (tail): {submission.stdout[-300:]}")

    # ─── Step 4: Run Monitor Agent ──────────────────────────────────────
    print("\n[Step 4] RunMonitorAgent: monitoring job status...")
    monitor = RunMonitorAgent()
    max_checks = 40  # Up to ~20 minutes of monitoring
    check_interval = 30  # seconds
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
        print(f"  Check #{check_num}: state={state}, terminal={snapshot.is_terminal}, success={snapshot.is_success}")
        if snapshot.failure_hints:
            print(f"    Hints: {snapshot.failure_hints}")
        if snapshot.log_tail:
            # Print last 200 chars of log
            print(f"    Log tail: ...{snapshot.log_tail[-200:]}")

        final_snapshot = snapshot

        if snapshot.is_terminal:
            break

        if not snapshot.needs_followup:
            break

        print(f"    Waiting {check_interval}s before next check...")
        time.sleep(check_interval)

    if final_snapshot:
        monitor.write_snapshot(final_snapshot, OUT_DIR / "04_run_monitor_final.json")
        print(f"\n  Final state: {final_snapshot.state}")
        print(f"  Is success: {final_snapshot.is_success}")
        print(f"  Missing outputs: {final_snapshot.missing_outputs}")

    # ─── Step 5: Result Ingestion ───────────────────────────────────────
    print("\n[Step 5] ResultIngestionAgent: collecting results...")
    ingestor = ResultIngestionAgent()
    ingested = ingestor.ingest_boltzgen_output(run_spec.output_dir, log_file=run_spec.log_file)
    ingestor.write_manifest(ingested, OUT_DIR / "05_result_ingestion.json")
    print(f"  Metrics files found: {len(ingested.metrics_files)}")
    print(f"  Candidates collected: {len(ingested.candidates)}")
    print(f"  Run-level issues: {ingested.run_level_issues}")

    # ─── Step 6: Evaluation Agent ───────────────────────────────────────
    print("\n[Step 6] EvaluationAgent: scoring candidates...")
    evaluator = EvaluationAgent()
    if ingested.candidates:
        evaluation = evaluator.evaluate_candidates(ingested.candidates)
    else:
        # If no real results yet, note it
        from binderloop.agents.evaluation_agent import EvaluationSummary
        evaluation = EvaluationSummary(0, 0, 0, {}, [], [], ["No candidates collected; job may still be running or output paths differ."])
        print("  WARNING: No candidates found. Job may still be running.")

    evaluator.write_summary(evaluation, OUT_DIR / "06_evaluation_summary.json")
    evaluator.write_scores_csv(evaluation, OUT_DIR / "06_scores_preview.csv")
    print(f"  Total candidates: {evaluation.total_candidates}")
    print(f"  Success: {evaluation.success_count}, Failure: {evaluation.failure_count}")
    print(f"  Tag counts: {evaluation.tag_counts}")
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
    print(f"  Structures analyzed: {struct_eval.total_structures}")
    print(f"  Reliable fraction: {struct_eval.reliable_seed_fraction:.2f}")
    print(f"  Aggregate tags: {struct_eval.aggregate_tags}")

    # ─── Step 8: Binder Quality Analysis (LLM) ─────────────────────────
    print("\n[Step 8] BinderQualityAnalysisAgent: LLM-assisted quality analysis...")
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
    print(f"  Overall: {quality_analysis.overall_assessment[:200]}")
    print(f"  High-quality modules: {len(quality_analysis.high_quality_modules)}")
    print(f"  Low-quality modules: {len(quality_analysis.low_quality_modules)}")

    # ─── Step 9: Hypothesis Generation ──────────────────────────────────
    print("\n[Step 9] HypothesisAgent: generating optimization hypotheses...")
    hyp_agent = HypothesisAgent(llm=llm)
    context["quality_analysis"] = asdict(quality_analysis)
    hypotheses = hyp_agent.propose(context)
    write_json(OUT_DIR / "09_hypotheses.json", asdict(hypotheses))
    print(f"  LLM used: {hypotheses.llm_used}")
    print(f"  Hypotheses count: {len(hypotheses.hypotheses)}")
    for h in hypotheses.hypotheses[:3]:
        print(f"    - {h.get('name')}: confidence={h.get('confidence')}")

    # ─── Step 10: Next Round Proposal ───────────────────────────────────
    print("\n[Step 10] ActiveLearningPolicyAgent: proposing next round...")
    policy_agent = ActiveLearningPolicyAgent()
    proposal = policy_agent.propose_next_boltzgen_params(
        evaluation, params, round_id=1,
        structural_summary=struct_eval,
        hypotheses=hypotheses.hypotheses,
        quality_analysis=asdict(quality_analysis),
    )
    policy_agent.write_proposal(proposal, OUT_DIR / "10_next_round_proposal.json")
    print(f"  Next round parameters proposed:")
    for r in proposal.rationale:
        print(f"    - {r}")

    # ─── Final Report ───────────────────────────────────────────────────
    report = {
        "pipeline": "IL-17A Full Pipeline Test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "IL-17A (chains A+B)",
        "binder_length": 70,
        "seed": 0,
        "task_flag": task_flag,
        "taiji_job_id": submission.taiji_job_id,
        "submission_returncode": submission.returncode,
        "final_state": final_snapshot.state if final_snapshot else "unknown",
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
    print("Pipeline complete! Report: outputs/il17a_full_pipeline_test/pipeline_report.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
