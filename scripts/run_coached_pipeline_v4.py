#!/usr/bin/env python3
"""Full Pipeline Re-analysis with Diagnostic Coaching.

This script:
1. Re-ingests results from the completed v3 Taiji job (which produced real BoltzGen outputs)
2. Runs the full evaluation pipeline with all agents
3. Uses the new DiagnosticCoachAgent (LLM-powered) to interpret results
4. Uses the new InputConfigurationAgent to derive next-round configuration
5. Submits the next round to Taiji with corrected parameters
6. Monitors and iterates

Acts as a "coaching loop" - the human expert's thinking is encapsulated in LLM agents.
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
    DiagnosticCoachAgent,
    InputConfigurationAgent,
)
from binderloop.agents.hypothesis_agent import HypothesisAgent
from binderloop.agents.binder_quality_analysis_agent import BinderQualityAnalysisAgent
from binderloop.config import load_config
from binderloop.llm import OpenAICompatibleClient
from binderloop.models.base import DesignJob
from binderloop.secrets import SecretStore

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/coached_pipeline_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Previous run output (already completed on Taiji)
PREV_BOLTZGEN_OUTPUT = ROOT / "outputs/il17a_full_pipeline_test_v3/round0_len70_seed0/taiji_project_package/outputs/boltzgen_output"
BOLTZGEN_CHECKPOINT_DIR = Path("/aceph/daweihuang/program/boltzgen/checkpoints")
BOLTZGEN_CACHE_DIR = Path("/aceph/daweihuang/program/boltzgen/cache")
BOLTZGEN_MOLDIR = BOLTZGEN_CACHE_DIR / "datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def section(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def main() -> int:
    print("=" * 70)
    print("  COACHED PIPELINE v4: Full Re-analysis + Next Round")
    print("  Target: IL-17A homodimer | Coach: LLM-powered DiagnosticCoachAgent")
    print("=" * 70)

    cfg = load_config(ROOT / "configs/il17a_full_pipeline_test.yaml")
    secrets = SecretStore.from_json(ROOT / "configs/llm_endpoints.local.json")
    ceph_secret = secrets.ceph_secret()
    llm = OpenAICompatibleClient.from_json(ROOT / "configs/llm_endpoints.local.json")

    print(f"\n  LLM available: {llm.available() if llm else False}")
    print(f"  CEPH_SECRET: {'set' if ceph_secret else 'MISSING'}")
    print(f"  Previous output: {PREV_BOLTZGEN_OUTPUT}")
    print(f"  Output dir: {OUT_DIR}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: Re-ingest and analyze the completed round 0 results
    # ═══════════════════════════════════════════════════════════════════════
    section("PHASE 1: Re-ingesting Round 0 Results (from completed Taiji job)")

    # Step 1: Ingest BoltzGen output
    print("\n  [1.1] ResultIngestionAgent: Parsing BoltzGen output...")
    ingestor = ResultIngestionAgent()
    log_file = str(PREV_BOLTZGEN_OUTPUT.parent.parent / "logs/boltzgen_full.log")
    ingested = ingestor.ingest_boltzgen_output(str(PREV_BOLTZGEN_OUTPUT), log_file=log_file)
    ingestor.write_manifest(ingested, OUT_DIR / "round0/01_result_ingestion.json")
    print(f"    Metrics files: {len(ingested.metrics_files)}")
    print(f"    Candidates: {len(ingested.candidates)}")
    print(f"    Final design files: {len(ingested.final_design_files)}")

    # Step 2: Evaluate candidates
    print("\n  [1.2] EvaluationAgent: Scoring and classifying candidates...")
    evaluator = EvaluationAgent()
    if ingested.candidates:
        evaluation = evaluator.evaluate_candidates(ingested.candidates)
    else:
        from binderloop.agents.evaluation_agent import EvaluationSummary
        evaluation = EvaluationSummary(0, 0, 0, {}, [], [], ["No candidates collected."])
    evaluator.write_summary(evaluation, OUT_DIR / "round0/02_evaluation_summary.json")
    evaluator.write_scores_csv(evaluation, OUT_DIR / "round0/02_scores.csv")
    print(f"    Total={evaluation.total_candidates} Pass={evaluation.success_count} Fail={evaluation.failure_count}")
    print(f"    Tags: {evaluation.tag_counts}")
    for obs in evaluation.observations:
        print(f"      • {obs}")

    # Step 3: Structure evaluation
    print("\n  [1.3] StructureEvaluationAgent: Analyzing CIF structures...")
    struct_agent = StructureEvaluationAgent()
    structure_files = [p for p in ingested.final_design_files if p.lower().endswith((".pdb", ".cif"))]
    struct_eval = struct_agent.analyze_structures(
        structure_files,
        binder_chain="D",
        target_chains=["A", "B"],
        hotspots=["A:67", "A:89", "B:49"],
    )
    struct_agent.write_batch(struct_eval, OUT_DIR / "round0/03_structure_evaluation.json")
    print(f"    Structures analyzed: {struct_eval.total_structures}")
    print(f"    Reliable fraction: {struct_eval.reliable_seed_fraction:.2f}")
    print(f"    Tags: {struct_eval.aggregate_tags}")
    for obs in struct_eval.observations:
        print(f"      • {obs}")

    # Step 4: Quality analysis (LLM)
    print("\n  [1.4] BinderQualityAnalysisAgent: Deep quality analysis...")
    quality_agent = BinderQualityAnalysisAgent(llm=llm)
    context = {
        "round_id": 0,
        "evaluation": asdict(evaluation),
        "structural_analysis": asdict(struct_eval),
        "memory": {},
        "messages": [],
    }
    quality = quality_agent.analyze(round_id=0, context=context)
    quality_agent.write_analysis(quality, OUT_DIR / "round0/04_quality_analysis.json")
    print(f"    LLM used: {quality.llm_used}")
    print(f"    Assessment: {quality.overall_assessment[:200]}")
    print(f"    High-quality modules: {len(quality.high_quality_modules)}")
    print(f"    Low-quality modules: {len(quality.low_quality_modules)}")

    # Step 5: Hypothesis generation
    print("\n  [1.5] HypothesisAgent: Generating failure hypotheses...")
    hyp_agent = HypothesisAgent(llm=llm)
    context["quality_analysis"] = asdict(quality)
    hypotheses = hyp_agent.propose(context)
    write_json(OUT_DIR / "round0/05_hypotheses.json", asdict(hypotheses))
    print(f"    LLM used: {hypotheses.llm_used}")
    print(f"    Hypotheses: {len(hypotheses.hypotheses)}")
    for h in hypotheses.hypotheses[:5]:
        print(f"      • [{h.get('confidence', '?')}] {h.get('name')}")

    # Step 6: Active learning policy
    print("\n  [1.6] ActiveLearningPolicyAgent: Proposing parameter updates...")
    policy = ActiveLearningPolicyAgent()
    base_params = {"hotspot_weight": 1.0, "num_designs": 10}
    base_params.update(cfg.search_space.boltzgen or {})
    proposal = policy.propose_next_boltzgen_params(
        evaluation, base_params, round_id=1,
        structural_summary=struct_eval,
        hypotheses=hypotheses.hypotheses,
        quality_analysis=asdict(quality),
    )
    policy.write_proposal(proposal, OUT_DIR / "round0/06_policy_proposal.json")
    print(f"    Rationale:")
    for r in proposal.rationale:
        print(f"      • {r}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Diagnostic Coaching (NEW - LLM reasoning about what to do)
    # ═══════════════════════════════════════════════════════════════════════
    section("PHASE 2: DiagnosticCoachAgent - Expert Reasoning")

    # Build metrics summary from the raw data
    metrics_summary = _build_metrics_summary(ingested.candidates)

    coach = DiagnosticCoachAgent(llm=llm)
    diagnostic = coach.diagnose(
        round_id=0,
        metrics_summary=metrics_summary,
        evaluation_summary=asdict(evaluation),
        structural_analysis=asdict(struct_eval),
        config={
            "binder_lengths": cfg.search_space.binder_lengths,
            "hotspots": cfg.target.hotspots,
            "num_designs": cfg.search_space.num_designs_per_round,
            "target": {"structure": cfg.target.structure_path, "chain_id": cfg.target.chain_id},
        },
    )
    coach.write_report(diagnostic, OUT_DIR / "round0/07_diagnostic_report.json")
    print(f"\n  LLM used: {diagnostic.llm_used}")
    print(f"  Diagnosis: {diagnostic.status_diagnosis[:300]}")
    print(f"\n  Root causes ({len(diagnostic.root_causes)}):")
    for rc in diagnostic.root_causes[:5]:
        print(f"    [{rc.get('confidence', '?')}] {rc.get('cause')}")
    print(f"\n  Pipeline health: {json.dumps(diagnostic.pipeline_health, indent=4)}")
    print(f"\n  Corrective actions ({len(diagnostic.corrective_actions)}):")
    for ca in diagnostic.corrective_actions[:5]:
        print(f"    [{ca.get('priority', '?')}] {ca.get('action', '')[:100]}")
        if ca.get("parameter_changes"):
            print(f"         params: {ca['parameter_changes']}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: InputConfigurationAgent - Derive Next Round Config
    # ═══════════════════════════════════════════════════════════════════════
    section("PHASE 3: InputConfigurationAgent - Next Round Configuration")

    config_agent = InputConfigurationAgent(llm=llm)
    next_config = config_agent.configure_next_round(
        target_name="IL-17A homodimer",
        current_config=base_params,
        diagnostic_report=asdict(diagnostic),
        evaluation_summary=asdict(evaluation),
        round_id=1,
    )
    config_agent.write_config(next_config, OUT_DIR / "round1/00_input_configuration.json")
    print(f"\n  LLM used: {next_config.llm_used}")
    print(f"  Reasoning: {next_config.reasoning[:300]}")
    print(f"\n  Recommended config:")
    for key, value in next_config.recommended_config.items():
        print(f"    {key}: {value}")
    print(f"\n  Parameter rationale:")
    for pr in next_config.parameter_rationale[:5]:
        print(f"    [{pr.get('confidence', '?')}] {pr.get('parameter')}: {pr.get('reason', '')[:80]}")
    print(f"\n  Iteration strategy: {json.dumps(next_config.iteration_strategy, indent=4)}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: Submit Next Round to Taiji
    # ═══════════════════════════════════════════════════════════════════════
    section("PHASE 4: Preparing & Submitting Round 1")

    # Merge configurations: policy proposal + diagnostic corrections + input config agent
    round1_params = dict(base_params)
    round1_params.update(proposal.params_update)
    round1_params.update(next_config.recommended_config)

    # Use longer binder length based on coaching advice
    binder_lengths = round1_params.pop("binder_lengths", [90, 110])
    if isinstance(binder_lengths, list) and len(binder_lengths) > 0:
        primary_length = binder_lengths[0]
    else:
        primary_length = 90

    round1_params.update({
        "task_id": f"il17a_coached_r1_{int(time.time())}",
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
        "cache": str(BOLTZGEN_CACHE_DIR),
        "moldir": str(BOLTZGEN_MOLDIR),
        "design_checkpoints": [
            str(BOLTZGEN_CHECKPOINT_DIR / "boltzgen1_diverse.ckpt"),
            str(BOLTZGEN_CHECKPOINT_DIR / "boltzgen1_adherence.ckpt"),
        ],
        "inverse_fold_checkpoint": str(BOLTZGEN_CHECKPOINT_DIR / "boltzgen1_ifold.ckpt"),
    })

    # Ensure key parameters are set
    round1_params.setdefault("num_designs", 15)
    round1_params.setdefault("budget", 10)
    round1_params.setdefault("protocol", "protein-anything")
    round1_params.setdefault("diffusion_batch_size", 1)
    round1_params.setdefault("inverse_fold_num_sequences", 2)
    round1_params.setdefault("run_filtering", True)
    round1_params.setdefault("keep_unfiltered_for_failure_analysis", True)
    round1_params.setdefault("binder_chain", "D")

    print(f"\n  Primary binder length: {primary_length}")
    print(f"  num_designs: {round1_params.get('num_designs')}")
    print(f"  hotspot_weight: {round1_params.get('hotspot_weight')}")
    print(f"  run_filtering: {round1_params.get('run_filtering')}")

    # Create design job
    job = DesignJob(
        job_id=round1_params["task_id"],
        target_structure=str(ROOT / "examples/bg_example/IL-17A.cif"),
        chain_id="A",
        hotspots=["A:67", "A:89", "B:49"],
        binder_length=primary_length,
        seed=42,
        params=round1_params,
        output_dir=str(OUT_DIR / f"round1/len{primary_length}_seed42"),
    )

    # Create run spec
    spec_agent = DesignSpecAgent(ROOT.parent / "boltzgen")
    run_spec = spec_agent.create_boltzgen_run_spec(job, params=round1_params)
    print(f"  Package: {run_spec.package_dir}")

    # Save parameter plan
    param_agent = DesignParameterAgent()
    param_agent.write_parameter_plan(round1_params, OUT_DIR / "round1/01_design_parameter_plan.yaml")

    # Build Taiji config
    task_flag = f"il17a_coached_r1_{int(time.time())}"
    package_dir = run_spec.package_dir
    run_script_rel = "scripts/run_boltzgen_full.sh"

    start_cmd = f"""bash -lc '
set -euo pipefail
mkdir -p /aceph/daweihuang
if ! mountpoint -q /aceph/daweihuang 2>/dev/null; then
  mount -t ceph 11.18.83.17:6789,11.18.83.31:6789,11.18.83.32:6789:/fandiwu/buddy1/daweihuang /aceph/daweihuang -o name=fandiwubuddy1,secret="${{CEPH_SECRET}}"
fi
cd {package_dir}
echo "[HARNESS] workspace=$(pwd)"
echo "[HARNESS] coached_round=1 length={primary_length} hotspot_weight={round1_params.get("hotspot_weight", 1.0)}"
bash {run_script_rel}
'"""

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
            "NUM_DESIGNS": str(round1_params.get("num_designs", 15)),
            "TAIJI_TIMEOUT": "5400",
        },
    }

    config_path = OUT_DIR / "round1/02_taiji_simple_config.json"
    config_path.write_text(json.dumps(taiji_config, ensure_ascii=False, indent=4), encoding="utf-8")

    # Submit
    import subprocess
    submit_cmd = ["taiji_client", "start", "-scfg", str(config_path)]
    print(f"\n  Submitting: {' '.join(submit_cmd)}")
    proc = subprocess.run(submit_cmd, text=True, capture_output=True, check=False)
    print(f"  returncode: {proc.returncode}")
    print(f"  stdout: {proc.stdout.strip()[:500]}")
    if proc.stderr:
        print(f"  stderr: {proc.stderr.strip()[:300]}")

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
        "round": 1,
        "binder_length": primary_length,
        "coached_params": {k: v for k, v in round1_params.items() if k not in ("design_checkpoints", "inverse_fold_checkpoint", "cache", "moldir", "target_include", "target_binding_types")},
    }
    write_json(OUT_DIR / "round1/03_submission_record.json", submission_record)
    print(f"\n  Taiji job ID: {taiji_job_id}")
    if taiji_job_id:
        print(f"  URL: http://taiji.woa.com/#/project-list/jizhi/task-inst-detail/{taiji_job_id}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: Monitor (with longer timeout)
    # ═══════════════════════════════════════════════════════════════════════
    section("PHASE 5: Monitoring Round 1 (extended timeout)")

    monitor = RunMonitorAgent()
    max_checks = 120  # 120 * 30s = 60 minutes max
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
        progress = f"[{ts}] #{check_num}/{max_checks}: state={state}"

        if snapshot.is_terminal:
            print(f"  {progress} *** TERMINAL ***")
        elif check_num % 5 == 0:
            print(f"  {progress}")

        if snapshot.log_tail and (check_num % 10 == 0 or snapshot.is_terminal):
            tail = snapshot.log_tail.strip()[-200:]
            print(f"    Log: {tail}")

        final_snapshot = snapshot
        if snapshot.is_terminal:
            break
        time.sleep(check_interval)

    if final_snapshot:
        monitor.write_snapshot(final_snapshot, OUT_DIR / "round1/04_monitor_final.json")
        print(f"\n  Final: state={final_snapshot.state} success={final_snapshot.is_success}")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 6: Collect Round 1 Results (if available)
    # ═══════════════════════════════════════════════════════════════════════
    section("PHASE 6: Round 1 Result Collection & Coaching Verdict")

    round1_output = Path(run_spec.output_dir) if hasattr(run_spec, 'output_dir') else OUT_DIR / f"round1/len{primary_length}_seed42/taiji_project_package/outputs/boltzgen_output"
    ingested_r1 = ingestor.ingest_boltzgen_output(str(round1_output), log_file=str(round1_output.parent.parent / "logs/boltzgen_full.log") if round1_output.exists() else None)
    ingestor.write_manifest(ingested_r1, OUT_DIR / "round1/05_result_ingestion.json")
    print(f"  Candidates collected: {len(ingested_r1.candidates)}")

    if ingested_r1.candidates:
        eval_r1 = evaluator.evaluate_candidates(ingested_r1.candidates)
    else:
        from binderloop.agents.evaluation_agent import EvaluationSummary
        eval_r1 = EvaluationSummary(0, 0, 0, {}, [], [], ["Round 1: No candidates yet (job may still be running)."])
    evaluator.write_summary(eval_r1, OUT_DIR / "round1/06_evaluation_summary.json")
    print(f"  Total={eval_r1.total_candidates} Pass={eval_r1.success_count}")

    # Run coach diagnostic on round 1
    diagnostic_r1 = coach.diagnose(
        round_id=1,
        monitor_snapshot=asdict(final_snapshot) if final_snapshot else None,
        metrics_summary=_build_metrics_summary(ingested_r1.candidates) if ingested_r1.candidates else None,
        evaluation_summary=asdict(eval_r1),
        config=round1_params,
    )
    coach.write_report(diagnostic_r1, OUT_DIR / "round1/07_diagnostic_report.json")
    print(f"\n  Coach Verdict: {diagnostic_r1.status_diagnosis[:300]}")
    print(f"  Pipeline Health: {json.dumps(diagnostic_r1.pipeline_health, indent=4)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    section("PIPELINE SUMMARY")
    summary = {
        "pipeline": "Coached Pipeline v4",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "IL-17A homodimer (A+B)",
        "rounds_completed": 1 if (final_snapshot and final_snapshot.is_terminal) else 0,
        "round0": {
            "source": "previous_v3_taiji_job",
            "candidates": len(ingested.candidates),
            "pass": evaluation.success_count,
            "fail": evaluation.failure_count,
            "structures": struct_eval.total_structures,
            "coach_diagnosis": diagnostic.status_diagnosis[:200],
        },
        "round1": {
            "task_flag": task_flag,
            "taiji_job_id": taiji_job_id,
            "binder_length": primary_length,
            "final_state": final_snapshot.state if final_snapshot else "not_monitored",
            "candidates": len(ingested_r1.candidates),
            "pass": eval_r1.success_count,
            "coach_diagnosis": diagnostic_r1.status_diagnosis[:200],
        },
        "agents_used": [
            "ResultIngestionAgent", "EvaluationAgent", "StructureEvaluationAgent",
            "BinderQualityAnalysisAgent", "HypothesisAgent", "ActiveLearningPolicyAgent",
            "DiagnosticCoachAgent (NEW)", "InputConfigurationAgent (NEW)",
            "DesignParameterAgent", "DesignSpecAgent", "RunMonitorAgent",
        ],
        "llm_calls": {
            "quality_analysis": quality.llm_used,
            "hypotheses": hypotheses.llm_used,
            "diagnostic_r0": diagnostic.llm_used,
            "input_config": next_config.llm_used,
            "diagnostic_r1": diagnostic_r1.llm_used,
        },
    }
    write_json(OUT_DIR / "coached_pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"\n  Full output: {OUT_DIR}")
    print("=" * 70)
    return 0


def _build_metrics_summary(candidates: list) -> dict:
    """Build a compact metrics summary from candidate list for the coach."""
    if not candidates:
        return {"count": 0}

    iptm_values = []
    plddt_values = []
    rmsd_values = []
    hbonds_values = []

    for c in candidates:
        iptm_values.append(float(c.get("design_to_target_iptm") or c.get("iptm") or 0))
        plddt_values.append(float(c.get("design_ptm") or c.get("plddt") or 0))
        rmsd_values.append(float(c.get("filter_rmsd") or 0))
        hbonds_values.append(float(c.get("plip_hbonds_refolded") or 0))

    def stats(vals):
        if not vals:
            return {"min": 0, "max": 0, "mean": 0}
        return {
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "mean": round(sum(vals) / len(vals), 4),
        }

    return {
        "count": len(candidates),
        "iptm": stats(iptm_values),
        "plddt": stats(plddt_values),
        "filter_rmsd": stats(rmsd_values),
        "plip_hbonds": stats(hbonds_values),
        "any_iptm_above_0.3": any(v > 0.3 for v in iptm_values),
        "any_iptm_above_0.4": any(v > 0.4 for v in iptm_values),
        "plddt_above_0.7_fraction": sum(1 for v in plddt_values if v > 0.7) / max(1, len(plddt_values)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
