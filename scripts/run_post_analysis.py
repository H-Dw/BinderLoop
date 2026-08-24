#!/usr/bin/env python3
"""Post-run analysis: collect Round 1 results and produce final coaching report.

Run this AFTER the coached_pipeline_v4 monitoring completes (or after Taiji job ends).
It re-ingests the output, runs all agents with the fixed JSON parsing, and
compares Round 0 vs Round 1 to assess whether the coaching improved results.
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents import (
    ResultIngestionAgent,
    EvaluationAgent,
    StructureEvaluationAgent,
    ActiveLearningPolicyAgent,
    DiagnosticCoachAgent,
    InputConfigurationAgent,
)
from binderloop.agents.hypothesis_agent import HypothesisAgent
from binderloop.agents.binder_quality_analysis_agent import BinderQualityAnalysisAgent
from binderloop.config import load_config
from binderloop.llm import OpenAICompatibleClient

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/coached_pipeline_v4"


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def section(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def build_metrics_summary(candidates: list) -> dict:
    if not candidates:
        return {"count": 0}
    iptm_vals = [float(c.get("design_to_target_iptm") or c.get("iptm") or 0) for c in candidates]
    plddt_vals = [float(c.get("design_ptm") or c.get("plddt") or 0) for c in candidates]
    rmsd_vals = [float(c.get("filter_rmsd") or 0) for c in candidates]
    hbonds_vals = [float(c.get("plip_hbonds_refolded") or 0) for c in candidates]
    def stats(v):
        return {"min": round(min(v), 4), "max": round(max(v), 4), "mean": round(sum(v)/len(v), 4)} if v else {}
    return {
        "count": len(candidates),
        "iptm": stats(iptm_vals),
        "plddt": stats(plddt_vals),
        "filter_rmsd": stats(rmsd_vals),
        "plip_hbonds": stats(hbonds_vals),
        "any_iptm_above_0.3": any(v > 0.3 for v in iptm_vals),
        "any_iptm_above_0.4": any(v > 0.4 for v in iptm_vals),
        "plddt_above_0.7_fraction": sum(1 for v in plddt_vals if v > 0.7) / max(1, len(plddt_vals)),
    }


def main() -> int:
    print("=" * 70)
    print("  POST-RUN ANALYSIS: Round 0 vs Round 1 Comparison")
    print("=" * 70)

    cfg = load_config(ROOT / "configs/il17a_full_pipeline_test.yaml")
    llm = OpenAICompatibleClient.from_json(ROOT / "configs/llm_endpoints.local.json")
    print(f"  LLM available: {llm.available() if llm else False}")

    # Locate Round 1 outputs
    r1_output = OUT_DIR / "round1/len90_seed42/taiji_project_package/outputs/boltzgen_output"
    if not r1_output.exists():
        print(f"  ERROR: Round 1 output not found at {r1_output}")
        return 1

    # ─── Re-ingest Round 1 ─────────────────────────────────────────────
    section("Round 1 Ingestion & Evaluation")
    ingestor = ResultIngestionAgent()
    log_file = str(r1_output.parent.parent / "logs/boltzgen_full.log")
    ingested = ingestor.ingest_boltzgen_output(str(r1_output), log_file=log_file)
    ingestor.write_manifest(ingested, OUT_DIR / "round1_analysis/01_result_ingestion.json")
    print(f"  Metrics files: {len(ingested.metrics_files)}")
    print(f"  Candidates: {len(ingested.candidates)}")
    print(f"  Final design files: {len(ingested.final_design_files)}")

    evaluator = EvaluationAgent()
    if ingested.candidates:
        evaluation = evaluator.evaluate_candidates(ingested.candidates)
    else:
        from binderloop.agents.evaluation_agent import EvaluationSummary
        evaluation = EvaluationSummary(0, 0, 0, {}, [], [], ["Round 1: No candidates."])
    evaluator.write_summary(evaluation, OUT_DIR / "round1_analysis/02_evaluation_summary.json")
    evaluator.write_scores_csv(evaluation, OUT_DIR / "round1_analysis/02_scores.csv")
    print(f"  Total={evaluation.total_candidates} Pass={evaluation.success_count} Fail={evaluation.failure_count}")
    print(f"  Tags: {evaluation.tag_counts}")
    for obs in evaluation.observations:
        print(f"    • {obs}")

    # ─── Structure Evaluation ──────────────────────────────────────────
    section("Round 1 Structure Analysis")
    struct_agent = StructureEvaluationAgent()
    structure_files = [p for p in ingested.final_design_files if p.lower().endswith((".pdb", ".cif"))]
    struct_eval = struct_agent.analyze_structures(
        structure_files, binder_chain="D", target_chains=["A", "B"], hotspots=["A:67", "A:89", "B:49"]
    )
    struct_agent.write_batch(struct_eval, OUT_DIR / "round1_analysis/03_structure_evaluation.json")
    print(f"  Structures: {struct_eval.total_structures}, Reliable: {struct_eval.reliable_seed_fraction:.2f}")
    print(f"  Tags: {struct_eval.aggregate_tags}")

    # ─── Quality Analysis (LLM) ───────────────────────────────────────
    section("Round 1 Quality Analysis (LLM)")
    quality_agent = BinderQualityAnalysisAgent(llm=llm)
    context = {"round_id": 1, "evaluation": asdict(evaluation), "structural_analysis": asdict(struct_eval), "memory": {}, "messages": []}
    quality = quality_agent.analyze(round_id=1, context=context)
    quality_agent.write_analysis(quality, OUT_DIR / "round1_analysis/04_quality_analysis.json")
    print(f"  LLM used: {quality.llm_used}")
    print(f"  Assessment: {quality.overall_assessment[:200]}")

    # ─── Hypotheses ────────────────────────────────────────────────────
    section("Round 1 Hypotheses")
    hyp_agent = HypothesisAgent(llm=llm)
    context["quality_analysis"] = asdict(quality)
    hypotheses = hyp_agent.propose(context)
    write_json(OUT_DIR / "round1_analysis/05_hypotheses.json", asdict(hypotheses))
    print(f"  LLM used: {hypotheses.llm_used}, Count={len(hypotheses.hypotheses)}")
    for h in hypotheses.hypotheses[:5]:
        print(f"    [{h.get('confidence')}] {h.get('name')}")

    # ─── Diagnostic Coach ─────────────────────────────────────────────
    section("Round 1 Diagnostic Coaching")
    coach = DiagnosticCoachAgent(llm=llm)
    metrics_summary = build_metrics_summary(ingested.candidates)
    diagnostic = coach.diagnose(
        round_id=1,
        metrics_summary=metrics_summary,
        evaluation_summary=asdict(evaluation),
        structural_analysis=asdict(struct_eval),
        config={"binder_lengths": [90], "hotspots": cfg.target.hotspots, "num_designs": 30},
    )
    coach.write_report(diagnostic, OUT_DIR / "round1_analysis/06_diagnostic_report.json")
    print(f"  LLM used: {diagnostic.llm_used}")
    print(f"  Diagnosis: {diagnostic.status_diagnosis[:300]}")
    print(f"  Pipeline Health: {json.dumps(diagnostic.pipeline_health, indent=4)}")
    print(f"  Root Causes ({len(diagnostic.root_causes)}):")
    for rc in diagnostic.root_causes:
        print(f"    [{rc.get('confidence')}] {rc.get('cause')}")
    print(f"  Corrective Actions ({len(diagnostic.corrective_actions)}):")
    for ca in diagnostic.corrective_actions:
        print(f"    [{ca.get('priority')}] {ca.get('action', '')[:100]}")

    # ─── Round 0 vs Round 1 Comparison ────────────────────────────────
    section("COMPARISON: Round 0 → Round 1")
    r0_metrics = build_metrics_summary(
        ResultIngestionAgent().ingest_boltzgen_output(
            str(ROOT / "outputs/il17a_full_pipeline_test_v3/round0_len70_seed0/taiji_project_package/outputs/boltzgen_output")
        ).candidates
    )
    r1_metrics = metrics_summary

    print(f"  {'Metric':<25} {'Round 0 (len=70)':<20} {'Round 1 (len=90)':<20} {'Change'}")
    print(f"  {'─'*25} {'─'*20} {'─'*20} {'─'*10}")
    for key in ["iptm", "plddt", "filter_rmsd", "plip_hbonds"]:
        r0v = r0_metrics.get(key, {}).get("mean", 0)
        r1v = r1_metrics.get(key, {}).get("mean", 0)
        delta = r1v - r0v
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {key:<25} {r0v:<20.4f} {r1v:<20.4f} {arrow} {delta:+.4f}")

    print(f"\n  {'Key Indicator':<25} {'Round 0':<20} {'Round 1':<20}")
    print(f"  {'─'*25} {'─'*20} {'─'*20}")
    print(f"  {'Candidates':<25} {r0_metrics.get('count', 0):<20} {r1_metrics.get('count', 0):<20}")
    print(f"  {'Pass count':<25} {0:<20} {evaluation.success_count:<20}")
    print(f"  {'Any iptm>0.3':<25} {r0_metrics.get('any_iptm_above_0.3', False)!s:<20} {r1_metrics.get('any_iptm_above_0.3', False)!s:<20}")
    print(f"  {'Any iptm>0.4':<25} {r0_metrics.get('any_iptm_above_0.4', False)!s:<20} {r1_metrics.get('any_iptm_above_0.4', False)!s:<20}")
    print(f"  {'pLDDT>0.7 fraction':<25} {r0_metrics.get('plddt_above_0.7_fraction', 0):<20.2f} {r1_metrics.get('plddt_above_0.7_fraction', 0):<20.2f}")

    # ─── Final Summary ─────────────────────────────────────────────────
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "round0": {"binder_length": 70, "hotspot_weight": 1.0, "num_designs": 10, **r0_metrics},
        "round1": {"binder_length": 90, "hotspot_weight": 2.5, "num_designs": 30, **r1_metrics},
        "evaluation_r1": {
            "total": evaluation.total_candidates,
            "pass": evaluation.success_count,
            "fail": evaluation.failure_count,
            "tags": evaluation.tag_counts,
        },
        "diagnostic_r1": asdict(diagnostic),
        "coaching_effectiveness": {
            "iptm_improved": (r1_metrics.get("iptm", {}).get("mean", 0) > r0_metrics.get("iptm", {}).get("mean", 0)),
            "candidates_increased": (r1_metrics.get("count", 0) > r0_metrics.get("count", 0)),
            "pass_improved": evaluation.success_count > 0,
        },
    }
    write_json(OUT_DIR / "round0_vs_round1_comparison.json", summary)
    print(f"\n  Summary saved: {OUT_DIR}/round0_vs_round1_comparison.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
