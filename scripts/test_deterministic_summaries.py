#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binderloop.analysis.scientific_summary import build_scientific_summary
from binderloop.execution_error_summary import build_execution_error_summary
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator

def main():
    rows=[{"design_to_target_iptm":"0.4","design_ptm":"bad","pass_filters":True},{"design_to_target_iptm":"nan"},{}]
    summary=build_scientific_summary(rows, evaluation={"success_count":1,"failure_count":2,"candidate_filtering":{"analysis_scope":"filtered_candidates","input_candidate_count":5,"analysis_candidate_count":3}})
    assert summary["schema_version"] == "1.0"
    assert summary["scope"]["name"] == "filtered_candidates"
    assert summary["metrics"]["iptm"] == {"valid_count":1,"missing_count":1,"invalid_count":1,"min":0.4,"max":0.4,"mean":0.4,"best":0.4}
    assert "mean" not in summary["metrics"]["design_ptm"]
    assert summary["gates"]["harness_compute_gate"]["pass_count"] == 1
    errors=build_execution_error_summary([{"status":"failed","attempts":2,"job":{"job_id":"j"},"error":"failed at /secret/run/file.txt: " + "x"*700}])
    assert "/secret" not in errors["failed_jobs"][0]["error"]
    assert len(errors["failed_jobs"][0]["error"]) <= 500
    monitor=BinderDesignOrchestrator._build_monitor_snapshot([{"status":"timeout","attempts":1,"error":"see /tmp/private.log"}])
    assert monitor["is_success"] is False and "/tmp" not in monitor["failure_hints"][0]
    metrics=BinderDesignOrchestrator._build_metrics_summary(rows)
    assert metrics["iptm"]["mean"] == 0.4 and "plddt" not in metrics
    print("OK: deterministic scientific and execution summaries")
if __name__ == "__main__": main()
