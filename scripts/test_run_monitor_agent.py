#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.run_monitor_agent import OPTIONAL_EXPECTED_OUTPUTS, RunMonitorAgent


def main() -> None:
    agent = RunMonitorAgent(log_host_index=1, command_timeout_seconds=17)

    with patch("binderloop.agents.run_monitor_agent.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess([], 0, stdout="log output", stderr="")
        output = agent._run_taiji_logs(
            task_flag="task",
            instance_id="instance",
            tail=80,
            simple_config_path="/tmp/simple.json",
            config_path=None,
        )
        assert output == "log output"
        assert run.call_args.kwargs["input"] == "1\n"
        assert run.call_args.kwargs["timeout"] == 17
        assert "stdin" not in run.call_args.kwargs

    with patch("binderloop.agents.run_monitor_agent.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess([], 0, stdout='{"state":"END"}', stderr="")
        detail = agent._run_taiji_detail(
            task_flag="task",
            instance_id="instance",
            simple_config_path="/tmp/simple.json",
            config_path=None,
        )
        assert detail["returncode"] == "0"
        assert run.call_args.kwargs["stdin"] is subprocess.DEVNULL
        assert run.call_args.kwargs["timeout"] == 17

    with patch("binderloop.agents.run_monitor_agent.subprocess.run") as run:
        run.side_effect = subprocess.TimeoutExpired(
            cmd=["taiji_client", "logs"],
            timeout=17,
            output=b"partial log",
            stderr=b"partial error",
        )
        output = agent._run_taiji_logs(
            task_flag="task",
            instance_id="instance",
            tail=80,
            simple_config_path=None,
            config_path=None,
        )
        assert "partial log" in output
        assert "partial error" in output
        assert "timed out after 17s" in output

    echoed_start_cmd = 'start_cmd=\'if [ ! -d /aceph/daweihuang ]; then echo "[HARNESS][ERROR] /aceph/daweihuang is not mounted and CEPH_SECRET is not set"; fi\'\n'
    assert "missing_ceph_mount_secret" not in agent._failure_hints(echoed_start_cmd, [])
    assert "missing_ceph_mount_secret" in agent._failure_hints(
        "[HARNESS][ERROR] /aceph/daweihuang is not mounted and CEPH_SECRET is not set", []
    )
    assert "generated_script_python_syntax_error" in agent._failure_hints(
        'SyntaxError: unterminated string literal\n  File "<stdin>", line 1', []
    )

    with patch.object(agent, "_run_taiji_detail", return_value={"stdout": '{"state":"RUNNING"}', "stderr": "", "returncode": "0"}), patch.object(
        agent, "_run_taiji_logs", return_value=""
    ), patch.object(agent, "_missing_expected_outputs") as missing:
        snapshot = agent.check_once(
            task_flag="task",
            instance_id="instance",
            expected_outputs={"metrics": "/ceph/results/**/*.csv"},
        )
        assert snapshot.state == "running"
        assert snapshot.missing_outputs == []
        missing.assert_not_called()

    with tempfile.TemporaryDirectory() as tmp:
        output_root = Path(tmp) / "outputs"
        metrics = output_root / "gpu_0" / "final_ranked_designs" / "final_designs_metrics_1.csv"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("id\ncandidate\n", encoding="utf-8")
        manifest = output_root / "result_manifest.json"
        manifest.write_text(
            json.dumps({"schema_version": 1, "files": [str(metrics.relative_to(output_root))]}),
            encoding="utf-8",
        )
        expected = {
            "boltzgen_output_dir": str(output_root),
            "result_manifest": str(manifest),
            "metrics": str(output_root / "**" / "*.csv"),
            "missing_designs": str(output_root / "gpu_*" / "intermediate_designs"),
        }
        with patch("binderloop.agents.run_monitor_agent.glob.glob", side_effect=AssertionError("manifest should avoid glob")):
            missing = agent._missing_expected_outputs(expected)
        assert missing == ["missing_designs"]

        # Authoritative manifests require both inventory membership and a concrete entity.
        steps = output_root / "steps.yaml"
        steps.write_text("schema_version: 1\ncontract: binder_harness_steps_manifest\n", encoding="utf-8")
        manifest.write_text(json.dumps({
            "schema_version": 6,
            "contract": {"name": "binder_harness_result_manifest", "version": 1},
            "execution_status": "success",
            "status": {"code": 0},
            "files": ["steps.yaml", str(metrics.relative_to(output_root))],
            "required_artifacts": ["steps.yaml"],
            "authoritative": {"inventory": "files", "entities": "artifacts"},
            "artifacts": [{"path": "steps.yaml", "kind": "steps_manifest", "authoritative": True}],
        }), encoding="utf-8")
        authoritative_expected = {**expected, "steps_manifest": str(steps)}
        missing, status = agent._evaluate_expected_outputs(authoritative_expected)
        assert missing == ["missing_designs"]
        assert status["state"] == "authoritative"
        steps.unlink()
        missing, status = agent._evaluate_expected_outputs(authoritative_expected)
        assert "steps_manifest" in missing
        assert "manifest_entity_missing:steps_manifest" in status["diagnostics"]

        # Legacy compatibility applies only to the exact root-level steps manifest.
        steps.write_text("gpu_distribution: {}\n", encoding="utf-8")
        manifest.write_text(json.dumps({"schema_version": 5, "files": [str(metrics.relative_to(output_root))]}), encoding="utf-8")
        missing, status = agent._evaluate_expected_outputs(authoritative_expected)
        assert "steps_manifest" not in missing
        assert "legacy_steps_manifest_files_omission_exact_fallback" in status["diagnostics"]
        assert not agent._is_success("end", 0, ["steps_manifest"])

        # Candidate lineage is an optional enhanced artifact. A successful
        # prediction with metrics must not be resubmitted only because the
        # four-stage candidate manifest is absent.
        missing_with_lineage = ["candidate_manifest"]
        assert agent._is_success("end", 0, missing_with_lineage)
        assert "candidate_manifest" in OPTIONAL_EXPECTED_OUTPUTS
        assert "final_ranked_designs" not in OPTIONAL_EXPECTED_OUTPUTS

        empty_lineage = output_root / "candidate_manifest.jsonl"
        empty_lineage.touch()
        zero_manifest = output_root / "result_manifest.json"
        zero_manifest.write_text(json.dumps({
            "schema_version": 2,
            "execution_status": "success",
            "quality_status": "no_filter_pass",
            "files": ["candidate_manifest.jsonl"],
            "candidate_manifests": ["candidate_manifest.jsonl"],
        }), encoding="utf-8")
        zero_expected = {
            "boltzgen_output_dir": str(output_root),
            "result_manifest": str(zero_manifest),
            "candidate_manifest": str(output_root / "**" / "candidate_manifest.jsonl"),
        }
        assert agent._missing_expected_outputs(zero_expected) == []
        assert agent._is_success("end", 0, [])
        zero_manifest.unlink()
        assert "result_manifest" in agent._missing_expected_outputs(zero_expected)

    print("OK: Taiji monitor commands are non-interactive, bounded, and manifest-first")


if __name__ == "__main__":
    main()
