import argparse
import copy
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location("closed_loop_runner", ROOT / "scripts" / "run_closed_loop_orchestrator.py")
runner = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader
MODULE_SPEC.loader.exec_module(runner)

from binderloop.agents import ConfigValidationAgent
from binderloop.models.base import DesignJob


def _job(tmp_path, params):
    target = tmp_path / "target.cif"
    target.write_text("data_target\n", encoding="utf-8")
    return DesignJob("v24_job", str(target), "A", ["A:1"], 80, params=copy.deepcopy(params), output_dir=str(tmp_path / "job"))


def _args(**overrides):
    values = dict(submit=False, conda_base="/data/miniconda3", conda_env_name="bg", secret_config=None,
        taiji_remote_run_root="", taiji_client="taiji_client", taiji_task_prefix=None, no_wait_taiji=True,
        taiji_wait_timeout=None, taiji_poll_seconds=5, result_sync_mode="symlink")
    values.update(overrides)
    return argparse.Namespace(**values)


@dataclass
class FakeRunSpec:
    task_id: str
    job_id: str
    design_spec_path: str
    run_script_path: str
    output_dir: str
    log_file: str
    command: list = field(default_factory=list)
    command_string: str = ""
    expected_outputs: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    package_dir: str = ""


class CapturingSpecAgent:
    def __init__(self): self.calls = []
    def create_boltzgen_run_spec(self, job, *, params, params_mode="overlay", execution_identity_context=None, **_):
        self.calls.append((copy.deepcopy(params), params_mode, copy.deepcopy(execution_identity_context)))
        package = Path(job.output_dir) / "package"
        for name in ("scripts", "logs", "outputs/boltzgen_output"):
            (package / name).mkdir(parents=True, exist_ok=True)
        script = package / "scripts/run.sh"
        script.write_text("exit 0\n", encoding="utf-8")
        return FakeRunSpec(task_id=job.job_id, job_id=job.job_id, design_spec_path="spec.yaml",
            run_script_path=str(script), output_dir=str(package / "outputs/boltzgen_output"),
            log_file=str(package / "logs/run.log"), params=copy.deepcopy(params), package_dir=str(package))


def test_real_validator_v24_metadata_normalizes_without_blocking(tmp_path):
    params = {
        "filter_biased": True, "protocol": "Protein-Anything", "steps": ["design", "folding"],
        "analysis_note": "v24-shaped unknown metadata", "job_identity": {"semantic_digest": "abc"},
        "execution_retry_source_job_id": "prior-v24-job", "num_designs": 80, "budget": 20,
    }
    job = _job(tmp_path, params)
    before = copy.deepcopy(job.params)
    spec = CapturingSpecAgent()
    record = runner._build_local_executor(spec_agent=spec, args=_args(), config_validator=ConfigValidationAgent())(job, 1)
    assert record["status"] == "dry_run"
    assert record["pre_submit"]["is_submittable"] is True
    assert record["pre_submit"]["requires_refinalization"] is False
    assert record["pre_submit"]["diff"]["normalization"]["filter_biased"]["after"] == "true"
    assert "analysis_note" in record["pre_submit"]["diff"]["metadata_stripping"]
    assert spec.calls[0][1] == "replacement"
    assert spec.calls[0][2]["job_identity"] == before["job_identity"]
    assert spec.calls[0][2]["execution_retry_source_job_id"] == "prior-v24-job"
    assert "job_identity" not in spec.calls[0][0]
    assert "execution_retry_source_job_id" not in spec.calls[0][0]
    assert "analysis_note" not in spec.calls[0][0]
    assert job.params == before


def test_typed_patch_excludes_metadata_unknown_stripping_and_safe_defaults():
    validation = {"schema_version": 1, "is_submittable": True, "issues": []}
    proposal = runner._typed_correction_proposal(
        {"filter_biased": True, "analysis_note": "drop", "job_identity": {"semantic_digest": "x"}},
        {"filter_biased": "true", "num_workers": 4},
        reason="post_failure_config_validation", validation=validation, safe_default_keys=["num_workers"],
    )
    patch = proposal["correction_patch"]
    assert patch["set"] == {"filter_biased": "true"}
    assert patch["remove"] == []
    assert patch["classification"] == "semantic_config_correction"
    assert patch["identity_effect"] == "requires_refinalization"
    assert len(patch["source_validation_digest"]) == 64
    assert proposal["corrected_params"]["num_workers"] == 4


def test_invalid_required_blocks_before_run_spec(tmp_path):
    class InvalidValidator(ConfigValidationAgent):
        def validate_full_job_config(self, config, **kwargs):
            result = super().validate_full_job_config(config, **kwargs)
            result.is_valid = False
            result.issues.append({"parameter": "required", "severity": "error", "resolved": False})
            return result
    spec = CapturingSpecAgent()
    record = runner._build_local_executor(spec_agent=spec, args=_args(), config_validator=InvalidValidator())(
        _job(tmp_path, {"filter_biased": "true"}), 1)
    assert record["submit_status"] == "blocked"
    assert not spec.calls


def test_local_and_taiji_use_same_preparation_path(tmp_path):
    job = _job(tmp_path, {"filter_biased": True, "devices": "3", "num_designs": 80, "budget": 20, "analysis_note": "drop"})
    validator = ConfigValidationAgent()
    local = runner._prepare_pre_submit_execution(job, 1, backend="local", backend_overrides={}, config_validator=validator)
    taiji = runner._prepare_pre_submit_execution(job, 1, backend="taiji",
        backend_overrides={"analysis_location": "taiji", "run_analysis_on_taiji": True}, config_validator=validator)
    assert local.validated_params["filter_biased"] == taiji.validated_params["filter_biased"] == "true"
    assert local.validated_params["devices"] == taiji.validated_params["devices"] == 3
    assert taiji.validated_params["analysis_location"] == "taiji"
    assert "analysis_note" not in local.validated_params
    assert local.orchestration_context == {}


@dataclass
class FakeSubmission:
    task_id: str = "v24_job"
    task_flag: str = "v24_flag"
    simple_config_path: str = "simple.json"
    submit_command: str = "never-executed"
    returncode: object = None
    stdout: str = ""
    stderr: str = ""
    taiji_job_id: object = None
    dry_run: bool = True


@dataclass
class FakeSubmitSpec:
    task_id: str
    task_flag: str
    simple_config_path: str
    submit_command: str
    run_script_path: str
    output_dir: str
    full_config_path: object = None
    simple_config: dict = field(default_factory=dict)


def test_local_executor_injects_inline_analysis(tmp_path):
    spec = CapturingSpecAgent()
    job = _job(tmp_path, {"filter_biased": "true", "num_designs": 8, "budget": 8})
    record = runner._build_local_executor(
        spec_agent=spec,
        args=_args(conda_executable="conda"),
        config_validator=ConfigValidationAgent(),
        backend="direct",
    )(job, 1)
    assert record["status"] == "dry_run"
    assert spec.calls[0][0]["analysis_location"] == "inline"


def test_taiji_dry_run_submit_uses_validated_resource_env(tmp_path, monkeypatch):
    job = _job(tmp_path, {"filter_biased": True, "devices": "3", "num_designs": 80, "budget": 20})
    spec_agent = CapturingSpecAgent()
    captured = {}

    class FakeTaijiAgent:
        def __init__(self, *_, **__): pass
        def create_boltzgen_taiji_spec(self, run_spec, **kwargs):
            captured["options"] = copy.deepcopy(kwargs["taiji_options"])
            return FakeSubmitSpec(job.job_id, kwargs["task_flag"], "simple.json", "never-executed",
                run_spec.run_script_path, run_spec.output_dir)
        def submit(self, submit_spec, dry_run=True):
            captured["submit_called"] = True
            assert dry_run is True
            return FakeSubmission(task_flag=submit_spec.task_flag)

    monkeypatch.setattr(runner, "TaijiExecutionAgent", FakeTaijiAgent)
    monkeypatch.setattr(runner, "_sync_package_to_remote_run_dir", lambda package, flag, root: Path(package))
    resource = SimpleNamespace(
        host_num=1, host_gpu_num=8, taiji_multi_host_mode="native", template_json=None, timeout_seconds=3600,
        to_taiji_options=lambda: {"host_num": 1, "host_gpu_num": 8, "envs": {}},
    )
    cfg = SimpleNamespace(resource=resource, task_name="v24")
    record = runner._build_taiji_executor(cfg, root=ROOT, spec_agent=spec_agent,
        args=_args(taiji_remote_run_root=str(tmp_path / "remote")), llm_config_path=None,
        config_validator=ConfigValidationAgent())(job, 1)
    assert captured["submit_called"] is True
    assert record["status"] == "dry_run"
    assert record["effective_devices"] == 3
    assert captured["options"]["host_gpu_num"] == 3
    assert captured["options"]["envs"]["HARNESS_GPUS_PER_HOST"] == "3"
    assert spec_agent.calls[0][0]["devices"] == 3
    assert spec_agent.calls[0][1] == "replacement"


def main() -> int:
    import pytest
    return int(pytest.main([str(Path(__file__).resolve()), "-q"]))


if __name__ == "__main__":
    raise SystemExit(main())


def test_round_pre_submit_summary_mixed_arms_and_artifact_counts(tmp_path):
    from binderloop.config import HarnessConfig, TargetSpec
    from binderloop.orchestration.orchestrator import BinderDesignOrchestrator

    cfg = HarnessConfig(target=TargetSpec(structure_path=str(tmp_path / "target.cif")))
    orchestrator = BinderDesignOrchestrator(cfg, out_dir=tmp_path / "out", max_parallel=1, max_retries=1)
    jobs = [
        DesignJob("job_a", "target.cif", "A", [], 60, params={"arm_id": "baseline_hold"}, output_dir=str(tmp_path / "a")),
        DesignJob("job_b", "target.cif", "A", [], 70, params={"arm_id": "probe"}, output_dir=str(tmp_path / "b")),
    ]
    records = [
        {
            "schema_version": 1, "job_id": "job_a", "backend": "local", "attempt": 1,
            "status": "dry_run", "submit_status": "not_requested",
            "pre_submit": {"schema_version": 1, "artifact": str(tmp_path / "a" / "validation.json"),
                "is_submittable": True, "requires_refinalization": False,
                "diff": {"schema_version": 1, "normalization": {"filter_biased": {"before": True, "after": "true"}},
                         "metadata_stripping": ["analysis_note"], "validator_additions": {}, "backend_overrides": {}},
                "validation": {"schema_version": 2, "issues": [], "missing_required_keys": []}},
        },
        {
            "schema_version": 1, "job_id": "job_b", "backend": "taiji", "attempt": 2,
            "status": "failed", "submit_status": "blocked",
            "pre_submit": {"schema_version": 1, "artifact": str(tmp_path / "b" / "validation.json"),
                "is_submittable": False, "requires_refinalization": True,
                "diff": {"schema_version": 1, "normalization": {}, "metadata_stripping": [],
                         "validator_additions": {"num_workers": 4}, "backend_overrides": {"analysis_location": "taiji"}},
                "validation": {"schema_version": 2, "issues": [{"parameter": "budget", "severity": "error", "resolved": False}],
                               "missing_required_keys": ["budget"]}},
        },
    ]
    summary = orchestrator._build_pre_submit_summary(3, jobs, records)
    assert summary["schema_version"] == 1
    assert summary["job_count"] == summary["arm_count"] == 2
    assert summary["backend_counts"] == {"local": 1, "taiji": 1}
    assert summary["validation_artifact_count"] == 2
    assert summary["normalization_count"] == summary["removal_count"] == 1
    assert summary["validator_addition_count"] == 1
    assert summary["requires_refinalization_count"] == 1
    assert summary["blocking_issue_count"] == summary["missing_required_job_count"] == 1
    assert summary["jobs"][1]["attempt"] == 2
    assert summary["jobs"][1]["schema_versions"] == {"execution_record": 1, "pre_submit": 1, "validation": 2, "diff": 1}
