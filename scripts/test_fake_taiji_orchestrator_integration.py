#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents import ConfigValidationAgent, DesignSpecAgent, TaijiExecutionAgent
from binderloop.agents.config_parameter_contract import partition_config_parameters
from binderloop.agents.design_spec_agent import _semantic_value
from binderloop.config import HarnessConfig, TargetSpec
from binderloop.execution_governance import stable_digest
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.resume import build_template_execution_identity, classify_template_replay
from scripts.run_closed_loop_orchestrator import _build_taiji_executor


def _args(remote_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        submit=False,
        conda_base="/data/miniconda3",
        conda_env_name="bg",
        secret_config=None,
        taiji_remote_run_root=str(remote_root),
        taiji_client="taiji_client",
        taiji_task_prefix="pytest_fake_taiji",
        no_wait_taiji=True,
        taiji_wait_timeout=None,
        taiji_poll_seconds=5,
        result_sync_mode="symlink",
    )


def _assert_execution_payload_is_clean(payload: dict) -> None:
    partitions = partition_config_parameters(payload)
    assert partitions["orchestration"] == {}
    assert partitions["unknown"] == {}


def _assert_metadata_artifacts(run_spec: dict, record: dict) -> dict:
    expected = run_spec["expected_outputs"]
    required = {"harness_execution_identity", "parameter_payload_parity", "parameter_consumption", "effective_execution_plan"}
    assert required <= set(expected)
    artifacts = {name: json.loads(Path(expected[name]).read_text(encoding="utf-8")) for name in required}

    identity = artifacts["harness_execution_identity"]
    assert identity["schema_version"] == 1
    assert identity["purpose"] == "execution_safety_only"
    assert identity["job_id"] == record["job_id"]
    for key in ("execution_job_id", "arm_id", "logical_branch_id", "semantic_digest", "parameter_payload_digest"):
        assert identity[key]
    spec = yaml.safe_load((Path(run_spec["package_dir"]) / "configs" / "boltzgen_design_spec.yaml").read_text(encoding="utf-8")) or {}
    assert identity["design_spec_digest"] == stable_digest(spec)

    parity = artifacts["parameter_payload_parity"]
    assert parity["record_digest"] == stable_digest({key: value for key, value in parity.items() if key != "record_digest"})
    assert parity["input_payload_digest"] == identity["parameter_payload_digest"]
    validated_payload = record["pre_submit"]["validated_execution_view"]
    assert identity["parameter_payload_digest"] == stable_digest(_semantic_value(validated_payload))
    parameter_plan = yaml.safe_load((Path(run_spec["package_dir"]) / "configs" / "boltzgen_parameter_plan.yaml").read_text(encoding="utf-8")) or {}
    _assert_execution_payload_is_clean(parameter_plan)
    assert parity["effective_payload_digest"] == stable_digest(_semantic_value(parameter_plan))

    assert artifacts["parameter_consumption"]["semantic_digest"]
    plan = artifacts["effective_execution_plan"]
    assert plan["job_id"] == record["job_id"]
    assert plan["semantic_digest"]
    assert plan["parameter_payload_parity"] == parity
    return identity


def _config(target: Path, *, template_enabled: bool = False) -> HarnessConfig:
    cfg = HarnessConfig(target=TargetSpec(str(target), "A", ["A:1"]))
    cfg.task_name = "fake_taiji_e2e"
    cfg.active_learning.branch_width = 2
    cfg.active_learning.max_rounds = 1
    cfg.search_space.binder_lengths = [50]
    cfg.search_space.num_designs_per_round = 2
    cfg.search_space.max_binders_per_round = 2
    cfg.search_space.boltzgen.update({
        "num_designs": 2,
        "budget": 2,
        "devices": 2,
        "filter_biased": True,
        "protocol": "Protein-Anything",
        "fragment_templates_enabled": template_enabled,
        "template_conditioned_fraction": 0.5 if template_enabled else 0.0,
    })
    cfg.resource.backend = "taiji"
    cfg.resource.host_num = 1
    cfg.resource.host_gpu_num = 2
    cfg.resource.max_parallel_jobs = 2
    cfg.resource.template_json = None
    cfg.resource.taiji_options = {"envs": {"FIXTURE_MODE": "v24-offline"}}
    return cfg


def test_one_round_two_arm_fake_taiji_orchestrator_contract(tmp_path: Path) -> None:
    target = tmp_path / "target.cif"
    target.write_text("data_target\n#\n", encoding="utf-8")
    cfg = _config(target)
    out = tmp_path / "run"
    spec_agent = DesignSpecAgent(tmp_path / "missing-boltzgen-checkout")
    executor = _build_taiji_executor(
        cfg,
        root=tmp_path,
        spec_agent=spec_agent,
        args=_args(tmp_path / "fake-taiji-remote"),
        llm_config_path=None,
        config_validator=ConfigValidationAgent(),
    )

    summary = BinderDesignOrchestrator(cfg, out_dir=out, max_rounds=1, max_retries=0).run(execute_job=executor)
    assert len(summary["rounds"]) == 1

    records = json.loads((out / "round_00" / "execution_records.json").read_text(encoding="utf-8"))
    assert len(records) == 2
    assert {record["status"] for record in records} == {"dry_run"}
    assert len({record["job_id"] for record in records}) == 2
    assert len({record["task_flag"] for record in records}) == 2

    identities = []
    for record in records:
        validation_path = Path(record["pre_submit"]["artifact"])
        assert validation_path.is_file()
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        assert validation["is_submittable"] is True
        assert validation["requires_refinalization"] is False

        run_spec = record["run_spec"]
        params = run_spec["params"]
        assert params["devices"] == 2
        assert params["filter_biased"] == "true"
        assert "job_identity" not in params
        assert "arm_id" not in params
        assert "logical_branch_id" not in params
        _assert_execution_payload_is_clean(params)

        identity = _assert_metadata_artifacts(run_spec, record)
        identities.append(identity)

        parity = json.loads(Path(run_spec["expected_outputs"]["parameter_payload_parity"]).read_text(encoding="utf-8"))
        assert parity["params_mode"] == "replacement"
        assert parity["inherited_job_param_keys"] == []
        assert parity["input_payload_digest"] and parity["effective_payload_digest"]

        simple = json.loads(Path(record["submit_spec"]["simple_config_path"]).read_text(encoding="utf-8"))
        assert simple["host_gpu_num"] == 2
        assert simple["envs"]["HARNESS_GPUS_PER_HOST"] == "2"
        assert simple["envs"]["HARNESS_HOST_COUNT"] == "1"
        assert simple["envs"]["HARNESS_MULTI_HOST_MODE"] == "split_jobs"
        assert simple["envs"]["FIXTURE_MODE"] == "v24-offline"
        assert record["submission"]["dry_run"] is True

    identity_keys = {(item["job_id"], item["arm_id"], item["logical_branch_id"], item["semantic_digest"]) for item in identities}
    assert len(identity_keys) == len(records)


def test_two_host_native_dry_run_contract(tmp_path: Path) -> None:
    target = tmp_path / "target.cif"
    target.write_text("data_target\n#\n", encoding="utf-8")
    cfg = _config(target)
    cfg.resource.host_num = 2
    cfg.resource.taiji_multi_host_mode = "native"
    out = tmp_path / "run"
    executor = _build_taiji_executor(
        cfg, root=tmp_path, spec_agent=DesignSpecAgent(tmp_path / "missing-boltzgen-checkout"),
        args=_args(tmp_path / "fake-taiji-remote"), llm_config_path=None,
        config_validator=ConfigValidationAgent(),
    )

    BinderDesignOrchestrator(cfg, out_dir=out, max_rounds=1, max_retries=0).run(execute_job=executor)
    records = json.loads((out / "round_00" / "execution_records.json").read_text(encoding="utf-8"))
    assert len(records) == 2
    assert {record["status"] for record in records} == {"dry_run"}
    for record in records:
        simple = json.loads(Path(record["submit_spec"]["simple_config_path"]).read_text(encoding="utf-8"))
        assert simple["host_num"] == 2
        assert simple["host_gpu_num"] == 2
        assert simple["envs"]["HARNESS_HOST_COUNT"] == "2"
        assert simple["envs"]["HARNESS_MULTI_HOST_MODE"] == "native"
        shard_plan_path = Path(record["local_package_dir"]) / "configs" / "cluster_shard_plan.json"
        assert shard_plan_path.is_file(), "2-host native submission must package a cluster shard plan"
        shard_plan = json.loads(shard_plan_path.read_text(encoding="utf-8"))
        assert shard_plan["mode"] == "native"
        assert shard_plan["host_count"] == 2
        assert shard_plan["worker_count"] == 4
        assert {shard["host"] for shard in shard_plan["shards"]} <= {0, 1}
        _assert_metadata_artifacts(record["run_spec"], record)


def test_v24_fixture_replay_template_modes_and_json_yaml_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "target.cif"
    template = tmp_path / "template.cif"
    target.write_text("data_target\n", encoding="utf-8")
    template.write_text("data_template\n", encoding="utf-8")
    base_template = {
        "template_id": "v24-fixture",
        "mode": "structure_redesign",
        "source_structure_file": str(template),
        "binder_chain": "A",
        "fixed_res_index": "1..3",
        "target_alignment": {"status": "aligned", "digest": "alignment-v24"},
        "length_transform": {"status": "identity", "digest": "length-v24"},
    }
    old = build_template_execution_identity(
        base_template,
        target_structure=target,
        target_chain="A",
        lineage_schema_version=2,
        lineage_manifest_digest="fixture-manifest-v24",
    )
    replayed = build_template_execution_identity(
        copy.deepcopy(base_template),
        target_structure=target,
        target_chain="A",
        lineage_schema_version=2,
        lineage_manifest_digest="fixture-manifest-v24",
    )
    verdict = classify_template_replay(old, replayed)
    assert verdict["status"] == "exact_replay"
    assert verdict["exact_attribution"] is True

    fixture = {
        "schema_version": 24,
        "template_on": {"fragment_templates_enabled": True, "binder_template": base_template},
        "template_off": {"fragment_templates_enabled": False, "template_conditioned_fraction": 0.0},
        "resume_identity": old,
    }
    json_path = tmp_path / "v24_fixture.json"
    yaml_path = tmp_path / "v24_fixture.yaml"
    json_path.write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(fixture, sort_keys=True), encoding="utf-8")
    from_json = json.loads(json_path.read_text(encoding="utf-8"))
    from_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert from_json == from_yaml == fixture
    assert from_json["template_on"]["binder_template"]["mode"] == "structure_redesign"
    assert from_json["template_off"]["fragment_templates_enabled"] is False
    assert classify_template_replay(from_json["resume_identity"], from_yaml["resume_identity"])["status"] == "exact_replay"


def test_real_taiji_agent_remains_dry_run_without_client(tmp_path: Path) -> None:
    assert TaijiExecutionAgent(taiji_client_bin="definitely-not-installed", dry_run=True).dry_run is True


def main() -> int:
    import pytest
    return int(pytest.main([str(Path(__file__).resolve()), "-q"]))


if __name__ == "__main__":
    raise SystemExit(main())
