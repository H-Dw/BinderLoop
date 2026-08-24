#!/usr/bin/env python3

import sys
import tempfile
import json
import hashlib
import os
import re
import subprocess
from unittest import mock
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents import DesignParameterAgent, DesignSpecAgent, TaijiExecutionAgent
from binderloop.config import load_config
from binderloop.models.base import DesignJob
from binderloop.package_layout import PROJECT_PACKAGE_DIRNAME, is_project_package_name, resolve_package_dir
from run_closed_loop_orchestrator import (
    _params_for_remote_boltzgen,
    _point_run_spec_to_package,
    _sync_package_to_remote_run_dir,
    _sync_remote_results_to_local,
)


ROOT = Path(__file__).resolve().parents[1]


def _run_contract_checks() -> None:
    cfg = load_config(ROOT / "configs/example_binder_task.yaml")
    params = DesignParameterAgent().choose_boltzgen_parameters(cfg)
    params.update(
        {
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
            "num_designs": 20,
            "budget": 5,
            "analysis_location": "taiji",
            "run_filtering": False,
            "secondary_structure": "alpha",
        }
    )
    job = DesignJob(
        job_id="bg_example_len50",
        target_structure=str(ROOT / "examples/bg_example/IL-17A.cif"),
        chain_id="A",
        hotspots=["A:67", "A:89", "B:49"],
        binder_length=50,
        seed=0,
        params=params,
        output_dir=str(ROOT / "outputs/bg_example_agents"),
    )
    run_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(job, params=params)
    packaged_target = Path(run_spec.package_dir) / "inputs" / Path(job.target_structure).name
    assert Path(run_spec.package_dir).name == PROJECT_PACKAGE_DIRNAME
    assert packaged_target.is_file()
    assert not packaged_target.is_symlink()
    assert packaged_target.read_bytes() == Path(job.target_structure).read_bytes()
    assert hashlib.sha256(packaged_target.read_bytes()).digest() == hashlib.sha256(
        Path(job.target_structure).read_bytes()
    ).digest()
    assert run_spec.expected_outputs["result_manifest"].endswith("/outputs/boltzgen_output/result_manifest.json")
    assert "--steps design inverse_folding folding design_folding analysis filtering" in run_spec.command_string
    assert run_spec.params["run_filtering"] is True
    design_spec = yaml.safe_load(Path(run_spec.design_spec_path).read_text(encoding="utf-8"))
    assert "secondary_structure" not in design_spec["entities"][0]["protein"]
    assert "--design_checkpoints" in run_spec.command_string
    assert "boltzgen1_diverse.ckpt" in run_spec.command_string
    assert "boltzgen1_adherence.ckpt" in run_spec.command_string
    assert "--inverse_fold_checkpoint" in run_spec.command_string
    assert "--folding_checkpoint" in run_spec.command_string
    assert "--affinity_checkpoint" in run_spec.command_string
    assert "--moldir" in run_spec.command_string
    assert "--seed" not in run_spec.command_string
    assert "random_state=" not in run_spec.command_string
    default_script = Path(run_spec.run_script_path).read_text(encoding="utf-8")
    assert "BOLTZGEN_VERBOSE_LOG" not in default_script
    assert 'BOLTZGEN_SILENCE_LOG="${BOLTZGEN_SILENCE_LOG:-0}"' in default_script
    assert "detail_logging=enabled" in default_script
    assert "detail_logs=screen_and_log" in default_script
    assert "harness_write_result_manifest" in default_script
    assert "harness_preflight_boltzgen_runtime" in default_script
    assert "harness_format_elapsed" in default_script
    assert 'echo "[HARNESS] binder_design elapsed_seconds=$binder_elapsed' in default_script
    assert 'echo "[HARNESS] elapsed_seconds=$HARNESS_ELAPSED_SECONDS' in default_script
    python_heredocs = re.findall(
        r"python\b[^\n]*<<'([A-Z0-9_]+)'\n(.*?)\n\1",
        default_script,
        flags=re.DOTALL,
    )
    assert python_heredocs
    assert {marker for marker, _ in python_heredocs} >= {
        "HARNESS_RUNTIME_PREFLIGHT", "HARNESS_RUNTIME_EXPORTS", "HARNESS_RESULT_MANIFEST",
    }
    for marker, source in python_heredocs:
        compile(source, f"run_boltzgen_full.sh:{marker}", "exec")
    assert 'PYTHONPATH="$PACKAGE_DIR/runtime/boltzgen_src' not in default_script
    assert "BOLTZGEN_LINEAGE_CONTEXT" not in default_script
    assert not (Path(run_spec.package_dir) / "runtime" / "boltzgen_src").exists()
    assert "os.replace(temporary_name" in default_script
    assert "root.rglob" not in default_script

    # The default API remains an overlay for legacy callers, while validated
    # replacement payloads preserve deletions and consume identity separately.
    view_job_params = {
        **params,
        "legacy_overlay_only": "preserved",
        "tombstoned_parameter": "must-not-revive",
        "arm_id": "stale-payload-arm",
    }
    view_job = DesignJob(
        **{
            **job.__dict__,
            "job_id": "bg_example_parameter_views",
            "params": view_job_params,
            "output_dir": str(ROOT / "outputs/bg_example_agents_parameter_views"),
        }
    )
    validated_replacement = {
        key: value
        for key, value in params.items()
        if key not in {"secondary_structure"}
    }
    validated_replacement.update({
        "job_identity": {"semantic_digest": "payload-metadata-must-not-hash"},
        "execution_retry_source_job_id": "payload-retry-context",
        "unknown_replacement_metadata": "tombstone-me",
    })
    replacement_identity = {
        "arm_id": "identity-arm",
        "exploration_arm": "identity-exploration",
        "logical_branch_id": "identity-branch",
        "execution_job_id": "identity-execution-job",
        "execution_slot": 7,
        "job_identity": {"semantic_digest": "identity-semantic-digest"},
    }
    replacement_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(
        view_job,
        params=validated_replacement,
        params_mode="replacement",
        execution_identity_context=replacement_identity,
    )
    assert "tombstoned_parameter" not in replacement_spec.params
    assert "legacy_overlay_only" not in replacement_spec.params
    assert "arm_id" not in replacement_spec.params
    assert "job_identity" not in replacement_spec.params
    assert "execution_retry_source_job_id" not in replacement_spec.params
    assert "unknown_replacement_metadata" not in replacement_spec.params
    assert not any(
        parameter in replacement_spec.params
        for parameter in ("template_requested", "template_staged", "template_applied")
    )
    replacement_identity_record = json.loads(
        Path(replacement_spec.expected_outputs["harness_execution_identity"]).read_text(encoding="utf-8")
    )
    assert replacement_identity_record["arm_id"] == "identity-arm"
    assert replacement_identity_record["exploration_arm"] == "identity-exploration"
    assert replacement_identity_record["logical_branch_id"] == "identity-branch"
    assert replacement_identity_record["execution_job_id"] == "identity-execution-job"
    assert replacement_identity_record["execution_slot"] == 7
    assert replacement_identity_record["semantic_digest"] == "identity-semantic-digest"
    assert replacement_identity_record["parameter_payload_digest"]
    replacement_parity = json.loads(
        Path(replacement_spec.expected_outputs["parameter_payload_parity"]).read_text(encoding="utf-8")
    )
    assert replacement_parity["params_mode"] == "replacement"
    assert replacement_parity["inherited_job_param_keys"] == []
    assert replacement_parity["input_payload_digest"]
    assert replacement_parity["effective_payload_digest"]
    assert replacement_parity["record_digest"]
    assert replacement_identity_record["parameter_payload_digest"] == replacement_parity["input_payload_digest"]
    assert "payload-metadata-must-not-hash" not in json.dumps(replacement_parity)

    overlay_job = DesignJob(
        **{
            **view_job.__dict__,
            "job_id": "bg_example_parameter_overlay_compat",
            "output_dir": str(ROOT / "outputs/bg_example_agents_parameter_overlay_compat"),
        }
    )
    overlay_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(
        overlay_job,
        params={"devices": 1, "arm_id": "legacy-patch-arm"},
    )
    assert overlay_spec.params["legacy_overlay_only"] == "preserved"
    assert overlay_spec.params["tombstoned_parameter"] == "must-not-revive"
    overlay_parity = json.loads(
        Path(overlay_spec.expected_outputs["parameter_payload_parity"]).read_text(encoding="utf-8")
    )
    assert overlay_parity["params_mode"] == "overlay"
    assert "legacy_overlay_only" in overlay_parity["inherited_job_param_keys"]
    overlay_identity_record = json.loads(
        Path(overlay_spec.expected_outputs["harness_execution_identity"]).read_text(encoding="utf-8")
    )
    assert overlay_identity_record["arm_id"] == "legacy-patch-arm"

    template_source = ROOT / "examples/bg_example/IL-17A.cif"
    template_params = {
        **params,
        "binder_template": {
            "mode": "structure_redesign",
            "template_id": "frag_test_high_quality_interface",
            "source_structure_file": str(template_source),
            "binder_chain": "A",
            "fixed_res_index": "10..18",
            "within_proximity": 7.5,
            "quality_score": 0.93,
            "source_digest": __import__("hashlib").sha256(template_source.read_bytes()).hexdigest(),
            "target_alignment": {"status": "aligned", "digest": "alignment-digest-test"},
            "source_to_effective_residue_map": {f"A:{i}": f"A:{i}" for i in range(1, 51)},
            "length_transform": {
                "status": "identity", "method": "unchanged", "digest": "transform-digest-test",
                "fixed_residue_tokens": [f"A:{i}" for i in range(10, 19)],
                "effective_length": 50,
            },
        },
        "devices": 2,
        "num_designs": 4,
        "max_binders_per_round": 4,
    }
    template_job = DesignJob(
        **{
            **job.__dict__,
            "job_id": "bg_example_len50_template_redesign",
            "params": template_params,
            "output_dir": str(ROOT / "outputs/bg_example_agents_template_redesign"),
        }
    )
    template_run_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(template_job, params=template_params)
    template_mask = Path(template_run_spec.package_dir) / "configs" / "boltzgen_redesign_mask.yaml"
    assert template_mask.exists()
    mask_data = yaml.safe_load(template_mask.read_text(encoding="utf-8"))
    fixed_chain = mask_data["restrictions"]["not_design"][0]["chain"]
    assert fixed_chain["binder"] == "A"
    assert fixed_chain["id"] == "A"
    assert fixed_chain["res_index"] == "10..18"
    assert fixed_chain["within_proximity"] == 7.5
    assert "--config inverse_folding data.cfg.design_mask_override=configs/boltzgen_redesign_mask.yaml" in template_run_spec.command_string
    assert template_run_spec.expected_outputs["redesign_mask"].endswith("configs/boltzgen_redesign_mask.yaml")
    assert Path(template_run_spec.expected_outputs["parameter_consumption"]).exists()
    assert Path(template_run_spec.expected_outputs["effective_execution_plan"]).exists()
    execution_plan = json.loads(Path(template_run_spec.expected_outputs["effective_execution_plan"]).read_text(encoding="utf-8"))
    parity = execution_plan["template_artifact_digests"]
    assert parity["source"] == __import__("hashlib").sha256(template_source.read_bytes()).hexdigest()
    assert parity["alignment"] == "alignment-digest-test"
    assert parity["length_transform"] == "transform-digest-test"
    assert parity["design_spec"] and parity["inverse_fold_mask"] and parity["residue_map"]
    consumption = json.loads(Path(template_run_spec.expected_outputs["parameter_consumption"]).read_text(encoding="utf-8"))
    assert "design_spec" in consumption["fields"]["binder_template"]["locations"]
    template_design_spec = yaml.safe_load(Path(template_run_spec.design_spec_path).read_text(encoding="utf-8"))
    template_file = template_design_spec["entities"][0]["file"]
    assert template_file["path"].startswith("../inputs/template_")
    assert template_file["include"] == [{"chain": {"id": "A"}}]
    assert template_file["design"] == [{"chain": {"id": "A"}}]
    assert template_file["not_design"] == [{"chain": {"id": "A", "res_index": "10..18"}}]
    packaged_template = Path(template_run_spec.package_dir) / "inputs" / f"template_{template_source.name}"
    assert packaged_template.is_file()
    assert not packaged_template.is_symlink()
    assert packaged_template.read_bytes() == template_source.read_bytes()
    assert hashlib.sha256(packaged_template.read_bytes()).digest() == hashlib.sha256(
        template_source.read_bytes()
    ).digest()
    template_script = Path(template_run_spec.run_script_path).read_text(encoding="utf-8")
    assert "data.cfg.design_mask_override=configs/boltzgen_redesign_mask.yaml" in template_script
    assert 'CUDA_VISIBLE_DEVICES="$g"' in template_script

    for gpu_count, total_designs in [(2, 7), (4, 10)]:
        multi_gpu_params = {
            **params,
            "devices": gpu_count,
            "num_designs": total_designs,
            "max_binders_per_round": total_designs,
            "binder_lengths": [50, 65, 80],
            "additional_filters": [
                {"feature": "ALA_fraction", "threshold": 0.3, "lower_is_better": True},
                "filter_rmsd_design<2.5",
            ],
        }
        multi_gpu_job = DesignJob(
            **{
                **job.__dict__,
                "job_id": f"bg_example_len50_multigpu_{gpu_count}",
                "params": multi_gpu_params,
                "output_dir": str(ROOT / f"outputs/bg_example_agents_multigpu_{gpu_count}"),
            }
        )
        multi_gpu_run_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(multi_gpu_job, params=multi_gpu_params)
        multi_gpu_script = Path(multi_gpu_run_spec.run_script_path).read_text(encoding="utf-8")
        assert "GPU_DISTRIBUTION_ENABLED=1" in multi_gpu_script
        assert f'GPU_COUNT="${{HARNESS_GPUS_PER_HOST:-{gpu_count}}}"' in multi_gpu_script
        assert f'total_designs={total_designs}' in multi_gpu_script
        assert 'CUDA_VISIBLE_DEVICES="$g"' in multi_gpu_script
        assert 'export BOLTZGEN_SHARD_ORDINAL="$idx"' in multi_gpu_script
        assert 'export BOLTZGEN_SHARD_HOST_ORDINAL="$HOST_RANK"' in multi_gpu_script
        assert 'export BOLTZGEN_SHARD_GPU_ORDINAL="$g"' in multi_gpu_script
        assert 'export BOLTZGEN_SHARD_HOSTNAME="${HOSTNAME:-unknown}"' in multi_gpu_script
        assert '--num_designs "$SHARD_NUM_DESIGNS"' in multi_gpu_script
        assert '--diffusion_batch_size "$SHARD_DIFFUSION_BATCH_SIZE"' in multi_gpu_script
        assert '--devices 1' in multi_gpu_script
        assert 'SHARD_DIFFUSION_BATCH_SIZE="${SHARD_BATCH[$idx]}"' in multi_gpu_script
        assert 'export BOLTZGEN_SHARD_ORDINAL="$idx"' in multi_gpu_script
        assert 'export BOLTZGEN_SHARD_HOST_ORDINAL="$HOST_RANK"' in multi_gpu_script
        assert 'export BOLTZGEN_SHARD_GPU_ORDINAL="$g"' in multi_gpu_script
        assert 'export BOLTZGEN_SHARD_HOSTNAME="${HOSTNAME:-unknown}"' in multi_gpu_script
        assert 'export BOLTZGEN_SHARD_LOGICAL_ORDINAL_START="${SHARD_ORD_START[$idx]}"' in multi_gpu_script
        assert 'SHARD_OUTPUT="outputs/boltzgen_output/gpu_${idx}"' in multi_gpu_script
        assert 'SHARD_LOG="logs/boltzgen_gpu_${idx}.log"' in multi_gpu_script
        assert "--additional_filters 'ALA_fraction<0.3' 'filter_rmsd_design<2.5'" in multi_gpu_script
        assert "elapsed_seconds=$shard_elapsed" in multi_gpu_script
        assert 'echo "[HARNESS] elapsed_seconds=$HARNESS_ELAPSED_SECONDS' in multi_gpu_script
        for length in [50, 65, 80]:
            spec_path = Path(multi_gpu_run_spec.package_dir) / "configs" / f"boltzgen_design_spec_len{length}.yaml"
            assert spec_path.exists(), spec_path
            spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            assert spec_data["entities"][0]["protein"]["sequence"] == str(length)
        assert "boltzgen_design_spec_len50.yaml" in multi_gpu_script
        assert "boltzgen_design_spec_len65.yaml" in multi_gpu_script
        assert "boltzgen_design_spec_len80.yaml" in multi_gpu_script
        remote_package_dir = Path(multi_gpu_run_spec.package_dir).parent / "remote_taiji_project_package"
        _point_run_spec_to_package(multi_gpu_run_spec, remote_package_dir)
        assert multi_gpu_run_spec.expected_outputs["intermediate_designs"].endswith("/gpu_*/intermediate_designs")
        assert multi_gpu_run_spec.expected_outputs["inverse_folded_designs"].endswith("/gpu_*/intermediate_designs_inverse_folded")

    native_params = {
        **params,
        "devices": 8,
        "host_count": 2,
        "taiji_submit_host_num": 2,
        "taiji_multi_host_mode": "native",
        "num_designs": 160,
        "max_binders_per_round": 160,
        "binder_lengths": [80, 85, 90, 95, 100, 105, 110, 115, 120],
    }
    native_job = DesignJob(
        **{
            **job.__dict__,
            "job_id": "bg_example_native_2x8",
            "params": native_params,
            "output_dir": str(ROOT / "outputs/bg_example_agents_native_2x8"),
        }
    )
    native_run_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(
        native_job,
        params=native_params,
    )
    native_package = Path(native_run_spec.package_dir)
    native_script = Path(native_run_spec.run_script_path).read_text(encoding="utf-8")
    native_plan = json.loads((native_package / "configs" / "cluster_shard_plan.json").read_text(encoding="utf-8"))
    assert native_plan["host_count"] == 2
    assert native_plan["gpus_per_host"] == 8
    assert native_plan["worker_count"] == 16
    assert native_plan["total_designs"] == 160
    assert native_plan["schema_version"] == 2
    assert native_plan["harness_execution_identity"] == "configs/harness_execution_identity.json"
    ranges = sorted((item["logical_ordinal_start"], item["logical_ordinal_end"]) for item in native_plan["shards"])
    assert ranges[0][0] == 0 and ranges[-1][1] == 160
    assert sum(end - start for start, end in ranges) == 160
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    lineage_context = json.loads((native_package / "configs" / "harness_execution_identity.json").read_text(encoding="utf-8"))
    assert lineage_context["schema_version"] == 1 and lineage_context["design_spec_digest"] and lineage_context["purpose"] == "execution_safety_only"
    assert lineage_context["job_id"]
    assert "BOLTZGEN_LINEAGE_CONTEXT" not in native_script
    assert sum(item["num_designs"] for item in native_plan["shards"]) == 160
    assert all(0 <= item["host"] < 2 and 0 <= item["gpu"] < 8 for item in native_plan["shards"])
    assert len({
        (item["host"], item["gpu"], item["shard_index"])
        for item in native_plan["shards"]
    }) == len(native_plan["shards"])
    assert "SHARD_HOST=(" in native_script
    assert "HOST_RANK_SOURCE=ceph_hostname_registry" in native_script
    assert 'SHARD_OUTPUT="outputs/boltzgen_output/host_${HOST_TAG}/gpu_${GPU_TAG}/shard_${SHARD_TAG}_len${SHARD_LEN[$idx]}"' in native_script
    assert 'export BOLTZGEN_SHARD_INDEX="$idx"' in native_script
    assert 'export BOLTZGEN_HOST_RANK="$HOST_RANK"' in native_script
    assert 'export BOLTZGEN_GPU_RANK="$g"' in native_script
    assert 'export BOLTZGEN_LOGICAL_ORDINAL_START="${SHARD_ORD_START[$idx]}"' in native_script
    assert 'export BOLTZGEN_IDENTITY_ROOT="$SHARD_OUTPUT"' not in native_script
    assert 'LOG_FILE="logs/host_${HOST_TAG}/boltzgen_full.log"' in native_script
    assert "result_manifest.json" in native_script
    assert native_run_spec.expected_outputs["intermediate_designs"].endswith(
        "/host_*/gpu_*/shard_*/intermediate_designs"
    )
    subprocess.run(["bash", "-n", native_run_spec.run_script_path], check=True)
    native_remote_package = native_package.parent / "remote_native_taiji_project_package"
    (native_remote_package / "scripts").mkdir(parents=True, exist_ok=True)
    _point_run_spec_to_package(native_run_spec, native_remote_package)
    assert native_run_spec.log_file.endswith("/logs/host_00/boltzgen_full.log")
    assert native_run_spec.expected_outputs["log_file"].endswith("/logs/host_00/boltzgen_full.log")
    native_submit = TaijiExecutionAgent(dry_run=True).create_boltzgen_taiji_spec(
        native_run_spec,
        output_json=native_package / "taiji_simple_config.json",
        taiji_options={
            "business_flag": "pathology_gpu_chongqing",
            "project_id": 192631,
            "image_full_name": "mirrors.tencent.com/davedwhuang/boltzgen:cu118",
            "host_num": 2,
            "host_gpu_num": 8,
        },
    )
    assert native_submit.simple_config["host_num"] == 2
    assert native_submit.simple_config["host_gpu_num"] == 8
    assert native_submit.simple_config["exec_start_in_all_mpi_pods"] is True

    heartbeat_params = {**params, "log_heartbeat_seconds": 999, "devices": 1}
    heartbeat_job = DesignJob(
        **{
            **job.__dict__,
            "job_id": "bg_example_len50_heartbeat_logging",
            "params": heartbeat_params,
            "output_dir": str(ROOT / "outputs/bg_example_agents_heartbeat_logging"),
        }
    )
    heartbeat_run_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(heartbeat_job, params=heartbeat_params)
    heartbeat_script = Path(heartbeat_run_spec.run_script_path).read_text(encoding="utf-8")
    assert 'BOLTZGEN_LOG_HEARTBEAT_SECONDS="${BOLTZGEN_LOG_HEARTBEAT_SECONDS:-360}"' in heartbeat_script
    assert "harness_log_runtime_context" in heartbeat_script
    assert "[HARNESS][HEARTBEAT]" in heartbeat_script
    assert "sed -u 's/^/[BOLTZGEN] /'" in heartbeat_script

    silence_params = {**params, "silence": True, "devices": 1}
    silence_job = DesignJob(
        **{
            **job.__dict__,
            "job_id": "bg_example_len50_silence_logging",
            "params": silence_params,
            "output_dir": str(ROOT / "outputs/bg_example_agents_silence_logging"),
        }
    )
    silence_run_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(silence_job, params=silence_params)
    silence_script = Path(silence_run_spec.run_script_path).read_text(encoding="utf-8")
    assert "BOLTZGEN_VERBOSE_LOG" not in silence_script
    assert 'BOLTZGEN_SILENCE_LOG="${BOLTZGEN_SILENCE_LOG:-1}"' in silence_script
    assert "detail_logs=log_file_only" in silence_script
    assert "sed -u 's/^/[BOLTZGEN] /' >> \"$LOG_FILE\"" in silence_script

    remote_params = _params_for_remote_boltzgen({**params, "analysis_location": "local"})
    assert remote_params["analysis_location"] == "taiji"
    assert remote_params["run_analysis_on_taiji"] is True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        failure_job = DesignJob(
            **{
                **job.__dict__,
                "job_id": "missing_target_manifest",
                "target_structure": str(tmp_dir / "missing_target.cif"),
                "params": {**params, "devices": 1},
                "output_dir": str(tmp_dir / "failed_run"),
            }
        )
        failure_spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(
            failure_job,
            params=failure_job.params,
        )
        failed = subprocess.run(["bash", failure_spec.run_script_path], text=True, capture_output=True, check=False)
        assert failed.returncode == 11
        failure_manifest = json.loads(
            (Path(failure_spec.output_dir) / "result_manifest.json").read_text(encoding="utf-8")
        )
        assert failure_manifest["schema_version"] == 6
        assert failure_manifest["collection_mode"] == "round_aggregate"
        assert failure_manifest["collection_mode"] == "round_aggregate"
        assert failure_manifest["candidate_attribution"] is False
        assert failure_manifest["attribution_scope"] == "job"
        assert failure_manifest["identity"]["job_id"] == failure_job.job_id
        assert failure_manifest["stage_classification"] is False
        assert "candidate_manifests" not in failure_manifest and "lineage_summary" not in failure_manifest and "files" in failure_manifest
        assert failure_manifest["status"]["code"] == 11
        assert failure_manifest["mode"] == "single_process"
        assert isinstance(failure_manifest.get("elapsed_seconds"), int)
        assert failure_manifest["elapsed_seconds"] >= 0
        assert failure_manifest.get("start_time")
        assert failure_manifest.get("end_time")

        remote_package = tmp_dir / "remote" / "project_package"
        local_package = tmp_dir / "local" / "project_package"
        (remote_package / "logs").mkdir(parents=True)
        (remote_package / "outputs" / "boltzgen_output").mkdir(parents=True)
        (remote_package / "logs" / "boltzgen_full.log").write_text("remote log", encoding="utf-8")
        (remote_package / "outputs" / "boltzgen_output" / "steps.yaml").write_text("remote steps", encoding="utf-8")
        (remote_package / "taiji_monitor_snapshot.json").write_text("{}", encoding="utf-8")
        local_package.mkdir(parents=True)
        old_link_target = tmp_dir / "must_not_delete"
        old_link_target.mkdir()
        (old_link_target / "sentinel.txt").write_text("preserve", encoding="utf-8")
        (local_package / "outputs").symlink_to(
            Path("..") / Path("..") / old_link_target.name,
            target_is_directory=True,
        )
        sync_record = _sync_remote_results_to_local(remote_package, local_package)
        assert (old_link_target / "sentinel.txt").read_text(encoding="utf-8") == "preserve"
        assert (local_package / "outputs").is_symlink()
        assert not (local_package / "outputs").readlink().is_absolute()
        assert (local_package / "logs" / "boltzgen_full.log").read_text(encoding="utf-8") == "remote log"
        assert (local_package / "outputs" / "boltzgen_output" / "steps.yaml").read_text(encoding="utf-8") == "remote steps"
        assert (local_package / "taiji_monitor_snapshot.json").exists()
        assert len(sync_record["linked"]) == 3
        assert not sync_record["copied"]

        materialized_package = tmp_dir / "materialized" / "project_package"
        materialized = _sync_remote_results_to_local(remote_package, materialized_package, mode="materialize")
        assert len(materialized["copied"]) == 3
        assert not (materialized_package / "outputs").is_symlink()
        assert (materialized_package / "outputs" / "boltzgen_output" / "steps.yaml").read_text(encoding="utf-8") == "remote steps"

        already_local = _sync_remote_results_to_local(remote_package, remote_package)
        assert already_local["already_local"] is True

        input_source = tmp_dir / "source.cif"
        input_source.write_text("data_source\n", encoding="utf-8")
        package_to_stage = tmp_dir / "package_to_stage"
        (package_to_stage / "inputs").mkdir(parents=True)
        (package_to_stage / "inputs" / "source.cif").write_bytes(input_source.read_bytes())
        (package_to_stage / "outputs" / "boltzgen_output").mkdir(parents=True)
        (package_to_stage / "outputs" / "boltzgen_output" / "stale.cif").write_text(
            "stale\n",
            encoding="utf-8",
        )
        (package_to_stage / "logs").mkdir()
        (package_to_stage / "logs" / "stale.log").write_text("stale\n", encoding="utf-8")
        staged = _sync_package_to_remote_run_dir(package_to_stage, "task", tmp_dir / "remote_root")
        assert staged.name == PROJECT_PACKAGE_DIRNAME
        staged_input = staged / "inputs" / "source.cif"
        assert staged_input.read_text(encoding="utf-8") == "data_source\n"
        assert not staged_input.is_symlink()
        # On one filesystem immutable, self-contained package files are staged
        # with hard links.
        local_script = package_to_stage / "run.sh"
        local_script.write_text("echo ok\n", encoding="utf-8")
        staged_again = _sync_package_to_remote_run_dir(package_to_stage, "task2", tmp_dir / "remote_root")
        assert (staged_again / "run.sh").stat().st_ino == local_script.stat().st_ino
        assert not (staged / "outputs" / "boltzgen_output" / "stale.cif").exists()
        assert not (staged / "logs" / "stale.log").exists()
    # Do not copy the example template's token/mount secret in tests. Real runs may
    # pass template_json=examples/bg_example/boltzgen_test_v100.json explicitly.
    submit_spec = TaijiExecutionAgent(dry_run=True).create_boltzgen_taiji_spec(
        run_spec,
        output_json=Path(run_spec.run_script_path).with_name("taiji_simple_config.json"),
        task_flag="binder_boltzgen_bg_example_len50",
        taiji_options={
            "business_flag": "pathology_gpu_chongqing",
            "project_id": 192631,
            "image_full_name": "mirrors.tencent.com/davedwhuang/boltzgen:cu118",
            "GPUName": "V100",
            "host_gpu_num": 1,
            "location": "cq",
        },
    )
    record = TaijiExecutionAgent(dry_run=True).submit(submit_spec)
    assert record.dry_run is True
    assert submit_spec.submit_command.startswith("taiji_client start -scfg ")
    start_cmd = submit_spec.simple_config["start_cmd"]
    assert "cd project_package" in start_cmd
    assert "taiji_project_package" in start_cmd
    print("OK: generated BoltzGen full-pipeline run spec and Taiji dry-run submit spec")
    print(run_spec.run_script_path)
    print(submit_spec.simple_config_path)


def test_boltzgen_taiji_agents_contract() -> None:
    _run_contract_checks()



def test_package_input_replaces_symlink_atomically() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        source = base / "source.cif"
        source.write_bytes(b"new self-contained input\n")
        old_source = base / "old-source.cif"
        old_source.write_bytes(b"old linked input\n")
        target = base / "package" / "inputs" / "target.cif"
        target.parent.mkdir(parents=True)
        target.symlink_to(old_source)

        real_replace = os.replace
        replace_observations = []

        def observe_replace(temporary_name, target_name):
            temporary = Path(temporary_name)
            destination = Path(target_name)
            replace_observations.append({
                "old_content": destination.read_bytes(),
                "temporary_is_regular": temporary.is_file() and not temporary.is_symlink(),
                "temporary_content": temporary.read_bytes(),
            })
            return real_replace(temporary_name, target_name)

        with mock.patch("binderloop.agents.design_spec_agent.os.replace", side_effect=observe_replace):
            packaged = DesignSpecAgent._package_input_file(source, target)

        assert packaged == target
        assert replace_observations == [{
            "old_content": b"old linked input\n",
            "temporary_is_regular": True,
            "temporary_content": source.read_bytes(),
        }]
        assert target.is_file()
        assert not target.is_symlink()
        assert target.read_bytes() == source.read_bytes()
        assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_package_input_refuses_existing_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        source = base / "source.cif"
        source.write_bytes(b"input\n")
        target = base / "inputs" / "target.cif"
        target.mkdir(parents=True)

        try:
            DesignSpecAgent._package_input_file(source, target)
        except IsADirectoryError as exc:
            assert "refusing to replace package input directory" in str(exc)
        else:
            raise AssertionError("package input directory was unexpectedly replaced")
        assert target.is_dir()


def test_package_input_copy_failure_preserves_existing_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        source = base / "source.cif"
        source.write_bytes(b"new input\n")
        target = base / "inputs" / "target.cif"
        target.parent.mkdir()
        target.write_bytes(b"existing input\n")

        with mock.patch(
            "binderloop.agents.design_spec_agent.shutil.copy2",
            side_effect=OSError("copy failed"),
        ):
            try:
                DesignSpecAgent._package_input_file(source, target)
            except OSError as exc:
                assert str(exc) == "copy failed"
            else:
                raise AssertionError("copy failure unexpectedly succeeded")

        assert target.read_bytes() == b"existing input\n"
        assert not target.is_symlink()
        assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_result_sync_relative_binding_and_no_copy_fallback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        remote = base / "remote" / "task" / "project_package"
        local = base / "arm" / "job" / "attempt_1" / "project_package"
        (remote / "outputs" / "boltzgen_output").mkdir(parents=True)
        (remote / "logs").mkdir()
        synced = _sync_remote_results_to_local(
            remote, local, job_id="job", attempt=1, task_flag="task", attempt_root=local.parent,
        )
        binding = synced["transport_binding"]
        assert binding["mode"] == "symlink"
        assert binding["local_package_dir"] == str(local)
        assert binding["local_output_alias"] == str(local / "outputs" / "boltzgen_output")
        assert binding["remote_package_dir"] == str(remote)
        assert binding["remote_output_root"] == str(remote / "outputs" / "boltzgen_output")
        assert binding["local_logs_alias"] == str(local / "logs")
        assert binding["remote_logs_root"] == str(remote / "logs")
        assert binding["link_text"] == str((local / "outputs").readlink())
        assert binding["logs_link_text"] == str((local / "logs").readlink())
        assert (binding["job_id"], binding["attempt"], binding["task_flag"]) == ("job", 1, "task")
        assert binding["attempt_root"] == str(local.parent)
        assert not Path(binding["link_text"]).is_absolute()
        assert not synced["copied"]

        failed_local = base / "failed" / "project_package"
        with mock.patch("pathlib.Path.symlink_to", side_effect=OSError("symlink denied")):
            try:
                _sync_remote_results_to_local(remote, failed_local)
            except RuntimeError as exc:
                assert "failed to create result transport symlink" in str(exc)
            else:
                raise AssertionError("symlink failure unexpectedly fell back to copy")
        assert not (failed_local / "outputs" / "boltzgen_output").exists()

        missing_logs_remote = base / "remote" / "missing_logs" / "project_package"
        (missing_logs_remote / "outputs" / "boltzgen_output").mkdir(parents=True)
        try:
            _sync_remote_results_to_local(missing_logs_remote, base / "missing_logs_local")
        except RuntimeError as exc:
            assert "outputs and logs links are both required" in str(exc)
        else:
            raise AssertionError("incomplete transport unexpectedly published a trusted binding")


def test_package_layout_prefers_canonical_then_legacy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert resolve_package_dir(root).name == PROJECT_PACKAGE_DIRNAME
        legacy = root / "taiji_project_package"
        legacy.mkdir()
        assert resolve_package_dir(root) == legacy
        canonical = root / PROJECT_PACKAGE_DIRNAME
        canonical.mkdir()
        assert resolve_package_dir(root) == canonical
        assert is_project_package_name("project_package")
        assert is_project_package_name("taiji_project_package")
        assert not is_project_package_name("other_package")


def test_direct_inline_analysis_keeps_filtering_in_shard_command() -> None:
    cfg = load_config(ROOT / "configs/example_binder_task.yaml")
    params = DesignParameterAgent().choose_boltzgen_parameters(cfg)
    params.update(
        {
            "analysis_location": "inline",
            "devices": 4,
            "num_designs": 8,
            "budget": 8,
            "max_binders_per_round": 8,
            "run_filtering": True,
            "steps": list(DesignSpecAgent.DEFAULT_FULL_STEPS),
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        job = DesignJob(
            job_id="inline_direct_len50",
            target_structure=str(ROOT / "examples/bg_example/IL-17A.cif"),
            chain_id="A",
            hotspots=["A:67"],
            binder_length=50,
            seed=0,
            params=params,
            output_dir=tmp,
        )
        spec = DesignSpecAgent(ROOT.parent / "boltzgen").create_boltzgen_run_spec(job, params=params)
        assert Path(spec.package_dir).name == PROJECT_PACKAGE_DIRNAME
        assert "--steps design inverse_folding folding design_folding analysis filtering" in spec.command_string
        script = Path(spec.run_script_path).read_text(encoding="utf-8")
        assert "--steps design inverse_folding folding design_folding analysis filtering" in script
        assert "elapsed_seconds=$shard_elapsed" in script
        assert 'echo "[HARNESS] elapsed_seconds=$HARNESS_ELAPSED_SECONDS' in script
        assert not (Path(spec.package_dir) / "scripts" / "run_boltzgen_analysis_local.sh").exists()
        assert spec.expected_outputs["final_ranked_designs"].endswith("/gpu_*/final_ranked_designs")
        subprocess.run(["bash", "-n", spec.run_script_path], check=True)


def main() -> int:
    import pytest
    return int(pytest.main([str(Path(__file__).resolve()), "-q"]))


if __name__ == "__main__":
    raise SystemExit(main())
