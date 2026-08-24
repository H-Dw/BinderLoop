
import json
import os
import shlex
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import yaml

from binderloop.models.base import DesignJob
from binderloop.agents.model_input_spec import normalize_additional_filters
from binderloop.agents.config_parameter_contract import parameter_contract_entry, partition_config_parameters
from binderloop.models.boltzgen_adapter import (
    BoltzGenAdapter,
    _build_redesign_schema,
    with_default_local_artifacts,
)
from binderloop.models.boltzgen_renderer import render_boltzgen_command
from binderloop.execution_governance import resolve_execution_plan, finalize_execution_plan, stable_digest
from binderloop.package_layout import PROJECT_PACKAGE_DIRNAME


@dataclass
class BoltzGenRunSpec:
    """Concrete runnable BoltzGen spec produced by DesignSpecAgent."""

    task_id: str
    job_id: str
    design_spec_path: str
    run_script_path: str
    output_dir: str
    log_file: str
    command: List[str]
    command_string: str
    expected_outputs: Dict[str, str]
    params: Dict[str, Any] = field(default_factory=dict)
    package_dir: Optional[str] = None


def _semantic_value(value: Any) -> Any:
    """Remove deployment-specific absolute paths from semantic identities."""
    if isinstance(value, Mapping):
        return {str(key): _semantic_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return "<absolute-runtime-path>"
    return value


class DesignSpecAgent:
    """Translate DesignJob + selected parameters into a full BoltzGen run spec.

    For Taiji we use a *project package* layout so input, generated spec, run
    script and output are all under the submitted project directory.  This avoids
    hard-coded remote `/aceph/...` target paths and fragile nested heredocs inside
    JSON `start_cmd`.
    """

    DEFAULT_GPU_STEPS = ["design", "inverse_folding", "folding", "design_folding"]
    DEFAULT_FULL_STEPS = ["design", "inverse_folding", "folding", "design_folding", "analysis", "filtering"]
    LOCAL_ANALYSIS_STEPS = ["analysis", "filtering"]

    def __init__(
        self,
        boltzgen_root: Union[str, Path] = "../boltzgen",
        *,
        checkpoint_dir: Union[str, Optional[Path]] = None,
        cache_dir: Union[str, Optional[Path]] = None,
        moldir: Union[str, Optional[Path]] = None,
    ):
        self.boltzgen_root = Path(boltzgen_root)
        self.checkpoint_dir = self._configured_artifact_path(checkpoint_dir)
        self.cache_dir = self._configured_artifact_path(cache_dir)
        self.moldir = self._configured_artifact_path(moldir)
        self.adapter = BoltzGenAdapter(str(self.boltzgen_root))

    @staticmethod
    def _configured_artifact_path(value: Union[str, Optional[Path]]):
        if not value:
            return None
        text = str(value)
        return PurePosixPath(text) if text.startswith("/") else Path(text).expanduser()

    def create_boltzgen_run_spec(
        self,
        job: DesignJob,
        *,
        params: Optional[Mapping[str, Any]] = None,
        params_mode: Literal["overlay", "replacement"] = "overlay",
        execution_identity_context: Optional[Mapping[str, Any]] = None,
        conda_base: str = "/data/miniconda3",
        conda_env_name: str = "bg",
    ) -> BoltzGenRunSpec:
        if params_mode not in {"overlay", "replacement"}:
            raise ValueError(f"unsupported params_mode: {params_mode}")
        supplied_params = dict(params or {})
        identity_context = dict(
            execution_identity_context
            if execution_identity_context is not None
            else supplied_params if params_mode == "overlay"
            else (job.params or {})
        )
        if params_mode == "replacement":
            # A validated replacement is the complete executable payload. Keep
            # orchestration/identity out of runner params and parity digests even
            # for direct callers; unknown keys are tombstoned. Metadata remains
            # available only through the identity artifact.
            supplied_partitions = partition_config_parameters(supplied_params)
            merged_params = {
                str(key): item
                for partition in ("runner", "adapter", "runtime")
                for key, item in supplied_partitions[partition].items()
            }
            identity_context = {
                **dict(supplied_partitions["orchestration"]),
                **identity_context,
            }
        else:
            # Historical callers use params as a patch over DesignJob.params.
            merged_params = dict(job.params or {})
            merged_params.update(supplied_params)
        input_payload_digest = stable_digest(_semantic_value(merged_params))
        merged_params = with_default_local_artifacts(
            merged_params,
            self.boltzgen_root,
            checkpoint_dir=self.checkpoint_dir,
            cache_dir=self.cache_dir,
            moldir=self.moldir,
        )
        # Closed-loop analysis, active learning, and trend plots consume final
        # ranked metrics. Keep BoltzGen filtering/ranking enabled even if an
        # upstream strategy proposed a diagnostic analysis-only run.
        merged_params["run_filtering"] = True
        if merged_params.get("weighted_hotspot_conditioning"):
            raise ValueError(
                "weighted_hotspot_conditioning is unsupported by current BoltzGen checkpoints; "
                "use binary BINDING/NOT_BINDING and post-generation ranking"
            )
        final_parameter_state = merged_params.pop("final_parameter_state", None)
        parameter_catalog = merged_params.pop("parameter_catalog", None)
        parameter_catalog_digest = str(merged_params.pop("parameter_catalog_digest", "") or "")
        initial_plan = resolve_execution_plan(
            merged_params,
            job_id=job.job_id,
            operational_bounds=merged_params.get("sampler_bounds"),
            final_parameter_state=final_parameter_state,
            parameter_catalog=parameter_catalog,
            parameter_catalog_digest=parameter_catalog_digest,
        )
        merged_params = dict(initial_plan.resolved_params)

        run_root = Path(job.output_dir)
        package_dir = Path(merged_params.get("package_dir") or run_root / PROJECT_PACKAGE_DIRNAME)
        input_dir = package_dir / "inputs"
        config_dir = package_dir / "configs"
        script_dir = package_dir / "scripts"
        output_dir = package_dir / "outputs" / "boltzgen_output"
        log_dir = package_dir / "logs"
        for d in [input_dir, config_dir, script_dir, output_dir, log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # BoltzGen executes from the image environment. Do not ship or prepend a
        # locally modified source tree: the filename producer and every downstream
        # reader must come from the same installed distribution.
        runtime_source = package_dir / "runtime" / "boltzgen_src"
        if runtime_source.exists():
            shutil.rmtree(runtime_source)

        # Package inputs as self-contained regular files so Taiji's project ZIP
        # never depends on links that resolve outside the submitted package.
        source_target = Path(job.target_structure)
        packaged_target = input_dir / source_target.name
        if source_target.exists():
            self._package_input_file(source_target, packaged_target)
        else:
            # Keep a clear failure message in the generated script rather than silently
            # using a remote path that may not exist.
            self._remove_package_entry(packaged_target)
            packaged_target.write_text("", encoding="utf-8")

        params_for_spec = dict(merged_params)
        params_for_spec["target_path_for_spec"] = "../inputs/" + packaged_target.name
        template = params_for_spec.get("binder_template")
        template_requested = isinstance(template, Mapping) and str(template.get("mode")) == "structure_redesign"
        effective_template = None
        merged_params["template_requested"] = bool(template_requested)
        merged_params["template_staged"] = False
        merged_params["template_applied"] = False
        merged_params.pop("template_drop_reason", None)
        if template_requested:
            # Keep each materialized structure paired with the digest computed for
            # that exact file.  Aligned/coherent coordinates are preferred because
            # structure-redesign templates must share the current target frame.
            template_sources = (
                ("coherent_aligned", template.get("coherent_frame_source_structure_file") or template.get("source_structure_file"), template.get("source_digest")),
                ("staged_unaligned", template.get("staged_source_structure_file") or template.get("unaligned_source_structure_file"), template.get("unaligned_source_digest")),
            )
            selected_source_kind, selected_source_value, selected_digest = next(
                ((kind, value, digest) for kind, value, digest in template_sources if value),
                ("missing", "", ""),
            )
            source_template = Path(str(selected_source_value or ""))
            if source_template.exists() and source_template.is_file():
                expected_source_digest = str(selected_digest or "")
                actual_source_digest = self._file_sha256(source_template)
                merged_params["template_source_parity"] = {
                    "selected_path_kind": selected_source_kind,
                    "source_path": str(source_template),
                    "expected_digest": expected_source_digest,
                    "actual_digest": actual_source_digest,
                }
                if not expected_source_digest or actual_source_digest != expected_source_digest:
                    raise ValueError("template_pre_submit_parity:source_digest_mismatch")
                packaged_template = input_dir / f"template_{source_template.name}"
                self._package_input_file(source_template, packaged_template)
                effective_template = {
                    **dict(template),
                    "source_structure_file": "../inputs/" + packaged_template.name,
                }
                params_for_spec["binder_template"] = effective_template
                merged_params["template_staged"] = True
                merged_params["template_applied"] = True
                merged_params["effective_template_id"] = str(template.get("template_id") or "")
            else:
                reason = f"unpackagable_template_source:{source_template}"
                raise ValueError(
                    f"template_materialization_failure:{reason}; template_exploit jobs may not degrade to template-free"
                )
        model_job = DesignJob(
            **{
                **job.__dict__,
                "target_structure": params_for_spec["target_path_for_spec"],
                "params": params_for_spec,
                "output_dir": str(output_dir),
            }
        )

        # Execution identity exists only to protect package/retry/resume safety.
        identity_job = dict(identity_context.get("job_identity") or {})
        execution_identity = {
            "schema_version": 1,
            "job_id": str(identity_context.get("job_id") or job.job_id),
            "arm_id": str(identity_context.get("arm_id") or identity_context.get("exploration_arm") or ""),
            "exploration_arm": str(identity_context.get("exploration_arm") or ""),
            "logical_branch_id": str(identity_context.get("logical_branch_id") or identity_context.get("branch_id") or ""),
            "execution_job_id": str(identity_context.get("execution_job_id") or job.job_id),
            "execution_slot": identity_context.get("execution_slot"),
            "semantic_digest": str(identity_context.get("semantic_digest") or identity_job.get("semantic_digest") or ""),
            "parameter_payload_digest": input_payload_digest,
            "purpose": "execution_safety_only",
        }

        # Write spec under configs/ with target path relative to configs/.
        spec_path = config_dir / "boltzgen_design_spec.yaml"
        self._write_design_spec_to_path(model_job, spec_path)
        redesign_mask_path = self._write_redesign_mask_to_path(model_job, config_dir / "boltzgen_redesign_mask.yaml")
        if effective_template is not None:
            self._validate_template_parity(effective_template, spec_path, redesign_mask_path)
            parity = self._template_parity_digests(effective_template, spec_path, redesign_mask_path)
            merged_params["template_artifact_digests"] = parity
        redesign_mask_cli_path = Path("configs") / redesign_mask_path.name if redesign_mask_path is not None else None
        execution_identity["design_spec_digest"] = stable_digest(
            yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        )
        execution_identity_path = config_dir / "harness_execution_identity.json"
        execution_identity_path.write_text(
            json.dumps(execution_identity, ensure_ascii=False, indent=2), encoding="utf-8"
        )


        command = self._build_packaged_command(
            spec_path=Path("configs") / spec_path.name,
            output_dir=Path("outputs") / "boltzgen_output",
            params=merged_params,
            redesign_mask_path=redesign_mask_cli_path,
        )
        steps = self._effective_steps(merged_params)
        command = self._ensure_complete_pipeline(command, merged_params)
        command_string = " ".join(shlex.quote(str(x)) for x in command)

        # Single-task-per-round multi-length fan-out: when GPU distribution is
        # active, the whole round runs inside one Taiji task. Each requested binder
        # length gets its own design spec, and the round budget is split into
        # shards distributed across the GPUs so all of them stay busy (BoltzGen has
        # one fixed length per spec, so multiple lengths require multiple specs).
        gpu_distribution = self._gpu_distribution(merged_params, steps)
        gpu_shards: Optional[List[Dict[str, Any]]] = None
        if gpu_distribution:
            round_lengths = gpu_distribution.get("lengths") or [int(model_job.binder_length)]
            per_length_specs = self._write_per_length_specs(model_job, config_dir, round_lengths)
            gpu_shards = self._build_gpu_shards(
                lengths=round_lengths,
                total_designs=gpu_distribution["num_designs"],
                gpu_count=gpu_distribution["devices"],
                host_count=gpu_distribution["host_count"],
            )
            resolved_batch_size = max(1, int(merged_params.get("diffusion_batch_size", 1) or 1))
            for shard in gpu_shards:
                shard["spec"] = per_length_specs[int(shard["length"])]
                shard["executor_diffusion_batch_size"] = resolved_batch_size
            shard_plan = {
                "schema_version": 2,
                "harness_execution_identity": "configs/harness_execution_identity.json",
                "mode": "native" if gpu_distribution["host_count"] > 1 else "single_host",
                "host_count": gpu_distribution["host_count"],
                "gpus_per_host": gpu_distribution["devices"],
                "worker_count": gpu_distribution["host_count"] * gpu_distribution["devices"],
                "total_designs": sum(int(shard["num_designs"]) for shard in gpu_shards),
                "shards": gpu_shards,
            }
            (config_dir / "cluster_shard_plan.json").write_text(
                json.dumps(shard_plan, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        silence_logging = self._bool_param(
            merged_params,
            "silence",
            "silent",
            "silence_logging",
            "boltzgen_silence_log",
            "boltzgen_silent_log",
            default=False,
        )
        heartbeat_seconds = self._heartbeat_seconds(merged_params)

        script_path = script_dir / "run_boltzgen_full.sh"
        native_host_count = int(gpu_distribution["host_count"]) if gpu_distribution else 1
        log_file = (
            log_dir / "host_00" / "boltzgen_full.log"
            if native_host_count > 1
            else log_dir / "boltzgen_full.log"
        )
        script_path.write_text(
            self._render_project_run_script(
                command=command,
                log_file=log_file,
                package_dir=package_dir,
                target_file=packaged_target,
                checkpoint_paths=self._local_checkpoint_paths(merged_params, steps),
                cache_dir=merged_params["cache"],
                moldir=merged_params["moldir"],
                gpu_shards=gpu_shards,
                host_count=native_host_count,
                gpus_per_host=int(gpu_distribution["devices"]) if gpu_distribution else 1,
                silence_logging=silence_logging,
                heartbeat_seconds=heartbeat_seconds,
                conda_base=conda_base,
                conda_env_name=conda_env_name,
            ),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        if "analysis" not in steps:
            local_analysis_params = {**merged_params, "analysis_location": "taiji", "steps": self.LOCAL_ANALYSIS_STEPS, "run_filtering": merged_params.get("run_filtering", True)}
            local_analysis_command = self._ensure_complete_pipeline(
                self._build_packaged_command(
                    spec_path=Path("configs") / spec_path.name,
                    output_dir=Path("outputs") / "boltzgen_output",
                    params={**merged_params, "reuse": True},
                    redesign_mask_path=redesign_mask_cli_path,
                ),
                local_analysis_params,
            )
            analysis_script = script_dir / "run_boltzgen_analysis_local.sh"
            analysis_script.write_text(
                self._render_project_run_script(
                    command=local_analysis_command,
                    log_file=log_dir / "boltzgen_analysis_local.log",
                    package_dir=package_dir,
                    target_file=packaged_target,
                    checkpoint_paths=self._local_checkpoint_paths(local_analysis_params, self.LOCAL_ANALYSIS_STEPS),
                    cache_dir=merged_params["cache"],
                    moldir=merged_params["moldir"],
                    gpu_shards=None,
                    host_count=1,
                    gpus_per_host=1,
                    silence_logging=silence_logging,
                    heartbeat_seconds=heartbeat_seconds,
                    conda_base=conda_base,
                    conda_env_name=conda_env_name,
                ),
                encoding="utf-8",
            )
            analysis_script.chmod(0o755)

        if params_mode == "replacement":
            effective_partitions = partition_config_parameters(merged_params)
            payload_params = {
                str(key): item
                for partition in ("runner", "adapter", "runtime")
                for key, item in effective_partitions[partition].items()
            }
        else:
            payload_params = merged_params
        param_path = config_dir / "boltzgen_parameter_plan.yaml"
        param_path.write_text(yaml.safe_dump(payload_params, allow_unicode=True), encoding="utf-8")
        payload_parity = {
            "schema_version": 1,
            "params_mode": params_mode,
            "input_payload_digest": input_payload_digest,
            "effective_payload_digest": stable_digest(_semantic_value(payload_params)),
            "inherited_job_param_keys": sorted(
                set(job.params or {}) - set(supplied_params)
            ) if params_mode == "overlay" else [],
        }
        payload_parity["record_digest"] = stable_digest(payload_parity)
        payload_parity_path = config_dir / "boltzgen_parameter_payload_parity.json"
        payload_parity_path.write_text(
            json.dumps(payload_parity, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        consumption_report = self._parameter_consumption_report(
            params=merged_params,
            design_spec_path=spec_path,
            command=command,
            gpu_shards=gpu_shards,
        )
        consumption_path = config_dir / "boltzgen_parameter_consumption.json"
        consumption_semantic = {
            "schema_version": 1,
            "fields": _semantic_value(consumption_report.get("fields", {})),
            "design_spec_entities": consumption_report.get("design_spec_entities", 0),
            "template_artifact_digests": dict(merged_params.get("template_artifact_digests") or {}),
        }
        consumption_report["semantic_digest"] = stable_digest(consumption_semantic)
        consumption_path.write_text(json.dumps(consumption_report, ensure_ascii=False, indent=2), encoding="utf-8")
        rendered_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        receipts = [
            {"field": key, "locations": item.get("locations", [])}
            for key, item in consumption_report.get("fields", {}).items()
            if item.get("locations") and item.get("locations") != ["rejected"]
        ]
        execution_plan = finalize_execution_plan(
            initial_plan,
            design_spec=rendered_spec,
            command=command,
            shards=gpu_shards,
            consumer_receipts=receipts,
        ).to_dict()
        execution_plan.update({
            "design_spec": "configs/" + spec_path.name,
            "parameter_consumption": "configs/" + consumption_path.name,
            "executor_derivations": {"gpu_shards": list(gpu_shards or [])},
            "template_artifact_digests": dict(merged_params.get("template_artifact_digests") or {}),
            "consumption_semantic_digest": consumption_report["semantic_digest"],
            "parameter_payload_parity": payload_parity,
        })
        execution_semantic = {
            "schema_version": execution_plan.get("schema_version"),
            "job_id": execution_plan.get("job_id"),
            "resolved_params": _semantic_value(execution_plan.get("resolved_params")),
            "artifact_digests": execution_plan.get("artifact_digests"),
            "template_artifact_digests": execution_plan.get("template_artifact_digests"),
            "parity": execution_plan.get("parity"),
            "consumption_semantic_digest": execution_plan.get("consumption_semantic_digest"),
        }
        execution_plan["semantic_digest"] = stable_digest(execution_semantic)
        execution_plan_path = config_dir / "effective_execution_plan.json"
        execution_plan_path.write_text(json.dumps(execution_plan, ensure_ascii=False, indent=2), encoding="utf-8")

        expected_outputs = {
            "package_dir": str(package_dir),
            "target_file": str(packaged_target),
            "boltzgen_output_dir": str(output_dir),
            "steps_manifest": str(output_dir / "steps.yaml"),
            "log_file": str(log_file),
            "parameter_consumption": str(consumption_path),
            "effective_execution_plan": str(execution_plan_path),
            "parameter_payload_parity": str(payload_parity_path),
            "harness_execution_identity": str(execution_identity_path),
        }
        native_multi_host = native_host_count > 1
        if "design" in steps:
            expected_outputs["intermediate_designs"] = (
                str(output_dir / "host_*" / "gpu_*" / "shard_*" / "intermediate_designs")
                if native_multi_host
                else str(output_dir / "gpu_*" / "intermediate_designs")
                if self._gpu_distribution(merged_params, steps)
                else str(output_dir / "intermediate_designs")
            )
        if "inverse_folding" in steps:
            expected_outputs["inverse_folded_designs"] = (
                str(output_dir / "host_*" / "gpu_*" / "shard_*" / "intermediate_designs_inverse_folded")
                if native_multi_host
                else str(output_dir / "gpu_*" / "intermediate_designs_inverse_folded")
                if self._gpu_distribution(merged_params, steps)
                else str(output_dir / "intermediate_designs_inverse_folded")
            )
        if "filtering" in steps:
            expected_outputs["final_ranked_designs"] = (
                str(output_dir / "host_*" / "gpu_*" / "shard_*" / "final_ranked_designs")
                if native_multi_host
                else str(output_dir / "gpu_*" / "final_ranked_designs")
                if self._gpu_distribution(merged_params, steps)
                else str(output_dir / "final_ranked_designs")
            )
        if "analysis" in steps or "filtering" in steps:
            expected_outputs["analysis_metrics_candidates"] = str(output_dir / "**" / "*.csv")
        if redesign_mask_path is not None:
            expected_outputs["redesign_mask"] = str(redesign_mask_path)
        expected_outputs["result_manifest"] = str(output_dir / "result_manifest.json")
        spec = BoltzGenRunSpec(
            task_id=merged_params.get("task_id", job.job_id),
            job_id=job.job_id,
            design_spec_path=str(spec_path),
            run_script_path=str(script_path),
            output_dir=str(output_dir),
            log_file=str(log_file),
            command=command,
            command_string=command_string,
            expected_outputs=expected_outputs,
            params=payload_params,
            package_dir=str(package_dir),
        )
        (package_dir / "boltzgen_run_manifest.json").write_text(json.dumps(asdict(spec), ensure_ascii=False, indent=2), encoding="utf-8")
        return spec

    @staticmethod
    def _file_sha256(path: Path) -> str:
        import hashlib
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _template_parity_digests(template: Mapping[str, Any], spec_path: Path, mask_path: Optional[Path]) -> Dict[str, str]:
        alignment = dict(template.get("target_alignment") or {})
        transform = dict(template.get("length_transform") or {})
        mapping = dict(template.get("source_to_effective_residue_map") or {})
        return {
            "source": str(template.get("source_digest") or ""),
            "alignment": str(alignment.get("digest") or ""),
            "residue_map": stable_digest(mapping),
            "length_transform": str(transform.get("digest") or ""),
            "design_spec": stable_digest(yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}),
            "inverse_fold_mask": stable_digest(yaml.safe_load(mask_path.read_text(encoding="utf-8")) or {}) if mask_path else "",
        }

    @staticmethod
    def _validate_template_parity(template: Mapping[str, Any], spec_path: Path, mask_path: Optional[Path]) -> None:
        alignment = dict(template.get("target_alignment") or {})
        transform = dict(template.get("length_transform") or {})
        mapping = dict(template.get("source_to_effective_residue_map") or {})
        required = [template.get("source_digest"), alignment.get("digest"), transform.get("digest"), mapping]
        if not all(required) or transform.get("status") not in {"identity", "applied", "validated"}:
            raise ValueError("template_pre_submit_parity:missing_source_alignment_map_or_transform")
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        file_entity = (spec.get("entities") or [{}])[0].get("file", {})
        spec_index = str((((file_entity.get("not_design") or [{}])[0].get("chain") or {}).get("res_index") or ""))
        mask = yaml.safe_load(mask_path.read_text(encoding="utf-8")) if mask_path else {}
        mask_index = str((((((mask or {}).get("restrictions") or {}).get("not_design") or [{}])[0].get("chain") or {}).get("res_index") or ""))
        if not spec_index or spec_index != mask_index:
            raise ValueError(f"template_pre_submit_parity:spec_mask_residue_mismatch:{spec_index}:{mask_index}")

    @staticmethod
    def _parameter_consumption_report(*, params: Mapping[str, Any], design_spec_path: Path, command: List[str], gpu_shards: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        try:
            spec = yaml.safe_load(design_spec_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            spec = {}
        command_tokens = {str(token).lstrip("-") for token in command}
        report: Dict[str, Any] = {"schema_version": 1, "fields": {}}
        for key, value in sorted(dict(params or {}).items()):
            contract = parameter_contract_entry(key) or {"type": "unknown", "consumer": "rejected"}
            locations: List[str] = []
            if key in command_tokens or ("--" + key) in [str(token) for token in command]:
                locations.append("CLI")
            if key in {"target_include", "target_binding_types", "structure_groups", "binder_chain", "binder_sequence", "binder_binding_types", "residue_constraints", "cyclic", "constraints", "total_len", "binder_template", "binder_templates", "binder_structure_groups"}:
                locations.append("design_spec")
            if key in {"expanded_binding_residues", "negative_binding_residues", "auxiliary_hotspots", "binding_residue_provenance", "selection_policy", "effective_intervention_digest"}:
                locations.append("harness_transform")
            if key in {"template_artifact_digests"}:
                locations.append("pre_submit_parity")
            if key in {"round_budget_weight", "round_budget_allocation", "template_conditioned_fraction"}:
                locations.append("allocation")
            if contract.get("type") in {"runtime_resource", "internal_metadata"} and not locations:
                locations.append("runtime" if contract.get("type") == "runtime_resource" else "metadata_only")
            if gpu_shards and key == "diffusion_batch_size":
                locations.append("executor_derived_per_shard")
            report["fields"][key] = {"value": value, "contract": contract, "locations": sorted(set(locations)) or ["rejected"]}
        report["design_spec_entities"] = len(spec.get("entities") or []) if isinstance(spec, Mapping) else 0
        return report

    def _write_design_spec_to_path(self, job: DesignJob, spec_path: Path) -> None:
        # Keep the adapter write and final replacement on the same filesystem so
        # readers never observe a partially copied YAML file.
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tmp_spec_writer_", dir=str(spec_path.parent)) as tmp:
            tmp_job = DesignJob(**{**job.__dict__, "output_dir": tmp})
            tmp_spec = self.adapter.write_design_spec(tmp_job)
            os.replace(str(tmp_spec), str(spec_path))

    @staticmethod
    def _package_input_file(source: Path, target: Path) -> Path:
        """Atomically materialize a self-contained regular package input file."""
        source = source.expanduser().resolve(strict=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink() and DesignSpecAgent._same_file(source, target):
            return target
        if target.exists() and not target.is_symlink() and not target.is_file():
            raise IsADirectoryError(f"refusing to replace package input directory: {target}")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        os.close(descriptor)
        try:
            shutil.copy2(str(source), temporary_name)
            os.replace(temporary_name, str(target))
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        return target

    @staticmethod
    def _same_file(source: Path, target: Path) -> bool:
        if os.path.abspath(str(source)) == os.path.abspath(str(target)):
            return True
        try:
            return os.path.samefile(str(source), str(target))
        except OSError:
            return False

    @staticmethod
    def _remove_package_entry(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            raise IsADirectoryError(f"refusing to replace package input directory: {path}")

    @staticmethod
    def _write_redesign_mask_to_path(job: DesignJob, mask_path: Path) -> Optional[Path]:
        params: Dict[str, Any] = dict(job.params or {})
        template = params.get("binder_template")
        proximity = params.get("binder_template_proximity", 8.0)
        schema = _build_redesign_schema(template or {}, default_proximity=float(proximity))
        if schema is None:
            return None
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.write_text(yaml.safe_dump(schema), encoding="utf-8")
        return mask_path

    def _build_packaged_command(
        self,
        *,
        spec_path: Path,
        output_dir: Path,
        params: Mapping[str, Any],
        redesign_mask_path: Optional[Path] = None,
    ) -> List[str]:
        return render_boltzgen_command(
            spec_path=spec_path,
            output_dir=output_dir,
            params=params,
            redesign_mask_path=redesign_mask_path,
        )

    def _ensure_complete_pipeline(self, command: List[str], params: Mapping[str, Any]) -> List[str]:
        command = list(map(str, command))
        steps = self._effective_steps(params)
        cleaned: List[str] = []
        i = 0
        step_names = set(self.DEFAULT_FULL_STEPS + ["affinity"])
        while i < len(command):
            if command[i] == "--steps":
                i += 1
                while i < len(command) and command[i] in step_names:
                    i += 1
                continue
            cleaned.append(command[i])
            i += 1
        cleaned += ["--steps", *steps]
        return cleaned

    def _effective_steps(self, params: Mapping[str, Any]) -> List[str]:
        analysis_location = str(params.get("analysis_location", "local")).lower()
        # taiji/remote: full pipeline on the cluster. inline/direct: each GPU shard
        # continues through analysis+filtering on the same host. local: GPU-only,
        # with a separate run_boltzgen_analysis_local.sh for a later pass.
        run_analysis_in_job = analysis_location in {"taiji", "remote", "inline", "direct"} or params.get("run_analysis_on_taiji")
        if run_analysis_in_job:
            steps = list(params.get("steps") or self.DEFAULT_FULL_STEPS)
        else:
            steps = [step for step in list(params.get("steps") or self.DEFAULT_GPU_STEPS) if step not in {"analysis", "filtering"}]
            if not steps:
                steps = list(self.DEFAULT_GPU_STEPS)
        if run_analysis_in_job and "filtering" not in steps:
            steps.append("filtering")
        return steps

    @staticmethod
    def _local_checkpoint_paths(params: Mapping[str, Any], steps: List[str]) -> List[str]:
        values: List[Any] = []
        step_set = set(steps)
        if "design" in step_set:
            design_checkpoints = params.get("design_checkpoints") or []
            if not isinstance(design_checkpoints, (list, tuple)):
                design_checkpoints = [design_checkpoints]
            values.extend(design_checkpoints)
        if "inverse_folding" in step_set or params.get("only_inverse_fold"):
            values.append(params.get("inverse_fold_checkpoint"))
        if "folding" in step_set or "design_folding" in step_set:
            values.append(params.get("folding_checkpoint"))
        if "affinity" in step_set or str(params.get("protocol", "")).lower() == "protein-small_molecule":
            values.append(params.get("affinity_checkpoint"))
        return [str(value) for value in values if value and not str(value).startswith("huggingface:")]

    @staticmethod
    def _gpu_distribution(params: Mapping[str, Any], steps: List[str]) -> Optional[Dict[str, Any]]:
        if "design" not in set(steps) or params.get("reuse"):
            return None
        if params.get("disable_gpu_distribution") or params.get("disable_gpu_sharding"):
            return None
        devices = max(1, int(params.get("devices") or 1))
        mode = str(params.get("taiji_multi_host_mode") or "native").strip().lower()
        mode = {"unified": "native", "fanout": "split_jobs", "split": "split_jobs"}.get(mode, mode)
        host_count = max(1, int(params.get("host_count") or params.get("taiji_submit_host_num") or 1))
        if mode != "native":
            host_count = 1
        num_designs = max(1, int(params.get("num_designs", params.get("num_designs_per_round", 1)) or 1))
        cap = max(1, int(params.get("max_binders_per_round") or num_designs))
        num_designs = min(num_designs, cap)
        lengths = sorted({int(x) for x in (params.get("binder_lengths") or []) if int(x) > 0})
        if devices * host_count <= 1 or (num_designs <= 1 and host_count <= 1):
            return None
        return {
            "devices": devices,
            "host_count": host_count,
            "num_designs": num_designs,
            "max_binders_per_round": cap,
            "lengths": lengths,
        }

    @staticmethod
    def _build_gpu_shards(
        *,
        lengths: List[int],
        total_designs: int,
        gpu_count: int,
        host_count: int = 1,
    ) -> List[Dict[str, Any]]:
        """Split a round's design budget across binder lengths and GPUs.

        Produces one shard per (length-chunk) work unit. The round budget is first
        divided across the requested binder lengths, then the largest units are
        split until there are at least ``gpu_count`` units so every GPU stays busy
        (this reproduces the previous single-length behaviour: one length + N GPUs
        -> N equal chunks). Units are finally packed onto GPUs with a largest-first
        greedy balance so per-GPU design load is even. Each shard runs as its own
        ``boltzgen run`` invocation (one fixed length per spec).
        """
        clean_lengths = sorted({int(x) for x in lengths if int(x) > 0})
        if not clean_lengths:
            clean_lengths = [0]
        total = max(1, int(total_designs))
        gpus_per_host = max(1, int(gpu_count))
        host_count = max(1, int(host_count))
        worker_count = gpus_per_host * host_count

        n_len = len(clean_lengths)
        base = total // n_len
        rem = total - base * n_len
        units: List[List[int]] = []  # [length, num_designs]
        for i, length in enumerate(clean_lengths):
            n = base + (1 if i < rem else 0)
            if n > 0:
                units.append([length, n])
        if not units:
            units = [[clean_lengths[0], total]]

        # Split the largest units until we have at least one per GPU (only when the
        # unit still has more than a single design to give).
        while len(units) < worker_count:
            idx = max(range(len(units)), key=lambda j: units[j][1])
            if units[idx][1] <= 1:
                break
            length, n = units[idx]
            half = n // 2
            units[idx] = [length, n - half]
            units.append([length, half])

        # Largest-first greedy assignment to balance per-GPU design load.
        gpu_load = [0] * worker_count
        assigned: List[Dict[str, Any]] = []
        for length, n in sorted(units, key=lambda u: -u[1]):
            worker = min(range(worker_count), key=lambda k: gpu_load[k])
            gpu_load[worker] += n
            assigned.append({
                # Interleave global workers across hosts so the largest shards do
                # not all land on host 0 when loads are initially tied.
                "host": worker % host_count,
                "gpu": worker // host_count,
                "global_worker": worker,
                "length": int(length),
                "num_designs": int(n),
            })

        assigned.sort(key=lambda s: (s["global_worker"], -s["num_designs"], s["length"]))
        for shard_index, shard in enumerate(assigned):
            shard["shard_index"] = shard_index
        logical_cursor = 0
        for shard in assigned:
            shard["logical_ordinal_start"] = logical_cursor
            logical_cursor += int(shard["num_designs"])
            shard["logical_ordinal_end"] = logical_cursor
        if sum(int(shard["num_designs"]) for shard in assigned) != total:
            raise ValueError("cluster shard plan does not preserve the total design budget")
        return assigned

    def _write_per_length_specs(self, model_job: DesignJob, config_dir: Path, lengths: List[int]) -> Dict[int, str]:
        """Write one BoltzGen design spec per binder length and return length->relpath.

        BoltzGen encodes a single fixed binder length per design spec
        (``protein.sequence: <int>``), so running multiple lengths inside one task
        requires one spec per length. When a structure-redesign template is active
        the spec is length-independent, so the per-length specs are identical; this
        is harmless and keeps the fan-out logic uniform.
        """
        specs: Dict[int, str] = {}
        for length in sorted({int(x) for x in lengths if int(x) > 0}) or [int(model_job.binder_length)]:
            spec_path = config_dir / f"boltzgen_design_spec_len{length}.yaml"
            length_job = DesignJob(**{**model_job.__dict__, "binder_length": int(length)})
            self._write_design_spec_to_path(length_job, spec_path)
            specs[int(length)] = "configs/" + spec_path.name
        return specs

    @staticmethod
    def _bool_param(params: Mapping[str, Any], *keys: str, default: bool = False) -> bool:
        for key in keys:
            if key not in params:
                continue
            value = params.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "y", "on", "enabled"}:
                return True
            if text in {"0", "false", "no", "n", "off", "disabled", ""}:
                return False
        return default

    @staticmethod
    def _heartbeat_seconds(params: Mapping[str, Any]) -> int:
        raw = (
            params.get("log_heartbeat_seconds")
            or params.get("boltzgen_log_heartbeat_seconds")
            or params.get("heartbeat_seconds")
            or 360
        )
        try:
            seconds = int(raw)
        except (TypeError, ValueError):
            seconds = 360
        # Detailed logging should confirm liveness at least once every 6 minutes.
        return min(360, max(1, seconds))

    @staticmethod
    def _replace_single_value_option(command: List[str], flag: str, value: str) -> List[str]:
        updated: List[str] = []
        i = 0
        replaced = False
        while i < len(command):
            updated.append(str(command[i]))
            if command[i] == flag:
                updated.append(value)
                i += 2
                replaced = True
                continue
            i += 1
        if not replaced:
            updated.extend([flag, value])
        return updated

    @staticmethod
    def _replace_positional_spec(command: List[str], value: str) -> List[str]:
        """Replace the positional design-spec path (the arg right after ``run``)."""
        updated = [str(x) for x in command]
        for i in range(len(updated) - 1):
            if updated[i] == "run":
                updated[i + 1] = value
                break
        return updated

    @staticmethod
    def _gpu_shard_command_string(command: List[str]) -> str:
        shard_command = DesignSpecAgent._replace_positional_spec(command, "__SHARD_SPEC__")
        shard_command = DesignSpecAgent._replace_single_value_option(shard_command, "--num_designs", "__SHARD_NUM_DESIGNS__")
        shard_command = DesignSpecAgent._replace_single_value_option(shard_command, "--diffusion_batch_size", "__SHARD_DIFFUSION_BATCH_SIZE__")
        shard_command = DesignSpecAgent._replace_single_value_option(shard_command, "--output", "__SHARD_OUTPUT__")
        shard_command = DesignSpecAgent._replace_single_value_option(shard_command, "--devices", "1")
        command_string = " ".join(shlex.quote(str(x)) for x in shard_command)
        return (
            command_string
            .replace("__SHARD_SPEC__", '"$SHARD_SPEC"')
            .replace("__SHARD_NUM_DESIGNS__", '"$SHARD_NUM_DESIGNS"')
            .replace("__SHARD_DIFFUSION_BATCH_SIZE__", '"$SHARD_DIFFUSION_BATCH_SIZE"')
            .replace("__SHARD_OUTPUT__", '"$SHARD_OUTPUT"')
        )

    @staticmethod
    def _render_host_rank_block(host_count: int, gpus_per_host: int) -> str:
        host_count = max(1, int(host_count))
        gpus_per_host = max(1, int(gpus_per_host))
        if host_count == 1:
            return f"""EXPECTED_HOST_COUNT=1
EXPECTED_GPUS_PER_HOST={gpus_per_host}
HOST_RANK=0
HOST_RANK_SOURCE=single_host
HOST_TAG=00
COORDINATION_RUN_TOKEN="${{HARNESS_RUN_TOKEN:-single_host}}"
COORDINATION_RUN_TOKEN="$(printf '%s' "$COORDINATION_RUN_TOKEN" | tr -c 'A-Za-z0-9_.-' '_')"
COORDINATION_DIR="outputs/boltzgen_output/.coordination/$COORDINATION_RUN_TOKEN"
mkdir -p "$COORDINATION_DIR"
"""
        return f"""EXPECTED_HOST_COUNT={host_count}
EXPECTED_GPUS_PER_HOST={gpus_per_host}
HOST_RANK=""
HOST_RANK_SOURCE=""
DETECTED_WORLD_SIZE=""
if [[ -n "${{OMPI_COMM_WORLD_RANK:-}}" ]]; then
  HOST_RANK="$OMPI_COMM_WORLD_RANK"
  DETECTED_WORLD_SIZE="${{OMPI_COMM_WORLD_SIZE:-}}"
  HOST_RANK_SOURCE=ompi
elif [[ -n "${{PMI_RANK:-}}" ]]; then
  HOST_RANK="$PMI_RANK"
  DETECTED_WORLD_SIZE="${{PMI_SIZE:-}}"
  HOST_RANK_SOURCE=pmi
elif [[ -n "${{NODE_RANK:-}}" ]]; then
  HOST_RANK="$NODE_RANK"
  DETECTED_WORLD_SIZE="${{WORLD_SIZE:-}}"
  HOST_RANK_SOURCE=node_rank
elif [[ -n "${{RANK:-}}" ]]; then
  HOST_RANK="$RANK"
  DETECTED_WORLD_SIZE="${{WORLD_SIZE:-}}"
  HOST_RANK_SOURCE=rank
fi

COORDINATION_RUN_TOKEN="${{HARNESS_RUN_TOKEN:-native_multi_host}}"
COORDINATION_RUN_TOKEN="$(printf '%s' "$COORDINATION_RUN_TOKEN" | tr -c 'A-Za-z0-9_.-' '_')"
COORDINATION_DIR="outputs/boltzgen_output/.coordination/$COORDINATION_RUN_TOKEN"
mkdir -p "$COORDINATION_DIR/hosts" "$COORDINATION_DIR/rank_claims" "$COORDINATION_DIR/status"
if [[ -z "$HOST_RANK" ]]; then
  HOST_RANK_SOURCE=ceph_hostname_registry
  HOST_ID="${{HOSTNAME:-$(hostname)}}"
  HOST_ID="$(printf '%s' "$HOST_ID" | tr -c 'A-Za-z0-9_.-' '_')"
  if ! mkdir "$COORDINATION_DIR/hosts/$HOST_ID" 2>/dev/null; then
    echo "[HARNESS][ERROR] duplicate hostname registration: $HOST_ID" >&2
    exit 18
  fi
  printf '%s\\n' "$HOST_ID" > "$COORDINATION_DIR/hosts/$HOST_ID/hostname"
  registration_deadline=$(( $(date +%s) + ${{HARNESS_HOST_REGISTRATION_TIMEOUT:-600}} ))
  while true; do
    shopt -s nullglob
    registered_hosts=("$COORDINATION_DIR"/hosts/*)
    shopt -u nullglob
    if (( ${{#registered_hosts[@]}} == EXPECTED_HOST_COUNT )); then
      break
    fi
    if (( ${{#registered_hosts[@]}} > EXPECTED_HOST_COUNT )); then
      echo "[HARNESS][ERROR] registered host count exceeds expected count" >&2
      exit 19
    fi
    if (( $(date +%s) >= registration_deadline )); then
      echo "[HARNESS][ERROR] timed out waiting for $EXPECTED_HOST_COUNT host registrations" >&2
      exit 20
    fi
    sleep 2
  done
  mapfile -t REGISTERED_HOSTS < <(
    for registered_path in "${{registered_hosts[@]}}"; do basename "$registered_path"; done | LC_ALL=C sort
  )
  for registered_index in "${{!REGISTERED_HOSTS[@]}}"; do
    if [[ "${{REGISTERED_HOSTS[$registered_index]}}" == "$HOST_ID" ]]; then
      HOST_RANK="$registered_index"
      break
    fi
  done
fi

if ! [[ "$HOST_RANK" =~ ^[0-9]+$ ]] || (( HOST_RANK < 0 || HOST_RANK >= EXPECTED_HOST_COUNT )); then
  echo "[HARNESS][ERROR] invalid host rank '$HOST_RANK' for host_count=$EXPECTED_HOST_COUNT" >&2
  exit 21
fi
if [[ -n "$DETECTED_WORLD_SIZE" ]] && [[ "$DETECTED_WORLD_SIZE" != "$EXPECTED_HOST_COUNT" ]]; then
  echo "[HARNESS][ERROR] detected world size $DETECTED_WORLD_SIZE does not match expected $EXPECTED_HOST_COUNT" >&2
  exit 22
fi
HOST_TAG="$(printf '%02d' "$HOST_RANK")"
if ! mkdir "$COORDINATION_DIR/rank_claims/rank_$HOST_TAG" 2>/dev/null; then
  echo "[HARNESS][ERROR] duplicate host rank claim: $HOST_RANK" >&2
  exit 23
fi
printf '%s\\n' "${{HOSTNAME:-unknown}}" > "$COORDINATION_DIR/rank_claims/rank_$HOST_TAG/hostname"
"""

    @staticmethod
    def _render_project_run_script(
        *,
        command: List[str],
        log_file: Path,
        package_dir: Path,
        target_file: Path,
        checkpoint_paths: List[str],
        cache_dir: str,
        moldir: str,
        gpu_shards: Optional[List[Dict[str, Any]]],
        host_count: int,
        gpus_per_host: int,
        silence_logging: bool,
        heartbeat_seconds: int,
        conda_base: str,
        conda_env_name: str,
    ) -> str:
        command_string = " ".join(shlex.quote(str(x)) for x in command)
        checkpoint_array = " ".join(shlex.quote(str(path)) for path in checkpoint_paths)
        first_checkpoint = checkpoint_paths[0] if checkpoint_paths else ""
        checkpoint_dir = (
            str(PurePosixPath(first_checkpoint).parent)
            if first_checkpoint.startswith("/")
            else str(Path(first_checkpoint).parent)
        ) if first_checkpoint else "/aceph/daweihuang/program/boltzgen/checkpoints"
        silence_default = "1" if silence_logging else "0"
        host_rank_block = DesignSpecAgent._render_host_rank_block(host_count, gpus_per_host)
        if host_count > 1:
            log_file_assignment = 'LOG_FILE="logs/host_${HOST_TAG}/boltzgen_full.log"'
        else:
            log_file_assignment = f'LOG_FILE="{log_file.relative_to(package_dir)}"'
        if gpu_shards:
            result_mode = "native_multi_host" if host_count > 1 else "single_host_gpu"
            gpu_distribution_block = DesignSpecAgent._render_gpu_distribution_block(
                shards=gpu_shards,
                shard_command=DesignSpecAgent._gpu_shard_command_string(command),
                host_count=host_count,
                gpus_per_host=gpus_per_host,
            )
        else:
            result_mode = "single_process"
            gpu_distribution_block = f"""echo "[HARNESS] command={command_string}"
set +e
binder_started_at="$(date +%s)"
if harness_silence_log_enabled; then
  {command_string} 2>&1 | sed -u 's/^/[BOLTZGEN] /' >> "$LOG_FILE"
  status=${{PIPESTATUS[0]}}
else
  {command_string} 2>&1 | sed -u 's/^/[BOLTZGEN] /'
  status=${{PIPESTATUS[0]}}
fi
binder_elapsed=$(( $(date +%s) - binder_started_at ))
echo "[HARNESS] binder_design elapsed_seconds=$binder_elapsed elapsed_human=$(harness_format_elapsed "$binder_elapsed") status=$status"
set -e
"""
        return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PACKAGE_DIR="$(pwd)"
CONDA_BASE="${{CONDA_BASE:-{conda_base}}}"
CONDA_ENV_NAME="${{CONDA_ENV_NAME:-{conda_env_name}}}"
PROJECT_ROOT="${{PROJECT_ROOT:-/aceph/daweihuang/dataset/proteo_benchmark}}"
CHECKPOINT_DIR="${{CHECKPOINT_DIR:-{checkpoint_dir}}}"
CACHE_DIR="${{CACHE_DIR:-{cache_dir}}}"
MOLDIR="${{MOLDIR:-{moldir}}}"
LOCAL_CHECKPOINTS=({checkpoint_array})
TARGET_FILE="{target_file.relative_to(package_dir)}"
BOLTZGEN_SILENCE_LOG="${{BOLTZGEN_SILENCE_LOG:-{silence_default}}}"
BOLTZGEN_LOG_HEARTBEAT_SECONDS="${{BOLTZGEN_LOG_HEARTBEAT_SECONDS:-{heartbeat_seconds}}}"
HARNESS_STARTED_AT="$(date +%s)"
HARNESS_START_TIME="$(date -Is)"
HARNESS_HEARTBEAT_PID=""
HARNESS_RESULT_MODE={result_mode}
export BOLTZGEN_LINEAGE_CAPABILITY="producer_unavailable_or_missing"
export BOLTZGEN_RUNTIME_VERSION=""
export BOLTZGEN_RUNTIME_DIGEST=""

mkdir -p outputs/boltzgen_output
{host_rank_block}
{log_file_assignment}
mkdir -p "$(dirname "$LOG_FILE")" outputs/boltzgen_output
if (( EXPECTED_HOST_COUNT == 1 )) || [[ "$HOST_RANK" == "0" ]]; then
  rm -f outputs/boltzgen_output/result_manifest.json
fi
exec > >(tee -a "$LOG_FILE") 2>&1

harness_silence_log_enabled() {{
  case "$BOLTZGEN_SILENCE_LOG" in
    1|true|TRUE|yes|YES|on|ON|enabled|ENABLED|silent|SILENT) return 0 ;;
    *) return 1 ;;
  esac
}}

harness_normalize_heartbeat_seconds() {{
  if ! [[ "$BOLTZGEN_LOG_HEARTBEAT_SECONDS" =~ ^[0-9]+$ ]]; then
    BOLTZGEN_LOG_HEARTBEAT_SECONDS=360
  fi
  if (( BOLTZGEN_LOG_HEARTBEAT_SECONDS < 1 || BOLTZGEN_LOG_HEARTBEAT_SECONDS > 360 )); then
    BOLTZGEN_LOG_HEARTBEAT_SECONDS=360
  fi
}}

harness_log_file_with_prefix() {{
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    echo "[HARNESS][VERBOSE] $label start path=$path"
    sed -u "s/^/[HARNESS][$label] /" "$path"
    echo "[HARNESS][VERBOSE] $label end path=$path"
  fi
}}

harness_preflight_boltzgen_runtime() {{
  local runtime_json="configs/boltzgen_runtime_capability.json"
  local cli_path
  cli_path="$(command -v boltzgen)" || {{
    echo "[HARNESS][ERROR] boltzgen CLI not found" >&2
    return 127
  }}
  python - "$runtime_json" "$cli_path" <<'HARNESS_RUNTIME_PREFLIGHT'
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
from pathlib import Path

out = Path(sys.argv[1])
cli_path = Path(sys.argv[2]).resolve()
result = {{"schema_version": 2, "status": "invalid", "failures": []}}
try:
    boltzgen = importlib.import_module("boltzgen")
    writer = importlib.import_module("boltzgen.task.predict.writer")
    analyzer = importlib.import_module("boltzgen.task.analyze.analyze")
    filter_module = importlib.import_module("boltzgen.task.filter.filter")
    files = [Path(inspect.getsourcefile(module) or "").resolve() for module in (writer, analyzer, filter_module)]
    if not all(path.is_file() for path in files):
        raise RuntimeError("core_runtime_source_missing")
    package_dir = Path.cwd().resolve()
    forbidden_roots = [package_dir / "runtime" / "boltzgen_src", package_dir]
    for module_path in files:
        if any(module_path == root or root in module_path.parents for root in forbidden_roots):
            raise RuntimeError(f"submitted_package_source_active:{{module_path}}")
    distribution = importlib.metadata.distribution("boltzgen")
    executable = Path(sys.executable).resolve()
    cli_real = cli_path.resolve()
    if executable.parent not in cli_real.parents and cli_real.parent != executable.parent:
        raise RuntimeError(f"cli_python_environment_mismatch:{{cli_real}}:{{executable}}")
    digest = hashlib.sha256()
    for path in sorted(files, key=str):
        digest.update(path.read_bytes())
    torch_info = {{}}
    try:
        import torch
        torch_info = {{
            "version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda or ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }}
    except Exception as exc:
        torch_info = {{"error": f"{{type(exc).__name__}}:{{exc}}"}}
    result.update({{
        "status": "validated",
        "runtime_source": "image",
        "version": str(distribution.version),
        "digest": digest.hexdigest(),
        "python_executable": str(executable),
        "python_version": platform.python_version(),
        "cli_path": str(cli_real),
        "package_file": str(Path(inspect.getsourcefile(boltzgen) or "").resolve()),
        "module_files": [str(path) for path in files],
        "torch": torch_info,
        "image_reference": str(os.environ.get("TAIJI_IMAGE_REFERENCE") or os.environ.get("IMAGE_FULL_NAME") or ""),
        "image_digest": str(os.environ.get("TAIJI_IMAGE_DIGEST") or ""),
    }})
except Exception as exc:
    result["failures"].append(f"{{type(exc).__name__}}:{{exc}}")
out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
if result["status"] != "validated":
    raise SystemExit(42)
HARNESS_RUNTIME_PREFLIGHT
  local preflight_status=$?
  if (( preflight_status != 0 )); then
    echo "[HARNESS][ERROR] BoltzGen runtime preflight failed status=$preflight_status" >&2
    return "$preflight_status"
  fi
  eval "$(python - "$runtime_json" <<'HARNESS_RUNTIME_EXPORTS'
import json, shlex, sys
value=json.load(open(sys.argv[1], encoding='utf-8'))
print('export BOLTZGEN_LINEAGE_CAPABILITY=not_provided_by_runtime')
print('export BOLTZGEN_RUNTIME_VERSION=' + shlex.quote(str(value.get('version') or '')))
print('export BOLTZGEN_RUNTIME_DIGEST=' + shlex.quote(str(value.get('digest') or '')))
HARNESS_RUNTIME_EXPORTS
)"
  echo "[HARNESS] boltzgen_runtime_source=image runtime_version=$BOLTZGEN_RUNTIME_VERSION runtime_digest=$BOLTZGEN_RUNTIME_DIGEST"
}}

harness_log_runtime_context() {{
  echo "[HARNESS][VERBOSE] boltzgen_cli=$(command -v boltzgen || true)"
  echo "[HARNESS][VERBOSE] conda_env=$CONDA_ENV_NAME cuda_visible_devices=${{CUDA_VISIBLE_DEVICES:-}}"
  echo "[HARNESS][VERBOSE] command_start"
  cat <<'HARNESS_BOLTZGEN_COMMAND'
{command_string}
HARNESS_BOLTZGEN_COMMAND
  echo "[HARNESS][VERBOSE] command_end"
  harness_log_file_with_prefix "DESIGN_SPEC" "configs/boltzgen_design_spec.yaml"
  harness_log_file_with_prefix "PARAMETER_PLAN" "configs/boltzgen_parameter_plan.yaml"
  harness_log_file_with_prefix "REDESIGN_MASK" "configs/boltzgen_redesign_mask.yaml"
}}

harness_start_heartbeat() {{
  (
    while true; do
      sleep "$BOLTZGEN_LOG_HEARTBEAT_SECONDS" || exit 0
      now="$(date +%s)"
      elapsed=$(( now - HARNESS_STARTED_AT ))
      if harness_silence_log_enabled; then
        {{
          echo "[HARNESS][HEARTBEAT] time=$(date -Is) elapsed_seconds=$elapsed status=running package_dir=$PACKAGE_DIR output_dir=outputs/boltzgen_output"
          for gpu_log in logs/boltzgen_gpu_*.log logs/host_"$HOST_TAG"/boltzgen_gpu_*.log; do
            [[ -e "$gpu_log" ]] || continue
            lines="$(wc -l < "$gpu_log" 2>/dev/null || echo 0)"
            bytes="$(wc -c < "$gpu_log" 2>/dev/null || echo 0)"
            echo "[HARNESS][HEARTBEAT] gpu_log=$gpu_log lines=$lines bytes=$bytes"
          done
        }} >> "$LOG_FILE"
      else
        echo "[HARNESS][HEARTBEAT] time=$(date -Is) elapsed_seconds=$elapsed status=running package_dir=$PACKAGE_DIR output_dir=outputs/boltzgen_output"
        for gpu_log in logs/boltzgen_gpu_*.log logs/host_"$HOST_TAG"/boltzgen_gpu_*.log; do
          [[ -e "$gpu_log" ]] || continue
          lines="$(wc -l < "$gpu_log" 2>/dev/null || echo 0)"
          bytes="$(wc -c < "$gpu_log" 2>/dev/null || echo 0)"
          echo "[HARNESS][HEARTBEAT] gpu_log=$gpu_log lines=$lines bytes=$bytes"
        done
      fi
    done
  ) &
  HARNESS_HEARTBEAT_PID="$!"
}}

harness_stop_heartbeat() {{
  if [[ -n "$HARNESS_HEARTBEAT_PID" ]]; then
    kill "$HARNESS_HEARTBEAT_PID" >/dev/null 2>&1 || true
    wait "$HARNESS_HEARTBEAT_PID" 2>/dev/null || true
  fi
}}

harness_format_elapsed() {{
  local seconds="$1"
  if ! [[ "$seconds" =~ ^[0-9]+$ ]]; then
    seconds=0
  fi
  local hours=$(( seconds / 3600 ))
  local minutes=$(( (seconds % 3600) / 60 ))
  local secs=$(( seconds % 60 ))
  if (( hours > 0 )); then
    printf '%dh%dm%ds' "$hours" "$minutes" "$secs"
  elif (( minutes > 0 )); then
    printf '%dm%ds' "$minutes" "$secs"
  else
    printf '%ds' "$secs"
  fi
}}

harness_record_elapsed() {{
  HARNESS_ENDED_AT="$(date +%s)"
  HARNESS_END_TIME="$(date -Is)"
  HARNESS_ELAPSED_SECONDS=$(( HARNESS_ENDED_AT - HARNESS_STARTED_AT ))
  if (( HARNESS_ELAPSED_SECONDS < 0 )); then
    HARNESS_ELAPSED_SECONDS=0
  fi
  export HARNESS_START_TIME HARNESS_END_TIME HARNESS_ELAPSED_SECONDS HARNESS_STARTED_AT
}}

harness_write_result_manifest() {{
  local result_status="$1"
  local result_mode="${{HARNESS_RESULT_MODE:-unknown}}"
  HARNESS_RESULT_STATUS="$result_status" \
  HARNESS_RESULT_MODE="$result_mode" \
  HARNESS_RESULT_HOST_COUNT="$EXPECTED_HOST_COUNT" \
  HARNESS_RESULT_GPUS_PER_HOST="$EXPECTED_GPUS_PER_HOST" \
  HOST_RANK="${{HOST_RANK:-0}}" \
  HARNESS_ELAPSED_SECONDS="${{HARNESS_ELAPSED_SECONDS:-}}" \
  HARNESS_START_TIME="${{HARNESS_START_TIME:-}}" \
  HARNESS_END_TIME="${{HARNESS_END_TIME:-}}" \
  python - <<'HARNESS_RESULT_MANIFEST'
import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

root = Path("outputs/boltzgen_output")
root.mkdir(parents=True, exist_ok=True)
identity_path = Path("configs/harness_execution_identity.json")
identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else {{}}
host_count = int(os.environ["HARNESS_RESULT_HOST_COUNT"])
gpus_per_host = int(os.environ["HARNESS_RESULT_GPUS_PER_HOST"])
host_rank = int(os.environ.get("HOST_RANK", "0") or 0)
result_status = int(os.environ["HARNESS_RESULT_STATUS"])
result_mode = os.environ["HARNESS_RESULT_MODE"]
elapsed_raw = os.environ.get("HARNESS_ELAPSED_SECONDS")
try:
    elapsed_seconds = int(elapsed_raw) if elapsed_raw not in (None, "") else None
except (TypeError, ValueError):
    elapsed_seconds = None
start_time = os.environ.get("HARNESS_START_TIME") or ""
end_time = os.environ.get("HARNESS_END_TIME") or ""


def atomic_json_write(target, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{{target.name}}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(target))
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def collect_files(prefix=None):
    scan_root = root / prefix if prefix else root
    if not scan_root.exists():
        return []
    files = []
    for directory, dirnames, filenames in os.walk(str(scan_root)):
        dirnames[:] = [name for name in dirnames if name != ".coordination"]
        directory_path = Path(directory)
        for name in filenames:
            if name in {{"result_manifest.json", "shard_result_manifest.json"}} or name.startswith((".result_manifest.", ".shard_result_manifest.")):
                continue
            files.append((directory_path / name).relative_to(root).as_posix())
    return sorted(files)


def read_csv_rows(paths):
    rows = []
    for relative in paths:
        try:
            with (root / relative).open(newline="", encoding="utf-8") as handle:
                for ordinal, row in enumerate(csv.DictReader(handle)):
                    rows.append({{**dict(row), "_metrics_relative_path": relative, "_metrics_row_ordinal": ordinal}})
        except (OSError, csv.Error):
            pass
    return rows


def truthy(value):
    return str(value or "").strip().lower() in {{"1", "true", "yes", "y", "pass", "passed"}}


def decorate_manifest(manifest):
    files = list(manifest.get("files") or [])
    unfiltered_metrics = [path for path in files if path.endswith("all_designs_metrics.csv")]
    selected_metrics = [path for path in files if "final_ranked_designs/" in path and Path(path).name.startswith("final_designs_metrics")]
    unfiltered_rows = read_csv_rows(unfiltered_metrics)
    selected_rows = read_csv_rows(selected_metrics)
    pass_flags_present = any("pass_filters" in row for row in selected_rows)
    filter_pass_count = sum(1 for row in selected_rows if truthy(row.get("pass_filters"))) if pass_flags_present else 0
    manifest["unfiltered_metric_count"] = len(unfiltered_rows)
    manifest["selected_metric_count"] = len(selected_rows)
    manifest["filter_pass_count"] = filter_pass_count
    manifest["selected_failed_filter_count"] = len(selected_rows) - filter_pass_count if pass_flags_present else None
    manifest["filter_pass_count_status"] = "evaluated" if pass_flags_present else "unavailable"
    manifest["execution_status"] = "success" if int((manifest.get("status") or {{}}).get("code", 1)) == 0 else "failed"
    manifest["quality_status"] = (
        "no_filter_pass"
        if manifest["execution_status"] == "success"
        and manifest["filter_pass_count_status"] == "evaluated"
        and manifest["filter_pass_count"] == 0
        and manifest["selected_metric_count"] > 0
        else "evaluated"
    )
    artifacts = []
    for relative in files:
        path = root / relative
        if not path.is_file():
            continue
        name = Path(relative).name
        is_metric = path.suffix.lower() == ".csv" and "metrics" in name.lower()
        is_structure = path.suffix.lower() in {{".cif", ".pdb", ".mmcif"}}
        is_steps_manifest = relative == "steps.yaml"
        if is_metric or is_structure or is_steps_manifest:
            artifacts.append({{
                "path": relative,
                "kind": "metrics" if is_metric else "structure" if is_structure else "steps_manifest",
                "authoritative": is_steps_manifest,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "job_id": identity.get("job_id"),
                "arm_id": identity.get("arm_id"),
                "logical_branch_id": identity.get("logical_branch_id"),
            }})
    manifest["artifacts"] = artifacts
    semantic_payload = {{
        "schema_version": manifest["schema_version"],
        "collection_mode": manifest["collection_mode"],
        "status": manifest["status"],
        "files": files,
    }}
    manifest["semantic_digest"] = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    runtime_path = Path("configs/boltzgen_runtime_capability.json")
    if runtime_path.is_file():
        try:
            manifest["runtime"] = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            manifest["runtime"] = {{"status": "invalid", "failures": [f"{{type(exc).__name__}}:{{exc}}"]}}
    return manifest


def base_manifest(collection_mode, files, status_code):
    return {{
        "schema_version": 6,
        "contract": {{"name": "binder_harness_result_manifest", "version": 1}},
        "collection_mode": collection_mode,
        "candidate_attribution": False,
        "attribution_scope": "job",
        "stage_classification": False,
        "identity": identity,
        "status": {{"code": int(status_code)}},
        "mode": result_mode,
        "host_count": host_count,
        "gpus_per_host": gpus_per_host,
        "elapsed_seconds": elapsed_seconds,
        "start_time": start_time,
        "end_time": end_time,
        "files": sorted(files),
        "run_manifest": "../../boltzgen_run_manifest.json",
    }}


plan = None
plan_path = Path("configs/cluster_shard_plan.json")
if plan_path.is_file():
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        plan = None

if host_count == 1:
    single_files = collect_files()
    manifest = base_manifest("round_aggregate", single_files, result_status)
    if result_status == 0:
        manifest["required_artifacts"] = ["steps.yaml"]
        manifest["authoritative"] = {{"inventory": "files", "entities": "artifacts"}}
    if plan:
        manifest["total_designs"] = int(plan.get("total_designs", 0))
        manifest["shards"] = list(plan.get("shards") or [])
    atomic_json_write(root / "result_manifest.json", decorate_manifest(manifest))
    raise SystemExit(0)

host_ref = f"host_{{host_rank:02d}}/shard_result_manifest.json"
host_files = collect_files(f"host_{{host_rank:02d}}")
host_manifest = base_manifest("host_shard", host_files, result_status)
host_manifest["host_rank"] = host_rank
if plan:
    host_manifest["shards"] = [
        shard for shard in (plan.get("shards") or []) if int(shard.get("host", -1)) == host_rank
    ]
    host_manifest["total_designs"] = sum(int(shard.get("num_designs", 0)) for shard in host_manifest["shards"])
atomic_json_write(root / host_ref, decorate_manifest(host_manifest))

# Every host publishes its own immutable snapshot. Only rank 0 aggregates, and
# only after all expected snapshots are present and parseable.
if host_rank != 0:
    raise SystemExit(0)

deadline = time.monotonic() + int(os.environ.get("HARNESS_MANIFEST_BARRIER_TIMEOUT", "600"))
shard_manifests = []
for expected_rank in range(host_count):
    ref = f"host_{{expected_rank:02d}}/shard_result_manifest.json"
    path = root / ref
    while True:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if int(value.get("host_rank", -1)) != expected_rank:
                raise ValueError("host rank mismatch")
            shard_manifests.append((ref, value))
            break
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out waiting for valid host result manifest: {{ref}}")
            time.sleep(0.2)

aggregate_files = sorted({{
    relative
    for _, shard_manifest in shard_manifests
    for relative in (shard_manifest.get("files") or [])
}} | {{ref for ref, _ in shard_manifests}} | ({{"steps.yaml"}} if result_status == 0 else set()))
host_statuses = [
    {{"host_rank": int(value["host_rank"]), "code": int((value.get("status") or {{}}).get("code", 1))}}
    for _, value in shard_manifests
]
aggregate_status = next((item["code"] for item in host_statuses if item["code"] != 0), result_status)
manifest = base_manifest("round_aggregate", aggregate_files, aggregate_status)
if aggregate_status == 0:
    manifest["required_artifacts"] = ["steps.yaml"]
    manifest["authoritative"] = {{"inventory": "files", "entities": "artifacts"}}
manifest["shard_manifests"] = [ref for ref, _ in shard_manifests]
manifest["host_statuses"] = host_statuses
if plan:
    manifest["total_designs"] = int(plan.get("total_designs", 0))
    manifest["shards"] = list(plan.get("shards") or [])
atomic_json_write(root / "result_manifest.json", decorate_manifest(manifest))
HARNESS_RESULT_MANIFEST
}}

harness_on_exit() {{
  local exit_status=$?
  trap - EXIT
  harness_stop_heartbeat
  if [[ -z "${{HARNESS_ELAPSED_SECONDS:-}}" ]]; then
    harness_record_elapsed
  fi
  if ! harness_write_result_manifest "$exit_status"; then
    echo "[HARNESS][WARN] failed to write host/result manifest" >&2
  fi
  exit "$exit_status"
}}
trap harness_on_exit EXIT

echo "[HARNESS] package_dir=$PACKAGE_DIR"
echo "[HARNESS] host_rank=$HOST_RANK host_count=$EXPECTED_HOST_COUNT rank_source=$HOST_RANK_SOURCE gpus_per_host=$EXPECTED_GPUS_PER_HOST"
echo "[HARNESS] checkpoint_dir=$CHECKPOINT_DIR"
echo "[HARNESS] cache_dir=$CACHE_DIR"
echo "[HARNESS] moldir=$MOLDIR"
echo "[HARNESS] start_time=$HARNESS_START_TIME"
if [[ ! -s "$TARGET_FILE" ]]; then
  echo "[HARNESS][ERROR] target file missing or empty: $TARGET_FILE" >&2
  exit 11
fi
if (( ${{#LOCAL_CHECKPOINTS[@]}} > 0 )) && [[ ! -f "${{LOCAL_CHECKPOINTS[0]}}" && -n "${{CEPH_SECRET:-}}" ]]; then
  echo "[HARNESS] checkpoint dir is not visible; attempting Ceph mount"
  mkdir -p /aceph/daweihuang
  if ! mountpoint -q /aceph/daweihuang 2>/dev/null; then
    if ! mount -t ceph 11.18.83.17:6789,11.18.83.31:6789,11.18.83.32:6789:/fandiwu/buddy1/daweihuang \
      /aceph/daweihuang -o name=fandiwubuddy1,secret="${{CEPH_SECRET}}"; then
      echo "[HARNESS][ERROR] failed to mount /aceph/daweihuang with CEPH_SECRET" >&2
      exit 14
    fi
  fi
fi
for checkpoint in "${{LOCAL_CHECKPOINTS[@]}}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "[HARNESS][ERROR] checkpoint missing: $checkpoint" >&2
    exit 12
  fi
done
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "[HARNESS][ERROR] cache dir missing: $CACHE_DIR" >&2
  exit 13
fi
if [[ ! -e "$MOLDIR" ]]; then
  echo "[HARNESS][ERROR] moldir missing: $MOLDIR" >&2
  exit 15
fi

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
fi

if ! command -v boltzgen >/dev/null 2>&1; then
  echo "[HARNESS][ERROR] boltzgen CLI not found" >&2
  exit 127
fi
harness_preflight_boltzgen_runtime

harness_normalize_heartbeat_seconds
echo "[HARNESS] detail_logging=enabled heartbeat_seconds=$BOLTZGEN_LOG_HEARTBEAT_SECONDS"
if harness_silence_log_enabled; then
  echo "[HARNESS] silence_logging=enabled detail_logs=log_file_only"
  harness_log_runtime_context >> "$LOG_FILE" 2>&1
else
  echo "[HARNESS] silence_logging=disabled detail_logs=screen_and_log"
  harness_log_runtime_context
fi
harness_start_heartbeat

{gpu_distribution_block}
harness_record_elapsed
echo "[HARNESS] elapsed_seconds=$HARNESS_ELAPSED_SECONDS elapsed_human=$(harness_format_elapsed "$HARNESS_ELAPSED_SECONDS") start_time=$HARNESS_START_TIME end_time=$HARNESS_END_TIME status=$status"
exit "$status"
"""

    @staticmethod
    def _render_gpu_distribution_block(
        *,
        shards: List[Dict[str, Any]],
        shard_command: str,
        host_count: int,
        gpus_per_host: int,
    ) -> str:
        host_count = max(1, int(host_count))
        gpu_count = max(1, int(gpus_per_host))
        total_designs = sum(int(s["num_designs"]) for s in shards)
        lengths = sorted({int(s["length"]) for s in shards})
        if any(int(s["host"]) >= host_count or int(s["gpu"]) >= gpu_count for s in shards):
            raise ValueError("cluster shard plan contains an out-of-range host or GPU")

        def _bash_array(name: str, values: List[str]) -> str:
            joined = " ".join(values)
            return f"{name}=({joined})"

        shard_host = _bash_array("SHARD_HOST", [str(int(s["host"])) for s in shards])
        shard_gpu = _bash_array("SHARD_GPU", [str(int(s["gpu"])) for s in shards])
        shard_len = _bash_array("SHARD_LEN", [str(int(s["length"])) for s in shards])
        shard_ndes = _bash_array("SHARD_NDES", [str(int(s["num_designs"])) for s in shards])
        shard_ord_start = _bash_array("SHARD_ORD_START", [str(int(s["logical_ordinal_start"])) for s in shards])
        shard_ord_end = _bash_array("SHARD_ORD_END", [str(int(s["logical_ordinal_end"])) for s in shards])
        shard_batch = _bash_array("SHARD_BATCH", [str(max(1, int(s.get("executor_diffusion_batch_size", 1)))) for s in shards])
        shard_spec = _bash_array("SHARD_SPEC_PATHS", [shlex.quote(str(s["spec"])) for s in shards])
        lengths_yaml = ", ".join(str(length) for length in lengths)

        return f"""GPU_DISTRIBUTION_ENABLED=1
GPU_COUNT="${{HARNESS_GPUS_PER_HOST:-{gpu_count}}}"
HOST_COUNT="${{HARNESS_HOST_COUNT:-{host_count}}}"
if (( GPU_COUNT != {gpu_count} )); then
  echo "[HARNESS][ERROR] runtime GPU_COUNT=$GPU_COUNT does not match planned gpus_per_host={gpu_count}" >&2
  exit 17
fi
if (( HOST_COUNT != {host_count} )); then
  echo "[HARNESS][ERROR] runtime HOST_COUNT=$HOST_COUNT does not match planned host_count={host_count}" >&2
  exit 24
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$visible_gpu_count" =~ ^[0-9]+$ ]] && (( visible_gpu_count < GPU_COUNT )); then
    echo "[HARNESS][ERROR] only $visible_gpu_count GPUs visible, but $GPU_COUNT are required" >&2
    exit 25
  fi
fi
{shard_host}
{shard_gpu}
{shard_len}
{shard_ndes}
{shard_ord_start}
{shard_ord_end}
{shard_batch}
{shard_spec}
SHARD_TOTAL="${{#SHARD_GPU[@]}}"
if (( ${{#SHARD_HOST[@]}} != SHARD_TOTAL || ${{#SHARD_LEN[@]}} != SHARD_TOTAL || ${{#SHARD_NDES[@]}} != SHARD_TOTAL || ${{#SHARD_BATCH[@]}} != SHARD_TOTAL )); then
  echo "[HARNESS][ERROR] cluster shard plan arrays have inconsistent lengths" >&2
  exit 26
fi
echo "[HARNESS] command_template_start"
cat <<'HARNESS_COMMAND_TEMPLATE'
{shard_command}
HARNESS_COMMAND_TEMPLATE
echo "[HARNESS] command_template_end"
echo "[HARNESS] gpu_distribution=enabled host_rank=$HOST_RANK host_count=$HOST_COUNT gpu_count=$GPU_COUNT shards=$SHARD_TOTAL total_designs={total_designs} lengths=[{lengths_yaml}]"
if (( HOST_COUNT > 1 && HOST_RANK == 0 )); then
  rm -f outputs/boltzgen_output/steps.yaml
fi
pids=()
worker_gpus=()
status=0
# Every MPI pod runs this script, but only executes shards assigned to its host
# rank. Local GPU workers run in parallel and serialize multiple local shards.
for ((g=0; g<GPU_COUNT; g++)); do
  (
    set +e
    worker_status=0
    for idx in "${{!SHARD_GPU[@]}}"; do
      if (( SHARD_HOST[idx] != HOST_RANK || SHARD_GPU[idx] != g )); then
        continue
      fi
      SHARD_SPEC="${{SHARD_SPEC_PATHS[$idx]}}"
      SHARD_NUM_DESIGNS="${{SHARD_NDES[$idx]}}"
      SHARD_DIFFUSION_BATCH_SIZE="${{SHARD_BATCH[$idx]}}"
      export BOLTZGEN_SHARD_ORDINAL="$idx"
      export BOLTZGEN_SHARD_HOST_ORDINAL="$HOST_RANK"
      export BOLTZGEN_SHARD_GPU_ORDINAL="$g"
      export BOLTZGEN_SHARD_HOSTNAME="${{HOSTNAME:-unknown}}"
      export BOLTZGEN_SHARD_LENGTH="${{SHARD_LEN[$idx]}}"
      export BOLTZGEN_SHARD_LOGICAL_ORDINAL_START="${{SHARD_ORD_START[$idx]}}"
      export BOLTZGEN_SHARD_LOGICAL_ORDINAL_END="${{SHARD_ORD_END[$idx]}}"
      export BOLTZGEN_SHARD_INDEX="$idx"
      export BOLTZGEN_HOST_RANK="$HOST_RANK"
      export BOLTZGEN_GPU_RANK="$g"
      export BOLTZGEN_LOGICAL_ORDINAL_START="${{SHARD_ORD_START[$idx]}}"
      export BOLTZGEN_LOGICAL_ORDINAL_END="${{SHARD_ORD_END[$idx]}}"
      if (( HOST_COUNT > 1 )); then
        GPU_TAG="$(printf '%02d' "$g")"
        SHARD_TAG="$(printf '%03d' "$idx")"
        SHARD_OUTPUT="outputs/boltzgen_output/host_${{HOST_TAG}}/gpu_${{GPU_TAG}}/shard_${{SHARD_TAG}}_len${{SHARD_LEN[$idx]}}"
        SHARD_LOG="logs/host_${{HOST_TAG}}/boltzgen_gpu_${{GPU_TAG}}_shard_${{SHARD_TAG}}.log"
      else
        SHARD_OUTPUT="outputs/boltzgen_output/gpu_${{idx}}"
        SHARD_LOG="logs/boltzgen_gpu_${{idx}}.log"
      fi
      mkdir -p "$SHARD_OUTPUT" "$(dirname "$SHARD_LOG")"
      echo "[HARNESS][host $HOST_RANK][GPU $g][shard $idx] len=${{SHARD_LEN[$idx]}} num_designs=$SHARD_NUM_DESIGNS spec=$SHARD_SPEC output=$SHARD_OUTPUT"
      shard_started_at="$(date +%s)"
      if harness_silence_log_enabled; then
        CUDA_VISIBLE_DEVICES="$g" {shard_command} 2>&1 \
          | sed -u "s/^/[HARNESS][host $HOST_RANK][GPU $g][shard $idx][BOLTZGEN] /" \
          | tee -a "$SHARD_LOG" >> "$LOG_FILE"
      else
        CUDA_VISIBLE_DEVICES="$g" {shard_command} 2>&1 \
          | sed -u "s/^/[HARNESS][host $HOST_RANK][GPU $g][shard $idx][BOLTZGEN] /" \
          | tee -a "$SHARD_LOG"
      fi
      shard_status=${{PIPESTATUS[0]}}
      shard_elapsed=$(( $(date +%s) - shard_started_at ))
      echo "[HARNESS][host $HOST_RANK][GPU $g][shard $idx] elapsed_seconds=$shard_elapsed elapsed_human=$(harness_format_elapsed "$shard_elapsed") status=$shard_status"
      if (( shard_status != 0 )); then
        echo "[HARNESS][host $HOST_RANK][GPU $g][shard $idx][ERROR] boltzgen failed status=$shard_status" >&2
        worker_status=$shard_status
      else
        echo "[HARNESS][host $HOST_RANK][GPU $g][shard $idx] completed"
      fi
    done
    exit "$worker_status"
  ) &
  pids+=("$!")
  worker_gpus+=("$g")
done

for index in "${{!pids[@]}}"; do
  pid="${{pids[$index]}}"
  g="${{worker_gpus[$index]}}"
  if wait "$pid"; then
    echo "[HARNESS][host $HOST_RANK][GPU $g] all shards completed"
  else
    worker_status=$?
    echo "[HARNESS][host $HOST_RANK][GPU $g][ERROR] worker failed status=$worker_status" >&2
    status=$worker_status
  fi
done

if (( HOST_COUNT == 1 )); then
  cat > outputs/boltzgen_output/steps.yaml <<EOF
schema_version: 1
contract: binder_harness_steps_manifest
gpu_distribution:
  enabled: true
  host_count: 1
  gpu_count: $GPU_COUNT
  shards: $SHARD_TOTAL
  total_designs: {total_designs}
  lengths: [{lengths_yaml}]
EOF
else
  status_tmp="$COORDINATION_DIR/status/host_${{HOST_TAG}}.status.tmp.$$"
  printf '%s\\n' "$status" > "$status_tmp"
  mv "$status_tmp" "$COORDINATION_DIR/status/host_${{HOST_TAG}}.status"
  manifest_tmp="$COORDINATION_DIR/status/host_${{HOST_TAG}}.json.tmp.$$"
  printf '{{"host_rank":%s,"status":%s,"finished_at":"%s"}}\\n' \
    "$HOST_RANK" "$status" "$(date -Is)" > "$manifest_tmp"
  mv "$manifest_tmp" "$COORDINATION_DIR/status/host_${{HOST_TAG}}.json"

  barrier_deadline=$(( $(date +%s) + ${{HARNESS_CLUSTER_BARRIER_TIMEOUT:-7200}} ))
  if (( HOST_RANK == 0 )); then
    for ((expected_host=0; expected_host<HOST_COUNT; expected_host++)); do
      expected_tag="$(printf '%02d' "$expected_host")"
      while [[ ! -f "$COORDINATION_DIR/status/host_${{expected_tag}}.status" ]]; do
        if (( $(date +%s) >= barrier_deadline )); then
          echo "[HARNESS][ERROR] timed out waiting for host $expected_host completion" >&2
          status=27
          break 2
        fi
        sleep 2
      done
    done
    if (( status == 0 )); then
      for ((expected_host=0; expected_host<HOST_COUNT; expected_host++)); do
        expected_tag="$(printf '%02d' "$expected_host")"
        host_status="$(tr -d '[:space:]' < "$COORDINATION_DIR/status/host_${{expected_tag}}.status")"
        if ! [[ "$host_status" =~ ^[0-9]+$ ]] || (( host_status != 0 )); then
          echo "[HARNESS][ERROR] host $expected_host failed with status=$host_status" >&2
          status=28
        fi
      done
    fi

    steps_tmp="outputs/boltzgen_output/steps.yaml.tmp.$$"
    cat > "$steps_tmp" <<EOF
schema_version: 1
contract: binder_harness_steps_manifest
gpu_distribution:
  enabled: true
  mode: native_multi_host
  host_count: $HOST_COUNT
  gpus_per_host: $GPU_COUNT
  worker_count: $(( HOST_COUNT * GPU_COUNT ))
  shards: $SHARD_TOTAL
  total_designs: {total_designs}
  lengths: [{lengths_yaml}]
  status: $status
EOF
    mv "$steps_tmp" outputs/boltzgen_output/steps.yaml

    printf '%s\\n' "$status" > "$COORDINATION_DIR/cluster.status"
  fi

  while [[ ! -f "$COORDINATION_DIR/cluster.status" ]]; do
    if (( $(date +%s) >= barrier_deadline )); then
      echo "[HARNESS][ERROR] timed out waiting for cluster finalization" >&2
      status=29
      break
    fi
    sleep 2
  done
  if [[ -f "$COORDINATION_DIR/cluster.status" ]]; then
    cluster_status="$(tr -d '[:space:]' < "$COORDINATION_DIR/cluster.status")"
    if [[ "$cluster_status" =~ ^[0-9]+$ ]] && (( cluster_status != 0 )); then
      status="$cluster_status"
    fi
  fi
fi
"""
