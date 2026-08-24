#!/usr/bin/env python3

import argparse
import copy
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping as _ABCMapping
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents import ConfigValidationAgent, DesignSpecAgent, RFD3SpecAgent, RunMonitorAgent, TaijiExecutionAgent
from binderloop.agents.config_parameter_contract import parameter_contract_entry, partition_config_parameters
from binderloop.config import apply_memory_cli_overrides, load_config, primary_design_model
from binderloop.llm import LLMConfigError, LLMHTTPError, LLMTransportError, OpenAICompatibleClient
from binderloop.models.base import DesignJob
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.orchestration.runner import conda_run_command
from binderloop.resume import ResumeMismatchError, atomic_write_json, build_run_manifest, validate_or_write_run_manifest
from binderloop.secrets import SecretStore, redact_sensitive
from binderloop.agents.run_monitor_agent import RUNNING_STATES
from binderloop.package_layout import PROJECT_PACKAGE_DIRNAME


TAIJI_REMOTE_RUN_ROOT = Path("/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_closed_loop")


def _apply_execution_args(cfg, args: argparse.Namespace) -> None:
    """Resolve explicit CLI overrides over the task's execution configuration."""
    backend_override = getattr(args, "backend", None)
    if backend_override:
        cfg.resource.backend = str(backend_override).strip().lower()
    model_name = primary_design_model(cfg)
    model_runtime = cfg.runtime.model_runtime(model_name)
    args.conda_base = getattr(args, "conda_base", None) or cfg.runtime.conda_base
    args.conda_executable = getattr(args, "conda_executable", None) or cfg.runtime.conda_executable
    args.conda_env_name = getattr(args, "conda_env_name", None) or model_runtime.conda_env
    weight_root = model_runtime.weights_path
    args.checkpoint_dir = getattr(args, "checkpoint_dir", None) or model_runtime.checkpoint_dir or weight_root
    args.cache_dir = getattr(args, "cache_dir", None) or model_runtime.cache_dir or weight_root
    args.moldir = getattr(args, "moldir", None) or model_runtime.moldir
    args.target_model = model_name


def _apply_memory_args(cfg, args: argparse.Namespace) -> None:
    apply_memory_cli_overrides(
        cfg.memory,
        enabled=bool(getattr(args, "memory_enabled", False)),
        index_items=bool(getattr(args, "memory_index_items", False)),
        retrieval=bool(getattr(args, "memory_retrieval", False)),
        semantic_rerank=bool(getattr(args, "memory_semantic_rerank", False)),
        compression=bool(getattr(args, "memory_compression", False)),
        apply_prompt_budget=bool(getattr(args, "memory_apply_prompt_budget", False)),
    )


def _apply_self_improvement_args(cfg, args: argparse.Namespace, *, root: Path, config_path: Path) -> None:
    requested = getattr(args, "self_improvement_enabled", None)
    requested_path = getattr(args, "self_improvement_skill", None)
    if requested is False and requested_path:
        raise SystemExit("--disable-self-improvement-skill conflicts with --self-improvement-skill")
    if requested_path:
        cfg.self_improvement.enabled = True
        cfg.self_improvement.skill_path = str(_resolve_path(root, requested_path))
        return
    if requested is not None:
        cfg.self_improvement.enabled = bool(requested)
        if not requested:
            cfg.self_improvement.skill_path = None
        return
    configured_path = cfg.self_improvement.skill_path
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        cfg.self_improvement.skill_path = str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run closed-loop Binder design orchestrator")
    parser.add_argument("--config", default="configs/example_binder_task.yaml")
    parser.add_argument("--out", default="outputs/closed_loop_orchestrator")
    parser.add_argument("--max-rounds", type=int, default=None, help="Override active_learning.max_rounds (total-round ceiling; used for resume extension)")
    parser.add_argument("--llm-config", help="Local JSON with OpenAI-compatible endpoints/API-key env names. Required with --require-llm.")
    parser.add_argument("--llm-model", help="Endpoint key from the LLM config to use as default_model for this run")
    parser.add_argument("--llm-thinking", help="Reasoning/thinking level for this run, e.g. low|medium|high or enabled")
    parser.add_argument("--require-llm", action="store_true", help="Fail fast after a live LLM API preflight if the endpoint is not configured/available instead of falling back to deterministic rules")
    parser.add_argument("--submit", action="store_true", help="Actually execute jobs using the selected backend. Without this, only runnable artifacts are generated.")
    parser.add_argument("--backend", "--execution-backend", choices=["direct", "local", "taiji", "dry_run"], help="Explicitly override resource.backend. Taiji is never selected implicitly.")
    parser.add_argument("--taiji-client", default="taiji_client")
    parser.add_argument("--taiji-task-prefix", default=None)
    parser.add_argument("--taiji-remote-run-root", default=str(TAIJI_REMOTE_RUN_ROOT), help="Ceph-visible root used to stage Taiji project packages.")
    parser.add_argument("--taiji-poll-seconds", type=int, default=120)
    parser.add_argument("--taiji-wait-timeout", type=int, default=None, help="Seconds to wait for each submitted Taiji job. Defaults to resource.timeout_seconds.")
    parser.add_argument("--no-wait-taiji", action="store_true", help="Submit Taiji jobs and return immediately instead of waiting for outputs.")
    parser.add_argument("--result-sync-mode", choices=["symlink", "copy", "materialize"], default="symlink", help="Expose completed Taiji logs/outputs through relative symlinks (default), or materialize local copies.")
    parser.add_argument("--secret-config", help="Local ignored JSON containing secrets such as CEPH_SECRET. Defaults to --llm-config.")
    parser.add_argument("--conda-base", help="Conda installation root used inside generated scripts. Defaults to runtime.conda_base.")
    parser.add_argument("--conda-executable", help="Conda executable used by the direct backend. Defaults to runtime.conda_executable.")
    parser.add_argument("--conda-env-name", help="BoltzGen Conda environment. Defaults to runtime.model_runtimes.boltzgen.conda_env.")
    parser.add_argument("--checkpoint-dir", help="Directory containing local BoltzGen checkpoints. Overrides runtime.model_runtimes.boltzgen.")
    parser.add_argument("--cache-dir", help="Directory used for BoltzGen local artifact cache. Overrides runtime.model_runtimes.boltzgen.")
    parser.add_argument("--moldir", help="Path to the local BoltzGen mols.zip artifact.")
    parser.add_argument("--boltzgen-heartbeat-seconds", type=int, default=None, help="Heartbeat interval for BoltzGen liveness logs. Values above 360 are clamped to 360 seconds.")
    parser.add_argument("--silence", "--boltzgen-silence-log", dest="boltzgen_silence_log", action="store_true", help="Keep detailed BoltzGen/agent logs in log files without mirroring them to the screen.")
    parser.add_argument("--force-new-run", action="store_true", help="Create a new run manifest only for an empty output directory. Existing resume artifacts are never overwritten.")
    parser.add_argument(
        "--memory-enabled",
        action="store_true",
        help="Opt-in convenience master: enable memory index_items + retrieval + compression + prompt budget. Does not enable semantic_rerank.",
    )
    parser.add_argument(
        "--memory-index-items",
        action="store_true",
        help="Opt-in: write/backfill indexed evidence cards into experiment memory.",
    )
    parser.add_argument(
        "--memory-retrieval",
        action="store_true",
        help="Opt-in: structured recall and inject recalled_items into agent prompts (affects context).",
    )
    parser.add_argument(
        "--memory-semantic-rerank",
        action="store_true",
        help="Opt-in: GPT semantic rerank of retrieval candidates (implies --memory-retrieval).",
    )
    parser.add_argument(
        "--memory-compression",
        action="store_true",
        help="Opt-in: performance-then-age compression of indexed memory items.",
    )
    parser.add_argument(
        "--memory-apply-prompt-budget",
        action="store_true",
        help="Opt-in: clamp the LLM endpoint max_prompt_bytes to memory.prompt_max_bytes.",
    )
    self_improvement_group = parser.add_mutually_exclusive_group()
    self_improvement_group.add_argument(
        "--enable-self-improvement-skill",
        dest="self_improvement_enabled",
        action="store_true",
        default=None,
        help="Enable the run-local self-improving Binder skill; create a unique skill when no path is supplied.",
    )
    self_improvement_group.add_argument(
        "--disable-self-improvement-skill",
        dest="self_improvement_enabled",
        action="store_false",
        help="Disable self-improvement even when the task config enables it.",
    )
    parser.add_argument(
        "--self-improvement-skill",
        help="Existing structured self-improvement YAML to copy as this run's read-only seed; implies enable.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(root, args.config)
    cfg = load_config(config_path)
    _apply_execution_args(cfg, args)
    _apply_memory_args(cfg, args)
    _apply_self_improvement_args(cfg, args, root=root, config_path=config_path)
    if args.boltzgen_heartbeat_seconds is not None:
        cfg.search_space.boltzgen["log_heartbeat_seconds"] = args.boltzgen_heartbeat_seconds
    if args.boltzgen_silence_log:
        cfg.search_space.boltzgen["silence"] = True
    llm_config_path = None
    if args.llm_config:
        llm_config_path = Path(args.llm_config).expanduser()
        if not llm_config_path.is_absolute():
            llm_config_path = root / llm_config_path
        llm = OpenAICompatibleClient.from_json(llm_config_path)
    else:
        llm = None
    if llm:
        llm.configure_default(model_key=args.llm_model, thinking=args.llm_thinking)
    logprobs_capability = {"status": "not_configured", "mode": "disabled"}
    if llm and llm.available():
        mode = str(llm.resolved_endpoint.capabilities.logprobs or "auto")
        logprobs_capability = {"status": "not_probed", "mode": mode, "endpoint_key": llm.resolved_endpoint_key}
        if mode == "required":
            logprobs_capability = llm.probe_logprobs()
            if logprobs_capability.get("status") != "supported":
                raise SystemExit(f"required logprobs capability preflight failed: {logprobs_capability.get('status')}: {logprobs_capability.get('reason', '')}")
    if args.require_llm:
        if not (llm and llm.available()):
            raise SystemExit(
                "--require-llm was set, but no enabled LLM endpoint is available. "
                "Check --llm-config/default_model/enabled/API key."
            )
        try:
            preflight = llm.preflight()
        except (LLMConfigError, LLMTransportError, LLMHTTPError) as exc:
            raise SystemExit(f"--require-llm preflight failed: {exc}") from exc
        print(
            "LLM preflight ok: "
            f"endpoint={preflight.get('endpoint_key')} "
            f"model={preflight.get('model')} "
            f"elapsed={preflight.get('elapsed_seconds')}s"
        )
    out_dir = _resolve_path(root, args.out)
    manifest = build_run_manifest(
        config_path=config_path,
        config=cfg,
        cli_identity={
            "max_rounds": args.max_rounds,
            "backend": cfg.resource.backend,
            "submit": bool(args.submit),
            "taiji_client": args.taiji_client,
            "taiji_task_prefix": args.taiji_task_prefix,
            "taiji_remote_run_root": args.taiji_remote_run_root,
            "taiji_wait_timeout": args.taiji_wait_timeout,
            "no_wait_taiji": bool(args.no_wait_taiji),
            "result_sync_mode": args.result_sync_mode,
            "conda_base": args.conda_base,
            "conda_executable": args.conda_executable,
            "conda_env_name": args.conda_env_name,
            "checkpoint_dir": args.checkpoint_dir,
            "cache_dir": args.cache_dir,
            "moldir": args.moldir,
            "boltzgen_heartbeat_seconds": args.boltzgen_heartbeat_seconds,
            "boltzgen_silence_log": bool(args.boltzgen_silence_log),
            "llm_config_path": str(llm_config_path) if llm_config_path else None,
            "llm_model": llm.resolved_endpoint_key if llm else None,
            "llm_model_override": args.llm_model,
            "llm_provider_model": llm.resolved_endpoint.model if llm else None,
            "llm_provider": llm.resolved_endpoint.provider if llm else None,
            "llm_thinking": llm.resolved_endpoint.thinking if llm else None,
            "llm_thinking_override": args.llm_thinking,
            "require_llm": bool(args.require_llm),
            "logprobs_capability": {
                "status": logprobs_capability.get("status"),
                "mode": logprobs_capability.get("mode"),
                "source": logprobs_capability.get("source"),
                "endpoint_key": logprobs_capability.get("endpoint_key"),
                "reason": logprobs_capability.get("reason"),
            },
            "memory_enabled": bool(args.memory_enabled),
            "memory_index_items": bool(args.memory_index_items),
            "memory_retrieval": bool(args.memory_retrieval),
            "memory_semantic_rerank": bool(args.memory_semantic_rerank),
            "memory_compression": bool(args.memory_compression),
            "memory_apply_prompt_budget": bool(args.memory_apply_prompt_budget),
            "self_improvement_enabled": bool(cfg.self_improvement.enabled),
            "self_improvement_skill_source": cfg.self_improvement.skill_path,
        },
    )
    try:
        validate_or_write_run_manifest(out_dir, manifest, force_new_run=bool(args.force_new_run))
    except ResumeMismatchError as exc:
        raise SystemExit(str(exc)) from exc
    executor = build_job_executor(cfg, root=root, args=args, llm_config_path=llm_config_path)
    summary = BinderDesignOrchestrator(
        cfg,
        out_dir=out_dir,
        max_rounds=args.max_rounds,
        llm=llm,
        require_llm=bool(args.require_llm),
    ).run(execute_job=executor)
    out_dir = Path(summary['out_dir'])
    print(f"Closed-loop summary: {out_dir}/orchestrator_summary.json")
    png_path = out_dir / "iteration_metrics_trends.png"
    if png_path.exists():
        print(f"Iteration metrics plot: {png_path}")
    else:
        print("Iteration metrics plot: (skipped - no round metrics with candidates found)")
    return 0


def build_job_executor(cfg, *, root: Path, args: argparse.Namespace, llm_config_path: Optional[Path]) -> Optional[Callable[[DesignJob, int], Dict[str, Any]]]:
    _apply_execution_args(cfg, args)
    backend = str(cfg.resource.backend or "direct").lower()
    if backend == "dry_run":
        return None

    target_model = str(getattr(args, "target_model", None) or primary_design_model(cfg)).strip().lower()
    if target_model == "rfd3":
        foundry_root = _resolve_path(root, getattr(cfg.runtime, "foundry_root", "models/foundry"))
        spec_agent = RFD3SpecAgent(
            foundry_root,
            weights_path=_resolve_optional_model_path(foundry_root, args.checkpoint_dir),
        )
        if backend == "taiji":
            raise SystemExit("RFD3 Foundry adapter supports direct/local backends in this release; Taiji remains BoltzGen-only")
    else:
        boltzgen_root = _resolve_path(root, cfg.runtime.boltzgen_root)
        spec_agent = DesignSpecAgent(
            boltzgen_root,
            checkpoint_dir=_resolve_optional_model_path(boltzgen_root, args.checkpoint_dir),
            cache_dir=_resolve_optional_model_path(boltzgen_root, args.cache_dir),
            moldir=_resolve_optional_model_path(boltzgen_root, args.moldir),
        )

    if llm_config_path:
        validation_llm = OpenAICompatibleClient.from_json(llm_config_path)
        validation_llm.configure_default(model_key=args.llm_model, thinking=args.llm_thinking)
    else:
        validation_llm = None
    config_validator = ConfigValidationAgent(validation_llm)

    if backend == "direct":
        return _build_local_executor(spec_agent=spec_agent, args=args, config_validator=config_validator, backend="direct", target_model=target_model)
    if backend == "local":
        return _build_local_executor(spec_agent=spec_agent, args=args, config_validator=config_validator, backend="local", target_model=target_model)
    if backend == "taiji":
        return _build_taiji_executor(cfg, root=root, spec_agent=spec_agent, args=args, llm_config_path=llm_config_path, config_validator=config_validator)

    raise SystemExit(f"Unsupported resource.backend={cfg.resource.backend!r}; expected direct, local, taiji, or dry_run")


@dataclass(frozen=True)
class PreSubmitPreparation:
    """Canonical, backend-neutral execution view produced before submission."""

    schema_version: int
    backend: str
    original_params: Dict[str, Any]
    backend_overrides: Dict[str, Any]
    candidate_params: Dict[str, Any]
    validated_params: Dict[str, Any]
    orchestration_context: Dict[str, Any]
    validation: Any
    is_submittable: bool
    requires_refinalization: bool
    diff: Dict[str, Any]
    artifact_path: str
    correction_proposal: Optional[Dict[str, Any]]


def _validation_payload(validation: Any) -> Dict[str, Any]:
    if hasattr(validation, "__dataclass_fields__"):
        return asdict(validation)
    if isinstance(validation, Mapping):
        return dict(validation)
    return dict(getattr(validation, "__dict__", {}) or {})


def _validation_value(validation: Any, key: str, default: Any = None) -> Any:
    if isinstance(validation, Mapping):
        return validation.get(key, default)
    return getattr(validation, key, default)


def _classify_execution_diff(original: Mapping[str, Any], validated: Mapping[str, Any], backend_overrides: Mapping[str, Any]) -> Dict[str, Any]:
    candidate = {**copy.deepcopy(dict(original or {})), **copy.deepcopy(dict(backend_overrides or {}))}
    normalized: Dict[str, Any] = {}
    stripped: List[str] = []
    added: Dict[str, Any] = {}
    for key in sorted(set(candidate) | set(validated)):
        if key not in validated:
            stripped.append(key)
        elif key not in candidate:
            added[key] = copy.deepcopy(validated[key])
        elif candidate[key] != validated[key]:
            normalized[key] = {"before": copy.deepcopy(candidate[key]), "after": copy.deepcopy(validated[key])}
    return {
        "schema_version": 1,
        "normalization": normalized,
        "metadata_stripping": stripped,
        "validator_additions": added,
        "backend_overrides": copy.deepcopy(dict(backend_overrides or {})),
        "has_execution_change": bool(normalized or stripped or added),
    }


def _semantic_patch_key(key: str) -> bool:
    entry = parameter_contract_entry(key)
    if entry is None:
        return False
    return str(entry.get("type") or "") not in {"internal_metadata", "deprecated_metadata"}


def _source_validation_digest(validation: Any) -> str:
    payload = _validation_payload(validation) if validation is not None else {}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _typed_correction_proposal(
    original: Mapping[str, Any], corrected: Mapping[str, Any], *, reason: str,
    validation: Any = None, classification: str = "semantic_config_correction",
    identity_effect: str = "requires_refinalization", safe_default_keys: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    original_dict, corrected_dict = dict(original or {}), dict(corrected or {})
    safe_defaults = set(safe_default_keys or [])
    patch_set = {
        key: copy.deepcopy(value) for key, value in corrected_dict.items()
        if key not in safe_defaults and _semantic_patch_key(key) and original_dict.get(key) != value
    }
    patch_remove = sorted(
        key for key in set(original_dict) - set(corrected_dict)
        if key not in safe_defaults and _semantic_patch_key(key)
    )
    if not patch_set and not patch_remove:
        return None
    correction_patch = {
        "set": patch_set, "remove": patch_remove, "classification": classification,
        "identity_effect": identity_effect, "source_validation_digest": _source_validation_digest(validation),
    }
    changes = dict(patch_set)
    if patch_remove:
        changes["__tombstones__"] = patch_remove
    return {
        "schema_version": 1, "proposal_type": "execution_params_replacement", "reason": reason,
        "requires_refinalization": identity_effect == "requires_refinalization",
        "correction_patch": correction_patch,
        "corrected_params": copy.deepcopy(corrected_dict),  # compatibility only
        "changes": changes,
    }



def _attempt_artifact_root(job: DesignJob, attempt: int) -> Path:
    root = Path(job.output_dir) / "attempts" / f"attempt_{int(attempt):02d}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _attempt_job(job: DesignJob, attempt: int) -> DesignJob:
    cloned = copy.deepcopy(job)
    cloned.output_dir = str(_attempt_artifact_root(job, attempt))
    return cloned


def _artifact_locators(job: DesignJob, attempt: int, attempt_root: Path) -> Dict[str, Any]:
    return {
        "identity_root": str(Path(job.output_dir)),
        "attempt_root": str(attempt_root),
        "attempt": int(attempt),
        "execution_record": str(attempt_root / "execution_record.json"),
        "legacy_execution_record": str(Path(job.output_dir) / "execution_record.json"),
    }

def _prepare_pre_submit_execution(
    job: DesignJob,
    attempt: int,
    *,
    backend: str,
    backend_overrides: Mapping[str, Any],
    config_validator: ConfigValidationAgent,
    target_model: str = "boltzgen",
) -> PreSubmitPreparation:
    original = copy.deepcopy(dict(job.params or {}))
    candidate = {**copy.deepcopy(original), **copy.deepcopy(dict(backend_overrides or {}))}
    context = {"backend": backend, "job_id": job.job_id, "attempt": attempt}
    # Prefer the richer contract when it lands, while retaining compatibility with
    # the legacy validator during concurrent development.
    validate = getattr(config_validator, "validate_execution_view", None)
    if callable(validate):
        validation = validate(original, backend_overrides=dict(backend_overrides or {}), target_model=target_model, context=context)
    else:
        validation = config_validator.validate_full_job_config(candidate, target_model=target_model, context=context)
    validated = copy.deepcopy(dict(
        _validation_value(validation, "validated_config", None)
        or _validation_value(validation, "execution_view", None)
        or _validation_value(validation, "corrected_config", {})
    ))
    # Submission payloads contain only fields consumed by the runner, adapter, or
    # runtime. Orchestration/identity metadata remains on the immutable DesignJob
    # and is passed separately as execution_identity_context.
    partitions = _validation_value(validation, "validated_partition", None)
    if isinstance(partitions, Mapping):
        validated = {
            str(key): copy.deepcopy(value)
            for partition in ("runner", "adapter", "runtime")
            for key, value in dict(partitions.get(partition) or {}).items()
        }
    original_partitions = partition_config_parameters(original)
    validated_orchestration = dict(partitions.get("orchestration") or {}) if isinstance(partitions, Mapping) else {}
    orchestration_context = {
        **copy.deepcopy(dict(original_partitions.get("orchestration") or {})),
        **copy.deepcopy(validated_orchestration),
    }
    is_submittable = bool(_validation_value(validation, "is_submittable", _validation_value(validation, "is_valid", False)))
    requires_refinalization = bool(_validation_value(validation, "requires_refinalization", False))
    diff = _classify_execution_diff(original, validated, backend_overrides)
    # Normalization and metadata tombstones are execution-view transformations,
    # not reasons to reject a finalized job. Only the validator's explicit gates block.
    correction = None
    if requires_refinalization:
        corrected_job = dict(validated)
        for key in backend_overrides:
            if key not in original:
                corrected_job.pop(key, None)
        correction = _typed_correction_proposal(original, corrected_job, reason="pre_submit_config_validation", validation=validation)
    artifact_path = Path(job.output_dir) / "pre_submit_config_validation.json"
    config_validator.write_result(validation, artifact_path)
    if original != dict(job.params or {}):
        raise RuntimeError("pre-submit preparation mutated immutable job.params")
    return PreSubmitPreparation(
        schema_version=1, backend=backend, original_params=original,
        backend_overrides=copy.deepcopy(dict(backend_overrides or {})), candidate_params=candidate,
        validated_params=validated, orchestration_context=orchestration_context,
        validation=validation, is_submittable=is_submittable,
        requires_refinalization=requires_refinalization, diff=diff, artifact_path=str(artifact_path),
        correction_proposal=correction,
    )


def _preparation_record(prepared: PreSubmitPreparation) -> Dict[str, Any]:
    return {
        "schema_version": prepared.schema_version,
        "artifact": prepared.artifact_path,
        "is_submittable": prepared.is_submittable,
        "requires_refinalization": prepared.requires_refinalization,
        "diff": prepared.diff,
        "backend_overrides": prepared.backend_overrides,
        "validated_execution_view": prepared.validated_params,
        "orchestration_context": prepared.orchestration_context,
        "validation": _validation_payload(prepared.validation),
    }


def _create_replacement_run_spec(spec_agent, job: DesignJob, params: Mapping[str, Any], args: argparse.Namespace, *, orchestration_context: Optional[Mapping[str, Any]] = None):
    kwargs: Dict[str, Any] = {
        "params": dict(params), "conda_base": args.conda_base, "conda_env_name": args.conda_env_name,
    }
    create = getattr(spec_agent, "create_rfd3_run_spec", None)
    if create is None:
        create = spec_agent.create_boltzgen_run_spec
    signature = inspect.signature(create)
    if "params_mode" in signature.parameters:
        kwargs["params_mode"] = "replacement"
    if "execution_identity_context" in signature.parameters:
        kwargs["execution_identity_context"] = copy.deepcopy(dict(orchestration_context or {}))
    return create(job, **kwargs)


def _blocked_pre_submit_record(job: DesignJob, attempt: int, prepared: PreSubmitPreparation) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "schema_version": 1, "job_id": job.job_id, "backend": prepared.backend, "attempt": attempt,
        "status": "failed", "submit_status": "blocked", "error": "pre-submit config validation failed",
        "retryable": False, "output_dir": job.output_dir, "attempt_root": job.output_dir,
        "artifact_locators": {"attempt_root": job.output_dir, "execution_record": str(Path(job.output_dir) / "execution_record.json")},
        "pre_submit": _preparation_record(prepared),
        "config_validation": _validation_payload(prepared.validation),
    }
    if prepared.correction_proposal:
        record["retry_correction_proposal"] = prepared.correction_proposal
    _write_json(Path(job.output_dir) / "execution_record.json", record)
    return record


def _build_local_executor(*, spec_agent, args: argparse.Namespace, config_validator: ConfigValidationAgent, backend: str = "local", target_model: str = "boltzgen") -> Callable[[DesignJob, int], Dict[str, Any]]:
    if backend not in {"direct", "local"}:
        raise ValueError(f"unsupported local executor backend: {backend}")

    def execute(job: DesignJob, attempt: int) -> Dict[str, Any]:
        identity_job = job
        job = _attempt_job(identity_job, attempt)
        attempt_root = Path(job.output_dir)
        locators = _artifact_locators(identity_job, attempt, attempt_root)
        prepared = _prepare_pre_submit_execution(
            job,
            attempt,
            backend=backend,
            backend_overrides={"analysis_location": "inline"},
            config_validator=config_validator,
            target_model=target_model,
        )
        if not prepared.is_submittable or prepared.requires_refinalization:
            return _blocked_pre_submit_record(job, attempt, prepared)
        run_spec = _create_replacement_run_spec(spec_agent, job, prepared.validated_params, args, orchestration_context=prepared.orchestration_context)
        record: Dict[str, Any] = {
            "schema_version": 1, "job_id": job.job_id, "backend": backend, "attempt": attempt,
            "run_spec": asdict(run_spec), "output_dir": run_spec.output_dir, "log_file": run_spec.log_file,
            "pre_submit": _preparation_record(prepared), "config_validation": _validation_payload(prepared.validation),
            "artifact_locators": locators, "identity_output_dir": identity_job.output_dir, "attempt_root": str(attempt_root),
            "status": "dry_run" if not args.submit else "running",
            "submit_status": "not_requested" if not args.submit else "started",
        }
        execution_command = ["bash", run_spec.run_script_path]
        if backend == "direct":
            execution_command = conda_run_command(
                execution_command,
                env_name=args.conda_env_name,
                conda_executable=args.conda_executable,
            )
        record["execution_command"] = execution_command
        if not args.submit:
            record["message"] = f"Generated {backend} {target_model} run script; pass --submit to execute it."
            _write_json(Path(job.output_dir) / "execution_record.json", record)
            return record
        proc = subprocess.run(execution_command, text=True, capture_output=True, check=False)
        record.update({"returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:],
                       "status": "completed" if proc.returncode == 0 else "failed",
                       "submit_status": "completed" if proc.returncode == 0 else "failed"})
        if proc.returncode != 0:
            _attach_post_failure_correction(record, job=job, execution_params=prepared.validated_params,
                                            config_validator=config_validator, error_context={"stdout_tail": record["stdout_tail"], "stderr_tail": record["stderr_tail"]},
                                            target_model=target_model)
        _write_json(Path(job.output_dir) / "execution_record.json", record)
        return record
    return execute


def _build_taiji_executor(cfg, *, root: Path, spec_agent: DesignSpecAgent, args: argparse.Namespace, llm_config_path: Optional[Path], config_validator: ConfigValidationAgent) -> Callable[[DesignJob, int], Dict[str, Any]]:
    secret_config_path = Path(args.secret_config).expanduser() if args.secret_config else llm_config_path
    if secret_config_path is not None and not secret_config_path.is_absolute():
        secret_config_path = root / secret_config_path
    secret_store = SecretStore.from_json(secret_config_path)
    template_json = _resolve_optional_path(root, cfg.resource.template_json)
    remote_run_root = Path(args.taiji_remote_run_root).expanduser()
    taiji_agent = TaijiExecutionAgent(taiji_client_bin=args.taiji_client, dry_run=not args.submit)
    monitor = RunMonitorAgent(taiji_client_bin=args.taiji_client)
    try:
        requested_host_num = max(1, int(getattr(cfg.resource, "host_num", 1) or 1))
    except (TypeError, ValueError):
        requested_host_num = 1
    multi_host_mode = str(getattr(cfg.resource, "taiji_multi_host_mode", "native") or "native").strip().lower()
    multi_host_mode = {"unified": "native", "fanout": "split_jobs", "split": "split_jobs"}.get(multi_host_mode, multi_host_mode)
    native_multi_host = requested_host_num > 1 and multi_host_mode == "native"
    submit_host_num = requested_host_num if native_multi_host else 1
    multi_host_wait_required = requested_host_num > 1

    def execute(job: DesignJob, attempt: int) -> Dict[str, Any]:
        identity_job = job
        job = _attempt_job(identity_job, attempt)
        attempt_root = Path(job.output_dir)
        locators = _artifact_locators(identity_job, attempt, attempt_root)
        overrides = {
            "analysis_location": "taiji",
            "run_analysis_on_taiji": True,
            "host_count": submit_host_num,
            "taiji_submit_host_num": submit_host_num,
            "taiji_multi_host_mode": "native" if native_multi_host else "split_jobs",
        }
        prepared = _prepare_pre_submit_execution(job, attempt, backend="taiji", backend_overrides=overrides, config_validator=config_validator)
        if not prepared.is_submittable or prepared.requires_refinalization:
            return _blocked_pre_submit_record(job, attempt, prepared)
        execution_params = prepared.validated_params
        run_spec = _create_replacement_run_spec(spec_agent, job, execution_params, args, orchestration_context=prepared.orchestration_context)
        local_package_dir = Path(run_spec.package_dir or Path(run_spec.run_script_path).parents[1])
        local_output_dir, local_log_file = local_package_dir / "outputs" / "boltzgen_output", Path(run_spec.log_file)
        task_flag = _task_flag(args.taiji_task_prefix or cfg.task_name, job.job_id, attempt)
        remote_package_dir = _sync_package_to_remote_run_dir(local_package_dir, task_flag, remote_run_root)
        _point_run_spec_to_package(run_spec, remote_package_dir)
        taiji_options = cfg.resource.to_taiji_options()
        try:
            effective_devices = max(1, int(execution_params.get("devices") or getattr(cfg.resource, "host_gpu_num", 1) or 1))
        except (TypeError, ValueError):
            effective_devices = max(1, int(getattr(cfg.resource, "host_gpu_num", 1) or 1))
        taiji_options.update({"model_local_file_path": str(remote_package_dir), "remote_project_dir": str(remote_package_dir),
                              "host_num": submit_host_num, "host_gpu_num": effective_devices, "exec_start_in_all_mpi_pods": True})
        envs = dict(taiji_options.get("envs") or {})
        envs.update({"HARNESS_HOST_COUNT": str(submit_host_num), "HARNESS_GPUS_PER_HOST": str(effective_devices),
                     "HARNESS_MULTI_HOST_MODE": "native" if native_multi_host else "split_jobs", "HARNESS_RUN_TOKEN": task_flag})
        ceph_secret = secret_store.ceph_secret()
        if ceph_secret:
            envs["CEPH_SECRET"] = ceph_secret
        taiji_options["envs"] = envs
        submit_spec = taiji_agent.create_boltzgen_taiji_spec(run_spec, template_json=template_json,
            output_json=Path(job.output_dir) / "taiji_simple_config.json", task_flag=task_flag, taiji_options=taiji_options)
        submission = taiji_agent.submit(submit_spec, dry_run=not args.submit)
        record: Dict[str, Any] = {
            "schema_version": 1, "job_id": job.job_id, "backend": "taiji", "attempt": attempt,
            "status": "dry_run" if submission.dry_run else "submitted",
            "submit_status": "dry_run" if submission.dry_run else "submitted", "run_spec": asdict(run_spec),
            "local_package_dir": str(local_package_dir), "local_output_dir": str(local_output_dir), "local_log_file": str(local_log_file),
            "remote_package_dir": str(remote_package_dir), "remote_output_dir": run_spec.output_dir, "remote_log_file": run_spec.log_file,
            "submit_spec": {k: v for k, v in asdict(submit_spec).items() if k != "simple_config"},
            "submission": redact_sensitive(asdict(submission)), "pre_submit": _preparation_record(prepared),
            "config_validation": _validation_payload(prepared.validation), "output_dir": str(local_output_dir),
            "log_file": str(local_log_file), "taiji_job_id": submission.taiji_job_id, "task_flag": task_flag,
            "effective_devices": effective_devices, "artifact_locators": locators,
            "identity_output_dir": identity_job.output_dir, "attempt_root": str(attempt_root),
        }
        if submission.returncode not in (0, None):
            record.update(status="failed", submit_status="failed", error="taiji_client start failed", retryable=_taiji_submission_retryable(submission))
        elif args.submit and (not args.no_wait_taiji or multi_host_wait_required):
            if args.no_wait_taiji and multi_host_wait_required:
                record["wait_override"] = "ignored --no-wait-taiji because host_num>1 must finish all hosts before analysis"
            snapshot = _wait_for_taiji_completion(monitor, submit_spec=submit_spec, run_spec=run_spec, instance_id=submission.taiji_job_id,
                timeout_seconds=args.taiji_wait_timeout or cfg.resource.timeout_seconds, poll_seconds=max(5, args.taiji_poll_seconds))
            record["monitor"], record["taiji_job_id"] = asdict(snapshot), snapshot.instance_id or submission.taiji_job_id
            record["result_sync"] = _sync_remote_results_to_local(
                remote_package_dir, local_package_dir, mode=args.result_sync_mode,
                job_id=job.job_id, attempt=attempt, task_flag=task_flag, attempt_root=attempt_root,
            )
            binding = record["result_sync"].get("transport_binding")
            if binding:
                record["transport_binding"] = binding
            if snapshot.is_success:
                record.update(status="completed", submit_status="completed")
            elif snapshot.is_terminal:
                record.update(status="failed", submit_status="failed", error=";".join(snapshot.failure_hints) or snapshot.state, retryable=_taiji_snapshot_retryable(snapshot))
            else:
                record.update(status="timeout", submit_status="timeout", error=f"Taiji job did not reach terminal state within {args.taiji_wait_timeout or cfg.resource.timeout_seconds}s", retryable=True)
        if str(record.get("status") or "").lower() in {"failed", "error", "timeout"}:
            context = _taiji_failure_context(record, local_log_file=local_log_file)
            _attach_post_failure_correction(record, job=job, execution_params=execution_params, config_validator=config_validator, error_context=context)
            failure_class = _classify_taiji_failure(record, context)
            if failure_class:
                record["failure_class"] = failure_class
                if failure_class == "boltzgen_image_filename_contract_incompatible":
                    record["retryable"] = False
        _write_json(Path(job.output_dir) / "execution_record.json", record)
        return record
    return execute


def _attach_post_failure_correction(record: Dict[str, Any], *, job: DesignJob, execution_params: Mapping[str, Any], config_validator: ConfigValidationAgent, error_context: Mapping[str, Any], target_model: str = "boltzgen") -> None:
    validation = config_validator.improve_after_failure(execution_params, target_model=target_model, error_context=error_context)
    artifact = Path(job.output_dir) / "post_failure_config_validation.json"
    config_validator.write_result(validation, artifact)
    payload = _validation_payload(validation)
    record["post_failure_config_validation"] = payload
    record["post_failure_artifact"] = str(artifact)
    corrected_execution = dict(_validation_value(validation, "validated_config", None) or _validation_value(validation, "corrected_config", {}))
    corrected_job = dict(corrected_execution)
    for key in ("analysis_location", "run_analysis_on_taiji"):
        if key not in job.params:
            corrected_job.pop(key, None)
    requires = bool(_validation_value(validation, "requires_refinalization", False))
    proposal = _typed_correction_proposal(job.params, corrected_job, reason="post_failure_config_validation", validation=validation)
    if proposal and (requires or proposal["changes"]):
        record["retry_correction_proposal"] = proposal
        record["retryable"] = False


def _job_params_from_remote_correction(original: Mapping[str, Any], corrected: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(corrected or {})
    for key in ("analysis_location", "run_analysis_on_taiji"):
        if key not in original:
            result.pop(key, None)
    return result


def _config_correction_proposal(original: Mapping[str, Any], corrected: Mapping[str, Any], *, reason: str) -> Optional[Dict[str, Any]]:
    return _typed_correction_proposal(original, corrected, reason=reason)

def _classify_taiji_failure(record: Mapping[str, Any], context: Mapping[str, Any]) -> Optional[str]:
    """Classify deterministic image/runtime filename-contract failures."""
    text = json.dumps({"record": dict(record), "context": dict(context)}, sort_keys=True, default=str).lower()
    filename_context = any(token in text for token in (
        "data_from_generated.py", "target_id_regex", "sample_id", "generated_structure", "generated structures",
    ))
    regex_group_failure = any(token in text for token in (
        "nonetype' object has no attribute 'group'", 'nonetype" object has no attribute "group"',
        "attributeerror", "re.search",
    ))
    if filename_context and regex_group_failure:
        return "boltzgen_image_filename_contract_incompatible"
    return None

def _taiji_submission_retryable(submission) -> bool:
    text = f"{getattr(submission, 'stdout', '')}\n{getattr(submission, 'stderr', '')}".lower()
    non_retryable_needles = [
        "invalid simple config",
        "invalid config",
        "json decode",
        "json.decoder",
        "expecting value",
        "permission denied",
        "unauthorized",
        "forbidden",
        "no such file",
        "file not found",
    ]
    return not any(needle in text for needle in non_retryable_needles)


def _taiji_failure_context(record: Mapping[str, Any], *, local_log_file: Path) -> Dict[str, Any]:
    context = {
        "status": record.get("status"),
        "error": record.get("error"),
        "submission": record.get("submission"),
        "monitor": record.get("monitor"),
        "task_flag": record.get("task_flag"),
        "taiji_job_id": record.get("taiji_job_id"),
    }
    if local_log_file.exists():
        context["boltzgen_log_tail"] = local_log_file.read_text(encoding="utf-8", errors="ignore")[-6000:]
    logs_root = local_log_file.parent.parent if local_log_file.parent.name.startswith("host_") else local_log_file.parent
    for gpu_log in sorted(logs_root.glob("**/boltzgen_gpu_*.log"))[:8]:
        context.setdefault("gpu_log_tails", {})[gpu_log.name] = gpu_log.read_text(encoding="utf-8", errors="ignore")[-4000:]
    return context


def _replace_param_domain(target: Dict[str, Any], corrected: Mapping[str, Any]) -> Dict[str, Any]:
    """Replace the validated config domain, applying omissions as tombstones."""
    replacement = dict(corrected or {})
    target.clear()
    target.update(replacement)
    return target


def _changed_config_values(original: Mapping[str, Any], corrected: Mapping[str, Any]) -> Dict[str, Any]:
    changed: Dict[str, Any] = {}
    original_dict = dict(original or {})
    corrected_dict = dict(corrected or {})
    for key, value in corrected_dict.items():
        if original_dict.get(key) != value:
            changed[key] = value
    tombstones = sorted(set(original_dict) - set(corrected_dict))
    if tombstones:
        changed["__tombstones__"] = tombstones
    return changed


def _taiji_snapshot_retryable(snapshot) -> bool:
    hints = {str(item) for item in (getattr(snapshot, "failure_hints", None) or [])}
    if "resource_scheduling_failure" in hints or "taiji_resource_or_queue_issue" in hints:
        return True
    non_retryable_hints = {
        "missing_boltzgen_cli",
        "missing_input_file",
        "missing_ceph_mount_secret",
        "conda_env_error",
        "boltzgen_config_error",
    }
    return not bool(hints & non_retryable_hints)


def _params_for_remote_boltzgen(params: Dict[str, Any]) -> Dict[str, Any]:
    remote_params = dict(params or {})
    remote_params["analysis_location"] = "taiji"
    remote_params["run_analysis_on_taiji"] = True
    return remote_params


def _wait_for_taiji_completion(
    monitor: RunMonitorAgent,
    *,
    submit_spec,
    run_spec,
    instance_id: Optional[str],
    timeout_seconds: int,
    poll_seconds: int,
):
    timeout_seconds = max(1, int(timeout_seconds))
    deadline = time.time() + timeout_seconds
    latest = None
    snapshot_path = Path(run_spec.package_dir or Path(run_spec.run_script_path).parents[1]) / "taiji_monitor_snapshot.json"
    while True:
        latest = monitor.check_once(
            task_flag=submit_spec.task_flag,
            instance_id=instance_id,
            expected_outputs=run_spec.expected_outputs,
            simple_config_path=submit_spec.simple_config_path,
            config_path=submit_spec.full_config_path,
        )
        monitor.write_snapshot(latest, snapshot_path)
        if latest.instance_id:
            instance_id = latest.instance_id
        if latest.is_terminal:
            return latest
        if time.time() >= deadline:
            latest = monitor.check_once(
                task_flag=submit_spec.task_flag,
                instance_id=instance_id,
                expected_outputs=run_spec.expected_outputs,
                simple_config_path=submit_spec.simple_config_path,
                config_path=submit_spec.full_config_path,
            )
            monitor.write_snapshot(latest, snapshot_path)
            if latest.instance_id:
                instance_id = latest.instance_id
            if latest.is_terminal:
                return latest
            if _taiji_remote_still_active(latest):
                print(
                    "[HARNESS][TAIJI] local wait timeout reached "
                    f"({timeout_seconds}s), but remote task is still active "
                    f"(task_flag={submit_spec.task_flag}, instance_id={instance_id or '-'}, state={latest.state}). "
                    "Continuing to wait instead of retrying/resubmitting.",
                    flush=True,
                )
                deadline = time.time() + timeout_seconds
            else:
                return latest
        time.sleep(min(poll_seconds, max(5, latest.recommended_followup_seconds)))


def _taiji_remote_still_active(snapshot) -> bool:
    return str(getattr(snapshot, "state", "") or "").lower() in RUNNING_STATES


def _sync_package_to_remote_run_dir(package_dir: Union[str, Path], task_flag: str, remote_root: Path) -> Path:
    package_dir = Path(package_dir)
    remote_package_dir = remote_root / task_flag / PROJECT_PACKAGE_DIRNAME
    if _same_path(package_dir, remote_package_dir):
        return remote_package_dir
    remote_package_dir.parent.mkdir(parents=True, exist_ok=True)
    _remove_path_entry(remote_package_dir)
    # Package inputs are self-contained regular files. Keep copytree configured
    # to materialize any legacy links that may still exist in older packages.
    # Logs and outputs can point at a previous remote attempt; never dereference
    # those trees back into a fresh package.
    def ignore_runtime_results(directory: str, names: List[str]) -> List[str]:
        if not _same_path(Path(directory), package_dir):
            return []
        return [
            name
            for name in ("logs", "outputs", "taiji_monitor_snapshot.json")
            if name in names
        ]

    shutil.copytree(
        package_dir,
        remote_package_dir,
        symlinks=False,
        ignore=ignore_runtime_results,
        # Both roots are normally on Ceph. Hard links avoid reading/writing the
        # same immutable package bytes twice while remaining valid if the local
        # attempt directory is later removed. Cross-filesystem staging falls
        # back to a normal metadata-preserving copy.
        copy_function=_link_or_copy_file,
    )
    (remote_package_dir / "logs").mkdir(parents=True, exist_ok=True)
    (remote_package_dir / "outputs" / "boltzgen_output").mkdir(parents=True, exist_ok=True)
    return remote_package_dir


def _link_or_copy_file(source: str, target: str) -> str:
    try:
        os.link(source, target, follow_symlinks=True)
        return target
    except OSError:
        return shutil.copy2(source, target, follow_symlinks=True)


def _sync_remote_results_to_local(
    remote_package_dir: Union[str, Path],
    local_package_dir: Union[str, Path],
    *,
    mode: str = "symlink",
    job_id: Optional[str] = None,
    attempt: Optional[int] = None,
    task_flag: Optional[str] = None,
    attempt_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Expose remote results locally without following links during cleanup.

    ``symlink`` creates relative links and is the metadata-light default.
    ``copy`` and ``materialize`` are aliases that create dereferenced local copies.
    """
    remote_package_dir = Path(remote_package_dir)
    local_package_dir = Path(local_package_dir)
    mode = str(mode).strip().lower()
    if mode not in {"symlink", "copy", "materialize"}:
        raise ValueError(f"unsupported result sync mode: {mode}")
    synced: Dict[str, Any] = {
        "remote_package_dir": str(remote_package_dir),
        "local_package_dir": str(local_package_dir),
        "mode": mode,
        "copied": [],
        "linked": [],
        "fallbacks": [],
        "skipped": [],
        "missing": [],
    }
    if _same_path(remote_package_dir, local_package_dir):
        synced["already_local"] = True
        return synced
    local_package_dir.mkdir(parents=True, exist_ok=True)
    for name in ["logs", "outputs"]:
        source = remote_package_dir / name
        target = local_package_dir / name
        if not source.exists():
            synced["missing"].append(str(source))
            continue
        _sync_result_entry(source, target, mode=mode, synced=synced)
    snapshot = remote_package_dir / "taiji_monitor_snapshot.json"
    if snapshot.exists():
        target = local_package_dir / snapshot.name
        _sync_result_entry(snapshot, target, mode=mode, synced=synced)
    if mode == "symlink":
        expected = {}
        for name in ("outputs", "logs"):
            source = remote_package_dir / name
            target = local_package_dir / name
            link_text = os.path.relpath(str(source.absolute()), str(target.parent.absolute()))
            expected[name] = {"source": str(source), "target": str(target), "link": link_text}
        if any(item not in synced["linked"] for item in expected.values()):
            raise RuntimeError("result transport binding incomplete: outputs and logs links are both required")
        outputs_link = local_package_dir / "outputs"
        logs_link = local_package_dir / "logs"
        if not outputs_link.is_symlink() or os.readlink(str(outputs_link)) != expected["outputs"]["link"]:
            raise RuntimeError("result transport binding verification failed for outputs")
        if not logs_link.is_symlink() or os.readlink(str(logs_link)) != expected["logs"]["link"]:
            raise RuntimeError("result transport binding verification failed for logs")
        synced["transport_binding"] = {
            "schema_version": 1,
            "mode": "symlink",
            "local_package_dir": str(local_package_dir),
            "local_output_alias": str(local_package_dir / "outputs" / "boltzgen_output"),
            "local_logs_alias": str(logs_link),
            "remote_package_dir": str(remote_package_dir),
            "remote_output_root": str(remote_package_dir / "outputs" / "boltzgen_output"),
            "remote_logs_root": str(remote_package_dir / "logs"),
            "link_text": expected["outputs"]["link"],
            "logs_link_text": expected["logs"]["link"],
            "job_id": job_id,
            "attempt": attempt,
            "task_flag": task_flag,
            "attempt_root": str(attempt_root) if attempt_root is not None else None,
        }
    return synced


def _sync_result_entry(source: Path, target: Path, *, mode: str, synced: Dict[str, Any]) -> None:
    if _same_path(source, target):
        synced["skipped"].append({"source": str(source), "target": str(target)})
        return
    _remove_path_entry(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        relative_source = os.path.relpath(str(source.absolute()), str(target.parent.absolute()))
        try:
            target.symlink_to(relative_source, target_is_directory=source.is_dir())
        except OSError as exc:
            raise RuntimeError(f"failed to create result transport symlink {target} -> {relative_source}: {exc}") from exc
        synced["linked"].append({"source": str(source), "target": str(target), "link": relative_source})
    else:
        _copy_result_entry(source, target, synced)


def _copy_result_entry(source: Path, target: Path, synced: Dict[str, Any]) -> None:
    if source.is_dir():
        shutil.copytree(source, target, symlinks=False)
        synced["copied"].append({"source": str(source), "target": str(target)})
    else:
        shutil.copy2(source, target, follow_symlinks=True)
        synced["copied"].append({"source": str(source), "target": str(target)})


def _remove_path_entry(path: Path) -> None:
    # Never pass a symlink to rmtree: doing so risks deleting or rejecting the
    # linked result tree instead of replacing the local directory entry.
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _same_path(left: Path, right: Path) -> bool:
    if os.path.abspath(str(left)) == os.path.abspath(str(right)):
        return True
    try:
        return os.path.samefile(str(left), str(right))
    except OSError:
        return False


def _point_run_spec_to_package(run_spec, package_dir: Path) -> None:
    previous_output_dir = Path(run_spec.output_dir)
    previous_package_dir = Path(run_spec.package_dir or Path(run_spec.run_script_path).parents[1])
    output_dir = package_dir / "outputs" / "boltzgen_output"
    previous_log_file = Path(run_spec.log_file)
    try:
        log_file = package_dir / previous_log_file.relative_to(previous_package_dir)
    except ValueError:
        log_file = package_dir / "logs" / "boltzgen_full.log"
    target_files = sorted((package_dir / "inputs").glob("*"))
    target_file = target_files[0] if target_files else package_dir / "inputs" / "target.cif"
    previous_expected = dict(run_spec.expected_outputs)
    run_spec.package_dir = str(package_dir)
    run_spec.design_spec_path = str(package_dir / "configs" / "boltzgen_design_spec.yaml")
    run_spec.run_script_path = str(package_dir / "scripts" / "run_boltzgen_full.sh")
    run_spec.output_dir = str(output_dir)
    run_spec.log_file = str(log_file)
    expected_outputs = {
        "package_dir": str(package_dir),
        "target_file": str(target_file),
        "boltzgen_output_dir": str(output_dir),
        "steps_manifest": str(output_dir / "steps.yaml"),
        "log_file": str(log_file),
    }
    for key, value in previous_expected.items():
        if key in expected_outputs:
            continue
        expected_outputs[key] = _rebase_expected_output(
            value,
            previous_output_dir=previous_output_dir,
            output_dir=output_dir,
            previous_package_dir=previous_package_dir,
            package_dir=package_dir,
        )
    run_spec.expected_outputs = expected_outputs


def _rebase_expected_output(
    value: str,
    *,
    previous_output_dir: Path,
    output_dir: Path,
    previous_package_dir: Path,
    package_dir: Path,
) -> str:
    text = str(value)
    previous_output = str(previous_output_dir)
    if text == previous_output or text.startswith(previous_output + "/"):
        return str(output_dir) + text[len(previous_output):]
    previous_package = str(previous_package_dir)
    if text == previous_package or text.startswith(previous_package + "/"):
        return str(package_dir) + text[len(previous_package):]
    return text


def _resolve_path(root: Path, value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _resolve_optional_path(root: Path, value: Union[str, Optional[Path]]) -> Optional[Path]:
    return _resolve_path(root, value) if value else None


def _resolve_optional_model_path(root: Path, value: Union[str, Optional[Path]]):
    if not value:
        return None
    text = str(value)
    return text if text.startswith("/") else _resolve_path(root, value)


def _write_json(path: Path, data: Any) -> Path:
    return atomic_write_json(path, redact_sensitive(data))


def _task_flag(prefix: str, job_id: str, attempt: int) -> str:
    safe_prefix = re.sub(r"[^A-Za-z0-9_]+", "_", prefix).strip("_") or "binder"
    digest = hashlib.sha1(job_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe_prefix}_{digest}_try{attempt}_{int(time.time())}"


if __name__ == "__main__":
    raise SystemExit(main())
