import ast
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical safety bounds (P0 guardrail).
#
# Hard limits on numeric search knobs. LLM agents may propose values inside
# these ranges, but the orchestrator clamps any proposal to this safe region
# before it ever reaches a runner. This prevents the v4 failure mode where an
# LLM pushed ``alpha`` from the 0.001 baseline to 0.7 (700x), which collapsed
# interface iPTM across the whole run.
#
# Fields:
#   min / max     : hard physical bounds; values outside are clamped.
#   default       : strong baseline; when no evidence supports a change, prefer this.
#   max_step_ratio: max multiplicative change vs current value per round (log_scale knobs).
#   max_step_abs  : max absolute change vs current value per round (linear knobs).
#   log_scale     : whether the knob lives on a multiplicative scale.
# ---------------------------------------------------------------------------
PARAM_BOUNDS: Dict[str, Dict[str, Any]] = {
    "alpha": {"min": 0.001, "max": 0.05, "default": 0.001, "max_step_ratio": 3.0, "log_scale": True},
    "exploration_ratio": {"min": 0.20, "max": 0.60, "default": 0.35, "max_step_abs": 0.15, "log_scale": False},
    "noise_scale": {"min": 0.6, "max": 0.9, "default": 0.7, "max_step_abs": 0.15, "log_scale": False},
    "step_scale": {"min": 0.6, "max": 1.0, "default": 0.8, "max_step_abs": 0.2, "log_scale": False},
    "template_conditioned_fraction": {"min": 0.0, "max": 0.8, "default": 0.5, "max_step_abs": 0.25, "log_scale": False},
}


# These keys are the knobs that the orchestrator can apply to the next-round
# HarnessConfig / BoltzGen params. LLM agents should only emit these keys when
# they intend to change executable configuration.
ADJUSTABLE_CONFIG_PARAMETERS: Dict[str, str] = {
    "diffusion_batch_size": "BoltzGen diffusion batch size.",
    "step_scale": "BoltzGen diffusion step scale.",
    "noise_scale": "BoltzGen diffusion noise scale.",
    "alpha": "BoltzGen diversity/adherence trade-off.",
    "inverse_fold_avoid": "Amino-acid symbols to avoid in inverse folding.",
    "filter_biased": "BoltzGen biased sequence filter choice flag.",
    "config_overrides": "Validated BoltzGen --config token groups.",
    "auxiliary_hotspots": "Harness-translated nearby target residues.",
    "epitope_crop_mode": "Harness target-context translator mode.",
    "template_conditioned_fraction": "Allocation intent used only with effective templates.",
}

DEPRECATED_METADATA_CONFIG_PARAMETERS: Dict[str, str] = {
    "hotspot_weight": "Deprecated numeric hotspot hint; audit only.",
    "prioritize_hotspots": "Deprecated hotspot intent; audit only.",
    "clash_filter": "Deprecated abstract clash hint; audit only.",
    "module_guided_repair": "Deprecated abstract module repair hint; audit only.",
    "module_guided_exploitation": "Deprecated abstract module exploitation hint; audit only.",
    "exploit_fragment_modules": "Deprecated fragment-module IDs; audit only.",
}



# These executable keys are orchestrator/internal-only.  They must not appear in
# LLM-facing contracts because a model can otherwise turn a non-success local
# fragment suggestion into an executable BoltzGen redesign template, bypassing
# the success-gated FragmentTemplateMiningAgent.
INTERNAL_ONLY_CONFIG_PARAMETERS: Dict[str, str] = {
    "binder_template": "Internal template-conditioned binder generation dict produced only by FragmentTemplateMiningAgent/orchestrator after PAE-gated provenance checks.",
    "binder_templates": "Internal Top-K template-conditioned binder generation dicts produced only by FragmentTemplateMiningAgent/orchestrator after PAE-gated provenance checks.",
    "binder_template_proximity": "Internal default Angstrom within_proximity radius around fixed template residues.",
    "target_include": "Internal harness-derived epitope crop emitted only by FragmentTemplateMiningAgent when epitope_crop_mode is enabled.",
    "target_binding_types": "Internal harness-derived binding-site crop emitted only by FragmentTemplateMiningAgent when epitope_crop_mode is enabled.",
    "structure_groups": "Internal carry-through for harness-derived epitope crops; not an LLM-editable target definition.",
    "exploration_ratio": "Internal active-learning rollback perturbation; not an LLM-editable BoltzGen parameter.",
    "length_delta_hint": "Internal active-learning branch perturbation hint; not an LLM-editable BoltzGen parameter.",
    "avoid_binder_lengths": "Internal active-learning length-avoidance hint; not an LLM-editable BoltzGen parameter.",
}

USER_OWNED_EXECUTABLE_CONFIG_PARAMETERS: Dict[str, str] = {
    "additional_filters": "User-owned BoltzGen metric filters. Accepted from static config and preserved as BoltzGen --additional_filters values; LLM agents must not edit this field.",
    "binder_lengths": "Harness-owned next-round length set. LLM/policy agents may propose discrete lengths, then the orchestrator clamps them to the user binder_length_range before writing one BoltzGen design spec per length.",
    "fragment_templates_enabled": "User-owned opt-in switch for executable fragment-template conditioned branches. Defaults false; LLM agents must not enable it.",
    "fragment_template_top_k": "User-owned maximum number of PAE-gated preserve templates to run as separate template-conditioned branches when fragment_templates_enabled is true.",
}

ADJUSTABLE_CONFIG_KEYS = frozenset(ADJUSTABLE_CONFIG_PARAMETERS)
# These sampler knobs are selected from finite catalogs by the probabilistic
# decision layer. Agent prose may discuss direction, but executable free-form
# values must be stripped before proposals reach orchestration.
PROBABILISTIC_SAMPLER_KEYS = frozenset({"alpha", "noise_scale", "step_scale"})
INTERNAL_ONLY_CONFIG_KEYS = frozenset(INTERNAL_ONLY_CONFIG_PARAMETERS)
USER_OWNED_EXECUTABLE_CONFIG_KEYS = frozenset(USER_OWNED_EXECUTABLE_CONFIG_PARAMETERS)
DEPRECATED_METADATA_CONFIG_KEYS = frozenset(DEPRECATED_METADATA_CONFIG_PARAMETERS)
PUBLIC_AGENT_CONFIG_KEYS = ADJUSTABLE_CONFIG_KEYS | frozenset({"binder_lengths"})

_BOLTZGEN_NATIVE_CLI_KEYS = frozenset({
    "protocol", "num_designs", "budget", "diffusion_batch_size", "step_scale", "noise_scale",
    "inverse_fold_num_sequences", "inverse_fold_avoid", "alpha", "refolding_rmsd_threshold",
    "devices", "num_workers", "filter_biased", "use_kernels", "cache", "moldir",
    "inverse_fold_checkpoint", "folding_checkpoint", "affinity_checkpoint", "design_checkpoints",
    "metrics_override", "size_buckets", "reuse", "skip_inverse_folding", "only_inverse_fold",
    "additional_filters", "config_overrides", "steps",
})
_BOLTZGEN_ADAPTER_KEYS = frozenset({
    "target_include", "target_binding_types", "structure_groups", "target_chain", "target_res_index",
    "not_binding", "binder_chain", "binder_sequence", "binder_binding_types", "residue_constraints",
    "cyclic", "constraints", "total_len", "binder_structure_groups", "binder_template",
    "binder_templates", "binder_template_proximity", "binder_lengths",
})
_BOLTZGEN_RUNTIME_KEYS = frozenset({
    "task_id", "num_designs_per_round", "max_binders_per_round", "binder_length_range",
    "binder_length_step", "host_count", "GPUName", "taiji_timeout", "taiji_multi_host_mode",
    "taiji_submit_host_num", "taiji_host_num_requested", "analysis_location", "run_analysis_on_taiji",
    "run_filtering", "keep_unfiltered_for_failure_analysis", "package_dir", "disable_gpu_distribution",
    "disable_gpu_sharding", "log_heartbeat_seconds", "boltzgen_log_heartbeat_seconds",
    "heartbeat_seconds", "silence", "silent", "silence_logging", "boltzgen_silence_log",
    "boltzgen_silent_log",
})
_BOLTZGEN_INTERNAL_METADATA_KEYS = frozenset({
    "exploration_ratio", "length_delta_hint", "avoid_binder_lengths",
    "template_conditioned_fraction", "fragment_templates_enabled",
    "fragment_template_top_k", "epitope_crop_mode", "auxiliary_hotspots", "round_budget_weight",
    "round_budget_allocation", "exploration_arm", "branch_id", "controlled_comparison",
    "multi_taiji_host_shard", "native_taiji_multi_host", "execution_retry_source_job_id",
    "execution_retry_preserve_budget", "resource_retry_degradation", "binder_length_guardrail",
    "blocked_strategy_arms", "retry_metadata", "binder_template_dropped", "binder_template_drop_reason",
    "strategy_intent", "binding_site_policy", "target_context_policy", "sampler_policy",
    "selection_policy", "effective_intervention_digest", "deprecated_strategy_audit",
    "expanded_binding_residues", "negative_binding_residues", "binding_residue_provenance",
    "template_requested", "template_staged", "template_applied", "template_drop_reason",
    "effective_template_id", "template_free_exploration", "template_index", "template_count",
    "fragment_template_gate", "fragment_interchain_pae_max", "fragment_template_min_quality",
    "auto_binder_length", "sampler_bounds", "weighted_hotspot_conditioning",
    "binder_template_dropped", "binder_template_drop_reason",
    "logical_branch_id", "logical_job_id", "arm_id", "parent_branch_id", "intent_digest",
    "harness_template_policy", "fragment_template_max_templates", "fragment_template_library_size",
    "fragment_template_max_fixed_fraction", "fragment_template_min_designable_residues",
    "fragment_template_min_alignment_coverage", "fragment_template_max_target_patch_rmsd",
    "fragment_template_require_pae", "fragment_template_package_failure_policy",
    "fragment_template_utility_decay", "fragment_template_cooldown_failures",
    "fragment_template_blacklist_failures",
    # v24 identity, execution, allocation, lineage, and audit payloads.
    "job_identity", "arm_rank", "arm_root", "arm_digest", "execution_job_id",
    "execution_slot", "identity_execution_slot", "target_identity_digest",
    "round_budget_resolution", "immutable_branch_plan", "arm_gpu_allocation",
    "template_application_plan", "template_execution_identity", "lineage_identity",
    "template_replay_classification", "replay_source_job_id",
    "replay_source_job_identity_digest", "execution_retry_source_job_id",
    "parameter_catalog", "parameter_catalog_digest", "final_parameter_state",
    "resolved_intervention_plan", "intent_digest", "sampler_policy_applied",
    "sampler_policy_status", "target_context_policy_status", "execution_retry_preserve_budget",
    "retry_metadata", "resource_retry_degradation", "blocked_strategy_arms",
    "native_taiji_multi_host", "multi_taiji_host_shard", "taiji_host_num_requested",
    "binder_length_guardrail", "round_budget_weight", "round_budget_allocation",
    "template_conditioned", "template_requested", "template_staged", "template_applied",
    "template_drop_reason", "effective_template_id", "effective_template_digest",
})

IDENTITY_POLICY_METADATA_KEYS = frozenset({
    "task_id", "branch_id", "logical_branch_id", "logical_job_id", "arm_id",
    "parent_branch_id", "exploration_arm", "intent_digest", "effective_intervention_digest",
})
TEMPLATE_POLICY_METADATA_KEYS = frozenset({
    key for key in _BOLTZGEN_INTERNAL_METADATA_KEYS
    if "template" in key
}) | frozenset({"harness_template_policy"})


def _policy_class(key: str) -> Optional[str]:
    if key in IDENTITY_POLICY_METADATA_KEYS:
        return "identity_policy"
    if key in TEMPLATE_POLICY_METADATA_KEYS:
        return "template_policy"
    return None


BOLTZGEN_PARAMETER_CONTRACT: Dict[str, Dict[str, Any]] = {}
for _key in sorted(_BOLTZGEN_NATIVE_CLI_KEYS):
    BOLTZGEN_PARAMETER_CONTRACT[_key] = {"owner": "runner_or_resolver", "type": "runner", "partition": "runner", "consumer": "boltzgen_cli", "translator": None, "llm_delta": _key in ADJUSTABLE_CONFIG_KEYS, "policy_class": _policy_class(_key)}
for _key in sorted(_BOLTZGEN_ADAPTER_KEYS):
    BOLTZGEN_PARAMETER_CONTRACT[_key] = {"owner": "user_or_harness", "type": "adapter_translated", "partition": "adapter", "consumer": "boltzgen_design_spec", "translator": "BoltzGenAdapter", "llm_delta": _key in PUBLIC_AGENT_CONFIG_KEYS, "policy_class": _policy_class(_key)}
for _key in sorted(_BOLTZGEN_RUNTIME_KEYS):
    BOLTZGEN_PARAMETER_CONTRACT[_key] = {"owner": "executor", "type": "runtime_resource", "partition": "runtime", "consumer": "DesignSpecAgent_or_executor", "translator": None, "llm_delta": False, "policy_class": _policy_class(_key)}
for _key in sorted(_BOLTZGEN_INTERNAL_METADATA_KEYS):
    BOLTZGEN_PARAMETER_CONTRACT[_key] = {"owner": "harness", "type": "internal_metadata", "partition": "orchestration", "consumer": "orchestrator_or_audit", "translator": None, "llm_delta": False, "policy_class": _policy_class(_key)}
for _key in sorted(DEPRECATED_METADATA_CONFIG_KEYS):
    BOLTZGEN_PARAMETER_CONTRACT[_key] = {"owner": "legacy_reader", "type": "deprecated_metadata", "partition": "orchestration", "consumer": "audit_only", "translator": None, "llm_delta": False, "policy_class": _policy_class(_key)}

BOLTZGEN_FULL_JOB_CONFIG_KEYS = frozenset(BOLTZGEN_PARAMETER_CONTRACT)
ALL_EXECUTABLE_CONFIG_KEYS = BOLTZGEN_FULL_JOB_CONFIG_KEYS


def parameter_contract_entry(key: str) -> Optional[Dict[str, Any]]:
    entry = BOLTZGEN_PARAMETER_CONTRACT.get(str(key))
    return dict(entry) if entry is not None else None


def partition_config_parameters(config: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Partition a config from the parameter contract, retaining unknown keys."""
    partitions: Dict[str, Dict[str, Any]] = {
        "runner": {}, "adapter": {}, "runtime": {}, "orchestration": {}, "unknown": {},
    }
    for key, value in dict(config or {}).items():
        entry = BOLTZGEN_PARAMETER_CONTRACT.get(str(key))
        partition = str((entry or {}).get("partition") or "unknown")
        if partition not in partitions:
            partition = "unknown"
        partitions[partition][str(key)] = value
    return partitions

# Primary families and safety companions remain classified for reporting and
# compatibility only. Central merge permits every normalized change to coexist;
# any future compatibility restriction must be explicit and independently audited.
PRIMARY_PARAMETER_FAMILIES: Dict[str, str] = {
    "alpha": "sampling", "noise_scale": "sampling", "step_scale": "sampling",
    "gamma_0": "sampling", "diffusion_batch_size": "sampling",
    "auxiliary_hotspots": "targeting", "epitope_crop_mode": "targeting",
    "inverse_fold_avoid": "sequence", "filter_biased": "sequence", "temperature": "sequence",
    "template_conditioned_fraction": "templating",
    "binder_lengths": "length",
}
SAFETY_COMPANION_KEYS = frozenset({"config_overrides"})


def enforce_single_primary_family(changes: Mapping[str, Any], *, sources: Optional[Mapping[str, str]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return every normalized change without experiment-arm rejections.

    The historical function name and ``sources`` argument remain for compatibility.
    Executable primary families and native safety controls such as
    ``config_overrides`` coexist through central merge. Deprecated hotspot/clash/module
    hints are excluded by the public contract. The rejection list stays empty unless a future explicit compatibility
    rule is introduced.
    """
    return dict(changes or {}), []


def _allowed_config_keys(*, include_internal: bool = False, allowed_keys: Optional[Iterable[str]] = None) -> frozenset:
    # Public agent outputs may include only true tuning knobs plus harness-owned
    # binder_lengths. User-owned executable fields such as additional_filters are
    # preserved by full-job validation but are not part of the LLM output surface.
    # Internal-only fields still require an explicitly trusted source.
    if allowed_keys is not None:
        return frozenset(str(key) for key in allowed_keys)
    public_keys = PUBLIC_AGENT_CONFIG_KEYS
    return ALL_EXECUTABLE_CONFIG_KEYS if include_internal else public_keys


def supported_config_changes(
    changes: Optional[Mapping[str, Any]],
    *,
    include_internal: bool = False,
    allowed_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Drop unsupported keys and normalize executable config value shapes.

    This is the LLM-facing sanitizer used by diagnostics, quality analysis,
    input configuration, policy merge, and resume paths.  It intentionally does
    more than key filtering: malformed values are removed before they can reach
    policy code that expects concrete Python types.
    """
    allowed = _allowed_config_keys(include_internal=include_internal, allowed_keys=allowed_keys)
    out: Dict[str, Any] = {}
    for key, value in dict(changes or {}).items():
        if key not in allowed:
            continue
        normalized, valid = _normalize_config_value(key, value)
        if valid:
            out[key] = normalized
    return out


def strip_probabilistic_sampler_keys(
    changes: Optional[Mapping[str, Any]],
    *,
    sampler_keys: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Remove only finite-catalog sampler keys, preserving every other key."""
    source = dict(changes or {})
    blocked = frozenset(str(key) for key in (sampler_keys if sampler_keys is not None else PROBABILISTIC_SAMPLER_KEYS))
    ignored = [key for key in source if key in blocked]
    return {key: value for key, value in source.items() if key not in blocked}, ignored


def unsupported_config_keys(
    changes: Optional[Mapping[str, Any]],
    *,
    include_internal: bool = False,
    allowed_keys: Optional[Iterable[str]] = None,
) -> list:
    allowed = _allowed_config_keys(include_internal=include_internal, allowed_keys=allowed_keys)
    return [k for k in dict(changes or {}) if k not in allowed]


def invalid_config_value_keys(
    changes: Optional[Mapping[str, Any]],
    *,
    include_internal: bool = False,
    allowed_keys: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return allowed keys whose values fail the executable config shape contract."""
    allowed = _allowed_config_keys(include_internal=include_internal, allowed_keys=allowed_keys)
    invalid: List[str] = []
    for key, value in dict(changes or {}).items():
        if key not in allowed:
            continue
        _, valid = _normalize_config_value(key, value)
        if not valid:
            invalid.append(f"{key}:invalid_shape")
    return invalid


def canonicalize_config_parameter_value(key: str, value: Any) -> Any:
    """Canonicalize one value using the executable parameter contract.

    Invalid values remain distinct instead of being silently converted or
    dropped; validation is still responsible for rejecting them.
    """
    normalized, valid = _normalize_config_value(str(key), value)
    return normalized if valid else value


def _normalize_config_value(key: str, value: Any) -> Tuple[Any, bool]:
    if value is None:
        return None, False
    if key == "binder_lengths":
        lengths = _coerce_int_list(value)
        return lengths, bool(lengths)
    if key == "config_overrides":
        overrides = _coerce_config_overrides(value)
        return overrides, bool(overrides) or _is_explicit_empty_config_overrides(value)
    if key in {"auxiliary_hotspots", "avoid_binder_lengths"}:
        values = _coerce_list(value)
        return [str(item).strip() for item in values if str(item).strip()], True
    if key == "filter_biased":
        boolean = _coerce_bool(value)
        return ("true" if boolean else "false"), boolean is not None
    if key == "use_kernels":
        token = _normalize_choice_token(value, frozenset({"auto", "true", "false"}))
        return token, token is not None
    if key in {"fragment_templates_enabled", "run_filtering", "auto_binder_length"}:
        boolean = _coerce_bool(value)
        return boolean, boolean is not None
    if key == "steps":
        values = _coerce_list(value)
        return [str(item).strip().lower() for item in values if str(item).strip()], bool(values)
    if key == "devices":
        integer = _coerce_int(value)
        return integer, integer is not None and integer > 0
    if key in {"diffusion_batch_size", "top_k", "max_rounds", "num_designs", "num_designs_per_round", "max_binders_per_round", "fragment_template_top_k", "inverse_fold_num_sequences", "num_workers", "budget", "length_delta_hint"}:
        integer = _coerce_int(value)
        return integer, integer is not None
    if key in {"step_scale", "noise_scale", "alpha", "gamma_0", "temperature", "exploration_ratio", "refolding_rmsd_threshold", "fragment_interchain_pae_max", "template_conditioned_fraction", "binder_template_proximity"}:
        number = _coerce_float(value)
        return number, number is not None
    return value, True


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        text = str(value).strip() if isinstance(value, str) else value
        if text == "":
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, bool):
            return None
        text = str(value).strip() if isinstance(value, str) else value
        if text == "":
            return None
        number = float(text)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _normalize_choice_token(value: Any, choices: frozenset) -> Optional[str]:
    if isinstance(value, bool):
        token = "true" if value else "false"
    else:
        token = str(value).strip().lower()
    return token if token in choices else None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "on"}:
            return True
        if text in {"false", "no", "n", "0", "off"}:
            return False
    return None


def _coerce_list(value: Any) -> List[Any]:
    if isinstance(value, str):
        parsed = _literal_or_jsonish(value)
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
        if parsed is not value:
            return [parsed]
        return [part for part in re.split(r"[,\s]+", value.strip()) if part]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _coerce_int_list(value: Any) -> List[int]:
    raw = _coerce_list(value)
    vals: List[int] = []
    for item in raw:
        integer = _coerce_int(item)
        if integer is not None:
            vals.append(integer)
    return sorted({v for v in vals if v > 0})


def _literal_or_jsonish(value: str) -> Any:
    text = value.strip()
    if not text:
        return value
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def _is_explicit_empty_config_overrides(value: Any) -> bool:
    """Return whether ``value`` intentionally represents no override groups."""
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return True
        parsed = _literal_or_jsonish(value)
        return parsed is not value and _is_explicit_empty_config_overrides(parsed)
    if isinstance(value, (list, tuple, set, Mapping)):
        return len(value) == 0
    return False


def _setting_tokens(value: Any) -> List[str]:
    """Render mapping/list/scalar settings without applying model semantics."""
    if isinstance(value, Mapping):
        return [f"{str(key).strip()}={str(item).strip()}" for key, item in value.items() if str(key).strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    token = str(value).strip()
    return [token] if token else []


def _string_override_group(value: str) -> List[List[str]]:
    text = value.strip()
    if not text:
        return []
    parsed = _literal_or_jsonish(value)
    if parsed is not value:
        return _coerce_config_overrides(parsed)
    tokens = [token.strip() for token in re.split(r"[,\s]+", text) if token.strip()]
    if not tokens:
        return []
    if all("=" in token for token in tokens):
        # Compatibility for historical single-setting strings such as
        # ``filter_bindingsite=true``. Infer the established filtering group for
        # any key=value-only string while preserving every sibling token.
        return [["filtering", *tokens]]
    if len(tokens) >= 2 and "=" not in tokens[0]:
        return [tokens]
    return []


def _coerce_config_overrides(value: Any) -> List[List[str]]:
    """Parse override inputs into lossless token groups.

    This contract-level coercer handles only representational forms (literal
    strings, mappings, flat groups, and nested groups). It deliberately does not
    validate BoltzGen step names or setting keys and never removes individual
    tokens; model-specific validation belongs in ``model_input_spec``.
    """
    parsed = _literal_or_jsonish(value) if isinstance(value, str) else value
    if isinstance(parsed, Mapping):
        section = parsed.get("section") or parsed.get("group")
        if section is not None:
            if "key" in parsed and "value" in parsed:
                setting_key = str(parsed.get("key")).strip()
                settings = [f"{setting_key}={str(parsed.get('value')).strip()}"] if setting_key else []
            else:
                setting_value = None
                for key in ("settings", "setting", "overrides", "override", "value", "key"):
                    if key in parsed:
                        setting_value = parsed[key]
                        break
                settings = _setting_tokens(setting_value) if setting_value is not None else []
            section_token = str(section).strip()
            return [[section_token, *settings]] if section_token and settings else []
        groups: List[List[str]] = []
        for group, settings_value in parsed.items():
            group_token = str(group).strip()
            settings = _setting_tokens(settings_value)
            if group_token and settings:
                groups.append([group_token, *settings])
        return groups
    if isinstance(parsed, str):
        return _string_override_group(parsed)
    if not isinstance(parsed, (list, tuple, set)):
        return []
    items = list(parsed)
    if not items:
        return []
    if all(not isinstance(part, (list, tuple, set, Mapping)) for part in items):
        tokens = [str(part).strip() for part in items if str(part).strip()]
        if len(tokens) == 1 and isinstance(items[0], str):
            return _string_override_group(items[0])
        if len(tokens) >= 2 and "=" not in tokens[0]:
            return [tokens]
        if tokens and all("=" in token for token in tokens):
            return [["filtering", *tokens]]
        return []
    out: List[List[str]] = []
    for item in items:
        out.extend(_coerce_config_overrides(item))
    return out


def render_config_parameter_contract(adjustable: Optional[Mapping[str, str]] = None) -> str:
    lines = [
        "Executable next-round config parameters you may change.",
        "Only use keys from this list in recommended_config, parameter_changes, or config_parameter_changes.",
        "Do not rewrite user-owned task/search/resource fields such as budget, num_designs, num_designs_per_round, max_binders_per_round, binder_length_range, inverse_fold_num_sequences, refolding_rmsd_threshold, exploit_fragment_modules, module_guided_exploitation, search_space, resource, target_include, target_binding_types, steps, run_filtering, additional_filters, fragment_templates_enabled, fragment_template_top_k, or hotspots; use auxiliary_hotspots for small nearby additions instead of rewriting hotspots.",
        "Never emit additional_filters from an LLM/policy output. Preserve static user filters exactly; do not add designfolding_iptm filters there.",
        "If refolded-binder stability needs attention, discuss designfolding-filter_rmsd or harness post-processing in rationale; do not invent filter_designfolding_iptm.",
        "Legacy hotspot_weight/prioritize_hotspots/clash_filter/module fields are audit-only and must never be emitted as executable changes.",
        "binder_lengths may be proposed as a discrete next-round length set, but only within the user binder_length_range; the orchestrator will clamp invalid lengths before submission.",
        "If a needed intervention cannot be represented by these keys, describe it as rationale/risk, not as an executable config change.",
        "Respect all user-provided task YAML values as hard constraints; the orchestrator will ignore attempts to rewrite them.",
        "When the user YAML has epitope_crop_mode disabled, do not enable hotspot_focus/engaged_focus/union/auto unless allow_agent_epitope_crop=true is present.",
    ]
    parameters = dict(adjustable or ADJUSTABLE_CONFIG_PARAMETERS)
    for key, description in parameters.items():
        lines.append(f"- {key}: {description}")
    return "\n".join(lines)


def render_param_bounds_contract(bounds: Optional[Mapping[str, Mapping[str, Any]]] = None) -> str:
    """LLM-readable description of hard physical bounds and per-round change limits."""
    lines = [
        "PHYSICAL SAFETY BOUNDS for numeric search knobs (hard limits, enforced by the orchestrator).",
        "You may tune these knobs, but ONLY within the stated [min, max] range AND within the per-round change limit.",
        "When no evidence supports a change, keep the default; the default is a strong baseline.",
    ]
    for key, b in dict(bounds or PARAM_BOUNDS).items():
        parts = [f"range=[{b['min']}, {b['max']}]", f"default={b['default']}"]
        if b.get("log_scale"):
            parts.append(f"per-round multiplicative change <= {b.get('max_step_ratio', 3.0)}x")
        elif b.get("max_step_abs") is not None:
            parts.append(f"per-round absolute change <= {b['max_step_abs']}")
        lines.append(f"- {key}: " + ", ".join(parts))
    if "alpha" in dict(bounds or PARAM_BOUNDS):
        lines.append(
            "Special rule for alpha: alpha is the diversity/adherence trade-off. High alpha destroys interface adherence. "
            "Only raise alpha when >30% of candidates are tagged diversity_collapse, and never above 0.05."
        )
    return "\n".join(lines)


def clamp_param_value(key: str, proposed: Any, *, current: Optional[float] = None) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Clamp a single numeric knob to its physical bound and per-round change rate.

    Returns ``(clamped_value, note)``. ``note`` is ``None`` when the value was
    already safe. Returns ``(None, None)`` when the key has no bound or the
    value is not numeric.
    """
    bound = PARAM_BOUNDS.get(key)
    if bound is None:
        return None, None
    try:
        value = float(proposed)
    except (TypeError, ValueError):
        return None, None

    original = value
    reasons: List[str] = []

    lo, hi = float(bound["min"]), float(bound["max"])
    if value < lo:
        value = lo
        reasons.append(f"below min {lo}")
    elif value > hi:
        value = hi
        reasons.append(f"above max {hi}")

    if current is not None:
        try:
            cur = float(current)
        except (TypeError, ValueError):
            cur = None
        if cur is not None and cur > 0:
            if bound.get("log_scale"):
                ratio = float(bound.get("max_step_ratio", 3.0))
                ceil_v = cur * ratio
                floor_v = cur / ratio
                if value > ceil_v:
                    value = min(hi, ceil_v)
                    reasons.append(f"change-rate capped to {ratio}x of current {cur}")
                elif value < floor_v:
                    value = max(lo, floor_v)
                    reasons.append(f"change-rate capped to 1/{ratio}x of current {cur}")
            elif bound.get("max_step_abs") is not None:
                step = float(bound["max_step_abs"])
                if value - cur > step:
                    value = min(hi, cur + step)
                    reasons.append(f"step capped to +{step} from current {cur}")
                elif cur - value > step:
                    value = max(lo, cur - step)
                    reasons.append(f"step capped to -{step} from current {cur}")

    if not reasons:
        return value, None
    note = {"parameter": key, "proposed": original, "clamped_to": round(value, 6), "reasons": reasons}
    return value, note



def _clamp_value_against_bound(key: str, proposed: Any, bound: Mapping[str, Any], *, current: Optional[float] = None) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    try:
        value = float(proposed)
    except (TypeError, ValueError):
        return None, None
    original = value
    reasons: List[str] = []
    lo, hi = float(bound["min"]), float(bound["max"])
    if value < lo:
        value = lo; reasons.append(f"below min {lo}")
    elif value > hi:
        value = hi; reasons.append(f"above max {hi}")
    if current is not None:
        try: cur = float(current)
        except (TypeError, ValueError): cur = None
        if cur is not None and cur > 0:
            if bound.get("log_scale"):
                ratio = float(bound.get("max_step_ratio", 3.0)); floor_v, ceil_v = cur / ratio, cur * ratio
                if value > ceil_v: value = min(hi, ceil_v); reasons.append(f"change-rate capped to {ratio}x of current {cur}")
                elif value < floor_v: value = max(lo, floor_v); reasons.append(f"change-rate capped to 1/{ratio}x of current {cur}")
            elif bound.get("max_step_abs") is not None:
                step = float(bound["max_step_abs"])
                if value > cur + step: value = min(hi, cur + step); reasons.append(f"step capped to +{step} from current {cur}")
                elif value < cur - step: value = max(lo, cur - step); reasons.append(f"step capped to -{step} from current {cur}")
    note = None if not reasons else {"parameter": key, "proposed": original, "clamped_to": round(value, 6), "reasons": reasons}
    return value, note

def clamp_config_with_inertia(changes: Optional[Mapping[str, Any]], *, current_config: Optional[Mapping[str, Any]] = None, bounds: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Apply physical bounds + per-round change-rate limits to a config dict.

    Returns ``(clamped_changes, clamp_notes)``.
    """
    out = dict(changes or {})
    cur = dict(current_config or {})
    notes: List[Dict[str, Any]] = []
    effective_bounds = dict(bounds or PARAM_BOUNDS)
    for key, bound in effective_bounds.items():
        if key not in out or out[key] is None:
            continue
        if bounds is None:
            value, note = clamp_param_value(key, out[key], current=cur.get(key))
        else:
            value, note = _clamp_value_against_bound(key, out[key], bound, current=cur.get(key))
        if value is None:
            continue
        out[key] = value
        if note is not None:
            notes.append(note)
    return out, notes