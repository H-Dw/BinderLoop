
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from binderloop.agents.config_parameter_contract import (
    PUBLIC_AGENT_CONFIG_KEYS,
    invalid_config_value_keys,
    parameter_contract_entry,
    partition_config_parameters,
    render_config_parameter_contract,
    supported_config_changes,
    unsupported_config_keys,
)
from binderloop.agents.context_compaction import compact_context_for_config_validation
from binderloop.agents.model_input_spec import (
    ModelInputSpec,
    analyze_runtime_error,
    apply_error_findings,
    check_one_param,
    get_model_input_spec,
)
from binderloop.llm import OpenAICompatibleClient


@dataclass
class ConfigValidationResult:
    """Result of checking a model input config before submission or after failure."""

    target_model: str
    activation: str
    llm_used: bool
    is_valid: bool
    corrected_config: Dict[str, Any] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    # Top-level summary of whether a downstream runtime failure was config-related
    # and which parameter(s) it pointed at. ``None`` for pure pre-submit checks
    # where no runner error was provided.
    runtime_error_analysis: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    # Additive v2 fields. Legacy readers may continue using is_valid and
    # corrected_config; submission code should prefer is_submittable.
    schema_version: int = 2
    is_submittable: Optional[bool] = None
    validated_partition: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    normalizations: List[Dict[str, Any]] = field(default_factory=list)
    removals: List[Dict[str, Any]] = field(default_factory=list)
    semantic_changes: List[Dict[str, Any]] = field(default_factory=list)
    missing_required_keys: List[str] = field(default_factory=list)
    requires_refinalization: bool = False

    def __post_init__(self) -> None:
        # Legacy callers did not provide this additive v2 field. In that case
        # inherit the legacy validity verdict; never default an invalid result
        # to submittable.
        self.is_submittable = bool(self.is_valid) if self.is_submittable is None else bool(self.is_submittable)


class ConfigValidationAgent:
    """Validate and repair executable model configs before they reach a runner.

    The LLM path performs model-specific semantic review.  The deterministic
    fallback enforces the most important executable shape constraints so config
    validation remains useful when no LLM endpoint is available.
    """

    SYSTEM_HEADER = """You are a strict input-config validator for automated protein binder design jobs.

Your job is to check whether the provided harness config can be submitted to the target model without violating the target model's expected parameter names, CLI/Hydra argument shape, and data formats.

Return JSON only with this schema:
{
  "is_valid": true,
  "corrected_config": {"only executable corrected keys": "..."},
  "issues": [
    {"parameter": "...", "severity": "info|warning|error", "problem": "...", "correction": "..."}
  ],
  "recommendations": [
    {"parameter": "...", "action": "...", "reason": "..."}
  ]
}
"""

    def __init__(self, llm: Optional[OpenAICompatibleClient] = None):
        self.llm = llm

    def _system_prompt(self, spec: ModelInputSpec) -> str:
        rules = "\n".join(f"- {rule}" for rule in spec.prompt_rules)
        prompt = f"{self.SYSTEM_HEADER}\nValidation requirements for target_model={spec.model}:\n{rules}\n"
        if spec.model == "boltzgen":
            prompt += "\n" + render_config_parameter_contract()
        return prompt

    def validate_for_submission(
        self,
        config: Mapping[str, Any],
        *,
        target_model: str = "boltzgen",
        context: Optional[Mapping[str, Any]] = None,
    ) -> ConfigValidationResult:
        return self.validate_full_job_config(config, target_model=target_model, context=context)

    def validate_full_job_config(
        self,
        config: Mapping[str, Any],
        *,
        target_model: str = "boltzgen",
        context: Optional[Mapping[str, Any]] = None,
    ) -> ConfigValidationResult:
        """Validate a complete runner job config before submission.

        Full job configs include harness/user-owned fields such as
        ``additional_filters``, target definitions, resource hints, and
        ``run_filtering``. The deterministic sanitizer is authoritative for
        submittability; any LLM review is advisory and must not veto submission.
        """
        return self._validate(
            config,
            target_model=target_model,
            activation="pre_submit",
            context=dict(context or {}),
            validation_mode="full_job_config",
        )

    def validate_agent_delta(
        self,
        config: Mapping[str, Any],
        *,
        target_model: str = "boltzgen",
        context: Optional[Mapping[str, Any]] = None,
    ) -> ConfigValidationResult:
        """Validate an LLM/agent-proposed config delta.

        This is the strict LLM output surface: user-owned full-job fields such
        as ``additional_filters``, target definitions, resource hints, steps,
        and run controls are stripped/reported instead of accepted.
        """
        return self._validate(
            config,
            target_model=target_model,
            activation="agent_delta",
            context=dict(context or {}),
            validation_mode="agent_delta",
        )

    def improve_after_failure(
        self,
        config: Mapping[str, Any],
        *,
        error_context: Mapping[str, Any],
        target_model: str = "boltzgen",
    ) -> ConfigValidationResult:
        return self._validate(
            config,
            target_model=target_model,
            activation="taiji_failure",
            context={"error_context": dict(error_context)},
            validation_mode="full_job_config",
        )

    def write_result(self, result: ConfigValidationResult, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _validate(
        self,
        config: Mapping[str, Any],
        *,
        target_model: str,
        activation: str,
        context: Mapping[str, Any],
        validation_mode: str,
    ) -> ConfigValidationResult:
        spec = get_model_input_spec(target_model)
        include_internal = validation_mode == "full_job_config"
        base_result = self._deterministic_validate(
            config,
            target_model=target_model,
            activation=activation,
            context=context,
            include_internal=include_internal,
        )
        base_result.raw = {**base_result.raw, "validation_mode": validation_mode}
        if not (self.llm and self.llm.available()):
            return base_result

        try:
            llm_user = compact_context_for_config_validation(
                target_model=target_model,
                activation=activation,
                config=config,
                deterministic_prefilter=asdict(base_result),
                context=context,
            )
            result = self.llm.chat_json(
                system=self._system_prompt(spec),
                user=llm_user,
                temperature=0.1,
                max_tokens=8000,
            )
        except Exception as exc:
            base_result.raw = {**base_result.raw, "llm_error": str(exc)}
            return base_result

        if not isinstance(result, dict) or "corrected_config" not in result:
            base_result.raw = {**base_result.raw, "llm_parse_failed": result}
            return base_result

        if validation_mode == "full_job_config":
            advisory_issues = _advisory_llm_issues(result.get("issues") or [])
            for key in unsupported_config_keys(dict(result.get("corrected_config") or {}), include_internal=True):
                advisory_issues.append({
                    "parameter": key,
                    "severity": "warning",
                    "problem": "LLM advisory proposed an unsupported full-job key; ignored.",
                    "correction": "Full-job pre-submit validation uses deterministic normalized config as source of truth.",
                    "resolved": True,
                    "advisory": True,
                })
            return ConfigValidationResult(
                target_model=target_model,
                activation=activation,
                llm_used=True,
                is_valid=base_result.is_valid,
                corrected_config=base_result.corrected_config,
                issues=list(base_result.issues) + advisory_issues,
                recommendations=list(result.get("recommendations") or []),
                runtime_error_analysis=base_result.runtime_error_analysis,
                raw={
                    "validation_mode": validation_mode,
                    "llm_result": result,
                    "deterministic_prefilter": asdict(base_result),
                    "llm_advisory_only": True,
                },
                is_submittable=base_result.is_submittable,
                validated_partition=base_result.validated_partition,
                normalizations=base_result.normalizations,
                removals=base_result.removals,
                semantic_changes=base_result.semantic_changes,
                missing_required_keys=base_result.missing_required_keys,
                requires_refinalization=base_result.requires_refinalization,
            )

        llm_corrected = dict(base_result.corrected_config)
        llm_update_raw = dict(result.get("corrected_config") or {})
        filter_keys = _spec_filter_keys(spec, include_internal=False)
        if spec.allowed_keys is not None:
            llm_update = supported_config_changes(llm_update_raw, include_internal=False, allowed_keys=filter_keys)
            ignored_llm_update = unsupported_config_keys(llm_update_raw, include_internal=False, allowed_keys=filter_keys)
        else:
            llm_update = dict(llm_update_raw)
            ignored_llm_update = []
        llm_corrected.update(llm_update)
        # Re-run the deterministic spec sanitizer on the LLM-merged config. This
        # is the key co-activation step: even if the LLM "corrects" a choice flag
        # back into a Python bool, the deterministic layer flattens it again.
        sanitized, sanitize_issues = _sanitize_config(llm_corrected, spec=spec, include_internal=False)
        issues = list(base_result.issues) + list(result.get("issues") or []) + sanitize_issues
        for key in ignored_llm_update:
            issues.append({
                "parameter": key,
                "severity": "warning",
                "problem": "LLM-proposed corrected_config key is unsupported or internal-only and was ignored.",
                "correction": "Only orchestrator/template-mining owned paths may set internal-only executable fields.",
                "resolved": True,
            })
        # Non-contract keys (e.g. harness-injected metadata like run_analysis_on_taiji)
        # are always stripped before submission, so an LLM marking them as a fatal
        # "error" must not block the round. Neutralise those so a submittable config
        # is not falsely rejected (this previously wasted whole rounds). Only models
        # with a registered executable-key contract are subject to this.
        if spec.allowed_keys is not None:
            for issue in issues:
                param = issue.get("parameter")
                if (
                    str(issue.get("severity") or "").lower() == "error"
                    and not issue.get("resolved")
                    and param is not None
                    and param not in spec.allowed_keys
                ):
                    issue["resolved"] = True
                    issue["resolution_note"] = "Non-executable key is stripped before submission; not a blocking error."
                elif _issue_resolved_by_sanitized_config(issue, sanitized, spec=spec):
                    issue["resolved"] = True
                    issue["resolution_note"] = "Final deterministic sanitizer output contains a valid executable value; not a blocking error."
        has_error = any(str(issue.get("severity") or "").lower() == "error" and not issue.get("resolved") for issue in issues)
        normalizations, removals, semantic_changes = _describe_config_changes(dict(config or {}), sanitized)
        # Submittability is decided by whether any *unresolved* error remains after
        # deterministic + LLM sanitization (and after non-executable keys are
        # neutralized above), NOT by a bare LLM ``is_valid: false`` verdict. A model
        # may flag a config as invalid without emitting a corresponding actionable
        # error issue; honoring that bare verdict would reject an already-clean,
        # submittable config. Since a pre-submit failure is treated as terminal
        # (non-retryable) downstream, a non-actionable veto would needlessly drop the
        # job for the whole round. Trust the deterministic error set instead.
        return ConfigValidationResult(
            target_model=target_model,
            activation=activation,
            llm_used=True,
            is_valid=not has_error,
            corrected_config=sanitized,
            issues=issues,
            recommendations=list(result.get("recommendations") or []),
            runtime_error_analysis=base_result.runtime_error_analysis,
            raw={"validation_mode": validation_mode, "llm_result": result, "deterministic_prefilter": asdict(base_result)},
            is_submittable=not has_error,
            validated_partition=partition_config_parameters(sanitized) if target_model == "boltzgen" else {"runner": dict(sanitized), "adapter": {}, "runtime": {}, "orchestration": {}, "unknown": {}},
            normalizations=normalizations,
            removals=removals,
            semantic_changes=semantic_changes,
            missing_required_keys=base_result.missing_required_keys,
            requires_refinalization=bool(semantic_changes),
        )

    @staticmethod
    def _deterministic_validate(
        config: Mapping[str, Any],
        *,
        target_model: str,
        activation: str,
        context: Mapping[str, Any],
        include_internal: bool,
    ) -> ConfigValidationResult:
        spec = get_model_input_spec(target_model)
        original = dict(config or {})
        corrected, issues = _sanitize_config(original, spec=spec, include_internal=include_internal)
        removed_keys = sorted(set(original) - set(corrected))
        readiness_required = include_internal and activation == "pre_submit" and target_model == "boltzgen"
        missing_required_keys = _missing_submission_keys(original, corrected) if readiness_required else []
        for key in missing_required_keys:
            issues.append({
                "parameter": key,
                "severity": "error",
                "problem": f"Required submission readiness key {key!r} is missing or not a positive integer.",
                "correction": "Finalize an explicit positive integer value before submission; no default is inferred.",
                "resolved": False,
            })
        # Retry-time intervention: when a downstream run failed, correlate the
        # runner's error with the config and repair the exact offending parameter.
        error_analysis: Dict[str, Any] = {}
        error_context = context.get("error_context")
        if error_context:
            error_text = json.dumps(error_context, ensure_ascii=False)
            findings = analyze_runtime_error(error_text)
            error_analysis = {
                "config_related": bool(findings),
                "findings": [
                    {"kind": f.kind, "parameter": f.parameter, "bad_value": f.bad_value, "allowed": f.allowed, "raw_line": f.raw_line}
                    for f in findings
                ],
            }
            if findings:
                issues.extend(apply_error_findings(corrected, findings, spec))

        normalizations, removals, semantic_changes = _describe_config_changes(original, corrected)
        validation_errors = [
            issue for issue in issues
            if str(issue.get("severity") or "").lower() == "error"
            and not issue.get("resolved")
            and issue.get("parameter") not in missing_required_keys
        ]
        is_valid = not validation_errors
        is_submittable = is_valid and not missing_required_keys

        raw: Dict[str, Any] = {
            "source": "deterministic_config_validation",
            "context_keys": sorted(context.keys()),
            "sanitized_partition": dict(corrected),
            "removed_keys": removed_keys,
            "tombstones": removed_keys,
        }
        if error_context:
            raw["runtime_error_analysis"] = error_analysis
        return ConfigValidationResult(
            target_model=target_model,
            activation=activation,
            llm_used=False,
            is_valid=is_valid,
            corrected_config=corrected,
            issues=issues,
            recommendations=[],
            runtime_error_analysis=error_analysis or None,
            raw=raw,
            is_submittable=is_submittable,
            validated_partition=partition_config_parameters(corrected) if target_model == "boltzgen" else {"runner": dict(corrected), "adapter": {}, "runtime": {}, "orchestration": {}, "unknown": {}},
            normalizations=normalizations,
            removals=removals,
            semantic_changes=semantic_changes,
            missing_required_keys=missing_required_keys,
            requires_refinalization=bool(semantic_changes),
        )



_REQUIRED_SUBMISSION_KEYS = ("num_designs", "budget")


def _positive_int(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and int(value) == float(value) and int(value) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def _missing_submission_keys(original: Mapping[str, Any], corrected: Mapping[str, Any]) -> List[str]:
    """Fail closed: readiness values must be explicit and survive sanitation."""
    return [key for key in _REQUIRED_SUBMISSION_KEYS if key not in original or not _positive_int(original.get(key)) or not _positive_int(corrected.get(key))]


def _change_class(key: str) -> Optional[str]:
    return (parameter_contract_entry(key) or {}).get("policy_class")


def _describe_config_changes(original: Mapping[str, Any], corrected: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Describe validation changes without treating metadata housekeeping as semantic."""
    normalizations: List[Dict[str, Any]] = []
    removals: List[Dict[str, Any]] = []
    semantic: List[Dict[str, Any]] = []
    for key in sorted(original):
        entry = parameter_contract_entry(key) or {}
        partition = str(entry.get("partition") or "unknown")
        policy_class = entry.get("policy_class")
        if key not in corrected:
            item = {"parameter": key, "before": original[key], "reason": "removed_by_validation", "partition": partition, "classification": "safe_stripping"}
            if policy_class:
                item["policy_class"] = policy_class
            removals.append(item)
            # Orchestration metadata and unknown keys are audit/housekeeping data;
            # stripping them must never force a new semantic finalization.
            continue
        if original[key] != corrected[key] or type(original[key]) is not type(corrected[key]):
            item = {"parameter": key, "before": original[key], "after": corrected[key], "partition": partition}
            if policy_class:
                item["policy_class"] = policy_class
            normalizations.append(item)
            if partition in {"runner", "adapter", "runtime"} and not _representationally_equivalent(key, original[key], corrected[key]):
                semantic.append(dict(item, change="normalization"))
    for key in sorted(set(corrected) - set(original)):
        entry = parameter_contract_entry(key) or {}
        partition = str(entry.get("partition") or "unknown")
        policy_class = entry.get("policy_class")
        item = {"parameter": key, "before": None, "after": corrected[key], "change": "addition", "partition": partition}
        if policy_class:
            item["policy_class"] = policy_class
        normalizations.append(dict(item, classification="safe_default"))
        # Deterministic defaults such as run_filtering=True are safe additions.
        # Metadata/unknown additions are also non-semantic; only an executable
        # runner/adapter/runtime addition can require refinalization.
        if partition in {"runner", "adapter", "runtime"} and key not in {"run_filtering"}:
            semantic.append(item)
    return normalizations, removals, semantic


def _representationally_equivalent(key: str, before: Any, after: Any) -> bool:
    if key == "filter_biased":
        token = str(before).strip().lower() if not isinstance(before, bool) else ("true" if before else "false")
        return token == after
    if isinstance(after, bool):
        return str(before).strip().lower() in ({"true", "1", "yes", "on"} if after else {"false", "0", "no", "off"})
    if isinstance(after, (int, float)) and not isinstance(after, bool):
        try:
            return float(before) == float(after)
        except (TypeError, ValueError):
            return False
    if isinstance(after, str):
        return str(before).strip().lower() == after
    return False


def _spec_filter_keys(spec: ModelInputSpec, *, include_internal: bool) -> Optional[frozenset]:
    """Return the sanitizer allowlist for a model spec and validation mode.

    Full-job validation uses the model's executable keys. Agent deltas keep only
    the public-agent surface intersected with that model, so user-owned fields
    such as additional_filters cannot leak into LLM/policy output.
    """
    if spec.allowed_keys is None:
        return None
    if include_internal:
        return frozenset(spec.allowed_keys)
    public = spec.public_agent_keys if spec.public_agent_keys is not None else PUBLIC_AGENT_CONFIG_KEYS
    return frozenset(spec.allowed_keys) & frozenset(public)


def _sanitize_config(
    config: Mapping[str, Any],
    *,
    spec: Optional[ModelInputSpec] = None,
    include_internal: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Deterministically check/repair a config against a downstream model spec.

    Defaults to the BoltzGen spec for backwards compatibility. Every rule is
    driven by ``spec`` so the same function validates any registered model
    (BoltzGen, ODesign, ...).
    """
    if spec is None:
        spec = get_model_input_spec("boltzgen")
    issues: List[Dict[str, Any]] = []

    # 1. Executable-key whitelist (only when the model declares a contract).
    if spec.allowed_keys is not None:
        allowed = _spec_filter_keys(spec, include_internal=include_internal)
        corrected = supported_config_changes(config, include_internal=include_internal, allowed_keys=allowed)
        for key in unsupported_config_keys(config, include_internal=include_internal, allowed_keys=allowed):
            issues.append({
                "parameter": key,
                "severity": "warning",
                "problem": "Unsupported executable config key was removed before submission.",
                "correction": "Move this idea to reasoning/metadata or add it to the explicit parameter contract first.",
                "resolved": True,
            })
        for key in invalid_config_value_keys(config, include_internal=include_internal, allowed_keys=allowed):
            issues.append({
                "parameter": key,
                "severity": "warning",
                "problem": "Executable config value had an invalid shape and was removed before submission.",
                "correction": "Emit values using the explicit JSON schema, e.g. binder_lengths as [60, 80] and config_overrides as [[section, key=value]].",
                "resolved": True,
            })
    else:
        corrected = dict(config or {})

    # 2. Per-parameter check/repair: drive a single rule unit over EVERY key that
    # is actually present, so validation is content-driven rather than tied to a
    # fixed list. New parameters are covered automatically once a rule class in
    # the spec references them.
    for key in list(corrected.keys()):
        check_one_param(key, corrected, issues, spec)

    # 3. Model-specific structural repairs (token lists, chain objects, ...).
    if spec.structural_normalizer is not None:
        spec.structural_normalizer(corrected, issues)
    if not include_internal:
        corrected.pop("run_filtering", None)

    return corrected, issues


def _advisory_llm_issues(raw_issues: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for raw in raw_issues:
        issue = dict(raw or {})
        original_severity = str(issue.get("severity") or "info").lower()
        if original_severity == "error":
            issue["original_severity"] = "error"
            issue["severity"] = "warning"
        issue["resolved"] = True
        issue["advisory"] = True
        issue.setdefault("resolution_note", "LLM advisory is not allowed to veto full-job pre-submit validation.")
        issues.append(issue)
    return issues


def _issue_resolved_by_sanitized_config(
    issue: Mapping[str, Any],
    sanitized: Mapping[str, Any],
    *,
    spec: Optional[ModelInputSpec] = None,
) -> bool:
    """Return True when an LLM-reported error is contradicted by sanitizer output.

    The deterministic sanitizer is the final source of truth for submittability.
    In addition to classic "missing key restored" conflicts, neutralize LLM
    complaints that a harness-owned/user-owned-but-executable field must not be
    emitted when the final sanitized config intentionally keeps that field.
    """
    if str(issue.get("severity") or "").lower() != "error" or issue.get("resolved"):
        return False
    parameter = issue.get("parameter")
    if parameter is None or parameter not in sanitized or sanitized.get(parameter) in (None, ""):
        return False
    text = " ".join(str(issue.get(key, "")) for key in ("problem", "correction")).lower()
    if ("missing" in text and ("restore" in text or "restored" in text)) or "restored" in text:
        return True
    allowed_keys = getattr(spec, "allowed_keys", None)
    if allowed_keys is not None and parameter in allowed_keys:
        conflict_terms = (
            "must not be emitted",
            "must not be submitted",
            "not an executable",
            "user-owned",
            "internal-only",
            "unsupported",
        )
        if any(term in text for term in conflict_terms):
            return True
    return False
