"""Declarative input-format specifications for downstream design models.

The :class:`ConfigValidationAgent` uses these specs to check and repair a
generated config against the *actual* input requirements of the downstream model
that will consume it (currently BoltzGen via Taiji, and ODesign via Hydra). Each
spec captures the model-specific traps that otherwise cause hard "invalid choice"
failures or silently malformed inputs:

* ``choice_flags``  - flags whose runner accepts only a small lowercase token set
  (a Python ``bool`` serialises to ``"True"`` and is rejected).
* ``single_choice`` - single-valued enum parameters.
* ``multi_choice``  - list-valued enum parameters (every token must be valid).
* ``int_keys`` / ``float_keys`` - numeric coercion targets.
* ``list_keys``     - keys that must be wrapped into a list.
* ``allowed_keys``  - executable-key whitelist; unknown keys are stripped. When
  ``None`` the whitelist step is skipped (the model has no executable contract
  registered yet).
* ``structural_normalizer`` - optional model-specific structural repair callable
  ``(corrected, issues) -> None`` for shapes that are not purely declarative.

Adding a new downstream model is a matter of registering a new spec here; the
validation agent and its deterministic fallback then cover it automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple


Issue = Dict[str, Any]
StructuralNormalizer = Callable[[Dict[str, Any], List[Issue]], None]


@dataclass(frozen=True)
class ModelInputSpec:
    """Declarative description of a downstream model's executable input format."""

    model: str
    runner_label: str
    choice_flags: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    single_choice: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    multi_choice: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    int_keys: Tuple[str, ...] = ()
    float_keys: Tuple[str, ...] = ()
    bool_keys: Tuple[str, ...] = ()
    list_keys: Tuple[str, ...] = ()
    allowed_keys: Optional[FrozenSet[str]] = None
    public_agent_keys: Optional[FrozenSet[str]] = None
    structural_normalizer: Optional[StructuralNormalizer] = None
    prompt_rules: Tuple[str, ...] = ()

    def flag_descriptor(self, key: str) -> str:
        return f"{self.runner_label} {key}"


def normalize_choice_flag(value: Any, choices: FrozenSet[str]) -> Optional[str]:
    """Coerce a bool/str choice-flag value to its canonical lowercase token.

    Returns the normalized token, or ``None`` when it cannot map to a valid choice.
    """
    if isinstance(value, bool):
        token = "true" if value else "false"
    else:
        token = str(value).strip().lower()
    return token if token in choices else None


_TRUE_TOKENS = frozenset({"1", "true", "yes", "y", "on", "enabled"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "n", "off", "disabled"})


def normalize_bool(value: Any) -> Optional[bool]:
    """Coerce a bool/int/str toggle to a real Python bool.

    Returns ``None`` when the value cannot be interpreted as a boolean. Used for
    harness-internal toggles (consumed as Python bools, not stringified CLI flags).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


def check_one_param(key: str, corrected: Dict[str, Any], issues: List[Issue], spec: "ModelInputSpec") -> None:
    """Check and repair a single config parameter against ``spec`` in place.

    This is the per-parameter unit the validator drives over *every* key actually
    present in the config, so checking/repair is content-driven rather than tied
    to a fixed list. A key may match several rule classes (e.g. ``steps`` is both
    a list and a multi-choice enum); the rules are applied in a safe order.
    """
    if corrected.get(key) is None:
        return

    # List coercion first so downstream list-aware rules see a list.
    if key in spec.list_keys and not isinstance(corrected[key], list):
        corrected[key] = [corrected[key]]
        issues.append({"parameter": key, "severity": "warning", "problem": "Expected a list value.", "correction": "Wrapped scalar value in a list.", "resolved": True})

    if key in spec.int_keys:
        try:
            corrected[key] = max(1, int(corrected[key]))
        except (TypeError, ValueError):
            corrected.pop(key, None)
            issues.append({"parameter": key, "severity": "error", "problem": "Expected a positive integer.", "correction": "Removed invalid value; upstream default must be used or LLM must provide an integer.", "resolved": True})
            return

    if key in spec.float_keys:
        try:
            value = float(corrected[key])
        except (TypeError, ValueError):
            corrected.pop(key, None)
            issues.append({"parameter": key, "severity": "error", "problem": "Expected a numeric value.", "correction": "Removed invalid value; upstream default must be used or LLM must provide a number.", "resolved": True})
            return
        if value != value or value in (float("inf"), float("-inf")):  # NaN / inf guard
            corrected.pop(key, None)
            issues.append({"parameter": key, "severity": "error", "problem": "Numeric value was NaN/inf.", "correction": "Removed non-finite value; upstream default must be used.", "resolved": True})
            return
        corrected[key] = value

    if key in spec.bool_keys:
        normalized_bool = normalize_bool(corrected[key])
        if normalized_bool is None:
            corrected.pop(key, None)
            issues.append({"parameter": key, "severity": "error", "problem": f"Expected a boolean toggle for {key}.", "correction": "Removed uninterpretable value; upstream default will be used.", "resolved": True})
            return
        if normalized_bool != corrected[key]:
            issues.append({"parameter": key, "severity": "warning", "problem": f"{key} should be a boolean; received {corrected[key]!r} ({type(corrected[key]).__name__}).", "correction": f"Normalized to {normalized_bool}.", "resolved": True})
        corrected[key] = normalized_bool

    if key in spec.choice_flags:
        choices = spec.choice_flags[key]
        normalized = normalize_choice_flag(corrected[key], choices)
        if normalized is None:
            issues.append({
                "parameter": key,
                "severity": "error",
                "problem": f"Value {corrected.get(key)!r} is not a valid choice for {spec.flag_descriptor(key)} (allowed: {sorted(choices)}).",
                "correction": "Removed invalid value; upstream default will be used.",
                "resolved": True,
            })
            corrected.pop(key, None)
            return
        if normalized != corrected[key]:
            issues.append({
                "parameter": key,
                "severity": "warning",
                "problem": f"{spec.flag_descriptor(key)} requires a lowercase string choice (one of {sorted(choices)}); received {corrected[key]!r} ({type(corrected[key]).__name__}).",
                "correction": f"Normalized to {normalized!r}.",
                "resolved": True,
            })
            corrected[key] = normalized

    if key in spec.single_choice:
        choices = spec.single_choice[key]
        token = str(corrected[key]).strip().lower()
        if token in choices:
            corrected[key] = token
        else:
            issues.append({
                "parameter": key,
                "severity": "error",
                "problem": f"{key} {corrected.get(key)!r} is not a valid choice for {spec.flag_descriptor(key)} (allowed: {sorted(choices)}).",
                "correction": "Removed invalid value; the upstream default will be used.",
                "resolved": True,
            })
            corrected.pop(key, None)
            return

    if key in spec.multi_choice:
        choices = spec.multi_choice[key]
        raw_values = corrected[key] if isinstance(corrected[key], list) else [corrected[key]]
        valid_values: List[str] = []
        dropped_values: List[str] = []
        for value in raw_values:
            token = str(value).strip().lower()
            if token in choices and token not in valid_values:
                valid_values.append(token)
            elif token not in choices:
                dropped_values.append(str(value))
        if dropped_values:
            issues.append({
                "parameter": key,
                "severity": "error",
                "problem": f"{key} contained invalid token(s) {dropped_values} for {spec.flag_descriptor(key)} (allowed: {sorted(choices)}).",
                "correction": "Dropped invalid token(s); kept the valid ordered subset.",
                "resolved": True,
            })
        if valid_values:
            corrected[key] = valid_values
        else:
            corrected.pop(key, None)
            if not dropped_values:
                issues.append({"parameter": key, "severity": "warning", "problem": f"{key} was empty.", "correction": "Removed; upstream default will be used.", "resolved": True})


def normalize_additional_filters(value: Any) -> List[str]:
    """Render BoltzGen additional_filters as CLI strings.

    BoltzGen expects ``--additional_filters feature>threshold feature<threshold``
    and parses those tokens into filtering config internally. Harness configs may
    also carry the already-parsed dict form; convert it back to CLI syntax here
    instead of sending it through ``--config filtering``.
    """
    if value in (None, ""):
        return []
    normalized, _ = normalize_boltzgen_additional_filters(value)
    return normalized


def normalize_boltzgen_additional_filters(value: Any) -> Tuple[List[str], List[Issue]]:
    """Normalize and validate BoltzGen additional_filters.

    ``additional_filters`` are hard filters over columns in BoltzGen's analysis
    metrics CSV. They are not the place for step-specific filtering task kwargs.
    In particular, ``designfolding_iptm`` is not a documented BoltzGen metrics
    column or filtering CLI override; the supported refolded-design stability
    signal from the bundled notebook/README is ``designfolding-filter_rmsd``.
    """
    issues: List[Issue] = []
    if value in (None, ""):
        return [], issues
    raw_items = value if isinstance(value, (list, tuple)) else [value]
    normalized: List[str] = []
    for item in raw_items:
        if item in (None, ""):
            continue
        if isinstance(item, Mapping):
            feature = str(item.get("feature") or "").strip()
            if not feature or item.get("threshold") is None:
                continue
            if _is_invalid_boltzgen_additional_filter_feature(feature):
                issues.append(_invalid_additional_filter_issue(feature))
                continue
            op = "<" if bool(item.get("lower_is_better")) else ">"
            normalized.append(f"{feature}{op}{item.get('threshold')}")
            continue
        token = str(item).strip()
        if not token:
            continue
        feature = _additional_filter_feature(token)
        if _is_invalid_boltzgen_additional_filter_feature(feature):
            issues.append(_invalid_additional_filter_issue(feature))
            continue
        normalized.append(token)
    return normalized, issues


def _additional_filter_feature(token: str) -> str:
    for op in (">", "<"):
        if op in token:
            return token.split(op, 1)[0].strip()
    return str(token or "").strip()


def _is_invalid_boltzgen_additional_filter_feature(feature: str) -> bool:
    normalized = str(feature or "").strip().lower().replace("-", "_")
    return normalized in {"designfolding_iptm", "design_folding_iptm", "filter_designfolding_iptm"}


def _invalid_additional_filter_issue(feature: str) -> Issue:
    return {
        "parameter": "additional_filters",
        "severity": "warning",
        "problem": (
            f"Dropped unsupported additional_filter feature {feature!r}. "
            "BoltzGen additional_filters operate on metrics CSV columns; "
            "designfolding_iptm is not a documented filtering column."
        ),
        "correction": (
            "Use a supported metrics column such as designfolding-filter_rmsd for "
            "refolded binder stability, or perform iPTM-based post-processing in the harness."
        ),
        "resolved": True,
    }


# ---------------------------------------------------------------------------
# Runtime-error correlation (retry-time intervention).
# ---------------------------------------------------------------------------
@dataclass
class ErrorFinding:
    """A config-related signal extracted from a downstream runtime error."""

    kind: str                       # invalid_choice | invalid_value | unrecognized_argument | config_override_format
    parameter: Optional[str]        # mapped config key (None when not mappable)
    bad_value: Optional[str] = None
    allowed: Optional[List[str]] = None
    value_type: Optional[str] = None  # for invalid_value (e.g. "int")
    raw_line: str = ""


_RE_INVALID_CHOICE = re.compile(r"argument\s+(--[\w\-]+):\s*invalid choice:\s*'([^']*)'(?:\s*\(choose from ([^)]*)\))?", re.IGNORECASE)
_RE_INVALID_VALUE = re.compile(r"argument\s+(--[\w\-]+):\s*invalid (\w+) value:\s*'([^']*)'", re.IGNORECASE)
_RE_UNRECOGNIZED = re.compile(r"unrecognized arguments:\s*(.+)", re.IGNORECASE)
# BoltzGen filtering/analysis tasks are instantiated via Hydra; an override key
# the task class does not accept surfaces as a TypeError such as
# ``Filter.__init__() got an unexpected keyword argument 'filter_rmsd_threshold'``.
_RE_UNEXPECTED_KWARG = re.compile(r"unexpected keyword argument\s+'([^']+)'", re.IGNORECASE)


def cli_flag_to_config_key(flag: str) -> str:
    """Map a CLI/Hydra flag token to its config key.

    ``--filter_biased`` -> ``filter_biased``; ``exp.use_msa`` -> ``use_msa``.
    """
    token = str(flag).strip().lstrip("-")
    if "=" in token:
        token = token.split("=", 1)[0]
    if "." in token:
        token = token.rsplit(".", 1)[1]
    return token


def analyze_runtime_error(error_text: str) -> List[ErrorFinding]:
    """Parse a downstream runner error blob into structured config-related findings.

    Model-agnostic: it recognises the stable argparse / BoltzGen error shapes that
    indicate a malformed config parameter. Returns an empty list when nothing
    config-related is found (i.e. the failure is likely infrastructure/runtime).
    """
    text = str(error_text or "")
    findings: List[ErrorFinding] = []
    seen: set = set()

    for match in _RE_INVALID_CHOICE.finditer(text):
        flag, bad, choose = match.group(1), match.group(2), match.group(3)
        key = cli_flag_to_config_key(flag)
        allowed = None
        if choose:
            allowed = [tok.strip().strip("'\"") for tok in choose.split(",") if tok.strip()]
        sig = ("invalid_choice", key, bad)
        if sig in seen:
            continue
        seen.add(sig)
        findings.append(ErrorFinding(kind="invalid_choice", parameter=key, bad_value=bad, allowed=allowed, raw_line=match.group(0).strip()))

    for match in _RE_INVALID_VALUE.finditer(text):
        flag, vtype, bad = match.group(1), match.group(2), match.group(3)
        key = cli_flag_to_config_key(flag)
        sig = ("invalid_value", key, bad)
        if sig in seen:
            continue
        seen.add(sig)
        findings.append(ErrorFinding(kind="invalid_value", parameter=key, bad_value=bad, value_type=vtype.lower(), raw_line=match.group(0).strip()))

    for match in _RE_UNRECOGNIZED.finditer(text):
        for tok in match.group(1).split():
            if tok.startswith("--"):
                key = cli_flag_to_config_key(tok)
                sig = ("unrecognized_argument", key, None)
                if sig in seen:
                    continue
                seen.add(sig)
                findings.append(ErrorFinding(kind="unrecognized_argument", parameter=key, raw_line=match.group(0).strip()))

    for match in _RE_UNEXPECTED_KWARG.finditer(text):
        bad_key = match.group(1).strip()
        sig = ("unexpected_kwarg", "config_overrides", bad_key)
        if not bad_key or sig in seen:
            continue
        seen.add(sig)
        findings.append(ErrorFinding(kind="config_override_unexpected_kwarg", parameter="config_overrides", bad_value=bad_key, raw_line=match.group(0).strip()))

    lowered = text.lower()
    if "invalid config" in lowered and "filtering," in lowered:
        findings.append(ErrorFinding(kind="config_override_format", parameter="config_overrides", raw_line="Invalid config: comma-joined --config token"))

    return findings


def apply_error_findings(corrected: Dict[str, Any], findings: List[ErrorFinding], spec: "ModelInputSpec") -> List[Issue]:
    """Apply targeted, deterministic repairs for runtime-error findings in place.

    Each repair points at the exact offending parameter and records the original
    runner error line so the fix is auditable.
    """
    issues: List[Issue] = []
    for finding in findings:
        key = finding.parameter
        if finding.kind == "config_override_format":
            corrected["config_overrides"] = [["filtering", "filter_bindingsite=true"]]
            issues.append({"parameter": "config_overrides", "severity": "warning", "problem": f"Runtime error indicates --config was passed as one comma-separated token. ({finding.raw_line})", "correction": "Rewrote to token-list form [[\"filtering\", \"filter_bindingsite=true\"]].", "resolved": True})
            continue
        if finding.kind == "config_override_unexpected_kwarg":
            bad_key = str(finding.bad_value or "").strip()
            existing = corrected.get("config_overrides") or []
            pruned: List[List[str]] = []
            removed = False
            for group in existing:
                tokens = list(group) if isinstance(group, (list, tuple)) else [group]
                kept = [tokens[0]] if tokens else []
                for token in tokens[1:]:
                    if str(token).split("=", 1)[0].strip() == bad_key:
                        removed = True
                        continue
                    kept.append(token)
                if len(kept) >= 2 and any("=" in str(t) for t in kept[1:]):
                    pruned.append([str(t) for t in kept])
            corrected["config_overrides"] = pruned
            issues.append({
                "parameter": "config_overrides",
                "severity": "warning" if removed else "info",
                "problem": f"BoltzGen rejected --config key {bad_key!r} as an unexpected keyword argument. ({finding.raw_line})",
                "correction": f"Removed override token(s) for {bad_key!r}; remaining overrides: {pruned!r}.",
                "resolved": True,
            })
            continue
        if not key:
            continue

        if finding.kind == "invalid_choice":
            allowed = frozenset(finding.allowed) if finding.allowed else spec.choice_flags.get(key) or spec.single_choice.get(key)
            present = key in corrected
            repaired = None
            if allowed is not None:
                repaired = normalize_choice_flag(corrected.get(key, finding.bad_value), allowed) if (key in spec.choice_flags or isinstance(corrected.get(key), bool)) else str(corrected.get(key, finding.bad_value)).strip().lower()
                if repaired not in allowed:
                    repaired = None
            if repaired is not None:
                corrected[key] = repaired
                issues.append({"parameter": key, "severity": "warning", "problem": f"Runtime error: {finding.raw_line}", "correction": f"Normalized {key} to valid choice {repaired!r}.", "resolved": True})
            elif present:
                corrected.pop(key, None)
                issues.append({"parameter": key, "severity": "warning", "problem": f"Runtime error: {finding.raw_line}", "correction": f"Removed invalid {key}; upstream default will be used.", "resolved": True})
        elif finding.kind == "invalid_value":
            if finding.value_type == "int":
                try:
                    corrected[key] = max(1, int(float(corrected.get(key, finding.bad_value))))
                    issues.append({"parameter": key, "severity": "warning", "problem": f"Runtime error: {finding.raw_line}", "correction": f"Coerced {key} to integer {corrected[key]}.", "resolved": True})
                    continue
                except (TypeError, ValueError):
                    pass
            corrected.pop(key, None)
            issues.append({"parameter": key, "severity": "warning", "problem": f"Runtime error: {finding.raw_line}", "correction": f"Removed invalid {key}; upstream default will be used.", "resolved": True})
        elif finding.kind == "unrecognized_argument":
            if key in corrected:
                corrected.pop(key, None)
                issues.append({"parameter": key, "severity": "warning", "problem": f"Runtime error: {finding.raw_line}", "correction": f"Removed unrecognized argument {key} not accepted by the runner.", "resolved": True})
    return issues


# ---------------------------------------------------------------------------
# BoltzGen
# ---------------------------------------------------------------------------
BOLTZGEN_PROTOCOLS: FrozenSet[str] = frozenset({
    "protein-anything",
    "peptide-anything",
    "protein-small_molecule",
    "nanobody-anything",
})

BOLTZGEN_STEPS: FrozenSet[str] = frozenset({
    "design",
    "inverse_folding",
    "design_folding",
    "folding",
    "affinity",
    "analysis",
    "filtering",
})

# ``--config`` override keys that BoltzGen's task classes do not accept. Passing
# them makes Hydra raise an ``InstantiationException`` (unexpected keyword
# argument) that fails the whole job. Keep this conservative: only list keys we
# have observed crash a run, so we never over-strip a legitimate override. The
# RMSD gate is controlled by ``refolding_rmsd_threshold`` / metric
# ``additional_filters`` instead, not by a ``filtering`` --config key.
BOLTZGEN_INVALID_CONFIG_OVERRIDE_KEYS: FrozenSet[str] = frozenset({
    "filter_rmsd_threshold",
    "iptm_threshold",
    "filter_designfolding_iptm",
})

# Orchestrator-consumed region/template strategy enums (not BoltzGen CLI args).
# ``disabled``/``off``/``none`` are accepted disable aliases for epitope_crop_mode.
EPITOPE_CROP_MODES: FrozenSet[str] = frozenset({
    "disabled", "off", "none", "auto", "hotspot_focus", "engaged_focus", "union",
})
FRAGMENT_TEMPLATE_GATES: FrozenSet[str] = frozenset({"interchain_pae", "iptm"})


def _boltzgen_structural_normalizer(corrected: Dict[str, Any], issues: List[Issue]) -> None:
    deprecated = ("hotspot_weight", "prioritize_hotspots", "clash_filter", "module_guided_repair", "module_guided_exploitation", "exploit_fragment_modules")
    audit = dict(corrected.get("deprecated_strategy_audit") or {})
    for key in deprecated:
        if key not in corrected:
            continue
        audit[key] = {"value": corrected.pop(key), "schema_version": "1.0", "status": "deprecated_audit_only"}
        issues.append({"parameter": key, "severity": "warning", "problem": "Deprecated strategy field has no downstream executable consumer.", "correction": "Moved to deprecated_strategy_audit metadata.", "resolved": True})
    if audit:
        corrected["deprecated_strategy_audit"] = audit
    if corrected.get("run_filtering") is not True:
        original = corrected.get("run_filtering")
        corrected["run_filtering"] = True
        issues.append({
            "parameter": "run_filtering",
            "severity": "warning",
            "problem": f"run_filtering is fixed to true for closed-loop BoltzGen runs; received {original!r}.",
            "correction": "Forced run_filtering to true so final_ranked_designs and downstream metrics are produced.",
            "resolved": True,
        })

    if "config_overrides" in corrected:
        normalized, override_issues = _normalize_config_overrides(corrected.get("config_overrides"))
        corrected["config_overrides"] = normalized
        issues.extend(override_issues)

    if "additional_filters" in corrected:
        corrected["additional_filters"], filter_issues = normalize_boltzgen_additional_filters(corrected.get("additional_filters"))
        issues.extend(filter_issues)
        if not corrected["additional_filters"]:
            corrected.pop("additional_filters", None)

    if "binder_lengths" in corrected:
        raw_lengths = corrected.get("binder_lengths") if isinstance(corrected.get("binder_lengths"), list) else [corrected.get("binder_lengths")]
        lengths: List[int] = []
        dropped = False
        for item in raw_lengths:
            try:
                value = int(item)
            except (TypeError, ValueError):
                dropped = True
                continue
            if value <= 0:
                dropped = True
                continue
            lengths.append(value)
        if lengths:
            corrected["binder_lengths"] = sorted(set(lengths))
        else:
            corrected.pop("binder_lengths", None)
            dropped = True
        if dropped:
            issues.append({
                "parameter": "binder_lengths",
                "severity": "warning",
                "problem": "binder_lengths must be positive integer lengths.",
                "correction": "Dropped invalid length value(s) and kept valid unique lengths.",
                "resolved": True,
            })

    for key in ["target_include", "target_binding_types"]:
        if key in corrected and corrected[key] is not None and not isinstance(corrected[key], list):
            corrected[key] = [corrected[key]]
            issues.append({"parameter": key, "severity": "warning", "problem": "Expected a list of chain constraint objects.", "correction": "Wrapped object in a list.", "resolved": True})

    # exploit_fragment_modules are opaque match keys. A legacy bug emitted
    # malformed ids like "frag_Union[<hash>" (unbalanced bracket). Repair stray
    # bracket/whitespace artifacts and drop empties so module-guided exploitation
    # receives clean tokens instead of meaningless garbage.
    if corrected.get("exploit_fragment_modules"):
        repaired_modules: List[str] = []
        repaired_any = False
        for token in corrected["exploit_fragment_modules"]:
            original = str(token)
            clean = original.replace("Union[", "").replace("[", "").replace("]", "").strip()
            if clean != original:
                repaired_any = True
            if clean:
                repaired_modules.append(clean)
        if repaired_any:
            issues.append({
                "parameter": "exploit_fragment_modules",
                "severity": "warning",
                "problem": "exploit_fragment_modules contained malformed token(s) with stray bracket/Union artifacts.",
                "correction": "Stripped bracket/Union artifacts from fragment module ids.",
                "resolved": True,
            })
        corrected["exploit_fragment_modules"] = repaired_modules


def _coalesce_flat_config_overrides(items: List[Any]) -> Tuple[List[Any], Optional[Issue]]:
    """Collapse a representational flat group into one nested override."""
    if not items or any(not isinstance(item, str) for item in items):
        return items, None
    head = str(items[0]).strip()
    rest = [str(token).strip() for token in items[1:] if str(token).strip()]
    if not head or not rest or "=" in head:
        return items, None
    coalesced = [[head, *rest]]
    issue = {
        "parameter": "config_overrides",
        "severity": "warning",
        "problem": "config_overrides was a flat list; BoltzGen expects a list of token lists, one per --config group.",
        "correction": f"Coalesced {items!r} into a single override group {coalesced!r}.",
        "resolved": True,
    }
    return coalesced, issue


def _nonempty_override_input(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool(raw.strip())
    if isinstance(raw, (list, tuple, set, Mapping)):
        return bool(raw)
    return True


def _normalize_config_overrides(raw: Any) -> Tuple[List[List[str]], List[Issue]]:
    """Validate BoltzGen override groups while preserving every valid sibling.

    This is the sole model-specific layer: step names are checked against
    ``BOLTZGEN_STEPS``, each setting must be a nonempty ``key=value`` token, and
    only explicitly forbidden BoltzGen keys are removed.
    """
    issues: List[Issue] = []
    had_input = _nonempty_override_input(raw)
    if not had_input:
        return [], issues
    items = raw if isinstance(raw, list) else [raw]
    coalesced, coalesce_issue = _coalesce_flat_config_overrides(list(items))
    if coalesce_issue is not None:
        items = coalesced
        issues.append(coalesce_issue)
    normalized: List[List[str]] = []
    for item in items:
        if isinstance(item, (list, tuple)):
            tokens = [str(token).strip() for token in item if str(token).strip()]
        elif isinstance(item, str):
            tokens = [token.strip() for token in re.split(r",|\s+", item) if token.strip()]
            issues.append({
                "parameter": "config_overrides",
                "severity": "warning",
                "problem": "Config override was provided as a single string; BoltzGen expects separate CLI tokens.",
                "correction": f"Converted {item!r} to token list {tokens!r}.",
                "resolved": True,
            })
        else:
            issues.append({
                "parameter": "config_overrides",
                "severity": "error",
                "problem": f"Config override group {item!r} was not a token list or string.",
                "correction": "Dropped malformed override group.",
                "resolved": True,
            })
            continue
        if not tokens:
            issues.append({
                "parameter": "config_overrides",
                "severity": "error",
                "problem": "Config override group was empty.",
                "correction": "Dropped empty override group.",
                "resolved": True,
            })
            continue
        step = tokens[0]
        if step not in BOLTZGEN_STEPS:
            issues.append({
                "parameter": "config_overrides",
                "severity": "error",
                "problem": f"BoltzGen --config step {step!r} is invalid (allowed: {sorted(BOLTZGEN_STEPS)}).",
                "correction": "Dropped override group with invalid step.",
                "resolved": True,
            })
            continue
        kept_settings: List[str] = []
        for token in tokens[1:]:
            if "=" not in token:
                issues.append({
                    "parameter": "config_overrides",
                    "severity": "error",
                    "problem": f"BoltzGen --config setting token {token!r} is missing '='.",
                    "correction": "Dropped malformed setting token and preserved valid siblings.",
                    "resolved": True,
                })
                continue
            key, setting_value = token.split("=", 1)
            key = key.strip()
            setting_value = setting_value.strip()
            if not key or not setting_value:
                issues.append({
                    "parameter": "config_overrides",
                    "severity": "error",
                    "problem": f"BoltzGen --config setting token {token!r} requires a nonempty key and value.",
                    "correction": "Dropped malformed setting token and preserved valid siblings.",
                    "resolved": True,
                })
                continue
            if key in BOLTZGEN_INVALID_CONFIG_OVERRIDE_KEYS:
                issues.append({
                    "parameter": "config_overrides",
                    "severity": "error",
                    "problem": f"BoltzGen does not accept --config key {key!r}; it raises an unexpected-keyword-argument error.",
                    "correction": f"Dropped unsupported override token {token!r} and preserved valid siblings.",
                    "resolved": True,
                })
                continue
            kept_settings.append(token)
        if kept_settings:
            normalized.append([step, *kept_settings])
        else:
            issues.append({
                "parameter": "config_overrides",
                "severity": "error",
                "problem": f"BoltzGen --config group for step {step!r} contained no valid key=value settings.",
                "correction": "Dropped override group after removing malformed or forbidden settings.",
                "resolved": True,
            })
    if had_input and not normalized:
        issues.append({
            "parameter": "config_overrides",
            "severity": "warning",
            "problem": "All nonempty config_overrides were removed during BoltzGen validation.",
            "correction": "Proceeding without --config overrides; provide a valid step and nonempty key=value settings to restore them.",
            "resolved": True,
        })
    return normalized, issues


def _normalize_string_list(raw: Any, *, key: str) -> Tuple[List[str], List[Issue]]:
    issues: List[Issue] = []
    if raw in (None, ""):
        return [], issues
    values = raw if isinstance(raw, list) else [raw]
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not isinstance(raw, list):
        issues.append({"parameter": key, "severity": "warning", "problem": "Expected a list of strings.", "correction": "Wrapped scalar value in a list.", "resolved": True})
    return normalized, issues


# ---------------------------------------------------------------------------
# ODesign (local Hydra inference). Hydra serialises booleans lowercase, so the
# same bool->"True" trap applies to its boolean overrides. Enum-like fields use
# a small fixed vocabulary in the reference inference demo.
# ---------------------------------------------------------------------------
ODESIGN_MODALITIES: FrozenSet[str] = frozenset({"protein", "peptide", "rna", "dna"})
ODESIGN_CENTER_METHODS: FrozenSet[str] = frozenset({"hotspot_center", "target_center", "binder_center"})
RFD3_STEPS: FrozenSet[str] = frozenset({"design", "inverse_folding", "folding"})
RFD3_ORI_STRATEGIES: FrozenSet[str] = frozenset({"hotspots", "com"})
RFD3_MPNN_MODELS: FrozenSet[str] = frozenset({"protein_mpnn"})
RFD3_ID_SCHEMES: FrozenSet[str] = frozenset({"auto", "label", "auth"})
RFD3_TARGET_ID_SCHEMES: FrozenSet[str] = frozenset({"native", "label", "auth"})


def _odesign_structural_normalizer(corrected: Dict[str, Any], issues: List[Issue]) -> None:
    # seeds must be an integer list for the Hydra "[s1,s2]" override.
    if "seeds" in corrected and corrected["seeds"] is not None:
        raw_seeds = corrected["seeds"] if isinstance(corrected["seeds"], list) else [corrected["seeds"]]
        clean_seeds: List[int] = []
        dropped = False
        for seed in raw_seeds:
            try:
                clean_seeds.append(int(seed))
            except (TypeError, ValueError):
                dropped = True
        if dropped:
            issues.append({"parameter": "seeds", "severity": "warning", "problem": "ODesign seeds must be integers.", "correction": "Dropped non-integer seed value(s).", "resolved": True})
        corrected["seeds"] = clean_seeds


def _build_boltzgen_spec() -> ModelInputSpec:
    # Imported lazily to avoid a circular import at module load.
    from binderloop.agents.config_parameter_contract import ALL_EXECUTABLE_CONFIG_KEYS, PUBLIC_AGENT_CONFIG_KEYS

    return ModelInputSpec(
        model="boltzgen",
        runner_label="BoltzGen CLI argument --",
        choice_flags={
            "filter_biased": frozenset({"true", "false"}),
            "use_kernels": frozenset({"auto", "true", "false"}),
        },
        single_choice={
            "protocol": BOLTZGEN_PROTOCOLS,
            "epitope_crop_mode": EPITOPE_CROP_MODES,
            "fragment_template_gate": FRAGMENT_TEMPLATE_GATES,
        },
        multi_choice={"steps": BOLTZGEN_STEPS},
        float_keys=("step_scale", "noise_scale", "alpha", "exploration_ratio", "fragment_interchain_pae_max", "template_conditioned_fraction", "binder_template_proximity"),
        bool_keys=("auto_binder_length", "fragment_templates_enabled"),
        int_keys=("num_designs", "budget", "diffusion_batch_size", "fragment_template_top_k"),
        list_keys=("binder_lengths",),
        allowed_keys=ALL_EXECUTABLE_CONFIG_KEYS,
        public_agent_keys=PUBLIC_AGENT_CONFIG_KEYS,
        structural_normalizer=_boltzgen_structural_normalizer,
        prompt_rules=(
            "config_overrides must be a list of token lists. Example: [[\"filtering\", \"filter_bindingsite=true\"]]. Do not emit [\"filtering, filter_bindingsite=true\"].",
            "Do not rewrite user-owned task/search/resource fields such as budget, num_designs, binder_length_range, inverse_fold_num_sequences, refolding_rmsd_threshold, fragment template payloads, target_include, target_binding_types, hotspots, or resource.",
            "Never emit additional_filters from LLM output. Static user additional_filters are preserved as BoltzGen CLI filter tokens (feature>threshold or feature<threshold).",
            "Do not put designfolding_iptm in additional_filters or config_overrides. BoltzGen documents designfolding-filter_rmsd for isolated-design refolding consistency; any designfolding iPTM gate must be harness post-processing unless explicitly supported by BoltzGen.",
            "Deprecated hotspot_weight/prioritize_hotspots/clash_filter/module fields are audit-only and must not be emitted as executable corrections. Specific residue additions belong in auxiliary_hotspots.",
            "binder_lengths is a harness-owned discrete length set. Keep/propose it only as a list of positive integers; the orchestrator clamps it to the user's binder_length_range and writes one design spec per length.",
            "num_designs and budget are required explicit positive integers for submission readiness; never infer defaults.",
            "diffusion_batch_size must be a positive integer when present.",
            "step_scale, noise_scale, alpha, and template_conditioned_fraction must be numeric when present.",
            "filter_biased must be the lowercase string \"true\" or \"false\" (NEVER a JSON/Python boolean and never capitalized \"True\"/\"False\"); --filter_biased only accepts {true, false} and a boolean serializes to \"True\" and is rejected with \"invalid choice\".",
            "use_kernels must be the lowercase string \"auto\", \"true\", or \"false\" (NEVER a JSON/Python boolean and never capitalized \"True\"/\"False\"); --use_kernels only accepts {auto, true, false}. YAML false must become \"false\", not \"False\".",
            "Do NOT \"fix\" a correct lowercase string value like \"true\" by converting it to a boolean; leave choice flags as lowercase strings.",
            "Do not emit secondary_structure for BoltzGen. The raw BoltzGen schema supports only exact per-residue secondary-structure syntax, and this harness does not generate that field.",
            "Unknown or unsupported executable keys must be removed from corrected_config rather than passed through.",
            "When a Taiji/BoltzGen error is provided, use it to propose the smallest safe correction that addresses the error.",
        ),
    )


def _build_odesign_spec() -> ModelInputSpec:
    return ModelInputSpec(
        model="odesign",
        runner_label="ODesign Hydra override ",
        choice_flags={
            "use_msa": frozenset({"true", "false"}),
            "invfold_use_beam": frozenset({"true", "false"}),
            "partial_diffusion_enable": frozenset({"true", "false"}),
            "motif_scaffolding": frozenset({"true", "false"}),
        },
        single_choice={
            "design_modality": ODESIGN_MODALITIES,
            "center_method": ODESIGN_CENTER_METHODS,
        },
        multi_choice={},
        int_keys=("N_sample", "num_samples", "N_step", "num_workers", "invfold_topk"),
        float_keys=("invfold_temp", "partial_diffusion_snr"),
        list_keys=("seeds",),
        allowed_keys=None,  # No executable-key contract registered for ODesign yet.
        structural_normalizer=_odesign_structural_normalizer,
        prompt_rules=(
            "ODesign is launched through Hydra overrides; boolean flags are serialized lowercase.",
            "use_msa, invfold_use_beam, partial_diffusion_enable, motif_scaffolding must be the lowercase string \"true\" or \"false\" (never a Python/JSON boolean or capitalized \"True\").",
            "design_modality must be one of: protein, peptide, rna, dna.",
            "center_method must be one of: hotspot_center, target_center, binder_center.",
            "N_sample, num_samples, N_step, num_workers, invfold_topk must be positive integers when present.",
            "invfold_temp and partial_diffusion_snr must be numeric when present.",
            "seeds must be a list of integers.",
        ),
    )


def _rfd3_structural_normalizer(corrected: Dict[str, Any], issues: List[Issue]) -> None:
    designed = corrected.get("designed_chains")
    if isinstance(designed, list):
        corrected["designed_chains"] = ",".join(str(item).strip() for item in designed if str(item).strip())
    hotspots = corrected.get("select_hotspots")
    if isinstance(hotspots, Mapping):
        corrected["select_hotspots"] = {str(key): str(value) for key, value in hotspots.items()}
    steps = corrected.get("steps")
    if isinstance(steps, list):
        kept = [str(item).strip().lower() for item in steps if str(item).strip().lower() in RFD3_STEPS]
        dropped = [str(item) for item in steps if str(item).strip().lower() not in RFD3_STEPS]
        if dropped:
            issues.append({"parameter": "steps", "severity": "error", "problem": f"Unsupported RFD3 steps {dropped}.", "correction": "Dropped invalid steps; kept design, inverse_folding, folding.", "resolved": True})
        if kept:
            corrected["steps"] = kept
        else:
            corrected.pop("steps", None)


def _build_rfd3_spec() -> ModelInputSpec:
    from binderloop.models.search_profile import get_model_search_profile
    profile = get_model_search_profile("rfd3")
    return ModelInputSpec(
        model="rfd3",
        runner_label="Foundry RFD3/MPNN/RF3 ",
        single_choice={
            "infer_ori_strategy": RFD3_ORI_STRATEGIES,
            "model_type": RFD3_MPNN_MODELS,
            "residue_id_scheme": RFD3_ID_SCHEMES,
            "rfd3_source_id_scheme": RFD3_ID_SCHEMES,
            "rfd3_residue_scheme": RFD3_TARGET_ID_SCHEMES,
        },
        multi_choice={"steps": RFD3_STEPS},
        int_keys=("num_designs", "n_batches", "diffusion_batch_size", "inverse_fold_num_sequences", "num_timesteps", "batch_size", "number_of_batches", "n_recycles", "num_steps", "budget", "dialect"),
        float_keys=("step_scale", "gamma_0", "temperature", "refolding_rmsd_threshold", "noise_scale", "early_stopping_plddt_threshold"),
        bool_keys=("is_non_loopy", "redesign_motif_sidechains", "is_legacy_weights", "skip_existing", "prevalidate_inputs", "dump_trajectories", "auto_binder_length", "low_memory_mode", "rfd3_adapt_structure", "rfd3_convert_residue_ids"),
        list_keys=("steps",),
        allowed_keys=profile.executable_keys(include_internal=True),
        public_agent_keys=frozenset(profile.adjustable_parameters) | frozenset({"binder_lengths"}),
        structural_normalizer=_rfd3_structural_normalizer,
        prompt_rules=(
            "RFD3 binder design is three Foundry CLIs: rfd3 design, mpnn protein_mpnn, rf3 fold.",
            "steps may only contain design, inverse_folding, and folding.",
            "infer_ori_strategy must be hotspots or com; protein binder design should use hotspots.",
            "model_type must be protein_mpnn. Do not emit ligand_mpnn in this harness path.",
            "step_scale default for PPI is 3.0 and gamma_0 default is 0.2. Do not apply BoltzGen 0.6-1.0 step_scale bounds.",
            "is_non_loopy should be true for structured binders.",
            "mmCIF residue lookups use label_seq_id; PDB lookups use auth/PDB numbering. residue_id_scheme auto-detects label vs auth and remaps contig plus hotspots.",
            "num_designs, n_batches, diffusion_batch_size, inverse_fold_num_sequences must be positive integers when present.",
        ),
    )


_SPEC_BUILDERS: Dict[str, Callable[[], ModelInputSpec]] = {
    "boltzgen": _build_boltzgen_spec,
    "odesign": _build_odesign_spec,
    "rfd3": _build_rfd3_spec,
}


def get_model_input_spec(target_model: str) -> ModelInputSpec:
    """Return the registered input spec for ``target_model``.

    Unknown models are configuration errors; silently treating a misspelled or
    unregistered model as BoltzGen can validate and submit the wrong job shape.
    """
    key = str(target_model or "").strip().lower()
    builder = _SPEC_BUILDERS.get(key)
    if builder is None:
        raise ValueError(f"Unsupported target model {target_model!r}; supported models={sorted(_SPEC_BUILDERS)}")
    return builder()


def supported_models() -> List[str]:
    return sorted(_SPEC_BUILDERS)
