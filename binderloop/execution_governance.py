"""Deterministic execution governance for concrete BoltzGen runs."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PARAM_BOUNDS = {
    "alpha": {"min": 0.001, "max": 0.05, "default": 0.001, "max_step_ratio": 3.0, "log_scale": True},
    "exploration_ratio": {"min": 0.20, "max": 0.60, "default": 0.35, "max_step_abs": 0.15, "log_scale": False},
    "noise_scale": {"min": 0.6, "max": 0.9, "default": 0.7, "max_step_abs": 0.15, "log_scale": False},
    "step_scale": {"min": 0.6, "max": 1.0, "default": 0.8, "max_step_abs": 0.2, "log_scale": False},
    "template_conditioned_fraction": {"min": 0.0, "max": 0.8, "default": 0.5, "max_step_abs": 0.25, "log_scale": False},
}

def parameter_contract_entry(key: str):
    from binderloop.agents.config_parameter_contract import parameter_contract_entry as lookup
    return lookup(key)

def clamp_config_with_inertia(*args, **kwargs):
    from binderloop.agents.config_parameter_contract import clamp_config_with_inertia as clamp
    return clamp(*args, **kwargs)

MULTI_GPU_BUDGET_FLOOR = 99999


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def stable_digest(value: Any) -> str:
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: int
    job_id: str
    resolved_params: Dict[str, Any]
    lineage: Dict[str, List[Dict[str, Any]]]
    applicability: Dict[str, Dict[str, Any]]
    candidate_upper_bound: int
    logical_num_designs: int
    artifact_digests: Dict[str, str] = field(default_factory=dict)
    parity: Dict[str, Any] = field(default_factory=dict)
    consumer_receipts: List[Dict[str, Any]] = field(default_factory=list)
    final_parameter_state: Dict[str, Any] = field(default_factory=dict)
    parameter_catalog_digest: str = ""
    plan_digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _positive_int(name: str, value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"execution resolver requires integer {name}")
    if result < 1:
        raise ValueError(f"execution resolver requires {name} >= 1")
    return result



@dataclass(frozen=True)
class RoundBudgetResolution:
    schema_version: int
    cap: int
    requested_conditioned_fraction: float
    effective_conditioned_fraction: float
    bucket_allocations: Dict[str, int]
    allocations: List[Dict[str, Any]]
    rejections: List[Dict[str, Any]]
    rematerialization: List[Dict[str, Any]]
    digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _largest_remainder(total: int, weights: Sequence[float], tie_keys: Sequence[str]) -> List[int]:
    if total <= 0 or not weights:
        return [0 for _ in weights]
    normalized = [max(0.0, float(value)) for value in weights]
    if sum(normalized) <= 0:
        normalized = [1.0 for _ in normalized]
    denominator = sum(normalized)
    quotas = [total * value / denominator for value in normalized]
    allocated = [int(value) for value in quotas]
    remaining = total - sum(allocated)
    order = sorted(range(len(quotas)), key=lambda i: (-(quotas[i] - allocated[i]), str(tie_keys[i])))
    for index in order[:remaining]:
        allocated[index] += 1
    return allocated


def resolve_round_budget(
    cap: int,
    candidates: Sequence[Mapping[str, Any]],
    *,
    requested_conditioned_fraction: float = 0.0,
) -> RoundBudgetResolution:
    """Resolve one global round budget without re-normalizing template shares.

    Candidate mappings require an ``id`` and may provide ``bucket``
    (template_conditioned/template_free/other), explanatory ``weight``, and
    ``valid``/``rejection_reason``. Invalid conditioned candidates are rejected;
    their intended capacity is rematerialized through the remaining valid
    buckets rather than submitted as an unconditioned template job.
    """
    cap = _positive_int("round cap", cap)
    try:
        requested = float(requested_conditioned_fraction)
    except (TypeError, ValueError):
        requested = 0.0
    requested = min(1.0, max(0.0, requested))
    valid: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        item = dict(raw)
        item_id = str(item.get("id") or f"candidate_{index}")
        bucket = str(item.get("bucket") or "other")
        if bucket not in {"template_conditioned", "template_free", "other"}:
            bucket = "other"
        if not bool(item.get("valid", True)):
            rejections.append({"id": item_id, "bucket": bucket, "reason": str(item.get("rejection_reason") or "invalid_candidate")})
            continue
        try:
            weight = max(0.0, float(item.get("weight", 1.0)))
        except (TypeError, ValueError):
            weight = 0.0
        valid.append({"id": item_id, "bucket": bucket, "weight": weight, "input_index": index})

    by_bucket = {name: [item for item in valid if item["bucket"] == name] for name in ("template_conditioned", "template_free", "other")}
    effective_requested = requested if by_bucket["template_conditioned"] else 0.0
    nonconditioned = cap * (1.0 - effective_requested)
    conditioned = cap * effective_requested
    free_weight = sum(item["weight"] for item in by_bucket["template_free"])
    other_weight = sum(item["weight"] for item in by_bucket["other"])
    if not by_bucket["template_free"]:
        free_weight = 0.0
    if not by_bucket["other"]:
        other_weight = 0.0
    if free_weight + other_weight <= 0 and (by_bucket["template_free"] or by_bucket["other"]):
        free_weight = 1.0 if by_bucket["template_free"] else 0.0
        other_weight = 1.0 if by_bucket["other"] else 0.0
    denominator = free_weight + other_weight
    bucket_weights = [conditioned, nonconditioned * free_weight / denominator if denominator else 0.0, nonconditioned * other_weight / denominator if denominator else 0.0]
    if not any(valid):
        raise ValueError("round budget resolver requires at least one valid candidate")
    bucket_counts = _largest_remainder(cap, bucket_weights, ["template_conditioned", "template_free", "other"])
    bucket_allocations = dict(zip(("template_conditioned", "template_free", "other"), bucket_counts))

    allocations: List[Dict[str, Any]] = []
    for bucket in ("template_conditioned", "template_free", "other"):
        members = by_bucket[bucket]
        shares = _largest_remainder(bucket_allocations[bucket], [item["weight"] for item in members], [f"{int(item['input_index']):012d}" for item in members])
        allocations.extend({"id": item["id"], "bucket": bucket, "num_designs": share, "weight": item["weight"], "input_index": item["input_index"]} for item, share in zip(members, shares))
    allocations.sort(key=lambda item: int(item["input_index"]))
    rematerialization = []
    if rejections:
        destinations = [name for name in ("template_free", "other") if bucket_allocations[name] > 0]
        rematerialization.append({"rejected_ids": [item["id"] for item in rejections], "policy": "reject_and_rematerialize", "destination_buckets": destinations})
    body = {
        "schema_version": 1,
        "cap": cap,
        "requested_conditioned_fraction": requested,
        "effective_conditioned_fraction": bucket_allocations["template_conditioned"] / float(cap),
        "bucket_allocations": bucket_allocations,
        "allocations": allocations,
        "rejections": rejections,
        "rematerialization": rematerialization,
    }
    if sum(item["num_designs"] for item in allocations) != cap:
        raise AssertionError("round budget resolver failed total conservation")
    return RoundBudgetResolution(**body, digest=stable_digest(body))


@dataclass(frozen=True)
class TemplateValidationResult:
    schema_version: int
    valid: bool
    status: str
    reason: str
    failures: Tuple[str, ...]
    template_id: str = ""
    source_structure: str = ""
    source_digest: str = ""
    digest: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_template_application(template: Optional[Mapping[str, Any]]) -> TemplateValidationResult:
    """Validate template applicability without consulting or assigning budget."""
    item = dict(template or {})
    failures: List[str] = []
    source = str(item.get("staged_source_structure_file") or item.get("source_structure_file") or "")
    source_digest = str(item.get("source_digest") or "")
    if not item:
        failures.append("no_template_requested")
    else:
        if item.get("staging_status") == "failed":
            failures.append("template_staging_failed")
        if not item.get("template_id") or not source or not source_digest:
            failures.append("missing_template_source_identity")
        source_path = Path(source) if source else None
        if source_path is not None and source_path.is_file() and source_digest:
            actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual_digest != source_digest:
                failures.append("source_digest_mismatch")
        alignment = dict(item.get("target_alignment") or {})
        if alignment.get("status") != "aligned" or not alignment.get("digest"):
            failures.append("alignment_not_evaluable")
        residue_map = dict(item.get("source_to_effective_residue_map") or {})
        if not residue_map:
            failures.append("missing_source_to_effective_residue_map")
        transform = dict(item.get("length_transform") or {})
        if transform.get("status") not in {"identity", "applied", "validated"} or not transform.get("digest"):
            failures.append("invalid_length_transform")
    failures = list(dict.fromkeys(failures))
    valid = not failures
    status = "validated" if valid else ("not_applicable" if failures == ["no_template_requested"] else "rejected")
    reason = "validated_template_application" if valid else ";".join(failures)
    body = {
        "schema_version": 1,
        "valid": valid,
        "status": status,
        "reason": reason,
        "failures": tuple(failures),
        "template_id": str(item.get("template_id") or ""),
        "source_structure": source,
        "source_digest": source_digest,
    }
    return TemplateValidationResult(**body, digest=stable_digest(body))


@dataclass(frozen=True)
class TemplateApplicationPlan:
    schema_version: int
    template_id: str
    source_digest: str
    source_structure: str
    source_target_identity: Dict[str, Any]
    current_target_identity: Dict[str, Any]
    alignment: Dict[str, Any]
    source_to_effective_residue_map: Dict[str, str]
    length_transform: Dict[str, Any]
    requested_round_fraction: float
    allocated_num_designs: int
    applicability: Dict[str, Any]
    artifact_digests: Dict[str, str] = field(default_factory=dict)
    consumer_receipts: List[Dict[str, Any]] = field(default_factory=list)
    digest: str = ""

    @property
    def current_target(self) -> str:
        return str(self.current_target_identity.get("structure") or "")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _target_identity(structure: str, chain: str = "", supplied_digest: str = "") -> Dict[str, Any]:
    path = Path(str(structure)) if structure else None
    digest = str(supplied_digest or "")
    if not digest and path is not None and path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"structure": str(structure or ""), "chain": str(chain or ""), "digest": digest}


def build_template_application_plan(
    template: Optional[Mapping[str, Any]],
    *,
    current_target: str,
    round_fraction: float,
    allocated_num_designs: int,
    current_target_chain: str = "",
) -> TemplateApplicationPlan:
    item = dict(template or {})
    validation = validate_template_application(item)
    applicability = {
        "applicable": validation.valid,
        "status": validation.status,
        "reason": validation.reason,
        "validation_digest": validation.digest,
    }
    source = str(item.get("staged_source_structure_file") or item.get("source_structure_file") or "")
    alignment = dict(item.get("target_alignment") or {})
    residue_map = dict(item.get("source_to_effective_residue_map") or {})
    transform = dict(item.get("length_transform") or {})
    source_target = dict(item.get("source_target_identity") or {})
    current_identity = dict(item.get("current_target_identity") or {})
    source_target = source_target or _target_identity(source, str(alignment.get("source_target_chain") or ""), str(item.get("source_digest") or ""))
    current_identity = current_identity or _target_identity(current_target, current_target_chain or str(alignment.get("current_target_chain") or ""))
    artifact_digests = {"source": str(item.get("source_digest") or ""), "alignment": str(alignment.get("digest") or ""), "residue_map": stable_digest(residue_map) if residue_map else "", "length_transform": str(transform.get("digest") or "")}
    body = {
        "schema_version": 1,
        "template_id": str(item.get("template_id") or ""),
        "source_digest": str(item.get("source_digest") or ""),
        "source_structure": source,
        "source_target_identity": source_target,
        "current_target_identity": current_identity,
        "alignment": alignment,
        "source_to_effective_residue_map": residue_map,
        "length_transform": transform,
        "requested_round_fraction": float(round_fraction),
        "allocated_num_designs": int(allocated_num_designs),
        "applicability": applicability,
        "artifact_digests": artifact_digests,
        "consumer_receipts": [],
    }
    return TemplateApplicationPlan(**body, digest=stable_digest(body))


def bind_template_application_budget(plan: TemplateApplicationPlan, allocated_num_designs: int, *, receipt: Optional[Mapping[str, Any]] = None) -> TemplateApplicationPlan:
    body = plan.to_dict()
    body.pop("digest", None)
    body["allocated_num_designs"] = max(0, int(allocated_num_designs))
    receipts = list(body.get("consumer_receipts") or [])
    if receipt is not None:
        receipts.append(dict(receipt))
    body["consumer_receipts"] = receipts
    return TemplateApplicationPlan(**body, digest=stable_digest(body))


def resolve_execution_plan(
    params: Mapping[str, Any],
    *,
    job_id: str = "",
    current_params: Optional[Mapping[str, Any]] = None,
    candidate_upper_bound: Optional[int] = None,
    operational_bounds: Optional[Mapping[str, Mapping[str, Any]]] = None,
    final_parameter_state: Optional[Mapping[str, Any]] = None,
    parameter_catalog: Optional[Mapping[str, Sequence[float]]] = None,
    parameter_catalog_digest: str = "",
) -> ExecutionPlan:
    """Resolve one immutable source of truth before adapter/executor rendering."""
    resolved = dict(params or {})
    lineage: Dict[str, List[Dict[str, Any]]] = {}

    def record(key: str, source: str, requested: Any, effective: Any, reason: str) -> None:
        contract = parameter_contract_entry(key) or {}
        lineage.setdefault(key, []).append({
            "source": source,
            "owner": contract.get("owner", "unregistered"),
            "requested": requested,
            "effective": effective,
            "reason": reason,
        })

    raw_num_designs = resolved.get("num_designs")
    if raw_num_designs is None:
        raw_num_designs = resolved.get("num_designs_per_round")
    num_designs = _positive_int("num_designs", raw_num_designs)
    resolved["num_designs"] = num_designs
    if "num_designs_per_round" in resolved:
        previous = resolved["num_designs_per_round"]
        resolved["num_designs_per_round"] = num_designs
        record("num_designs_per_round", "resolver", previous, num_designs, "canonicalized to logical num_designs")
    record("num_designs", "resolver", raw_num_designs, num_designs, "logical design budget source of truth")

    inverse_fold = _positive_int("inverse_fold_num_sequences", resolved.get("inverse_fold_num_sequences", 1))
    resolved["inverse_fold_num_sequences"] = inverse_fold
    upper = _positive_int("candidate_upper_bound", candidate_upper_bound or num_designs * inverse_fold)
    if upper < num_designs:
        raise ValueError("candidate_upper_bound cannot be below logical num_designs")

    effective_bounds = {key: dict(value) for key, value in PARAM_BOUNDS.items()}
    for key, user_bound in dict(operational_bounds or resolved.pop("sampler_bounds", {}) or {}).items():
        if key not in effective_bounds or not isinstance(user_bound, Mapping):
            continue
        absolute = effective_bounds[key]
        narrowed = dict(absolute)
        narrowed["min"] = max(float(absolute["min"]), float(user_bound.get("min", absolute["min"])))
        narrowed["max"] = min(float(absolute["max"]), float(user_bound.get("max", absolute["max"])))
        if narrowed["min"] > narrowed["max"]:
            raise ValueError(f"{key} operational bounds do not overlap the absolute envelope")
        if user_bound.get("max_step_abs") is not None:
            narrowed["max_step_abs"] = min(float(user_bound["max_step_abs"]), narrowed["max"] - narrowed["min"])
        effective_bounds[key] = narrowed

    final_state = dict(final_parameter_state or {})
    if final_state or parameter_catalog_digest:
        from binderloop.parameter_decision import PROBABILISTIC_SAMPLER_KEYS
        catalog = dict(parameter_catalog or {})
        sampler_keys = frozenset(catalog) or PROBABILISTIC_SAMPLER_KEYS
        for key, value in final_state.items():
            if key not in sampler_keys:
                raise ValueError(f"new execution schema contains unsupported final parameter: {key}")
            allowed = tuple(float(item) for item in catalog.get(key, ()))
            if not allowed or float(value) not in allowed:
                raise ValueError(f"{key} final value is not an exact catalog member")
            if key in resolved and float(resolved[key]) != float(value):
                raise ValueError(f"{key} resolved value differs from immutable final state")
            resolved[key] = float(value)
        non_sampler = {key: value for key, value in resolved.items() if key not in sampler_keys}
        clamped, notes = clamp_config_with_inertia(non_sampler, current_config=current_params, bounds=effective_bounds)
        resolved = {**clamped, **{key: resolved[key] for key in sampler_keys if key in resolved}}
    else:
        clamped, notes = clamp_config_with_inertia(resolved, current_config=current_params, bounds=effective_bounds)
        resolved = clamped
    for note in notes:
        key = str(note["parameter"])
        record(key, "resolver_clamp", note["proposed"], note["clamped_to"], "; ".join(note["reasons"]))
    for key in ("step_scale", "noise_scale"):
        if key in resolved:
            value = float(resolved[key])
            bounds = effective_bounds[key]
            if not float(bounds["min"]) <= value <= float(bounds["max"]):
                raise ValueError(f"{key} escaped resolver bounds")

    devices = _positive_int("devices", resolved.get("devices", 1))
    hosts = _positive_int("host_count", resolved.get("host_count", resolved.get("taiji_submit_host_num", 1)))
    resolved["devices"] = devices
    requested_budget = resolved.get("budget")
    if requested_budget is None:
        raise ValueError("execution resolver requires explicit budget; adapter fallbacks are forbidden")
    requested_budget = _positive_int("budget", requested_budget)
    budget_floor = max(MULTI_GPU_BUDGET_FLOOR, upper) if devices * hosts > 1 else upper
    effective_budget = max(requested_budget, budget_floor)
    resolved["budget"] = effective_budget
    record("budget", "resolver", requested_budget, effective_budget, "multi-GPU candidate-preservation floor" if devices * hosts > 1 else "candidate upper-bound floor")

    templates = resolved.get("binder_templates") or ([resolved["binder_template"]] if resolved.get("binder_template") else [])
    applicability = {
        "template_conditioned_fraction": {
            "applicable": bool(templates),
            "reason": "effective_templates_available" if templates else "not_applicable:no_effective_templates",
        }
    }
    if not templates and "template_conditioned_fraction" in resolved:
        requested_fraction = resolved.pop("template_conditioned_fraction")
        record("template_conditioned_fraction", "applicability", requested_fraction, None, "not_applicable:no_effective_templates")

    base = {
        "schema_version": 1,
        "job_id": str(job_id),
        "resolved_params": resolved,
        "lineage": lineage,
        "applicability": applicability,
        "candidate_upper_bound": upper,
        "logical_num_designs": num_designs,
        "artifact_digests": {},
        "parity": {},
        "consumer_receipts": [],
        "final_parameter_state": final_state,
        "parameter_catalog_digest": str(parameter_catalog_digest or ""),
    }
    return ExecutionPlan(**base, plan_digest=stable_digest(base))


def finalize_execution_plan(
    plan: ExecutionPlan,
    *,
    design_spec: Any,
    command: Sequence[str],
    shards: Optional[Sequence[Mapping[str, Any]]],
    consumer_receipts: Sequence[Mapping[str, Any]],
) -> ExecutionPlan:
    shard_list = [dict(item) for item in (shards or [])]
    cli_num_designs = _command_int(command, "--num_designs")
    shard_total = sum(int(item.get("num_designs", 0)) for item in shard_list) if shard_list else cli_num_designs
    parity = {
        "logical_num_designs": plan.logical_num_designs,
        "cli_num_designs": cli_num_designs,
        "shard_num_designs": shard_total,
        "num_designs_conserved": plan.logical_num_designs == cli_num_designs == shard_total,
    }
    if not parity["num_designs_conserved"]:
        raise ValueError(f"execution num_designs parity failure: {parity}")
    artifact_digests = {
        "resolved_params": stable_digest(plan.resolved_params),
        "design_spec": stable_digest(design_spec),
        "cli": stable_digest(list(command)),
        "shards": stable_digest(shard_list),
    }
    body = plan.to_dict()
    body.update({"artifact_digests": artifact_digests, "parity": parity, "consumer_receipts": [dict(r) for r in consumer_receipts]})
    body.pop("plan_digest", None)
    return ExecutionPlan(**body, plan_digest=stable_digest(body))


def _command_int(command: Sequence[str], flag: str) -> int:
    tokens = list(map(str, command))
    try:
        return _positive_int(flag, tokens[tokens.index(flag) + 1])
    except (ValueError, IndexError):
        raise ValueError(f"rendered command is missing a valid {flag}")
