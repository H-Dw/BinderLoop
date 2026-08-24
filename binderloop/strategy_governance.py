"""Pre-budget resolution of executable closed-loop strategy interventions.

This module deliberately describes only harness strategy semantics.  It does not
add model flags; immutable plans contain a projection of values already supported
by the BoltzGen renderer, design spec, template translator, or selection layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import statistics
import re
from enum import Enum
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from binderloop.models.base import DesignJob
from binderloop.agents.config_parameter_contract import canonicalize_config_parameter_value


class ArmApplicability(str, Enum):
    ELIGIBLE = "eligible"
    NOT_APPLICABLE = "not_applicable"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"
    DUPLICATE_EFFECTIVE_INTERVENTION = "duplicate_effective_intervention"


@dataclass(frozen=True)
class CandidateIntervention:
    arm: str
    family: str
    bundle: Tuple[str, ...] = ()
    direction: str = "hold"
    evidence: Tuple[str, ...] = ()
    proposed_changes: Mapping[str, Any] = field(default_factory=dict)
    branch_role: str = "probe"

    @property
    def intent_digest(self) -> str:
        return intent_semantic_digest(self)


@dataclass(frozen=True)
class ResolvedInterventionPlan:
    schema_version: int
    candidate: CandidateIntervention
    applicability: ArmApplicability
    reason: str
    baseline_semantics: Mapping[str, Any]
    resolved_semantics: Mapping[str, Any]
    intent_digest: str
    effective_intervention_digest: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["applicability"] = self.applicability.value
        return value


@dataclass(frozen=True)
class ImmutableBranchPlan:
    schema_version: int
    branch_id: str
    parent_branch_id: str
    branch_role: str
    applicability: ArmApplicability
    intent_digest: str
    effective_intervention_digest: str
    semantic_projection: Mapping[str, Any]
    allocated_designs: int
    plan_digest: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["applicability"] = self.applicability.value
        return value


FAMILY_KEYS: Mapping[str, frozenset[str]] = MappingProxyType({
    "sampler": frozenset({"alpha", "noise_scale", "step_scale", "gamma_0", "diffusion_batch_size", "inverse_fold_num_sequences", "inverse_fold_avoid"}),
    "length": frozenset({"binder_lengths", "binder_length"}),
    "target_context": frozenset({"auxiliary_hotspots", "expanded_binding_residues", "negative_binding_residues", "target_binding_types", "target_include", "structure_groups", "epitope_crop_mode"}),
    "template": frozenset({"binder_template", "binder_templates", "template_conditioned_fraction"}),
    "selection": frozenset({"selection_policy", "additional_filters", "filter_biased"}),
    "sequence": frozenset({"inverse_fold_avoid", "inverse_fold_num_sequences", "filter_biased", "temperature"}),
})
SUPPORTED_BUNDLES = frozenset({
    frozenset({"sampler"}), frozenset({"length"}), frozenset({"target_context"}),
    frozenset({"target_context", "length"}), frozenset({"target_context", "sampler"}),
    frozenset({"length", "sampler"}), frozenset({"target_context", "length", "sampler"}),
    frozenset({"template", "length"}),
    frozenset({"sequence"}), frozenset({"sequence", "length"}),
})

# Metadata is intentionally absent.  Adding a strategy label cannot make a no-op
# branch unique or executable.
_MODEL_KEYS = frozenset({"alpha", "noise_scale", "step_scale", "gamma_0", "diffusion_batch_size", "inverse_fold_num_sequences", "inverse_fold_avoid", "filter_biased", "temperature", "additional_filters", "config_overrides", "protocol", "devices", "num_workers"})
_SAMPLER_KEYS = frozenset({"alpha", "noise_scale", "step_scale", "gamma_0", "diffusion_batch_size"})
_SEQUENCE_KEYS = frozenset({"inverse_fold_num_sequences", "inverse_fold_avoid", "filter_biased", "temperature"})


def _normal(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normal(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if isinstance(value, set):
        return sorted(_normal(v) for v in value)
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(_normal(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def intent_semantic_digest(candidate: CandidateIntervention) -> str:
    return _digest({"arm": candidate.arm, "family": candidate.family, "bundle": list(candidate.bundle), "direction": candidate.direction, "evidence": list(candidate.evidence), "branch_role": candidate.branch_role})


def semantic_projection(job: DesignJob, *, allocated_designs: Optional[int] = None) -> Dict[str, Any]:
    params = dict(job.params or {})
    projection: Dict[str, Any] = {
        "model_params": {key: canonicalize_config_parameter_value(key, params[key]) for key in sorted(_MODEL_KEYS) if key in params},
        "binder_lengths": sorted({int(v) for v in (params.get("binder_lengths") or [job.binder_length])}),
        "target": {
            "structure": job.target_structure,
            "chain": job.chain_id,
            "hotspots": sorted({str(v) for v in job.hotspots or []}),
            "include": params.get("target_include"),
            "structure_groups": params.get("structure_groups"),
            "binding_types": params.get("target_binding_types"),
            "expanded_binding_residues": params.get("expanded_binding_residues"),
            "negative_binding_residues": params.get("negative_binding_residues"),
            "epitope_crop_mode": params.get("epitope_crop_mode"),
        },
        "template_plan": params.get("template_application_plan") or params.get("binder_template"),
        "selection_policy": params.get("selection_policy"),
        "execution_policy": {
            "binding_site_policy": params.get("binding_site_policy"),
            "target_context_policy": params.get("target_context_policy"),
            "sampler_policy": params.get("sampler_policy"),
        },
    }
    if allocated_designs is not None:
        projection["branch_budget"] = int(allocated_designs)
    return _normal(projection)


def effective_semantic_digest(job: DesignJob, *, allocated_designs: Optional[int] = None) -> str:
    """Digest executable semantics only; attribution labels are deliberately absent."""
    return _digest(semantic_projection(job, allocated_designs=allocated_designs))


def attribution_identity_digest(job: DesignJob) -> str:
    """Digest arm/branch attribution independently from executable semantics."""
    params = dict(job.params or {})
    return _digest({
        "arm_id": str(params.get("arm_id") or params.get("exploration_arm") or ""),
        "logical_branch_id": str(params.get("logical_branch_id") or params.get("branch_id") or ""),
        "branch_role": "control" if str(params.get("arm_id") or params.get("exploration_arm") or "") == "baseline_hold" else "probe",
        "intent_digest": str(params.get("intent_digest") or (params.get("resolved_intervention_plan") or {}).get("intent_digest") or ""),
    })


def job_identity_semantic_digest(job: DesignJob) -> str:
    """Digest the fully materialized execution semantics used by job identity.

    Pre-budget strategy deduplication intentionally uses ``effective_semantic_digest``
    without a budget. Identity finalization, however, happens after budget and
    execution-plan materialization, so the allocated branch budget and execution
    partition must be included.
    """
    params = dict(job.params or {})
    immutable = dict(params.get("immutable_branch_plan") or {})
    allocated = immutable.get("allocated_designs", params.get("num_designs"))
    try:
        allocated_designs = int(allocated) if allocated not in (None, "") else None
    except (TypeError, ValueError):
        allocated_designs = None
    projection = semantic_projection(job, allocated_designs=allocated_designs)
    projection["attribution_identity_digest"] = attribution_identity_digest(job)
    shard = dict(params.get("multi_taiji_host_shard") or {})
    if shard:
        projection["execution_partition"] = {
            "kind": "taiji_host_shard",
            "shard_count": int(shard.get("shard_count") or 1),
            "shard_index": int(shard.get("shard_index") or 0),
            "num_designs": int(shard.get("num_designs") or allocated_designs or 0),
        }
    identity_slot = params.get("identity_execution_slot")
    if identity_slot not in (None, ""):
        projection["identity_execution_slot"] = int(identity_slot)
    native = dict(params.get("native_taiji_multi_host") or {})
    if native:
        projection["execution_partition"] = {
            "kind": "taiji_native_multi_host",
            "host_count": int(native.get("host_count") or params.get("host_count") or 1),
            "gpus_per_host": int(native.get("gpus_per_host") or params.get("devices") or 1),
        }
    return _digest(projection)


def _changed_families(baseline: DesignJob, proposed: DesignJob) -> Tuple[str, ...]:
    base = semantic_projection(baseline)
    candidate = semantic_projection(proposed)
    changed: List[str] = []
    if any(base["model_params"].get(key) != candidate["model_params"].get(key) for key in _SAMPLER_KEYS):
        changed.append("sampler")
    if base["binder_lengths"] != candidate["binder_lengths"]:
        changed.append("length")
    if base["target"] != candidate["target"]:
        changed.append("target_context")
    if base["template_plan"] != candidate["template_plan"]:
        changed.append("template")
    if base["selection_policy"] != candidate["selection_policy"]:
        changed.append("selection")
    if any(base["model_params"].get(key) != candidate["model_params"].get(key) for key in _SEQUENCE_KEYS):
        changed.append("sequence")
    return tuple(changed)


def assess_candidate_intervention(
    candidate: CandidateIntervention, baseline: DesignJob, proposed: DesignJob, *,
    blocked_digests: Iterable[str] = (), seen_effective_digests: Iterable[str] = (),
) -> ResolvedInterventionPlan:
    bundle = _changed_families(baseline, proposed)
    applicability = ArmApplicability.ELIGIBLE
    reason = "semantic_delta_resolved"
    if not bundle:
        applicability, reason = ArmApplicability.NOT_APPLICABLE, "no_effective_semantic_delta"
    elif frozenset(bundle) not in SUPPORTED_BUNDLES:
        applicability, reason = ArmApplicability.UNSUPPORTED, "unsupported_interaction_bundle"
    if candidate.arm == "target_context_focus" and "target_context" not in bundle:
        applicability, reason = ArmApplicability.NOT_APPLICABLE, "target_context_translator_produced_no_semantic_diff"
    if candidate.arm == "sampler_explore" and "sampler" not in bundle:
        applicability, reason = ArmApplicability.NOT_APPLICABLE, "sampler_candidate_has_no_legal_nonzero_cli_delta"
    if candidate.arm == "sequence_repair" and "sequence" not in bundle:
        applicability, reason = ArmApplicability.NOT_APPLICABLE, "sequence_tool_produced_no_semantic_diff"
    effective = effective_semantic_digest(proposed)
    if effective in {str(v) for v in blocked_digests}:
        applicability, reason = ArmApplicability.BLOCKED, "effective_intervention_digest_in_cooldown"
    elif effective in {str(v) for v in seen_effective_digests}:
        applicability, reason = ArmApplicability.DUPLICATE_EFFECTIVE_INTERVENTION, "duplicate_effective_intervention"
    resolved_candidate = CandidateIntervention(
        arm=candidate.arm, family="+".join(bundle) or candidate.family, bundle=bundle,
        direction=candidate.direction, evidence=candidate.evidence,
        proposed_changes=candidate.proposed_changes, branch_role=candidate.branch_role,
    )
    return ResolvedInterventionPlan(1, resolved_candidate, applicability, reason, semantic_projection(baseline), semantic_projection(proposed), resolved_candidate.intent_digest, effective)


def _copy_job(source: DesignJob, *, job_id: str, output_dir: str, params: Mapping[str, Any], binder_length: Optional[int] = None) -> DesignJob:
    return DesignJob(job_id=job_id, target_structure=source.target_structure, chain_id=source.chain_id, hotspots=list(source.hotspots), binder_length=int(binder_length if binder_length is not None else source.binder_length), seed=source.seed, params=dict(params), output_dir=output_dir)


JOB_IDENTITY_SCHEMA_VERSION = 2


def safe_path_component(value: Any, *, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip(".-")
    return (text[:80] or fallback)


def resolved_within(path: Any, root: Any) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(root).resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def arm_output_root(output_root: str, round_id: int, rank: int, arm_id: str, arm_digest: str) -> Path:
    base = Path(output_root) / f"r{int(round_id)}" / "arms"
    target = base / f"{int(rank):02d}_{safe_path_component(arm_id, fallback='arm')}_{safe_path_component(arm_digest[:12], fallback='digest')}"
    if not resolved_within(target, base):
        raise ValueError("unsafe_arm_output_path")
    return target


def materialize_deterministic_job_identities(
    jobs: Sequence[DesignJob], *, round_id: int, output_root: str, finalized: bool = False,
) -> List[DesignJob]:
    """Assign deterministic arm, logical-branch and execution identities."""
    rows = list(jobs)
    # Before execution fanout, multiple independent arms/logical branches are
    # valid. Reject only repeated identities within the same logical branch:
    # those are unpartitioned copies, not distinct strategy proposals.
    seen_logical: set[tuple[str, str]] = set()
    if not finalized:
        for job in rows:
            params = dict(job.params or {})
            arm_id = str(params.get("arm_id") or params.get("exploration_arm") or "baseline_hold")
            logical_branch_id = str(
                params.get("logical_branch_id") or params.get("branch_id")
                or f"r{int(round_id)}_{safe_path_component(arm_id)}"
            )
            identity = (logical_branch_id, effective_semantic_digest(job))
            if identity in seen_logical:
                raise ValueError("duplicate_logical_job_identity")
            seen_logical.add(identity)
    seen_execution: set[str] = set()
    arm_ranks: Dict[str, int] = {}
    for index, job in enumerate(rows):
        params = dict(job.params or {})
        arm_id = str(params.get("arm_id") or params.get("exploration_arm") or "baseline_hold")
        params["arm_id"] = arm_id
        params["exploration_arm"] = str(params.get("exploration_arm") or arm_id)
        arm_rank = int(params.get("arm_rank", arm_ranks.setdefault(arm_id, len(arm_ranks))))
        params["arm_rank"] = arm_rank
        logical_branch_id = str(params.get("logical_branch_id") or params.get("branch_id") or f"r{int(round_id)}_{safe_path_component(arm_id)}")
        params["logical_branch_id"] = logical_branch_id
        params["branch_id"] = logical_branch_id
        arm_digest = str(params.get("arm_digest") or effective_semantic_digest(job))
        params["arm_digest"] = arm_digest
        root = arm_output_root(output_root, round_id, arm_rank, arm_id, arm_digest)
        params["arm_root"] = str(root)

        # Digest only after every attribution identity field is visible through
        # job.params.  The digest functions project explicit semantic/attribution
        # fields, so the generated digest and metadata fields cannot self-reference.
        job.params = params
        execution_semantic = effective_semantic_digest(job)
        attribution_digest = attribution_identity_digest(job)
        params["execution_semantic_digest"] = execution_semantic
        params["attribution_identity_digest"] = attribution_digest
        logical_digest = _digest({"execution_semantic_digest": execution_semantic, "attribution_identity_digest": attribution_digest})
        generated_logical_id = f"r{int(round_id)}_arm{arm_rank:02d}_{logical_digest[:12]}"
        logical_job_id = str(params.get("logical_job_id") or generated_logical_id) if finalized else generated_logical_id
        params["logical_job_id"] = logical_job_id
        if finalized:
            execution_slot = int(params.get("execution_slot", index))
            params["execution_slot"] = execution_slot
            job.params = params
            digest = job_identity_semantic_digest(job)
            execution_job_id = f"{logical_job_id}_job{execution_slot:02d}_{digest[:10]}"
            target = root / "jobs" / safe_path_component(execution_job_id, fallback="job")
        else:
            digest = logical_digest
            execution_slot = None
            execution_job_id = logical_job_id
            target = root / "logical"
        if digest in seen_execution:
            raise ValueError("duplicate_execution_identity")
        seen_execution.add(digest)
        if not resolved_within(target, root):
            raise ValueError("unsafe_job_output_path")
        params["execution_job_id"] = execution_job_id
        params["job_identity"] = {
            "schema_version": JOB_IDENTITY_SCHEMA_VERSION, "job_id": execution_job_id,
            "arm_id": arm_id, "branch_id": logical_branch_id, "logical_branch_id": logical_branch_id,
            "execution_slot": execution_slot, "semantic_digest": digest,
            "execution_semantic_digest": execution_semantic,
            "attribution_identity_digest": attribution_digest,
            "finalized": bool(finalized), "purpose": "arm_scoped_execution_identity",
        }
        job.job_id = execution_job_id
        job.output_dir = str(target)
        job.params = params
    return rows




def deduplicate_effective_jobs(jobs: Sequence[DesignJob]) -> List[DesignJob]:
    seen: set[tuple[str, str]] = set()
    kept: List[DesignJob] = []
    for job in jobs:
        digest = effective_semantic_digest(job)
        identity = (digest, attribution_identity_digest(job))
        if identity in seen:
            continue
        seen.add(identity)
        job.params["effective_intervention_digest"] = digest
        kept.append(job)
    return kept


def finalize_immutable_branch_plan(job: DesignJob, allocated_designs: int) -> ImmutableBranchPlan:
    params = dict(job.params or {})
    resolved = dict(params.get("resolved_intervention_plan") or {})
    projection = semantic_projection(job, allocated_designs=allocated_designs)
    effective = _digest(projection)
    payload = {"schema_version": 1, "branch_id": str(params.get("branch_id") or job.job_id), "parent_branch_id": str(params.get("parent_branch_id") or ""), "branch_role": "control" if params.get("exploration_arm") == "baseline_hold" else "probe", "applicability": ArmApplicability.ELIGIBLE.value, "intent_digest": str(params.get("intent_digest") or resolved.get("intent_digest") or ""), "effective_intervention_digest": effective, "semantic_projection": projection, "allocated_designs": int(allocated_designs)}
    return ImmutableBranchPlan(plan_digest=_digest(payload), **{**payload, "applicability": ArmApplicability.ELIGIBLE})


# ---------------------------------------------------------------------------
# Durable owner state and matched binding-site outcome governance.
# These types live beside the immutable intervention-plan types so the module can
# remain the single consolidation point for closed-loop strategy semantics.

@dataclass
class LengthPolicyState:
    schema_version: str = "1.0"
    initial_lengths: List[int] = field(default_factory=list)
    current_lengths: List[int] = field(default_factory=list)
    allowed_range: List[int] = field(default_factory=list)
    baseline_round_id: Optional[int] = None
    baseline_lengths: List[int] = field(default_factory=list)
    round_lengths: Dict[str, List[int]] = field(default_factory=dict)
    last_recommendation: Dict[str, Any] = field(default_factory=dict)
    last_transition: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def initialize(cls, lengths: Sequence[int], allowed_range: Sequence[int] = ()) -> "LengthPolicyState":
        normalized = _normalized_lengths(lengths)
        bounds = _normalized_lengths(allowed_range)
        return cls(initial_lengths=normalized, current_lengths=normalized,
                   allowed_range=[min(bounds), max(bounds)] if bounds else [],
                   baseline_lengths=normalized)

    def clamp(self, lengths: Sequence[int]) -> List[int]:
        values = _normalized_lengths(lengths)
        if self.allowed_range:
            lo, hi = min(self.allowed_range), max(self.allowed_range)
            values = [value for value in values if lo <= value <= hi]
        return values or list(self.initial_lengths)

    def record_round(self, round_id: int, lengths: Sequence[int]) -> None:
        normalized = self.clamp(lengths)
        self.round_lengths[str(int(round_id))] = normalized
        if self.baseline_round_id is None:
            self.baseline_round_id = int(round_id)
            self.baseline_lengths = list(normalized)
        self.current_lengths = list(normalized)

    def select_next(self, *, round_id: int, recommended_lengths: Sequence[int],
                    branch_action: str, branch_from_round: int,
                    recommendation: Optional[Mapping[str, Any]] = None) -> List[int]:
        before = list(self.current_lengths)
        if branch_action in {"replay_best", "branch_from_best"}:
            selected = self.clamp(self.round_lengths.get(str(int(branch_from_round))) or self.baseline_lengths)
            reason = "restored_branch_baseline"
        else:
            selected = self.clamp(recommended_lengths or self.current_lengths)
            reason = "accepted_length_policy_recommendation"
        self.current_lengths = list(selected)
        self.last_recommendation = dict(recommendation or {})
        self.last_transition = {"round_id": int(round_id), "branch_action": str(branch_action),
                                "branch_from_round": int(branch_from_round), "before": before,
                                "after": list(selected), "reason": reason}
        return list(selected)


@dataclass
class BindingSiteResolution:
    schema_version: str = "1.0"
    primary: List[str] = field(default_factory=list)
    expanded: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)
    provenance: Dict[str, Dict[str, str]] = field(default_factory=dict)
    effective_binding_types: List[Dict[str, Any]] = field(default_factory=list)
    semantic_digest: str = ""
    retracted_expanded: List[str] = field(default_factory=list)

    @classmethod
    def rebuild(cls, *, primary_residues: Sequence[Any], expanded_residues: Sequence[Any] = (),
                negative_residues: Sequence[Any] = (), original_binding_types: Sequence[Mapping[str, Any]] = (),
                expanded_source: str = "harness_expanded_hotspot") -> "BindingSiteResolution":
        original_positive, original_negative = _binding_type_residues(original_binding_types)
        primary = _residue_tokens(primary_residues) | original_positive
        negative = _residue_tokens(negative_residues) | original_negative
        expanded = _residue_tokens(expanded_residues) - primary - negative
        provenance = {token: {"role": "primary", "source": "user"} for token in sorted(primary, key=_residue_sort_key)}
        provenance.update({token: {"role": "expanded", "source": expanded_source} for token in sorted(expanded, key=_residue_sort_key)})
        provenance.update({token: {"role": "negative", "source": "user"} for token in sorted(negative, key=_residue_sort_key)})
        payload = {"primary": sorted(primary, key=_residue_sort_key),
                   "expanded": sorted(expanded, key=_residue_sort_key),
                   "negative": sorted(negative, key=_residue_sort_key),
                   "semantics": "binary_BINDING_NOT_BINDING_v1"}
        return cls(primary=payload["primary"], expanded=payload["expanded"], negative=payload["negative"],
                   provenance=provenance, effective_binding_types=_materialize_binary_binding_types(primary | expanded, negative),
                   semantic_digest=_digest(payload)[:16])

    def without_expanded(self, residues: Sequence[Any]) -> "BindingSiteResolution":
        removed = _residue_tokens(residues)
        resolved = self.rebuild(primary_residues=self.primary,
                                expanded_residues=[value for value in self.expanded if value not in removed],
                                negative_residues=self.negative,
                                expanded_source="retained_expanded_hotspot")
        resolved.retracted_expanded = sorted(set(self.retracted_expanded) | removed, key=_residue_sort_key)
        return resolved


@dataclass
class MatchedHotspotOutcome:
    schema_version: str = "1.0"
    control: Dict[str, Optional[float]] = field(default_factory=dict)
    expanded: Dict[str, Optional[float]] = field(default_factory=dict)
    deltas: Dict[str, Optional[float]] = field(default_factory=dict)
    matched_pairs: int = 0
    credible_benefit: bool = False
    rationale: List[str] = field(default_factory=list)


def compare_matched_hotspot_outcome(control_rows: Sequence[Mapping[str, Any]],
                                    expanded_rows: Sequence[Mapping[str, Any]], *,
                                    min_pairs: int = 1) -> MatchedHotspotOutcome:
    control_map = {_hotspot_match_key(row, index): row for index, row in enumerate(control_rows)}
    expanded_map = {_hotspot_match_key(row, index): row for index, row in enumerate(expanded_rows)}
    shared = sorted(set(control_map) & set(expanded_map))
    pairs = [(control_map[key], expanded_map[key]) for key in shared]
    if not pairs and control_rows and expanded_rows:
        pairs = list(zip(list(control_rows)[:min(len(control_rows), len(expanded_rows))],
                         list(expanded_rows)[:min(len(control_rows), len(expanded_rows))]))
    control = _hotspot_cohort_metrics([pair[0] for pair in pairs])
    expanded = _hotspot_cohort_metrics([pair[1] for pair in pairs])
    deltas: Dict[str, Optional[float]] = {}
    for key in control:
        if control[key] is None or expanded[key] is None:
            deltas[key] = None
        elif key == "interface_pae":
            deltas[key] = round(float(control[key]) - float(expanded[key]), 6)
        else:
            deltas[key] = round(float(expanded[key]) - float(control[key]), 6)
    gains = [float(value) for value in deltas.values() if value is not None]
    material_gain = any(value >= 0.02 for value in gains)
    material_harm = any(value < -0.02 for value in gains)
    credible = len(pairs) >= max(1, int(min_pairs)) and bool(gains) and material_gain and not material_harm
    rationale = [f"matched_pairs={len(pairs)}",
                 "credible benefit requires >=0.02 primary-endpoint gain and no endpoint regression below -0.02"]
    rationale.append("expanded hotspot showed credible matched benefit" if credible else
                     "no credible matched benefit for the expanded hotspot")
    return MatchedHotspotOutcome(control=control, expanded=expanded, deltas=deltas,
                                 matched_pairs=len(pairs), credible_benefit=credible, rationale=rationale)


def retract_unbeneficial_expanded_hotspots(previous: BindingSiteResolution,
                                            current: BindingSiteResolution,
                                            outcome: MatchedHotspotOutcome) -> Tuple[BindingSiteResolution, List[str]]:
    newly_added = sorted(set(current.expanded) - set(previous.expanded), key=_residue_sort_key)
    if outcome.credible_benefit or not newly_added:
        return current, []
    return current.without_expanded(newly_added), newly_added


def _normalized_lengths(values: Sequence[Any]) -> List[int]:
    result: List[int] = []
    for value in values or []:
        try: number = int(value)
        except (TypeError, ValueError): continue
        if number > 0 and number not in result: result.append(number)
    return sorted(result)


def _parse_residue(value: Any, default_chain: str = "") -> Optional[str]:
    text = str(value or "").strip()
    if not text: return None
    chain, raw = text.split(":", 1) if ":" in text else (default_chain, text)
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == "-")
    try: residue = int(digits)
    except ValueError: return None
    return f"{chain.strip()}:{residue}" if chain.strip() else None


def _residue_tokens(values: Sequence[Any], default_chain: str = "") -> set[str]:
    return {token for token in (_parse_residue(value, default_chain) for value in values or []) if token}


def _residue_sort_key(token: str) -> Tuple[str, int]:
    chain, residue = token.rsplit(":", 1)
    return chain, int(residue)


def _binding_type_residues(entries: Sequence[Mapping[str, Any]]) -> Tuple[set[str], set[str]]:
    positive: set[str] = set(); negative: set[str] = set()
    for item in entries or []:
        chain = item.get("chain") if isinstance(item, Mapping) and isinstance(item.get("chain"), Mapping) else {}
        chain_id = str(chain.get("id") or "")
        for key, target in (("binding", positive), ("not_binding", negative)):
            for raw in str(chain.get(key) or "").split(","):
                token = _parse_residue(raw, chain_id)
                if token: target.add(token)
    return positive, negative


def _materialize_binary_binding_types(positive: set[str], negative: set[str]) -> List[Dict[str, Any]]:
    by_chain: Dict[str, Dict[str, List[int]]] = {}
    for role, values in (("binding", positive), ("not_binding", negative)):
        for token in sorted(values, key=_residue_sort_key):
            chain, residue = _residue_sort_key(token)
            by_chain.setdefault(chain, {"binding": [], "not_binding": []})[role].append(residue)
    result = []
    for chain in sorted(by_chain):
        payload: Dict[str, Any] = {"id": chain}
        for role in ("binding", "not_binding"):
            if by_chain[chain][role]: payload[role] = ",".join(str(value) for value in by_chain[chain][role])
        result.append({"chain": payload})
    return result


def _hotspot_match_key(row: Mapping[str, Any], index: int) -> str:
    for key in ("matched_group_id", "matched_control_id", "match_id", "parent_id", "binder_length", "length"):
        if row.get(key) not in (None, ""): return f"{key}:{row.get(key)}"
    return f"index:{index}"


def _metric_number(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if row.get(key) not in (None, ""):
            try: return float(row[key])
            except (TypeError, ValueError): pass
    return None


def _hotspot_cohort_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    def median(*keys: str) -> Optional[float]:
        values = [_metric_number(row, *keys) for row in rows]
        numeric = [value for value in values if value is not None]
        return round(float(statistics.median(numeric)), 6) if numeric else None
    strict = []
    for row in rows:
        explicit = row.get("strict_positive")
        if explicit is None:
            iptm = _metric_number(row, "design_to_target_iptm", "iptm", "interface_confidence") or 0.0
            pae = _metric_number(row, "min_design_to_target_pae", "min_interaction_pae")
            ptm = _metric_number(row, "design_ptm", "ptm") or 0.0
            rmsd = _metric_number(row, "designfolding_filter_rmsd", "filter_rmsd", "designfolding-filter-rmsd")
            explicit = iptm >= 0.5 and pae is not None and pae <= 10.0 and ptm >= 0.7 and rmsd is not None and rmsd <= 2.5
        strict.append(1.0 if explicit else 0.0)
    return {"interface_iptm": median("design_to_target_iptm", "iptm", "interface_confidence"),
            "interface_pae": median("min_design_to_target_pae", "min_interaction_pae"),
            "primary_coverage": median("primary_hotspot_coverage", "primary_coverage", "hotspot_coverage"),
            "binding_pose": median("binding_pose", "binding_pose_score", "interface_confidence"),
            "strict_yield": round(sum(strict) / len(strict), 6) if strict else None}
