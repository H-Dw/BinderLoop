"""Finite-catalog, probability-based sampler parameter decisions.

The decision layer deliberately returns catalog members only.  It never creates
an interpolated or probability-weighted parameter set.
"""

from dataclasses import dataclass, field
import itertools
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

HOLD_CURRENT = "HOLD_CURRENT"
DEFAULT_SAMPLER_AXES = ("alpha", "noise_scale", "step_scale")


@dataclass(frozen=True, order=True)
class ParameterCandidate:
    """Exact catalog member. BoltzGen keeps the 3-arg constructor; other models use kwargs."""

    values: Tuple[Tuple[str, float], ...] = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 3 and not kwargs:
            mapping = {"alpha": args[0], "noise_scale": args[1], "step_scale": args[2]}
        elif len(args) == 1 and isinstance(args[0], Mapping) and not kwargs:
            mapping = dict(args[0])
        elif kwargs and not args:
            mapping = dict(kwargs)
        elif not args and not kwargs:
            mapping = {}
        else:
            raise TypeError("ParameterCandidate expects (alpha, noise_scale, step_scale) or mapping/kwargs")
        items = tuple(sorted((str(key), float(value)) for key, value in mapping.items()))
        for name, value in items:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "values", items)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.values)

    def __getattr__(self, name: str) -> float:
        mapping = self.as_dict()
        if name in mapping:
            return mapping[name]
        raise AttributeError(name)

    @property
    def key(self) -> str:
        return "|".join(f"{name}={value:g}" for name, value in self.values)


@dataclass(frozen=True)
class DecisionThresholds:
    top_probability: float = 0.65
    margin: float = 0.20
    max_normalized_entropy: float = 0.75

    def __post_init__(self) -> None:
        for name in ("top_probability", "margin", "max_normalized_entropy"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
            object.__setattr__(self, name, value)


@dataclass
class ParameterDecisionSpec:
    alpha_candidates: Sequence[float] = field(default_factory=lambda: [0.001, 0.003, 0.009, 0.027, 0.05])
    noise_scale_candidates: Sequence[float] = field(default_factory=lambda: [0.6, 0.7, 0.8, 0.9])
    step_scale_candidates: Sequence[float] = field(default_factory=lambda: [0.6, 0.8, 1.0])
    gamma_0_candidates: Sequence[float] = field(default_factory=tuple)
    sampler_axes: Optional[Sequence[str]] = None
    top_probability_threshold: float = 0.65
    margin_threshold: float = 0.20
    max_normalized_entropy: float = 0.75
    joint_evidence_fallback_mode: str = "off"

    def __post_init__(self) -> None:
        self.alpha_candidates = _optional_axis("alpha_candidates", self.alpha_candidates)
        self.noise_scale_candidates = _optional_axis("noise_scale_candidates", self.noise_scale_candidates)
        self.step_scale_candidates = _optional_axis("step_scale_candidates", self.step_scale_candidates)
        self.gamma_0_candidates = _optional_axis("gamma_0_candidates", self.gamma_0_candidates)
        if self.sampler_axes is not None:
            axes = tuple(str(item).strip() for item in self.sampler_axes if str(item).strip())
            if not axes:
                raise ValueError("sampler_axes cannot be empty")
            self.sampler_axes = axes
        for key, values in self.active_axes().items():
            if not values:
                raise ValueError(f"{key}_candidates cannot be empty for an active sampler axis")
        mode = str(self.joint_evidence_fallback_mode or "off").strip().lower()
        if mode not in {"off", "shadow", "active"}:
            raise ValueError("joint_evidence_fallback_mode must be one of: off, shadow, active")
        self.joint_evidence_fallback_mode = mode
        self.thresholds  # validate thresholds eagerly

    def active_sampler_keys(self) -> Tuple[str, ...]:
        if self.sampler_axes:
            return tuple(self.sampler_axes)
        return DEFAULT_SAMPLER_AXES

    def active_axes(self) -> Dict[str, Tuple[float, ...]]:
        out: Dict[str, Tuple[float, ...]] = {}
        for key in self.active_sampler_keys():
            values = tuple(getattr(self, f"{key}_candidates", ()) or ())
            out[key] = tuple(float(item) for item in values)
        return out

    @property
    def catalog(self) -> Tuple[ParameterCandidate, ...]:
        axes = self.active_axes()
        names = list(axes)
        return tuple(
            ParameterCandidate(**dict(zip(names, combo)))
            for combo in itertools.product(*(axes[name] for name in names))
        )

    @property
    def thresholds(self) -> DecisionThresholds:
        return DecisionThresholds(
            self.top_probability_threshold,
            self.margin_threshold,
            self.max_normalized_entropy,
        )


def _optional_axis(name: str, values: Optional[Sequence[float]]) -> Tuple[float, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if len(tuple(values)) == 0:
        return ()
    return _validate_axis(name, values)


def _validate_axis(name: str, values: Sequence[float]) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain only numbers")
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


CandidateState = Union[ParameterCandidate, str]


@dataclass(frozen=True)
class ParameterDecision:
    proposed: CandidateState
    final: CandidateState
    probabilities: Mapping[CandidateState, float]
    top_probability: float
    margin: float
    normalized_entropy: float
    held: bool
    reason: str


def normalize_probabilities(weights: Mapping[CandidateState, float]) -> Dict[CandidateState, float]:
    """Validate and normalize non-negative finite weights."""
    cleaned: Dict[CandidateState, float] = {}
    for state, raw_weight in weights.items():
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("probability weights must be finite and non-negative")
        cleaned[state] = weight
    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("probability weights must have a positive sum")
    return {state: weight / total for state, weight in cleaned.items()}


def remove_invalid_and_renormalize(
    weights: Mapping[CandidateState, float],
    catalog: Iterable[ParameterCandidate],
) -> Dict[CandidateState, float]:
    """Drop states outside the catalog (except HOLD_CURRENT), then renormalize."""
    valid = set(_validated_catalog(catalog))
    retained = {
        state: weight
        for state, weight in weights.items()
        if state == HOLD_CURRENT or (isinstance(state, ParameterCandidate) and state in valid)
    }
    if not retained:
        raise ValueError("no valid candidate probability remains")
    return normalize_probabilities(retained)


def map_proposed_to_final(
    proposed: CandidateState,
    catalog: Iterable[ParameterCandidate],
) -> CandidateState:
    """Map a proposal to the exact canonical catalog member; never interpolate."""
    if proposed == HOLD_CURRENT:
        return HOLD_CURRENT
    for candidate in _validated_catalog(catalog):
        if proposed == candidate:
            return candidate
    raise ValueError("proposed parameter set is not an exact catalog member")


def decide_parameters(
    weights: Mapping[CandidateState, float],
    catalog: Iterable[ParameterCandidate],
    thresholds: Optional[DecisionThresholds] = None,
) -> ParameterDecision:
    """Choose a catalog member only when all conservative confidence gates pass."""
    canonical_catalog = _validated_catalog(catalog)
    probabilities = remove_invalid_and_renormalize(weights, canonical_catalog)
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    proposed, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top - runner_up
    entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0.0)
    normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    gates = thresholds or DecisionThresholds()

    failures = []
    if top < gates.top_probability:
        failures.append("top_probability")
    if margin < gates.margin:
        failures.append("margin")
    if normalized_entropy > gates.max_normalized_entropy:
        failures.append("entropy")

    if proposed == HOLD_CURRENT:
        final: CandidateState = HOLD_CURRENT
        reason = "hold_has_highest_probability"
    elif failures:
        final = HOLD_CURRENT
        reason = "conservative_gate_failed:" + ",".join(failures)
    else:
        final = map_proposed_to_final(proposed, canonical_catalog)
        reason = "confidence_gates_passed"

    return ParameterDecision(
        proposed=proposed,
        final=final,
        probabilities=probabilities,
        top_probability=top,
        margin=margin,
        normalized_entropy=normalized_entropy,
        held=final == HOLD_CURRENT,
        reason=reason,
    )


def _validated_catalog(catalog: Iterable[ParameterCandidate]) -> Tuple[ParameterCandidate, ...]:
    result = tuple(catalog)
    if not result:
        raise ValueError("parameter catalog cannot be empty")
    if any(not isinstance(candidate, ParameterCandidate) for candidate in result):
        raise ValueError("parameter catalog must contain ParameterCandidate values")
    if len(set(result)) != len(result):
        raise ValueError("parameter catalog cannot contain duplicates")
    return result


PROBABILISTIC_SAMPLER_KEYS = frozenset(DEFAULT_SAMPLER_AXES)


def sampler_keys_for_spec(spec: Optional[ParameterDecisionSpec] = None) -> Tuple[str, ...]:
    if spec is None:
        return DEFAULT_SAMPLER_AXES
    return spec.active_sampler_keys()


def parameter_axis(spec: ParameterDecisionSpec, key: str) -> Tuple[float, ...]:
    """Return one finite candidate axis by executable parameter name."""
    axes = spec.active_axes()
    if key not in axes:
        raise ValueError(f"unsupported probabilistic sampler key: {key}")
    return tuple(axes[key])


def filter_parameter_candidates(
    candidates: Sequence[float], *, current: Optional[float] = None,
    bounds: Optional[Mapping[str, float]] = None,
) -> Tuple[float, ...]:
    """Filter by physical bounds and inertia without clamping any value."""
    limits = dict(bounds or {})
    out = []
    for raw in candidates:
        value = float(raw)
        if limits.get("min") is not None and value < float(limits["min"]):
            continue
        if limits.get("max") is not None and value > float(limits["max"]):
            continue
        if current is not None:
            now = float(current)
            if limits.get("max_step_abs") is not None and abs(value - now) > float(limits["max_step_abs"]) + 1e-12:
                continue
            if limits.get("max_step_ratio") is not None and now > 0:
                ratio = float(limits["max_step_ratio"])
                if value > now * ratio + 1e-12 or value < now / ratio - 1e-12:
                    continue
        out.append(value)
    return tuple(out)


def decide_parameter_distribution(
    label_probabilities: Mapping[str, float], *, labels_to_values: Mapping[str, Union[float, str]],
    candidates: Sequence[float], current: Optional[float], thresholds: Optional[DecisionThresholds] = None,
    bounds: Optional[Mapping[str, float]] = None, capability_status: str = "supported",
    capability_mode: str = "auto",
) -> Dict[str, Any]:
    """Resolve one labelled distribution to an exact scalar catalog value or HOLD."""
    from typing import Any
    status = str(capability_status or "indeterminate")
    mode = str(capability_mode or "auto")
    if status != "supported":
        if mode == "required":
            raise RuntimeError(f"required logprobs capability is {status}")
        return {"proposed": HOLD_CURRENT, "final": HOLD_CURRENT, "reason": f"capability_{status}", "probabilities": {HOLD_CURRENT: 1.0}, "eligible_candidates": []}
    eligible = filter_parameter_candidates(candidates, current=current, bounds=bounds)
    eligible_set = set(eligible)
    mapped: Dict[Union[float, str], float] = {}
    dropped = []
    for label, probability in label_probabilities.items():
        value = labels_to_values.get(str(label))
        if value == HOLD_CURRENT:
            mapped[HOLD_CURRENT] = mapped.get(HOLD_CURRENT, 0.0) + float(probability)
        elif value is not None and float(value) in eligible_set:
            scalar = float(value)
            mapped[scalar] = mapped.get(scalar, 0.0) + float(probability)
        else:
            dropped.append(str(label))
    if not mapped:
        return {"proposed": HOLD_CURRENT, "final": HOLD_CURRENT, "reason": "no_valid_candidate_probability", "probabilities": {HOLD_CURRENT: 1.0}, "eligible_candidates": list(eligible), "dropped_labels": dropped}
    probabilities = normalize_probabilities(mapped)
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    proposed, top = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top - runner
    entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
    normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    gates = thresholds or DecisionThresholds()
    passed = top >= gates.top_probability and margin >= gates.margin and normalized_entropy <= gates.max_normalized_entropy
    final = proposed if passed and proposed != HOLD_CURRENT else HOLD_CURRENT
    if final != HOLD_CURRENT and float(final) not in eligible_set:
        raise AssertionError("final scalar is not an exact eligible catalog member")
    return {"proposed": proposed, "final": final, "reason": "confidence_gates_passed" if final != HOLD_CURRENT else ("hold_has_highest_probability" if proposed == HOLD_CURRENT else "conservative_gate_failed"), "probabilities": probabilities, "top_probability": top, "margin": margin, "normalized_entropy": normalized_entropy, "eligible_candidates": list(eligible), "dropped_labels": dropped}


@dataclass(frozen=True)
class JointParameterEvidence:
    """One completed observation for an exact, joint sampler state.

    ``comparison_group`` identifies a matched experiment wave.  A challenger is
    treated as exploitation-grade evidence only when the same group contains a
    completed control observation.  Unmatched observations may inform
    exploration statistics, but never the support gate or exploitation quality
    posterior.  Their observed cost may still inform the shared resource model.
    """

    candidate: ParameterCandidate
    successes: int
    trials: int
    replicate_id: str = ""
    comparison_group: str = ""
    is_control: bool = False
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ParameterCandidate):
            raise TypeError("candidate must be a ParameterCandidate")
        trials = int(self.trials)
        successes = int(self.successes)
        if trials < 0 or successes < 0 or successes > trials:
            raise ValueError("joint evidence requires 0 <= successes <= trials")
        cost = float(self.cost)
        if not math.isfinite(cost) or cost <= 0.0:
            raise ValueError("joint evidence cost must be finite and positive")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "replicate_id", str(self.replicate_id or ""))
        object.__setattr__(self, "comparison_group", str(self.comparison_group or ""))
        object.__setattr__(self, "is_control", bool(self.is_control))
        object.__setattr__(self, "cost", cost)


@dataclass(frozen=True)
class JointSelectionPolicy:
    """Conservative policy for evidence-aware joint catalog selection."""

    minimum_replicates: int = 2
    minimum_trials: int = 4
    minimum_matched_controls: int = 2
    minimum_conservative_effect: float = 0.0
    exploitation_fraction: float = 0.5
    uncertainty_weight: float = 0.35
    novelty_weight: float = 0.20
    diversity_weight: float = 0.25
    cost_weight: float = 0.10
    default_candidate_cost: float = 1.0
    max_total_cost: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("minimum_replicates", "minimum_trials", "minimum_matched_controls"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be at least one")
            object.__setattr__(self, name, value)
        fraction = float(self.exploitation_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("exploitation_fraction must be within [0, 1]")
        object.__setattr__(self, "exploitation_fraction", fraction)
        minimum_effect = float(self.minimum_conservative_effect)
        if not math.isfinite(minimum_effect):
            raise ValueError("minimum_conservative_effect must be finite")
        object.__setattr__(self, "minimum_conservative_effect", minimum_effect)
        for name in ("uncertainty_weight", "novelty_weight", "diversity_weight", "cost_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        default_cost = float(self.default_candidate_cost)
        if not math.isfinite(default_cost) or default_cost <= 0.0:
            raise ValueError("default_candidate_cost must be finite and positive")
        object.__setattr__(self, "default_candidate_cost", default_cost)
        if self.max_total_cost is not None:
            budget = float(self.max_total_cost)
            if not math.isfinite(budget) or budget <= 0.0:
                raise ValueError("max_total_cost must be finite and positive")
            object.__setattr__(self, "max_total_cost", budget)


@dataclass(frozen=True)
class JointCandidateScore:
    """Auditable evidence summary used by the deterministic selector."""

    candidate: ParameterCandidate
    successes: int
    trials: int
    replicates: int
    matched_controls: int
    posterior_mean: float
    posterior_uncertainty: float
    matched_successes: int
    matched_trials: int
    matched_replicates: int
    matched_posterior_mean: float
    matched_posterior_uncertainty: float
    matched_control_mean: float
    conservative_effect: float
    estimated_cost: float
    supported: bool


def joint_parameter_evidence_from_rounds(
    rounds: Sequence[Any], *, spec: ParameterDecisionSpec,
    required_target_identity_digest: Optional[str] = None,
    required_catalog_digest: Optional[str] = None,
    required_execution_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[JointParameterEvidence, ...]:
    """Extract unconfounded, completed joint-state evidence from memory rounds.

    The extractor deliberately accepts persisted mappings as well as the
    in-memory dataclasses.  It ignores partial vectors, incomplete executions,
    confounded arms, and non-sampling interventions.  No model-specific command
    or adapter detail is consulted.
    """

    catalog = set(spec.catalog)
    sampler_keys = spec.active_sampler_keys()
    required_context = {
        str(key): str(value)
        for key, value in dict(required_execution_context or {}).items()
        if value not in (None, "")
    }
    extracted: list[Tuple[int, JointParameterEvidence]] = []
    seen = set()

    def as_mapping(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        raw = getattr(value, "__dict__", None)
        return raw if isinstance(raw, Mapping) else {}

    for ordinal, round_value in enumerate(rounds or ()):
        round_row = as_mapping(round_value)
        round_id = int(round_row.get("round_id", ordinal))
        outcomes = [as_mapping(item) for item in (round_row.get("arm_outcomes") or ())]
        jobs = [as_mapping(item) for item in (round_row.get("jobs") or ())]
        pending: list[JointParameterEvidence] = []
        for job in jobs:
            params = as_mapping(job.get("params"))
            if required_target_identity_digest is not None and str(
                params.get("target_identity_digest") or ""
            ) != str(required_target_identity_digest):
                continue
            if any(str(params.get(key) or "") != value for key, value in required_context.items()):
                continue
            arm_id = str(params.get("arm_id") or params.get("exploration_arm") or "")
            branch_id = str(params.get("logical_branch_id") or params.get("branch_id") or "")
            matches = [item for item in outcomes if str(item.get("arm_id") or "") == arm_id]
            arm_job_count = 0
            for other_job in jobs:
                other_params = as_mapping(other_job.get("params"))
                other_arm_id = str(
                    other_params.get("arm_id")
                    or other_params.get("exploration_arm")
                    or ""
                )
                if other_arm_id == arm_id:
                    arm_job_count += 1
            if branch_id:
                branch_matches = [item for item in matches if str(item.get("branch_id") or "") == branch_id]
                if branch_matches:
                    matches = branch_matches
                elif arm_job_count != 1:
                    # Legacy arm-level outcomes are safe only when the arm maps
                    # to exactly one job.  Never attribute one aggregate result
                    # to multiple branch-specific parameter vectors.
                    continue
            elif arm_job_count != 1:
                # Branchless legacy rows are equally ambiguous when one arm
                # name describes more than one parameter vector.
                continue
            if len(matches) != 1:
                continue
            outcome = matches[0]
            status = str(outcome.get("status") or "").strip().lower()
            requested = int(outcome.get("requested_budget") or 0)
            completed = int(outcome.get("completed_budget") or 0)
            completed_statuses = {"closed", "complete", "completed", "success"}
            if status:
                if status not in completed_statuses:
                    continue
            elif not (requested > 0 and completed >= requested):
                # Missing status is accepted only with explicit, complete
                # requested/completed accounting.  Trials alone do not prove
                # that execution reached a scientific terminal state.
                continue
            if requested > 0 and completed < requested:
                continue
            if list(outcome.get("confounders") or ()):
                continue
            trials = int(outcome.get("trials") or 0)
            successes = int(outcome.get("successes") or 0)
            if trials <= 0 or successes < 0 or successes > trials:
                continue

            state_source = params.get("final_parameter_state")
            state = as_mapping(state_source) if isinstance(state_source, Mapping) else params
            if any(key not in state or state.get(key) in (None, "") for key in sampler_keys):
                continue
            try:
                candidate = ParameterCandidate({key: float(state[key]) for key in sampler_keys})
            except (TypeError, ValueError):
                continue
            if candidate not in catalog:
                continue

            is_control = bool(outcome.get("is_baseline")) or arm_id in {"baseline", "baseline_hold"}
            if (
                required_catalog_digest is not None
                and not is_control
                and str(params.get("parameter_catalog_digest") or "") != str(required_catalog_digest)
            ):
                continue
            intent = as_mapping(params.get("strategy_intent"))
            is_sampler = (
                bool(params.get("random_sampler_fallback"))
                or str(params.get("exploration_arm") or "") == "sampler_explore"
                or str(intent.get("kind") or "") == "sampling"
                or arm_id.startswith("sampler_")
            )
            if not (is_control or is_sampler):
                continue
            evidence_key = (round_id, arm_id, branch_id, candidate.key)
            if evidence_key in seen:
                continue
            seen.add(evidence_key)
            cost = float(completed or requested or trials)
            pending.append(JointParameterEvidence(
                candidate=candidate,
                successes=successes,
                trials=trials,
                replicate_id=f"R{round_id}:{arm_id}:{branch_id or candidate.key}",
                is_control=is_control,
                cost=cost,
            ))

        has_control = any(item.is_control for item in pending)
        # A control-only round contains no candidate signal and must not switch
        # the fallback away from its backward-compatible seeded shuffle.
        if has_control and not any(not item.is_control for item in pending):
            continue
        comparison_group = f"round:{round_id}" if has_control else ""
        for item in pending:
            extracted.append((round_id, JointParameterEvidence(
                candidate=item.candidate,
                successes=item.successes,
                trials=item.trials,
                replicate_id=item.replicate_id,
                comparison_group=comparison_group,
                is_control=item.is_control,
                cost=item.cost,
            )))

    extracted.sort(key=lambda item: (item[0], item[1].replicate_id, item[1].candidate.key))
    return tuple(item for _, item in extracted)


def _coerce_joint_evidence(value: Any) -> JointParameterEvidence:
    if isinstance(value, JointParameterEvidence):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("joint evidence must contain JointParameterEvidence values or mappings")
    candidate_value = value.get("candidate", value.get("parameter_state", value.get("values")))
    candidate = candidate_value if isinstance(candidate_value, ParameterCandidate) else ParameterCandidate(candidate_value or {})
    return JointParameterEvidence(
        candidate=candidate,
        successes=int(value.get("successes") or 0),
        trials=int(value.get("trials") or 0),
        replicate_id=str(value.get("replicate_id") or value.get("evidence_id") or ""),
        comparison_group=str(value.get("comparison_group") or ""),
        is_control=bool(value.get("is_control")),
        cost=float(value.get("cost") or 1.0),
    )


def _wilson_bounds(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    rate = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (rate + z2 / (2.0 * trials)) / denominator
    margin = z * math.sqrt((rate * (1.0 - rate) / trials) + z2 / (4.0 * trials * trials)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def joint_candidate_scores(
    candidates: Sequence[ParameterCandidate], evidence: Sequence[Any], *,
    policy: Optional[JointSelectionPolicy] = None,
) -> Dict[ParameterCandidate, JointCandidateScore]:
    """Aggregate evidence by full vector; never mix marginal per-axis results."""

    rules = policy or JointSelectionPolicy()
    unique_rows: Dict[str, JointParameterEvidence] = {}
    for index, raw_item in enumerate(evidence or ()):
        item = _coerce_joint_evidence(raw_item)
        identity = item.replicate_id or f"anonymous:{index}"
        existing = unique_rows.get(identity)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting joint evidence for replicate_id={identity}")
        unique_rows[identity] = item
    rows = tuple(unique_rows.values())
    controls_by_group: Dict[str, list[JointParameterEvidence]] = {}
    for item in rows:
        if item.is_control and item.comparison_group and item.trials > 0:
            controls_by_group.setdefault(item.comparison_group, []).append(item)
    result: Dict[ParameterCandidate, JointCandidateScore] = {}
    for candidate in candidates:
        candidate_rows = [item for item in rows if item.candidate == candidate and item.trials > 0]
        successes = sum(item.successes for item in candidate_rows)
        trials = sum(item.trials for item in candidate_rows)
        replicate_ids = {item.replicate_id or f"row:{index}" for index, item in enumerate(candidate_rows)}
        matched_rows = [
            item for item in candidate_rows
            if not item.is_control and item.comparison_group in controls_by_group
        ]
        matched_successes = sum(item.successes for item in matched_rows)
        matched_trials = sum(item.trials for item in matched_rows)
        matched_replicate_ids = {
            item.replicate_id or f"matched:{index}"
            for index, item in enumerate(matched_rows)
        }
        matched_groups = sorted({item.comparison_group for item in matched_rows})
        control_rows = [item for group in matched_groups for item in controls_by_group[group]]
        control_successes = sum(item.successes for item in control_rows)
        control_trials = sum(item.trials for item in control_rows)
        posterior_alpha = 1.0 + successes
        posterior_beta = 1.0 + max(0, trials - successes)
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        posterior_uncertainty = math.sqrt(
            posterior_alpha * posterior_beta
            / ((posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1.0))
        )
        matched_posterior_alpha = 1.0 + matched_successes
        matched_posterior_beta = 1.0 + max(0, matched_trials - matched_successes)
        matched_posterior_mean = matched_posterior_alpha / (
            matched_posterior_alpha + matched_posterior_beta
        )
        matched_posterior_uncertainty = math.sqrt(
            matched_posterior_alpha * matched_posterior_beta
            / (
                (matched_posterior_alpha + matched_posterior_beta) ** 2
                * (matched_posterior_alpha + matched_posterior_beta + 1.0)
            )
        )
        control_mean = (1.0 + control_successes) / (2.0 + control_trials) if control_trials else 0.5
        candidate_low, _ = _wilson_bounds(matched_successes, matched_trials)
        _, control_high = _wilson_bounds(control_successes, control_trials)
        conservative_effect = candidate_low - control_high if matched_groups else -1.0
        estimated_cost = (
            sum(item.cost for item in candidate_rows) / len(candidate_rows)
            if candidate_rows else rules.default_candidate_cost
        )
        supported = (
            len(matched_replicate_ids) >= rules.minimum_replicates
            and matched_trials >= rules.minimum_trials
            and len(matched_groups) >= rules.minimum_matched_controls
            and conservative_effect >= rules.minimum_conservative_effect
        )
        result[candidate] = JointCandidateScore(
            candidate=candidate,
            successes=successes,
            trials=trials,
            replicates=len(replicate_ids),
            matched_controls=len(matched_groups),
            posterior_mean=posterior_mean,
            posterior_uncertainty=posterior_uncertainty,
            matched_successes=matched_successes,
            matched_trials=matched_trials,
            matched_replicates=len(matched_replicate_ids),
            matched_posterior_mean=matched_posterior_mean,
            matched_posterior_uncertainty=matched_posterior_uncertainty,
            matched_control_mean=control_mean,
            conservative_effect=conservative_effect,
            estimated_cost=estimated_cost,
            supported=supported,
        )
    return result


def _joint_candidate_distance(
    left: ParameterCandidate, right: ParameterCandidate, spec: ParameterDecisionSpec,
) -> float:
    axes = spec.active_axes()
    left_values = left.as_dict()
    right_values = right.as_dict()
    distances = []
    for key, values in axes.items():
        if len(values) <= 1:
            distances.append(0.0)
            continue
        positions = {value: index for index, value in enumerate(values)}
        distances.append(abs(positions[left_values[key]] - positions[right_values[key]]) / (len(values) - 1))
    return sum(distances) / len(distances) if distances else 0.0


def select_joint_parameter_states(
    spec: ParameterDecisionSpec, candidates: Sequence[ParameterCandidate], *,
    evidence: Sequence[Any], count: int, seed: int = 0,
    policy: Optional[JointSelectionPolicy] = None,
    selected: Sequence[ParameterCandidate] = (),
) -> Tuple[ParameterCandidate, ...]:
    """Greedily select a cost-aware, diverse exploitation/exploration batch."""

    rules = policy or JointSelectionPolicy()
    anchors = list(dict.fromkeys(selected or ()))
    anchor_set = set(anchors)
    remaining = [
        item for item in dict.fromkeys(candidates)
        if item not in anchor_set
    ]
    if not remaining or count <= 0:
        return ()
    scores = joint_candidate_scores(remaining, evidence, policy=rules)
    chosen: list[ParameterCandidate] = []
    supported_count = sum(1 for item in scores.values() if item.supported)
    exploit_target = min(supported_count, int(math.ceil(max(0, int(count)) * rules.exploitation_fraction)))
    spent = 0.0

    def tie_breaker(candidate: ParameterCandidate) -> int:
        import hashlib
        return int(hashlib.sha256(f"{int(seed)}|{candidate.key}".encode("utf-8")).hexdigest()[:16], 16)

    while remaining and len(chosen) < max(0, int(count)):
        want_exploit = sum(1 for item in chosen if scores[item].supported) < exploit_target
        pool = [item for item in remaining if scores[item].supported] if want_exploit else list(remaining)
        if not pool:
            pool = list(remaining)
        affordable = [
            item for item in pool
            if rules.max_total_cost is None or spent + scores[item].estimated_cost <= rules.max_total_cost + 1e-12
        ]
        if not affordable and want_exploit:
            # An unaffordable supported candidate must not suppress a legal,
            # lower-cost exploratory fill for the remaining batch budget.
            pool = list(remaining)
            affordable = [
                item for item in pool
                if rules.max_total_cost is None
                or spent + scores[item].estimated_cost <= rules.max_total_cost + 1e-12
            ]
        if not affordable:
            break

        def rank(candidate: ParameterCandidate) -> Tuple[float, float, float, int, str]:
            item = scores[candidate]
            uncertainty = min(1.0, 4.0 * item.posterior_uncertainty)
            novelty = 1.0 / math.sqrt(item.replicates + 1.0)
            cost_efficiency = rules.default_candidate_cost / (rules.default_candidate_cost + item.estimated_cost)
            diversity = min(
                (_joint_candidate_distance(candidate, anchor, spec) for anchor in anchors + chosen),
                default=1.0,
            )
            if want_exploit and item.supported:
                uncertainty = min(1.0, 4.0 * item.matched_posterior_uncertainty)
                effect = item.matched_posterior_mean - item.matched_control_mean
                base = effect + 0.25 * item.conservative_effect + rules.uncertainty_weight * uncertainty
            else:
                base = (
                    rules.uncertainty_weight * uncertainty
                    + rules.novelty_weight * novelty
                    + 0.10 * item.posterior_mean
                )
            value = base + rules.diversity_weight * diversity + rules.cost_weight * cost_efficiency
            return value, diversity, -item.estimated_cost, tie_breaker(candidate), candidate.key

        selected_candidate = max(affordable, key=rank)
        chosen.append(selected_candidate)
        spent += scores[selected_candidate].estimated_cost
        remaining.remove(selected_candidate)
    return tuple(chosen)


def deterministic_sampler_states(
    spec: ParameterDecisionSpec, *, current: Optional[Mapping[str, float]] = None,
    count: int = 1, seed: int = 0, bounds: Optional[Mapping[str, Mapping[str, float]]] = None,
    evidence: Optional[Sequence[Any]] = None, policy: Optional[JointSelectionPolicy] = None,
    selected: Sequence[ParameterCandidate] = (),
) -> Tuple[ParameterCandidate, ...]:
    """Draw distinct legal joint states deterministically, excluding current.

    With no evidence this preserves the original seeded-shuffle behavior exactly.
    Once evidence is supplied, selection becomes cost-aware and treats the full
    parameter vector as the experimental unit.
    """
    current_values = dict(current or {})
    limits = dict(bounds or {})
    eligible = []
    for candidate in spec.catalog:
        values = candidate.as_dict()
        if current_values and all(
            key in current_values and float(current_values[key]) == value for key, value in values.items()
        ):
            continue
        if any(
            value not in filter_parameter_candidates(
                parameter_axis(spec, key), current=current_values.get(key), bounds=limits.get(key),
            )
            for key, value in values.items()
        ):
            continue
        eligible.append(candidate)
    evidence_rows = tuple(evidence or ())
    if evidence_rows:
        return select_joint_parameter_states(
            spec, eligible, evidence=evidence_rows, count=count, seed=seed,
            policy=policy, selected=selected,
        )
    import random
    rng = random.Random(int(seed))
    rng.shuffle(eligible)
    return tuple(eligible[:max(0, int(count))])


def parameter_catalog_digest(spec: ParameterDecisionSpec) -> str:
    """Stable digest of axes and thresholds without importing resume helpers."""
    import hashlib, json
    payload = {
        **{key: list(values) for key, values in spec.active_axes().items()},
        "thresholds": {"top": spec.top_probability_threshold, "margin": spec.margin_threshold, "entropy": spec.max_normalized_entropy},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
