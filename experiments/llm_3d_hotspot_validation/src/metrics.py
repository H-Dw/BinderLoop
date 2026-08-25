"""Dependency-free metrics for blinded residue-hotspot validation.

The module deliberately works with canonical residue identifiers and generic
coordinates.  It contains no benchmark targets, labels, or target-specific
assumptions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations, product
import math
import random
import re
from statistics import fmean
from typing import Callable, Iterable, Mapping, Sequence


_TOKEN_RE = re.compile(
    r"^(?P<chain>[A-Za-z0-9_.-]+):"
    r"(?P<number>(?:0|-[1-9][0-9]*|[1-9][0-9]*))"
    r"(?P<icode>[A-Z]?)$"
)


class PredictionSchemaError(ValueError):
    """Raised when a prediction does not satisfy the blinded schema."""


@dataclass(frozen=True)
class PredictionSet:
    primary: tuple[str, str, str]
    alternates: tuple[str, str, str]

    @property
    def ranked(self) -> tuple[str, ...]:
        return self.primary + self.alternates


@dataclass(frozen=True)
class PredictionValidation:
    compliant: bool
    recognized: bool
    recognition_rate: float
    errors: tuple[str, ...]
    primary: tuple[str, ...]
    alternates: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentPrediction:
    """A validated, metadata-bearing prediction record for experiment tables."""

    case: str
    condition: str
    replicate: int
    primary_hotspots: tuple[str, str, str]
    alternate_hotspots: tuple[str, str, str]
    recognition_status: bool
    compliance: bool

    @property
    def prediction(self) -> PredictionSet:
        return PredictionSet(self.primary_hotspots, self.alternate_hotspots)


def parse_residue_token(token: str) -> tuple[str, int, str]:
    """Parse ``CHAIN:NUMBER[ICODE]`` and reject non-canonical spellings."""

    if not isinstance(token, str):
        raise ValueError("residue token must be a string")
    match = _TOKEN_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"invalid canonical residue token: {token!r}")
    return match.group("chain"), int(match.group("number")), match.group("icode")


def canonical_residue_token(chain: str, number: int, insertion_code: str = "") -> str:
    """Construct a canonical residue token and validate each component."""

    if not isinstance(number, int) or isinstance(number, bool):
        raise ValueError("residue number must be an integer")
    token = f"{chain}:{number}{insertion_code}"
    parse_residue_token(token)
    return token


def is_canonical_residue_token(token: object) -> bool:
    try:
        parse_residue_token(token)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


def assess_prediction_schema(
    payload: object,
    recognized_tokens: Iterable[str] | None = None,
    *,
    require_recognized: bool = True,
) -> PredictionValidation:
    """Return recognition and schema-compliance status without raising."""

    errors: list[str] = []
    primary: tuple[str, ...] = ()
    alternates: tuple[str, ...] = ()
    if not isinstance(payload, Mapping):
        errors.append("prediction must be a mapping")
    else:
        keys = set(payload)
        required = {"primary", "alternates"}
        if keys != required:
            errors.append("prediction keys must be exactly 'primary' and 'alternates'")
        for name in ("primary", "alternates"):
            value = payload.get(name)
            if not isinstance(value, (list, tuple)):
                errors.append(f"{name} must be an ordered list or tuple")
                continue
            if len(value) != 3:
                errors.append(f"{name} must contain exactly 3 residues")
            if not all(isinstance(item, str) for item in value):
                errors.append(f"{name} entries must all be strings")
                continue
            if name == "primary":
                primary = tuple(value)
            else:
                alternates = tuple(value)

    tokens = primary + alternates
    valid_tokens: list[str] = []
    for index, token in enumerate(tokens, start=1):
        if is_canonical_residue_token(token):
            valid_tokens.append(token)
        else:
            errors.append(f"rank {index} has invalid residue token {token!r}")

    if len(alternates) == 3 and len(set(alternates)) != 3:
        errors.append("alternates must contain 3 unique residues")
    if len(tokens) == 6 and len(set(tokens)) != 6:
        errors.append("all 6 ranked residues must be unique")

    recognized_set: set[str] | None = None
    if recognized_tokens is not None:
        recognized_set = set(recognized_tokens)
        invalid_universe = sorted(token for token in recognized_set if not is_canonical_residue_token(token))
        if invalid_universe:
            errors.append("recognized token universe contains non-canonical entries")

    if recognized_set is None:
        recognized_count = len(valid_tokens)
    else:
        recognized_count = sum(token in recognized_set for token in valid_tokens)
        unknown = sorted(set(valid_tokens) - recognized_set)
        if require_recognized and unknown:
            errors.append("unrecognized residue tokens: " + ", ".join(unknown))
    recognition_rate = recognized_count / len(tokens) if tokens else 0.0
    recognized = len(tokens) == 6 and recognized_count == 6
    return PredictionValidation(
        compliant=not errors,
        recognized=recognized,
        recognition_rate=recognition_rate,
        errors=tuple(errors),
        primary=primary,
        alternates=alternates,
    )


def validate_prediction_schema(
    payload: object,
    recognized_tokens: Iterable[str] | None = None,
    *,
    require_recognized: bool = True,
) -> PredictionSet:
    """Validate and return a typed six-residue prediction."""

    report = assess_prediction_schema(
        payload,
        recognized_tokens,
        require_recognized=require_recognized,
    )
    if not report.compliant:
        raise PredictionSchemaError("; ".join(report.errors))
    return PredictionSet(
        primary=(report.primary[0], report.primary[1], report.primary[2]),
        alternates=(report.alternates[0], report.alternates[1], report.alternates[2]),
    )


def validate_experiment_prediction_schema(
    payload: object,
    recognized_tokens: Iterable[str] | None = None,
    *,
    require_recognized: bool = True,
) -> ExperimentPrediction:
    """Validate the full experiment record, including explicit status booleans.

    ``False`` is a valid value for both status fields; the fields describe the
    recorded upstream outcome and are not inferred from Python truthiness.
    """

    if not isinstance(payload, Mapping):
        raise PredictionSchemaError("experiment prediction must be a mapping")
    required = {
        "case",
        "condition",
        "replicate",
        "primary_hotspots",
        "alternate_hotspots",
        "recognition_status",
        "compliance",
    }
    if set(payload) != required:
        raise PredictionSchemaError(
            "experiment prediction keys must be exactly: " + ", ".join(sorted(required))
        )
    case = payload["case"]
    condition = payload["condition"]
    replicate = payload["replicate"]
    recognition_status = payload["recognition_status"]
    compliance = payload["compliance"]
    errors: list[str] = []
    if not isinstance(case, str) or not case.strip():
        errors.append("case must be a non-empty string")
    if not isinstance(condition, str) or not condition.strip():
        errors.append("condition must be a non-empty string")
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate < 1:
        errors.append("replicate must be a positive integer")
    if not isinstance(recognition_status, bool):
        errors.append("recognition_status must be a boolean")
    if not isinstance(compliance, bool):
        errors.append("compliance must be a boolean")
    try:
        typed = validate_prediction_schema(
            {
                "primary": payload["primary_hotspots"],
                "alternates": payload["alternate_hotspots"],
            },
            recognized_tokens,
            require_recognized=require_recognized,
        )
    except PredictionSchemaError as exc:
        errors.append(str(exc))
        typed = None
    if errors:
        raise PredictionSchemaError("; ".join(errors))
    assert typed is not None
    return ExperimentPrediction(
        case=case,
        condition=condition,
        replicate=replicate,
        primary_hotspots=typed.primary,
        alternate_hotspots=typed.alternates,
        recognition_status=recognition_status,
        compliance=compliance,
    )


@dataclass(frozen=True)
class ExactMetrics:
    k: int
    h: int
    precision: float
    recall: float
    f1: float
    jaccard: float
    average_precision: float
    enrichment: float
    hypergeometric_p: float


@dataclass(frozen=True)
class TopMetrics:
    top3: ExactMetrics
    top6: ExactMetrics


def hypergeometric_survival(
    observed: int,
    population_size: int,
    success_states: int,
    draws: int,
) -> float:
    """Compute P[X >= observed] exactly from integer binomial coefficients."""

    values = (observed, population_size, success_states, draws)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise ValueError("hypergeometric arguments must be integers")
    if population_size < 0 or not 0 <= success_states <= population_size:
        raise ValueError("invalid population or success-state count")
    if not 0 <= draws <= population_size:
        raise ValueError("draw count must be within the population")
    lower = max(0, draws - (population_size - success_states))
    upper = min(draws, success_states)
    if observed <= lower:
        return 1.0
    if observed > upper:
        return 0.0
    denominator = math.comb(population_size, draws)
    terms = (
        math.comb(success_states, hits)
        * math.comb(population_size - success_states, draws - hits)
        / denominator
        for hits in range(observed, upper + 1)
    )
    return min(1.0, math.fsum(terms))


def exact_ranked_metrics(
    ranked_prediction: Sequence[str],
    reference: Iterable[str],
    universe_size: int,
    *,
    k: int | None = None,
) -> ExactMetrics:
    """Calculate exact set/ranking metrics at ``k``."""

    ranked = tuple(ranked_prediction)
    cutoff = len(ranked) if k is None else k
    if not isinstance(cutoff, int) or cutoff <= 0 or cutoff > len(ranked):
        raise ValueError("k must be between 1 and the prediction length")
    selected = ranked[:cutoff]
    if len(set(selected)) != len(selected):
        raise ValueError("ranked predictions must be unique")
    for token in selected:
        parse_residue_token(token)
    truth = set(reference)
    for token in truth:
        parse_residue_token(token)
    if not isinstance(universe_size, int) or universe_size < len(truth | set(selected)):
        raise ValueError("universe_size is smaller than the observed token universe")

    hit_flags = [token in truth for token in selected]
    h = sum(hit_flags)
    precision = h / cutoff
    recall = h / len(truth) if truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    union_size = len(set(selected) | truth)
    jaccard = h / union_size if union_size else 1.0
    ap_denominator = min(cutoff, len(truth))
    if ap_denominator:
        cumulative = 0
        ap_terms: list[float] = []
        for rank, is_hit in enumerate(hit_flags, start=1):
            cumulative += int(is_hit)
            if is_hit:
                ap_terms.append(cumulative / rank)
        average_precision = math.fsum(ap_terms) / ap_denominator
    else:
        average_precision = 0.0
    prevalence = len(truth) / universe_size if universe_size else 0.0
    enrichment = precision / prevalence if prevalence else 0.0
    p_value = hypergeometric_survival(h, universe_size, len(truth), cutoff)
    return ExactMetrics(
        k=cutoff,
        h=h,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        average_precision=average_precision,
        enrichment=enrichment,
        hypergeometric_p=p_value,
    )


def top3_top6_metrics(
    prediction: PredictionSet | Mapping[str, object],
    reference: Iterable[str],
    universe_size: int,
) -> TopMetrics:
    typed = prediction if isinstance(prediction, PredictionSet) else validate_prediction_schema(prediction)
    return TopMetrics(
        top3=exact_ranked_metrics(typed.ranked, reference, universe_size, k=3),
        top6=exact_ranked_metrics(typed.ranked, reference, universe_size, k=6),
    )


def consensus_prediction(runs: Sequence[PredictionSet | Mapping[str, object]]) -> PredictionSet:
    """Aggregate exactly three runs by frequency, reciprocal rank, then token."""

    if len(runs) != 3:
        raise ValueError("consensus requires exactly 3 runs")
    typed = [run if isinstance(run, PredictionSet) else validate_prediction_schema(run) for run in runs]
    frequency: Counter[str] = Counter()
    reciprocal_rank: defaultdict[str, float] = defaultdict(float)
    for run in typed:
        for rank, token in enumerate(run.ranked, start=1):
            frequency[token] += 1
            reciprocal_rank[token] += 1.0 / rank
    ordered = sorted(frequency, key=lambda token: (-frequency[token], -reciprocal_rank[token], token))
    if len(ordered) < 6:
        raise ValueError("the three runs must contain at least 6 distinct residues")
    return PredictionSet(
        primary=(ordered[0], ordered[1], ordered[2]),
        alternates=(ordered[3], ordered[4], ordered[5]),
    )


@dataclass(frozen=True)
class Atom:
    element: str
    x: float
    y: float
    z: float


AtomLike = Atom | Sequence[object]
ResidueAtoms = Mapping[str, Iterable[AtomLike]]


def _heavy_coordinates(atoms: Iterable[AtomLike]) -> tuple[tuple[float, float, float], ...]:
    coordinates: list[tuple[float, float, float]] = []
    for atom in atoms:
        if isinstance(atom, Atom):
            element = atom.element
            xyz = (atom.x, atom.y, atom.z)
        else:
            fields = tuple(atom)
            if len(fields) == 3:
                element = "C"
                xyz = fields
            elif len(fields) == 4 and isinstance(fields[0], str):
                element = fields[0]
                xyz = fields[1:]
            else:
                raise ValueError("atoms must be Atom, (x,y,z), or (element,x,y,z)")
        if element.strip().upper() in {"H", "D", "T"}:
            continue
        point = tuple(float(value) for value in xyz)
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            raise ValueError("atom coordinates must be three finite values")
        coordinates.append((point[0], point[1], point[2]))
    return tuple(coordinates)


def heavy_atom_residue_distance(atoms_a: Iterable[AtomLike], atoms_b: Iterable[AtomLike]) -> float:
    """Minimum Euclidean distance between non-hydrogen atoms in two residues."""

    coords_a = _heavy_coordinates(atoms_a)
    coords_b = _heavy_coordinates(atoms_b)
    if not coords_a or not coords_b:
        return math.inf
    squared = min(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        for a in coords_a
        for b in coords_b
    )
    return math.sqrt(squared)


def _directed_nearest(
    source: Sequence[str],
    target: Sequence[str],
    residue_atoms: ResidueAtoms,
) -> tuple[float, ...]:
    if not source:
        return ()
    if not target:
        return tuple(math.inf for _ in source)
    return tuple(
        min(
            heavy_atom_residue_distance(residue_atoms.get(left, ()), residue_atoms.get(right, ()))
            for right in target
        )
        for left in source
    )


def _nearest_rank_percentile(values: Sequence[float], proportion: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = max(0, math.ceil(proportion * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True)
class PocketDistanceMetrics:
    minimum: float
    chamfer: float
    d90: float
    hausdorff: float


def pocket_distance_metrics(
    predicted: Iterable[str],
    reference: Iterable[str],
    residue_atoms: ResidueAtoms,
) -> PocketDistanceMetrics:
    """Symmetric residue-pocket distances based on heavy-atom minima."""

    left = tuple(sorted(set(predicted)))
    right = tuple(sorted(set(reference)))
    for token in left + right:
        parse_residue_token(token)
    if not left or not right:
        return PocketDistanceMetrics(math.inf, math.inf, math.inf, math.inf)
    left_nearest = _directed_nearest(left, right, residue_atoms)
    right_nearest = _directed_nearest(right, left, residue_atoms)
    all_nearest = left_nearest + right_nearest
    return PocketDistanceMetrics(
        minimum=min(all_nearest),
        chamfer=(fmean(left_nearest) + fmean(right_nearest)) / 2.0,
        d90=_nearest_rank_percentile(all_nearest, 0.90),
        hausdorff=max(all_nearest),
    )


@dataclass(frozen=True)
class TolerantMetrics:
    threshold: float
    predicted_hits: int
    reference_hits: int
    precision: float
    recall: float
    f1: float


def distance_tolerant_metrics(
    predicted: Iterable[str],
    reference: Iterable[str],
    residue_atoms: ResidueAtoms,
    threshold: float,
) -> TolerantMetrics:
    """Bidirectional thresholded overlap; matching is not one-to-one."""

    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")
    left = tuple(sorted(set(predicted)))
    right = tuple(sorted(set(reference)))
    left_nearest = _directed_nearest(left, right, residue_atoms)
    right_nearest = _directed_nearest(right, left, residue_atoms)
    predicted_hits = sum(distance <= threshold for distance in left_nearest)
    reference_hits = sum(distance <= threshold for distance in right_nearest)
    precision = predicted_hits / len(left) if left else 0.0
    recall = reference_hits / len(right) if right else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TolerantMetrics(
        threshold=threshold,
        predicted_hits=predicted_hits,
        reference_hits=reference_hits,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def standard_distance_tolerances(
    predicted: Iterable[str],
    reference: Iterable[str],
    residue_atoms: ResidueAtoms,
) -> dict[float, TolerantMetrics]:
    return {
        threshold: distance_tolerant_metrics(predicted, reference, residue_atoms, threshold)
        for threshold in (4.0, 6.0, 8.0)
    }


def expand_residue_pocket(
    seeds: Iterable[str],
    residue_atoms: ResidueAtoms,
    radius: float = 6.0,
) -> frozenset[str]:
    """Expand seed residues to all residues within a heavy-atom radius (H6)."""

    if radius < 0 or not math.isfinite(radius):
        raise ValueError("radius must be finite and non-negative")
    seed_set = frozenset(seeds)
    missing = sorted(seed_set - set(residue_atoms))
    if missing:
        raise KeyError("missing seed coordinates: " + ", ".join(missing))
    for token in seed_set:
        parse_residue_token(token)
    expanded: set[str] = set(seed_set)
    for candidate, candidate_atoms in residue_atoms.items():
        parse_residue_token(candidate)
        if any(
            heavy_atom_residue_distance(candidate_atoms, residue_atoms[seed]) <= radius
            for seed in seed_set
        ):
            expanded.add(candidate)
    return frozenset(expanded)


@dataclass(frozen=True)
class PocketOverlap:
    predicted_h6: frozenset[str]
    reference_h6: frozenset[str]
    jaccard: float
    dice: float
    sasa_weighted_jaccard: float
    sasa_weighted_dice: float


def h6_pocket_overlap(
    predicted_seeds: Iterable[str],
    reference_seeds: Iterable[str],
    residue_atoms: ResidueAtoms,
    sasa: Mapping[str, float],
    *,
    radius: float = 6.0,
) -> PocketOverlap:
    """Compare H6-expanded pockets with unweighted and SASA-weighted scores."""

    predicted = expand_residue_pocket(predicted_seeds, residue_atoms, radius)
    reference = expand_residue_pocket(reference_seeds, residue_atoms, radius)
    union = predicted | reference
    intersection = predicted & reference
    missing = sorted(union - set(sasa))
    if missing:
        raise KeyError("missing SASA weights: " + ", ".join(missing))
    weights: dict[str, float] = {}
    for token in union:
        value = float(sasa[token])
        if value < 0 or not math.isfinite(value):
            raise ValueError("SASA weights must be finite and non-negative")
        weights[token] = value
    jaccard = len(intersection) / len(union) if union else 1.0
    dice_denominator = len(predicted) + len(reference)
    dice = 2.0 * len(intersection) / dice_denominator if dice_denominator else 1.0
    intersection_weight = math.fsum(weights[token] for token in intersection)
    predicted_weight = math.fsum(weights[token] for token in predicted)
    reference_weight = math.fsum(weights[token] for token in reference)
    union_weight = predicted_weight + reference_weight - intersection_weight
    weighted_jaccard = intersection_weight / union_weight if union_weight else 1.0
    weighted_dice_denominator = predicted_weight + reference_weight
    weighted_dice = (
        2.0 * intersection_weight / weighted_dice_denominator
        if weighted_dice_denominator
        else 1.0
    )
    return PocketOverlap(
        predicted_h6=predicted,
        reference_h6=reference,
        jaccard=jaccard,
        dice=dice,
        sasa_weighted_jaccard=weighted_jaccard,
        sasa_weighted_dice=weighted_dice,
    )


@dataclass(frozen=True)
class ChainPermutationResult:
    remapped: tuple[str, ...]
    chain_mapping: tuple[tuple[str, str], ...]
    score: tuple[float, ...]


ResidueCorrespondence = Mapping[tuple[str, str], str]


def _remap_token(
    token: str,
    chain_mapping: Mapping[str, str],
    residue_correspondence: ResidueCorrespondence | None = None,
) -> str | None:
    chain, number, insertion = parse_residue_token(token)
    destination = chain_mapping.get(chain, chain)
    if destination == chain:
        return token
    if residue_correspondence is not None:
        return residue_correspondence.get((token, destination))
    return canonical_residue_token(destination, number, insertion)


def _all_chain_mappings(equivalent_chain_groups: Sequence[Sequence[str]]) -> tuple[dict[str, str], ...]:
    seen: set[str] = set()
    choices: list[list[dict[str, str]]] = []
    for raw_group in equivalent_chain_groups:
        group = tuple(raw_group)
        if not group or len(set(group)) != len(group):
            raise ValueError("equivalent chain groups must be non-empty and unique")
        for chain in group:
            if not isinstance(chain, str) or not chain or ":" in chain:
                raise ValueError("invalid chain identifier")
            if chain in seen:
                raise ValueError("equivalent chain groups must be disjoint")
            seen.add(chain)
        choices.append(
            [dict(zip(group, destination)) for destination in permutations(group)]
        )
    if not choices:
        return ({},)
    return tuple(
        {source: destination for mapping in combination for source, destination in mapping.items()}
        for combination in product(*choices)
    )


def global_chain_permutation(
    ranked_prediction: Sequence[str],
    reference: Iterable[str],
    equivalent_chain_groups: Sequence[Sequence[str]],
    score_fn: Callable[[tuple[str, ...]], float | Sequence[float]] | None = None,
    residue_correspondence: ResidueCorrespondence | None = None,
) -> ChainPermutationResult:
    """Choose one global chain permutation, never residue-wise remappings.

    ``residue_correspondence[(source_token, destination_chain)]`` supplies the
    destination token when equivalent chains use different local ordinals or
    have different cropped residue sets.  A non-identity permutation is invalid
    if any selected token lacks a correspondence.  Identity never needs a
    correspondence and therefore always remains a candidate.
    """

    ranked = tuple(ranked_prediction)
    truth = frozenset(reference)
    for token in ranked + tuple(truth):
        parse_residue_token(token)
    if residue_correspondence is not None:
        for key, destination_token in residue_correspondence.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not isinstance(key[1], str)
            ):
                raise ValueError(
                    "residue correspondence keys must be (source_token, destination_chain)"
                )
            parse_residue_token(key[0])
            destination_chain, _, _ = parse_residue_token(destination_token)
            if destination_chain != key[1]:
                raise ValueError(
                    "residue correspondence destination token has the wrong chain"
                )

    def default_score(remapped: tuple[str, ...]) -> tuple[float, ...]:
        return (float(len(set(remapped) & truth)),)

    scorer = score_fn or default_score
    candidates: list[ChainPermutationResult] = []
    for mapping in _all_chain_mappings(equivalent_chain_groups):
        candidate = tuple(
            _remap_token(token, mapping, residue_correspondence) for token in ranked
        )
        if any(token is None for token in candidate):
            continue
        remapped = tuple(token for token in candidate if token is not None)
        raw_score = scorer(remapped)
        score = (float(raw_score),) if isinstance(raw_score, (int, float)) else tuple(float(x) for x in raw_score)
        candidates.append(
            ChainPermutationResult(
                remapped=remapped,
                chain_mapping=tuple(sorted(mapping.items())),
                score=score,
            )
        )
    if not candidates:  # Defensive: identity should always survive validation.
        raise ValueError("no valid global chain permutation remains")
    candidates.sort(key=lambda item: (tuple(-value for value in item.score), item.remapped, item.chain_mapping))
    return candidates[0]


@dataclass(frozen=True)
class SymmetryMetrics:
    strict: TopMetrics
    symmetry_adjusted: TopMetrics
    chain_mapping: tuple[tuple[str, str], ...]
    remapped_prediction: PredictionSet


def symmetry_adjusted_top_metrics(
    prediction: PredictionSet | Mapping[str, object],
    reference: Iterable[str],
    universe_size: int,
    equivalent_chain_groups: Sequence[Sequence[str]],
    residue_correspondence: ResidueCorrespondence | None = None,
) -> SymmetryMetrics:
    """Report strict metrics and the best globally symmetry-adjusted metrics."""

    typed = prediction if isinstance(prediction, PredictionSet) else validate_prediction_schema(prediction)
    truth = frozenset(reference)
    strict = top3_top6_metrics(typed, truth, universe_size)

    def score(remapped: tuple[str, ...]) -> tuple[float, float]:
        return (
            float(len(set(remapped[:3]) & truth)),
            float(len(set(remapped[:6]) & truth)),
        )

    optimized = global_chain_permutation(
        typed.ranked,
        truth,
        equivalent_chain_groups,
        score,
        residue_correspondence,
    )
    adjusted_prediction = PredictionSet(
        primary=(optimized.remapped[0], optimized.remapped[1], optimized.remapped[2]),
        alternates=(optimized.remapped[3], optimized.remapped[4], optimized.remapped[5]),
    )
    adjusted = top3_top6_metrics(adjusted_prediction, truth, universe_size)
    return SymmetryMetrics(strict, adjusted, optimized.chain_mapping, adjusted_prediction)


def empirical_rsasa_quintiles(
    universe: Iterable[str],
    rsasa: Mapping[str, float],
) -> dict[str, int]:
    """Assign deterministic empirical quintiles, breaking ties by residue token."""

    tokens = sorted(set(universe))
    if not tokens:
        return {}
    missing = sorted(set(tokens) - set(rsasa))
    if missing:
        raise KeyError("missing rSASA values: " + ", ".join(missing))
    ordered: list[tuple[float, str]] = []
    for token in tokens:
        parse_residue_token(token)
        value = float(rsasa[token])
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise ValueError("rSASA values must be finite and within [0, 1]")
        ordered.append((value, token))
    ordered.sort()
    count = len(ordered)
    return {token: min(4, 5 * rank // count) for rank, (_, token) in enumerate(ordered)}


def _stratum(token: str, quintiles: Mapping[str, int]) -> tuple[str, int]:
    chain, _, _ = parse_residue_token(token)
    return chain, quintiles[token]


def stratified_resamples(
    selected: Iterable[str],
    universe: Iterable[str],
    rsasa: Mapping[str, float],
    *,
    draws: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Sample sets preserving joint chain and empirical-rSASA-quintile counts."""

    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    universe_set = frozenset(universe)
    selected_set = frozenset(selected)
    if not selected_set <= universe_set:
        raise ValueError("selected residues must be a subset of the universe")
    quintiles = empirical_rsasa_quintiles(universe_set, rsasa)
    pools: defaultdict[tuple[str, int], list[str]] = defaultdict(list)
    required: Counter[tuple[str, int]] = Counter()
    for token in universe_set:
        pools[_stratum(token, quintiles)].append(token)
    for token in selected_set:
        required[_stratum(token, quintiles)] += 1
    for pool in pools.values():
        pool.sort()
    rng = random.Random(seed)
    samples: list[tuple[str, ...]] = []
    for _ in range(draws):
        sample: list[str] = []
        for stratum in sorted(required):
            sample.extend(rng.sample(pools[stratum], required[stratum]))
        samples.append(tuple(sorted(sample)))
    return tuple(samples)


@dataclass(frozen=True)
class MonteCarloResult:
    observed: float
    null_values: tuple[float, ...]
    p_greater_equal: float
    draws: int
    seed: int
    stratum_counts: tuple[tuple[tuple[str, int], int], ...]


def stratified_monte_carlo(
    selected: Iterable[str],
    universe: Iterable[str],
    rsasa: Mapping[str, float],
    statistic: Callable[[frozenset[str]], float],
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> MonteCarloResult:
    """Evaluate a statistic against a deterministic stratified null."""

    universe_set = frozenset(universe)
    selected_set = frozenset(selected)
    quintiles = empirical_rsasa_quintiles(universe_set, rsasa)
    counts = Counter(_stratum(token, quintiles) for token in selected_set)
    samples = stratified_resamples(selected_set, universe_set, rsasa, draws=draws, seed=seed)
    observed = float(statistic(selected_set))
    null_values = tuple(float(statistic(frozenset(sample))) for sample in samples)
    exceedances = sum(value >= observed for value in null_values)
    p_value = (exceedances + 1.0) / (draws + 1.0)
    return MonteCarloResult(
        observed=observed,
        null_values=null_values,
        p_greater_equal=p_value,
        draws=draws,
        seed=seed,
        stratum_counts=tuple(sorted(counts.items())),
    )


@dataclass(frozen=True)
class PairedPermutationResult:
    estimate: float
    p_two_sided: float
    n_targets: int
    exact: bool
    permutations: int
    seed: int | None


def paired_sign_flip_test(
    method_a: Sequence[float],
    method_b: Sequence[float],
    *,
    draws: int = 100_000,
    seed: int = 0,
    exact_limit: int = 20,
) -> PairedPermutationResult:
    """Target-level paired two-sided sign-flip/permutation comparison."""

    if len(method_a) != len(method_b) or not method_a:
        raise ValueError("paired methods must have the same non-zero target count")
    differences = tuple(float(a) - float(b) for a, b in zip(method_a, method_b))
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("paired values must be finite")
    observed = fmean(differences)
    threshold = abs(observed) - 1e-15
    n = len(differences)
    if n <= exact_limit:
        total = 1 << n
        exceedances = 0
        for mask in range(total):
            permuted = fmean(
                value if mask & (1 << index) else -value
                for index, value in enumerate(differences)
            )
            exceedances += abs(permuted) >= threshold
        p_value = exceedances / total
        return PairedPermutationResult(observed, p_value, n, True, total, None)
    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(draws):
        permuted = fmean(value if rng.getrandbits(1) else -value for value in differences)
        exceedances += abs(permuted) >= threshold
    p_value = (exceedances + 1.0) / (draws + 1.0)
    return PairedPermutationResult(observed, p_value, n, False, draws, seed)


def holm_adjust(
    p_values: Sequence[float] | Mapping[str, float],
) -> tuple[float, ...] | dict[str, float]:
    """Holm step-down family-wise error adjustment."""

    is_mapping = isinstance(p_values, Mapping)
    items = list(p_values.items()) if is_mapping else list(enumerate(p_values))
    for _, value in items:
        if not 0.0 <= float(value) <= 1.0 or not math.isfinite(float(value)):
            raise ValueError("p-values must be finite and within [0, 1]")
    ordered = sorted(items, key=lambda item: (float(item[1]), str(item[0])))
    adjusted: dict[object, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[label] = running
    if is_mapping:
        return {str(label): adjusted[label] for label, _ in items}
    return tuple(adjusted[index] for index in range(len(items)))


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    confidence: float
    draws: int
    seed: int


def _linear_percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_bootstrap(draws: int, confidence: float) -> None:
    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")


def target_bootstrap_ci(
    target_values: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    statistic: Callable[[Sequence[float]], float] = fmean,
) -> BootstrapCI:
    """Percentile CI from target-level resampling with replacement."""

    _validate_bootstrap(draws, confidence)
    values = tuple(float(value) for value in target_values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("target values must be non-empty and finite")
    rng = random.Random(seed)
    count = len(values)
    distribution = tuple(
        float(statistic(tuple(values[rng.randrange(count)] for _ in range(count))))
        for _ in range(draws)
    )
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        estimate=float(statistic(values)),
        lower=_linear_percentile(distribution, alpha),
        upper=_linear_percentile(distribution, 1.0 - alpha),
        confidence=confidence,
        draws=draws,
        seed=seed,
    )


def hierarchical_bootstrap_ci(
    target_observations: Mapping[str, Sequence[float]],
    *,
    draws: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Resample targets, then observations within targets, with equal target weight."""

    _validate_bootstrap(draws, confidence)
    observations: dict[str, tuple[float, ...]] = {
        target: tuple(float(value) for value in values)
        for target, values in target_observations.items()
    }
    if not observations or any(not values for values in observations.values()):
        raise ValueError("every target must contain at least one observation")
    if any(not math.isfinite(value) for values in observations.values() for value in values):
        raise ValueError("observations must be finite")
    targets = tuple(sorted(observations))
    rng = random.Random(seed)
    distribution: list[float] = []
    for _ in range(draws):
        sampled_targets = tuple(targets[rng.randrange(len(targets))] for _ in targets)
        within_means: list[float] = []
        for target in sampled_targets:
            values = observations[target]
            resampled = tuple(values[rng.randrange(len(values))] for _ in values)
            within_means.append(fmean(resampled))
        distribution.append(fmean(within_means))
    alpha = (1.0 - confidence) / 2.0
    estimate = fmean(fmean(values) for values in observations.values())
    return BootstrapCI(
        estimate=estimate,
        lower=_linear_percentile(distribution, alpha),
        upper=_linear_percentile(distribution, 1.0 - alpha),
        confidence=confidence,
        draws=draws,
        seed=seed,
    )
