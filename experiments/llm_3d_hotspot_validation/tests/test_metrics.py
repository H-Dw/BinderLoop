from collections import Counter
import math
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (  # noqa: E402
    Atom,
    PredictionSchemaError,
    assess_prediction_schema,
    consensus_prediction,
    distance_tolerant_metrics,
    empirical_rsasa_quintiles,
    exact_ranked_metrics,
    h6_pocket_overlap,
    hierarchical_bootstrap_ci,
    holm_adjust,
    hypergeometric_survival,
    paired_sign_flip_test,
    parse_residue_token,
    pocket_distance_metrics,
    standard_distance_tolerances,
    stratified_monte_carlo,
    stratified_resamples,
    symmetry_adjusted_top_metrics,
    target_bootstrap_ci,
    top3_top6_metrics,
    validate_experiment_prediction_schema,
    validate_prediction_schema,
)


def prediction(*tokens):
    assert len(tokens) == 6
    return {"primary": list(tokens[:3]), "alternates": list(tokens[3:])}


def test_known_hypergeometric_survival_cases():
    # N=10, K=3, n=2: P(X>=1)=1-C(7,2)/C(10,2), P(X>=2)=C(3,2)/C(10,2).
    assert hypergeometric_survival(0, 10, 3, 2) == 1.0
    assert hypergeometric_survival(1, 10, 3, 2) == pytest.approx(8 / 15)
    assert hypergeometric_survival(2, 10, 3, 2) == pytest.approx(1 / 15)
    assert hypergeometric_survival(3, 10, 3, 2) == 0.0


def test_exact_top3_top6_metrics_include_ranking_and_enrichment():
    ranked = tuple(f"A:{index}" for index in range(1, 7))
    truth = {"A:1", "A:3", "A:5"}
    result = top3_top6_metrics(prediction(*ranked), truth, universe_size=30)

    assert result.top3.h == 2
    assert result.top3.precision == pytest.approx(2 / 3)
    assert result.top3.recall == pytest.approx(2 / 3)
    assert result.top3.f1 == pytest.approx(2 / 3)
    assert result.top3.jaccard == pytest.approx(1 / 2)
    assert result.top3.average_precision == pytest.approx(5 / 9)
    assert result.top3.enrichment == pytest.approx(20 / 3)

    assert result.top6.h == 3
    assert result.top6.precision == pytest.approx(1 / 2)
    assert result.top6.recall == 1.0
    assert result.top6.f1 == pytest.approx(2 / 3)
    assert result.top6.jaccard == pytest.approx(1 / 2)
    assert result.top6.average_precision == pytest.approx(34 / 45)
    assert result.top6.enrichment == 5.0


@pytest.mark.parametrize(
    "bad_payload, message",
    [
        ({"primary": ["A:1", "A:2"], "alternates": ["A:3", "A:4", "A:5"]}, "exactly 3"),
        (prediction("A:1", "A:2", "A:3", "A:4", "A:4", "A:5"), "unique"),
        (prediction("A:1", "A:2", "not-a-token", "A:4", "A:5", "A:6"), "invalid"),
        ({"primary": {"A:1", "A:2", "A:3"}, "alternates": ["A:4", "A:5", "A:6"]}, "ordered"),
        ({"primary": ["A:1", "A:2", "A:3"], "alternates": ["A:4", "A:5", "A:6"], "extra": 1}, "keys"),
    ],
)
def test_schema_errors_are_explicit(bad_payload, message):
    with pytest.raises(PredictionSchemaError, match=message):
        validate_prediction_schema(bad_payload)


def test_schema_reports_recognition_and_compliance_separately():
    payload = prediction("A:1", "A:2", "A:3", "A:4", "A:5", "A:6")
    recognized = {f"A:{index}" for index in range(1, 6)}
    report = assess_prediction_schema(payload, recognized, require_recognized=True)
    assert report.recognition_rate == pytest.approx(5 / 6)
    assert not report.recognized
    assert not report.compliant
    with pytest.raises(PredictionSchemaError, match="unrecognized"):
        validate_prediction_schema(payload, recognized)


def test_full_experiment_payload_preserves_false_status_booleans():
    payload = {
        "case": "synthetic_case_01",
        "condition": "blind_control",
        "replicate": 2,
        "primary_hotspots": ["A:1", "A:2", "A:3"],
        "alternate_hotspots": ["A:4", "A:5", "A:6"],
        "recognition_status": False,
        "compliance": False,
    }
    result = validate_experiment_prediction_schema(payload)
    assert result.case == "synthetic_case_01"
    assert result.condition == "blind_control"
    assert result.replicate == 2
    assert result.recognition_status is False
    assert result.compliance is False
    assert result.prediction.ranked == tuple(f"A:{index}" for index in range(1, 7))


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("case", ""),
        ("condition", 7),
        ("replicate", 0),
        ("recognition_status", 0),
        ("compliance", "false"),
    ],
)
def test_full_experiment_payload_rejects_bad_metadata_types(field, bad_value):
    payload = {
        "case": "synthetic_case_01",
        "condition": "blind_control",
        "replicate": 1,
        "primary_hotspots": ["A:1", "A:2", "A:3"],
        "alternate_hotspots": ["A:4", "A:5", "A:6"],
        "recognition_status": True,
        "compliance": True,
    }
    payload[field] = bad_value
    with pytest.raises(PredictionSchemaError):
        validate_experiment_prediction_schema(payload)


def test_canonical_tokens_reject_noncanonical_spelling():
    assert parse_residue_token("chain_A:-12B") == ("chain_A", -12, "B")
    with pytest.raises(ValueError):
        parse_residue_token("A:01")
    with pytest.raises(ValueError):
        parse_residue_token("A:12b")


def test_consensus_uses_frequency_then_reciprocal_rank_then_token():
    runs = [
        prediction("A:1", "A:2", "A:3", "A:4", "A:5", "A:6"),
        prediction("A:2", "A:1", "A:4", "A:3", "A:6", "A:5"),
        prediction("A:2", "A:4", "A:1", "A:6", "A:3", "A:5"),
    ]
    result = consensus_prediction(runs)
    assert result.ranked == ("A:2", "A:1", "A:4", "A:3", "A:6", "A:5")


def synthetic_atoms():
    return {
        "A:1": (Atom("C", 0, 0, 0), Atom("H", 100, 0, 0)),
        "A:2": (Atom("C", 5, 0, 0),),
        "A:3": (Atom("C", 10.5, 0, 0),),
        "A:100": (Atom("C", 3, 0, 0),),
        "B:1": (Atom("C", 40, 0, 0),),
    }


def test_exact_and_disjoint_pocket_distances_and_tolerance_grid():
    atoms = synthetic_atoms()
    exact = pocket_distance_metrics({"A:1", "A:2"}, {"A:1", "A:2"}, atoms)
    assert exact.minimum == 0.0
    assert exact.chamfer == 0.0
    assert exact.d90 == 0.0
    assert exact.hausdorff == 0.0
    assert standard_distance_tolerances({"A:1"}, {"A:1"}, atoms)[4.0].f1 == 1.0

    disjoint = pocket_distance_metrics({"A:1"}, {"B:1"}, atoms)
    assert disjoint.minimum == 40.0
    assert disjoint.chamfer == 40.0
    assert disjoint.d90 == 40.0
    assert disjoint.hausdorff == 40.0
    assert distance_tolerant_metrics({"A:1"}, {"B:1"}, atoms, 8.0).f1 == 0.0


def test_sequence_far_residues_can_be_spatially_near_and_hydrogen_is_ignored():
    atoms = synthetic_atoms()
    tolerant = distance_tolerant_metrics({"A:1"}, {"A:100"}, atoms, 4.0)
    assert tolerant.precision == 1.0
    assert tolerant.recall == 1.0
    assert tolerant.f1 == 1.0
    # The hydrogen at x=100 must not affect the heavy-atom minimum.
    assert pocket_distance_metrics({"A:1"}, {"A:100"}, atoms).minimum == 3.0


def test_h6_expansion_and_unweighted_and_sasa_weighted_overlap():
    atoms = synthetic_atoms()
    sasa = {token: 1.0 for token in atoms}
    sasa["A:2"] = 8.0
    result = h6_pocket_overlap({"A:1"}, {"A:3"}, atoms, sasa)
    assert result.predicted_h6 == frozenset({"A:1", "A:2", "A:100"})
    assert result.reference_h6 == frozenset({"A:2", "A:3"})
    assert result.jaccard == pytest.approx(1 / 4)
    assert result.dice == pytest.approx(2 / 5)
    assert result.sasa_weighted_jaccard == pytest.approx(8 / 11)
    assert result.sasa_weighted_dice == pytest.approx(16 / 19)


def test_symmetry_uses_one_global_chain_permutation_and_retains_strict_metrics():
    payload = prediction("A:1", "A:2", "C:1", "B:3", "C:2", "C:3")
    truth = {"B:1", "B:2", "A:3"}
    result = symmetry_adjusted_top_metrics(payload, truth, 20, [("A", "B")])
    assert result.strict.top3.h == 0
    assert result.strict.top6.h == 0
    assert result.symmetry_adjusted.top3.h == 2
    assert result.symmetry_adjusted.top6.h == 3
    assert dict(result.chain_mapping) == {"A": "B", "B": "A"}
    assert result.remapped_prediction.ranked[:2] == ("B:1", "B:2")


def test_symmetry_optimization_prioritizes_primary_top3_before_top6():
    payload = prediction("A:1", "A:2", "A:3", "A:4", "A:5", "A:6")
    # Identity has Top3=2, Top6=2; swapping A/B has Top3=1, Top6=4.
    truth = {"A:1", "A:2", "B:3", "B:4", "B:5", "B:6"}
    result = symmetry_adjusted_top_metrics(payload, truth, 20, [("A", "B")])
    assert result.symmetry_adjusted.top3.h == 2
    assert result.symmetry_adjusted.top6.h == 2
    assert dict(result.chain_mapping) == {"A": "A", "B": "B"}


def test_stratified_null_preserves_joint_counts_and_is_deterministic():
    universe = {f"{chain}:{index}" for chain in ("A", "B") for index in range(1, 11)}
    rsasa = {f"{chain}:{index}": index / 10 for chain in ("A", "B") for index in range(1, 11)}
    selected = {"A:1", "A:7", "B:2", "B:8"}
    quintiles = empirical_rsasa_quintiles(universe, rsasa)

    def counts(tokens):
        return Counter((parse_residue_token(token)[0], quintiles[token]) for token in tokens)

    samples = stratified_resamples(selected, universe, rsasa, draws=25, seed=314)
    assert all(counts(sample) == counts(selected) for sample in samples)
    truth = {"A:1", "B:2"}
    statistic = lambda sample: float(len(sample & truth))
    first = stratified_monte_carlo(selected, universe, rsasa, statistic, draws=50, seed=2718)
    second = stratified_monte_carlo(selected, universe, rsasa, statistic, draws=50, seed=2718)
    assert first == second
    assert 0.0 < first.p_greater_equal <= 1.0


def test_target_level_comparison_holm_and_bootstraps_are_deterministic():
    paired = paired_sign_flip_test([3.0, 2.0], [1.0, 1.0])
    assert paired.exact
    assert paired.estimate == 1.5
    assert paired.p_two_sided == 0.5
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx((0.03, 0.06, 0.06))

    target_a = target_bootstrap_ci([1.0, 2.0, 3.0], draws=100, seed=9)
    target_b = target_bootstrap_ci([1.0, 2.0, 3.0], draws=100, seed=9)
    assert target_a == target_b
    assert target_a.lower <= target_a.estimate <= target_a.upper

    hierarchy_a = hierarchical_bootstrap_ci(
        {"t1": [1.0, 2.0], "t2": [3.0, 4.0]}, draws=100, seed=10
    )
    hierarchy_b = hierarchical_bootstrap_ci(
        {"t1": [1.0, 2.0], "t2": [3.0, 4.0]}, draws=100, seed=10
    )
    assert hierarchy_a == hierarchy_b
    assert math.isclose(hierarchy_a.estimate, 2.5)


def test_disjoint_exact_ranked_pocket_has_zero_overlap_metrics():
    result = exact_ranked_metrics(
        ["A:1", "A:2", "A:3"], {"B:1", "B:2"}, universe_size=10
    )
    assert result.h == 0
    assert result.precision == result.recall == result.f1 == result.jaccard == 0.0
    assert result.average_precision == result.enrichment == 0.0
    assert result.hypergeometric_p == 1.0
