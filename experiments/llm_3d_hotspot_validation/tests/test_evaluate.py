from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from src.evaluate import (  # noqa: E402
    EvaluationError,
    _features,
    _local_token,
    _metric_bundle,
    _pocket_matched_null,
    _residue_atoms,
    derive_equivalent_chain_groups,
    derive_equivalent_chain_symmetry,
    evaluate_benchmark,
    joint_matched_null,
    map_author_labels_to_local,
    primary_decision,
    verify_prediction_freeze,
    write_reports,
)
from src.metrics import (  # noqa: E402
    PredictionSet,
    global_chain_permutation,
    symmetry_adjusted_top_metrics,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _synthetic_mapping() -> dict[str, object]:
    residues = []
    for local_chain, auth_chain, label_chain in (
        ("T1", "AUTH_A", "LAB_A"),
        ("T2", "AUTH_B", "LAB_B"),
    ):
        for number in range(1, 7):
            residues.append(
                {
                    "local": {"chain_id": local_chain, "seq_id": number},
                    "auth": {
                        "asym_id": auth_chain,
                        "seq_id": str(number),
                        "insertion_code": "",
                        "comp_id": "ALA",
                    },
                    "label": {
                        "asym_id": label_chain,
                        "seq_id": str(number),
                        "comp_id": "ALA",
                        "entity_id": "ENTITY_1",
                    },
                }
            )
    return {"schema_version": 1, "residues": residues, "atoms": []}


def _synthetic_features() -> dict[str, object]:
    residues = []
    for chain in ("T1", "T2"):
        for number in range(1, 7):
            residues.append(
                {
                    "token": f"{chain}:{number}",
                    "relative_sasa": (number - 1) / 5,
                    "residue_sasa_angstrom2": float(number),
                }
            )
    return {"schema_version": 1, "residues": residues}


def _synthetic_cif() -> str:
    header = """\
data_synthetic_evaluation
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.auth_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.pdbx_PDB_model_num
"""
    rows = []
    atom_id = 1
    for chain_index, chain in enumerate(("T1", "T2")):
        for number in range(1, 7):
            x = 4.0 * (number - 1)
            y = 1.0 * chain_index
            rows.append(
                f"ATOM {atom_id} C CA CA . ALA ALA {chain} {chain} {number} {number} ? "
                f"{x:.1f} {y:.1f} 0.0 1.00 1"
            )
            atom_id += 1
    return header + "\n".join(rows) + "\n#\n"


def test_label_file_is_refused_before_complete_freeze(tmp_path: Path) -> None:
    manifest = tmp_path / "prediction_freeze_manifest.json"
    plan = tmp_path / "run_plan.json"
    labels = tmp_path / "labels.json"
    _write_json(manifest, {"freeze_complete": False, "runs": []})
    _write_json(plan, {"runs": []})
    # Invalid label bytes make it evident that the label file was never parsed.
    labels.write_text("this must remain sealed", encoding="utf-8")

    with pytest.raises(EvaluationError, match="labels remain sealed"):
        evaluate_benchmark(manifest, plan, labels, per_run_mc_draws=2, joint_mc_draws=2)


def test_pipeline_artifact_freeze_contract_is_accepted_and_hashed(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    process = root / "process"
    prediction = root / "runs" / "opaque" / "output" / "prediction.json"
    _write_json(prediction, {"synthetic": True})
    plan = process / "run_plan.json"
    manifest = process / "prediction_freeze_manifest.json"
    _write_json(
        plan,
        {
            "runs": [
                {
                    "run_id": "opaque",
                    "task_path": "runs/opaque",
                    "case_id": "case_synthetic",
                    "condition": "anonymous_no_web",
                    "replicate": 1,
                }
            ]
        },
    )
    _write_json(
        manifest,
        {
            "frozen_at": "2000-01-01T00:00:00Z",
            "expected_predictions": 1,
            "validated_predictions": 1,
            "labels_absent": True,
            "artifacts": [
                {
                    "path": "runs/opaque/output/prediction.json",
                    "role": "output",
                    "run_id": "opaque",
                    "sha256": _digest(prediction),
                }
            ],
        },
    )
    state, plan_by_key = verify_prediction_freeze(manifest, plan)
    assert state.runs[0].prediction_path == prediction.resolve()
    assert set(plan_by_key) == {("case_synthetic", "anonymous_no_web", 1)}


def test_author_mapping_and_conservative_equivalent_chain_derivation() -> None:
    mapping = _synthetic_mapping()
    labels = [
        {"auth_asym_id": "AUTH_B", "auth_seq_id": "1"},
        {"auth_asym_id": "AUTH_B", "auth_seq_id": 2},
    ]
    assert map_author_labels_to_local(labels, mapping) == frozenset({"T2:1", "T2:2"})
    assert derive_equivalent_chain_groups(mapping) == (("T1", "T2"),)

    missing_entity = json.loads(json.dumps(mapping))
    for residue in missing_entity["residues"]:
        residue["label"].pop("entity_id")
    assert derive_equivalent_chain_groups(missing_entity) == ()


def test_evaluator_rejects_legacy_local_chain_tokens(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="T<positive-int>"):
        _local_token({"local": {"chain_id": "L1", "seq_id": 1}})

    features = _synthetic_features()
    features["residues"][0]["token"] = "L1:1"
    with pytest.raises(EvaluationError, match="T<positive-int>"):
        _features(features)

    structure_path = tmp_path / "legacy.cif"
    structure_path.write_text(_synthetic_cif().replace("T1 T1", "L1 L1"), encoding="utf-8")
    with pytest.raises(EvaluationError, match="T<positive-int>"):
        _residue_atoms(structure_path)


def test_symmetry_is_one_global_permutation_and_strict_is_retained() -> None:
    prediction = {
        "primary": ["T1:1", "T1:2", "T1:3"],
        "alternates": ["T1:4", "T1:5", "T1:6"],
    }
    result = symmetry_adjusted_top_metrics(
        prediction,
        {"T2:1", "T2:2"},
        universe_size=12,
        equivalent_chain_groups=derive_equivalent_chain_groups(_synthetic_mapping()),
    )
    assert result.strict.top3.h == 0
    assert result.symmetry_adjusted.top3.h == 2
    assert dict(result.chain_mapping) == {"T1": "T2", "T2": "T1"}


def test_frozen_explicit_three_auth_chain_group_maps_to_local_and_scores_globally() -> None:
    mapping = _synthetic_mapping()
    for number in range(1, 7):
        mapping["residues"].append(
            {
                "local": {"chain_id": "T3", "seq_id": number},
                "auth": {
                    "asym_id": "AUTH_C",
                    "seq_id": str(number),
                    "insertion_code": "",
                    "comp_id": "ALA",
                },
                "label": {
                    "asym_id": "LAB_C",
                    "seq_id": str(number),
                    "comp_id": "ALA",
                },
            }
        )
    groups = derive_equivalent_chain_groups(
        mapping, [["AUTH_A", "AUTH_B", "AUTH_C"]]
    )
    assert groups == (("T1", "T2", "T3"),)
    result = symmetry_adjusted_top_metrics(
        {
            "primary": ["T1:1", "T1:2", "T1:3"],
            "alternates": ["T1:4", "T1:5", "T1:6"],
        },
        {"T3:1", "T3:2"},
        universe_size=18,
        equivalent_chain_groups=groups,
    )
    assert result.strict.top3.h == 0
    assert result.symmetry_adjusted.top3.h == 2
    assert dict(result.chain_mapping)["T1"] == "T3"

    universe = {f"{chain}:{number}" for chain in ("T1", "T2", "T3") for number in range(1, 7)}
    rsasa = {token: (int(token.split(":")[1]) - 1) / 5 for token in universe}
    null = joint_matched_null(
        [
            {
                "target_id": "three_copy_target",
                "selected": {"T1:1", "T1:2", "T1:3"},
                "truth": {"T3:1", "T3:2"},
                "universe": universe,
                "rsasa": rsasa,
                "equivalent_chain_groups": groups,
            }
        ],
        draws=19,
        seed=7,
    )
    assert null.observed == 2


def test_explicit_group_aligns_different_crop_starts_by_label_sequence() -> None:
    residues = []
    components = {9: "GLY", 10: "ALA", 11: "SER", 12: "TYR", 13: "ARG", 14: "LEU"}
    # AUTH_A starts at source label position 10; AUTH_B includes one preceding
    # residue.  Local ordinals therefore differ by one throughout the overlap.
    for local_number, label_seq in enumerate(range(10, 15), start=1):
        residues.append(
            {
                "local": {"chain_id": "T1", "seq_id": local_number},
                "auth": {
                    "asym_id": "AUTH_A",
                    "seq_id": str(100 + local_number),
                    "insertion_code": "",
                    "comp_id": components[label_seq],
                },
                "label": {
                    "asym_id": "LAB_A",
                    "seq_id": str(label_seq),
                    "comp_id": components[label_seq],
                },
            }
        )
    for local_number, label_seq in enumerate(range(9, 15), start=1):
        residues.append(
            {
                "local": {"chain_id": "T2", "seq_id": local_number},
                "auth": {
                    "asym_id": "AUTH_B",
                    "seq_id": str(200 + local_number),
                    "insertion_code": "",
                    "comp_id": components[label_seq],
                },
                "label": {
                    "asym_id": "LAB_B",
                    "seq_id": str(label_seq),
                    "comp_id": components[label_seq],
                },
            }
        )
    residues.append(
        {
            "local": {"chain_id": "T3", "seq_id": 1},
            "auth": {
                "asym_id": "AUTH_X",
                "seq_id": "1",
                "insertion_code": "",
                "comp_id": "ALA",
            },
            "label": {"asym_id": "LAB_X", "seq_id": "1", "comp_id": "ALA"},
        }
    )
    mapping = {"schema_version": 1, "residues": residues, "atoms": []}
    symmetry = derive_equivalent_chain_symmetry(
        mapping, [["AUTH_A", "AUTH_B"]]
    )
    assert symmetry.groups == (("T1", "T2"),)
    assert symmetry.residue_correspondence[("T1:1", "T2")] == "T2:2"
    assert symmetry.residue_correspondence[("T2:2", "T1")] == "T1:1"
    assert ("T2:1", "T1") not in symmetry.residue_correspondence

    prediction = PredictionSet(
        primary=("T1:1", "T1:2", "T1:3"),
        alternates=("T1:4", "T1:5", "T3:1"),
    )
    truth = frozenset({"T2:2", "T2:3"})
    exact = symmetry_adjusted_top_metrics(
        prediction,
        truth,
        universe_size=12,
        equivalent_chain_groups=symmetry.groups,
        residue_correspondence=symmetry.residue_correspondence,
    )
    assert exact.strict.top3.h == 0
    assert exact.symmetry_adjusted.top3.h == 2
    assert exact.remapped_prediction.ranked[:3] == ("T2:2", "T2:3", "T2:4")

    universe = frozenset(
        [f"T1:{number}" for number in range(1, 6)]
        + [f"T2:{number}" for number in range(1, 7)]
        + ["T3:1"]
    )
    atoms = {
        token: (("C", float(100 + index), 0.0, 0.0),)
        for index, token in enumerate(sorted(universe))
    }
    for number in range(1, 7):
        atoms[f"T2:{number}"] = (("C", float(number), 0.0, 0.0),)
    rsasa = {token: (index % 5) / 4 for index, token in enumerate(sorted(universe))}
    sasa = {token: 1.0 for token in universe}
    bundle = _metric_bundle(
        prediction,
        truth,
        universe,
        rsasa,
        sasa,
        atoms,
        symmetry.groups,
        symmetry.residue_correspondence,
        mc_draws=13,
        seed=23,
    )
    assert bundle["spatial"]["symmetry_top3"]["distances"]["minimum"] == 0.0
    assert bundle["spatial"]["strict_top3"]["distances"]["minimum"] > 50.0
    assert bundle["matched_null"]["observed"] == 2.0
    assert bundle["pocket_matched_null"]["draws"] == 13
    assert "p_greater_equal" in bundle["pocket_matched_null"]
    assert "rsasa_weighted_jaccard" in bundle["spatial"]["symmetry_top3"]["h6_overlap"]

    joint = joint_matched_null(
        [
            {
                "target_id": "offset_crop",
                "selected": prediction.primary,
                "truth": truth,
                "universe": universe,
                "rsasa": rsasa,
                "equivalent_chain_groups": symmetry.groups,
                "residue_correspondence": symmetry.residue_correspondence,
            }
        ],
        draws=17,
        seed=29,
    )
    assert joint.observed == 2.0

    # T2:1 has no counterpart in T1, so the swap is invalid and the identity
    # permutation remains available as the required fallback.
    identity = global_chain_permutation(
        ["T2:1"],
        {"T1:1"},
        symmetry.groups,
        residue_correspondence=symmetry.residue_correspondence,
    )
    assert identity.remapped == ("T2:1",)
    assert dict(identity.chain_mapping) == {"T1": "T1", "T2": "T2"}

    mismatched = json.loads(json.dumps(mapping))
    next(
        item
        for item in mismatched["residues"]
        if item["local"] == {"chain_id": "T2", "seq_id": 2}
    )["label"]["comp_id"] = "VAL"
    with pytest.raises(EvaluationError, match="auth/label component|residue identity"):
        derive_equivalent_chain_symmetry(mismatched, [["AUTH_A", "AUTH_B"]])

    fallback_mapping = {
        "residues": [
            {
                "local": {"chain_id": "T1", "seq_id": 1},
                "auth": {
                    "asym_id": "AUTH_A", "seq_id": "7", "insertion_code": "", "comp_id": "ALA"
                },
                "label": {"asym_id": "LAB_A", "seq_id": "7", "comp_id": "ALA"},
            },
            {
                "local": {"chain_id": "T2", "seq_id": 1},
                "auth": {
                    "asym_id": "AUTH_B", "seq_id": "7", "insertion_code": "", "comp_id": "ALA"
                },
                "label": {"asym_id": "LAB_B", "seq_id": None, "comp_id": "ALA"},
            },
        ]
    }
    fallback = derive_equivalent_chain_symmetry(
        fallback_mapping, [["AUTH_A", "AUTH_B"]]
    )
    assert fallback.residue_correspondence[("T1:1", "T2")] == "T2:1"


def test_joint_matched_null_is_global_and_deterministic() -> None:
    universe = {f"A:{number}" for number in range(1, 21)}
    rsasa = {f"A:{number}": (number - 1) / 19 for number in range(1, 21)}
    cases = [
        {
            "target_id": "synthetic_target",
            "selected": {"A:1", "A:6", "A:11"},
            "truth": {"A:1", "A:2", "A:3"},
            "universe": universe,
            "rsasa": rsasa,
            "equivalent_chain_groups": (),
        }
    ]
    first = joint_matched_null(cases, draws=75, seed=41)
    second = joint_matched_null(cases, draws=75, seed=41)
    assert first == second
    assert first.draws == 75
    assert 0.0 < first.p_greater_equal <= 1.0


def test_pocket_matched_null_scores_observed_and_every_null_with_global_symmetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.evaluate as evaluate_module

    universe = frozenset(f"T1:{number}" for number in range(1, 7))
    selected = ("T1:1", "T1:2", "T1:3")
    atoms = {
        token: (("C", float(int(token.split(":")[1]) * 4), 0.0, 0.0),)
        for token in universe
    }
    rsasa = {token: (int(token.split(":")[1]) - 1) / 5 for token in universe}
    calls = 0
    original = evaluate_module.global_chain_permutation

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "global_chain_permutation", counted)
    result = _pocket_matched_null(
        selected,
        frozenset(selected),
        universe,
        rsasa,
        atoms,
        (),
        None,
        draws=4,
        seed=19,
    )
    assert result["observed"] == 1.0
    assert 0.0 < result["p_greater_equal"] <= 1.0
    assert result["draws"] == 4
    assert result["stratum_counts"]
    assert calls == 5  # Once for observed plus once for every null sample.


def test_primary_decision_implements_both_preregistered_gates() -> None:
    assert primary_decision(6, 2.5, 0.01)["decision"] == "supported"
    assert (
        primary_decision(6, 6.0, 0.01)["decision"]
        == "not_supported_in_this_benchmark"
    )
    assert (
        primary_decision(6, 2.5, 0.05)["decision"]
        == "not_supported_in_this_benchmark"
    )


def _full_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    mapping_path = tmp_path / "inputs" / "mapping.json"
    features_path = tmp_path / "inputs" / "features.json"
    structure_path = tmp_path / "inputs" / "local.cif"
    _write_json(mapping_path, _synthetic_mapping())
    _write_json(features_path, _synthetic_features())
    structure_path.write_text(_synthetic_cif(), encoding="utf-8")

    plan_runs = []
    frozen_runs = []
    for condition in ("named_no_web", "anonymous_no_web", "anonymous_generic_packet"):
        for replicate in (1, 2, 3):
            key = f"{condition}_{replicate}"
            prediction_path = tmp_path / "predictions" / f"{key}.json"
            _write_json(
                prediction_path,
                {
                    "schema_version": "1.0",
                    "case_id": "synthetic_target",
                    "condition": condition,
                    "replicate": replicate,
                    "primary_hotspots": ["T1:1", "T1:2", "T1:3"],
                    "alternate_hotspots": ["T1:4", "T1:5", "T1:6"],
                    "recognition_status": "none",
                    "compliance": {
                        "labels_seen": False,
                        "target_search_used": False,
                        "other_runs_seen": False,
                    },
                },
            )
            plan_runs.append(
                {
                    "target_id": "synthetic_target",
                    "condition": condition,
                    "replicate": replicate,
                    "mapping_path": str(mapping_path.relative_to(tmp_path)),
                    "features_path": str(features_path.relative_to(tmp_path)),
                    "structure_path": str(structure_path.relative_to(tmp_path)),
                }
            )
            frozen_runs.append(
                {
                    "target_id": "synthetic_target",
                    "condition": condition,
                    "replicate": replicate,
                    "prediction_path": str(prediction_path.relative_to(tmp_path)),
                    "sha256": _digest(prediction_path),
                }
            )
    plan = tmp_path / "run_plan.json"
    manifest = tmp_path / "prediction_freeze_manifest.json"
    labels = tmp_path / "labels.json"
    _write_json(plan, {"schema_version": 1, "runs": plan_runs})
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "freeze_complete": True,
            "frozen_at": "2000-01-01T00:00:00Z",
            "runs": frozen_runs,
        },
    )
    _write_json(
        labels,
        {
            "schema_version": 1,
            "targets": [
                {
                    "target_id": "synthetic_target",
                    "residues": [
                        {"auth_asym_id": "AUTH_B", "auth_seq_id": "1"},
                        {"auth_asym_id": "AUTH_B", "auth_seq_id": "2"},
                    ],
                }
            ],
        },
    )
    return manifest, plan, labels


def test_synthetic_end_to_end_reports_have_required_sections(tmp_path: Path) -> None:
    manifest, plan, labels = _full_synthetic_inputs(tmp_path)
    result = evaluate_benchmark(
        manifest,
        plan,
        labels,
        per_run_mc_draws=11,
        joint_mc_draws=31,
        bootstrap_draws=25,
        seed=17,
    )
    assert result["freeze"]["verified_run_count"] == 9
    assert result["primary"]["observed"] == 2
    assert len(result["consensus"]) == 3
    assert result["consensus"][0]["metrics"]["strict"]["top3"]["h"] == 0
    assert result["consensus"][0]["metrics"]["symmetry_adjusted"]["top3"]["h"] == 2

    output = tmp_path / "results"
    write_reports(result, output)
    expected = (
        output / "summary.md",
        output / "condition_comparison.md",
        output / "leakage_audit.md",
        output / "per_target" / "synthetic_target.md",
    )
    for path in expected:
        text = path.read_text(encoding="utf-8")
        assert "## ARS Material Passport" in text
        assert "## Evidence" in text
        assert "## Inference" in text
        assert "## Limitations" in text
    assert (output / "metrics.json").is_file()
    assert (output / "metrics.csv").is_file()


def _terminal_outcome_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Create a wholly synthetic 72-outcome freeze (40 prediction, 32 refusal)."""

    root = tmp_path / "terminal_experiment"
    process = root / "process"
    mapping_path = root / "inputs" / "mapping.json"
    features_path = root / "inputs" / "features.json"
    structure_path = root / "inputs" / "local.cif"
    _write_json(mapping_path, _synthetic_mapping())
    _write_json(features_path, _synthetic_features())
    structure_path.parent.mkdir(parents=True, exist_ok=True)
    structure_path.write_text(_synthetic_cif(), encoding="utf-8")

    plan_runs = []
    terminal_runs = []
    artifacts = []
    labels_targets = []
    target_manifest_entries = []
    first_failure: Path | None = None
    conditions = ("named_no_web", "anonymous_no_web", "anonymous_generic_packet")
    for target_index in range(1, 9):
        target_id = f"case_{target_index:02d}"
        labels_targets.append(
            {
                "target_id": target_id,
                "residues": [
                    {"auth_asym_id": "AUTH_B", "auth_seq_id": "1"},
                    {"auth_asym_id": "AUTH_B", "auth_seq_id": "2"},
                ],
            }
        )
        target_manifest_entries.append(
            {
                "case_id": target_id,
                "display_name": f"Synthetic Material {target_index}",
                "equivalent_auth_chain_groups": [["AUTH_A", "AUTH_B"]],
            }
        )
        for condition in conditions:
            for replicate in (1, 2, 3):
                run_id = f"{target_id}_{condition}_{replicate}"
                plan_runs.append(
                    {
                        "run_id": run_id,
                        "target_id": target_id,
                        "condition": condition,
                        "replicate": replicate,
                        "mapping_path": "../inputs/mapping.json",
                        "features_path": "../inputs/features.json",
                        "structure_path": "../inputs/local.cif",
                    }
                )
                success = condition == "named_no_web" or replicate == 1
                if condition == "anonymous_generic_packet" and target_index == 8:
                    success = False
                if (
                    condition == "anonymous_generic_packet"
                    and target_index == 7
                    and replicate == 2
                ):
                    success = True
                if success:
                    excluded = (
                        condition == "named_no_web"
                        and replicate == 1
                        and target_index <= 3
                    )
                    prediction = root / "runs" / run_id / "output" / "prediction.json"
                    process_md = root / "runs" / run_id / "process.md"
                    _write_json(
                        prediction,
                        {
                            "case_id": target_id,
                            "condition": condition,
                            "replicate": replicate,
                            "primary_hotspots": ["T1:1", "T1:2", "T1:3"],
                            "alternate_hotspots": ["T1:4", "T1:5", "T1:6"],
                            "recognition_status": "none",
                            "compliance": {
                                "labels_seen": False,
                                "target_search_used": False,
                                "other_runs_seen": False,
                            },
                        },
                    )
                    process_md.write_text("synthetic process record\n", encoding="utf-8")
                    terminal_run = {
                            "run_id": run_id,
                            "outcome": "excluded_prediction" if excluded else "prediction",
                            "prediction_path": str(prediction.relative_to(root)),
                            "process_path": str(process_md.relative_to(root)),
                        }
                    if excluded:
                        exclusion = process / "exclusions" / f"{run_id}.md"
                        exclusion.parent.mkdir(parents=True, exist_ok=True)
                        exclusion.write_text("synthetic compliance exclusion\n", encoding="utf-8")
                        terminal_run["exclusion_path"] = str(exclusion.relative_to(root))
                        terminal_run["exclusion_reason"] = "predefined_compliance_exclusion"
                        artifacts.append(
                            {
                                "path": str(exclusion.relative_to(root)),
                                "run_id": run_id,
                                "role": "exclusion",
                                "sha256": _digest(exclusion),
                            }
                        )
                    terminal_runs.append(terminal_run)
                    for path, role in ((prediction, "output"), (process_md, "process")):
                        artifacts.append(
                            {
                                "path": str(path.relative_to(root)),
                                "run_id": run_id,
                                "role": role,
                                "sha256": _digest(path),
                            }
                        )
                else:
                    failure = process / "failures" / f"{run_id}.md"
                    failure.parent.mkdir(parents=True, exist_ok=True)
                    failure.write_text("synthetic refusal\n", encoding="utf-8")
                    first_failure = first_failure or failure
                    terminal_runs.append(
                        {
                            "run_id": run_id,
                            "outcome": "terminal_failure",
                            "failure_path": str(failure.relative_to(root)),
                        }
                    )
                    artifacts.append(
                        {
                            "path": str(failure.relative_to(root)),
                            "run_id": run_id,
                            "role": "failure",
                            "sha256": _digest(failure),
                        }
                    )

    assert len(plan_runs) == len(terminal_runs) == 72
    assert sum(
        item["outcome"] in {"prediction", "excluded_prediction"}
        for item in terminal_runs
    ) == 40
    assert sum(item["outcome"] == "excluded_prediction" for item in terminal_runs) == 3
    plan = process / "run_plan.json"
    manifest = process / "prediction_freeze_manifest.json"
    labels = process / "labels.json"
    target_manifest = process / "target_manifest.json"
    _write_json(plan, {"runs": plan_runs})
    _write_json(
        manifest,
        {
            "frozen_at": "2000-01-01T00:00:00Z",
            "labels_absent": True,
            "all_terminal": True,
            "expected_runs": 72,
            "validated_predictions": 40,
            "eligible_predictions": 37,
            "excluded_predictions": 3,
            "terminal_failures": 32,
            "runs": terminal_runs,
            "artifacts": artifacts,
        },
    )
    _write_json(labels, {"targets": labels_targets})
    _write_json(target_manifest, {"targets": target_manifest_entries})
    assert first_failure is not None
    return manifest, plan, labels, target_manifest, first_failure


def test_terminal_outcomes_gate_primary_and_keep_available_case_exploratory(
    tmp_path: Path,
) -> None:
    manifest, plan, labels, target_manifest, first_failure = _terminal_outcome_inputs(tmp_path)
    result = evaluate_benchmark(
        manifest,
        plan,
        labels,
        target_manifest_path=target_manifest,
        per_run_mc_draws=3,
        joint_mc_draws=7,
        bootstrap_draws=5,
        seed=101,
    )

    assert result["freeze"]["verified_run_count"] == 72
    assert result["freeze"]["verified_terminal_outcome_count"] == 72
    assert result["freeze"]["verified_prediction_hash_count"] == 40
    assert len(result["outcomes"]) == 72
    assert len(result["runs"]) == 37
    assert sum(item["outcome"] == "excluded" for item in result["outcomes"]) == 3
    assert result["outcome_summary_by_condition"]["anonymous_no_web"] == {
        "attempted": 24,
        "valid": 8,
        "refusals": 16,
        "excluded": 0,
        "other_failures": 0,
        "valid_rate": 1 / 3,
        "refusal_rate": 2 / 3,
        "failure_rate": 2 / 3,
    }
    assert result["primary"]["decision"] == "not_evaluable_due_to_refusals"
    assert "supported" not in result["primary"]
    assert result["primary"]["observed"] is None
    assert len(result["consensus"]) == 5  # Three named cells lose an excluded prediction.
    assert len(result["available_case_consensus"]) == 23
    assert result["exploratory"]["available_case_joint_matched_null"]["target_count"] == 8
    assert all(item["analysis"].startswith("exploratory") for item in result["contrasts"])
    assert [item["paired_target_count"] for item in result["contrasts"]] == [8, 7]
    assert result["target_names"]["case_01"] == "Synthetic Material 1"
    assert all(run["metrics"]["matched_null"]["draws"] == 3 for run in result["runs"])
    assert all(
        run["metrics"]["pocket_matched_null"]["draws"] == 3
        for run in result["runs"]
    )
    zero_cell = next(
        item for item in result["workflow_hit_rates"]
        if item["target_id"] == "case_08"
        and item["condition"] == "anonymous_generic_packet"
    )
    assert len(result["workflow_hit_rates"]) == 24
    assert zero_cell["valid_runs"] == 0
    assert zero_cell["strict_hit_runs"] == zero_cell["symmetry_hit_runs"] == 0
    assert zero_cell["strict_workflow_hit_rate"] == 0.0
    assert zero_cell["symmetry_workflow_hit_rate"] == 0.0
    assert zero_cell["strict_conditional_hit_rate"] is None
    assert zero_cell["symmetry_conditional_hit_rate"] is None

    output = tmp_path / "terminal_reports"
    write_reports(result, output)
    summary = (output / "summary.md").read_text(encoding="utf-8")
    target = (output / "per_target" / "case_01.md").read_text(encoding="utf-8")
    leakage = (output / "leakage_audit.md").read_text(encoding="utf-8")
    assert "not_evaluable_due_to_refusals" in summary
    assert "No supported/not-supported conclusion" in summary
    assert "Exploratory available-case" in summary
    assert "Refusal-related selection bias" in summary
    assert "predefined_compliance_exclusion" in summary
    assert "Synthetic Material 1" in target
    assert "predefined_compliance_exclusion" in target
    for heading in (
        "h/K", "Precision", "Recall", "F1", "Jaccard", "AP", "Enrichment",
        "Hypergeom p", "Bidirectional Chamfer", "D90", "Hausdorff", "4Å P",
        "6Å F1", "8Å R", "Pocket Dice", "rSASA-weighted Jaccard",
        "Pocket p≥",
    ):
        assert heading in target
    assert "stored/frozen prediction artifacts" in summary
    assert "eligible predictions only" in leakage

    # The freeze verifier hashes every artifact, including refusal records.
    first_failure.write_text("tampered synthetic refusal\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="hash mismatch"):
        verify_prediction_freeze(manifest, plan)
