from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


EXPERIMENT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT / "src"
sys.path.insert(0, str(SRC))

import pipeline  # noqa: E402


def _synthetic_cif() -> str:
    tags = """\
data_private_source
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
    rows: list[str] = []
    atom_id = 1
    names = ("ALA", "SER", "TYR", "ARG", "LEU", "ASP")
    for seq_id, residue_name in enumerate(names, start=1):
        x = float(seq_id * 3)
        for atom_name, element, offset in (
            ("N", "N", -1.0),
            ("CA", "C", 0.0),
            ("C", "C", 1.0),
            ("O", "O", 1.7),
            ("CB", "C", 0.5),
        ):
            rows.append(
                f"ATOM {atom_id} {element} {atom_name} {atom_name} . "
                f"{residue_name} {residue_name} SOURCE_LABEL SOURCE_AUTH "
                f"{seq_id} {seq_id} ? {x + offset:.1f} "
                f"{(seq_id % 2) * 1.5:.1f} {0.8 if atom_name == 'CB' else 0.0:.1f} 1.00 1"
            )
            atom_id += 1
    return tags + "\n".join(rows) + "\n#\n"


def _make_experiment(root: Path) -> Path:
    process = root / "process"
    raw = process / "raw_cif"
    raw.mkdir(parents=True)
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "src" / "pipeline.py").write_text("# pipeline sentinel\n", encoding="utf-8")
    (root / "src" / "metrics.py").write_text("# scoring sentinel\n", encoding="utf-8")
    (root / "tests" / "test_pipeline.py").write_text("# test sentinel\n", encoding="utf-8")
    (root / "run_benchmark.py").write_text("# entrypoint sentinel\n", encoding="utf-8")
    targets = []
    for index in range(8):
        pdb_id = f"x{index:03d}"
        targets.append(
            {
                "case_id": f"case_{index:08x}",
                "target_name": f"SecretTarget{index}",
                "pdb_id": pdb_id,
                "contexts": [{"auth_chain": "SOURCE_AUTH", "start": 1, "end": 6}],
            }
        )
        (raw / f"{pdb_id}.cif").write_text(_synthetic_cif(), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "targets": targets,
        "conditions": list(pipeline.CONDITIONS),
        "replicates": list(pipeline.REPLICATES),
        "base_seed": 12345,
    }
    (process / "target_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (process / "prediction_schema.json").write_text(
        (EXPERIMENT / "process" / "prediction_schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (process / "prompt_template.md").write_text(
        (EXPERIMENT / "process" / "prompt_template.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (process / "generic_knowledge_packet.md").write_text(
        "# Frozen generic method\n\nRank solvent-accessible geometric patches.\n",
        encoding="utf-8",
    )
    (process / "preregistration.md").write_text(
        "# Synthetic preregistration\n", encoding="utf-8"
    )
    (process / "generic_search_audit.md").write_text(
        "# Synthetic audit\n", encoding="utf-8"
    )
    (process / "download_manifest.json").write_text(
        '{"downloads": []}\n', encoding="utf-8"
    )
    (process / "leakage_preflight.md").write_text(
        """# Leakage Preflight Checklist

- [ ] Target-only anonymous structures generated and metadata scanner passed.
- [ ] Local residue mapping retained outside assigned anonymous inputs.
- [ ] Generic packet has no benchmark names, PDB IDs, partners, fingerprints, or
      target-specific sources.
- [ ] All prompts and common code hashed.
- [ ] Ground-truth label file absent immediately before first dispatch.
- [ ] Every dispatch uses `fork_turns="none"` and an opaque task name.
- [ ] All 72 outputs validated and frozen before label creation.
""",
        encoding="utf-8",
    )
    return root


def _valid_payload(run: dict[str, object], input_dir: Path) -> dict[str, object]:
    features = json.loads((input_dir / "features.json").read_text(encoding="utf-8"))
    tokens = [item["token"] for item in features["residues"]]
    assert len(tokens) == 6
    return {
        "schema_version": "1.0",
        "case_id": run["case_id"],
        "condition": run["condition"],
        "replicate": run["replicate"],
        "primary_hotspots": tokens[:3],
        "alternate_hotspots": tokens[3:],
        "pocket_groups": [tokens[:3], tokens[3:]],
        "structural_rationale": "Synthetic geometry-only rationale.",
        "recognition_status": "none",
        "compliance": {
            "labels_seen": False,
            "target_search_used": False,
            "other_runs_seen": False,
            "files_read": ["input/structure.cif", "input/features.json"],
            "commands_run": [],
            "files_created": ["output/prediction.json"],
        },
    }


def _write_all_predictions(root: Path) -> list[dict[str, object]]:
    plan = json.loads((root / "process" / "run_plan.json").read_text(encoding="utf-8"))
    for run in plan["runs"]:
        run_dir = root / run["task_path"]
        payload = _valid_payload(run, run_dir / "input")
        (run_dir / "output" / "prediction.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        (run_dir / "output" / "process.md").write_text(
            "# Process\n\nRanked opaque local geometry; compliance self-check passed.\n",
            encoding="utf-8",
        )
    return plan["runs"]


def _replace_prediction_with_failure(root: Path, run: dict[str, object]) -> Path:
    run_dir = root / str(run["task_path"])
    (run_dir / "output" / "prediction.json").unlink()
    (run_dir / "output" / "process.md").unlink()
    failure_path = root / "process" / "failures" / f"{run['run_id']}.md"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        f"# Formal run failure: `{run['run_id']}`\n\n"
        "The assigned run ended in a refusal and produced no prediction.\n",
        encoding="utf-8",
    )
    return failure_path


def _write_exclusion(root: Path, run: dict[str, object]) -> Path:
    exclusion_path = root / "process" / "exclusions" / f"{run['run_id']}.md"
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    exclusion_path.write_text(
        f"# Predefined compliance exclusion: `{run['run_id']}`\n\n"
        "The original prediction is retained unchanged but excluded from analysis.\n",
        encoding="utf-8",
    )
    return exclusion_path


@pytest.mark.parametrize(
    ("target_count", "expected"),
    [(1, 9), (2, 18), (8, 72)],
)
def test_run_plan_cartesian_combinatorics_are_opaque(
    target_count: int, expected: int
) -> None:
    manifest = {
        "targets": [{"case_id": f"case_{index:04d}"} for index in range(target_count)],
        "conditions": list(pipeline.CONDITIONS),
        "replicates": list(pipeline.REPLICATES),
        "base_seed": 77,
    }
    plan = pipeline.build_run_plan(manifest)
    assert plan["expected_run_count"] == expected
    assert len(plan["runs"]) == expected
    assert len({run["run_id"] for run in plan["runs"]}) == expected
    assert all(run["task_path"] == f"runs/{run['run_id']}" for run in plan["runs"])
    assert all("case_" not in run["run_id"] for run in plan["runs"])


@pytest.mark.parametrize(
    ("token", "valid"),
    [
        ("T1:1", True),
        ("T12:345", True),
        ("L1:1", False),
        ("T0:1", False),
        ("T1:0", False),
        ("T1:-1", False),
    ],
)
def test_anonymous_local_token_contract(token: str, valid: bool) -> None:
    assert bool(pipeline._LOCAL_TOKEN_RE.fullmatch(token)) is valid


def test_prepare_refuses_when_any_label_file_exists(tmp_path: Path) -> None:
    root = _make_experiment(tmp_path / "experiment")
    (root / "labels.json").write_text("{}", encoding="utf-8")
    with pytest.raises(pipeline.BenchmarkStateError, match="label-blind gate"):
        pipeline.prepare(root, sphere_points=4)
    assert not (root / "process" / "run_plan.json").exists()


def test_prepare_isolates_identity_and_places_condition_packets(tmp_path: Path) -> None:
    root = _make_experiment(tmp_path / "experiment")
    result = pipeline.prepare(root, sphere_points=4)
    assert result.run_count == 72
    plan = json.loads(result.run_plan_path.read_text(encoding="utf-8"))
    assert len(plan["runs"]) == 72
    assert len(list((root / "runs").iterdir())) == 72

    structures: dict[tuple[str, int], set[bytes]] = {}
    for run in plan["runs"]:
        input_dir = root / run["task_path"] / "input"
        names = {path.name for path in input_dir.iterdir()}
        assert {"structure.cif", "features.json", "prediction_schema.json", "prompt.md"} <= names
        structure_text = (input_dir / "structure.cif").read_text(encoding="utf-8")
        features_text = (input_dir / "features.json").read_text(encoding="utf-8")
        assert "T1" in structure_text and "L1" not in structure_text
        assert '"token":"T1:' in features_text and "L1" not in features_text
        if run["condition"] == "named_no_web":
            assert "identity_card.json" in names
            assert "generic_knowledge_packet.md" not in names
            assert "SecretTarget" in (input_dir / "identity_card.json").read_text()
        elif run["condition"] == "anonymous_generic_packet":
            assert "generic_knowledge_packet.md" in names
            assert "identity_card.json" not in names
        else:
            assert "identity_card.json" not in names
            assert "generic_knowledge_packet.md" not in names
        if run["condition"].startswith("anonymous_"):
            assigned = "\n".join(
                path.read_text(encoding="utf-8") for path in input_dir.iterdir()
            ).casefold()
            assert "secrettarget" not in assigned
            assert "source_auth" not in assigned
            assert "_atom_site.auth_" not in assigned
            assert not any(f"x{index:03d}" in assigned for index in range(8))
        key = (run["case_id"], run["replicate"])
        structures.setdefault(key, set()).add((input_dir / "structure.cif").read_bytes())
    assert all(len(values) == 1 for values in structures.values())
    for case_dir in (root / "process" / "prepared").iterdir():
        assert (case_dir / "private" / "local_mapping.json").is_file()
        assert len(list((case_dir / "variants").glob("v*/structure.cif"))) == 3
    assigned_prompt = (
        root / plan["runs"][0]["task_path"] / "input" / "prompt.md"
    ).read_text(encoding="utf-8")
    assert "output/prediction.json" in assigned_prompt
    assert "output/process.md" in assigned_prompt
    assert "T<positive-int>:<positive-int>" in assigned_prompt
    assert "do not name, propose, or discuss any guessed target identity" in assigned_prompt


def test_validation_checks_full_schema_and_allowed_local_tokens(tmp_path: Path) -> None:
    root = _make_experiment(tmp_path / "experiment")
    pipeline.prepare(root, sphere_points=4)
    runs = _write_all_predictions(root)
    report = pipeline.validate(root)
    assert report["all_valid"] is True
    assert report["successful_artifacts"] == 72
    assert report["eligible_predictions"] == 72
    assert report["excluded_predictions"] == 0

    bad_run = runs[0]
    process_path = root / bad_run["task_path"] / "output" / "process.md"
    process_path.unlink()
    report = pipeline.validate(root)
    assert report["all_valid"] is False
    assert report["successful_artifacts"] == 71
    errors = next(item["errors"] for item in report["runs"] if item["run_id"] == bad_run["run_id"])
    assert "output/process.md is missing" in errors

    process_path.write_text(" \n", encoding="utf-8")
    report = pipeline.validate(root)
    assert report["all_valid"] is False
    assert report["successful_artifacts"] == 71
    errors = next(item["errors"] for item in report["runs"] if item["run_id"] == bad_run["run_id"])
    assert "output/process.md must be non-empty" in errors
    process_path.write_text("# Process\n\nOpaque local structural evidence.\n", encoding="utf-8")

    path = root / bad_run["task_path"] / "output" / "prediction.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["compliance"]["labels_seen"] = True
    payload["alternate_hotspots"][-1] = "T99:999"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = pipeline.validate(root)
    assert report["all_valid"] is False
    assert report["successful_artifacts"] == 71
    errors = next(item["errors"] for item in report["runs"] if item["run_id"] == bad_run["run_id"])
    assert any("must equal False" in error for error in errors)
    assert any("not present in assigned features" in error for error in errors)


def test_terminal_failure_validation_and_mixed_freeze(tmp_path: Path) -> None:
    root = _make_experiment(tmp_path / "experiment")
    pipeline.prepare(root, sphere_points=4)
    runs = _write_all_predictions(root)
    first_failure = _replace_prediction_with_failure(root, runs[0])
    second_failure = _replace_prediction_with_failure(root, runs[1])

    report = pipeline.validate(root)
    assert report["terminal_outcomes"] == 72
    assert report["valid_predictions"] == 70
    assert report["eligible_predictions"] == 70
    assert report["excluded_predictions"] == 0
    assert report["terminal_failures"] == 2
    assert report["unaccounted"] == 0
    assert report["dual_outcome"] == 0
    assert report["all_valid_predictions"] is False
    assert report["all_terminal"] is True
    assert report["all_valid"] is False
    by_run = {item["run_id"]: item for item in report["runs"]}
    assert by_run[runs[0]["run_id"]]["outcome"] == "terminal_failure"
    assert by_run[runs[0]["run_id"]]["terminal"] is True

    first_failure.write_text(" \n", encoding="utf-8")
    report = pipeline.validate(root)
    assert report["terminal_outcomes"] == 71
    assert report["terminal_failures"] == 1
    assert report["unaccounted"] == 1
    assert report["all_terminal"] is False
    errors = next(
        item["errors"] for item in report["runs"] if item["run_id"] == runs[0]["run_id"]
    )
    assert "failure document must be non-empty" in errors
    first_failure.write_text(
        f"# Formal run failure: `{runs[0]['run_id']}`\n", encoding="utf-8"
    )

    first_failure.write_text(
        f"# Formal run failure: `{runs[1]['run_id']}`\n", encoding="utf-8"
    )
    report = pipeline.validate(root)
    assert report["terminal_outcomes"] == 71
    assert report["unaccounted"] == 1
    errors = next(
        item["errors"] for item in report["runs"] if item["run_id"] == runs[0]["run_id"]
    )
    assert "failure document must identify exactly its matching run_id" in errors
    first_failure.write_text(
        f"# Formal run failure: `{runs[0]['run_id']}`\n", encoding="utf-8"
    )

    bad_failure = first_failure.parent / "run_00000000000000000000.md"
    bad_failure.write_text("# Not a planned run\n", encoding="utf-8")
    report = pipeline.validate(root)
    assert report["all_terminal"] is False
    assert any("exactly one planned run ID" in error for error in report["plan_errors"])
    bad_failure.unlink()

    run_dir = root / str(runs[1]["task_path"])
    payload = _valid_payload(runs[1], run_dir / "input")
    (run_dir / "output" / "prediction.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    report = pipeline.validate(root)
    assert report["dual_outcome"] == 1
    assert report["terminal_outcomes"] == 71
    assert report["all_terminal"] is False
    dual = next(
        item for item in report["runs"] if item["run_id"] == runs[1]["run_id"]
    )
    assert dual["outcome"] == "dual_outcome"
    assert any("cannot coexist" in error for error in dual["errors"])
    (run_dir / "output" / "prediction.json").unlink()

    manifest = pipeline.freeze(root)
    assert manifest["expected_runs"] == 72
    assert manifest["validated_runs"] == 72
    assert manifest["validated_predictions"] == 70
    assert manifest["eligible_predictions"] == 70
    assert manifest["excluded_predictions"] == 0
    assert manifest["terminal_failures"] == 2
    assert manifest["labels_absent"] is True
    assert manifest["all_terminal"] is True
    manifest_by_run = {item["run_id"]: item for item in manifest["runs"]}
    assert len(manifest_by_run) == 72
    assert manifest_by_run[runs[0]["run_id"]] == {
        "run_id": runs[0]["run_id"],
        "outcome": "terminal_failure",
        "failure_path": f"process/failures/{runs[0]['run_id']}.md",
    }
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    assert artifacts[first_failure.relative_to(root).as_posix()]["role"] == "process"
    assert artifacts[second_failure.relative_to(root).as_posix()]["role"] == "process"
    assert sum(
        item["role"] == "output" and item["path"].endswith("prediction.json")
        for item in manifest["artifacts"]
    ) == 70
    assert "documented refusals were not imputed" in (
        root / "process" / "leakage_preflight.md"
    ).read_text(encoding="utf-8")


def test_excluded_predictions_are_retained_and_not_dual_outcomes(tmp_path: Path) -> None:
    root = _make_experiment(tmp_path / "experiment")
    pipeline.prepare(root, sphere_points=4)
    runs = _write_all_predictions(root)
    first_exclusion = _write_exclusion(root, runs[0])
    second_exclusion = _write_exclusion(root, runs[1])

    report = pipeline.validate(root)
    assert report["terminal_outcomes"] == 72
    assert report["valid_predictions"] == 72
    assert report["eligible_predictions"] == 70
    assert report["excluded_predictions"] == 2
    assert report["terminal_failures"] == 0
    assert report["unaccounted"] == 0
    assert report["dual_outcome"] == 0
    assert report["all_valid_predictions"] is True
    assert report["all_terminal"] is True
    by_run = {item["run_id"]: item for item in report["runs"]}
    excluded = by_run[runs[0]["run_id"]]
    assert excluded["outcome"] == "excluded_prediction"
    assert excluded["valid"] is True
    assert excluded["eligible"] is False
    assert excluded["terminal"] is True

    first_exclusion.write_text(" \n", encoding="utf-8")
    report = pipeline.validate(root)
    assert report["terminal_outcomes"] == 71
    assert report["excluded_predictions"] == 1
    assert report["unaccounted"] == 1
    errors = next(
        item["errors"] for item in report["runs"] if item["run_id"] == runs[0]["run_id"]
    )
    assert "exclusion document must be non-empty" in errors

    first_exclusion.write_text(
        f"# Wrong exclusion: `{runs[1]['run_id']}`\n", encoding="utf-8"
    )
    report = pipeline.validate(root)
    errors = next(
        item["errors"] for item in report["runs"] if item["run_id"] == runs[0]["run_id"]
    )
    assert "exclusion document must identify exactly its matching run_id" in errors
    first_exclusion = _write_exclusion(root, runs[0])

    unknown = first_exclusion.parent / "run_00000000000000000000.md"
    unknown.write_text(
        "# Unknown exclusion: `run_00000000000000000000`\n", encoding="utf-8"
    )
    report = pipeline.validate(root)
    assert report["all_terminal"] is False
    assert any(
        "exclusion document must be named for exactly one planned run ID" in error
        for error in report["plan_errors"]
    )
    unknown.unlink()

    failure_path = _replace_prediction_with_failure(root, runs[1])
    report = pipeline.validate(root)
    assert report["dual_outcome"] == 1
    assert report["terminal_outcomes"] == 71
    conflict = next(
        item for item in report["runs"] if item["run_id"] == runs[1]["run_id"]
    )
    assert conflict["outcome"] == "dual_outcome"
    assert any("cannot both record" in error for error in conflict["errors"])
    failure_path.unlink()
    run_dir = root / str(runs[1]["task_path"])
    (run_dir / "output" / "prediction.json").write_text(
        json.dumps(_valid_payload(runs[1], run_dir / "input")), encoding="utf-8"
    )
    (run_dir / "output" / "process.md").write_text(
        "# Process\n\nOriginal opaque analysis retained.\n", encoding="utf-8"
    )

    manifest = pipeline.freeze(root)
    assert manifest["validated_predictions"] == 72
    assert manifest["eligible_predictions"] == 70
    assert manifest["excluded_predictions"] == 2
    assert manifest["terminal_failures"] == 0
    manifest_by_run = {item["run_id"]: item for item in manifest["runs"]}
    assert manifest_by_run[runs[0]["run_id"]] == {
        "run_id": runs[0]["run_id"],
        "outcome": "excluded_prediction",
        "exclusion_reason": "predefined_compliance_exclusion",
        "prediction_path": f"runs/{runs[0]['run_id']}/output/prediction.json",
        "process_path": f"runs/{runs[0]['run_id']}/output/process.md",
        "exclusion_path": f"process/exclusions/{runs[0]['run_id']}.md",
    }
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    assert artifacts[first_exclusion.relative_to(root).as_posix()]["role"] == "process"
    assert artifacts[second_exclusion.relative_to(root).as_posix()]["role"] == "process"
    for run in runs[:2]:
        assert f"runs/{run['run_id']}/output/prediction.json" in artifacts
        assert f"runs/{run['run_id']}/output/process.md" in artifacts
    assert sum(
        item["role"] == "output" and item["path"].endswith("prediction.json")
        for item in manifest["artifacts"]
    ) == 72


def test_freeze_refuses_incomplete_then_hashes_all_artifacts(tmp_path: Path) -> None:
    root = _make_experiment(tmp_path / "experiment")
    pipeline.prepare(root, sphere_points=4)
    with pytest.raises(pipeline.BenchmarkStateError, match="freeze refused"):
        pipeline.freeze(root)
    runs = _write_all_predictions(root)
    scratch = root / runs[0]["task_path"] / "scratch"
    (scratch / "helper.py").write_text("print('opaque analysis')\n", encoding="utf-8")
    (scratch / "analysis.log").write_text("completed\n", encoding="utf-8")
    (root / "src" / "schema.txt").write_text("frozen source support\n", encoding="utf-8")
    (root / "tests" / "fixture.bin").write_bytes(b"frozen-test-support\n")
    manifest = pipeline.freeze(root)
    assert manifest["labels_absent"] is True
    assert manifest["expected_runs"] == 72
    assert manifest["validated_runs"] == 72
    assert manifest["validated_predictions"] == 72
    assert manifest["eligible_predictions"] == 72
    assert manifest["excluded_predictions"] == 0
    assert manifest["terminal_failures"] == 0
    assert sum(
        item["role"] == "output" and item["path"].endswith("prediction.json")
        for item in manifest["artifacts"]
    ) == 72
    assert any(item["role"] == "prompt" for item in manifest["artifacts"])
    assert any(item["role"] == "input" for item in manifest["artifacts"])
    assert any(item["role"] == "common_code" for item in manifest["artifacts"])
    by_path = {item["path"]: item for item in manifest["artifacts"]}
    assert by_path["process/target_manifest.json"]["role"] == "process"
    assert by_path["process/preregistration.md"]["role"] == "process"
    assert by_path["process/download_manifest.json"]["role"] == "process"
    assert by_path["process/generic_search_audit.md"]["role"] == "process"
    assert by_path["process/generic_knowledge_packet.md"]["role"] == "process"
    assert by_path["process/raw_cif/x000.cif"]["role"] == "process"
    assert by_path["process/structure_qc.json"]["role"] == "process"
    assert by_path["process/preparation_checksums.json"]["role"] == "process"
    assert any(
        path.endswith("/private/local_mapping.json") and item["role"] == "process"
        for path, item in by_path.items()
    )
    assert by_path["src/pipeline.py"]["role"] == "source_code"
    assert by_path["src/metrics.py"]["role"] == "source_code"
    assert by_path["src/schema.txt"]["role"] == "source_code"
    assert by_path["tests/test_pipeline.py"]["role"] == "test_code"
    assert by_path["tests/fixture.bin"]["role"] == "test_code"
    assert by_path["run_benchmark.py"]["role"] == "entrypoint_code"
    opaque_run = runs[0]["run_id"]
    assert by_path[f"runs/{opaque_run}/scratch/helper.py"]["role"] == "scratch"
    assert by_path[f"runs/{opaque_run}/scratch/analysis.log"]["role"] == "scratch"
    assert by_path[f"runs/{opaque_run}/output/process.md"]["role"] == "output"
    assert "process/prediction_freeze_manifest.json" not in by_path
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"])
    for relative in (
        "process/leakage_preflight.md",
        "process/validation_report.json",
        "src/pipeline.py",
        f"runs/{opaque_run}/scratch/helper.py",
    ):
        assert by_path[relative]["sha256"] == hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest()
    assert (root / "process" / "prediction_freeze_manifest.json").is_file()
    assert "[x] All 72 outputs validated" in (
        root / "process" / "leakage_preflight.md"
    ).read_text(encoding="utf-8")
