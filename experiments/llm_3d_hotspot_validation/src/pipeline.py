"""Label-blind preparation, validation, and prediction-freeze pipeline.

This module is intentionally unable to score predictions.  It prepares opaque
inputs, validates only their declared JSON contract and local residue universe,
and hashes artifacts before any label file is permitted to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

try:  # Support both package imports and the experiment's direct-src tests.
    from . import structure
except ImportError:  # pragma: no cover - exercised by direct-src imports
    import structure  # type: ignore


EXPECTED_RUNS = 72
CONDITIONS = (
    "named_no_web",
    "anonymous_no_web",
    "anonymous_generic_packet",
)
REPLICATES = (1, 2, 3)
_OPAQUE_RUN_RE = re.compile(r"^run_[0-9a-f]{20}$")
_OPAQUE_RUN_TOKEN_RE = re.compile(r"\brun_[0-9a-f]{20}\b")
_LOCAL_TOKEN_RE = re.compile(r"^T[1-9][0-9]*:[1-9][0-9]*$")
_LABEL_FILE_RE = re.compile(
    r"(?:^|[_.-])(?:labels?|ground[_.-]?truth|hotspot[_.-]?truth)(?:[_.-]|$)",
    re.IGNORECASE,
)
_LABEL_DATA_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pickle",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


class BenchmarkStateError(RuntimeError):
    """Raised when a stage would violate the benchmark freeze protocol."""


@dataclass(frozen=True)
class PreparationResult:
    run_count: int
    run_plan_path: Path
    qc_path: Path
    checksums_path: Path


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_label_files(experiment_root: str | Path) -> tuple[Path, ...]:
    """Find label-like data files by name without opening their contents."""

    root = Path(experiment_root).resolve()
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() in _LABEL_DATA_SUFFIXES and _LABEL_FILE_RE.search(path.name):
            matches.append(path)
    return tuple(sorted(matches, key=lambda item: item.as_posix()))


def assert_no_label_files(experiment_root: str | Path) -> None:
    matches = find_label_files(experiment_root)
    if matches:
        relative = [str(path.relative_to(Path(experiment_root).resolve())) for path in matches]
        raise BenchmarkStateError(
            "label-blind gate failed; remove label files before this stage: "
            + ", ".join(relative)
        )


def _opaque_digest(*parts: object, length: int = 20) -> str:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:length]


def build_run_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the deterministic Cartesian run plan using opaque task paths."""

    targets = manifest.get("targets")
    conditions = manifest.get("conditions")
    replicates = manifest.get("replicates")
    base_seed = manifest.get("base_seed")
    if not isinstance(targets, list) or not targets:
        raise ValueError("manifest targets must be a non-empty list")
    if tuple(conditions or ()) != CONDITIONS:
        raise ValueError(f"manifest conditions must be exactly {list(CONDITIONS)!r}")
    if tuple(replicates or ()) != REPLICATES:
        raise ValueError(f"manifest replicates must be exactly {list(REPLICATES)!r}")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("manifest base_seed must be an integer")

    case_ids: list[str] = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise ValueError("every manifest target must be an object")
        case_id = target.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every target needs a non-empty case_id")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")

    runs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for case_id in case_ids:
        for condition in CONDITIONS:
            for replicate in REPLICATES:
                digest = _opaque_digest(base_seed, case_id, condition, replicate)
                run_id = f"run_{digest}"
                if run_id in seen_ids:
                    raise ValueError("opaque run-id collision")
                seen_ids.add(run_id)
                runs.append(
                    {
                        "run_id": run_id,
                        "task_name": f"task_{_opaque_digest('task', digest)}",
                        "task_path": f"runs/{run_id}",
                        "case_id": case_id,
                        "condition": condition,
                        "replicate": replicate,
                        "variant": replicate,
                    }
                )
    return {
        "schema_version": "1.0",
        "expected_run_count": len(runs),
        "base_seed": base_seed,
        "runs": runs,
    }


_CONDITION_ADDENDA = {
    "named_no_web": """## Assigned condition: named_no_web

Read `identity_card.json`, which is the only assigned identity-bearing artifact.
You may use learned parametric knowledge but must not browse or query a database.
Return local residue tokens only.
""",
    "anonymous_no_web": """## Assigned condition: anonymous_no_web

No identity card or methods packet is assigned. Do not infer, name, or search for
the target identity. Use only the anonymous structure and derived features.
""",
    "anonymous_generic_packet": """## Assigned condition: anonymous_generic_packet

Read `generic_knowledge_packet.md` as the frozen target-agnostic method. No
identity card is assigned, and no live lookup or identity inference is allowed.
""",
}


def render_run_prompt(
    template_text: str, *, case_id: str, condition: str, replicate: int
) -> str:
    """Append assignment fields and the frozen condition addendum to a prompt."""

    if condition not in _CONDITION_ADDENDA:
        raise ValueError(f"unknown benchmark condition {condition!r}")
    if replicate not in REPLICATES:
        raise ValueError(f"invalid replicate {replicate!r}")
    assignment = (
        "\n\n## Opaque assignment\n\n"
        f"- `case_id`: `{case_id}`\n"
        f"- `condition`: `{condition}`\n"
        f"- `replicate`: `{replicate}`\n\n"
        "Copy these exact values into `prediction.json`.\n\n"
    )
    return template_text.rstrip() + assignment + _CONDITION_ADDENDA[condition].rstrip() + "\n"


def create_run_prompt(
    template_path: str | Path,
    destination: str | Path,
    *,
    case_id: str,
    condition: str,
    replicate: int,
) -> Path:
    destination_path = Path(destination)
    _write_text(
        destination_path,
        render_run_prompt(
            Path(template_path).read_text(encoding="utf-8"),
            case_id=case_id,
            condition=condition,
            replicate=replicate,
        ),
    )
    return destination_path


def _auth_ranges(target: Mapping[str, Any]) -> tuple[structure.AuthResidueRange, ...]:
    contexts = target.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"target {target.get('case_id')!r} has no contexts")
    ranges: list[structure.AuthResidueRange] = []
    for context in contexts:
        if not isinstance(context, Mapping):
            raise ValueError("target contexts must be objects")
        ranges.append(
            structure.AuthResidueRange(
                str(context["auth_chain"]),
                int(context["start"]),
                int(context["end"]),
                str(context.get("start_insertion_code", "")),
                str(context.get("end_insertion_code", "")),
            )
        )
    return tuple(ranges)


def _identity_card(target: Mapping[str, Any], mapping: structure.LocalMapping) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "case_id": target["case_id"],
        "target_name": target["target_name"],
        "pdb_id": target["pdb_id"],
        "contexts": target["contexts"],
        "local_residue_mapping": [item.to_dict() for item in mapping.residues],
    }


def _source_terms(target: Mapping[str, Any], mapping: structure.LocalMapping) -> set[str]:
    terms = {str(target["target_name"]).casefold(), str(target["pdb_id"]).casefold()}
    for residue in mapping.residues:
        for value in (residue.auth_asym_id, residue.label_asym_id):
            if len(value) >= 3:  # one-letter chain IDs are not safe text sentinels
                terms.add(value.casefold())
    return {term for term in terms if term}


def _scan_anonymous_input(input_dir: Path, forbidden_terms: Iterable[str]) -> list[str]:
    findings: list[str] = []
    identity_fields = (
        "_atom_site.auth_",
        "_entry.",
        "_struct.title",
        "_entity.",
        "_citation.",
        "_database_",
        "_entity_src_",
        "_pdbx_entity_src_",
    )
    terms = tuple(sorted(set(forbidden_terms)))
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        text = path.read_text(encoding="utf-8").casefold()
        for term in terms:
            if term in text:
                findings.append(f"{path.name}: source identity token {term!r}")
        if path.name in {"structure.cif", "features.json"}:
            for marker in identity_fields:
                if marker.casefold() in text:
                    findings.append(f"{path.name}: source metadata marker {marker!r}")
    return findings


def _variant_seed(base_seed: int, case_id: str, replicate: int) -> int:
    return int(_opaque_digest("rigid", base_seed, case_id, replicate, length=16), 16)


def _copy_common_code(root: Path) -> list[Path]:
    common_dir = root / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    destination = common_dir / "structure.py"
    shutil.copyfile(Path(structure.__file__).resolve(), destination)
    init = common_dir / "__init__.py"
    _write_text(init, "\"\"\"Frozen, identity-free structure inspection code.\"\"\"\n")
    return [init, destination]


def prepare_benchmark(
    experiment_root: str | Path, *, sphere_points: int = 960
) -> PreparationResult:
    """Prepare all target variants and 72 isolated run directories without labels."""

    root = Path(experiment_root).resolve()
    process = root / "process"
    assert_no_label_files(root)
    if sphere_points < 4:
        raise ValueError("sphere_points must be at least 4")
    manifest = _read_json(process / "target_manifest.json")
    if not isinstance(manifest, Mapping):
        raise ValueError("target manifest must be a JSON object")
    plan = build_run_plan(manifest)
    if plan["expected_run_count"] != EXPECTED_RUNS:
        raise BenchmarkStateError(
            f"frozen benchmark requires exactly {EXPECTED_RUNS} runs, got "
            f"{plan['expected_run_count']}"
        )

    # Re-preparation is allowed before predictions exist, but never after dispatch.
    existing_predictions = tuple((root / "runs").glob("*/output/prediction.json"))
    if existing_predictions:
        raise BenchmarkStateError("refusing to rewrite assigned inputs after predictions exist")

    prompt_template = process / "prompt_template.md"
    prediction_schema = process / "prediction_schema.json"
    generic_packet = process / "generic_knowledge_packet.md"
    for required in (prompt_template, prediction_schema, generic_packet):
        if not required.is_file():
            raise FileNotFoundError(required)
    common_paths = _copy_common_code(root)

    prepared_root = process / "prepared"
    case_records: dict[str, dict[str, Any]] = {}
    qc_rows: list[dict[str, Any]] = []
    for target in manifest["targets"]:
        case_id = str(target["case_id"])
        pdb_id = str(target["pdb_id"])
        raw_cif = process / "raw_cif" / f"{pdb_id.lower()}.cif"
        if not raw_cif.is_file():
            raise FileNotFoundError(raw_cif)
        parsed = structure.parse_mmcif_file(raw_cif)
        cropped = structure.crop_by_auth_ranges(parsed, _auth_ranges(target))
        mapping = structure.make_local_mapping(cropped)
        case_dir = prepared_root / case_id
        private_dir = case_dir / "private"
        variants_dir = case_dir / "variants"
        mapping_path = private_dir / "local_mapping.json"
        identity_path = private_dir / "identity_card.json"
        _write_json(mapping_path, mapping.to_dict())
        _write_json(identity_path, _identity_card(target, mapping))
        source_terms = _source_terms(target, mapping)
        variants: dict[int, dict[str, Any]] = {}

        for replicate in REPLICATES:
            seed = _variant_seed(int(manifest["base_seed"]), case_id, replicate)
            rigid = structure.seeded_rigid_transform(seed, translation_scale=25.0)
            transformed = structure.transform_residues(cropped, rigid)
            variant_dir = variants_dir / f"v{replicate}"
            structure_path = variant_dir / "structure.cif"
            features_path = variant_dir / "features.json"
            cif_text, _ = structure.local_mmcif_text(transformed, mapping)
            feature_text = structure.identity_free_feature_json(
                transformed, sphere_points=sphere_points
            )
            _write_text(structure_path, cif_text)
            _write_text(features_path, feature_text + "\n")
            variant_checksums = {
                "structure.cif": structure.sha256_file(structure_path),
                "features.json": structure.sha256_file(features_path),
            }
            checksums_path = variant_dir / "checksums.json"
            _write_json(checksums_path, variant_checksums)
            determinant = float(__import__("numpy").linalg.det(rigid.rotation))
            qc = {
                "schema_version": "1.0",
                "case_id": case_id,
                "variant": replicate,
                "residue_count": len(cropped),
                "atom_count": sum(len(item.atoms) for item in cropped),
                "local_chain_count": len({item.local_chain_id for item in mapping.residues}),
                "rigid_seed": seed,
                "rotation_determinant": round(determinant, 12),
                "source_identity_absent": not _scan_anonymous_input(
                    variant_dir, source_terms
                ),
                "checksums": variant_checksums,
            }
            if not qc["source_identity_absent"]:
                raise BenchmarkStateError(f"source metadata leakage in {case_id} variant {replicate}")
            qc_path = variant_dir / "qc.json"
            _write_json(qc_path, qc)
            variants[replicate] = {
                "structure": structure_path,
                "features": features_path,
                "qc": qc_path,
                "checksums": checksums_path,
            }
            qc_rows.append(qc)
        case_records[case_id] = {
            "mapping": mapping_path,
            "identity": identity_path,
            "source_terms": source_terms,
            "variants": variants,
        }

    run_ids = {item["run_id"] for item in plan["runs"]}
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        item.name for item in runs_root.iterdir() if item.is_dir() and item.name not in run_ids
    )
    if unexpected:
        raise BenchmarkStateError("unexpected pre-existing run directories: " + ", ".join(unexpected))

    for run in plan["runs"]:
        run_dir = root / run["task_path"]
        if not _OPAQUE_RUN_RE.fullmatch(run_dir.name):
            raise AssertionError("non-opaque generated run path")
        input_dir = run_dir / "input"
        scratch_dir = run_dir / "scratch"
        output_dir = run_dir / "output"
        for directory in (input_dir, scratch_dir, output_dir):
            directory.mkdir(parents=True, exist_ok=True)
        case = case_records[run["case_id"]]
        variant = case["variants"][run["variant"]]
        shutil.copyfile(variant["structure"], input_dir / "structure.cif")
        shutil.copyfile(variant["features"], input_dir / "features.json")
        shutil.copyfile(prediction_schema, input_dir / "prediction_schema.json")
        create_run_prompt(
            prompt_template,
            input_dir / "prompt.md",
            case_id=run["case_id"],
            condition=run["condition"],
            replicate=run["replicate"],
        )
        optional = {
            "identity_card.json": run["condition"] == "named_no_web",
            "generic_knowledge_packet.md": run["condition"] == "anonymous_generic_packet",
        }
        for filename, wanted in optional.items():
            path = input_dir / filename
            if not wanted and path.exists():
                path.unlink()
        if optional["identity_card.json"]:
            shutil.copyfile(case["identity"], input_dir / "identity_card.json")
        if optional["generic_knowledge_packet.md"]:
            shutil.copyfile(generic_packet, input_dir / "generic_knowledge_packet.md")
        if run["condition"].startswith("anonymous_"):
            findings = _scan_anonymous_input(input_dir, case["source_terms"])
            if findings:
                raise BenchmarkStateError(
                    f"anonymous input leakage for {run['run_id']}: " + "; ".join(findings)
                )

    run_plan_path = process / "run_plan.json"
    _write_json(run_plan_path, plan)
    qc_document = {
        "schema_version": "1.0",
        "label_files_absent": True,
        "expected_variants": len(manifest["targets"]) * len(REPLICATES),
        "passed_variants": sum(bool(row["source_identity_absent"]) for row in qc_rows),
        "variants": qc_rows,
    }
    qc_json_path = process / "structure_qc.json"
    _write_json(qc_json_path, qc_document)
    _write_structure_qc_markdown(process / "structure_qc.md", qc_document)

    checksum_paths: list[Path] = [run_plan_path, qc_json_path, *common_paths]
    checksum_paths.extend(path for path in prepared_root.rglob("*") if path.is_file())
    checksum_paths.extend(path for path in runs_root.glob("*/input/*") if path.is_file())
    checksums = {
        path.relative_to(root).as_posix(): structure.sha256_file(path)
        for path in sorted(set(checksum_paths), key=lambda item: item.as_posix())
    }
    checksums_path = process / "preparation_checksums.json"
    _write_json(checksums_path, {"schema_version": "1.0", "artifacts": checksums})
    _update_preflight(process / "leakage_preflight.md", prepared=True, frozen=False)
    assert_no_label_files(root)
    return PreparationResult(EXPECTED_RUNS, run_plan_path, qc_json_path, checksums_path)


def _write_structure_qc_markdown(path: Path, document: Mapping[str, Any]) -> None:
    lines = [
        "# Structure Preparation QC",
        "",
        f"- Label files absent: `{str(document['label_files_absent']).lower()}`",
        f"- Variants passed: `{document['passed_variants']}/{document['expected_variants']}`",
        "",
        "| Opaque case | Variant | Residues | Atoms | Chains | Identity-free |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in document["variants"]:
        lines.append(
            f"| `{row['case_id']}` | {row['variant']} | {row['residue_count']} | "
            f"{row['atom_count']} | {row['local_chain_count']} | "
            f"{'pass' if row['source_identity_absent'] else 'FAIL'} |"
        )
    _write_text(path, "\n".join(lines) + "\n")


def _update_preflight(
    path: Path,
    *,
    prepared: bool,
    frozen: bool,
    terminal_failures: int = 0,
    excluded_predictions: int = 0,
) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if prepared:
        for phrase in (
            "Target-only anonymous structures generated and metadata scanner passed.",
            "Local residue mapping retained outside assigned anonymous inputs.",
            "Generic packet has no benchmark names, PDB IDs, partners, fingerprints, or",
            "All prompts and common code hashed.",
            "Ground-truth label file absent immediately before first dispatch.",
        ):
            text = text.replace(f"- [ ] {phrase}", f"- [x] {phrase}")
    if frozen:
        old_line = "- [ ] All 72 outputs validated and frozen before label creation."
        previous_terminal_line = (
            "- [x] All 72 runs reached a validated terminal outcome and were "
            "frozen before label creation; documented refusals were not imputed."
        )
        if terminal_failures and excluded_predictions:
            terminal_line = (
                "- [x] All 72 runs reached a validated terminal outcome and were "
                "frozen before label creation; documented refusals were not imputed "
                "and excluded predictions were retained unchanged."
            )
            text = text.replace(old_line, terminal_line)
            text = text.replace(previous_terminal_line, terminal_line)
        elif terminal_failures:
            text = text.replace(old_line, previous_terminal_line)
        elif excluded_predictions:
            terminal_line = (
                "- [x] All 72 runs reached a validated terminal outcome and were "
                "frozen before label creation; excluded predictions were retained "
                "unchanged."
            )
            text = text.replace(old_line, terminal_line)
            text = text.replace(previous_terminal_line, terminal_line)
        else:
            text = text.replace(old_line, old_line.replace("[ ]", "[x]"))
    _write_text(path, text)


def _json_type_matches(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _schema_errors(value: object, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Validate the JSON-Schema keywords used by the frozen experiment schema."""

    errors: list[str] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _json_type_matches(value, expected_type):
        return [f"{path}: expected {expected_type}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is not in the allowed enum")
    pattern = schema.get("pattern")
    if isinstance(value, str) and isinstance(pattern, str) and re.search(pattern, value) is None:
        errors.append(f"{path}: does not match required pattern {pattern!r}")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors.extend(_schema_errors(item, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: additional property {key!r} is not allowed")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: permits at most {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True, allow_nan=False) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: items must be unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(_schema_errors(item, item_schema, f"{path}[{index}]"))
    return errors


def validate_prediction_payload(
    payload: object,
    schema: Mapping[str, Any],
    allowed_tokens: Iterable[str],
    *,
    expected_case_id: str,
    expected_condition: str,
    expected_replicate: int,
) -> tuple[str, ...]:
    errors = _schema_errors(payload, schema)
    if not isinstance(payload, Mapping):
        return tuple(errors)
    for key, expected in (
        ("case_id", expected_case_id),
        ("condition", expected_condition),
        ("replicate", expected_replicate),
    ):
        if payload.get(key) != expected:
            errors.append(f"$.{key}: does not match this opaque assignment")
    universe = set(allowed_tokens)
    if not universe or any(not _LOCAL_TOKEN_RE.fullmatch(token) for token in universe):
        errors.append("assigned features contain an invalid local-token universe")
    ranked: list[object] = []
    for field in ("primary_hotspots", "alternate_hotspots"):
        value = payload.get(field)
        if isinstance(value, list):
            ranked.extend(value)
    if len(ranked) == 6 and len(set(item for item in ranked if isinstance(item, str))) != 6:
        errors.append("primary and alternate hotspots must be six distinct tokens")
    token_values = list(ranked)
    pocket_groups = payload.get("pocket_groups")
    if isinstance(pocket_groups, list):
        for group in pocket_groups:
            if isinstance(group, list):
                token_values.extend(group)
    for token in token_values:
        if isinstance(token, str) and token not in universe:
            errors.append(f"residue token {token!r} is not present in assigned features")
    return tuple(dict.fromkeys(errors))


def _allowed_tokens(input_dir: Path) -> set[str]:
    features = _read_json(input_dir / "features.json")
    if not isinstance(features, Mapping) or not isinstance(features.get("residues"), list):
        raise ValueError("assigned features.json has no residues array")
    tokens = {
        item.get("token")
        for item in features["residues"]
        if isinstance(item, Mapping) and isinstance(item.get("token"), str)
    }
    if len(tokens) != len(features["residues"]):
        raise ValueError("assigned features contain missing or duplicate residue tokens")
    return tokens


def _run_documents(
    process_dir: Path, directory_name: str, planned_run_ids: set[str]
) -> tuple[dict[str, Path], list[str]]:
    """Index strictly named, direct per-run documents without accepting extras."""

    documents_dir = process_dir / directory_name
    document_label = directory_name[:-1] if directory_name.endswith("s") else directory_name
    if not documents_dir.exists():
        return {}, []
    if documents_dir.is_symlink() or not documents_dir.is_dir():
        return {}, [f"process/{directory_name} must be a real directory"]

    documents: dict[str, Path] = {}
    errors: list[str] = []
    for path in sorted(documents_dir.iterdir(), key=lambda item: item.name):
        relative = path.relative_to(process_dir.parent).as_posix()
        if path.is_symlink() or not path.is_file():
            errors.append(f"unexpected {document_label}-document entry: {relative}")
            continue
        run_id = path.stem
        if path.name != f"{run_id}.md" or run_id not in planned_run_ids:
            errors.append(
                f"{document_label} document must be named for exactly one planned "
                f"run ID: {relative}"
            )
            continue
        documents[run_id] = path
    return documents, errors


def validate_benchmark(experiment_root: str | Path) -> dict[str, Any]:
    """Validate one eligible, excluded, or failed terminal outcome per run."""

    root = Path(experiment_root).resolve()
    process = root / "process"
    assert_no_label_files(root)
    plan = _read_json(process / "run_plan.json")
    schema = _read_json(process / "prediction_schema.json")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("runs"), list):
        raise BenchmarkStateError("run_plan.json is malformed")
    if not isinstance(schema, Mapping):
        raise BenchmarkStateError("prediction_schema.json is malformed")
    runs = plan["runs"]
    run_ids = [item.get("run_id") for item in runs if isinstance(item, Mapping)]
    plan_error: list[str] = []
    if len(runs) != EXPECTED_RUNS or plan.get("expected_run_count") != EXPECTED_RUNS:
        plan_error.append(f"run plan must contain exactly {EXPECTED_RUNS} runs")
    opaque_run_ids = [
        run_id
        for run_id in run_ids
        if isinstance(run_id, str) and _OPAQUE_RUN_RE.fullmatch(run_id)
    ]
    if len(opaque_run_ids) != len(run_ids) or len(opaque_run_ids) != len(
        set(opaque_run_ids)
    ):
        plan_error.append("run IDs must be unique and opaque")
    planned_run_ids = set(opaque_run_ids)
    actual_dirs = (
        {item.name for item in (root / "runs").iterdir() if item.is_dir()}
        if (root / "runs").is_dir()
        else set()
    )
    if actual_dirs != planned_run_ids:
        plan_error.append("run directory set does not exactly match run_plan.json")
    failure_documents, failure_errors = _run_documents(
        process, "failures", planned_run_ids
    )
    plan_error.extend(failure_errors)
    exclusion_documents, exclusion_errors = _run_documents(
        process, "exclusions", planned_run_ids
    )
    plan_error.extend(exclusion_errors)

    records: list[dict[str, Any]] = []
    eligible_predictions = 0
    excluded_predictions = 0
    terminal_failures = 0
    unaccounted = 0
    dual_outcome = 0
    for run in runs:
        if not isinstance(run, Mapping):
            records.append(
                {
                    "run_id": None,
                    "outcome": "unaccounted",
                    "valid": False,
                    "terminal": False,
                    "errors": ["invalid run record"],
                }
            )
            unaccounted += 1
            continue
        run_id = str(run.get("run_id", ""))
        run_dir = root / "runs" / run_id
        errors: list[str] = []
        prediction_path = run_dir / "output" / "prediction.json"
        process_path = run_dir / "output" / "process.md"
        failure_path = failure_documents.get(run_id)
        exclusion_path = exclusion_documents.get(run_id)
        prediction_present = prediction_path.exists() or prediction_path.is_symlink()
        failure_present = failure_path is not None
        exclusion_present = exclusion_path is not None
        failure_document_errors: list[str] = []
        if failure_present:
            try:
                failure_text = failure_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                failure_document_errors.append(f"failure document is unreadable: {exc}")
            else:
                if not failure_text.strip():
                    failure_document_errors.append("failure document must be non-empty")
                mentioned_run_ids = set(_OPAQUE_RUN_TOKEN_RE.findall(failure_text))
                if mentioned_run_ids != {run_id}:
                    failure_document_errors.append(
                        "failure document must identify exactly its matching run_id"
                    )
        exclusion_document_errors: list[str] = []
        if exclusion_present:
            try:
                exclusion_text = exclusion_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                exclusion_document_errors.append(
                    f"exclusion document is unreadable: {exc}"
                )
            else:
                if not exclusion_text.strip():
                    exclusion_document_errors.append(
                        "exclusion document must be non-empty"
                    )
                mentioned_run_ids = set(_OPAQUE_RUN_TOKEN_RE.findall(exclusion_text))
                if mentioned_run_ids != {run_id}:
                    exclusion_document_errors.append(
                        "exclusion document must identify exactly its matching run_id"
                    )

        if failure_present and exclusion_present:
            outcome = "dual_outcome"
            dual_outcome += 1
            errors.extend(failure_document_errors)
            errors.extend(exclusion_document_errors)
            errors.append(
                "process/failures and process/exclusions cannot both record the same run"
            )
        elif prediction_present and failure_present:
            outcome = "dual_outcome"
            dual_outcome += 1
            errors.extend(failure_document_errors)
            errors.append(
                "process/failures failure document cannot coexist with "
                "output/prediction.json"
            )
        elif failure_present:
            errors.extend(failure_document_errors)
            if errors:
                outcome = "unaccounted"
                unaccounted += 1
            else:
                outcome = "terminal_failure"
                terminal_failures += 1
        elif prediction_present:
            if prediction_path.is_symlink() or not prediction_path.is_file():
                errors.append("output/prediction.json must be a real file")
            else:
                try:
                    payload = _read_json(prediction_path)
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    errors.append(f"output/prediction.json is not valid JSON: {exc}")
                else:
                    try:
                        tokens = _allowed_tokens(run_dir / "input")
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        errors.append(f"assigned feature validation failed: {exc}")
                    else:
                        try:
                            expected_replicate = int(run.get("replicate", -1))
                        except (TypeError, ValueError):
                            errors.append("run plan replicate is invalid")
                        else:
                            errors.extend(
                                validate_prediction_payload(
                                    payload,
                                    schema,
                                    tokens,
                                    expected_case_id=str(run.get("case_id")),
                                    expected_condition=str(run.get("condition")),
                                    expected_replicate=expected_replicate,
                                )
                            )
            if process_path.is_symlink():
                errors.append("output/process.md must be a real file")
            elif not process_path.is_file():
                errors.append("output/process.md is missing")
            else:
                try:
                    process_text = process_path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    errors.append(f"output/process.md is unreadable: {exc}")
                else:
                    if not process_text.strip():
                        errors.append("output/process.md must be non-empty")
            errors.extend(exclusion_document_errors)
            if errors:
                outcome = "unaccounted"
                unaccounted += 1
            elif exclusion_present:
                outcome = "excluded_prediction"
                excluded_predictions += 1
            else:
                outcome = "prediction"
                eligible_predictions += 1
        else:
            outcome = "unaccounted"
            unaccounted += 1
            errors.extend(exclusion_document_errors)
            if exclusion_present:
                errors.append(
                    "process/exclusions requires the original output/prediction.json"
                )
            errors.append(
                "no terminal outcome: provide a valid prediction plus process.md "
                f"or process/failures/{run_id}.md"
            )
        records.append(
            {
                "run_id": run_id,
                "outcome": outcome,
                "valid": outcome in {"prediction", "excluded_prediction"},
                "eligible": outcome == "prediction",
                "terminal": outcome
                in {"prediction", "excluded_prediction", "terminal_failure"},
                "errors": errors,
            }
        )

    valid_predictions = eligible_predictions + excluded_predictions
    terminal_outcomes = valid_predictions + terminal_failures
    complete_record_set = len(records) == EXPECTED_RUNS
    all_valid_predictions = (
        not plan_error
        and complete_record_set
        and valid_predictions == EXPECTED_RUNS
        and terminal_failures == 0
        and unaccounted == 0
        and dual_outcome == 0
    )
    all_terminal = (
        not plan_error
        and complete_record_set
        and terminal_outcomes == EXPECTED_RUNS
        and unaccounted == 0
        and dual_outcome == 0
    )
    report = {
        "schema_version": "1.2",
        "validated_at": _utc_now(),
        "label_files_absent": True,
        "expected_runs": EXPECTED_RUNS,
        "terminal_outcomes": terminal_outcomes,
        "valid_predictions": valid_predictions,
        "eligible_predictions": eligible_predictions,
        "excluded_predictions": excluded_predictions,
        "terminal_failures": terminal_failures,
        "unaccounted": unaccounted,
        "dual_outcome": dual_outcome,
        "all_valid_predictions": all_valid_predictions,
        "all_terminal": all_terminal,
        # Compatibility aliases for the original all-prediction report.
        "expected_artifacts": EXPECTED_RUNS,
        "successful_artifacts": valid_predictions,
        "all_valid": all_valid_predictions,
        "plan_errors": plan_error,
        "runs": records,
    }
    _write_json(process / "validation_report.json", report)
    return report


def freeze_predictions(experiment_root: str | Path) -> dict[str, Any]:
    """Freeze the state after all 72 runs have a validated terminal outcome."""

    root = Path(experiment_root).resolve()
    process = root / "process"
    assert_no_label_files(root)
    report = validate_benchmark(root)
    if not report["all_terminal"] or report["terminal_outcomes"] != EXPECTED_RUNS:
        raise BenchmarkStateError(
            "freeze refused: "
            f"{report['terminal_outcomes']}/{EXPECTED_RUNS} terminal outcomes; "
            f"valid_predictions={report['valid_predictions']}, "
            f"eligible_predictions={report['eligible_predictions']}, "
            f"excluded_predictions={report['excluded_predictions']}, "
            f"terminal_failures={report['terminal_failures']}, "
            f"unaccounted={report['unaccounted']}, "
            f"dual_outcome={report['dual_outcome']}"
        )
    # This is the last process-file mutation before hashing. Nothing under
    # process/ may change after the complete snapshot below is calculated.
    _update_preflight(
        process / "leakage_preflight.md",
        prepared=True,
        frozen=True,
        terminal_failures=report["terminal_failures"],
        excluded_predictions=report["excluded_predictions"],
    )
    plan = _read_json(process / "run_plan.json")
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    freeze_path = process / "prediction_freeze_manifest.json"

    def add(path: Path, role: str, run_id: str | None = None) -> None:
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            return
        seen.add(relative)
        record: dict[str, Any] = {
            "path": relative,
            "role": role,
            "sha256": structure.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if run_id is not None:
            record["run_id"] = run_id
        artifacts.append(record)

    def tree_files(directory: Path) -> list[Path]:
        files: list[Path] = []
        if not directory.is_dir():
            return files
        for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink():
                raise BenchmarkStateError(
                    f"freeze refused: symlink cannot be included safely: "
                    f"{path.relative_to(root).as_posix()}"
                )
            if path.is_file():
                files.append(path)
        return files

    for run in plan["runs"]:
        run_id = run["run_id"]
        run_dir = root / "runs" / run_id
        for path in tree_files(run_dir / "input"):
            add(path, "prompt" if path.name == "prompt.md" else "input", run_id)
        for path in tree_files(run_dir / "scratch"):
            add(path, "scratch", run_id)
        for path in tree_files(run_dir / "output"):
            add(path, "output", run_id)
    for path in tree_files(root / "common"):
        add(path, "common_code")

    # The process tree is the private experiment record.  It includes raw CIFs,
    # private local mappings, manifests, preregistration/audits, generic material,
    # and preparation/validation QC.  The manifest being constructed is the sole
    # exclusion so that it does not attempt to hash itself (including on re-freeze).
    for path in tree_files(process):
        if path.resolve() != freeze_path.resolve():
            add(path, "process")

    # Freeze the exact implementation and preregistered scoring/test surface.
    for path in tree_files(root / "src"):
        add(path, "source_code")
    for path in tree_files(root / "tests"):
        add(path, "test_code")
    for path, role in (
        (root / "run_benchmark.py", "entrypoint_code"),
        (root / "__init__.py", "experiment_code"),
    ):
        if path.is_symlink():
            raise BenchmarkStateError(
                f"freeze refused: symlink cannot be included safely: {path.name}"
            )
        if path.is_file():
            add(path, role)
    if not any(item["role"] == "common_code" for item in artifacts):
        raise BenchmarkStateError("freeze refused: no frozen common-code artifact exists")
    prediction_artifact_count = sum(
        item["role"] == "output" and item["path"].endswith("prediction.json")
        for item in artifacts
    )
    if prediction_artifact_count != report["valid_predictions"]:
        raise BenchmarkStateError(
            "freeze refused: frozen prediction artifact count does not match validation"
        )

    run_outcomes: list[dict[str, Any]] = []
    for record in report["runs"]:
        run_id = record["run_id"]
        outcome = record["outcome"]
        if outcome == "prediction":
            run_outcomes.append(
                {
                    "run_id": run_id,
                    "outcome": outcome,
                    "prediction_path": f"runs/{run_id}/output/prediction.json",
                    "process_path": f"runs/{run_id}/output/process.md",
                }
            )
        elif outcome == "excluded_prediction":
            run_outcomes.append(
                {
                    "run_id": run_id,
                    "outcome": outcome,
                    "exclusion_reason": "predefined_compliance_exclusion",
                    "prediction_path": f"runs/{run_id}/output/prediction.json",
                    "process_path": f"runs/{run_id}/output/process.md",
                    "exclusion_path": f"process/exclusions/{run_id}.md",
                }
            )
        elif outcome == "terminal_failure":
            run_outcomes.append(
                {
                    "run_id": run_id,
                    "outcome": outcome,
                    "failure_path": f"process/failures/{run_id}.md",
                }
            )
        else:  # all_terminal above makes this an internal consistency failure.
            raise BenchmarkStateError(
                f"freeze refused: non-terminal validation record for {run_id}"
            )

    manifest = {
        "schema_version": "1.2",
        "frozen_at": _utc_now(),
        "expected_runs": EXPECTED_RUNS,
        "validated_runs": report["terminal_outcomes"],
        "terminal_outcomes": report["terminal_outcomes"],
        "expected_predictions": EXPECTED_RUNS,
        "validated_predictions": report["valid_predictions"],
        "eligible_predictions": report["eligible_predictions"],
        "excluded_predictions": report["excluded_predictions"],
        "terminal_failures": report["terminal_failures"],
        "all_terminal": True,
        "labels_absent": True,
        "label_file_search": {"root": ".", "matches": []},
        "runs": run_outcomes,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    _write_json(freeze_path, manifest)
    assert_no_label_files(root)
    return manifest


# Short stage names form the public API used by the CLI and tests.
prepare = prepare_benchmark
validate = validate_benchmark
freeze = freeze_predictions


__all__ = [
    "BenchmarkStateError",
    "CONDITIONS",
    "EXPECTED_RUNS",
    "PreparationResult",
    "REPLICATES",
    "assert_no_label_files",
    "build_run_plan",
    "create_run_prompt",
    "find_label_files",
    "freeze",
    "freeze_predictions",
    "prepare",
    "prepare_benchmark",
    "render_run_prompt",
    "validate",
    "validate_benchmark",
    "validate_prediction_payload",
]
