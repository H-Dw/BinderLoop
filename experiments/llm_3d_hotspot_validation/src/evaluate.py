"""Post-freeze evaluation and reporting for the blinded hotspot benchmark.

This module contains no target identities or labels.  It intentionally refuses
to open the label file until the prediction freeze has been validated against
the run plan and every frozen prediction hash.  The supported JSON contracts
are documented by the normalisers below and exercised with synthetic fixtures
in ``tests/test_evaluate.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

try:  # Package import.
    from .metrics import (
        PredictionSet,
        consensus_prediction,
        empirical_rsasa_quintiles,
        expand_residue_pocket,
        global_chain_permutation,
        h6_pocket_overlap,
        hierarchical_bootstrap_ci,
        holm_adjust,
        paired_sign_flip_test,
        pocket_distance_metrics,
        standard_distance_tolerances,
        stratified_monte_carlo,
        symmetry_adjusted_top_metrics,
        target_bootstrap_ci,
        validate_prediction_schema,
    )
    from .structure import parse_mmcif_file
except ImportError:  # Direct ``python src/evaluate.py`` execution.
    from metrics import (  # type: ignore
        PredictionSet,
        consensus_prediction,
        empirical_rsasa_quintiles,
        expand_residue_pocket,
        global_chain_permutation,
        h6_pocket_overlap,
        hierarchical_bootstrap_ci,
        holm_adjust,
        paired_sign_flip_test,
        pocket_distance_metrics,
        standard_distance_tolerances,
        stratified_monte_carlo,
        symmetry_adjusted_top_metrics,
        target_bootstrap_ci,
        validate_prediction_schema,
    )
    from structure import parse_mmcif_file  # type: ignore


PRIMARY_CONDITION = "anonymous_no_web"
CONDITIONS = (
    "named_no_web",
    PRIMARY_CONDITION,
    "anonymous_generic_packet",
)
CONTRASTS = (
    ("named_no_web", PRIMARY_CONDITION),
    ("anonymous_generic_packet", PRIMARY_CONDITION),
)
_LOCAL_TOKEN_RE = re.compile(r"^T[1-9][0-9]*:[1-9][0-9]*$")


class EvaluationError(RuntimeError):
    """Raised for a freeze, input-contract, or evaluation-integrity failure."""


@dataclass(frozen=True)
class FrozenRun:
    target_id: str
    condition: str
    replicate: int
    prediction_path: Path | None
    sha256: str | None
    outcome: str = "success"
    raw_outcome: str = "success"
    outcome_path: Path | None = None
    outcome_sha256: str | None = None
    outcome_reason: str | None = None


@dataclass(frozen=True)
class FreezeState:
    manifest_path: Path
    frozen_at: datetime
    runs: tuple[FrozenRun, ...]
    manifest_sha256: str
    verified_artifact_hash_count: int = 0
    terminal_outcome_count: int = 0


@dataclass(frozen=True)
class JointNullResult:
    observed: float
    null_mean: float
    p_greater_equal: float
    draws: int
    seed: int
    null_values: tuple[float, ...]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"invalid JSON in {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as exc:
        raise EvaluationError(f"frozen prediction is missing: {path}") from exc
    return digest.hexdigest()


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"freeze manifest requires non-empty {field}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise EvaluationError(f"invalid ISO-8601 {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise EvaluationError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _run_key(item: Mapping[str, Any]) -> tuple[str, str, int]:
    target = item.get("target_id", item.get("case_id", item.get("case")))
    condition = item.get("condition")
    replicate = item.get("replicate")
    if not isinstance(target, str) or not target:
        raise EvaluationError("every run requires target_id (or case_id)")
    if not isinstance(condition, str) or not condition:
        raise EvaluationError("every run requires condition")
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate < 1:
        raise EvaluationError("every run requires a positive integer replicate")
    return target, condition, replicate


def _resolve(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"missing path field {field}")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _outcome_status(payload: Any) -> tuple[str, str]:
    """Return ``(canonical, recorded)`` for a frozen terminal outcome.

    Terminal-outcome producers have used a few field names.  We accept those
    encodings but never infer success or refusal from the presence/absence of a
    prediction file: the recorded terminal state is the authority.
    """

    value: object = payload
    if isinstance(payload, Mapping):
        for field in ("terminal_outcome", "outcome", "status", "result", "kind"):
            candidate = payload.get(field)
            if isinstance(candidate, Mapping):
                for nested in ("status", "outcome", "result", "kind"):
                    if candidate.get(nested) is not None:
                        candidate = candidate[nested]
                        break
            if candidate is not None:
                value = candidate
                break
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError("terminal outcome requires a recorded status")
    recorded = value.strip()
    normalized = recorded.lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "success", "succeeded", "complete", "completed", "valid", "prediction",
        "prediction_written", "ok",
    }:
        return "success", recorded
    if normalized in {
        "refusal", "refused", "model_refusal", "safety_refusal", "declined",
        "terminal_failure",
    }:
        return "refusal", recorded
    if normalized in {
        "excluded", "excluded_prediction", "prediction_excluded", "ineligible_prediction",
    }:
        return "excluded", recorded
    if normalized in {
        "failure", "failed", "error", "timeout", "timed_out", "cancelled",
        "canceled", "invalid", "no_output", "interrupted",
    }:
        return "failure", recorded
    # Unknown recorded terminal states remain failures rather than being
    # silently promoted to valid predictions.
    return "failure", recorded


def _outcome_reason(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    candidates: list[object] = []
    for field in (
        "exclusion_reason", "reason", "failure_reason", "detail", "details", "message"
    ):
        candidates.append(payload.get(field))
    for field in ("terminal_outcome", "outcome", "result"):
        nested = payload.get(field)
        if isinstance(nested, Mapping):
            for nested_field in (
                "exclusion_reason", "reason", "failure_reason", "detail", "message"
            ):
                candidates.append(nested.get(nested_field))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, (Mapping, list)) and candidate:
            return json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    return None


def _hashed_path(
    base: Path,
    path_value: object,
    hash_value: object,
    *,
    path_field: str,
    context: str,
) -> tuple[Path, str]:
    path = _resolve(base, path_value, path_field)
    if not isinstance(hash_value, str) or len(hash_value) != 64:
        raise EvaluationError(f"{context} requires a SHA-256")
    actual = _sha256_file(path)
    if actual.lower() != hash_value.lower():
        raise EvaluationError(f"frozen artifact hash mismatch for {context}")
    return path, actual.lower()


def _run_asset_paths(
    plan: Mapping[str, Any], plan_file: Path
) -> tuple[Path, Path, Path]:
    """Resolve either explicit synthetic paths or canonical pipeline paths."""

    explicit = (plan.get("mapping_path"), plan.get("features_path"), plan.get("structure_path"))
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise EvaluationError("run-plan asset paths must be supplied together")
        return (
            _resolve(plan_file.parent, explicit[0], "mapping_path"),
            _resolve(plan_file.parent, explicit[1], "features_path"),
            _resolve(plan_file.parent, explicit[2], "structure_path"),
        )
    case_id = plan.get("case_id", plan.get("target_id"))
    task_path = plan.get("task_path")
    if not isinstance(case_id, str) or not isinstance(task_path, str):
        raise EvaluationError("implicit pipeline assets require case_id and task_path")
    experiment_root = plan_file.parent.parent if plan_file.parent.name == "process" else plan_file.parent
    mapping = experiment_root / "process" / "prepared" / case_id / "private" / "local_mapping.json"
    input_dir = experiment_root / task_path / "input"
    return mapping.resolve(), (input_dir / "features.json").resolve(), (input_dir / "structure.cif").resolve()


def _explicit_equivalence_by_target(
    target_manifest_path: Path | None,
) -> dict[str, tuple[tuple[str, ...], ...]] | None:
    """Read preregistered auth-chain groups without reading any label material."""

    if target_manifest_path is None:
        return None
    payload = _read_json(target_manifest_path)
    targets = payload.get("targets") if isinstance(payload, Mapping) else None
    if not isinstance(targets, list):
        raise EvaluationError("target manifest requires targets[]")
    output: dict[str, tuple[tuple[str, ...], ...]] = {}
    for target in targets:
        if not isinstance(target, Mapping):
            raise EvaluationError("target-manifest entries must be objects")
        target_id = target.get("case_id", target.get("target_id"))
        raw_groups = target.get("equivalent_auth_chain_groups", [])
        if not isinstance(target_id, str) or not isinstance(raw_groups, list):
            raise EvaluationError("target equivalence declarations are malformed")
        groups: list[tuple[str, ...]] = []
        for group in raw_groups:
            if not isinstance(group, list) or not all(isinstance(chain, str) for chain in group):
                raise EvaluationError("equivalent_auth_chain_groups must be arrays of strings")
            groups.append(tuple(group))
        output[target_id] = tuple(groups)
    return output


def _target_names_by_id(target_manifest_path: Path | None) -> dict[str, str]:
    """Read report-facing target names from the frozen target manifest."""

    if target_manifest_path is None:
        return {}
    payload = _read_json(target_manifest_path)
    targets = payload.get("targets") if isinstance(payload, Mapping) else None
    if not isinstance(targets, list):
        raise EvaluationError("target manifest requires targets[]")
    names: dict[str, str] = {}
    for target in targets:
        if not isinstance(target, Mapping):
            raise EvaluationError("target-manifest entries must be objects")
        target_id = target.get("case_id", target.get("target_id"))
        if not isinstance(target_id, str) or not target_id:
            raise EvaluationError("target-manifest entries require case_id or target_id")
        name = next(
            (
                target.get(field)
                for field in (
                    "display_name", "target_name", "name", "material_name", "public_name"
                )
                if isinstance(target.get(field), str) and str(target.get(field)).strip()
            ),
            target_id,
        )
        names[target_id] = str(name).strip()
    return names


def verify_prediction_freeze(
    manifest_path: str | Path,
    run_plan_path: str | Path,
) -> tuple[FreezeState, dict[str, Any]]:
    """Validate a legacy prediction freeze or a terminal-outcome freeze.

    A terminal-outcome freeze has one hashed outcome artifact for every planned
    run and a hashed prediction only for outcomes recorded as successful.  The
    legacy all-predictions contract remains supported.  Every manifest artifact
    is hashed before this function returns and therefore before labels can be
    opened by :func:`evaluate_benchmark`.
    """

    manifest_file = Path(manifest_path).resolve()
    plan_file = Path(run_plan_path).resolve()
    manifest = _read_json(manifest_file)
    plan = _read_json(plan_file)
    if not isinstance(manifest, Mapping):
        raise EvaluationError("prediction freeze is absent or not complete; labels remain sealed")
    raw_plan_runs = plan.get("runs") if isinstance(plan, Mapping) else None
    completion_claimed = (
        manifest.get("freeze_complete") is True
        or manifest.get("all_terminal") is True
        or manifest.get("labels_absent") is True
    )
    if not completion_claimed:
        raise EvaluationError("prediction freeze is absent or not complete; labels remain sealed")
    if not isinstance(raw_plan_runs, list) or not raw_plan_runs:
        raise EvaluationError("run plan requires a non-empty runs list")
    compact_runs = manifest.get("runs")
    compact_complete = (
        manifest.get("freeze_complete") is True
        and manifest.get("all_terminal") is not True
        and isinstance(compact_runs, list)
        and bool(compact_runs)
    )
    count_complete = False
    for expected_field, validated_field in (
        ("expected_terminal_outcomes", "validated_terminal_outcomes"),
        ("expected_outcomes", "validated_outcomes"),
        ("expected_runs", "validated_runs"),
        ("expected_predictions", "validated_predictions"),
    ):
        expected_count = manifest.get(expected_field)
        validated_count = manifest.get(validated_field)
        if (
            isinstance(expected_count, int)
            and not isinstance(expected_count, bool)
            and validated_count == expected_count == len(raw_plan_runs)
        ):
            count_complete = True
            break
    expected_runs = manifest.get("expected_runs")
    validated_predictions = manifest.get("validated_predictions")
    terminal_failures = manifest.get("terminal_failures")
    excluded_predictions = manifest.get("excluded_predictions", 0)
    terminal_complete = (
        manifest.get("all_terminal") is True
        and manifest.get("labels_absent") is True
        and isinstance(expected_runs, int)
        and not isinstance(expected_runs, bool)
        and expected_runs == len(raw_plan_runs)
        and isinstance(validated_predictions, int)
        and not isinstance(validated_predictions, bool)
        and isinstance(terminal_failures, int)
        and not isinstance(terminal_failures, bool)
        and validated_predictions + terminal_failures == expected_runs
        and isinstance(excluded_predictions, int)
        and not isinstance(excluded_predictions, bool)
        and 0 <= excluded_predictions <= validated_predictions
        and isinstance(manifest.get("runs"), list)
        and len(manifest["runs"]) == expected_runs
    )
    artifact_complete = (
        isinstance(manifest.get("artifacts"), list)
        and bool(manifest.get("artifacts"))
        and manifest.get("labels_absent") is True
        and (count_complete or manifest.get("freeze_complete") is True or terminal_complete)
    )
    if not compact_complete and not artifact_complete:
        raise EvaluationError("prediction freeze is absent or not complete; labels remain sealed")
    timestamp_value = manifest.get("frozen_at", manifest.get("freeze_completed_at"))
    frozen_at = _parse_timestamp(timestamp_value, "frozen_at")

    plan_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in raw_plan_runs:
        if not isinstance(raw, Mapping):
            raise EvaluationError("run-plan entries must be objects")
        key = _run_key(raw)
        if key in plan_by_key:
            raise EvaluationError(f"duplicate run-plan key: {key}")
        plan_by_key[key] = dict(raw)

    frozen: list[FrozenRun] = []
    frozen_keys: set[tuple[str, str, int]] = set()
    verified_hashes = 0
    if compact_complete:
        for raw in compact_runs:
            if not isinstance(raw, Mapping):
                raise EvaluationError("freeze-manifest run entries must be objects")
            key = _run_key(raw)
            if key in frozen_keys:
                raise EvaluationError(f"duplicate frozen run key: {key}")
            frozen_keys.add(key)
            outcome_path_value = raw.get("outcome_path", raw.get("terminal_outcome_path"))
            outcome_hash_value = raw.get(
                "outcome_sha256", raw.get("terminal_outcome_sha256")
            )
            if outcome_path_value is not None or outcome_hash_value is not None:
                outcome_path, outcome_hash = _hashed_path(
                    manifest_file.parent,
                    outcome_path_value,
                    outcome_hash_value,
                    path_field="outcome_path",
                    context=f"terminal outcome {key}",
                )
                verified_hashes += 1
                outcome_payload = _read_json(outcome_path)
                outcome, raw_outcome = _outcome_status(outcome_payload)
                outcome_reason = _outcome_reason(outcome_payload)
                if raw.get("outcome", raw.get("status")) is not None:
                    declared, _ = _outcome_status(raw)
                    if declared != outcome:
                        raise EvaluationError(
                            f"terminal outcome status mismatch for run {key}"
                        )
            elif raw.get("outcome", raw.get("status")) is not None:
                raise EvaluationError(
                    f"terminal-outcome run {key} requires a hashed outcome artifact"
                )
            else:
                # Backward-compatible all-predictions compact freeze.
                outcome_path = None
                outcome_hash = None
                outcome = raw_outcome = "success"
                outcome_reason = None

            outcome_reason = _outcome_reason(raw) or outcome_reason

            prediction_path: Path | None = None
            prediction_hash: str | None = None
            path_value = raw.get("prediction_path", raw.get("path"))
            expected_hash = raw.get("prediction_sha256", raw.get("sha256"))
            if path_value is not None or expected_hash is not None:
                prediction_path, prediction_hash = _hashed_path(
                    manifest_file.parent,
                    path_value,
                    expected_hash,
                    path_field="prediction_path",
                    context=f"prediction {key}",
                )
                verified_hashes += 1
            if outcome in {"success", "excluded"} and prediction_path is None:
                raise EvaluationError(
                    f"prediction-bearing terminal outcome {key} has no frozen prediction"
                )
            frozen.append(
                FrozenRun(
                    *key,
                    prediction_path,
                    prediction_hash,
                    outcome,
                    raw_outcome,
                    outcome_path,
                    outcome_hash,
                    outcome_reason,
                )
            )
    else:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise EvaluationError("pipeline freeze manifest requires artifacts[]")
        root_value = manifest.get("artifact_root")
        if root_value is None:
            artifact_root = (
                manifest_file.parent.parent
                if manifest_file.parent.name == "process"
                else manifest_file.parent
            )
        else:
            artifact_root = _resolve(manifest_file.parent, root_value, "artifact_root")
        prediction_artifacts: dict[str, tuple[Path, str]] = {}
        outcome_artifacts: dict[str, tuple[Path, str, str, str, str | None]] = {}
        artifacts_by_path: dict[Path, tuple[str, str | None]] = {}
        artifact_paths_by_run: dict[str, set[Path]] = {}
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise EvaluationError("freeze artifacts must be objects")
            artifact_path = _resolve(artifact_root, artifact.get("path"), "artifact.path")
            expected_hash = artifact.get("sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise EvaluationError(f"frozen artifact requires SHA-256: {artifact_path}")
            actual_hash = _sha256_file(artifact_path)
            if actual_hash.lower() != expected_hash.lower():
                raise EvaluationError(f"frozen artifact hash mismatch: {artifact_path}")
            verified_hashes += 1
            run_id = artifact.get("run_id")
            artifacts_by_path[artifact_path] = (
                actual_hash.lower(), run_id if isinstance(run_id, str) else None
            )
            if isinstance(run_id, str):
                artifact_paths_by_run.setdefault(run_id, set()).add(artifact_path)
            role = str(artifact.get("role", "")).strip().lower().replace("-", "_")
            is_prediction = (
                artifact_path.name.lower() == "prediction.json"
                and role not in {"input", "label"}
            )
            is_outcome = (
                role in {"terminal_outcome", "outcome", "run_outcome", "status"}
                or artifact_path.name.lower() in {
                    "terminal_outcome.json", "terminal-outcome.json", "outcome.json"
                }
            )
            if is_prediction and isinstance(run_id, str):
                if run_id in prediction_artifacts:
                    raise EvaluationError(f"duplicate frozen prediction for run {run_id}")
                prediction_artifacts[run_id] = (artifact_path, actual_hash.lower())
            if is_outcome and isinstance(run_id, str):
                if run_id in outcome_artifacts:
                    raise EvaluationError(f"duplicate terminal outcome for run {run_id}")
                outcome_payload = _read_json(artifact_path)
                outcome, raw_outcome = _outcome_status(outcome_payload)
                outcome_artifacts[run_id] = (
                    artifact_path, actual_hash.lower(), outcome, raw_outcome,
                    _outcome_reason(outcome_payload),
                )

        manifest_outcomes: dict[str, Mapping[str, Any]] = {}
        if terminal_complete:
            for raw in manifest.get("runs", []):
                if not isinstance(raw, Mapping):
                    raise EvaluationError("terminal freeze runs must be objects")
                run_id = raw.get("run_id")
                if not isinstance(run_id, str) or not run_id:
                    raise EvaluationError("terminal freeze run requires run_id")
                if run_id in manifest_outcomes:
                    raise EvaluationError(f"duplicate terminal freeze run_id: {run_id}")
                _outcome_status(raw)  # Validate the recorded terminal state now.
                manifest_outcomes[run_id] = raw
            recorded_eligible = sum(
                _outcome_status(raw)[0] == "success"
                for raw in manifest_outcomes.values()
            )
            recorded_excluded = sum(
                _outcome_status(raw)[0] == "excluded"
                for raw in manifest_outcomes.values()
            )
            if recorded_eligible + recorded_excluded != validated_predictions:
                raise EvaluationError(
                    "terminal freeze validated_predictions does not match recorded outcomes"
                )
            if recorded_excluded != excluded_predictions:
                raise EvaluationError(
                    "terminal freeze excluded_predictions does not match recorded outcomes"
                )
            eligible_predictions = manifest.get("eligible_predictions")
            if eligible_predictions is not None and (
                not isinstance(eligible_predictions, int)
                or isinstance(eligible_predictions, bool)
                or eligible_predictions != recorded_eligible
            ):
                raise EvaluationError(
                    "terminal freeze eligible_predictions does not match recorded outcomes"
                )
            if len(manifest_outcomes) - recorded_eligible - recorded_excluded != terminal_failures:
                raise EvaluationError(
                    "terminal freeze terminal_failures does not match recorded outcomes"
                )

        def declared_paths(raw: Mapping[str, Any]) -> set[Path]:
            values: set[Path] = set()
            for field, value in raw.items():
                if not isinstance(value, str):
                    continue
                lowered = field.lower()
                if (
                    lowered.endswith("path")
                    or lowered.endswith("relpath")
                    or value.lower().endswith((".json", ".md"))
                ):
                    values.add(_resolve(artifact_root, value, field))
            return values

        manifest_terminal_mode = bool(manifest_outcomes)
        terminal_mode = bool(outcome_artifacts or manifest_outcomes)
        for key, planned in plan_by_key.items():
            run_id = planned.get("run_id")
            if not isinstance(run_id, str):
                raise EvaluationError(f"run-plan key {key} lacks an opaque run_id")
            linked_paths = set(artifact_paths_by_run.get(run_id, ()))
            if manifest_terminal_mode:
                if run_id not in manifest_outcomes:
                    raise EvaluationError(f"no manifest terminal outcome for run-plan key {key}")
                manifest_outcome = manifest_outcomes.pop(run_id)
                outcome, raw_outcome = _outcome_status(manifest_outcome)
                outcome_reason = _outcome_reason(manifest_outcome)
                outcome_path = None
                outcome_hash = None
                declared = declared_paths(manifest_outcome)
                missing_declared = declared - set(artifacts_by_path)
                if missing_declared:
                    raise EvaluationError(
                        f"terminal run {run_id} references unhashed artifacts: "
                        + ", ".join(str(path) for path in sorted(missing_declared))
                    )
                linked_paths.update(declared)
            elif terminal_mode:
                if run_id not in outcome_artifacts:
                    raise EvaluationError(f"no terminal outcome artifact for run-plan key {key}")
                (
                    outcome_path, outcome_hash, outcome, raw_outcome, outcome_reason
                ) = outcome_artifacts.pop(run_id)
            else:
                outcome_path = None
                outcome_hash = None
                outcome = raw_outcome = "success"
                outcome_reason = None
            prediction = prediction_artifacts.pop(run_id, None)
            if prediction is None:
                declared_predictions = sorted(
                    path for path in linked_paths if path.name.lower() == "prediction.json"
                )
                if len(declared_predictions) > 1:
                    raise EvaluationError(f"multiple frozen predictions for run {run_id}")
                if declared_predictions:
                    path = declared_predictions[0]
                    prediction = (path, artifacts_by_path[path][0])
            if outcome in {"success", "excluded"} and prediction is None:
                raise EvaluationError(
                    f"prediction-bearing terminal outcome {key} has no frozen prediction artifact"
                )
            if manifest_terminal_mode and outcome in {"success", "excluded"} and not any(
                path.name.lower() == "process.md" for path in linked_paths
            ):
                raise EvaluationError(
                    f"successful terminal outcome {key} has no hashed process.md"
                )
            if manifest_terminal_mode and outcome == "excluded":
                if not outcome_reason:
                    raise EvaluationError(
                        f"excluded prediction {key} requires a structured exclusion reason"
                    )
                if not any(
                    path.suffix.lower() == ".md" and "exclusions" in {
                        part.lower() for part in path.parts
                    }
                    for path in linked_paths
                ):
                    raise EvaluationError(
                        f"excluded prediction {key} has no hashed exclusion Markdown artifact"
                    )
            if manifest_terminal_mode and outcome in {"refusal", "failure"} and not any(
                path.suffix.lower() == ".md" and "failures" in {
                    part.lower() for part in path.parts
                }
                for path in linked_paths
            ):
                raise EvaluationError(
                    f"terminal failure {key} has no hashed failure Markdown artifact"
                )
            prediction_path, prediction_hash = prediction or (None, None)
            frozen_keys.add(key)
            frozen.append(
                FrozenRun(
                    *key,
                    prediction_path,
                    prediction_hash,
                    outcome,
                    raw_outcome,
                    outcome_path,
                    outcome_hash,
                    outcome_reason,
                )
            )
        if outcome_artifacts:
            raise EvaluationError(
                "freeze contains outcomes absent from the run plan: "
                + ", ".join(sorted(outcome_artifacts))
            )
        if manifest_outcomes:
            raise EvaluationError(
                "freeze contains manifest outcomes absent from the run plan: "
                + ", ".join(sorted(manifest_outcomes))
            )
        if prediction_artifacts:
            raise EvaluationError(
                "freeze contains predictions absent from the run plan: "
                + ", ".join(sorted(prediction_artifacts))
            )

    if frozen_keys != set(plan_by_key):
        missing = sorted(set(plan_by_key) - frozen_keys)
        extra = sorted(frozen_keys - set(plan_by_key))
        raise EvaluationError(
            f"freeze/run-plan mismatch; missing={missing!r}, extra={extra!r}"
        )
    state = FreezeState(
        manifest_path=manifest_file,
        frozen_at=frozen_at,
        runs=tuple(sorted(frozen, key=lambda item: (item.target_id, item.condition, item.replicate))),
        manifest_sha256=_sha256_file(manifest_file),
        verified_artifact_hash_count=verified_hashes,
        terminal_outcome_count=(
            len(frozen) if terminal_complete or any(run.outcome_path for run in frozen) else 0
        ),
    )
    return state, plan_by_key


def assert_labels_postdate_freeze(labels_path: str | Path, freeze: FreezeState) -> Path:
    """Use metadata only to keep labels sealed until the validated freeze."""

    label_file = Path(labels_path).resolve()
    if label_file in {run.prediction_path for run in freeze.runs}:
        raise EvaluationError("label path must not be a frozen prediction artifact")
    try:
        label_time = datetime.fromtimestamp(label_file.stat().st_mtime, tz=timezone.utc)
    except FileNotFoundError as exc:
        raise EvaluationError(f"post-freeze label file does not exist: {label_file}") from exc
    # Permit one second for coarse filesystem timestamp resolution.
    if label_time.timestamp() + 1.0 < freeze.frozen_at.timestamp():
        raise EvaluationError("label file predates the prediction freeze; refusing to open it")
    return label_file


def _normalise_insertion(value: object) -> str:
    return "" if value in (None, "", ".", "?") else str(value)


def _mapping_residues(mapping: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    residues = mapping.get("residues")
    if not isinstance(residues, list) or not residues:
        raise EvaluationError("mapping JSON requires a non-empty residues list")
    if not all(isinstance(item, Mapping) for item in residues):
        raise EvaluationError("mapping residues must be objects")
    return residues  # type: ignore[return-value]


def _local_token(item: Mapping[str, Any]) -> str:
    local = item.get("local") if isinstance(item.get("local"), Mapping) else item
    chain = local.get("chain_id", local.get("local_chain_id"))
    number = local.get("seq_id", local.get("local_seq_id"))
    try:
        number_int = int(number)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("mapping residue has an invalid local sequence ID") from exc
    if not isinstance(chain, str) or not chain:
        raise EvaluationError("mapping residue has an invalid local chain ID")
    token = f"{chain}:{number_int}"
    if not _LOCAL_TOKEN_RE.fullmatch(token):
        raise EvaluationError(
            "mapping residue local token must match "
            "T<positive-int>:<positive-int>"
        )
    return token


def _auth_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
    auth = item.get("auth") if isinstance(item.get("auth"), Mapping) else item
    chain = auth.get("asym_id", auth.get("auth_asym_id"))
    number = auth.get("seq_id", auth.get("auth_seq_id"))
    insertion = auth.get("insertion_code", item.get("insertion_code", ""))
    if not isinstance(chain, str) or not chain or number is None:
        raise EvaluationError("mapping residue has an invalid author key")
    return chain, str(number), _normalise_insertion(insertion)


def _label_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
    chain = item.get("auth_asym_id", item.get("chain", item.get("chain_id")))
    number = item.get("auth_seq_id", item.get("residue_number", item.get("seq_id")))
    insertion = item.get("insertion_code", "")
    if not isinstance(chain, str) or not chain or number is None:
        raise EvaluationError("label residue requires author chain and residue number")
    return chain, str(number), _normalise_insertion(insertion)


def map_author_labels_to_local(
    author_labels: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Any],
) -> frozenset[str]:
    """Map author-chain residue labels to opaque local residue tokens."""

    index: dict[tuple[str, str, str], str] = {}
    for item in _mapping_residues(mapping):
        key = _auth_key(item)
        if key in index:
            raise EvaluationError(f"ambiguous author residue in mapping: {key}")
        index[key] = _local_token(item)
    local: set[str] = set()
    for label in author_labels:
        if not isinstance(label, Mapping):
            raise EvaluationError("labels must be residue objects, not free-form strings")
        key = _label_key(label)
        try:
            local.add(index[key])
        except KeyError as exc:
            raise EvaluationError(f"author label is outside the included target: {key}") from exc
    if not local:
        raise EvaluationError("each target requires at least one mapped label")
    return frozenset(local)


def _entity_id(item: Mapping[str, Any], chain_entities: Mapping[str, object]) -> str | None:
    local_chain = _local_token(item).split(":", 1)[0]
    candidates: list[object] = [
        item.get("source_entity_id"),
        item.get("entity_id"),
        chain_entities.get(local_chain),
    ]
    for nested_name in ("source", "label", "auth"):
        nested = item.get(nested_name)
        if isinstance(nested, Mapping):
            candidates.append(nested.get("entity_id"))
    for candidate in candidates:
        if candidate not in (None, ""):
            return str(candidate)
    return None


@dataclass(frozen=True)
class EquivalentChainSymmetry:
    groups: tuple[tuple[str, ...], ...]
    residue_correspondence: Mapping[tuple[str, str], str]


@dataclass(frozen=True)
class _MappedResidue:
    token: str
    local_chain: str
    local_number: int
    auth_chain: str
    auth_seq: str
    insertion_code: str
    label_seq: str | None
    component: str


def _mapped_residues_by_chain(
    mapping: Mapping[str, Any],
) -> dict[str, tuple[_MappedResidue, ...]]:
    output: dict[str, list[_MappedResidue]] = {}
    for item in _mapping_residues(mapping):
        token = _local_token(item)
        local_chain, number_text = token.split(":", 1)
        auth_chain, auth_seq, insertion = _auth_key(item)
        label = item.get("label") if isinstance(item.get("label"), Mapping) else item
        auth = item.get("auth") if isinstance(item.get("auth"), Mapping) else item
        raw_label_seq = label.get("seq_id", label.get("label_seq_id"))
        label_seq = None if raw_label_seq in (None, "", ".", "?") else str(raw_label_seq)
        label_component = label.get("comp_id", label.get("label_comp_id"))
        auth_component = auth.get("comp_id", auth.get("auth_comp_id"))
        if label_component in (None, "") and auth_component in (None, ""):
            raise EvaluationError("mapping residue lacks component identity")
        if (
            label_component not in (None, "")
            and auth_component not in (None, "")
            and str(label_component) != str(auth_component)
        ):
            raise EvaluationError("mapping residue auth/label component identities disagree")
        component = str(
            label_component if label_component not in (None, "") else auth_component
        )
        output.setdefault(local_chain, []).append(
            _MappedResidue(
                token=token,
                local_chain=local_chain,
                local_number=int(number_text),
                auth_chain=auth_chain,
                auth_seq=auth_seq,
                insertion_code=insertion,
                label_seq=label_seq,
                component=component,
            )
        )
    return {
        chain: tuple(sorted(records, key=lambda record: record.local_number))
        for chain, records in output.items()
    }


def _group_residue_correspondence(
    groups: Sequence[Sequence[str]],
    residues_by_chain: Mapping[str, Sequence[_MappedResidue]],
) -> dict[tuple[str, str], str]:
    correspondence: dict[tuple[str, str], str] = {}
    for group in groups:
        for source_chain in group:
            source_records = tuple(residues_by_chain[source_chain])
            for destination_chain in group:
                if source_chain == destination_chain:
                    continue
                destination_records = tuple(residues_by_chain[destination_chain])
                destination_label: dict[str, _MappedResidue] = {}
                destination_auth: dict[tuple[str, str], _MappedResidue] = {}
                for record in destination_records:
                    if record.label_seq is not None:
                        if record.label_seq in destination_label:
                            raise EvaluationError(
                                f"duplicate label sequence ID in local chain {destination_chain}"
                            )
                        destination_label[record.label_seq] = record
                    auth_key = (record.auth_seq, record.insertion_code)
                    if auth_key in destination_auth:
                        raise EvaluationError(
                            f"duplicate author residue ID in local chain {destination_chain}"
                        )
                    destination_auth[auth_key] = record

                # Shared source positions must agree in amino-acid identity even
                # when one cropped chain contains additional positions.
                for source in source_records:
                    if source.label_seq is not None and source.label_seq in destination_label:
                        destination = destination_label[source.label_seq]
                        if source.component != destination.component:
                            raise EvaluationError(
                                "declared equivalent chains disagree in residue identity "
                                f"at label sequence {source.label_seq}"
                            )
                    auth_key = (source.auth_seq, source.insertion_code)
                    if auth_key in destination_auth:
                        destination = destination_auth[auth_key]
                        if (
                            (source.label_seq is None or destination.label_seq is None)
                            and source.component != destination.component
                        ):
                            raise EvaluationError(
                                "declared equivalent chains disagree in residue identity "
                                f"at author residue {source.auth_seq}{source.insertion_code}"
                            )

                for source in source_records:
                    destination: _MappedResidue | None = None
                    if source.label_seq is not None:
                        destination = destination_label.get(source.label_seq)
                    if destination is None:
                        fallback = destination_auth.get(
                            (source.auth_seq, source.insertion_code)
                        )
                        # Auth numbering is a fallback only when label numbering
                        # is absent on at least one side; conflicting present
                        # label IDs never fall back to ordinal/author coincidence.
                        if fallback is not None and (
                            source.label_seq is None or fallback.label_seq is None
                        ):
                            destination = fallback
                    if destination is not None:
                        if source.component != destination.component:
                            raise EvaluationError(
                                "declared equivalent chains have an amino-acid mismatch"
                            )
                        correspondence[(source.token, destination_chain)] = destination.token
    return correspondence


def derive_equivalent_chain_symmetry(
    mapping: Mapping[str, Any],
    equivalent_auth_chain_groups: Sequence[Sequence[str]] | None = None,
) -> EquivalentChainSymmetry:
    """Compile equivalent groups and residue-level cross-chain correspondence."""

    residues_by_chain = _mapped_residues_by_chain(mapping)
    if equivalent_auth_chain_groups is not None:
        auth_to_local: dict[str, set[str]] = {}
        for local_chain, records in residues_by_chain.items():
            for record in records:
                auth_to_local.setdefault(record.auth_chain, set()).add(local_chain)
        local_groups: list[tuple[str, ...]] = []
        seen_auth: set[str] = set()
        seen_local: set[str] = set()
        for raw_group in equivalent_auth_chain_groups:
            if not isinstance(raw_group, (list, tuple)) or len(raw_group) < 2:
                raise EvaluationError("explicit equivalent auth-chain groups need at least two chains")
            group = tuple(raw_group)
            if any(not isinstance(chain, str) or not chain for chain in group):
                raise EvaluationError("explicit equivalent auth-chain identifiers must be strings")
            if len(set(group)) != len(group) or seen_auth.intersection(group):
                raise EvaluationError("explicit equivalent auth-chain groups must be unique and disjoint")
            mapped: list[str] = []
            for auth_chain in group:
                local = auth_to_local.get(auth_chain, set())
                if len(local) != 1:
                    raise EvaluationError(
                        f"equivalent author chain {auth_chain!r} does not map to exactly one local chain"
                    )
                mapped.append(next(iter(local)))
            if len(set(mapped)) != len(mapped) or seen_local.intersection(mapped):
                raise EvaluationError("equivalent auth chains do not map to disjoint local chains")
            seen_auth.update(group)
            seen_local.update(mapped)
            local_groups.append(tuple(mapped))
        groups = tuple(local_groups)
        return EquivalentChainSymmetry(
            groups=groups,
            residue_correspondence=_group_residue_correspondence(
                groups, residues_by_chain
            ),
        )

    chain_entities: dict[str, object] = {}
    raw_chains = mapping.get("chains")
    if isinstance(raw_chains, list):
        for chain in raw_chains:
            if isinstance(chain, Mapping):
                local = chain.get("local_chain_id", chain.get("chain_id"))
                entity = chain.get("source_entity_id", chain.get("entity_id"))
                if isinstance(local, str) and entity not in (None, ""):
                    chain_entities[local] = entity
    mapping_by_token = {_local_token(item): item for item in _mapping_residues(mapping)}
    signatures: dict[tuple[str, tuple[tuple[int, str, str], ...]], list[str]] = {}
    for chain, records in residues_by_chain.items():
        entity: str | None = None
        signature_records: list[tuple[int, str, str]] = []
        for record in records:
            item = mapping_by_token[record.token]
            current_entity = _entity_id(item, chain_entities)
            if record.label_seq is None or current_entity is None:
                entity = None
                signature_records = []
                break
            if entity is not None and entity != current_entity:
                entity = None
                signature_records = []
                break
            entity = current_entity
            signature_records.append(
                (record.local_number, record.label_seq, record.component)
            )
        if entity is not None:
            signature = (entity, tuple(sorted(signature_records)))
            signatures.setdefault(signature, []).append(chain)
    groups = tuple(
        tuple(sorted(chains))
        for _, chains in sorted(signatures.items(), key=lambda item: repr(item[0]))
        if len(chains) > 1
    )
    return EquivalentChainSymmetry(
        groups=groups,
        residue_correspondence=_group_residue_correspondence(groups, residues_by_chain),
    )


def derive_equivalent_chain_groups(
    mapping: Mapping[str, Any],
    equivalent_auth_chain_groups: Sequence[Sequence[str]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Backward-compatible group-only view of the compiled symmetry."""

    return derive_equivalent_chain_symmetry(
        mapping, equivalent_auth_chain_groups
    ).groups


def _labels_by_target(payload: Any) -> dict[str, Sequence[Mapping[str, Any]]]:
    if not isinstance(payload, Mapping):
        raise EvaluationError("labels JSON must be an object")
    output: dict[str, Sequence[Mapping[str, Any]]] = {}
    if isinstance(payload.get("targets"), list):
        for target in payload["targets"]:
            if not isinstance(target, Mapping):
                raise EvaluationError("label target entries must be objects")
            target_id = target.get("target_id", target.get("case_id"))
            labels = target.get("residues", target.get("labels", target.get("hotspots")))
            if not isinstance(target_id, str) or not isinstance(labels, list):
                raise EvaluationError("label targets require target_id and residues")
            output[target_id] = labels
    elif isinstance(payload.get("labels"), Mapping):
        for target_id, labels in payload["labels"].items():
            if not isinstance(target_id, str) or not isinstance(labels, list):
                raise EvaluationError("labels mapping values must be lists")
            output[target_id] = labels
    else:
        raise EvaluationError("labels JSON requires targets[] or labels{}")
    return output


def _prediction_set(payload: Any, universe: Iterable[str]) -> PredictionSet:
    if not isinstance(payload, Mapping):
        raise EvaluationError("prediction JSON must be an object")
    if "primary_hotspots" in payload or "alternate_hotspots" in payload:
        selected = {
            "primary": payload.get("primary_hotspots"),
            "alternates": payload.get("alternate_hotspots"),
        }
    else:
        selected = {"primary": payload.get("primary"), "alternates": payload.get("alternates")}
    try:
        return validate_prediction_schema(selected, universe, require_recognized=True)
    except ValueError as exc:
        raise EvaluationError(f"invalid frozen prediction: {exc}") from exc


def _available_case_consensus(predictions: Sequence[PredictionSet]) -> PredictionSet:
    """Exploratory n=1..3 consensus with the preregistered rank rules.

    Residues are ordered by descending raw run frequency, then descending sum
    of reciprocal within-run ranks, then ascending opaque residue token.  The
    final token sort is the deterministic tie rule.
    """

    if not 1 <= len(predictions) <= 3:
        raise ValueError("available-case consensus requires 1 to 3 predictions")
    frequency: Counter[str] = Counter()
    reciprocal_rank: defaultdict[str, float] = defaultdict(float)
    for prediction in predictions:
        for rank, token in enumerate(prediction.ranked, start=1):
            frequency[token] += 1
            reciprocal_rank[token] += 1.0 / rank
    ordered = sorted(
        frequency,
        key=lambda token: (-frequency[token], -reciprocal_rank[token], token),
    )
    if len(ordered) < 6:
        raise EvaluationError("available predictions must contain at least 6 distinct residues")
    return PredictionSet(
        primary=(ordered[0], ordered[1], ordered[2]),
        alternates=(ordered[3], ordered[4], ordered[5]),
    )


def _features(payload: Any) -> tuple[frozenset[str], dict[str, float], dict[str, float]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("residues"), list):
        raise EvaluationError("feature JSON requires residues[]")
    universe: set[str] = set()
    rsasa: dict[str, float] = {}
    sasa: dict[str, float] = {}
    for item in payload["residues"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("token"), str):
            raise EvaluationError("feature residues require token")
        token = item["token"]
        if not _LOCAL_TOKEN_RE.fullmatch(token):
            raise EvaluationError(
                "feature residue token must match T<positive-int>:<positive-int>"
            )
        try:
            relative = float(item["relative_sasa"])
            absolute = float(item["residue_sasa_angstrom2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(f"feature residue {token} lacks SASA fields") from exc
        if token in universe or not 0.0 <= relative <= 1.0 or absolute < 0.0:
            raise EvaluationError(f"invalid feature residue {token}")
        universe.add(token)
        rsasa[token] = relative
        sasa[token] = absolute
    if not universe:
        raise EvaluationError("feature universe is empty")
    return frozenset(universe), rsasa, sasa


def _residue_atoms(path: Path) -> dict[str, tuple[tuple[object, ...], ...]]:
    atoms: dict[str, tuple[tuple[object, ...], ...]] = {}
    try:
        residues = parse_mmcif_file(path)
    except Exception as exc:
        raise EvaluationError(f"could not parse local structure {path}: {exc}") from exc
    for residue in residues:
        try:
            number = int(residue.auth_seq_id)
        except ValueError as exc:
            raise EvaluationError("local structure residue numbering must be integral") from exc
        insertion = _normalise_insertion(residue.insertion_code)
        token = f"{residue.auth_asym_id}:{number}{insertion}"
        if not _LOCAL_TOKEN_RE.fullmatch(token):
            raise EvaluationError(
                "local structure residue token must match "
                "T<positive-int>:<positive-int>"
            )
        atoms[token] = tuple(
            (
                atom.element,
                float(atom.coordinate[0]),
                float(atom.coordinate[1]),
                float(atom.coordinate[2]),
            )
            for atom in residue.atoms
        )
    return atoms


def _symmetry_hit_count(
    selected: Iterable[str],
    truth: frozenset[str],
    groups: Sequence[Sequence[str]],
    residue_correspondence: Mapping[tuple[str, str], str] | None = None,
) -> float:
    optimized = global_chain_permutation(
        tuple(sorted(selected)),
        truth,
        groups,
        residue_correspondence=residue_correspondence,
    )
    return float(len(set(optimized.remapped) & truth))


def _pocket_matched_null(
    selected: Sequence[str],
    truth: frozenset[str],
    universe: frozenset[str],
    rsasa: Mapping[str, float],
    residue_atoms: Mapping[str, Iterable[Sequence[object]]],
    groups: Sequence[Sequence[str]],
    residue_correspondence: Mapping[tuple[str, str], str] | None,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Matched null for symmetry-aware Top3 H6 pocket Jaccard.

    Single-residue H6 neighborhoods are cached, so every Monte Carlo draw only
    unions three small sets.  Each observed or null selection is still passed
    through one allowed *global* chain permutation, chosen to maximize pocket
    Jaccard; no residue-wise symmetry remapping is permitted.
    """

    neighborhoods = {
        token: expand_residue_pocket((token,), residue_atoms, radius=6.0)
        for token in sorted(universe)
    }
    truth_pocket = frozenset().union(*(neighborhoods[token] for token in truth))

    def pocket_jaccard(tokens: tuple[str, ...]) -> float:
        predicted_pocket = frozenset().union(*(neighborhoods[token] for token in tokens))
        union = predicted_pocket | truth_pocket
        return len(predicted_pocket & truth_pocket) / len(union) if union else 1.0

    def statistic(sample: frozenset[str]) -> float:
        optimized = global_chain_permutation(
            tuple(sorted(sample)),
            truth,
            groups,
            score_fn=pocket_jaccard,
            residue_correspondence=residue_correspondence,
        )
        return float(optimized.score[0])

    mc = stratified_monte_carlo(
        selected,
        universe,
        rsasa,
        statistic,
        draws=draws,
        seed=seed,
    )
    return {
        "statistic": "symmetry-aware Top3 H6 pocket Jaccard",
        "observed": mc.observed,
        "null_mean": fmean(mc.null_values),
        "p_greater_equal": mc.p_greater_equal,
        "draws": mc.draws,
        "seed": mc.seed,
        "stratum_counts": [
            {"chain": key[0], "rsasa_quintile": key[1], "count": count}
            for key, count in mc.stratum_counts
        ],
    }


class _MatchedSampler:
    def __init__(self, selected: Iterable[str], universe: Iterable[str], rsasa: Mapping[str, float]):
        self.selected = frozenset(selected)
        self.universe = frozenset(universe)
        if not self.selected <= self.universe:
            raise EvaluationError("matched-null selection is outside its universe")
        quintiles = empirical_rsasa_quintiles(self.universe, rsasa)
        pools: dict[tuple[str, int], list[str]] = {}
        required: dict[tuple[str, int], int] = {}
        for token in self.universe:
            chain = token.split(":", 1)[0]
            pools.setdefault((chain, quintiles[token]), []).append(token)
        for token in self.selected:
            chain = token.split(":", 1)[0]
            key = (chain, quintiles[token])
            required[key] = required.get(key, 0) + 1
        self.pools = {key: tuple(sorted(value)) for key, value in pools.items()}
        self.required = tuple(sorted(required.items()))

    def draw(self, rng: random.Random) -> frozenset[str]:
        sample: list[str] = []
        for key, count in self.required:
            sample.extend(rng.sample(self.pools[key], count))
        return frozenset(sample)


def joint_matched_null(
    cases: Sequence[Mapping[str, Any]],
    *,
    draws: int = 99_999,
    seed: int = 0,
) -> JointNullResult:
    """Jointly sample the primary consensus total under the matched null."""

    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if not cases:
        raise ValueError("joint null requires at least one target")
    prepared: list[
        tuple[
            str,
            _MatchedSampler,
            frozenset[str],
            tuple[tuple[str, ...], ...],
            Mapping[tuple[str, str], str] | None,
        ]
    ] = []
    seen: set[str] = set()
    for case in cases:
        target_id = case.get("target_id")
        if not isinstance(target_id, str) or not target_id or target_id in seen:
            raise ValueError("joint-null targets must have unique target_id values")
        seen.add(target_id)
        truth = frozenset(case["truth"])
        groups = tuple(tuple(group) for group in case.get("equivalent_chain_groups", ()))
        raw_correspondence = case.get("residue_correspondence")
        correspondence = (
            None
            if raw_correspondence is None
            else dict(raw_correspondence)
        )
        sampler = _MatchedSampler(case["selected"], case["universe"], case["rsasa"])
        prepared.append((target_id, sampler, truth, groups, correspondence))
    prepared.sort(key=lambda item: item[0])
    observed = math.fsum(
        _symmetry_hit_count(item.selected, truth, groups, correspondence)
        for _, item, truth, groups, correspondence in prepared
    )
    rng = random.Random(seed)
    null_values: list[float] = []
    exceedances = 0
    for _ in range(draws):
        total = math.fsum(
            _symmetry_hit_count(
                sampler.draw(rng), truth, groups, correspondence
            )
            for _, sampler, truth, groups, correspondence in prepared
        )
        null_values.append(total)
        exceedances += total >= observed
    return JointNullResult(
        observed=observed,
        null_mean=fmean(null_values),
        p_greater_equal=(exceedances + 1.0) / (draws + 1.0),
        draws=draws,
        seed=seed,
        null_values=tuple(null_values),
    )


def primary_decision(observed: float, null_mean: float, p_value: float) -> dict[str, Any]:
    supported = observed > null_mean and p_value < 0.05
    return {
        "supported": supported,
        "decision": "supported" if supported else "not_supported_in_this_benchmark",
        "rule": "observed > matched-null expectation and one-sided joint p < 0.05",
    }


def _metric_bundle(
    prediction: PredictionSet,
    truth: frozenset[str],
    universe: frozenset[str],
    rsasa: Mapping[str, float],
    sasa: Mapping[str, float],
    residue_atoms: Mapping[str, Iterable[Sequence[object]]],
    groups: Sequence[Sequence[str]],
    residue_correspondence: Mapping[tuple[str, str], str] | None = None,
    *,
    mc_draws: int | None,
    seed: int,
) -> dict[str, Any]:
    symmetry = symmetry_adjusted_top_metrics(
        prediction,
        truth,
        len(universe),
        groups,
        residue_correspondence,
    )
    strict_top3 = prediction.primary
    strict_top6 = prediction.ranked
    adjusted_top3 = symmetry.remapped_prediction.primary
    adjusted_top6 = symmetry.remapped_prediction.ranked
    missing_atoms = (set(prediction.ranked) | set(truth) | set(universe)) - set(residue_atoms)
    if missing_atoms:
        raise EvaluationError("local structure is missing residues: " + ", ".join(sorted(missing_atoms)))

    spatial: dict[str, Any] = {}
    for name, selected in (
        ("strict_top3", strict_top3),
        ("strict_top6", strict_top6),
        ("symmetry_top3", adjusted_top3),
        ("symmetry_top6", adjusted_top6),
    ):
        distances = pocket_distance_metrics(selected, truth, residue_atoms)
        tolerant = standard_distance_tolerances(selected, truth, residue_atoms)
        # The preregistered pocket weighting is relative SASA (rSASA).  Keep
        # the older generic ``sasa_weighted_*`` keys below as value aliases for
        # downstream readers that consumed schema v1.
        overlap = h6_pocket_overlap(selected, truth, residue_atoms, rsasa)
        spatial[name] = {
            "distances": asdict(distances),
            "tolerant": {str(int(key)): asdict(value) for key, value in tolerant.items()},
            "h6_overlap": {
                "jaccard": overlap.jaccard,
                "dice": overlap.dice,
                "rsasa_weighted_jaccard": overlap.sasa_weighted_jaccard,
                "rsasa_weighted_dice": overlap.sasa_weighted_dice,
                "predicted_h6_size": len(overlap.predicted_h6),
                "reference_h6_size": len(overlap.reference_h6),
            },
        }
    output = {
        "prediction": {"primary": list(prediction.primary), "alternates": list(prediction.alternates)},
        "strict": _jsonable(symmetry.strict),
        "symmetry_adjusted": _jsonable(symmetry.symmetry_adjusted),
        "chain_mapping": dict(symmetry.chain_mapping),
        "residue_correspondence_count": len(residue_correspondence or {}),
        "symmetry_remapped_prediction": list(symmetry.remapped_prediction.ranked),
        "spatial": spatial,
    }
    if mc_draws is not None:
        statistic = lambda sample: _symmetry_hit_count(
            sample, truth, groups, residue_correspondence
        )
        mc = stratified_monte_carlo(
            prediction.primary,
            universe,
            rsasa,
            statistic,
            draws=mc_draws,
            seed=seed,
        )
        output["matched_null"] = {
            "observed": mc.observed,
            "null_mean": fmean(mc.null_values),
            "p_greater_equal": mc.p_greater_equal,
            "draws": mc.draws,
            "seed": mc.seed,
            "stratum_counts": [
                {"chain": key[0], "rsasa_quintile": key[1], "count": count}
                for key, count in mc.stratum_counts
            ],
        }
        output["pocket_matched_null"] = _pocket_matched_null(
            prediction.primary,
            truth,
            universe,
            rsasa,
            residue_atoms,
            groups,
            residue_correspondence,
            draws=mc_draws,
            seed=seed + 1_000_000,
        )
    return output


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_benchmark(
    manifest_path: str | Path,
    run_plan_path: str | Path,
    labels_path: str | Path,
    *,
    target_manifest_path: str | Path | None = None,
    per_run_mc_draws: int = 9_999,
    joint_mc_draws: int = 99_999,
    bootstrap_draws: int = 10_000,
    seed: int = 20260824,
) -> dict[str, Any]:
    """Evaluate successful predictions while preserving every terminal outcome."""

    freeze, plan_by_key = verify_prediction_freeze(manifest_path, run_plan_path)
    expected_targets = {run.target_id for run in freeze.runs}
    plan_file = Path(run_plan_path).resolve()
    if target_manifest_path is None:
        candidate = plan_file.parent / "target_manifest.json"
        resolved_target_manifest = candidate if candidate.is_file() else None
    else:
        resolved_target_manifest = Path(target_manifest_path).resolve()
    explicit_equivalence = _explicit_equivalence_by_target(resolved_target_manifest)
    target_names = _target_names_by_id(resolved_target_manifest)
    if explicit_equivalence is not None and set(explicit_equivalence) != expected_targets:
        raise EvaluationError("target manifest does not exactly cover frozen benchmark targets")
    if target_names and set(target_names) != expected_targets:
        raise EvaluationError("target manifest names do not exactly cover frozen benchmark targets")
    target_names = {target: target_names.get(target, target) for target in expected_targets}
    frozen_by_key = {(run.target_id, run.condition, run.replicate): run for run in freeze.runs}

    outcomes = [
        {
            "target_id": run.target_id,
            "target_name": target_names[run.target_id],
            "condition": run.condition,
            "replicate": run.replicate,
            "outcome": run.outcome,
            "recorded_outcome": run.raw_outcome,
            "outcome_reason": run.outcome_reason,
            "outcome_sha256": run.outcome_sha256,
            "prediction_sha256": run.sha256,
            "prediction_valid": run.outcome == "success",
            "prediction_excluded": run.outcome == "excluded",
        }
        for run in freeze.runs
    ]
    outcome_summary: dict[str, dict[str, Any]] = {}
    for condition in sorted({run.condition for run in freeze.runs}):
        condition_runs = [run for run in freeze.runs if run.condition == condition]
        attempted = len(condition_runs)
        valid = sum(run.outcome == "success" for run in condition_runs)
        refusals = sum(run.outcome == "refusal" for run in condition_runs)
        excluded = sum(run.outcome == "excluded" for run in condition_runs)
        other_failures = attempted - valid - refusals - excluded
        outcome_summary[condition] = {
            "attempted": attempted,
            "valid": valid,
            "refusals": refusals,
            "excluded": excluded,
            "other_failures": other_failures,
            "valid_rate": valid / attempted,
            "refusal_rate": refusals / attempted,
            "failure_rate": (attempted - valid) / attempted,
        }

    # Read predictions and derive both consensus variants while labels remain
    # sealed.  Failed/refused runs are deliberately not opened as predictions.
    prepared: dict[tuple[str, str, int], dict[str, Any]] = {}
    blind_cells: dict[tuple[str, str], list[tuple[int, PredictionSet]]] = {}
    for key in sorted(plan_by_key):
        frozen_run = frozen_by_key[key]
        if frozen_run.outcome != "success":
            continue
        if frozen_run.prediction_path is None or frozen_run.sha256 is None:
            raise EvaluationError(f"successful run {key} lacks a frozen prediction")
        plan = plan_by_key[key]
        mapping_path, features_path, structure_path = _run_asset_paths(plan, plan_file)
        mapping = _read_json(mapping_path)
        features_payload = _read_json(features_path)
        if not isinstance(mapping, Mapping):
            raise EvaluationError("mapping JSON must be an object")
        universe, rsasa, sasa = _features(features_payload)
        atoms = _residue_atoms(structure_path)
        declared_groups = (
            None if explicit_equivalence is None else explicit_equivalence.get(key[0], ())
        )
        symmetry_definition = derive_equivalent_chain_symmetry(mapping, declared_groups)
        prediction_payload = _read_json(frozen_run.prediction_path)
        if not isinstance(prediction_payload, Mapping):
            raise EvaluationError("prediction JSON must be an object")
        recorded_target = prediction_payload.get(
            "target_id", prediction_payload.get("case_id", prediction_payload.get("case"))
        )
        for field, recorded, expected in (
            ("target/case", recorded_target, key[0]),
            ("condition", prediction_payload.get("condition"), key[1]),
            ("replicate", prediction_payload.get("replicate"), key[2]),
        ):
            if recorded is not None and recorded != expected:
                raise EvaluationError(
                    f"frozen prediction {field} does not match run-plan key {key}"
                )
        prediction = _prediction_set(prediction_payload, universe)
        prepared[key] = {
            "mapping": mapping,
            "universe": universe,
            "rsasa": rsasa,
            "sasa": sasa,
            "atoms": atoms,
            "groups": symmetry_definition.groups,
            "residue_correspondence": symmetry_definition.residue_correspondence,
            "prediction": prediction,
            "prediction_payload": prediction_payload,
        }
        blind_cells.setdefault((key[0], key[1]), []).append((key[2], prediction))

    blind_consensus: dict[tuple[str, str], PredictionSet] = {}
    available_consensus: dict[tuple[str, str], PredictionSet] = {}
    for cell, cell_predictions in sorted(blind_cells.items()):
        cell_predictions.sort(key=lambda item: item[0])
        predictions = [prediction for _, prediction in cell_predictions]
        available_consensus[cell] = _available_case_consensus(predictions)
        if len(predictions) == 3:
            blind_consensus[cell] = consensus_prediction(predictions)

    def consensus_digest(values: Mapping[tuple[str, str], PredictionSet]) -> str:
        serialized = json.dumps(
            [
                {
                    "target_id": target,
                    "condition": condition,
                    "primary": list(prediction.primary),
                    "alternates": list(prediction.alternates),
                }
                for (target, condition), prediction in sorted(values.items())
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(serialized).hexdigest()

    blind_consensus_sha256 = consensus_digest(blind_consensus)
    available_consensus_sha256 = consensus_digest(available_consensus)

    label_file = assert_labels_postdate_freeze(labels_path, freeze)
    labels_payload = _read_json(label_file)  # The sole post-validation unsealing point.
    labels_by_target = _labels_by_target(labels_payload)
    if set(labels_by_target) != expected_targets:
        raise EvaluationError("label target set does not exactly match the frozen benchmark")

    assets: dict[tuple[str, str, int], dict[str, Any]] = {}
    evaluated_runs: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(prepared)):
        item = prepared[key]
        truth = map_author_labels_to_local(labels_by_target[key[0]], item["mapping"])
        if not truth <= item["universe"]:
            raise EvaluationError(f"mapped labels fall outside feature universe for {key[0]}")
        bundle = _metric_bundle(
            item["prediction"], truth, item["universe"], item["rsasa"], item["sasa"],
            item["atoms"], item["groups"], item["residue_correspondence"],
            mc_draws=per_run_mc_draws, seed=seed + index,
        )
        evaluated_runs.append(
            {
                "target_id": key[0],
                "target_name": target_names[key[0]],
                "condition": key[1],
                "replicate": key[2],
                "prediction_sha256": frozen_by_key[key].sha256,
                "metrics": bundle,
                "audit": {
                    "recognition_status": item["prediction_payload"].get("recognition_status"),
                    "compliance": item["prediction_payload"].get("compliance"),
                },
            }
        )
        assets[key] = {
            "truth": truth,
            "universe": item["universe"],
            "rsasa": item["rsasa"],
            "sasa": item["sasa"],
            "atoms": item["atoms"],
            "groups": item["groups"],
            "residue_correspondence": item["residue_correspondence"],
        }

    runs_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in evaluated_runs:
        runs_by_cell.setdefault((run["target_id"], run["condition"]), []).append(run)

    def score_consensus(
        source: Mapping[tuple[str, str], PredictionSet], analysis: str
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cell_index, ((target, condition), prediction) in enumerate(sorted(source.items())):
            cell_runs = sorted(runs_by_cell[(target, condition)], key=lambda item: item["replicate"])
            asset = assets[(target, condition, cell_runs[0]["replicate"])]
            bundle = _metric_bundle(
                prediction, asset["truth"], asset["universe"], asset["rsasa"],
                asset["sasa"], asset["atoms"], asset["groups"],
                asset["residue_correspondence"], mc_draws=None,
                seed=seed + 100_000 + cell_index,
            )
            rows.append(
                {
                    "target_id": target,
                    "target_name": target_names[target],
                    "condition": condition,
                    "valid_predictions": len(cell_runs),
                    "analysis": analysis,
                    "metrics": bundle,
                }
            )
        return rows

    consensus_rows = score_consensus(blind_consensus, "preregistered_three_run_consensus")
    available_rows = score_consensus(
        available_consensus, "exploratory_available_case_consensus"
    )

    workflow_hits: list[dict[str, Any]] = []
    planned_cells = sorted({(key[0], key[1]) for key in plan_by_key})
    for target, condition in planned_cells:
        cell_runs = runs_by_cell.get((target, condition), [])
        strict_hit_runs = sum(
            run["metrics"]["strict"]["top3"]["h"] > 0 for run in cell_runs
        )
        adjusted_hit_runs = sum(
            run["metrics"]["symmetry_adjusted"]["top3"]["h"] > 0
            for run in cell_runs
        )
        valid = len(cell_runs)
        workflow_hits.append(
            {
                "target_id": target,
                "target_name": target_names[target],
                "condition": condition,
                "valid_runs": valid,
                "preregistered_denominator": 3,
                "strict_hit_runs": strict_hit_runs,
                "strict_workflow_hit_rate": strict_hit_runs / 3.0,
                "strict_conditional_hit_rate": (
                    strict_hit_runs / valid if valid else None
                ),
                "symmetry_hit_runs": adjusted_hit_runs,
                "symmetry_workflow_hit_rate": adjusted_hit_runs / 3.0,
                "symmetry_conditional_hit_rate": (
                    adjusted_hit_runs / valid if valid else None
                ),
                # Backward-compatible names used by the original report.
                "strict_any_hit_per_3": strict_hit_runs / 3.0,
                "symmetry_any_hit_per_3": adjusted_hit_runs / 3.0,
            }
        )

    planned_primary = {
        target: [key for key in plan_by_key if key[0] == target and key[1] == PRIMARY_CONDITION]
        for target in expected_targets
    }
    strict_by_cell = {(row["target_id"], row["condition"]): row for row in consensus_rows}
    primary_complete = bool(expected_targets) and all(
        len(planned_primary[target]) == 3
        and (target, PRIMARY_CONDITION) in strict_by_cell
        and strict_by_cell[(target, PRIMARY_CONDITION)]["valid_predictions"] == 3
        for target in expected_targets
    )

    def joint_cases(rows: Sequence[Mapping[str, Any]], condition: str) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for row in rows:
            if row["condition"] != condition:
                continue
            target = str(row["target_id"])
            cell_runs = runs_by_cell[(target, condition)]
            asset = assets[(target, condition, cell_runs[0]["replicate"])]
            cases.append(
                {
                    "target_id": target,
                    "selected": row["metrics"]["prediction"]["primary"],
                    "truth": asset["truth"],
                    "universe": asset["universe"],
                    "rsasa": asset["rsasa"],
                    "equivalent_chain_groups": asset["groups"],
                    "residue_correspondence": asset["residue_correspondence"],
                }
            )
        return cases

    if primary_complete:
        primary_joint = joint_matched_null(
            joint_cases(consensus_rows, PRIMARY_CONDITION),
            draws=joint_mc_draws,
            seed=seed + 200_000,
        )
        primary = {
            "condition": PRIMARY_CONDITION,
            "analysis": "preregistered_strict",
            "endpoint": "three-run consensus Top3 symmetry-aware total exact hits",
            "expected_cells": len(expected_targets),
            "complete_cells": len(expected_targets),
            "observed": primary_joint.observed,
            "matched_null_mean": primary_joint.null_mean,
            "p_greater_equal": primary_joint.p_greater_equal,
            "draws": primary_joint.draws,
            "seed": primary_joint.seed,
            **primary_decision(
                primary_joint.observed,
                primary_joint.null_mean,
                primary_joint.p_greater_equal,
            ),
        }
    else:
        complete_primary_cells = sum(
            (target, PRIMARY_CONDITION) in strict_by_cell for target in expected_targets
        )
        primary = {
            "condition": PRIMARY_CONDITION,
            "analysis": "preregistered_strict",
            "endpoint": "three-run consensus Top3 symmetry-aware total exact hits",
            "expected_cells": len(expected_targets),
            "complete_cells": complete_primary_cells,
            "decision": "not_evaluable_due_to_refusals",
            "rule": "all anonymous_no_web target cells must contain 3/3 valid predictions",
            "observed": None,
            "matched_null_mean": None,
            "p_greater_equal": None,
            "draws": None,
            "seed": None,
        }

    available_primary_cases = joint_cases(available_rows, PRIMARY_CONDITION)
    if available_primary_cases:
        available_joint = joint_matched_null(
            available_primary_cases,
            draws=joint_mc_draws,
            seed=seed + 250_000,
        )
        exploratory_joint: dict[str, Any] = {
            "analysis": "exploratory_available_case",
            "condition": PRIMARY_CONDITION,
            "target_count": len(available_primary_cases),
            "observed": available_joint.observed,
            "matched_null_mean": available_joint.null_mean,
            "p_greater_equal": available_joint.p_greater_equal,
            "draws": available_joint.draws,
            "seed": available_joint.seed,
        }
    else:
        exploratory_joint = {
            "analysis": "exploratory_available_case",
            "condition": PRIMARY_CONDITION,
            "target_count": 0,
            "status": "not_evaluable_no_available_predictions",
        }

    condition_statistics: dict[str, Any] = {}
    for condition in sorted({row["condition"] for row in available_rows}):
        target_values = [
            float(row["metrics"]["symmetry_adjusted"]["top3"]["h"])
            for row in available_rows if row["condition"] == condition
        ]
        observations: dict[str, list[float]] = {}
        for run in evaluated_runs:
            if run["condition"] == condition:
                observations.setdefault(run["target_id"], []).append(
                    float(run["metrics"]["symmetry_adjusted"]["top3"]["h"])
                )
        condition_statistics[condition] = {
            "analysis": "exploratory_available_case",
            "target_count": len(target_values),
            "target_macro_bootstrap": _jsonable(
                target_bootstrap_ci(target_values, draws=bootstrap_draws, seed=seed + 300_000)
            ),
            "hierarchical_bootstrap": _jsonable(
                hierarchical_bootstrap_ci(observations, draws=bootstrap_draws, seed=seed + 400_000)
            ),
        }

    target_condition_hits = {
        (row["target_id"], row["condition"]): float(
            row["metrics"]["symmetry_adjusted"]["top3"]["h"]
        )
        for row in available_rows
    }
    contrasts: list[dict[str, Any]] = []
    evaluable_contrast_indexes: list[int] = []
    for left, right in CONTRASTS:
        targets = sorted(
            target for target in expected_targets
            if (target, left) in target_condition_hits
            and (target, right) in target_condition_hits
        )
        item: dict[str, Any] = {
            "contrast": f"{left} - {right}",
            "left": left,
            "right": right,
            "analysis": "exploratory_paired_available_case",
            "paired_target_count": len(targets),
            "paired_targets": targets,
        }
        if targets:
            test = paired_sign_flip_test(
                [target_condition_hits[(target, left)] for target in targets],
                [target_condition_hits[(target, right)] for target in targets],
                seed=seed + 500_000,
            )
            item.update(_jsonable(test))
            evaluable_contrast_indexes.append(len(contrasts))
        else:
            item.update(
                {
                    "status": "not_evaluable_no_paired_available_targets",
                    "estimate": None,
                    "p_two_sided": None,
                }
            )
        contrasts.append(item)
    adjusted = holm_adjust(
        [contrasts[index]["p_two_sided"] for index in evaluable_contrast_indexes]
    ) if evaluable_contrast_indexes else ()
    for item in contrasts:
        item["holm_p"] = None
    for index, adjusted_p in zip(evaluable_contrast_indexes, adjusted):
        contrasts[index]["holm_p"] = adjusted_p

    return {
        "schema_version": 2,
        "material_passport": {
            "origin": "academic-research-suite / post-freeze-evaluator",
            "origin_mode": "run",
            "verification_status": "COMPUTED_FROM_FROZEN_SYNTHETIC_OR_USER_SUPPLIED_INPUTS",
            "version": "hotspot_blind_v2_terminal_outcomes",
        },
        "freeze": {
            "manifest_path": str(freeze.manifest_path),
            "manifest_sha256": freeze.manifest_sha256,
            "frozen_at": freeze.frozen_at.isoformat(),
            "verified_run_count": len(freeze.runs),
            "verified_terminal_outcome_count": freeze.terminal_outcome_count,
            "verified_prediction_hash_count": sum(
                run.prediction_path is not None for run in freeze.runs
            ),
            "verified_artifact_hash_count": freeze.verified_artifact_hash_count,
            "label_path": str(label_file),
            "label_sha256": _sha256_file(label_file),
            "labels_opened_after_freeze_validation": True,
            "blind_consensus_sha256": blind_consensus_sha256,
            "blind_consensus_fixed_before_labels_opened": True,
            "available_case_consensus_sha256": available_consensus_sha256,
            "available_case_consensus_fixed_before_labels_opened": True,
            "target_manifest_path": (
                str(resolved_target_manifest) if resolved_target_manifest is not None else None
            ),
        },
        "configuration": {
            "per_run_mc_draws": per_run_mc_draws,
            "joint_mc_draws": joint_mc_draws,
            "bootstrap_draws": bootstrap_draws,
            "seed": seed,
        },
        "runs": evaluated_runs,
        "outcomes": outcomes,
        "outcome_summary_by_condition": outcome_summary,
        "target_names": dict(sorted(target_names.items())),
        "consensus": consensus_rows,
        "available_case_consensus": available_rows,
        "consensus_rule": {
            "preregistered": "computed only for cells with exactly 3 valid predictions",
            "exploratory_available_case": (
                "n=1..3; raw frequency descending, reciprocal-rank sum descending, "
                "opaque residue token ascending"
            ),
        },
        "workflow_hit_rates": workflow_hits,
        "empirical_any_hit_per_3": workflow_hits,
        "primary": primary,
        "exploratory": {"available_case_joint_matched_null": exploratory_joint},
        "condition_statistics": condition_statistics,
        "contrasts": contrasts,
        "selection_bias": {
            "present": any(run.outcome != "success" for run in freeze.runs),
            "statement": (
                "Refusals, exclusions, and other failures can select an easier or otherwise "
                "non-representative subset. Available-case estimates condition on eligible, "
                "successful prediction production and must not be interpreted as full-workflow performance."
            ),
        },
        "limitations": [
            "The benchmark measures recovery of supplied residue labels, not affinity, specificity, foldability, or design success.",
            "Equivalent-chain adjustment is allowed only where source entity and full label-sequence correspondence are identical.",
            "Monte Carlo and bootstrap intervals retain simulation error at their configured draw counts.",
            "Refusal-related missingness can induce selection bias; exploratory available-case results condition on successful runs.",
            "Compliance-excluded predictions remain in the hash audit but are omitted from every score and consensus.",
        ],
    }


def _passport(result: Mapping[str, Any]) -> str:
    passport = result["material_passport"]
    return (
        "## ARS Material Passport\n\n"
        f"- Origin: {passport['origin']}\n"
        f"- Origin Mode: {passport['origin_mode']}\n"
        f"- Verification Status: {passport['verification_status']}\n"
        f"- Version: {passport['version']}\n"
    )


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _summary_markdown(result: Mapping[str, Any]) -> str:
    primary = result["primary"]
    condition_rows = []
    for condition, stats in sorted(result["condition_statistics"].items()):
        macro = stats["target_macro_bootstrap"]
        hierarchy = stats["hierarchical_bootstrap"]
        condition_rows.append(
            f"| {condition} | {_fmt(macro['estimate'])} | [{_fmt(macro['lower'])}, {_fmt(macro['upper'])}] "
            f"| {_fmt(hierarchy['estimate'])} | [{_fmt(hierarchy['lower'])}, {_fmt(hierarchy['upper'])}] |"
        )
    failure_rows = [
        f"| {condition} | {item['attempted']} | {item['valid']} | {item['refusals']} | "
        f"{item['excluded']} | {item['other_failures']} | {_fmt(item['valid_rate'])} | "
        f"{_fmt(item['refusal_rate'])} |"
        for condition, item in sorted(result["outcome_summary_by_condition"].items())
    ]
    exclusion_rows = [
        f"| {row['target_name']} (`{row['target_id']}`) | {row['condition']} | "
        f"{row['replicate']} | {row.get('outcome_reason') or 'unspecified'} |"
        for row in result.get("outcomes", []) if row.get("outcome") == "excluded"
    ]
    target_rows = [
        f"| `{target_id}` | {name} |"
        for target_id, name in sorted(result.get("target_names", {}).items())
    ]
    if primary["decision"] == "not_evaluable_due_to_refusals":
        primary_text = (
            f"Preregistered decision: **not_evaluable_due_to_refusals**. Only "
            f"{primary['complete_cells']}/{primary['expected_cells']} `{primary['condition']}` "
            "target cells had 3/3 valid predictions. No supported/not-supported conclusion is reported."
        )
    else:
        primary_text = (
            f"For `{primary['condition']}`, the three-run consensus Top3 symmetry-aware total was "
            f"{_fmt(primary['observed'])}; the joint matched-null mean was "
            f"{_fmt(primary['matched_null_mean'])}, with one-sided p={_fmt(primary['p_greater_equal'])} "
            f"from {primary['draws']} draws. Preregistered decision: **{primary['decision']}**. "
            f"The rule was: {primary['rule']}."
        )
    exploratory = result["exploratory"]["available_case_joint_matched_null"]
    if exploratory.get("target_count", 0):
        exploratory_text = (
            f"Across {exploratory['target_count']} anonymous targets with at least one valid "
            f"prediction, the available-case consensus total was {_fmt(exploratory['observed'])}; "
            f"matched-null mean {_fmt(exploratory['matched_null_mean'])}, one-sided "
            f"p={_fmt(exploratory['p_greater_equal'])} from {exploratory['draws']} draws."
        )
    else:
        exploratory_text = "No anonymous target had an available prediction."
    return (
        "# Blind Hotspot Benchmark Summary\n\n"
        + _passport(result)
        + "\n## Evidence\n\n"
        + "### Terminal outcomes and failure rates\n\n"
        + f"The validated freeze contains {result['freeze']['verified_run_count']} terminal outcomes, "
        + f"{result['freeze']['verified_prediction_hash_count']} stored/frozen prediction artifacts, and "
        + f"{result['freeze']['verified_artifact_hash_count']} verified artifact hashes.\n\n"
        + "| Condition | Attempted | Valid | Refusals | Excluded | Other failures | Valid rate | Refusal rate |\n"
        + "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(failure_rows)
        + ("\n\n### Excluded predictions\n\n"
           + "Excluded predictions remain hash-verified but are never scored or used in either consensus.\n\n"
           + "| Target | Condition | Replicate | Exclusion reason |\n|---|---|---:|---|\n"
           + "\n".join(exclusion_rows) if exclusion_rows else "")
        + "\n\n### Target manifest mapping\n\n"
        + "| Frozen target ID | Target name |\n|---|---|\n"
        + "\n".join(target_rows)
        + "\n\n## Inference\n\n"
        + "### Preregistered strict primary endpoint\n\n"
        + primary_text
        + "\n\n### Exploratory available-case results\n\n"
        + "**Exploratory:** " + exploratory_text + "\n\n"
        + "| Condition | Target-macro mean | 95% CI | Hierarchical mean | 95% CI |\n"
        + "|---|---:|---:|---:|---:|\n"
        + "\n".join(condition_rows)
        + "\n\nConsensus rule: n=1..3 valid predictions; raw frequency, then reciprocal-rank sum, "
        + "then opaque-token lexical tie-break.\n\n"
        + "## Refusal-related selection bias\n\n"
        + result["selection_bias"]["statement"] + "\n\n"
        + "## Limitations\n\n"
        + "\n".join(f"- {item}" for item in result["limitations"])
        + "\n"
    )


def _condition_markdown(result: Mapping[str, Any]) -> str:
    rows = [
        f"| {item['contrast']} | {item['paired_target_count']} | {_fmt(item['estimate'])} | "
        f"{_fmt(item['p_two_sided'])} | {_fmt(item['holm_p'])} |"
        for item in result["contrasts"]
    ]
    return (
        "# Exploratory Condition Comparisons\n\n"
        + _passport(result)
        + "\n## Evidence\n\n"
        + "All comparisons are exploratory. Paired target-level contrasts use only targets with "
        + "available-case consensus in both conditions and symmetry-aware Top3 exact hits.\n\n"
        + "| Contrast | Paired targets | Mean paired difference | Raw p | Holm p |\n"
        + "|---|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n## Inference\n\n"
        + "Holm-adjusted values control the family-wise error rate across the evaluable exploratory contrasts; they are not a preregistered primary conclusion.\n\n"
        + "## Refusal-related selection bias\n\n"
        + result["selection_bias"]["statement"] + "\n\n"
        + "## Limitations\n\n"
        + "The paired tests operate at target level; target count, not run count, determines inferential replication.\n"
    )


def _leakage_markdown(result: Mapping[str, Any]) -> str:
    freeze = result["freeze"]
    statuses: dict[str, int] = {}
    compliance_violations = 0
    excluded_predictions = sum(
        row.get("outcome") == "excluded" for row in result.get("outcomes", [])
    )
    for run in result["runs"]:
        status = str(run["audit"].get("recognition_status"))
        statuses[status] = statuses.get(status, 0) + 1
        compliance = run["audit"].get("compliance")
        if isinstance(compliance, Mapping) and any(
            compliance.get(key) is not False
            for key in ("labels_seen", "target_search_used", "other_runs_seen")
        ):
            compliance_violations += 1
    return (
        "# Leakage Audit\n\n"
        + _passport(result)
        + "\n## Evidence\n\n"
        + f"Freeze manifest SHA-256: `{freeze['manifest_sha256']}`. All "
        + f"{freeze['verified_run_count']} planned outcomes were accounted for; "
        + f"{freeze['verified_artifact_hash_count']} artifact hashes, including "
        + f"{freeze['verified_prediction_hash_count']} stored/frozen prediction artifacts "
        + f"(eligible plus {excluded_predictions} excluded), matched before the label file was opened. "
        + f"The label-blind consensus was fixed first with SHA-256 `{freeze['blind_consensus_sha256']}`. "
        + f"Recorded recognition statuses: {json.dumps(statuses, sort_keys=True)}. "
        + f"Recorded compliance violations among eligible predictions only: {compliance_violations}. "
        + "Excluded predictions are reported separately and are not hidden by this eligible-only count.\n\n"
        + "## Inference\n\n"
        + "The evaluator verified temporal and cryptographic ordering. Self-reported run metadata is audit evidence, not proof of absence of learned-memory recognition.\n\n"
        + "## Limitations\n\n"
        + "Filesystem timestamps can establish ordering only to filesystem resolution; the prediction hashes and exact run-plan match provide the stronger freeze check.\n"
    )


def _target_markdown(result: Mapping[str, Any], target_id: str) -> str:
    target_name = result.get("target_names", {}).get(target_id, target_id)
    consensus = [row for row in result["consensus"] if row["target_id"] == target_id]
    available = [
        row for row in result["available_case_consensus"] if row["target_id"] == target_id
    ]
    runs = [row for row in result["runs"] if row["target_id"] == target_id]
    strict_rows = [
        f"| {row['condition']} | {row['metrics']['strict']['top3']['h']} | "
        f"{row['metrics']['symmetry_adjusted']['top3']['h']} | "
        f"{row['metrics']['strict']['top6']['h']} | "
        f"{row['metrics']['symmetry_adjusted']['top6']['h']} | "
        f"{_fmt(row['metrics']['symmetry_adjusted']['top3']['hypergeometric_p'])} | "
        f"{_fmt(row['metrics']['symmetry_adjusted']['top6']['hypergeometric_p'])} |"
        for row in sorted(consensus, key=lambda item: item["condition"])
    ]
    available_rows = [
        f"| {row['condition']} | {row['valid_predictions']} | "
        f"{row['metrics']['strict']['top3']['h']} | "
        f"{row['metrics']['symmetry_adjusted']['top3']['h']} | "
        f"{row['metrics']['strict']['top6']['h']} | "
        f"{row['metrics']['symmetry_adjusted']['top6']['h']} | "
        f"{_fmt(row['metrics']['symmetry_adjusted']['top3']['hypergeometric_p'])} | "
        f"{_fmt(row['metrics']['symmetry_adjusted']['top6']['hypergeometric_p'])} |"
        for row in sorted(available, key=lambda item: item["condition"])
    ]
    exact_run_rows = []
    spatial_run_rows = []
    tolerance_run_rows = []
    pocket_run_rows = []
    null_run_rows = []
    for row in sorted(runs, key=lambda item: (item["condition"], item["replicate"])):
        metrics = row["metrics"]
        for mode_label, exact_mode, spatial_mode in (
            ("strict", "strict", "strict"),
            ("symmetry-aware", "symmetry_adjusted", "symmetry"),
        ):
            for cutoff in ("top3", "top6"):
                exact = metrics[exact_mode][cutoff]
                spatial = metrics["spatial"][f"{spatial_mode}_{cutoff}"]
                exact_run_rows.append(
                    f"| {row['condition']} | {row['replicate']} | {mode_label} | "
                    f"{cutoff.title()} | {exact['h']} | {_fmt(exact['h'] / exact['k'])} | "
                    f"{_fmt(exact['precision'])} | {_fmt(exact['recall'])} | "
                    f"{_fmt(exact['f1'])} | {_fmt(exact['jaccard'])} | "
                    f"{_fmt(exact['average_precision'])} | {_fmt(exact['enrichment'])} | "
                    f"{_fmt(exact['hypergeometric_p'])} |"
                )
                distance = spatial["distances"]
                spatial_run_rows.append(
                    f"| {row['condition']} | {row['replicate']} | {mode_label} | "
                    f"{cutoff.title()} | {_fmt(distance['minimum'])} | "
                    f"{_fmt(distance['chamfer'])} | {_fmt(distance['d90'])} | "
                    f"{_fmt(distance['hausdorff'])} |"
                )
                tolerance = spatial["tolerant"]
                tolerance_run_rows.append(
                    f"| {row['condition']} | {row['replicate']} | {mode_label} | "
                    f"{cutoff.title()} | "
                    + " | ".join(
                        _fmt(tolerance[threshold][field])
                        for threshold in ("4", "6", "8")
                        for field in ("precision", "recall", "f1")
                    )
                    + " |"
                )
                pocket = spatial["h6_overlap"]
                pocket_run_rows.append(
                    f"| {row['condition']} | {row['replicate']} | {mode_label} | "
                    f"{cutoff.title()} | {_fmt(pocket['jaccard'])} | "
                    f"{_fmt(pocket['dice'])} | "
                    f"{_fmt(pocket['rsasa_weighted_jaccard'])} | "
                    f"{_fmt(pocket['rsasa_weighted_dice'])} |"
                )
        exact_null = metrics["matched_null"]
        pocket_null = metrics["pocket_matched_null"]
        null_run_rows.append(
            f"| {row['condition']} | {row['replicate']} | "
            f"{_fmt(exact_null['observed'])} | {_fmt(exact_null['null_mean'])} | "
            f"{_fmt(exact_null['p_greater_equal'])} | {exact_null['draws']} | "
            f"{_fmt(pocket_null['observed'])} | {_fmt(pocket_null['null_mean'])} | "
            f"{_fmt(pocket_null['p_greater_equal'])} | {pocket_null['draws']} |"
        )
    hit_rows = [
        f"| {item['condition']} | {item['valid_runs']} | {item['symmetry_hit_runs']} | "
        f"{_fmt(item['symmetry_workflow_hit_rate'])} | "
        f"{_fmt(item['symmetry_conditional_hit_rate'])} |"
        for item in sorted(result["workflow_hit_rates"], key=lambda item: item["condition"])
        if item["target_id"] == target_id
    ]
    outcome_rows = []
    for condition in sorted({row["condition"] for row in result["outcomes"] if row["target_id"] == target_id}):
        cell = [
            row for row in result["outcomes"]
            if row["target_id"] == target_id and row["condition"] == condition
        ]
        outcome_rows.append(
            f"| {condition} | {len(cell)} | {sum(row['outcome'] == 'success' for row in cell)} | "
            f"{sum(row['outcome'] == 'refusal' for row in cell)} | "
            f"{sum(row['outcome'] == 'excluded' for row in cell)} | "
            f"{sum(row['outcome'] == 'failure' for row in cell)} |"
        )
    exclusion_rows = [
        f"| {row['condition']} | {row['replicate']} | {row.get('outcome_reason') or 'unspecified'} |"
        for row in result["outcomes"]
        if row["target_id"] == target_id and row["outcome"] == "excluded"
    ]
    strict_table = "\n".join(strict_rows) if strict_rows else "| _No complete 3/3 cell_ | | | | | | |"
    available_table = "\n".join(available_rows) if available_rows else "| _No available prediction_ | | | | | | | |"
    exact_run_table = "\n".join(exact_run_rows) if exact_run_rows else "| _No successful run_ | | | | | | | | | | | | | |"
    spatial_run_table = "\n".join(spatial_run_rows) if spatial_run_rows else "| _No successful run_ | | | | | | | |"
    tolerance_run_table = "\n".join(tolerance_run_rows) if tolerance_run_rows else "| _No successful run_ | | | | | | | | | | | | |"
    pocket_run_table = "\n".join(pocket_run_rows) if pocket_run_rows else "| _No successful run_ | | | | | | | |"
    null_run_table = "\n".join(null_run_rows) if null_run_rows else "| _No successful run_ | | | | | | | | | |"
    hit_table = "\n".join(hit_rows) if hit_rows else "| _No cell with a prediction_ | | | | |"
    return (
        f"# Target {target_name} (`{target_id}`)\n\n"
        + _passport(result)
        + "\n## Evidence\n\n"
        + "### Terminal outcomes\n\n"
        + "| Condition | Attempted | Valid | Refusals | Excluded | Other failures |\n"
        + "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(outcome_rows)
        + ("\n\n### Excluded prediction reasons\n\n"
           + "| Condition | Replicate | Reason |\n|---|---:|---|\n"
           + "\n".join(exclusion_rows) if exclusion_rows else "")
        + "\n\n## Preregistered three-run consensus\n\n"
        + "A row exists only when all three predictions in that target-condition cell are valid.\n\n"
        + "| Condition | Strict Top3 hits | Symmetry Top3 hits | Strict Top6 hits | Symmetry Top6 hits | Top3 hypergeometric p | Top6 hypergeometric p |\n"
        + "|---|---:|---:|---:|---:|---:|---:|\n"
        + strict_table
        + "\n\n## Exploratory available-case consensus\n\n"
        + "**Exploratory:** n=1..3; raw frequency, reciprocal-rank sum, then opaque-token tie-break.\n\n"
        + "| Condition | Valid n | Strict Top3 hits | Symmetry Top3 hits | Strict Top6 hits | Symmetry Top6 hits | Top3 hypergeometric p | Top6 hypergeometric p |\n"
        + "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + available_table
        + "\n\n## Successful-run exact and hypergeometric metrics\n\n"
        + "| Condition | Rep | Mode | Cutoff | h | h/K | Precision | Recall | F1 | Jaccard | AP | Enrichment | Hypergeom p |\n"
        + "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + exact_run_table
        + "\n\n## Successful-run spatial distance metrics\n\n"
        + "| Condition | Rep | Mode | Cutoff | Minimum Å | Bidirectional Chamfer Å | D90 Å | Hausdorff Å |\n"
        + "|---|---:|---|---|---:|---:|---:|---:|\n"
        + spatial_run_table
        + "\n\n## Successful-run 4/6/8 Å tolerant overlap\n\n"
        + "| Condition | Rep | Mode | Cutoff | 4Å P | 4Å R | 4Å F1 | 6Å P | 6Å R | 6Å F1 | 8Å P | 8Å R | 8Å F1 |\n"
        + "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + tolerance_run_table
        + "\n\n## Successful-run H6 pocket overlap\n\n"
        + "The weighted columns use residue relative solvent accessibility (rSASA), not absolute SASA.\n\n"
        + "| Condition | Rep | Mode | Cutoff | Pocket Jaccard | Pocket Dice | rSASA-weighted Jaccard | rSASA-weighted Dice |\n"
        + "|---|---:|---|---|---:|---:|---:|---:|\n"
        + pocket_run_table
        + "\n\n## Successful-run matched-null metrics\n\n"
        + "Exact MC is symmetry-aware Top3 exact h. Pocket MC is symmetry-aware Top3 H6 pocket Jaccard; both preserve chain × rSASA-quintile composition.\n\n"
        + "| Condition | Rep | Exact observed | Exact null mean | Exact p≥ | Exact draws | Pocket observed | Pocket null mean | Pocket p≥ | Pocket draws |\n"
        + "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + null_run_table
        + "\n\n## Workflow and conditional hit rates\n\n"
        + "Workflow rate is hit runs/3; conditional rate is hit runs/valid runs. A hit is a symmetry-aware Top3 exact hit > 0.\n\n"
        + "| Condition | Valid runs | Hit runs | Workflow hit rate | Conditional hit rate |\n"
        + "|---|---:|---:|---:|---:|\n"
        + hit_table
        + "\n\n## Inference\n\n"
        + "Target-level strict and exploratory values are descriptive components of the benchmark analyses.\n\n"
        + "### Refusal-related selection bias\n\n"
        + result["selection_bias"]["statement"]
        + "\n\n## Limitations\n\n"
        + "A residue-label hit is not a direct experimental measurement of binding or design quality.\n"
    )


def _csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("runs", "consensus", "available_case_consensus"):
        for item in result[kind]:
            metrics = item["metrics"]
            row = {
                "row_type": (
                    "run" if kind == "runs" else
                    "preregistered_consensus" if kind == "consensus" else
                    "exploratory_available_case_consensus"
                ),
                "target_id": item["target_id"],
                "target_name": item.get("target_name", item["target_id"]),
                "condition": item["condition"],
                "replicate": item.get("replicate", ""),
                "valid_predictions": item.get("valid_predictions", ""),
            }
            for mode in ("strict", "symmetry_adjusted"):
                for cutoff in ("top3", "top6"):
                    exact = metrics[mode][cutoff]
                    for field in (
                        "h", "precision", "recall", "f1", "jaccard",
                        "average_precision", "enrichment", "hypergeometric_p",
                    ):
                        row[f"{mode}_{cutoff}_{field}"] = exact[field]
            null = metrics.get("matched_null", {})
            row["matched_null_p"] = null.get("p_greater_equal", "")
            pocket_null = metrics.get("pocket_matched_null", {})
            row["pocket_matched_null_observed"] = pocket_null.get("observed", "")
            row["pocket_matched_null_p"] = pocket_null.get("p_greater_equal", "")
            row["pocket_matched_null_draws"] = pocket_null.get("draws", "")
            rows.append(row)
    return rows


def write_reports(result: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    per_target = output / "per_target"
    per_target.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = _csv_rows(result)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("row_type,target_id,condition,replicate\n")
    (output / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")
    (output / "condition_comparison.md").write_text(
        _condition_markdown(result), encoding="utf-8"
    )
    (output / "leakage_audit.md").write_text(_leakage_markdown(result), encoding="utf-8")
    targets = sorted(result.get("target_names", {}))
    for target in targets:
        safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in target)
        (per_target / f"{safe_name}.md").write_text(
            _target_markdown(result, target), encoding="utf-8"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--run-plan", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument(
        "--target-manifest",
        type=Path,
        help="frozen target manifest with optional equivalent_auth_chain_groups",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-run-mc-draws", type=int, default=9_999)
    parser.add_argument("--joint-mc-draws", type=int, default=99_999)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args(argv)
    result = evaluate_benchmark(
        args.freeze_manifest,
        args.run_plan,
        args.labels,
        target_manifest_path=args.target_manifest,
        per_run_mc_draws=args.per_run_mc_draws,
        joint_mc_draws=args.joint_mc_draws,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    write_reports(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
