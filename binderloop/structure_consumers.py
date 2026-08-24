"""Explicit, fail-closed candidate-to-structure attribution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _normalized_path(value: Any) -> str:
    return str(value or "")


def _explicit_structure(row: Any, available: Sequence[str]) -> Optional[str]:
    raw = _row_value(row, "raw", {}) or {}
    candidates = []
    if isinstance(raw, Mapping):
        candidates.append(raw.get("structure_file"))
    candidates.append(_row_value(row, "structure_file"))
    available_set = {_normalized_path(value) for value in available}
    for value in candidates:
        path = _normalized_path(value)
        if path and path in available_set:
            return path
    return None


def _legacy_tokens(row: Any) -> List[str]:
    tokens: List[str] = []
    for key in ("id", "file_name", "design", "name", "candidate_id", "local_candidate_id"):
        value = _row_value(row, key)
        if value:
            tokens.extend((str(value), Path(str(value)).stem))
    return sorted({token for token in tokens if token}, key=len, reverse=True)


def _legacy_matches(row: Any, available: Sequence[str]) -> List[str]:
    tokens = _legacy_tokens(row)
    return [
        str(path) for path in available
        if any(Path(str(path)).stem == token or Path(str(path)).stem.endswith(token) for token in tokens)
    ]


def structure_files_for_candidates(candidates: Sequence[Any], structure_files: Sequence[str]) -> List[str]:
    """Resolve explicit bindings first; scoped rows without one fail closed."""
    selected: List[str] = []
    for row in candidates or []:
        explicit = _explicit_structure(row, structure_files)
        if explicit:
            selected.append(explicit)
            continue
        if _row_value(row, "identity_quality") == "scoped_source" or _row_value(row, "job_id"):
            continue
        selected.extend(_legacy_matches(row, structure_files))
    selected_set = set(selected)
    return [str(path) for path in structure_files or [] if str(path) in selected_set]


def success_structure_files(evaluation: Any, structure_files: Sequence[str]) -> List[str]:
    """Return explicitly attributed structures for compute-gate successes."""
    selected: List[str] = []
    for candidate in list(getattr(evaluation, "top_candidates", []) or []):
        tags = list(_row_value(candidate, "tags", []) or [])
        if tags != ["pass_compute_gate"]:
            continue
        explicit = _explicit_structure(candidate, structure_files)
        if explicit:
            selected.append(explicit)
    selected_set = set(selected)
    return [str(path) for path in structure_files or [] if str(path) in selected_set]


def _pae(row: Any) -> Optional[float]:
    for key in ("min_design_to_target_pae", "min_interaction_pae", "interaction_pae"):
        raw = _row_value(row, key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 < value < 1000.0:
            return value
    return None


def structure_interchain_pae(candidates: Sequence[Any], structure_files: Sequence[str]) -> Dict[str, float]:
    """Bind inter-chain PAE to explicit paths, with exact legacy suffix fallback."""
    result: Dict[str, float] = {}
    for row in candidates or []:
        pae = _pae(row)
        if pae is None:
            continue
        explicit = _explicit_structure(row, structure_files)
        if explicit:
            result[explicit] = min(pae, result.get(explicit, pae))
            continue
        if _row_value(row, "identity_quality") == "scoped_source" or _row_value(row, "job_id"):
            continue
        matches = _legacy_matches(row, structure_files)
        if len(matches) == 1:
            path = matches[0]
            result[path] = min(pae, result.get(path, pae))
    return {str(path): result[str(path)] for path in structure_files or [] if str(path) in result}
