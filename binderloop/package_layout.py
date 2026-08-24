"""Canonical on-disk names for a generated Binder harness project package."""

from pathlib import Path
from typing import List, Sequence, Union

PROJECT_PACKAGE_DIRNAME = "project_package"
LEGACY_PROJECT_PACKAGE_DIRNAMES: Sequence[str] = ("taiji_project_package",)


def is_project_package_name(name: str) -> bool:
    return name in {PROJECT_PACKAGE_DIRNAME, *LEGACY_PROJECT_PACKAGE_DIRNAMES}


def package_dir_candidates(run_root: Union[str, Path]) -> List[Path]:
    root = Path(run_root)
    return [root / PROJECT_PACKAGE_DIRNAME, *[root / name for name in LEGACY_PROJECT_PACKAGE_DIRNAMES]]


def resolve_package_dir(run_root: Union[str, Path]) -> Path:
    """Prefer the canonical package directory; fall back to a legacy name if it exists."""
    candidates = package_dir_candidates(run_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
