import hashlib
import json
import os
import stat
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


RESUME_SCHEMA_VERSION = 4
PAYLOAD_CONTRACT_VERSION = 1
VALIDATION_CONTRACT_VERSION = 1
CORRECTION_PATCH_SCHEMA_VERSION = 1


class ResumeMismatchError(RuntimeError):
    """Raised when an output directory belongs to a different run identity."""


class ArtifactDigestCache:
    """Process-local artifact records with explicit write coherence.

    Reads populate the cache lazily. After any write, including
    ``atomic_write_text``/``atomic_write_json``, the caller must explicitly
    call ``invalidate(path)`` or ``update(path)``. A new process necessarily
    creates a new cache and reads the artifact again.
    """

    def __init__(self) -> None:
        self._records: Dict[str, dict] = {}
        self._lock = threading.RLock()

    def record(self, path: Union[str, Path]) -> dict:
        key = self._key(path)
        with self._lock:
            if key not in self._records:
                self._records[key] = _artifact_record_uncached(path)
            return dict(self._records[key])

    def update(self, path: Union[str, Path]) -> dict:
        """Read and store the current state after a caller-controlled write."""
        key = self._key(path)
        with self._lock:
            record = _artifact_record_uncached(path)
            self._records[key] = record
            return dict(record)

    def store(self, path: Union[str, Path], record: Mapping[str, Any]) -> dict:
        """Store caller-computed metadata without reading the artifact again."""
        key = self._key(path)
        with self._lock:
            value = dict(record)
            value.setdefault("path", str(Path(path)))
            self._records[key] = value
            return dict(value)

    def invalidate(self, path: Optional[Union[str, Path]] = None) -> None:
        """Forget one artifact, or all artifacts when path is omitted."""
        with self._lock:
            if path is None:
                self._records.clear()
            else:
                self._records.pop(self._key(path), None)

    @staticmethod
    def _key(path: Union[str, Path]) -> str:
        return str(Path(path))


def canonical_data(value: Any) -> Any:
    """Convert dataclasses/paths into stable JSON-compatible data."""
    if is_dataclass(value):
        return canonical_data(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): canonical_data(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonical_data(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(canonical_data(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Union[str, Path], text: str, *, encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def atomic_write_json(
    path: Union[str, Path], payload: Any, *, indent: int = 2,
    cache: Optional[ArtifactDigestCache] = None,
) -> Path:
    text = json.dumps(canonical_data(payload), ensure_ascii=False, indent=indent) + "\n"
    encoded = text.encode("utf-8")
    written = atomic_write_text(path, text)
    if cache is not None:
        cache.store(written, {
            "path": str(written), "exists": True,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
        })
    return written


def _artifact_record_uncached(path: Union[str, Path]) -> dict:
    artifact = Path(path)
    record = {"path": str(artifact), "exists": False}
    try:
        metadata = artifact.stat()
    except (FileNotFoundError, OSError):
        return record
    record["exists"] = True
    if stat.S_ISREG(metadata.st_mode):
        record.update({
            "sha256": file_sha256(artifact),
            "size_bytes": metadata.st_size,
        })
    return record


def artifact_record(path: Union[str, Path], cache: Optional[ArtifactDigestCache] = None) -> dict:
    if cache is None:
        return _artifact_record_uncached(path)
    return cache.record(path)


def artifacts_match(records: Any, cache: Optional[ArtifactDigestCache] = None) -> bool:
    if not isinstance(records, list):
        return False
    for record in records:
        if not isinstance(record, Mapping):
            return False
        path = Path(str(record.get("path") or ""))
        if cache is not None:
            current = cache.record(path)
            if not current.get("exists") or "sha256" not in current:
                return False
            if record.get("sha256") and current.get("sha256") != record.get("sha256"):
                return False
            if record.get("size_bytes") is not None and current.get("size_bytes") != int(record["size_bytes"]):
                return False
            continue
        current = _artifact_record_uncached(path)
        if not current.get("exists") or "sha256" not in current:
            return False
        if record.get("sha256") and current.get("sha256") != record.get("sha256"):
            return False
        if record.get("size_bytes") is not None and current.get("size_bytes") != int(record["size_bytes"]):
            return False
    return True


def _resolve_structure_path(structure_path: Union[str, Path], *, config_path: Optional[Path] = None) -> Path:
    path = Path(str(structure_path)).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if config_path is not None:
        candidates.append(Path(config_path).parent / path)
        candidates.append(Path(config_path).resolve().parents[1] / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path(config_path).parent / path).resolve() if config_path is not None else path.resolve()


def extract_target_identity(config: Any, *, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Build the resume fingerprint for target + user hard constraints.

    Rounds / max_rounds are intentionally excluded so a completed run can be
    extended by raising the total-round ceiling.
    """
    cfg = canonical_data(config)
    if not isinstance(cfg, dict):
        raise ResumeMismatchError("effective config must be an object")
    target = dict(cfg.get("target") or {})
    task = dict(cfg.get("task") or {})
    search = dict(cfg.get("search_space") or {})
    structure_path = (
        task.get("target_structure_path")
        or target.get("structure_path")
        or ""
    )
    resolved = None
    structure_sha = None
    if structure_path:
        resolved = _resolve_structure_path(structure_path, config_path=Path(config_path) if config_path else None)
        if resolved.exists():
            structure_sha = file_sha256(resolved)
    binder_length_range = task.get("binder_length_range")
    if binder_length_range is None:
        binder_length_range = search.get("binder_length_range")
    binder_length_step = task.get("binder_length_step")
    if binder_length_step is None:
        binder_length_step = search.get("binder_length_step", 10)
    owner = dict(cfg.get("owner") or {})
    hard = dict(owner.get("task_hard_constraints") or {})
    max_binders = hard.get("num_designs")
    if max_binders is None:
        max_binders = task.get("max_binders_per_round")
    if max_binders is None:
        max_binders = search.get("max_binders_per_round") or search.get("num_designs_per_round")
    identity = {
        "task_name": str(cfg.get("task_name") or task.get("task_name") or "binder_task"),
        "structure_path": str(structure_path),
        "structure_sha256": structure_sha,
        "chain_id": str(task.get("target_chain_id") or target.get("chain_id") or "A"),
        "hotspots": canonical_data(task.get("hotspots") or target.get("hotspots") or []),
        "include": canonical_data(task.get("target_include") or target.get("include") or []),
        "binding_types": canonical_data(task.get("target_binding_types") or target.get("binding_types") or []),
        "structure_groups": task.get("structure_groups") if task.get("structure_groups") is not None else target.get("structure_groups"),
        "profile": canonical_data(target.get("profile") or {}),
        "binder_length_range": canonical_data(binder_length_range),
        "binder_length_step": int(binder_length_step or 10),
        "task_num_designs": int(max_binders) if max_binders is not None else None,
    }
    return canonical_data(identity)


def extract_target_identity_from_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Read target identity from schema v2, or derive it from schema v1."""
    if not isinstance(manifest, Mapping):
        raise ResumeMismatchError("run manifest must be an object")
    identity = manifest.get("identity") if isinstance(manifest.get("identity"), Mapping) else {}
    if isinstance(identity.get("target_identity"), Mapping):
        return canonical_data(identity["target_identity"])
    # Schema v1 fallback: rebuild from effective_config.
    effective = identity.get("effective_config")
    config_source = identity.get("config_source") if isinstance(identity.get("config_source"), Mapping) else {}
    config_path = config_source.get("path")
    return extract_target_identity(effective, config_path=config_path)


def diff_target_identity(existing: Mapping[str, Any], current: Mapping[str, Any]) -> List[str]:
    diffs: List[str] = []
    keys = sorted(set(existing) | set(current))
    for key in keys:
        left = canonical_data(existing.get(key))
        right = canonical_data(current.get(key))
        if left != right:
            diffs.append(f"{key}: existing={left!r} current={right!r}")
    return diffs


def build_run_manifest(*, config_path: Union[str, Path], config: Any, cli_identity: Mapping[str, Any]) -> dict:
    config_path = Path(config_path)
    config_source = {
        "path": str(config_path),
        "sha256": file_sha256(config_path),
    }
    target_identity = extract_target_identity(config, config_path=config_path)
    identity = {
        "config_source": config_source,
        "effective_config": canonical_data(config),
        "cli_identity": canonical_data(dict(cli_identity)),
        "target_identity": target_identity,
    }
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        # Execution contracts are additive audit metadata. They are deliberately
        # excluded from target identity so upgrades never invalidate or resubmit
        # completed rounds and terminal execution records in an existing run.
        "payload_contract_version": PAYLOAD_CONTRACT_VERSION,
        "validation_contract_version": VALIDATION_CONTRACT_VERSION,
        "correction_patch_schema_version": CORRECTION_PATCH_SCHEMA_VERSION,
        "identity_hash": stable_hash(target_identity),
        "identity": identity,
    }


def validate_or_write_run_manifest(out_dir: Union[str, Path], manifest: Mapping[str, Any], *, force_new_run: bool = False) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "run_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force_new_run:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ResumeMismatchError(f"Existing run manifest is not valid JSON: {path}: {exc}") from exc
        try:
            existing_target = extract_target_identity_from_manifest(existing)
            current_target = extract_target_identity_from_manifest(manifest)
        except ResumeMismatchError:
            raise
        except Exception as exc:
            raise ResumeMismatchError(f"Unable to compare run identities in {path}: {exc}") from exc
        diffs = diff_target_identity(existing_target, current_target)
        if diffs:
            raise ResumeMismatchError(
                "Output directory already contains a different target/hard-constraint identity. "
                "Differences:\n- "
                + "\n- ".join(diffs)
                + "\nUse a new --out path, or pass --force-new-run only after confirming the directory is not a resume target."
            )
        # Same target identity: refresh audit snapshot (allows higher max_rounds).
        return atomic_write_json(path, manifest)
    if force_new_run and _has_resume_artifacts(out_dir):
        raise ResumeMismatchError(
            "--force-new-run refuses to overwrite a directory with existing round/memory artifacts. "
            "Choose a fresh --out path instead."
        )
    return atomic_write_json(path, manifest)


def _has_resume_artifacts(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    if (out_dir / "memory").exists() or (out_dir / "orchestrator_summary.json").exists():
        return True
    return any(child.is_dir() and child.name.startswith("round_") for child in out_dir.iterdir())


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


TEMPLATE_REPLAY_IDENTITY_VERSION = 1
TEMPLATE_REPLAY_STATES = frozenset({"exact_replay", "rematerialize_replay", "reject_replay", "lineage_unavailable"})
_TEMPLATE_PATH_KEYS = frozenset({"path", "file", "structure", "source_structure", "source_structure_file", "staged_source_structure_file", "current_target", "target_structure", "output_dir", "package_dir", "manifest_path"})
_TEMPLATE_OPERATIONAL_KEYS = frozenset({"host", "hostname", "gpu", "gpus", "devices", "shard", "shard_index", "rank", "worker", "output_dir", "package_dir", "log_file"})


def _semantic_projection(value: Any) -> Any:
    """Remove placement/path spelling while retaining content identities."""
    if isinstance(value, Mapping):
        projected = {}
        for key, item in value.items():
            name = str(key)
            if name in _TEMPLATE_OPERATIONAL_KEYS or name in _TEMPLATE_PATH_KEYS or (name == "structure" and isinstance(item, (str, Path))):
                continue
            if name == "digest" and any(k in value for k in _TEMPLATE_PATH_KEYS):
                # Recompute the digest from projected content; a digest produced
                # from an absolute-path-bearing object is not semantic identity.
                continue
            projected[name] = _semantic_projection(item)
        return {key: projected[key] for key in sorted(projected)}
    if isinstance(value, (list, tuple)):
        return [_semantic_projection(item) for item in value]
    if isinstance(value, Path):
        return None
    return value


def _content_identity(path_value: Any, supplied_digest: Any = "") -> Dict[str, Any]:
    path = Path(str(path_value or "")).expanduser() if path_value else None
    actual = file_sha256(path) if path is not None and path.is_file() else ""
    return {
        "sha256": actual or str(supplied_digest or ""),
        "content_available": bool(actual or supplied_digest),
    }


def build_template_execution_identity(
    params: Mapping[str, Any], *, target_structure: Any = "", target_chain: str = "",
    output_dir: Any = "", lineage_schema_version: Any = None,
    lineage_manifest_digest: str = "",
) -> Dict[str, Any]:
    """Build path/placement-independent template execution identity.

    Semantic identity controls attribution replay. Operational identity remains
    auditable but never rejects a replay merely because host/GPU/path changed.
    """
    values = dict(params or {})
    template = dict(values.get("binder_template") or {})
    plan = dict(values.get("template_application_plan") or {})
    alignment = dict(template.get("target_alignment") or plan.get("alignment") or {})
    residue_map = dict(template.get("source_to_effective_residue_map") or plan.get("source_to_effective_residue_map") or {})
    transform = dict(template.get("length_transform") or plan.get("length_transform") or {})
    source_path = template.get("staged_source_structure_file") or template.get("source_structure_file") or plan.get("source_structure") or ""
    source_digest = template.get("source_digest") or plan.get("source_digest") or ""
    target_identity = dict(plan.get("current_target_identity") or values.get("current_target_identity") or {})
    target_path = target_structure or target_identity.get("structure") or values.get("target_structure") or ""
    target_digest = target_identity.get("digest") or values.get("target_identity_digest") or ""
    lineage = dict(values.get("lineage_identity") or {})
    schema = lineage_schema_version if lineage_schema_version is not None else lineage.get("schema_version", values.get("lineage_schema_version"))
    manifest_digest = str(lineage_manifest_digest or lineage.get("manifest_digest") or values.get("lineage_manifest_digest") or values.get("candidate_manifest_digest") or "")
    semantic = {
        "identity_version": TEMPLATE_REPLAY_IDENTITY_VERSION,
        "policy": _semantic_projection(values.get("harness_template_policy") or values.get("template_policy") or {}),
        "target": {"content": _content_identity(target_path, target_digest), "chain": str(target_chain or target_identity.get("chain") or values.get("target_chain") or "")},
        "source": {"content": _content_identity(source_path, source_digest), "template_id": str(template.get("template_id") or plan.get("template_id") or "")},
        "alignment": _semantic_projection(alignment),
        "residue_map": _semantic_projection(residue_map),
        "length_transform": _semantic_projection(transform),
        "template_application_plan": _semantic_projection(plan),
        "matched_group": _semantic_projection({"matched_group_id": values.get("matched_group_id"), "matched_comparison": values.get("matched_comparison")}),
    }
    # Persist each attribution-bearing component digest as part of the semantic
    # contract. This makes replay decisions explainable and prevents callers
    # from accidentally treating only the aggregate digest as authoritative.
    semantic["component_digests"] = {
        name: stable_hash(semantic[name])
        for name in (
            "policy", "target", "source", "alignment", "residue_map",
            "length_transform", "template_application_plan", "matched_group",
        )
    }
    operational = {
        "target_path": str(target_path or ""), "source_path": str(source_path or ""),
        "output_dir": str(output_dir or values.get("output_dir") or ""),
        "host": values.get("host") or values.get("hostname"), "devices": values.get("devices"),
        "gpu": values.get("gpu"), "shard": values.get("shard_index") or values.get("shard"),
    }
    try:
        normalized_schema = int(schema) if not isinstance(schema, str) or schema.isdigit() else 0
    except (TypeError, ValueError):
        normalized_schema = 0
    # Only the current candidate-lineage v2 contract enables exact attribution.
    # Historical v22/v1 and unversioned manifests remain audit-only.
    lineage_available = bool(normalized_schema == 2 and manifest_digest)
    return {
        "schema_version": TEMPLATE_REPLAY_IDENTITY_VERSION,
        "semantic": semantic, "semantic_digest": stable_hash(semantic),
        "operational": operational, "operational_digest": stable_hash(operational),
        "lineage": {"schema_version": schema, "manifest_digest": manifest_digest},
        "lineage_available": lineage_available,
        "lineage_status": "available" if lineage_available else "lineage_unavailable",
        "materialization_available": bool(
            (not source_path or (Path(str(source_path)).expanduser().is_file()))
            and (not target_path or (Path(str(target_path)).expanduser().is_file()))
        ),
    }


def classify_template_replay(existing: Mapping[str, Any], current: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify safe template execution replay separately from exact attribution.

    Candidate lineage is post-execution diagnostic metadata. Its absence or a
    changed manifest digest may disable exact four-stage attribution, but must not
    reject a replay whose template source, target, alignment, residue mapping,
    length transform, application plan, and matched-control semantics are equal.
    """
    old, new = dict(existing or {}), dict(current or {})

    def execution_semantic(identity: Mapping[str, Any]) -> Dict[str, Any]:
        semantic = dict(identity.get("semantic") or {})
        # Backward compatibility for identity v1 snapshots that embedded lineage
        # in the execution semantic contract.
        semantic.pop("lineage", None)
        components = dict(semantic.get("component_digests") or {})
        components.pop("lineage", None)
        if components:
            semantic["component_digests"] = components
        return semantic

    old_sem, new_sem = execution_semantic(old), execution_semantic(new)
    differences = diff_target_identity(old_sem, new_sem)
    if differences:
        return {"status": "reject_replay", "exact_attribution": False, "audit_ingestion_allowed": True, "reason": "template_semantic_identity_mismatch", "differences": differences}
    if stable_hash(old_sem) != stable_hash(new_sem):
        return {"status": "reject_replay", "exact_attribution": False, "audit_ingestion_allowed": True, "reason": "template_semantic_digest_mismatch"}
    if not new.get("materialization_available", True):
        return {"status": "rematerialize_replay", "exact_attribution": False, "audit_ingestion_allowed": True, "reason": "semantic_match_materialization_missing"}
    old_lineage = dict(old.get("lineage") or (old.get("semantic") or {}).get("lineage") or {})
    new_lineage = dict(new.get("lineage") or (new.get("semantic") or {}).get("lineage") or {})
    exact_attribution = bool(
        old.get("lineage_available") and new.get("lineage_available")
        and old_lineage == new_lineage
    )
    return {
        "status": "exact_replay",
        "exact_attribution": exact_attribution,
        "audit_ingestion_allowed": True,
        "reason": "template_semantic_identity_match" if exact_attribution else "template_semantic_identity_match_lineage_unavailable_or_changed",
        "operational_changed": old.get("operational_digest") != new.get("operational_digest"),
    }
