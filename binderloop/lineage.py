"""Versioned BoltzGen candidate lineage contract (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

SCHEMA_VERSION = 3
STAGES = ("initial_design", "inverse_folded", "before_refolding", "final_refold")


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity(kind: str, *parts: Any) -> str:
    digest = canonical_digest({"kind": kind, "parts": [str(x) for x in parts]})[:32]
    return f"{kind}_{digest}"


def make_identity_root(run_namespace: str, branch: str) -> str:
    """Explicit deployment-independent root for all candidate identities."""
    return _identity("root", run_namespace, branch)


def make_run_id(run_namespace: str, branch: str) -> str:
    return _identity("run", make_identity_root(run_namespace, branch))


def make_backbone_id(run_namespace: str, branch: str, logical_ordinal: int, *, identity_root: str = "") -> str:
    return _identity("bb", identity_root or make_identity_root(run_namespace, branch), int(logical_ordinal))


def make_inverse_fold_sequence_id(backbone_id: str, sequence_ordinal: int, sequence: str = "") -> str:
    return _identity("seq", backbone_id, int(sequence_ordinal), sequence)


def make_candidate_id(run_namespace: str, branch: str, logical_ordinal: int, backbone_id: str, sequence_id: str = "", *, identity_root: str = "") -> str:
    # Deliberately excludes host/GPU/shard/rank/path.
    return _identity("cand", identity_root or make_identity_root(run_namespace, branch), int(logical_ordinal), backbone_id, sequence_id)


def _slug(value: Any, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip()).strip("-").lower()
    return slug or fallback


def canonical_alias(logical_ordinal: int, sequence_ordinal: int, global_candidate_id: str,
                    *, run_namespace: str = "run", branch: str = "branch") -> str:
    """Stable readable v3 locator, independent of rank, shard, host, and paths."""
    candidate_id = str(global_candidate_id).strip()
    if not candidate_id:
        raise ValueError("global_candidate_id is required for canonical_alias")
    short_id = candidate_id.rsplit("_", 1)[-1][:12]
    return (f"{_slug(run_namespace, 'run')}__{_slug(branch, 'branch')}__"
            f"bb{int(logical_ordinal)}__s{int(sequence_ordinal)}__{short_id}")


def artifact_stem(source_stem: str, alias: str, stage: str) -> str:
    """Return the canonical v3 artifact stem; source_stem is legacy-only input."""
    if stage not in STAGES:
        raise ValueError(f"unsupported lineage stage: {stage}")
    return f"{alias}__{stage}"


def safe_relative_path(value: str) -> bool:
    path = Path(str(value))
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def make_stage_record(*, context: Mapping[str, Any], stage: str, logical_ordinal: int,
                      backbone_id: str, candidate_id: str, parent_candidate_id: Optional[str],
                      inverse_fold_sequence_id: Optional[str] = None,
                      artifacts: Optional[Mapping[str, Any]] = None,
                      structural: bool = True, source_key: str = "",
                      execution: Optional[Mapping[str, Any]] = None,
                      sequence_ordinal: int = 0, alias: str = "") -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unsupported lineage stage: {stage}")
    identity_root = str(context.get("identity_root") or make_identity_root(str(context["run_namespace"]), str(context["branch"])))
    canonical_name = alias or canonical_alias(logical_ordinal, sequence_ordinal, candidate_id, run_namespace=str(context["run_namespace"]), branch=str(context["branch"]))
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "candidate_stage",
        "run_id": str(context["run_id"]),
        "identity_root": identity_root,
        "run_namespace": str(context["run_namespace"]),
        "branch": str(context["branch"]),
        "round_id": str(context.get("round_id", "")),
        "job_id": str(context.get("job_id", "")),
        "template_id": str(context.get("template_id", "")),
        "stage": stage,
        "stage_index": STAGES.index(stage),
        "global_candidate_id": candidate_id,
        "canonical_alias": canonical_name,
        "parent_candidate_id": parent_candidate_id,
        "root_candidate_id": str(context.get("root_candidate_id") or (candidate_id if stage == "initial_design" else "")),
        "backbone_id": backbone_id,
        "inverse_fold_sequence_id": inverse_fold_sequence_id,
        "logical_ordinal": int(logical_ordinal),
        "sequence_ordinal": int(sequence_ordinal),
        "structural": bool(structural),
        "structure_status": "structural" if structural else "sequence_only",
        "source_key": str(source_key),
        "artifacts": dict(artifacts or {}),
        "execution": dict(execution or {}),
        "digests": dict(context.get("digests") or {}),
    }
    record["record_digest"] = canonical_digest({k: v for k, v in record.items() if k != "record_digest"})
    return record


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"lineage record {line_number} is not an object")
                records.append(value)
    return records


def validate_records(records: Iterable[Mapping[str, Any]], *, expected_run_id: str = "") -> dict[str, Any]:
    rows = [dict(row) for row in records]
    if not rows:
        return {
            "status": "no_candidates",
            "exact_attribution": False,
            "record_count": 0,
            "candidate_count": 0,
            "initial_count": 0,
            "digest": canonical_digest([]),
        }
    keys: set[tuple[str, str]] = set()
    candidates: dict[str, list[dict[str, Any]]] = {}
    initial_ids: set[str] = set()
    for row in rows:
        if int(row.get("schema_version", 0)) not in {2, SCHEMA_VERSION} or row.get("record_type") != "candidate_stage":
            raise ValueError("invalid candidate stage record")
        stage = str(row.get("stage"))
        if stage not in STAGES or int(row.get("stage_index", -1)) != STAGES.index(stage):
            raise ValueError(f"invalid stage ordering metadata: {stage}")
        cid = str(row.get("global_candidate_id") or "")
        if not cid:
            raise ValueError("missing global_candidate_id")
        if int(row.get("schema_version", 0) or 0) >= SCHEMA_VERSION:
            if not row.get("identity_root") or not row.get("canonical_alias"):
                raise ValueError(f"v3 candidate identity binding missing: {cid}:{stage}")
            expected_alias = canonical_alias(int(row["logical_ordinal"]), int(row.get("sequence_ordinal", 0)), cid, run_namespace=str(row.get("run_namespace") or ""), branch=str(row.get("branch") or ""))
            if row["canonical_alias"] != expected_alias:
                raise ValueError(f"candidate alias mismatch: {cid}:{stage}")
        key = (cid, stage)
        if key in keys:
            raise ValueError(f"duplicate candidate stage: {cid}:{stage}")
        keys.add(key)
        if expected_run_id and row.get("run_id") != expected_run_id:
            raise ValueError("candidate run_id mismatch")
        expected = canonical_digest({k: v for k, v in row.items() if k != "record_digest"})
        if row.get("record_digest") != expected:
            raise ValueError(f"candidate record digest mismatch: {cid}:{stage}")
        for value in dict(row.get("artifacts") or {}).values():
            artifact_path = value.get("path") if isinstance(value, Mapping) else value
            if artifact_path not in (None, "") and not safe_relative_path(str(artifact_path)):
                raise ValueError(f"unsafe candidate artifact path: {artifact_path}")
        candidates.setdefault(cid, []).append(row)
        if stage == "initial_design":
            if row.get("parent_candidate_id") not in (None, ""):
                raise ValueError("initial candidate may not have a parent")
            initial_ids.add(cid)
    all_ids = set(candidates)
    strict_v3 = all(int(row.get("schema_version", 0) or 0) >= SCHEMA_VERSION for row in rows)
    roots = {str(row.get("identity_root") or "") for row in rows}
    if strict_v3 and (len(roots) != 1 or "" in roots):
        raise ValueError("candidate identity_root mismatch or missing")
    contexts = {(str(row.get("run_id") or ""), str(row.get("run_namespace") or ""), str(row.get("branch") or "")) for row in rows}
    if strict_v3 and (len(contexts) != 1 or any(not value for value in next(iter(contexts)))):
        raise ValueError("candidate run context mismatch or missing")
    for cid, stages in candidates.items():
        by_stage = {str(row["stage"]): row for row in stages}
        for row in stages:
            parent = row.get("parent_candidate_id")
            if parent and parent not in all_ids:
                raise ValueError(f"missing parent candidate: {parent}")
            if row["stage"] in {"before_refolding", "final_refold"} and parent != cid:
                raise ValueError(f"sequence candidate identity not preserved: {cid}")
        if strict_v3 and "final_refold" in by_stage:
            required = {"inverse_folded", "before_refolding", "final_refold"}
            if not required.issubset(by_stage):
                raise ValueError(f"incomplete final candidate chain: {cid}")
            inverse = by_stage["inverse_folded"]
            initial_id = str(inverse.get("parent_candidate_id") or "")
            initial = candidates.get(initial_id, [])
            if len(initial) != 1 or initial[0].get("stage") != "initial_design":
                raise ValueError(f"final candidate missing initial_design: {cid}")
            invariant = ("run_id", "identity_root", "run_namespace", "branch", "backbone_id", "logical_ordinal", "inverse_fold_sequence_id", "canonical_alias")
            for key in invariant:
                values = {str(by_stage[stage].get(key) or "") for stage in required}
                if len(values) != 1:
                    raise ValueError(f"final candidate chain mismatch: {cid}:{key}")
            if str(initial[0].get("backbone_id") or "") != str(inverse.get("backbone_id") or ""):
                raise ValueError(f"final candidate backbone chain mismatch: {cid}")
    status = "validated"
    final_count = sum(1 for stages in candidates.values() if any(row["stage"] == "final_refold" for row in stages))
    return {"status": status, "exact_attribution": bool(final_count), "record_count": len(rows), "candidate_count": len(candidates), "initial_count": len(initial_ids), "final_count": final_count, "digest": canonical_digest(rows)}
