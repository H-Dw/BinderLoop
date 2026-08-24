"""Fail-closed post-ingest parity validation for template attribution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from binderloop.execution_governance import stable_digest
from binderloop.lineage import safe_relative_path, validate_records

REQUIRED_TEMPLATE_DIGESTS = ("source", "alignment", "residue_map", "length_transform", "design_spec", "inverse_fold_mask")


@dataclass(frozen=True)
class PostIngestParity:
    schema_version: int
    status: str
    failures: tuple
    checks: Dict[str, Any]
    digest: str

    @property
    def evaluable(self) -> bool:
        return self.status == "validated"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_post_ingest_parity(root: Path, manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any]],
                                *, expected: Optional[Mapping[str, Any]] = None) -> PostIngestParity:
    """Validate every identity needed before structural attribution.

    ``expected`` is the pre-submit execution-plan identity when available.  Paths
    are transport metadata and are validated separately; semantic equality is
    established only by digests.
    """
    root = Path(root); rows = [dict(row) for row in records]
    failures = []; checks: Dict[str, Any] = {}
    run = dict(manifest.get("run") or {})
    identity = dict(manifest.get("identity") or {})
    digests = dict(manifest.get("digests") or {})
    expected = dict(expected or {})
    expected_run = dict(expected.get("run") or {})
    expected_digests = dict(expected.get("digests") or expected.get("template_artifact_digests") or {})
    declares_candidate_attribution = manifest.get("candidate_attribution") is True
    job_level_only = manifest.get("candidate_attribution") is False
    checks["attribution_scope"] = str(manifest.get("attribution_scope") or ("job" if job_level_only else "candidate" if declares_candidate_attribution else "legacy"))
    if job_level_only and checks["attribution_scope"] != "job":
        failures.append("job_attribution_scope_invalid")
    if declares_candidate_attribution and not rows and not manifest.get("candidate_manifests"):
        failures.append("candidate_attribution_unsubstantiated")
    if not job_level_only:
        for key in ("run_id", "run_namespace", "branch", "round_id", "job_id", "template_id"):
            if key != "template_id" and not run.get(key): failures.append(f"run_identity_missing:{key}")
            if expected_run.get(key) not in (None, "") and str(run.get(key)) != str(expected_run[key]): failures.append(f"run_identity_mismatch:{key}")
        for key in REQUIRED_TEMPLATE_DIGESTS:
            actual = str(digests.get(key) or digests.get({"length_transform":"transform", "inverse_fold_mask":"mask"}.get(key, key)) or "")
            wanted = str(expected_digests.get(key) or "")
            if run.get("template_id") and not actual: failures.append(f"template_digest_missing:{key}")
            if wanted and actual != wanted: failures.append(f"template_digest_mismatch:{key}")
    elif expected_run.get("job_id") not in (None, "") and str(identity.get("job_id") or "") != str(expected_run["job_id"]):
        failures.append("run_identity_mismatch:job_id")
    try:
        lineage = validate_records(rows, expected_run_id=str(run.get("run_id") or ""))
        checks["lineage"] = lineage
        declared_lineage = dict(manifest.get("lineage_summary") or {})
        if declared_lineage.get("record_count") is not None and int(declared_lineage["record_count"]) != lineage["record_count"]: failures.append("lineage_record_count_mismatch")
        if digests.get("lineage") and str(digests["lineage"]) != lineage["digest"]: failures.append("lineage_digest_mismatch")
    except (TypeError, ValueError) as exc:
        failures.append(f"lineage_invalid:{exc}")
    schema_version = int(manifest.get("schema_version", 1) or 1)
    listed = set(map(str, manifest.get("files") or []))
    if schema_version >= 3:
        for relative in listed:
            if not safe_relative_path(relative):
                failures.append(f"result_file_path_invalid:{relative}")
            elif not (root / relative).is_file():
                failures.append(f"result_file_missing:{relative}")
        final_records = {
            str(row.get("canonical_alias") or ""): row for row in rows
            if row.get("stage") == "final_refold"
        }
        metric_paths = [root / relative for relative in listed if Path(relative).name.startswith("final_designs_metrics") and relative.endswith(".csv")]
        if final_records and not metric_paths:
            failures.append("final_metrics_missing")
        metric_aliases = set()
        for metric_path in metric_paths:
            with metric_path.open(newline="", encoding="utf-8") as handle:
                for csv_row in csv.DictReader(handle):
                    alias = str(csv_row.get("canonical_alias") or csv_row.get("id") or "").strip()
                    record = final_records.get(alias)
                    if record is None:
                        failures.append(f"final_metrics_alias_missing:{alias}")
                        continue
                    metric_aliases.add(alias)
                    structure = dict(record.get("artifacts") or {}).get("structure") or {}
                    digest = str(structure.get("sha256") if isinstance(structure, Mapping) else "")
                    if str(csv_row.get("id") or "") != alias:
                        failures.append(f"final_metrics_id_alias_mismatch:{alias}")
                    if str(csv_row.get("global_candidate_id") or "") != str(record.get("global_candidate_id") or ""):
                        failures.append(f"final_metrics_global_id_mismatch:{alias}")
                    if str(csv_row.get("artifact_digest") or "") != digest:
                        failures.append(f"final_metrics_artifact_digest_mismatch:{alias}")
        if metric_paths and metric_aliases != set(final_records):
            failures.append("final_metrics_manifest_candidate_mismatch")
    alias_to_global: Dict[str, str] = {}
    for row in rows:
        if int(row.get("schema_version", 0) or 0) < 3:
            continue
        alias = str(row.get("canonical_alias") or "").strip()
        cid = str(row.get("global_candidate_id") or "").strip()
        if not alias:
            failures.append(f"canonical_alias_missing:{cid}:{row.get('stage')}")
            continue
        prior = alias_to_global.get(alias)
        if prior is not None and prior != cid:
            failures.append(f"canonical_alias_global_id_mismatch:{alias}:{prior}:{cid}")
        alias_to_global[alias] = cid
    checks["canonical_alias_count"] = len(alias_to_global)
    for ref in list(manifest.get("candidate_manifests") or []) + list(manifest.get("shard_manifests") or []):
        if not safe_relative_path(str(ref)): failures.append(f"unsafe_manifest_path:{ref}")
        elif str(ref) not in listed and str(ref) not in set(map(str, manifest.get("shard_manifests") or [])): failures.append(f"manifest_path_unlisted:{ref}")
        elif not (root / str(ref)).is_file(): failures.append(f"manifest_path_missing:{ref}")
    for row in rows:
        for kind, raw in dict(row.get("artifacts") or {}).items():
            value = dict(raw) if isinstance(raw, Mapping) else {"path": raw}
            relative = str(value.get("path") or "")
            if not safe_relative_path(relative): failures.append(f"artifact_path_invalid:{row.get('global_candidate_id')}:{row.get('stage')}:{kind}"); continue
            path = root / relative
            if not path.is_file(): failures.append(f"artifact_missing:{row.get('global_candidate_id')}:{row.get('stage')}:{kind}"); continue
            wanted = str(value.get("sha256") or value.get("digest") or "")
            if not wanted: failures.append(f"artifact_digest_missing:{row.get('global_candidate_id')}:{row.get('stage')}:{kind}")
            elif _sha256(path) != wanted: failures.append(f"artifact_digest_mismatch:{row.get('global_candidate_id')}:{row.get('stage')}:{kind}")
    shards = list(manifest.get("shards") or [])
    if shards:
        shard_digest = stable_digest(shards)
        checks["shard_digest"] = shard_digest
        if digests.get("shards") and str(digests["shards"]) != shard_digest: failures.append("shard_digest_mismatch")
    body = {"schema_version": 1, "status": "validated" if not failures else "not_evaluable", "failures": tuple(failures), "checks": checks}
    return PostIngestParity(**body, digest=stable_digest(body))
