"""Versioned, lineage-driven fragment-template motif attribution.

All residue lookup is performed with the typed source-to-effective mapping.  The
module deliberately fails closed: identity/digest errors are ``not_evaluable``;
missing and sequence-only stages are ``not_available``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from binderloop.analysis.structure_features import AA3_TO_1, AtomRecord, parse_structure
from binderloop.execution_governance import stable_digest
from binderloop.lineage import STAGES, validate_records
from binderloop.templates.residue_identity import ResidueIdentity, parse_residue_identity

SCHEMA_VERSION = 1
COMPARISONS = (
    ("source", "initial_design"),
    ("initial_design", "inverse_folded"),
    ("inverse_folded", "before_refolding"),
    ("before_refolding", "final_refold"),
    ("source", "final_refold"),
)


@dataclass(frozen=True)
class AttributionValue:
    status: str
    value: Any = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotifAttributionRecord:
    schema_version: int
    record_type: str
    global_candidate_id: str
    root_candidate_id: str
    template_id: str
    from_stage: str
    to_stage: str
    status: str
    reason: str
    metrics: Dict[str, Dict[str, Any]]
    inputs: Dict[str, Any]
    record_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _value(status: str, value: Any = None, reason: str = "") -> Dict[str, Any]:
    return AttributionValue(status, value, reason).to_dict()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(root: Path, record: Mapping[str, Any], kind: str) -> Tuple[Optional[Path], str]:
    raw = dict(record.get("artifacts") or {}).get(kind)
    expected = ""
    if isinstance(raw, Mapping):
        expected = str(raw.get("sha256") or raw.get("digest") or "")
        raw = raw.get("path")
    if not raw:
        return None, "artifact_path_missing"
    path = root / str(raw)
    if not path.is_file():
        return None, "artifact_missing"
    if expected and _sha256(path) != expected:
        return None, "artifact_digest_mismatch"
    return path, ""


def _typed_mapping(raw: Mapping[str, Any]) -> Dict[ResidueIdentity, ResidueIdentity]:
    result: Dict[ResidueIdentity, ResidueIdentity] = {}
    for source, effective in dict(raw or {}).items():
        left = parse_residue_identity(source)
        right = parse_residue_identity(effective)
        if left in result or right in result.values():
            raise ValueError("non_bijective_effective_mapping")
        result[left] = right
    if not result:
        raise ValueError("effective_mapping_missing")
    return result


def _ca_by_identity(path: Path) -> Dict[ResidueIdentity, AtomRecord]:
    return {
        ResidueIdentity(atom.chain, atom.resseq, atom.icode): atom
        for atom in parse_structure(path) if atom.name == "CA"
    }


def _residue_atoms(path: Path) -> Dict[ResidueIdentity, List[AtomRecord]]:
    groups: Dict[ResidueIdentity, List[AtomRecord]] = {}
    for atom in parse_structure(path):
        groups.setdefault(ResidueIdentity(atom.chain, atom.resseq, atom.icode), []).append(atom)
    return groups


def _coords(records: Mapping[ResidueIdentity, AtomRecord], ids: Sequence[ResidueIdentity]) -> List[Tuple[float, float, float]]:
    return [records[item].coord for item in ids if item in records]


def _transform(source: Sequence[Tuple[float, float, float]], target: Sequence[Tuple[float, float, float]]):
    if len(source) != len(target) or len(source) < 3:
        raise ValueError("insufficient_alignment_atoms")
    import numpy as np
    src = np.asarray(source, dtype=float); dst = np.asarray(target, dtype=float)
    sc = src.mean(0); dc = dst.mean(0)
    u, _, vt = np.linalg.svd((src-sc).T @ (dst-dc)); rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1; rotation = u @ vt
    translation = dc - sc @ rotation
    aligned = src @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned-dst) ** 2, axis=1))))
    return rotation, translation, rmsd


def _rmsd(left, right) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("coordinate_cardinality_mismatch")
    return math.sqrt(sum(sum((a[i]-b[i])**2 for i in range(3)) for a,b in zip(left,right))/len(left))


def _sequence(groups: Mapping[ResidueIdentity, Sequence[AtomRecord]], ids: Sequence[ResidueIdentity]) -> str:
    return "".join(AA3_TO_1.get(groups[item][0].resname, "X") for item in ids if groups.get(item))


def _contacts(groups: Mapping[ResidueIdentity, Sequence[AtomRecord]], binder: Sequence[ResidueIdentity], targets: Sequence[ResidueIdentity], cutoff: float = 5.0) -> set:
    result = set(); limit = cutoff * cutoff
    for bid in binder:
        for tid in targets:
            if any((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2 <= limit for a in groups.get(bid, ()) for b in groups.get(tid, ())):
                result.add((bid.token, tid.token))
    return result


def _clashes(groups, binder, targets, cutoff: float = 2.0) -> Tuple[int, float]:
    count = pairs = 0; limit = cutoff * cutoff
    for bid in binder:
        for tid in targets:
            for a in groups.get(bid, ()):
                for b in groups.get(tid, ()):
                    if a.name.startswith("H") or b.name.startswith("H"): continue
                    pairs += 1
                    if (a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2 <= limit: count += 1
    return count, count / max(1, pairs)


def compare_structures(left_path: Path, right_path: Path, *, left_motif: Sequence[ResidueIdentity],
                       right_motif: Sequence[ResidueIdentity], left_patch: Sequence[ResidueIdentity],
                       right_patch: Sequence[ResidueIdentity], primary_contacts: Sequence[Tuple[str, str]] = ()) -> Dict[str, Dict[str, Any]]:
    left_ca = _ca_by_identity(left_path); right_ca = _ca_by_identity(right_path)
    motif_pairs = [(a,b) for a,b in zip(left_motif,right_motif) if a in left_ca and b in right_ca]
    patch_pairs = [(a,b) for a,b in zip(left_patch,right_patch) if a in left_ca and b in right_ca]
    motif_coverage = len(motif_pairs) / max(1, len(left_motif))
    patch_coverage = len(patch_pairs) / max(1, len(left_patch))
    if len(motif_pairs) < 3:
        raise ValueError("insufficient_mapped_motif_ca")
    motif_left = [left_ca[a].coord for a,_ in motif_pairs]; motif_right = [right_ca[b].coord for _,b in motif_pairs]
    _, _, self_rmsd = _transform(motif_right, motif_left)
    patch_rmsd = pose_rmsd = None
    if len(patch_pairs) >= 3:
        patch_left = [left_ca[a].coord for a,_ in patch_pairs]; patch_right = [right_ca[b].coord for _,b in patch_pairs]
        rotation, translation, patch_rmsd = _transform(patch_right, patch_left)
        import numpy as np
        posed = (np.asarray(motif_right) @ rotation + translation).tolist()
        pose_rmsd = _rmsd(motif_left, posed)
    left_groups = _residue_atoms(left_path); right_groups = _residue_atoms(right_path)
    left_ids = [a for a,_ in motif_pairs]; right_ids = [b for _,b in motif_pairs]
    left_seq = _sequence(left_groups, left_ids); right_seq = _sequence(right_groups, right_ids)
    identity = sum(a == b for a,b in zip(left_seq,right_seq)) / max(1, len(left_seq))
    left_contacts = _contacts(left_groups, left_ids, left_patch)
    right_contacts = _contacts(right_groups, right_ids, right_patch)
    expected_primary = set(primary_contacts) or left_contacts
    mapped_right = set()
    binder_map = {b.token:a.token for a,b in motif_pairs}; patch_map = {b.token:a.token for a,b in patch_pairs}
    for b,t in right_contacts:
        if b in binder_map and t in patch_map: mapped_right.add((binder_map[b], patch_map[t]))
    retained = expected_primary & mapped_right
    clash_count, clash_density = _clashes(right_groups, right_ids, right_patch)
    return {
        "motif_self_aligned_rmsd": _value("available", round(self_rmsd, 6)),
        "target_patch_aligned_motif_rmsd": _value("available", round(pose_rmsd, 6)) if pose_rmsd is not None else _value("not_available", reason="insufficient_target_patch_ca"),
        "target_patch_rmsd": _value("available", round(patch_rmsd, 6)) if patch_rmsd is not None else _value("not_available", reason="insufficient_target_patch_ca"),
        "target_patch_coverage": _value("available", round(patch_coverage, 6)),
        "motif_mapping_coverage": _value("available", round(motif_coverage, 6)),
        "mapped_sequence_identity": _value("available", round(identity, 6)),
        "primary_contact_retention": _value("available", round(len(retained)/max(1,len(expected_primary)), 6)) if expected_primary else _value("not_available", reason="primary_contacts_unspecified"),
        "contact_retention": _value("available", round(len(left_contacts & mapped_right)/max(1,len(left_contacts)), 6)) if left_contacts else _value("not_available", reason="reference_contacts_absent"),
        "clash_count": _value("available", clash_count),
        "clash_density": _value("available", round(clash_density, 8)),
        "quality": _value("available", {"matched_motif_ca": len(motif_pairs), "matched_patch_ca": len(patch_pairs), "right_contact_count": len(right_contacts)}),
    }


def _record(candidate_id: str, root_id: str, template_id: str, left: str, right: str,
            status: str, reason: str, metrics: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    body = {"schema_version": SCHEMA_VERSION, "record_type": "template_motif_attribution",
            "global_candidate_id": candidate_id, "root_candidate_id": root_id,
            "template_id": template_id, "from_stage": left, "to_stage": right,
            "status": status, "reason": reason, "metrics": metrics, "inputs": inputs}
    return MotifAttributionRecord(**body, record_digest=stable_digest(body)).to_dict()


def attribute_candidate_lineages(root: Path, records: Iterable[Mapping[str, Any]], template: Mapping[str, Any],
                                 *, parity_error: str = "") -> Dict[str, Any]:
    rows = [dict(row) for row in records]
    if not rows:
        return {"schema_version": SCHEMA_VERSION, "status": "lineage_unavailable", "comparisons": [], "reason": "historical_output_without_lineage"}
    validate_records(rows)
    try:
        mapping = _typed_mapping(template.get("source_to_effective_residue_map") or {})
        motif_source = [parse_residue_identity(x, default_chain=str(template.get("binder_chain") or "A")) for x in template.get("binder_residue_ids") or mapping.keys()]
        motif_effective = [mapping[item] for item in motif_source]
        alignment = dict(template.get("target_alignment") or {})
        patch_source = [parse_residue_identity(x, default_chain=str(alignment.get("source_target_chain") or "")) for x in (template.get("target_contact_residues") or alignment.get("residue_map", {}).keys())]
        patch_effective = [parse_residue_identity(alignment.get("residue_map", {}).get(item.token, item.token)) for item in patch_source]
    except (KeyError, TypeError, ValueError) as exc:
        parity_error = parity_error or str(exc)
        mapping = {}; motif_source = motif_effective = patch_source = patch_effective = []
    source = Path(str(template.get("staged_source_structure_file") or template.get("source_structure_file") or ""))
    by_candidate: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows: by_candidate.setdefault(str(row["global_candidate_id"]), {})[str(row["stage"])] = row
    output = []
    for cid, stages in sorted(by_candidate.items()):
        root_id = str(next(iter(stages.values())).get("root_candidate_id") or cid)
        root_stages = by_candidate.get(root_id, {})
        for left, right in COMPARISONS:
            left_row = None if left == "source" else (stages.get(left) or root_stages.get(left))
            right_row = stages.get(right)
            inputs = {"left_stage": left, "right_stage": right, "left_record_digest": (left_row or {}).get("record_digest", ""), "right_record_digest": (right_row or {}).get("record_digest", ""), "mapping_digest": stable_digest({a.token:b.token for a,b in mapping.items()}) if mapping else ""}
            if parity_error:
                output.append(_record(cid, root_id, str(template.get("template_id") or ""), left, right, "not_evaluable", parity_error, {}, inputs)); continue
            if right_row is None or (left != "source" and left_row is None):
                output.append(_record(cid, root_id, str(template.get("template_id") or ""), left, right, "not_available", "lineage_stage_missing", {}, inputs)); continue
            if not bool(right_row.get("structural", True)) or (left_row is not None and not bool(left_row.get("structural", True))):
                output.append(_record(cid, root_id, str(template.get("template_id") or ""), left, right, "not_available", "sequence_only_stage", {}, inputs)); continue
            left_path, left_error = (source, "") if left == "source" else _artifact_path(root, left_row, "structure")
            right_path, right_error = _artifact_path(root, right_row, "structure")
            if left_error or right_error or left_path is None or right_path is None or not left_path.is_file():
                reason = left_error or right_error or "source_artifact_missing"
                status = "not_evaluable" if "digest" in reason else "not_available"
                output.append(_record(cid, root_id, str(template.get("template_id") or ""), left, right, status, reason, {}, inputs)); continue
            try:
                metrics = compare_structures(left_path, right_path,
                    left_motif=motif_source if left == "source" else motif_effective, right_motif=motif_effective,
                    left_patch=patch_source if left == "source" else patch_effective, right_patch=patch_effective)
                output.append(_record(cid, root_id, str(template.get("template_id") or ""), left, right, "evaluated", "", metrics, inputs))
            except (OSError, ValueError) as exc:
                output.append(_record(cid, root_id, str(template.get("template_id") or ""), left, right, "not_evaluable", str(exc), {}, inputs))
    body = {"schema_version": SCHEMA_VERSION, "status": "evaluated", "comparisons": output}
    body["document_digest"] = stable_digest(body)
    return body
