"""Local, identity-free residue tables for LLM hotspot selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from binderloop.analysis.structure_features import (
    AA3_TO_1,
    HYDROPHOBIC,
    NEGATIVE,
    POLAR,
    POSITIVE,
    AtomRecord,
    parse_structure,
    _dist,
    _residue_groups,
)

KYTE_DOOLITTLE = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}
AROMATIC = {"PHE", "TRP", "TYR", "HIS"}
NEIGHBOR_CUTOFF = 8.0
PATCH_CUTOFF = 8.0


@dataclass
class ResidueDescriptor:
    token: str
    chain: str
    resseq: int
    aa1: str
    aa3: str
    hydrophobicity: float
    charge: float
    aromatic: bool
    polar: bool
    hydrophobic: bool
    ca_neighbor_count: int
    exposure_percentile: float
    patch_hydrophobic_fraction: float
    patch_charged_fraction: float
    patch_aromatic_count: int

    def to_prompt_row(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "aa": self.aa1,
            "hydrophobicity": round(self.hydrophobicity, 3),
            "charge": self.charge,
            "aromatic": self.aromatic,
            "polar": self.polar,
            "hydrophobic": self.hydrophobic,
            "ca_neighbors": self.ca_neighbor_count,
            "exposure_pct": round(self.exposure_percentile, 3),
            "patch_hydrophobic": round(self.patch_hydrophobic_fraction, 3),
            "patch_charged": round(self.patch_charged_fraction, 3),
            "patch_aromatic": self.patch_aromatic_count,
        }


@dataclass
class TargetResidueTable:
    chain_id: str
    residue_count: int
    sequence: str
    residues: List[ResidueDescriptor] = field(default_factory=list)
    truncated: bool = False

    def prompt_payload(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "residue_count": self.residue_count,
            "sequence": self.sequence,
            "truncated": self.truncated,
            "residues": [item.to_prompt_row() for item in self.residues],
        }

    def tokens(self) -> List[str]:
        return [item.token for item in self.residues]

    def by_token(self) -> Dict[str, ResidueDescriptor]:
        return {item.token: item for item in self.residues}


def build_target_residue_table(
    structure_file: Union[str, Path],
    *,
    chain_id: str,
    max_residues: int = 200,
) -> TargetResidueTable:
    atoms = parse_structure(structure_file)
    chain = str(chain_id or "").strip() or "A"
    chain_atoms = [atom for atom in atoms if atom.chain == chain]
    groups = _residue_groups(chain_atoms)
    ordered_ids = sorted(groups, key=_residue_sort_key)
    ca_coords: Dict[str, Tuple[float, float, float]] = {}
    descriptors: List[ResidueDescriptor] = []
    for residue_id in ordered_ids:
        residue_atoms = groups[residue_id]
        resname = residue_atoms[0].resname
        coord = _ca_coord(residue_atoms)
        if coord is not None:
            ca_coords[residue_id] = coord
        descriptors.append(
            ResidueDescriptor(
                token=residue_id if ":" in residue_id else f"{chain}:{residue_id}",
                chain=chain,
                resseq=_resseq_of(residue_id),
                aa1=AA3_TO_1.get(resname, "X"),
                aa3=resname,
                hydrophobicity=float(KYTE_DOOLITTLE.get(resname, 0.0)),
                charge=_charge(resname),
                aromatic=resname in AROMATIC,
                polar=resname in POLAR,
                hydrophobic=resname in HYDROPHOBIC,
                ca_neighbor_count=0,
                exposure_percentile=0.0,
                patch_hydrophobic_fraction=0.0,
                patch_charged_fraction=0.0,
                patch_aromatic_count=0,
            )
        )
    neighbor_counts = _neighbor_counts(ca_coords)
    for item in descriptors:
        item.ca_neighbor_count = int(neighbor_counts.get(item.token, 0))
    _assign_exposure(descriptors)
    _assign_patch_composition(descriptors, ca_coords)
    sequence = "".join(item.aa1 for item in descriptors)
    truncated = False
    selected = descriptors
    if max_residues and len(descriptors) > int(max_residues):
        truncated = True
        selected = _truncate_exposed_clusters(descriptors, ca_coords, int(max_residues))
    return TargetResidueTable(
        chain_id=chain,
        residue_count=len(descriptors),
        sequence=sequence,
        residues=selected,
        truncated=truncated,
    )


def deterministic_surface_hotspots(
    table: TargetResidueTable,
    *,
    min_hotspots: int = 3,
    max_hotspots: int = 6,
) -> List[str]:
    """Pick a compact exposed hydrophobic/aromatic patch without LLM."""
    ranked = sorted(
        table.residues,
        key=lambda item: (
            -(item.exposure_percentile),
            -(1.0 if item.aromatic or item.hydrophobic else 0.0),
            item.ca_neighbor_count,
            item.resseq,
        ),
    )
    if not ranked:
        return []
    coords = {item.token: (item.resseq,) for item in table.residues}
    seed = ranked[0]
    selected = [seed]
    remaining = [item for item in ranked[1:]]
    target_count = max(int(min_hotspots), min(int(max_hotspots), max(1, len(ranked))))
    while len(selected) < target_count and remaining:
        best_index = 0
        best_score = None
        for index, candidate in enumerate(remaining):
            seq_dist = min(abs(candidate.resseq - item.resseq) for item in selected)
            score = (seq_dist, -candidate.exposure_percentile, candidate.resseq)
            if best_score is None or score < best_score:
                best_score = score
                best_index = index
        selected.append(remaining.pop(best_index))
    selected.sort(key=lambda item: item.resseq)
    _ = coords
    return [item.token for item in selected[: int(max_hotspots)]]


def parse_hotspot_token(value: Any) -> Tuple[str, Optional[int]]:
    text = str(value or "").strip()
    if not text:
        return "", None
    if ":" in text:
        chain, raw = text.split(":", 1)
    else:
        chain, raw = "", text
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == "-")
    try:
        return chain.strip(), int(digits)
    except ValueError:
        return chain.strip(), None


def format_hotspot_token(chain: str, resseq: int) -> str:
    return f"{chain}:{int(resseq)}"


def sanitize_hotspot_tokens(
    raw: Any,
    *,
    allowed_tokens: Sequence[str],
    chain_id: str,
    min_hotspots: int,
    max_hotspots: int,
    previous: Optional[Sequence[str]] = None,
    max_change_per_round: int = 2,
    fallback: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Keep residues on the parsed chain, clamp count, and apply change inertia."""
    notes: List[str] = []
    allowed = {str(token) for token in allowed_tokens if str(token).strip()}
    allowed_by_number = {}
    for token in allowed:
        _chain, number = parse_hotspot_token(token)
        if number is not None:
            allowed_by_number.setdefault(number, token)
    chain = str(chain_id or "").strip()
    values = raw if isinstance(raw, (list, tuple, set)) else ([raw] if raw else [])
    cleaned: List[str] = []
    seen = set()
    for item in values:
        token_chain, number = parse_hotspot_token(item)
        if number is None:
            notes.append("dropped_unparseable:%s" % item)
            continue
        token = format_hotspot_token(token_chain or chain, number)
        if chain and token_chain and token_chain != chain:
            notes.append("dropped_wrong_chain:%s" % token)
            continue
        if token not in allowed:
            mapped = allowed_by_number.get(number)
            if mapped is None:
                notes.append("dropped_unknown_residue:%s" % token)
                continue
            token = mapped
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
        if len(cleaned) >= int(max_hotspots):
            if len(values) > int(max_hotspots):
                notes.append("clamped_to_max_hotspots")
            break
    previous_list = [str(item) for item in (previous or []) if str(item).strip()]
    if previous_list and int(max_change_per_round) >= 0:
        cleaned, inertia_notes = _apply_inertia(cleaned, previous_list, int(max_change_per_round), int(max_hotspots))
        notes.extend(inertia_notes)
    if len(cleaned) < int(min_hotspots):
        filler = [token for token in (fallback or previous_list or list(allowed_tokens)) if token not in cleaned]
        needed = int(min_hotspots) - len(cleaned)
        if needed > 0 and filler:
            cleaned.extend(filler[:needed])
            notes.append("padded_to_min_hotspots")
    if not cleaned and fallback:
        cleaned = list(fallback)[: int(max_hotspots)]
        notes.append("used_fallback_hotspots")
    return cleaned[: int(max_hotspots)], notes


def _apply_inertia(
    proposed: Sequence[str],
    previous: Sequence[str],
    max_change: int,
    max_hotspots: int,
) -> Tuple[List[str], List[str]]:
    previous_set = list(dict.fromkeys(previous))
    proposed_set = list(dict.fromkeys(proposed))
    kept = [token for token in previous_set if token in proposed_set]
    additions = [token for token in proposed_set if token not in previous_set]
    removals = [token for token in previous_set if token not in proposed_set]
    allowed_change = max(0, int(max_change))
    notes: List[str] = []
    if len(additions) + len(removals) <= allowed_change:
        merged = list(proposed_set)[:max_hotspots]
        return merged, notes
    kept_additions = additions[:allowed_change]
    leftover = max(0, allowed_change - len(kept_additions))
    dropped = removals[:leftover] if leftover else []
    retained_previous = [token for token in previous_set if token not in dropped]
    out = list(dict.fromkeys(retained_previous + kept_additions))[:max_hotspots]
    notes.append("inertia_capped_changes")
    _ = kept
    return out, notes


def _ca_coord(atoms: Sequence[AtomRecord]) -> Optional[Tuple[float, float, float]]:
    for name in ("CA", "CB", "N", "C"):
        for atom in atoms:
            if atom.name.upper() == name:
                return atom.coord
    return atoms[0].coord if atoms else None


def _residue_sort_key(residue_id: str) -> Tuple[int, str]:
    _chain, number = parse_hotspot_token(residue_id)
    return (number if number is not None else 10**9, residue_id)


def _resseq_of(residue_id: str) -> int:
    _chain, number = parse_hotspot_token(residue_id)
    return int(number or 0)


def _charge(resname: str) -> float:
    if resname in POSITIVE:
        return 1.0 if resname != "HIS" else 0.5
    if resname in NEGATIVE:
        return -1.0
    return 0.0


def _neighbor_counts(ca_coords: Mapping[str, Tuple[float, float, float]]) -> Dict[str, int]:
    tokens = list(ca_coords)
    counts = {token: 0 for token in tokens}
    for i, left in enumerate(tokens):
        for right in tokens[i + 1 :]:
            if _dist(ca_coords[left], ca_coords[right]) <= NEIGHBOR_CUTOFF:
                counts[left] += 1
                counts[right] += 1
    return counts


def _assign_exposure(descriptors: Sequence[ResidueDescriptor]) -> None:
    if not descriptors:
        return
    ranked = sorted(descriptors, key=lambda item: (item.ca_neighbor_count, item.resseq))
    n = max(1, len(ranked) - 1)
    for index, item in enumerate(ranked):
        # Low neighbor count => high exposure percentile.
        item.exposure_percentile = 1.0 - (index / n if n else 0.0)


def _assign_patch_composition(
    descriptors: Sequence[ResidueDescriptor],
    ca_coords: Mapping[str, Tuple[float, float, float]],
) -> None:
    by_token = {item.token: item for item in descriptors}
    for item in descriptors:
        coord = ca_coords.get(item.token)
        if coord is None:
            continue
        neighbors = [item]
        for other in descriptors:
            other_coord = ca_coords.get(other.token)
            if other_coord is None or other.token == item.token:
                continue
            if _dist(coord, other_coord) <= PATCH_CUTOFF:
                neighbors.append(other)
        count = max(1, len(neighbors))
        item.patch_hydrophobic_fraction = sum(1 for row in neighbors if row.hydrophobic) / count
        item.patch_charged_fraction = sum(1 for row in neighbors if row.charge != 0) / count
        item.patch_aromatic_count = sum(1 for row in neighbors if row.aromatic)
        _ = by_token


def _truncate_exposed_clusters(
    descriptors: Sequence[ResidueDescriptor],
    ca_coords: Mapping[str, Tuple[float, float, float]],
    max_residues: int,
) -> List[ResidueDescriptor]:
    ranked = sorted(descriptors, key=lambda item: (-item.exposure_percentile, item.resseq))
    selected: List[ResidueDescriptor] = []
    selected_tokens = set()
    for seed in ranked:
        if len(selected) >= max_residues:
            break
        if seed.token in selected_tokens:
            continue
        cluster = [seed]
        seed_coord = ca_coords.get(seed.token)
        if seed_coord is not None:
            for other in ranked:
                if other.token == seed.token or other.token in selected_tokens:
                    continue
                other_coord = ca_coords.get(other.token)
                if other_coord is None:
                    continue
                if _dist(seed_coord, other_coord) <= PATCH_CUTOFF * 1.5:
                    cluster.append(other)
        for item in cluster:
            if len(selected) >= max_residues:
                break
            if item.token in selected_tokens:
                continue
            selected.append(item)
            selected_tokens.add(item.token)
    selected.sort(key=lambda item: item.resseq)
    return selected


def compact_round_hotspot_evidence(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep residue-number evidence only; never file paths or protein names."""
    payload = dict(context or {})
    evaluation = dict(payload.get("evaluation") or {})
    structural = dict(payload.get("structural_analysis") or {})
    contacts = []
    for item in structural.get("summaries") or []:
        if not isinstance(item, Mapping):
            continue
        preview = item.get("contacts_preview") or []
        for contact in preview[:24]:
            if not isinstance(contact, Mapping):
                continue
            target = str(contact.get("target_residue") or "").strip()
            if target:
                contacts.append(target)
    engaged = []
    for key in ("aggregate_engaged_residues", "engaged_residues"):
        values = structural.get(key) or payload.get(key) or []
        if isinstance(values, Mapping):
            values = values.get("residues") or values.get("tokens") or []
        engaged.extend(str(item).strip() for item in values if str(item).strip())
    return {
        "success_count": evaluation.get("success_count"),
        "total_candidates": evaluation.get("total_candidates"),
        "success_rate": evaluation.get("success_rate"),
        "round_rank_key": list(payload.get("round_rank_key") or evaluation.get("round_rank_key") or []),
        "tag_counts": {
            key: int(value)
            for key, value in dict(evaluation.get("tag_counts") or {}).items()
            if key in {"hotspot_miss", "hotspot_not_covered", "binding_pose_failure", "pass_compute_gate"}
        },
        "aggregate_tags": {
            key: int(value)
            for key, value in dict(structural.get("aggregate_tags") or {}).items()
            if "hotspot" in str(key) or key in {"weak_or_tiny_interface", "structure_features_pass"}
        },
        "contact_target_residues": list(dict.fromkeys(contacts))[:40],
        "engaged_residues": list(dict.fromkeys(engaged))[:40],
        "previous_hotspots": list(payload.get("previous_hotspots") or []),
    }
