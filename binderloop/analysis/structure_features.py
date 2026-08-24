import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _fmean(values: Iterable[float]) -> float:
    """``statistics.fmean`` is only available on Python 3.8+.

    The runtime environment can be Python 3.6, so fall back to a manual mean
    over a materialised list (also handles generator inputs safely).
    """
    _fast = getattr(statistics, "fmean", None)
    if _fast is not None:
        return _fast(values)
    data = list(values)
    return float(sum(data)) / len(data) if data else 0.0


AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "TYR"}
POSITIVE = {"LYS", "ARG", "HIS"}
NEGATIVE = {"ASP", "GLU"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "TRP"}


@dataclass(frozen=True)
class AtomRecord:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    element: str = ""

    @property
    def residue_id(self) -> str:
        return f"{self.chain}:{self.resseq}{self.icode.strip()}"

    @property
    def coord(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class ResidueContact:
    binder_residue: str
    target_residue: str
    min_distance: float
    contact_type: str


@dataclass
class BinderFragmentQuality:
    fragment_id: str
    start_residue: int
    end_residue: int
    residue_ids: List[str]
    residue_count: int
    interface_contact_count: int
    interface_residue_count: int
    hotspot_contact_count: int
    clash_count: int
    hbond_like_count: int
    salt_bridge_like_count: int
    hydrophobic_contact_count: int
    hydrophobic_fraction: float
    polar_fraction: float
    local_chain_break_count: int
    quality_score: float
    quality_label: str
    quality_rank: List[float] = field(default_factory=list)
    gate_failures: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    suggested_action: str = ""
    sequence: str = ""
    ca_coordinates: List[List[float]] = field(default_factory=list)


@dataclass
class StructureFeatureSummary:
    structure_file: str
    binder_chain: str
    target_chains: List[str]
    chain_detection_note: str
    atom_count: int
    binder_residue_count: int
    target_residue_count: int
    interface_contact_count: int
    interface_residue_count: int
    hotspot_contacts: Dict[str, int]
    hotspot_min_distances: Dict[str, float]
    clash_count: int
    clash_density: float
    hbond_like_count: int
    salt_bridge_like_count: int
    hydrophobic_contact_count: int
    binder_radius_of_gyration: float
    binder_end_to_end_distance: float
    binder_contact_order: float
    binder_surface_proxy: float
    interface_hydrophobic_fraction: float
    interface_polar_fraction: float
    chain_break_count: int
    reliability_score: float
    reliability_tags: List[str] = field(default_factory=list)
    contacts_preview: List[ResidueContact] = field(default_factory=list)
    fragment_qualities: List[BinderFragmentQuality] = field(default_factory=list)
    high_quality_fragments: List[BinderFragmentQuality] = field(default_factory=list)
    low_quality_fragments: List[BinderFragmentQuality] = field(default_factory=list)
    primary_coverage: Dict[str, float] = field(default_factory=dict)
    expanded_coverage: Dict[str, float] = field(default_factory=dict)
    negative_coverage: Dict[str, float] = field(default_factory=dict)
    heavy_atom_clash_count: int = 0
    heavy_atom_clash_density: float = 0.0
    clash_gate_pass: bool = True
    clash_rank: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["contacts_preview"] = [asdict(c) for c in self.contacts_preview]
        data["fragment_qualities"] = [asdict(f) for f in self.fragment_qualities]
        data["high_quality_fragments"] = [asdict(f) for f in self.high_quality_fragments]
        data["low_quality_fragments"] = [asdict(f) for f in self.low_quality_fragments]
        return data


@dataclass
class TargetStructureSummary:
    structure_file: str
    chain_id: str
    chain_residue_counts: Dict[str, int]
    chain_residue_spans: Dict[str, str]
    requested_hotspots: List[str]
    hotspots_present: List[str]
    hotspots_missing: List[str]
    boltzgen_target_include: List[Dict]
    boltzgen_target_binding_types: List[Dict]
    observations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_structure(path: Union[str, Path]) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    path = Path(path)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
            try:
                resname = line[17:20].strip().upper()
                if resname and resname not in AA3:
                    continue
                atoms.append(AtomRecord(int(line[6:11] or 0), line[12:16].strip(), resname, line[21].strip() or "_", int(line[22:26]), line[26].strip(), float(line[30:38]), float(line[38:46]), float(line[46:54]), line[76:78].strip() if len(line) >= 78 else ""))
                continue
            except ValueError:
                pass
        if line.startswith(("ATOM ", "HETATM ")):
            parts = line.split()
            try:
                if len(parts) >= 13:
                    parsed = _parse_mmcif_atom_parts(parts)
                    if parsed is None:
                        continue
                    serial, atom_name, resname, chain, resseq, xyz, element = parsed
                    atoms.append(AtomRecord(serial, atom_name, resname, chain, resseq, "", *xyz, element=element))
            except Exception:
                continue
    return atoms


def analyze_target_structure(structure_file: Union[str, Path], *, chain_id: str = "A", hotspots: Optional[Sequence[str]] = None) -> TargetStructureSummary:
    atoms = parse_structure(structure_file)
    residue_groups = _residue_groups(atoms)
    residues_by_chain: Dict[str, List[int]] = {}
    for residue_id in residue_groups:
        chain, _, token = residue_id.partition(":")
        number = _residue_number_from_id(residue_id)
        if token and number:
            residues_by_chain.setdefault(chain, []).append(number)

    chain_counts = {chain: len(set(numbers)) for chain, numbers in residues_by_chain.items()}
    chain_spans = {}
    for chain, numbers in residues_by_chain.items():
        unique_numbers = sorted(set(numbers))
        if unique_numbers:
            chain_spans[chain] = f"{unique_numbers[0]}..{unique_numbers[-1]}"

    requested = list(hotspots or [])
    normalized = [_normalize_hotspot(h) for h in requested]
    known_residues = set(residue_groups)
    present = [raw for raw, norm in zip(requested, normalized) if norm in known_residues]
    missing = [raw for raw, norm in zip(requested, normalized) if norm not in known_residues]
    include = [{"chain": {"id": chain_id, "res_index": chain_spans[chain_id]}}] if chain_id in chain_spans else [{"chain": {"id": chain_id}}]
    binding_values = [_residue_number_from_id(h) for h in normalized if h in known_residues]
    binding_types = [{"chain": {"id": chain_id, "binding": ",".join(str(v) for v in binding_values)}}] if binding_values else []

    observations: List[str] = []
    if not atoms:
        observations.append("No target atoms parsed; verify CIF/PDB path and format before running design.")
    elif chain_id not in chain_counts:
        observations.append(f"Requested target chain {chain_id} was not found; available chains={sorted(chain_counts)}.")
    else:
        observations.append(f"Target chain {chain_id} spans residues {chain_spans.get(chain_id, '?')} with {chain_counts[chain_id]} residues.")
    if missing:
        observations.append(f"Hotspots missing from parsed coordinates: {missing}.")

    return TargetStructureSummary(
        structure_file=str(structure_file),
        chain_id=chain_id,
        chain_residue_counts=chain_counts,
        chain_residue_spans=chain_spans,
        requested_hotspots=requested,
        hotspots_present=present,
        hotspots_missing=missing,
        boltzgen_target_include=include,
        boltzgen_target_binding_types=binding_types,
        observations=observations,
    )


def detect_binder_target_chains(
    residue_counts: Dict[str, int],
    *,
    binder_length: Optional[int] = None,
    configured_binder_chain: Optional[str] = None,
    configured_target_chains: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str], str]:
    """Resolve the binder vs target chains from the *actual* structure.

    BoltzGen relabels output chains by entity order (A, B, ...), which does not
    match the design-spec ``id`` values (e.g. binder ``D`` / target ``E``).
    Trusting the configured IDs therefore yields an empty interface and false
    ``hotspot_not_covered`` / ``weak_or_tiny_interface`` tags. We instead detect
    the binder chain by residue count (closest to the requested binder length)
    and treat the remaining chains as target.
    """
    chains = [c for c, n in residue_counts.items() if n > 0]
    if not chains:
        return (configured_binder_chain or "A"), list(configured_target_chains or []), "no_chains_parsed"
    configured_targets = {str(c) for c in (configured_target_chains or [])}
    if len(chains) == 1:
        only = chains[0]
        if configured_binder_chain and only == configured_binder_chain:
            return only, [], "single_chain_assumed_binder"
        if only in configured_targets:
            return only, [], "single_chain_assumed_target_no_binder"
        return only, [], "single_chain_assumed_binder"
    if binder_length and int(binder_length) > 0:
        binder = min(chains, key=lambda c: (abs(residue_counts[c] - int(binder_length)), residue_counts[c]))
        note = f"binder_by_length(len={int(binder_length)})"
    elif configured_binder_chain in chains and (not configured_targets or configured_targets.intersection(chains)):
        binder = configured_binder_chain
        note = "binder_by_configured_id"
    elif configured_targets and not configured_targets.intersection(chains) and "A" in chains:
        # BoltzGen writes output chains by entity order. For binder design specs
        # the generated binder entity is first, so output chain A is the binder
        # and configured target chain IDs (for example E) are not preserved.
        binder = "A"
        note = "binder_by_boltzgen_output_order"
    else:
        binder = min(chains, key=lambda c: residue_counts[c])
        note = "binder_by_smallest_chain"
    targets = [c for c in chains if c != binder]
    if configured_binder_chain and configured_binder_chain != binder:
        note += f";overrode_configured_binder={configured_binder_chain}"
    return binder, targets, note


def analyze_binder_structure(structure_file: Union[str, Path], *, binder_chain: str = "B", target_chains: Optional[Sequence[str]] = None, hotspots: Optional[Sequence[str]] = None, primary_residues: Optional[Sequence[str]] = None, expanded_residues: Optional[Sequence[str]] = None, negative_residues: Optional[Sequence[str]] = None, binder_length: Optional[int] = None, auto_detect_chains: bool = True, contact_cutoff: float = 5.0, clash_cutoff: float = 2.0, clash_density_max: float = 0.02, fragment_window: int = 8, fragment_stride: int = 4) -> StructureFeatureSummary:
    atoms = parse_structure(structure_file)
    hotspots = list(hotspots or [])
    detection_note = "configured_chains"
    if auto_detect_chains:
        residue_counts = {chain: len({a.residue_id for a in atoms if a.chain == chain}) for chain in {a.chain for a in atoms}}
        binder_chain, target_chains, detection_note = detect_binder_target_chains(
            residue_counts,
            binder_length=binder_length,
            configured_binder_chain=binder_chain,
            configured_target_chains=target_chains,
        )
    else:
        chains = sorted({a.chain for a in atoms})
        target_chains = list(target_chains) if target_chains is not None else [c for c in chains if c != binder_chain]
    target_chains = list(target_chains)
    binder_atoms = [a for a in atoms if a.chain == binder_chain]
    target_atoms = [a for a in atoms if a.chain in target_chains]
    contacts = _contacts(binder_atoms, target_atoms, cutoff=contact_cutoff)
    clashes = _contacts(binder_atoms, target_atoms, cutoff=clash_cutoff)
    heavy_atom_clashes, heavy_atom_pairs = _heavy_atom_clashes(binder_atoms, target_atoms, cutoff=clash_cutoff)
    binder_res = _residue_groups(binder_atoms)
    target_res = _residue_groups(target_atoms)
    interface_binder_res = {c.binder_residue for c in contacts}
    interface_resnames = [_resname_of_residue(binder_res, rid) for rid in interface_binder_res]
    hotspot_numbers = {_hotspot_residue_number(h) for h in hotspots if _hotspot_residue_number(h) is not None}
    hotspot_contacts: Dict[str, int] = {}
    hotspot_min: Dict[str, float] = {}
    for hotspot in hotspots:
        distances = [c.min_distance for c in contacts if _contact_matches_hotspot(c.target_residue, hotspot)]
        hotspot_contacts[hotspot] = len(distances)
        hotspot_min[hotspot] = min(distances) if distances else math.inf
    hbond_like = sum(1 for c in contacts if c.contact_type == "polar")
    salt_like = sum(1 for c in contacts if c.contact_type == "salt_bridge")
    hydrophobic = sum(1 for c in contacts if c.contact_type == "hydrophobic")
    rg = _radius_of_gyration([a.coord for a in binder_atoms if a.name == "CA"] or [a.coord for a in binder_atoms])
    e2e = _end_to_end([a for a in binder_atoms if a.name == "CA"])
    chain_breaks = _chain_breaks([a for a in binder_atoms if a.name == "CA"])
    hyd_frac = sum(1 for r in interface_resnames if r in HYDROPHOBIC) / max(1, len(interface_resnames))
    polar_frac = sum(1 for r in interface_resnames if r in POLAR or r in POSITIVE or r in NEGATIVE) / max(1, len(interface_resnames))
    reliability, tags = _reliability(len(interface_binder_res), len(clashes) / max(1, len(contacts)), hotspot_contacts, chain_breaks, rg, e2e, hyd_frac)
    fragment_qualities = _fragment_qualities(
        binder_res=binder_res,
        contacts=contacts,
        clashes=clashes,
        hotspot_numbers=hotspot_numbers,
        fragment_window=fragment_window,
        fragment_stride=fragment_stride,
    )
    high_fragments = [f for f in fragment_qualities if f.quality_label == "high"][:5]
    low_fragments = [f for f in fragment_qualities if f.quality_label == "low"][:5]
    primary = list(primary_residues if primary_residues is not None else hotspots)
    expanded = list(expanded_residues or [])
    negative = list(negative_residues or [])
    primary_coverage = _residue_coverage(primary, contacts)
    expanded_coverage = _residue_coverage(expanded, contacts)
    negative_coverage = _residue_coverage(negative, contacts)
    heavy_density = heavy_atom_clashes / max(1, heavy_atom_pairs)
    clash_gate_pass = heavy_density <= float(clash_density_max)
    clash_rank = [1.0 if clash_gate_pass else 0.0, -float(heavy_atom_clashes), -float(heavy_density)]
    return StructureFeatureSummary(str(structure_file), binder_chain, target_chains, detection_note, len(atoms), len(binder_res), len(target_res), len(contacts), len(interface_binder_res), hotspot_contacts, hotspot_min, len(clashes), len(clashes) / max(1, len(contacts)), hbond_like, salt_like, hydrophobic, rg, e2e, _contact_order(contacts), len(interface_binder_res) / max(1, len(binder_res)), hyd_frac, polar_frac, chain_breaks, reliability, tags, contacts[:50], fragment_qualities, high_fragments, low_fragments, primary_coverage, expanded_coverage, negative_coverage, heavy_atom_clashes, heavy_density, clash_gate_pass, clash_rank)




def motif_retention_metrics(
    reference_structure: Union[str, Path],
    candidate_structure: Union[str, Path],
    *,
    reference_chain: str,
    candidate_chain: str,
    residue_ids: Sequence[str],
    reference_sequence: str = "",
    reference_target_contacts: Optional[Sequence[str]] = None,
    contact_cutoff: float = 5.0,
) -> Dict[str, Union[float, int, str, List[str]]]:
    """Measure whether a template motif survives a generated/refolded structure.

    RMSD is computed after a rigid Kabsch alignment of matched CA atoms. Sequence
    identity and target-contact retention are reported independently so a branch
    cannot claim motif retention from one scalar alone.
    """
    reference_atoms = parse_structure(reference_structure)
    candidate_atoms = parse_structure(candidate_structure)
    numbers = [_residue_number_from_id(str(value)) for value in residue_ids or []]
    numbers = [value for value in numbers if value != 0]
    ref_ca = {atom.resseq: atom for atom in reference_atoms if atom.chain == reference_chain and atom.name == "CA" and atom.resseq in numbers}
    cand_ca = {atom.resseq: atom for atom in candidate_atoms if atom.chain == candidate_chain and atom.name == "CA" and atom.resseq in numbers}
    matched = sorted(set(ref_ca).intersection(cand_ca))
    rmsd = math.inf
    if matched:
        ref = [ref_ca[value].coord for value in matched]
        cand = [cand_ca[value].coord for value in matched]
        rmsd = _kabsch_rmsd(ref, cand)
    candidate_res = _residue_groups([atom for atom in candidate_atoms if atom.chain == candidate_chain])
    candidate_sequence = "".join(AA3_TO_1.get(_resname_of_residue(candidate_res, f"{candidate_chain}:{value}"), "X") for value in matched)
    expected_sequence = str(reference_sequence or "")[: len(matched)]
    sequence_identity = sum(a == b for a, b in zip(expected_sequence, candidate_sequence)) / max(1, len(expected_sequence)) if expected_sequence else 0.0
    candidate_targets = [atom for atom in candidate_atoms if atom.chain != candidate_chain]
    candidate_binder = [atom for atom in candidate_atoms if atom.chain == candidate_chain and atom.resseq in numbers]
    contacts = _contacts(candidate_binder, candidate_targets, cutoff=contact_cutoff)
    observed_contacts = sorted({contact.target_residue for contact in contacts})
    expected_contacts = {str(value) for value in (reference_target_contacts or [])}
    retained = sorted(expected_contacts.intersection(observed_contacts))
    contact_retention = len(retained) / max(1, len(expected_contacts)) if expected_contacts else 0.0
    return {
        "matched_ca_count": len(matched),
        "motif_rmsd": round(float(rmsd), 6) if math.isfinite(rmsd) else math.inf,
        "sequence_identity": round(float(sequence_identity), 6),
        "contact_retention": round(float(contact_retention), 6),
        "retained_target_contacts": retained,
        "observed_target_contacts": observed_contacts,
    }


def _kabsch_rmsd(reference: Sequence[Tuple[float, float, float]], candidate: Sequence[Tuple[float, float, float]]) -> float:
    if len(reference) != len(candidate) or not reference:
        return math.inf
    try:
        import numpy as np
        ref = np.asarray(reference, dtype=float)
        cand = np.asarray(candidate, dtype=float)
        ref -= ref.mean(axis=0)
        cand -= cand.mean(axis=0)
        u, _, vt = np.linalg.svd(cand.T @ ref)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = u @ vt
        aligned = cand @ rotation
        return float(np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1))))
    except Exception:
        return math.sqrt(sum(sum((a[i] - b[i]) ** 2 for i in range(3)) for a, b in zip(reference, candidate)) / len(reference))

def _fragment_qualities(
    *,
    binder_res: Dict[str, List[AtomRecord]],
    contacts: Sequence[ResidueContact],
    clashes: Sequence[ResidueContact],
    hotspot_numbers: Sequence[int],
    fragment_window: int,
    fragment_stride: int,
) -> List[BinderFragmentQuality]:
    ordered_ids = sorted(binder_res, key=lambda rid: _residue_number_from_id(rid))
    if not ordered_ids:
        return []
    window = max(3, int(fragment_window or 8))
    stride = max(1, int(fragment_stride or max(1, window // 2)))
    spans: List[List[str]] = []
    if len(ordered_ids) <= window:
        spans = [ordered_ids]
    else:
        for start in range(0, len(ordered_ids), stride):
            chunk = ordered_ids[start:start + window]
            if len(chunk) < max(3, window // 2) and spans:
                break
            spans.append(chunk)
            if start + window >= len(ordered_ids):
                break
    hotspot_number_set = {int(n) for n in hotspot_numbers}
    qualities = [_score_fragment(chunk, binder_res, contacts, clashes, hotspot_number_set) for chunk in spans]
    return sorted(qualities, key=lambda f: tuple(f.quality_rank), reverse=True)


def _score_fragment(
    residue_ids: Sequence[str],
    binder_res: Dict[str, List[AtomRecord]],
    contacts: Sequence[ResidueContact],
    clashes: Sequence[ResidueContact],
    hotspot_numbers: set,
) -> BinderFragmentQuality:
    residues = set(residue_ids)
    frag_contacts = [c for c in contacts if c.binder_residue in residues]
    frag_clashes = [c for c in clashes if c.binder_residue in residues]
    interface_res = {c.binder_residue for c in frag_contacts}
    hotspot_contacts = [c for c in frag_contacts if _residue_number_from_id(c.target_residue) in hotspot_numbers]
    resnames = [_resname_of_residue(binder_res, rid) for rid in residue_ids]
    hydrophobic_fraction = sum(1 for r in resnames if r in HYDROPHOBIC) / max(1, len(resnames))
    polar_fraction = sum(1 for r in resnames if r in POLAR or r in POSITIVE or r in NEGATIVE) / max(1, len(resnames))
    hbond_like = sum(1 for c in frag_contacts if c.contact_type == "polar")
    salt_like = sum(1 for c in frag_contacts if c.contact_type == "salt_bridge")
    hydrophobic = sum(1 for c in frag_contacts if c.contact_type == "hydrophobic")
    ca_atoms = [atoms[0] for rid in residue_ids for atoms in [sorted([a for a in binder_res.get(rid, []) if a.name == "CA"], key=lambda a: a.resseq)] if atoms]
    local_breaks = _chain_breaks(ca_atoms)
    interface_density = len(interface_res) / max(1, len(residue_ids))
    contact_density = len(frag_contacts) / max(1, len(residue_ids))
    clash_density = len(frag_clashes) / max(1, len(frag_contacts))
    chemistry_balance = 1.0 - min(1.0, abs(hydrophobic_fraction - 0.45) / 0.45)
    score = 0.25 + 0.25 * min(1.0, interface_density) + 0.20 * min(1.0, contact_density / 3.0) + 0.15 * min(1.0, len(hotspot_contacts) / 2.0) + 0.10 * chemistry_balance
    score -= 0.25 * min(1.0, clash_density)
    score -= 0.15 * min(1.0, local_breaks)
    score = max(0.0, min(1.0, score))
    gate_failures: List[str] = []
    if local_breaks:
        gate_failures.append("local_chain_break")
    if clash_density > 0.15:
        gate_failures.append("local_clash_risk")
    if interface_density < 0.25:
        gate_failures.append("insufficient_interface_coverage")
    gate_pass = int(not gate_failures)
    quality_rank = (
        gate_pass,
        interface_density,
        contact_density,
        float(len(hotspot_contacts)),
        float(hbond_like + salt_like),
        chemistry_balance,
        -clash_density,
        -float(local_breaks),
    )
    reasons: List[str] = []
    if interface_density >= 0.5:
        reasons.append("dense_target_interface")
    if len(hotspot_contacts) > 0:
        reasons.append("contacts_target_hotspot")
    if hbond_like + salt_like >= 2:
        reasons.append("specific_polar_or_salt_contacts")
    if hydrophobic >= 2 and 0.25 <= hydrophobic_fraction <= 0.75:
        reasons.append("balanced_hydrophobic_packing")
    if clash_density > 0.15:
        reasons.append("local_clash_risk")
    if local_breaks:
        reasons.append("local_chain_break")
    if interface_density < 0.25:
        reasons.append("weak_local_interface")
    if hydrophobic_fraction > 0.85:
        reasons.append("over_hydrophobic_patch")
    if polar_fraction < 0.10 and hbond_like + salt_like == 0:
        reasons.append("few_specific_polar_contacts")
    if gate_pass and (interface_density >= 0.5 or len(hotspot_contacts) > 0):
        label = "high"
        action = "Preserve or exploit this fragment/conditioning pattern; keep nearby hotspot/interface constraints."
    elif not gate_pass:
        label = "low"
        action = "Avoid direct exploitation; repair by changing local length, hotspot weighting, clash filtering, or sequence/secondary-structure constraints."
    else:
        label = "medium"
        action = "Keep as neutral evidence; test with modest perturbations before exploitation."
    start = _residue_number_from_id(residue_ids[0])
    end = _residue_number_from_id(residue_ids[-1])
    sequence = "".join(AA3_TO_1.get(r, "X") for r in resnames)
    ca_coordinates = [[round(a.x, 3), round(a.y, 3), round(a.z, 3)] for a in ca_atoms]
    return BinderFragmentQuality(
        fragment_id=f"{residue_ids[0]}-{residue_ids[-1]}",
        start_residue=start,
        end_residue=end,
        residue_ids=list(residue_ids),
        residue_count=len(residue_ids),
        interface_contact_count=len(frag_contacts),
        interface_residue_count=len(interface_res),
        hotspot_contact_count=len(hotspot_contacts),
        clash_count=len(frag_clashes),
        hbond_like_count=hbond_like,
        salt_bridge_like_count=salt_like,
        hydrophobic_contact_count=hydrophobic,
        hydrophobic_fraction=round(hydrophobic_fraction, 3),
        polar_fraction=round(polar_fraction, 3),
        local_chain_break_count=local_breaks,
        quality_score=round(score, 3),
        quality_label=label,
        quality_rank=[round(value, 6) for value in quality_rank],
        gate_failures=gate_failures,
        reasons=reasons,
        suggested_action=action,
        sequence=sequence,
        ca_coordinates=ca_coordinates,
    )


def _residue_number_from_id(residue_id: str) -> int:
    try:
        token = residue_id.split(":", 1)[1]
        digits = "".join(ch for ch in token if ch.isdigit() or ch == "-")
        return int(digits)
    except Exception:
        return 0

def _find_xyz(parts: Sequence[str]) -> Optional[Tuple[float, float, float]]:
    floats = []
    for i, part in enumerate(parts):
        try:
            floats.append((i, float(part)))
        except ValueError:
            pass
    for (i1, x), (i2, y), (i3, z) in zip(floats, floats[1:], floats[2:]):
        if i2 == i1 + 1 and i3 == i2 + 1 and i1 >= 8:
            return (x, y, z)
    return (floats[-3][1], floats[-2][1], floats[-1][1]) if len(floats) >= 3 else None


def _first_int(parts: Sequence[str]) -> Optional[int]:
    for p in parts:
        try:
            return int(float(p))
        except ValueError:
            continue
    return None


def _parse_mmcif_atom_parts(parts: Sequence[str]) -> Optional[Tuple[int, str, str, str, int, Tuple[float, float, float], str]]:
    """Parse a whitespace-tokenized mmCIF atom_site row.

    The common atom_site order is:
    group_PDB id type_symbol label_atom_id label_alt_id label_comp_id
    label_asym_id label_entity_id label_seq_id ... Cartn_x Cartn_y Cartn_z
    ... auth_seq_id auth_asym_id ...

    Older fallback parsing accidentally used label_entity_id as the residue
    number, collapsing every residue in a chain to residue 1.
    """
    if len(parts) >= 19 and parts[5].upper() in AA3:
        xyz = _find_xyz(parts)
        if xyz is None:
            return None
        resseq = _first_int([parts[8], parts[16], parts[7]])
        if resseq is None:
            return None
        return int(parts[1]), parts[3], parts[5].upper(), parts[6], resseq, xyz, parts[2]

    resname = parts[3].upper()
    if resname not in AA3:
        return None
    xyz = _find_xyz(parts)
    if xyz is None:
        return None
    resseq = _first_int(parts[5:8])
    if resseq is None:
        return None
    return int(parts[1]), parts[2], resname, parts[4], resseq, xyz, parts[2]


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _residue_groups(atoms: Iterable[AtomRecord]) -> Dict[str, List[AtomRecord]]:
    groups: Dict[str, List[AtomRecord]] = {}
    for atom in atoms:
        groups.setdefault(atom.residue_id, []).append(atom)
    return groups


def _resname_of_residue(groups: Dict[str, List[AtomRecord]], residue_id: str) -> str:
    atoms = groups.get(residue_id) or []
    return atoms[0].resname if atoms else "UNK"


def _residue_bounds(atoms_by_residue: Dict[str, List[AtomRecord]]) -> Dict[str, Tuple[Tuple[float, float, float], float]]:
    """Per-residue (centroid, radius) of its heavy atoms for spatial pre-screening."""
    bounds: Dict[str, Tuple[Tuple[float, float, float], float]] = {}
    for residue_id, atoms in atoms_by_residue.items():
        if not atoms:
            continue
        cx = _fmean(a.x for a in atoms)
        cy = _fmean(a.y for a in atoms)
        cz = _fmean(a.z for a in atoms)
        centroid = (cx, cy, cz)
        radius = max(_dist(a.coord, centroid) for a in atoms)
        bounds[residue_id] = (centroid, radius)
    return bounds


def _contacts(binder_atoms: Sequence[AtomRecord], target_atoms: Sequence[AtomRecord], cutoff: float) -> List[ResidueContact]:
    """Residue-pair minimum heavy-atom distances within ``cutoff``.

    Correct binder/target chain resolution means both sides are now non-empty
    every round, so the naive all-atom double loop became the analysis hot path.
    We pre-screen residue pairs by (centroid, radius): a residue pair can only
    contain an atom pair within ``cutoff`` if their centroids are within
    ``cutoff + radius_b + radius_t``. This is exact (no contacts are missed) and
    skips the vast majority of far-apart atom comparisons.
    """
    binder_by_res: Dict[str, List[AtomRecord]] = {}
    for atom in binder_atoms:
        if atom.name.upper().startswith("H"):
            continue
        binder_by_res.setdefault(atom.residue_id, []).append(atom)
    target_by_res: Dict[str, List[AtomRecord]] = {}
    for atom in target_atoms:
        if atom.name.upper().startswith("H"):
            continue
        target_by_res.setdefault(atom.residue_id, []).append(atom)

    binder_bounds = _residue_bounds(binder_by_res)
    target_bounds = _residue_bounds(target_by_res)

    best: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for brid, batoms in binder_by_res.items():
        bcentroid, bradius = binder_bounds[brid]
        for trid, tatoms in target_by_res.items():
            tcentroid, tradius = target_bounds[trid]
            if _dist(bcentroid, tcentroid) > cutoff + bradius + tradius:
                continue
            best_d = math.inf
            best_pair: Optional[Tuple[AtomRecord, AtomRecord]] = None
            for b in batoms:
                for t in tatoms:
                    d = _dist(b.coord, t.coord)
                    if d < best_d:
                        best_d = d
                        best_pair = (b, t)
            if best_d <= cutoff and best_pair is not None:
                best[(brid, trid)] = (best_d, _contact_type(best_pair[0], best_pair[1]))
    return [ResidueContact(b, t, d, c) for (b, t), (d, c) in sorted(best.items(), key=lambda item: item[1][0])]


def _heavy_atom_clashes(binder_atoms: Sequence[AtomRecord], target_atoms: Sequence[AtomRecord], cutoff: float) -> Tuple[int, int]:
    """Count cross-chain clashes using an exact uniform-grid neighbor search."""
    binder = [atom for atom in binder_atoms if not atom.name.upper().startswith("H") and str(atom.element or "").upper() != "H"]
    target = [atom for atom in target_atoms if not atom.name.upper().startswith("H") and str(atom.element or "").upper() != "H"]
    pair_count = len(binder) * len(target)
    if not binder or not target or cutoff <= 0:
        return 0, pair_count
    cell = float(cutoff)
    grid: Dict[Tuple[int, int, int], List[AtomRecord]] = {}
    for atom in target:
        key = tuple(int(math.floor(value / cell)) for value in atom.coord)
        grid.setdefault(key, []).append(atom)
    count = 0
    offsets = (-1, 0, 1)
    for left in binder:
        base = tuple(int(math.floor(value / cell)) for value in left.coord)
        for dx in offsets:
            for dy in offsets:
                for dz in offsets:
                    for right in grid.get((base[0] + dx, base[1] + dy, base[2] + dz), ()):
                        if _dist(left.coord, right.coord) <= cutoff:
                            count += 1
    return count, pair_count


def _residue_coverage(residues: Sequence[str], contacts: Sequence[ResidueContact]) -> Dict[str, float]:
    requested = [str(value) for value in residues or [] if str(value).strip()]
    covered = [value for value in requested if any(_contact_matches_hotspot(contact.target_residue, value) for contact in contacts)]
    return {
        "requested_count": float(len(requested)),
        "covered_count": float(len(covered)),
        "coverage_fraction": float(len(covered)) / max(1, len(requested)),
        "covered_residues": covered,
        "missed_residues": [value for value in requested if value not in covered],
    }


def _contact_type(a: AtomRecord, b: AtomRecord) -> str:
    if (a.resname in POSITIVE and b.resname in NEGATIVE) or (a.resname in NEGATIVE and b.resname in POSITIVE):
        return "salt_bridge"
    if a.resname in HYDROPHOBIC and b.resname in HYDROPHOBIC:
        return "hydrophobic"
    if a.resname in POLAR or b.resname in POLAR or a.resname in POSITIVE | NEGATIVE or b.resname in POSITIVE | NEGATIVE:
        return "polar"
    return "van_der_waals"


def _radius_of_gyration(coords: Sequence[Tuple[float, float, float]]) -> float:
    if not coords:
        return 0.0
    cx = _fmean(x for x, _, _ in coords); cy = _fmean(y for _, y, _ in coords); cz = _fmean(z for _, _, z in coords)
    return math.sqrt(_fmean((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 for x, y, z in coords))


def _end_to_end(ca_atoms: Sequence[AtomRecord]) -> float:
    if len(ca_atoms) < 2:
        return 0.0
    atoms = sorted(ca_atoms, key=lambda a: a.resseq)
    return _dist(atoms[0].coord, atoms[-1].coord)


def _contact_order(contacts: Sequence[ResidueContact]) -> float:
    vals = []
    for c in contacts:
        try:
            vals.append(abs(int(c.binder_residue.split(":", 1)[1]) - int(c.target_residue.split(":", 1)[1])))
        except Exception:
            pass
    return _fmean(vals) if vals else 0.0


def _chain_breaks(ca_atoms: Sequence[AtomRecord]) -> int:
    atoms = sorted(ca_atoms, key=lambda a: a.resseq)
    return sum(1 for left, right in zip(atoms, atoms[1:]) if right.resseq - left.resseq > 1 or _dist(left.coord, right.coord) > 4.5)


def _normalize_hotspot(hotspot: str) -> str:
    return hotspot if ":" in hotspot else hotspot


def _hotspot_residue_number(hotspot: str) -> Optional[int]:
    """Extract the integer residue number from a hotspot token like 'E:153' or '153'."""
    token = str(hotspot)
    if ":" in token:
        token = token.split(":", 1)[1]
    elif "/" in token:
        token = token.split("/", 1)[1]
    digits = "".join(ch for ch in token if ch.isdigit() or ch == "-")
    try:
        return int(digits)
    except ValueError:
        return None


def _contact_matches_hotspot(target_residue: str, hotspot: str) -> bool:
    """Match a contact's target residue to a hotspot, tolerant of chain relabeling.

    The configured hotspot chain (e.g. 'E') usually differs from the BoltzGen
    output target chain (e.g. 'B'), so an exact 'chain:resid' comparison fails.
    We match on residue number, and additionally honour an exact match when the
    chain letters happen to agree.
    """
    if target_residue == _normalize_hotspot(hotspot):
        return True
    hotspot_number = _hotspot_residue_number(hotspot)
    if hotspot_number is None:
        return False
    return _residue_number_from_id(target_residue) == hotspot_number


def _reliability(interface_count: int, clash_density: float, hotspot_contacts: Dict[str, int], chain_breaks: int, rg: float, end_to_end: float, hydrophobic_fraction: float) -> Tuple[float, List[str]]:
    score = 1.0; tags: List[str] = []
    if interface_count < 6:
        score -= 0.25; tags.append("weak_or_tiny_interface")
    if clash_density > 0.15:
        score -= min(0.35, clash_density); tags.append("interface_clash_risk")
    if hotspot_contacts and any(v == 0 for v in hotspot_contacts.values()):
        score -= 0.25; tags.append("hotspot_not_covered")
    if chain_breaks:
        score -= 0.25; tags.append("binder_chain_break")
    if rg < 3.0 or (end_to_end and rg / max(end_to_end, 1.0) > 1.5):
        score -= 0.10; tags.append("binder_geometry_suspicious")
    if hydrophobic_fraction > 0.85:
        score -= 0.10; tags.append("over_hydrophobic_interface")
    if not tags:
        tags.append("structure_features_pass")
    return max(0.0, min(1.0, score)), tags