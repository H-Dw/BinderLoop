"""Convert harness residue IDs onto the scheme AtomWorks/RFD3 will look up.

RFD3 selects residues as ``{chain_id}{res_id}`` on the loaded AtomArray.
AtomWorks sets those fields from mmCIF ``label_asym_id`` / ``label_seq_id``.
PDB files have only author/PDB numbering, so ``res_id`` is the auth number.

Harness configs often mix BoltzGen **label** IDs (PD-L1 ``A:40`` / range
``1-116``) with literature or official RFD3 **auth** IDs (``A56`` / ``17-132``).
This module:

1. Builds the label ↔ auth map from the target structure.
2. Detects the source scheme of contig/hotspot tokens.
3. Rewrites the target range and hotspot keys into RFD3's native scheme.
4. Optionally rewrites a CIF so ``label_*`` columns match a chosen scheme.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


SCHEME_LABEL = "label"
SCHEME_AUTH = "auth"
SCHEME_NATIVE = "native"
SCHEME_AUTO = "auto"
VALID_SCHEMES = (SCHEME_LABEL, SCHEME_AUTH)
AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

_RANGE_TOKEN = re.compile(r"^(\d+)\s*(?:\.\.|-|–|—|:)\s*(\d+)$")
_HOTSPOT_TOKEN = re.compile(
    r"^(?:(?P<chain>[A-Za-z0-9]+)\s*[:/]\s*)?(?P<number>-?\d+)(?P<icode>[A-Za-z]?)$"
)
_RFD3_TOKEN = re.compile(r"^(?P<chain>[A-Za-z]+)(?P<number>-?\d+)(?P<icode>[A-Za-z]?)$")
_CONTIG_TARGET = re.compile(r"^(?P<chain>[A-Za-z]+)(?P<start>\d+)\s*-\s*(?P<end>\d+)\s*$")


@dataclass(frozen=True)
class ResidueIdRecord:
    label_chain: str
    auth_chain: str
    label_seq_id: int
    auth_seq_id: int
    ins_code: str = ""
    resname: str = ""
    atom_names: Tuple[str, ...] = ()

    def chain_for(self, scheme: str) -> str:
        return self.label_chain if scheme == SCHEME_LABEL else self.auth_chain

    def number_for(self, scheme: str) -> int:
        return self.label_seq_id if scheme == SCHEME_LABEL else self.auth_seq_id

    def token_for(self, scheme: str) -> str:
        icode = self.ins_code or ""
        return f"{self.chain_for(scheme)}{self.number_for(scheme)}{icode}"


@dataclass
class StructureIdMap:
    path: str
    format: str
    native_scheme: str
    residues: List[ResidueIdRecord] = field(default_factory=list)

    def residues_for_chain(self, chain: str, scheme: str) -> List[ResidueIdRecord]:
        chain = str(chain or "").strip()
        out = [item for item in self.residues if item.chain_for(scheme) == chain]
        if out:
            return out
        return [
            item for item in self.residues
            if item.label_chain == chain or item.auth_chain == chain
        ]

    def lookup(self, chain: str, number: int, scheme: str, icode: str = "") -> ResidueIdRecord:
        icode = str(icode or "").strip()
        matches = [
            item for item in self.residues_for_chain(chain, scheme)
            if item.number_for(scheme) == int(number) and (item.ins_code or "") == icode
        ]
        if not matches:
            available = sorted({item.number_for(scheme) for item in self.residues_for_chain(chain, scheme)})
            span = f"{available[0]}-{available[-1]}" if available else "none"
            raise KeyError(
                f"residue {chain}{number}{icode} not in {scheme} numbering for {self.path} (available {span})"
            )
        return matches[0]

    def contains(self, chain: str, number: int, scheme: str, icode: str = "") -> bool:
        try:
            self.lookup(chain, number, scheme, icode)
            return True
        except KeyError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format,
            "native_scheme": self.native_scheme,
            "residue_count": len(self.residues),
            "residues": [asdict(item) for item in self.residues],
        }


@dataclass
class AdaptedRFD3Identifiers:
    structure_path: str
    native_scheme: str
    source_scheme: str
    target_scheme: str
    chain_id: str
    res_index: Optional[str]
    hotspots: List[str]
    select_hotspots: Dict[str, str]
    contig_target: Optional[str]
    conversions: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_scheme(value: Any, *, allow_auto: bool = False, allow_native: bool = False) -> Optional[str]:
    if value in (None, ""):
        return SCHEME_AUTO if allow_auto else None
    token = str(value).strip().lower()
    if token in {"label", "label_seq_id", "label_seq"}:
        return SCHEME_LABEL
    if token in {"auth", "author", "auth_seq_id", "pdb"}:
        return SCHEME_AUTH
    if allow_native and token in {SCHEME_NATIVE, "rfd3", "atomworks"}:
        return SCHEME_NATIVE
    if allow_auto and token in {SCHEME_AUTO, "detect"}:
        return SCHEME_AUTO
    raise ValueError(f"unsupported residue ID scheme {value!r}")


def parse_residue_token(token: str, *, default_chain: str) -> Tuple[str, int, str]:
    text = str(token or "").strip()
    if not text:
        raise ValueError("empty residue token")
    match = _HOTSPOT_TOKEN.match(text)
    if match:
        chain = match.group("chain") or default_chain
        return str(chain), int(match.group("number")), str(match.group("icode") or "")
    match = _RFD3_TOKEN.match(text)
    if match:
        return match.group("chain"), int(match.group("number")), str(match.group("icode") or "")
    raise ValueError(f"invalid residue token {token!r}")


def parse_res_index(value: Any) -> Optional[Tuple[int, int]]:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("..", "-").replace(":", "-")
    match = _RANGE_TOKEN.match(text)
    if match:
        lo, hi = int(match.group(1)), int(match.group(2))
        return (min(lo, hi), max(lo, hi))
    if text.isdigit():
        number = int(text)
        return (number, number)
    return None


def rfd3_native_scheme(path: Union[str, Path], *, file_format: Optional[str] = None) -> str:
    """Return the numbering AtomWorks will assign to ``res_id`` for this file."""
    kind = file_format or _detect_format(path)
    return SCHEME_LABEL if kind == "cif" else SCHEME_AUTH


def parse_structure_id_map(path: Union[str, Path]) -> StructureIdMap:
    path = Path(path)
    kind = _detect_format(path)
    if kind == "cif":
        residues = _parse_cif_residues(path)
        native = SCHEME_LABEL
    else:
        residues = _parse_pdb_residues(path)
        native = SCHEME_AUTH
    if not residues:
        raise ValueError(f"no polymer residues parsed from {path}")
    return StructureIdMap(path=str(path), format=kind, native_scheme=native, residues=residues)


def detect_source_scheme(
    id_map: StructureIdMap,
    *,
    chain_id: str,
    numbers: Sequence[int],
    atom_hints: Optional[Mapping[Tuple[str, int], Sequence[str]]] = None,
) -> str:
    """Infer whether supplied residue numbers are label or auth.

    Numbers that exist in only one scheme vote for that scheme. Overlapping
    numbers (common on cropped CIFs) are disambiguated with atom-name hints
    from ``select_hotspots``.
    """
    unique_label = unique_auth = 0
    both = 0
    missing = 0
    for number in numbers:
        in_label = id_map.contains(chain_id, int(number), SCHEME_LABEL)
        in_auth = id_map.contains(chain_id, int(number), SCHEME_AUTH)
        if in_label and in_auth:
            both += 1
        elif in_label:
            unique_label += 1
        elif in_auth:
            unique_auth += 1
        else:
            missing += 1
    if unique_label and not unique_auth:
        return SCHEME_LABEL
    if unique_auth and not unique_label:
        return SCHEME_AUTH
    if unique_label and unique_auth:
        raise ValueError(
            f"residue numbers mix label-only and auth-only IDs on chain {chain_id} of {id_map.path}"
        )
    hint_label, hint_auth = _atom_hint_votes(id_map, chain_id, atom_hints or {})
    if hint_label and not hint_auth:
        return SCHEME_LABEL
    if hint_auth and not hint_label:
        return SCHEME_AUTH
    if both:
        return id_map.native_scheme
    if missing and not (unique_label or unique_auth or both):
        raise ValueError(
            f"residue numbers {list(numbers)} not found in label or auth numbering for chain {chain_id} of {id_map.path}"
        )
    return id_map.native_scheme


def convert_residue(
    id_map: StructureIdMap,
    token: str,
    *,
    default_chain: str,
    source_scheme: str,
    target_scheme: str,
) -> ResidueIdRecord:
    chain, number, icode = parse_residue_token(token, default_chain=default_chain)
    return id_map.lookup(chain, number, source_scheme, icode)


def convert_res_index(
    id_map: StructureIdMap,
    value: Any,
    *,
    chain_id: str,
    source_scheme: str,
    target_scheme: str,
) -> Tuple[str, str]:
    parsed = parse_res_index(value)
    if parsed is None:
        raise ValueError(f"cannot parse residue range {value!r}")
    lo, hi = parsed
    mapped: List[ResidueIdRecord] = []
    for record in id_map.residues_for_chain(chain_id, source_scheme):
        number = record.number_for(source_scheme)
        if lo <= number <= hi:
            mapped.append(record)
    if not mapped:
        raise ValueError(
            f"range {lo}-{hi} on chain {chain_id} is empty in {source_scheme} numbering of {id_map.path}"
        )
    target_chain = mapped[0].chain_for(target_scheme)
    numbers = [item.number_for(target_scheme) for item in mapped]
    return target_chain, f"{min(numbers)}-{max(numbers)}"


def convert_contig_target(
    contig: str,
    id_map: StructureIdMap,
    *,
    default_chain: str,
    source_scheme: str,
    target_scheme: str,
) -> str:
    text = str(contig or "").strip()
    if not text:
        return text
    if "/0," in text:
        prefix, _, tail = text.partition("/0,")
        converted = _convert_contig_span(tail.strip(), id_map, default_chain, source_scheme, target_scheme)
        return f"{prefix}/0,{converted}"
    return _convert_contig_span(text, id_map, default_chain, source_scheme, target_scheme)


def adapt_rfd3_identifiers(
    structure_path: Union[str, Path],
    *,
    chain_id: str,
    res_index: Any = None,
    target_include: Any = None,
    contig: Any = None,
    hotspots: Optional[Sequence[str]] = None,
    select_hotspots: Optional[Mapping[str, Any]] = None,
    source_scheme: Any = SCHEME_AUTO,
    target_scheme: Any = SCHEME_NATIVE,
    adapt_structure: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
) -> AdaptedRFD3Identifiers:
    """Adapt structure residue IDs and the matching hotspot keys for RFD3."""
    path = Path(structure_path)
    notes: List[str] = []
    conversions: List[Dict[str, Any]] = []
    if not path.exists():
        notes.append(f"structure {path} is missing; residue IDs were left unchanged")
        return AdaptedRFD3Identifiers(
            structure_path=str(path),
            native_scheme=rfd3_native_scheme(path),
            source_scheme=str(source_scheme or SCHEME_AUTO),
            target_scheme=str(target_scheme or SCHEME_NATIVE),
            chain_id=str(chain_id),
            res_index=str(res_index) if res_index not in (None, "") else None,
            hotspots=list(hotspots or []),
            select_hotspots={str(key): str(value) for key, value in dict(select_hotspots or {}).items()},
            contig_target=None,
            notes=notes,
        )

    id_map = parse_structure_id_map(path)
    native = id_map.native_scheme
    requested_source = normalize_scheme(source_scheme, allow_auto=True) or SCHEME_AUTO
    requested_target = normalize_scheme(target_scheme, allow_native=True) or SCHEME_NATIVE
    resolved_target = native if requested_target == SCHEME_NATIVE else requested_target

    hotspot_tokens = [str(item) for item in (hotspots or []) if str(item).strip()]
    select_map = {str(key): value for key, value in dict(select_hotspots or {}).items() if str(key).strip()}
    numbers: List[int] = []
    atom_hints: Dict[Tuple[str, int], List[str]] = {}
    for token in list(hotspot_tokens) + list(select_map):
        try:
            chain, number, _icode = parse_residue_token(token, default_chain=chain_id)
        except ValueError:
            continue
        numbers.append(number)
        atoms = _atom_names(select_map.get(token) if token in select_map else None)
        if atoms:
            atom_hints[(chain, number)] = atoms
    parsed_range = parse_res_index(res_index) or parse_res_index(_res_index_from_include(target_include, chain_id))
    if parsed_range:
        numbers.extend(parsed_range)
    contig_span = _contig_span_numbers(contig)
    if contig_span:
        numbers.extend(contig_span)

    if requested_source == SCHEME_AUTO:
        resolved_source = detect_source_scheme(
            id_map, chain_id=chain_id, numbers=numbers, atom_hints=atom_hints
        ) if numbers else native
        notes.append(f"detected source residue scheme {resolved_source}")
    else:
        resolved_source = requested_source

    out_chain = chain_id
    out_index = str(res_index).strip() if res_index not in (None, "") else None
    if parsed_range is not None:
        out_chain, out_index = convert_res_index(
            id_map, f"{parsed_range[0]}-{parsed_range[1]}",
            chain_id=chain_id, source_scheme=resolved_source, target_scheme=resolved_target,
        )
        conversions.append({
            "kind": "res_index",
            "from": f"{chain_id}{parsed_range[0]}-{parsed_range[1]}",
            "to": f"{out_chain}{out_index}",
            "source_scheme": resolved_source,
            "target_scheme": resolved_target,
        })

    converted_hotspots: List[str] = []
    for token in hotspot_tokens:
        record = convert_residue(
            id_map, token, default_chain=chain_id,
            source_scheme=resolved_source, target_scheme=resolved_target,
        )
        converted = record.token_for(resolved_target)
        converted_hotspots.append(converted)
        conversions.append({
            "kind": "hotspot",
            "from": token,
            "to": converted,
            "resname": record.resname,
            "source_scheme": resolved_source,
            "target_scheme": resolved_target,
        })

    converted_select: Dict[str, str] = {}
    for key, value in select_map.items():
        record = convert_residue(
            id_map, key, default_chain=chain_id,
            source_scheme=resolved_source, target_scheme=resolved_target,
        )
        converted = record.token_for(resolved_target)
        converted_select[converted] = str(value)
        conversions.append({
            "kind": "select_hotspots",
            "from": key,
            "to": converted,
            "atoms": str(value),
            "resname": record.resname,
            "source_scheme": resolved_source,
            "target_scheme": resolved_target,
        })

    contig_target = f"{out_chain}{out_index}" if out_index else None
    if contig not in (None, "") and "/0," in str(contig):
        rewritten = convert_contig_target(
            str(contig), id_map, default_chain=chain_id,
            source_scheme=resolved_source, target_scheme=resolved_target,
        )
        _prefix, _, tail = rewritten.partition("/0,")
        contig_target = tail.strip() or contig_target

    adapted_path = str(path)
    if adapt_structure and resolved_target != native and id_map.format == "cif":
        dest_dir = Path(output_dir) if output_dir is not None else path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{path.stem}.rfd3_{resolved_target}.cif"
        write_adapted_cif(path, dest, target_scheme=resolved_target)
        adapted_path = str(dest)
        notes.append(f"wrote adapted structure {dest} with {resolved_target} IDs copied into label columns")
    elif adapt_structure and resolved_target != native:
        notes.append("structure rewrite is only implemented for mmCIF; PDB numbering was left unchanged")

    return AdaptedRFD3Identifiers(
        structure_path=adapted_path,
        native_scheme=native,
        source_scheme=resolved_source,
        target_scheme=resolved_target,
        chain_id=out_chain,
        res_index=out_index,
        hotspots=converted_hotspots,
        select_hotspots=converted_select,
        contig_target=contig_target,
        conversions=conversions,
        notes=notes,
    )


def write_conversion_report(adapted: AdaptedRFD3Identifiers, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adapted.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def write_adapted_cif(source: Union[str, Path], dest: Union[str, Path], *, target_scheme: str) -> Path:
    """Copy ``label_seq_id`` / ``label_asym_id`` from auth columns when targeting auth."""
    if target_scheme != SCHEME_AUTH:
        raise ValueError("CIF adaptation currently only copies auth IDs into label columns")
    source = Path(source)
    dest = Path(dest)
    text = source.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start, columns, data_end = _find_atom_site_loop(lines)
    col_index = {name: idx for idx, name in enumerate(columns)}
    required = ("label_seq_id", "label_asym_id", "auth_seq_id", "auth_asym_id")
    missing = [name for name in required if name not in col_index]
    if missing:
        raise ValueError(f"{source} atom_site loop is missing {missing}")
    rewritten = list(lines)
    for row_idx in range(start + 1 + len(columns), data_end):
        raw = lines[row_idx]
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        tokens = _cif_split(raw)
        if len(tokens) < len(columns):
            continue
        if tokens[0] not in {"ATOM", "HETATM"}:
            continue
        tokens[col_index["label_seq_id"]] = tokens[col_index["auth_seq_id"]]
        tokens[col_index["label_asym_id"]] = tokens[col_index["auth_asym_id"]]
        rewritten[row_idx] = " ".join(tokens)
    dest.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return dest


def _detect_format(path: Union[str, Path]) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return "cif"
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    try:
        head = Path(path).read_text(encoding="utf-8", errors="ignore")[:4096]
    except OSError:
        return "pdb"
    for line in head.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("data_") or stripped.startswith("_atom_site.") or stripped.startswith("loop_"):
            return "cif"
        if stripped.startswith(("ATOM", "HETATM", "HEADER", "CRYST1")):
            return "pdb"
        break
    return "pdb"


def _parse_cif_residues(path: Path) -> List[ResidueIdRecord]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start, columns, data_end = _find_atom_site_loop(lines)
    col = {name: idx for idx, name in enumerate(columns)}
    grouped: Dict[Tuple[str, int, str, str, int], Dict[str, Any]] = {}
    for row_idx in range(start + 1 + len(columns), data_end):
        tokens = _cif_split(lines[row_idx])
        if len(tokens) < len(columns) or tokens[0] not in {"ATOM", "HETATM"}:
            continue
        resname = _cif_value(tokens, col, "label_comp_id").upper()
        if resname not in AA3:
            continue
        label_seq = _optional_int(_cif_value(tokens, col, "label_seq_id"))
        auth_seq = _optional_int(_cif_value(tokens, col, "auth_seq_id"))
        if label_seq is None or auth_seq is None:
            continue
        label_chain = _cif_value(tokens, col, "label_asym_id") or "A"
        auth_chain = _cif_value(tokens, col, "auth_asym_id") or label_chain
        ins = _cif_value(tokens, col, "pdbx_PDB_ins_code")
        if ins in {".", "?", ""}:
            ins = ""
        atom_name = _cif_value(tokens, col, "label_atom_id")
        key = (label_chain, label_seq, ins, auth_chain, auth_seq)
        item = grouped.setdefault(key, {"resname": resname, "atoms": []})
        if atom_name:
            item["atoms"].append(atom_name)
    residues = []
    for (label_chain, label_seq, ins, auth_chain, auth_seq), payload in grouped.items():
        residues.append(
            ResidueIdRecord(
                label_chain=label_chain,
                auth_chain=auth_chain,
                label_seq_id=label_seq,
                auth_seq_id=auth_seq,
                ins_code=ins,
                resname=payload["resname"],
                atom_names=tuple(dict.fromkeys(payload["atoms"])),
            )
        )
    residues.sort(key=lambda item: (item.label_chain, item.label_seq_id, item.ins_code))
    return residues


def _parse_pdb_residues(path: Path) -> List[ResidueIdRecord]:
    grouped: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 26:
            continue
        resname = line[17:20].strip().upper()
        if resname not in AA3:
            continue
        try:
            number = int(line[22:26])
        except ValueError:
            continue
        chain = (line[21].strip() or "A")
        icode = line[26].strip() if len(line) > 26 else ""
        atom_name = line[12:16].strip()
        key = (chain, number, icode)
        item = grouped.setdefault(key, {"resname": resname, "atoms": []})
        if atom_name:
            item["atoms"].append(atom_name)
    residues = []
    for (chain, number, icode), payload in grouped.items():
        residues.append(
            ResidueIdRecord(
                label_chain=chain,
                auth_chain=chain,
                label_seq_id=number,
                auth_seq_id=number,
                ins_code=icode,
                resname=payload["resname"],
                atom_names=tuple(dict.fromkeys(payload["atoms"])),
            )
        )
    residues.sort(key=lambda item: (item.auth_chain, item.auth_seq_id, item.ins_code))
    return residues


def _find_atom_site_loop(lines: Sequence[str]) -> Tuple[int, List[str], int]:
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "loop_":
            nxt = idx + 1
            if nxt < len(lines) and lines[nxt].strip().startswith("_atom_site."):
                start = idx
                break
    if start is None:
        raise ValueError("mmCIF is missing an atom_site loop")
    columns: List[str] = []
    cursor = start + 1
    while cursor < len(lines):
        stripped = lines[cursor].strip()
        if stripped.startswith("_atom_site."):
            columns.append(stripped.split(".", 1)[1].split()[0])
            cursor += 1
            continue
        break
    data_end = cursor
    while data_end < len(lines):
        stripped = lines[data_end].strip()
        if stripped.startswith("loop_") or (stripped.startswith("_") and not stripped.startswith("_atom_site.")):
            break
        if stripped.startswith("data_"):
            break
        data_end += 1
    if not columns:
        raise ValueError("mmCIF atom_site loop has no columns")
    return start, columns, data_end


def _cif_split(line: str) -> List[str]:
    tokens: List[str] = []
    buf: List[str] = []
    quote = None
    for char in line.strip():
        if quote:
            if char == quote:
                tokens.append("".join(buf))
                buf = []
                quote = None
            else:
                buf.append(char)
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
            continue
        buf.append(char)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _cif_value(tokens: Sequence[str], col: Mapping[str, int], name: str) -> str:
    idx = col.get(name)
    if idx is None or idx >= len(tokens):
        return ""
    value = tokens[idx]
    return "" if value in {".", "?"} else value


def _optional_int(value: str) -> Optional[int]:
    if value in {"", ".", "?"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _atom_names(value: Any) -> List[str]:
    if value in (None, "", "ALL"):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip() and str(item).strip().upper() != "ALL"]
    return [part.strip() for part in str(value).split(",") if part.strip() and part.strip().upper() != "ALL"]


def _atom_hint_votes(
    id_map: StructureIdMap,
    chain_id: str,
    atom_hints: Mapping[Tuple[str, int], Sequence[str]],
) -> Tuple[int, int]:
    label_hits = auth_hits = 0
    for (chain, number), atoms in atom_hints.items():
        needed = {name.upper() for name in atoms}
        if not needed:
            continue
        cid = chain or chain_id
        for scheme, bucket in ((SCHEME_LABEL, "label"), (SCHEME_AUTH, "auth")):
            if not id_map.contains(cid, int(number), scheme):
                continue
            record = id_map.lookup(cid, int(number), scheme)
            have = {name.upper() for name in record.atom_names}
            if needed.issubset(have):
                if bucket == "label":
                    label_hits += 1
                else:
                    auth_hits += 1
    return label_hits, auth_hits


def _res_index_from_include(include: Any, chain_id: str) -> Optional[str]:
    if not include:
        return None
    items = include if isinstance(include, list) else [include]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        chain = item.get("chain") if isinstance(item.get("chain"), Mapping) else item
        if not isinstance(chain, Mapping):
            continue
        if str(chain.get("id") or chain_id) != chain_id:
            continue
        value = chain.get("res_index") or chain.get("residue_index")
        if value not in (None, ""):
            return str(value)
    return None


def _contig_span_numbers(contig: Any) -> Optional[Tuple[int, int]]:
    if contig in (None, ""):
        return None
    text = str(contig)
    tail = text.split("/0,", 1)[1] if "/0," in text else text
    match = _CONTIG_TARGET.match(tail.strip())
    if not match:
        return None
    lo, hi = int(match.group("start")), int(match.group("end"))
    return (min(lo, hi), max(lo, hi))


def _convert_contig_span(
    span: str,
    id_map: StructureIdMap,
    default_chain: str,
    source_scheme: str,
    target_scheme: str,
) -> str:
    match = _CONTIG_TARGET.match(span.strip())
    if not match:
        return span
    chain = match.group("chain") or default_chain
    _out_chain, converted = convert_res_index(
        id_map, f"{match.group('start')}-{match.group('end')}",
        chain_id=chain, source_scheme=source_scheme, target_scheme=target_scheme,
    )
    return f"{_out_chain}{converted}"
