"""Structure preparation primitives for a blinded 3-D hotspot benchmark.

This module deliberately contains no hotspot-selection logic.  Source identifiers
are retained only in :class:`LocalMapping`; local mmCIF and feature JSON outputs
use opaque chain IDs and contiguous residue numbers.

SASA is calculated with a deterministic Shrake--Rupley implementation.  Defaults
are a 1.4 A solvent probe, 960 Fibonacci-sphere points, and Bondi-style van der
Waals radii (C 1.70, N 1.55, O 1.52, S/P 1.80, Se 1.90 A; 1.70 A fallback).
Relative SASA uses the theoretical Gly-X-Gly maxima from Tien et al., PLoS ONE
8:e80635 (2013), doi:10.1371/journal.pone.0080635, listed in
``MAXIMUM_RESIDUE_ASA`` below.  rSASA is intentionally not clipped at 1.0.

Only the Python standard library and NumPy are required.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)

RESIDUE_ONE_LETTER: Mapping[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})

VDW_RADII_ANGSTROM: Mapping[str, float] = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
    "SE": 1.90,
}

# Theoretical maximum ASA values (A^2) from Tien et al. (2013), Table 1.
MAXIMUM_RESIDUE_ASA: Mapping[str, float] = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}

# Exclusive, coarse residue classes used only as aggregated patch composition.
PATCH_COMPOSITION_CLASSES: Mapping[str, frozenset[str]] = {
    "hydrophobic": frozenset({"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"}),
    "polar": frozenset({"SER", "THR", "ASN", "GLN", "CYS"}),
    "positive": frozenset({"LYS", "ARG", "HIS"}),
    "negative": frozenset({"ASP", "GLU"}),
    "special": frozenset({"GLY", "PRO"}),
}

_MISSING_CIF_VALUES = frozenset({"", ".", "?"})
_FLOAT_WITH_UNCERTAINTY = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\(\d+\)$"
)


def _immutable_coordinate(value: Sequence[float] | np.ndarray) -> np.ndarray:
    coordinate = np.asarray(value, dtype=np.float64)
    if coordinate.shape != (3,):
        raise ValueError(f"coordinate must have shape (3,), got {coordinate.shape}")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError("coordinate must contain only finite values")
    coordinate = coordinate.copy()
    coordinate.flags.writeable = False
    return coordinate


def _normalise_missing(value: str | None) -> str:
    if value is None or value in _MISSING_CIF_VALUES:
        return ""
    return value


def _normalise_seq_id(value: str | None) -> str | None:
    normalised = _normalise_missing(value)
    return normalised or None


@dataclass(frozen=True, slots=True)
class Atom:
    """One selected, model-1, heavy ATOM record with both ID namespaces."""

    element: str
    label_atom_id: str
    auth_atom_id: str
    coordinate: np.ndarray
    occupancy: float
    altloc: str
    auth_asym_id: str
    auth_seq_id: str
    insertion_code: str
    auth_comp_id: str
    label_asym_id: str
    label_seq_id: str | None
    label_comp_id: str
    model_num: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinate", _immutable_coordinate(self.coordinate))
        object.__setattr__(self, "element", self.element.upper())
        object.__setattr__(self, "altloc", _normalise_missing(self.altloc))
        object.__setattr__(self, "insertion_code", _normalise_missing(self.insertion_code))
        if not math.isfinite(self.occupancy) or self.occupancy < 0.0:
            raise ValueError("occupancy must be finite and non-negative")
        if self.model_num != 1:
            raise ValueError("Atom objects in prepared structures must be from model 1")

    @property
    def atom_name(self) -> str:
        return self.label_atom_id or self.auth_atom_id

    @property
    def is_backbone(self) -> bool:
        return self.atom_name.upper() in BACKBONE_ATOM_NAMES


@dataclass(frozen=True, slots=True)
class Residue:
    """A source residue after model, amino-acid, heavy-atom, and altloc filtering."""

    auth_asym_id: str
    auth_seq_id: str
    insertion_code: str
    auth_comp_id: str
    label_asym_id: str
    label_seq_id: str | None
    label_comp_id: str
    atoms: tuple[Atom, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "insertion_code", _normalise_missing(self.insertion_code))
        object.__setattr__(self, "atoms", tuple(self.atoms))
        if not self.atoms:
            raise ValueError("residue must contain at least one heavy atom")

    @property
    def residue_name(self) -> str:
        if self.auth_comp_id.upper() in STANDARD_AMINO_ACIDS:
            return self.auth_comp_id.upper()
        return self.label_comp_id.upper()

    @property
    def auth_key(self) -> tuple[str, str, str]:
        return (self.auth_asym_id, self.auth_seq_id, self.insertion_code)

    @property
    def label_key(self) -> tuple[str, str | None]:
        return (self.label_asym_id, self.label_seq_id)

    @property
    def ca_atom(self) -> Atom | None:
        return next((atom for atom in self.atoms if atom.atom_name.upper() == "CA"), None)


@dataclass(frozen=True, slots=True)
class AuthResidueRange:
    """Inclusive author-numbered crop range.

    Blank insertion-code endpoints mean "all insertion codes at this sequence
    number".  Supplying an endpoint insertion code narrows that boundary.
    """

    chain_id: str
    start: int
    end: int
    start_insertion_code: str = ""
    end_insertion_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "start_insertion_code", _normalise_missing(self.start_insertion_code)
        )
        object.__setattr__(
            self, "end_insertion_code", _normalise_missing(self.end_insertion_code)
        )
        if self.start > self.end:
            raise ValueError("author range start must not exceed end")

    def contains(self, residue: Residue) -> bool:
        if residue.auth_asym_id != self.chain_id:
            return False
        try:
            sequence_number = int(residue.auth_seq_id)
        except ValueError as exc:
            raise ValueError(
                f"author sequence ID {residue.auth_seq_id!r} is not an integer and cannot be ranged"
            ) from exc
        if sequence_number < self.start or sequence_number > self.end:
            return False
        insertion = residue.insertion_code
        if (
            sequence_number == self.start
            and self.start_insertion_code
            and _insertion_sort_key(insertion) < _insertion_sort_key(self.start_insertion_code)
        ):
            return False
        if (
            sequence_number == self.end
            and self.end_insertion_code
            and _insertion_sort_key(insertion) > _insertion_sort_key(self.end_insertion_code)
        ):
            return False
        return True


@dataclass(frozen=True, slots=True)
class LocalResidueMapping:
    local_chain_id: str
    local_seq_id: int
    auth_asym_id: str
    auth_seq_id: str
    insertion_code: str
    auth_comp_id: str
    label_asym_id: str
    label_seq_id: str | None
    label_comp_id: str

    @property
    def local_key(self) -> tuple[str, int]:
        return (self.local_chain_id, self.local_seq_id)

    @property
    def auth_key(self) -> tuple[str, str, str]:
        return (self.auth_asym_id, self.auth_seq_id, self.insertion_code)

    @property
    def label_key(self) -> tuple[str, str | None]:
        return (self.label_asym_id, self.label_seq_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "local": {"chain_id": self.local_chain_id, "seq_id": self.local_seq_id},
            "auth": {
                "asym_id": self.auth_asym_id,
                "seq_id": self.auth_seq_id,
                "insertion_code": self.insertion_code,
                "comp_id": self.auth_comp_id,
            },
            "label": {
                "asym_id": self.label_asym_id,
                "seq_id": self.label_seq_id,
                "comp_id": self.label_comp_id,
            },
        }


@dataclass(frozen=True, slots=True)
class LocalAtomMapping:
    local_atom_id: int
    local_chain_id: str
    local_seq_id: int
    local_atom_id_string: str
    auth_asym_id: str
    auth_seq_id: str
    insertion_code: str
    auth_comp_id: str
    auth_atom_id: str
    label_asym_id: str
    label_seq_id: str | None
    label_comp_id: str
    label_atom_id: str
    selected_source_altloc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "local": {
                "atom_id": self.local_atom_id,
                "chain_id": self.local_chain_id,
                "seq_id": self.local_seq_id,
                "atom_name": self.local_atom_id_string,
            },
            "auth": {
                "asym_id": self.auth_asym_id,
                "seq_id": self.auth_seq_id,
                "insertion_code": self.insertion_code,
                "comp_id": self.auth_comp_id,
                "atom_id": self.auth_atom_id,
            },
            "label": {
                "asym_id": self.label_asym_id,
                "seq_id": self.label_seq_id,
                "comp_id": self.label_comp_id,
                "atom_id": self.label_atom_id,
            },
            "selected_source_altloc": self.selected_source_altloc,
        }


@dataclass(frozen=True, slots=True)
class LocalMapping:
    """Complete local-to-auth/label mapping for all emitted residues and atoms."""

    residues: tuple[LocalResidueMapping, ...]
    atoms: tuple[LocalAtomMapping, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "residues", tuple(self.residues))
        object.__setattr__(self, "atoms", tuple(self.atoms))
        for name, keys in (
            ("local residue", [item.local_key for item in self.residues]),
            ("auth residue", [item.auth_key for item in self.residues]),
            (
                "label residue",
                [item.label_key for item in self.residues if item.label_seq_id is not None],
            ),
            ("local atom", [item.local_atom_id for item in self.atoms]),
        ):
            if len(keys) != len(set(keys)):
                raise ValueError(f"{name} identifiers are not unique")

    def residue_from_local(self, chain_id: str, seq_id: int) -> LocalResidueMapping:
        return self._find_residue("local_key", (chain_id, seq_id))

    def residue_from_auth(
        self, chain_id: str, seq_id: str | int, insertion_code: str = ""
    ) -> LocalResidueMapping:
        key = (chain_id, str(seq_id), _normalise_missing(insertion_code))
        return self._find_residue("auth_key", key)

    def residue_from_label(
        self, chain_id: str, seq_id: str | int
    ) -> LocalResidueMapping:
        return self._find_residue("label_key", (chain_id, str(seq_id)))

    def atom_from_local(self, local_atom_id: int) -> LocalAtomMapping:
        for item in self.atoms:
            if item.local_atom_id == local_atom_id:
                return item
        raise KeyError(local_atom_id)

    def _find_residue(self, attribute: str, key: object) -> LocalResidueMapping:
        for item in self.residues:
            if getattr(item, attribute) == key:
                return item
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "residues": [item.to_dict() for item in self.residues],
            "atoms": [item.to_dict() for item in self.atoms],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """A proper rigid transform ``x' = R x + t``."""

    rotation: np.ndarray
    translation: np.ndarray
    seed: int | None = None

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("rigid transform must contain only finite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
            raise ValueError("rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-10):
            raise ValueError("rotation must be proper (determinant +1)")
        rotation = rotation.copy()
        translation = translation.copy()
        rotation.flags.writeable = False
        translation.flags.writeable = False
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, coordinates: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(coordinates, dtype=np.float64)
        if array.shape[-1:] != (3,):
            raise ValueError("coordinates must end in an axis of length 3")
        return array @ self.rotation.T + self.translation

    def inverse(self) -> "RigidTransform":
        inverse_rotation = self.rotation.T
        inverse_translation = -(inverse_rotation @ self.translation)
        return RigidTransform(inverse_rotation, inverse_translation, self.seed)


@dataclass(frozen=True, slots=True)
class _ParsedAtom:
    order: int
    atom: Atom


def _tokenize_mmcif(text: str) -> Iterator[str]:
    """Yield mmCIF tokens, including quoted and semicolon-delimited values."""

    i = 0
    size = len(text)
    while i < size:
        while i < size and text[i].isspace():
            i += 1
        if i >= size:
            return
        if text[i] == "#":
            newline = text.find("\n", i)
            i = size if newline < 0 else newline + 1
            continue
        at_line_start = i == 0 or text[i - 1] == "\n"
        if text[i] == ";" and at_line_start:
            value_start = i + 1
            search = value_start
            close = -1
            while search < size:
                candidate = text.find("\n;", search)
                if candidate < 0:
                    break
                close = candidate + 1
                break
            if close < 0:
                raise ValueError("unterminated semicolon-delimited mmCIF value")
            value = text[value_start:close]
            if value.startswith("\r\n"):
                value = value[2:]
            elif value.startswith("\n"):
                value = value[1:]
            value = value.rstrip("\r\n")
            yield value
            line_end = text.find("\n", close + 1)
            i = size if line_end < 0 else line_end + 1
            continue
        if text[i] in {"'", '"'}:
            quote = text[i]
            i += 1
            start = i
            while i < size:
                if text[i] == quote and (i + 1 == size or text[i + 1].isspace()):
                    yield text[start:i]
                    i += 1
                    break
                i += 1
            else:
                raise ValueError("unterminated quoted mmCIF value")
            continue
        start = i
        while i < size and not text[i].isspace():
            i += 1
        yield text[start:i]


def _atom_site_rows(text: str) -> list[dict[str, str]]:
    tokens = list(_tokenize_mmcif(text))
    rows: list[dict[str, str]] = []
    i = 0
    while i < len(tokens):
        if tokens[i].lower() != "loop_":
            i += 1
            continue
        i += 1
        tags: list[str] = []
        while i < len(tokens) and tokens[i].startswith("_"):
            tags.append(tokens[i].lower())
            i += 1
        if not tags:
            raise ValueError("mmCIF loop_ has no tags")
        values: list[str] = []
        while i < len(tokens):
            token = tokens[i]
            lower = token.lower()
            at_row_boundary = len(values) % len(tags) == 0
            is_control = (
                lower in {"loop_", "stop_", "global_"}
                or lower.startswith("data_")
                or lower.startswith("save_")
                or token.startswith("_")
            )
            if at_row_boundary and is_control:
                break
            values.append(token)
            i += 1
        if len(values) % len(tags):
            raise ValueError(
                f"mmCIF loop has {len(values)} values for {len(tags)} columns"
            )
        if any(tag.startswith("_atom_site.") for tag in tags):
            if not all(tag.startswith("_atom_site.") for tag in tags):
                raise ValueError("mixed-category loop containing _atom_site tags is unsupported")
            for start in range(0, len(values), len(tags)):
                rows.append(dict(zip(tags, values[start : start + len(tags)])))
        if i < len(tokens) and tokens[i].lower() == "stop_":
            i += 1
    if not rows:
        raise ValueError("mmCIF contains no _atom_site loop rows")
    return rows


def _row_value(
    row: Mapping[str, str], *tags: str, default: str | None = None
) -> str | None:
    for tag in tags:
        value = row.get(tag.lower())
        if value is not None and value not in _MISSING_CIF_VALUES:
            return value
    return default


def _as_float(value: str | None, field: str, default: float | None = None) -> float:
    if value is None or value in _MISSING_CIF_VALUES:
        if default is not None:
            return default
        raise ValueError(f"missing required mmCIF numeric field {field}")
    match = _FLOAT_WITH_UNCERTAINTY.match(value)
    if match:
        value = match.group(1)
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value {value!r} for {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value for {field}")
    return result


def _as_model_number(value: str | None) -> int:
    if value is None or value in _MISSING_CIF_VALUES:
        return 1
    number = _as_float(value, "pdbx_PDB_model_num")
    if not number.is_integer():
        raise ValueError(f"model number must be an integer, got {value!r}")
    return int(number)


def _infer_element(atom_name: str) -> str:
    stripped = atom_name.strip().upper()
    stripped = stripped.lstrip("0123456789")
    if not stripped:
        return ""
    if stripped.startswith("SE"):
        return "SE"
    return stripped[0]


def _altloc_tie_key(altloc: str) -> tuple[int, str]:
    normalised = _normalise_missing(altloc).upper()
    if not normalised:
        return (0, "")
    if normalised == "A":
        return (1, "A")
    return (2, normalised)


def parse_mmcif(text: str) -> tuple[Residue, ...]:
    """Parse and select standard amino-acid, model-1, heavy ``ATOM`` rows.

    Altlocs are resolved independently per atom site by highest occupancy.  Exact
    occupancy ties prefer blank, then ``A``, then other altloc IDs in lexical
    order.  File order is the final tie breaker and determines output order.
    """

    selected: dict[tuple[object, ...], _ParsedAtom] = {}
    for order, row in enumerate(_atom_site_rows(text)):
        group = (_row_value(row, "_atom_site.group_pdb", default="ATOM") or "").upper()
        if group != "ATOM":
            continue
        model_num = _as_model_number(_row_value(row, "_atom_site.pdbx_pdb_model_num"))
        if model_num != 1:
            continue

        label_comp = (
            _row_value(row, "_atom_site.label_comp_id", "_atom_site.auth_comp_id") or ""
        ).upper()
        auth_comp = (
            _row_value(row, "_atom_site.auth_comp_id", "_atom_site.label_comp_id") or ""
        ).upper()
        if label_comp not in STANDARD_AMINO_ACIDS and auth_comp not in STANDARD_AMINO_ACIDS:
            continue

        label_atom = _row_value(row, "_atom_site.label_atom_id", "_atom_site.auth_atom_id") or ""
        auth_atom = _row_value(row, "_atom_site.auth_atom_id", "_atom_site.label_atom_id") or ""
        if not label_atom and not auth_atom:
            raise ValueError("ATOM row is missing both label_atom_id and auth_atom_id")
        element = (
            _row_value(row, "_atom_site.type_symbol") or _infer_element(label_atom or auth_atom)
        ).upper()
        if element in {"H", "D"}:
            continue

        label_chain = _row_value(row, "_atom_site.label_asym_id", "_atom_site.auth_asym_id")
        auth_chain = _row_value(row, "_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        auth_seq = _row_value(row, "_atom_site.auth_seq_id", "_atom_site.label_seq_id")
        label_seq = _normalise_seq_id(_row_value(row, "_atom_site.label_seq_id"))
        if label_chain is None or auth_chain is None or auth_seq is None:
            raise ValueError("ATOM row is missing a chain or residue identifier")

        insertion_code = _normalise_missing(
            _row_value(row, "_atom_site.pdbx_pdb_ins_code", default="")
        )
        altloc = _normalise_missing(
            _row_value(
                row,
                "_atom_site.label_alt_id",
                "_atom_site.pdbx_pdb_alt_id",
                default="",
            )
        )
        occupancy = _as_float(
            _row_value(row, "_atom_site.occupancy"), "occupancy", default=1.0
        )
        coordinate = np.array(
            [
                _as_float(_row_value(row, "_atom_site.cartn_x"), "Cartn_x"),
                _as_float(_row_value(row, "_atom_site.cartn_y"), "Cartn_y"),
                _as_float(_row_value(row, "_atom_site.cartn_z"), "Cartn_z"),
            ],
            dtype=np.float64,
        )
        atom = Atom(
            element=element,
            label_atom_id=label_atom,
            auth_atom_id=auth_atom,
            coordinate=coordinate,
            occupancy=occupancy,
            altloc=altloc,
            auth_asym_id=auth_chain,
            auth_seq_id=auth_seq,
            insertion_code=insertion_code,
            auth_comp_id=auth_comp,
            label_asym_id=label_chain,
            label_seq_id=label_seq,
            label_comp_id=label_comp,
            model_num=1,
        )
        residue_site = (
            auth_chain,
            auth_seq,
            insertion_code,
            label_chain,
            label_seq,
            auth_comp,
            label_comp,
        )
        atom_site = residue_site + (label_atom or auth_atom,)
        candidate = _ParsedAtom(order, atom)
        incumbent = selected.get(atom_site)
        if incumbent is None:
            selected[atom_site] = candidate
        elif occupancy > incumbent.atom.occupancy:
            selected[atom_site] = candidate
        elif occupancy == incumbent.atom.occupancy and _altloc_tie_key(altloc) < _altloc_tie_key(
            incumbent.atom.altloc
        ):
            selected[atom_site] = candidate

    if not selected:
        raise ValueError("mmCIF contains no model-1 standard amino-acid heavy ATOM rows")

    grouped: dict[tuple[object, ...], list[_ParsedAtom]] = {}
    first_order: dict[tuple[object, ...], int] = {}
    for atom_site, parsed_atom in selected.items():
        residue_site = atom_site[:-1]
        grouped.setdefault(residue_site, []).append(parsed_atom)
        first_order[residue_site] = min(first_order.get(residue_site, parsed_atom.order), parsed_atom.order)

    residues: list[Residue] = []
    for residue_site in sorted(grouped, key=first_order.__getitem__):
        auth_chain, auth_seq, insertion, label_chain, label_seq, auth_comp, label_comp = residue_site
        parsed_atoms = sorted(grouped[residue_site], key=lambda item: item.order)
        residues.append(
            Residue(
                auth_asym_id=str(auth_chain),
                auth_seq_id=str(auth_seq),
                insertion_code=str(insertion),
                auth_comp_id=str(auth_comp),
                label_asym_id=str(label_chain),
                label_seq_id=None if label_seq is None else str(label_seq),
                label_comp_id=str(label_comp),
                atoms=tuple(item.atom for item in parsed_atoms),
            )
        )
    return tuple(residues)


def parse_mmcif_file(path: str | Path) -> tuple[Residue, ...]:
    return parse_mmcif(Path(path).read_text(encoding="utf-8"))


def _insertion_sort_key(insertion_code: str) -> tuple[int, str]:
    insertion = _normalise_missing(insertion_code).upper()
    return (0, "") if not insertion else (1, insertion)


def _coerce_auth_range(
    value: AuthResidueRange | Sequence[object],
) -> AuthResidueRange:
    if isinstance(value, AuthResidueRange):
        return value
    fields = tuple(value)
    if len(fields) == 3:
        return AuthResidueRange(str(fields[0]), int(fields[1]), int(fields[2]))
    if len(fields) == 5:
        return AuthResidueRange(
            str(fields[0]), int(fields[1]), int(fields[2]), str(fields[3]), str(fields[4])
        )
    raise TypeError("author ranges must be AuthResidueRange or 3-/5-item sequences")


def crop_by_auth_ranges(
    residues: Sequence[Residue],
    ranges: Iterable[AuthResidueRange | Sequence[object]],
) -> tuple[Residue, ...]:
    """Keep only residues in inclusive author-chain ranges, preserving file order.

    Consequently, every unlisted chain (including binding partners) is stripped.
    Overlapping ranges do not duplicate residues.
    """

    normalised_ranges = tuple(_coerce_auth_range(item) for item in ranges)
    if not normalised_ranges:
        raise ValueError("at least one author range is required")
    cropped = tuple(
        residue
        for residue in residues
        if any(author_range.contains(residue) for author_range in normalised_ranges)
    )
    if not cropped:
        raise ValueError("author crop selected no residues")
    return cropped


def make_local_mapping(residues: Sequence[Residue]) -> LocalMapping:
    """Assign opaque chains ``T1``, ``T2``, ... and contiguous per-chain seq IDs."""

    if not residues:
        raise ValueError("cannot map an empty structure")
    chain_numbers: dict[str, int] = {}
    chain_residue_counts: dict[str, int] = {}
    residue_mappings: list[LocalResidueMapping] = []
    atom_mappings: list[LocalAtomMapping] = []
    local_atom_id = 1
    for residue in residues:
        if residue.auth_asym_id not in chain_numbers:
            chain_numbers[residue.auth_asym_id] = len(chain_numbers) + 1
            chain_residue_counts[residue.auth_asym_id] = 0
        local_chain = f"T{chain_numbers[residue.auth_asym_id]}"
        chain_residue_counts[residue.auth_asym_id] += 1
        local_seq = chain_residue_counts[residue.auth_asym_id]
        residue_mappings.append(
            LocalResidueMapping(
                local_chain_id=local_chain,
                local_seq_id=local_seq,
                auth_asym_id=residue.auth_asym_id,
                auth_seq_id=residue.auth_seq_id,
                insertion_code=residue.insertion_code,
                auth_comp_id=residue.auth_comp_id,
                label_asym_id=residue.label_asym_id,
                label_seq_id=residue.label_seq_id,
                label_comp_id=residue.label_comp_id,
            )
        )
        for atom in residue.atoms:
            atom_mappings.append(
                LocalAtomMapping(
                    local_atom_id=local_atom_id,
                    local_chain_id=local_chain,
                    local_seq_id=local_seq,
                    local_atom_id_string=atom.atom_name,
                    auth_asym_id=atom.auth_asym_id,
                    auth_seq_id=atom.auth_seq_id,
                    insertion_code=atom.insertion_code,
                    auth_comp_id=atom.auth_comp_id,
                    auth_atom_id=atom.auth_atom_id,
                    label_asym_id=atom.label_asym_id,
                    label_seq_id=atom.label_seq_id,
                    label_comp_id=atom.label_comp_id,
                    label_atom_id=atom.label_atom_id,
                    selected_source_altloc=atom.altloc,
                )
            )
            local_atom_id += 1
    return LocalMapping(tuple(residue_mappings), tuple(atom_mappings))


def _cif_token(value: str) -> str:
    if not value:
        return "."
    if any(character.isspace() for character in value) or value[0] in "_#$;[]":
        if "'" not in value:
            return f"'{value}'"
        if '"' not in value:
            return f'"{value}"'
        raise ValueError(f"cannot encode mmCIF token containing both quote styles: {value!r}")
    return value


def local_mmcif_text(
    residues: Sequence[Residue], mapping: LocalMapping | None = None
) -> tuple[str, LocalMapping]:
    """Return a minimal, metadata-free local mmCIF and its private source mapping."""

    residues = tuple(residues)
    mapping = make_local_mapping(residues) if mapping is None else mapping
    if len(mapping.residues) != len(residues):
        raise ValueError("mapping residue count does not match structure")
    if len(mapping.atoms) != sum(len(residue.atoms) for residue in residues):
        raise ValueError("mapping atom count does not match structure")

    tags = (
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.pdbx_PDB_model_num",
    )
    lines = ["data_local", "#", "loop_", *tags]
    atom_offset = 0
    for residue_index, residue in enumerate(residues):
        residue_mapping = mapping.residues[residue_index]
        for atom in residue.atoms:
            atom_mapping = mapping.atoms[atom_offset]
            x, y, z = atom.coordinate
            values = (
                "ATOM",
                str(atom_mapping.local_atom_id),
                _cif_token(atom.element),
                _cif_token(atom_mapping.local_atom_id_string),
                ".",
                _cif_token(residue.residue_name),
                _cif_token(residue_mapping.local_chain_id),
                str(residue_mapping.local_seq_id),
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{atom.occupancy:.3f}",
                "1",
            )
            lines.append(" ".join(values))
            atom_offset += 1
    lines.append("#")
    return "\n".join(lines) + "\n", mapping


def write_local_mmcif(
    residues: Sequence[Residue], path: str | Path, mapping: LocalMapping | None = None
) -> LocalMapping:
    text, completed_mapping = local_mmcif_text(residues, mapping)
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return completed_mapping


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return sha256_text(canonical)


def seeded_rigid_transform(seed: int, translation_scale: float = 10.0) -> RigidTransform:
    """Generate a deterministic uniform proper rotation and bounded translation.

    Rotation uses Shoemake's three-uniform quaternion construction.  Each
    translation component is uniform on ``[-translation_scale, translation_scale]``.
    """

    if not math.isfinite(translation_scale) or translation_scale < 0.0:
        raise ValueError("translation_scale must be finite and non-negative")
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random(3)
    x = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    y = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    z = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    w = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    translation = rng.uniform(-translation_scale, translation_scale, size=3)
    return RigidTransform(rotation, translation, seed)


def transform_residues(
    residues: Sequence[Residue], transform: RigidTransform
) -> tuple[Residue, ...]:
    transformed: list[Residue] = []
    for residue in residues:
        atoms = tuple(
            replace(atom, coordinate=transform.apply(atom.coordinate)) for atom in residue.atoms
        )
        transformed.append(replace(residue, atoms=atoms))
    return tuple(transformed)


def _fibonacci_sphere(number_of_points: int) -> np.ndarray:
    if number_of_points <= 0:
        raise ValueError("number_of_points must be positive")
    indices = np.arange(number_of_points, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / number_of_points
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))


def _flatten_atoms(residues: Sequence[Residue]) -> tuple[Atom, ...]:
    return tuple(atom for residue in residues for atom in residue.atoms)


def shrake_rupley_sasa(
    atoms: Sequence[Atom], *, probe_radius: float = 1.4, sphere_points: int = 960
) -> np.ndarray:
    """Return deterministic per-atom solvent-accessible areas in square angstroms."""

    atoms = tuple(atoms)
    if not atoms:
        return np.empty(0, dtype=np.float64)
    if not math.isfinite(probe_radius) or probe_radius < 0.0:
        raise ValueError("probe_radius must be finite and non-negative")
    unit_sphere = _fibonacci_sphere(sphere_points)
    coordinates = np.vstack([atom.coordinate for atom in atoms])
    expanded_radii = np.array(
        [VDW_RADII_ANGSTROM.get(atom.element.upper(), 1.70) + probe_radius for atom in atoms],
        dtype=np.float64,
    )
    areas = np.zeros(len(atoms), dtype=np.float64)
    center_distances_squared = np.sum(
        (coordinates[:, None, :] - coordinates[None, :, :]) ** 2, axis=2
    )
    for atom_index, (center, expanded_radius) in enumerate(zip(coordinates, expanded_radii)):
        possible = center_distances_squared[atom_index] < (
            expanded_radius + expanded_radii
        ) ** 2
        possible[atom_index] = False
        neighbor_indices = np.flatnonzero(possible)
        surface_points = center + expanded_radius * unit_sphere
        if neighbor_indices.size:
            displacement = (
                surface_points[:, None, :] - coordinates[neighbor_indices][None, :, :]
            )
            squared = np.einsum("pni,pni->pn", displacement, displacement)
            occluded = np.any(
                squared < expanded_radii[neighbor_indices][None, :] ** 2, axis=1
            )
            accessible_count = int(np.count_nonzero(~occluded))
        else:
            accessible_count = sphere_points
        areas[atom_index] = (
            4.0 * math.pi * expanded_radius * expanded_radius * accessible_count / sphere_points
        )
    return areas


def _minimum_residue_distances(residues: Sequence[Residue]) -> np.ndarray:
    count = len(residues)
    result = np.full((count, count), np.inf, dtype=np.float64)
    np.fill_diagonal(result, 0.0)
    coordinates = [np.vstack([atom.coordinate for atom in residue.atoms]) for residue in residues]
    for left in range(count):
        for right in range(left + 1, count):
            displacement = coordinates[left][:, None, :] - coordinates[right][None, :, :]
            minimum = float(np.sqrt(np.min(np.einsum("ijk,ijk->ij", displacement, displacement))))
            result[left, right] = minimum
            result[right, left] = minimum
    return result


def _patch_class(residue_name: str) -> str:
    for class_name, members in PATCH_COMPOSITION_CLASSES.items():
        if residue_name in members:
            return class_name
    raise ValueError(f"no patch-composition class for residue {residue_name!r}")


def _round_float(value: float, precision: int) -> float:
    rounded = round(float(value), precision)
    return 0.0 if rounded == 0.0 else rounded


def identity_free_residue_features(
    residues: Sequence[Residue],
    *,
    neighbor_cutoffs: Sequence[float] = (4.0, 6.0, 8.0),
    probe_radius: float = 1.4,
    sphere_points: int = 960,
    precision: int = 6,
) -> dict[str, Any]:
    """Build source-ID-free geometric and chemical residue features.

    Residues and neighbors are addressed only by opaque local tokens such as
    ``T1:7``; no auth/label identifiers are emitted.  Neighbor entries are sorted
    by minimum heavy-atom distance and then local token.  Patch composition is an
    aggregate over other residues within each cutoff and uses the documented
    coarse classes above.  ``relative_sasa_raw`` reports the direct ratio to the
    Tien et al. maximum; ``relative_sasa`` is its evaluator-safe [0, 1] clamp.
    """

    residues = tuple(residues)
    if not residues:
        raise ValueError("cannot featurize an empty structure")
    cutoffs = tuple(float(value) for value in neighbor_cutoffs)
    if not cutoffs or any(not math.isfinite(value) or value <= 0.0 for value in cutoffs):
        raise ValueError("neighbor cutoffs must be finite and positive")
    if tuple(sorted(set(cutoffs))) != cutoffs:
        raise ValueError("neighbor cutoffs must be strictly increasing")
    if precision < 0:
        raise ValueError("precision must be non-negative")

    mapping = make_local_mapping(residues)
    atoms = _flatten_atoms(residues)
    atom_sasa = shrake_rupley_sasa(
        atoms, probe_radius=probe_radius, sphere_points=sphere_points
    )
    minimum_distances = _minimum_residue_distances(residues)
    class_names = tuple(PATCH_COMPOSITION_CLASSES)
    residue_classes = tuple(_patch_class(residue.residue_name) for residue in residues)

    output_residues: list[dict[str, Any]] = []
    atom_offset = 0
    for residue_index, residue in enumerate(residues):
        local = mapping.residues[residue_index]
        ca = residue.ca_atom
        if ca is None:
            raise ValueError(
                f"local residue {local.local_chain_id}:{local.local_seq_id} has no CA atom"
            )
        sidechain_atoms = [atom for atom in residue.atoms if not atom.is_backbone]
        sidechain_fallback = not sidechain_atoms
        sidechain_centroid = (
            ca.coordinate
            if sidechain_fallback
            else np.mean(np.vstack([atom.coordinate for atom in sidechain_atoms]), axis=0)
        )
        residue_atom_count = len(residue.atoms)
        residue_atom_sasa = atom_sasa[atom_offset : atom_offset + residue_atom_count]

        neighbor_features: dict[str, list[dict[str, Any]]] = {}
        neighbor_counts: dict[str, int] = {}
        patch_composition: dict[str, Any] = {}
        for cutoff in cutoffs:
            cutoff_key = f"{cutoff:.1f}"
            neighbor_indices = sorted(
                (
                    other
                    for other in range(len(residues))
                    if other != residue_index
                    and minimum_distances[residue_index, other] <= cutoff
                ),
                key=lambda other: (
                    minimum_distances[residue_index, other],
                    mapping.residues[other].local_chain_id,
                    mapping.residues[other].local_seq_id,
                ),
            )
            neighbor_features[cutoff_key] = [
                {
                    "token": (
                        f"{mapping.residues[other].local_chain_id}:"
                        f"{mapping.residues[other].local_seq_id}"
                    ),
                    "min_heavy_atom_distance": _round_float(
                        minimum_distances[residue_index, other], precision
                    ),
                }
                for other in neighbor_indices
            ]
            neighbor_counts[cutoff_key] = len(neighbor_indices)
            counts = {class_name: 0 for class_name in class_names}
            for other in neighbor_indices:
                counts[residue_classes[other]] += 1
            denominator = len(neighbor_indices)
            fractions = {
                class_name: _round_float(
                    counts[class_name] / denominator if denominator else 0.0, precision
                )
                for class_name in class_names
            }
            patch_composition[cutoff_key] = {"counts": counts, "fractions": fractions}

        residue_sasa = float(np.sum(residue_atom_sasa))
        relative_sasa_raw = residue_sasa / MAXIMUM_RESIDUE_ASA[residue.residue_name]
        output_residues.append(
            {
                "local_index": residue_index + 1,
                "local_chain_id": local.local_chain_id,
                "local_seq_id": local.local_seq_id,
                "token": f"{local.local_chain_id}:{local.local_seq_id}",
                "residue_name": residue.residue_name,
                "residue_one_letter": RESIDUE_ONE_LETTER[residue.residue_name],
                "chemistry_class": residue_classes[residue_index],
                "ca_angstrom": [_round_float(value, precision) for value in ca.coordinate],
                "sidechain_heavy_atom_centroid_angstrom": [
                    _round_float(value, precision) for value in sidechain_centroid
                ],
                "sidechain_centroid_fallback_to_ca": sidechain_fallback,
                "neighbors": neighbor_features,
                "neighbor_counts": neighbor_counts,
                "patch_composition": patch_composition,
                "per_atom_sasa_angstrom2": [
                    {
                        "local_atom_id": mapping.atoms[atom_offset + local_atom_index].local_atom_id,
                        "sasa": _round_float(value, precision),
                    }
                    for local_atom_index, value in enumerate(residue_atom_sasa)
                ],
                "residue_sasa_angstrom2": _round_float(residue_sasa, precision),
                "relative_sasa_raw": _round_float(relative_sasa_raw, precision),
                "relative_sasa": _round_float(
                    min(1.0, max(0.0, relative_sasa_raw)), precision
                ),
            }
        )
        atom_offset += residue_atom_count

    return {
        "schema_version": 1,
        "coordinate_frame": "opaque_local",
        "neighbor_cutoffs_angstrom": list(cutoffs),
        "patch_composition_class_names": list(PATCH_COMPOSITION_CLASSES),
        "sasa_method": {
            "algorithm": "Shrake-Rupley",
            "probe_radius_angstrom": probe_radius,
            "sphere_points": sphere_points,
            "sphere_sampling": "deterministic_fibonacci",
            "vdw_radii_angstrom": dict(VDW_RADII_ANGSTROM),
            "unknown_element_radius_angstrom": 1.70,
            "relative_sasa_reference": "Tien et al. PLoS ONE 8:e80635 (2013)",
            "relative_sasa_fields": {
                "relative_sasa_raw": "unclipped residue SASA / residue-specific maximum",
                "relative_sasa": "relative_sasa_raw clamped to [0, 1]",
            },
        },
        "residues": output_residues,
    }


def identity_free_feature_json(
    residues: Sequence[Residue],
    *,
    neighbor_cutoffs: Sequence[float] = (4.0, 6.0, 8.0),
    probe_radius: float = 1.4,
    sphere_points: int = 960,
    precision: int = 6,
) -> str:
    features = identity_free_residue_features(
        residues,
        neighbor_cutoffs=neighbor_cutoffs,
        probe_radius=probe_radius,
        sphere_points=sphere_points,
        precision=precision,
    )
    return json.dumps(
        features,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


__all__ = [
    "Atom",
    "AuthResidueRange",
    "BACKBONE_ATOM_NAMES",
    "LocalAtomMapping",
    "LocalMapping",
    "LocalResidueMapping",
    "MAXIMUM_RESIDUE_ASA",
    "PATCH_COMPOSITION_CLASSES",
    "Residue",
    "RESIDUE_ONE_LETTER",
    "RigidTransform",
    "STANDARD_AMINO_ACIDS",
    "VDW_RADII_ANGSTROM",
    "crop_by_auth_ranges",
    "identity_free_feature_json",
    "identity_free_residue_features",
    "local_mmcif_text",
    "make_local_mapping",
    "parse_mmcif",
    "parse_mmcif_file",
    "seeded_rigid_transform",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "shrake_rupley_sasa",
    "transform_residues",
    "write_local_mmcif",
]
