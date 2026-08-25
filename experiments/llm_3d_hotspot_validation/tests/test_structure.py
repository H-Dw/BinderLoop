from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import structure  # noqa: E402


SYNTHETIC_CIF = """\
data_synthetic
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.auth_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.pdbx_PDB_model_num
ATOM 1 N N N . ALA ALA LABA AUTH_A 1 10 ? 0.0 0.0 0.0 1.00 1
ATOM 2 C CA CA A ALA ALA LABA AUTH_A 1 10 ? 9.0 9.0 9.0 0.50 1
ATOM 3 C CA CA . ALA ALA LABA AUTH_A 1 10 ? 1.0 0.0 0.0 0.50 1
ATOM 4 C C C . ALA ALA LABA AUTH_A 1 10 ? 2.0 0.0 0.0 1.00 1
ATOM 5 O O O . ALA ALA LABA AUTH_A 1 10 ? 3.0 0.0 0.0 1.00 1
ATOM 6 C CB CB A ALA ALA LABA AUTH_A 1 10 ? 1.0 1.0 0.0 0.70 1
ATOM 7 C CB CB B ALA ALA LABA AUTH_A 1 10 ? 1.0 2.0 0.0 0.80 1
ATOM 8 H H H . ALA ALA LABA AUTH_A 1 10 ? 1.0 0.0 1.0 1.00 1
ATOM 9 C CA CA . ALA ALA LABA AUTH_A 1 10 ? 99.0 99.0 99.0 1.00 2
ATOM 10 N N N . GLY GLY LABA AUTH_A 2 10 A 4.0 0.0 0.0 1.00 1
ATOM 11 C CA CA . GLY GLY LABA AUTH_A 2 10 A 5.0 0.0 0.0 1.00 1
ATOM 12 C C C . GLY GLY LABA AUTH_A 2 10 A 6.0 0.0 0.0 1.00 1
ATOM 13 O O O . GLY GLY LABA AUTH_A 2 10 A 7.0 0.0 0.0 1.00 1
ATOM 14 N N N . SER SER LABP PARTNER 1 50 ? 0.0 5.0 0.0 1.00 1
ATOM 15 C CA CA . SER SER LABP PARTNER 1 50 ? 1.0 5.0 0.0 1.00 1
ATOM 16 C C C . SER SER LABP PARTNER 1 50 ? 2.0 5.0 0.0 1.00 1
ATOM 17 O O O . SER SER LABP PARTNER 1 50 ? 3.0 5.0 0.0 1.00 1
ATOM 18 C CB CB . SER SER LABP PARTNER 1 50 ? 1.0 6.0 0.0 1.00 1
HETATM 19 C C1 C1 . LIG LIG LIGCHAIN LIGAUTH 1 999 ? 0.0 0.0 8.0 1.00 1
#
"""


def _coordinates(residues: tuple[structure.Residue, ...]) -> np.ndarray:
    return np.vstack([atom.coordinate for residue in residues for atom in residue.atoms])


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def test_parse_auth_label_insertion_model_and_altloc_policy() -> None:
    residues = structure.parse_mmcif(SYNTHETIC_CIF)

    assert len(residues) == 3
    first, inserted, partner = residues
    assert first.auth_key == ("AUTH_A", "10", "")
    assert first.label_key == ("LABA", "1")
    assert inserted.auth_key == ("AUTH_A", "10", "A")
    assert inserted.label_key == ("LABA", "2")
    assert partner.auth_asym_id == "PARTNER"

    atoms = {atom.atom_name: atom for atom in first.atoms}
    assert set(atoms) == {"N", "CA", "C", "O", "CB"}
    np.testing.assert_allclose(atoms["CA"].coordinate, [1.0, 0.0, 0.0])
    assert atoms["CA"].altloc == ""  # blank wins an occupancy tie against A
    np.testing.assert_allclose(atoms["CB"].coordinate, [1.0, 2.0, 0.0])
    assert atoms["CB"].altloc == "B"  # occupancy outranks the A preference
    assert all(atom.model_num == 1 and atom.element != "H" for atom in atoms.values())


def test_author_crop_strips_partner_and_handles_insertion_boundaries() -> None:
    residues = structure.parse_mmcif(SYNTHETIC_CIF)

    crop = structure.crop_by_auth_ranges(residues, [("AUTH_A", 10, 10)])
    assert [residue.insertion_code for residue in crop] == ["", "A"]
    assert {residue.auth_asym_id for residue in crop} == {"AUTH_A"}

    inserted_only = structure.crop_by_auth_ranges(
        residues, [structure.AuthResidueRange("AUTH_A", 10, 10, "A", "A")]
    )
    assert len(inserted_only) == 1
    assert inserted_only[0].insertion_code == "A"


def test_local_writer_strips_metadata_and_mapping_is_complete() -> None:
    crop = structure.crop_by_auth_ranges(
        structure.parse_mmcif(SYNTHETIC_CIF), [("AUTH_A", 10, 10)]
    )
    cif_text, mapping = structure.local_mmcif_text(crop)

    assert "AUTH_A" not in cif_text
    assert "LABA" not in cif_text
    assert "PARTNER" not in cif_text
    assert "_atom_site.auth_" not in cif_text
    assert "_atom_site.pdbx_PDB_ins_code" not in cif_text
    assert [item.local_key for item in mapping.residues] == [("T1", 1), ("T1", 2)]
    assert mapping.residue_from_local("T1", 2).auth_key == ("AUTH_A", "10", "A")
    assert mapping.residue_from_auth("AUTH_A", 10, "A").label_key == ("LABA", "2")
    assert mapping.residue_from_label("LABA", 1).auth_key == ("AUTH_A", "10", "")
    assert len(mapping.atoms) == sum(len(residue.atoms) for residue in crop)
    assert mapping.atom_from_local(1).auth_atom_id == "N"

    reparsed = structure.parse_mmcif(cif_text)
    assert [(residue.auth_asym_id, residue.auth_seq_id) for residue in reparsed] == [
        ("T1", "1"),
        ("T1", "2"),
    ]
    assert "T1" in cif_text
    assert "L1" not in cif_text


def test_local_mapping_assigns_sequential_t_chains() -> None:
    mapping = structure.make_local_mapping(structure.parse_mmcif(SYNTHETIC_CIF))
    assert [item.local_key for item in mapping.residues] == [
        ("T1", 1),
        ("T1", 2),
        ("T2", 1),
    ]


def test_seeded_proper_rigid_transform_is_deterministic_and_distance_invariant() -> None:
    residues = structure.parse_mmcif(SYNTHETIC_CIF)
    original = _coordinates(residues)
    first = structure.seeded_rigid_transform(421, translation_scale=25.0)
    second = structure.seeded_rigid_transform(421, translation_scale=25.0)

    np.testing.assert_allclose(first.rotation, second.rotation, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first.translation, second.translation, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.linalg.det(first.rotation), 1.0, atol=1e-12, rtol=0.0)
    transformed = structure.transform_residues(residues, first)
    moved = _coordinates(transformed)
    np.testing.assert_allclose(
        _pairwise_distances(original), _pairwise_distances(moved), atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(first.inverse().apply(moved), original, atol=1e-10, rtol=1e-10)


def test_identity_free_features_sasa_and_hashes_are_deterministic() -> None:
    crop = structure.crop_by_auth_ranges(
        structure.parse_mmcif(SYNTHETIC_CIF), [("AUTH_A", 10, 10)]
    )
    first = structure.identity_free_feature_json(crop, sphere_points=64)
    second = structure.identity_free_feature_json(crop, sphere_points=64)

    assert first == second
    assert "AUTH_A" not in first and "LABA" not in first
    parsed = json.loads(first)
    assert parsed["neighbor_cutoffs_angstrom"] == [4.0, 6.0, 8.0]
    assert parsed["residues"][0]["token"] == "T1:1"
    assert parsed["residues"][0]["residue_name"] == "ALA"
    assert parsed["residues"][0]["residue_one_letter"] == "A"
    assert parsed["residues"][0]["chemistry_class"] == "hydrophobic"
    assert parsed["residues"][0]["neighbors"]["4.0"] == [
        {"token": "T1:2", "min_heavy_atom_distance": 1.0}
    ]
    assert parsed["residues"][1]["neighbors"]["4.0"] == [
        {"token": "T1:1", "min_heavy_atom_distance": 1.0}
    ]
    assert all(item["local_chain_id"] == "T1" for item in parsed["residues"])
    assert "L1" not in first
    assert parsed["residues"][1]["sidechain_centroid_fallback_to_ca"] is True
    assert (
        parsed["residues"][1]["sidechain_heavy_atom_centroid_angstrom"]
        == parsed["residues"][1]["ca_angstrom"]
    )
    assert all(item["residue_sasa_angstrom2"] > 0.0 for item in parsed["residues"])
    assert all(item["relative_sasa_raw"] > 0.0 for item in parsed["residues"])
    assert all(0.0 <= item["relative_sasa"] <= 1.0 for item in parsed["residues"])
    assert structure.sha256_text(first) == structure.sha256_bytes(first.encode("utf-8"))
    assert structure.sha256_json(parsed) == structure.sha256_text(first)
