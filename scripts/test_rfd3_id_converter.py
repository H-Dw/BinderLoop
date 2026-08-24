#!/usr/bin/env python3
"""Tests for RFD3 label/auth residue ID conversion."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binderloop.models.rfd3_adapter import RFD3Adapter
from binderloop.models.rfd3_id_converter import (
    adapt_rfd3_identifiers,
    detect_source_scheme,
    parse_structure_id_map,
    write_adapted_cif,
)
from binderloop.models.base import DesignJob


PDL1 = ROOT / "examples" / "bg_example" / "PD-L1.cif"
PDB = ROOT / "models" / "foundry" / "models" / "rfd3" / "docs" / "input_pdbs" / "5o45_cropped.pdb"


def test_pdl1_label_auth_map() -> None:
    id_map = parse_structure_id_map(PDL1)
    assert id_map.native_scheme == "label"
    assert id_map.format == "cif"
    tyr = id_map.lookup("A", 40, "label")
    assert tyr.auth_seq_id == 56
    assert tyr.resname == "TYR"
    assert "OH" in tyr.atom_names
    assert id_map.lookup("A", 56, "auth").label_seq_id == 40


def test_auto_detect_auth_from_official_hotspots() -> None:
    id_map = parse_structure_id_map(PDL1)
    scheme = detect_source_scheme(
        id_map,
        chain_id="A",
        numbers=[56, 115, 123, 17, 132],
        atom_hints={("A", 56): ["CG", "OH"], ("A", 115): ["CG", "SD"], ("A", 123): ["CD2", "OH"]},
    )
    assert scheme == "auth"


def test_auto_detect_label_from_boltzgen_hotspots() -> None:
    id_map = parse_structure_id_map(PDL1)
    scheme = detect_source_scheme(id_map, chain_id="A", numbers=[40, 99, 107, 1, 116])
    assert scheme == "label"


def test_adapt_auth_hotspots_on_cif_to_label(tmp_path: Path) -> None:
    adapted = adapt_rfd3_identifiers(
        PDL1,
        chain_id="A",
        res_index="17-132",
        hotspots=["A:56", "A:115", "A:123"],
        select_hotspots={"A56": "CG,OH", "A115": "CG,SD", "A123": "CD2,OH"},
        output_dir=tmp_path,
    )
    assert adapted.source_scheme == "auth"
    assert adapted.target_scheme == "label"
    assert adapted.res_index == "1-116"
    assert adapted.hotspots == ["A40", "A99", "A107"]
    assert adapted.select_hotspots == {"A40": "CG,OH", "A99": "CG,SD", "A107": "CD2,OH"}
    assert adapted.contig_target == "A1-116"


def test_adapt_label_hotspots_on_cif_stay_label() -> None:
    adapted = adapt_rfd3_identifiers(
        PDL1,
        chain_id="A",
        res_index="1-116",
        hotspots=["A:40", "A:99", "A:107"],
        source_scheme="label",
    )
    assert adapted.res_index == "1-116"
    assert adapted.hotspots == ["A40", "A99", "A107"]


def test_pdb_keeps_auth_ids() -> None:
    adapted = adapt_rfd3_identifiers(
        PDB,
        chain_id="A",
        res_index="17-131",
        hotspots=["A:56", "A:115", "A:123"],
        select_hotspots={"A56": "CG,OH"},
    )
    assert adapted.native_scheme == "auth"
    assert adapted.res_index == "17-131"
    assert adapted.hotspots[0] == "A56"
    assert adapted.select_hotspots["A56"] == "CG,OH"


def test_label_hotspots_on_pdb_convert_to_auth() -> None:
    adapted = adapt_rfd3_identifiers(
        PDB,
        chain_id="A",
        res_index="1-116",
        hotspots=["A:40", "A:99", "A:107"],
        source_scheme="label",
        target_scheme="native",
    )
    # 5o45 PDB has no separate label track; label==auth, so 1-116 stays 1-116.
    assert adapted.native_scheme == "auth"
    assert adapted.hotspots == ["A40", "A99", "A107"]


def test_rewrite_cif_copies_auth_into_label(tmp_path: Path) -> None:
    dest = tmp_path / "pdl1_auth.cif"
    write_adapted_cif(PDL1, dest, target_scheme="auth")
    rewritten = parse_structure_id_map(dest)
    tyr = rewritten.lookup("A", 56, "label")
    assert tyr.resname == "TYR"
    assert tyr.auth_seq_id == 56
    adapted = adapt_rfd3_identifiers(
        PDL1,
        chain_id="A",
        res_index="17-132",
        hotspots=["A:56"],
        select_hotspots={"A56": "CG,OH"},
        target_scheme="auth",
        adapt_structure=True,
        output_dir=tmp_path,
    )
    assert adapted.structure_path.endswith(".rfd3_auth.cif")
    assert adapted.hotspots == ["A56"]
    assert adapted.res_index == "17-132"


def test_adapter_writes_converted_spec(tmp_path: Path) -> None:
    job = DesignJob(
        job_id="idconv",
        target_structure=str(PDL1),
        chain_id="A",
        hotspots=["A:56", "A:115", "A:123"],
        binder_length=50,
        params={
            "target_res_index": "17-132",
            "select_hotspots": {"A56": "CG,OH", "A115": "CG,SD", "A123": "CD2,OH"},
        },
        output_dir=str(tmp_path / "job"),
    )
    spec_path = RFD3Adapter().write_design_spec(job)
    payload = spec_path.read_text(encoding="utf-8")
    assert "50,/0,A1-116" in payload
    assert "A40:" in payload
    report = json.loads((tmp_path / "job" / "rfd3_id_conversion.json").read_text(encoding="utf-8"))
    assert report["source_scheme"] == "auth"
    assert report["target_scheme"] == "label"
