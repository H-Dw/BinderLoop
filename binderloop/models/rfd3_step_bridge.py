#!/usr/bin/env python3
"""Stdlib-only glue between Foundry RFD3, ProteinMPNN, and RF3 outputs.

This module is copied into a job package and executed with the Foundry Python
interpreter, so it must not import binderloop or third-party packages.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


TRAJECTORY_MARKERS = ("denoised", "noisy", "trajectory", "traj")
STRUCTURE_SUFFIXES = (".cif.gz", ".cif", ".pdb.gz", ".pdb")


def _is_trajectory(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in TRAJECTORY_MARKERS)


def discover_design_structures(design_dir: Path) -> List[Path]:
    """Return RFD3 design models, excluding optional diffusion trajectories."""
    root = Path(design_dir)
    if not root.exists():
        return []
    found: List[Path] = []
    for pattern in ("*_model_*.cif.gz", "*_model_*.cif", "*.cif.gz", "*.cif", "*.pdb"):
        for path in sorted(root.rglob(pattern)):
            if not path.is_file() or _is_trajectory(path):
                continue
            if path not in found:
                found.append(path)
    return found


def discover_fold_confidences(fold_dir: Path) -> List[Path]:
    root = Path(fold_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*_summary_confidences.json") if path.is_file())


def write_mpnn_config(
    design_structures: Sequence[Path],
    *,
    out_path: Path,
    out_directory: Path,
    params: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a multi-input Foundry MPNN ``config_json`` for binder-only redesign."""
    params = dict(params or {})
    designed = params.get("designed_chains") or params.get("binder_chain") or "A"
    if isinstance(designed, str):
        designed_chains = [part.strip() for part in designed.replace(" ", "").split(",") if part.strip()]
    else:
        designed_chains = [str(item).strip() for item in designed if str(item).strip()]
    batch_size = max(1, int(params.get("inverse_fold_num_sequences") or params.get("batch_size") or 1))
    number_of_batches = max(1, int(params.get("number_of_batches") or 1))
    temperature = float(params.get("temperature", 0.1))
    checkpoint = str(params.get("mpnn_checkpoint") or params.get("checkpoint_path") or "proteinmpnn_v_48_020.pt")
    inputs: List[Dict[str, Any]] = []
    for path in design_structures:
        inputs.append(
            {
                "structure_path": str(path),
                "name": path.name.split(".")[0],
                "batch_size": batch_size,
                "number_of_batches": number_of_batches,
                "temperature": temperature,
                "designed_chains": designed_chains,
                "fixed_chains": None,
            }
        )
    payload = {
        "model_type": str(params.get("model_type") or "protein_mpnn"),
        "checkpoint_path": checkpoint,
        "is_legacy_weights": True if params.get("is_legacy_weights", True) not in {False, "False", "false", 0, "0"} else False,
        "out_directory": str(out_directory),
        "write_fasta": True,
        "write_structures": True,
        "inputs": inputs,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _first_number(payload: Mapping[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key not in payload or payload[key] in (None, ""):
            continue
        value = payload[key]
        if isinstance(value, Mapping):
            for nested in ("mean", "value", "score", "avg"):
                if nested in value:
                    try:
                        return float(value[nested])
                    except (TypeError, ValueError):
                        continue
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def parse_rf3_confidence(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        payload = {}
    iptm = _first_number(payload, ("iptm", "ipTM", "interface_ptm", "i_ptm"))
    ptm = _first_number(payload, ("ptm", "pTM"))
    plddt = _first_number(payload, ("plddt", "pLDDT", "mean_plddt", "average_plddt"))
    ranking = _first_number(payload, ("ranking_score", "rankingScore", "score"))
    return {
        "design": Path(path).name.replace("_summary_confidences.json", ""),
        "path": str(path),
        "iptm": iptm,
        "ptm": ptm,
        "plddt": plddt,
        "ranking_score": ranking,
        "pass_filters": None,
    }


def write_metrics_csv(rows: Sequence[Mapping[str, Any]], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["design", "path", "iptm", "ptm", "plddt", "ranking_score", "refolding_rmsd", "pass_filters"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return out_path


def assemble_mpnn_config(args: argparse.Namespace) -> int:
    design_dir = Path(args.design_dir)
    structures = discover_design_structures(design_dir)
    params = json.loads(args.params_json) if args.params_json else {}
    if args.params_file:
        params.update(json.loads(Path(args.params_file).read_text(encoding="utf-8")))
    write_mpnn_config(
        structures,
        out_path=Path(args.out),
        out_directory=Path(args.out_directory),
        params=params,
    )
    return 0 if structures else 2


def aggregate_metrics(args: argparse.Namespace) -> int:
    fold_dir = Path(args.fold_dir)
    rows = [parse_rf3_confidence(path) for path in discover_fold_confidences(fold_dir)]
    threshold = None if args.rmsd_threshold in (None, "") else float(args.rmsd_threshold)
    for row in rows:
        iptm = row.get("iptm")
        rmsd = row.get("refolding_rmsd")
        passed = True
        if iptm is not None and args.iptm_threshold not in (None, "") and float(iptm) < float(args.iptm_threshold):
            passed = False
        if rmsd is not None and threshold is not None and float(rmsd) > threshold:
            passed = False
        row["pass_filters"] = passed
    write_metrics_csv(rows, Path(args.out))
    return 0 if rows else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foundry RFD3 step bridge")
    sub = parser.add_subparsers(dest="command", required=True)

    assemble = sub.add_parser("assemble-mpnn", help="Write ProteinMPNN config_json from RFD3 designs")
    assemble.add_argument("--design-dir", required=True)
    assemble.add_argument("--out", required=True)
    assemble.add_argument("--out-directory", required=True)
    assemble.add_argument("--params-json", default="")
    assemble.add_argument("--params-file", default="")
    assemble.set_defaults(func=assemble_mpnn_config)

    aggregate = sub.add_parser("aggregate-metrics", help="Write final_designs_metrics.csv from RF3 confidences")
    aggregate.add_argument("--fold-dir", required=True)
    aggregate.add_argument("--out", required=True)
    aggregate.add_argument("--iptm-threshold", default="")
    aggregate.add_argument("--rmsd-threshold", default="")
    aggregate.set_defaults(func=aggregate_metrics)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
