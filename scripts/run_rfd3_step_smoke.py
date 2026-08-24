#!/usr/bin/env python3
"""Dry-run or live per-step smoke for the Foundry RFD3 pathway.

Default is dry-run (print conda-wrapped commands). ``--submit`` executes each
selected CLI in conda env ``foundry``. Isolated live smokes do not start from a
full 200-step PPI campaign:

- design: ``n_batches=1 diffusion_batch_size=1`` (optional ``--num-timesteps``)
- inverse_folding: official cropped PDB if no RFD3 CIFs are present
- folding: official RF3 ``5vht_from_json.json`` if no MPNN outputs are present
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binderloop.config import load_config
from binderloop.models.base import DesignJob
from binderloop.models.rfd3_adapter import RFD3Adapter
from binderloop.models.rfd3_renderer import render_mpnn_command, render_rf3_fold_command, with_default_foundry_artifacts
from binderloop.models.rfd3_step_bridge import discover_design_structures
from binderloop.orchestration.runner import conda_run_command


STEPS = ("design", "inverse_folding", "folding")
OFFICIAL_MPNN_PDB = ROOT / "models" / "foundry" / "models" / "rfd3" / "docs" / "input_pdbs" / "5o45_cropped.pdb"
OFFICIAL_RF3_JSON = ROOT / "models" / "foundry" / "models" / "rf3" / "tests" / "data" / "5vht_from_json.json"


def _job(out_dir: Path, cfg, *, n_batches: int, diffusion_batch_size: int, num_timesteps) -> DesignJob:
    params = with_default_foundry_artifacts(
        dict(cfg.search_space.rfd3 or {}),
        weights_path=cfg.runtime.model_runtime("rfd3").weights_path,
    )
    params["n_batches"] = int(n_batches)
    params["diffusion_batch_size"] = int(diffusion_batch_size)
    if num_timesteps is not None:
        params["num_timesteps"] = int(num_timesteps)
    return DesignJob(
        job_id="rfd3_smoke",
        target_structure=str(Path(cfg.target.structure_path)),
        chain_id=cfg.target.chain_id,
        hotspots=list(cfg.target.hotspots),
        binder_length=int((cfg.search_space.binder_lengths or [50])[0]),
        params=params,
        output_dir=str(out_dir),
    )


def _probe_foundry(conda_executable: str, env_name: str) -> dict:
    report = {}
    for cli in ("rfd3", "mpnn", "rf3"):
        command = conda_run_command([cli, "--help"], env_name=env_name, conda_executable=conda_executable)
        try:
            proc = subprocess.run(command, cwd=str(ROOT), universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            report[cli] = {"returncode": proc.returncode, "ok": proc.returncode == 0}
        except FileNotFoundError as exc:
            report[cli] = {"returncode": None, "ok": False, "error": str(exc)}
    return report


def _command_for_step(adapter: RFD3Adapter, job: DesignJob, step: str) -> list:
    layout = adapter.output_layout(job)
    params = with_default_foundry_artifacts(job.params or {})
    if step == "design":
        return adapter.build_step_command(job, "design")
    if step == "inverse_folding":
        designs = discover_design_structures(layout["design"])
        if designs:
            return adapter.build_step_command(job, "inverse_folding")
        layout["inverse_fold"].mkdir(parents=True, exist_ok=True)
        print(f"[inverse_folding] no design CIFs; using official PDB {OFFICIAL_MPNN_PDB}")
        return render_mpnn_command(
            params=params,
            out_directory=layout["inverse_fold"],
            structure_path=OFFICIAL_MPNN_PDB,
        )
    fold_inputs = layout["inverse_fold"]
    has_mpnn = fold_inputs.is_dir() and any(fold_inputs.iterdir())
    if has_mpnn:
        return adapter.build_step_command(job, "folding")
    print(f"[folding] no MPNN outputs; using official RF3 fixture {OFFICIAL_RF3_JSON}")
    layout["fold"].mkdir(parents=True, exist_ok=True)
    return render_rf3_fold_command(inputs=OFFICIAL_RF3_JSON, output_dir=layout["fold"], params=params)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RFD3 Foundry steps separately")
    parser.add_argument("--config", default="configs/example_rfd3_binder_task.yaml")
    parser.add_argument("--step", choices=(*STEPS, "all"), default="all")
    parser.add_argument("--out", default="outputs/rfd3_step_smoke")
    parser.add_argument("--submit", action="store_true", help="Execute the Foundry CLI in conda env foundry")
    parser.add_argument("--n-batches", type=int, default=1, help="Live design smoke batch count (default 1)")
    parser.add_argument("--diffusion-batch-size", type=int, default=1, help="Live design smoke diffusion batch (default 1)")
    parser.add_argument("--num-timesteps", type=int, default=None, help="Optional shorter diffusion for design smoke")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    runtime = cfg.runtime.model_runtime("rfd3")
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    foundry_root = getattr(cfg.runtime, "foundry_root", None) or str(ROOT / "models" / "foundry")
    adapter = RFD3Adapter(str(Path(foundry_root)))
    job = _job(
        out_dir / "job",
        cfg,
        n_batches=args.n_batches,
        diffusion_batch_size=args.diffusion_batch_size,
        num_timesteps=args.num_timesteps,
    )
    selected = list(STEPS) if args.step == "all" else [args.step]

    print("Foundry CLI probe:")
    probe = _probe_foundry(cfg.runtime.conda_executable, runtime.conda_env)
    for name, item in probe.items():
        print(f"  {name}: {'ok' if item.get('ok') else 'unavailable'} returncode={item.get('returncode')}")

    for step in selected:
        command = _command_for_step(adapter, job, step)
        wrapped = conda_run_command(command, env_name=runtime.conda_env, conda_executable=cfg.runtime.conda_executable)
        print(f"\n[{step}] {' '.join(str(token) for token in wrapped)}")
        if not args.submit:
            continue
        proc = subprocess.run(wrapped, cwd=str(adapter.root), check=False)
        print(f"[{step}] exit={proc.returncode}")
        if proc.returncode != 0:
            return proc.returncode
    if not args.submit:
        print("\nDry-run only. Pass --submit to execute each selected Foundry CLI in conda env foundry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
