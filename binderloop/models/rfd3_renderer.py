"""Command renderers for the official Foundry RFD3 → ProteinMPNN → RF3 pathway.

RFD3 and RF3 use Hydra ``key=value`` overrides. ProteinMPNN uses argparse and
legacy-weight booleans that must be the tokens ``True``/``False`` (not Hydra
lowercase ``true``/``false``).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, List, Mapping, Optional, Sequence, Union


RFD3_STEPS = ("design", "inverse_folding", "folding")
DEFAULT_RFD3_PPI_STEP_SCALE = 3.0
DEFAULT_RFD3_PPI_GAMMA_0 = 0.2
DEFAULT_DIFFUSION_BATCH_SIZE = 8
DEFAULT_MPNN_TEMPERATURE = 0.1
DEFAULT_MPNN_MODEL_TYPE = "protein_mpnn"

DEFAULT_CHECKPOINT_FILENAMES = {
    "rfd3": ("rfd3_latest.ckpt", "rfd3.ckpt"),
    "mpnn": ("proteinmpnn_v_48_020.pt",),
    "rf3": ("rf3_foundry_01_24_latest_remapped.ckpt", "rf3.ckpt"),
}
FOUNDRY_ALIASES = {"rfd3": "rfd3", "rf3": "rf3"}


def _as_path(value: Union[str, Path, PurePosixPath]) -> Path:
    text = str(value)
    if text.startswith("/"):
        return Path(text)
    path = Path(value).expanduser()
    return path if path.is_absolute() else path


def _hydra_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return "true"
    if token in {"0", "false", "no", "n", "off"}:
        return "false"
    return token


def _mpnn_bool(value: Any) -> str:
    """ProteinMPNN ``str2bool`` only accepts ``True``/``1`` and ``False``/``0``."""
    if isinstance(value, bool):
        return "True" if value else "False"
    token = str(value).strip()
    if token in {"True", "1", "true", "yes", "y", "on"}:
        return "True"
    return "False"


def _positive_int(params: Mapping[str, Any], key: str, default: Optional[int] = None) -> int:
    if key not in params or params[key] in (None, ""):
        if default is None:
            raise ValueError(f"resolved RFD3 params require {key}")
        value = default
    else:
        if isinstance(params[key], bool):
            raise ValueError(f"resolved RFD3 params require integer {key}")
        value = int(params[key])
    if value < 1:
        raise ValueError(f"resolved RFD3 params require {key} >= 1")
    return value


def resolve_rfd3_batching(params: Mapping[str, Any]) -> tuple:
    """Map harness ``num_designs`` onto official ``n_batches * diffusion_batch_size``."""
    batch = _positive_int(params, "diffusion_batch_size", DEFAULT_DIFFUSION_BATCH_SIZE)
    if params.get("n_batches") not in (None, ""):
        n_batches = _positive_int(params, "n_batches")
    else:
        num_designs = _positive_int(params, "num_designs", batch)
        n_batches = max(1, (num_designs + batch - 1) // batch)
    return n_batches, batch


def _first_existing(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def with_default_foundry_artifacts(
    params: Mapping[str, Any],
    *,
    weights_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Resolve Foundry checkpoints from ``weights_path``, else official aliases."""
    merged = dict(params or {})
    weight_root = merged.get("weights_path") or merged.get("checkpoint_dir") or weights_path
    root = _as_path(weight_root) if weight_root else None

    rfd3_ckpt = merged.get("rfd3_checkpoint") or merged.get("ckpt_path")
    if not rfd3_ckpt and root is not None:
        found = _first_existing(root, DEFAULT_CHECKPOINT_FILENAMES["rfd3"])
        rfd3_ckpt = str(found) if found is not None else None
    merged["rfd3_checkpoint"] = str(rfd3_ckpt or FOUNDRY_ALIASES["rfd3"])
    merged["ckpt_path"] = merged["rfd3_checkpoint"]

    mpnn_ckpt = merged.get("mpnn_checkpoint") or merged.get("checkpoint_path")
    if not mpnn_ckpt and root is not None:
        found = _first_existing(root, DEFAULT_CHECKPOINT_FILENAMES["mpnn"])
        mpnn_ckpt = str(found) if found is not None else str(root / DEFAULT_CHECKPOINT_FILENAMES["mpnn"][0])
    merged["mpnn_checkpoint"] = str(mpnn_ckpt or DEFAULT_CHECKPOINT_FILENAMES["mpnn"][0])
    merged["checkpoint_path"] = merged["mpnn_checkpoint"]

    rf3_ckpt = merged.get("rf3_checkpoint") or merged.get("folding_checkpoint")
    if not rf3_ckpt and root is not None:
        found = _first_existing(root, DEFAULT_CHECKPOINT_FILENAMES["rf3"])
        rf3_ckpt = str(found) if found is not None else None
    merged["rf3_checkpoint"] = str(rf3_ckpt or FOUNDRY_ALIASES["rf3"])
    if root is not None and merged.get("weights_path") in (None, ""):
        merged["weights_path"] = str(root)
    return merged


def render_rfd3_design_command(
    *,
    spec_path: Path,
    output_dir: Path,
    params: Mapping[str, Any],
) -> List[str]:
    n_batches, batch = resolve_rfd3_batching(params)
    step_scale = params.get("step_scale", DEFAULT_RFD3_PPI_STEP_SCALE)
    gamma_0 = params.get("gamma_0", DEFAULT_RFD3_PPI_GAMMA_0)
    cmd = [
        "rfd3",
        "design",
        f"out_dir={output_dir}",
        f"inputs={spec_path}",
        f"ckpt_path={params.get('rfd3_checkpoint') or params.get('ckpt_path') or FOUNDRY_ALIASES['rfd3']}",
        f"n_batches={n_batches}",
        f"diffusion_batch_size={batch}",
        f"skip_existing={_hydra_bool(params.get('skip_existing', False))}",
        f"prevalidate_inputs={_hydra_bool(params.get('prevalidate_inputs', True))}",
        f"inference_sampler.step_scale={step_scale}",
        f"inference_sampler.gamma_0={gamma_0}",
    ]
    if params.get("num_timesteps") is not None:
        cmd.append(f"inference_sampler.num_timesteps={int(params['num_timesteps'])}")
    if params.get("noise_scale") is not None:
        cmd.append(f"inference_sampler.noise_scale={params['noise_scale']}")
    if params.get("dump_trajectories") is not None:
        cmd.append(f"dump_trajectories={_hydra_bool(params.get('dump_trajectories'))}")
    if params.get("low_memory_mode") is not None:
        cmd.append(f"low_memory_mode={_hydra_bool(params.get('low_memory_mode'))}")
    if params.get("global_prefix"):
        cmd.append(f"global_prefix={params['global_prefix']}")
    return cmd


def render_mpnn_command(*, params: Mapping[str, Any], out_directory: Path, config_json: Optional[Path] = None, structure_path: Optional[Path] = None) -> List[str]:
    """Render Foundry ``mpnn`` for binder-only ProteinMPNN inverse folding."""
    cmd = [
        "mpnn",
        "--model_type",
        str(params.get("model_type") or DEFAULT_MPNN_MODEL_TYPE),
        "--is_legacy_weights",
        _mpnn_bool(params.get("is_legacy_weights", True)),
        "--checkpoint_path",
        str(params.get("mpnn_checkpoint") or params.get("checkpoint_path") or DEFAULT_CHECKPOINT_FILENAMES["mpnn"][0]),
        "--out_directory",
        str(out_directory),
        "--write_fasta",
        _mpnn_bool(params.get("write_fasta", True)),
        "--write_structures",
        _mpnn_bool(params.get("write_structures", True)),
    ]
    if config_json is not None:
        cmd += ["--config_json", str(config_json)]
        return cmd
    if structure_path is None:
        raise ValueError("mpnn command requires config_json or structure_path")
    cmd += ["--structure_path", str(structure_path)]
    designed = params.get("designed_chains") or params.get("binder_chain") or "A"
    if isinstance(designed, (list, tuple)):
        designed = ",".join(str(item).strip() for item in designed if str(item).strip())
    cmd += ["--designed_chains", str(designed)]
    batch = int(params.get("inverse_fold_num_sequences") or params.get("batch_size") or 1)
    cmd += ["--batch_size", str(max(1, batch))]
    cmd += ["--number_of_batches", str(max(1, int(params.get("number_of_batches") or 1)))]
    cmd += ["--temperature", str(params.get("temperature", DEFAULT_MPNN_TEMPERATURE))]
    if params.get("seed") is not None:
        cmd += ["--seed", str(int(params["seed"]))]
    return cmd


def render_rf3_fold_command(*, inputs: Path, output_dir: Path, params: Mapping[str, Any]) -> List[str]:
    cmd = [
        "rf3",
        "fold",
        f"inputs={inputs}",
        f"out_dir={output_dir}",
        f"ckpt_path={params.get('rf3_checkpoint') or params.get('folding_checkpoint') or FOUNDRY_ALIASES['rf3']}",
    ]
    if params.get("n_recycles") is not None:
        cmd.append(f"n_recycles={int(params['n_recycles'])}")
    if params.get("early_stopping_plddt_threshold") is not None:
        cmd.append(f"early_stopping_plddt_threshold={params['early_stopping_plddt_threshold']}")
    if params.get("num_steps") is not None:
        cmd.append(f"num_steps={int(params['num_steps'])}")
    if params.get("seed") is not None:
        cmd.append(f"seed={int(params['seed'])}")
    return cmd
