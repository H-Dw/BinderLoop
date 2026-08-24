"""Single command renderer for all BoltzGen execution paths."""
from pathlib import Path
from typing import Any, List, Mapping, Optional

from binderloop.agents.model_input_spec import normalize_additional_filters
from binderloop.models.boltzgen_adapter import _cli_flag_value


def _required_positive_int(params: Mapping[str, Any], key: str) -> int:
    if key not in params or isinstance(params[key], bool):
        raise ValueError(f"resolved BoltzGen params require {key}")
    value = int(params[key])
    if value < 1:
        raise ValueError(f"resolved BoltzGen params require {key} >= 1")
    return value


def render_boltzgen_command(*, spec_path: Path, output_dir: Path, params: Mapping[str, Any], redesign_mask_path: Optional[Path] = None) -> List[str]:
    num_designs = _required_positive_int(params, "num_designs")
    budget = _required_positive_int(params, "budget")
    cmd = ["boltzgen", "run", str(spec_path), "--output", str(output_dir), "--protocol", str(params.get("protocol", "protein-anything")), "--num_designs", str(num_designs), "--budget", str(budget)]
    for key in ("diffusion_batch_size", "step_scale", "noise_scale", "inverse_fold_num_sequences", "alpha", "refolding_rmsd_threshold", "devices", "num_workers", "filter_biased", "use_kernels", "cache", "moldir", "inverse_fold_checkpoint", "folding_checkpoint", "affinity_checkpoint"):
        if params.get(key) is not None:
            cmd += [f"--{key}", _cli_flag_value(key, params[key])]
    if params.get("inverse_fold_avoid"):
        cmd += ["--inverse_fold_avoid", str(params["inverse_fold_avoid"])]
    for key in ("design_checkpoints", "metrics_override", "size_buckets"):
        values = params.get(key)
        if values:
            cmd += [f"--{key}", *map(str, values if isinstance(values, (list, tuple)) else [values])]
    for key in ("reuse", "skip_inverse_folding", "only_inverse_fold"):
        if params.get(key): cmd.append(f"--{key}")
    filters = normalize_additional_filters(params.get("additional_filters"))
    if filters: cmd += ["--additional_filters", *filters]
    for override in params.get("config_overrides", []) or []:
        cmd += ["--config", *map(str, override)] if isinstance(override, (list, tuple)) else ["--config", str(override)]
    if redesign_mask_path is not None:
        cmd += ["--config", "inverse_folding", f"data.cfg.design_mask_override={redesign_mask_path}"]
    return cmd
