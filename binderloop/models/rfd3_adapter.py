"""Adapter that turns a harness DesignJob into official Foundry RFD3 artifacts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

from .base import DesignJob, ModelAdapter
from .rfd3_renderer import (
    DEFAULT_RFD3_PPI_GAMMA_0,
    DEFAULT_RFD3_PPI_STEP_SCALE,
    RFD3_STEPS,
    render_mpnn_command,
    render_rf3_fold_command,
    render_rfd3_design_command,
    with_default_foundry_artifacts,
)
from .rfd3_id_converter import adapt_rfd3_identifiers, write_conversion_report
from .rfd3_step_bridge import discover_design_structures, write_mpnn_config


_RANGE_TOKEN = re.compile(r"^(\d+)\s*(?:\.\.|-|–|—|:)\s*(\d+)$")


def _residue_number(token: str) -> str:
    token = str(token).strip()
    if ":" in token:
        return token.split(":", 1)[1]
    if "/" in token:
        return token.split("/", 1)[1]
    return token


def _hotspot_chain(token: str, default_chain: str) -> str:
    token = str(token).strip()
    if ":" in token:
        return token.split(":", 1)[0] or default_chain
    if "/" in token:
        return token.split("/", 1)[0] or default_chain
    return default_chain


def hotspot_to_rfd3_residue(token: str, default_chain: str) -> str:
    """Convert ``A:56`` / ``A/56`` / ``56`` into RFD3 InputSelection ``A56``."""
    chain = _hotspot_chain(token, default_chain)
    residue = _residue_number(token)
    residue = residue.replace("..", "-")
    if residue.upper().startswith(chain.upper()) and residue[len(chain):].lstrip(":-/").isdigit():
        return f"{chain}{residue[len(chain):].lstrip(':-/')}"
    return f"{chain}{residue}"


def _parse_res_index(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("..", "-").replace(":", "-")
    match = _RANGE_TOKEN.match(text)
    if match:
        lo, hi = int(match.group(1)), int(match.group(2))
        return f"{min(lo, hi)}-{max(lo, hi)}"
    if text.isdigit():
        return f"{text}-{text}"
    return text


def _range_from_target_include(include: Any, chain_id: str) -> Optional[str]:
    if not include:
        return None
    items = include if isinstance(include, list) else [include]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        chain = item.get("chain") if isinstance(item.get("chain"), Mapping) else item
        if not isinstance(chain, Mapping):
            continue
        cid = str(chain.get("id") or chain_id)
        if cid != chain_id:
            continue
        parsed = _parse_res_index(chain.get("res_index") or chain.get("residue_index"))
        if parsed:
            return parsed
    return None


def _range_from_structure(path: Union[str, Path], chain_id: str) -> Optional[str]:
    try:
        from binderloop.analysis.structure_features import parse_structure
    except Exception:
        return None
    try:
        atoms = parse_structure(path)
    except Exception:
        return None
    residues = sorted({atom.resseq for atom in atoms if atom.chain == chain_id})
    if not residues:
        residues = sorted({atom.resseq for atom in atoms})
    if not residues:
        return None
    return f"{residues[0]}-{residues[-1]}"


def build_binder_contig(
    *,
    binder_length: int,
    target_chain: str,
    params: Mapping[str, Any],
    target_structure: Optional[str] = None,
) -> str:
    if params.get("contig"):
        return str(params["contig"]).strip()
    target_range = (
        _parse_res_index(params.get("target_res_index"))
        or _range_from_target_include(params.get("target_include") or params.get("include"), target_chain)
    )
    if target_range is None and target_structure:
        target_range = _range_from_structure(target_structure, target_chain)
    if target_range is None:
        raise ValueError("RFD3 contig requires target_res_index, target_include, an explicit contig, or a parseable target structure")
    auto = bool(params.get("auto_binder_length"))
    length_range = params.get("binder_length_range")
    if auto and length_range not in (None, "", []):
        if isinstance(length_range, Mapping):
            lo = int(length_range.get("min", length_range.get("start", binder_length)))
            hi = int(length_range.get("max", length_range.get("end", binder_length)))
        elif isinstance(length_range, str):
            parsed = _parse_res_index(length_range) or f"{binder_length}-{binder_length}"
            lo, hi = (int(part) for part in parsed.split("-", 1))
        else:
            seq = list(length_range)
            lo = int(seq[0])
            hi = int(seq[1] if len(seq) > 1 else seq[0])
        binder_token = f"{min(lo, hi)}-{max(lo, hi)}"
    else:
        binder_token = str(int(binder_length))
    return f"{binder_token},/0,{target_chain}{target_range}"


def build_select_hotspots(hotspots: Sequence[str], *, default_chain: str, params: Mapping[str, Any]) -> Dict[str, str]:
    supplied = params.get("select_hotspots")
    if isinstance(supplied, Mapping) and supplied:
        return {str(key): str(value) for key, value in supplied.items()}
    if isinstance(supplied, str) and supplied.strip():
        return {token.strip(): "ALL" for token in supplied.split(",") if token.strip()}
    selected: Dict[str, str] = {}
    for token in hotspots or []:
        residue = hotspot_to_rfd3_residue(token, default_chain)
        selected[residue] = "ALL"
    if not selected:
        raise ValueError("RFD3 binder design requires hotspots or params.select_hotspots")
    return selected


def _normalized_steps(params: Mapping[str, Any]) -> List[str]:
    raw = params.get("steps") or list(RFD3_STEPS)
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    steps = [str(item).strip().lower() for item in raw if str(item).strip()]
    unknown = [step for step in steps if step not in RFD3_STEPS]
    if unknown:
        raise ValueError(f"unsupported RFD3 steps {unknown}; allowed={list(RFD3_STEPS)}")
    return steps or list(RFD3_STEPS)


def _bash_join(command: Sequence[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(token)) for token in command)


def render_rfd3_pipeline_script(
    *,
    spec_path: Path,
    design_dir: Path,
    inverse_fold_dir: Path,
    fold_dir: Path,
    metrics_path: Path,
    mpnn_config_path: Path,
    bridge_path: Path,
    params: Mapping[str, Any],
    conda_base: str = "/data/miniconda3",
    conda_env_name: str = "foundry",
    log_file: Optional[Path] = None,
) -> str:
    steps = _normalized_steps(params)
    design_cmd = render_rfd3_design_command(spec_path=spec_path, output_dir=design_dir, params=params)
    mpnn_cmd = render_mpnn_command(params=params, out_directory=inverse_fold_dir, config_json=mpnn_config_path)
    fold_cmd = render_rf3_fold_command(inputs=inverse_fold_dir, output_dir=fold_dir, params=params)
    params_json = json.dumps(
        {
            "designed_chains": params.get("designed_chains") or params.get("binder_chain") or "A",
            "binder_chain": params.get("binder_chain") or "A",
            "inverse_fold_num_sequences": params.get("inverse_fold_num_sequences") or params.get("batch_size") or 1,
            "temperature": params.get("temperature", 0.1),
            "mpnn_checkpoint": params.get("mpnn_checkpoint") or params.get("checkpoint_path"),
            "model_type": params.get("model_type") or "protein_mpnn",
            "is_legacy_weights": params.get("is_legacy_weights", True),
        },
        separators=(",", ":"),
    )
    log_redirect = f' >> "{log_file}" 2>&1' if log_file is not None else ""
    blocks = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'CONDA_BASE="${{CONDA_BASE:-{conda_base}}}"',
        f'CONDA_ENV_NAME="${{CONDA_ENV_NAME:-{conda_env_name}}}"',
        'if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then',
        '  # shellcheck disable=SC1091',
        '  source "$CONDA_BASE/etc/profile.d/conda.sh"',
        '  conda activate "$CONDA_ENV_NAME"',
        "fi",
        "for cli in rfd3 mpnn rf3; do",
        '  if ! command -v "$cli" >/dev/null 2>&1; then',
        '    echo "[HARNESS][ERROR] $cli CLI not found in conda env $CONDA_ENV_NAME" >&2',
        "    exit 127",
        "  fi",
        "done",
        f'mkdir -p "{design_dir}" "{inverse_fold_dir}" "{fold_dir}" "{Path(metrics_path).parent}" "{Path(mpnn_config_path).parent}"',
    ]
    if "design" in steps:
        blocks.append(f'echo "[HARNESS] rfd3 design"')
        blocks.append(_bash_join(design_cmd) + log_redirect)
    if "inverse_folding" in steps:
        blocks.append(f'echo "[HARNESS] assemble ProteinMPNN config_json"')
        blocks.append(
            _bash_join(
                [
                    "python",
                    str(bridge_path),
                    "assemble-mpnn",
                    "--design-dir",
                    str(design_dir),
                    "--out",
                    str(mpnn_config_path),
                    "--out-directory",
                    str(inverse_fold_dir),
                    "--params-json",
                    params_json,
                ]
            )
            + log_redirect
        )
        blocks.append(f'echo "[HARNESS] mpnn protein_mpnn"')
        blocks.append(_bash_join(mpnn_cmd) + log_redirect)
    if "folding" in steps:
        blocks.append(f'echo "[HARNESS] rf3 fold"')
        blocks.append(_bash_join(fold_cmd) + log_redirect)
        blocks.append(f'echo "[HARNESS] aggregate RF3 metrics"')
        rmsd = params.get("refolding_rmsd_threshold", 2.0)
        blocks.append(
            _bash_join(
                [
                    "python",
                    str(bridge_path),
                    "aggregate-metrics",
                    "--fold-dir",
                    str(fold_dir),
                    "--out",
                    str(metrics_path),
                    "--rmsd-threshold",
                    str(rmsd),
                ]
            )
            + log_redirect
        )
    blocks.append('echo "[HARNESS] rfd3 pipeline complete"')
    return "\n".join(blocks) + "\n"


class RFD3Adapter(ModelAdapter):
    """Write an official RFD3 YAML spec and a Foundry three-step run script."""

    name = "rfd3"

    def __init__(self, root: str = "../foundry", python_bin: str = "python"):
        self.root = Path(root)
        self.python_bin = python_bin

    def write_design_spec(self, job: DesignJob) -> Path:
        out = Path(job.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        spec_path = out / "rfd3_design_spec.yaml"
        from binderloop.models.search_profile import get_model_search_profile
        params = get_model_search_profile(self.name).filter_params(job.params).params
        target_chain = str(params.get("target_chain") or job.chain_id or "A")
        target_path = params.get("target_path_for_spec") or job.target_structure
        target_path = str(Path(target_path).expanduser())
        if not target_path.startswith("/") and params.get("target_path_for_spec") is None:
            resolved = Path(job.target_structure).expanduser()
            target_path = str(resolved.resolve()) if resolved.exists() else str(resolved)
        convert_ids = params.get("rfd3_convert_residue_ids", True)
        if isinstance(convert_ids, str):
            convert_ids = convert_ids.strip().lower() not in {"0", "false", "no", "off"}
        if convert_ids:
            adapted = adapt_rfd3_identifiers(
                target_path,
                chain_id=target_chain,
                res_index=params.get("target_res_index"),
                target_include=params.get("target_include") or params.get("include"),
                contig=params.get("contig"),
                hotspots=job.hotspots,
                select_hotspots=params.get("select_hotspots") if isinstance(params.get("select_hotspots"), Mapping) else None,
                source_scheme=params.get("residue_id_scheme") or params.get("rfd3_source_id_scheme") or "auto",
                target_scheme=params.get("rfd3_residue_scheme") or "native",
                adapt_structure=bool(params.get("rfd3_adapt_structure", False)),
                output_dir=out,
            )
            write_conversion_report(adapted, out / "rfd3_id_conversion.json")
            target_path = adapted.structure_path
            target_chain = adapted.chain_id or target_chain
            if adapted.res_index:
                params["target_res_index"] = adapted.res_index
            params["target_chain"] = target_chain
            if adapted.select_hotspots:
                params["select_hotspots"] = adapted.select_hotspots
            if adapted.contig_target and params.get("contig"):
                prefix, sep, _tail = str(params["contig"]).partition("/0,")
                params["contig"] = f"{prefix}{sep}{adapted.contig_target}" if sep else adapted.contig_target
            hotspot_tokens = adapted.hotspots or list(job.hotspots)
        else:
            hotspot_tokens = list(job.hotspots)
        contig = build_binder_contig(
            binder_length=int(job.binder_length),
            target_chain=target_chain,
            params=params,
            target_structure=target_path,
        )
        hotspots = build_select_hotspots(hotspot_tokens, default_chain=target_chain, params=params)
        example_name = str(params.get("sample_name") or params.get("task_id") or job.job_id or "binder")
        example_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", example_name) or "binder"
        spec_body: Dict[str, Any] = {
            "dialect": int(params.get("dialect", 2)),
            "input": target_path,
            "contig": contig,
            "select_hotspots": hotspots,
            "infer_ori_strategy": str(params.get("infer_ori_strategy") or "hotspots"),
            "is_non_loopy": bool(params.get("is_non_loopy", True)),
            "redesign_motif_sidechains": bool(params.get("redesign_motif_sidechains", False)),
        }
        for key in ("select_hbond_donor", "select_hbond_acceptor", "select_fixed_atoms", "ori_token", "length", "ligand"):
            if params.get(key) not in (None, "", {}):
                spec_body[key] = params[key]
        spec = {example_name: spec_body}
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        return spec_path

    def output_layout(self, job: DesignJob) -> Dict[str, Path]:
        root = Path(job.output_dir)
        return {
            "root": root,
            "spec": root / "rfd3_design_spec.yaml",
            "script": root / "run_rfd3_pipeline.sh",
            "bridge": root / "rfd3_step_bridge.py",
            "design": root / "design",
            "inverse_fold": root / "inverse_fold",
            "fold": root / "fold",
            "mpnn_config": root / "mpnn_inputs.json",
            "metrics": root / "final_designs_metrics.csv",
            "log": root / "rfd3_pipeline.log",
        }

    def _stage_bridge(self, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).with_name("rfd3_step_bridge.py"), dest)
        return dest

    def write_pipeline_script(self, job: DesignJob) -> Path:
        layout = self.output_layout(job)
        params = with_default_foundry_artifacts(job.params or {})
        self._stage_bridge(layout["bridge"])
        script = render_rfd3_pipeline_script(
            spec_path=layout["spec"],
            design_dir=layout["design"],
            inverse_fold_dir=layout["inverse_fold"],
            fold_dir=layout["fold"],
            metrics_path=layout["metrics"],
            mpnn_config_path=layout["mpnn_config"],
            bridge_path=layout["bridge"],
            params=params,
            conda_base=str(params.get("conda_base") or "/data/miniconda3"),
            conda_env_name=str(params.get("conda_env_name") or "foundry"),
            log_file=layout["log"],
        )
        layout["script"].write_text(script, encoding="utf-8")
        layout["script"].chmod(layout["script"].stat().st_mode | 0o111)
        return layout["script"]

    def build_step_command(self, job: DesignJob, step: str) -> List[str]:
        step = str(step or "").strip().lower()
        if step not in RFD3_STEPS:
            raise ValueError(f"unsupported RFD3 step {step!r}; allowed={list(RFD3_STEPS)}")
        layout = self.output_layout(job)
        params = with_default_foundry_artifacts(job.params or {})
        spec_path = self.write_design_spec(job) if step == "design" or not layout["spec"].exists() else layout["spec"]
        layout["design"].mkdir(parents=True, exist_ok=True)
        layout["inverse_fold"].mkdir(parents=True, exist_ok=True)
        layout["fold"].mkdir(parents=True, exist_ok=True)
        if step == "design":
            return render_rfd3_design_command(spec_path=spec_path, output_dir=layout["design"], params=params)
        if step == "inverse_folding":
            structures = discover_design_structures(layout["design"])
            write_mpnn_config(
                structures,
                out_path=layout["mpnn_config"],
                out_directory=layout["inverse_fold"],
                params=params,
            )
            return render_mpnn_command(params=params, out_directory=layout["inverse_fold"], config_json=layout["mpnn_config"])
        return render_rf3_fold_command(inputs=layout["inverse_fold"], output_dir=layout["fold"], params=params)

    def build_command(self, job: DesignJob) -> List[str]:
        self.write_design_spec(job)
        script = self.write_pipeline_script(job)
        return ["bash", str(script)]

    def expected_outputs(self, job: DesignJob) -> Dict[str, str]:
        layout = self.output_layout(job)
        return {key: str(path) for key, path in layout.items()}
