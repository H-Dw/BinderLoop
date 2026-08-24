
from pathlib import Path
from typing import List
import json

from .base import DesignJob, ModelAdapter


def _hotspot_to_odesign(token: str, default_chain: str) -> str:
    token = str(token).strip()
    if "/" in token:
        return token
    if ":" in token:
        c, r = token.split(":", 1)
        return f"{c}/{r}"
    return f"{default_chain}/{token}"


class ODesignAdapter(ModelAdapter):
    """Adapter for local ODesign Hydra inference.

    It writes an ODesign input JSON, then emits a command equivalent to the
    repository's inference_demo.sh with Hydra overrides.
    """

    name = "odesign"

    def __init__(self, root: str = "../ODesign", python_bin: str = "python"):
        self.root = Path(root)
        self.python_bin = python_bin

    def write_input_json(self, job: DesignJob) -> Path:
        out = Path(job.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        params = dict(job.params or {})
        target_chain = params.get("target_chain", job.chain_id)
        binder_chain = params.get("binder_chain", "A")
        target_range = params.get("target_range", f"{target_chain}")
        # ODesign examples specify generated binder as a length range like "64-64".
        binder_sequence = params.get("binder_sequence", f"{job.binder_length}-{job.binder_length}")
        hotspots = ",".join(_hotspot_to_odesign(h, target_chain) for h in job.hotspots)
        case = {
            "name": params.get("sample_name", job.job_id),
            "ref_file": job.target_structure,
            "chains": [
                {"chain_type": "proteinChain", "sequence": target_range},
                {"chain_type": "proteinChain", "sequence": binder_sequence},
            ],
            "hotspot": hotspots,
            "center_method": params.get("center_method", "hotspot_center"),
        }
        if params.get("motif_scaffolding") is not None:
            case["motif_scaffolding"] = bool(params["motif_scaffolding"])
        if params.get("partial_diff"):
            case["partial_diff"] = params["partial_diff"]
        cfg_path = out / "odesign_input.json"
        cfg_path.write_text(json.dumps([case], ensure_ascii=False, indent=2), encoding="utf-8")
        return cfg_path

    def build_command(self, job: DesignJob) -> List[str]:
        input_json = self.write_input_json(job)
        params = dict(job.params or {})
        script = self.root / "scripts" / "inference.py"
        infer_model_name = params.get("infer_model_name", "odesign_base_prot_flex")
        design_modality = params.get("design_modality", "protein")
        exp_name = params.get("exp_name", job.job_id)
        seeds = params.get("seeds", [job.seed])
        if isinstance(seeds, str):
            seed_arg = seeds
        else:
            seed_arg = "[" + ",".join(str(s) for s in seeds) + "]"
        cmd = [
            self.python_bin,
            str(script),
            f"exp=train_{infer_model_name}",
            f"data_root_dir={params.get('data_root_dir', './data')}",
            f"ckpt_root_dir={params.get('ckpt_root_dir', './ckpt')}",
            f"exp.infer_model_name={infer_model_name}",
            f"exp.design_modality={design_modality}",
            f"exp.input_json_path={input_json}",
            f"exp.exp_name={exp_name}",
            f"exp.seeds={seed_arg}",
            f"exp.model.sample_diffusion.N_sample={params.get('N_sample', params.get('num_samples', 5))}",
            f"exp.use_msa={str(params.get('use_msa', False)).lower()}",
            f"exp.num_workers={params.get('num_workers', 4)}",
            f"exp.invfold_topk={params.get('invfold_topk', 1)}",
            f"exp.invfold_temp={params.get('invfold_temp', 1.0)}",
            f"exp.invfold_use_beam={str(params.get('invfold_use_beam', True)).lower()}",
        ]
        if "N_step" in params:
            cmd.append(f"exp.model.sample_diffusion.N_step={params['N_step']}")
        if params.get("partial_diffusion_enable") is not None:
            cmd.append(
                "exp.model.inference_noise_schedulers.coordinate.partial_diffusion.enable="
                + str(params.get("partial_diffusion_enable")).lower()
            )
        if params.get("partial_diffusion_snr") is not None:
            cmd.append(
                "exp.model.inference_noise_schedulers.coordinate.partial_diffusion.snr="
                + str(params.get("partial_diffusion_snr"))
            )
        return cmd

    # Backward-compatible alias for older tests.
    write_inference_config = write_input_json
