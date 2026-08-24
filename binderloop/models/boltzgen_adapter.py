
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Union, Optional
import yaml

from .base import DesignJob, ModelAdapter


# BoltzGen CLI flags whose argparse uses a fixed lowercase choice set. A Python
# bool would serialise to "True"/"False" via str() and be rejected with
# "invalid choice"; coerce booleans/strings to the canonical lowercase token.
_CHOICE_FLAGS = {"filter_biased", "use_kernels"}


def _cli_flag_value(key: str, value: Any) -> str:
    """Render a scalar CLI flag value, normalising boolean choice flags to
    lowercase string tokens that BoltzGen's argparse accepts."""
    if key in _CHOICE_FLAGS:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip().lower()
    return str(value)


def _residue_number(token: str) -> str:
    """Convert A:45 / A/45 / 45 into the residue index string used by BoltzGen specs."""
    token = str(token).strip()
    if ":" in token:
        return token.split(":", 1)[1]
    if "/" in token:
        return token.split("/", 1)[1]
    return token


def _tokens_to_range(tokens: List[str]) -> str:
    values = sorted({int(_residue_number(token)) for token in tokens})
    if not values:
        return ""
    groups=[]; start=previous=values[0]
    for value in values[1:]:
        if value == previous + 1: previous=value; continue
        groups.append(str(start) if start == previous else f"{start}..{previous}"); start=previous=value
    groups.append(str(start) if start == previous else f"{start}..{previous}")
    return ",".join(groups)


def _structure_redesign_template(params: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    template = params.get("binder_template")
    if not isinstance(template, Mapping):
        return None
    if str(template.get("mode")) != "structure_redesign":
        return None
    if not template.get("source_structure_file") or not template.get("binder_chain") or not template.get("fixed_res_index"):
        return None
    return template


def _build_redesign_schema(template: Mapping[str, Any], *, default_proximity: float = 8.0) -> Optional[Dict[str, Any]]:
    """Translate a structure_redesign binder_template into a BoltzGen redesign
    schema (``restrictions.not_design``).

    BoltzGen's ``parse_redesign_schema`` first marks every binder token as fixed
    (``not_design``), then re-enables design for tokens *outside* the
    ``within_proximity`` radius of the listed residues. The net effect is: the
    high-quality template fragment (and its local neighbourhood) is preserved,
    and only the remaining binder region is re-generated/re-folded. This schema
    must live in its own yaml file (the keys are NOT part of the main design-spec
    ``yaml_keys`` whitelist) and is injected via the inverse_fold config.
    """
    if not isinstance(template, Mapping):
        return None
    if str(template.get("mode")) != "structure_redesign":
        return None
    binder_chain = template.get("binder_chain")
    transform = dict(template.get("length_transform") or {})
    fixed_tokens = list(transform.get("fixed_residue_tokens") or [])
    mapped_range = _tokens_to_range(fixed_tokens) if fixed_tokens else ""
    fixed_res_index = mapped_range or template.get("inverse_fold_res_index") or template.get("fixed_res_index")
    if not binder_chain or not fixed_res_index:
        return None
    proximity = template.get("within_proximity", default_proximity)
    chain_id = template.get("id", binder_chain)
    return {
        "restrictions": {
            "not_design": [
                {
                    "chain": {
                        "binder": str(binder_chain),
                        "id": str(chain_id),
                        "res_index": str(fixed_res_index),
                        "within_proximity": float(proximity),
                    }
                }
            ]
        }
    }


DEFAULT_CHECKPOINT_FILENAMES = {
    "design_diverse": "boltzgen1_diverse.ckpt",
    "design_adherence": "boltzgen1_adherence.ckpt",
    "inverse_fold": "boltzgen1_ifold.ckpt",
    "folding": "boltz2_conf_final.ckpt",
    "affinity": "boltz2_aff.ckpt",
}
DEFAULT_MOLDIR_CACHE_RELATIVE = (
    "datasets--boltzgen--inference-data/"
    "snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"
)
DEFAULT_MOLDIR_RELATIVE = f"cache/{DEFAULT_MOLDIR_CACHE_RELATIVE}"


def _resolve_boltzgen_path(root: Path, value: Union[str, Path]) -> Path:
    text = str(value)
    if text.startswith("/"):
        return PurePosixPath(text)
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _resolve_local_artifact(root: Path, value: Any) -> str:
    text = str(value)
    return text if text.startswith("huggingface:") else str(_resolve_boltzgen_path(root, text))


def _resolve_local_artifact_list(root: Path, values: Any) -> List[str]:
    if values in (None, [], ()):
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [_resolve_local_artifact(root, value) for value in values]


def with_default_local_artifacts(
    params: Mapping[str, Any],
    boltzgen_root: Union[str, Path],
    *,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    moldir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Add local BoltzGen artifacts so CLI does not fall back to HuggingFace."""
    root = Path(boltzgen_root).expanduser()
    merged = dict(params or {})

    checkpoint_root = _resolve_boltzgen_path(root, merged.get("checkpoint_dir") or checkpoint_dir or "checkpoints")
    if merged.get("design_checkpoints"):
        merged["design_checkpoints"] = _resolve_local_artifact_list(root, merged["design_checkpoints"])
    else:
        merged["design_checkpoints"] = [
            str(checkpoint_root / DEFAULT_CHECKPOINT_FILENAMES["design_diverse"]),
            str(checkpoint_root / DEFAULT_CHECKPOINT_FILENAMES["design_adherence"]),
        ]
    for key, filename in [
        ("inverse_fold_checkpoint", DEFAULT_CHECKPOINT_FILENAMES["inverse_fold"]),
        ("folding_checkpoint", DEFAULT_CHECKPOINT_FILENAMES["folding"]),
        ("affinity_checkpoint", DEFAULT_CHECKPOINT_FILENAMES["affinity"]),
    ]:
        if merged.get(key):
            merged[key] = _resolve_local_artifact(root, merged[key])
        else:
            merged[key] = str(checkpoint_root / filename)

    cache_root = _resolve_boltzgen_path(root, merged.get("cache") or cache_dir or "cache")
    merged["cache"] = str(cache_root)
    if merged.get("moldir"):
        merged["moldir"] = _resolve_local_artifact(root, merged["moldir"])
    else:
        merged["moldir"] = str(_resolve_boltzgen_path(root, moldir) if moldir else cache_root / DEFAULT_MOLDIR_CACHE_RELATIVE)
    return merged


class BoltzGenAdapter(ModelAdapter):
    """Adapter for the local BoltzGen CLI.

    It writes a valid-looking BoltzGen design spec YAML, then emits the upstream
    `boltzgen run <spec> --output ...` command with configurable CLI flags.
    """

    name = "boltzgen"

    def __init__(self, root: str = "../boltzgen", python_bin: str = "python"):
        self.root = Path(root)
        self.python_bin = python_bin

    def write_design_spec(self, job: DesignJob) -> Path:
        out = Path(job.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        spec_path = out / "boltzgen_design_spec.yaml"

        params: Dict[str, Any] = dict(job.params or {})
        from binderloop.models.search_profile import get_model_search_profile
        params = get_model_search_profile(self.name).filter_params(params).params
        target_chain = params.get("target_chain", job.chain_id)
        binder_chain = params.get("binder_chain", "B")
        binder_sequence = params.get("binder_sequence", str(job.binder_length))
        template = _structure_redesign_template(params)
        if template is not None:
            binder_chain = str(template["binder_chain"])
        hotspot_indices = [_residue_number(h) for h in job.hotspots]
        binding = ",".join(hotspot_indices) if hotspot_indices else None

        include = params.get("target_include") or params.get("include")
        if include is None:
            if params.get("target_res_index"):
                include = [{"chain": {"id": target_chain, "res_index": params["target_res_index"]}}]
            else:
                include = [{"chain": {"id": target_chain}}]
        target_file: Dict[str, Any] = {
            "path": job.target_structure,
            "include": include,
        }
        binding_types = params.get("target_binding_types")
        if binding_types is not None:
            target_file["binding_types"] = binding_types
        elif binding:
            target_file["binding_types"] = [{"chain": {"id": target_chain, "binding": binding}}]
        if params.get("not_binding"):
            target_file.setdefault("binding_types", []).append(
                {"chain": {"id": target_chain, "not_binding": params["not_binding"]}}
            )
        if params.get("structure_groups"):
            target_file["structure_groups"] = params["structure_groups"]

        designed_protein: Dict[str, Any] = {
            "id": binder_chain,
            "sequence": binder_sequence,
        }
        if params.get("binder_binding_types"):
            designed_protein["binding_types"] = params["binder_binding_types"]
        if params.get("residue_constraints"):
            designed_protein["residue_constraints"] = params["residue_constraints"]
        if params.get("cyclic") is not None:
            designed_protein["cyclic"] = bool(params["cyclic"])

        spec: Dict[str, Any] = {
            "entities": [
                {"protein": designed_protein},
                {"file": target_file},
            ]
        }
        if template is not None:
            template_file: Dict[str, Any] = {
                "path": str(template["source_structure_file"]),
                "include": [{"chain": {"id": binder_chain}}],
                "design": [{"chain": {"id": binder_chain}}],
                "not_design": [
                    {
                        "chain": {
                            "id": binder_chain,
                            "res_index": str(_tokens_to_range(list((template.get("length_transform") or {}).get("fixed_residue_tokens") or [])) or template.get("backbone_fixed_res_index") or template["fixed_res_index"]),
                        }
                    }
                ],
            }
            template_groups = template.get("binder_structure_groups") or params.get("binder_structure_groups")
            if template_groups:
                template_file["structure_groups"] = template_groups
            insertions = template.get("design_insertions")
            if insertions:
                template_file["design_insertions"] = list(insertions)
            spec["entities"] = [
                {"file": template_file},
                {"file": target_file},
            ]
        if params.get("constraints"):
            spec["constraints"] = params["constraints"]
        if params.get("total_len"):
            spec.setdefault("constraints", []).append({"total_len": params["total_len"]})

        spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
        return spec_path

    def write_redesign_mask(self, job: DesignJob) -> Optional[Path]:
        """Write the structure-redesign mask yaml if a structure_redesign template
        is configured. Returns the file path, or None when not applicable."""
        params: Dict[str, Any] = dict(job.params or {})
        template = params.get("binder_template")
        proximity = params.get("binder_template_proximity", 8.0)
        schema = _build_redesign_schema(template or {}, default_proximity=float(proximity))
        if schema is None:
            return None
        out = Path(job.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        mask_path = out / "boltzgen_redesign_mask.yaml"
        mask_path.write_text(yaml.safe_dump(schema), encoding="utf-8")
        return mask_path

    def build_command(self, job: DesignJob) -> List[str]:
        from binderloop.models.boltzgen_renderer import render_boltzgen_command

        spec_path = self.write_design_spec(job)
        redesign_mask_path = self.write_redesign_mask(job)
        params = with_default_local_artifacts(job.params or {}, self.root)
        return render_boltzgen_command(
            spec_path=spec_path,
            output_dir=Path(job.output_dir),
            params=params,
            redesign_mask_path=redesign_mask_path.resolve() if redesign_mask_path is not None else None,
        )
