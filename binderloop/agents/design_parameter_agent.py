
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml

from binderloop.config import HarnessConfig
from binderloop.agents.model_input_spec import normalize_choice_flag

_USE_KERNELS_CHOICES = frozenset({"auto", "true", "false"})


@dataclass
class BoltzGenParameterPlan:
    """Parameter plan chosen by DesignParameterAgent for one BoltzGen design run."""

    protocol: str = "protein-anything"
    num_designs: int = 100
    budget: int = 30
    steps: List[str] = field(
        default_factory=lambda: ["design", "inverse_folding", "folding", "design_folding", "analysis", "filtering"]
    )
    run_filtering: bool = True
    keep_unfiltered_for_failure_analysis: bool = True
    diffusion_batch_size: Optional[int] = None
    design_checkpoints: Optional[List[str]] = None
    step_scale: Union[float, Optional[str]] = None
    noise_scale: Union[float, Optional[str]] = None
    devices: Optional[int] = None
    num_workers: int = 1
    use_kernels: str = "auto"
    inverse_fold_num_sequences: int = 1
    inverse_fold_avoid: Optional[str] = None
    alpha: Optional[float] = 0.001
    filter_biased: str = "true"
    refolding_rmsd_threshold: Optional[float] = 2.0
    metrics_override: List[str] = field(default_factory=list)
    additional_filters: List[str] = field(default_factory=list)
    size_buckets: List[str] = field(default_factory=list)
    cache: Optional[str] = None
    moldir: Optional[str] = None
    config_overrides: List[List[str]] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)

    def to_params(self) -> Dict[str, Any]:
        data = asdict(self)
        # Keep only non-empty values for cleaner manifests/commands.
        return {k: v for k, v in data.items() if v not in (None, [], {})}


class DesignParameterAgent:
    """Choose reasonable BoltzGen parameters from a compact harness config.

    The input YAML can stay concise, but this agent expands it into a richer,
    explicit parameter plan.  The choices are heuristic and auditable; callers can
    override any field through ``search_space.boltzgen``.
    """

    def choose_boltzgen_parameters(
        self,
        cfg: HarnessConfig,
        *,
        target_metadata: Optional[Mapping[str, Any]] = None,
        previous_feedback: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        target_metadata = dict(target_metadata or {})
        previous_feedback = dict(previous_feedback or {})
        overrides: Dict[str, Any] = dict(cfg.search_space.boltzgen or {})

        lengths = list(cfg.search_space.binder_lengths or [])
        max_len = max(lengths) if lengths else 100
        hotspots = list(cfg.target.hotspots or [])
        cap = max(1, int(cfg.search_space.max_binders_per_round or cfg.search_space.num_designs_per_round or 1))
        num_designs = min(cap, int(overrides.get("num_designs", cfg.search_space.num_designs_per_round or cap)))

        plan = BoltzGenParameterPlan()
        plan.num_designs = max(1, num_designs)
        plan.budget = int(overrides.get("budget", min(30, max(5, plan.num_designs // 4))))
        plan.devices = overrides.get("devices", max(1, int(getattr(cfg.resource, "host_gpu_num", 1) or 1)))
        plan.cache = overrides.get("cache")
        plan.moldir = overrides.get("moldir")

        # Protocol selection.
        binder_type = str(overrides.get("binder_type", target_metadata.get("binder_type", "protein"))).lower()
        has_small_molecule = bool(target_metadata.get("has_small_molecule") or overrides.get("has_small_molecule"))
        if "nanobody" in binder_type:
            plan.protocol = "nanobody-anything"
            plan.inverse_fold_avoid = overrides.get("inverse_fold_avoid", "C")
            plan.rationale.append("binder_type=nanobody -> use nanobody-anything and avoid free cysteines by default")
        elif "peptide" in binder_type or max_len <= 35:
            plan.protocol = "peptide-anything"
            plan.inverse_fold_avoid = overrides.get("inverse_fold_avoid", "C")
            plan.rationale.append("short peptide-like binder -> use peptide-anything")
        elif has_small_molecule:
            plan.protocol = "protein-small_molecule"
            plan.rationale.append("target context includes small molecule -> use protein-small_molecule")
        else:
            plan.protocol = "protein-anything"
            plan.inverse_fold_avoid = overrides.get("inverse_fold_avoid", "")
            plan.rationale.append("default protein binder setting -> use protein-anything")

        # Batch sizing. BoltzGen notes that too-large batches can collapse length sampling.
        if "diffusion_batch_size" in overrides:
            plan.diffusion_batch_size = overrides["diffusion_batch_size"]
            plan.rationale.append("diffusion_batch_size overridden by config")
        elif plan.num_designs < 100:
            plan.diffusion_batch_size = 1
            plan.rationale.append("num_designs < 100 -> batch size 1 for length/randomness diversity")
        elif max_len >= 130 or len(set(lengths)) > 4:
            plan.diffusion_batch_size = 4
            plan.rationale.append("large/varied length search -> moderate batch size to avoid same-length batches")
        else:
            plan.diffusion_batch_size = 10
            plan.rationale.append("standard search -> batch size 10")

        # Exploration/exploitation sampling knobs.
        exploration = float(cfg.active_learning.exploration_ratio)
        if "step_scale" in overrides:
            plan.step_scale = overrides["step_scale"]
        elif previous_feedback.get("diversity_collapse"):
            plan.step_scale = 1.8
            plan.rationale.append("feedback diversity_collapse -> increase step_scale")
        if "noise_scale" in overrides:
            plan.noise_scale = overrides["noise_scale"]
        elif exploration >= 0.4:
            plan.noise_scale = 1.0
            plan.rationale.append("high exploration_ratio -> keep/increase noise scale")

        # Inverse folding and filtering.
        plan.inverse_fold_num_sequences = int(overrides.get("inverse_fold_num_sequences", 1 if plan.num_designs >= 100 else 2))
        plan.refolding_rmsd_threshold = float(overrides.get("refolding_rmsd_threshold", 2.0 if max_len <= 120 else 2.5))
        plan.alpha = float(overrides.get("alpha", 0.001 if plan.protocol != "peptide-anything" else 0.01))
        plan.filter_biased = str(overrides.get("filter_biased", "true")).lower()
        if "use_kernels" in overrides:
            token = normalize_choice_flag(overrides["use_kernels"], _USE_KERNELS_CHOICES)
            if token is not None:
                plan.use_kernels = token
                plan.rationale.append(f"use_kernels overridden by config -> {token}")
        plan.run_filtering = bool(overrides.get("run_filtering", True))
        plan.keep_unfiltered_for_failure_analysis = bool(overrides.get("keep_unfiltered_for_failure_analysis", True))

        if len(hotspots) > 0:
            plan.config_overrides.append(["filtering", "filter_bindingsite=true"])
            plan.rationale.append("hotspots provided -> enable binding-site-aware filtering")
        if plan.run_filtering:
            if "additional_filters" in overrides:
                filters = overrides.get("additional_filters") or []
                if not isinstance(filters, (list, tuple)):
                    filters = [filters]
                plan.additional_filters = [str(item) for item in filters if str(item).strip()]
                if plan.additional_filters:
                    plan.rationale.append("user-configured additional_filters preserved for BoltzGen filtering")
            plan.rationale.append("run filtering but leave hard filter thresholds user-owned")
        else:
            plan.steps = ["design", "inverse_folding", "folding", "design_folding", "analysis"]
            plan.additional_filters = []
            plan.rationale.append("filtering disabled -> stop after analysis for failure-case capture")

        # User overrides win last. This lets config expose any BoltzGen CLI option supported below.
        merged = plan.to_params()
        for key, value in overrides.items():
            if key in {"binder_type", "has_small_molecule"}:
                continue
            merged[key] = value
        merged["num_designs"] = min(cap, max(1, int(merged.get("num_designs", plan.num_designs))))
        merged["num_designs_per_round"] = min(cap, max(1, int(merged.get("num_designs_per_round", merged["num_designs"]))))
        merged["max_binders_per_round"] = cap
        merged.setdefault("GPUName", getattr(cfg.resource, "gpu_name", "V100"))
        merged.setdefault("taiji_timeout", getattr(cfg.resource, "timeout_seconds", 3600))
        merged.setdefault("rationale", plan.rationale)
        if "use_kernels" in merged:
            token = normalize_choice_flag(merged["use_kernels"], _USE_KERNELS_CHOICES)
            if token is not None:
                merged["use_kernels"] = token
        return merged

    def write_parameter_plan(self, params: Mapping[str, Any], path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(dict(params), allow_unicode=True), encoding="utf-8")
        return path
