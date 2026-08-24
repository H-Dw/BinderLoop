"""Per-backbone search profiles: sampler catalogs, isolation, and tool bindings.

Job params must not be a flat merge of BoltzGen ∪ RFD3. Each profile declares
the keys it may execute, the keys it must strip, and the default sequence/refold
tools. Mixed GPU DAGs are rejected until a later pipeline composes tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from binderloop.agents.config_parameter_contract import (
    ADJUSTABLE_CONFIG_PARAMETERS,
    ALL_EXECUTABLE_CONFIG_KEYS,
    INTERNAL_ONLY_CONFIG_KEYS,
    PARAM_BOUNDS,
    PUBLIC_AGENT_CONFIG_KEYS,
    USER_OWNED_EXECUTABLE_CONFIG_KEYS,
    _BOLTZGEN_INTERNAL_METADATA_KEYS,
    _BOLTZGEN_RUNTIME_KEYS,
)
from binderloop.models.refolding import (
    RefoldTool,
    RefoldToolError,
    get_refold_tool,
    normalize_refold_tool_name,
)
from binderloop.models.sequence import (
    SequenceTool,
    SequenceToolError,
    get_sequence_tool,
    normalize_sequence_tool_name,
)


class SearchProfileError(ValueError):
    """Invalid backbone/tool pairing or isolated-parameter contract."""


BOLTZGEN_ONLY_KEYS = frozenset({
    "alpha", "protocol", "filter_biased", "additional_filters", "config_overrides",
    "use_kernels", "binder_template", "binder_templates", "binder_template_proximity",
    "epitope_crop_mode", "inverse_fold_avoid", "inverse_fold_checkpoint",
    "affinity_checkpoint", "moldir", "cache", "design_checkpoints",
    "metrics_override", "size_buckets", "skip_inverse_folding", "only_inverse_fold",
    "reuse", "target_binding_types", "structure_groups", "binder_structure_groups",
    "binder_binding_types", "residue_constraints", "cyclic", "constraints", "total_len",
    "fragment_templates_enabled", "fragment_template_top_k", "fragment_template_gate",
    "fragment_interchain_pae_max", "template_conditioned_fraction",
    "fragment_template_min_quality", "fragment_template_max_templates",
    "fragment_template_library_size", "fragment_template_max_fixed_fraction",
    "fragment_template_min_designable_residues", "fragment_template_min_alignment_coverage",
    "fragment_template_max_target_patch_rmsd", "fragment_template_require_pae",
    "fragment_template_package_failure_policy", "fragment_template_utility_decay",
    "fragment_template_cooldown_failures", "fragment_template_blacklist_failures",
})

RFD3_ONLY_KEYS = frozenset({
    "gamma_0", "is_non_loopy", "infer_ori_strategy", "n_batches", "contig", "dialect",
    "designed_chains", "select_hotspots", "select_hbond_donor", "select_hbond_acceptor",
    "is_legacy_weights", "temperature", "residue_id_scheme", "rfd3_source_id_scheme",
    "rfd3_residue_scheme", "rfd3_adapt_structure", "rfd3_convert_residue_ids",
    "n_recycles", "ckpt_path", "rfd3_checkpoint", "mpnn_checkpoint", "rf3_checkpoint",
    "model_type", "sample_name", "redesign_motif_sidechains", "skip_existing",
    "prevalidate_inputs", "dump_trajectories", "low_memory_mode", "global_prefix",
    "early_stopping_plddt_threshold", "number_of_batches", "write_fasta",
    "write_structures", "batch_size", "target_res_index",
})

SHARED_JOB_KEYS = (
    frozenset({
        "task_id", "num_designs", "num_designs_per_round", "max_binders_per_round",
        "binder_lengths", "binder_length_range", "binder_length_step", "binder_length",
        "binder_chain", "target_chain", "target_include", "include", "steps",
        "diffusion_batch_size", "step_scale", "noise_scale", "inverse_fold_num_sequences",
        "refolding_rmsd_threshold", "budget", "run_filtering",
        "keep_unfiltered_for_failure_analysis", "auto_binder_length",
        "auxiliary_hotspots", "hotspots", "checkpoint_dir", "weights_path",
        "conda_env_name", "conda_base", "folding_checkpoint",
        "sequence_tool", "refold_tool", "sequence_policy", "sampler_policy",
        "binding_site_policy", "target_context_policy", "selection_policy",
        "diversity_collapse", "search_profile_model", "search_profile_stripped_keys",
    })
    | _BOLTZGEN_RUNTIME_KEYS
    | _BOLTZGEN_INTERNAL_METADATA_KEYS
    | INTERNAL_ONLY_CONFIG_KEYS
    | USER_OWNED_EXECUTABLE_CONFIG_KEYS
)

BOLTZGEN_RESTORE_KEYS = frozenset({
    "hotspot_weight", "budget", "protocol", "diffusion_batch_size",
    "step_scale", "noise_scale", "inverse_fold_num_sequences", "inverse_fold_avoid",
    "alpha", "refolding_rmsd_threshold", "filter_biased", "steps", "analysis_location",
    "num_workers", "use_kernels", "run_filtering", "keep_unfiltered_for_failure_analysis",
    "additional_filters", "config_overrides", "clash_filter", "target_include",
    "target_binding_types", "structure_groups", "binder_chain", "binder_structure_prior",
    "residue_constraints", "binder_binding_types", "length_delta_hint",
    "avoid_binder_lengths", "prioritize_hotspots", "auxiliary_hotspots",
    "exploit_fragment_modules", "module_guided_exploitation", "module_guided_repair",
    "epitope_crop_mode", "auto_binder_length", "fragment_template_gate",
    "fragment_interchain_pae_max", "fragment_templates_enabled",
    "fragment_template_top_k", "binder_template", "binder_templates",
    "binder_template_proximity", "template_conditioned_fraction",
})

RFD3_RESTORE_KEYS = frozenset({
    "budget", "diffusion_batch_size", "step_scale", "noise_scale", "gamma_0",
    "n_batches", "inverse_fold_num_sequences", "temperature", "refolding_rmsd_threshold",
    "steps", "run_filtering", "target_include", "binder_chain", "target_chain",
    "target_res_index", "contig", "select_hotspots", "is_non_loopy",
    "infer_ori_strategy", "designed_chains", "model_type", "is_legacy_weights",
    "residue_id_scheme", "rfd3_residue_scheme", "auxiliary_hotspots",
    "n_recycles", "dialect", "auto_binder_length",
})

RFD3_PARAM_BOUNDS: Dict[str, Dict[str, Any]] = {
    "step_scale": {"min": 1.0, "max": 4.0, "default": 3.0, "max_step_abs": 1.0, "log_scale": False},
    "gamma_0": {"min": 0.0, "max": 1.0, "default": 0.2, "max_step_abs": 0.2, "log_scale": False},
    "noise_scale": {"min": 0.0, "max": 2.0, "default": 1.0, "max_step_abs": 0.3, "log_scale": False},
    "exploration_ratio": dict(PARAM_BOUNDS["exploration_ratio"]),
}

RFD3_ADJUSTABLE_PARAMETERS: Dict[str, str] = {
    "diffusion_batch_size": "RFD3 diffusion batch size (official PPI default 8).",
    "step_scale": "RFD3 inference_sampler.step_scale. PPI recipe is 3.0, not BoltzGen 0.6-1.0.",
    "gamma_0": "RFD3 inference_sampler.gamma_0. Official PPI default is 0.2.",
    "noise_scale": "Optional RFD3 inference_sampler.noise_scale.",
    "temperature": "ProteinMPNN sampling temperature.",
    "auxiliary_hotspots": "Harness-translated nearby target residues (add-only).",
}

BOLTZGEN_SAMPLER_AXES = ("alpha", "noise_scale", "step_scale")
RFD3_SAMPLER_AXES = ("step_scale", "gamma_0")
RFD3_DEFAULT_STEP_SCALE_CANDIDATES = (1.5, 2.5, 3.0, 3.5)
RFD3_DEFAULT_GAMMA_0_CANDIDATES = (0.1, 0.2, 0.4, 0.6)

CANONICAL_ARMS = frozenset({
    "baseline_hold", "site_primary_condition", "site_expanded_condition",
    "site_negative_exclusion", "target_context_focus", "sampler_explore",
    "template_exploit", "sequence_repair",
})


@dataclass(frozen=True)
class FilterResult:
    params: Dict[str, Any]
    stripped: Tuple[str, ...]


@dataclass
class ModelSearchProfile:
    model: str
    sampler_axes: Tuple[str, ...]
    param_bounds: Dict[str, Dict[str, Any]]
    search_space_attr: str
    restore_keys: FrozenSet[str]
    supported_arms: FrozenSet[str]
    backbone_keys: FrozenSet[str]
    forbidden_keys: FrozenSet[str]
    default_sequence_tool: str
    default_refold_tool: str
    compatible_sequence_tools: FrozenSet[str]
    compatible_refold_tools: FrozenSet[str]
    adjustable_parameters: Dict[str, str] = field(default_factory=dict)
    sequence_tool: Optional[SequenceTool] = None
    refold_tool: Optional[RefoldTool] = None

    def bound_tools(self, sequence: SequenceTool, refold: RefoldTool) -> "ModelSearchProfile":
        return replace(self, sequence_tool=sequence, refold_tool=refold)

    def resolved_sequence_tool(self) -> SequenceTool:
        return self.sequence_tool or get_sequence_tool(self.default_sequence_tool)

    def resolved_refold_tool(self) -> RefoldTool:
        return self.refold_tool or get_refold_tool(self.default_refold_tool)

    def executable_keys(self, *, include_internal: bool = False) -> FrozenSet[str]:
        seq = self.resolved_sequence_tool()
        fold = self.resolved_refold_tool()
        public = frozenset(self.adjustable_parameters) | frozenset({"binder_lengths"})
        if not include_internal:
            return public
        return (
            (ALL_EXECUTABLE_CONFIG_KEYS | self.backbone_keys | seq.allowed_keys | fold.allowed_keys | SHARED_JOB_KEYS)
            - self.forbidden_keys
        )

    def filter_params(self, params: Optional[Mapping[str, Any]]) -> FilterResult:
        seq = self.resolved_sequence_tool()
        fold = self.resolved_refold_tool()
        forbidden = self.forbidden_keys | seq.forbidden_keys | fold.forbidden_keys
        allowed = self.backbone_keys | seq.allowed_keys | fold.allowed_keys | SHARED_JOB_KEYS | frozenset(self.sampler_axes)
        stripped: List[str] = []
        out: Dict[str, Any] = {}
        for key, value in dict(params or {}).items():
            if key in forbidden:
                stripped.append(str(key))
                continue
            if key in allowed or key not in (BOLTZGEN_ONLY_KEYS | RFD3_ONLY_KEYS):
                out[key] = value
            else:
                stripped.append(str(key))
        out["sequence_tool"] = seq.name
        out["refold_tool"] = fold.name
        return FilterResult(params=out, stripped=tuple(sorted(set(stripped))))

    def load_search_space(self, cfg: Any) -> Dict[str, Any]:
        block = dict(getattr(getattr(cfg, "search_space", None), self.search_space_attr, None) or {})
        return dict(block)

    def materialize_sampler(
        self,
        params: Mapping[str, Any],
        *,
        final_state: Optional[Mapping[str, Any]] = None,
        catalog_axes: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        updated = dict(params or {})
        state = dict(final_state or updated.get("final_parameter_state") or {})
        axes = set(self.sampler_axes)
        if state:
            for key, value in state.items():
                if key not in axes:
                    raise SearchProfileError(f"final_parameter_state contains unsupported key for {self.model}: {key}")
                exact = float(value)
                if catalog_axes is not None:
                    allowed = {float(item) for item in catalog_axes.get(key) or ()}
                    if exact not in allowed:
                        raise SearchProfileError(f"{key} final value is not an exact catalog member")
                if key in updated and updated.get(key) not in (None, "") and float(updated[key]) != exact and not updated.get("random_sampler_fallback"):
                    raise SearchProfileError(f"{key} differs from immutable final_parameter_state")
                updated[key] = exact
            updated["final_parameter_state"] = {key: float(state[key]) for key in state if key in axes}
            updated["sampler_policy_applied"] = True
            updated["sampler_policy_status"] = str(updated.get("sampler_policy_status") or "applied:final_probabilistic_state")
        if updated.get("diversity_collapse") and str(updated.get("sampler_policy") or "") == "explore":
            updated["diffusion_batch_size"] = 1
        for key in list(updated):
            if key in (BOLTZGEN_ONLY_KEYS | RFD3_ONLY_KEYS) and key not in axes and key not in self.backbone_keys:
                if key in self.forbidden_keys:
                    updated.pop(key, None)
        return updated

    def materialize_sequence(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        updated = dict(params or {})
        policy = str(updated.get("sequence_policy") or "")
        if policy != "repair":
            return updated
        update = self.resolved_sequence_tool().materialize(policy, updated)
        updated.update(update)
        return updated

    def materialize_site(self, job: Any) -> Any:
        params = dict(job.params or {})
        policy = str(params.get("binding_site_policy") or "")
        if self.model != "rfd3" or policy != "primary_expanded":
            job.params = params
            return job
        extra = list(params.get("expanded_binding_residues") or params.get("auxiliary_hotspots") or [])
        if not extra:
            job.params = params
            return job
        from binderloop.models.rfd3_adapter import hotspot_to_rfd3_residue
        from binderloop.models.rfd3_id_converter import adapt_rfd3_identifiers
        supplied = dict(params.get("select_hotspots") or {}) if isinstance(params.get("select_hotspots"), Mapping) else {}
        chain = str(params.get("target_chain") or getattr(job, "chain_id", None) or "A")
        structure = str(getattr(job, "target_structure", None) or params.get("target_structure") or "")
        converted_extra: List[str] = []
        try:
            adapted = adapt_rfd3_identifiers(
                structure,
                chain_id=chain,
                hotspots=[str(token) for token in extra],
                source_scheme=params.get("residue_id_scheme") or params.get("rfd3_source_id_scheme") or "auto",
                target_scheme=params.get("rfd3_residue_scheme") or "native",
            )
            converted_extra = [str(token) for token in (adapted.hotspots or []) if str(token).strip()]
            if adapted.notes:
                params["rfd3_site_id_notes"] = list(adapted.notes)
        except Exception:
            converted_extra = []
        if not converted_extra:
            converted_extra = [hotspot_to_rfd3_residue(str(token), chain) for token in extra]
        normalized: List[str] = []
        for residue in converted_extra:
            token = str(residue).strip()
            if ":" in token or "/" in token:
                token = hotspot_to_rfd3_residue(token, chain)
            if token:
                normalized.append(token)
        for residue in normalized:
            supplied.setdefault(residue, "ALL")
        params["select_hotspots"] = supplied
        job.params = params
        return job

    def ingest(self, output_dir: Any, **kwargs: Any) -> Any:
        return self.resolved_refold_tool().ingest(output_dir, **kwargs)


def _boltzgen_profile() -> ModelSearchProfile:
    return ModelSearchProfile(
        model="boltzgen",
        sampler_axes=BOLTZGEN_SAMPLER_AXES,
        param_bounds={key: dict(value) for key, value in PARAM_BOUNDS.items()},
        search_space_attr="boltzgen",
        restore_keys=BOLTZGEN_RESTORE_KEYS,
        supported_arms=CANONICAL_ARMS,
        backbone_keys=ALL_EXECUTABLE_CONFIG_KEYS - RFD3_ONLY_KEYS,
        forbidden_keys=RFD3_ONLY_KEYS,
        default_sequence_tool="boltz_ifold",
        default_refold_tool="boltz2",
        compatible_sequence_tools=frozenset({"boltz_ifold"}),
        compatible_refold_tools=frozenset({"boltz2"}),
        adjustable_parameters=dict(ADJUSTABLE_CONFIG_PARAMETERS),
    )


def _rfd3_profile() -> ModelSearchProfile:
    backbone = frozenset({
        "steps", "diffusion_batch_size", "n_batches", "ckpt_path", "rfd3_checkpoint",
        "checkpoint_dir", "weights_path", "infer_ori_strategy", "is_non_loopy",
        "redesign_motif_sidechains", "dialect", "step_scale", "gamma_0",
        "num_timesteps", "noise_scale", "skip_existing", "prevalidate_inputs",
        "dump_trajectories", "low_memory_mode", "global_prefix", "auto_binder_length",
        "binder_chain", "target_chain", "target_res_index", "contig", "select_hotspots",
        "select_hbond_donor", "select_hbond_acceptor", "sample_name",
        "residue_id_scheme", "rfd3_source_id_scheme", "rfd3_residue_scheme",
        "rfd3_adapt_structure", "rfd3_convert_residue_ids", "num_designs", "budget",
        "target_include",
    })
    return ModelSearchProfile(
        model="rfd3",
        sampler_axes=RFD3_SAMPLER_AXES,
        param_bounds={key: dict(value) for key, value in RFD3_PARAM_BOUNDS.items()},
        search_space_attr="rfd3",
        restore_keys=RFD3_RESTORE_KEYS,
        supported_arms=CANONICAL_ARMS - frozenset({"template_exploit"}),
        backbone_keys=backbone,
        forbidden_keys=BOLTZGEN_ONLY_KEYS,
        default_sequence_tool="protein_mpnn",
        default_refold_tool="rf3",
        compatible_sequence_tools=frozenset({"protein_mpnn"}),
        compatible_refold_tools=frozenset({"rf3"}),
        adjustable_parameters=dict(RFD3_ADJUSTABLE_PARAMETERS),
    )


_PROFILE_BUILDERS = {
    "boltzgen": _boltzgen_profile,
    "rfd3": _rfd3_profile,
}


def resolve_sequence_tool(cfg: Any, profile: ModelSearchProfile) -> SequenceTool:
    requested = getattr(getattr(cfg, "sequence", None), "tool", None) if cfg is not None else None
    name = normalize_sequence_tool_name(requested) or profile.default_sequence_tool
    if name not in profile.compatible_sequence_tools:
        raise SearchProfileError(
            f"sequence.tool {name!r} is incompatible with backbone {profile.model}; "
            f"allowed={sorted(profile.compatible_sequence_tools)}"
        )
    try:
        return get_sequence_tool(name)
    except SequenceToolError as exc:
        raise SearchProfileError(str(exc)) from exc


def resolve_refold_tool(cfg: Any, profile: ModelSearchProfile) -> RefoldTool:
    requested = getattr(getattr(cfg, "refolding", None), "tool", None) if cfg is not None else None
    name = normalize_refold_tool_name(requested) or profile.default_refold_tool
    if name not in profile.compatible_refold_tools:
        raise SearchProfileError(
            f"refolding.tool {name!r} is incompatible with backbone {profile.model}; "
            f"allowed={sorted(profile.compatible_refold_tools)}"
        )
    try:
        return get_refold_tool(name)
    except RefoldToolError as exc:
        raise SearchProfileError(str(exc)) from exc


def get_model_search_profile(model: str, cfg: Any = None) -> ModelSearchProfile:
    key = str(model or "boltzgen").strip().lower()
    builder = _PROFILE_BUILDERS.get(key)
    if builder is None:
        raise SearchProfileError(f"unsupported search profile model {model!r}; supported={sorted(_PROFILE_BUILDERS)}")
    profile = builder()
    if cfg is not None:
        profile = profile.bound_tools(resolve_sequence_tool(cfg, profile), resolve_refold_tool(cfg, profile))
    return profile


def isolate_model_params(cfg: Any, model_name: str, job_params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build adapter params from the named model's search_space plus shared job fields."""
    from binderloop.config import primary_design_model
    bind_cfg = cfg if cfg is not None and primary_design_model(cfg) == str(model_name).strip().lower() else None
    profile = get_model_search_profile(model_name, cfg=bind_cfg)
    merged = dict(profile.load_search_space(cfg))
    for key, value in dict(job_params or {}).items():
        if key in profile.forbidden_keys:
            continue
        merged[key] = value
    return profile.filter_params(merged).params


def absolute_bounds_for_axis(name: str, lo: float, hi: float) -> Dict[str, Any]:
    """Accept a [min, max] window if it fits any registered model bound for ``name``."""
    catalogs = (("boltzgen", PARAM_BOUNDS), ("rfd3", RFD3_PARAM_BOUNDS))
    for _model, bounds in catalogs:
        bound = bounds.get(name)
        if bound is None:
            continue
        if lo >= float(bound["min"]) and hi <= float(bound["max"]):
            return dict(bound)
    raise SearchProfileError(
        f"sampler_bounds.{name} [{lo}, {hi}] is outside every model contract"
    )
