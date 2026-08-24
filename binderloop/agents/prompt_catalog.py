"""Declarative prompt tags, shared SYSTEM sections, and per-role specs.

Agents declare ``required_tags`` instead of assembling ad-hoc context dicts.
``compose_system`` is the single place for domain knowledge that used to be
copy-pasted into every SYSTEM string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from binderloop.agents.config_parameter_contract import (
    render_config_parameter_contract,
    render_param_bounds_contract,
)


PROMPT_VERSION = "round-context-v1"

CONTEXT_TAGS: Tuple[str, ...] = (
    "task.role",
    "task.goal",
    "task.round_id",
    "schema.output",
    "facts.metric",
    "facts.gates",
    "facts.core_rank",
    "examples.al_current",
    "examples.al_prior",
    "examples.al_clusters",
    "candidates.clusters",
    "candidates.representatives",
    "candidates.leaves",
    "structure.aggregates",
    "structure.fragments_diverse",
    "structure.templates_ids",
    "structure.phenotype_clusters",
    "config.current",
    "config.bounds",
    "constraints.hard",
    "target.profile",
    "execution.monitor",
    "execution.errors",
    "memory.retrieved",
    "skills.directives",
    "upstream.quality",
    "upstream.hypotheses",
    "upstream.diagnostic",
    "upstream.arm_comparison",
    "arms.evidence",
    "arms.blocked",
    "ledger.compact",
)

SYSTEM_SECTIONS: Dict[str, str] = {
    "knowledge.success_gate": (
        "The unified strict success gate is iPTM>=0.50, interface PAE<=10A, "
        "design pTM>=0.70, and refold RMSD<=2.5A. No single metric alone defines success. "
        "Treat metric_facts as immutable. Explicitly distinguish additional_filter_pass "
        "(e.g. pass_iptm_filter for iptm>0.35), BoltzGen pass_filters, and harness success_count; "
        "never infer that one gate implies another."
    ),
    "knowledge.al_three_class": (
        "If active_learning_examples or examples.al_clusters is present, keep three labels "
        "distinct: strict_positive_examples pass iPTM>=0.50, PAE<=10A, pTM>=0.70, and RMSD<=2.5A; "
        "near_miss_examples are boundary evidence and never successes; other_negative_examples "
        "are the remaining failures. Use prior_rounds only as accumulated evidence, not as "
        "current-round outcomes. Phenotype cluster cards summarize similar candidates; "
        "do not treat cluster size as extra successes."
    ),
    "knowledge.chain_id_note": (
        "BoltzGen output structures relabel chains by entity order: the generated binder is "
        "commonly output chain A and target chains are shifted/reassigned. Do not diagnose a "
        "chain-ID mismatch solely because output target_chains differ from configured "
        "target.chain_id; use chain_detection_note, hotspot_contacts, and target residue numbers."
    ),
    "contract.sampler_direction": (
        "For alpha, noise_scale, and step_scale, describe only increase/decrease/hold direction "
        "in prose; never emit them as numeric executable config values."
    ),
    "hypothesis.role": (
        "You are a protein binder design research agent. Return JSON only. "
        "Every hypothesis must include one or more exact failure_modes from the closed enum. "
        "The name is descriptive only and is never used for routing. Be cautious and do not "
        "invent unavailable measurements."
    ),
    "hypothesis.schema": (
        'Schema: {"hypotheses":[{"name":...,"failure_modes":'
        '["hotspot_miss|binding_pose_failure|clash|folding_failure|diversity_collapse|no_dominant_failure"],'
        '"evidence":...,"confidence":0-1,"intervention":...,"config_parameter_changes":{"key":"value"},'
        '"expected_signal_next_round":...,"risk":...}]}. '
        "Use config_parameter_changes only for executable keys from the contract; otherwise keep "
        "proposed interventions as prose."
    ),
    "quality.role": (
        "You are a senior computational protein binder design analyst. "
        "Use only supplied evidence. Do not invent unavailable measurements. "
        "Distinguish candidate-level quality from local fragment/module quality. "
        "Prefer actionable next-round guidance."
    ),
    "quality.schema": (
        "Return JSON only with keys: "
        '{"overall_assessment":"...","high_quality_modules":[{"module_id":"...","evidence":["..."],'
        '"likely_causes":["..."],"reuse_guidance":"...","confidence":0-1}],'
        '"low_quality_modules":[{"module_id":"...","evidence":["..."],"likely_causes":["..."],'
        '"repair_guidance":"...","confidence":0-1}],'
        '"causal_factors":[{"factor":"...","evidence":["..."],"impact":"positive|negative|mixed","confidence":0-1}],'
        '"next_round_guidance":[{"action":"...","target_modules":["..."],"parameter_or_constraint_change":"...",'
        '"config_parameter_changes":{"key":"value"},"expected_signal":"...","risk":"..."}]}. '
        "Use config_parameter_changes only for executable keys from the contract; otherwise keep "
        "guidance as prose in parameter_or_constraint_change."
    ),
    "diagnostic.role": (
        "You are an expert computational protein engineering coach monitoring an automated "
        "binder design pipeline (BoltzGen/BoltzDesign). You receive a snapshot of the current "
        "pipeline state including job execution status, BoltzGen metrics, structural evaluations, "
        "and historical round data. Be specific, quantitative, and actionable. Avoid generic suggestions."
    ),
    "diagnostic.schema": (
        "Return JSON only with this schema: "
        '{"status_diagnosis":"...","root_causes":[{"cause":"...","evidence":["..."],"confidence":0-1,'
        '"category":"execution|design_quality|constraint|sampling|resource"}],'
        '"metric_interpretation":{"iptm_assessment":"...","plddt_assessment":"...","rmsd_assessment":"...",'
        '"interface_assessment":"...","overall_binding_quality":"none|weak|moderate|strong","key_bottleneck":"..."},'
        '"corrective_actions":[{"action":"...","parameter_changes":{"key":"value"},"priority":"critical|high|medium|low",'
        '"expected_improvement":"...","risk":"..."}],'
        '"monitoring_recommendations":[{"check":"...","frequency_seconds":60,"abort_condition":"...","success_condition":"..."}],'
        '"pipeline_health":{"execution_ok":true,"design_generating":true,"filtering_working":true,'
        '"interface_forming":true,"hotspot_engaging":true,"ready_for_next_round":true,"recommended_wait_seconds":0}}. '
        "Every key inside corrective_actions[].parameter_changes must be an executable config parameter "
        "from the contract."
    ),
    "hotspot.role": (
        "You select a compact protein-binder hotspot patch from an anonymous residue table. "
        "The table has chain IDs, residue numbers, amino-acid letters, hydrophobicity, charge, "
        "aromatic/polar flags, CA neighbor counts, exposure percentiles, and local patch composition. "
        "Do not search the web or use protein names, PDB IDs, file paths, or literature epitopes. "
        "Reason only from geometry and physicochemical properties. Prefer a spatially clustered "
        "solvent-exposed patch rather than isolated buried residues."
    ),
    "hotspot.schema": (
        "Return JSON only with keys: "
        '{"hotspots":["A:12","A:14","A:18"],"rationale":"...","expected_signal_next_round":"...",'
        '"changes_from_previous":["+A:20","-A:9"]}. '
        "hotspots must use chain:residue tokens from the supplied table. "
        "rationale must discuss surface exposure, clustering, hydrophobicity/charge/aromatic character; "
        "never name a protein or cite papers."
    ),
}


def compose_system(
    *section_ids: str,
    extra: str = "",
    adjustable: Optional[Mapping[str, str]] = None,
    bounds: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    """Concatenate named SYSTEM sections plus optional dynamic contracts."""
    parts = []
    for section_id in section_ids:
        if section_id == "contract.config":
            parts.append(render_config_parameter_contract(adjustable))
        elif section_id == "contract.param_bounds":
            parts.append(render_param_bounds_contract(bounds))
        else:
            text = SYSTEM_SECTIONS.get(section_id)
            if not text:
                raise KeyError("unknown SYSTEM section: %s" % section_id)
            parts.append(text)
    if extra:
        parts.append(extra)
    return "\n\n".join(part.strip() for part in parts if str(part).strip())


@dataclass(frozen=True)
class AgentPromptSpec:
    role: str
    goal: str
    required_tags: Tuple[str, ...]
    system_sections: Tuple[str, ...]
    schema_fields: Tuple[str, ...] = ()
    include_leaves: bool = False
    extra_system: str = ""


AGENT_PROMPT_SPECS: Dict[str, AgentPromptSpec] = {
    "HypothesisAgent": AgentPromptSpec(
        role="HypothesisAgent",
        goal="propose_failure_hypotheses",
        required_tags=(
            "task.round_id",
            "facts.metric",
            "examples.al_clusters",
            "candidates.clusters",
            "structure.aggregates",
            "config.current",
        ),
        system_sections=(
            "hypothesis.role",
            "hypothesis.schema",
            "knowledge.al_three_class",
            "knowledge.success_gate",
            "knowledge.chain_id_note",
            "contract.config",
            "contract.sampler_direction",
        ),
        schema_fields=("hypotheses",),
    ),
    "BinderQualityAnalysisAgent": AgentPromptSpec(
        role="BinderQualityAnalysisAgent",
        goal="analyze_binder_quality",
        required_tags=(
            "task.round_id",
            "facts.metric",
            "examples.al_clusters",
            "candidates.clusters",
            "structure.fragments_diverse",
            "target.profile",
            "config.current",
            "constraints.hard",
        ),
        system_sections=(
            "quality.role",
            "quality.schema",
            "knowledge.al_three_class",
            "knowledge.success_gate",
            "knowledge.chain_id_note",
            "contract.config",
            "contract.sampler_direction",
        ),
        schema_fields=("overall_assessment", "next_round_guidance", "causal_factors"),
        extra_system=(
            "Treat context.evaluation.metric_facts as immutable facts, and explicitly distinguish "
            "additional_filter_pass, BoltzGen pass_filters, and harness success_count."
        ),
    ),
    "DiagnosticCoachAgent": AgentPromptSpec(
        role="DiagnosticCoachAgent",
        goal="diagnose_pipeline_health",
        required_tags=(
            "task.round_id",
            "execution.monitor",
            "facts.metric",
            "examples.al_clusters",
            "structure.aggregates",
            "config.current",
            "memory.retrieved",
        ),
        system_sections=(
            "diagnostic.role",
            "diagnostic.schema",
            "knowledge.success_gate",
            "knowledge.al_three_class",
            "knowledge.chain_id_note",
            "contract.config",
            "contract.sampler_direction",
        ),
        schema_fields=("corrective_actions",),
        extra_system=(
            "iPTM below 0.2 is weak interface evidence, but labels must follow the full four-metric gate. "
            "design_ptm > 0.7 indicates reasonable fold confidence. "
            "designfolding-filter_rmsd < 2.5 means the design refolds reliably. "
            "filter_rmsd < 5.0 is acceptable; > 10 indicates severe backbone deviation. "
            "plip_hbonds_refolded > 3 indicates some polar contacts forming. "
            "Low iptm with good plddt = binder folds OK but does not bind the target. "
            "High filter_rmsd with low designfolding-filter_rmsd = backbone is fine but target placement failed. "
            "Treat pipeline_state.metrics_summary and pipeline_state.evaluation.metric_facts as immutable facts."
        ),
    ),
    "InputConfigurationAgent": AgentPromptSpec(
        role="InputConfigurationAgent",
        goal="configure_next_round",
        required_tags=(
            "task.round_id",
            "config.current",
            "constraints.hard",
            "facts.metric",
            "upstream.quality",
            "upstream.hypotheses",
            "upstream.diagnostic",
            "memory.retrieved",
            "target.profile",
        ),
        system_sections=(
            "knowledge.success_gate",
            "knowledge.al_three_class",
            "contract.config",
            "contract.param_bounds",
            "contract.sampler_direction",
        ),
        schema_fields=("parameter_delta", "recommended_config"),
    ),
    "ConfigValidationAgent": AgentPromptSpec(
        role="ConfigValidationAgent",
        goal="validate_submittable_config",
        required_tags=("config.current", "execution.errors"),
        system_sections=(),
        schema_fields=("is_valid", "corrected_config"),
    ),
    "BlockedArmReviewAgent": AgentPromptSpec(
        role="BlockedArmReviewAgent",
        goal="review_blocked_arms",
        required_tags=("arms.blocked", "arms.evidence", "ledger.compact"),
        system_sections=(),
        schema_fields=("reviews",),
    ),
    "HotspotSelectionAgent": AgentPromptSpec(
        role="HotspotSelectionAgent",
        goal="select_primary_hotspots",
        required_tags=(),
        system_sections=("hotspot.role", "hotspot.schema", "knowledge.success_gate"),
        schema_fields=("hotspots", "rationale", "expected_signal_next_round", "changes_from_previous"),
        extra_system=(
            "Closed-loop identity hiding is mandatory: never infer or state the protein name. "
            "If round_evidence is present, adjust the previous patch using contact residues and "
            "success counts only. Stay within min_hotspots/max_hotspots and prefer small changes."
        ),
    ),
}


def spec_for(role: str) -> AgentPromptSpec:
    spec = AGENT_PROMPT_SPECS.get(role)
    if spec is None:
        raise KeyError("unknown agent prompt role: %s" % role)
    return spec
