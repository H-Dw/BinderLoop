
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from binderloop.agents.config_parameter_contract import invalid_config_value_keys, strip_probabilistic_sampler_keys, supported_config_changes, unsupported_config_keys
from binderloop.agents.context_compaction import compact_context_for_quality
from binderloop.agents.prompt_catalog import compose_system, spec_for
from binderloop.agents.role import LLMStructuredAgent
from binderloop.llm import OpenAICompatibleClient
from binderloop.resume import atomic_write_json


_QUALITY_SPEC = spec_for("BinderQualityAnalysisAgent")


@dataclass
class BinderQualityAnalysis:
    round_id: int
    llm_used: bool
    overall_assessment: str
    high_quality_modules: List[Dict[str, Any]] = field(default_factory=list)
    low_quality_modules: List[Dict[str, Any]] = field(default_factory=list)
    causal_factors: List[Dict[str, Any]] = field(default_factory=list)
    next_round_guidance: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class BinderQualityAnalysisAgent(LLMStructuredAgent):
    """Per-round binder quality analyst.

    It converts numeric metrics + coordinate-level fragment summaries into a
    quality narrative that can be consumed by policy and next-round strategy.
    LLM mode is preferred when configured; deterministic fallback preserves the
    same JSON shape for offline tests.
    """

    name = "BinderQualityAnalysisAgent"
    required_tags = _QUALITY_SPEC.required_tags
    output_schema = _QUALITY_SPEC.schema_fields
    system_sections = _QUALITY_SPEC.system_sections
    extra_system = _QUALITY_SPEC.extra_system
    temperature = 0.2
    max_tokens = 8000
    thinking = "low"
    SYSTEM = compose_system(*_QUALITY_SPEC.system_sections, extra=_QUALITY_SPEC.extra_system)

    def __init__(self, llm: Optional[OpenAICompatibleClient] = None, *, require_llm: bool = False):
        super().__init__(llm, require_llm=require_llm)

    def analyze(self, *, round_id: int, context: Mapping[str, Any]) -> BinderQualityAnalysis:
        compact = self._compact_context(context)
        prompt_context = dict(compact)
        skills = prompt_context.pop("active_skills", None)
        call = self.call_json(
            system=self.composed_system(active_skills=skills),
            user={"round_id": round_id, "context": prompt_context},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking=self.thinking,
        )
        if not call.llm_used:
            fallback = self._fallback(round_id, compact)
            if call.source == "llm_unavailable":
                return fallback
            fallback.raw = {
                "llm_error": call.raw.get("llm_error") or call.error or "transport_or_config",
                "source": call.source or "deterministic_fallback_after_llm_error",
            }
            return fallback
        result = call.value
        if isinstance(result, dict) and not result.get("parse_error"):
            has_guidance = isinstance(result.get("next_round_guidance"), list)
            has_assessment = bool(result.get("overall_assessment"))
            has_factors = isinstance(result.get("causal_factors"), list)
            if has_guidance or has_assessment or has_factors:
                facts = dict((compact.get("evaluation") or {}).get("metric_facts") or compact.get("metric_facts") or {})
                fact_check_issues = self.fact_check(json.dumps(result, ensure_ascii=False), facts)
                if fact_check_issues:
                    fallback = self._fallback(round_id, compact)
                    fallback.raw = {
                        "source": "deterministic_fallback_after_fact_check",
                        "llm_result": result,
                        "fact_check_issues": fact_check_issues,
                        "facts_used": facts,
                        "context_digest": self.digest(compact),
                    }
                    return fallback
                next_round_guidance = self._sanitize_guidance(list(result.get("next_round_guidance") or []))
                return BinderQualityAnalysis(
                    round_id=round_id,
                    llm_used=True,
                    overall_assessment=str(result.get("overall_assessment", "")),
                    high_quality_modules=list(result.get("high_quality_modules") or []),
                    low_quality_modules=list(result.get("low_quality_modules") or []),
                    causal_factors=list(result.get("causal_factors") or []),
                    next_round_guidance=next_round_guidance,
                    raw={**result, "facts_used": facts, "context_digest": self.digest(compact)},
                )
        fallback = self._fallback(round_id, compact)
        fallback.raw = {"llm_parse_failed": result}
        return fallback

    @staticmethod
    def _sanitize_guidance(guidance: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for item in guidance:
            row = dict(item or {})
            changes = dict(row.get("config_parameter_changes") or {})
            changes, sampler_ignored = strip_probabilistic_sampler_keys(changes)
            ignored = sampler_ignored + unsupported_config_keys(changes) + invalid_config_value_keys(changes)
            row["config_parameter_changes"] = supported_config_changes(changes)
            if ignored:
                row["ignored_config_parameter_changes"] = ignored
            sanitized.append(row)
        return sanitized

    def write_analysis(self, analysis: BinderQualityAnalysis, path: Union[str, Path]) -> Path:
        return atomic_write_json(path, asdict(analysis))

    @staticmethod
    def _compact_context(context: Mapping[str, Any]) -> Dict[str, Any]:
        # Delegate to the shared compactor so heavy fields (ca_coordinates, raw
        # metrics, full message logs) are stripped consistently across agents.
        return compact_context_for_quality(context)

    @staticmethod
    def _fallback(round_id: int, context: Mapping[str, Any]) -> BinderQualityAnalysis:
        structural = dict(context.get("structural_analysis") or {})
        evaluation = dict(context.get("evaluation") or {})
        high: List[Dict[str, Any]] = []
        low: List[Dict[str, Any]] = []
        factors: List[Dict[str, Any]] = []
        guidance: List[Dict[str, Any]] = []
        for summary in structural.get("summaries", []) or []:
            struct_id = f"aggregate_structure_{len(high) + len(low) + 1}"
            for frag in summary.get("high_quality_fragments", []) or []:
                high.append({
                    "module_id": f"{struct_id}:{frag.get('fragment_id')}",
                    "evidence": frag.get("reasons", []) + [f"quality_score={frag.get('quality_score')}", f"interface_contacts={frag.get('interface_contact_count')}", f"hotspot_contacts={frag.get('hotspot_contact_count')}"] ,
                    "likely_causes": _causes_from_reasons(frag.get("reasons", []), positive=True),
                    "reuse_guidance": frag.get("suggested_action", "Preserve this fragment pattern in the next round."),
                    "confidence": min(0.9, max(0.4, float(frag.get("quality_score") or 0.0))),
                })
            for frag in summary.get("low_quality_fragments", []) or []:
                low.append({
                    "module_id": f"{struct_id}:{frag.get('fragment_id')}",
                    "evidence": frag.get("reasons", []) + [f"quality_score={frag.get('quality_score')}", f"clashes={frag.get('clash_count')}"] ,
                    "likely_causes": _causes_from_reasons(frag.get("reasons", []), positive=False),
                    "repair_guidance": frag.get("suggested_action", "Repair or avoid this fragment pattern in the next round."),
                    "confidence": min(0.9, max(0.35, 1.0 - float(frag.get("quality_score") or 0.0))),
                })
        tags = dict(evaluation.get("tag_counts") or {})
        struct_tags = dict(structural.get("aggregate_tags") or {})
        if high:
            factors.append({"factor": "reusable_local_interface_modules", "evidence": [f"{len(high)} high-quality fragments detected"], "impact": "positive", "confidence": 0.65})
            guidance.append({"action": "exploit_high_quality_fragments", "target_modules": [h["module_id"] for h in high[:5]], "parameter_or_constraint_change": "Report high-scoring fragments; FragmentTemplateMiningAgent decides whether and how to template from PAE-gated fragments.", "config_parameter_changes": {}, "expected_signal": "higher reliability_score and retained hotspot/interface contacts", "risk": "over-exploitation may reduce topology diversity"})
        if low or tags or struct_tags:
            evidence = [f"metric_tags={tags}", f"structure_tags={struct_tags}"]
            low_module_ids = [l["module_id"] for l in low[:5]]
            factors.append({"factor": "repairable_failure_modules", "evidence": evidence, "impact": "negative", "confidence": 0.65})
            guidance.append({"action": "repair_low_quality_fragments", "target_modules": low_module_ids, "parameter_or_constraint_change": "Increase clash-aware filtering/hotspot weighting for affected modules; reduce exploitation of chain-break or weak-interface fragments. Low-quality module IDs are analysis metadata, not BoltzGen executable config.", "config_parameter_changes": {}, "selection_policy": {"kind": "cross_chain_heavy_atom_clash"}, "analysis_metadata": {"avoid_fragment_modules": low_module_ids}, "expected_signal": "lower clash_density, fewer hotspot_not_covered/weak_or_tiny_interface tags", "risk": "stricter filters may discard diverse candidates"})
        if not guidance:
            guidance.append({"action": "collect_richer_structure_evidence", "target_modules": [], "parameter_or_constraint_change": "Retain intermediates and run at least one round with structure outputs available.", "config_parameter_changes": {"diffusion_batch_size": 1}, "expected_signal": "fragment-level quality labels become available", "risk": "no strong causal claim without structures"})
        assessment = "No structures were available; quality analysis is based on metrics and logs only." if not structural.get("summaries") else f"Analyzed {structural.get('total_structures', 0)} structures; reliable_seed_fraction={structural.get('reliable_seed_fraction', 0)}."
        return BinderQualityAnalysis(round_id, False, assessment, high[:20], low[:20], factors, guidance, {"source": "deterministic_fallback"})


def _causes_from_reasons(reasons: List[str], *, positive: bool) -> List[str]:
    mapping = {
        "dense_target_interface": "many binder residues participate in the target interface",
        "contacts_target_hotspot": "local geometry reaches specified hotspot residues",
        "specific_polar_or_salt_contacts": "specific polar/salt-bridge-like contacts stabilize the interface",
        "balanced_hydrophobic_packing": "hydrophobic packing is present without becoming over-hydrophobic",
        "local_clash_risk": "close contacts suggest steric conflict or over-packed local geometry",
        "local_chain_break": "local backbone continuity appears unreliable",
        "weak_local_interface": "few residues in this segment contact the target",
        "over_hydrophobic_patch": "interface chemistry may be too hydrophobic and nonspecific",
        "few_specific_polar_contacts": "lack of polar/salt contacts makes binding specificity uncertain",
    }
    causes = [mapping.get(r, r) for r in reasons]
    if not causes:
        causes = ["combined local geometry and metric evidence is favorable" if positive else "combined local geometry and metric evidence is weak"]
    return causes
