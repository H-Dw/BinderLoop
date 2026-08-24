
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from binderloop.agents.config_parameter_contract import invalid_config_value_keys, strip_probabilistic_sampler_keys, supported_config_changes, unsupported_config_keys
from binderloop.agents.context_compaction import compact_context_for_hypothesis
from binderloop.agents.prompt_catalog import compose_system, spec_for
from binderloop.agents.role import LLMStructuredAgent
from binderloop.llm import OpenAICompatibleClient


FAILURE_MODES = frozenset({
    "hotspot_miss",
    "binding_pose_failure",
    "clash",
    "folding_failure",
    "diversity_collapse",
    "no_dominant_failure",
})

_HYPOTHESIS_SPEC = spec_for("HypothesisAgent")


@dataclass
class HypothesisSet:
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    llm_used: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


class HypothesisAgent(LLMStructuredAgent):
    """Generate failure hypotheses via LLM, with deterministic fallback rules."""

    name = "HypothesisAgent"
    required_tags = _HYPOTHESIS_SPEC.required_tags
    output_schema = _HYPOTHESIS_SPEC.schema_fields
    system_sections = _HYPOTHESIS_SPEC.system_sections
    extra_system = _HYPOTHESIS_SPEC.extra_system
    temperature = 0.25
    SYSTEM = compose_system(*_HYPOTHESIS_SPEC.system_sections, extra=_HYPOTHESIS_SPEC.extra_system)

    def __init__(self, llm: Optional[OpenAICompatibleClient] = None, *, require_llm: bool = False):
        super().__init__(llm, require_llm=require_llm)

    def propose(self, context: Mapping[str, Any]) -> HypothesisSet:
        compact = compact_context_for_hypothesis(context)
        prompt = dict(compact)
        skills = prompt.pop("active_skills", None)
        call = self.call_json(
            system=self.composed_system(active_skills=skills),
            user={"context": prompt},
            temperature=self.temperature,
        )
        if not call.llm_used:
            raw: Dict[str, Any] = {}
            if call.source == "llm_unavailable":
                return HypothesisSet(self._fallback(context), False)
            if call.error:
                raw = {"llm_error": call.error, "source": call.source, **call.raw}
                return HypothesisSet(self._fallback(context), False, raw)
            return HypothesisSet(self._fallback(context), False, {"llm_parse_failed": call.raw})
        result = call.value or {}
        if isinstance(result.get("hypotheses"), list):
            facts = dict((compact.get("evaluation") or {}).get("metric_facts") or {})
            sanitized = self._sanitize_hypotheses(result["hypotheses"])
            augmented = self._augment_failure_coverage(sanitized, context)
            return HypothesisSet(
                augmented,
                True,
                {**result, "facts_used": facts, "context_digest": self.digest(compact)},
            )
        return HypothesisSet(self._fallback(context), False, {"llm_parse_failed": result})

    @staticmethod
    def _sanitize_hypotheses(hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for hypothesis in hypotheses:
            item = dict(hypothesis or {})
            changes = dict(item.get("config_parameter_changes") or {})
            changes, sampler_ignored = strip_probabilistic_sampler_keys(changes)
            ignored = sampler_ignored + unsupported_config_keys(changes) + invalid_config_value_keys(changes)
            item["config_parameter_changes"] = supported_config_changes(changes)
            modes = item.get("failure_modes")
            if isinstance(modes, str):
                modes = [modes]
            item["failure_modes"] = [
                str(mode) for mode in (modes or [])
                if str(mode) in FAILURE_MODES
            ] or ["no_dominant_failure"]
            if ignored:
                item["ignored_config_parameter_changes"] = ignored
            sanitized.append(item)
        return sanitized

    @staticmethod
    def _augment_failure_coverage(hypotheses: List[Dict[str, Any]], context: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Add deterministic hypotheses for obvious failure evidence omitted by the LLM."""
        augmented = list(hypotheses or [])
        covered = {
            str(mode)
            for hypothesis in augmented
            for mode in (hypothesis.get("failure_modes") or [])
            if str(mode) != "no_dominant_failure"
        }
        repairs = [
            hypothesis for hypothesis in HypothesisAgent._fallback(context)
            if any(mode not in covered for mode in hypothesis.get("failure_modes") or [])
            and hypothesis.get("failure_modes") != ["no_dominant_failure"]
        ]
        for repair in repairs:
            item = dict(repair)
            item["source"] = "deterministic_coverage_repair"
            augmented.append(item)
            covered.update(str(mode) for mode in item.get("failure_modes") or [])
        if covered:
            augmented = [
                item for item in augmented
                if item.get("failure_modes") != ["no_dominant_failure"]
                or item.get("source") != "deterministic_coverage_repair"
            ]
        return augmented

    @staticmethod
    def _fallback(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
        evaluation = context.get("evaluation") or {}; structure = context.get("structural_analysis") or {}
        tags = dict(evaluation.get("tag_counts") or {}); struct_tags = dict(structure.get("aggregate_tags") or {})
        total = max(1, int(evaluation.get("total_candidates") or structure.get("total_structures") or 1))
        hyps: List[Dict[str, Any]] = []
        def add(name, failure_mode, evidence, confidence, intervention, signal, risk, changes=None):
            hyps.append({"name": name, "failure_modes": [failure_mode], "evidence": evidence, "confidence": round(confidence, 3), "intervention": intervention, "config_parameter_changes": supported_config_changes(changes or {}), "expected_signal_next_round": signal, "risk": risk, "source": "deterministic_fallback"})
        if tags.get("hotspot_miss", 0) / total > 0.25 or struct_tags.get("hotspot_not_covered", 0):
            add("hotspot_conditioning_too_weak_or_patch_misaligned", "hotspot_miss", "hotspot misses or coordinate-level hotspot_not_covered are frequent.", 0.75, "Increase hotspot/binding-site conditioning; try hotspot subset ensembles and patch expansion.", "Higher hotspot_contact and fewer hotspot_not_covered tags.", "Over-constraining can reduce diversity or create clashes.", {"auxiliary_hotspots": []})
        if tags.get("folding_failure", 0) / total > 0.25 or struct_tags.get("binder_chain_break", 0):
            add("binder_backbone_or_sequence_not_self_consistent", "folding_failure", "Refolding/PLDDT failures or chain breaks suggest unreliable seed backbones.", 0.7, "Lower diffusion noise while keeping user-owned inverse-fold and binder-length settings unchanged.", "Improved pLDDT/refolding RMSD and fewer chain breaks.", "May miss rare valid topologies.", {"noise_scale": 0.9})
        if tags.get("binding_pose_failure", 0) / total > 0.25 or struct_tags.get("weak_or_tiny_interface", 0):
            add("pose_search_not_forming_sufficient_interface", "binding_pose_failure", "Low interface confidence or tiny interface indicates pose failure.", 0.65, "Increase interface-contact constraints, keep more candidates pre-filtering, explore longer/scaffolded binders.", "More interface residues and improved ipTM.", "Larger binders can add nonspecific contacts.", {"diffusion_batch_size": 1})
        if struct_tags.get("interface_clash_risk", 0):
            add("interface_packing_too_aggressive", "clash", "Coordinate analysis found clash-prone interfaces.", 0.68, "Add clash-aware filters or soften hotspot/packing constraints.", "Lower clash_density while preserving hotspot contacts.", "May discard salvageable designs.", {})
        if tags.get("diversity_collapse", 0) / total > 0.25:
            add("search_distribution_collapsed", "diversity_collapse", "Diversity-collapse tags are frequent.", 0.7, "Increase BoltzGen diversity knobs while preserving user-owned search size and length policy.", "Higher diversity with stable interface metrics.", "More exploration slows convergence.", {"diffusion_batch_size": 1, "alpha": 0.002})
        if not hyps:
            add("no_single_dominant_failure_mode", "no_dominant_failure", "No dominant failure mode is identified.", 0.45, "Run balanced exploitation plus exploratory length/topology arms and collect richer structure metrics.", "One arm separates in reliability/interface metrics.", "Broad search may spend budget slowly.", {"diffusion_batch_size": 1})
        return hyps
