from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from binderloop.agents.config_parameter_contract import invalid_config_value_keys, render_config_parameter_contract, render_param_bounds_contract, strip_probabilistic_sampler_keys, supported_config_changes, unsupported_config_keys
from binderloop.agents.context_compaction import compact_context_for_input_config, compact_context_for_target_config, context_digest
from binderloop.llm import OpenAICompatibleClient, LLMConfigError, LLMTransportError
from binderloop.parameter_decision import HOLD_CURRENT, ParameterDecisionSpec
from binderloop.resume import atomic_write_json
from binderloop.skills import compose_agent_system


def _build_system_prompt(
    *,
    sampler_axes: Optional[Sequence[str]] = None,
    adjustable: Optional[Mapping[str, str]] = None,
    bounds: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    axes = tuple(str(item) for item in (sampler_axes or ("alpha", "noise_scale", "step_scale")) if str(item))
    axis_text = ", ".join(axes) if axes else "catalog sampler axes"
    alpha_rule = ""
    if "alpha" in axes:
        alpha_rule = (
            "\n- alpha must stay near 0.001 unless >30% of candidates are tagged diversity_collapse. "
            "A large alpha (e.g. 0.5-0.7) destroys interface adherence and collapses iPTM."
        )
    domain = f"""You are an expert protein binder design strategist. Given information about a target protein, you derive optimal binder-design pipeline configurations for the active backbone.

Return JSON only with this schema. The primary executable output is sparse parameter_delta; recommended_config is an optional legacy alias and must contain the same sparse delta:
{{
  "reasoning": "2-3 sentence explanation of your overall strategy",
  "parameter_delta": {{
    "only_changed_executable_keys": "values"
  }},
  "evidence_finding_ids": ["validated finding IDs supporting the delta"],
  "hold_reasons": ["why unchanged parameters are held"],
  "expected_signals": ["measurable next-round signals"],
  "recommended_config": {{
    "executable_keys_from_the_contract_below": "values"
  }},
  "parameter_rationale": [
    {{"parameter": "...", "value": "...", "reason": "...", "confidence": 0-1}}
  ],
  "risk_assessment": [
    {{"risk": "...", "likelihood": "low|medium|high", "mitigation": "..."}}
  ],
  "iteration_strategy": {{}}
}}

Domain knowledge:
- Hotspot targeting is expressed through primary/expanded BINDING residues and measured coverage; numeric hotspot weights are unsupported.
- {axis_text} may be described only by direction; never output numeric values for them in recommended_config
- Strict success requires iPTM>=0.50, interface PAE<=10A, design pTM>=0.70, and refold RMSD<=2.5A; no single metric alone defines success
- design_ptm > 0.7 needed for reliable fold
- designfolding-filter_rmsd < 2.5 needed for refolding consistency
- Target-specific structural facts must come only from the current user context (target_profile, target_info, current_config.target, structural_analysis, or constraints). Do not infer oligomer state, chain bridging requirements, hotspots, residue IDs, or target chains from examples, prior tasks, file names, or static prompt memory. If a target fact is absent, state uncertainty in rationale instead of assuming it.
- Do not optimize interface quality alone. If design_ptm drops severely or designfolding-filter_rmsd rises, preserve foldability by broadening/adjusting binder_lengths within the user range, reducing excessive binding/crop pressure, and disabling overly tight hotspot/crop constraints before pushing more interface pressure.
- Do not propose or modify budget, round sample counts, inverse_fold_num_sequences, refolding_rmsd_threshold, fragment-template switches, search_space, resource, target definition, run_filtering, steps, or additional_filters; these are user-owned or orchestrator-owned values. You may propose binder_lengths within the user range and a few auxiliary_hotspots near existing user hotspots; never remove or replace user hotspots.
- Whole-binder decisions use one strict lexicographic CoreRankKey: primary gate pass, worst normalized margin, iPTM descending, PAE ascending, RMSD ascending. Never compensate one failed metric with another, and never use H-bonds/SASA/hotspot-contact as a decision tie-break.
- If evaluation_summary.pressure_conflict.active is true, do not emit auxiliary_hotspots additions, epitope_crop_mode tightening, filter_bindingsite=true, template_conditioned_fraction increases, or narrowed binder_lengths.
- If constraints.epitope_crop_disabled_hard_constraint is true, epitope_crop_mode must remain disabled and target_include/target_binding_types must remain the original user target.
- If evaluation_summary.active_learning_examples is present, use the distinct current_round strict_positive_examples, near_miss_examples, and other_negative_examples to select next-round changes. Near misses are boundary evidence, not successes. Use prior_rounds as accumulated support and keep it distinct from the current round.

Be quantitative and specific. Base your recommendations on the provided evidence. When metric_facts is present, treat it as immutable. Distinguish additional_filter_pass, BoltzGen pass_filters, and harness success_count.
"""
    return (
        domain
        + "\n" + render_config_parameter_contract(adjustable)
        + "\nOnly put executable keys from the list above in recommended_config; put non-executable ideas in reasoning, parameter_rationale, or risk_assessment."
        + "\n\n" + render_param_bounds_contract(bounds)
        + """

CRITICAL TUNING DISCIPLINE (the orchestrator enforces these as hard limits; violating them wastes a round):
- Never move a numeric knob outside its [min, max] range, and never exceed its per-round change limit. The default value is a strong baseline; only deviate with explicit evidence from the provided diagnostics/tags."""
        + alpha_rule
        + """
- If a "tuning_feedback" block is present in the input, it reports how your PREVIOUS round's parameter moves affected the reward. If a previous increase of a knob was followed by a reward DROP, do NOT repeat that move; revert toward the value that produced the best reward so far.
- If a "pressure_conflict" block is present and active, it is authoritative over hotspot/contact/crop repair suggestions.
"""
    )


@dataclass
class InputConfiguration:
    """LLM-derived input configuration for a binder design run."""
    target_name: str
    llm_used: bool
    reasoning: str
    recommended_config: Dict[str, Any] = field(default_factory=dict)
    parameter_rationale: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: List[Dict[str, Any]] = field(default_factory=list)
    iteration_strategy: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    parameter_proposals: Dict[str, Any] = field(default_factory=dict)
    parameter_delta: Dict[str, Any] = field(default_factory=dict)
    evidence_finding_ids: List[str] = field(default_factory=list)
    hold_reasons: List[str] = field(default_factory=list)
    expected_signals: List[str] = field(default_factory=list)


class InputConfigurationAgent:
    """LLM-powered agent that derives optimal pipeline input configurations.

    This encapsulates the expert reasoning needed to:
    1. Analyze a target protein structure and identify binding opportunities
    2. Choose appropriate binder lengths based on target interface topology
    3. Set hotspot residues and binding constraints
    4. Configure backbone sampler parameters for the specific target
    5. Plan a multi-round search strategy

    Rather than requiring manual expert configuration, this agent reasons
    about the target and produces a complete configuration with justification.
    """

    SYSTEM = _build_system_prompt()

    def __init__(
        self,
        llm: Optional[OpenAICompatibleClient] = None,
        *,
        require_llm: bool = False,
        parameter_candidates: Optional[Mapping[str, Any]] = None,
        adjustable_parameters: Optional[Mapping[str, str]] = None,
        param_bounds: Optional[Mapping[str, Mapping[str, Any]]] = None,
        sampler_axes: Optional[Sequence[str]] = None,
    ):
        self.llm = llm
        self.require_llm = require_llm
        self.parameter_candidates = _normalize_parameter_candidates(parameter_candidates)
        self.adjustable_parameters = dict(adjustable_parameters) if adjustable_parameters is not None else None
        self.param_bounds = (
            {str(key): dict(value) for key, value in dict(param_bounds).items()}
            if param_bounds is not None else None
        )
        self.sampler_axes = tuple(str(item) for item in sampler_axes) if sampler_axes is not None else None

    def _system_prompt(self) -> str:
        if self.adjustable_parameters is None and self.param_bounds is None and self.sampler_axes is None:
            return self.SYSTEM
        axes = self.sampler_axes or tuple(self.parameter_candidates) or ("alpha", "noise_scale", "step_scale")
        return _build_system_prompt(
            sampler_axes=axes,
            adjustable=self.adjustable_parameters,
            bounds=self.param_bounds,
        )

    def configure(
        self,
        *,
        target_name: str,
        target_info: Mapping[str, Any],
        target_profile: Optional[Mapping[str, Any]] = None,
        previous_results: Optional[Mapping[str, Any]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        parameter_candidates: Optional[Mapping[str, Any]] = None,
    ) -> InputConfiguration:
        """Derive optimal input configuration for the given target."""
        context = {
            "target_name": target_name,
            "target_info": dict(target_info),
        }
        if target_profile:
            context["target_profile"] = dict(target_profile)
        if previous_results:
            context["previous_results"] = dict(previous_results)
        if constraints:
            context["constraints"] = dict(constraints)

        if self.require_llm and not (self.llm and self.llm.available()):
            raise RuntimeError(
                "InputConfigurationAgent: --require-llm is set but no LLM endpoint is available. "
                "Cannot fall back to deterministic rules."
            )
        if self.llm and self.llm.available():
            llm_context = compact_context_for_target_config(context)
            try:
                prompt_context = dict(llm_context)
                prompt_context.pop("active_skills", None)
                result = self.llm.chat_json(
                    system=compose_agent_system(
                        self._system_prompt(),
                        active_skills=llm_context.get("active_skills"),
                    ),
                    user=prompt_context,
                    temperature=0.2,
                    max_tokens=8000,
                )
            except (LLMConfigError, LLMTransportError):
                if self.require_llm:
                    raise
                fallback = self._deterministic_config(target_name, context)
                fallback.raw = {"llm_error": "transport_or_config", "source": "deterministic_fallback_after_llm_error"}
                return fallback
            except Exception as exc:
                if self.require_llm:
                    raise
                fallback = self._deterministic_config(target_name, context)
                fallback.raw = {"llm_error": str(exc), "source": "deterministic_fallback_after_llm_error"}
                return fallback
            if isinstance(result, dict) and ("parameter_delta" in result or "recommended_config" in result):
                recommended_config, ignored_keys = self._sanitize_recommended_config(dict(result.get("parameter_delta") or result.get("recommended_config") or {}))
                proposals = self._collect_parameter_proposals(analysis=result, context=llm_context, current_config=target_info, parameter_candidates=parameter_candidates)
                return InputConfiguration(
                    target_name=target_name,
                    llm_used=True,
                    reasoning=str(result.get("reasoning", "")),
                    recommended_config=recommended_config,
                    parameter_rationale=list(result.get("parameter_rationale") or []),
                    risk_assessment=list(result.get("risk_assessment") or []),
                    iteration_strategy={},
                    raw={**result, "parameter_delta": recommended_config, "recommended_config": recommended_config, "ignored_recommended_config_keys": ignored_keys, "context_digest": context_digest(llm_context), "ignored_iteration_strategy": bool(result.get("iteration_strategy"))},
                    parameter_proposals=proposals,
                    parameter_delta=recommended_config,
                    evidence_finding_ids=_bounded_string_list(result.get("evidence_finding_ids"), 12, 80),
                    hold_reasons=_bounded_string_list(result.get("hold_reasons"), 8, 240),
                    expected_signals=_bounded_string_list(result.get("expected_signals"), 8, 240),
                )
            return self._handle_llm_parse_failure(target_name, result, mode="initial")
        return self._deterministic_config(target_name, context)

    def configure_next_round(
        self,
        *,
        target_name: str,
        current_config: Mapping[str, Any],
        diagnostic_report: Mapping[str, Any],
        evaluation_summary: Mapping[str, Any],
        round_id: int,
        structural_analysis: Optional[Mapping[str, Any]] = None,
        quality_analysis: Optional[Mapping[str, Any]] = None,
        hypotheses: Optional[List[Mapping[str, Any]]] = None,
        memory_summary: Optional[Mapping[str, Any]] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        tuning_feedback: Optional[Mapping[str, Any]] = None,
        target_profile: Optional[Mapping[str, Any]] = None,
        active_skills: Optional[List[Mapping[str, Any]]] = None,
        parameter_candidates: Optional[Mapping[str, Any]] = None,
    ) -> InputConfiguration:
        """Derive corrected configuration for the next round based on diagnostic feedback."""
        context = {
            "target_name": target_name,
            "current_config": dict(current_config),
            "diagnostic_report": dict(diagnostic_report),
            "evaluation_summary": dict(evaluation_summary),
            "round_id": round_id,
            "task": "configure_next_round_based_on_diagnostics",
        }
        if structural_analysis:
            context["structural_analysis"] = dict(structural_analysis)
        if quality_analysis:
            context["quality_analysis"] = dict(quality_analysis)
        if hypotheses:
            context["hypotheses"] = list(hypotheses)
        if memory_summary:
            context["memory_summary"] = dict(memory_summary)
        if constraints:
            context["constraints"] = dict(constraints)
        if tuning_feedback:
            context["tuning_feedback"] = dict(tuning_feedback)
        if target_profile:
            context["target_profile"] = dict(target_profile)
        if active_skills:
            context["active_skills"] = list(active_skills)

        if self.require_llm and not (self.llm and self.llm.available()):
            raise RuntimeError(
                "InputConfigurationAgent: --require-llm is set but no LLM endpoint is available. "
                "Cannot fall back to deterministic rules."
            )
        if self.llm and self.llm.available():
            llm_context = compact_context_for_input_config(
                target_name=target_name,
                current_config=current_config,
                diagnostic_report=diagnostic_report,
                evaluation_summary=evaluation_summary,
                round_id=round_id,
                target_profile=target_profile,
                structural_analysis=structural_analysis,
                quality_analysis=quality_analysis,
                hypotheses=hypotheses,
                memory_summary=memory_summary,
                constraints=constraints,
                tuning_feedback=tuning_feedback,
                active_skills=active_skills,
            )
            try:
                prompt_context = dict(llm_context)
                prompt_context.pop("active_skills", None)
                result = self.llm.chat_json(
                    system=compose_agent_system(
                        self._system_prompt(),
                        active_skills=active_skills,
                    ),
                    user=prompt_context,
                    temperature=0.2,
                    max_tokens=8000,
                )
            except (LLMConfigError, LLMTransportError):
                if self.require_llm:
                    raise
                fallback = self._deterministic_next_round(target_name, context)
                fallback.raw = {"llm_error": "transport_or_config", "source": "deterministic_next_round_fallback_after_llm_error"}
                return fallback
            except Exception as exc:
                if self.require_llm:
                    raise
                fallback = self._deterministic_next_round(target_name, context)
                fallback.raw = {"llm_error": str(exc), "source": "deterministic_next_round_fallback_after_llm_error"}
                return fallback
            if isinstance(result, dict) and ("parameter_delta" in result or "recommended_config" in result):
                recommended_config, ignored_keys = self._sanitize_recommended_config(dict(result.get("parameter_delta") or result.get("recommended_config") or {}))
                proposals = self._collect_parameter_proposals(analysis=result, context=llm_context, current_config=current_config, parameter_candidates=parameter_candidates)
                return InputConfiguration(
                    target_name=target_name,
                    llm_used=True,
                    reasoning=str(result.get("reasoning", "")),
                    recommended_config=recommended_config,
                    parameter_rationale=list(result.get("parameter_rationale") or []),
                    risk_assessment=list(result.get("risk_assessment") or []),
                    iteration_strategy={},
                    raw={**result, "parameter_delta": recommended_config, "recommended_config": recommended_config, "ignored_recommended_config_keys": ignored_keys, "facts_used": (llm_context.get("evaluation_summary") or {}).get("metric_facts"), "context_digest": context_digest(llm_context), "ignored_iteration_strategy": bool(result.get("iteration_strategy"))},
                    parameter_proposals=proposals,
                    parameter_delta=recommended_config,
                    evidence_finding_ids=_bounded_string_list(result.get("evidence_finding_ids"), 12, 80),
                    hold_reasons=_bounded_string_list(result.get("hold_reasons"), 8, 240),
                    expected_signals=_bounded_string_list(result.get("expected_signals"), 8, 240),
                )
            return self._handle_llm_parse_failure(target_name, result, mode="next_round")
        return self._deterministic_next_round(target_name, context)

    def write_config(self, config: InputConfiguration, path: Union[str, Path]) -> Path:
        return atomic_write_json(path, asdict(config))

    def _handle_llm_parse_failure(self, target_name: str, result: Any, *, mode: str) -> InputConfiguration:
        if self.require_llm:
            raise RuntimeError(f"InputConfigurationAgent LLM JSON parse failed during {mode}; refusing deterministic fallback because require_llm=True")
        return InputConfiguration(
            target_name=target_name,
            llm_used=False,
            reasoning="LLM returned invalid JSON or an invalid schema; no executable configuration changes were applied.",
            recommended_config={},
            parameter_rationale=[],
            risk_assessment=[{
                "risk": "llm_parse_failed",
                "likelihood": "high",
                "mitigation": "Retry with JSON repair or inspect raw.llm_parse_failed before applying next-round configuration.",
            }],
            iteration_strategy={},
            raw={
                "llm_attempted": True,
                "llm_completed": True,
                "llm_parse_ok": False,
                "llm_used_for_config": False,
                "fallback_reason": "json_parse_failed_no_config_applied",
                "llm_parse_failed": result,
                "source": "safe_noop_after_llm_parse_failure",
                "mode": mode,
            },
        )

    @staticmethod
    def _sanitize_recommended_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        config, sampler_ignored = strip_probabilistic_sampler_keys(config)
        ignored = sampler_ignored + unsupported_config_keys(config) + invalid_config_value_keys(config)
        return supported_config_changes(config), sorted(set(ignored))

    def _collect_parameter_proposals(self, *, analysis: Mapping[str, Any], context: Mapping[str, Any], current_config: Mapping[str, Any], parameter_candidates: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        catalogs = _normalize_parameter_candidates(parameter_candidates, base=self.parameter_candidates)
        method = getattr(self.llm, "chat_label_distribution", None)
        proposals: Dict[str, Any] = {}
        for parameter in catalogs:
            values = list(catalogs[parameter]); current = current_config.get(parameter)
            try:
                if current is not None and float(current) not in values: values.append(float(current))
            except (TypeError, ValueError):
                pass
            values = sorted(set(values)); labels = [HOLD_CURRENT] + [f"C{i}" for i in range(len(values))]
            label_map = {HOLD_CURRENT: HOLD_CURRENT, **{f"C{i}": value for i, value in enumerate(values)}}
            if not callable(method):
                proposals[parameter] = _unavailable_parameter_evidence(parameter, label_map, "chat_label_distribution unavailable"); continue
            try:
                evidence = method(system=f"Choose one closed label for {parameter}; use HOLD_CURRENT when evidence is weak. Return only the label.", user={"parameter": parameter, "current_value": current, "label_candidates": label_map, "analysis": dict(analysis), "context": dict(context)}, labels=labels)
                distribution = dict(evidence.get("distribution") or {}) if isinstance(evidence, Mapping) else {}
                selected = str(evidence.get("label") or HOLD_CURRENT) if isinstance(evidence, Mapping) else HOLD_CURRENT
                available = bool(distribution) and selected in label_map
                proposals[parameter] = {"status": "available" if available else "unavailable", "selected_label": selected if available else HOLD_CURRENT, "proposed_value": label_map.get(selected, HOLD_CURRENT) if available else HOLD_CURRENT, "labels": label_map, "distribution": distribution, "evidence": evidence, "execute": False}
            except Exception as exc:
                proposals[parameter] = _unavailable_parameter_evidence(parameter, label_map, str(exc))
        return proposals

    @staticmethod
    def _deterministic_config(target_name: str, context: Dict[str, Any]) -> InputConfiguration:
        """Deterministic fallback for initial configuration."""
        target_info = context.get("target_info") or {}
        hotspots = list(target_info.get("hotspots") or (context.get("target_profile") or {}).get("hotspots") or [])

        config = {"diffusion_batch_size": 1}
        risks = [
            {"risk": "Initial BoltzGen tuning is conservative", "likelihood": "medium", "mitigation": "Let closed-loop evidence adjust only allowed BoltzGen knobs"},
        ]
        if not hotspots:
            risks.append({
                "risk": "No target hotspots were provided",
                "likelihood": "medium",
                "mitigation": "Use target_profile/target_info from the current task; do not assume residue IDs from other targets.",
            })
        return InputConfiguration(
            target_name=target_name,
            llm_used=False,
            reasoning="Deterministic fallback: preserve user task YAML and apply only conservative BoltzGen tuning.",
            recommended_config=config,
            parameter_delta=config,
            parameter_rationale=[
                {"parameter": "diffusion_batch_size", "value": "1", "reason": "Preserve sampling diversity without changing user-owned search size", "confidence": 0.5},
            ],
            risk_assessment=risks,
            iteration_strategy={
                "round_1_focus": "Survey landscape with multiple lengths",
                "round_2_focus": "Exploit best-performing length/seed combinations",
                "round_3_focus": "Refine top candidates with increased sampling",
                "convergence_criteria": "iptm > 0.4 for at least 1 candidate",
                "abort_criteria": "No iptm > 0.15 after 3 rounds",
            },
            raw={"source": "deterministic_fallback"},
        )

    @staticmethod
    def _deterministic_next_round(target_name: str, context: Dict[str, Any]) -> InputConfiguration:
        """Deterministic fallback for next-round configuration."""
        diag = context.get("diagnostic_report") or {}
        current = dict(context.get("current_config") or {})
        evaluation = dict(context.get("evaluation_summary") or {})
        structural = dict(context.get("structural_analysis") or {})
        quality = dict(context.get("quality_analysis") or {})
        hypotheses = list(context.get("hypotheses") or [])
        memory = dict(context.get("memory_summary") or {})
        constraints = dict(context.get("constraints") or {})
        pressure_conflict = dict(evaluation.get("pressure_conflict") or {})
        crop_locked_disabled = bool(constraints.get("epitope_crop_disabled_hard_constraint")) or (
            str(current.get("epitope_crop_mode") or "disabled").strip().lower() in {"", "disabled", "off", "none", "false", "0"}
            and not bool(current.get("allow_agent_epitope_crop", False))
        )
        corrective = list(diag.get("corrective_actions") or [])
        structural_tags = dict(structural.get("aggregate_tags") or {})

        # Apply only supported BoltzGen tuning actions; user-owned task/search/resource
        # values from current_config are context, not an editable output surface.
        new_config: Dict[str, Any] = {}
        rationale: List[Dict[str, Any]] = []

        for action in corrective:
            changes = supported_config_changes(action.get("parameter_changes") or {})
            for key, value in changes.items():
                if pressure_conflict.get("active") and _is_pressure_increase(key, value, current):
                    continue
                new_config[key] = value
                rationale.append({
                    "parameter": key,
                    "value": str(value),
                    "reason": action.get("action", "diagnostic correction"),
                    "confidence": 0.6,
                })

        for guidance in quality.get("next_round_guidance", []) or []:
            changes = supported_config_changes(guidance.get("config_parameter_changes") or {})
            for key, value in changes.items():
                if pressure_conflict.get("active") and _is_pressure_increase(key, value, current):
                    continue
                new_config[key] = value
                rationale.append({
                    "parameter": key,
                    "value": str(value),
                    "reason": guidance.get("action", "quality analysis guidance"),
                    "confidence": 0.6,
                })

        for hypothesis in hypotheses:
            changes = supported_config_changes(hypothesis.get("config_parameter_changes") or {})
            for key, value in changes.items():
                if pressure_conflict.get("active") and _is_pressure_increase(key, value, current):
                    continue
                new_config[key] = value
                rationale.append({
                    "parameter": key,
                    "value": str(value),
                    "reason": hypothesis.get("name", "hypothesis guidance"),
                    "confidence": float(hypothesis.get("confidence") or 0.5),
                })

        total = int(evaluation.get("total_candidates") or 0)
        success = int(evaluation.get("success_count") or 0)
        tag_counts = dict(evaluation.get("tag_counts") or {})
        if not pressure_conflict.get("active") and not corrective and success == 0:
            new_config["diffusion_batch_size"] = 1
            rationale.append({"parameter": "diffusion_batch_size", "value": "1", "reason": "No passing candidates; preserve diverse diagnostic sampling", "confidence": 0.55})

        if not pressure_conflict.get("active") and (structural_tags.get("hotspot_not_covered") or tag_counts.get("hotspot_miss", 0) > max(1, total) * 0.3):
            if not crop_locked_disabled:
                new_config["epitope_crop_mode"] = "hotspot_focus"
            auxiliary = _auxiliary_hotspots_from_structural_evidence(structural, list(current.get("hotspots") or []))
            if auxiliary:
                new_config["auxiliary_hotspots"] = auxiliary
            new_config.setdefault("config_overrides", [])
            if ["filtering", "filter_bindingsite=true"] not in new_config["config_overrides"]:
                new_config["config_overrides"].append(["filtering", "filter_bindingsite=true"])
            rationale.append({
                "parameter": "binding_site_policy",
                "value": str({"auxiliary_hotspots": new_config.get("auxiliary_hotspots"), "epitope_crop_mode": new_config.get("epitope_crop_mode")}),
                "reason": "Coordinate analysis shows poor hotspot coverage; retain user hotspots, focus future crops around them, and optionally append nearby contacted target residues as auxiliary hotspots",
                "confidence": 0.65,
            })

        if structural_tags.get("weak_or_tiny_interface") or tag_counts.get("binding_pose_failure", 0) > max(1, total) * 0.3:
            new_config["diffusion_batch_size"] = 1
            new_config["config_overrides"] = new_config.get("config_overrides") or []
            rationale.append({
                "parameter": "diffusion_batch_size",
                "value": str(new_config.get("diffusion_batch_size")),
                "reason": "Weak/tiny interface suggests retaining diverse pose samples without changing user-owned length or budget",
                "confidence": 0.6,
            })

        if structural_tags.get("interface_clash_risk") or tag_counts.get("clash", 0) > max(1, total) * 0.2:
            new_config["diffusion_batch_size"] = 1
            rationale.append({
                "parameter": "diffusion_batch_size",
                "value": "1",
                "reason": "Interface clash risk was observed; keep batch size low and use measured clash selection downstream",
                "confidence": 0.7,
            })

        foldability_risk = structural_tags.get("binder_chain_break") or tag_counts.get("folding_failure", 0) > max(1, total) * 0.3
        if foldability_risk:
            if str(new_config.get("epitope_crop_mode") or current.get("epitope_crop_mode") or "").strip().lower() in {"hotspot_focus", "engaged_focus"}:
                new_config["epitope_crop_mode"] = "disabled"
            length_range = _coerce_length_range(current.get("binder_length_range"), current.get("binder_lengths"))
            widened = _broaden_lengths(current.get("binder_lengths"), length_range, int(current.get("binder_length_step") or 10))
            if widened:
                new_config["binder_lengths"] = widened
            rationale.append({
                "parameter": "foldability_adjustment",
                "value": str({k: new_config.get(k) for k in ("binder_lengths", "epitope_crop_mode") if k in new_config}),
                "reason": "Folding/chain-break signal observed; relax hotspot/crop pressure and broaden length options within the user range while leaving inverse-fold count/refolding threshold user-owned",
                "confidence": 0.55,
            })

        # Repeated length failures are reported in rationale only; binder length
        # search values come from the user task YAML and orchestrator policy.
        if crop_locked_disabled:
            new_config["epitope_crop_mode"] = "disabled"
        if pressure_conflict.get("active"):
            _strip_pressure_increases(new_config, current)

        return InputConfiguration(
            target_name=target_name,
            llm_used=False,
            reasoning="Deterministic correction from diagnostics and structural tags, limited to allowed BoltzGen tuning knobs.",
            recommended_config=supported_config_changes(strip_probabilistic_sampler_keys(new_config)[0]),
            parameter_delta=supported_config_changes(strip_probabilistic_sampler_keys(new_config)[0]),
            parameter_rationale=rationale,
            risk_assessment=[],
            iteration_strategy={},
            raw={"source": "deterministic_next_round_fallback"},
        )


def _normalize_parameter_candidates(supplied: Optional[Mapping[str, Any]], *, base: Optional[Mapping[str, Any]] = None) -> Dict[str, Tuple[float, ...]]:
    defaults = ParameterDecisionSpec()
    default_axes = {"alpha": tuple(defaults.alpha_candidates), "noise_scale": tuple(defaults.noise_scale_candidates), "step_scale": tuple(defaults.step_scale_candidates)}
    if supplied or base:
        keys = []
        for source in (base or {}, supplied or {}):
            for key, raw in dict(source).items():
                name = str(key)[:-11] if str(key).endswith("_candidates") else str(key)
                if name not in keys:
                    keys.append(name)
        out: Dict[str, Tuple[float, ...]] = {}
        for name in keys:
            raw = None
            for source in (supplied or {}, base or {}):
                if source.get(name) is not None:
                    raw = source.get(name)
                    break
                if source.get(f"{name}_candidates") is not None:
                    raw = source.get(f"{name}_candidates")
                    break
            if raw is None:
                continue
            values = tuple(float(value) for value in raw)
            if not values:
                raise ValueError(f"{name} candidates cannot be empty")
            out[name] = values
        if out:
            if set(out).issubset({"alpha", "noise_scale", "step_scale"}):
                merged = dict(default_axes)
                merged.update(out)
                return merged
            return out
    return dict(default_axes)


def _unavailable_parameter_evidence(parameter: str, labels: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    return {"status": "unavailable", "parameter": parameter, "selected_label": HOLD_CURRENT, "proposed_value": HOLD_CURRENT, "labels": dict(labels), "distribution": {HOLD_CURRENT: 1.0}, "evidence": {"reason": reason}, "execute": False}


def _coerce_length_range(raw_range: Any, lengths: Any) -> Optional[Tuple[int, int]]:
    if raw_range:
        if isinstance(raw_range, str):
            token = raw_range.replace("..", "-").replace(":", "-")
            lo_s, hi_s = token.split("-", 1)
            return int(lo_s), int(hi_s)
        if isinstance(raw_range, dict):
            return int(raw_range.get("min") or raw_range.get("start")), int(raw_range.get("max") or raw_range.get("end"))
        seq = list(raw_range)
        if len(seq) >= 2:
            return int(seq[0]), int(seq[1])
    if lengths:
        vals = [int(x) for x in lengths]
        return min(vals), max(vals)
    return None


def _is_pressure_increase(key: str, value: Any, current: Mapping[str, Any]) -> bool:
    if key in {"auxiliary_hotspots"}:
        return bool(value)
    if key == "epitope_crop_mode":
        return str(value or "").strip().lower() not in {"", "disabled", "off", "none", "false", "0"}
    if key == "config_overrides":
        return _contains_filter_bindingsite(value)
    if key == "template_conditioned_fraction":
        return True
    return False


def _strip_pressure_increases(config: Dict[str, Any], current: Mapping[str, Any]) -> None:
    config.pop("auxiliary_hotspots", None)
    if str(config.get("epitope_crop_mode") or "").strip().lower() not in {"", "disabled", "off", "none", "false", "0"}:
        config["epitope_crop_mode"] = "disabled"
    if _contains_filter_bindingsite(config.get("config_overrides")):
        config["config_overrides"] = [
            item for item in (config.get("config_overrides") or [])
            if "filter_bindingsite=true" not in " ".join(str(part).strip().lower() for part in (item if isinstance(item, (list, tuple)) else [item]))
        ]
    config.pop("template_conditioned_fraction", None)


def _contains_filter_bindingsite(value: Any) -> bool:
    for item in value or []:
        text = " ".join(str(part).strip().lower() for part in (item if isinstance(item, (list, tuple)) else [item]))
        if "filter_bindingsite=true" in text:
            return True
    return False


def _bounded_lengths(lengths: Any, length_range: Optional[Tuple[int, int]]) -> List[int]:
    vals = sorted({int(x) for x in (lengths or [])})
    if not length_range:
        return vals
    lo, hi = length_range
    return sorted({min(max(v, lo), hi) for v in vals if lo <= min(max(v, lo), hi) <= hi})


def _shift_lengths(lengths: List[int], length_range: Optional[Tuple[int, int]], delta: int) -> List[int]:
    shifted = [int(x) + int(delta) for x in lengths]
    if length_range:
        lo, hi = length_range
        shifted = [min(max(x, lo), hi) for x in shifted]
    return sorted({max(30, min(220, x)) for x in shifted})


def _broaden_lengths(lengths: Any, length_range: Optional[Tuple[int, int]], step: int) -> List[int]:
    vals = sorted({int(x) for x in (lengths or [])})
    if not vals or not length_range:
        return vals
    lo, hi = length_range
    step = max(1, int(step or 10))
    widened = set(vals)
    for value in vals:
        if value - step >= lo:
            widened.add(value - step)
        if value + step <= hi:
            widened.add(value + step)
    return sorted(widened)


def _merge_unique(current: Any, additions: List[str]) -> List[str]:
    values = list(current or [])
    for item in additions:
        if item not in values:
            values.append(item)
    return values


def _hotspots_from_structural_evidence(structural: Mapping[str, Any], current_hotspots: List[str]) -> List[str]:
    summaries = list(structural.get("summaries") or [])
    if not summaries:
        return current_hotspots
    contacted: Dict[str, int] = {}
    for summary in summaries:
        for hotspot, count in (summary.get("hotspot_contacts") or {}).items():
            if int(count or 0) > 0:
                contacted[hotspot] = contacted.get(hotspot, 0) + int(count)
    if not contacted:
        return current_hotspots
    prioritized = sorted(contacted, key=lambda h: contacted[h], reverse=True)
    for hotspot in current_hotspots:
        if hotspot not in prioritized:
            prioritized.append(hotspot)
    return prioritized[: max(3, min(len(prioritized), len(current_hotspots) or 5))]


def _auxiliary_hotspots_from_structural_evidence(structural: Mapping[str, Any], current_hotspots: List[str], *, max_items: int = 3, max_distance: int = 15) -> List[str]:
    user = [_parse_hotspot_token(h) for h in current_hotspots]
    user = [(c, r) for c, r in user if c and r is not None]
    if not user:
        return []
    existing = {f"{c}:{r}" for c, r in user}
    counts: Dict[str, int] = {}
    for summary in list(structural.get("summaries") or []):
        for contact in list(summary.get("contacts_preview") or [])[:50]:
            target = str(contact.get("target_residue") or "")
            chain, resid = _parse_hotspot_token(target)
            if not chain or resid is None:
                continue
            token = f"{chain}:{resid}"
            if token in existing:
                continue
            if any(chain == uc and abs(int(resid) - int(ur)) <= int(max_distance) for uc, ur in user):
                counts[token] = counts.get(token, 0) + 1
    return sorted(counts, key=lambda h: counts[h], reverse=True)[:max_items]


def _parse_hotspot_token(value: Any) -> Tuple[str, Optional[int]]:
    text = str(value or "").strip()
    if not text:
        return "", None
    if ":" in text:
        chain, raw = text.split(":", 1)
    else:
        chain, raw = "", text
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == "-")
    try:
        return chain.strip(), int(digits)
    except ValueError:
        return chain.strip(), None


def _failed_lengths_from_memory(memory: Mapping[str, Any]) -> Dict[int, int]:
    failed: Dict[int, int] = {}
    for record in memory.get("recent_rounds", []) or []:
        evaluation = record.get("evaluation") or {}
        if int(evaluation.get("success_count") or 0) > 0:
            continue
        for job in record.get("jobs", []) or []:
            try:
                length = int(job.get("binder_length"))
            except (TypeError, ValueError):
                continue
            failed[length] = failed.get(length, 0) + 1
    return failed

def _bounded_string_list(value: Any, maximum: int, max_length: int) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value[:maximum] if isinstance(item, str) and item and len(item) <= max_length]
