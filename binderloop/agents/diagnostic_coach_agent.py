
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from binderloop.agents.config_parameter_contract import invalid_config_value_keys, render_config_parameter_contract, strip_probabilistic_sampler_keys, supported_config_changes, unsupported_config_keys
from binderloop.agents.context_compaction import compact_context_for_diagnostic, context_digest, fact_check_text_against_metric_facts
from binderloop.llm import OpenAICompatibleClient, LLMConfigError, LLMTransportError
from binderloop.resume import atomic_write_json
from binderloop.skills import compose_agent_system


@dataclass
class DiagnosticReport:
    """Structured output from the DiagnosticCoachAgent."""
    round_id: int
    llm_used: bool
    status_diagnosis: str
    root_causes: List[Dict[str, Any]] = field(default_factory=list)
    metric_interpretation: Dict[str, Any] = field(default_factory=dict)
    corrective_actions: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_health: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class DiagnosticCoachAgent:
    """LLM-powered diagnostic coach that analyzes pipeline state and prescribes corrections.

    This agent encapsulates the 'thinking' that a human expert would apply when
    monitoring the binder design pipeline:
    1. Interpreting monitor/job status (queue delays, resource issues)
    2. Analyzing metric distributions to identify systemic problems
    3. Cross-referencing structural evaluation with metric failure modes
    4. Prescribing concrete parameter changes (not just generic advice)
    5. Deciding when to wait vs. re-submit vs. abort
    """

    SYSTEM = """You are an expert computational protein engineering coach monitoring an automated binder design pipeline (BoltzGen/BoltzDesign).

You receive a snapshot of the current pipeline state including: job execution status, BoltzGen metrics (iptm, plddt, rmsd, etc.), structural evaluations, and historical round data.

Return JSON only with this schema:
{
  "status_diagnosis": "one-paragraph summary of overall state",
  "root_causes": [
    {"cause": "...", "evidence": ["..."], "confidence": 0-1, "category": "execution|design_quality|constraint|sampling|resource"}
  ],
  "metric_interpretation": {
    "iptm_assessment": "...",
    "plddt_assessment": "...",
    "rmsd_assessment": "...",
    "interface_assessment": "...",
    "overall_binding_quality": "none|weak|moderate|strong",
    "key_bottleneck": "..."
  },
  "corrective_actions": [
    {"action": "...", "parameter_changes": {"key": "value"}, "priority": "critical|high|medium|low", "expected_improvement": "...", "risk": "..."}
  ],
  "monitoring_recommendations": [
    {"check": "...", "frequency_seconds": 60, "abort_condition": "...", "success_condition": "..."}
  ],
  "pipeline_health": {
    "execution_ok": true/false,
    "design_generating": true/false,
    "filtering_working": true/false,
    "interface_forming": true/false,
    "hotspot_engaging": true/false,
    "ready_for_next_round": true/false,
    "recommended_wait_seconds": 0
  }
}

Key domain knowledge:
- The unified strict success gate is iPTM>=0.50, interface PAE<=10A, design pTM>=0.70, and refold RMSD<=2.5A
- iPTM below 0.2 is weak interface evidence, but labels must follow the full four-metric gate
- design_ptm > 0.7 indicates reasonable fold confidence
- designfolding-filter_rmsd < 2.5 means the design refolds reliably
- filter_rmsd < 5.0 is acceptable; > 10 indicates severe backbone deviation
- plip_hbonds_refolded > 3 indicates some polar contacts forming
- For multimeric or multi-domain targets, use only the supplied target/config evidence to decide whether binders should bridge chains or focus on one patch
- Low iptm with good plddt = binder folds OK but doesn't bind the target
- High filter_rmsd with low designfolding-filter_rmsd = backbone is fine but target placement failed
- BoltzGen output structures relabel chains by entity order: the generated binder is commonly output chain A and target chains are shifted/reassigned. Do not diagnose a chain-ID mismatch solely because output target_chains differ from configured target.chain_id; use chain_detection_note, hotspot_contacts, and target residue numbers.

Be specific, quantitative, and actionable. Avoid generic suggestions.
If active_learning_examples is present, compare strict_positive_examples, boundary-only near_miss_examples, and other_negative_examples. Near misses may guide diagnosis but must not be relabelled or counted as success. Use prior_rounds as historical context only.
Treat pipeline_state.metrics_summary and pipeline_state.evaluation.metric_facts as immutable facts. Explicitly distinguish additional_filter_pass (e.g. pass_iptm_filter for iptm>0.35), BoltzGen pass_filters, and harness success_count; never infer that one gate implies another.
""" + "\n" + render_config_parameter_contract() + "\nEvery key inside corrective_actions[].parameter_changes must be one of the executable config parameters above. For alpha, noise_scale, and step_scale, describe only increase/decrease/hold direction in action prose; never emit them in parameter_changes."

    def __init__(self, llm: Optional[OpenAICompatibleClient] = None, *, require_llm: bool = False):
        self.llm = llm
        self.require_llm = require_llm

    def diagnose(
        self,
        *,
        round_id: int,
        monitor_snapshot: Optional[Mapping[str, Any]] = None,
        metrics_summary: Optional[Mapping[str, Any]] = None,
        evaluation_summary: Optional[Mapping[str, Any]] = None,
        structural_analysis: Optional[Mapping[str, Any]] = None,
        job_history: Optional[Sequence[Mapping[str, Any]]] = None,
        config: Optional[Mapping[str, Any]] = None,
        active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
        candidate_clusters: Optional[Mapping[str, Any]] = None,
    ) -> DiagnosticReport:
        """Produce a diagnostic report from available pipeline state."""
        context = self._build_context(
            round_id=round_id,
            monitor_snapshot=monitor_snapshot,
            metrics_summary=metrics_summary,
            evaluation_summary=evaluation_summary,
            structural_analysis=structural_analysis,
            job_history=job_history,
            config=config,
            candidate_clusters=candidate_clusters,
        )
        if self.require_llm and not (self.llm and self.llm.available()):
            raise RuntimeError(
                "DiagnosticCoachAgent: --require-llm is set but no LLM endpoint is available. "
                "Cannot fall back to deterministic rules."
            )
        if self.llm and self.llm.available():
            try:
                result = self.llm.chat_json(
                    system=compose_agent_system(
                        self.SYSTEM,
                        active_skills=active_skills,
                    ),
                    user={"round_id": round_id, "pipeline_state": context},
                    temperature=0.15,
                    max_tokens=8000,
                )
            except (LLMConfigError, LLMTransportError):
                if self.require_llm:
                    raise
                fallback = self._deterministic_diagnosis(round_id, context)
                fallback.raw = {"llm_error": "transport_or_config", "source": "deterministic_fallback_after_llm_error"}
                return fallback
            except Exception as exc:
                if self.require_llm:
                    raise
                fallback = self._deterministic_diagnosis(round_id, context)
                fallback.raw = {"llm_error": str(exc), "source": "deterministic_fallback_after_llm_error"}
                return fallback
            if isinstance(result, dict) and "corrective_actions" in result:
                corrective_actions = self._sanitize_corrective_actions(list(result.get("corrective_actions") or []))
                facts = (context.get("evaluation") or {}).get("metric_facts") or context.get("metrics_summary") or {}
                repaired, validation = self._repair_fact_invalid_fields(
                    round_id=round_id, context=context, result=result, facts=facts
                )
                corrective_actions = self._sanitize_corrective_actions(list(repaired.get("corrective_actions") or []))
                return DiagnosticReport(
                    round_id=round_id,
                    llm_used=True,
                    status_diagnosis=str(repaired.get("status_diagnosis", "")),
                    root_causes=list(repaired.get("root_causes") or []),
                    metric_interpretation=dict(repaired.get("metric_interpretation") or {}),
                    corrective_actions=corrective_actions,
                    monitoring_recommendations=list(repaired.get("monitoring_recommendations") or []),
                    pipeline_health=dict(repaired.get("pipeline_health") or {}),
                    raw={
                        **repaired,
                        **validation,
                        "facts_used": facts,
                        "context_digest": context_digest(context),
                    },
                )
            # LLM returned something unparseable - fallback
            fallback = self._deterministic_diagnosis(round_id, context)
            fallback.raw = {"llm_parse_failed": result}
            return fallback
        return self._deterministic_diagnosis(round_id, context)

    @classmethod
    def _repair_fact_invalid_fields(
        cls,
        *,
        round_id: int,
        context: Dict[str, Any],
        result: Mapping[str, Any],
        facts: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Repair only fields whose text contradicts immutable metric facts.

        Independent advice survives validation. Rejected claims and replacement
        provenance remain machine-auditable in ``raw``.
        """
        fallback = cls._deterministic_diagnosis(round_id, context)
        fallback_values = {
            "status_diagnosis": fallback.status_diagnosis,
            "root_causes": fallback.root_causes,
            "metric_interpretation": fallback.metric_interpretation,
            "corrective_actions": fallback.corrective_actions,
            "monitoring_recommendations": fallback.monitoring_recommendations,
            "pipeline_health": fallback.pipeline_health,
        }
        repaired = dict(result)
        repaired_fields: List[str] = []
        rejected_claims: List[Dict[str, Any]] = []
        validated_core: Dict[str, Any] = {}
        for field_name in fallback_values:
            value = repaired.get(field_name)
            issues = fact_check_text_against_metric_facts(
                json.dumps(value, ensure_ascii=False, default=str), facts
            )
            if issues:
                rejected_claims.extend(
                    {"field": field_name, "claim": issue} for issue in issues
                )
                repaired[field_name] = fallback_values[field_name]
                repaired_fields.append(field_name)
            else:
                validated_core[field_name] = value
        return repaired, {
            "validation_source": "field_level_fact_repair",
            "validated_core": validated_core,
            "repaired_fields": repaired_fields,
            "rejected_claims": rejected_claims,
        }

    @staticmethod
    def _sanitize_corrective_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for action in actions:
            item = dict(action or {})
            changes = dict(item.get("parameter_changes") or {})
            changes, sampler_ignored = strip_probabilistic_sampler_keys(changes)
            ignored = sampler_ignored + unsupported_config_keys(changes) + invalid_config_value_keys(changes)
            item["parameter_changes"] = supported_config_changes(changes)
            if ignored:
                item["ignored_parameter_changes"] = ignored
            sanitized.append(item)
        return sanitized

    def write_report(self, report: DiagnosticReport, path: Union[str, Path]) -> Path:
        return atomic_write_json(path, asdict(report))

    @staticmethod
    def _build_context(
        *,
        round_id: int,
        monitor_snapshot: Optional[Mapping[str, Any]],
        metrics_summary: Optional[Mapping[str, Any]],
        evaluation_summary: Optional[Mapping[str, Any]],
        structural_analysis: Optional[Mapping[str, Any]],
        job_history: Optional[Sequence[Mapping[str, Any]]],
        config: Optional[Mapping[str, Any]],
        candidate_clusters: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Delegate to the shared compactor so heavy/unbounded fields are stripped
        # consistently and the request stays well under the context window.
        return compact_context_for_diagnostic(
            round_id=round_id,
            monitor_snapshot=monitor_snapshot,
            metrics_summary=metrics_summary,
            evaluation_summary=evaluation_summary,
            structural_analysis=structural_analysis,
            job_history=job_history,
            config=config,
            candidate_clusters=candidate_clusters,
        )

    @staticmethod
    def _deterministic_diagnosis(round_id: int, context: Dict[str, Any]) -> DiagnosticReport:
        """Rule-based fallback when LLM is unavailable."""
        evaluation = context.get("evaluation") or {}
        monitor = context.get("monitor") or {}
        metrics = context.get("metrics_summary") or {}

        root_causes: List[Dict[str, Any]] = []
        corrective_actions: List[Dict[str, Any]] = []
        pipeline_health: Dict[str, Any] = {
            "execution_ok": True,
            "design_generating": True,
            "filtering_working": True,
            "interface_forming": False,
            "hotspot_engaging": False,
            "ready_for_next_round": False,
            "recommended_wait_seconds": 0,
        }

        total = int(evaluation.get("total_candidates") or 0)
        success = int(evaluation.get("success_count") or 0)
        tags = dict(evaluation.get("tag_counts") or {})

        # Check execution status
        if monitor.get("state") and not monitor.get("is_terminal"):
            pipeline_health["execution_ok"] = False
            pipeline_health["recommended_wait_seconds"] = 120
            root_causes.append({
                "cause": "Job still running or in queue",
                "evidence": [f"state={monitor.get('state')}"],
                "confidence": 0.9,
                "category": "resource",
            })

        # Use metrics_summary for accurate iptm assessment (raw design_to_target_iptm)
        avg_iptm = 0.0
        avg_plddt = 0.0

        if total == 0:
            pipeline_health["design_generating"] = False
            root_causes.append({
                "cause": "No candidates generated",
                "evidence": ["total_candidates=0"],
                "confidence": 0.85,
                "category": "execution",
            })
            corrective_actions.append({
                "action": "Check Taiji logs for execution errors; verify boltzgen steps completed",
                "parameter_changes": {"diffusion_batch_size": 1},
                "priority": "critical",
                "expected_improvement": "At least see intermediate outputs for diagnosis",
                "risk": "None - diagnostic only",
            })
        else:
            # Use metrics_summary (raw iptm values) if available, otherwise fall back to evaluation metrics
            if metrics and metrics.get("iptm"):
                avg_iptm = float(metrics["iptm"].get("mean", 0))
                any_above_04 = bool(metrics.get("any_iptm_above_0.4"))
                any_above_03 = bool(metrics.get("any_iptm_above_0.3"))
            else:
                # Fall back to evaluation metrics (which may use refolded iptm)
                iptm_values = []
                for cand in evaluation.get("top_candidates", []):
                    m = cand.get("metrics") or {}
                    iptm_values.append(float(m.get("interface_confidence", 0)))
                avg_iptm = sum(iptm_values) / max(1, len(iptm_values))
                any_above_04 = any(v > 0.4 for v in iptm_values)
                any_above_03 = any(v > 0.3 for v in iptm_values)

            if metrics and metrics.get("plddt"):
                avg_plddt = float(metrics["plddt"].get("mean", 0))
            else:
                plddt_values = []
                for cand in evaluation.get("top_candidates", []):
                    m = cand.get("metrics") or {}
                    plddt_values.append(float(m.get("binder_plddt", 0)))
                avg_plddt = sum(plddt_values) / max(1, len(plddt_values))

            # Key check: if NO candidates pass AND success_count==0, something is wrong
            if success == 0:
                # No pass candidates at all - binding is fundamentally failing
                pipeline_health["interface_forming"] = False
                binding_tag_count = tags.get("binding_pose_failure", 0)
                root_causes.append({
                    "cause": f"All {total} candidates fail compute gates (0 pass); binding_pose_failure={binding_tag_count}/{total}",
                    "evidence": [
                        f"success_count=0/{total}",
                        f"avg_iptm={avg_iptm:.3f} (design_to_target_iptm)",
                        f"binding_pose_failure={binding_tag_count}",
                        f"tags={tags}",
                    ],
                    "confidence": 0.85,
                    "category": "design_quality",
                })
                corrective_actions.append({
                    "action": "Strengthen interface conditioning with allowed BoltzGen knobs",
                    "parameter_changes": {
                        "hotspot_weight": 2.0,
                        "diffusion_batch_size": 1,
                    },
                    "priority": "high",
                    "expected_improvement": "iptm should rise; at least some candidates should pass",
                    "risk": "Longer binders may fold less reliably",
                })

            if avg_iptm < 0.2:
                pipeline_health["interface_forming"] = False
                if not any(rc.get("category") == "design_quality" for rc in root_causes):
                    root_causes.append({
                        "cause": "Binders fold but do not bind target (raw design_to_target_iptm < 0.2)",
                        "evidence": [f"avg_design_to_target_iptm={avg_iptm:.3f}"],
                        "confidence": 0.8,
                        "category": "design_quality",
                    })

            if avg_plddt > 0.7:
                pipeline_health["design_generating"] = True

            if tags.get("hotspot_miss", 0) > total * 0.3 or (metrics and not metrics.get("any_iptm_above_0.3")):
                pipeline_health["hotspot_engaging"] = False
                root_causes.append({
                    "cause": "Hotspot residues not contacted by binder",
                    "evidence": [f"hotspot_miss={tags.get('hotspot_miss', 0)}/{total}", f"no_iptm_above_0.3={not any_above_03}"],
                    "confidence": 0.75,
                    "category": "constraint",
                })
                corrective_actions.append({
                    "action": "Increase hotspot weight and expand hotspot conditioning patch",
                    "parameter_changes": {"hotspot_weight": 2.5, "binder_chain": "D"},
                    "priority": "high",
                    "expected_improvement": "Hotspot contact should improve",
                    "risk": "Over-constraining may reduce diversity",
                })

            # Clash analysis
            if tags.get("clash", 0) > total * 0.3:
                root_causes.append({
                    "cause": "High clash rate in generated designs",
                    "evidence": [f"clash={tags.get('clash', 0)}/{total}"],
                    "confidence": 0.7,
                    "category": "design_quality",
                })
                corrective_actions.append({
                    "action": "Enable clash-aware filtering and reduce packing density",
                    "parameter_changes": {"clash_filter": True},
                    "priority": "medium",
                    "expected_improvement": "Fewer clashing designs in final set",
                    "risk": "May discard some salvageable candidates",
                })

            if success > 0:
                pipeline_health["ready_for_next_round"] = True
                pipeline_health["interface_forming"] = True

        status = f"Round {round_id}: {total} candidates, {success} pass. "
        if total == 0:
            status += "Critical: no candidates generated at all."
        elif not pipeline_health["interface_forming"]:
            status += "Critical: binders fold but do not form target interface."
        elif not pipeline_health["hotspot_engaging"]:
            status += "Moderate: interface forming but hotspots not engaged."
        elif pipeline_health["ready_for_next_round"]:
            status += "Ready for next round with exploitation of successful seeds."
        else:
            status += "Intermediate state; collect more data."

        return DiagnosticReport(
            round_id=round_id,
            llm_used=False,
            status_diagnosis=status,
            root_causes=root_causes,
            metric_interpretation={
                "iptm_assessment": f"avg={avg_iptm:.3f}",
                "plddt_assessment": f"avg={avg_plddt:.3f}",
                "overall_binding_quality": "none" if total == 0 else ("weak" if avg_iptm < 0.2 else ("moderate" if avg_iptm < 0.4 else "strong")),
                "key_bottleneck": "no_candidates" if total == 0 else ("interface_engagement" if success == 0 else "optimization"),
            },
            corrective_actions=corrective_actions,
            monitoring_recommendations=[],
            pipeline_health=pipeline_health,
            raw={"source": "deterministic_fallback"},
        )
