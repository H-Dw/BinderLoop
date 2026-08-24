"""Performance-triggered, evidence-bounded multi-agent quality analysis."""

import concurrent.futures
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from binderloop.agents.binder_quality_analysis_agent import (
    BinderQualityAnalysis,
    BinderQualityAnalysisAgent,
)
from binderloop.agents.context_compaction import (
    compact_context_for_quality,
    context_digest,
)
from binderloop.llm import OpenAICompatibleClient
from binderloop.resume import atomic_write_json, stable_hash
from binderloop.skills import compose_agent_system
from binderloop.structured_llm import call_structured_json


QUALITY_MANAGER_CONFIG_KEYS = frozenset({
    "hotspot_weight",
    "diffusion_batch_size",
    "step_scale",
    "noise_scale",
    "alpha",
    "clash_filter",
    "prioritize_hotspots",
    "auxiliary_hotspots",
    "module_guided_repair",
    "epitope_crop_mode",
    "template_conditioned_fraction",
    "binder_lengths",
})


@dataclass
class FinalStrategyDecision:
    round_id: int
    selected_arm_id: Optional[str]
    update_direction: str
    accepted_evidence_ids: List[str] = field(default_factory=list)
    rejected_evidence_ids: List[str] = field(default_factory=list)
    physical_rationale: str = ""
    risks: List[str] = field(default_factory=list)
    uncertainty: List[str] = field(default_factory=list)
    biochemical_assessment: str = "not_assessed"
    developability_assessment: str = "not_assessed"
    executable_config_update: Dict[str, Any] = field(default_factory=dict)
    llm_used: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass
class QualitySpecialistBatch:
    """Frozen specialist inputs so Wave A can run roles without the manager."""

    round_id: int
    compact: Dict[str, Any]
    packets: Dict[str, Any]
    registry: Dict[str, Any]
    roles: Tuple[str, ...]
    mode_decision: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    telemetry: List[Dict[str, Any]] = field(default_factory=list)
    fallback_analysis: Optional[BinderQualityAnalysis] = None


@dataclass
class QualityAnalysisModeDecision:
    mode: str
    reason: str
    active: bool
    current_reward: Optional[float]
    previous_reward: Optional[float]
    recovery_target_reward: Optional[float]
    trigger_round_id: Optional[int] = None
    trigger_reasons: List[Dict[str, Any]] = None
    failure_signature: Optional[str] = None
    consecutive_multi_rounds: int = 0
    exit_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "reason": self.reason, "active": self.active,
            "current_reward": self.current_reward, "previous_reward": self.previous_reward,
            "recovery_target_reward": self.recovery_target_reward,
            "trigger_round_id": self.trigger_round_id,
            "trigger_reasons": list(self.trigger_reasons or []),
            "failure_signature": self.failure_signature,
            "consecutive_multi_rounds": self.consecutive_multi_rounds,
            "exit_reason": self.exit_reason,
        }


class QualityCollaborationController:
    """Adaptive, durable single↔multi quality-analysis state machine."""

    @staticmethod
    def decide(memory: Any, outcome: Mapping[str, Any], spec: Any,
               signals: Optional[Mapping[str, Any]] = None) -> QualityAnalysisModeDecision:
        state = dict(getattr(memory, "quality_collaboration_state", {}) or {})
        signals = dict(signals or {})
        round_id = int(outcome.get("round_id", -1))
        current = _optional_float(outcome.get("reward"))
        valid = not bool(outcome.get("execution_failed")) and current is not None
        active = bool(state.get("active"))
        prior = sorted([dict(x) for x in (getattr(memory, "round_metrics", []) or [])
                        if int(x.get("round_id", -1)) < round_id and not x.get("execution_failed")
                        and x.get("reward") is not None], key=lambda x: int(x.get("round_id", -1)))
        previous = _optional_float(prior[-1].get("reward")) if prior else None
        enabled = bool(getattr(spec, "enabled", False))
        baseline = _optional_float(state.get("trigger_baseline_reward"))
        ratio = min(.98, max(.95, float(getattr(spec, "recovery_ratio", .97) or .97)))
        target = baseline * ratio if baseline is not None else None

        if not enabled:
            return QualityCollaborationController._finish(memory, state, round_id,
                QualityAnalysisModeDecision("single", "quality collaboration disabled", False, current, previous, target))
        if not valid:
            decision = QualityAnalysisModeDecision("multi" if active else "single",
                "execution failure has no quality signal; state and counters preserved", active,
                current, previous, target, _optional_int(state.get("trigger_round_id")), [],
                state.get("failure_signature"), int(state.get("consecutive_multi_rounds", 0)))
            return QualityCollaborationController._finish(memory, state, round_id, decision, preserve=True)

        reasons = QualityCollaborationController._triggers(outcome, signals, previous, spec)
        signature = QualityCollaborationController._signature(outcome, signals)
        new_signature = bool(signature and signature != state.get("failure_signature"))
        high_value = any(r["code"] == "high_value_decision" for r in reasons)
        repeated = bool(signals.get("conclusion_repeated"))
        no_guidance = bool(signals.get("no_actionable_guidance"))
        count = int(state.get("consecutive_multi_rounds", 0))
        max_multi = max(1, int(getattr(spec, "max_consecutive_multi_rounds", 2) or 2))
        exit_reason = None

        if active:
            recovered = target is not None and current + float(getattr(spec, "recovery_tolerance", 1e-6) or 0) >= target
            if recovered: exit_reason = "trigger_baseline_recovered"
            elif repeated: exit_reason = "conclusion_repeated"
            elif no_guidance: exit_reason = "no_actionable_guidance"
            elif count >= max_multi and not (new_signature or high_value): exit_reason = "maximum_consecutive_multi_rounds"
            if exit_reason:
                decision = QualityAnalysisModeDecision("single", exit_reason, False, current, previous, target,
                    _optional_int(state.get("trigger_round_id")), [], signature, 0, exit_reason)
            else:
                count += 1
                decision = QualityAnalysisModeDecision("multi", "collaboration remains active", True, current,
                    previous, target, _optional_int(state.get("trigger_round_id")), reasons, signature, count)
        elif reasons:
            baseline = max([x for x in (previous, max((float(x["reward"]) for x in prior), default=None)) if x is not None], default=current)
            target = baseline * ratio if baseline is not None else None
            decision = QualityAnalysisModeDecision("multi", "; ".join(r["detail"] for r in reasons), True,
                current, previous, target, round_id, reasons, signature, 1)
            state["trigger_baseline_reward"] = baseline
        else:
            decision = QualityAnalysisModeDecision("single", "no adaptive collaboration trigger", False,
                current, previous, target, None, [], signature, 0)
        return QualityCollaborationController._finish(memory, state, round_id, decision)

    @staticmethod
    def _triggers(outcome, signals, previous, spec):
        result = []
        def add(code, detail, evidence=None):
            result.append({"code": code, "detail": detail, "evidence": dict(evidence or {})})
        current = _optional_float(outcome.get("reward"))
        band = max(float(getattr(spec, "reward_noise_band", .02) or 0),
                   float(getattr(spec, "performance_drop_tolerance", 0) or 0))
        if previous is not None and current is not None and previous-current > band:
            add("reward_credible_decline", "reward decline exceeded noise band", {"previous": previous, "current": current, "noise_band": band})
        for key, bad, threshold in (("compute_gate_yield", "lower", getattr(spec,"compute_gate_degradation_ratio",.9)),
                                    ("hotspot_yield", "lower", getattr(spec,"hotspot_degradation_ratio",.9)),
                                    ("mean_pae", "higher", getattr(spec,"pae_degradation_ratio",1.1))):
            cur, prev = _optional_float(signals.get(key)), _optional_float(signals.get("previous_"+key))
            degraded = prev is not None and cur is not None and ((bad=="lower" and cur < prev*float(threshold)) or (bad=="higher" and cur > prev*float(threshold)))
            if degraded: add("compute_gate_degradation", key+" degraded", {"previous":prev,"current":cur})
        if signals.get("metric_conflict"): add("metric_conflict", "decision metrics conflict", {"conflict": signals.get("metric_conflict")})
        if signals.get("zero_filter_pass_with_unfiltered_evidence"):
            add("zero_filter_pass_review", "zero filter pass with recoverable unfiltered quality evidence", {"unfiltered_evidence": True})
        high = list(signals.get("high_value_events") or [])
        if high: add("high_value_decision", "high-value decision pending", {"events": high})
        failures = list(signals.get("single_analysis_failures") or [])
        if failures: add("single_analysis_unreliable", "single-agent analysis requires verification", {"failures": failures})
        return result

    @staticmethod
    def _signature(outcome, signals):
        tags = signals.get("failure_tags") or outcome.get("failure_tags") or []
        if isinstance(tags, Mapping): tags = [k for k,v in tags.items() if v and not str(k).startswith("pass_")]
        if not tags: return None
        return hashlib.sha256(json.dumps(sorted(map(str,tags))).encode()).hexdigest()[:16]

    @staticmethod
    def _finish(memory, state, round_id, decision, preserve=False):
        history = list(state.get("history") or [])
        history.append({"round_id": round_id, **decision.to_dict()})
        if not preserve:
            state.update({"active":decision.active, "trigger_round_id":decision.trigger_round_id,
                "recovery_target_reward":decision.recovery_target_reward,
                "failure_signature":decision.failure_signature,
                "consecutive_multi_rounds":decision.consecutive_multi_rounds,
                "last_mode":decision.mode, "last_reason":decision.reason})
            if not decision.active: state.pop("trigger_baseline_reward", None)
        state.update({"last_round_id":round_id, "history":history[-50:]})
        memory.quality_collaboration_state = state
        return decision


class BinderQualityCollaborationAgent:
    """Three evidence specialists plus one physical-consistency manager."""

    POSITIVE_SYSTEM = """You are SuccessMechanismAgent. Return JSON only: {"findings":[{"finding_id":"P1","statement":"short evidence-bound statement","scope":"local_fragment|whole_binder|population","signal":"reusable|non_reusable|uncertain","evidence_ids":[],"counterevidence_ids":[],"confidence":0.0}]}. Output only findings, at most 3. Do not output parameters, config, recommendations, mechanisms, risks, or unrelated knowledge. Near misses are not successes and local fragments are not whole-binder evidence."""

    NEGATIVE_SYSTEM = """You are FailureMechanismAgent. Return JSON only: {"findings":[{"finding_id":"N1","statement":"short evidence-bound statement","scope":"candidate|cluster|population","failure_type":"pose|pae|hotspot|foldability|clash|filtering|uncertain","repair_family":"target_context|sampler|length|selection|none","evidence_ids":[],"counterevidence_ids":[],"confidence":0.0}]}. Output only findings, at most 3. Do not output parameter names, values, config, recommendations, risks, or unrelated knowledge."""

    TRAJECTORY_SYSTEM = """You are TrajectoryMemoryAgent. Return JSON only: {"findings":[{"finding_id":"T1","statement":"short evidence-bound lesson","scope":"trajectory","source_round_ids":[],"family":"sampler|length|target_context|template|selection|none","outcome":"improved|regressed|mixed|not_identifiable","causal_strength":"none|weak|moderate","evidence_ids":[],"counterevidence_ids":[],"confidence":0.0}]}. Output only findings, at most 3. Do not output concrete parameter values or config. Every numeric statement must cite source rounds; observational multi-parameter changes are not controlled causality."""

    FINAL_STRATEGY_SYSTEM = """You are the existing PhysicsDebateManager making the final biochemical/biophysical arm decision. Return JSON only with exactly: {"selected_arm_id":null,"update_direction":"preserve_winner|explore_alternative|hold","accepted_evidence_ids":[],"rejected_evidence_ids":[],"physical_rationale":"...","risks":[],"uncertainty":[],"biochemical_assessment":"favorable|unfavorable|mixed|not_assessed","developability_assessment":"favorable|unfavorable|mixed|not_assessed"}. Select only the arm authorized by historical resolution or hold. Cite only supplied evidence IDs. Do not emit config or parameter changes. Unmeasured biochemical/developability properties must be not_assessed."""

    MANAGER_SYSTEM = """You are PhysicsDebateManager. Return JSON only: {"accepted_finding_ids":[],"rejected_finding_ids":[],"strategy_intents":[{"intent_id":"I1","kind":"preserve|relax|explore|repair|hold","evidence_ids":[],"expected_signal":"short text","risk":"short text","priority":1}],"uncertainties":[]}. Output only these fields and at most 3 strategy intents. Do not output parameter names, values, config objects, claim rewrites, revisions, or final analysis."""

    def __init__(
        self,
        llm: Optional[OpenAICompatibleClient],
        *,
        request_timeout_seconds: int = 105,
        failure_cooldown_seconds: int = 20,
        specialist_max_tokens: int = 1400,
        manager_max_tokens: int = 1800,
        reasoning_budget_tokens: int = 0,
        visible_json_budget_tokens: int = 8000,
        max_completion_tokens: int = 65_536,
        specialist_reasoning_mode: str = "low",
        specialist_output_tokens: Optional[int] = None,
        manager_reasoning_mode: str = "low",
        manager_output_tokens: Optional[int] = None,
        max_revisions: Optional[int] = None,
        final_max_tokens: Optional[int] = None,
        max_api_calls: int = 6,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.request_timeout_seconds = max(30, int(request_timeout_seconds))
        self.failure_cooldown_seconds = max(
            0,
            int(failure_cooldown_seconds),
        )
        self.specialist_max_tokens = max(256, int(specialist_output_tokens or specialist_max_tokens))
        self.manager_max_tokens = max(256, int(manager_output_tokens or manager_max_tokens))
        self.specialist_reasoning_mode = str(specialist_reasoning_mode or "low")
        self.manager_reasoning_mode = str(manager_reasoning_mode or "low")
        self.reasoning_budget_tokens = max(0, int(reasoning_budget_tokens or 0))
        self.visible_json_budget_tokens = max(1024, int(visible_json_budget_tokens or 8000))
        self.max_completion_tokens = min(65_536, max(self.visible_json_budget_tokens, int(max_completion_tokens or 65_536)))
        self.deprecated_options_ignored = {key: value for key, value in {"max_revisions": max_revisions, "final_max_tokens": final_max_tokens}.items() if value is not None}
        self.max_api_calls = max(1, int(max_api_calls))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def final_strategy_decision(self, *, round_id: int, arm_comparison: Mapping[str, Any], history_resolution: Optional[Mapping[str, Any]] = None, history_evidence: Sequence[Mapping[str, Any]] = (), measured_assessments: Optional[Mapping[str, Any]] = None) -> FinalStrategyDecision:
        comparison=dict(arm_comparison or {}); history=dict(history_resolution or {}); measured=dict(measured_assessments or {})
        current_ids={str(x) for x in comparison.get("evidence_ids",[]) or []}; history_ids={str(x) for x in history.get("accepted_history_evidence_ids",[]) or []}
        registry=current_ids|history_ids; closed={str(x) for x in comparison.get("closed_arm_ids",[]) or []}
        authorized=str(history.get("selected_arm_id") or "") or None
        selected=authorized if comparison.get("status")=="winner" and history.get("action")=="select" and authorized in closed else None
        direction=str(history.get("update_direction") or "hold") if selected else "hold"
        structure_chemistry=dict(measured.get("structure_interface_chemistry") or {})
        biochemical_measured=bool(measured.get("biochemical_measured") or structure_chemistry.get("biochemical_measured"))
        developability_measured=bool(measured.get("developability_measured") or structure_chemistry.get("developability_measured"))
        fallback=FinalStrategyDecision(round_id,selected,direction,sorted(registry),[],"Validated current-arm comparison and historical ledger agree on the selected arm." if selected else "Current and historical evidence do not support a unique safe arm direction; hold.",[str(x) for x in comparison.get("negative_differences",[]) or []],sorted({str(x) for x in comparison.get("confounders",[]) or []}|{str(x) for x in history.get("conflicts",[]) or []}),str(measured.get("biochemical_assessment") or "not_assessed") if biochemical_measured else "not_assessed",str(measured.get("developability_assessment") or "not_assessed") if developability_measured else "not_assessed",{},False,{"source":"deterministic_physics_manager_decision","history_resolution":history,"preferred_arm_linkage":selected,"measured_structure_interface_chemistry":structure_chemistry})
        if not (self.llm and self.llm.available()): return fallback
        payload={"round_id":round_id,"current_completed_arm_comparison":comparison,"historical_resolution":history,"authorized_selected_arm_id":selected,"closed_arm_ids":sorted(closed),"validated_evidence_ids":sorted(registry),"measured_structure_interface_chemistry":structure_chemistry,"biochemical_measured":biochemical_measured,"developability_measured":developability_measured}
        telemetry=[]
        result=self._call(role="final_strategy_decision",system=self.FINAL_STRATEGY_SYSTEM,payload=payload,max_tokens=min(self.manager_max_tokens,self.visible_json_budget_tokens),reasoning_mode=self.manager_reasoning_mode,telemetry=telemetry)
        required={"selected_arm_id","update_direction","accepted_evidence_ids","rejected_evidence_ids","physical_rationale","risks","uncertainty","biochemical_assessment","developability_assessment"}
        if not isinstance(result,Mapping) or set(result)!=required: return fallback
        proposed=str(result.get("selected_arm_id") or "") or None; update=str(result.get("update_direction") or "")
        accepted=[str(x) for x in result.get("accepted_evidence_ids",[]) or []]; rejected=[str(x) for x in result.get("rejected_evidence_ids",[]) or []]
        assessments={"favorable","unfavorable","mixed","not_assessed"}
        valid=(update in {"preserve_winner","explore_alternative","hold"} and set(accepted+rejected).issubset(registry) and ((selected is None and proposed is None and update=="hold") or (selected is not None and proposed==selected and proposed in closed)) and isinstance(result.get("physical_rationale"),str) and bool(result.get("physical_rationale")) and isinstance(result.get("risks"),list) and isinstance(result.get("uncertainty"),list) and result.get("biochemical_assessment") in assessments and result.get("developability_assessment") in assessments)
        if not valid: return fallback
        biochemical=str(result.get("biochemical_assessment")) if biochemical_measured else "not_assessed"
        developability=str(result.get("developability_assessment")) if developability_measured else "not_assessed"
        return FinalStrategyDecision(round_id,proposed,update,accepted,rejected,str(result["physical_rationale"]),[str(x) for x in result["risks"]][:8],[str(x) for x in result["uncertainty"]][:8],biochemical,developability,{},True,{"source":"validated_llm_physics_manager_decision","context_digest":stable_hash(payload),"telemetry":telemetry,"preferred_arm_linkage":proposed})

    def prepare_specialists(
        self,
        *,
        round_id: int,
        context: Mapping[str, Any],
        memory: Any,
        mode_decision: Mapping[str, Any],
    ) -> QualitySpecialistBatch:
        compact = compact_context_for_quality(context)
        packets = self._build_packets(
            round_id=round_id, compact=compact, memory=memory,
            mode_decision=mode_decision, round_outcome=_as_dict(context.get("reward")),
        )
        registry = _evidence_registry(packets)
        roles = ("positive", "negative", "trajectory")[:max(0, min(3, self.max_api_calls))]
        batch = QualitySpecialistBatch(
            round_id=int(round_id), compact=dict(compact), packets=packets,
            registry=registry, roles=roles, mode_decision=dict(mode_decision or {}),
        )
        if not (self.llm and self.llm.available()):
            batch.fallback_analysis = self._fallback(
                round_id, compact, mode_decision, reason="llm_unavailable",
                collaboration=_delivery_metadata("fallback", [], False, registry),
            )
        return batch

    def run_specialist(
        self,
        batch: QualitySpecialistBatch,
        role: str,
    ) -> tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Wave A specialist call. Packets are frozen; roles never read each other."""
        systems = {"positive": self.POSITIVE_SYSTEM, "negative": self.NEGATIVE_SYSTEM,
                   "trajectory": self.TRAJECTORY_SYSTEM}
        role_telemetry: List[Dict[str, Any]] = []
        if role not in systems or role not in batch.packets:
            return role, None, role_telemetry
        result = self._call(
            role=role,
            system=compose_agent_system(
                systems[role],
                active_skills=_skills_for_role(batch.compact.get("active_skills"), role),
                role=role, max_directives=3,
            ),
            payload=batch.packets[role],
            max_tokens=min(self.specialist_max_tokens, self.visible_json_budget_tokens),
            reasoning_mode=self.specialist_reasoning_mode, telemetry=role_telemetry,
        )
        normalized = _normalize_specialist_output(result, role=role) if result else None
        return role, normalized, role_telemetry

    def absorb_specialist_results(
        self,
        batch: QualitySpecialistBatch,
        specialist_results: Mapping[str, tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]],
    ) -> QualitySpecialistBatch:
        """Merge Wave A specialist results in role order so artifacts stay deterministic."""
        for role in batch.roles:
            result, role_telemetry = specialist_results.get(role, (None, []))
            batch.telemetry.extend(list(role_telemetry or []))
            if result:
                batch.outputs[role] = result
        return batch

    def assemble_with_manager(self, batch: QualitySpecialistBatch) -> BinderQualityAnalysis:
        """Wave B manager arbitration over frozen specialist findings."""
        if batch.fallback_analysis is not None:
            return batch.fallback_analysis
        outputs = dict(batch.outputs)
        telemetry = list(batch.telemetry)
        registry = batch.registry
        packets = batch.packets
        compact = batch.compact
        mode_decision = batch.mode_decision
        cited = set()
        for output in outputs.values():
            for claim in output.get("claims") or []:
                cited.update(str(x) for x in claim.get("evidence_ids") or [])
                cited.update(str(x) for x in claim.get("counterevidence_ids") or [])
        manager_slice = {key: registry[key] for key in sorted(cited) if key in registry}
        validation = _validate_claims(outputs, registry, packets["physics"])
        deterministic = _deterministic_arbitrate(outputs, packets)
        conflicts = deterministic.get("conflicts") or []
        manager: Optional[Dict[str, Any]] = {"strategy_intents": [], "manager_uncertainties": []}
        llm_manager_used = False
        if outputs:
            llm_result = self._call(
                role="manager_deliberation",
                system=compose_agent_system(
                    self.MANAGER_SYSTEM,
                    active_skills=_skills_for_role(compact.get("active_skills"), "manager"), role="manager", max_directives=3,
                ),
                payload={"specialist_findings": outputs, "evidence_registry": manager_slice,
                         "three_class_evidence": packets["manager_evidence_classes"],
                         "detected_conflicts": conflicts},
                max_tokens=min(self.manager_max_tokens, self.visible_json_budget_tokens), reasoning_mode=self.manager_reasoning_mode, telemetry=telemetry,
            )
            if llm_result:
                known_findings = [str(claim.get("claim_id")) for value in outputs.values() for claim in value.get("claims", [])]
                manager = _normalize_manager_output(llm_result, registry, known_findings)
                llm_manager_used = True
        accepted = _accepted_claims(outputs, deterministic, validation)
        grade = _collaboration_grade(len(outputs), llm_manager_used)
        if not outputs:
            return self._fallback(batch.round_id, compact, mode_decision, reason="all_specialists_unavailable",
                                  collaboration=_delivery_metadata(grade, [], False, registry, telemetry))
        analysis = _assemble_analysis(batch.round_id, compact, accepted, outputs, manager)
        analysis.raw.update({
            "source": "deterministic_collaboration_assembler",
            "quality_analysis_mode": dict(mode_decision),
            "facts_used": dict(_as_dict(compact.get("evaluation")).get("metric_facts") or {}),
            "context_digest": context_digest(compact),
            "collaboration": {
                **_delivery_metadata(grade, sorted(outputs), llm_manager_used, registry, telemetry),
                "specialist_outputs": outputs, "manager_deliberation": manager,
                "deprecated_options_ignored": dict(self.deprecated_options_ignored),
                "manager_evidence_ids": sorted(manager_slice), "claim_validation": validation,
            },
        })
        return analysis

    def analyze(
        self,
        *,
        round_id: int,
        context: Mapping[str, Any],
        memory: Any,
        mode_decision: Mapping[str, Any],
    ) -> BinderQualityAnalysis:
        batch = self.prepare_specialists(
            round_id=round_id, context=context, memory=memory, mode_decision=mode_decision,
        )
        if batch.fallback_analysis is not None:
            return batch.fallback_analysis
        specialist_results: Dict[str, tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(batch.roles) or 1)) as pool:
            futures = {pool.submit(self.run_specialist, batch, role): role for role in batch.roles}
            for future in concurrent.futures.as_completed(futures):
                role, result, role_telemetry = future.result()
                specialist_results[role] = (result, role_telemetry)
        self.absorb_specialist_results(batch, specialist_results)
        return self.assemble_with_manager(batch)

    def _call(self, *, role: str, system: str, payload: Mapping[str, Any],
              max_tokens: int, reasoning_mode: str, telemetry: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if len(telemetry) >= self.max_api_calls or not self.llm:
            return None
        model = getattr(getattr(self.llm, "settings", None), "default_model", "unknown")
        prompt_version = "collaboration-delivery-v3-structured"
        key = stable_hash({"role": role, "payload": payload, "model": model,
                           "prompt": system, "prompt_version": prompt_version})
        cache_path = self.cache_dir / (key + ".json") if self.cache_dir else None
        if cache_path and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                telemetry.append({"role": role, "ok": True, "cache_hit": True, "cache_key": key,
                                  "reasoning_tokens": 0, "visible_output_tokens": 0,
                                  "parse_status": "cache_hit", "finish_reason": "cached"})
                return _as_dict(cached.get("result"))
            except Exception:
                pass
        required_by_role = {
            "positive": ("findings",), "negative": ("findings",), "trajectory": ("findings",),
            "manager_deliberation": ("accepted_finding_ids","rejected_finding_ids","strategy_intents","uncertainties"),
            "final_strategy_decision": ("selected_arm_id","update_direction","accepted_evidence_ids","rejected_evidence_ids","physical_rationale","risks","uncertainty","biochemical_assessment","developability_assessment"),
        }
        required = required_by_role.get(role, ())
        outcome = call_structured_json(
            self.llm, system=system, user=payload, required_fields=required,
            temperature=0.1, max_completion_tokens=self.max_completion_tokens,
            visible_json_tokens=max(1024, int(max_tokens)), thinking=reasoning_mode,
            reasoning_budget_tokens=self.reasoning_budget_tokens or None, repair=True,
        )
        for attempt in outcome.attempts:
            telemetry.append({"role": role, "cache_hit": False, "cache_key": key,
                              "parse_status": "ok" if attempt.get("ok") else ("invalid_json" if attempt.get("parse_error") else "invalid_schema"),
                              **attempt})
        if outcome.value is not None and cache_path:
            atomic_write_json(cache_path, {"key": key, "role": role, "model": model,
                                           "prompt_version": prompt_version, "result": outcome.value,
                                           "attempts": outcome.attempts})
        return outcome.value

    def _build_packets(
        self,
        *,
        round_id: int,
        compact: Mapping[str, Any],
        memory: Any,
        mode_decision: Mapping[str, Any],
        round_outcome: Mapping[str, Any],
    ) -> Dict[str, Any]:
        evaluation = _as_dict(compact.get("evaluation"))
        examples = _as_dict(compact.get("active_learning_examples"))
        current_examples = _as_dict(examples.get("current_round"))
        strict_positives = [
            {"evidence_id": f"R{round_id}:STRICT_POS:{index + 1}", **{k: v for k, v in dict(item).items() if k not in {"candidate_id", "source"}}}
            for index, item in enumerate(current_examples.get("strict_positive_examples")
                                         or current_examples.get("positive_examples") or [])
            if str(item.get("label") or "strict_positive") == "strict_positive"
        ]
        near_misses = [
            {"evidence_id": f"R{round_id}:NEAR_MISS:{index + 1}", **{k: v for k, v in dict(item).items() if k not in {"candidate_id", "source"}}}
            for index, item in enumerate(current_examples.get("near_miss_examples") or [])
        ]
        harness_successes = [{"evidence_id": f"R{round_id}:HARNESS_SUCCESS_COUNT", "count": evaluation.get("success_count", 0)}]
        positives = strict_positives
        provisional = []
        if not strict_positives:
            provisional = [
                {
                    **dict(item),
                    "evidence_role": "provisional_reference",
                    "success_counted": False,
                    "label": "near_miss",
                }
                for item in near_misses
            ]
        negative_source = list(
            current_examples.get("other_negative_examples")
            or current_examples.get("hard_negative_examples")
            or []
        )
        negatives = [
            {"evidence_id": f"R{round_id}:OTHER_NEG:{index + 1}",
             "failure_cluster": _failure_cluster(item), **{k: v for k, v in dict(item).items() if k not in {"candidate_id", "source"}}}
            for index, item in enumerate(negative_source)
        ]
        summaries = [
            _collaboration_structure_evidence(round_id, index + 1, item)
            for index, item in enumerate(
                _as_dict(compact.get("structural_analysis")).get("summaries")
                or []
            )
        ]
        # Structures and metric examples are independent aggregate samples.
        positive_structures = summaries
        local_only_structures = summaries
        current_metrics = {
            "evidence_id": f"R{round_id}:METRICS",
            "round_id": round_id,
            "metric_facts": evaluation.get("metric_facts"),
            "tag_counts": evaluation.get("tag_counts"),
            "candidate_filtering": evaluation.get("candidate_filtering"),
            "outcome": dict(round_outcome),
        }
        history = _history_cards(memory, before_round_id=round_id)
        previous = history[-1] if history else {}
        current_config = dict(compact.get("current_config") or {})
        trajectory_views = _trajectory_views(history, current_config=current_config)
        current_strict_population = _strict_positive_population(
            round_id, strict_positives
        )
        previous_strict_population = _as_dict(previous.get("strict_positive_population"))
        historical_best_card = _as_dict(trajectory_views.get("historical_best"))
        historical_best_strict_population = _as_dict(
            historical_best_card.get("strict_positive_population")
        )
        # Only the trajectory specialist receives detailed historical cards. Other
        # specialists operate on current-round evidence, and the manager receives
        # only evidence IDs actually cited by validated findings.
        previous_config = dict(previous.get("config") or {})
        referenced_rounds: Dict[str, Any] = {}
        for label, card in (("previous", previous), ("historical_best", historical_best_card),
                            ("same_config_best", _as_dict(trajectory_views.get("same_config_best"))),
                            ("same_config_worst", _as_dict(trajectory_views.get("same_config_worst")))):
            if card:
                referenced_rounds[label] = {
                    "evidence_id": card.get("evidence_id"), "round_id": card.get("round_id"),
                    "reward": card.get("reward"), "best_iptm": card.get("best_iptm"),
                    "success_count": card.get("success_count"), "failure_tags": card.get("failure_tags"),
                }
        same_config = _as_dict(trajectory_views.get("same_config"))
        trajectory = {
            "cards": {
                "current": {"evidence_id": f"R{round_id}:TRAJECTORY:CURRENT", "round_id": round_id, "metrics": current_metrics, "config": {"evidence_id": f"R{round_id}:CONFIG", **current_config}, "strict_positive_population": current_strict_population},
                "previous": previous or None,
                "historical_best": historical_best_card or None,
                "same_config": same_config or None,
            },
            "previous_to_current_diff": {"evidence_id": f"R{round_id - 1}-R{round_id}:PARAM_DIFF", "changes": _parameter_diff(previous_config, current_config)},
            "quality_mode_decision": dict(mode_decision),
        }
        evidence_classes = {
            "strict_positive_examples": strict_positives,
            "near_miss_examples": near_misses,
            "other_negative_examples": negatives,
        }
        physics = {
            "immutable_current_round": current_metrics,
            "rules": [
                {
                    "evidence_id": "PHYS:LOCAL_GLOBAL",
                    "rule": (
                        "Local fragment quality cannot establish whole-binder success."
                    ),
                },
                {
                    "evidence_id": "PHYS:IPTM",
                    "rule": (
                        "Higher iPTM supports interface confidence but does not prove affinity."
                    ),
                },
                {
                    "evidence_id": "PHYS:PAE",
                    "rule": (
                        "Lower interface PAE supports better localized relative geometry."
                    ),
                },
                {
                    "evidence_id": "PHYS:RMSD",
                    "rule": (
                        "Low refold RMSD supports consistency, not binding by itself."
                    ),
                },
                {
                    "evidence_id": "PHYS:CAUSALITY",
                    "rule": (
                        "Observational multi-parameter rounds do not identify controlled causality."
                    ),
                },
            ],
            "executable_config_keys": sorted(
                set(current_config) & QUALITY_MANAGER_CONFIG_KEYS
            ),
            "evidence_taxonomy": {
                "strict_positive_count": len(strict_positives),
                "near_miss_count": len(near_misses),
                "other_negative_count": len(negatives),
                "near_miss_success_counted": False,
            },
        }
        return {
            "positive": {
                "current_metrics": current_metrics,
                "strict_metric_positives": strict_positives,
                "near_miss_boundary_examples": near_misses,
                "provisional_reference": provisional,
                "harness_successes": harness_successes,
                "whole_binder_structures": positive_structures[:6],
                "local_reusable_evidence_ids": [item.get("evidence_id") for item in local_only_structures[:4]],
            },
            "negative": {
                "current_metrics": current_metrics,
                "other_negative_examples": negatives,
            },
            "trajectory": trajectory,
            "physics": physics,
            "manager_evidence_classes": evidence_classes,
            "specialist_activation_audit": {
                "mode": str(mode_decision.get("mode") or "single"),
                "positive_input_count": len(strict_positives) + len(provisional),
                "negative_input_count": len(negatives),
                "trajectory_input_count": len(history),
                "provisional_reference_count": len(provisional),
                "specialists_expected": str(mode_decision.get("mode") or "single") == "multi",
                "inactive_reason": None if str(mode_decision.get("mode") or "single") == "multi" else "quality_collaboration_single_mode",
            },
        }

    @staticmethod
    def _fallback(
        round_id: int,
        compact: Mapping[str, Any],
        mode_decision: Mapping[str, Any],
        *,
        reason: str,
        collaboration: Optional[Mapping[str, Any]] = None,
    ) -> BinderQualityAnalysis:
        fallback = BinderQualityAnalysisAgent._fallback(round_id, compact)
        fallback.raw = {
            **dict(fallback.raw or {}),
            "source": "deterministic_fallback_after_multi_agent",
            "quality_analysis_mode": dict(mode_decision),
            "fallback_reason": reason,
            "collaboration": dict(collaboration or {}),
            "context_digest": context_digest(compact),
        }
        return fallback



def _collaboration_structure_evidence(round_id: int, index: int, item: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a path-free structure record with explicit candidate provenance."""
    source = dict(item)
    candidate_id = str(source.get("candidate_id") or "")
    result = {key: value for key, value in source.items() if key not in {"structure_file", "filename", "path", "messages", "history", "metrics"}}
    result["evidence_id"] = f"R{round_id}:STRUCT:{index}"
    if candidate_id:
        result["candidate_id"] = candidate_id
        result["candidate_evidence_ids"] = [f"R{round_id}:CANDIDATE:{candidate_id}"]
    return result


def _harness_success_candidate_ids(evaluation: Mapping[str, Any]) -> set:
    result = set()
    for item in list(evaluation.get("top_candidates") or []):
        row = _as_dict(item)
        tags = list(row.get("tags") or [])
        if tags == ["pass_compute_gate"] or bool(row.get("harness_success")):
            candidate_id = str(row.get("candidate_id") or row.get("name") or "")
            if candidate_id:
                result.add(candidate_id)
    return result


def _failure_cluster(item: Mapping[str, Any]) -> str:
    text = " ".join(map(str, [item.get("label_reason"), item.get("primary_failure"),
                              item.get("failure_reason"), item.get("tags"), item.get("margins")])).lower()
    for cluster, markers in (("high_pae", ("pae",)), ("hotspot", ("hotspot", "contact")),
                             ("foldability", ("fold", "rmsd", "ptm")),
                             ("clash", ("clash", "geometry", "steric")),
                             ("filtering", ("filter",)), ("pose", ("pose", "iptm", "interface"))):
        if any(marker in text for marker in markers):
            return cluster
    return "pose"


def _trajectory_views(history: Sequence[Mapping[str, Any]], *, current_config: Mapping[str, Any]) -> Dict[str, Any]:
    cards = [dict(item) for item in history]
    scored = [item for item in cards if _optional_float(item.get("reward")) is not None]
    same = [item for item in scored if dict(item.get("config") or {}) == dict(current_config)]
    key = lambda item: float(item.get("reward"))
    return {
        "most_recent": cards[-1] if cards else None,
        "historical_best": max(scored, key=key) if scored else None,
        "same_config": max(same, key=key) if same else None,
        "same_config_best": max(same, key=key) if same else None,
        "same_config_worst": min(same, key=key) if same else None,
    }


def _deterministic_arbitrate(outputs: Mapping[str, Any], packets: Mapping[str, Any]) -> Dict[str, Any]:
    registry = _collect_evidence_registry(packets)
    verdicts, conflicts = [], []
    seen_claims = []
    for role, output in outputs.items():
        for claim in _as_dict(output).get("claims") or []:
            row = dict(claim); claim_id = str(row.get("claim_id") or "")
            evidence = [str(value) for value in row.get("evidence_ids") or []]
            unknown = sorted(set(evidence) - set(registry))
            text = str(row.get("claim") or "").lower()
            scope = str(row.get("scope") or "")
            reason, verdict = "evidence registered", "accept"
            if unknown:
                verdict, reason = "reject", "unregistered evidence: " + ", ".join(unknown)
            elif scope == "local_fragment" and any(word in text for word in ("binder success", "whole-binder success", "affinity")):
                verdict, reason = "reject", "local evidence cannot establish whole-binder success"
            verdicts.append({"claim_id": claim_id, "verdict": verdict, "reason": reason,
                             "trusted_evidence_ids": [eid for eid in evidence if eid in registry]})
            for prior_role, prior in seen_claims:
                if prior_role != role and set(evidence) & set(prior.get("evidence_ids") or []) and _claims_conflict(prior, row):
                    conflicts.append({"roles": [prior_role, role], "claim_ids": [prior.get("claim_id"), claim_id],
                                      "evidence_ids": sorted(set(evidence) & set(prior.get("evidence_ids") or []))})
            seen_claims.append((role, row))
    return {"claim_verdicts": verdicts, "conflicts": conflicts,
            "evidence_registry": registry, "arbitration_mode": "deterministic"}


def _collect_evidence_registry(value: Any) -> Dict[str, Dict[str, Any]]:
    result = {}
    if isinstance(value, Mapping):
        if value.get("evidence_id"):
            result[str(value["evidence_id"])] = dict(value)
        for child in value.values(): result.update(_collect_evidence_registry(child))
    elif isinstance(value, list):
        for child in value: result.update(_collect_evidence_registry(child))
    return result


def _claims_conflict(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_text, right_text = str(left.get("claim") or "").lower(), str(right.get("claim") or "").lower()
    positives = ("pass", "success", "improved", "favorable", "low pae")
    negatives = ("fail", "worse", "high pae", "poor", "clash")
    return ((any(x in left_text for x in positives) and any(x in right_text for x in negatives)) or
            (any(x in right_text for x in positives) and any(x in left_text for x in negatives)))


def _sanitize_collaboration_guidance(
    guidance: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    sanitized = BinderQualityAnalysisAgent._sanitize_guidance(
        [dict(item) for item in guidance]
    )
    result: List[Dict[str, Any]] = []
    for row in sanitized:
        item = dict(row)
        changes = dict(item.get("config_parameter_changes") or {})
        disallowed = sorted(set(changes) - QUALITY_MANAGER_CONFIG_KEYS)
        item["config_parameter_changes"] = {
            key: value
            for key, value in changes.items()
            if key in QUALITY_MANAGER_CONFIG_KEYS
        }
        ignored = list(item.get("ignored_config_parameter_changes") or [])
        ignored.extend(disallowed)
        if ignored:
            item["ignored_config_parameter_changes"] = sorted(set(ignored))
        result.append(item)
    return result


def _history_cards(memory: Any, *, before_round_id: int) -> List[Dict[str, Any]]:
    metrics = {
        int(item.get("round_id", -1)): dict(item)
        for item in (getattr(memory, "round_metrics", []) or [])
    }
    cards = []
    for record in sorted(
        getattr(memory, "rounds", []) or [],
        key=lambda value: int(getattr(value, "round_id", -1)),
    ):
        round_id = int(getattr(record, "round_id", -1))
        if round_id < 0 or round_id >= before_round_id:
            continue
        metric = metrics.get(round_id, {})
        if bool(metric.get("execution_failed")):
            continue
        evaluation = _as_dict(getattr(record, "evaluation", {}))
        active_examples = _as_dict(
            getattr(record, "active_learning_examples", None)
            or evaluation.get("active_learning_examples")
        )
        round_examples = _as_dict(active_examples.get("current_round") or active_examples)
        strict_examples = list(
            round_examples.get("strict_positive_examples")
            or round_examples.get("positive_examples")
            or []
        )
        cards.append({
            "evidence_id": f"MEM:R{round_id}",
            "round_id": round_id,
            "reward": metric.get("reward", getattr(record, "reward", None)),
            "best_iptm": metric.get("best_iptm"),
            "median_iptm": metric.get("median_iptm"),
            "core_objective": metric.get("core_objective"),
            "success_count": metric.get("success_count"),
            "arm": metric.get("arm_signature"),
            "failure_tags": [
                key
                for key, value in _as_dict(evaluation.get("tag_counts")).items()
                if value and not str(key).startswith("pass_")
            ],
            "config": dict(getattr(record, "config_snapshot", {}) or {}),
            "strict_positive_population": _strict_positive_population(
                round_id, strict_examples
            ),
        })
    recent = cards[-5:]
    scored = [item for item in cards if _optional_float(item.get("reward")) is not None]
    historical_best = max(scored, key=lambda item: float(item["reward"])) if scored else None
    if historical_best and historical_best not in recent:
        recent = [historical_best] + recent
    return recent


def _strict_positive_population(
    round_id: int,
    examples: Sequence[Mapping[str, Any]],
    *,
    representative_limit: int = 3,
) -> Dict[str, Any]:
    """Bounded population evidence for trajectory comparisons."""

    rows = [dict(item) for item in examples or []]
    metric_names = (
        "design_to_target_iptm",
        "min_design_to_target_pae",
        "design_ptm",
        "designfolding_filter_rmsd",
    )
    metric_summary: Dict[str, Any] = {}
    for name in metric_names:
        values = [
            value
            for value in (
                _optional_float(_as_dict(item.get("metrics")).get(name))
                for item in rows
            )
            if value is not None
        ]
        if values:
            metric_summary[name] = {
                "min": round(min(values), 6),
                "mean": round(sum(values) / len(values), 6),
                "max": round(max(values), 6),
            }
    representatives = sorted(
        rows,
        key=lambda item: _optional_float(
            _as_dict(item.get("metrics")).get("core_objective")
        ) or -1.0,
        reverse=True,
    )[: max(0, int(representative_limit))]
    return {
        "evidence_id": f"R{round_id}:STRICT_POS:POPULATION",
        "round_id": round_id,
        "strict_positive_count": len(rows),
        "metric_summary": metric_summary,
        "representative_samples": representatives,
    }


def _parameter_diff(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    result = {}
    for key in sorted(set(previous) | set(current)):
        old, new = previous.get(key), current.get(key)
        if old == new:
            continue
        row: Dict[str, Any] = {"before": old, "after": new}
        if (
            isinstance(old, (int, float))
            and not isinstance(old, bool)
            and isinstance(new, (int, float))
            and not isinstance(new, bool)
        ):
            row["delta"] = round(float(new) - float(old), 9)
        result[str(key)] = row
    return result


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
        if value.startswith("json"):
            value = value[4:].strip()
    start, end = value.find("{"), value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    try:
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, dict) else None
    except Exception:
        return None


def _find_evidence_items(value: Any, wanted: set) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    if isinstance(value, Mapping):
        evidence_id = str(value.get("evidence_id") or "")
        if evidence_id and evidence_id in wanted:
            result.append(dict(value))
            seen.add(evidence_id)
        for child in value.values():
            for item in _find_evidence_items(child, wanted):
                item_id = str(item.get("evidence_id") or "")
                if item_id and item_id not in seen:
                    result.append(item)
                    seen.add(item_id)
    elif isinstance(value, list):
        for child in value:
            for item in _find_evidence_items(child, wanted):
                item_id = str(item.get("evidence_id") or "")
                if item_id and item_id not in seen:
                    result.append(item)
                    seen.add(item_id)
    return result


def _collect_evidence_ids(value: Any) -> set:
    result = set()
    if isinstance(value, Mapping):
        if value.get("evidence_id"):
            result.add(str(value["evidence_id"]))
        for child in value.values():
            result.update(_collect_evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_collect_evidence_ids(child))
    return result


def _referenced_evidence_ids(value: Any) -> set:
    result = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {
                "evidence_ids",
                "counterevidence_ids",
                "trusted_evidence_ids",
            }:
                result.update(str(item) for item in child or [])
            else:
                result.update(_referenced_evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_referenced_evidence_ids(child))
    return result


def _packet_audit(packets: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: {
            "bytes": len(
                json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
            ),
            "evidence_count": len(_collect_evidence_ids(value)),
            "digest": hashlib.sha256(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()[:16],
        }
        for key, value in packets.items()
    }


def _evidence_registry(packets: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            evidence_id = value.get("evidence_id")
            if evidence_id:
                registry[str(evidence_id)] = dict(value)
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(packets)
    return registry


def _validate_claims(outputs: Mapping[str, Any], registry: Mapping[str, Any], physics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for role, output in outputs.items():
        for claim in output.get("claims") or []:
            ids = [str(x) for x in claim.get("evidence_ids") or []]
            scope = str(claim.get("scope") or "")
            records = [registry[x] for x in ids if x in registry]
            issues = []
            if not ids or len(records) != len(ids): issues.append("missing_or_unknown_evidence")
            if not scope and role != "trajectory": issues.append("missing_scope")
            text = str(claim.get("claim") or "").lower()
            if "affinity" in text: issues.append("physical_boundary_affinity_not_measured")
            if scope == "whole_binder" and any(":STRUCT:" in x for x in ids): issues.append("local_evidence_for_whole_binder_claim")
            if scope == "population" and not any(x.endswith(":METRICS") for x in ids): issues.append("population_denominator_missing")
            rows.append({"role": role, "claim_id": str(claim.get("claim_id") or ""),
                         "valid": not issues, "issues": issues, "evidence_ids": ids,
                         "scope": scope})
    return rows


def _accepted_claims(outputs: Mapping[str, Any], manager: Optional[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    valid = {(x["role"], x["claim_id"]) for x in validation if x.get("valid")}
    verdicts = {str(x.get("claim_id")): str(x.get("verdict")) for x in (manager or {}).get("claim_verdicts") or []}
    result = []
    for role, output in outputs.items():
        for claim in output.get("claims") or []:
            cid = str(claim.get("claim_id") or "")
            if (role, cid) in valid and verdicts.get(cid, "accept") == "accept":
                result.append({"role": role, **dict(claim)})
    return result


def _skills_for_role(skills: Any, role: str) -> List[Dict[str, Any]]:
    """Project skill directives to the role that can act on them."""
    rows = [dict(item) for item in (skills or []) if isinstance(item, Mapping)]
    def roles(item):
        metadata = _as_dict(item.get("role_metadata"))
        values = metadata.get("roles") or item.get("roles") or []
        return [values] if isinstance(values, str) else list(values)
    if role == "manager":
        return [item for item in rows if "manager" in roles(item) and (_as_dict(item.get("role_metadata")).get("arbitration_only") or item.get("type") == "deterministic_policy" or item.get("origin") == "run_local_self_improvement")]
    return [item for item in rows if role in roles(item)]


def _normalization_issue(code: str, **details: Any) -> Dict[str, Any]:
    return {"code": code, **details}


def _strict_string_list(value: Any, *, field: str, maximum: int, max_length: int, audit: List[Dict[str, Any]]) -> Optional[List[str]]:
    if not isinstance(value, list) or len(value) > maximum:
        audit.append(_normalization_issue("invalid_list", field=field, maximum=maximum))
        return None
    result: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item or len(item) > max_length:
            audit.append(_normalization_issue("invalid_string_item", field=field, max_length=max_length))
            return None
        result.append(item)
    return result


def _source_rounds_match(source_round_ids: Sequence[int], evidence_ids: Sequence[str]) -> bool:
    cited_rounds = set()
    for evidence_id in evidence_ids:
        match = __import__("re").match(r"^(?:MEM:)?R(-?\d+)(?::|$)", evidence_id)
        if match:
            cited_rounds.add(int(match.group(1)))
    return set(source_round_ids).issubset(cited_rounds)


def _normalize_specialist_output(output: Mapping[str, Any], *, role: str) -> Dict[str, Any]:
    """Strict specialist contract: reject malformed records and preserve audit."""
    audit: List[Dict[str, Any]] = []
    if not isinstance(output, Mapping):
        return {"claims": [], "validation_audit": [_normalization_issue("invalid_output_type")]}
    if set(output) != {"findings"}:
        audit.append(_normalization_issue("illegal_top_level_fields", fields=sorted(set(output) - {"findings"})))
    findings = output.get("findings")
    if not isinstance(findings, list) or len(findings) > 3:
        audit.append(_normalization_issue("invalid_findings", maximum=3))
        return {"claims": [], "validation_audit": audit}
    schemas = {
        "positive": ({"finding_id", "statement", "scope", "signal", "evidence_ids", "counterevidence_ids", "confidence"}, {"local_fragment", "whole_binder", "population"}, {"signal": {"reusable", "non_reusable", "uncertain"}}),
        "negative": ({"finding_id", "statement", "scope", "failure_type", "repair_family", "evidence_ids", "counterevidence_ids", "confidence"}, {"candidate", "cluster", "population"}, {"failure_type": {"pose", "pae", "hotspot", "foldability", "clash", "filtering", "uncertain"}, "repair_family": {"target_context", "sampler", "length", "selection", "none"}}),
        "trajectory": ({"finding_id", "statement", "scope", "source_round_ids", "family", "outcome", "causal_strength", "evidence_ids", "counterevidence_ids", "confidence"}, {"trajectory"}, {"family": {"sampler", "length", "target_context", "template", "selection", "none"}, "outcome": {"improved", "regressed", "mixed", "not_identifiable"}, "causal_strength": {"none", "weak", "moderate"}}),
    }
    allowed, scopes, enums = schemas.get(role, (set(), set(), {}))
    claims: List[Dict[str, Any]] = []
    seen = set()
    for index, raw in enumerate(findings):
        if not isinstance(raw, Mapping):
            audit.append(_normalization_issue("invalid_finding_type", index=index)); continue
        item = dict(raw)
        if set(item) != allowed:
            audit.append(_normalization_issue("illegal_finding_fields", index=index, fields=sorted(set(item) ^ allowed))); continue
        finding_id, statement, scope = item.get("finding_id"), item.get("statement"), item.get("scope")
        if not isinstance(finding_id, str) or not finding_id or len(finding_id) > 32 or finding_id in seen:
            audit.append(_normalization_issue("invalid_or_duplicate_finding_id", index=index)); continue
        if not isinstance(statement, str) or not statement or len(statement) > 320 or scope not in scopes:
            audit.append(_normalization_issue("invalid_statement_or_scope", index=index)); continue
        evidence = _strict_string_list(item.get("evidence_ids"), field="evidence_ids", maximum=8, max_length=160, audit=audit)
        counter = _strict_string_list(item.get("counterevidence_ids"), field="counterevidence_ids", maximum=8, max_length=160, audit=audit)
        confidence = item.get("confidence")
        if not evidence or counter is None or isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            audit.append(_normalization_issue("invalid_evidence_or_confidence", index=index)); continue
        if any(item.get(field) not in choices for field, choices in enums.items()):
            audit.append(_normalization_issue("invalid_enum", index=index)); continue
        rounds = item.get("source_round_ids", [])
        if role == "trajectory":
            if not isinstance(rounds, list) or len(rounds) > 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in rounds) or not _source_rounds_match(rounds, evidence + counter):
                audit.append(_normalization_issue("invalid_source_round_ids", index=index)); continue
        seen.add(finding_id)
        claims.append({"claim_id": finding_id, "claim": statement, "scope": scope, "evidence_ids": evidence, "counterevidence_ids": counter, "confidence": float(confidence), **{field: item.get(field) for field in enums}, "source_round_ids": list(rounds)})
    return {"claims": claims, "validation_audit": audit}


def _normalize_manager_output(output: Mapping[str, Any], registry: Mapping[str, Any], finding_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Strict manager contract with exact fields, enums, IDs and evidence."""
    audit: List[Dict[str, Any]] = []
    result = {"accepted_finding_ids": [], "rejected_finding_ids": [], "strategy_intents": [], "uncertainties": [], "validation_audit": audit}
    if not isinstance(output, Mapping):
        audit.append(_normalization_issue("invalid_output_type")); return result
    allowed = {"accepted_finding_ids", "rejected_finding_ids", "strategy_intents", "uncertainties"}
    if set(output) != allowed:
        audit.append(_normalization_issue("illegal_top_level_fields", fields=sorted(set(output) ^ allowed))); return result
    known = set(finding_ids or [])
    accepted = _strict_string_list(output.get("accepted_finding_ids"), field="accepted_finding_ids", maximum=9, max_length=32, audit=audit)
    rejected = _strict_string_list(output.get("rejected_finding_ids"), field="rejected_finding_ids", maximum=9, max_length=32, audit=audit)
    uncertainties = _strict_string_list(output.get("uncertainties"), field="uncertainties", maximum=3, max_length=240, audit=audit)
    if accepted is None or rejected is None or uncertainties is None or len(set(accepted + rejected)) != len(accepted + rejected) or (known and not set(accepted + rejected).issubset(known)):
        audit.append(_normalization_issue("invalid_finding_references")); return result
    intents = output.get("strategy_intents")
    if not isinstance(intents, list) or len(intents) > 3:
        audit.append(_normalization_issue("invalid_strategy_intents", maximum=3)); return result
    seen = set()
    for index, raw in enumerate(intents):
        fields = {"intent_id", "kind", "evidence_ids", "expected_signal", "risk", "priority"}
        if not isinstance(raw, Mapping) or set(raw) != fields:
            audit.append(_normalization_issue("illegal_intent_fields", index=index)); continue
        row = dict(raw); intent_id = row.get("intent_id"); kind = row.get("kind"); priority = row.get("priority")
        ids = _strict_string_list(row.get("evidence_ids"), field="evidence_ids", maximum=8, max_length=160, audit=audit)
        if not isinstance(intent_id, str) or not intent_id or len(intent_id) > 32 or intent_id in seen or kind not in {"preserve", "relax", "explore", "repair", "hold"} or not ids or any(value not in registry for value in ids):
            audit.append(_normalization_issue("invalid_intent_identity_or_evidence", index=index)); continue
        if any(not isinstance(row.get(field), str) or not row.get(field) or len(row.get(field)) > 240 for field in ("expected_signal", "risk")) or isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= 9:
            audit.append(_normalization_issue("invalid_intent_content", index=index)); continue
        seen.add(intent_id); result["strategy_intents"].append(dict(row))
    result.update({"accepted_finding_ids": accepted, "rejected_finding_ids": rejected, "uncertainties": uncertainties})
    return result

def _collaboration_grade(count: int, manager: bool) -> str:
    if count == 3: return "full"
    if manager: return "manager_plus_specialists"
    if count >= 2: return "two_specialists"
    if count == 1: return "one_specialist_plus_deterministic"
    return "fallback"


def _delivery_metadata(grade: str, roles: Sequence[str], manager: bool, registry: Mapping[str, Any], telemetry: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    return {"collaboration_grade": grade,
            "availability": {"specialists": list(roles), "manager": manager,
                             "deterministic_assembler": True},
            "authority": {"evidence": "original_evidence_registry", "claim_verdicts": "deterministic_evidence_validator",
                          "formal_analysis": "deterministic_assembler"},
            "evidence_registry_digest": stable_hash(registry), "telemetry": list(telemetry or [])}


def _assemble_analysis(round_id: int, compact: Mapping[str, Any], claims: Sequence[Mapping[str, Any]], outputs: Mapping[str, Any], manager: Optional[Mapping[str, Any]]) -> BinderQualityAnalysis:
    fallback = BinderQualityAnalysisAgent._fallback(round_id, compact)
    strategies = [{"action": item.get("kind"), "evidence_ids": list(item.get("evidence_ids") or []), "expected_signal": item.get("expected_signal"), "risk": item.get("risk"), "config_parameter_changes": {}} for item in list((manager or {}).get("strategy_intents") or [])]
    guidance = _sanitize_collaboration_guidance(strategies) if strategies else fallback.next_round_guidance
    summary = f"Deterministically assembled {len(claims)} validated claims from {len(outputs)} specialist(s)."
    factors = [{"factor": str(c.get("claim") or ""), "evidence_ids": list(c.get("evidence_ids") or []),
                "evidence": [], "impact": "mixed", "confidence": c.get("confidence", 0.5)} for c in claims[:5]]
    return BinderQualityAnalysis(round_id, bool(outputs), summary, fallback.high_quality_modules,
                                 fallback.low_quality_modules, factors or fallback.causal_factors, guidance, {})


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
