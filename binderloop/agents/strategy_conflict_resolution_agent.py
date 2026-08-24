"""Evidence- and physics-grounded arbitration for soft strategy conflicts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from binderloop.agents.config_parameter_contract import supported_config_changes
from binderloop.llm import LLMConfigError, LLMTransportError, OpenAICompatibleClient
from binderloop.resume import stable_hash
from binderloop.parameter_decision import PROBABILISTIC_SAMPLER_KEYS
from binderloop.skills import compose_agent_system
from binderloop.structured_llm import call_structured_json


DECISION_ACTIONS = frozenset(
    {"choose", "blend", "hold", "drop", "rerun", "revert_to_best", "insufficient_evidence"}
)


@dataclass
class StrategyConflictResolution:
    round_id: int
    llm_used: bool
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    params_update: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArmHistoryResolution:
    round_id: int
    action: str
    selected_arm_id: Optional[str] = None
    update_direction: str = "hold"
    accepted_current_evidence_ids: List[str] = field(default_factory=list)
    accepted_history_evidence_ids: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    llm_used: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]: return asdict(self)


class StrategyConflictResolutionAgent:
    ARM_HISTORY_SYSTEM = """You are the existing StrategyConflictResolutionAgent performing the second, historical comparison. Return JSON only with exactly: {"action":"select|hold","selected_arm_id":null,"update_direction":"preserve_winner|explore_alternative|hold","accepted_current_evidence_ids":[],"accepted_history_evidence_ids":[],"conflicts":[{"code":"...","arm_id":"...","evidence_ids":[]}]}. Select only a supplied current executed closed arm. Cite only supplied evidence IDs. Never emit parameter/config changes. Hold when evidence conflicts or is insufficient."""

    SYSTEM = """You are StrategyConflictResolutionAgent for a protein-binder closed loop.
Resolve only the supplied soft parameter-family conflicts. Return JSON only:
{
 "summary":"...",
 "decisions":[{
   "parameter_family":"...",
   "action":"choose|blend|hold|revert_to_best|insufficient_evidence",
   "selected_rule_ids":[],
   "suspended_rule_ids":[],
   "evidence_ids":[],
   "physical_rationale":"...",
   "parameter_changes":{},
   "expected_signals":[],
   "watch_signals":[],
   "confidence":0.0
 }],
 "controlled_comparisons":[{"parameter_family":"...","arms":[{"arm_id":"...","parameter_changes":{}}],"reason":"..."}]
}
Use exact cross-round outcomes and the historical best, not rule priority or recency alone.
Check iPTM/interface confidence, PAE, pTM/foldability, refold RMSD, hotspot/geometry,
diversity and rollback risk. Do not claim causation for confounded multi-parameter rounds.
Select one coherent round-level parameter vector. Hold or revert when evidence is weak. Never override hard bounds, immutable metric facts, ownership,
pressure-conflict controls, target definition, budget, or template provenance."""

    def __init__(self, llm: Optional[OpenAICompatibleClient], *, require_llm: bool = False):
        self.llm = llm
        self.require_llm = bool(require_llm)

    def resolve_arm_direction(self, *, round_id: int, arm_comparison: Mapping[str, Any], ledger_history: Mapping[str, Any]) -> ArmHistoryResolution:
        comparison=dict(arm_comparison or {}); history=dict(ledger_history or {})
        current_ids=sorted({str(x) for x in comparison.get("evidence_ids",[]) or []})
        historical_rows=list(history.get("recent_rounds") or [])
        history_ids=sorted({str(item.get("evidence_id") or f"MEM:R{item.get('round_id')}") for item in historical_rows})
        closed=set(str(x) for x in comparison.get("closed_arm_ids",[]) or [])
        selected=str(comparison.get("winner_arm_id") or "") or None; conflicts=[]
        if selected:
            for item in historical_rows:
                for outcome in item.get("per_arm_outcomes",[]) or (item.get("outcome") or {}).get("per_arm_outcomes",[]) or []:
                    if str(outcome.get("arm_id"))==selected and bool(outcome.get("regressed")):
                        conflicts.append({"code":"historical_regression","arm_id":selected,"round_id":item.get("round_id"),"evidence_ids":[str(item.get("evidence_id") or f"MEM:R{item.get('round_id')}")]})
        fallback=ArmHistoryResolution(round_id,"hold",None,"hold",current_ids,history_ids,conflicts,False,{"source":"deterministic_history_resolution"})
        if comparison.get("status")=="winner" and selected in closed and not conflicts:
            fallback=ArmHistoryResolution(round_id,"select",selected,str(comparison.get("update_direction") or "preserve_winner"),current_ids,history_ids,[],False,{"source":"deterministic_history_resolution"})
        if not (self.llm and self.llm.available()):
            if self.require_llm: raise RuntimeError("StrategyConflictResolutionAgent arm history resolution requires an available LLM")
            return fallback
        payload={"round_id":round_id,"current_completed_comparison":comparison,"current_executed_closed_arm_ids":sorted(closed),"current_evidence_ids":current_ids,"history_evidence_ids":history_ids,"ledger_history":{"recent_rounds":historical_rows[-5:]}}
        required=("action","selected_arm_id","update_direction","accepted_current_evidence_ids","accepted_history_evidence_ids","conflicts")
        def validate(result):
            action=str(result.get("action") or ""); proposed=str(result.get("selected_arm_id") or "") or None; direction=str(result.get("update_direction") or "")
            accepted_current=[str(x) for x in result.get("accepted_current_evidence_ids",[]) or []] if isinstance(result.get("accepted_current_evidence_ids"),list) else []
            accepted_history=[str(x) for x in result.get("accepted_history_evidence_ids",[]) or []] if isinstance(result.get("accepted_history_evidence_ids"),list) else []
            invalid=[]
            if action not in {"select","hold"}: invalid.append("action")
            if direction not in {"preserve_winner","explore_alternative","hold"}: invalid.append("update_direction")
            if not isinstance(result.get("conflicts"),list): invalid.append("conflicts")
            if not ((action=="hold" and proposed is None and direction=="hold") or (action=="select" and proposed in closed and proposed==selected)): invalid.append("decision_consistency")
            return {"invalid_fields":invalid,"illegal_arm_ids":([proposed] if proposed and proposed not in closed else []),"illegal_evidence_ids":sorted((set(accepted_current)-set(current_ids))|(set(accepted_history)-set(history_ids)))}
        outcome=call_structured_json(self.llm,system=self.ARM_HISTORY_SYSTEM,user=payload,required_fields=required,field_validator=validate,temperature=.05,max_completion_tokens=1_000_000,visible_json_tokens=4096,thinking="low",repair=True,valid_arm_ids=sorted(closed),valid_evidence_ids=current_ids+history_ids)
        if outcome.value is None:
            fallback.raw={"source":"deterministic_history_resolution","fallback_reason":"invalid_llm_output_after_targeted_repair","llm_error":outcome.error,"llm_attempts":outcome.attempts,"context_digest":stable_hash(payload)}
            return fallback
        result=outcome.value; action=str(result["action"]); proposed=str(result.get("selected_arm_id") or "") or None; direction=str(result["update_direction"])
        accepted_current=[str(x) for x in result.get("accepted_current_evidence_ids",[]) or []]; accepted_history=[str(x) for x in result.get("accepted_history_evidence_ids",[]) or []]
        llm_conflicts=[]
        for item in result.get("conflicts",[]) or []:
            row=dict(item or {}); ids=[str(x) for x in row.get("evidence_ids",[]) or []]
            if set(ids).issubset(set(current_ids)|set(history_ids)): llm_conflicts.append({"code":str(row.get("code") or "conflict"),"arm_id":str(row.get("arm_id") or ""),"evidence_ids":ids})
        return ArmHistoryResolution(round_id,action,proposed,direction,accepted_current,accepted_history,llm_conflicts,True,{"source":"validated_llm_history_resolution","context_digest":stable_hash(payload),"llm_repaired":outcome.repaired,"llm_attempts":outcome.attempts})

    def resolve(
        self,
        *,
        round_id: int,
        conflicts: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> StrategyConflictResolution:
        if not conflicts:
            return StrategyConflictResolution(round_id, False, summary="No soft strategy conflict.")
        if not (self.llm and self.llm.available()):
            if self.require_llm:
                raise RuntimeError("StrategyConflictResolutionAgent requires an available LLM")
            return self._safe_fallback(round_id, conflicts, context, reason="llm_unavailable")
        payload = {
            "round_id": int(round_id),
            "conflicts": [dict(item) for item in conflicts],
            "trajectory_and_physics": dict(context),
            "executable_config_keys_only": True,
        }
        try:
            result = self.llm.chat_json(
                system=compose_agent_system(self.SYSTEM, active_skills=active_skills),
                user=payload,
                temperature=0.05,
                max_tokens=4000,
                thinking="low",
            )
        except (LLMConfigError, LLMTransportError) as exc:
            if self.require_llm:
                raise
            return self._safe_fallback(round_id, conflicts, context, reason=str(exc))
        decisions = []
        merged: Dict[str, Any] = {}
        allowed_families = {
            str(item.get("parameter_family") or "")
            for item in conflicts
            if item.get("parameter_family")
        }
        seen_families = set()
        for raw in result.get("decisions") or []:
            item = dict(raw or {})
            action = str(item.get("action") or "")
            family = str(item.get("parameter_family") or "")
            if (
                action not in DECISION_ACTIONS
                or family not in allowed_families
                or family in seen_families
            ):
                continue
            seen_families.add(family)
            changes = supported_config_changes(item.get("parameter_changes") or {})
            family_keys = _keys_for_family(family)
            if family_keys is not None:
                changes = {key: value for key, value in changes.items() if key in family_keys}
            sampler_keys = set(changes).intersection(PROBABILISTIC_SAMPLER_KEYS)
            if sampler_keys:
                changes = {key: value for key, value in changes.items() if key not in PROBABILISTIC_SAMPLER_KEYS}
                if action not in {"hold", "drop", "rerun", "insufficient_evidence"}:
                    action = "hold"
                    item["action"] = action
                item["probabilistic_sampler_veto"] = sorted(sampler_keys)
            item["parameter_changes"] = changes
            try:
                item["confidence"] = min(1.0, max(0.0, float(item.get("confidence") or 0.0)))
            except (TypeError, ValueError):
                item["confidence"] = 0.0
            if action in {"choose", "blend", "revert_to_best"} and item["confidence"] >= 0.55:
                merged.update(changes)
            decisions.append(item)
        if not decisions:
            return self._safe_fallback(
                round_id,
                conflicts,
                context,
                reason="llm_returned_no_valid_decisions",
            )
        return StrategyConflictResolution(
            round_id=round_id,
            llm_used=True,
            decisions=decisions,
            params_update=merged,
            summary=str(result.get("summary") or ""),
            raw={"source": "llm_soft_conflict_resolution", "context_digest": stable_hash(payload), "llm_result": result},
        )

    @staticmethod
    def _safe_fallback(
        round_id: int,
        conflicts: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        *,
        reason: str,
    ) -> StrategyConflictResolution:
        current = dict(context.get("current_config") or {})
        historical_best = dict((context.get("historical_best") or {}).get("config") or {})
        decisions: List[Dict[str, Any]] = []
        hold_update: Dict[str, Any] = {}
        for conflict in conflicts:
            keys = [str(value) for value in conflict.get("keys", []) or []]
            best_values = {
                key: historical_best.get(key)
                for key in keys
                if key in historical_best
            }
            fallback_values = best_values or {
                key: current.get(key) for key in keys if key in current
            }
            normalized = supported_config_changes(fallback_values)
            vetoed_sampler_keys = sorted(set(normalized).intersection(PROBABILISTIC_SAMPLER_KEYS))
            held = {k: v for k, v in normalized.items() if k not in PROBABILISTIC_SAMPLER_KEYS}
            hold_update.update(held)
            decisions.append({
                "parameter_family": conflict.get("parameter_family"),
                "action": "revert_to_best" if best_values else "hold",
                "selected_rule_ids": [],
                "suspended_rule_ids": list(conflict.get("rule_ids") or []),
                "evidence_ids": list(conflict.get("evidence_ids") or []),
                "physical_rationale": (
                    "Insufficient verified arbitration evidence; revert the conflicted family to historical best."
                    if best_values
                    else "Insufficient verified arbitration evidence; hold current safe values."
                ),
                "parameter_changes": held,
                "probabilistic_sampler_veto": vetoed_sampler_keys,
                "expected_signals": [],
                "watch_signals": [],
                "confidence": 1.0,
            })
        return StrategyConflictResolution(
            round_id=round_id,
            llm_used=False,
            decisions=decisions,
            params_update=hold_update,
            summary="Safe hold fallback for unresolved soft conflicts.",
            raw={"source": "deterministic_hold_fallback", "reason": reason},
        )


def detect_strategy_conflicts(
    *,
    merge_report: Mapping[str, Any],
    proposed_update: Mapping[str, Any],
    tuning_feedback: Mapping[str, Any],
    pressure_conflict: Mapping[str, Any],
    learned_document: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Detect only meaningful soft conflicts; hard constraints remain elsewhere."""
    conflicts: Dict[str, Dict[str, Any]] = {}
    for item in merge_report.get("ownership_conflicts") or []:
        key = str(item.get("key") or "")
        family = _parameter_family(key)
        if not key or _values_compatible(item.get("kept_value"), item.get("rejected_value")):
            continue
        conflicts.setdefault(family, _new_conflict(family))["sources"].append(dict(item))
        conflicts[family]["keys"].append(key)
    for penalty in tuning_feedback.get("penalized_moves") or []:
        key = str(penalty.get("parameter") or "")
        if key and key in proposed_update:
            family = _parameter_family(key)
            row = conflicts.setdefault(family, _new_conflict(family))
            row["keys"].append(key)
            row["sources"].append({"source": "penalized_move", **dict(penalty)})
    if pressure_conflict.get("active"):
        pressure_keys = {
            "auxiliary_hotspots",
            "epitope_crop_mode", "template_conditioned_fraction",
        }
        for key in sorted(pressure_keys.intersection(proposed_update)):
            family = _parameter_family(key)
            row = conflicts.setdefault(family, _new_conflict(family))
            row["keys"].append(key)
            row["sources"].append({"source": "pressure_conflict", "detail": dict(pressure_conflict)})
    if learned_document:
        directional_rules: Dict[str, List[Dict[str, Any]]] = {}
        for section in (learned_document.get("modules") or {}).values():
            for rule_id, rule in (section.get("rules") or {}).items():
                if rule.get("status") not in {"seed_active", "active"}:
                    continue
                signature = dict(rule.get("canonical_signature") or {})
                for family, direction in dict(signature.get("action_directions") or {}).items():
                    directional_rules.setdefault(str(family), []).append({
                        "rule_id": str(rule_id),
                        "direction": str(direction),
                    })
        opposites = {
            ("increase", "decrease"),
            ("enable", "disable"),
            ("broaden", "narrow"),
        }
        for family, rules in directional_rules.items():
            for left_index, left in enumerate(rules):
                for right in rules[left_index + 1:]:
                    pair = (left["direction"], right["direction"])
                    if pair not in opposites and tuple(reversed(pair)) not in opposites:
                        continue
                    row = conflicts.setdefault(family, _new_conflict(family))
                    row["rule_ids"].extend([left["rule_id"], right["rule_id"]])
                    row["sources"].append({
                        "source": "learned_opposite_directions",
                        "left": left,
                        "right": right,
                    })
        for conflict_id, conflict in (learned_document.get("conflict_sets") or {}).items():
            if str(conflict.get("status") or "open") != "open":
                continue
            family = "learned_rule_conflict"
            row = conflicts.setdefault(family, _new_conflict(family))
            row["rule_ids"].extend(str(value) for value in conflict.get("rule_ids") or [])
            row["evidence_ids"].append(str(conflict_id))
            row["sources"].append({"source": "learned_semantic_conflict", "detail": dict(conflict)})
    result = []
    for row in conflicts.values():
        if learned_document:
            for section in (learned_document.get("modules") or {}).values():
                for rule_id, rule in (section.get("rules") or {}).items():
                    signature = dict(rule.get("canonical_signature") or {})
                    families = set(signature.get("parameter_families") or [])
                    if row["parameter_family"] in families and rule.get("status") in {"seed_active", "active"}:
                        row["rule_ids"].append(str(rule_id))
        row["keys"] = sorted(set(row["keys"]))
        row["rule_ids"] = sorted(set(row["rule_ids"]))
        row["evidence_ids"] = sorted(set(row["evidence_ids"]))
        row["conflict_id"] = stable_hash(row)[:20]
        result.append(row)
    return sorted(result, key=lambda item: str(item["parameter_family"]))


def _new_conflict(family: str) -> Dict[str, Any]:
    return {
        "parameter_family": family,
        "keys": [],
        "rule_ids": [],
        "evidence_ids": [],
        "sources": [],
    }


def _parameter_family(key: str) -> str:
    if key in {"auxiliary_hotspots", "config_overrides"}:
        return "hotspot_pressure"
    if key in {"binder_lengths"}:
        return "binder_length"
    if key in {"alpha", "noise_scale", "diffusion_batch_size", "step_scale"}:
        return "sampling_exploration"
    if key in {"epitope_crop_mode"}:
        return "target_crop"
    if key in {"template_conditioned_fraction"}:
        return "template_or_module"
    if key in {"filter_biased"}:
        return "filtering"
    return key or "unknown"


def _keys_for_family(family: str) -> Optional[set]:
    return {
        "hotspot_pressure": {"auxiliary_hotspots", "config_overrides"},
        "binder_length": {"binder_lengths"},
        "sampling_exploration": {"alpha", "noise_scale", "diffusion_batch_size", "step_scale"},
        "target_crop": {"epitope_crop_mode"},
        "template_or_module": {"template_conditioned_fraction"},
        "filtering": {"filter_biased"},
    }.get(str(family))


def _normalize_comparison(value: Mapping[str, Any]) -> Dict[str, Any]:
    family = str(value.get("parameter_family") or "")
    family_keys = _keys_for_family(family)
    arms = []
    for index, raw_arm in enumerate(value.get("arms") or []):
        if not isinstance(raw_arm, Mapping):
            continue
        changes = {k: v for k, v in supported_config_changes(raw_arm.get("parameter_changes") or {}).items() if k not in PROBABILISTIC_SAMPLER_KEYS}
        if family_keys is not None:
            changes = {key: item for key, item in changes.items() if key in family_keys}
        if not changes:
            continue
        arms.append({
            "arm_id": str(raw_arm.get("arm_id") or raw_arm.get("name") or "arm_%d" % (index + 1)),
            "parameter_changes": changes,
        })
    if len(arms) < 2:
        return {}
    return {
        "parameter_family": family,
        "arms": arms,
        "reason": str(value.get("reason") or ""),
    }


def _values_compatible(left: Any, right: Any) -> bool:
    if left == right:
        return True
    try:
        lvalue, rvalue = float(left), float(right)
        scale = max(1.0, abs(lvalue), abs(rvalue))
        return abs(lvalue - rvalue) <= 0.02 * scale
    except (TypeError, ValueError):
        return False

