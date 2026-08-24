"""LLM updater for structured, run-local Binder self-improvement skills."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from binderloop.llm import LLMConfigError, LLMTransportError, OpenAICompatibleClient
from binderloop.resume import atomic_write_json, stable_hash
from binderloop.skills import compose_agent_system
from binderloop.skills.self_improvement import (
    EXPERIENCE_MODULES,
    RULE_RELATIONS,
    UPDATE_OPERATIONS,
    semantic_candidates,
)


@dataclass
class SelfImprovementUpdate:
    round_id: int
    llm_used: bool
    operations: List[Dict[str, Any]] = field(default_factory=list)
    semantic_relations: List[Dict[str, Any]] = field(default_factory=list)
    rejected_operations: List[Dict[str, Any]] = field(default_factory=list)
    sanitization_notes: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SelfImprovementSkillAgent:
    """Propose typed document operations; never write the skill file directly."""

    SYSTEM = """You are SelfImprovementSkillAgent for a protein-binder closed loop.
Return JSON only:
{
  "summary": "...",
  "operations": [
    {
      "operation_id": "...",
      "op": "UPSERT|REVISE|MERGE|UPVOTE|DOWNVOTE|RETIRE",
      "module": "successful_patterns|failure_avoidance|parameter_effects|structural_context_rules|exploration_exploitation|rollback_recovery|transfer_candidates",
      "from_module": "optional prior module when reclassifying an existing rule",
      "rule_id": "...",
      "rule": {
        "title": "...",
        "condition": "...",
        "strategy": "...",
        "expected_signals": [],
        "watch_signals": [],
        "contraindications": [],
        "status": "candidate",
        "canonical_signature": {
          "experience_type": "...",
          "parameter_families": [],
          "action_directions": {},
          "trigger_phenotypes": [],
          "expected_signals": [],
          "watch_signals": [],
          "contraindications": []
        }
      },
      "patch": {},
      "source_rule_ids": [],
      "weight": 1.0,
      "reason": "..."
    }
  ]
}
Use only immutable supplied evidence and strategies that were actually exposed. Infrastructure
failures produce no scientific lesson. Multi-parameter observations are correlations, not causal
proof. Rules must be target-agnostic: no target names, paths, chain/residue identifiers, candidate
IDs, template IDs, or target-specific absolute lengths. Express structure as phenotypes and actions
as relative directions/range quantiles. Route each lesson to its fixed experience module. Match and
revise an existing rule when possible; do not append prose or invent a new rule ID for paraphrases.
The deterministic writer owns promotion, retirement, conflict state, and file mutation."""

    MATCH_SYSTEM = """You compare proposed and existing Binder strategy rules.
Return JSON only:
{"relations":[{"left_rule_id":"...","right_rule_id":"...","relation":"equivalent|subsumes|subsumed_by|complementary|contradictory|distinct","confidence":0.0,"evidence":["field-level reason"]}]}.
Use the complete rule bodies and canonical signatures. Equivalent means the same condition and
action despite wording. Contradictory means incompatible actions under materially overlapping
conditions. Do not merge merely because both mention the same parameter family."""

    def __init__(
        self,
        llm: Optional[OpenAICompatibleClient],
        *,
        require_llm: bool = False,
        semantic_candidate_limit: int = 8,
        semantic_confidence_threshold: float = 0.72,
        prompt_max_bytes: int = 24_000,
        cache_dir: Optional[Path] = None,
        reward_improvement_threshold: float = 0.01,
        strong_improvement_threshold: float = 0.05,
    ) -> None:
        self.llm = llm
        self.require_llm = bool(require_llm)
        self.semantic_candidate_limit = max(1, int(semantic_candidate_limit))
        self.semantic_confidence_threshold = min(1.0, max(0.0, float(semantic_confidence_threshold)))
        self.prompt_max_bytes = max(1024, int(prompt_max_bytes))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.reward_improvement_threshold = max(0.0, float(reward_improvement_threshold))
        self.strong_improvement_threshold = max(
            self.reward_improvement_threshold,
            float(strong_improvement_threshold),
        )
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def propose_update(
        self,
        *,
        round_id: int,
        document: Mapping[str, Any],
        evidence: Mapping[str, Any],
        governance_skills: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> SelfImprovementUpdate:
        if not isinstance(evidence.get("outcome"), Mapping) or not isinstance(
            evidence.get("evaluation"), Mapping
        ):
            return SelfImprovementUpdate(
                round_id=round_id,
                llm_used=False,
                summary="Required immutable outcome/evaluation evidence is missing; update skipped.",
                raw={"source": "invalid_evidence_deterministic_skip"},
            )
        if bool((evidence.get("outcome") or {}).get("execution_failed")):
            return SelfImprovementUpdate(
                round_id=round_id,
                llm_used=False,
                summary="Execution failure has no scientific strategy signal; update skipped.",
                raw={"source": "execution_failure_deterministic_skip"},
            )
        if not (self.llm and self.llm.available()):
            if self.require_llm:
                raise RuntimeError("SelfImprovementSkillAgent requires an available LLM")
            return SelfImprovementUpdate(
                round_id=round_id,
                llm_used=False,
                summary="LLM unavailable; preserved the prior self-improvement skill unchanged.",
                raw={"source": "llm_unavailable_safe_noop"},
            )
        prompt = {
            "round_id": int(round_id),
            "task": "update_structured_self_improvement_skill",
            "experience": deidentify_experience(evidence),
            "current_skill": _compact_document(document),
            "governance_skills": _compact_governance_skills(governance_skills),
        }
        try:
            result = self.llm.chat_json(
                system=self.SYSTEM,
                user=prompt,
                temperature=0.1,
                max_tokens=5000,
                max_prompt_bytes=self.prompt_max_bytes,
                thinking="low",
            )
        except (LLMConfigError, LLMTransportError) as exc:
            if self.require_llm:
                raise
            return SelfImprovementUpdate(
                round_id=round_id,
                llm_used=False,
                summary="LLM update failed; preserved prior skill.",
                raw={"source": "llm_error_safe_noop", "error": str(exc)},
            )
        operations, rejected, sanitization_notes = self._normalize_operations(
            result.get("operations") or [],
            round_id=round_id,
        )
        operations, evidence_rejections = self._gate_operations_by_evidence(
            operations,
            evidence=evidence,
            round_id=round_id,
        )
        rejected.extend(evidence_rejections)
        semantic_relations: List[Dict[str, Any]] = []
        comparisons = self._semantic_comparisons(document, operations)
        if comparisons:
            try:
                comparison_digest = stable_hash(comparisons)
                cache_path = (
                    self.cache_dir / (comparison_digest + ".json")
                    if self.cache_dir
                    else None
                )
                if cache_path and cache_path.exists():
                    semantic_result = json.loads(cache_path.read_text(encoding="utf-8"))
                else:
                    semantic_result = self.llm.chat_json(
                        system=compose_agent_system(
                            self.MATCH_SYSTEM,
                            active_skills=governance_skills,
                        ),
                        user={"comparisons": comparisons},
                        temperature=0.0,
                        max_tokens=3000,
                        max_prompt_bytes=self.prompt_max_bytes,
                        thinking="low",
                    )
                    if cache_path:
                        atomic_write_json(cache_path, semantic_result)
                semantic_relations = self._normalize_relations(
                    semantic_result.get("relations") or [],
                    comparisons=comparisons,
                )
                operations = self._coalesce_equivalent_upserts(operations, semantic_relations, comparisons)
            except (LLMConfigError, LLMTransportError) as exc:
                if self.require_llm:
                    raise
                rejected.append({
                    "operation": {"comparison_count": len(comparisons)},
                    "reason": "semantic_match_unavailable:%s" % exc,
                })
        return SelfImprovementUpdate(
            round_id=round_id,
            llm_used=True,
            operations=operations,
            semantic_relations=semantic_relations,
            rejected_operations=rejected,
            sanitization_notes=sanitization_notes,
            summary=str(result.get("summary") or ""),
            raw={
                "source": "llm_structured_update",
                "context_digest": stable_hash(prompt),
                "llm_result": result,
                "semantic_comparison_count": len(comparisons),
            },
        )

    def _normalize_operations(
        self,
        raw_operations: Sequence[Mapping[str, Any]],
        *,
        round_id: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        sanitization_notes: List[Dict[str, Any]] = []
        for index, raw in enumerate(raw_operations or []):
            item = dict(raw or {})
            op = str(item.get("op") or item.get("operation") or "").upper()
            module = str(item.get("module") or "")
            if op not in UPDATE_OPERATIONS or module not in EXPERIENCE_MODULES:
                rejected.append({"operation": item, "reason": "invalid_op_or_module"})
                continue
            item["op"] = op
            item["module"] = module
            item["operation_id"] = str(
                item.get("operation_id")
                or stable_hash({"round_id": round_id, "index": index, "operation": item})[:20]
            )
            for key in ("rule", "patch"):
                if isinstance(item.get(key), Mapping):
                    before = dict(item[key])
                    item[key] = _generalize_value(before)
                    if item[key] != before:
                        sanitization_notes.append({
                            "operation_id": item["operation_id"],
                            "field": key,
                            "action": "target_specific_tokens_generalized",
                        })
            if op == "UPSERT" and isinstance(item.get("rule"), Mapping):
                rule = dict(item["rule"])
                generated_rule_id = "rule_" + stable_hash({
                    "module": module,
                    "canonical_signature": rule.get("canonical_signature"),
                    "condition": rule.get("condition"),
                    "strategy": rule.get("strategy"),
                })[:16]
                item["rule_id"] = generated_rule_id
                rule["rule_id"] = generated_rule_id
                rule["status"] = "candidate"
                rule["support_count"] = 0
                rule["contradiction_count"] = 0
                rule["utility"] = 0.0
                item["rule"] = rule
            if op == "REVISE" and isinstance(item.get("patch"), Mapping):
                item["patch"] = _strip_lifecycle_fields(dict(item["patch"]))
            if op == "MERGE" and isinstance(item.get("rule"), Mapping):
                item["rule"] = _strip_lifecycle_fields(dict(item["rule"]))
            if op in {"UPVOTE", "DOWNVOTE"}:
                try:
                    item["weight"] = min(1.0, max(0.1, float(item.get("weight") or 1.0)))
                except (TypeError, ValueError):
                    item["weight"] = 1.0
            accepted.append(item)
        return accepted, rejected, sanitization_notes

    def _gate_operations_by_evidence(
        self,
        operations: Sequence[Mapping[str, Any]],
        *,
        evidence: Mapping[str, Any],
        round_id: int,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        outcome = dict(evidence.get("outcome") or {})
        if outcome.get("execution_failed"):
            return [], [
                {"operation": dict(item), "reason": "execution_failure_has_no_scientific_signal"}
                for item in operations
            ]
        exposure = dict(evidence.get("strategy_exposure") or {})
        cited = {str(value) for value in exposure.get("cited_rule_ids", []) or []}
        current_reward = _optional_float(outcome.get("reward"))
        prior_rewards = [
            _optional_float(item.get("reward"))
            for item in evidence.get("recent_rounds", []) or []
            if _optional_float(item.get("reward")) is not None
        ]
        reward_delta = (
            None
            if current_reward is None or not prior_rewards
            else current_reward - float(prior_rewards[-1])
        )
        rollback_action = str((evidence.get("rollback") or {}).get("action") or "advance")
        watched_regression = _has_watched_regression(evidence)
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        evidence_ref = {
            "round_id": int(round_id),
            "exposure_id": exposure.get("exposure_id"),
            "evidence_digest": evidence.get("evidence_digest"),
            "reward_delta": reward_delta,
            "rollback_action": rollback_action,
        }
        for raw in operations:
            item = dict(raw)
            op = str(item.get("op"))
            rule_id = str(item.get("rule_id") or "")
            if op in {"UPVOTE", "DOWNVOTE", "RETIRE"} and rule_id not in cited:
                rejected.append({"operation": item, "reason": "rule_was_not_cited_as_used"})
                continue
            if op == "UPVOTE" and (
                reward_delta is None
                or reward_delta <= self.reward_improvement_threshold
                or watched_regression
            ):
                rejected.append({"operation": item, "reason": "no_verified_positive_reward_delta"})
                continue
            if op in {"DOWNVOTE", "RETIRE"} and not (
                (
                    reward_delta is not None
                    and reward_delta < -self.reward_improvement_threshold
                )
                or rollback_action in {"replay_best", "branch_from_best"}
            ):
                rejected.append({"operation": item, "reason": "no_verified_regression_or_rollback"})
                continue
            if op == "UPSERT" and isinstance(item.get("rule"), Mapping):
                rule = dict(item["rule"])
                rule_families = set(
                    (rule.get("canonical_signature") or {}).get("parameter_families") or []
                )
                applied_families = {
                    _parameter_family_from_key(key)
                    for key in (exposure.get("applied_update") or {})
                }
                if (
                    reward_delta is not None
                    and reward_delta >= self.strong_improvement_threshold
                    and not watched_regression
                    and exposure.get("exposure_id")
                    and exposure.get("applied_update")
                    and bool(rule_families.intersection(applied_families))
                ):
                    rule["strong_evidence"] = True
                    rule["support_count"] = 1
                    rule["utility"] = round(float(reward_delta), 6)
                rule["evidence_refs"] = [evidence_ref]
                rule["source_round_ids"] = [int(round_id)]
                item["rule"] = rule
            elif op == "REVISE" and isinstance(item.get("patch"), Mapping):
                patch = dict(item["patch"])
                patch["last_evidence_ref"] = evidence_ref
                item["patch"] = patch
            accepted.append(item)
        return accepted, rejected

    def _semantic_comparisons(
        self,
        document: Mapping[str, Any],
        operations: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        comparisons: List[Dict[str, Any]] = []
        for operation in operations:
            if operation.get("op") != "UPSERT" or not isinstance(operation.get("rule"), Mapping):
                continue
            proposed = dict(operation["rule"])
            proposed_id = str(operation.get("rule_id") or proposed.get("rule_id") or "")
            if not proposed_id:
                continue
            proposed["rule_id"] = proposed_id
            candidates = semantic_candidates(
                document,
                proposed,
                module=str(operation.get("module")),
                limit=self.semantic_candidate_limit,
            )
            for candidate in candidates:
                comparisons.append({
                    "left_rule_id": proposed_id,
                    "left_module": operation.get("module"),
                    "left_rule": proposed,
                    "right_rule_id": candidate["rule_id"],
                    "right_module": candidate["module"],
                    "right_rule": candidate["rule"],
                    "prefilter_score": candidate["score"],
                })
        return comparisons

    def _normalize_relations(
        self,
        relations: Sequence[Mapping[str, Any]],
        *,
        comparisons: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        allowed_pairs = {
            (str(item["left_rule_id"]), str(item["right_rule_id"]))
            for item in comparisons
        }
        normalized: List[Dict[str, Any]] = []
        for raw in relations or []:
            item = dict(raw or {})
            pair = (str(item.get("left_rule_id") or ""), str(item.get("right_rule_id") or ""))
            relation = str(item.get("relation") or "")
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence") or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            if pair not in allowed_pairs or relation not in RULE_RELATIONS:
                continue
            normalized.append({
                "left_rule_id": pair[0],
                "right_rule_id": pair[1],
                "relation": relation,
                "confidence": confidence,
                "evidence": [str(value) for value in item.get("evidence", []) or []][:8],
                "analysis_digest": stable_hash(item)[:20],
            })
        return normalized

    def _coalesce_equivalent_upserts(
        self,
        operations: Sequence[Mapping[str, Any]],
        relations: Sequence[Mapping[str, Any]],
        comparisons: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        comparison_map = {
            (str(row["left_rule_id"]), str(row["right_rule_id"])): row
            for row in comparisons
        }
        replacements: Dict[str, Dict[str, Any]] = {}
        for relation in relations:
            if relation["confidence"] < self.semantic_confidence_threshold:
                continue
            if relation["relation"] not in {"equivalent", "subsumes", "subsumed_by"}:
                continue
            pair = (relation["left_rule_id"], relation["right_rule_id"])
            row = comparison_map.get(pair)
            if not row:
                continue
            semantic_provenance = {
                "relation": relation["relation"],
                "confidence": relation["confidence"],
                "evidence": list(relation.get("evidence") or []),
                "proposed_rule_digest": stable_hash(row["left_rule"])[:20],
            }
            patch = (
                {
                    "semantic_match_provenance": semantic_provenance,
                    "last_evidence_ref": (row["left_rule"].get("evidence_refs") or [None])[-1],
                }
                if relation["relation"] == "subsumed_by"
                else {
                    **dict(row["left_rule"]),
                    "semantic_match_provenance": semantic_provenance,
                }
            )
            patch = _strip_lifecycle_fields(dict(patch))
            replacements[pair[0]] = {
                "operation_id": "semantic_revise_" + stable_hash(relation)[:16],
                "op": "REVISE",
                "module": row["right_module"],
                "rule_id": pair[1],
                "patch": patch,
                "reason": "semantic_%s" % relation["relation"],
            }
        output: List[Dict[str, Any]] = []
        for operation in operations:
            rule_id = str(operation.get("rule_id") or (operation.get("rule") or {}).get("rule_id") or "")
            output.append(replacements.get(rule_id, dict(operation)))
        return output


def deidentify_experience(value: Mapping[str, Any]) -> Dict[str, Any]:
    return _deidentify_mapping(dict(value or {}))


def _deidentify_mapping(data: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in data.items():
        token = str(key).lower()
        if any(part in token for part in (
            "structure_path", "structure_file", "target_name", "target_structure",
            "candidate_id", "template_id", "fragment_id", "output_dir",
            "artifact_path", "job_id", "task_name",
        )):
            continue
        if token in {"chain_id", "binder_chain", "primary_chain_id", "notes"}:
            continue
        if token in {"target_chains", "target_include", "target_binding_types"}:
            result[key + "_count"] = len(value or []) if isinstance(value, (list, tuple, set)) else int(bool(value))
            continue
        if token in {"hotspots", "auxiliary_hotspots", "effective_hotspots"}:
            result[key + "_count"] = len(value or []) if isinstance(value, (list, tuple, set)) else int(bool(value))
            continue
        if token == "binder_lengths" and isinstance(value, (list, tuple)):
            result["binder_length_profile"] = _relative_length_profile(value, data.get("binder_length_range"))
            continue
        if isinstance(value, Mapping):
            result[key] = _deidentify_mapping(value)
        elif isinstance(value, list):
            result[key] = [
                _deidentify_mapping(item) if isinstance(item, Mapping) else _generalize_value(item)
                for item in value[:40]
            ]
        else:
            result[key] = _generalize_value(value)
    return result


def _relative_length_profile(values: Sequence[Any], allowed_range: Any) -> List[str]:
    try:
        if isinstance(allowed_range, Mapping):
            lower = float(allowed_range.get("min") or allowed_range.get("start"))
            upper = float(allowed_range.get("max") or allowed_range.get("end"))
        else:
            lower, upper = float(allowed_range[0]), float(allowed_range[-1])
        width = max(1.0, upper - lower)
        labels = []
        for value in values:
            ratio = (float(value) - lower) / width
            labels.append("low" if ratio < 0.34 else "mid" if ratio < 0.67 else "high")
        return sorted(set(labels))
    except Exception:
        return ["relative_unspecified"]


_CHAIN_RESIDUE = re.compile(r"\b[A-Za-z]:-?\d+\b")
_PATH_OR_FILE = re.compile(r"(?:[/~][^\s\"']+|[A-Za-z0-9_.-]+\.(?:cif|pdb|yaml|yml|json))", re.IGNORECASE)
_ENTITY_ID = re.compile(
    r"\b(?:candidate[-_][A-Za-z0-9_.:-]+|rank[-_]?\d[A-Za-z0-9_.:-]*|design[-_](?:\d|spec|len|rank)[A-Za-z0-9_.:-]*|fragment[-_][A-Za-z0-9_.:-]+|frag[-_][A-Za-z0-9_.:-]+|template[-_](?:\d|frag|rank|seed|[0-9a-f]{8})[A-Za-z0-9_.:-]*)\b",
    re.IGNORECASE,
)


def _generalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _generalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_generalize_value(item) for item in value]
    if not isinstance(value, str):
        return value
    text = _CHAIN_RESIDUE.sub("<hotspot-residue>", value)
    text = _PATH_OR_FILE.sub("<target-artifact>", text)
    text = _ENTITY_ID.sub("<candidate-or-template>", text)
    return text


def _compact_document(document: Mapping[str, Any]) -> Dict[str, Any]:
    modules: Dict[str, Any] = {}
    for module in EXPERIENCE_MODULES:
        section = dict((document.get("modules") or {}).get(module) or {})
        rules = list((section.get("rules") or {}).values())
        rules.sort(
            key=lambda rule: (
                -float(rule.get("utility") or 0.0),
                str(rule.get("rule_id") or ""),
            )
        )
        modules[module] = {"summary": section.get("summary"), "rules": rules[:12]}
    return {
        "identity": dict(document.get("identity") or {}),
        "modules": modules,
        "semantic_relations": dict(document.get("semantic_relations") or {}),
        "conflict_sets": dict(document.get("conflict_sets") or {}),
    }


def _compact_governance_skills(skills: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "priority": item.get("priority"),
            "guidance": list(item.get("guidance") or [])[:6],
            "output_schema": dict(item.get("output_schema") or {}),
            "deterministic_controls": dict(item.get("deterministic_controls") or {}),
        }
        for item in (dict(value) for value in (skills or []))
    ]


def _strip_lifecycle_fields(value: Dict[str, Any]) -> Dict[str, Any]:
    for key in (
        "rule_id",
        "status",
        "support_count",
        "contradiction_count",
        "utility",
        "retirement_reason",
        "contested_reason",
    ):
        value.pop(key, None)
    return value


def _optional_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _has_watched_regression(evidence: Mapping[str, Any]) -> bool:
    trends = dict((evidence.get("evaluation") or {}).get("core_metric_trends") or {})
    delta = dict(trends.get("delta") or {})
    checks = (
        ("best_design_ptm", "lt", -0.03),
        ("mean_design_ptm", "lt", -0.03),
        ("best_refold_rmsd", "gt", 0.25),
        ("mean_refold_rmsd", "gt", 0.25),
        ("best_min_pae", "gt", 0.5),
        ("mean_min_pae", "gt", 0.5),
    )
    for key, direction, threshold in checks:
        value = _optional_float(delta.get(key))
        if value is None:
            continue
        if direction == "lt" and value < threshold:
            return True
        if direction == "gt" and value > threshold:
            return True
    return False


def _parameter_family_from_key(key: str) -> str:
    token = str(key)
    if token in {"hotspot_weight", "auxiliary_hotspots", "prioritize_hotspots", "config_overrides"}:
        return "hotspot_pressure"
    if token == "binder_lengths":
        return "binder_length"
    if token in {"alpha", "noise_scale", "diffusion_batch_size", "step_scale"}:
        return "sampling_exploration"
    if token == "epitope_crop_mode":
        return "target_crop"
    if token in {"template_conditioned_fraction", "module_guided_repair"}:
        return "template_or_module"
    if token in {"clash_filter", "filter_biased"}:
        return "filtering"
    return token

