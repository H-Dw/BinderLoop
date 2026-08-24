"""Run-local, structured self-improvement skill documents.

The LLM never edits these YAML files directly.  It proposes typed operations;
``SkillDocumentEditor`` validates and applies them before atomically rewriting
the canonical document.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from binderloop.resume import atomic_write_json, atomic_write_text, file_sha256, stable_hash


SELF_IMPROVEMENT_SCHEMA_VERSION = "1.0"
EXPERIENCE_MODULES: Tuple[str, ...] = (
    "successful_patterns",
    "failure_avoidance",
    "parameter_effects",
    "structural_context_rules",
    "exploration_exploitation",
    "rollback_recovery",
    "transfer_candidates",
)
RULE_RELATIONS = frozenset(
    {"equivalent", "subsumes", "subsumed_by", "complementary", "contradictory", "distinct"}
)
RULE_STATUSES = frozenset({"candidate", "seed_active", "active", "contested", "retired"})
UPDATE_OPERATIONS = frozenset({"UPSERT", "REVISE", "MERGE", "UPVOTE", "DOWNVOTE", "RETIRE"})

_TARGET_TOKEN_PATTERNS = (
    re.compile(r"(?:^|[\s\"'])[/~][^\s\"']+"),
    re.compile(r"\b[A-Za-z0-9_.-]+\.(?:cif|pdb|yaml|yml|json)\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]:-?\d+\b"),
    re.compile(r"\bcandidate[-_][A-Za-z0-9_.:-]+\b", re.IGNORECASE),
    re.compile(r"\brank[-_]?\d[A-Za-z0-9_.:-]*\b", re.IGNORECASE),
    re.compile(r"\bdesign[-_](?:\d|spec|len|rank)[A-Za-z0-9_.:-]*\b", re.IGNORECASE),
    re.compile(r"\b(?:fragment|frag)[-_][A-Za-z0-9_.:-]+\b", re.IGNORECASE),
    re.compile(r"\btemplate[-_](?:\d|frag|rank|seed|[0-9a-f]{8})[A-Za-z0-9_.:-]*\b", re.IGNORECASE),
)


class SelfImprovementSkillError(ValueError):
    """Raised for invalid skill documents, operations, or lifecycle state."""


@dataclass
class LearnedStrategyRule:
    rule_id: str
    title: str
    condition: str
    strategy: str
    canonical_signature: Dict[str, Any]
    expected_signals: List[str]
    watch_signals: List[str]
    contraindications: List[str]
    status: str = "candidate"
    support_count: int = 0
    contradiction_count: int = 0
    utility: float = 0.0
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence_refs"] = list(self.evidence_refs or [])
        metadata = dict(data.pop("metadata") or {})
        data.update(metadata)
        return data


@dataclass(frozen=True)
class SelfImprovementSkillHandle:
    path: str
    mode: str
    source_path: Optional[str]
    source_sha256: Optional[str]
    generation_id: str
    state_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_skill_document(*, generation_id: str, source: str = "new") -> Dict[str, Any]:
    now = int(time.time())
    return {
        "schema_version": SELF_IMPROVEMENT_SCHEMA_VERSION,
        "identity": {
            "skill_id": "binder-self-improvement-" + generation_id,
            "generation_id": generation_id,
            "revision": 0,
            "created_at": now,
            "updated_at": now,
        },
        "governance": {
            "scope": "run_local",
            "source": source,
            "immutable_controls": [
                "round_reward",
                "RollbackController",
                "config_parameter_contract",
                "binder_length_bounds",
                "template_provenance_gate",
            ],
        },
        "modules": {
            module: {"summary": "", "rules": {}}
            for module in EXPERIENCE_MODULES
        },
        "semantic_relations": {},
        "conflict_sets": {},
        "provenance": {"applied_operation_ids": []},
    }


def canonical_rule_signature(rule: Mapping[str, Any], *, module: str) -> Dict[str, Any]:
    signature = dict(rule.get("canonical_signature") or {})
    parameter_families = signature.get("parameter_families") or rule.get("parameter_families") or []
    action_directions = signature.get("action_directions") or rule.get("action_directions") or {}
    triggers = signature.get("trigger_phenotypes") or rule.get("trigger_phenotypes") or []
    expected = signature.get("expected_signals") or rule.get("expected_signals") or []
    watch = signature.get("watch_signals") or rule.get("watch_signals") or []
    contraindications = signature.get("contraindications") or rule.get("contraindications") or []
    return {
        "experience_type": str(signature.get("experience_type") or module),
        "parameter_families": sorted({str(value) for value in parameter_families if str(value)}),
        "action_directions": {
            str(key): _normalize_direction(value)
            for key, value in sorted(dict(action_directions or {}).items())
        },
        "trigger_phenotypes": sorted({str(value) for value in triggers if str(value)}),
        "expected_signals": sorted({str(value) for value in expected if str(value)}),
        "watch_signals": sorted({str(value) for value in watch if str(value)}),
        "contraindications": sorted({str(value) for value in contraindications if str(value)}),
    }


def semantic_candidates(
    document: Mapping[str, Any],
    candidate_rule: Mapping[str, Any],
    *,
    module: str,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Deterministically shortlist rules before an LLM relation judgment."""
    validated = validate_skill_document(document)
    candidate_signature = canonical_rule_signature(candidate_rule, module=module)
    related_modules = {
        module,
        *({
            "successful_patterns": {"parameter_effects", "structural_context_rules"},
            "failure_avoidance": {"parameter_effects", "rollback_recovery"},
            "parameter_effects": {"successful_patterns", "failure_avoidance"},
            "structural_context_rules": {"successful_patterns", "exploration_exploitation"},
            "exploration_exploitation": {"structural_context_rules", "rollback_recovery"},
            "rollback_recovery": {"failure_avoidance", "exploration_exploitation"},
            "transfer_candidates": set(EXPERIENCE_MODULES),
        }.get(module, set()))
    }
    ranked: List[Tuple[float, int, str, str, Dict[str, Any]]] = []
    for existing_module in EXPERIENCE_MODULES:
        if existing_module not in related_modules:
            continue
        for rule_id, rule in validated["modules"][existing_module]["rules"].items():
            score = _signature_overlap(
                candidate_signature,
                canonical_rule_signature(rule, module=existing_module),
            )
            score = max(score, 0.6 * _rule_text_overlap(candidate_rule, rule))
            if score <= 0:
                continue
            source_rounds = [
                int(value)
                for value in rule.get("source_round_ids", []) or []
                if str(value).lstrip("-").isdigit()
            ]
            recency = max(source_rounds or [-1])
            ranked.append((score, recency, existing_module, rule_id, dict(rule)))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    return [
        {"module": item[2], "rule_id": item[3], "score": round(item[0], 6), "rule": item[4]}
        for item in ranked[: max(1, int(limit))]
    ]


def _signature_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    scores: List[float] = []
    for key in ("parameter_families", "trigger_phenotypes", "expected_signals", "watch_signals"):
        lvalues = set(left.get(key) or [])
        rvalues = set(right.get(key) or [])
        if not lvalues or not rvalues:
            continue
        scores.append(len(lvalues & rvalues) / float(len(lvalues | rvalues)))
    left_dirs = dict(left.get("action_directions") or {})
    right_dirs = dict(right.get("action_directions") or {})
    common = set(left_dirs) & set(right_dirs)
    if common:
        scores.append(
            sum(1.0 if left_dirs[key] == right_dirs[key] else 0.25 for key in common)
            / float(len(common))
        )
    return max(scores) if scores else 0.0


def _rule_text_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    def tokens(rule: Mapping[str, Any]) -> set:
        text = " ".join(
            str(rule.get(key) or "")
            for key in ("title", "condition", "strategy")
        ).lower()
        return {
            token
            for token in re.findall(r"[a-z0-9_]+", text)
            if len(token) >= 4
        }

    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(len(left_tokens | right_tokens))


def apply_semantic_relations(
    document: Mapping[str, Any],
    relations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    data = validate_skill_document(document)
    relation_store = dict(data.get("semantic_relations") or {})
    conflict_store = dict(data.get("conflict_sets") or {})
    for raw in relations or []:
        item = dict(raw or {})
        relation = str(item.get("relation") or "")
        if relation not in RULE_RELATIONS:
            continue
        left = str(item.get("left_rule_id") or "")
        right = str(item.get("right_rule_id") or "")
        if (
            not left
            or not right
            or left == right
            or not _rule_exists(data, left)
            or not _rule_exists(data, right)
        ):
            continue
        relation_id = stable_hash({"left": min(left, right), "right": max(left, right), "relation": relation})[:20]
        relation_store[relation_id] = {
            "left_rule_id": left,
            "right_rule_id": right,
            "relation": relation,
            "confidence": _bounded_float(item.get("confidence"), 0.0, 1.0),
            "evidence": [str(value) for value in item.get("evidence", []) or []][:8],
            "analysis_digest": str(item.get("analysis_digest") or ""),
        }
        if relation == "contradictory":
            conflict_store[relation_id] = {
                "rule_ids": sorted({left, right}),
                "status": "open",
                "created_at": int(time.time()),
            }
            _set_rule_status(data, left, "contested")
            _set_rule_status(data, right, "contested")
    data["semantic_relations"] = relation_store
    data["conflict_sets"] = conflict_store
    return validate_skill_document(data)


def apply_lifecycle(
    document: Mapping[str, Any],
    *,
    promotion_min_support: int,
    retirement_contradictions: int,
    max_rules: int,
) -> Dict[str, Any]:
    data = validate_skill_document(document)
    all_rules: List[Tuple[str, str, Dict[str, Any]]] = []
    for module in EXPERIENCE_MODULES:
        rules = data["modules"][module]["rules"]
        for rule_id, raw in list(rules.items()):
            rule = dict(raw)
            support = int(rule.get("support_count") or 0)
            contradictions = int(rule.get("contradiction_count") or 0)
            utility = float(rule.get("utility") or 0.0)
            status = str(rule.get("status") or "candidate")
            if (
                status == "candidate"
                and utility > 0
                and (
                    support >= int(promotion_min_support)
                    or bool(rule.get("strong_evidence"))
                )
            ):
                rule["status"] = "active"
            if status not in {"retired", "contested"} and (
                contradictions >= int(retirement_contradictions)
                or utility <= -float(retirement_contradictions)
            ):
                rule["status"] = "retired"
                rule.setdefault("retirement_reason", "contradiction_threshold")
            rules[rule_id] = rule
            all_rules.append((module, rule_id, rule))
    overflow = max(0, len(all_rules) - int(max_rules))
    if overflow:
        removable = sorted(
            [
                item for item in all_rules
                if str(item[2].get("status")) not in {"seed_active", "active", "contested"}
            ],
            key=lambda item: (
                str(item[2].get("status")) != "retired",
                float(item[2].get("utility") or 0.0),
                int(item[2].get("support_count") or 0),
                item[1],
            ),
        )
        for module, rule_id, _ in removable[:overflow]:
            data["modules"][module]["rules"].pop(rule_id, None)
    return validate_skill_document(data)


def active_prompt_rules(
    document: Mapping[str, Any],
    *,
    limit: int = 6,
    context_phenotypes: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    data = validate_skill_document(document)
    current_phenotypes = {str(value) for value in (context_phenotypes or []) if str(value)}
    rows: List[Dict[str, Any]] = []
    for module in EXPERIENCE_MODULES:
        for rule in data["modules"][module]["rules"].values():
            if str(rule.get("status")) not in {"seed_active", "active"}:
                continue
            row = {
                "rule_id": rule.get("rule_id"),
                "module": module,
                "title": rule.get("title"),
                "condition": rule.get("condition"),
                "strategy": rule.get("strategy"),
                "canonical_signature": rule.get("canonical_signature"),
                "expected_signals": list(rule.get("expected_signals") or []),
                "watch_signals": list(rule.get("watch_signals") or []),
                "contraindications": list(rule.get("contraindications") or []),
                "utility": float(rule.get("utility") or 0.0),
                "support_count": int(rule.get("support_count") or 0),
                "status": rule.get("status"),
            }
            trigger_phenotypes = set(
                (row.get("canonical_signature") or {}).get("trigger_phenotypes") or []
            )
            row["relevance"] = (
                len(trigger_phenotypes & current_phenotypes)
                / float(max(1, len(trigger_phenotypes | current_phenotypes)))
                if current_phenotypes and trigger_phenotypes
                else 0.0
            )
            rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["relevance"]),
            -float(row["utility"]),
            -int(row["support_count"]),
            str(row["rule_id"]),
        )
    )
    return rows[: max(1, int(limit))]


def mark_rules_contested(
    document: Mapping[str, Any],
    rule_ids: Iterable[str],
    *,
    reason: str,
) -> Dict[str, Any]:
    data = validate_skill_document(document)
    for rule_id in {str(value) for value in rule_ids if str(value)}:
        for module in EXPERIENCE_MODULES:
            rules = data["modules"][module]["rules"]
            if rule_id not in rules:
                continue
            rule = dict(rules[rule_id])
            if rule.get("status") != "retired":
                rule["status"] = "contested"
                rule["contested_reason"] = str(reason)
                rule["contested_at"] = int(time.time())
                rules[rule_id] = rule
            break
    return validate_skill_document(data)


def record_conflict_decisions(
    document: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    data = validate_skill_document(document)
    for conflict in data.get("conflict_sets", {}).values():
        if str(conflict.get("status") or "open") != "open":
            continue
        conflict_rules = {str(value) for value in conflict.get("rule_ids") or []}
        for decision in decisions or []:
            decision_rules = {
                str(value)
                for key in ("selected_rule_ids", "suspended_rule_ids")
                for value in decision.get(key, []) or []
            }
            if conflict_rules and not conflict_rules.intersection(decision_rules):
                continue
            conflict["status"] = "testing"
            conflict["decision"] = dict(decision)
            conflict["updated_at"] = int(time.time())
            break
    return validate_skill_document(data)


def settle_conflicts_from_operations(
    document: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    data = validate_skill_document(document)
    upvoted = {
        str(item.get("rule_id"))
        for item in operations or []
        if str(item.get("op")) == "UPVOTE"
    }
    downvoted = {
        str(item.get("rule_id"))
        for item in operations or []
        if str(item.get("op")) in {"DOWNVOTE", "RETIRE"}
    }
    for conflict in data.get("conflict_sets", {}).values():
        if str(conflict.get("status") or "") != "testing":
            continue
        rules = {str(value) for value in conflict.get("rule_ids") or []}
        if rules.intersection(upvoted):
            conflict["status"] = "resolved"
            conflict["resolution"] = "positive_followup_for_selected_rule"
            conflict["updated_at"] = int(time.time())
        elif rules.intersection(downvoted):
            conflict["status"] = "open"
            conflict["resolution"] = "tested_choice_regressed"
            conflict["updated_at"] = int(time.time())
    return validate_skill_document(data)


def _set_rule_status(document: Dict[str, Any], rule_id: str, status: str) -> None:
    for module in EXPERIENCE_MODULES:
        rules = document["modules"][module]["rules"]
        if rule_id in rules:
            rule = dict(rules[rule_id])
            rule["status"] = status
            rules[rule_id] = rule
            return


def _rule_exists(document: Mapping[str, Any], rule_id: str) -> bool:
    return any(
        str(rule_id) in ((document.get("modules") or {}).get(module, {}).get("rules") or {})
        for module in EXPERIENCE_MODULES
    )


def _bounded_float(value: Any, lower: float, upper: float) -> float:
    try:
        return min(upper, max(lower, float(value)))
    except (TypeError, ValueError):
        return lower


def _normalize_direction(value: Any) -> str:
    token = str(value or "").strip().lower()
    aliases = {
        "+": "increase", "up": "increase", "raise": "increase", "higher": "increase",
        "-": "decrease", "down": "decrease", "lower": "decrease", "reduce": "decrease",
        "same": "hold", "keep": "hold", "unchanged": "hold",
    }
    token = aliases.get(token, token)
    return token if token in {"increase", "decrease", "hold", "enable", "disable", "broaden", "narrow"} else "contextual"


def validate_skill_document(document: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(document, Mapping):
        raise SelfImprovementSkillError("self-improvement skill root must be a mapping")
    data = copy.deepcopy(dict(document))
    if str(data.get("schema_version")) != SELF_IMPROVEMENT_SCHEMA_VERSION:
        raise SelfImprovementSkillError(
            "self-improvement skill schema_version must be %s" % SELF_IMPROVEMENT_SCHEMA_VERSION
        )
    identity = data.get("identity")
    if not isinstance(identity, Mapping):
        raise SelfImprovementSkillError("self-improvement skill identity must be a mapping")
    for key in ("skill_id", "generation_id", "revision"):
        if key not in identity:
            raise SelfImprovementSkillError("self-improvement skill identity.%s is required" % key)
    modules = data.get("modules")
    if not isinstance(modules, Mapping):
        raise SelfImprovementSkillError("self-improvement skill modules must be a mapping")
    unknown_modules = sorted(set(modules) - set(EXPERIENCE_MODULES))
    missing_modules = sorted(set(EXPERIENCE_MODULES) - set(modules))
    if unknown_modules or missing_modules:
        raise SelfImprovementSkillError(
            "self-improvement skill modules mismatch; missing=%s unknown=%s"
            % (missing_modules, unknown_modules)
        )
    for module in EXPERIENCE_MODULES:
        section = modules[module]
        if not isinstance(section, Mapping) or not isinstance(section.get("rules"), Mapping):
            raise SelfImprovementSkillError("modules.%s must contain a rules mapping" % module)
        normalized_rules: Dict[str, Any] = {}
        for rule_id, raw_rule in section.get("rules", {}).items():
            if not isinstance(raw_rule, Mapping):
                raise SelfImprovementSkillError("rule %s must be a mapping" % rule_id)
            rule = dict(raw_rule)
            rule["rule_id"] = str(rule.get("rule_id") or rule_id)
            if rule["rule_id"] != str(rule_id):
                raise SelfImprovementSkillError("rule map key must equal rule_id: %s" % rule_id)
            status = str(rule.get("status") or "candidate")
            if status not in RULE_STATUSES:
                raise SelfImprovementSkillError("rule %s has invalid status %s" % (rule_id, status))
            rule["status"] = status
            rule["canonical_signature"] = canonical_rule_signature(rule, module=module)
            _assert_target_agnostic_rule(rule)
            normalized_rules[str(rule_id)] = rule
        section = dict(section)
        section["summary"] = str(section.get("summary") or "")
        section["rules"] = normalized_rules
        modules[module] = section
    data["modules"] = dict(modules)
    relations = data.get("semantic_relations") or {}
    if not isinstance(relations, Mapping):
        raise SelfImprovementSkillError("semantic_relations must be a mapping")
    for relation in relations.values():
        if not isinstance(relation, Mapping) or str(relation.get("relation")) not in RULE_RELATIONS:
            raise SelfImprovementSkillError("semantic relation must use the fixed relation taxonomy")
    if not isinstance(data.get("conflict_sets") or {}, Mapping):
        raise SelfImprovementSkillError("conflict_sets must be a mapping")
    provenance = data.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise SelfImprovementSkillError("provenance must be a mapping")
    provenance = dict(provenance)
    provenance["applied_operation_ids"] = [
        str(value) for value in provenance.get("applied_operation_ids", []) or []
    ]
    data["provenance"] = provenance
    data.setdefault("governance", {})
    data.setdefault("semantic_relations", {})
    data.setdefault("conflict_sets", {})
    return data


def _assert_target_agnostic_rule(rule: Mapping[str, Any]) -> None:
    prompt_visible = {
        key: value
        for key, value in rule.items()
        if key not in {"provenance", "evidence_refs", "source_round_ids", "artifact_digests", "source_skill_ids"}
    }
    text = json.dumps(prompt_visible, ensure_ascii=False, default=str)
    for pattern in _TARGET_TOKEN_PATTERNS:
        match = pattern.search(text)
        if match:
            raise SelfImprovementSkillError(
                "target-specific token is not allowed in learned rule: %s" % match.group(0)
            )


class SkillDocumentEditor:
    """Apply typed operations to a canonical skill document."""

    def __init__(self, document: Mapping[str, Any]):
        self.document = validate_skill_document(document)

    def apply(self, operations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        applied_ids = set(self.document["provenance"].get("applied_operation_ids") or [])
        changed = False
        for raw_operation in operations or []:
            operation = dict(raw_operation or {})
            operation_id = str(operation.get("operation_id") or stable_hash(operation)[:20])
            if operation_id in applied_ids:
                continue
            self._apply_one(operation)
            applied_ids.add(operation_id)
            changed = True
        self.document["provenance"]["applied_operation_ids"] = sorted(applied_ids)
        identity = dict(self.document["identity"])
        identity["revision"] = int(identity.get("revision") or 0) + (1 if changed else 0)
        identity["updated_at"] = int(time.time())
        self.document["identity"] = identity
        return validate_skill_document(self.document)

    def _apply_one(self, operation: Mapping[str, Any]) -> None:
        op = str(operation.get("op") or operation.get("operation") or "").upper()
        if op not in UPDATE_OPERATIONS:
            raise SelfImprovementSkillError("unsupported self-improvement operation: %s" % op)
        module = str(operation.get("module") or "")
        if module not in EXPERIENCE_MODULES:
            raise SelfImprovementSkillError("operation module must be one of the fixed experience modules")
        rules = self.document["modules"][module]["rules"]
        rule_id = str(operation.get("rule_id") or "")
        if op == "UPSERT":
            raw_rule = operation.get("rule")
            if not isinstance(raw_rule, Mapping):
                raise SelfImprovementSkillError("UPSERT requires a rule mapping")
            if not rule_id:
                rule_id = str(raw_rule.get("rule_id") or "")
            if not rule_id:
                raise SelfImprovementSkillError("UPSERT requires rule_id")
            rule = dict(raw_rule)
            rule["rule_id"] = rule_id
            rule.setdefault("status", "candidate")
            rule.setdefault("support_count", 0)
            rule.setdefault("contradiction_count", 0)
            rule.setdefault("utility", 0.0)
            rule["canonical_signature"] = canonical_rule_signature(rule, module=module)
            rules[rule_id] = rule
            return
        if op == "REVISE":
            if rule_id not in rules:
                from_module = str(operation.get("from_module") or "")
                if (
                    from_module in EXPERIENCE_MODULES
                    and rule_id in self.document["modules"][from_module]["rules"]
                ):
                    rules[rule_id] = self.document["modules"][from_module]["rules"].pop(rule_id)
                else:
                    raise SelfImprovementSkillError("REVISE references unknown rule_id %s" % rule_id)
            patch = operation.get("patch")
            if not isinstance(patch, Mapping):
                raise SelfImprovementSkillError("REVISE requires a patch mapping")
            updated = _deep_merge(dict(rules[rule_id]), dict(patch))
            updated["rule_id"] = rule_id
            updated["canonical_signature"] = canonical_rule_signature(updated, module=module)
            rules[rule_id] = updated
            return
        if op == "MERGE":
            source_ids = [str(value) for value in operation.get("source_rule_ids", []) or []]
            if len(source_ids) < 2 or any(value not in rules for value in source_ids):
                raise SelfImprovementSkillError("MERGE requires at least two existing source_rule_ids")
            target_id = rule_id or source_ids[0]
            merged = operation.get("rule")
            if not isinstance(merged, Mapping):
                raise SelfImprovementSkillError("MERGE requires a merged rule mapping")
            merged_rule = dict(merged)
            merged_rule["rule_id"] = target_id
            merged_rule["support_count"] = sum(int(rules[x].get("support_count") or 0) for x in source_ids)
            merged_rule["contradiction_count"] = sum(int(rules[x].get("contradiction_count") or 0) for x in source_ids)
            merged_rule["utility"] = max(float(rules[x].get("utility") or 0.0) for x in source_ids)
            source_statuses = {str(rules[x].get("status") or "candidate") for x in source_ids}
            if "contested" in source_statuses:
                merged_rule["status"] = "contested"
            elif "active" in source_statuses:
                merged_rule["status"] = "active"
            elif "seed_active" in source_statuses:
                merged_rule["status"] = "seed_active"
            else:
                merged_rule["status"] = "candidate"
            merged_rule["merged_from_rule_ids"] = sorted(set(source_ids))
            merged_rule["evidence_refs"] = [
                dict(ref)
                for source_id in source_ids
                for ref in (rules[source_id].get("evidence_refs") or [])
                if isinstance(ref, Mapping)
            ]
            merged_rule["canonical_signature"] = canonical_rule_signature(merged_rule, module=module)
            for source_id in source_ids:
                rules.pop(source_id, None)
            rules[target_id] = merged_rule
            return
        if rule_id not in rules:
            raise SelfImprovementSkillError("%s references unknown rule_id %s" % (op, rule_id))
        rule = dict(rules[rule_id])
        if op == "UPVOTE":
            rule["support_count"] = int(rule.get("support_count") or 0) + 1
            rule["utility"] = round(float(rule.get("utility") or 0.0) + float(operation.get("weight") or 1.0), 6)
            if rule.get("status") == "contested":
                rule["status"] = "candidate"
                rule["contested_resolution"] = "returned_to_candidate_after_positive_exposure"
        elif op == "DOWNVOTE":
            rule["contradiction_count"] = int(rule.get("contradiction_count") or 0) + 1
            rule["utility"] = round(float(rule.get("utility") or 0.0) - float(operation.get("weight") or 1.0), 6)
        elif op == "RETIRE":
            rule["status"] = "retired"
            rule["retirement_reason"] = str(operation.get("reason") or "retired_by_update")
        rules[rule_id] = rule


def _deep_merge(left: Dict[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(left.get(key), Mapping):
            left[key] = _deep_merge(dict(left[key]), value)
        else:
            left[key] = copy.deepcopy(value)
    return left


class SelfImprovementSkillStore:
    """Create/copy/load one unique working skill inside a run directory."""

    STATE_FILENAME = "self_improvement_skill_state.json"

    def __init__(self, handle: SelfImprovementSkillHandle):
        self.handle = handle
        self.path = Path(handle.path)

    @classmethod
    def prepare(
        cls,
        *,
        enabled: bool,
        source_path: Optional[str],
        out_dir: Path,
        source_base: Optional[Path] = None,
    ) -> Optional["SelfImprovementSkillStore"]:
        if not enabled:
            return None
        memory_dir = Path(out_dir) / "memory"
        state_path = memory_dir / cls.STATE_FILENAME
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            handle = SelfImprovementSkillHandle(**state)
            working = Path(handle.path)
            if not working.exists():
                raise SelfImprovementSkillError("recorded self-improvement skill is missing: %s" % working)
            if source_path:
                requested = _resolve_source_path(source_path, source_base)
                requested_sha = file_sha256(requested)
                if handle.source_sha256 != requested_sha:
                    raise SelfImprovementSkillError(
                        "resume self-improvement source differs from the existing run"
                    )
            validate_skill_document(yaml.safe_load(working.read_text(encoding="utf-8")) or {})
            return cls(handle)

        source: Optional[Path] = None
        source_sha: Optional[str] = None
        source_document: Optional[Dict[str, Any]] = None
        mode = "new"
        if source_path:
            source = _resolve_source_path(source_path, source_base)
            if not source.exists():
                raise SelfImprovementSkillError("self-improvement skill not found: %s" % source)
            source_document = validate_skill_document(
                yaml.safe_load(source.read_text(encoding="utf-8")) or {}
            )
            source_sha = file_sha256(source)
            mode = "reuse_copy"

        skill_dir = memory_dir / "self_improvement_skills"
        skill_dir.mkdir(parents=True, exist_ok=True)
        generation_id = ""
        working_path: Optional[Path] = None
        for _ in range(8):
            generation_id = uuid.uuid4().hex
            filename = "binder-self-improvement-%s-%s.yaml" % (
                time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                generation_id[:12],
            )
            candidate_path = skill_dir / filename
            try:
                _reserve_unique_file(candidate_path)
                working_path = candidate_path
                break
            except SelfImprovementSkillError:
                continue
        if working_path is None:
            raise SelfImprovementSkillError(
                "unable to reserve a unique self-improvement skill filename after 8 attempts"
            )
        try:
            if source_document is None:
                document = default_skill_document(generation_id=generation_id, source="new")
            else:
                document = copy.deepcopy(source_document)
                identity = dict(document["identity"])
                identity["generation_id"] = generation_id
                identity["skill_id"] = "binder-self-improvement-" + generation_id
                identity["revision"] = 0
                identity["created_at"] = int(time.time())
                identity["updated_at"] = int(time.time())
                document["identity"] = identity
                governance = dict(document.get("governance") or {})
                governance["scope"] = "run_local"
                governance["source"] = "reused_copy"
                document["governance"] = governance
            atomic_write_text(working_path, _dump_skill_yaml(validate_skill_document(document)))
            handle = SelfImprovementSkillHandle(
                path=str(working_path),
                mode=mode,
                source_path=str(source) if source else None,
                source_sha256=source_sha,
                generation_id=generation_id,
                state_path=str(state_path),
            )
            atomic_write_json(state_path, handle.to_dict())
            return cls(handle)
        except Exception:
            try:
                working_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def load(self) -> Dict[str, Any]:
        return validate_skill_document(yaml.safe_load(self.path.read_text(encoding="utf-8")) or {})

    def save(self, document: Mapping[str, Any]) -> Path:
        validated = validate_skill_document(document)
        return atomic_write_text(self.path, _dump_skill_yaml(validated))

    def apply_operations(self, operations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        updated = SkillDocumentEditor(self.load()).apply(operations)
        self.save(updated)
        return updated

    def bootstrap_from_skills(self, skills: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        document = self.load()
        if any(section["rules"] for section in document["modules"].values()):
            return document
        operations: List[Dict[str, Any]] = []
        for skill in skills:
            if str(skill.get("type")) != "strategy":
                continue
            skill_id = str(skill.get("id") or "")
            params = dict(skill.get("params") or {})
            module = _module_for_seed_skill(skill_id)
            rule_id = "seed_" + stable_hash({"skill_id": skill_id})[:16]
            arm_name = str(params.get("arm_name") or "")
            families = sorted(
                {
                    _parameter_family(key)
                    for key in params
                    if _parameter_family(key)
                }
            )
            operations.append({
                "operation_id": "bootstrap_" + rule_id,
                "op": "UPSERT",
                "module": module,
                "rule_id": rule_id,
                "rule": {
                    "rule_id": rule_id,
                    "title": str(skill.get("description") or skill_id),
                    "condition": "Apply when the declared structural or performance trigger is present.",
                    "strategy": "; ".join(str(x) for x in skill.get("guidance", []) or []),
                    "expected_signals": list((skill.get("expected_signals") or {}).get("improve") or []),
                    "watch_signals": list((skill.get("expected_signals") or {}).get("watch") or []),
                    "contraindications": [str(skill.get("risk") or "")] if skill.get("risk") else [],
                    "status": "seed_active",
                    "support_count": 1,
                    "contradiction_count": 0,
                    "utility": 1.0,
                    "source_skill_ids": [skill_id],
                    "canonical_signature": {
                        "experience_type": module,
                        "parameter_families": families or ([arm_name] if arm_name else []),
                        "action_directions": {},
                        "trigger_phenotypes": list((skill.get("trigger") or {}).get("tags_any") or []),
                        "expected_signals": list((skill.get("expected_signals") or {}).get("improve") or []),
                        "watch_signals": list((skill.get("expected_signals") or {}).get("watch") or []),
                        "contraindications": [],
                    },
                },
            })
        return self.apply_operations(operations) if operations else document


def _resolve_source_path(value: str, source_base: Optional[Path]) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and source_base is not None:
        path = Path(source_base) / path
    return path.resolve()


def _reserve_unique_file(path: Path) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except FileExistsError as exc:
        raise SelfImprovementSkillError("unique self-improvement filename collision: %s" % path) from exc
    os.close(descriptor)


def _dump_skill_yaml(document: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(document),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def _module_for_seed_skill(skill_id: str) -> str:
    token = skill_id.lower()
    if "rollback" in token:
        return "rollback_recovery"
    if "explore" in token or "exploit" in token:
        return "exploration_exploitation"
    if "hotspot" in token or "pose" in token or "template" in token:
        return "structural_context_rules"
    return "successful_patterns"


def _parameter_family(key: str) -> str:
    token = str(key)
    mapping = {
        "arm_name": "search_arm",
        "suggested_binder_lengths": "binder_length",
        "length_policy": "binder_length",
        "hotspot_weight_direction": "hotspot_pressure",
        "hotspot_weight_policy": "hotspot_pressure",
        "crop_policy": "target_crop",
        "template_policy": "template_use",
        "budget_policy": "budget",
    }
    return mapping.get(token, token if token else "")

