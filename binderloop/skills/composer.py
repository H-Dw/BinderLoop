"""Priority-aware prompt rendering for static and run-local skills."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence


def compose_agent_system(
    base_system: str,
    *,
    active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
    max_skill_bytes: int = 24_000,
    role: Optional[str] = None,
    max_directives: int = 3,
    structured_output: bool = True,
    schema_fields: Optional[Sequence[str]] = None,
) -> str:
    """Render deterministic controls, learned rules, then supporting skills.

    System-level contracts in ``base_system`` remain authoritative.  The
    run-local skill is the highest *advisory* layer, never a permission grant.
    """
    skills = [dict(item or {}) for item in (active_skills or [])]
    if role:
        skills = [item for item in skills if _skill_applies_to_role(item, role)]
    if not skills:
        return base_system
    skills.sort(key=lambda item: (-int(item.get("priority") or 0), str(item.get("id") or "")))
    deterministic = [item for item in skills if item.get("type") == "deterministic_policy"]
    learned = [item for item in skills if item.get("origin") == "run_local_self_improvement"]
    supporting = [
        item for item in skills
        if item not in deterministic and item not in learned
    ]
    header = ("\n\nSKILL INSTRUCTION PRECEDENCE:\n"
              "1. The base system output schema and immutable facts are authoritative.\n"
              "2. Deterministic controls and manager arbitration directives constrain semantics only.\n"
              "3. Run-local learned rules and supporting directives are advisory.\n"
              "Skills must never add top-level response fields, wrap the response, rename fields, "
              "or change field types required by the base schema.\n"
              "Conflicting lower-priority directives are ignored, never merged.")
    fields = {str(value) for value in (schema_fields or [])}
    if not structured_output or {"learned_rule_ids", "learned_skill_nonuse_reason"}.intersection(fields):
        header += ("\nReport learned-rule use only through schema fields explicitly provided "
                   "by the base system; otherwise do not add reporting fields.")
    ordered = deterministic + learned + supporting
    directives: List[str] = []
    for skill in ordered:
        directives.extend(_render_directives(skill, role=role))
    directives = directives[:max(0, int(max_directives))]
    budget = max(0, int(max_skill_bytes))
    rendered = header if len(header.encode("utf-8")) <= budget else ""
    used = len(rendered.encode("utf-8"))
    omitted = len(directives) < sum(len(_render_directives(skill, role=role)) for skill in ordered)
    for directive in directives:
        block = "\n- " + directive
        if used + len(block.encode("utf-8")) > budget:
            omitted = True
            continue
        rendered += block
        used += len(block.encode("utf-8"))
    marker = "\n...[lower-priority skill directives truncated at directive boundary]"
    if omitted and used + len(marker.encode("utf-8")) <= budget:
        rendered += marker
    return str(base_system) + rendered


def _render_skills(skills: Sequence[Mapping[str, Any]]) -> str:
    rows: List[str] = []
    for skill in skills:
        guidance = [str(value) for value in skill.get("guidance", []) or []][:8]
        learned_rules = list(skill.get("learned_rules") or [])
        row: Dict[str, Any] = {
            "id": skill.get("id"),
            "priority": skill.get("priority"),
            "trigger_reason": skill.get("trigger_reason"),
            "guidance": guidance,
        }
        if learned_rules:
            row["rules"] = learned_rules
        if skill.get("deterministic_controls"):
            row["deterministic_controls"] = skill.get("deterministic_controls")
        rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return "\n".join("- " + row for row in rows)



def _skill_applies_to_role(skill: Mapping[str, Any], role: str) -> bool:
    metadata = skill.get("role_metadata") or {}
    if isinstance(metadata, Mapping):
        roles = metadata.get("roles") or metadata.get("role") or []
    else:
        roles = []
    if isinstance(roles, str):
        roles = [roles]
    legacy = skill.get("roles") or []
    if isinstance(legacy, str):
        legacy = [legacy]
    selected = {str(value) for value in list(roles) + list(legacy)}
    return not selected or role in selected or "*" in selected

def _render_directives(skill: Mapping[str, Any], *, role: Optional[str]) -> List[str]:
    metadata = skill.get("role_metadata") or {}
    role_directives = []
    if isinstance(metadata, Mapping) and role:
        value = metadata.get("directives") or {}
        if isinstance(value, Mapping):
            role_directives = value.get(role) or value.get("*") or []
    source = role_directives or skill.get("guidance") or []
    rows = []
    for index, value in enumerate(source):
        text = str(value).strip()
        if not text:
            continue
        rows.append(json.dumps({"skill_id": skill.get("id"), "priority": skill.get("priority"), "directive_index": index, "directive": text}, ensure_ascii=False, sort_keys=True))
    return rows
