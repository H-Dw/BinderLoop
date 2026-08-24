from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

import yaml


LLM_REASONING = "llm_reasoning"
STRATEGY = "strategy"
DETERMINISTIC_POLICY = "deterministic_policy"
VALID_SKILL_TYPES = {LLM_REASONING, STRATEGY, DETERMINISTIC_POLICY}


@dataclass
class SkillDefinition:
    """Structured guidance used by agents and policy modules.

    A skill is intentionally declarative. It can contribute prompt guidance,
    expected output schema, trigger rules and allowed config-key hints, but it
    must not directly mutate the executable config or override deterministic
    policy decisions.
    """

    skill_id: str
    skill_type: str
    description: str = ""
    applies_to: List[str] = field(default_factory=list)
    trigger: Dict[str, Any] = field(default_factory=dict)
    required_inputs: List[str] = field(default_factory=list)
    guidance: List[str] = field(default_factory=list)
    runtime_logic: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    allowed_config_keys: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    expected_signals: Dict[str, Any] = field(default_factory=dict)
    deterministic_controls: Dict[str, Any] = field(default_factory=dict)
    risk: str = ""
    priority: int = 0
    conflict_group: str = ""
    depends_on: List[str] = field(default_factory=list)
    excludes: List[str] = field(default_factory=list)
    origin: str = "static"
    version: str = "1"
    role_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "SkillDefinition":
        data = dict(item or {})
        skill_id = str(data.pop("id", data.pop("skill_id", ""))).strip()
        skill_type = str(data.pop("type", data.pop("skill_type", ""))).strip()
        if not skill_id:
            raise ValueError("Skill definition is missing id")
        if skill_type not in VALID_SKILL_TYPES:
            raise ValueError("Skill %s has invalid type %r" % (skill_id, skill_type))
        default_priority = {
            DETERMINISTIC_POLICY: 1000,
            STRATEGY: 500,
            LLM_REASONING: 300,
        }.get(skill_type, 0)
        return cls(
            skill_id=skill_id,
            skill_type=skill_type,
            description=str(data.pop("description", "")),
            applies_to=[str(x) for x in data.pop("applies_to", []) or []],
            trigger=dict(data.pop("trigger", {}) or {}),
            required_inputs=[str(x) for x in data.pop("required_inputs", []) or []],
            guidance=[str(x) for x in data.pop("guidance", []) or []],
            runtime_logic=dict(data.pop("runtime_logic", {}) or {}),
            output_schema=dict(data.pop("output_schema", {}) or {}),
            allowed_config_keys=[str(x) for x in data.pop("allowed_config_keys", []) or []],
            params=dict(data.pop("params", {}) or {}),
            expected_signals=dict(data.pop("expected_signals", {}) or {}),
            deterministic_controls=dict(data.pop("deterministic_controls", {}) or {}),
            risk=str(data.pop("risk", "")),
            priority=int(data.pop("priority", default_priority) or default_priority),
            conflict_group=str(data.pop("conflict_group", "")),
            depends_on=[str(x) for x in data.pop("depends_on", []) or []],
            excludes=[str(x) for x in data.pop("excludes", []) or []],
            origin=str(data.pop("origin", "static")),
            version=str(data.pop("version", "1")),
            role_metadata=dict(data.pop("role_metadata", {}) or {}),
            metadata=dict(data),
        )

    def to_activation(self, *, trigger_reason: str, missing_required_inputs: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Compact, context-safe representation for agent prompts/artifacts."""
        return {
            "id": self.skill_id,
            "type": self.skill_type,
            "description": self.description,
            "trigger_reason": trigger_reason,
            "required_inputs": list(self.required_inputs),
            "guidance": list(self.guidance),
            "runtime_logic": dict(self.runtime_logic),
            "output_schema": dict(self.output_schema),
            "allowed_config_keys": list(self.allowed_config_keys),
            "params": dict(self.params),
            "expected_signals": dict(self.expected_signals),
            "deterministic_controls": dict(self.deterministic_controls),
            "risk": self.risk,
            "priority": self.priority,
            "conflict_group": self.conflict_group,
            "depends_on": list(self.depends_on),
            "excludes": list(self.excludes),
            "origin": self.origin,
            "version": self.version,
            "role_metadata": dict(self.role_metadata),
            "missing_required_inputs": list(missing_required_inputs or []),
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["id"] = data.pop("skill_id")
        data["type"] = data.pop("skill_type")
        return data


class SkillRegistry:
    """Select structured skills for the current agent context."""

    def __init__(self, skills: Optional[Sequence[SkillDefinition]] = None, *, source_paths: Optional[Sequence[Union[str, Path]]] = None):
        self.skills = list(skills or [])
        self.source_paths = [str(p) for p in (source_paths or [])]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SkillRegistry":
        path = Path(path)
        if path.is_dir():
            files = sorted([p for p in path.rglob("*.yaml") if p.is_file()] + [p for p in path.rglob("*.yml") if p.is_file()])
            return cls.from_paths(files)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, Mapping) and data.get("include"):
            include_paths = []
            for item in data.get("include") or []:
                include_path = Path(str(item))
                if not include_path.is_absolute():
                    include_path = path.parent / include_path
                include_paths.append(include_path)
            registry = cls.from_paths(include_paths)
            registry.source_paths.insert(0, str(path))
            return registry
        if isinstance(data, Mapping) and data.get("skill"):
            raw_skills = [data.get("skill")]
        elif isinstance(data, Mapping) and data.get("id") and data.get("type"):
            raw_skills = [data]
        else:
            raw_skills = data.get("skills", data if isinstance(data, list) else [])
        if not isinstance(raw_skills, list):
            raise ValueError("Skill registry YAML must contain a top-level skills list")
        skills = [SkillDefinition.from_mapping(item) for item in raw_skills]
        return cls(skills, source_paths=[path])

    @classmethod
    def from_paths(cls, paths: Iterable[Union[str, Path]]) -> "SkillRegistry":
        skills: List[SkillDefinition] = []
        source_paths: List[str] = []
        seen_skill_ids: Set[str] = set()
        for path in paths:
            p = Path(path)
            if not p.exists():
                continue
            registry = cls.from_yaml(p)
            for skill in registry.skills:
                if skill.skill_id in seen_skill_ids:
                    continue
                seen_skill_ids.add(skill.skill_id)
                skills.append(skill)
            source_paths.extend(registry.source_paths)
        return cls(skills, source_paths=source_paths)

    @classmethod
    def empty(cls) -> "SkillRegistry":
        return cls([])

    def select(
        self,
        *,
        agent_name: str,
        context: Mapping[str, Any],
        skill_types: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        allowed_types = set(skill_types or VALID_SKILL_TYPES)
        candidates: List[Dict[str, Any]] = []
        for skill in self.skills:
            if skill.skill_type not in allowed_types:
                continue
            if not self._applies_to(skill, agent_name):
                continue
            matched, reason = self._trigger_matches(skill.trigger, context)
            if not matched:
                continue
            missing = self._missing_required_inputs(skill.required_inputs, context)
            if missing and bool(skill.runtime_logic.get("strict_required_inputs")):
                continue
            candidates.append(
                skill.to_activation(
                    trigger_reason=reason,
                    missing_required_inputs=missing,
                )
            )
        candidates.sort(
            key=lambda item: (
                -int(item.get("priority") or 0),
                str(item.get("id") or ""),
            )
        )
        activations: List[Dict[str, Any]] = []
        candidate_ids = {str(item.get("id")) for item in candidates}
        selected_ids: Set[str] = set()
        selected_groups: Set[str] = set()
        for candidate in candidates:
            dependencies = set(candidate.get("depends_on") or [])
            if dependencies and not dependencies.issubset(candidate_ids):
                continue
            if selected_ids.intersection(set(candidate.get("excludes") or [])):
                continue
            group = str(candidate.get("conflict_group") or "")
            if group and group in selected_groups:
                continue
            activations.append(candidate)
            selected_ids.add(str(candidate.get("id")))
            if group:
                selected_groups.add(group)
            if limit is not None and len(activations) >= int(limit):
                break
        return activations

    def by_type(self, skill_type: str) -> List[Dict[str, Any]]:
        return [skill.to_dict() for skill in self.skills if skill.skill_type == skill_type]

    def audit_summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {key: 0 for key in sorted(VALID_SKILL_TYPES)}
        for skill in self.skills:
            counts[skill.skill_type] = counts.get(skill.skill_type, 0) + 1
        return {
            "source_paths": list(self.source_paths),
            "skill_count": len(self.skills),
            "counts_by_type": counts,
            "skill_ids": [skill.skill_id for skill in self.skills],
        }

    @staticmethod
    def _applies_to(skill: SkillDefinition, agent_name: str) -> bool:
        applies = skill.applies_to or ["*"]
        return "*" in applies or agent_name in applies

    def _trigger_matches(self, trigger: Mapping[str, Any], context: Mapping[str, Any]) -> tuple:
        trigger = dict(trigger or {})
        if not trigger or bool(trigger.get("always")):
            return True, "always"

        tag_paths = list(trigger.get("tag_paths") or [
            "evaluation.tag_counts",
            "evaluation.metric_facts.tag_counts",
            "structural_analysis.aggregate_tags",
        ])
        context_tags = self._collect_tags(context, tag_paths)

        tags_any = {str(x) for x in trigger.get("tags_any", []) or []}
        if tags_any and context_tags.intersection(tags_any):
            return True, "matched_tags_any:" + ",".join(sorted(context_tags.intersection(tags_any)))

        tags_all = {str(x) for x in trigger.get("tags_all", []) or []}
        if tags_all and tags_all.issubset(context_tags):
            return True, "matched_tags_all:" + ",".join(sorted(tags_all))

        if self._thresholds_match(trigger.get("metric_thresholds") or {}, context):
            return True, "matched_metric_thresholds"

        if self._paths_truthy(trigger.get("paths_truthy") or [], context):
            return True, "matched_paths_truthy"

        if bool(trigger.get("fallback_when_no_match")):
            return True, "fallback_when_no_match"
        return False, "no_trigger_match"

    @classmethod
    def _collect_tags(cls, context: Mapping[str, Any], paths: Sequence[str]) -> Set[str]:
        tags: Set[str] = set()
        for path in paths:
            value = cls._get_path(context, path)
            if isinstance(value, Mapping):
                for key, count in value.items():
                    try:
                        active = float(count or 0) > 0
                    except (TypeError, ValueError):
                        active = bool(count)
                    if active:
                        tags.add(str(key))
            elif isinstance(value, list):
                tags.update(str(x) for x in value if x)
        return tags

    @classmethod
    def _thresholds_match(cls, thresholds: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        matched_any = False
        for path, rule in dict(thresholds or {}).items():
            value = cls._get_path(context, str(path))
            try:
                current = float(value)
            except (TypeError, ValueError):
                continue
            if isinstance(rule, Mapping):
                if "lt" in rule and current < float(rule["lt"]):
                    matched_any = True
                if "lte" in rule and current <= float(rule["lte"]):
                    matched_any = True
                if "gt" in rule and current > float(rule["gt"]):
                    matched_any = True
                if "gte" in rule and current >= float(rule["gte"]):
                    matched_any = True
            elif current == float(rule):
                matched_any = True
        return matched_any

    @classmethod
    def _paths_truthy(cls, paths: Sequence[str], context: Mapping[str, Any]) -> bool:
        for path in paths:
            if cls._get_path(context, str(path)):
                return True
        return False

    @staticmethod
    def _get_path(data: Any, path: str) -> Any:
        current = data
        for part in str(path or "").split("."):
            if not part:
                continue
            if isinstance(current, Mapping):
                current = current.get(part)
            else:
                return None
        return current

    @classmethod
    def _missing_required_inputs(
        cls,
        required_inputs: Sequence[str],
        context: Mapping[str, Any],
    ) -> List[str]:
        return [
            str(path)
            for path in required_inputs or []
            if not cls._path_has_value(context, str(path).split("."))
        ]

    @classmethod
    def _path_has_value(cls, current: Any, parts: Sequence[str]) -> bool:
        if not parts:
            return current is not None
        if isinstance(current, Mapping):
            key = parts[0]
            if key not in current:
                return False
            return cls._path_has_value(current.get(key), parts[1:])
        if isinstance(current, list):
            return any(cls._path_has_value(item, parts) for item in current)
        return False
