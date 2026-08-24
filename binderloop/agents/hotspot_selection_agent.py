"""LLM agent that proposes primary hotspots from structure/physchem evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from binderloop.agents.prompt_catalog import compose_system, spec_for
from binderloop.agents.role import LLMStructuredAgent
from binderloop.analysis.hotspot_descriptors import (
    TargetResidueTable,
    compact_round_hotspot_evidence,
    deterministic_surface_hotspots,
    sanitize_hotspot_tokens,
)
from binderloop.config import HotspotSelectionSpec
from binderloop.llm import OpenAICompatibleClient

_HOTSPOT_SPEC = spec_for("HotspotSelectionAgent")
_PATH_RE = re.compile(r"(?i)(?:[A-Za-z]:)?(?:/|\\)[^\s\"']+\.(?:cif|pdb|mmcif|ent)")
_IDENTITY_KEYS = {
    "task_name", "protein_name", "pdb_id", "pdb", "uniprot", "structure_path",
    "structure_file", "notes", "target_name", "filename", "path",
}


@dataclass
class HotspotSelection:
    hotspots: List[str] = field(default_factory=list)
    rationale: str = ""
    expected_signal_next_round: str = ""
    changes_from_previous: List[str] = field(default_factory=list)
    llm_used: bool = False
    source: str = "deterministic_surface_heuristic"
    allow_web_search: bool = False
    identity_hidden: bool = True
    leakage_risk: Optional[str] = None
    sanitize_notes: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hotspots": list(self.hotspots),
            "rationale": self.rationale,
            "expected_signal_next_round": self.expected_signal_next_round,
            "changes_from_previous": list(self.changes_from_previous),
            "llm_used": self.llm_used,
            "source": self.source,
            "allow_web_search": self.allow_web_search,
            "identity_hidden": self.identity_hidden,
            "leakage_risk": self.leakage_risk,
            "sanitize_notes": list(self.sanitize_notes),
        }


class HotspotSelectionAgent(LLMStructuredAgent):
    name = "HotspotSelectionAgent"
    required_tags = _HOTSPOT_SPEC.required_tags
    output_schema = _HOTSPOT_SPEC.schema_fields
    system_sections = _HOTSPOT_SPEC.system_sections
    extra_system = _HOTSPOT_SPEC.extra_system
    temperature = 0.15
    SYSTEM = compose_system(*_HOTSPOT_SPEC.system_sections, extra=_HOTSPOT_SPEC.extra_system)

    def __init__(
        self,
        llm: Optional[OpenAICompatibleClient] = None,
        *,
        spec: Optional[HotspotSelectionSpec] = None,
        require_llm: Optional[bool] = None,
    ) -> None:
        self.spec = spec or HotspotSelectionSpec()
        super().__init__(llm, require_llm=self.spec.require_llm if require_llm is None else bool(require_llm))

    def select(
        self,
        *,
        residue_table: TargetResidueTable,
        previous_hotspots: Optional[Sequence[str]] = None,
        round_evidence: Optional[Mapping[str, Any]] = None,
        active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
        chain_id: Optional[str] = None,
    ) -> HotspotSelection:
        fallback = deterministic_surface_hotspots(
            residue_table,
            min_hotspots=self.spec.min_hotspots,
            max_hotspots=self.spec.max_hotspots,
        )
        previous = [str(item) for item in (previous_hotspots or []) if str(item).strip()]
        user = anonymize_hotspot_prompt({
            "residue_table": residue_table.prompt_payload(),
            "round_evidence": compact_round_hotspot_evidence({
                **dict(round_evidence or {}),
                "previous_hotspots": previous,
            }),
            "selection_limits": {
                "min_hotspots": self.spec.min_hotspots,
                "max_hotspots": self.spec.max_hotspots,
                "max_change_per_round": self.spec.max_change_per_round,
                "require_same_chain": True,
            },
        })
        call = self.call_json(
            system=self.composed_system(active_skills=active_skills),
            user=user,
            temperature=self.temperature,
            model_key=self.spec.model,
            allow_web_search=bool(self.spec.allow_web_search),
        )
        leakage_risk = "web_search_enabled" if self.spec.allow_web_search else None
        if not call.llm_used:
            selected, notes = sanitize_hotspot_tokens(
                previous or fallback,
                allowed_tokens=residue_table.tokens(),
                chain_id=chain_id or residue_table.chain_id,
                min_hotspots=self.spec.min_hotspots,
                max_hotspots=self.spec.max_hotspots,
                previous=previous,
                max_change_per_round=self.spec.max_change_per_round,
                fallback=fallback,
            )
            source = "deterministic_surface_heuristic"
            if call.source:
                source = "deterministic_fallback_after_%s" % call.source
            return HotspotSelection(
                hotspots=selected,
                rationale="Deterministic exposed hydrophobic/aromatic patch; LLM was not used.",
                expected_signal_next_round="Higher hotspot_contact if the surface patch is a true interface.",
                changes_from_previous=_diff_tokens(previous, selected),
                llm_used=False,
                source=source,
                allow_web_search=bool(self.spec.allow_web_search),
                leakage_risk=leakage_risk,
                sanitize_notes=notes,
                raw={"llm_error": call.error, "source": call.source, **dict(call.raw or {})},
            )
        result = dict(call.value or {})
        selected, notes = sanitize_hotspot_tokens(
            result.get("hotspots"),
            allowed_tokens=residue_table.tokens(),
            chain_id=chain_id or residue_table.chain_id,
            min_hotspots=self.spec.min_hotspots,
            max_hotspots=self.spec.max_hotspots,
            previous=previous,
            max_change_per_round=self.spec.max_change_per_round,
            fallback=fallback,
        )
        return HotspotSelection(
            hotspots=selected,
            rationale=str(result.get("rationale") or ""),
            expected_signal_next_round=str(result.get("expected_signal_next_round") or ""),
            changes_from_previous=list(result.get("changes_from_previous") or _diff_tokens(previous, selected)),
            llm_used=True,
            source="llm",
            allow_web_search=bool(self.spec.allow_web_search),
            leakage_risk=leakage_risk,
            sanitize_notes=notes,
            raw=result,
        )


def anonymize_hotspot_prompt(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop identity/path fields so closed-loop prompts stay anonymous."""
    return _scrub(payload)


def prompt_contains_identity(payload: Any, *, forbidden_tokens: Optional[Sequence[str]] = None) -> List[str]:
    text = _flatten_strings(payload)
    hits: List[str] = []
    if _PATH_RE.search(text):
        hits.append("file_path")
    for token in forbidden_tokens or []:
        if token and str(token) in text:
            hits.append(str(token))
    return hits


def _scrub(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            token = str(key)
            if token in _IDENTITY_KEYS or token.endswith("_path") or token.endswith("_file"):
                continue
            out[token] = _scrub(item)
        return out
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        if _PATH_RE.search(value):
            return "[redacted_path]"
        return value
    return value


def _flatten_strings(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_flatten_strings(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_strings(item) for item in value)
    return str(value or "")


def _diff_tokens(previous: Sequence[str], current: Sequence[str]) -> List[str]:
    prev = set(previous)
    cur = set(current)
    added = sorted(cur - prev)
    removed = sorted(prev - cur)
    changes = ["+%s" % token for token in added] + ["-%s" % token for token in removed]
    return changes
