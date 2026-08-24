"""Thin agent-role abstractions inspired by OpenAI Agents / LangGraph shapes.

This is not a dependency on those SDKs. LLM roles are one-shot structured JSON
nodes with deterministic fallbacks, not ReAct tool loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from binderloop.agents.context_compaction import context_digest, fact_check_text_against_metric_facts
from binderloop.agents.prompt_catalog import compose_system
from binderloop.llm import LLMConfigError, LLMTransportError, OpenAICompatibleClient
from binderloop.skills import compose_agent_system


class AgentRole:
    """Base role: named, tagged, schema-bearing unit of work."""

    name: str = ""
    required_tags: Tuple[str, ...] = ()
    output_schema: Tuple[str, ...] = ()

    def run(self, store: Mapping[str, Any]) -> Any:
        raise NotImplementedError


@dataclass
class StructuredCall:
    value: Optional[Dict[str, Any]] = None
    llm_used: bool = False
    error: Optional[str] = None
    source: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMStructuredAgent(AgentRole):
    """Shared chat_json / require_llm / fallback plumbing for JSON agents."""

    system_sections: Tuple[str, ...] = ()
    extra_system: str = ""
    temperature: float = 0.2
    max_tokens: int = 8000
    thinking: Optional[str] = None

    def __init__(
        self,
        llm: Optional[OpenAICompatibleClient] = None,
        *,
        require_llm: bool = False,
    ) -> None:
        self.llm = llm
        self.require_llm = bool(require_llm)

    def system_prompt(self, **kwargs: Any) -> str:
        return compose_system(*self.system_sections, extra=self.extra_system, **kwargs)

    def composed_system(self, *, active_skills: Any = None, role: Optional[str] = None) -> str:
        return compose_agent_system(
            self.system_prompt(),
            active_skills=active_skills,
            role=role or self.name or None,
        )

    def _ensure_llm(self) -> None:
        if self.require_llm and not (self.llm and self.llm.available()):
            raise RuntimeError(
                "%s: --require-llm is set but no LLM endpoint is available. "
                "Cannot fall back to deterministic rules." % (self.name or self.__class__.__name__)
            )

    def call_json(
        self,
        *,
        system: str,
        user: Mapping[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        thinking: Optional[str] = None,
        model_key: Optional[str] = None,
        allow_web_search: Optional[bool] = None,
    ) -> StructuredCall:
        self._ensure_llm()
        if not (self.llm and self.llm.available()):
            return StructuredCall(source="llm_unavailable")
        try:
            result = self.llm.chat_json(
                system=system,
                user=user,
                model_key=model_key,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                thinking=self.thinking if thinking is None else thinking,
                allow_web_search=allow_web_search,
            )
        except (LLMConfigError, LLMTransportError) as exc:
            if self.require_llm:
                raise
            return StructuredCall(
                error=str(exc),
                source="deterministic_fallback_after_llm_error",
                raw={"llm_error": "transport_or_config"},
            )
        except Exception as exc:
            if self.require_llm:
                raise
            return StructuredCall(
                error=str(exc),
                source="deterministic_fallback_after_llm_error",
                raw={"llm_error": str(exc)},
            )
        if isinstance(result, dict):
            return StructuredCall(value=result, llm_used=True, source="llm")
        return StructuredCall(raw={"llm_parse_failed": result}, source="llm_parse_failed")

    @staticmethod
    def digest(payload: Mapping[str, Any]) -> str:
        return context_digest(payload)

    @staticmethod
    def fact_check(text: str, metric_facts: Optional[Mapping[str, Any]]) -> list:
        return fact_check_text_against_metric_facts(text, metric_facts)
