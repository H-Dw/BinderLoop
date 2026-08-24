"""Runtime skill registry for binder-harness agents.

Skills are structured guidance and trigger rules. They never execute model
changes directly; agents and deterministic controllers keep their existing
sanitizers, bounds and scoring logic as the source of truth.
"""

from .registry import SkillDefinition, SkillRegistry
from .self_improvement import (
    SelfImprovementSkillError,
    SelfImprovementSkillHandle,
    SelfImprovementSkillStore,
    SkillDocumentEditor,
    LearnedStrategyRule,
)
from .composer import compose_agent_system

__all__ = [
    "SkillDefinition",
    "SkillRegistry",
    "SelfImprovementSkillError",
    "SelfImprovementSkillHandle",
    "SelfImprovementSkillStore",
    "SkillDocumentEditor",
    "LearnedStrategyRule",
    "compose_agent_system",
]
