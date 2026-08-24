"""Build a tagged RoundContextStore and assemble per-role prompts.

Existing ``compact_context_for_*`` helpers remain the projection backend so
live agent call sites stay compatible while scripts can take tagged slices
without importing each agent class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from binderloop.agents.context_compaction import (
    compact_config,
    compact_context_for_config_validation,
    compact_context_for_diagnostic,
    compact_context_for_hypothesis,
    compact_context_for_input_config,
    compact_context_for_quality,
    compact_diagnostic_report,
    compact_hypotheses,
    compact_memory,
    compact_quality_analysis,
    compact_structural_analysis,
    compact_target_profile,
    enforce_byte_budget,
    MAX_PROMPT_BYTES,
)
from binderloop.analysis.candidate_clusters import (
    aggregate_candidate_phenotypes,
    compact_cluster_cards,
)
from binderloop.agents.prompt_catalog import (
    PROMPT_VERSION,
    AgentPromptSpec,
    compose_system,
    spec_for,
)
from binderloop.skills import compose_agent_system


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))


@dataclass
class RoundContextStore:
    """One materialized round of tagged slices."""

    slices: Dict[str, Any] = field(default_factory=dict)
    compact_by_role: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    prompt_version: str = PROMPT_VERSION

    def get(self, tag: str, default: Any = None) -> Any:
        return self.slices.get(tag, default)

    def project(self, spec: AgentPromptSpec) -> Dict[str, Any]:
        user: Dict[str, Any] = {
            "prompt_version": self.prompt_version,
            "role": spec.role,
            "task": {"goal": spec.goal, "round_id": self.slices.get("task.round_id")},
        }
        for tag in spec.required_tags:
            if tag == "task.round_id":
                continue
            if tag == "candidates.leaves" and not spec.include_leaves:
                continue
            if tag in self.slices and self.slices.get(tag) not in (None, {}, []):
                user[tag] = self.slices[tag]
        return user


def load_round_artifacts(round_dir) -> Dict[str, Any]:
    """Load durable round JSON into the dict ``build_store`` expects."""
    from pathlib import Path

    root = Path(round_dir)
    def _read(name: str, default: Any = None) -> Any:
        path = root / name
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    evaluation = _as_dict(_read("evaluation_summary.json", {}))
    structural = _as_dict(_read("structure_evaluation.json", {}))
    al_examples = _as_dict(_read("active_learning_examples.json", {}))
    checkpoint = _as_dict(_read("round_checkpoint.json", {}))
    clusters = _as_dict(_read("candidate_clusters.json", {}))
    quality = _as_dict(_read("binder_quality_analysis.json", {}))
    hypotheses = _as_dict(_read("hypotheses.json", {}))
    diagnostic = _as_dict(_read("diagnostic_report.json", {}))
    skills = _as_dict(_read("active_skills.json", {}))
    execution_records = list(_read("execution_records.json", []) or [])
    jobs = list(checkpoint.get("current_jobs") or [])
    if not jobs and execution_records:
        jobs = [dict(item.get("job") or {}) for item in execution_records if item.get("job")]
    job = _as_dict(jobs[0] if jobs else {})
    current_config = _as_dict(job.get("params"))
    current_config.update({
        "task_name": current_config.get("task_id") or current_config.get("task_name") or "target",
        "target": current_config.get("target") or {
            "structure_path": job.get("target_structure"),
            "chain_id": job.get("chain_id"),
            "hotspots": job.get("hotspots") or [],
        },
        "binder_lengths": current_config.get("binder_lengths")
        or current_config.get("binder_length_range")
        or ([job.get("binder_length")] if job.get("binder_length") else []),
    })
    memory_path = root.parent / "memory" / "experiment_memory.json"
    memory = _as_dict(json.loads(memory_path.read_text(encoding="utf-8"))) if memory_path.exists() else {}
    if not memory:
        memory = {
            "recent_rounds": [],
            "experiment_id": str(root.parent.name),
        }
    round_id = int(checkpoint.get("round_id") or 0)
    if al_examples:
        evaluation["active_learning_examples"] = al_examples
    return {
        "round_id": round_id,
        "evaluation": evaluation,
        "evaluation_summary": evaluation,
        "metric_facts": evaluation.get("metric_facts"),
        "active_learning_examples": al_examples,
        "structural_analysis": structural,
        "structure_evaluation": structural,
        "candidate_clusters": clusters,
        "current_config": current_config,
        "config": current_config,
        "memory": memory,
        "memory_summary": memory,
        "target_profile": _as_dict(current_config.get("target")),
        "target_analysis": _as_dict(current_config.get("target")),
        "constraints": {
            "max_binders_per_round": current_config.get("max_binders_per_round"),
            "binder_length_range": current_config.get("binder_length_range"),
            "epitope_crop_disabled_hard_constraint": current_config.get("epitope_crop_mode") == "disabled",
        },
        "quality_analysis": quality,
        "hypotheses": hypotheses.get("hypotheses") or hypotheses,
        "diagnostic_report": diagnostic,
        "diagnostic": diagnostic,
        "monitor": {
            "state": "completed",
            "is_terminal": True,
            "is_success": True,
            "status_counts": {"completed": len(execution_records)},
        },
        "active_skills": skills,
        "arm_comparison": _as_dict(_read("arm_comparison.json", {})),
        "blocked_arms": _read("blocked_arms.json", []),
        "ledger.compact": _as_dict(_read("arm_history_resolution.json", {})),
    }


def build_store(round_artifacts: Mapping[str, Any]) -> RoundContextStore:
    """Project round artifacts (or a live orchestrator context) into tagged slices."""
    data = _as_dict(round_artifacts)
    evaluation = _as_dict(data.get("evaluation") or data.get("evaluation_summary"))
    structural = _as_dict(data.get("structural_analysis") or data.get("structure_evaluation"))
    al_examples = _as_dict(
        data.get("active_learning_examples")
        or evaluation.get("active_learning_examples")
    )
    raw_clusters = data.get("candidate_clusters")
    if not (isinstance(raw_clusters, Mapping) and (raw_clusters.get("clusters") or raw_clusters.get("cluster_count"))):
        raw_clusters = aggregate_candidate_phenotypes(
            round_id=int(data.get("round_id") or 0),
            evaluation=evaluation,
            active_learning_examples=al_examples,
            structural_analysis=structural,
        )
    prompt_clusters = compact_cluster_cards(raw_clusters)
    data = dict(data)
    data["candidate_clusters"] = prompt_clusters
    clusters = prompt_clusters
    quality = compact_context_for_quality(data)
    hypothesis = compact_context_for_hypothesis(data)
    diagnostic = compact_context_for_diagnostic(
        round_id=int(data.get("round_id") or 0),
        monitor_snapshot=data.get("monitor") or data.get("monitor_snapshot"),
        metrics_summary=data.get("metrics_summary") or evaluation.get("metric_facts"),
        evaluation_summary=evaluation or None,
        structural_analysis=structural or None,
        job_history=data.get("job_history") or _as_dict(data.get("memory")).get("recent_rounds"),
        config=data.get("current_config") or data.get("config"),
        candidate_clusters=prompt_clusters,
    )
    memory = compact_memory(data.get("memory") or data.get("memory_summary"))
    slices: Dict[str, Any] = {
        "task.round_id": data.get("round_id"),
        "facts.metric": evaluation.get("metric_facts") or data.get("metric_facts") or quality.get("evaluation", {}).get("metric_facts"),
        "facts.gates": _as_dict(evaluation.get("metric_facts")).get("gate_denominators"),
        "examples.al_current": _as_dict(al_examples.get("current_round")),
        "examples.al_prior": _as_dict(al_examples.get("prior_rounds")),
        "examples.al_clusters": clusters.get("clusters"),
        "candidates.clusters": clusters,
        "candidates.representatives": clusters.get("representatives"),
        "candidates.leaves": list(_as_dict(raw_clusters).get("leaves") or data.get("candidates.leaves") or []),
        "structure.aggregates": compact_structural_analysis(structural, include_summaries=False),
        "structure.fragments_diverse": quality.get("structural_analysis"),
        "structure.templates_ids": data.get("fragment_templates"),
        "structure.phenotype_clusters": _as_dict(data.get("candidate_clusters")).get("structure_clusters"),
        "config.current": compact_config(data.get("current_config") or data.get("config")),
        "constraints.hard": _as_dict(data.get("constraints")),
        "target.profile": compact_target_profile(data.get("target_profile") or data.get("target_analysis")),
        "execution.monitor": diagnostic.get("monitor"),
        "execution.errors": _as_dict(data.get("execution_failure") or data.get("error_context")),
        "memory.retrieved": memory,
        "upstream.quality": compact_quality_analysis(data.get("quality_analysis")),
        "upstream.hypotheses": compact_hypotheses(data.get("hypotheses")),
        "upstream.diagnostic": compact_diagnostic_report(data.get("diagnostic_report") or data.get("diagnostic")),
        "upstream.arm_comparison": _as_dict(data.get("arm_comparison")),
        "arms.evidence": data.get("arm_evidence") or data.get("arms.evidence"),
        "arms.blocked": data.get("blocked_arms") or data.get("arms.blocked"),
        "ledger.compact": _as_dict(data.get("ledger_history") or data.get("ledger.compact")),
    }
    return RoundContextStore(
        slices={key: value for key, value in slices.items() if value not in (None, {}, [])},
        compact_by_role={
            "HypothesisAgent": hypothesis,
            "BinderQualityAnalysisAgent": quality,
            "DiagnosticCoachAgent": diagnostic,
            "InputConfigurationAgent": compact_context_for_input_config(
                target_name=str(data.get("target_name") or "target"),
                current_config=_as_dict(data.get("current_config") or data.get("config")),
                diagnostic_report=_as_dict(data.get("diagnostic_report") or data.get("diagnostic")),
                evaluation_summary=evaluation,
                round_id=int(data.get("round_id") or 0) + 1,
                target_profile=data.get("target_profile") or data.get("target_analysis"),
                structural_analysis=structural or None,
                quality_analysis=data.get("quality_analysis"),
                hypotheses=data.get("hypotheses"),
                memory_summary=data.get("memory") or data.get("memory_summary"),
                constraints=data.get("constraints"),
                tuning_feedback=data.get("tuning_feedback"),
            ) if (data.get("current_config") or data.get("config")) else {},
            "ConfigValidationAgent": compact_context_for_config_validation(
                target_model=str(data.get("target_model") or "boltzgen"),
                activation=str(data.get("activation") or "pre_submit"),
                config=_as_dict(data.get("current_config") or data.get("config")),
                deterministic_prefilter=_as_dict(data.get("deterministic_prefilter")),
                context=data,
            ) if (data.get("current_config") or data.get("config") or data.get("deterministic_prefilter")) else {},
        },
    )


def assemble(
    role: str,
    store: RoundContextStore,
    *,
    active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
    max_bytes: int = MAX_PROMPT_BYTES,
    tagged: bool = True,
) -> Dict[str, Any]:
    """Return ``{system, user, spec, user_bytes}`` for one agent role."""
    spec = spec_for(role)
    system = compose_system(*spec.system_sections, extra=spec.extra_system)
    if active_skills:
        system = compose_agent_system(system, active_skills=active_skills, role=role)
    if tagged:
        user = store.project(spec)
    else:
        user = dict(store.compact_by_role.get(role) or store.project(spec))
        user.pop("active_skills", None)
    user = enforce_byte_budget(user, max_bytes=max_bytes)
    return {
        "role": spec.role,
        "goal": spec.goal,
        "prompt_version": store.prompt_version,
        "required_tags": list(spec.required_tags),
        "system": system,
        "user": user,
        "user_bytes": _json_bytes(user),
        "system_bytes": len(system.encode("utf-8")),
    }


def assemble_legacy_user(role: str, store: RoundContextStore) -> Dict[str, Any]:
    """Backend compact dict used by live agent call sites."""
    return dict(store.compact_by_role.get(role) or {})
