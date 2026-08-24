
import concurrent.futures
import copy
import json
import hashlib
import math
import os
import re
import random
import threading
import time
import warnings
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import yaml

from binderloop.active_learning.strategy import StrategyLevelActiveLearner
from binderloop.active_learning.examples import build_active_learning_examples, prior_examples_from_memory
from binderloop.active_learning.rollback import RollbackController, RollbackDecision, RoundOutcome, round_reward
from binderloop.agents import ActiveLearningPolicyAgent, BinderQualityAnalysisAgent, EvaluationAgent, ResultIngestionAgent, DiagnosticCoachAgent, FragmentTemplateMiningAgent, InputConfigurationAgent, SelfImprovementSkillAgent, SelfImprovementUpdate, StrategyConflictResolutionAgent, StrategyConflictResolution, StrategyArmRankingAgent, BlockedArmReviewAgent, BlockedArmReviewDecision, HotspotSelectionAgent
from binderloop.agents.context_compaction import blocked_arm_ledger_view, build_metric_facts, compact_structural_aggregate_from_object, top_candidates_by_core, top_candidates_by_iptm
from binderloop.agents.active_learning_policy_agent import NextRoundParameterProposal
from binderloop.agents.binder_length_policy_agent import BinderLengthPolicyAgent, BinderLengthRecommendation
from binderloop.agents.binder_quality_analysis_agent import BinderQualityAnalysis
from binderloop.agents.binder_quality_collaboration_agent import (
    BinderQualityCollaborationAgent,
    QualityCollaborationController,
)
from binderloop.agents.evaluation_agent import CandidateEvaluation, EvaluationSummary
from binderloop.agents.fragment_template_mining_agent import FragmentTemplate, FragmentTemplateBatch
from binderloop.agents.result_ingestion_agent import IngestedBoltzGenRun, ResultPathSafetyError, TransportBindingError
from binderloop.package_layout import is_project_package_name, package_dir_candidates
from binderloop.strategy_governance import (
    ArmApplicability,
    CandidateIntervention,
    assess_candidate_intervention,
    deduplicate_effective_jobs,
    effective_semantic_digest,
    finalize_immutable_branch_plan,
    job_identity_semantic_digest,
    materialize_deterministic_job_identities,
    safe_path_component,
    resolved_within,
)
from binderloop.agents.config_parameter_contract import (
    PARAM_BOUNDS,
    clamp_config_with_inertia,
    enforce_single_primary_family,
    invalid_config_value_keys,
    parameter_contract_entry,
    partition_config_parameters,
    supported_config_changes,
    unsupported_config_keys,
)
from binderloop.agents.diagnostic_coach_agent import DiagnosticReport
from binderloop.agents.hypothesis_agent import HypothesisAgent, HypothesisSet
from binderloop.agents.input_configuration_agent import InputConfiguration
from binderloop.agents.memory_compression_agent import MemoryCompressionAgent
from binderloop.agents.memory_retrieval_agent import MemoryRetrievalAgent, MemoryRetrievalQuery
from binderloop.agents.strategy_conflict_resolution_agent import detect_strategy_conflicts
from binderloop.agents.structure_evaluation_agent import StructureBatchEvaluation, StructureEvaluationAgent
from binderloop.analysis.candidate_clusters import aggregate_candidate_phenotypes, compact_cluster_cards
from binderloop.analysis.core_objective import (
    core_metric_stats,
    monitoring_scalar_from_round_rank,
    round_core_objective,
    round_rank_key,
)
from binderloop.analysis.structure_features import analyze_target_structure
from binderloop.analysis.hotspot_descriptors import build_target_residue_table
from binderloop.communication import AgentMessage, MessageBus
from binderloop.config import HarnessConfig, binder_generation_cap, _expand_length_range, primary_design_model
from binderloop.models.search_profile import get_model_search_profile
from binderloop.parameter_decision import (
    HOLD_CURRENT, PROBABILISTIC_SAMPLER_KEYS, ParameterCandidate, ParameterDecisionSpec,
    decide_parameter_distribution,
    deterministic_sampler_states, joint_parameter_evidence_from_rounds, parameter_axis,
    parameter_catalog_digest, sampler_keys_for_spec,
)
from binderloop.execution_error_summary import sanitize_error_text
from binderloop.execution_governance import (
    bind_template_application_budget,
    build_template_application_plan,
    resolve_round_budget,
    validate_template_application,
)
from binderloop.memory import (
    ExperimentMemoryStore,
    build_round_memory_item,
    parameter_diff,
    target_memory_key,
)
from binderloop.orchestration.round_graph import RoundGraph
from binderloop.models.base import DesignJob
from binderloop.templates.outcome_ledger import OutcomeLedger, update_ledger_from_round
from binderloop.resume import (
    ArtifactDigestCache,
    artifact_record,
    artifacts_match,
    atomic_write_json,
    atomic_write_text,
    build_template_execution_identity,
    classify_template_replay,
    stable_hash,
)
from binderloop.skills import SkillRegistry, SelfImprovementSkillStore
from binderloop.skills.self_improvement import (
    active_prompt_rules,
    apply_lifecycle,
    apply_semantic_relations,
    mark_rules_contested,
    record_conflict_decisions,
    settle_conflicts_from_operations,
    validate_skill_document,
)
from binderloop.visualization.iteration_metrics_plot import (
    IterationMetricsInputError,
    IterationMetricsNoDataError,
    IterationMetricsRoundCache,
    build_round_analysis_bundle,
    plot_iteration_metrics,
)


class ModuleOutputValidationError(RuntimeError):
    """Raised when a module output cannot satisfy the next module's input contract."""


class BinderDesignOrchestrator:
    """Closed-loop global scheduler with memory, bounded parallelism and retry."""

    RECOVERY_ACTIONS = frozenset({"replay_best", "retest_best_config", "branch_from_best"})
    FAILURE_STATUSES = {"failed", "error", "not_executed", "timeout", "cancelled", "canceled"}
    # Complete set understood by the current harness when reconstructing legacy
    # (pre-boltzgen_config) snapshots. New snapshots preserve the entire dict.
    BOLTZGEN_RESTORE_KEYS = frozenset({
        "hotspot_weight", "budget", "protocol", "diffusion_batch_size",
        "step_scale", "noise_scale", "inverse_fold_num_sequences", "inverse_fold_avoid",
        "alpha", "refolding_rmsd_threshold", "filter_biased", "steps", "analysis_location",
        "num_workers", "use_kernels", "run_filtering", "keep_unfiltered_for_failure_analysis",
        "additional_filters", "config_overrides", "clash_filter", "target_include",
        "target_binding_types", "structure_groups", "binder_chain", "binder_structure_prior",
        "residue_constraints", "binder_binding_types", "length_delta_hint",
        "avoid_binder_lengths", "prioritize_hotspots", "auxiliary_hotspots",
        "exploit_fragment_modules", "module_guided_exploitation", "module_guided_repair",
        "epitope_crop_mode", "auto_binder_length", "fragment_template_gate",
        "fragment_interchain_pae_max", "fragment_templates_enabled",
        "fragment_template_top_k", "binder_template", "binder_templates",
        "binder_template_proximity", "template_conditioned_fraction",
    })
    RESOURCE_CONFIG_KEYS = ("GPUName", "devices", "taiji_timeout")
    RESOURCE_SCHEDULING_FAILURE_NEEDLES = (
        "resource_scheduling_failure",
        "taiji_resource_or_queue_issue",
        "pending timeout",
        "state pending timeout",
        "resource exhausted",
        "no available resource",
        "quota",
        "queued",
        "evicted",
    )

    def __init__(self, cfg: HarnessConfig, *, out_dir: Union[str, Optional[Path]] = None, max_rounds: Optional[int] = None, max_parallel: Optional[int] = None, max_retries: Optional[int] = None, llm=None, require_llm: bool = False):
        self.cfg = cfg
        self.require_llm = bool(require_llm)
        self.out_dir = Path(out_dir or cfg.runtime.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._round_design_cap = binder_generation_cap(cfg)
        self._round_num_designs = min(max(1, int(cfg.search_space.num_designs_per_round or 1)), self._round_design_cap)
        self.max_rounds = max_rounds or cfg.active_learning.max_rounds or 5
        requested_parallel = max_parallel or cfg.resource.max_parallel_jobs or 1
        self.max_parallel = min(max(1, requested_parallel), max(1, cfg.resource.max_parallel_jobs))
        host_shards = self._taiji_host_shard_count()
        if host_shards > 1 and not self._native_taiji_multi_host():
            # Multi-host Taiji runs are represented as several single-host jobs.
            # Each job still fans out across its own GPUs internally; the executor
            # waits for all submitted jobs before analysis.
            parallel_limit = min(host_shards, max(1, cfg.resource.max_parallel_jobs))
            self.max_parallel = min(max(1, requested_parallel), parallel_limit) if max_parallel is not None else parallel_limit
        elif self._native_taiji_multi_host() or int(getattr(cfg.resource, "host_gpu_num", 1) or 1) > 1:
            # Multi-GPU BoltzGen jobs use all requested GPUs inside one shell
            # script. Native multi-host mode also represents the whole cluster as
            # one Taiji task, so another parallel submission would over-reserve.
            self.max_parallel = 1
        configured_retries = max_retries if max_retries is not None else cfg.active_learning.max_retries
        # Keep the historical behavior conservative: this value is the maximum
        # number of submit attempts per job, including the initial submit.
        self.max_retries = max(1, int(configured_retries if configured_retries is not None else 3))
        self.bus = MessageBus(self.out_dir / "agent_messages.jsonl")
        self.memory_store = ExperimentMemoryStore(self.out_dir / "memory", experiment_id=f"binder_{int(time.time())}")
        self._artifact_digest_cache = ArtifactDigestCache()
        self._io_telemetry: Dict[str, Any] = {
            "json_writes": 0, "json_bytes_written": 0,
            "ingestion_seconds": 0.0, "structure_seconds": 0.0,
        }
        self.template_outcome_ledger = OutcomeLedger.open(self.out_dir / "template_outcome_ledger.json")
        target_path = Path(str(self.cfg.target.structure_path or ""))
        target_sha = artifact_record(target_path).get("sha256") if target_path.is_file() else ""
        self._target_identity_digest = stable_hash({"structure_sha256": target_sha, "chain": self.cfg.target.chain_id})
        self.learner = StrategyLevelActiveLearner(exploration_ratio=cfg.active_learning.exploration_ratio)
        self.rollback = RollbackController(
            enabled=bool(getattr(cfg.active_learning, "enable_backtracking", True)),
            regression_tolerance=float(getattr(cfg.active_learning, "regression_tolerance", 0.15)),
            patience=int(getattr(cfg.active_learning, "rollback_patience", 1)),
        )
        self.ingestor = ResultIngestionAgent()
        self.evaluator = EvaluationAgent()
        self.structure_agent = StructureEvaluationAgent()
        self.fragment_template_agent = FragmentTemplateMiningAgent()
        self.length_policy_agent = BinderLengthPolicyAgent()
        self.policy_agent = ActiveLearningPolicyAgent()
        self.hypothesis_agent = HypothesisAgent(llm=llm, require_llm=self.require_llm)
        self.quality_agent = BinderQualityAnalysisAgent(llm=llm, require_llm=self.require_llm)
        self.round_graph = RoundGraph()
        collaboration_cfg = getattr(cfg, "quality_collaboration", None)
        self.quality_collaboration_agent = BinderQualityCollaborationAgent(
            llm=llm,
            request_timeout_seconds=int(
                getattr(collaboration_cfg, "request_timeout_seconds", 105) or 105
            ),
            failure_cooldown_seconds=int(
                getattr(collaboration_cfg, "failure_cooldown_seconds", 20) or 0
            ),
            specialist_max_tokens=int(
                getattr(collaboration_cfg, "specialist_max_tokens", 1400) or 1400
            ),
            manager_max_tokens=int(
                getattr(collaboration_cfg, "manager_max_tokens", 1800) or 1800
            ),
            specialist_reasoning_mode=str(getattr(collaboration_cfg, "specialist_reasoning_mode", "low") or "low"),
            specialist_output_tokens=None,
            manager_reasoning_mode=str(getattr(collaboration_cfg, "manager_reasoning_mode", "low") or "low"),
            manager_output_tokens=None,
            reasoning_budget_tokens=int(getattr(collaboration_cfg, "reasoning_budget_tokens", 0) or 0),
            visible_json_budget_tokens=int(getattr(collaboration_cfg, "visible_json_budget_tokens", 8000) or 8000),
            max_completion_tokens=int(getattr(collaboration_cfg, "max_completion_tokens", 65_536) or 65_536),
            max_revisions=getattr(collaboration_cfg, "max_revisions", None),
            final_max_tokens=getattr(collaboration_cfg, "final_max_tokens", None),
            max_api_calls=int(
                getattr(collaboration_cfg, "max_api_calls", 6) or 6
            ),
            cache_dir=self.out_dir / "quality_collaboration_cache",
        )
        self.diagnostic_coach = DiagnosticCoachAgent(llm=llm, require_llm=self.require_llm)
        decision_spec = getattr(getattr(cfg, "owner", None), "parameter_decision", None)
        profile = get_model_search_profile(primary_design_model(cfg), cfg=cfg)
        sampler_keys = sampler_keys_for_spec(decision_spec) if decision_spec is not None else profile.sampler_axes
        self.input_config_agent = InputConfigurationAgent(
            llm=llm, require_llm=self.require_llm,
            parameter_candidates=({key: parameter_axis(decision_spec, key) for key in sampler_keys} if decision_spec is not None else None),
            adjustable_parameters=profile.adjustable_parameters,
            param_bounds=profile.param_bounds,
            sampler_axes=tuple(sampler_keys),
        )
        self._llm = llm
        self.self_improvement_agent = SelfImprovementSkillAgent(
            llm=llm,
            require_llm=self.require_llm,
            semantic_candidate_limit=int(
                getattr(getattr(cfg, "self_improvement", None), "semantic_candidate_limit", 8) or 8
            ),
            semantic_confidence_threshold=float(
                getattr(getattr(cfg, "self_improvement", None), "semantic_confidence_threshold", 0.72) or 0.72
            ),
            prompt_max_bytes=int(
                getattr(getattr(cfg, "self_improvement", None), "prompt_max_bytes", 24_000) or 24_000
            ),
            cache_dir=self.out_dir / "self_improvement_semantic_cache",
            reward_improvement_threshold=float(
                getattr(getattr(cfg, "self_improvement", None), "reward_improvement_threshold", 0.01) or 0.0
            ),
            strong_improvement_threshold=float(
                getattr(getattr(cfg, "self_improvement", None), "strong_improvement_threshold", 0.05) or 0.05
            ),
        )
        self.strategy_conflict_agent = StrategyConflictResolutionAgent(
            llm=llm,
            require_llm=self.require_llm,
        )
        self.strategy_arm_ranking_agent = StrategyArmRankingAgent(
            llm=llm,
            require_llm=self.require_llm,
        )
        self.blocked_arm_review_agent = BlockedArmReviewAgent(
            llm=llm,
            require_llm=self.require_llm,
        )
        hotspot_spec = getattr(cfg, "hotspot_selection", None)
        self.hotspot_selection_agent = None
        self._llm_selected_hotspots: List[str] = []
        self._latest_hotspot_selection: Dict[str, Any] = {}
        self._hotspot_residue_table = None
        if hotspot_spec is not None and bool(getattr(hotspot_spec, "enabled", False)):
            self.hotspot_selection_agent = HotspotSelectionAgent(
                llm=llm,
                spec=hotspot_spec,
                require_llm=bool(getattr(hotspot_spec, "require_llm", True) or self.require_llm),
            )
        memory_cfg = getattr(cfg, "memory", None)
        self._memory_cfg = memory_cfg
        wants_index = bool(memory_cfg and memory_cfg.wants_index_items())
        wants_retrieval = bool(memory_cfg and memory_cfg.wants_retrieval())
        wants_compression = bool(memory_cfg and memory_cfg.wants_compression())
        wants_prompt_budget = bool(memory_cfg and memory_cfg.wants_prompt_budget())
        configured_prompt_budget = (
            int(getattr(memory_cfg, "prompt_max_bytes", 0) or 0) if wants_prompt_budget else 0
        )
        if llm and configured_prompt_budget > 0:
            endpoint_key = llm.settings.default_model
            endpoint = llm.settings.endpoints.get(endpoint_key)
            if endpoint is not None:
                endpoint.max_prompt_bytes = min(
                    configured_prompt_budget,
                    int(endpoint.max_prompt_bytes or configured_prompt_budget),
                )
        retrieval_llm = (
            llm if wants_retrieval and memory_cfg and memory_cfg.wants_semantic_rerank() else None
        )
        compression_llm = llm if wants_compression else None
        self.memory_index_enabled = wants_index
        self.memory_retrieval_agent = MemoryRetrievalAgent(
            llm=retrieval_llm,
            candidate_limit=int(getattr(memory_cfg, "retrieval_candidate_limit", 24) or 24),
            top_k=int(getattr(memory_cfg, "retrieval_top_k", 8) or 8),
            mmr_lambda=float(getattr(memory_cfg, "mmr_lambda", 0.7) or 0.7),
        ) if wants_retrieval else None
        self.memory_compression_agent = MemoryCompressionAgent(
            llm=compression_llm,
            max_active_items=int(getattr(memory_cfg, "max_active_items", 24) or 24),
            batch_size=int(getattr(memory_cfg, "compression_batch_size", 6) or 6),
            max_summary_chars=int(getattr(memory_cfg, "max_summary_chars", 1200) or 1200),
        ) if wants_compression else None
        self.skill_registry = self._load_skill_registry()
        self.self_improvement_store = SelfImprovementSkillStore.prepare(
            enabled=bool(getattr(getattr(cfg, "self_improvement", None), "enabled", False)),
            source_path=getattr(getattr(cfg, "self_improvement", None), "skill_path", None),
            out_dir=self.out_dir,
            source_base=Path(__file__).resolve().parents[2],
        )
        self.self_improvement_document: Optional[Dict[str, Any]] = None
        if self.self_improvement_store is not None:
            self.self_improvement_document = self.self_improvement_store.bootstrap_from_skills(
                self.skill_registry.by_type("strategy")
            )
            self._record_self_improvement_manifest()
        self._latest_pressure_conflict: Dict[str, Any] = {}
        self._preferred_arm_id: Optional[str] = None
        self._active_memory = None
        self._target_analysis_cache: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, Any]] = {}
        self._last_plotted_round_count = -1
        self._iteration_metrics_cache = IterationMetricsRoundCache()
        self._original_target_include = list(cfg.target.include or [])
        self._original_target_binding_types = list(cfg.target.binding_types or [])
        self._original_structure_groups = cfg.target.structure_groups
        self._initial_epitope_crop_mode = str((cfg.search_space.boltzgen or {}).get("epitope_crop_mode", "disabled")).strip().lower()

    def _load_skill_registry(self) -> SkillRegistry:
        path_value = getattr(self.cfg.runtime, "skill_registry_path", None)
        if not path_value:
            return SkillRegistry.empty()
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / path
        try:
            return SkillRegistry.from_yaml(path)
        except FileNotFoundError:
            return SkillRegistry.empty()

    def _record_self_improvement_manifest(self) -> None:
        if self.self_improvement_store is None:
            return
        path = self.out_dir / "run_manifest.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        document = self.self_improvement_document or self.self_improvement_store.load()
        payload["self_improvement_skill"] = {
            **self.self_improvement_store.handle.to_dict(),
            "working_sha256": artifact_record(self.self_improvement_store.path).get("sha256"),
            "revision": int((document.get("identity") or {}).get("revision") or 0),
        }
        atomic_write_json(path, payload)
        self._artifact_digest_cache.invalidate(path)

    def _select_agent_skills(self, agent_name: str, context: Mapping[str, Any], skill_types: Iterable[str]) -> List[Dict[str, Any]]:
        activations = self.skill_registry.select(
            agent_name=agent_name,
            context=context,
            skill_types=skill_types,
        )
        learned_activation = self._learned_skill_activation(agent_name, context)
        if learned_activation is not None and "llm_reasoning" in set(skill_types):
            activations.append(learned_activation)
            activations.sort(
                key=lambda item: (-int(item.get("priority") or 0), str(item.get("id") or ""))
            )
        return self._filter_strategy_skills_by_evidence(activations, context)

    def _learned_skill_activation(
        self,
        agent_name: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        consumers = {
            "BinderQualityAnalysisAgent",
            "HypothesisAgent",
            "DiagnosticCoachAgent",
            "InputConfigurationAgent",
            "StrategyConflictResolutionAgent",
        }
        if agent_name not in consumers or not self.self_improvement_document:
            return None
        context = dict(context or {})
        evaluation = dict(context.get("evaluation") or context.get("evaluation_summary") or {})
        structural = dict(context.get("structural_analysis") or {})
        phenotypes = {
            str(key)
            for source in (
                evaluation.get("tag_counts") or {},
                structural.get("aggregate_tags") or {},
            )
            for key, value in dict(source).items()
            if value
        }
        if (evaluation.get("pressure_conflict") or context.get("pressure_conflict") or {}).get("active"):
            phenotypes.add("pressure_conflict")
        rules = active_prompt_rules(
            self.self_improvement_document,
            limit=int(getattr(self.cfg.self_improvement, "max_active_rules", 6) or 6),
            context_phenotypes=phenotypes,
        )
        if not rules:
            return None
        guidance = [
            "[%s] IF %s THEN %s"
            % (
                rule.get("rule_id"),
                rule.get("condition") or "the rule phenotype matches",
                rule.get("strategy") or "follow the recorded strategy",
            )
            for rule in rules
        ]
        return {
            "id": "run-local-self-improvement",
            "type": "llm_reasoning",
            "description": "Validated run-local self-improvement strategy rules.",
            "trigger_reason": "run_local_active_rules",
            "required_inputs": [],
            "guidance": guidance,
            "runtime_logic": {"role": "highest_advisory_priority"},
            "output_schema": {"citation_field": "learned_rule_ids"},
            "allowed_config_keys": [],
            "params": {},
            "expected_signals": {},
            "deterministic_controls": {},
            "risk": "Previously useful rules may become stale; cite evidence and surface conflicts.",
            "priority": 900,
            "conflict_group": "",
            "depends_on": [],
            "excludes": [],
            "origin": "run_local_self_improvement",
            "version": str((self.self_improvement_document.get("identity") or {}).get("revision", 0)),
            "missing_required_inputs": [],
            "learned_rules": rules,
        }

    def _filter_strategy_skills_by_evidence(self, skills: List[Dict[str, Any]], context: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if not skills:
            return []
        strategy_enabled = bool(getattr(self.cfg.active_learning, "enable_strategy_skills", False))
        exploitation_allowed = self._contrastive_exploitation_allowed(context)
        gated_ids = {"strategy-contrastive-positive-exploit", "strategy-template-strict-exploit"}
        filtered: List[Dict[str, Any]] = []
        for skill in skills:
            if str(skill.get("type")) != "strategy":
                filtered.append(skill)
                continue
            if not strategy_enabled:
                continue
            if str(skill.get("id")) in gated_ids and not exploitation_allowed:
                continue
            filtered.append(skill)
        return filtered

    def _contrastive_exploitation_allowed(self, context: Mapping[str, Any]) -> bool:
        if not bool(getattr(self.cfg.active_learning, "enable_exploitation_arms", False)):
            return False
        examples = dict(context.get("active_learning_examples") or {})
        current = dict(examples.get("current_round") or {})
        counts = dict(current.get("counts") or {})
        try:
            current_positive = int(counts.get("strict_positive", counts.get("positive", 0)) or 0)
        except (TypeError, ValueError):
            current_positive = 0
        min_positive = max(1, int(getattr(self.cfg.active_learning, "min_current_positives_for_exploit", 2) or 2))
        rollback = dict(context.get("rollback") or {})
        pressure_conflict = dict((context.get("evaluation") or {}).get("pressure_conflict") or {})
        if (
            bool(rollback.get("is_regression"))
            or str(rollback.get("action") or "") in self.RECOVERY_ACTIONS | {"stop"}
            or pressure_conflict.get("active")
        ):
            return False
        return current_positive >= min_positive

    def _filter_fragment_template_update_by_evidence(
        self,
        update: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        filtered = dict(update or {})
        if self._contrastive_exploitation_allowed(context):
            return filtered
        for key in (
            "binder_template",
            "binder_templates",
            "binder_template_proximity",
            "exploit_fragment_modules",
            "module_guided_exploitation",
            "template_conditioned_fraction",
        ):
            filtered.pop(key, None)
        return filtered

    def _skills_audit_payload(self, activations_by_agent: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        payload = {
            "registry": self.skill_registry.audit_summary(),
            "activations_by_agent": {
                agent: list(skills or [])
                for agent, skills in activations_by_agent.items()
            },
            "policy": {
                "llm_reasoning_skills": "Prompt/context guidance and output schemas only.",
                "strategy_skills": "Materialized by StrategyLevelActiveLearner into arms/jobs after config sanitization.",
                "deterministic_policy_skills": "Audit/guardrail guidance only; they do not override deterministic controllers.",
            },
        }
        if self.self_improvement_store is not None and self.self_improvement_document:
            selected_audit_rules = active_prompt_rules(
                self.self_improvement_document,
                limit=self.cfg.self_improvement.max_active_rules,
            )
            selected_audit_ids = {
                str(rule.get("rule_id"))
                for rule in selected_audit_rules
                if rule.get("rule_id")
            }
            rule_statuses = [
                {
                    "rule_id": str(rule_id),
                    "module": module,
                    "status": str(rule.get("status") or "candidate"),
                    "utility": float(rule.get("utility") or 0.0),
                    "support_count": int(rule.get("support_count") or 0),
                    "contradiction_count": int(rule.get("contradiction_count") or 0),
                    "inactive_reason": (
                        None
                        if str(rule_id) in selected_audit_ids
                        else "top_k_or_relevance_trimmed"
                        if rule.get("status") in {"seed_active", "active"}
                        else "candidate_not_promoted"
                        if rule.get("status") == "candidate"
                        else "soft_conflict_pending_resolution"
                        if rule.get("status") == "contested"
                        else "retired"
                    ),
                }
                for module, section in self.self_improvement_document.get("modules", {}).items()
                for rule_id, rule in (section.get("rules") or {}).items()
            ]
            payload["self_improvement"] = {
                "handle": self.self_improvement_store.handle.to_dict(),
                "revision": int((self.self_improvement_document.get("identity") or {}).get("revision") or 0),
                "active_rule_ids": [
                    str(rule.get("rule_id"))
                    for rule in selected_audit_rules
                    if rule.get("rule_id")
                ],
                "semantic_relation_count": len(self.self_improvement_document.get("semantic_relations") or {}),
                "semantic_relations": list(
                    (self.self_improvement_document.get("semantic_relations") or {}).values()
                ),
                "open_conflict_count": sum(
                    1
                    for item in (self.self_improvement_document.get("conflict_sets") or {}).values()
                    if str(item.get("status") or "open") == "open"
                ),
                "conflict_sets": dict(self.self_improvement_document.get("conflict_sets") or {}),
                "rule_statuses": rule_statuses,
            }
        return payload

    def run(self, execute_job: Optional[Callable[[DesignJob, int], Dict[str, Any]]] = None) -> Dict[str, Any]:
        memory = self.memory_store.load(target=asdict(self.cfg.target))
        self._active_memory = memory
        if self.memory_index_enabled:
            self._backfill_indexed_memory(memory)
            self.memory_store.save(memory)
        self._seed_rollback_history(memory)
        if self._hotspot_selection_enabled():
            self._prepare_llm_hotspots_for_run()
        initial_jobs = materialize_deterministic_job_identities(
            deduplicate_effective_jobs(self._initial_jobs()), round_id=0, output_root=str(self.out_dir),
        )
        initial_jobs = self._enforce_round_cap(initial_jobs, round_id=0)
        summary: Dict[str, Any] = {"out_dir": str(self.out_dir), "rounds": []}
        start_round, current_jobs, recovered_rounds = self._recover_completed_rounds(initial_jobs)
        if self._hotspot_selection_enabled():
            self._restore_llm_hotspots_after_recover(start_round)
        summary["rounds"].extend(recovered_rounds)
        if recovered_rounds:
            self._write_summary(summary)
        if start_round >= self.max_rounds:
            # Total-round ceiling already satisfied; keep existing artifacts and exit.
            summary["resumed_complete"] = True
            summary["completed_rounds"] = start_round
            summary["max_rounds"] = self.max_rounds
            self._write_summary(summary)
            return summary

        for round_id in range(start_round, self.max_rounds):
            round_dir = self.out_dir / f"round_{round_id:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = self._load_round_checkpoint(round_dir, round_id) or {}
            if checkpoint:
                raw_jobs = checkpoint.get("current_jobs")
                if isinstance(raw_jobs, list):
                    current_jobs = self._jobs_from_dicts(raw_jobs)
                if str(checkpoint.get("status") or "").lower() == "failed":
                    # Restarting a failed round: do not let the old status/error
                    # override later checkpoint writes via payload unpacking.
                    checkpoint.pop("error", None)
                    checkpoint.pop("status", None)
                    checkpoint.pop("module_retries", None)
            current_jobs = self._bind_execution_identities_if_needed(current_jobs, round_id=round_id)
            identity_state = self._validate_job_identities(current_jobs, allow_legacy=True)
            if identity_state["status"] == "legacy_identity_ambiguous":
                checkpoint["job_identity_compatibility"] = identity_state
            checkpoint.update({
                "round_id": round_id,
                "current_jobs": [asdict(job) for job in current_jobs],
            })
            if self.self_improvement_store is not None:
                checkpoint["self_improvement_skill"] = {
                    **self.self_improvement_store.handle.to_dict(),
                    "revision": int(
                        ((self.self_improvement_document or {}).get("identity") or {}).get("revision")
                        or 0
                    ),
                }
            checkpoint.setdefault("artifacts", [])
            ledger_snapshot = self.template_outcome_ledger.target_snapshot(self._target_identity_digest, round_id=round_id)
            checkpoint["template_outcome_ledger_input"] = {"path": str(self.template_outcome_ledger.path), "digest": ledger_snapshot["digest"], "snapshot": ledger_snapshot}
            checkpoint.setdefault("modules", {})
            if self._hotspot_selection_enabled():
                used = self._persist_round_hotspot_selection(round_dir, round_id, phase="used_this_round")
                checkpoint["llm_hotspot_selection"] = dict(used)
                checkpoint["llm_hotspot_selection_path"] = str(round_dir / "llm_hotspot_selection.json")
            stage = "round_started"
            try:
                extend_memory = bool(getattr(self.cfg.runtime, "extend_memory", False))
                self.memory_store.record_jobs(memory, round_id, self._logical_jobs_for_memory(current_jobs), extend_memory=extend_memory)
                memory_summary = self.memory_store.summarize_for_agent(memory, extend_memory=extend_memory)
                self.memory_store.save(memory)
                self.bus.publish(AgentMessage("Orchestrator", "all", "status", {"event": "round_started", "job_count": len(current_jobs)}, round_id=round_id))
                self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)

                stage = "jobs_executed"
                def execute_jobs_module() -> Dict[str, Any]:
                    records = self._run_jobs(current_jobs, round_id, execute_job, attempts_path=round_dir / "execution_attempts.json")
                    path = self._write_json(round_dir / "execution_records.json", records)
                    checkpoint.update({"execution_records_path": str(path)})
                    self._append_artifact(checkpoint, path)
                    self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)
                    return {"records": records, "path": str(path)}

                result_collection_attempts = max(1, self.max_rounds - round_id)
                execution_module = self._run_validated_module(
                    module_name=stage,
                    round_id=round_id,
                    round_dir=round_dir,
                    checkpoint=checkpoint,
                    action=execute_jobs_module,
                    validator=lambda result: self._validate_execution_module(result, current_jobs),
                    loader=lambda record: {"records": self._load_json_path(round_dir / "execution_records.json"), "path": str(round_dir / "execution_records.json")},
                    max_attempts=result_collection_attempts,
                )
                execution_records = execution_module["records"]
                checkpoint["semantic_failure_fingerprint_version"] = 1
                checkpoint["semantic_failure_fingerprints"] = [
                    {
                        "version": int(record.get("semantic_failure_fingerprint_version") or 1),
                        "job_id": str(record.get("job_id") or (record.get("job") or {}).get("job_id") or ""),
                        "fingerprint": str(record.get("semantic_failure_fingerprint") or ""),
                        "scope": str(record.get("semantic_failure_scope") or ""),
                    }
                    for record in execution_records
                    if str(record.get("semantic_failure_fingerprint") or "")
                ]
                pre_submit_summary = self._build_pre_submit_summary(round_id, current_jobs, execution_records)
                pre_submit_summary_path = round_dir / "pre_submit_summary.json"
                pre_submit_summary["artifact"]["path"] = str(pre_submit_summary_path)
                pre_submit_summary["artifact_path"] = str(pre_submit_summary_path)
                self._write_json(pre_submit_summary_path, pre_submit_summary)
                checkpoint["pre_submit_summary"] = pre_submit_summary
                checkpoint["pre_submit_summary_path"] = str(pre_submit_summary_path)
                self._append_artifact(checkpoint, pre_submit_summary_path)
                execution_state = self._classify_execution_state(current_jobs, execution_records)
                execution_state["pre_submit_summary"] = pre_submit_summary
                execution_state["pre_submit_summary_path"] = str(pre_submit_summary_path)
                execution_state_path = self._write_json(round_dir / "execution_state.json", execution_state)
                checkpoint["execution_state"] = execution_state
                self._append_artifact(checkpoint, execution_state_path)
                self.bus.publish(AgentMessage("Orchestrator", "all", "status", {"event": "execution_state_classified", **execution_state}, round_id=round_id, artifacts=[str(execution_state_path)]))

                stage = "results_ingested"
                def ingest_results_module() -> Dict[str, Any]:
                    started = time.perf_counter()
                    items = self._ingest_execution_outputs(current_jobs, execution_records)
                    self._io_telemetry["ingestion_seconds"] = round(time.perf_counter() - started, 6)
                    path = self._write_json(round_dir / "ingestions.json", items)
                    checkpoint.update({"ingestions_path": str(path)})
                    self._append_artifact(checkpoint, path)
                    self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)
                    return {"ingestions": items, "path": str(path)}

                ingestion_module = self._run_validated_module(
                    module_name=stage,
                    round_id=round_id,
                    round_dir=round_dir,
                    checkpoint=checkpoint,
                    action=ingest_results_module,
                    validator=lambda result: self._validate_ingestion_module(result, current_jobs),
                    loader=lambda record: {"ingestions": self._load_json_path(round_dir / "ingestions.json"), "path": str(round_dir / "ingestions.json")},
                    max_attempts=result_collection_attempts,
                    retry_predicate=self._ingestion_error_retryable,
                )
                ingestions = ingestion_module["ingestions"]
                execution_state["quality_candidate_count"] = sum(len(item.get("candidates") or []) for item in ingestions)
                execution_state["quality_complete"] = execution_state["quality_candidate_count"] > 0
                execution_state["state"] = "complete" if execution_state["complete"] else "quality-complete" if execution_state["quality_complete"] else "partial"
                self._write_json(round_dir / "execution_state.json", execution_state)

                stage = "evaluated"
                def evaluate_module() -> Dict[str, Any]:
                    rows = [row for item in ingestions for row in item.get("candidates", [])]
                    filtered_rows, candidate_filtering = self._analysis_candidates(rows)
                    analysis_rows = filtered_rows if filtered_rows else list(rows)
                    summary_eval = self.evaluator.evaluate_candidates(analysis_rows)
                    summary_eval.candidate_filtering = candidate_filtering
                    if candidate_filtering.get("filtering_applied") and not filtered_rows and rows:
                        summary_eval.observations.append(
                            "User additional_filters rejected every candidate; downstream analysis uses raw rejected candidates for failure diagnosis."
                        )
                    path = self.evaluator.write_summary(summary_eval, round_dir / "evaluation_summary.json")
                    # Build explicit arm-scoped records; the round aggregate never scans sibling arms.
                    arm_rows: Dict[str, List[Mapping[str, Any]]] = {}
                    for row in rows:
                        arm_rows.setdefault(str(row.get("arm_id") or row.get("exploration_arm") or "legacy"), []).append(row)
                    logical_by_arm = {str((job.params or {}).get("arm_id") or (job.params or {}).get("exploration_arm") or "legacy"): job for job in self._logical_jobs_for_memory(current_jobs)}
                    execution_by_arm: Dict[str, List[DesignJob]] = {}
                    for job in current_jobs:
                        execution_by_arm.setdefault(str((job.params or {}).get("arm_id") or (job.params or {}).get("exploration_arm") or "legacy"), []).append(job)
                    arm_records = []
                    arm_structure_chemistry: Dict[str, Any] = {}
                    # Parse each inventory-validated structure once per round. Arm
                    # and round views below aggregate these immutable summaries.
                    all_structure_files = self._collect_structure_files(ingestions)
                    round_structure_summary = self.structure_agent.analyze_trusted_structures(
                        all_structure_files, binder_chain=self._guess_binder_chain(),
                        target_chains=[self.cfg.target.chain_id], hotspots=self._effective_hotspots(),
                        primary_residues=list(self.cfg.target.hotspots or []),
                        expanded_residues=self._effective_auxiliary_hotspots(),
                        negative_residues=list((self.cfg.search_space.boltzgen or {}).get("negative_binding_residues") or []),
                        binder_length=self._binder_length_hint(), auto_detect_chains=True,
                    )
                    structure_summary_by_path = {
                        str(item.get("structure_file") or ""): item
                        for item in round_structure_summary.summaries
                    }
                    round_arms_root = round_dir / "arms"
                    round_arms_root.mkdir(parents=True, exist_ok=True)
                    for arm_id, logical_job in sorted(logical_by_arm.items(), key=lambda item: int((item[1].params or {}).get("arm_rank", 0))):
                        params = dict(logical_job.params or {}); arm_candidates = list(arm_rows.get(arm_id, []))
                        arm_rank = int(params.get("arm_rank", 0)); arm_digest = str(params.get("arm_digest") or params.get("effective_intervention_digest") or "unknown")
                        arm_dir_name = f"{arm_rank:02d}_{safe_path_component(arm_id, fallback='arm')}_{safe_path_component(arm_digest[:12], fallback='digest')}"
                        arm_dir = round_arms_root / arm_dir_name; analysis_dir = arm_dir / "analysis"
                        if not resolved_within(analysis_dir, round_arms_root): raise ValueError("unsafe_round_arm_analysis_path")
                        analysis_dir.mkdir(parents=True, exist_ok=True)
                        arm_root = str(params.get("arm_root") or "")
                        manifest = {"schema_version":1,"round_id":round_id,"arm_id":arm_id,"arm_rank":arm_rank,"arm_digest":arm_digest,"logical_branch_id":params.get("logical_branch_id"),"logical_job_id":params.get("logical_job_id"),"execution_arm_root":arm_root,"round_arm_root":str(arm_dir),"analysis_root":str(analysis_dir),"execution_job_ids":[job.job_id for job in execution_by_arm.get(arm_id,[])],"containment":{"analysis_under_round_arm":resolved_within(analysis_dir,arm_dir),"execution_jobs_under_arm_root":all(resolved_within(job.output_dir,arm_root) for job in execution_by_arm.get(arm_id,[])) if arm_root else False}}
                        if not all(manifest["containment"].values()): raise ValueError(f"arm containment failed:{arm_id}")
                        self._write_json(arm_dir / "arm_manifest.json", manifest)
                        arm_ingestions = [item for item in ingestions if str(item.get("arm_id") or item.get("exploration_arm") or "legacy") == arm_id]
                        self._write_json(analysis_dir / "ingestion.json", {"arm_id":arm_id,"candidate_count":len(arm_candidates),"runs":arm_ingestions,"candidates":arm_candidates})
                        arm_eval = self.evaluator.evaluate_candidates(arm_candidates)
                        self.evaluator.write_summary(arm_eval, analysis_dir / "evaluation_summary.json")
                        ranked = sorted([asdict(item) for item in arm_eval.top_candidates + arm_eval.failed_examples], key=lambda item: tuple(item.get("core_rank_key") or []), reverse=True)
                        self._write_json(analysis_dir / "final_ranked_designs" / "ranked_candidates.json", {"arm_id":arm_id,"ranking":"core_rank_key_descending","candidates":ranked,"structure_attribution":"explicit" if all(item.get("raw",{}).get("structure_file") for item in ranked) else "unresolved_no_structure_copy"})
                        arm_structure_files = [value for item in arm_ingestions for value in item.get("structure_files", [])]
                        arm_structure = self.structure_agent.aggregate_summaries(
                            structure_summary_by_path[value]
                            for value in arm_structure_files
                            if value in structure_summary_by_path
                        )
                        self.structure_agent.write_batch(arm_structure, analysis_dir / "structure_evaluation.json")
                        arm_structure_chemistry[arm_id] = {"total_structures": int(getattr(arm_structure, "total_structures", 0) or 0), "aggregate_tags": dict(getattr(arm_structure, "aggregate_tags", {}) or {}), "interface_data_quality": dict(getattr(arm_structure, "interface_data_quality", {}) or {}), "observations": list(getattr(arm_structure, "observations", []) or []), "biochemical_measured": False, "developability_measured": False}
                        self._write_json(analysis_dir / "fragment_templates.json", {"schema_version":1,"arm_id":arm_id,"arm_scoped":True,"source_structure_files":arm_structure_files,"templates":[],"status":"deferred_to_round_template_governance"})
                        requested=sum(int((job.params or {}).get("num_designs") or 0) for job in execution_by_arm.get(arm_id,[]))
                        failed_ids={str(record.get("job_id") or (record.get("job") or {}).get("job_id") or "") for record in execution_records if str(record.get("status") or "").lower() in self.FAILURE_STATUSES}
                        completed=sum(int((job.params or {}).get("num_designs") or 0) for job in execution_by_arm.get(arm_id,[]) if job.job_id not in failed_ids)
                        endpoint_rows=[item.metrics for item in arm_eval.top_candidates]
                        endpoint = {"strict_yield":arm_eval.success_count/max(1,arm_eval.total_candidates),"core_objective":max([float(item.get("core_objective",0)) for item in endpoint_rows] or [0]),"interface_confidence":max([float(item.get("interface_confidence",0)) for item in endpoint_rows] or [0]),"interface_pae":min([float(item.get("min_design_to_target_pae",1e9)) for item in endpoint_rows] or [1e9]),"refold_rmsd":min([float(item.get("designfolding_filter_rmsd",1e9)) for item in endpoint_rows] or [1e9])}
                        evidence={"evidence_id":f"R{round_id}:ARM:{arm_id}","arm_id":arm_id,"arm_rank":arm_rank,"status":"closed" if completed==requested and requested>0 else "incomplete","requested_budget":requested,"completed_budget":completed,"trials":arm_eval.total_candidates,"successes":arm_eval.success_count,"endpoints":endpoint,"positive_features":[item.candidate_id for item in arm_eval.top_candidates[:3]],"negative_features":sorted(arm_eval.tag_counts),"confounders":[] if completed==requested else ["incomplete_execution"],"evidence_ids":[f"R{round_id}:ARM:{arm_id}:EVALUATION"],"branch_id":params.get("logical_branch_id"),"config_digest":stable_hash(params.get("final_parameter_state") or params),"intervention_digest":params.get("effective_intervention_digest"),"is_baseline":arm_id=="baseline_hold","arm_manifest":str(arm_dir / "arm_manifest.json")}
                        self._write_json(analysis_dir / "binder_quality_analysis.json", evidence)
                        arm_records.append(evidence)
                    cards={"round_id":round_id,"arms":arm_records,"structure_chemistry":arm_structure_chemistry}; self._write_json(round_dir / "arm_evidence_cards.json", cards)
                    self._append_artifact(checkpoint, path)
                    self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)
                    return {"candidates": analysis_rows, "filtered_candidates": filtered_rows, "all_candidates": rows, "candidate_filtering": candidate_filtering, "evaluation": summary_eval, "path": str(path), "arm_evidence_cards": cards, "arm_structure_chemistry": arm_structure_chemistry, "round_structure_summary": round_structure_summary}

                def load_evaluation_module(record: Mapping[str, Any]) -> Dict[str, Any]:
                    rows = [row for item in ingestions for row in item.get("candidates", [])]
                    filtered_rows, candidate_filtering = self._analysis_candidates(rows)
                    analysis_rows = filtered_rows if filtered_rows else list(rows)
                    evaluation_payload = self._load_json_path(round_dir / "evaluation_summary.json")
                    cards = self._load_json_if_exists(round_dir / "arm_evidence_cards.json")
                    return {
                        "candidates": analysis_rows,
                        "filtered_candidates": filtered_rows,
                        "all_candidates": rows,
                        "candidate_filtering": candidate_filtering,
                        "evaluation": self._evaluation_from_dict(evaluation_payload),
                        "path": str(round_dir / "evaluation_summary.json"),
                        "arm_evidence_cards": cards,
                        "arm_structure_chemistry": dict((cards or {}).get("structure_chemistry") or {}),
                    }

                evaluation_module = self._run_validated_module(
                    module_name=stage,
                    round_id=round_id,
                    round_dir=round_dir,
                    checkpoint=checkpoint,
                    action=evaluate_module,
                    validator=self._validate_evaluation_module,
                    loader=load_evaluation_module,
                )
                candidates = evaluation_module["candidates"]
                filtered_candidates = evaluation_module.get("filtered_candidates", candidates)
                all_candidates = evaluation_module.get("all_candidates", candidates)
                candidate_filtering = evaluation_module.get("candidate_filtering") or {}
                population_metadata = {
                    "collection_mode": "round_aggregate",
                    "metrics_rows_observed": sum(int(item.get("metrics_rows_read") or 0) for item in ingestions),
                    "metrics_rows_over_limit": any(bool(item.get("metrics_rows_over_limit")) for item in ingestions),
                    "metrics_rows_truncated": False,
                    "structure_files_observed": sum(int(item.get("structure_file_count") or 0) for item in ingestions),
                    "structure_scope": "native_final_refold_only",
                    "evidence_selection_policy": "full_ingestion_then_skill_guided_compaction",
                }
                candidate_filtering["population_metadata"] = population_metadata
                if population_metadata["metrics_rows_over_limit"]:
                    self.bus.publish(AgentMessage(
                        "ResultIngestionAgent", "all", "warning",
                        {"event": "metrics_evidence_budget_exceeded", **population_metadata},
                        round_id=round_id,
                    ))
                recovered_ingestions = [item for item in ingestions if item.get("candidate_scope") == "unfiltered_zero_pass_recovery"]
                if recovered_ingestions:
                    candidate_filtering.update({
                        "candidate_scope": "unfiltered_zero_pass_recovery",
                        "quality_status": "no_filter_pass",
                        "unfiltered_evaluable_count": sum(int(item.get("unfiltered_metric_count") or 0) for item in recovered_ingestions),
                        "filtered_candidate_count": sum(int(item.get("selected_metric_count") or 0) for item in recovered_ingestions),
                    })
                evaluation = evaluation_module["evaluation"]
                evaluation.candidate_filtering = candidate_filtering
                self._annotate_candidate_count_semantics(evaluation, current_jobs)
                evaluation_path = self.evaluator.write_summary(evaluation, round_dir / "evaluation_summary.json")
                self._artifact_digest_cache.invalidate(evaluation_path)

                stage = "structures_analyzed"
                def analyze_structures_module() -> Dict[str, Any]:
                    started = time.perf_counter()
                    structure_files = self._collect_structure_files(ingestions)
                    structure_summary = evaluation_module.get("round_structure_summary")
                    if not isinstance(structure_summary, StructureBatchEvaluation):
                        structure_summary = self.structure_agent.analyze_trusted_structures(
                            structure_files, binder_chain=self._guess_binder_chain(), target_chains=[self.cfg.target.chain_id],
                            hotspots=self._effective_hotspots(), primary_residues=list(self.cfg.target.hotspots or []),
                            expanded_residues=self._effective_auxiliary_hotspots(),
                            negative_residues=list((self.cfg.search_space.boltzgen or {}).get("negative_binding_residues") or []),
                            binder_length=self._binder_length_hint(), auto_detect_chains=True,
                        )
                    structure_path = self.structure_agent.write_batch(structure_summary, round_dir / "structure_evaluation.json")
                    self._append_artifact(checkpoint, structure_path)
                    motif_path = self._write_json(round_dir / "template_motif_attribution.json", {
                        "schema_version": 1,
                        "record_type": "round_aggregate_template_evidence",
                        "status": "aggregate_only",
                        "candidate_attribution": False,
                    })
                    self._append_artifact(checkpoint, motif_path)
                    ledger_snapshot = self.template_outcome_ledger.target_snapshot(
                        self._target_identity_digest, round_id=round_id
                    )
                    snapshot_path = self._write_json(
                        round_dir / "template_outcome_ledger_snapshot.json", ledger_snapshot
                    )
                    self._append_artifact(checkpoint, snapshot_path)
                    # Map each design structure to its inter-chain (design-to-target)
                    # PAE so the fragment miner and length policy can use the local
                    # interaction-confidence signal instead of the global complex iPTM.
                    interchain_pae_by_structure: Dict[str, float] = {}
                    success_structure_files: List[str] = []
                    template_batch = self.fragment_template_agent.mine_templates(
                        structure_summary,
                        round_id=round_id,
                        prior_templates=list(memory.template_library or []),
                        target_chain=self.cfg.target.chain_id,
                        requested_hotspots=self._effective_hotspots(),
                        structure_groups=self.cfg.target.structure_groups,
                        crop_mode=str((self.cfg.search_space.boltzgen or {}).get("epitope_crop_mode", "disabled")),
                        success_structure_files=success_structure_files,
                        gate_metric=self._fragment_template_gate(),
                        interchain_pae_by_structure=interchain_pae_by_structure,
                        interchain_pae_max=self._fragment_interchain_pae_max(),
                        templates_enabled=self._fragment_templates_enabled(),
                        template_top_k=self._fragment_template_top_k(),
                        template_artifact_dir=self.out_dir / "template_artifacts",
                        min_template_quality=float((self.cfg.search_space.boltzgen or {}).get("fragment_template_min_quality", 0.70) or 0.70),
                        current_target_structure=self.cfg.target.structure_path,
                        min_alignment_coverage=float((self.cfg.search_space.boltzgen or {}).get("fragment_template_min_alignment_coverage", 0.75) or 0.75),
                        max_target_patch_rmsd=float((self.cfg.search_space.boltzgen or {}).get("fragment_template_max_target_patch_rmsd", 2.5) or 2.5),
                        require_pae=bool((self.cfg.search_space.boltzgen or {}).get("fragment_template_require_pae", True)),
                        max_fixed_fraction=float((self.cfg.search_space.boltzgen or {}).get("fragment_template_max_fixed_fraction", 0.5) or 0.5),
                        min_designable_residues=int((self.cfg.search_space.boltzgen or {}).get("fragment_template_min_designable_residues", 8) or 8),
                        within_proximity=float((self.cfg.search_space.boltzgen or {}).get("binder_template_proximity", 8.0) or 8.0),
                        outcome_ledger_snapshot=ledger_snapshot,
                    )
                    memory.template_library = list(template_batch.library)
                    templates_path = self.fragment_template_agent.write_templates(template_batch, round_dir / "fragment_templates.json")
                    self._append_artifact(checkpoint, templates_path)
                    # Structure-quality-driven binder length range selection for next round.
                    allowed_min, allowed_max, length_step = self._binder_length_bounds()
                    length_recommendation = self.length_policy_agent.recommend(
                        structure_summary,
                        current_lengths=list(self.cfg.search_space.binder_lengths or []),
                        allowed_min=allowed_min,
                        allowed_max=allowed_max,
                        step=length_step,
                        interchain_pae_by_structure=interchain_pae_by_structure,
                        enabled=self._auto_binder_length_enabled(),
                    )
                    length_path = self.length_policy_agent.write_recommendation(length_recommendation, round_dir / "binder_length_recommendation.json")
                    self._append_artifact(checkpoint, length_path)
                    self._io_telemetry["structure_seconds"] = round(time.perf_counter() - started, 6)
                    self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)
                    return {"structures": structure_files, "structural_analysis": structure_summary, "structure_path": str(structure_path), "fragment_templates": template_batch, "fragment_templates_path": str(templates_path), "length_recommendation": length_recommendation, "length_recommendation_path": str(length_path)}

                def load_structure_module(record: Mapping[str, Any]) -> Dict[str, Any]:
                    structure_path = round_dir / "structure_evaluation.json"
                    templates_path = round_dir / "fragment_templates.json"
                    length_path = round_dir / "binder_length_recommendation.json"
                    structure_files = self._collect_structure_files(ingestions)
                    return {
                        "structures": structure_files,
                        "structural_analysis": self._structure_batch_from_dict(self._load_json_path(structure_path)),
                        "structure_path": str(structure_path),
                        "fragment_templates": self._fragment_batch_from_dict(self._load_json_path(templates_path)),
                        "fragment_templates_path": str(templates_path),
                        "length_recommendation": self._length_recommendation_from_dict(self._load_json_path(length_path)),
                        "length_recommendation_path": str(length_path),
                    }

                structure_module = self._run_validated_module(
                    module_name=stage,
                    round_id=round_id,
                    round_dir=round_dir,
                    checkpoint=checkpoint,
                    action=analyze_structures_module,
                    validator=self._validate_structure_module,
                    loader=load_structure_module,
                )
                struct_eval = structure_module["structural_analysis"]
                fragment_templates = structure_module["fragment_templates"]
                length_recommendation = structure_module["length_recommendation"]
                memory.template_library = list(fragment_templates.library)

                stage = "agents_analyzed"
                current_config_snapshot = self._current_config_snapshot()
                best_round_iptm = self._best_candidate_iptm(candidates)
                median_round_iptm = self._topk_candidate_iptm_median(candidates, top_k=self.cfg.active_learning.top_k)
                round_core_score = round_core_objective(candidates, top_k=self.cfg.active_learning.top_k)
                # Historical-best ranking uses the complete evaluable population.
                # Task analysis filters remain diagnostic and cannot change its denominator.
                round_rank = round_rank_key(all_candidates, top_k=self.cfg.active_learning.top_k)
                round_core_stats = self._candidate_metric_stats(candidates)
                execution_failed, execution_failure_reason = self._detect_round_execution_failure(
                    total_candidates=len(all_candidates or []),
                    execution_records=execution_records,
                )
                design_quality_status = (
                    "design_quality_failure:no_filter_pass"
                    if candidate_filtering.get("quality_status") == "no_filter_pass"
                    else "evaluated"
                )
                failed_job_count = sum(1 for record in execution_records if str(record.get("status") or "").lower() in self.FAILURE_STATUSES)
                requested_budget = sum(int((job.params or {}).get("num_designs") or 0) for job in current_jobs)
                completed_budget = sum(
                    int((job.params or {}).get("num_designs") or 0)
                    for job in current_jobs
                    if str(next((r.get("status") for r in execution_records if str(r.get("job_id") or (r.get("job") or {}).get("job_id")) == job.job_id), "")).lower() not in self.FAILURE_STATUSES
                )
                candidate_filtering["design_quality_status"] = design_quality_status
                candidate_filtering["execution_degraded"] = bool(failed_job_count)
                candidate_filtering["successful_budget_fraction"] = (completed_budget / requested_budget) if requested_budget else 0.0
                execution_state["quality_complete"] = not execution_failed and bool(all_candidates)
                if execution_state["quality_complete"] and not execution_state["complete"]:
                    execution_state["state"] = "quality-complete"
                self._write_json(round_dir / "execution_state.json", execution_state)
                arm_signature = self._round_arm_signature(current_jobs) or "baseline"
                branch_id = f"round_{round_id}"
                config_digest = hashlib.sha256(json.dumps(current_config_snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
                is_baseline = round_id == 0 or arm_signature == "baseline"
                intervention_digest = "" if is_baseline else hashlib.sha256(
                    json.dumps({"arm": arm_signature, "config": config_digest}, sort_keys=True).encode("utf-8")
                ).hexdigest()[:16]
                round_outcome = RoundOutcome(
                    round_id=round_id,
                    reward=monitoring_scalar_from_round_rank(round_rank),
                    best_iptm=best_round_iptm,
                    median_iptm=median_round_iptm,
                    core_objective=round_core_score,
                    core_metric_stats=round_core_stats,
                    round_rank_key=list(round_rank),
                    success_count=evaluation.success_count,
                    arm_signature=arm_signature, branch_id=branch_id, config_digest=config_digest,
                    intervention_digest=intervention_digest, is_baseline=is_baseline,
                    strict_successes=int(round_rank[0] or 0), strict_trials=len(all_candidates or []),
                    raw_candidate_count=len(all_candidates or []),
                    analysis_candidate_count=len(candidates or []),
                    raw_strict_yield=(float(round_rank[0]) / len(all_candidates)) if all_candidates else 0.0,
                    conditional_strict_yield=(float(evaluation.success_count or 0) / len(candidates)) if candidates else 0.0,
                    execution_failed=execution_failed,
                    execution_failure_reason=execution_failure_reason,
                )
                if self._hotspot_selection_enabled():
                    self._annotate_round_hotspot_metrics(round_dir, round_id, evaluation, round_outcome)
                if execution_failed:
                    self.bus.publish(AgentMessage("RollbackController", "all", "status", {"event": "execution_failure_excluded_from_reward", "round_id": round_id, "reason": execution_failure_reason}, round_id=round_id))
                elif not execution_state["complete"]:
                    self.bus.publish(AgentMessage("Orchestrator", "all", "warning", {"event": "partial_execution_quality_evaluable", **execution_state}, round_id=round_id))
                rollback_decision = self.rollback.observe(round_outcome)
                self._record_round_metric(memory, round_outcome)
                if rollback_decision.blocked_arm_signature:
                    self.memory_store.record_blocked_combination(
                        memory, round_id=round_id,
                        arm_ids=[name for name in rollback_decision.blocked_arm_signature.split(";") if name],
                        reason=rollback_decision.action,
                        intervention_digest=rollback_decision.blocked_intervention_digest or "",
                    )
                # Commit the completed round outcome before any next-round planning.
                # Later stages enrich this record, but a planning failure cannot erase
                # completed execution/evaluation evidence or the historical-best state.
                early_record = self.memory_store.upsert_round(memory, round_id)
                early_record.evaluation = asdict(evaluation)
                early_record.reward = round_outcome.reward
                early_record.config_snapshot = current_config_snapshot
                early_record.rollback_decision = rollback_decision.to_dict()
                early_record.finished_at = time.time()
                self.memory_store.upsert_ledger_round(
                    memory, round_id=round_id, outcome=round_outcome.to_dict(),
                    policy_snapshot=current_config_snapshot, failed_arms=[],
                    candidate_denominators={
                        "strict_trials": round_outcome.strict_trials,
                        "raw_candidate_count": round_outcome.raw_candidate_count,
                        "analysis_candidate_count": round_outcome.analysis_candidate_count,
                    },
                )
                memory.experiment_ledger.best_config_retests = dict(self.rollback.best_config_retests)
                self.memory_store.save(memory)
                checkpoint["current_round_complete"] = True
                checkpoint["completed_round_outcome"] = round_outcome.to_dict()
                self._write_checkpoint(round_dir, round_id, "round_outcome_persisted", "running", checkpoint)
                quality_signals = self._quality_collaboration_signals(
                    memory=memory,
                    evaluation=asdict(evaluation),
                    candidates=all_candidates,
                    current_config=current_config_snapshot,
                    rollback=rollback_decision.to_dict(),
                )
                quality_mode_decision = QualityCollaborationController.decide(
                    memory,
                    round_outcome.to_dict(),
                    getattr(self.cfg, "quality_collaboration", None),
                    signals=quality_signals,
                )
                quality_mode_path = self._write_json(
                    round_dir / "quality_analysis_mode.json",
                    quality_mode_decision.to_dict(),
                )
                self._append_artifact(checkpoint, quality_mode_path)
                self.bus.publish(AgentMessage(
                    "QualityCollaborationController",
                    "all",
                    "status",
                    {
                        "event": "quality_analysis_mode_selected",
                        **quality_mode_decision.to_dict(),
                    },
                    round_id=round_id,
                ))
                rollback_path = self._write_json(round_dir / "rollback_decision.json", {"outcome": round_outcome.to_dict(), "decision": rollback_decision.to_dict()})
                self._append_artifact(checkpoint, rollback_path)
                self.bus.publish(AgentMessage("RollbackController", "all", "status", {"event": "rollback_decision", **rollback_decision.to_dict()}, round_id=round_id))
                hard_constraints = self._hard_constraints()
                evaluation_context = asdict(evaluation)
                metric_facts = build_metric_facts(evaluation_context, candidates=all_candidates)
                self._latest_pressure_conflict = self._build_pressure_conflict(memory, all_candidates, current_config_snapshot)
                metric_facts["population_metadata"] = dict(candidate_filtering.get("population_metadata") or {})
                evaluation_context["metric_facts"] = metric_facts
                evaluation_context["population_metadata"] = dict(candidate_filtering.get("population_metadata") or {})
                evaluation_context["core_metric_trends"] = self._core_metric_trends(memory, all_candidates)
                evaluation_context["core_metric_stats"] = self._candidate_metric_stats(all_candidates)
                if self._latest_pressure_conflict:
                    evaluation_context["pressure_conflict"] = self._latest_pressure_conflict
                evaluation_context["candidate_scope"] = "raw_candidates_for_failure_analysis" if (candidate_filtering.get("filtering_applied") and not filtered_candidates and all_candidates) else candidate_filtering.get("analysis_scope", "all_candidates")
                evaluation_context["top_by_score"] = evaluation_context.get("top_candidates", [])
                evaluation_context["top_by_core"] = top_candidates_by_core(all_candidates)
                evaluation_context["top_by_iptm"] = top_candidates_by_iptm(all_candidates)
                if candidate_filtering.get("filtering_applied"):
                    evaluation_context["filtered_top_by_iptm"] = top_candidates_by_iptm(filtered_candidates)
                active_learning_examples = build_active_learning_examples(
                    round_id=round_id,
                    current_candidates=all_candidates,
                    prior_rounds=prior_examples_from_memory(memory, before_round_id=round_id),
                    additional_filters=(self.cfg.search_space.boltzgen or {}).get("additional_filters", []),
                    prior_positive_decay_after_zero_rounds=int(getattr(self.cfg.active_learning, "prior_positive_decay_after_zero_rounds", 2) or 2),
                    near_miss_top_k=int(getattr(self.cfg.active_learning, "near_miss_top_k", 4) or 4),
                    near_miss_min_confidence=float(getattr(self.cfg.active_learning, "near_miss_min_confidence", 0.30) or 0.30),
                    near_miss_weight=float(getattr(self.cfg.active_learning, "near_miss_weight", 0.25) or 0.25),
                    reward=round_outcome.reward,
                    rollback=rollback_decision.to_dict(),
                )
                examples_path = self._write_json(round_dir / "active_learning_examples.json", active_learning_examples)
                self._append_artifact(checkpoint, examples_path)
                evaluation_context["active_learning_examples"] = active_learning_examples
                structural_payload = self._compact_structure_evidence(struct_eval)
                candidate_clusters = aggregate_candidate_phenotypes(
                    round_id=round_id,
                    evaluation=evaluation_context,
                    active_learning_examples=active_learning_examples,
                    structural_analysis=structural_payload,
                )
                cluster_path = self._write_json(round_dir / "candidate_clusters.json", candidate_clusters)
                self._append_artifact(checkpoint, cluster_path)
                cluster_cards = compact_cluster_cards(candidate_clusters)
                current_example_counts = (active_learning_examples.get("current_round") or {}).get("counts", {})
                prior_example_counts = (active_learning_examples.get("prior_rounds") or {}).get("counts", {})
                provisional_reference_count = int(current_example_counts.get("near_miss", 0) or 0) if int(current_example_counts.get("strict_positive", 0) or 0) == 0 else 0
                metric_facts["active_learning_examples"] = {
                    "current_strict_positive_count": current_example_counts.get("strict_positive", 0),
                    "current_near_miss_count": current_example_counts.get("near_miss", 0),
                    "current_other_negative_count": current_example_counts.get("other_negative", 0),
                    "unfiltered_evaluable_count": candidate_filtering.get("unfiltered_evaluable_count", 0),
                    "provisional_reference_count": provisional_reference_count,
                    "prior_strict_positive_count": prior_example_counts.get("strict_positive", 0),
                    "prior_near_miss_count": prior_example_counts.get("near_miss", 0),
                    "prior_other_negative_count": prior_example_counts.get("other_negative", 0),
                }
                memory_summary = self._retrieve_memory_summary(
                    memory,
                    fallback_summary=memory_summary,
                    evaluation=evaluation_context,
                    current_config=current_config_snapshot,
                    current_jobs=current_jobs,
                )
                self_improvement_skills: List[Dict[str, Any]] = []
                self_improvement_update: Optional[SelfImprovementUpdate] = None
                if self.self_improvement_store is not None:
                    self_improvement_evidence = self._build_self_improvement_evidence(
                        round_id=round_id,
                        memory=memory,
                        current_jobs=current_jobs,
                        current_config=current_config_snapshot,
                        evaluation=evaluation_context,
                        structural_analysis=asdict(struct_eval),
                        outcome=round_outcome.to_dict(),
                        rollback=rollback_decision.to_dict(),
                    )
                    self_improvement_skill_context = {
                        "round_id": round_id,
                        "experience": self_improvement_evidence,
                        "current_skill": self.self_improvement_document or {},
                    }
                    self_improvement_skills = self._select_agent_skills(
                        "SelfImprovementSkillAgent",
                        self_improvement_skill_context,
                        ["llm_reasoning", "deterministic_policy"],
                    )

                    def self_improvement_module() -> Dict[str, Any]:
                        evidence_path = self._write_json(
                            round_dir / "self_improvement_evidence.json",
                            self_improvement_evidence,
                        )
                        update = self.self_improvement_agent.propose_update(
                            round_id=round_id,
                            document=self.self_improvement_document or self.self_improvement_store.load(),
                            evidence=self_improvement_evidence,
                            governance_skills=self_improvement_skills,
                        )
                        document = self.self_improvement_store.apply_operations(update.operations)
                        document = apply_semantic_relations(document, update.semantic_relations)
                        spec = self.cfg.self_improvement
                        document = apply_lifecycle(
                            document,
                            promotion_min_support=spec.promotion_min_support,
                            retirement_contradictions=spec.retirement_contradictions,
                            max_rules=spec.max_rules,
                        )
                        document = settle_conflicts_from_operations(
                            document,
                            update.operations,
                        )
                        update.raw["final_document_digest"] = stable_hash(document)
                        update.raw["applied_operation_ids"] = [
                            str(value)
                            for value in (document.get("provenance") or {}).get(
                                "applied_operation_ids", []
                            )
                        ]
                        self.self_improvement_store.save(document)
                        self.self_improvement_document = document
                        self._record_self_improvement_manifest()
                        checkpoint["self_improvement_skill"] = {
                            **self.self_improvement_store.handle.to_dict(),
                            "revision": int((document.get("identity") or {}).get("revision") or 0),
                        }
                        update_path = self._write_json(
                            round_dir / "self_improvement_update.json",
                            update.to_dict(),
                        )
                        snapshot_path = round_dir / "self_improvement_skill_snapshot.yaml"
                        atomic_write_text(
                            snapshot_path,
                            yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
                        )
                        self._artifact_digest_cache.invalidate(snapshot_path)
                        for path in (evidence_path, update_path, snapshot_path):
                            self._append_artifact(checkpoint, path)
                        return {
                            "update": update,
                            "document": document,
                            "evidence_path": str(evidence_path),
                            "update_path": str(update_path),
                            "snapshot_path": str(snapshot_path),
                        }

                    self_improvement_module_result = self._run_validated_module(
                        module_name="self_improvement_updated",
                        round_id=round_id,
                        round_dir=round_dir,
                        checkpoint=checkpoint,
                        action=self_improvement_module,
                        validator=self._validate_self_improvement_module,
                        loader=lambda record: self._load_self_improvement_module(round_dir),
                    )
                    self_improvement_update = self_improvement_module_result["update"]
                    self.self_improvement_document = self_improvement_module_result["document"]

                learned_strategy_rules = (
                    active_prompt_rules(
                        self.self_improvement_document or {},
                        limit=self.cfg.self_improvement.max_active_rules,
                    )
                    if self.self_improvement_document
                    else []
                )
                context = {"round_id": round_id, "evaluation": evaluation_context, "metric_facts": metric_facts, "active_learning_examples": active_learning_examples, "candidate_clusters": cluster_cards, "structural_analysis": self._compact_structure_evidence(struct_eval), "fragment_templates": self._compact_fragment_template_evidence(fragment_templates), "memory": memory_summary, "target_analysis": self._target_analysis(), "current_config": current_config_snapshot, "constraints": hard_constraints, "execution_failure": {"failed": execution_failed, "reason": execution_failure_reason}, "messages": [m.to_dict() for m in self.bus.query(round_id=round_id)][-32:], "rollback": rollback_decision.to_dict(), "reward": round_outcome.to_dict(), "quality_analysis_mode": quality_mode_decision.to_dict(), "learned_strategy_rules": learned_strategy_rules}
                active_skills_by_agent: Dict[str, List[Dict[str, Any]]] = {
                    "SelfImprovementSkillAgent": self_improvement_skills,
                    "BinderDesignOrchestrator": self._select_agent_skills("BinderDesignOrchestrator", context, ["deterministic_policy"]),
                    "RollbackController": self._select_agent_skills("RollbackController", context, ["deterministic_policy"]),
                    "FragmentTemplateMiningAgent": self._select_agent_skills("FragmentTemplateMiningAgent", context, ["strategy", "deterministic_policy"]),
                    "BinderLengthPolicyAgent": self._select_agent_skills("BinderLengthPolicyAgent", context, ["deterministic_policy"]),
                    "BinderQualityAnalysisAgent": self._select_agent_skills("BinderQualityAnalysisAgent", context, ["llm_reasoning", "deterministic_policy"]),
                }
                frozen_context = self._frozen_round_reasoning_context(context)
                metrics_summary = self._build_metrics_summary(all_candidates)
                metrics_summary.update({k: v for k, v in metric_facts.items() if k not in metrics_summary})
                active_skills_by_agent["HypothesisAgent"] = self._select_agent_skills(
                    "HypothesisAgent", frozen_context, ["llm_reasoning", "strategy"],
                )
                active_skills_by_agent["DiagnosticCoachAgent"] = self._select_agent_skills(
                    "DiagnosticCoachAgent", frozen_context, ["llm_reasoning", "deterministic_policy"],
                )
                skills_path = self._write_json(round_dir / "active_skills.json", self._skills_audit_payload(active_skills_by_agent))
                self._append_artifact(checkpoint, skills_path)
                arm_evidence_cards = dict(evaluation_module.get("arm_evidence_cards") or {})
                arm_records = list(arm_evidence_cards.get("arms") or [])
                arm_structure_chemistry = dict(
                    evaluation_module.get("arm_structure_chemistry")
                    or arm_evidence_cards.get("structure_chemistry")
                    or {}
                )
                quality_mode = quality_mode_decision.to_dict()
                quality_context = dict(frozen_context)
                quality_context["active_skills"] = active_skills_by_agent.get("BinderQualityAnalysisAgent", [])
                specialist_batch = None
                if not execution_failed and quality_mode_decision.mode == "multi":
                    specialist_batch = self.quality_collaboration_agent.prepare_specialists(
                        round_id=round_id, context=quality_context, memory=memory, mode_decision=quality_mode,
                    )

                def compute_hypotheses() -> HypothesisSet:
                    if execution_failed:
                        return HypothesisSet(
                            hypotheses=[], llm_used=False,
                            raw={"source": "execution_failure_noop", "execution_failure_reason": execution_failure_reason},
                        )
                    hypothesis_context = dict(frozen_context)
                    hypothesis_context["active_skills"] = active_skills_by_agent.get("HypothesisAgent", [])
                    return self.hypothesis_agent.propose(hypothesis_context)

                def compute_diagnostic() -> DiagnosticReport:
                    if execution_failed:
                        return DiagnosticReport(
                            round_id=round_id, llm_used=False,
                            status_diagnosis=f"Execution failure ({execution_failure_reason}); design-quality diagnosis skipped.",
                            root_causes=[{"cause": "execution_or_infrastructure_failure", "evidence": [execution_failure_reason], "confidence": 0.9, "category": "execution"}],
                            metric_interpretation={"active_learning_signal": "none", "reason": "execution failure produced no reliable design-quality evidence"},
                            corrective_actions=[], monitoring_recommendations=[{"action": "fix_or_retry_infrastructure", "reason": execution_failure_reason}],
                            pipeline_health={"healthy": False, "failure_class": "execution_or_infrastructure", "execution_failure_reason": execution_failure_reason},
                            raw={"source": "execution_failure_noop"},
                        )
                    return self.diagnostic_coach.diagnose(
                        round_id=round_id, monitor_snapshot=self._build_monitor_snapshot(execution_records),
                        metrics_summary=metrics_summary, evaluation_summary=evaluation_context,
                        structural_analysis=asdict(struct_eval), job_history=memory_summary.get("recent_rounds", []),
                        config=current_config_snapshot,
                        active_skills=active_skills_by_agent.get("DiagnosticCoachAgent", []),
                        candidate_clusters=frozen_context.get("candidate_clusters"),
                    )

                def compute_arm_comparison():
                    return self.strategy_arm_ranking_agent.compare_completed_arms(
                        round_id=round_id, arm_evidence=arm_records,
                    )

                def compute_single_quality() -> BinderQualityAnalysis:
                    if execution_failed:
                        return BinderQualityAnalysis(
                            round_id=round_id, llm_used=False,
                            overall_assessment=f"Execution failure ({execution_failure_reason}); quality analysis skipped to avoid contaminating active learning.",
                            high_quality_modules=[], low_quality_modules=[], causal_factors=[], next_round_guidance=[],
                            raw={"source": "execution_failure_noop", "execution_failure_reason": execution_failure_reason, "quality_analysis_mode": quality_mode},
                        )
                    analysis = self.quality_agent.analyze(round_id=round_id, context=quality_context)
                    analysis.raw = {**dict(analysis.raw or {}), "quality_analysis_mode": quality_mode}
                    return analysis

                def compute_specialist(role: str):
                    return self.quality_collaboration_agent.run_specialist(specialist_batch, role)

                wave_holder: Dict[str, Any] = {}
                if self._round_llm_wave_artifacts_ready(round_dir):
                    self.bus.publish(AgentMessage(
                        "Orchestrator", "all", "status",
                        {"event": "llm_dependency_waves_resumed", "waves": ["A", "B"]},
                        round_id=round_id,
                    ))
                    wave_holder = self._load_llm_wave_holder(round_dir)
                    self._preferred_arm_id = dict(wave_holder.get("final_strategy_decision") or {}).get("selected_arm_id")
                else:
                    # Wave A: frozen round evidence only. Specialists, hypothesis,
                    # diagnostic, and arm comparison never consume each other's outputs.
                    wave_a_tasks: Dict[str, Callable[[], Any]] = {
                        "hypotheses": compute_hypotheses,
                        "diagnostic": compute_diagnostic,
                        "arm_comparison": compute_arm_comparison,
                    }
                    if execution_failed or quality_mode_decision.mode != "multi" or specialist_batch is None:
                        wave_a_tasks["quality_single"] = compute_single_quality
                    elif specialist_batch.fallback_analysis is not None:
                        wave_a_tasks["quality_single"] = lambda: specialist_batch.fallback_analysis
                    elif specialist_batch.roles:
                        for role in specialist_batch.roles:
                            wave_a_tasks[f"quality_{role}"] = (lambda captured=role: compute_specialist(captured))
                    else:
                        wave_a_tasks["quality_single"] = compute_single_quality
                    wave_a = self._run_llm_wave(wave_name="A", round_id=round_id, tasks=wave_a_tasks)
                    comparison = wave_a["arm_comparison"]
                    hypotheses = wave_a["hypotheses"]
                    diagnostic = wave_a["diagnostic"]
                    if "quality_single" in wave_a:
                        pending_quality = wave_a["quality_single"]
                    else:
                        specialist_results = {}
                        for role in specialist_batch.roles:
                            _role_name, normalized, telemetry = wave_a[f"quality_{role}"]
                            specialist_results[role] = (normalized, telemetry)
                        self.quality_collaboration_agent.absorb_specialist_results(specialist_batch, specialist_results)
                        pending_quality = None

                    def compute_quality_manager() -> BinderQualityAnalysis:
                        if pending_quality is not None:
                            return pending_quality
                        return self.quality_collaboration_agent.assemble_with_manager(specialist_batch)

                    def compute_arm_history():
                        return self.strategy_conflict_agent.resolve_arm_direction(
                            round_id=round_id, arm_comparison=comparison.to_dict(),
                            ledger_history=self.memory_store.ledger_prompt_view(memory),
                        )

                    # Wave B: manager reads specialist findings; history reads comparison.
                    wave_b = self._run_llm_wave(
                        wave_name="B", round_id=round_id,
                        tasks={"quality_manager": compute_quality_manager, "arm_history": compute_arm_history},
                    )
                    quality_analysis = wave_b["quality_manager"]
                    history_resolution = wave_b["arm_history"]
                    final_decision = self.quality_collaboration_agent.final_strategy_decision(
                        round_id=round_id, arm_comparison=comparison.to_dict(),
                        history_resolution=history_resolution.to_dict(),
                        history_evidence=self.memory_store.ledger_prompt_view(memory).get("recent_rounds", []),
                        measured_assessments={
                            "structure_interface_chemistry": {
                                "per_arm": arm_structure_chemistry,
                                "structure_measured": any(int(value.get("total_structures", 0)) > 0 for value in arm_structure_chemistry.values()),
                                "biochemical_measured": False, "developability_measured": False,
                            }
                        },
                    )
                    wave_holder.update({
                        "quality_analysis": quality_analysis, "hypotheses": hypotheses, "diagnostic": diagnostic,
                        "arm_comparison": comparison.to_dict(), "arm_history_resolution": history_resolution.to_dict(),
                        "final_strategy_decision": final_decision.to_dict(),
                    })
                    quality_path = self.quality_agent.write_analysis(quality_analysis, round_dir / "binder_quality_analysis.json")
                    hypotheses_wave_path = self._write_json(round_dir / "hypotheses.json", asdict(hypotheses))
                    diagnostic_wave_path = self.diagnostic_coach.write_report(diagnostic, round_dir / "diagnostic_report.json")
                    comparison_path = self._write_json(round_dir / "arm_comparison.json", comparison.to_dict())
                    history_path = self._write_json(round_dir / "arm_history_resolution.json", history_resolution.to_dict())
                    attempts_path = self._write_json(round_dir / "arm_history_llm_attempts.json", {
                        "round_id": round_id, "llm_used": history_resolution.llm_used,
                        "fallback_reason": dict(history_resolution.raw or {}).get("fallback_reason"),
                        "error": dict(history_resolution.raw or {}).get("llm_error"),
                        "attempts": dict(history_resolution.raw or {}).get("llm_attempts", []),
                    })
                    final_path = self._write_json(round_dir / "final_strategy_decision.json", final_decision.to_dict())
                    for path in (quality_path, hypotheses_wave_path, diagnostic_wave_path, comparison_path, history_path, attempts_path, final_path):
                        self._append_artifact(checkpoint, path)
                    self._preferred_arm_id = final_decision.selected_arm_id

                def quality_analysis_module() -> Dict[str, Any]:
                    analysis = wave_holder.get("quality_analysis")
                    if analysis is None:
                        analysis = self._quality_analysis_from_dict(
                            self._load_json_path(round_dir / "binder_quality_analysis.json")
                        )
                    path = self.quality_agent.write_analysis(analysis, round_dir / "binder_quality_analysis.json")
                    self._append_artifact(checkpoint, path)
                    return {"quality_analysis": analysis, "path": str(path)}

                quality_module = self._run_validated_module(
                    module_name="binder_quality_analyzed",
                    round_id=round_id,
                    round_dir=round_dir,
                    checkpoint=checkpoint,
                    action=quality_analysis_module,
                    validator=self._validate_quality_module,
                    loader=lambda record: {
                        "quality_analysis": self._quality_analysis_from_dict(self._load_json_path(round_dir / "binder_quality_analysis.json")),
                        "path": str(round_dir / "binder_quality_analysis.json"),
                    },
                )
                quality_analysis = quality_module["quality_analysis"]
                self._observe_llm_fallback(round_dir, round_id, "BinderQualityAnalysisAgent", quality_analysis, checkpoint)
                context["quality_analysis"] = asdict(quality_analysis)

                if wave_holder:
                    hypotheses = wave_holder["hypotheses"]
                    diagnostic = wave_holder["diagnostic"]
                else:
                    hypotheses = self._hypotheses_from_dict(self._load_json_path(round_dir / "hypotheses.json"))
                    diagnostic = self._diagnostic_from_dict(self._load_json_path(round_dir / "diagnostic_report.json"))

                hypotheses_path = self._write_json(round_dir / "hypotheses.json", asdict(hypotheses))
                diagnostic_path = self.diagnostic_coach.write_report(diagnostic, round_dir / "diagnostic_report.json")
                self._append_artifact(checkpoint, hypotheses_path)
                self._append_artifact(checkpoint, diagnostic_path)
                checkpoint.setdefault("modules", {})["hypotheses_proposed"] = {
                    "status": "completed", "path": str(hypotheses_path),
                }
                checkpoint.setdefault("modules", {})["diagnosed"] = {
                    "status": "completed", "path": str(diagnostic_path),
                }
                self._observe_llm_fallback(round_dir, round_id, "HypothesisAgent", hypotheses, checkpoint)
                self._observe_llm_fallback(round_dir, round_id, "DiagnosticCoachAgent", diagnostic, checkpoint)
                context["hypotheses"] = hypotheses.hypotheses
                stage = "diagnosed"
                self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)
                self.bus.publish(AgentMessage("DiagnosticCoach", "all", "diagnostic", {"status_diagnosis": diagnostic.status_diagnosis, "pipeline_health": diagnostic.pipeline_health, "corrective_actions_count": len(diagnostic.corrective_actions)}, round_id=round_id))
                self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)

                # Wave C: InputConfiguration and later policy/strategy consume Wave A+B
                # conclusions. They remain sequential because each mutates the next-round plan.
                stage = "next_round_configured"
                input_config_skill_context = dict(context)
                input_config_skill_context.update({
                    "diagnostic_report": asdict(diagnostic),
                    "hypotheses": hypotheses.hypotheses,
                    "quality_analysis": asdict(quality_analysis),
                    "target_profile": self._target_profile_context(),
                })
                active_skills_by_agent["InputConfigurationAgent"] = self._select_agent_skills("InputConfigurationAgent", input_config_skill_context, ["llm_reasoning", "strategy", "deterministic_policy"])
                def input_configuration_module() -> Dict[str, Any]:
                    if execution_failed:
                        config = InputConfiguration(
                            target_name=self.cfg.task_name,
                            llm_used=False,
                            reasoning=f"Execution failure ({execution_failure_reason}); no design-parameter changes are proposed.",
                            recommended_config={},
                            parameter_rationale=[],
                            risk_assessment=[{"risk": "execution_failure", "likelihood": "high", "mitigation": "Retry/fix infrastructure before changing design parameters."}],
                            iteration_strategy={},
                            raw={"source": "execution_failure_noop", "execution_failure_reason": execution_failure_reason},
                        )
                    else:
                        tuning_feedback = self._build_tuning_feedback(memory, current_config_snapshot)
                        config = self.input_config_agent.configure_next_round(
                            target_name=self.cfg.task_name,
                            current_config=current_config_snapshot,
                            diagnostic_report=asdict(diagnostic),
                            evaluation_summary=evaluation_context,
                            structural_analysis=asdict(struct_eval),
                            quality_analysis=asdict(quality_analysis),
                            hypotheses=hypotheses.hypotheses,
                            memory_summary=memory_summary,
                            constraints=hard_constraints,
                            round_id=round_id + 1,
                            tuning_feedback=tuning_feedback,
                            target_profile=self._target_profile_context(),
                            active_skills=active_skills_by_agent.get("InputConfigurationAgent", []),
                        )
                    path = self.input_config_agent.write_config(config, round_dir / "next_round_input_configuration.json")
                    self._append_artifact(checkpoint, path)
                    return {"input_config": config, "path": str(path)}

                input_config_module = self._run_validated_module(
                    module_name=stage,
                    round_id=round_id,
                    round_dir=round_dir,
                    checkpoint=checkpoint,
                    action=input_configuration_module,
                    validator=self._validate_input_config_module,
                    loader=lambda record: {
                        "input_config": self._input_config_from_dict(self._load_json_path(round_dir / "next_round_input_configuration.json")),
                        "path": str(round_dir / "next_round_input_configuration.json"),
                    },
                )
                input_config = input_config_module["input_config"]
                self._observe_llm_fallback(round_dir, round_id, "InputConfigurationAgent", input_config, checkpoint)
                self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)

                stage = "policy_updated"
                policy_skill_context = dict(input_config_skill_context)
                policy_skill_context["input_configuration"] = asdict(input_config)
                active_skills_by_agent["ActiveLearningPolicyAgent"] = self._select_agent_skills("ActiveLearningPolicyAgent", policy_skill_context, ["strategy", "deterministic_policy"])
                def policy_update_module() -> Dict[str, Any]:
                    recovery_requested = rollback_decision.action in self.RECOVERY_ACTIONS
                    conflict_resolution = StrategyConflictResolution(
                        round_id=round_id + 1,
                        llm_used=False,
                        summary="Conflict resolution not required.",
                    )
                    strategy_conflicts: List[Dict[str, Any]] = []
                    if recovery_requested:
                        # Quality analysis remains useful audit evidence, but no
                        # degraded-round recommendation may mutate the live config.
                        policy_proposal = NextRoundParameterProposal(
                            round_id=round_id + 1,
                            params_update={},
                            rationale=[
                                f"Quality recovery requested; suppressing round {round_id} policy inputs. "
                                f"Next jobs will restore and exactly replay best round {rollback_decision.best_round}."
                            ],
                            analysis_metadata={
                                "rollback_policy_suppressed": True,
                                "rollback_action": rollback_decision.action,
                                "rollback_replay_source_round": rollback_decision.best_round,
                                "current_round_inputs": "audit_only_suppressed",
                            },
                        )
                        merge_report = self._rollback_suppressed_merge_report(
                            rollback_decision=rollback_decision,
                            input_config=input_config.recommended_config,
                            binder_length_update=length_recommendation.recommended_config,
                            fragment_template_update=fragment_templates.recommended_config,
                        )
                    elif execution_failed:
                        policy_proposal = NextRoundParameterProposal(
                            round_id=round_id + 1,
                            params_update={},
                            rationale=[f"Execution failure ({execution_failure_reason}); suppressing design-parameter updates to avoid contaminating active learning."],
                            analysis_metadata={"execution_failure_reason": execution_failure_reason},
                        )
                    else:
                        policy_proposal = self.policy_agent.propose_next_params(
                            evaluation,
                            self._base_params(),
                            round_id=round_id + 1,
                            model=self._search_profile().model,
                            structural_summary=struct_eval,
                            hypotheses=hypotheses.hypotheses,
                            quality_analysis=asdict(quality_analysis),
                            diagnostic_report=asdict(diagnostic),
                            memory_summary=memory_summary,
                            max_binders_per_round=binder_generation_cap(self.cfg),
                            active_skills=active_skills_by_agent.get("ActiveLearningPolicyAgent", []),
                        )
                    if not recovery_requested:
                        fragment_template_update = self._filter_fragment_template_update_by_evidence(
                            fragment_templates.recommended_config,
                            policy_skill_context,
                        )
                        self._pending_policy_proposal = policy_proposal
                        policy_proposal.params_update, merge_report = self._merge_next_round_updates(
                            ("input_configuration", input_config.recommended_config),
                            ("binder_length_policy", length_recommendation.recommended_config),
                            ("policy_proposal", policy_proposal.params_update),
                            ("fragment_template_mining", fragment_template_update),
                            apply=False,
                        )
                        if (
                            not execution_failed
                            and self.self_improvement_store is not None
                            and bool(getattr(self.cfg.self_improvement, "conflict_resolution_enabled", True))
                        ):
                            tuning_feedback = self._build_tuning_feedback(memory, current_config_snapshot)
                            strategy_conflicts = detect_strategy_conflicts(
                                merge_report=merge_report,
                                proposed_update=policy_proposal.params_update,
                                tuning_feedback=tuning_feedback,
                                pressure_conflict=self._latest_pressure_conflict,
                                learned_document=self.self_improvement_document,
                            )
                            contested_rule_ids = sorted({
                                str(rule_id)
                                for conflict in strategy_conflicts
                                for rule_id in conflict.get("rule_ids", []) or []
                                if str(rule_id)
                            })
                            if contested_rule_ids and self.self_improvement_document:
                                self.self_improvement_document = mark_rules_contested(
                                    self.self_improvement_document,
                                    contested_rule_ids,
                                    reason="soft_parameter_family_conflict",
                                )
                                self.self_improvement_store.save(self.self_improvement_document)
                                self._record_self_improvement_manifest()
                                post_conflict_snapshot = round_dir / "self_improvement_skill_post_conflict_snapshot.yaml"
                                atomic_write_text(
                                    post_conflict_snapshot,
                                    yaml.safe_dump(
                                        self.self_improvement_document,
                                        allow_unicode=True,
                                        sort_keys=False,
                                    ),
                                )
                                self._artifact_digest_cache.invalidate(post_conflict_snapshot)
                                self._append_artifact(checkpoint, post_conflict_snapshot)
                            conflict_skill_context = {
                                "conflicts": strategy_conflicts,
                                "current_config": current_config_snapshot,
                                "recent_rounds": memory_summary.get("recent_rounds", []),
                                "evaluation": evaluation_context,
                                "structure_phenotype": self_improvement_evidence.get("structure_phenotype", {}),
                                "quality_analysis": asdict(quality_analysis),
                                "tuning_feedback": tuning_feedback,
                                "historical_best": {
                                    "round_id": tuning_feedback.get("best_round_id"),
                                    "reward": tuning_feedback.get("best_reward"),
                                    "config": tuning_feedback.get("best_round_config"),
                                },
                                "pressure_conflict": dict(self._latest_pressure_conflict or {}),
                                "hard_constraints": hard_constraints,
                                "max_binders_per_round": binder_generation_cap(self.cfg),
                            }
                            active_skills_by_agent["StrategyConflictResolutionAgent"] = self._select_agent_skills(
                                "StrategyConflictResolutionAgent",
                                conflict_skill_context,
                                ["llm_reasoning", "deterministic_policy"],
                            )
                            if strategy_conflicts:
                                conflict_resolution = self.strategy_conflict_agent.resolve(
                                    round_id=round_id + 1,
                                    conflicts=strategy_conflicts,
                                    context=conflict_skill_context,
                                    active_skills=active_skills_by_agent.get("StrategyConflictResolutionAgent", []),
                                )
                                if self.self_improvement_document:
                                    self.self_improvement_document = record_conflict_decisions(
                                        self.self_improvement_document,
                                        conflict_resolution.decisions,
                                    )
                                    self.self_improvement_store.save(self.self_improvement_document)
                                    self._record_self_improvement_manifest()
                                    checkpoint["self_improvement_skill"] = {
                                        **self.self_improvement_store.handle.to_dict(),
                                        "revision": int(
                                            (self.self_improvement_document.get("identity") or {}).get("revision")
                                            or 0
                                        ),
                                    }
                                    post_conflict_snapshot = round_dir / "self_improvement_skill_post_conflict_snapshot.yaml"
                                    atomic_write_text(
                                        post_conflict_snapshot,
                                        yaml.safe_dump(
                                            self.self_improvement_document,
                                            allow_unicode=True,
                                            sort_keys=False,
                                        ),
                                    )
                                    self._artifact_digest_cache.invalidate(post_conflict_snapshot)
                                    self._append_artifact(checkpoint, post_conflict_snapshot)
                                policy_proposal.analysis_metadata["strategy_conflict_resolution"] = {
                                    "conflict_count": len(strategy_conflicts),
                                    "llm_used": conflict_resolution.llm_used,
                                    "decisions": conflict_resolution.decisions,
                                    }
                        sampler_resolution = self._resolve_probabilistic_sampler(input_config)
                        sampler_artifacts = {}
                        for artifact_name in ("source_proposals", "proposed_state", "guardrail_mapping", "final_executable_state"):
                            artifact_path = self._write_json(round_dir / (artifact_name + ".json"), sampler_resolution[artifact_name])
                            self._append_artifact(checkpoint, artifact_path)
                            sampler_artifacts[artifact_name + "_path"] = str(artifact_path)
                        checkpoint.update({
                            "final_parameter_state_path": sampler_artifacts["final_executable_state_path"],
                            "final_parameter_state_digest": stable_hash(sampler_resolution["final_executable_state"]),
                            "parameter_catalog_digest": sampler_resolution["catalog_digest"],
                        })
                        policy_proposal.analysis_metadata["probabilistic_sampler"] = {**sampler_resolution, **sampler_artifacts}
                        policy_proposal.final_params_update = dict(sampler_resolution["final_executable_state"])
                        final_updates: List[Tuple[str, Mapping[str, Any]]] = [
                            ("input_configuration", input_config.recommended_config),
                            ("binder_length_policy", length_recommendation.recommended_config),
                            ("policy_proposal", policy_proposal.params_update),
                            ("fragment_template_mining", fragment_template_update),
                            ("probabilistic_sampler_final", sampler_resolution["final_executable_state"]),
                        ]
                        if conflict_resolution.params_update:
                            final_updates.append(
                                ("strategy_conflict_resolution", conflict_resolution.params_update)
                            )
                        policy_proposal.params_update, merge_report = self._merge_next_round_updates(
                            *final_updates,
                            apply=True,
                        )
                        policy_proposal.applied_params_update = dict(policy_proposal.params_update)
                    conflicts_path = self._write_json(
                        round_dir / "strategy_conflicts.json",
                        strategy_conflicts,
                    )
                    resolution_path = self._write_json(
                        round_dir / "strategy_conflict_resolution.json",
                        conflict_resolution.to_dict(),
                    )
                    self._append_artifact(checkpoint, conflicts_path)
                    self._append_artifact(checkpoint, resolution_path)
                    policy_path = self.policy_agent.write_proposal(policy_proposal, round_dir / "next_round_parameter_proposal.json")
                    self._append_artifact(checkpoint, policy_path)
                    report_path = self._write_json(round_dir / "next_round_config_merge_report.json", merge_report)
                    self._append_artifact(checkpoint, report_path)
                    config_path = round_dir / "next_round_config.yaml"
                    self._write_next_round_config(config_path, policy_proposal.params_update)
                    self._append_artifact(checkpoint, config_path)
                    return {"proposal": policy_proposal, "config_merge_report": merge_report, "conflict_resolution": conflict_resolution, "proposal_path": str(policy_path), "merge_report_path": str(report_path), "conflicts_path": str(conflicts_path), "conflict_resolution_path": str(resolution_path), "config_path": str(config_path)}

                policy_module = self._run_validated_module(
                    module_name=stage,
                    round_id=round_id,
                    round_dir=round_dir,
                    checkpoint=checkpoint,
                    action=policy_update_module,
                    validator=self._validate_policy_module,
                    loader=lambda record: {
                        "proposal": self._policy_proposal_from_dict(self._load_json_path(round_dir / "next_round_parameter_proposal.json")),
                        "config_merge_report": self._load_json_path(round_dir / "next_round_config_merge_report.json"),
                        "conflict_resolution": self._conflict_resolution_from_dict(self._load_json_path(round_dir / "strategy_conflict_resolution.json")),
                        "proposal_path": str(round_dir / "next_round_parameter_proposal.json"),
                        "merge_report_path": str(round_dir / "next_round_config_merge_report.json"),
                        "conflicts_path": str(round_dir / "strategy_conflicts.json"),
                        "conflict_resolution_path": str(round_dir / "strategy_conflict_resolution.json"),
                        "config_path": str(round_dir / "next_round_config.yaml"),
                    },
                )
                proposal = policy_module["proposal"]
                config_merge_report = policy_module["config_merge_report"]
                conflict_resolution = policy_module["conflict_resolution"]
                merge_report_path = Path(policy_module["merge_report_path"])
                if self._hotspot_selection_enabled() and rollback_decision.action != "stop":
                    self._refine_llm_hotspots(
                        round_id=round_id,
                        round_dir=round_dir,
                        evaluation=evaluation,
                        structural=struct_eval,
                        round_outcome=round_outcome,
                    )

                stage = "next_jobs_proposed"
                strategy_skill_context = dict(policy_skill_context)
                strategy_skill_context["policy_proposal"] = asdict(proposal)
                strategy_skill_context["config_merge_report"] = config_merge_report
                active_skills_by_agent["StrategyLevelActiveLearner"] = self._select_agent_skills("StrategyLevelActiveLearner", strategy_skill_context, ["strategy", "deterministic_policy"])
                def next_jobs_module() -> Dict[str, Any]:
                    proposed_jobs: List[DesignJob] = []
                    applied_update: Dict[str, Any] = dict(proposal.params_update)
                    replay_source_job_ids: List[str] = []
                    replay_snapshot: Dict[str, Any] = {}
                    # Always prepare continuation seeds unless early-stop. max_rounds
                    # only gates whether those jobs are executed in this process.
                    prepare_continuation = rollback_decision.action != "stop"
                    if prepare_continuation and failed_job_count:
                        applied_update = {}
                        proposed_jobs = self._retry_jobs_after_execution_failure(current_jobs, execution_records, next_round_id=round_id + 1)
                        proposed_jobs = self._enforce_binder_length_range(proposed_jobs)
                        # Retry/retest continuations are logical strategy jobs at this point.
                        # Bind their preserved budgets, execution partitions, slots, output
                        # roots and immutable execution identity before the next round can
                        # submit them or create an attempt-ledger entry.
                        proposed_jobs = self._enforce_round_cap(proposed_jobs, round_id=round_id + 1)
                        self._write_next_round_config(round_dir / "next_round_config.yaml", applied_update)
                    elif prepare_continuation and rollback_decision.action in self.RECOVERY_ACTIONS:
                        proposed_jobs, applied_update, replay_snapshot, replay_source_job_ids = (
                            self._prepare_exact_rollback_replay(
                                memory, rollback_decision, next_round_id=round_id + 1,
                            )
                        )
                        if rollback_decision.action == "branch_from_best":
                            # Restore the best baseline but create a fresh strategy branch;
                            # do not consume the next round on another identical replay.
                            parent_jobs = self._logical_jobs_for_memory(proposed_jobs)
                            blocked_signature = tuple(sorted(name for name in (rollback_decision.blocked_arm_signature or "").split(";") if name))
                            proposed_jobs = self.learner.propose_next(
                                round_id + 1, parent_jobs, [], str(self.out_dir),
                                top_k=self.cfg.active_learning.top_k, policy_update={},
                                blocked_arm_combinations=[blocked_signature] if blocked_signature else [], branch_width=self.cfg.active_learning.branch_width,
                                enable_exploitation_arms=bool(getattr(self.cfg.active_learning, "enable_exploitation_arms", False)),
                                structural_summary=struct_eval, hypotheses=hypotheses.hypotheses,
                                quality_analysis=asdict(quality_analysis), pressure_conflict=self._latest_pressure_conflict,
                                active_skills=active_skills_by_agent.get("StrategyLevelActiveLearner", []),
                                selection_context=self._strategy_selection_context(
                                    evaluation=evaluation, active_learning_examples=active_learning_examples,
                                    round_outcome=round_outcome, rollback_decision=rollback_decision,
                                    fragment_templates=fragment_templates, round_id=round_id,
                                ),
                            ).jobs
                            proposed_jobs = self._enforce_binder_length_range(proposed_jobs)
                            proposed_jobs = self._finalize_semantic_job_identities(
                                proposed_jobs, round_id=round_id + 1,
                            )
                            proposed_jobs = self._enforce_round_cap(proposed_jobs, round_id=round_id + 1)
                        else:
                            # Exact retest/replay jobs are still logical after
                            # `_prepare_exact_rollback_replay`. Bind execution
                            # slots, job output directories and finalized
                            # identity before the next round can submit them.
                            proposed_jobs = self._bind_execution_identities_if_needed(
                                proposed_jobs, round_id=round_id + 1,
                            )
                        self._write_next_round_config(round_dir / "next_round_config.yaml", applied_update)
                    elif prepare_continuation:
                        parent_jobs = self._logical_jobs_for_memory(current_jobs)
                        example_current = dict((active_learning_examples or {}).get("current_round") or {})
                        example_counts = dict(example_current.get("counts") or {})
                        selection_context = self._strategy_selection_context(
                            evaluation=evaluation, active_learning_examples=active_learning_examples,
                            round_outcome=round_outcome, rollback_decision=rollback_decision,
                            fragment_templates=fragment_templates, round_id=round_id,
                        )
                        strategy_skills = active_skills_by_agent.get("StrategyLevelActiveLearner", [])
                        exploitation_enabled = bool(getattr(self.cfg.active_learning, "enable_exploitation_arms", False))
                        candidate_arms = self.learner.candidate_arms(
                            structural_summary=struct_eval,
                            hypotheses=hypotheses.hypotheses,
                            quality_analysis=asdict(quality_analysis),
                            pressure_conflict=self._latest_pressure_conflict,
                            active_skills=strategy_skills,
                            enable_exploitation_arms=exploitation_enabled,
                            selection_context=selection_context,
                        )
                        ranking_prefilter = []
                        if self._epitope_crop_disabled_hard_constraint():
                            retained = []
                            for arm in candidate_arms:
                                if str(arm.get("name") or "") == "target_context_focus":
                                    ranking_prefilter.append({"arm_id": "target_context_focus", "applicability": ArmApplicability.NOT_APPLICABLE.value, "reason": "target_crop_hard_disabled"})
                                else:
                                    retained.append(arm)
                            candidate_arms = retained
                        ranking = self.strategy_arm_ranking_agent.rank(
                            round_id=round_id + 1,
                            arms=candidate_arms,
                            context={
                                "active_learning_examples": active_learning_examples,
                                "failure_tag_counts": dict(evaluation.tag_counts or {}),
                                "core_metric_trends": evaluation_context.get("core_metric_trends"),
                                "pressure_conflict": self._latest_pressure_conflict,
                                "hypotheses": hypotheses.hypotheses,
                                "preferred_arm_id": self._preferred_arm_id,
                            },
                        )
                        ranking_path = self._write_json(round_dir / "strategy_arm_ranking.json", ranking.to_dict())
                        self._append_artifact(checkpoint, ranking_path)
                        blocked_arms = {
                            str(name)
                            for job in parent_jobs
                            for name in ((job.params or {}).get("blocked_strategy_arms") or [])
                            if str(name)
                        } | set(self.memory_store.soft_blocked_arms(memory, round_id + 1))
                        blocked_arms = self._review_and_unfreeze_arms(
                            memory=memory, round_dir=round_dir, next_round_id=round_id + 1,
                            blocked_arms=blocked_arms, arm_evidence_cards=dict(evaluation_module.get("arm_evidence_cards") or {}),
                            selection_context=selection_context, hypotheses=hypotheses.hypotheses,
                            structural_summary=struct_eval, quality_analysis=asdict(quality_analysis),
                        )
                        strategy_proposal = self.learner.propose_next(
                            round_id + 1,
                            parent_jobs,
                            evaluation.top_candidates,
                            str(self.out_dir),
                            top_k=self.cfg.active_learning.top_k,
                            policy_update=proposal.params_update,
                            structural_summary=struct_eval,
                            hypotheses=hypotheses.hypotheses,
                            quality_analysis=asdict(quality_analysis),
                            blocked_arms=blocked_arms,
                            blocked_arm_combinations=self.memory_store.active_blocked_combinations(memory),
                            pressure_conflict=self._latest_pressure_conflict,
                            active_skills=strategy_skills,
                            branch_width=self.cfg.active_learning.branch_width,
                            enable_exploitation_arms=exploitation_enabled,
                            selection_context=selection_context,
                            ranked_arm_names=ranking.ordered_arm_names,
                            defer_branch_width=True,
                        )
                        proposed_jobs = strategy_proposal.jobs
                        sampler_meta = dict((proposal.analysis_metadata or {}).get("probabilistic_sampler") or {})
                        final_sampler = dict(sampler_meta.get("final_executable_state") or {})
                        decision_spec = getattr(getattr(self.cfg, "owner", None), "parameter_decision", None)
                        if final_sampler and decision_spec is not None:
                            catalog_axes = {key: list(parameter_axis(decision_spec, key)) for key in self._sampler_keys()}
                            for job in proposed_jobs:
                                if str((job.params or {}).get("arm_id") or (job.params or {}).get("exploration_arm") or "") != "sampler_explore":
                                    continue
                                job.params.update(final_sampler)
                                job.params["final_parameter_state"] = dict(final_sampler)
                                job.params["parameter_catalog"] = catalog_axes
                                job.params["parameter_catalog_digest"] = str(sampler_meta.get("catalog_digest") or parameter_catalog_digest(decision_spec))
                        proposed_jobs = self._materialize_job_binding_types(proposed_jobs)
                        proposed_jobs = self._materialize_sampler_and_context_intents(proposed_jobs)
                        proposed_jobs = self._resolve_job_pressure_conflicts(proposed_jobs)
                        proposed_jobs = self._govern_exploration_jobs(
                            proposed_jobs,
                            current_jobs=current_jobs,
                            next_round_id=round_id + 1,
                            strict_positive_count=int(example_counts.get("strict_positive") or 0),
                            blocked_digests=self.memory_store.blocked_interventions(memory, round_id + 1),
                            prefilter_records=ranking_prefilter,
                        )
                        filtering_path = self._write_json(round_dir / "next_job_filtering_report.json", dict(getattr(self, "_last_next_job_filtering_report", {}) or {}))
                        self._append_artifact(checkpoint, filtering_path)
                        proposed_jobs = self._enforce_binder_length_range(proposed_jobs)
                        proposed_jobs = self._finalize_semantic_job_identities(
                            proposed_jobs, round_id=round_id + 1,
                        )
                        proposed_jobs = self._enforce_round_cap(proposed_jobs, round_id=round_id + 1)
                    next_identity_state = self._validate_job_identities(proposed_jobs) if proposed_jobs else {"schema_version": 1, "status": "empty", "failures": []}
                    path = self._write_json(round_dir / "next_jobs.json", [asdict(job) for job in proposed_jobs])
                    checkpoint.update({
                        "next_job_identity_state": next_identity_state,
                        "applied_params_update": dict(applied_update),
                        "config_merge_report_path": str(merge_report_path),
                        "next_jobs_path": str(path),
                        "rollback_action": rollback_decision.action,
                        "rollback_branch_from_round": rollback_decision.branch_from_round,
                        "rollback_exact_replay": bool(rollback_decision.action in self.RECOVERY_ACTIONS),
                        "rollback_replay_source_round": (
                            int(rollback_decision.best_round)
                            if rollback_decision.action in self.RECOVERY_ACTIONS else None
                        ),
                        "rollback_replay_source_job_ids": replay_source_job_ids,
                        "rollback_replay_config_snapshot": replay_snapshot,
                        "execution_failure_retry": bool(failed_job_count and proposed_jobs),
                        "continuation_prepared": bool(prepare_continuation and proposed_jobs),
                    })
                    self._append_artifact(checkpoint, path)
                    self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)
                    return {"next_jobs": proposed_jobs, "path": str(path)}

                expect_next_jobs = rollback_decision.action != "stop"
                next_jobs_module_result = self._run_validated_module(
                    module_name=stage,
                    round_id=round_id,
                    round_dir=round_dir,
                    checkpoint=checkpoint,
                    action=next_jobs_module,
                    validator=lambda result: self._validate_next_jobs_module(result, expect_jobs=expect_next_jobs),
                    loader=lambda record: {
                        "next_jobs": self._jobs_from_dicts(self._load_json_path(round_dir / "next_jobs.json")),
                        "path": str(round_dir / "next_jobs.json"),
                    },
                    retry_predicate=self._identity_or_budget_error_retryable,
                )
                next_jobs = next_jobs_module_result["next_jobs"]
                strategy_exposure = {
                    "schema_version": "1.0",
                    "origin_round_id": round_id,
                    "execution_round_id": round_id + 1,
                    "available_rule_ids": [
                        str(rule.get("rule_id"))
                        for rule in learned_strategy_rules
                        if rule.get("rule_id")
                    ],
                    "cited_rule_ids": self._collect_learned_rule_ids(
                        quality_analysis=asdict(quality_analysis),
                        hypotheses=hypotheses.hypotheses,
                        diagnostic=asdict(diagnostic),
                        input_configuration=asdict(input_config),
                        policy=asdict(proposal),
                    ),
                    "learned_skill_nonuse_reasons": self._collect_learned_nonuse_reasons(
                        quality_analysis=asdict(quality_analysis),
                        hypotheses=hypotheses.hypotheses,
                        diagnostic=asdict(diagnostic),
                        input_configuration=asdict(input_config),
                    ),
                    "applied_update": dict(proposal.params_update),
                    "config_merge_report": config_merge_report,
                    "strategy_conflict_resolution": conflict_resolution.to_dict(),
                    "next_round": {
                        "selected_arm": str((next_jobs[0].params or {}).get("exploration_arm") or "baseline_hold") if next_jobs else "",
                        "parameter_vector": self._current_config_snapshot() if next_jobs else {},
                        "logical_job_count": len(next_jobs),
                    },
                }
                if strategy_exposure["available_rule_ids"]:
                    if not any((
                        quality_analysis.llm_used,
                        hypotheses.llm_used,
                        diagnostic.llm_used,
                        input_config.llm_used,
                    )):
                        strategy_exposure["citation_compliance"] = "deterministic_fallback_no_llm"
                    else:
                        strategy_exposure["citation_compliance"] = (
                            "cited"
                            if strategy_exposure["cited_rule_ids"]
                            else "explicit_nonuse"
                            if strategy_exposure["learned_skill_nonuse_reasons"]
                            else "missing_required_citation_or_nonuse_reason"
                        )
                else:
                    strategy_exposure["citation_compliance"] = "no_active_learned_rules"
                strategy_exposure["exposure_id"] = stable_hash(strategy_exposure)[:24]
                exposure_path = self._write_json(
                    round_dir / "next_strategy_exposure.json",
                    strategy_exposure,
                )
                self._append_artifact(checkpoint, exposure_path)
                checkpoint["strategy_exposure_path"] = str(exposure_path)
                checkpoint["strategy_exposure_id"] = strategy_exposure["exposure_id"]
                self._write_checkpoint(round_dir, round_id, stage, "running", checkpoint)
                self._write_json(skills_path, self._skills_audit_payload(active_skills_by_agent))
                self.bus.publish(AgentMessage("SkillRegistry", "all", "status", {"event": "skills_activated", "agents": sorted(active_skills_by_agent), "skill_count": sum(len(v or []) for v in active_skills_by_agent.values())}, round_id=round_id, artifacts=[str(skills_path)]))

                record = self.memory_store.upsert_round(memory, round_id)
                record.ingestion = [self._compact_ingestion_record(item) for item in ingestions]
                record.evaluation = {**self._compact_evaluation(evaluation), "active_learning_examples": active_learning_examples}
                record.structural_analysis = [self._compact_structure_evidence(struct_eval)]
                record.active_learning_examples = active_learning_examples
                record.quality_analysis = asdict(quality_analysis)
                record.hypotheses = hypotheses.hypotheses
                record.decisions = [asdict(proposal), {"input_configuration": asdict(input_config)}, {"fragment_templates": asdict(fragment_templates)}, {"binder_length_recommendation": asdict(length_recommendation)}, {"config_merge_report": config_merge_report}, {"strategy_conflict_resolution": conflict_resolution.to_dict()}, {"active_skills": self._skills_audit_payload(active_skills_by_agent)}, {"quality_analysis_mode": quality_mode_decision.to_dict()}, {"self_improvement_update": self_improvement_update.to_dict() if self_improvement_update else None}, {"strategy_exposure": strategy_exposure}]
                record.retry_events = [r for r in execution_records if r.get("attempts", 0) > 1]
                record.artifacts = list(checkpoint["artifacts"])
                record.reward = round_outcome.reward
                record.config_snapshot = current_config_snapshot
                record.rollback_decision = {**rollback_decision.to_dict(), **{
                    "arm_signature": round_outcome.arm_signature, "branch_id": round_outcome.branch_id,
                    "config_digest": round_outcome.config_digest, "intervention_digest": round_outcome.intervention_digest,
                }}
                arm_evidence_cards = dict(evaluation_module.get("arm_evidence_cards") or {})
                arm_comparison = dict(wave_holder.get("arm_comparison") or self._load_json_if_exists(round_dir / "arm_comparison.json") or {})
                arm_history_resolution = dict(wave_holder.get("arm_history_resolution") or self._load_json_if_exists(round_dir / "arm_history_resolution.json") or {})
                final_strategy_decision = dict(wave_holder.get("final_strategy_decision") or self._load_json_if_exists(round_dir / "final_strategy_decision.json") or {})
                record.arm_outcomes = [dict(item) for item in arm_evidence_cards.get("arms", [])]
                record.arm_evidence_cards = arm_evidence_cards
                record.arm_comparison = arm_comparison
                record.arm_history_resolution = arm_history_resolution
                record.final_strategy_decision = final_strategy_decision
                self.memory_store.upsert_ledger_round(
                    memory, round_id=round_id, outcome=round_outcome.to_dict(),
                    policy_snapshot=current_config_snapshot,
                    failed_arms=[],
                    candidate_denominators={
                        "strict_trials": round_outcome.strict_trials,
                        "raw_candidate_count": round_outcome.raw_candidate_count,
                        "analysis_candidate_count": round_outcome.analysis_candidate_count,
                    },
                    next_hypotheses=hypotheses.hypotheses,
                )
                for arm_outcome in record.arm_outcomes:
                    self.memory_store.record_governance_outcome(
                        memory, round_id=round_id, branch_id=str(arm_outcome.get("branch_id") or f"r{round_id}_{arm_outcome.get('arm_id')}"),
                        arm_id=str(arm_outcome.get("arm_id") or "baseline_hold"), successes=int(arm_outcome.get("successes") or 0),
                        trials=int(arm_outcome.get("trials") or 0), config_digest=str(arm_outcome.get("config_digest") or ""),
                        intervention_digest=str(arm_outcome.get("intervention_digest") or ""), is_baseline=bool(arm_outcome.get("is_baseline")),
                        regressed=bool(rollback_decision.is_regression and arm_comparison.get("winner_arm_id") and str(arm_outcome.get("arm_id")) != str(arm_comparison.get("winner_arm_id"))),
                    )
                memory.experiment_ledger.best_config_retests = dict(self.rollback.best_config_retests)
                record.finished_at = time.time()
                self._index_and_compress_round_memory(
                    memory,
                    round_id=round_id,
                    evaluation=evaluation_context,
                    outcome=round_outcome.to_dict(),
                    current_config=current_config_snapshot,
                    artifact_refs=checkpoint["artifacts"],
                )
                self.memory_store.record_message_bus(memory, self.bus.read_all())
                self.memory_store.save(memory)

                llm_fallbacks_path = round_dir / "llm_fallbacks.json"
                llm_fallbacks = self._load_json_path(llm_fallbacks_path) if llm_fallbacks_path.exists() else []
                summary_round = {"round_id": round_id, "execution": execution_state, "pre_submit_summary": pre_submit_summary, "pre_submit_summary_path": str(pre_submit_summary_path), "llm_fallbacks": llm_fallbacks, "evaluation": asdict(evaluation), "active_learning_examples": active_learning_examples, "structural_analysis": asdict(struct_eval), "fragment_templates": asdict(fragment_templates), "binder_length_recommendation": asdict(length_recommendation), "quality_analysis": asdict(quality_analysis), "quality_analysis_mode": quality_mode_decision.to_dict(), "hypotheses": asdict(hypotheses), "proposal": asdict(proposal), "diagnostic": asdict(diagnostic), "input_configuration": asdict(input_config), "config_merge_report": config_merge_report, "strategy_conflict_resolution": conflict_resolution.to_dict(), "active_skills": self._skills_audit_payload(active_skills_by_agent), "self_improvement_update": self_improvement_update.to_dict() if self_improvement_update else None, "strategy_exposure": strategy_exposure, "arm_evidence_cards": arm_evidence_cards, "arm_comparison": arm_comparison, "arm_history_resolution": arm_history_resolution, "final_strategy_decision": final_strategy_decision, "rollback": rollback_decision.to_dict(), "reward": round_outcome.to_dict()}
                if self._hotspot_selection_enabled():
                    summary_round["llm_hotspot_selection"] = dict(self._latest_hotspot_selection or {})
                summary["rounds"].append(self._compact_round_summary(summary_round))
                round_bundle = build_round_analysis_bundle(
                    round_id=round_id,
                    candidates=candidates,
                    population_candidates=all_candidates,
                    evaluation=asdict(evaluation),
                    structure=asdict(struct_eval),
                    templates=asdict(fragment_templates),
                )
                if self._hotspot_selection_enabled():
                    round_bundle["llm_hotspot_selection"] = dict(self._latest_hotspot_selection or {})
                bundle_path = self._write_json(
                    round_dir / "round_analysis_bundle.json",
                    round_bundle,
                )
                self._append_artifact(checkpoint, bundle_path)
                self._iteration_metrics_cache.add_bundle(
                    self.out_dir,
                    round_id,
                    round_bundle,
                )
                io_telemetry_path = self._write_json(round_dir / "io_telemetry.json", {
                    "schema_version": 1, "round_id": round_id, **dict(self._io_telemetry),
                    "ingestion_runs": len(ingestions),
                    "metrics_files": sum(len(item.get("all_metrics_files") or []) for item in ingestions),
                    "structure_files": sum(int(item.get("structure_file_count") or 0) for item in ingestions),
                    "result_transport_copy_count": sum(len(((record.get("result_sync") or {}).get("copied") or [])) for record in execution_records),
                    "result_transport_link_count": sum(len(((record.get("result_sync") or {}).get("linked") or [])) for record in execution_records),
                })
                self._append_artifact(checkpoint, io_telemetry_path)
                self._store_completed_round_summary(
                    checkpoint=checkpoint,
                    round_dir=round_dir,
                    round_id=round_id,
                    summary_round=summary_round,
                )
                self._write_checkpoint(round_dir, round_id, "round_completed", "completed", checkpoint)
                self._write_summary(summary)

                if rollback_decision.action == "stop":
                    self.bus.publish(AgentMessage("RollbackController", "all", "status", {"event": "early_stop", "best_round": rollback_decision.best_round}, round_id=round_id))
                    break
                if round_id + 1 >= self.max_rounds:
                    break
                current_jobs = next_jobs
            except Exception as exc:
                checkpoint["error"] = {"type": type(exc).__name__, "message": str(exc)}
                if checkpoint.get("current_round_complete"):
                    checkpoint["next_round_planning_failed"] = stage == "next_jobs_proposed"
                self._write_checkpoint(round_dir, round_id, stage, "failed", checkpoint)
                self._write_summary(summary)
                raise
        self._write_summary(summary)
        return summary

    def _write_json(self, path: Path, payload: Any) -> Path:
        written = atomic_write_json(path, payload, cache=self._artifact_digest_cache)
        record = self._artifact_digest_cache.record(written)
        self._io_telemetry["json_writes"] = int(self._io_telemetry.get("json_writes", 0)) + 1
        self._io_telemetry["json_bytes_written"] = int(self._io_telemetry.get("json_bytes_written", 0)) + int(record.get("size_bytes") or 0)
        return written

    @staticmethod
    def _frozen_round_reasoning_context(context: Mapping[str, Any]) -> Dict[str, Any]:
        """Strip later-wave conclusions so Wave A agents only see frozen round evidence."""
        frozen = dict(context or {})
        for key in (
            "quality_analysis", "hypotheses", "diagnostic_report", "diagnostic",
            "arm_comparison", "arm_history_resolution", "final_strategy_decision",
            "input_configuration",
        ):
            frozen.pop(key, None)
        return frozen

    @staticmethod
    def _round_llm_wave_artifact_paths(round_dir: Path) -> Tuple[Path, ...]:
        return (
            round_dir / "binder_quality_analysis.json",
            round_dir / "hypotheses.json",
            round_dir / "diagnostic_report.json",
            round_dir / "arm_comparison.json",
            round_dir / "arm_history_resolution.json",
            round_dir / "final_strategy_decision.json",
        )

    @classmethod
    def _round_llm_wave_artifacts_ready(cls, round_dir: Path, checkpoint: Optional[Mapping[str, Any]] = None) -> bool:
        del checkpoint
        return all(path.exists() for path in cls._round_llm_wave_artifact_paths(round_dir))

    def _load_llm_wave_holder(self, round_dir: Path) -> Dict[str, Any]:
        return {
            "quality_analysis": self._quality_analysis_from_dict(self._load_json_path(round_dir / "binder_quality_analysis.json")),
            "hypotheses": self._hypotheses_from_dict(self._load_json_path(round_dir / "hypotheses.json")),
            "diagnostic": self._diagnostic_from_dict(self._load_json_path(round_dir / "diagnostic_report.json")),
            "arm_comparison": dict(self._load_json_path(round_dir / "arm_comparison.json") or {}),
            "arm_history_resolution": dict(self._load_json_path(round_dir / "arm_history_resolution.json") or {}),
            "final_strategy_decision": dict(self._load_json_path(round_dir / "final_strategy_decision.json") or {}),
        }

    def _run_llm_wave(
        self,
        *,
        wave_name: str,
        round_id: int,
        tasks: Mapping[str, Callable[[], Any]],
    ) -> Dict[str, Any]:
        """Run independent LLM callables in one pool. Persist/checkpoint stays on this thread."""
        if not tasks:
            return {}
        graph = getattr(self, "round_graph", None) or RoundGraph()
        wave = graph.run_wave(wave_name, tasks)
        results = wave.results
        errors = wave.errors
        self.bus.publish(AgentMessage(
            "Orchestrator", "all", "status",
            {
                "event": "llm_dependency_wave_completed",
                "wave": wave_name,
                "tasks": sorted(tasks),
                "failed_tasks": sorted(errors),
            },
            round_id=round_id,
        ))
        if errors:
            first = next(iter(errors.values()))
            raise RuntimeError(
                "llm wave %s failed: %s" % (wave_name, {name: "%s: %s" % (type(exc).__name__, exc) for name, exc in errors.items()})
            ) from first
        return results

    def _run_validated_module(
        self,
        *,
        module_name: str,
        round_id: int,
        round_dir: Path,
        checkpoint: Dict[str, Any],
        action: Callable[[], Any],
        validator: Callable[[Any], None],
        loader: Optional[Callable[[Mapping[str, Any]], Any]] = None,
        max_attempts: Optional[int] = None,
        retry_predicate: Optional[Callable[[Exception], bool]] = None,
    ) -> Any:
        """Run one pipeline module until its output satisfies the next input contract."""
        input_artifacts = self._checkpoint_artifact_records(checkpoint)
        if loader is not None:
            resumed = self._try_resume_validated_module(
                module_name=module_name,
                round_id=round_id,
                checkpoint=checkpoint,
                loader=loader,
                validator=validator,
                input_artifacts=input_artifacts,
            )
            if resumed is not None:
                self.bus.publish(AgentMessage("Orchestrator", "all", "status", {"event": "module_resumed", "module": module_name}, round_id=round_id))
                return resumed
        attempts = max(1, int(max_attempts if max_attempts is not None else self.max_retries))
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                result = action()
                validator(result)
                self._record_module_completed(
                    checkpoint=checkpoint,
                    module_name=module_name,
                    result=result,
                    input_artifacts=input_artifacts,
                    round_dir=round_dir,
                    round_id=round_id,
                )
                if attempt > 1:
                    self.bus.publish(AgentMessage("Orchestrator", "all", "status", {"event": "module_retry_recovered", "module": module_name, "attempt": attempt}, round_id=round_id))
                return result
            except Exception as exc:
                last_error = exc
                retry_record = {
                    "module": module_name,
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                checkpoint.setdefault("module_retries", []).append(retry_record)
                retryable = retry_predicate(exc) if retry_predicate is not None else True
                retry_record["retryable"] = bool(retryable)
                if attempt >= attempts or not retryable:
                    event = "module_retry_exhausted" if retryable else "module_non_retryable_failure"
                    self.bus.publish(AgentMessage("Orchestrator", "all", "failure", {"event": event, **retry_record}, round_id=round_id))
                    self._write_checkpoint(round_dir, round_id, module_name, "failed", checkpoint)
                    raise
                self.bus.publish(AgentMessage("Orchestrator", "all", "retry", {"event": "module_output_invalid", "next_attempt": attempt + 1, **retry_record}, round_id=round_id))
                self._write_checkpoint(round_dir, round_id, module_name, "retrying", checkpoint)
        if last_error is not None:
            raise last_error
        raise ModuleOutputValidationError(f"{module_name}: no module attempts were executed")

    def _try_resume_validated_module(
        self,
        *,
        module_name: str,
        round_id: int,
        checkpoint: Mapping[str, Any],
        loader: Callable[[Mapping[str, Any]], Any],
        validator: Callable[[Any], None],
        input_artifacts: List[Dict[str, Any]],
    ) -> Optional[Any]:
        module_record = ((checkpoint.get("modules") or {}).get(module_name) or {})
        if not isinstance(module_record, Mapping) or module_record.get("status") != "completed":
            return None
        if not artifacts_match(module_record.get("artifacts"), cache=self._artifact_digest_cache):
            return None
        if not artifacts_match(module_record.get("input_artifacts"), cache=self._artifact_digest_cache):
            return None
        expected_inputs = {
            str(item.get("path")): item.get("sha256")
            for item in module_record.get("input_artifacts") or []
            if isinstance(item, Mapping) and item.get("path") and item.get("sha256")
        }
        current_inputs = {
            str(item.get("path")): item.get("sha256")
            for item in input_artifacts
            if isinstance(item, Mapping) and item.get("path") and item.get("sha256")
        }
        for path, digest in expected_inputs.items():
            if current_inputs.get(path) != digest:
                return None
        try:
            result = loader(module_record)
            validator(result)
        except Exception as exc:
            self.bus.publish(AgentMessage("Orchestrator", "all", "status", {"event": "module_resume_rejected", "module": module_name, "error": str(exc)}, round_id=round_id))
            return None
        return result

    def _record_module_completed(
        self,
        *,
        checkpoint: Dict[str, Any],
        module_name: str,
        result: Any,
        input_artifacts: List[Dict[str, Any]],
        round_dir: Path,
        round_id: int,
    ) -> None:
        artifacts = [
            artifact_record(path, cache=self._artifact_digest_cache)
            for path in self._module_output_paths(result)
        ]
        if not artifacts:
            artifacts = self._checkpoint_artifact_records(checkpoint)
        output_digest = stable_hash([
            {"path": item.get("path"), "sha256": item.get("sha256"), "size_bytes": item.get("size_bytes")}
            for item in artifacts
        ])
        checkpoint.setdefault("modules", {})[module_name] = {
            "status": "completed",
            "phase": "output_validated",
            "schema_version": 2,
            "completed_at": time.time(),
            "input_digest": stable_hash(input_artifacts),
            "output_digest": output_digest,
            "publish_status": "not_published",
            "input_artifacts": input_artifacts,
            "artifacts": artifacts,
        }
        self._write_checkpoint(round_dir, round_id, module_name, "running", checkpoint)

    @staticmethod
    def _module_output_paths(result: Any) -> List[Path]:
        if not isinstance(result, Mapping):
            return []
        paths: List[Path] = []
        for key, value in result.items():
            if key == "path" or key.endswith("_path"):
                if value:
                    paths.append(Path(str(value)))
            elif key.endswith("_paths") and isinstance(value, list):
                paths.extend(Path(str(item)) for item in value if item)
        seen = set()
        unique: List[Path] = []
        for path in paths:
            text = str(path)
            if text not in seen:
                seen.add(text)
                unique.append(path)
        return unique

    def _checkpoint_artifact_records(self, checkpoint: Mapping[str, Any]) -> List[Dict[str, Any]]:
        records = []
        for path in checkpoint.get("artifacts") or []:
            try:
                artifact = artifact_record(path, cache=self._artifact_digest_cache)
            except OSError:
                continue
            if artifact.get("exists"):
                records.append(artifact)
        return records

    @staticmethod
    def _append_artifact(checkpoint: Dict[str, Any], path: Union[str, Path]) -> None:
        artifacts = checkpoint.setdefault("artifacts", [])
        value = str(path)
        if value not in artifacts:
            artifacts.append(value)

    def _validate_json_artifact(self, path: Union[str, Path], *, expected_type: Optional[type] = None) -> Any:
        artifact = Path(path)
        if not artifact.exists():
            raise ModuleOutputValidationError(f"expected artifact does not exist: {artifact}")
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModuleOutputValidationError(f"artifact is not valid JSON: {artifact}: {exc}") from exc
        if expected_type is not None and not isinstance(payload, expected_type):
            raise ModuleOutputValidationError(f"artifact {artifact} expected {expected_type.__name__}, got {type(payload).__name__}")
        return payload

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ModuleOutputValidationError(message)

    def _write_summary(self, summary: Mapping[str, Any]) -> Path:
        payload = dict(summary)
        round_count = len(summary.get("rounds") or [])
        plot_status: Optional[Dict[str, Any]] = None
        if round_count != self._last_plotted_round_count:
            raw_plot_status = self._write_iteration_metric_plots()
            if isinstance(raw_plot_status, Mapping):
                plot_status = dict(raw_plot_status)
                stable_plot_status = plot_status.get("status") in {"completed", "no_data"}
            else:
                # Compatibility for injected/legacy helpers that returned bool.
                stable_plot_status = bool(raw_plot_status)
            if stable_plot_status:
                self._last_plotted_round_count = round_count
        else:
            status_path = self.out_dir / "iteration_metrics_plot_status.json"
            if status_path.exists():
                try:
                    loaded = json.loads(status_path.read_text(encoding="utf-8"))
                except Exception:
                    loaded = None
                if isinstance(loaded, dict):
                    plot_status = loaded
        if plot_status is not None:
            payload["iteration_metrics_plot_status"] = {
                key: copy.deepcopy(plot_status[key])
                for key in ("status", "event", "error", "artifacts")
                if key in plot_status
            }
        return self._write_json(self.out_dir / "orchestrator_summary.json", payload)

    @staticmethod
    def _job_identity_digest(job: DesignJob) -> str:
        metadata = dict((job.params or {}).get("job_identity") or {})
        return str(metadata.get("semantic_digest") or job_identity_semantic_digest(job))

    @classmethod
    def _validate_job_identities(cls, jobs: Sequence[DesignJob], *, allow_legacy: bool = False) -> Dict[str, Any]:
        rows = list(jobs or [])
        job_ids = [str(job.job_id or "") for job in rows]
        branch_ids = [str((job.params or {}).get("branch_id") or job.job_id or "") for job in rows]
        outputs = [str(Path(job.output_dir).resolve()) for job in rows]
        failures: List[str] = []
        if any(not value for value in job_ids): failures.append("job_id_missing")
        if any(not value for value in branch_ids): failures.append("branch_id_missing")
        for code, values in (("duplicate_job_id", job_ids), ("duplicate_branch_id", branch_ids), ("duplicate_output_dir", outputs)):
            duplicates = sorted({value for value in values if value and values.count(value) > 1})
            if duplicates: failures.append(code + ":" + ",".join(duplicates))
        known = set(job_ids)
        for job in rows:
            metadata = dict((job.params or {}).get("job_identity") or {})
            if metadata:
                if str(metadata.get("job_id") or "") != job.job_id or str(metadata.get("branch_id") or "") != str((job.params or {}).get("branch_id") or ""):
                    failures.append(f"job_identity_metadata_mismatch:{job.job_id}")
                expected_digest = (
                    job_identity_semantic_digest(job)
                    if bool(metadata.get("finalized"))
                    else stable_hash({"execution_semantic_digest": effective_semantic_digest(job), "attribution_identity_digest": str((job.params or {}).get("attribution_identity_digest") or "")})
                )
                if str(metadata.get("semantic_digest") or "") != expected_digest:
                    failures.append(f"job_identity_digest_mismatch:{job.job_id}")
        legacy = bool(failures) and allow_legacy and all(item.startswith(("duplicate_job_id", "duplicate_branch_id", "duplicate_output_dir")) for item in failures)
        if failures and not legacy:
            raise ValueError(";".join(failures))
        return {"schema_version": 1, "status": "legacy_identity_ambiguous" if legacy else "validated", "failures": failures}

    def _finalize_semantic_job_identities(
        self, jobs: Sequence[DesignJob], *, round_id: int,
    ) -> List[DesignJob]:
        """Finalize identity after the last semantic mutation in a phase."""
        rows = deduplicate_effective_jobs(list(jobs or []))
        if not rows:
            return []
        return materialize_deterministic_job_identities(
            rows, round_id=int(round_id), output_root=str(self.out_dir),
        )

    def _finalize_execution_job_identities(
        self, jobs: Sequence[DesignJob], *, round_id: int,
    ) -> List[DesignJob]:
        """Finalize IDs only after budget, plans and execution partitions exist."""
        rows = list(jobs or [])
        if not rows:
            return []
        finalized = materialize_deterministic_job_identities(
            rows, round_id=int(round_id), output_root=str(self.out_dir), finalized=True,
        )
        self._validate_job_identities(finalized)
        return finalized

    def _bind_execution_identities_if_needed(
        self, jobs: Sequence[DesignJob], *, round_id: int,
    ) -> List[DesignJob]:
        """Bind execution slots and job output dirs onto logical continuation jobs.

        Exact best-config replay stops at semantic identity so the cloned
        strategy can keep its preserved budget. Submit and resume require
        ``job_identity.finalized`` plus a unique directory under ``jobs/``.
        """
        rows = list(jobs or [])
        if not rows:
            return []
        if all(bool(((job.params or {}).get("job_identity") or {}).get("finalized")) for job in rows):
            return rows
        return self._finalize_execution_job_identities(rows, round_id=int(round_id))

    @staticmethod
    def _identity_round_id(jobs: Sequence[DesignJob]) -> int:
        for job in jobs or []:
            for value in (job.job_id, ((job.params or {}).get("job_identity") or {}).get("job_id")):
                text = str(value or "")
                if text.startswith("r") and "_" in text:
                    try:
                        return int(text[1:text.index("_")])
                    except ValueError:
                        pass
        return 0

    @staticmethod
    def _index_records_by_job_id(records: Sequence[Mapping[str, Any]], *, label: str) -> Dict[str, Mapping[str, Any]]:
        result: Dict[str, Mapping[str, Any]] = {}
        for row in records or []:
            job_id = str(row.get("job_id") or (row.get("job") or {}).get("job_id") or "")
            if not job_id:
                raise ValueError(f"{label}_job_id_missing")
            if job_id in result:
                raise ValueError(f"duplicate_{label}_job_id:{job_id}")
            result[job_id] = row
        return result

    @classmethod
    def _records_for_jobs(
        cls, jobs: Sequence[DesignJob], records: Sequence[Mapping[str, Any]], *, label: str, allow_legacy_order: bool = False,
    ) -> List[Mapping[str, Any]]:
        job_rows = list(jobs or []); record_rows = list(records or [])
        identity = cls._validate_job_identities(job_rows, allow_legacy=allow_legacy_order)
        if identity["status"] == "legacy_identity_ambiguous":
            if len(job_rows) != len(record_rows):
                raise ValueError(f"legacy_identity_ambiguous_{label}_count_mismatch")
            for job, record in zip(job_rows, record_rows):
                record_id = str(record.get("job_id") or (record.get("job") or {}).get("job_id") or "")
                if record_id != job.job_id:
                    raise ValueError(f"legacy_identity_ambiguous_{label}_order_mismatch")
            return record_rows
        index = cls._index_records_by_job_id(record_rows, label=label)
        if set(index) != {job.job_id for job in job_rows}:
            raise ValueError(f"{label}_job_identity_set_mismatch")
        return [index[job.job_id] for job in job_rows]

    def _validate_execution_module(self, result: Mapping[str, Any], jobs: List[DesignJob]) -> None:
        records = result.get("records")
        payload = self._validate_json_artifact(result.get("path", ""), expected_type=list)
        self._require(isinstance(records, list), "execution_records must be a list")
        self._require(len(records) == len(jobs), f"execution_records count {len(records)} does not match job count {len(jobs)}")
        self._require(len(payload) == len(records), "execution_records artifact content differs from in-memory records")
        ordered_records = self._records_for_jobs(jobs, records, label="execution_record")
        for job, record in zip(jobs, ordered_records):
            expected_digest = self._job_identity_digest(job)
            actual_digest = str(record.get("job_identity_digest") or expected_digest)
            self._require(actual_digest == expected_digest, f"execution_record_identity_mismatch:{job.job_id}")
            self._require(isinstance(record, Mapping), "each execution record must be an object")
            status = str(record.get("status") or "")
            self._require(bool(status), "execution record missing status")
            self._require(bool(record.get("job_id") or (record.get("job") or {}).get("job_id")), "execution record missing job_id")
            if status.lower() not in self.FAILURE_STATUSES and status.lower() != "dry_run":
                self._require(bool(record.get("output_dir") or record.get("local_output_dir")), "successful execution record missing output_dir")

    @staticmethod
    def _identity_or_budget_error_retryable(exc: Exception) -> bool:
        message = str(exc).lower()
        deterministic = (
            "duplicate_job_id", "duplicate_branch_id", "duplicate_output_dir",
            "job_identity", "identity_mismatch", "dangling_baseline_reference",
            "normal logical round budget mismatch", "round budget resolver failed",
            "attempt_ledger_output_mismatch", "attempt_ledger_identity_mismatch",
            "execution_job_identity_not_finalized", "next_jobs_execution_identity_not_finalized",
            "shard_source_",
        )
        return not any(token in message for token in deterministic)

    @staticmethod
    def _ingestion_error_retryable(exc: Exception) -> bool:
        if isinstance(exc, (ResultPathSafetyError, TransportBindingError)):
            return False
        if isinstance(exc, (FileNotFoundError, TimeoutError, BlockingIOError, InterruptedError)):
            return True
        if isinstance(exc, OSError):
            return True
        message = str(exc).lower()
        deterministic = ("unsafe path", "outside declared arm_root", "transport binding mismatch", "result manifest files must be a list", "invalid v2 result manifest", "schema", "digest mismatch")
        return not any(token in message for token in deterministic)

    def _validate_ingestion_module(self, result: Mapping[str, Any], jobs: List[DesignJob]) -> None:
        ingestions = result.get("ingestions")
        payload = self._validate_json_artifact(result.get("path", ""), expected_type=list)
        self._require(isinstance(ingestions, list), "ingestions must be a list")
        self._require(len(ingestions) == len(jobs), f"ingestions count {len(ingestions)} does not match job count {len(jobs)}")
        self._require(len(payload) == len(ingestions), "ingestions artifact content differs from in-memory ingestions")
        for item in ingestions:
            self._require(isinstance(item, Mapping), "each ingestion entry must be an object")
            self._require("output_dir" in item, "ingestion entry missing output_dir")
            for key in ["metrics_files", "structure_files", "candidates", "run_level_issues"]:
                self._require(isinstance(item.get(key), list), f"ingestion entry {key} must be a list")

    def _validate_evaluation_module(self, result: Mapping[str, Any]) -> None:
        evaluation = result.get("evaluation")
        payload = self._validate_json_artifact(result.get("path", ""), expected_type=dict)
        data = asdict(evaluation) if evaluation is not None else {}
        self._require(isinstance(payload.get("tag_counts"), dict), "evaluation tag_counts must be an object")
        self._require(isinstance(payload.get("top_candidates"), list), "evaluation top_candidates must be a list")
        self._require(isinstance(payload.get("failed_examples"), list), "evaluation failed_examples must be a list")
        self._require(isinstance(payload.get("observations"), list), "evaluation observations must be a list")
        self._require(int(data.get("total_candidates", -1)) >= 0, "evaluation total_candidates must be non-negative")

    def _validate_structure_module(self, result: Mapping[str, Any]) -> None:
        struct_eval = result.get("structural_analysis")
        payload = self._validate_json_artifact(result.get("structure_path", ""), expected_type=dict)
        self._validate_json_artifact(result.get("fragment_templates_path", ""), expected_type=dict)
        length_payload = self._validate_json_artifact(result.get("length_recommendation_path", ""), expected_type=dict)
        self._require(isinstance(length_payload.get("recommended_lengths"), list), "binder length recommendation recommended_lengths must be a list")
        self._require(isinstance(length_payload.get("recommended_config"), dict), "binder length recommendation recommended_config must be an object")
        data = asdict(struct_eval) if struct_eval is not None else {}
        self._require(isinstance(payload.get("summaries"), list), "structure summaries must be a list")
        self._require(isinstance(payload.get("aggregate_tags"), dict), "structure aggregate_tags must be an object")
        self._require(isinstance(payload.get("observations"), list), "structure observations must be a list")
        self._require(int(data.get("total_structures", -1)) >= 0, "structure total_structures must be non-negative")

    def _validate_self_improvement_module(self, result: Mapping[str, Any]) -> None:
        evidence = self._validate_json_artifact(result.get("evidence_path", ""), expected_type=dict)
        update = self._validate_json_artifact(result.get("update_path", ""), expected_type=dict)
        snapshot_path = Path(str(result.get("snapshot_path") or ""))
        self._require(snapshot_path.exists(), "self-improvement snapshot does not exist")
        try:
            snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
            validate_skill_document(snapshot)
        except Exception as exc:
            raise ModuleOutputValidationError(
                "self-improvement snapshot is invalid: %s" % exc
            ) from exc
        self._require(isinstance(evidence.get("outcome"), dict), "self-improvement evidence missing outcome")
        self._require(isinstance(update.get("operations"), list), "self-improvement update operations must be a list")
        self._require(isinstance(update.get("semantic_relations"), list), "self-improvement semantic_relations must be a list")
        self._require(isinstance(result.get("document"), Mapping), "self-improvement result missing document")

    def _load_self_improvement_module(self, round_dir: Path) -> Dict[str, Any]:
        evidence_path = round_dir / "self_improvement_evidence.json"
        update_path = round_dir / "self_improvement_update.json"
        snapshot_path = round_dir / "self_improvement_skill_snapshot.yaml"
        update_payload = self._load_json_path(update_path)
        update = SelfImprovementUpdate(
            round_id=int(update_payload.get("round_id") or 0),
            llm_used=bool(update_payload.get("llm_used")),
            operations=list(update_payload.get("operations") or []),
            semantic_relations=list(update_payload.get("semantic_relations") or []),
            rejected_operations=list(update_payload.get("rejected_operations") or []),
            sanitization_notes=list(update_payload.get("sanitization_notes") or []),
            summary=str(update_payload.get("summary") or ""),
            raw=dict(update_payload.get("raw") or {}),
        )
        document = validate_skill_document(
            yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
        )
        if self.self_improvement_store is not None:
            self.self_improvement_store.save(document)
        return {
            "update": update,
            "document": document,
            "evidence_path": str(evidence_path),
            "update_path": str(update_path),
            "snapshot_path": str(snapshot_path),
        }

    def _validate_quality_module(self, result: Mapping[str, Any]) -> None:
        payload = self._validate_json_artifact(result.get("path", ""), expected_type=dict)
        self._require(isinstance(payload.get("overall_assessment"), str), "quality overall_assessment must be a string")
        for key in ["high_quality_modules", "low_quality_modules", "causal_factors", "next_round_guidance"]:
            self._require(isinstance(payload.get(key), list), f"quality {key} must be a list")
        self._validate_learned_skill_usage(payload, module_name="quality")

    def _validate_hypotheses_module(self, result: Mapping[str, Any]) -> None:
        payload = self._validate_json_artifact(result.get("path", ""), expected_type=dict)
        self._require(isinstance(payload.get("hypotheses"), list), "hypotheses must be a list")
        self._validate_learned_skill_usage(payload, module_name="hypotheses")

    def _validate_diagnostic_module(self, result: Mapping[str, Any]) -> None:
        payload = self._validate_json_artifact(result.get("path", ""), expected_type=dict)
        self._require(isinstance(payload.get("status_diagnosis"), str), "diagnostic status_diagnosis must be a string")
        self._require(isinstance(payload.get("pipeline_health"), dict), "diagnostic pipeline_health must be an object")
        self._require(isinstance(payload.get("corrective_actions"), list), "diagnostic corrective_actions must be a list")
        self._validate_learned_skill_usage(payload, module_name="diagnostic")

    def _validate_input_config_module(self, result: Mapping[str, Any]) -> None:
        payload = self._validate_json_artifact(result.get("path", ""), expected_type=dict)
        self._require(isinstance(payload.get("recommended_config"), dict), "input configuration recommended_config must be an object")
        self._require(isinstance(payload.get("parameter_rationale"), list), "input configuration parameter_rationale must be a list")
        self._validate_learned_skill_usage(payload, module_name="input_configuration")

    def _validate_learned_skill_usage(
        self,
        payload: Mapping[str, Any],
        *,
        module_name: str,
    ) -> None:
        if not payload.get("llm_used") or not self.self_improvement_document:
            return
        active_ids = {
            str(rule.get("rule_id"))
            for rule in active_prompt_rules(
                self.self_improvement_document,
                limit=self.cfg.self_improvement.max_active_rules,
            )
            if rule.get("rule_id")
        }
        if not active_ids:
            return
        cited = set(self._collect_learned_rule_ids(payload=payload))
        nonuse = self._collect_learned_nonuse_reasons(payload=payload)
        self._require(
            bool(cited or nonuse),
            "%s LLM output must cite learned_rule_ids or learned_skill_nonuse_reason"
            % module_name,
        )
        unknown = sorted(cited - active_ids)
        self._require(
            not unknown,
            "%s cited unknown/inactive learned rules: %s"
            % (module_name, ", ".join(unknown)),
        )

    def _validate_policy_module(self, result: Mapping[str, Any]) -> None:
        proposal_payload = self._validate_json_artifact(result.get("proposal_path", ""), expected_type=dict)
        merge_payload = self._validate_json_artifact(result.get("merge_report_path", ""), expected_type=dict)
        conflicts_payload = self._validate_json_artifact(result.get("conflicts_path", ""), expected_type=list)
        resolution_payload = self._validate_json_artifact(result.get("conflict_resolution_path", ""), expected_type=dict)
        config_path = Path(str(result.get("config_path") or ""))
        self._require(config_path.exists(), f"next round config does not exist: {config_path}")
        self._require(isinstance(proposal_payload.get("params_update"), dict), "policy proposal params_update must be an object")
        self._require(isinstance(merge_payload.get("applied_update"), dict), "config merge report applied_update must be an object")
        self._require(isinstance(conflicts_payload, list), "strategy conflicts must be a list")
        self._require(isinstance(resolution_payload.get("decisions"), list), "conflict resolution decisions must be a list")

    def _validate_next_jobs_module(self, result: Mapping[str, Any], *, expect_jobs: bool) -> None:
        payload = self._validate_json_artifact(result.get("path", ""), expected_type=list)
        next_jobs = result.get("next_jobs")
        self._require(isinstance(next_jobs, list), "next_jobs must be a list")
        self._require(len(payload) == len(next_jobs), "next_jobs artifact content differs from in-memory next jobs")
        if expect_jobs:
            self._require(len(next_jobs) > 0, "next round expected jobs but next_jobs is empty")
        if next_jobs:
            self._validate_job_identities(next_jobs)
            unfinalized = [
                job.job_id for job in next_jobs
                if not bool(((job.params or {}).get("job_identity") or {}).get("finalized"))
            ]
            self._require(
                not unfinalized,
                "next_jobs_execution_identity_not_finalized:" + ",".join(unfinalized),
            )
        allowed_lengths = set(self._allowed_binder_lengths())
        for job in next_jobs:
            self._require(isinstance(job, DesignJob), "each next job must be a DesignJob")
            if allowed_lengths:
                self._require(int(job.binder_length) in allowed_lengths, f"next job binder_length {job.binder_length} violates binder_length_range")

    def _write_iteration_metric_plots(self) -> Dict[str, Any]:
        status_path = self.out_dir / "iteration_metrics_plot_status.json"
        try:
            artifacts = plot_iteration_metrics(
                self.out_dir,
                cache=self._iteration_metrics_cache,
            )
        except IterationMetricsNoDataError as exc:
            payload = {"status": "no_data", "event": "iteration_plot_no_data", "error_type": type(exc).__name__, "error": str(exc), "updated_at": time.time()}
            self._write_json(status_path, payload)
            self.bus.publish(AgentMessage("Orchestrator", "all", "status", payload, artifacts=[str(status_path)]))
            return payload
        except (IterationMetricsInputError, ValueError) as exc:
            payload = {"status": "failed", "event": "iteration_plot_input_failed", "error_type": type(exc).__name__, "error": str(exc), "updated_at": time.time()}
            self._write_json(status_path, payload)
            self.bus.publish(AgentMessage("Orchestrator", "all", "error", payload, artifacts=[str(status_path)]))
            return payload
        except Exception as exc:
            payload = {"status": "failed", "event": "iteration_plot_failed", "error_type": type(exc).__name__, "error": str(exc), "updated_at": time.time()}
            self._write_json(status_path, payload)
            self.bus.publish(AgentMessage("Orchestrator", "all", "error", payload, artifacts=[str(status_path)]))
            return payload
        payload = {"status": "completed", "event": "iteration_metrics_plotted", "artifacts": {key: str(value) for key, value in artifacts.items()}, "updated_at": time.time()}
        self._write_json(status_path, payload)
        self.bus.publish(
            AgentMessage(
                "Orchestrator",
                "all",
                "status",
                {
                    "event": "iteration_metrics_plotted",
                    "plot_png": str(artifacts["plot_png"]),
                    "stats_json": str(artifacts["stats_json"]),
                },
            )
        )
        return payload

    @staticmethod
    def _compact_structure_evidence(batch: Any, *, limit: int = 24) -> Dict[str, Any]:
        data = asdict(batch) if hasattr(batch, "summaries") else dict(batch or {})
        summaries = list(data.get("summaries") or [])
        ranked = sorted(summaries, key=lambda item: float(item.get("reliability_score") or 0.0), reverse=True)
        data["summaries"] = ranked[:max(1, int(limit))]
        data["summaries_total"] = len(summaries)
        data["summaries_truncated"] = len(summaries) > len(data["summaries"])
        return data

    @staticmethod
    def _compact_fragment_template_evidence(batch: Any, *, limit: int = 12) -> Dict[str, Any]:
        data = asdict(batch) if hasattr(batch, "templates") else dict(batch or {})
        for key in ("templates", "library"):
            if key not in data:
                continue
            values = list(data.get(key) or [])
            data[key] = values[:max(1, int(limit))]
            data[f"{key}_total"] = len(values)
        return data

    @staticmethod
    def _compact_ingestion_record(item: Mapping[str, Any]) -> Dict[str, Any]:
        keep = ("output_dir", "job_id", "arm_id", "logical_branch_id", "execution_job_id", "execution_slot", "candidate_scope", "selected_metric_count", "unfiltered_metric_count", "filter_pass_count", "selected_failed_filter_count", "core_ingestion_status", "metrics_rows_read", "structure_file_count", "population_metadata", "run_level_issues")
        return {key: copy.deepcopy(item.get(key)) for key in keep if key in item}

    @staticmethod
    def _compact_evaluation(evaluation: Any) -> Dict[str, Any]:
        data = asdict(evaluation) if not isinstance(evaluation, Mapping) else dict(evaluation)
        data.pop("failed_examples", None)
        data["top_candidates"] = list(data.get("top_candidates") or [])[:12]
        return data

    @classmethod
    def _compact_round_summary(cls, summary: Mapping[str, Any]) -> Dict[str, Any]:
        keep = ("round_id", "execution", "pre_submit_summary_path", "quality_analysis_mode", "config_merge_report", "strategy_conflict_resolution", "strategy_exposure", "arm_evidence_cards", "arm_comparison", "final_strategy_decision", "rollback", "reward", "llm_hotspot_selection")
        result = {key: copy.deepcopy(summary.get(key)) for key in keep if key in summary}
        if "evaluation" in summary:
            result["evaluation"] = cls._compact_evaluation(summary.get("evaluation") or {})
        result["artifact_refs"] = {
            key: value for key, value in summary.items()
            if key.endswith("_path") and isinstance(value, str)
        }
        return result

    def _write_checkpoint(self, round_dir: Path, round_id: int, stage: str, status: str, payload: Mapping[str, Any]) -> Path:
        body = dict(payload)
        if body.get("pre_submit_summary_path"):
            body.pop("pre_submit_summary", None)
        if body.get("execution_state") and (round_dir / "execution_state.json").exists():
            body.pop("execution_state", None)
            body["execution_state_path"] = str(round_dir / "execution_state.json")
        if body.get("next_jobs_path"):
            body.pop("next_jobs", None)
        checkpoint = {
            "round_id": round_id, "stage": stage, "phase": stage,
            "status": status, "updated_at": time.time(), **body,
        }
        return self._write_json(round_dir / "round_checkpoint.json", checkpoint)

    def _load_round_checkpoint(self, round_dir: Path, round_id: int) -> Optional[Dict[str, Any]]:
        checkpoint_path = round_dir / "round_checkpoint.json"
        if not checkpoint_path.exists():
            return None
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        try:
            checkpoint_round = int(checkpoint.get("round_id", -1)) if isinstance(checkpoint, dict) else -1
        except (TypeError, ValueError):
            checkpoint_round = -1
        if not isinstance(checkpoint, dict) or checkpoint_round != int(round_id):
            return None
        if checkpoint.get("status") == "completed":
            return None
        checkpoint.setdefault("artifacts", [])
        checkpoint.setdefault("modules", {})
        return checkpoint

    def _store_completed_round_summary(
        self,
        *,
        checkpoint: Dict[str, Any],
        round_dir: Path,
        round_id: int,
        summary_round: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Persist detailed evidence separately and expose a compact round summary."""
        details_path = self._write_json(round_dir / "round_details.json", dict(summary_round))
        details_record = artifact_record(details_path, cache=self._artifact_digest_cache)
        compact = self._compact_round_summary(summary_round)
        compact["round_details_ref"] = {
            "path": str(details_path), "sha256": str(details_record.get("sha256") or ""),
            "size_bytes": int(details_record.get("size_bytes") or 0),
        }
        round_summary_path = self._write_json(round_dir / "round_summary.json", compact)
        round_summary_record = artifact_record(round_summary_path, cache=self._artifact_digest_cache)
        checkpoint.pop("summary_round", None)
        checkpoint["round_summary_ref"] = {
            "schema_version": 1,
            "round_id": round_id,
            "path": str(round_summary_path),
            "sha256": str(round_summary_record.get("sha256") or ""),
            "size_bytes": int(round_summary_record.get("size_bytes") or 0),
        }
        final_decision = summary_round.get("final_strategy_decision")
        selected_arm_id = final_decision.get("selected_arm_id") if isinstance(final_decision, Mapping) else None
        checkpoint["preferred_arm_id"] = str(selected_arm_id) if selected_arm_id else None
        return dict(checkpoint["round_summary_ref"])

    def _load_checkpoint_round_summary(
        self,
        checkpoint: Mapping[str, Any],
        round_dir: Path,
        round_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Load a completed round summary from compact or legacy checkpoints."""
        summary_ref = checkpoint.get("round_summary_ref")
        if isinstance(summary_ref, Mapping):
            summary_path_value = summary_ref.get("path")
            if not summary_path_value:
                raise RuntimeError(f"round summary reference is missing a path for round_{round_id:02d}")
            summary_path = Path(str(summary_path_value))
            if not summary_path.is_absolute():
                summary_path = round_dir / summary_path
            try:
                summary_bytes = summary_path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"cannot load round summary artifact for round_{round_id:02d}: {summary_path}"
                ) from exc
            expected_digest = str(summary_ref.get("sha256") or "")
            actual_digest = hashlib.sha256(summary_bytes).hexdigest()
            if expected_digest and actual_digest != expected_digest:
                raise RuntimeError(f"round summary digest mismatch for round_{round_id:02d}")
            expected_size = summary_ref.get("size_bytes")
            if expected_size is not None and int(expected_size) != len(summary_bytes):
                raise RuntimeError(f"round summary size mismatch for round_{round_id:02d}")
            try:
                summary_round = json.loads(summary_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid round summary artifact for round_{round_id:02d}") from exc
            if not isinstance(summary_round, dict):
                raise RuntimeError(f"round summary artifact is not an object for round_{round_id:02d}")
            try:
                artifact_round_id = int(summary_round.get("round_id", -1))
                reference_round_id = int(summary_ref.get("round_id", round_id))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid round id in round summary reference for round_{round_id:02d}") from exc
            if artifact_round_id != round_id or reference_round_id != round_id:
                raise RuntimeError(f"round summary round id mismatch for round_{round_id:02d}")
            return dict(summary_round)
        legacy_summary = checkpoint.get("summary_round")
        return dict(legacy_summary) if isinstance(legacy_summary, dict) else None

    def _recover_completed_rounds(self, initial_jobs: List[DesignJob]) -> Tuple[int, List[DesignJob], List[Dict[str, Any]]]:
        current_jobs = list(initial_jobs)
        recovered_rounds: List[Dict[str, Any]] = []
        next_round = 0
        for round_id in range(self.max_rounds):
            checkpoint_path = self.out_dir / f"round_{round_id:02d}" / "round_checkpoint.json"
            if not checkpoint_path.exists():
                break
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except Exception:
                break
            if checkpoint.get("status") != "completed":
                break
            if int(checkpoint.get("round_id", -1)) != int(round_id):
                raise RuntimeError(
                    f"Completed round checkpoint is discontinuous or mislabeled at round_{round_id:02d}: "
                    f"expected round_id={round_id}, got {checkpoint.get('round_id')!r}"
                )
            summary_round = self._load_checkpoint_round_summary(checkpoint, checkpoint_path.parent, round_id)
            if summary_round is not None:
                if not isinstance(summary_round.get("pre_submit_summary"), dict):
                    pre_submit_summary = self._load_checkpoint_pre_submit_summary(checkpoint, checkpoint_path.parent)
                    if pre_submit_summary is not None:
                        summary_round["pre_submit_summary"] = pre_submit_summary
                        summary_round.setdefault("pre_submit_summary_path", str(checkpoint_path.parent / "pre_submit_summary.json"))
                recovered_rounds.append(summary_round)
                recovered_decision = dict(summary_round.get("final_strategy_decision") or {})
                recovered_arm_id = recovered_decision.get("selected_arm_id") or checkpoint.get("preferred_arm_id")
                if recovered_arm_id:
                    self._preferred_arm_id = str(recovered_arm_id)
            replay_snapshot = checkpoint.get("rollback_replay_config_snapshot")
            update = checkpoint.get("applied_params_update")
            final_state_path = checkpoint.get("final_parameter_state_path")
            final_state = None
            if final_state_path and Path(str(final_state_path)).exists():
                final_state = self._load_json_path(final_state_path)
                expected_digest = str(checkpoint.get("final_parameter_state_digest") or "")
                if expected_digest and stable_hash(final_state) != expected_digest:
                    raise RuntimeError(f"final parameter state digest mismatch for round_{round_id:02d}")
            if checkpoint.get("rollback_exact_replay") and isinstance(replay_snapshot, dict) and replay_snapshot:
                self._restore_exact_config_snapshot(replay_snapshot)
            elif isinstance(update, dict):
                restored_update = dict(update)
                if isinstance(final_state, dict):
                    restored_update.update(final_state)
                self._apply_next_round_update(restored_update)
            raw_next_jobs = checkpoint.get("next_jobs")
            if isinstance(raw_next_jobs, list) and raw_next_jobs:
                current_jobs = self._jobs_from_dicts(raw_next_jobs)
            else:
                next_jobs_path = checkpoint.get("next_jobs_path")
                loaded_jobs: List[DesignJob] = []
                if next_jobs_path and Path(next_jobs_path).exists():
                    try:
                        loaded_jobs = self._jobs_from_dicts(json.loads(Path(next_jobs_path).read_text(encoding="utf-8")))
                    except Exception:
                        loaded_jobs = []
                if loaded_jobs:
                    current_jobs = loaded_jobs
                elif str(checkpoint.get("rollback_action") or "") == "stop":
                    # Early-stop rounds intentionally have no continuation.
                    current_jobs = []
                else:
                    # Legacy last-round artifacts wrote empty next_jobs. Rebuild a
                    # continuation seed once and atomically backfill the checkpoint.
                    current_jobs = self._rebuild_legacy_continuation_jobs(round_id, checkpoint)
            next_round = round_id + 1
        # Scan beyond max_rounds only to detect discontinuous completed history that
        # would be skipped by a lowered max_rounds; do not recover those rounds.
        return next_round, current_jobs, recovered_rounds

    def _rebuild_legacy_continuation_jobs(self, round_id: int, checkpoint: Mapping[str, Any]) -> List[DesignJob]:
        """Rebuild next_jobs for legacy final rounds that stored an empty list."""
        round_dir = self.out_dir / f"round_{round_id:02d}"
        parent_jobs: List[DesignJob] = []
        raw_current = checkpoint.get("current_jobs")
        if isinstance(raw_current, list) and raw_current:
            parent_jobs = self._jobs_from_dicts(raw_current)
        if not parent_jobs:
            raise RuntimeError(
                f"Cannot resume past round_{round_id:02d}: legacy checkpoint has empty next_jobs "
                "and no current_jobs to rebuild a continuation seed. Re-run with a fresh --out "
                "or restore a complete round_checkpoint.json."
            )
        summary_round = self._load_checkpoint_round_summary(checkpoint, round_dir, round_id) or {}
        evaluation = summary_round.get("evaluation") if isinstance(summary_round.get("evaluation"), dict) else {}
        top_candidates = evaluation.get("top_candidates") or []
        struct_eval = summary_round.get("structural_analysis")
        hypotheses = (summary_round.get("hypotheses") or {}).get("hypotheses") or []
        quality_analysis = summary_round.get("quality_analysis") or {}
        proposal = summary_round.get("proposal") if isinstance(summary_round.get("proposal"), dict) else {}
        policy_update = proposal.get("params_update") if isinstance(proposal.get("params_update"), dict) else {}
        applied = checkpoint.get("applied_params_update") if isinstance(checkpoint.get("applied_params_update"), dict) else policy_update
        proposed_jobs = self.learner.propose_next(
            round_id + 1,
            parent_jobs,
            top_candidates,
            str(self.out_dir),
            top_k=self.cfg.active_learning.top_k,
            policy_update=applied,
            structural_summary=struct_eval,
            hypotheses=hypotheses,
            quality_analysis=quality_analysis,
            branch_width=self.cfg.active_learning.branch_width,
            enable_exploitation_arms=bool(getattr(self.cfg.active_learning, "enable_exploitation_arms", False)),
        ).jobs
        proposed_jobs = self._materialize_job_binding_types(proposed_jobs)
        proposed_jobs = self._materialize_sampler_and_context_intents(proposed_jobs)
        proposed_jobs = self._resolve_job_pressure_conflicts(proposed_jobs)
        resume_memory = self.memory_store.load(target=asdict(self.cfg.target))
        proposed_jobs = self._govern_exploration_jobs(
            proposed_jobs, current_jobs=parent_jobs, next_round_id=round_id + 1, strict_positive_count=0,
            blocked_digests=self.memory_store.blocked_interventions(resume_memory, round_id + 1),
        )
        proposed_jobs = self._enforce_binder_length_range(proposed_jobs)
        proposed_jobs = self._enforce_round_cap(proposed_jobs, round_id=round_id + 1)
        if not proposed_jobs:
            raise RuntimeError(
                f"Cannot resume past round_{round_id:02d}: failed to rebuild non-empty continuation jobs "
                "from legacy empty next_jobs."
            )
        path = self._write_json(round_dir / "next_jobs.json", [asdict(job) for job in proposed_jobs])
        updated = dict(checkpoint)
        updated["next_jobs"] = [asdict(job) for job in proposed_jobs]
        updated["next_jobs_path"] = str(path)
        updated["continuation_prepared"] = True
        updated["continuation_rebuilt_from_legacy"] = True
        self._write_checkpoint(round_dir, round_id, "round_completed", "completed", updated)
        return proposed_jobs

    @staticmethod
    def _job_from_dict(item: Mapping[str, Any]) -> DesignJob:
        allowed = {field.name for field in fields(DesignJob)}
        payload = {key: value for key, value in dict(item or {}).items() if key in allowed}
        payload.setdefault("params", {})
        payload.setdefault("output_dir", "outputs/job")
        return DesignJob(**payload)

    @classmethod
    def _jobs_from_dicts(cls, items: Iterable[Mapping[str, Any]]) -> List[DesignJob]:
        return [cls._job_from_dict(item) for item in items]

    @staticmethod
    def _load_json_path(path: Union[str, Path]) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def _load_json_if_exists(cls, path: Union[str, Path], default: Any = None) -> Any:
        target = Path(path)
        if not target.exists():
            return {} if default is None else default
        return cls._load_json_path(target)

    @staticmethod
    def _candidate_from_dict(data: Mapping[str, Any]) -> CandidateEvaluation:
        return CandidateEvaluation(**dict(data))

    @classmethod
    def _evaluation_from_dict(cls, data: Mapping[str, Any]) -> EvaluationSummary:
        payload = dict(data)
        payload["top_candidates"] = [cls._candidate_from_dict(item) for item in payload.get("top_candidates", [])]
        payload["failed_examples"] = [cls._candidate_from_dict(item) for item in payload.get("failed_examples", [])]
        return EvaluationSummary(**payload)

    @staticmethod
    def _structure_batch_from_dict(data: Mapping[str, Any]) -> StructureBatchEvaluation:
        return StructureBatchEvaluation(**dict(data))

    @staticmethod
    def _fragment_batch_from_dict(data: Mapping[str, Any]) -> FragmentTemplateBatch:
        payload = dict(data)
        payload["templates"] = [FragmentTemplate(**dict(item)) for item in payload.get("templates", [])]
        return FragmentTemplateBatch(**payload)

    @staticmethod
    def _length_recommendation_from_dict(data: Mapping[str, Any]) -> BinderLengthRecommendation:
        return BinderLengthRecommendation(**dict(data))

    @staticmethod
    def _quality_analysis_from_dict(data: Mapping[str, Any]) -> BinderQualityAnalysis:
        allowed = {item.name for item in fields(BinderQualityAnalysis)}
        return BinderQualityAnalysis(**{key: value for key, value in dict(data or {}).items() if key in allowed})

    @staticmethod
    def _hypotheses_from_dict(data: Mapping[str, Any]) -> HypothesisSet:
        allowed = {item.name for item in fields(HypothesisSet)}
        return HypothesisSet(**{key: value for key, value in dict(data or {}).items() if key in allowed})

    @staticmethod
    def _diagnostic_from_dict(data: Mapping[str, Any]) -> DiagnosticReport:
        allowed = {item.name for item in fields(DiagnosticReport)}
        return DiagnosticReport(**{key: value for key, value in dict(data or {}).items() if key in allowed})

    @staticmethod
    def _input_config_from_dict(data: Mapping[str, Any]) -> InputConfiguration:
        allowed = {item.name for item in fields(InputConfiguration)}
        return InputConfiguration(**{key: value for key, value in dict(data).items() if key in allowed})

    @staticmethod
    def _conflict_resolution_from_dict(data: Mapping[str, Any]) -> StrategyConflictResolution:
        return StrategyConflictResolution(
            round_id=int(data.get("round_id") or 0),
            llm_used=bool(data.get("llm_used")),
            decisions=list(data.get("decisions") or []),
            params_update=dict(data.get("params_update") or {}),
            summary=str(data.get("summary") or ""),
            raw=dict(data.get("raw") or {}),
        )

    @staticmethod
    def _policy_proposal_from_dict(data: Mapping[str, Any]) -> NextRoundParameterProposal:
        if not isinstance(data, Mapping):
            raise TypeError("policy proposal must be an object")
        payload = dict(data)
        raw_update = payload.get("params_update", payload.get("config_updates", {}))
        if not isinstance(raw_update, Mapping):
            raise TypeError("policy proposal params_update must be an object")
        invalid = invalid_config_value_keys(raw_update)
        if invalid:
            raise ValueError(f"invalid policy config values: {', '.join(invalid)}")
        try:
            round_id = int(payload.get("round_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("policy proposal round_id must be an integer") from exc
        rationale = payload.get("rationale", [])
        if isinstance(rationale, str):
            rationale = [rationale]
        elif not isinstance(rationale, (list, tuple)):
            raise TypeError("policy proposal rationale must be a string or list")
        metadata = payload.get("analysis_metadata", payload.get("metadata", {}))
        if not isinstance(metadata, Mapping):
            raise TypeError("policy proposal metadata must be an object")
        return NextRoundParameterProposal(
            round_id=round_id,
            params_update=supported_config_changes(raw_update),
            rationale=[str(item) for item in rationale],
            analysis_metadata=dict(metadata),
            final_params_update=dict(payload.get("final_params_update") or {}),
            applied_params_update=dict(payload.get("applied_params_update") or {}),
        )

    def _retry_jobs_after_execution_failure(
        self,
        current_jobs: List[DesignJob],
        execution_records: List[Dict[str, Any]],
        *,
        next_round_id: int,
    ) -> List[DesignJob]:
        """Continue a multi-arm round without collapsing successful siblings.

        Successful arms are repeated unchanged, failed arms apply typed corrections,
        and circuit-broken arms are replaced by distinct fresh arms. The resulting
        continuation always restores ``branch_width`` before budget allocation.
        """
        ordered_records = self._records_for_jobs(
            current_jobs, execution_records or [], label="execution_retry_record", allow_legacy_order=True,
        )
        width = max(1, int(self.cfg.active_learning.branch_width))
        continuations: List[DesignJob] = []
        blocked_arms: set[str] = set()
        retained_arms: set[str] = set()
        total = max(1, len(current_jobs))
        arm_statuses: Dict[str, List[bool]] = {}
        for job, record in zip(current_jobs, ordered_records):
            params = dict(job.params or {})
            arm_id = str(params.get("arm_id") or params.get("exploration_arm") or job.job_id)
            arm_statuses.setdefault(arm_id, []).append(str(record.get("status") or "").lower() in self.FAILURE_STATUSES)
        for index, (job, record) in enumerate(zip(current_jobs, ordered_records)):
            failed = str(record.get("status") or "").lower() in self.FAILURE_STATUSES
            params = copy.deepcopy(dict(job.params or {}))
            arm_id = str(params.get("arm_id") or params.get("exploration_arm") or job.job_id)
            # Split-host shards are execution slots of one logical arm. If only
            # some shards failed, retry those failed slots and retain the arm once.
            if not failed and any(arm_statuses.get(arm_id, [])):
                continue
            semantic_fingerprint = str(record.get("semantic_failure_fingerprint") or self._semantic_failure_fingerprint(record))
            semantic_scope = self._semantic_failure_scope(job)
            circuit_broken = failed and self._repeated_cross_round_semantic_failure(
                semantic_fingerprint, semantic_scope, before_round_id=next_round_id - 2,
            )
            if circuit_broken:
                record["semantic_retry_circuit_breaker"] = "cross_round_identical_semantic_failure"
                record["retry_circuit_breaker"] = record.get("retry_circuit_breaker") or "cross_round_identical_semantic_failure"
                if arm_id:
                    blocked_arms.add(arm_id)
                continue
            proposal = record.get("retry_correction_proposal")
            if failed and isinstance(proposal, Mapping) and bool(proposal.get("requires_refinalization")):
                params, patch_audit = self._apply_retry_correction_patch(params, proposal)
                if patch_audit:
                    retry_metadata = dict(params.get("retry_metadata") or {})
                    retry_metadata["correction_patch"] = patch_audit
                    params["retry_metadata"] = retry_metadata
            prior_values = {
                "job_id": job.job_id, "output_dir": job.output_dir,
                "branch_id": copy.deepcopy((job.params or {}).get("branch_id")),
                "logical_branch_id": copy.deepcopy((job.params or {}).get("logical_branch_id")),
                "logical_job_id": copy.deepcopy((job.params or {}).get("logical_job_id")),
                "execution_job_id": copy.deepcopy((job.params or {}).get("execution_job_id")),
                "job_identity": copy.deepcopy((job.params or {}).get("job_identity")),
            }
            params["continuation_kind"] = "failed_arm_retry" if failed else "successful_arm_retest"
            params["lineage"] = {
                "source_job_id": job.job_id,
                "source_round_id": next_round_id - 1,
                "source_arm_id": arm_id,
                "kind": params["continuation_kind"],
            }
            params["execution_retry_source_job_id"] = job.job_id
            params["execution_retry_preserve_budget"] = True
            retry_metadata = dict(params.get("retry_metadata") or {})
            retry_metadata["refinalization"] = {"required": True, "prior_values": prior_values}
            params["retry_metadata"] = retry_metadata
            payload = asdict(job)
            suffix = "round" if total == 1 else f"arm_{index:02d}_{arm_id or index}"
            payload.update({"job_id": f"r{next_round_id}_{suffix}", "params": params, "output_dir": f"{self.out_dir}/r{next_round_id}/arms/{suffix}"})
            continuations.append(self._job_from_dict(payload))
            if arm_id:
                retained_arms.add(arm_id)

        missing = width - len(continuations)
        if missing > 0:
            fresh_blocked = blocked_arms | retained_arms
            fresh = self.learner.propose_next(
                next_round_id, self._logical_jobs_for_memory(current_jobs), [], str(self.out_dir),
                top_k=self.cfg.active_learning.top_k, policy_update={}, blocked_arms=fresh_blocked,
                branch_width=missing,
                enable_exploitation_arms=bool(getattr(self.cfg.active_learning, "enable_exploitation_arms", False)),
            ).jobs
            for job in fresh:
                arm_id = str((job.params or {}).get("arm_id") or (job.params or {}).get("exploration_arm") or "")
                job.params["continuation_kind"] = "fresh_complementary_arm"
                job.params["lineage"] = {
                    "source_job_ids": [item.job_id for item in current_jobs],
                    "source_round_id": next_round_id - 1,
                    "replaced_blocked_arm_ids": sorted(blocked_arms),
                    "source_arm_id": arm_id,
                    "kind": "fresh_complementary_arm",
                }
            continuations.extend(fresh)
        if len(continuations) != width:
            raise ValueError(f"execution continuation branch_width={width} requires {width} jobs; got {len(continuations)}")
        finalized = self._finalize_semantic_job_identities(continuations, round_id=next_round_id)
        for clone in finalized:
            retry_metadata = dict(clone.params.get("retry_metadata") or {})
            refinalization = dict(retry_metadata.get("refinalization") or {})
            if refinalization:
                refinalization["final_values"] = {
                    "job_id": clone.job_id, "output_dir": clone.output_dir,
                    "branch_id": copy.deepcopy(clone.params.get("branch_id")),
                    "logical_branch_id": copy.deepcopy(clone.params.get("logical_branch_id")),
                    "logical_job_id": copy.deepcopy(clone.params.get("logical_job_id")),
                    "execution_job_id": copy.deepcopy(clone.params.get("execution_job_id")),
                    "job_identity": copy.deepcopy(clone.params.get("job_identity")),
                }
                retry_metadata["refinalization"] = refinalization
                clone.params["retry_metadata"] = retry_metadata
        return finalized

    @staticmethod
    def _set_patch_path(params: Dict[str, Any], raw_path: Any, value: Any) -> None:
        parts = [part for part in str(raw_path).split(".") if part]
        if not parts:
            return
        target = params
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = copy.deepcopy(value)

    @staticmethod
    def _remove_patch_path(params: Dict[str, Any], raw_path: Any) -> None:
        parts = [part for part in str(raw_path).split(".") if part]
        if not parts:
            return
        target: Any = params
        for part in parts[:-1]:
            if not isinstance(target, Mapping) or not isinstance(target.get(part), Mapping):
                return
            target = target[part]
        if isinstance(target, dict):
            target.pop(parts[-1], None)

    @staticmethod
    def _retry_contract_metadata(key: Any) -> Dict[str, Any]:
        """Read partition and retry policy from the contract's public API."""
        name = str(key)
        entry = dict(parameter_contract_entry(name) or {})
        partition = str(entry.get("partition") or "")
        if not partition:
            try:
                partitioned = partition_config_parameters({name: None})
            except (TypeError, ValueError):
                partitioned = {}
            for candidate, values in dict(partitioned or {}).items():
                if isinstance(values, Mapping) and name in values:
                    partition = str(candidate)
                    break
        entry["partition"] = partition or "unknown"
        return entry

    @classmethod
    def _retry_orchestrator_owned_path(cls, raw_path: Any) -> bool:
        parts = [part for part in str(raw_path).split(".") if part]
        if not parts:
            return False
        contract = cls._retry_contract_metadata(parts[0])
        retry_policy = contract.get("retry_policy")
        if isinstance(retry_policy, Mapping):
            retry_policy = retry_policy.get("correction") or retry_policy.get("mode") or retry_policy.get("action")
        policy = str(retry_policy or "").strip().lower()
        explicitly_preserved = policy in {
            "preserve", "preserved", "immutable", "orchestrator_owned",
            "ignore_correction", "deny", "fail_closed",
        }
        return explicitly_preserved or contract.get("partition") == "orchestration"

    @classmethod
    def _apply_retry_correction_patch(
        cls, original_params: Mapping[str, Any], proposal: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply executable corrections without allowing metadata ownership transfer."""
        params = copy.deepcopy(dict(original_params or {}))
        patch = proposal.get("typed_patch") or proposal.get("correction_patch") or proposal.get("patch")
        if not isinstance(patch, Mapping) and any(key in proposal for key in ("set", "remove", "classification", "identity_effect", "source_validation_digest")):
            patch = proposal
        if isinstance(patch, Mapping):
            raw_version = patch.get("version", patch.get("schema_version", 1))
            try:
                version = int(raw_version)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid_retry_correction_patch_version:{raw_version!r}") from exc
            if version != 1:
                raise ValueError(f"unsupported_retry_correction_patch_version:{version}")
            set_values = patch.get("set") or {}
            remove_values = patch.get("remove") or []
            if isinstance(remove_values, str):
                remove_values = [remove_values]
            ignored: List[Dict[str, Any]] = []
            applied_set: Dict[str, Any] = {}
            applied_remove: List[str] = []
            if isinstance(set_values, Mapping):
                for path, value in set_values.items():
                    path_text = str(path)
                    if cls._retry_orchestrator_owned_path(path_text):
                        ignored.append({"operation": "set", "path": path_text, "reason": "orchestrator_owned_metadata"})
                        continue
                    cls._set_patch_path(params, path_text, value)
                    applied_set[path_text] = copy.deepcopy(value)
            if isinstance(remove_values, (list, tuple, set)):
                for path in remove_values:
                    path_text = str(path)
                    if cls._retry_orchestrator_owned_path(path_text):
                        ignored.append({"operation": "remove", "path": path_text, "reason": "orchestrator_owned_metadata"})
                        continue
                    cls._remove_patch_path(params, path_text)
                    applied_remove.append(path_text)
            return params, {
                "version": version,
                "set": applied_set,
                "remove": applied_remove,
                "classification": patch.get("classification"),
                "identity_effect": patch.get("identity_effect"),
                "source_validation_digest": patch.get("source_validation_digest"),
                "ignored_orchestrator_metadata_overrides": ignored,
            }

        corrected = proposal.get("corrected_params")
        if not isinstance(corrected, Mapping):
            return params, {}
        corrected_params = copy.deepcopy(dict(corrected))
        original_partitions = partition_config_parameters(params)
        corrected_partitions = partition_config_parameters(corrected_params)
        execution_partitions = ("runner", "adapter", "runtime")
        merged = copy.deepcopy(params)
        for partition in execution_partitions:
            original_execution = dict(original_partitions.get(partition) or {})
            corrected_execution = dict(corrected_partitions.get(partition) or {})
            for key in original_execution:
                merged.pop(key, None)
            merged.update(copy.deepcopy(corrected_execution))
        attempted_metadata = []
        for partition, values in dict(corrected_partitions or {}).items():
            if partition in execution_partitions:
                continue
            for key, value in dict(values or {}).items():
                if key not in params or params.get(key) != value:
                    attempted_metadata.append({
                        "operation": "set",
                        "path": str(key),
                        "partition": str(partition),
                        "reason": "legacy_corrected_params_execution_only",
                    })
        return merged, {
            "version": 0,
            "legacy_corrected_params": True,
            "replaced_partitions": list(execution_partitions),
            "preserved_partitions": [
                str(name) for name in dict(original_partitions or {}) if name not in execution_partitions
            ],
            "ignored_metadata_overrides": attempted_metadata,
        }

    @staticmethod
    def _semantic_failure_scope(job: DesignJob) -> str:
        params = dict(job.params or {})
        shard = dict(params.get("multi_taiji_host_shard") or {})
        return stable_hash({
            "arm": str(params.get("arm_id") or params.get("exploration_arm") or ""),
            "execution_partition": {
                "kind": "taiji_host_shard" if shard else "logical_job",
                "shard_index": shard.get("shard_index"),
            },
        })

    def _repeated_cross_round_semantic_failure(
        self, fingerprint: str, scope: str, *, before_round_id: int,
    ) -> bool:
        """Return true after the immediately preceding round failed equivalently."""
        if not fingerprint or before_round_id < 0:
            return False
        path = self.out_dir / f"round_{before_round_id:02d}" / "execution_records.json"
        if not path.exists():
            return False
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        for prior in records if isinstance(records, list) else []:
            if str(prior.get("status") or "").lower() not in self.FAILURE_STATUSES:
                continue
            prior_fingerprint = str(prior.get("semantic_failure_fingerprint") or self._semantic_failure_fingerprint(prior))
            prior_job = prior.get("job") if isinstance(prior.get("job"), Mapping) else {}
            try:
                prior_scope = str(prior.get("semantic_failure_scope") or self._semantic_failure_scope(self._job_from_dict(prior_job)))
            except (TypeError, ValueError):
                prior_scope = ""
            if prior_fingerprint == fingerprint and prior_scope == scope:
                return True
        return False

    def _run_jobs(
        self,
        jobs: List[DesignJob],
        round_id: int,
        execute_job: Optional[Callable[[DesignJob, int], Dict[str, Any]]],
        *,
        attempts_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        self._validate_job_identities(jobs)
        unfinalized = [
            job.job_id for job in jobs
            if not bool(((job.params or {}).get("job_identity") or {}).get("finalized"))
        ]
        if unfinalized:
            raise ValueError("execution_job_identity_not_finalized:" + ",".join(unfinalized))
        attempts_path = attempts_path or self.out_dir / f"round_{round_id:02d}" / "execution_attempts.json"
        attempts_lock = threading.Lock()
        attempts_ledger = self._load_attempt_ledger(attempts_path, round_id)
        incomplete_attempts = self._incomplete_attempts(attempts_ledger)
        if incomplete_attempts:
            self._reconcile_incomplete_attempts(attempts_path, attempts_ledger, attempts_lock)
            incomplete_attempts = self._incomplete_attempts(attempts_ledger)
        if incomplete_attempts:
            blocked = ", ".join(
                f"{item['job_id']}#attempt{item['attempt']}" for item in incomplete_attempts[:10]
            )
            raise RuntimeError(
                "Refusing to submit new jobs because previous attempts are still marked "
                f"started in {attempts_path}: {blocked}. Inspect/cancel the corresponding "
                "remote Taiji tasks or archive/remove the output directory only after "
                "confirming it is safe to start a fresh run."
            )

        def run_one(job: DesignJob) -> Dict[str, Any]:
            if execute_job is None:
                if str(self.cfg.resource.backend or "").lower() == "dry_run":
                    result = {"job_id": job.job_id, "status": "dry_run", "output_dir": job.output_dir}
                    self.bus.publish(AgentMessage("Orchestrator", "RunMonitorAgent", "status", result, round_id=round_id, job_id=job.job_id))
                    return {"job": asdict(job), "job_identity_digest": self._job_identity_digest(job), "attempts": 0, **result}
                raise RuntimeError("No job executor configured for non-dry-run orchestrator run")
            last_error = None
            final_record: Optional[Dict[str, Any]] = None
            while True:
                existing_record = self._terminal_attempt_record(attempts_ledger, attempts_lock, job)
                if existing_record is not None:
                    self.bus.publish(AgentMessage("Orchestrator", "RunMonitorAgent", "status", existing_record, round_id=round_id, job_id=job.job_id))
                    return existing_record

                prior_attempts = self._attempt_count(attempts_ledger, attempts_lock, job.job_id)
                if prior_attempts >= self.max_retries:
                    result = {
                        "job": asdict(job),
                        "attempts": prior_attempts,
                        "status": "failed",
                        "error": f"retry limit reached before submit: {prior_attempts}/{self.max_retries} attempts already recorded",
                    }
                    self._record_terminal_attempt(attempts_path, attempts_ledger, attempts_lock, job, result)
                    self.bus.publish(AgentMessage("Orchestrator", "RunMonitorAgent", "status", result, round_id=round_id, job_id=job.job_id))
                    return result

                attempt = prior_attempts + 1
                self._record_attempt_started(attempts_path, attempts_ledger, attempts_lock, job, attempt)
                try:
                    # The ledger-bound job is immutable after attempt start. A
                    # defensive copy also contains accidental third-party executor
                    # mutation; semantic corrections must be returned as proposals.
                    result = execute_job(copy.deepcopy(job), attempt)
                    if str(result.get("status") or "").lower() not in self.FAILURE_STATUSES:
                        final_record = {"job": asdict(job), "job_identity_digest": self._job_identity_digest(job), "attempts": attempt, **result}
                        self._record_attempt_finished(attempts_path, attempts_ledger, attempts_lock, job, attempt, final_record)
                        self._record_terminal_attempt(attempts_path, attempts_ledger, attempts_lock, job, final_record)
                        self.bus.publish(AgentMessage("Orchestrator", "RunMonitorAgent", "status", final_record, round_id=round_id, job_id=job.job_id))
                        return final_record
                    last_error = str(result.get("error") or result.get("status"))
                    final_record = {"job": asdict(job), "job_identity_digest": self._job_identity_digest(job), "attempts": attempt, **result}
                    if self._is_resource_scheduling_failure(final_record):
                        proposal = self._resource_retry_correction_proposal(job)
                        final_record["failure_class"] = "resource_scheduling_failure"
                        if proposal:
                            final_record["resource_retry_degradation"] = {
                                "reason": proposal["reason"],
                                "before": proposal["before"],
                                "after": proposal["corrected_params"],
                                "changes": proposal["changes"],
                            }
                            final_record["retry_correction_proposal"] = proposal
                            final_record["retryable"] = False
                    final_record["failure_fingerprint"] = self._failure_fingerprint(final_record)
                    final_record["semantic_failure_fingerprint_version"] = 1
                    final_record["semantic_failure_fingerprint"] = self._semantic_failure_fingerprint(final_record)
                    final_record["semantic_failure_scope"] = self._semantic_failure_scope(job)
                    identical = self._identical_failure_fingerprint(attempts_ledger, attempts_lock, job.job_id, final_record["failure_fingerprint"])
                    if identical:
                        final_record["retryable"] = False
                        final_record["retry_circuit_breaker"] = "identical_failure_fingerprint"
                    self._record_attempt_finished(attempts_path, attempts_ledger, attempts_lock, job, attempt, final_record)
                    if not self._execution_failure_retryable(final_record):
                        self._record_terminal_attempt(attempts_path, attempts_ledger, attempts_lock, job, final_record)
                        self.bus.publish(AgentMessage("Orchestrator", "RunMonitorAgent", "status", final_record, round_id=round_id, job_id=job.job_id))
                        return final_record
                except Exception as exc:
                    last_error = str(exc)
                    final_record = {"job": asdict(job), "job_identity_digest": self._job_identity_digest(job), "attempts": attempt, "status": "failed", "error": last_error}
                    final_record["failure_fingerprint"] = self._failure_fingerprint(final_record)
                    final_record["semantic_failure_fingerprint_version"] = 1
                    final_record["semantic_failure_fingerprint"] = self._semantic_failure_fingerprint(final_record)
                    final_record["semantic_failure_scope"] = self._semantic_failure_scope(job)
                    if self._identical_failure_fingerprint(attempts_ledger, attempts_lock, job.job_id, final_record["failure_fingerprint"]):
                        final_record["retryable"] = False
                        final_record["retry_circuit_breaker"] = "identical_failure_fingerprint"
                    self._record_attempt_finished(attempts_path, attempts_ledger, attempts_lock, job, attempt, final_record)

                if attempt >= self.max_retries:
                    # Preserve the executor/exception failure record verbatim at the
                    # retry cap. Reconstructing a minimal record here discards the
                    # validation, runtime, scope, and fingerprint context computed
                    # immediately above and makes the returned record disagree with
                    # the attempt ledger.
                    capped_record = dict(final_record or {})
                    capped_record["attempts"] = attempt
                    capped_record.setdefault("job", asdict(job))
                    capped_record.setdefault("job_identity_digest", self._job_identity_digest(job))
                    capped_record.setdefault("status", "failed")
                    capped_record.setdefault("error", last_error)
                    self._record_terminal_attempt(attempts_path, attempts_ledger, attempts_lock, job, capped_record)
                    self.bus.publish(AgentMessage("Orchestrator", "RunMonitorAgent", "status", capped_record, round_id=round_id, job_id=job.job_id))
                    return capped_record
                self.bus.publish(AgentMessage("Orchestrator", "all", "retry", {"attempt": attempt, "next_attempt": attempt + 1, "max_attempts": self.max_retries, "error": last_error, "next_params": {key: job.params.get(key) for key in self.RESOURCE_CONFIG_KEYS if key in job.params}}, round_id=round_id, job_id=job.job_id))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, self.max_parallel)) as pool:
            return list(pool.map(run_one, jobs))

    @staticmethod
    def _execution_transport_binding(record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        existing = record.get("transport_binding") or (record.get("result_sync") or {}).get("transport_binding")
        if isinstance(existing, Mapping):
            return dict(existing)
        sync = record.get("result_sync")
        if not isinstance(sync, Mapping) or str(sync.get("mode") or "").lower() != "symlink":
            return None
        required_record = ("job_id", "attempt", "task_flag", "attempt_root", "local_output_dir", "remote_output_dir")
        if any(record.get(key) in (None, "") for key in required_record):
            return None
        local_package_value = sync.get("local_package_dir") or record.get("local_package_dir")
        remote_package_value = sync.get("remote_package_dir") or record.get("remote_package_dir")
        if not local_package_value or not remote_package_value:
            return None
        local_package = Path(str(local_package_value))
        remote_package = Path(str(remote_package_value))
        if record.get("local_package_dir") and Path(str(record["local_package_dir"])) != local_package:
            return None
        if record.get("remote_package_dir") and Path(str(record["remote_package_dir"])) != remote_package:
            return None
        local_output = Path(str(record["local_output_dir"]))
        remote_output = Path(str(record["remote_output_dir"]))
        attempt_root = Path(str(record["attempt_root"]))
        task_flag = str(record["task_flag"])
        lexical = lambda value: Path(os.path.abspath(os.path.normpath(str(value))))
        if lexical(local_output) != lexical(local_package / "outputs" / "boltzgen_output"):
            return None
        if lexical(remote_output) != lexical(remote_package / "outputs" / "boltzgen_output"):
            return None
        if not is_project_package_name(remote_package.name) or remote_package.parent.name != task_flag:
            return None
        try:
            lexical(local_package).relative_to(lexical(attempt_root))
        except ValueError:
            return None
        linked = sync.get("linked")
        if not isinstance(linked, list):
            return None
        mappings: Dict[str, Dict[str, str]] = {}
        for name in ("outputs", "logs"):
            source = remote_package / name
            target = local_package / name
            expected_link = os.path.relpath(str(source.absolute()), str(target.parent.absolute()))
            matches = [item for item in linked if isinstance(item, Mapping)
                       and lexical(Path(str(item.get("source") or ""))) == lexical(source)
                       and lexical(Path(str(item.get("target") or ""))) == lexical(target)
                       and str(item.get("link") or "") == expected_link
                       and not Path(str(item.get("link") or "")).is_absolute()]
            if len(matches) != 1:
                return None
            mappings[name] = {"link": expected_link}
        return {
            "schema_version": 1, "mode": "symlink",
            "local_package_dir": str(local_package),
            "local_output_alias": str(local_package / "outputs" / "boltzgen_output"),
            "local_logs_alias": str(local_package / "logs"),
            "remote_package_dir": str(remote_package),
            "remote_output_root": str(remote_package / "outputs" / "boltzgen_output"),
            "remote_logs_root": str(remote_package / "logs"),
            "link_text": mappings["outputs"]["link"],
            "logs_link_text": mappings["logs"]["link"],
            "job_id": str(record["job_id"]), "attempt": record["attempt"],
            "task_flag": task_flag, "attempt_root": str(attempt_root),
            "provenance": "legacy_result_sync_v1_migration",
        }

    def _ingest_execution_outputs(self, jobs: List[DesignJob], execution_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ordered_records = self._records_for_jobs(
            jobs, execution_records, label="ingestion_execution_record", allow_legacy_order=True,
        )
        ingestions: List[Dict[str, Any]] = []
        for job, record in zip(jobs, ordered_records):
            status = str(record.get("status") or "").lower()
            explicit_output_dir = record.get("local_output_dir") or record.get("output_dir")
            log_file = record.get("local_log_file") or record.get("log_file")
            if status in self.FAILURE_STATUSES:
                output_dir = explicit_output_dir or job.output_dir
                issues = ["execution_failed_output_not_ingested"]
                if not explicit_output_dir:
                    issues.append("execution_failed_no_explicit_output_dir")
                if record.get("error"):
                    issues.append(str(record.get("error")))
                failure_hints = list(record.get("failure_hints") or [])
                monitor = record.get("monitor") if isinstance(record.get("monitor"), Mapping) else {}
                failure_hints.extend(monitor.get("failure_hints") or [])
                issues.extend(str(hint) for hint in failure_hints if hint)
                issues = list(dict.fromkeys(issues))
                params = dict(job.params or {})
                ingestions.append(asdict(IngestedBoltzGenRun(
                    output_dir=str(output_dir),
                    job_id=job.job_id,
                    arm_id=str(params.get("arm_id") or ""),
                    exploration_arm=str(params.get("exploration_arm") or ""),
                    logical_branch_id=str(params.get("logical_branch_id") or params.get("branch_id") or ""),
                    execution_job_id=str(params.get("execution_job_id") or job.job_id),
                    execution_slot=params.get("execution_slot"),
                    arm_root=str(params.get("arm_root") or job.output_dir),
                    log_file=str(log_file) if log_file else None,
                    run_level_issues=issues,
                )))
                continue
            output_dir = explicit_output_dir or job.output_dir
            params = dict(job.params or {})
            shard = dict(params.get("multi_taiji_host_shard") or {})
            transport_binding = self._execution_transport_binding(record)
            ingestions.append(asdict(self._search_profile().ingest(
                output_dir, log_file=log_file, identity_context={
                    "job_id": job.job_id,
                    "execution_job_id": params.get("execution_job_id") or job.job_id,
                    "execution_slot": params.get("execution_slot"),
                    "host_shard": shard.get("shard_id"),
                    "arm_id": params.get("arm_id") or params.get("exploration_arm"),
                    "exploration_arm": params.get("exploration_arm"),
                    "logical_branch_id": params.get("logical_branch_id") or params.get("branch_id"),
                    "arm_root": params.get("arm_root") or job.output_dir,
                    "output_root": str(output_dir),
                    "attempt": record.get("attempt"),
                    "attempt_root": record.get("attempt_root"),
                    "task_flag": record.get("task_flag"),
                    "transport_binding": transport_binding,
                },
                ingestor=self.ingestor,
            )))
        return ingestions

    def _load_attempt_ledger(self, path: Path, round_id: int) -> Dict[str, Any]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("jobs", {})
                    return data
            except Exception:
                pass
        return {"round_id": round_id, "max_attempts_per_job": self.max_retries, "jobs": {}}

    @staticmethod
    def _incomplete_attempts(ledger: Mapping[str, Any]) -> List[Dict[str, Any]]:
        incomplete: List[Dict[str, Any]] = []
        for job_id, job_entry in (ledger.get("jobs") or {}).items():
            if not isinstance(job_entry, Mapping) or job_entry.get("terminal_record"):
                continue
            attempts = job_entry.get("attempts") or []
            if not attempts:
                continue
            latest = attempts[-1]
            if str(latest.get("status") or "").lower() == "started":
                incomplete.append({"job_id": job_id, "attempt": latest.get("attempt")})
        return incomplete

    def _reconcile_incomplete_attempts(self, path: Path, ledger: Dict[str, Any], lock: threading.Lock) -> None:
        for job_id, job_entry in list((ledger.get("jobs") or {}).items()):
            if not isinstance(job_entry, Mapping) or job_entry.get("terminal_record"):
                continue
            attempts = list(job_entry.get("attempts") or [])
            if not attempts or str(attempts[-1].get("status") or "").lower() != "started":
                continue
            job_payload = dict(job_entry.get("job") or {})
            if not job_payload:
                continue
            try:
                job = self._job_from_dict(job_payload)
            except TypeError:
                continue
            attempt = int(attempts[-1].get("attempt") or len(attempts))
            record = self._reconciled_execution_record(job, attempt)
            if record is None:
                continue
            self._record_attempt_finished(path, ledger, lock, job, attempt, record)
            status = str(record.get("status") or "").lower()
            if status not in self.FAILURE_STATUSES or not self._execution_failure_retryable(record) or attempt >= self.max_retries:
                self._record_terminal_attempt(path, ledger, lock, job, record)

    def _reconciled_execution_record(self, job: DesignJob, attempt: int) -> Optional[Dict[str, Any]]:
        candidates = [
            Path(job.output_dir) / "attempts" / f"attempt_{int(attempt):02d}" / "execution_record.json",
            Path(job.output_dir) / "execution_record.json",
        ]
        for execution_record in candidates:
            if not execution_record.exists():
                continue
            try:
                record = json.loads(execution_record.read_text(encoding="utf-8"))
            except Exception:
                record = None
            if isinstance(record, dict) and self._record_matches_attempt(record, job, attempt):
                status = str(record.get("status") or "").lower()
                if status and status not in {"started", "submitted", "running", "pending", "queued"}:
                    record.setdefault("artifact_locators", {"attempt_root": str(execution_record.parent), "execution_record": str(execution_record)})
                    return {"job": asdict(job), "attempts": attempt, **record}
        snapshot_record = self._reconciled_taiji_snapshot(job, attempt)
        if snapshot_record is not None:
            return snapshot_record
        return None

    @staticmethod
    def _record_matches_attempt(record: Mapping[str, Any], job: DesignJob, attempt: int) -> bool:
        record_job_id = str(record.get("job_id") or (record.get("job") or {}).get("job_id") or "")
        if record_job_id and record_job_id != job.job_id:
            return False
        try:
            record_attempt = int(record.get("attempt") or attempt)
        except (TypeError, ValueError):
            record_attempt = attempt
        return record_attempt == int(attempt)

    def _reconciled_taiji_snapshot(self, job: DesignJob, attempt: int) -> Optional[Dict[str, Any]]:
        attempt_root = Path(job.output_dir) / "attempts" / f"attempt_{int(attempt):02d}"
        package_candidates = package_dir_candidates(attempt_root) + package_dir_candidates(Path(job.output_dir))
        package_dir = next((candidate for candidate in package_candidates if (candidate / "taiji_monitor_snapshot.json").exists()), package_candidates[0])
        snapshot_path = package_dir / "taiji_monitor_snapshot.json"
        if not snapshot_path.exists():
            return None
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(snapshot, Mapping) or not bool(snapshot.get("is_terminal")):
            return None
        local_output_dir = package_dir / "outputs" / "boltzgen_output"
        local_log_file = package_dir / "logs" / "boltzgen_full.log"
        status = "completed" if snapshot.get("is_success") else "failed"
        record: Dict[str, Any] = {
            "job": asdict(job),
            "job_id": job.job_id,
            "backend": "taiji",
            "attempt": attempt,
            "attempts": attempt,
            "status": status,
            "output_dir": str(local_output_dir),
            "local_output_dir": str(local_output_dir),
            "log_file": str(local_log_file),
            "local_log_file": str(local_log_file),
            "monitor": dict(snapshot),
            "taiji_job_id": snapshot.get("instance_id"),
        }
        if status != "completed":
            hints = snapshot.get("failure_hints") or []
            record["error"] = ";".join(str(item) for item in hints) or str(snapshot.get("state") or "taiji terminal failure")
            record["retryable"] = True
        return record

    @staticmethod
    def _attempt_count(ledger: Mapping[str, Any], lock: threading.Lock, job_id: str) -> int:
        with lock:
            job_entry = (ledger.get("jobs") or {}).get(job_id) or {}
            return len(job_entry.get("attempts") or [])

    def _terminal_attempt_record(self, ledger: Mapping[str, Any], lock: threading.Lock, job: DesignJob) -> Optional[Dict[str, Any]]:
        with lock:
            job_entry = (ledger.get("jobs") or {}).get(job.job_id) or {}
            terminal_record = job_entry.get("terminal_record")
            if isinstance(terminal_record, dict):
                record = dict(terminal_record)
                record.setdefault("job", asdict(job))
                record.setdefault("attempts", len(job_entry.get("attempts") or []))
                if self._terminal_record_should_resubmit(record, job_entry):
                    return None
                return record
        return None

    def _terminal_record_should_resubmit(self, record: Mapping[str, Any], job_entry: Mapping[str, Any]) -> bool:
        status = str(record.get("status") or "").lower()
        if status not in self.FAILURE_STATUSES:
            return False
        attempts = len(job_entry.get("attempts") or [])
        if attempts >= self.max_retries:
            return False
        return self._execution_failure_retryable(record)

    @staticmethod
    def _execution_failure_retryable(record: Mapping[str, Any]) -> bool:
        proposal = record.get("retry_correction_proposal")
        if isinstance(proposal, Mapping) and bool(proposal.get("requires_refinalization")):
            # A changed semantic/resource payload cannot reuse this finalized ID,
            # output directory, or ledger entry. Carry it across rounds instead.
            return False
        # A resource/queue scheduling failure is worth a retry only when no
        # identity-changing degradation proposal can be made.
        # even if the executor marked the record non-retryable, because retrying
        # targets the infrastructure rather than the (valid) job config.
        if BinderDesignOrchestrator._is_resource_scheduling_failure(record):
            return True
        # An explicit ``retryable: False`` from the executor is authoritative. The
        # taiji executor already classifies submission/snapshot/pre-submit failures
        # (e.g. "pre-submit config validation failed", missing_input_file,
        # boltzgen_config_error) and only sets ``retryable: True`` when it produced
        # changed corrected params for the next attempt. Previously this method
        # re-promoted such failures back to retryable via a broad error-string
        # heuristic, which resubmitted genuinely non-retryable jobs and wasted
        # try2/try3 GPU submissions. Honor the explicit decision instead.
        retryable_flag = record.get("retryable")
        if retryable_flag is False:
            return False
        return True

    @classmethod
    def _is_resource_scheduling_failure(cls, record: Mapping[str, Any]) -> bool:
        text = " ".join(
            str(value or "").lower()
            for value in [
                record.get("error"),
                record.get("failure_class"),
                ";".join(str(item) for item in ((record.get("monitor") or {}).get("failure_hints") or [])) if isinstance(record.get("monitor"), Mapping) else "",
            ]
        )
        return any(needle in text for needle in cls.RESOURCE_SCHEDULING_FAILURE_NEEDLES)

    def _resource_retry_correction_proposal(self, job: DesignJob) -> Dict[str, Any]:
        """Propose degraded resources without mutating a finalized job."""
        params = copy.deepcopy(job.params or {})
        before = {key: params.get(key) for key in self.RESOURCE_CONFIG_KEYS if key in params}
        default_gpu = str(getattr(self.cfg.resource, "gpu_name", "") or "V100")
        default_timeout = max(1, int(getattr(self.cfg.resource, "timeout_seconds", 3600) or 3600))
        try:
            current_devices = max(1, int(params.get("devices") or getattr(self.cfg.resource, "host_gpu_num", 1) or 1))
        except (TypeError, ValueError):
            current_devices = max(1, int(getattr(self.cfg.resource, "host_gpu_num", 1) or 1))
        try:
            current_timeout = max(1, int(params.get("taiji_timeout") or default_timeout))
        except (TypeError, ValueError):
            current_timeout = default_timeout
        params["GPUName"] = default_gpu
        params["devices"] = max(1, current_devices // 2) if current_devices > 1 else 1
        params["taiji_timeout"] = max(current_timeout, default_timeout * 2)
        native_multi_host = params.get("native_taiji_multi_host")
        if isinstance(native_multi_host, Mapping):
            params["native_taiji_multi_host"] = {**dict(native_multi_host), "gpus_per_host": params["devices"]}
        changes = {key: params.get(key) for key in self.RESOURCE_CONFIG_KEYS if before.get(key) != params.get(key)}
        if not changes:
            return {}
        return {
            "reason": "resource_scheduling_failure",
            "requires_refinalization": True,
            "before": before,
            # Kept for old checkpoints/readers; new consumers apply the typed
            # patch over the immutable source params.
            "corrected_params": params,
            "correction_patch": {
                "version": 1,
                "set": changes,
                "remove": [],
                "classification": "resource_scheduling_failure",
                "identity_effect": "execution_semantics_changed",
                "source_validation_digest": stable_hash(before),
            },
            "changes": changes,
        }

    @staticmethod
    def _failure_fingerprint(record: Mapping[str, Any]) -> str:
        monitor = record.get("monitor") if isinstance(record.get("monitor"), Mapping) else {}
        run_spec = record.get("run_spec") if isinstance(record.get("run_spec"), Mapping) else {}
        runtime = record.get("runtime") if isinstance(record.get("runtime"), Mapping) else {}
        payload = {
            "stage": str(record.get("stage") or record.get("failure_stage") or "execution").lower(),
            "status": str(record.get("status") or "").lower(),
            "failure_class": str(record.get("failure_class") or "").lower(),
            "normalized_traceback": sanitize_error_text(str(record.get("error") or "")).strip().lower(),
            "failure_hints": sorted(str(item).lower() for item in (monitor.get("failure_hints") or [])),
            "job_semantic_digest": str(record.get("job_identity_digest") or ((record.get("job") or {}).get("params") or {}).get("job_identity", {}).get("semantic_digest") or ""),
            "runtime_digest": str(runtime.get("digest") or runtime.get("runtime_digest") or run_spec.get("runtime_digest") or ""),
            "image_digest": str(runtime.get("image_digest") or run_spec.get("image_digest") or ""),
            "proposal_changes": ((record.get("retry_correction_proposal") or {}).get("changes")
                                 if isinstance(record.get("retry_correction_proposal"), Mapping) else None),
        }
        return stable_hash(payload)

    @staticmethod
    def _semantic_failure_fingerprint(record: Mapping[str, Any]) -> str:
        """Fingerprint failure meaning while excluding identity/path bookkeeping."""
        monitor = record.get("monitor") if isinstance(record.get("monitor"), Mapping) else {}
        runtime = record.get("runtime") if isinstance(record.get("runtime"), Mapping) else {}
        run_spec = record.get("run_spec") if isinstance(record.get("run_spec"), Mapping) else {}
        pre_submit = record.get("pre_submit") if isinstance(record.get("pre_submit"), Mapping) else {}
        validation = record.get("config_validation") if isinstance(record.get("config_validation"), Mapping) else {}
        if not validation and isinstance(pre_submit.get("validation"), Mapping):
            validation = pre_submit.get("validation")
        if not validation and isinstance(pre_submit.get("config_validation"), Mapping):
            validation = pre_submit.get("config_validation")

        def canonical_text(value: Any) -> str:
            text = str(sanitize_error_text(str(value or "")) or "").strip().lower()
            text = re.sub(r"(?:[a-zA-Z]:)?[/\\][^\s:;,]+", "<path>", text)
            text = re.sub(r"\br(?:ound[_-]?)?\d+\b", "<round>", text)
            return re.sub(r"\b(?:job|attempt|try)[_#:= -]*[a-z0-9_.-]+", "<bookkeeping>", text)

        error = canonical_text(record.get("error"))

        blocking_parameters = []
        for issue in validation.get("issues") or []:
            if not isinstance(issue, Mapping):
                continue
            if str(issue.get("severity") or "").lower() == "error" and not issue.get("resolved"):
                blocking_parameters.append({
                    "parameter": str(issue.get("parameter") or ""),
                    "problem": canonical_text(issue.get("problem")),
                    "correction": canonical_text(issue.get("correction")),
                })
        blocking_parameters.sort(key=lambda item: (item["parameter"], item["problem"], item["correction"]))
        missing_required = sorted({str(key) for key in (validation.get("missing_required_keys") or [])})
        semantic_changes = []
        for change in validation.get("semantic_changes") or []:
            if not isinstance(change, Mapping):
                continue
            if str(change.get("change") or "").lower() == "normalization":
                continue
            if str(change.get("partition") or "").lower() not in {"runner", "adapter", "runtime"}:
                continue
            semantic_changes.append({
                key: change.get(key)
                for key in ("parameter", "before", "after", "change", "partition", "policy_class")
                if key in change
            })
        semantic_changes.sort(key=lambda item: str(item.get("parameter") or ""))
        payload = {
            "version": 1,
            "stage": str(record.get("stage") or record.get("failure_stage") or "execution").lower(),
            "status": str(record.get("status") or "").lower(),
            "failure_class": str(record.get("failure_class") or "").lower(),
            "normalized_traceback": error,
            "failure_hints": sorted(str(item).lower() for item in (monitor.get("failure_hints") or [])),
            "runtime_digest": str(runtime.get("digest") or runtime.get("runtime_digest") or run_spec.get("runtime_digest") or ""),
            "image_digest": str(runtime.get("image_digest") or run_spec.get("image_digest") or ""),
            "blocking_parameters": blocking_parameters,
            "missing_required_keys": missing_required,
            "semantic_changes": semantic_changes,
        }
        return stable_hash(payload)

    @staticmethod
    def _identical_failure_fingerprint(
        ledger: Mapping[str, Any], lock: threading.Lock, job_id: str, fingerprint: str,
    ) -> bool:
        with lock:
            attempts = list((((ledger.get("jobs") or {}).get(job_id) or {}).get("attempts") or []))
            prior = [str(item.get("failure_fingerprint") or "") for item in attempts[:-1]]
        return bool(fingerprint and prior and prior[-1] == fingerprint)

    def _record_attempt_started(self, path: Path, ledger: Dict[str, Any], lock: threading.Lock, job: DesignJob, attempt: int) -> None:
        with lock:
            job_entry = self._job_attempt_entry(ledger, job)
            # A retryable failed Taiji terminal record may be reactivated on rerun.
            # Once a new attempt starts, remove the stale terminal marker so an
            # interrupted resubmission is detected as incomplete instead of hidden.
            job_entry.pop("terminal_record", None)
            job_entry.pop("terminal_at", None)
            attempt_root = Path(job.output_dir) / "attempts" / f"attempt_{int(attempt):02d}"
            job_entry.setdefault("attempts", []).append({
                "attempt": attempt, "status": "started", "started_at": time.time(),
                "artifact_locators": {
                    "identity_root": str(Path(job.output_dir)),
                    "attempt_root": str(attempt_root),
                    "execution_record": str(attempt_root / "execution_record.json"),
                    "legacy_execution_record": str(Path(job.output_dir) / "execution_record.json"),
                },
            })
            self._write_attempt_ledger(path, ledger)

    def _record_attempt_finished(self, path: Path, ledger: Dict[str, Any], lock: threading.Lock, job: DesignJob, attempt: int, record: Mapping[str, Any]) -> None:
        with lock:
            job_entry = self._job_attempt_entry(ledger, job)
            attempts = job_entry.setdefault("attempts", [])
            if not attempts or attempts[-1].get("attempt") != attempt:
                attempts.append({"attempt": attempt, "status": "started", "started_at": time.time()})
            attempts[-1].update({
                "status": str(record.get("status") or "unknown"),
                "error": record.get("error"),
                "finished_at": time.time(),
                "retryable": record.get("retryable"),
                "failure_fingerprint": record.get("failure_fingerprint"),
                "semantic_failure_fingerprint_version": int(record.get("semantic_failure_fingerprint_version") or 1),
                "semantic_failure_fingerprint": record.get("semantic_failure_fingerprint"),
                "semantic_failure_scope": record.get("semantic_failure_scope"),
                "retry_circuit_breaker": record.get("retry_circuit_breaker"),
                "artifact_locators": copy.deepcopy(record.get("artifact_locators") or attempts[-1].get("artifact_locators") or {}),
            })
            self._write_attempt_ledger(path, ledger)

    def _record_terminal_attempt(self, path: Path, ledger: Dict[str, Any], lock: threading.Lock, job: DesignJob, record: Mapping[str, Any]) -> None:
        with lock:
            job_entry = self._job_attempt_entry(ledger, job)
            job_entry["terminal_record"] = dict(record)
            job_entry["terminal_at"] = time.time()
            self._write_attempt_ledger(path, ledger)

    @staticmethod
    def _job_attempt_entry(ledger: Dict[str, Any], job: DesignJob) -> Dict[str, Any]:
        jobs = ledger.setdefault("jobs", {})
        existing = jobs.get(job.job_id)
        metadata = dict((job.params or {}).get("job_identity") or {})
        digest = BinderDesignOrchestrator._job_identity_digest(job)
        current_digest = (
            job_identity_semantic_digest(job)
            if bool(metadata.get("finalized"))
            else effective_semantic_digest(job)
        )
        if metadata and digest != current_digest:
            raise ValueError(f"attempt_ledger_identity_mismatch:{job.job_id}")
        if existing is not None:
            stored_job = dict(existing.get("job") or {})
            stored_digest = str(existing.get("job_identity_digest") or "")
            if stored_digest and stored_digest != digest:
                raise ValueError(f"attempt_ledger_identity_mismatch:{job.job_id}")
            if stored_job and str(stored_job.get("output_dir") or "") != str(job.output_dir):
                raise ValueError(f"attempt_ledger_output_mismatch:{job.job_id}")
            existing.setdefault("job_identity_digest", digest)
            return existing
        jobs[job.job_id] = {"job": asdict(job), "job_identity_digest": digest, "attempts": []}
        return jobs[job.job_id]

    def _write_attempt_ledger(self, path: Path, ledger: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(path, ledger)

    def _initial_jobs(self) -> List[DesignJob]:
        jobs = self.learner.initial_jobs(self.cfg.target.structure_path, self.cfg.target.chain_id, list(self.cfg.target.hotspots or []), self.cfg.search_space.binder_lengths, str(self.out_dir), self._base_params(), branch_width=self.cfg.active_learning.branch_width)
        return self._materialize_sampler_and_context_intents(self._materialize_job_binding_types(jobs))

    def _taiji_host_shard_count(self) -> int:
        backend = str(getattr(self.cfg.resource, "backend", "") or "").lower()
        if backend != "taiji":
            return 1
        try:
            host_num = int(getattr(self.cfg.resource, "host_num", 1) or 1)
        except (TypeError, ValueError):
            host_num = 1
        return max(1, host_num)

    def _taiji_multi_host_mode(self) -> str:
        mode = str(getattr(self.cfg.resource, "taiji_multi_host_mode", "native") or "native").strip().lower()
        return {"unified": "native", "fanout": "split_jobs", "split": "split_jobs"}.get(mode, mode)

    def _native_taiji_multi_host(self) -> bool:
        return self._taiji_host_shard_count() > 1 and self._taiji_multi_host_mode() == "native"

    def _search_profile(self):
        return get_model_search_profile(primary_design_model(self.cfg), cfg=self.cfg)

    def _sampler_keys(self) -> Tuple[str, ...]:
        spec = getattr(getattr(self.cfg, "owner", None), "parameter_decision", None)
        if spec is not None:
            return sampler_keys_for_spec(spec)
        return tuple(self._search_profile().sampler_axes)

    def _profile_allowed_keys(self, *, include_internal: bool = False):
        return self._search_profile().executable_keys(include_internal=include_internal)

    def _profile_param_bounds(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self._search_profile().param_bounds.items()}

    def _base_params(self) -> Dict[str, Any]:
        cap = self._round_design_cap
        params: Dict[str, Any] = {
            "task_id": self.cfg.task_name,
            "num_designs": self._round_num_designs,
            "num_designs_per_round": self._round_num_designs,
            "max_binders_per_round": cap,
            "binder_length_range": self.cfg.search_space.binder_length_range,
            "binder_length_step": self.cfg.search_space.binder_length_step,
            "devices": max(1, int(getattr(self.cfg.resource, "host_gpu_num", 1) or 1)),
            "host_count": self._taiji_host_shard_count(),
            "taiji_multi_host_mode": self._taiji_multi_host_mode(),
            "GPUName": self.cfg.resource.gpu_name,
            "taiji_timeout": self.cfg.resource.timeout_seconds,
        }
        owner = getattr(self.cfg, "owner", None)
        sampler_bounds = getattr(owner, "sampler_bounds", None) if owner is not None else None
        if sampler_bounds is not None:
            params["sampler_bounds"] = {
                key: dict(value)
                for key in self._sampler_keys()
                for value in [getattr(sampler_bounds, key, None)]
                if value is not None
            }
        if self.cfg.target.include:
            params["target_include"] = self.cfg.target.include
        binding_types, provenance = self._build_target_binding_types(
            primary=self.cfg.target.hotspots or [],
            expanded=self._effective_auxiliary_hotspots(),
            negative=(self.cfg.search_space.boltzgen or {}).get("negative_binding_residues") or [],
            policy="primary_expanded",
        )
        if binding_types:
            params["target_binding_types"] = binding_types
        params["binding_residue_provenance"] = provenance
        params["exploration_arm"] = "baseline_hold"
        params["strategy_intent"] = {"kind": "hold"}
        if self.cfg.target.structure_groups:
            params["structure_groups"] = self.cfg.target.structure_groups
        canonical_binding_types = params.get("target_binding_types")
        canonical_provenance = params.get("binding_residue_provenance")
        profile = self._search_profile()
        params.update(profile.load_search_space(self.cfg))
        params["sequence_tool"] = profile.resolved_sequence_tool().name
        params["refold_tool"] = profile.resolved_refold_tool().name
        params = profile.filter_params(params).params
        params["search_profile_model"] = profile.model
        if profile.model != "rfd3" and canonical_binding_types is not None:
            params["target_binding_types"] = canonical_binding_types
        params["binding_residue_provenance"] = canonical_provenance
        params.pop("secondary_structure", None)
        params["run_filtering"] = True
        params["devices"] = max(1, int(getattr(self.cfg.resource, "host_gpu_num", 1) or 1))
        params["host_count"] = self._taiji_host_shard_count()
        params["taiji_multi_host_mode"] = self._taiji_multi_host_mode()
        params["GPUName"] = self.cfg.resource.gpu_name
        params["taiji_timeout"] = self.cfg.resource.timeout_seconds
        if bool(getattr(self.cfg.task, "freeze_round_budget", True)):
            params["num_designs"] = self._round_num_designs
            params["num_designs_per_round"] = self._round_num_designs
        else:
            params["num_designs"] = min(cap, max(1, int(params.get("num_designs", cap))))
            params["num_designs_per_round"] = min(cap, max(1, int(params.get("num_designs_per_round", params["num_designs"]))))
        params["max_binders_per_round"] = cap
        return params

    def _enforce_round_cap(self, jobs: List[DesignJob], *, round_id: Optional[int] = None) -> List[DesignJob]:
        cap = self._round_design_cap
        selected = sorted(self._balanced_round_jobs(jobs), key=lambda job: int((job.params or {}).get("arm_rank", 0)))
        if not selected:
            return []
        fractions: List[float] = []
        candidates: List[Dict[str, Any]] = []
        for index, job in enumerate(selected):
            params = dict(job.params or {})
            params.setdefault("target_identity_digest", self._target_identity_digest)
            job.params["target_identity_digest"] = params["target_identity_digest"]
            try:
                fractions.append(float(params.get("template_conditioned_fraction", 0.0)))
            except (TypeError, ValueError):
                pass
            conditioned = bool(params.get("template_conditioned"))
            bucket = "template_conditioned" if conditioned else "other"
            valid = True
            rejection_reason = ""
            if conditioned:
                validation = validate_template_application(params.get("binder_template"))
                job.params["template_validation"] = validation.to_dict()
                valid = validation.valid
                rejection_reason = validation.reason if not valid else ""
                if not valid:
                    template = dict(params.get("binder_template") or {})
                    tid = str(template.get("template_id") or "")
                    if tid:
                        failure = "alignment_failure" if "alignment_not_evaluable" in validation.failures else "package_failure"
                        failure_round = self._identity_round_id([job]) if round_id is None else int(round_id)
                        self.template_outcome_ledger.record_failure(
                            self._target_identity_digest, tid, round_id=failure_round,
                            failure_type=failure, detail=rejection_reason,
                        )
                        self.template_outcome_ledger.save()
            candidates.append({
                "id": job.job_id or f"job_{index}", "bucket": bucket, "weight": 1.0,
                "valid": valid, "rejection_reason": rejection_reason,
            })
        requested_fraction = max(fractions) if fractions else 0.0
        resolution = resolve_round_budget(cap, candidates, requested_conditioned_fraction=requested_fraction)
        allocation_by_index = {int(item["input_index"]): item for item in resolution.allocations}
        effective_gpus = max(1, int(self.cfg.resource.host_gpu_num or 1) * (int(self.cfg.resource.host_num or 1) if self.cfg.resource.backend == "taiji" else 1))
        arm_count = len(resolution.allocations)
        slot_shares = self._split_design_count(effective_gpus, min(effective_gpus, arm_count)) if effective_gpus >= arm_count else [1] * arm_count
        resolved_jobs: List[DesignJob] = []
        for index, job in enumerate(selected):
            allocation_record = allocation_by_index.get(index)
            if allocation_record is None:
                continue
            budget = int(allocation_record.get("num_designs", 0) or 0)
            if budget <= 0:
                continue
            if bool(job.params.get("execution_retry_preserve_budget")):
                try:
                    job.params["num_designs"] = min(max(1, int(job.params.get("num_designs", budget))), budget)
                except (TypeError, ValueError):
                    job.params["num_designs"] = budget
            else:
                job.params["num_designs"] = budget
            job.params["num_designs_per_round"] = job.params["num_designs"]
            job.params["max_binders_per_round"] = cap
            job.params["round_budget_resolution"] = resolution.to_dict()
            arm_rank = int(job.params.get("arm_rank", index))
            allocation_rank = len(resolved_jobs)
            job.params["arm_gpu_allocation"] = {
                "effective_gpu_count": effective_gpus,
                "arm_wave": allocation_rank // effective_gpus if effective_gpus < arm_count else 0,
                "gpu_slot_start": (allocation_rank % effective_gpus) if effective_gpus < arm_count else sum(slot_shares[:allocation_rank]),
                "gpu_slot_count": slot_shares[allocation_rank] if effective_gpus >= arm_count else 1,
            }
            job.params["round_budget_allocation"] = {
                "selected_jobs": len(resolution.allocations),
                "job_index": int(allocation_record["input_index"]),
                "allocation_job_id": str(allocation_record["id"]),
                "job_identity_digest": self._job_identity_digest(job),
                "binder_length": int(job.binder_length),
                "binder_lengths": [int(x) for x in (job.params.get("binder_lengths") or [job.binder_length])],
                "num_designs": int(job.params["num_designs"]),
                "round_cap": cap,
                "strategy": "multi_job_round_multi_length_gpu_fanout",
                "budget_resolver": "global_largest_remainder_round_budget",
                "requested_conditioned_fraction": resolution.requested_conditioned_fraction,
                "effective_conditioned_fraction": resolution.effective_conditioned_fraction,
                "budget_resolution_digest": resolution.digest,
                "legacy_round_budget_weight": job.params.get("round_budget_weight"),
                "arm_rank": arm_rank,
            }
            if bool(job.params.get("template_conditioned")):
                policy = dict(job.params.get("harness_template_policy") or {})
                plan = build_template_application_plan(
                    job.params.get("binder_template"), current_target=job.target_structure,
                    current_target_chain=job.chain_id,
                    round_fraction=float(policy.get("round_conditioned_fraction", requested_fraction) or 0.0),
                    allocated_num_designs=int(job.params["num_designs"]),
                )
                if not plan.applicability.get("applicable"):
                    raise AssertionError("invalid template survived round budget resolution")
                plan = bind_template_application_budget(
                    plan, int(job.params["num_designs"]),
                    receipt={"consumer": "round_budget_resolver", "budget_resolution_digest": resolution.digest},
                )
                job.params["template_application_plan"] = plan.to_dict()
                lineage = dict(job.params.get("lineage_identity") or {})
                job.params["template_execution_identity"] = build_template_execution_identity(
                    job.params, target_structure=job.target_structure, target_chain=job.chain_id,
                    output_dir=job.output_dir, lineage_schema_version=lineage.get("schema_version"),
                    lineage_manifest_digest=str(lineage.get("manifest_digest") or ""),
                )
            if self._native_taiji_multi_host():
                host_count = self._taiji_host_shard_count()
                job.params["host_count"] = host_count
                job.params["taiji_submit_host_num"] = host_count
                job.params["taiji_multi_host_mode"] = "native"
                job.params["native_taiji_multi_host"] = {"enabled": True, "host_count": host_count, "gpus_per_host": max(1, int(job.params.get("devices") or 1)), "execution_scope": "whole_cluster"}
                job.params["round_budget_allocation"]["strategy"] = "single_taiji_multi_host_gpu_fanout"
                job.params["round_budget_allocation"]["execution_scope"] = "whole_cluster"
            immutable_plan = finalize_immutable_branch_plan(job, int(job.params["num_designs"]))
            job.params["immutable_branch_plan"] = immutable_plan.to_dict()
            job.params["effective_intervention_digest"] = immutable_plan.effective_intervention_digest
            resolved_jobs.append(job)
        self._assert_normal_round_budget(resolved_jobs)
        execution_jobs = self._split_multi_host_taiji_jobs(resolved_jobs)
        identity_round = self._identity_round_id(selected) if round_id is None else int(round_id)
        return self._finalize_execution_job_identities(execution_jobs, round_id=identity_round)

    def _assert_normal_round_budget(self, jobs: List[DesignJob]) -> None:
        """Assert exact logical backbone budget before host-level splitting.

        Execution retry jobs explicitly marked ``execution_retry_preserve_budget``
        retain their remaining budget and are excluded from this normal-round
        invariant. Candidate row counts may differ due to inverse folding/failures.
        """
        if any(bool(job.params.get("execution_retry_preserve_budget")) for job in jobs):
            return
        normal_jobs = list(jobs)
        if not normal_jobs:
            return
        allocated = sum(int(job.params.get("num_designs", 0) or 0) for job in normal_jobs)
        assert allocated == self._round_design_cap, (
            f"normal logical round budget mismatch: allocated={allocated}, "
            f"requested={self._round_design_cap}"
        )

    def _split_multi_host_taiji_jobs(self, jobs: List[DesignJob]) -> List[DesignJob]:
        requested_hosts = self._taiji_host_shard_count()
        if requested_hosts <= 1 or self._taiji_multi_host_mode() != "split_jobs":
            return jobs
        expanded: List[DesignJob] = []
        for job in jobs:
            params = dict(job.params or {})
            if params.get("multi_taiji_host_shard"):
                expanded.append(job)
                continue
            try:
                total_designs = max(1, int(params.get("num_designs", params.get("num_designs_per_round", 1)) or 1))
            except (TypeError, ValueError):
                total_designs = 1
            shard_count = min(requested_hosts, total_designs)
            if shard_count <= 1:
                params["taiji_host_num_requested"] = requested_hosts
                params["taiji_submit_host_num"] = 1
                job.params = params
                expanded.append(job)
                continue
            shares = self._split_design_count(total_designs, shard_count)
            source_allocation = dict(params.get("round_budget_allocation") or {})
            for shard_index, num_designs in enumerate(shares):
                shard_params = dict(params)
                shard_params["num_designs"] = int(num_designs)
                shard_params["num_designs_per_round"] = int(num_designs)
                shard_params["taiji_submit_host_num"] = 1
                # Host shards are independent execution branches. Keep the
                # logical source lineage in shard metadata, but give each
                # executable shard its own branch identity so identity
                # validation cannot mistake distinct outputs for duplicates.
                shard_branch_id = f"{params.get('logical_branch_id') or params.get('branch_id') or job.job_id}_host{shard_index + 1:02d}"
                shard_params["logical_branch_id"] = shard_branch_id
                shard_params["branch_id"] = shard_branch_id
                shard_params["multi_taiji_host_shard"] = {
                    "enabled": True,
                    "requested_host_num": requested_hosts,
                    "submitted_host_num": 1,
                    "shard_count": shard_count,
                    "shard_index": shard_index,
                    "shard_id": f"{shard_index + 1}_of_{shard_count}",
                    "source_job_id": job.job_id,
                    "source_job_identity_digest": self._job_identity_digest(job),
                    "source_output_dir": job.output_dir,
                    "source_num_designs": total_designs,
                    "num_designs": int(num_designs),
                    "note": "host_num>1 is executed as multiple single-host Taiji jobs",
                }
                allocation = dict(source_allocation)
                allocation.update({
                    "num_designs": int(num_designs),
                    "multi_taiji_host_shard": shard_params["multi_taiji_host_shard"],
                    "strategy": "multi_host_single_host_taiji_fanout",
                })
                shard_params["round_budget_allocation"] = allocation
                suffix = f"host{shard_index + 1:02d}"
                shard_job = DesignJob(
                    job_id=f"{job.job_id}_{suffix}",
                    target_structure=job.target_structure,
                    chain_id=job.chain_id,
                    hotspots=list(job.hotspots),
                    binder_length=job.binder_length,
                    seed=job.seed,
                    params=shard_params,
                    output_dir=str(Path(job.output_dir) / suffix),
                )
                shard_plan = finalize_immutable_branch_plan(shard_job, int(num_designs))
                shard_job.params["immutable_branch_plan"] = shard_plan.to_dict()
                shard_job.params["effective_intervention_digest"] = shard_plan.effective_intervention_digest
                expanded.append(shard_job)
        return expanded

    @staticmethod
    def _logical_jobs_for_memory(jobs: List[DesignJob]) -> List[DesignJob]:
        """Collapse execution-only multi-host shards back into logical round jobs."""
        grouped: Dict[str, List[DesignJob]] = {}
        ordered_keys: List[str] = []
        passthrough: List[DesignJob] = []
        for job in jobs or []:
            shard = dict((job.params or {}).get("multi_taiji_host_shard") or {})
            source_id = str(shard.get("source_job_id") or "")
            source_digest = str(shard.get("source_job_identity_digest") or source_id)
            if not source_id:
                passthrough.append(job)
                continue
            if source_digest not in grouped:
                grouped[source_digest] = []
                ordered_keys.append(source_digest)
            grouped[source_digest].append(job)

        logical: List[DesignJob] = list(passthrough)
        for source_digest in ordered_keys:
            shards = grouped[source_digest]
            first = shards[0]
            first_shard = dict((first.params or {}).get("multi_taiji_host_shard") or {})
            source_id = str(first_shard.get("source_job_id") or "")
            for shard_job in shards:
                shard_meta = dict((shard_job.params or {}).get("multi_taiji_host_shard") or {})
                if str(shard_meta.get("source_job_identity_digest") or source_id) != source_digest:
                    raise ValueError(f"shard_source_identity_mismatch:{source_id}")
                if str(shard_meta.get("source_job_id") or "") != source_id or str(shard_meta.get("source_output_dir") or "") != str(first_shard.get("source_output_dir") or ""):
                    raise ValueError(f"shard_source_metadata_mismatch:{source_id}")
            params = dict(first.params or {})
            total_designs = first_shard.get("source_num_designs")
            if total_designs in (None, ""):
                total_designs = sum(int((job.params or {}).get("num_designs") or 0) for job in shards)
            total_designs = max(1, int(total_designs or 1))
            source_output_dir = str(first_shard.get("source_output_dir") or first.output_dir)
            params.pop("multi_taiji_host_shard", None)
            params.pop("taiji_submit_host_num", None)
            params["num_designs"] = total_designs
            params["num_designs_per_round"] = total_designs
            allocation = dict(params.get("round_budget_allocation") or {})
            allocation.pop("multi_taiji_host_shard", None)
            allocation["num_designs"] = total_designs
            allocation["strategy"] = "logical_multi_host_taiji_fanout"
            allocation["execution_shard_count"] = len(shards)
            params["round_budget_allocation"] = allocation
            logical.append(DesignJob(
                job_id=source_id,
                target_structure=first.target_structure,
                chain_id=first.chain_id,
                hotspots=list(first.hotspots),
                binder_length=first.binder_length,
                seed=first.seed,
                params=params,
                output_dir=source_output_dir,
            ))
        return logical

    @staticmethod
    def _split_design_count(total_designs: int, shard_count: int) -> List[int]:
        total = max(1, int(total_designs))
        shards = max(1, min(int(shard_count), total))
        base = total // shards
        remainder = total % shards
        return [base + (1 if index < remainder else 0) for index in range(shards)]

    @staticmethod
    def _balanced_round_jobs(jobs: List[DesignJob]) -> List[DesignJob]:
        """Interleave jobs by binder length to cover length space early.

        With the single-task-per-round model this is typically a single job, but
        the helper stays length-ordered so any multi-job fallback still front-loads
        length coverage. BoltzGen has no seed control, so there is no seed key.
        """
        by_length: Dict[int, List[DesignJob]] = {}
        for job in jobs:
            by_length.setdefault(int(job.binder_length), []).append(job)
        for items in by_length.values():
            items.sort(key=lambda job: str(job.job_id))
        ordered: List[DesignJob] = []
        lengths = sorted(by_length)
        offset = 0
        while True:
            added = False
            for length in lengths:
                items = by_length[length]
                if offset < len(items):
                    ordered.append(items[offset])
                    added = True
            if not added:
                break
            offset += 1
        return ordered

    def _allowed_binder_lengths(self) -> List[int]:
        rng = self.cfg.search_space.binder_length_range
        if rng is None:
            return []
        try:
            return sorted({int(length) for length in _expand_length_range(rng, self.cfg.search_space.binder_length_step)})
        except (TypeError, ValueError):
            return []

    def _enforce_binder_length_range(self, jobs: List[DesignJob]) -> List[DesignJob]:
        """Clamp proposed job binder lengths to the user's hard binder_length_range."""
        allowed = self._allowed_binder_lengths()
        if not allowed:
            return jobs
        allowed_set = set(allowed)

        def _snap(length: int) -> int:
            return length if length in allowed_set else min(allowed, key=lambda a: (abs(a - length), a))

        for job in jobs:
            current = int(job.binder_length)
            if current not in allowed_set:
                clamped = _snap(current)
                job.binder_length = clamped
                job.params["binder_length_guardrail"] = {
                    "requested": current,
                    "applied": clamped,
                    "allowed_lengths": allowed,
                    "reason": "binder_length_range hard constraint",
                }
            # Keep the policy-selected length subset (e.g. from BinderLengthPolicyAgent
            # / auto_binder_length), only snapping each entry into the user's hard
            # range. Do NOT overwrite it with the full expanded range, otherwise every
            # round would be forced back to all lengths and length narrowing would be
            # lost. The whole subset is fanned across GPUs inside one Taiji task.
            requested_lengths = [int(x) for x in (job.params.get("binder_lengths") or [job.binder_length])]
            snapped = sorted({_snap(length) for length in requested_lengths})
            job.params["binder_lengths"] = snapped or list(allowed)
        return jobs

    def _review_and_unfreeze_arms(
        self, *, memory: Any, round_dir: Path, next_round_id: int, blocked_arms: set[str],
        arm_evidence_cards: Mapping[str, Any], selection_context: Mapping[str, Any],
        hypotheses: Sequence[Mapping[str, Any]], structural_summary: Any, quality_analysis: Mapping[str, Any],
    ) -> set[str]:
        if not blocked_arms:
            return set()
        states_by_arm = {
            arm: dict(memory.experiment_ledger.arm_blocks.get(arm) or {
                "arm_id": arm, "status": "soft_blocked", "reason": "inherited_soft_block",
            })
            for arm in sorted(blocked_arms)
        }
        evidence = []
        for item in arm_evidence_cards.get("arms", []) or []:
            row = dict(item)
            row.setdefault("evidence_id", f"R{next_round_id-1}:ARM:{row.get('arm_id')}:EVALUATION")
            evidence.append(row)
        complete_arms = {
            str(item.get("arm_id") or "")
            for item in evidence
            if str(item.get("arm_id") or "") in blocked_arms
            and str(item.get("status") or "").lower() == "closed"
            and int(item.get("completed_budget") or 0) >= int(item.get("requested_budget") or 0) > 0
            and int(item.get("trials") or 0) > 0
        }
        review_evidence = [item for item in evidence if str(item.get("arm_id") or "") in complete_arms]
        context = {
            "selection_context": dict(selection_context),
            "hypotheses": [dict(v) for v in hypotheses],
            "structural_summary": compact_structural_aggregate_from_object(structural_summary),
            "quality_analysis": dict(quality_analysis),
            "ledger_history": blocked_arm_ledger_view(memory.experiment_ledger, complete_arms, max_rounds=5),
        }
        reviews_by_arm = {
            arm_id: {
                "arm_id": arm_id, "recommendation": "insufficient_evidence",
                "accepted_evidence_ids": [], "counterevidence_ids": [],
                "risk_codes": ["no_direct_complete_closed_evidence"],
                "reason": "No direct, complete, closed evidence for this blocked arm.",
            }
            for arm_id in blocked_arms - complete_arms
        }
        llm_decision = None
        if complete_arms:
            llm_decision = self.blocked_arm_review_agent.review(
                round_id=next_round_id,
                blocked_arms=[states_by_arm[arm] for arm in sorted(complete_arms)],
                evidence=review_evidence, context=context,
            )
            for review in llm_decision.reviews:
                arm_id = str(review.get("arm_id") or "")
                if arm_id in complete_arms and arm_id not in reviews_by_arm:
                    reviews_by_arm[arm_id] = dict(review)
        for arm_id in complete_arms:
            reviews_by_arm.setdefault(arm_id, {
                "arm_id": arm_id, "recommendation": "insufficient_evidence",
                "accepted_evidence_ids": [], "counterevidence_ids": [],
                "risk_codes": ["missing_review"],
                "reason": "No valid review was returned for this blocked arm.",
            })
        decision = BlockedArmReviewDecision(
            round_id=next_round_id, reviews=[reviews_by_arm[arm] for arm in sorted(blocked_arms)],
            llm_used=bool(llm_decision and getattr(llm_decision, "llm_used", False)),
            raw=dict(getattr(llm_decision, "raw", {}) or {}) if llm_decision else {
                "source": "deterministic_keep_blocked",
                "fallback_reason": "no_direct_complete_closed_evidence", "llm_attempts": [],
            },
        )
        self._write_json(round_dir / "blocked_arm_review.json", decision.to_dict())
        remaining = set(blocked_arms)
        evidence_arm_by_id = {
            str(item.get("evidence_id")): str(item.get("arm_id") or "")
            for item in review_evidence if str(item.get("evidence_id") or "")
        }
        for review in decision.reviews:
            arm_id = str(review.get("arm_id") or "")
            accepted = [str(v) for v in review.get("accepted_evidence_ids") or []]
            accepted_match_arm = bool(accepted) and all(evidence_arm_by_id.get(item) == arm_id for item in accepted)
            state = dict(memory.experiment_ledger.arm_blocks.get(arm_id) or {})
            cooldown_expired = int(state.get("cooldown_until_round") or 0) <= int(next_round_id)
            if review.get("recommendation") == "eligible_for_unfreeze" and accepted_match_arm and cooldown_expired:
                remaining.discard(arm_id)
                self.memory_store.apply_arm_unfreeze(
                    memory, arm_id=arm_id, round_id=next_round_id, evidence_ids=accepted,
                    reason=str(review.get("reason") or "validated_llm_review"),
                )
        return remaining

    def _strategy_selection_context(
        self, *, evaluation: EvaluationSummary, active_learning_examples: Mapping[str, Any],
        round_outcome: RoundOutcome, rollback_decision: RollbackDecision,
        fragment_templates: FragmentTemplateBatch, round_id: int,
    ) -> Dict[str, Any]:
        current = dict((active_learning_examples or {}).get("current_round") or {})
        counts = dict(current.get("counts") or {})
        return {
            "strict_positive_count": int(counts.get("strict_positive") or round_outcome.strict_successes),
            "min_positives_for_exploit": int(getattr(self.cfg.active_learning, "min_current_positives_for_exploit", 2) or 2),
            "failure_tag_counts": dict(evaluation.tag_counts or {}),
            "round_rank_key": list(round_outcome.round_rank_key or []),
            "round_rank_improved": int(rollback_decision.best_round) == int(round_id),
            "round_rank_non_regressed": not bool(rollback_decision.is_regression),
            "plateau": round_id > 0 and not bool(rollback_decision.is_regression) and int(rollback_decision.best_round) != int(round_id),
            "effective_templates_available": bool((fragment_templates.recommended_config or {}).get("binder_templates") or (fragment_templates.recommended_config or {}).get("binder_template")),
        }

    def _govern_exploration_jobs(
        self,
        jobs: Sequence[DesignJob],
        *,
        current_jobs: Sequence[DesignJob],
        next_round_id: int,
        strict_positive_count: int,
        blocked_digests: Iterable[str] = (),
        prefilter_records: Optional[Sequence[Mapping[str, Any]]] = None,
        memory: Optional[Any] = None,
    ) -> List[DesignJob]:
        """Assess ranked arms one-by-one; only eligible arms consume width."""
        del strict_positive_count
        requested_width = max(1, int(self.cfg.active_learning.branch_width))
        baseline = list(current_jobs)[0] if current_jobs else None
        baseline_reference = self._materialize_job_binding_types([copy.deepcopy(baseline)])[0] if baseline is not None else None
        records: List[Dict[str, Any]] = [dict(item) for item in (prefilter_records or [])]
        governed: List[DesignJob] = []
        seen_effective: set[str] = set()
        blocked = {str(value) for value in blocked_digests if str(value)}
        if memory is not None:
            blocked.update(self.memory_store.blocked_interventions(memory, next_round_id))
        input_jobs = list(jobs or [])
        for rank, job in enumerate(input_jobs):
            if len(governed) >= requested_width:
                records.append({"rank": rank, "arm_id": str((job.params or {}).get("arm_id") or ""), "applicability": "not_selected", "reason": "branch_width_filled"})
                continue
            params = dict(job.params or {})
            arm_id = str(params.get("arm_id") or params.get("exploration_arm") or "")
            reason = "semantic_delta_resolved"
            template_validation = None
            if bool(params.get("template_conditioned")):
                template_validation = validate_template_application(params.get("binder_template"))
                params["template_validation"] = template_validation.to_dict()
                job.params = params
            if template_validation is not None and not template_validation.valid:
                applicability = ArmApplicability.NOT_APPLICABLE
                reason = f"invalid_template_application:{template_validation.reason}"
            elif arm_id == "target_context_focus" and self._epitope_crop_disabled_hard_constraint():
                applicability = ArmApplicability.NOT_APPLICABLE
                reason = "target_crop_hard_disabled"
            elif arm_id and arm_id not in self._search_profile().supported_arms:
                applicability = ArmApplicability.UNSUPPORTED
                reason = "unsupported_arm_for_search_profile"
            elif arm_id == "template_exploit" and not params.get("template_conditioned"):
                applicability = ArmApplicability.NOT_APPLICABLE
                reason = "requires_effective_template"
            elif arm_id == "site_primary_condition" and not list(job.hotspots or []):
                applicability = ArmApplicability.NOT_APPLICABLE
                reason = "requires_primary_binding_residues"
            elif arm_id == "site_expanded_condition" and not list(params.get("expanded_binding_residues") or params.get("auxiliary_hotspots") or []):
                applicability = ArmApplicability.NOT_APPLICABLE
                reason = "requires_expanded_binding_residues"
            elif arm_id == "site_negative_exclusion" and not list(params.get("negative_binding_residues") or []):
                applicability = ArmApplicability.NOT_APPLICABLE
                reason = "requires_negative_binding_residues"
            elif arm_id == "sampler_explore" and not bool(params.get("sampler_policy_applied")):
                applicability = ArmApplicability.NOT_APPLICABLE
                reason = "requires_final_sampler_state"
            elif baseline is None:
                applicability = ArmApplicability.NOT_APPLICABLE
                reason = "missing_baseline_parent"
            elif arm_id == "baseline_hold":
                params["branch_role"] = "control"
                params["is_control"] = True
                params.pop("final_parameter_state", None)
                params.pop("parameter_catalog", None)
                params.pop("parameter_catalog_digest", None)
                job.params = params
                effective = effective_semantic_digest(job)
                parent_effective = effective_semantic_digest(baseline_reference)
                if effective != parent_effective:
                    applicability, reason = ArmApplicability.NOT_APPLICABLE, "baseline_execution_semantics_differ_from_parent"
                elif effective in blocked:
                    applicability, reason = ArmApplicability.BLOCKED, "effective_intervention_digest_in_cooldown"
                elif effective in seen_effective:
                    applicability, reason = ArmApplicability.DUPLICATE_EFFECTIVE_INTERVENTION, "duplicate_effective_intervention"
                else:
                    applicability, reason = ArmApplicability.ELIGIBLE, "parent_execution_semantics_verified"
            elif not arm_id:
                applicability, reason = ArmApplicability.ELIGIBLE, "legacy_single_job_compatibility"
                params["arm_id"] = "sampler_explore" if any(key in params for key in self._sampler_keys()) else "baseline_hold"
                params["exploration_arm"] = params["arm_id"]
                params.setdefault("branch_id", f"round_{next_round_id}")
                job.params = params
            else:
                intent = dict(params.get("strategy_intent") or {})
                candidate = CandidateIntervention(arm=arm_id, family=str(intent.get("kind") or "unknown"), direction=str(intent.get("direction") or intent.get("mode") or "apply"), proposed_changes=params, branch_role="probe")
                plan = assess_candidate_intervention(candidate, baseline_reference, job, blocked_digests=blocked, seen_effective_digests=seen_effective)
                params["resolved_intervention_plan"] = plan.to_dict()
                job.params = params
                applicability, reason = plan.applicability, plan.reason
            job.params["strategy_applicability"] = applicability.value
            job.params["strategy_applicability_reason"] = reason
            effective = effective_semantic_digest(job)
            records.append({"rank": rank, "arm_id": arm_id, "applicability": applicability.value, "reason": reason, "execution_semantic_digest": effective})
            if applicability == ArmApplicability.ELIGIBLE:
                governed.append(job)
                seen_effective.add(effective)

        fallback_count = 0
        fallback_attempted = False
        self._last_joint_sampler_evidence_report = None
        if baseline_reference is not None and len(governed) < requested_width:
            fallback_attempted = True
            fallback_jobs = self._deterministic_sampler_fallback_jobs(
                baseline_reference, round_id=next_round_id, count=requested_width - len(governed),
                blocked_digests=blocked, seen_digests=seen_effective,
            )
            for job in fallback_jobs:
                effective = effective_semantic_digest(job)
                governed.append(job)
                seen_effective.add(effective)
                fallback_count += 1
                fallback_reason = (
                    "joint_evidence_sampler_fallback"
                    if job.params.get("fallback_selection_policy") == "joint_evidence"
                    else "deterministic_random_sampler_fallback"
                )
                records.append({"rank": len(records), "arm_id": job.params["arm_id"], "applicability": ArmApplicability.ELIGIBLE.value, "reason": fallback_reason, "execution_semantic_digest": effective})

        effective_width = len(governed)
        for rank, job in enumerate(governed):
            job.params.pop("controlled_comparison", None)
            job.params["arm_rank"] = rank
            job.params.setdefault("round_budget_weight", 1.0)
            job.params.setdefault("continuation_kind", "fresh_arm")
            job.params["requested_branch_width"] = requested_width
            job.params["effective_branch_width"] = effective_width
            if effective_width < requested_width:
                job.params["branch_width_degradation"] = {"requested": requested_width, "realized": effective_width, "reason": "insufficient_distinct_executable_arms"}
            job.params.setdefault("lineage", {"source_job_id": job.job_id, "source_round_id": next_round_id - 1})
        self._last_next_job_filtering_report = {
            "schema_version": 1, "round_id": int(next_round_id), "requested_branch_width": requested_width,
            "realized_branch_width": effective_width, "fallback_sampler_count": fallback_count,
            "blocked_intervention_digests": sorted(blocked), "arms": records,
        }
        evidence_report = getattr(self, "_last_joint_sampler_evidence_report", None)
        if fallback_attempted and isinstance(evidence_report, Mapping) and evidence_report.get("mode") != "off":
            self._last_next_job_filtering_report["joint_sampler_evidence"] = dict(evidence_report)
        return governed

    def _fallback_sampler_seed(self, baseline: DesignJob, *, round_id: int, catalog_digest: str, slot: int) -> int:
        payload = {
            "run_identity": str(self.out_dir.resolve()),
            "target_identity": self._target_identity_digest,
            "round_id": int(round_id),
            "parent_execution_digest": effective_semantic_digest(baseline),
            "parameter_catalog_digest": str(catalog_digest),
            "slot": int(slot),
        }
        return int(stable_hash(payload)[:16], 16)

    def _joint_sampler_evidence(
        self, baseline: DesignJob, *, spec: ParameterDecisionSpec, catalog_digest: str,
    ) -> Tuple[str, Tuple[Any, ...], Dict[str, Any]]:
        """Resolve evidence under an explicit off/shadow/active policy gate.

        The memory header is only a coarse target guard.  Promotion-grade rows
        additionally need job-level target, catalog, model, sequence-tool and
        refold-tool provenance.  This is deliberately not described as
        current-run isolation: same-target evidence can span compatible runs.
        """

        mode = str(getattr(spec, "joint_evidence_fallback_mode", "off") or "off").strip().lower()
        report: Dict[str, Any] = {
            "schema_version": 1,
            "mode": mode,
            "catalog_digest": str(catalog_digest),
            "evidence_count": 0,
            "matched_control_groups": 0,
            "scope": "disabled" if mode == "off" else "memory_header_target_guard",
            "activation_reason": "policy_disabled" if mode == "off" else "no_compatible_evidence",
            "shadow_recommendations": [],
        }
        if mode == "off":
            return mode, (), report

        active_memory = self._active_memory
        try:
            current_target_key = target_memory_key(asdict(self.cfg.target))
            memory_target_key = (
                target_memory_key(getattr(active_memory, "target", {}) or {})
                if active_memory is not None else ""
            )
        except Exception as exc:
            if mode != "shadow":
                raise
            report.update({
                "activation_reason": "shadow_memory_header_validation_failed",
                "shadow_error_type": type(exc).__name__,
            })
            return mode, (), report
        report.update({
            "current_target_key": current_target_key,
            "memory_target_key": memory_target_key,
        })
        if active_memory is None or memory_target_key != current_target_key:
            report["activation_reason"] = "memory_header_target_mismatch"
            return mode, (), report

        try:
            profile = self._search_profile()
            execution_context = {
                "search_profile_model": profile.model,
                "sequence_tool": profile.resolved_sequence_tool().name,
                "refold_tool": profile.resolved_refold_tool().name,
            }
        except Exception as exc:
            if mode != "shadow":
                raise
            report.update({
                "activation_reason": "shadow_current_context_resolution_failed",
                "shadow_error_type": type(exc).__name__,
            })
            return mode, (), report
        try:
            evidence = joint_parameter_evidence_from_rounds(
                getattr(active_memory, "rounds", ()),
                spec=spec,
                required_target_identity_digest=self._target_identity_digest,
                required_catalog_digest=catalog_digest,
                required_execution_context=execution_context,
            )
        except Exception as exc:
            if mode != "shadow":
                raise
            # Shadow mode must never block the production seeded fallback.
            # Retain only the exception class as sanitized policy telemetry.
            report.update({
                "activation_reason": "shadow_evidence_extraction_failed",
                "shadow_error_type": type(exc).__name__,
            })
            return mode, (), report
        matched_control_groups = {
            item.comparison_group for item in evidence
            if item.comparison_group and not item.is_control
        }
        report.update({
            "evidence_count": len(evidence),
            "matched_control_groups": len(matched_control_groups),
            "scope": "memory_header_and_job_provenance_match",
            "required_target_identity_digest": self._target_identity_digest,
            "required_execution_context": execution_context,
            "activation_reason": "compatible_evidence_available" if evidence else "no_compatible_evidence",
        })
        return mode, evidence, report

    def _deterministic_sampler_fallback_jobs(
        self, baseline: DesignJob, *, round_id: int, count: int, blocked_digests: Iterable[str], seen_digests: Iterable[str],
    ) -> List[DesignJob]:
        owner = getattr(self.cfg, "owner", None)
        spec = getattr(owner, "parameter_decision", None) if owner is not None else None
        if spec is None:
            spec = ParameterDecisionSpec()
        bounds_obj = getattr(owner, "sampler_bounds", None) if owner is not None else None
        bounds = {key: dict(value) for key in self._sampler_keys() for value in [getattr(bounds_obj, key, None) if bounds_obj is not None else None] if value is not None}
        current = {key: float((baseline.params or {}).get(key)) for key in self._sampler_keys() if (baseline.params or {}).get(key) not in (None, "")}
        catalog_digest = parameter_catalog_digest(spec)
        evidence_mode, joint_evidence, evidence_report = self._joint_sampler_evidence(
            baseline, spec=spec, catalog_digest=catalog_digest,
        )
        self._last_joint_sampler_evidence_report = evidence_report
        active_evidence = joint_evidence if evidence_mode == "active" else ()
        blocked = {str(value) for value in blocked_digests}
        seen = {str(value) for value in seen_digests}
        current_round_control_retained = effective_semantic_digest(baseline) in seen
        result: List[DesignJob] = []
        catalog_size = max(1, len(spec.catalog))
        for slot in range(max(0, int(count))):
            chosen = None
            seed = self._fallback_sampler_seed(baseline, round_id=round_id, catalog_digest=catalog_digest, slot=slot)
            for attempt in range(catalog_size):
                selected_states = [
                    state for state in (
                        (job.params or {}).get("final_parameter_state") for job in result
                    ) if isinstance(state, Mapping)
                ]
                states = deterministic_sampler_states(
                    spec, current=current, count=catalog_size, seed=seed + attempt, bounds=bounds,
                    evidence=active_evidence, selected=[ParameterCandidate(state) for state in selected_states],
                )
                if evidence_mode == "shadow" and joint_evidence and attempt == 0:
                    try:
                        shadow_states = deterministic_sampler_states(
                            spec, current=current, count=catalog_size, seed=seed, bounds=bounds,
                            evidence=joint_evidence,
                            selected=[ParameterCandidate(state) for state in selected_states],
                        )
                    except Exception as exc:
                        evidence_report.update({
                            "activation_reason": "shadow_selection_failed",
                            "shadow_error_type": type(exc).__name__,
                            "shadow_recommendations": [],
                        })
                        joint_evidence = ()
                    else:
                        evidence_report["shadow_recommendations"].append({
                            "slot": slot,
                            "states": [state.as_dict() for state in shadow_states[:min(5, len(shadow_states))]],
                        })
                if not states:
                    break
                for state in states:
                    values = state.as_dict()
                    params = dict(baseline.params or {})
                    params.update(values)
                    params.update({
                        "final_parameter_state": dict(values), "parameter_catalog_digest": catalog_digest,
                        "sampler_policy": "explore", "sampler_policy_applied": True,
                        "sampler_policy_status": "applied:joint_evidence_fallback" if active_evidence else "applied:deterministic_random_fallback",
                        "random_sampler_fallback": True, "random_sampler_seed": seed,
                        "random_sampler_slot": slot, "arm_id": f"sampler_explore_fallback_{slot:02d}",
                        "exploration_arm": "sampler_explore", "logical_branch_id": f"r{round_id}_sampler_fallback_{slot:02d}",
                        "branch_id": f"r{round_id}_sampler_fallback_{slot:02d}",
                        "strategy_intent": {"kind": "sampling", "direction": "explore", "fallback": True},
                    })
                    if active_evidence:
                        params.update({
                            "fallback_selection_policy": "joint_evidence",
                            "joint_sampler_evidence_count": len(joint_evidence),
                            "joint_sampler_matched_control_groups": evidence_report["matched_control_groups"],
                            "joint_sampler_evidence_scope": evidence_report["scope"],
                            "matched_control_semantics": "same_target_same_round_baseline",
                            "current_round_control_owned_by_governance": current_round_control_retained,
                            "current_sampler_state_excluded": all(key in current for key in spec.active_sampler_keys()),
                        })
                    candidate_job = DesignJob(f"r{round_id}_sampler_fallback_{slot:02d}", baseline.target_structure, baseline.chain_id, list(baseline.hotspots), baseline.binder_length, seed=baseline.seed, params=params, output_dir=f"{self.out_dir}/r{round_id}/arms/pending_sampler_fallback_{slot:02d}")
                    candidate_job = self._materialize_job_binding_types([candidate_job])[0]
                    candidate_job = self._materialize_sampler_and_context_intents([candidate_job])[0]
                    intervention = CandidateIntervention(
                        arm="sampler_explore", family="sampling", direction="explore",
                        proposed_changes=dict(candidate_job.params or {}), branch_role="probe",
                    )
                    plan = assess_candidate_intervention(
                        intervention, baseline, candidate_job,
                        blocked_digests=blocked, seen_effective_digests=seen,
                    )
                    candidate_job.params["resolved_intervention_plan"] = plan.to_dict()
                    candidate_job.params["strategy_applicability"] = plan.applicability.value
                    candidate_job.params["strategy_applicability_reason"] = plan.reason
                    if plan.applicability == ArmApplicability.ELIGIBLE:
                        chosen = candidate_job
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                digest = effective_semantic_digest(chosen)
                result.append(chosen)
                seen.add(digest)
        return result

    def _materialize_sampler_and_context_intents(self, jobs: List[DesignJob]) -> List[DesignJob]:
        owner = getattr(self.cfg, "owner", None)
        configured_bounds = getattr(owner, "sampler_bounds", None) if owner is not None else None
        decision_spec = getattr(owner, "parameter_decision", None) if owner is not None else None
        profile = self._search_profile()
        sampler_keys = self._sampler_keys()
        for job in jobs or []:
            params = dict(job.params or {})
            effective_decision_spec = decision_spec or (ParameterDecisionSpec() if params.get("random_sampler_fallback") else None)
            if effective_decision_spec is not None and profile.model == "rfd3" and not getattr(effective_decision_spec, "sampler_axes", None):
                effective_decision_spec = decision_spec
            bounds = {
                key: dict(value)
                for key in sampler_keys
                for value in [getattr(configured_bounds, key, None) if configured_bounds is not None else None]
                if value is not None
            }
            if bounds:
                params["sampler_bounds"] = bounds
            if str(params.get("sampler_policy") or "") == "explore":
                final_state = params.get("final_parameter_state")
                if params.get("random_sampler_fallback") and not (isinstance(final_state, Mapping) and final_state):
                    if effective_decision_spec is not None:
                        axes = effective_decision_spec.active_axes()
                        names = list(axes)
                        catalog = []
                        import itertools
                        for combo in itertools.product(*(axes[name] for name in names)):
                            catalog.append(dict(zip(names, (float(item) for item in combo))))
                        current = {key: float(params.get(key)) for key in sampler_keys if params.get(key) not in (None, "")}
                        eligible = [state for state in catalog if any(current.get(key) != value for key, value in state.items())]
                        if eligible:
                            seed = int(params.get("random_sampler_seed") or 0)
                            rng = random.Random(seed)
                            draw_index = rng.randrange(len(eligible))
                            final_state = eligible[draw_index]
                            params["final_parameter_state"] = dict(final_state)
                            params["random_sampler_provenance"] = {"seed": seed, "candidate_count": len(eligible), "candidate_catalog_digest": stable_hash(eligible), "draw_index": draw_index, "final_state": dict(final_state)}
                if isinstance(final_state, Mapping) and final_state:
                    if effective_decision_spec is None:
                        raise ValueError("final_parameter_state requires configured parameter catalog")
                    catalog_axes = {key: list(parameter_axis(effective_decision_spec, key)) for key in sampler_keys}
                    params = profile.materialize_sampler(params, final_state=final_state, catalog_axes=catalog_axes)
                    params["sampler_policy_status"] = str(params.get("sampler_policy_status") or "applied:final_probabilistic_state")
                else:
                    # Strategy intent is audit metadata only. Numeric sampler values
                    # may only be materialized from the authoritative final state.
                    params["sampler_policy_applied"] = False
                    params["sampler_policy_status"] = "not_applicable:missing_final_probabilistic_state"
                    params["strategy_intent"] = {"kind": "hold", "reason": "missing_final_probabilistic_state"}
            params = profile.materialize_sequence(params)
            job.params = params
            job = profile.materialize_site(job)
            params = dict(job.params or {})
            if str(params.get("target_context_policy") or "") == "focus":
                if self._epitope_crop_disabled_hard_constraint():
                    params["target_context_policy_status"] = "not_applicable:target_definition_frozen"
                else:
                    params["epitope_crop_mode"] = "hotspot_focus"
                    params["target_context_policy_status"] = "applied:epitope_crop_mode"
            filtered = profile.filter_params(params)
            params = dict(filtered.params)
            if filtered.stripped:
                params["search_profile_stripped_keys"] = list(filtered.stripped)
            job.params = params
        return jobs

    def _resolve_job_pressure_conflicts(self, jobs: List[DesignJob]) -> List[DesignJob]:
        """Apply pressure-conflict rules after strategy arms mutate job params."""
        conflict = dict(getattr(self, "_latest_pressure_conflict", {}) or {})
        if not conflict.get("active"):
            return jobs
        for job in jobs or []:
            resolved, notes = self._resolve_pressure_conflicts(
                dict(job.params or {}),
                {key: "strategy_job" for key in (job.params or {})},
            )
            if notes:
                resolved["pressure_conflict_job_notes"] = notes
            job.params = resolved
            effective_hotspots = [str(h) for h in (self.cfg.target.hotspots or []) if str(h).strip()]
            for hotspot in self._sanitize_auxiliary_hotspots(resolved.get("auxiliary_hotspots")):
                if hotspot not in effective_hotspots:
                    effective_hotspots.append(hotspot)
            job.hotspots = effective_hotspots
        return jobs

    @staticmethod
    def _annotate_candidate_count_semantics(evaluation: Any, jobs: List[DesignJob]) -> None:
        requested = 0
        expected_upper = 0
        for job in jobs or []:
            params = dict(job.params or {})
            try:
                num_designs = max(1, int(params.get("num_designs", params.get("num_designs_per_round", 0)) or 0))
            except (TypeError, ValueError):
                num_designs = 0
            try:
                inverse_fold = max(1, int(params.get("inverse_fold_num_sequences", 1) or 1))
            except (TypeError, ValueError):
                inverse_fold = 1
            requested += num_designs
            expected_upper += num_designs * inverse_fold
        evaluation.requested_backbone_designs = requested
        evaluation.expected_candidate_upper_bound = expected_upper
        filtering = dict(getattr(evaluation, "candidate_filtering", {}) or {})
        if filtering.get("filtering_applied"):
            evaluation.candidate_count_semantics = (
                "total_candidates counts the analysis cohort after user additional_filters; "
                "candidate_filtering.input_candidate_count counts raw downstream metric rows; "
                "requested_backbone_designs counts BoltzGen --num_designs backbones."
            )
        else:
            evaluation.candidate_count_semantics = "total_candidates counts downstream metric rows; requested_backbone_designs counts BoltzGen --num_designs backbones."

    def _quality_collaboration_signals(self, *, memory: Any, evaluation: Dict[str, Any], candidates: List[Dict[str, Any]], current_config: Dict[str, Any], rollback: Dict[str, Any]) -> Dict[str, Any]:
        """Build deterministic activation evidence; failed executions never call this as quality evidence."""
        total = max(1, int(evaluation.get("total_candidates") or len(candidates or []) or 0))
        tags = dict(evaluation.get("tag_counts") or {})
        passed = int(tags.get("pass_compute_gate") or evaluation.get("success_count") or 0)
        pae_values = []
        hotspot_values = []
        for candidate in candidates or []:
            metrics = dict(candidate.get("metrics") or candidate)
            for key in ("design_to_target_pae", "interchain_pae", "min_interaction_pae"):
                if metrics.get(key) is not None:
                    try: pae_values.append(float(metrics[key]))
                    except (TypeError, ValueError): pass
                    break
            for key in ("hotspot_contact", "hotspot_coverage", "hotspot_contact_fraction"):
                if metrics.get(key) is not None:
                    try: hotspot_values.append(float(metrics[key]))
                    except (TypeError, ValueError): pass
                    break
        prior = [dict(x) for x in (getattr(memory, "quality_collaboration_state", {}) or {}).get("signal_history", [])]
        previous = prior[-1] if prior else {}
        config_history = [r for r in (getattr(memory, "rounds", []) or []) if getattr(r, "config_snapshot", None)]
        signal = {
            "compute_gate_yield": passed / total,
            "mean_pae": sum(pae_values) / len(pae_values) if pae_values else None,
            "hotspot_yield": sum(hotspot_values) / len(hotspot_values) if hotspot_values else None,
            "failure_tags": tags,
            "zero_filter_pass_with_unfiltered_evidence": bool(
                (evaluation.get("candidate_filtering") or {}).get("quality_status") == "no_filter_pass"
                and len(candidates or []) > 0
            ),
            "high_value_events": (
                ["rollback_replay"]
                if rollback.get("action") in self.RECOVERY_ACTIONS else []
            ),
        }
        if signal["zero_filter_pass_with_unfiltered_evidence"]:
            signal["high_value_events"].append("zero_filter_pass_unfiltered_quality_review")
        previous_analysis = dict(getattr(config_history[-1], "quality_analysis", {}) or {}) if config_history else {}
        previous_raw = dict(previous_analysis.get("raw") or {})
        single_failures = []
        if previous_raw.get("fact_check_issues"):
            single_failures.append("fact_check_failed")
        confidences = []
        for section in ("high_quality_modules", "low_quality_modules", "causal_factors"):
            for item in previous_analysis.get(section) or []:
                if isinstance(item, dict) and item.get("confidence") is not None:
                    try: confidences.append(float(item["confidence"]))
                    except (TypeError, ValueError): pass
        threshold = float(getattr(self.cfg.quality_collaboration, "low_confidence_threshold", .55) or .55)
        if confidences and max(confidences) < threshold:
            single_failures.append("low_confidence")
        guidance = list(previous_analysis.get("next_round_guidance") or [])
        if previous_raw.get("suggestion_conflict"):
            single_failures.append("suggestion_conflict")
        signal["single_analysis_failures"] = single_failures
        signal["no_actionable_guidance"] = bool(previous_analysis and not guidance)
        prior_digest = (getattr(memory, "quality_collaboration_state", {}) or {}).get("last_guidance_digest")
        guidance_digest = hashlib.sha256(json.dumps(guidance, sort_keys=True, default=str).encode()).hexdigest()[:16] if guidance else None
        signal["conclusion_repeated"] = bool(guidance_digest and guidance_digest == prior_digest)
        for key in ("compute_gate_yield", "mean_pae", "hotspot_yield"):
            signal["previous_" + key] = previous.get(key)
        # Conflicting movement among core quality indicators merits independent review.
        deltas = []
        for key, direction in (("compute_gate_yield", 1), ("hotspot_yield", 1), ("mean_pae", -1)):
            if signal.get(key) is not None and previous.get(key) is not None:
                delta = (float(signal[key]) - float(previous[key])) * direction
                if abs(delta) > 1e-9: deltas.append(delta > 0)
        if deltas and any(deltas) and not all(deltas):
            signal["metric_conflict"] = "core indicators moved in opposing directions"
        previous_config = dict(config_history[-1].config_snapshot or {}) if config_history else {}
        changed = [k for k in current_config if previous_config.get(k) != current_config.get(k)]
        high_impact = {"binder_lengths", "binder_length_range", "binder_template", "binder_templates", "epitope_crop_mode", "template_conditioned_fraction"}
        important = sorted(set(changed) & high_impact)
        if len(important) >= int(getattr(self.cfg.quality_collaboration, "high_impact_parameter_count", 2) or 2):
            signal["high_value_events"].append("multiple_high_impact_parameter_changes:" + ",".join(important))
        state = dict(getattr(memory, "quality_collaboration_state", {}) or {})
        history = list(state.get("signal_history") or [])
        history.append({k: signal.get(k) for k in ("compute_gate_yield", "mean_pae", "hotspot_yield")})
        state["signal_history"] = history[-20:]
        if guidance_digest:
            state["last_guidance_digest"] = guidance_digest
        memory.quality_collaboration_state = state
        return signal

    def _hotspot_selection_enabled(self) -> bool:
        spec = getattr(self.cfg, "hotspot_selection", None)
        return bool(spec is not None and getattr(spec, "enabled", False) and self.hotspot_selection_agent is not None)

    def _hotspot_selection_spec(self):
        return getattr(self.cfg, "hotspot_selection", None)

    def _hotspot_residue_table_cached(self):
        if self._hotspot_residue_table is None:
            spec = self._hotspot_selection_spec()
            self._hotspot_residue_table = build_target_residue_table(
                self.cfg.target.structure_path,
                chain_id=str(self.cfg.target.chain_id or "A"),
                max_residues=int(getattr(spec, "max_residues_in_prompt", 200) or 200),
            )
        return self._hotspot_residue_table

    def _apply_llm_hotspots(self, hotspots: Sequence[str], *, selection: Optional[Mapping[str, Any]] = None) -> List[str]:
        tokens = [str(item).strip() for item in hotspots if str(item).strip()]
        self._llm_selected_hotspots = list(tokens)
        self.cfg.target.hotspots = list(tokens)
        if getattr(self.cfg, "task", None) is not None:
            self.cfg.task.hotspots = list(tokens)
        payload = dict(selection or {})
        payload["hotspots"] = list(tokens)
        payload["identity_hidden"] = True
        self._latest_hotspot_selection = payload
        current_path = self.out_dir / "llm_hotspot_selection_current.json"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(current_path, payload)
        return tokens

    def _prepare_llm_hotspots_for_run(self) -> None:
        restored = self._load_latest_hotspot_artifact()
        if restored and restored.get("hotspots"):
            self._apply_llm_hotspots(restored.get("hotspots") or [], selection=restored)
            return
        self._run_llm_hotspot_selection(round_id=0, previous=[], evidence={}, phase="pre_round")

    def _restore_llm_hotspots_after_recover(self, start_round: int) -> None:
        previous = self.out_dir / f"round_{start_round - 1:02d}" / "llm_hotspot_selection_next.json"
        used = self.out_dir / f"round_{start_round:02d}" / "llm_hotspot_selection.json"
        payload = None
        for path in (used, previous, self.out_dir / "llm_hotspot_selection_current.json"):
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    payload = None
                if isinstance(payload, dict) and payload.get("hotspots"):
                    self._apply_llm_hotspots(payload.get("hotspots") or [], selection=payload)
                    return
        if not self._llm_selected_hotspots:
            self._run_llm_hotspot_selection(round_id=max(0, int(start_round)), previous=[], evidence={}, phase="pre_round")

    def _load_latest_hotspot_artifact(self) -> Optional[Dict[str, Any]]:
        candidates = [self.out_dir / "llm_hotspot_selection_current.json"]
        for round_id in range(self.max_rounds - 1, -1, -1):
            round_dir = self.out_dir / f"round_{round_id:02d}"
            candidates.extend([
                round_dir / "llm_hotspot_selection_next.json",
                round_dir / "llm_hotspot_selection.json",
            ])
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("hotspots"):
                return dict(payload)
        return None

    def _run_llm_hotspot_selection(
        self,
        *,
        round_id: int,
        previous: Sequence[str],
        evidence: Mapping[str, Any],
        phase: str,
    ) -> Dict[str, Any]:
        table = self._hotspot_residue_table_cached()
        skill_context = {
            "residue_table": table.prompt_payload(),
            "round_evidence": dict(evidence or {}),
        }
        if previous:
            skill_context["round_evidence"]["previous_hotspots"] = list(previous)
        skills = self._select_agent_skills("HotspotSelectionAgent", skill_context, ["llm_reasoning"])
        selection = self.hotspot_selection_agent.select(
            residue_table=table,
            previous_hotspots=previous,
            round_evidence=evidence,
            active_skills=skills,
            chain_id=str(self.cfg.target.chain_id or table.chain_id),
        )
        payload = selection.to_dict()
        payload.update({
            "round_id": int(round_id),
            "phase": phase,
            "identity_hidden": True,
        })
        applied = self._apply_llm_hotspots(selection.hotspots, selection=payload)
        if not applied:
            raise RuntimeError("LLM hotspot selection produced an empty hotspot set")
        self.bus.publish(AgentMessage(
            "HotspotSelectionAgent", "Orchestrator", "status",
            {"event": "llm_hotspots_selected", "hotspots": applied, "source": selection.source, "phase": phase},
            round_id=round_id,
        ))
        return payload

    def _persist_round_hotspot_selection(self, round_dir: Path, round_id: int, *, phase: str) -> Dict[str, Any]:
        payload = dict(self._latest_hotspot_selection or {})
        payload["round_id"] = int(round_id)
        payload["phase"] = phase
        payload["hotspots"] = list(self._llm_selected_hotspots or self.cfg.target.hotspots or [])
        path = round_dir / "llm_hotspot_selection.json"
        if path.exists() and phase == "used_this_round":
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = None
            if isinstance(existing, dict) and existing.get("hotspots"):
                payload = dict(existing)
                self._apply_llm_hotspots(payload.get("hotspots") or [], selection=payload)
                return payload
        self._write_json(path, payload)
        self._latest_hotspot_selection = payload
        return payload

    def _annotate_round_hotspot_metrics(self, round_dir: Path, round_id: int, evaluation: Any, round_outcome: Any) -> None:
        path = round_dir / "llm_hotspot_selection.json"
        payload = dict(self._latest_hotspot_selection or {})
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = dict(self._latest_hotspot_selection or {})
        total = int(getattr(evaluation, "total_candidates", 0) or 0)
        success = int(getattr(evaluation, "success_count", 0) or 0)
        payload["round_metrics"] = {
            "success_count": success,
            "total_candidates": total,
            "success_rate": (success / total) if total else 0.0,
            "round_rank_key": list(getattr(round_outcome, "round_rank_key", None) or []),
        }
        payload["round_id"] = int(round_id)
        self._write_json(path, payload)
        self._latest_hotspot_selection = payload

    def _refine_llm_hotspots(self, *, round_id: int, round_dir: Path, evaluation: Any, structural: Any, round_outcome: Any) -> Dict[str, Any]:
        previous = list(self._llm_selected_hotspots or self.cfg.target.hotspots or [])
        evidence = {
            "evaluation": {
                "success_count": int(getattr(evaluation, "success_count", 0) or 0),
                "total_candidates": int(getattr(evaluation, "total_candidates", 0) or 0),
                "success_rate": (
                    float(getattr(evaluation, "success_count", 0) or 0)
                    / float(getattr(evaluation, "total_candidates", 0) or 1)
                ),
                "tag_counts": dict(getattr(evaluation, "tag_counts", None) or {}),
                "round_rank_key": list(getattr(round_outcome, "round_rank_key", None) or []),
            },
            "structural_analysis": asdict(structural) if hasattr(structural, "__dataclass_fields__") else dict(structural or {}),
            "previous_hotspots": previous,
            "round_rank_key": list(getattr(round_outcome, "round_rank_key", None) or []),
        }
        payload = self._run_llm_hotspot_selection(
            round_id=round_id + 1,
            previous=previous,
            evidence=evidence,
            phase="post_round_refine",
        )
        next_path = round_dir / "llm_hotspot_selection_next.json"
        self._write_json(next_path, payload)
        return payload

    def _effective_auxiliary_hotspots(self) -> List[str]:
        return self._sanitize_auxiliary_hotspots((self.cfg.search_space.boltzgen or {}).get("auxiliary_hotspots"))

    def _effective_hotspots(self) -> List[str]:
        base = [str(h) for h in (self.cfg.target.hotspots or []) if str(h).strip()]
        out = list(base)
        for hotspot in self._effective_auxiliary_hotspots():
            if hotspot not in out:
                out.append(hotspot)
        return out

    def _sanitize_auxiliary_hotspots(self, raw: Any, *, max_items: int = 3, max_distance: int = 15) -> List[str]:
        base = [self._parse_hotspot_token(h) for h in (self.cfg.target.hotspots or [])]
        base = [(c, r) for c, r in base if c and r is not None]
        if not base:
            return []
        existing = {f"{c}:{r}" for c, r in base}
        values = raw if isinstance(raw, (list, tuple, set)) else ([raw] if raw else [])
        cleaned: List[str] = []
        for item in values:
            chain, resid = self._parse_hotspot_token(item)
            if not chain or resid is None:
                continue
            token = f"{chain}:{resid}"
            if token in existing or token in cleaned:
                continue
            if any(chain == bc and abs(int(resid) - int(br)) <= int(max_distance) for bc, br in base):
                cleaned.append(token)
            if len(cleaned) >= int(max_items):
                break
        return cleaned

    @staticmethod
    def _parse_hotspot_token(value: Any) -> Tuple[str, Optional[int]]:
        text = str(value or "").strip()
        if not text:
            return "", None
        if ":" in text:
            chain, raw = text.split(":", 1)
        else:
            chain, raw = "", text
        digits = "".join(ch for ch in raw if ch.isdigit() or ch == "-")
        try:
            return chain.strip(), int(digits)
        except ValueError:
            return chain.strip(), None

    def _build_target_binding_types(
        self,
        *,
        primary: Sequence[Any],
        expanded: Sequence[Any],
        negative: Sequence[Any],
        policy: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build BINDING/NOT_BINDING from one residue source with provenance."""
        known = set()
        try:
            analysis = self._target_analysis()
            spans = analysis.get("chain_residue_spans") or {}
            for chain, span in spans.items():
                if ".." in str(span):
                    lo, hi = str(span).split("..", 1)
                    known.update((str(chain), value) for value in range(int(lo), int(hi) + 1))
        except Exception:
            known = set()

        accepted: Dict[str, List[str]] = {"primary": [], "expanded": [], "negative": []}
        rejected: List[Dict[str, Any]] = []
        occupied = set()
        for source, values in (("primary", primary), ("expanded", expanded), ("negative", negative)):
            for raw in values or []:
                chain, resid = self._parse_hotspot_token(raw)
                chain = chain or str(self.cfg.target.chain_id)
                token = "%s:%s" % (chain, resid) if resid is not None else str(raw)
                if resid is None:
                    rejected.append({"residue": str(raw), "source": source, "reason": "invalid_residue"})
                    continue
                key = (chain, int(resid))
                if known and key not in known:
                    rejected.append({"residue": token, "source": source, "reason": "missing_from_target"})
                    continue
                if key in occupied:
                    rejected.append({"residue": token, "source": source, "reason": "duplicate_or_conflicting_precedence"})
                    continue
                occupied.add(key)
                accepted[source].append(token)

        positives = list(accepted["primary"])
        if policy in {"primary_expanded", "primary_expanded_negative"}:
            positives.extend(accepted["expanded"])
        negatives = accepted["negative"] if policy in {"primary_negative", "primary_expanded_negative"} else []
        by_chain: Dict[str, Dict[str, List[int]]] = {}
        for token in positives:
            chain, resid = self._parse_hotspot_token(token)
            by_chain.setdefault(chain, {"binding": [], "not_binding": []})["binding"].append(int(resid))
        for token in negatives:
            chain, resid = self._parse_hotspot_token(token)
            by_chain.setdefault(chain, {"binding": [], "not_binding": []})["not_binding"].append(int(resid))
        binding_types = []
        for chain in sorted(by_chain):
            payload: Dict[str, Any] = {"id": chain}
            if by_chain[chain]["binding"]:
                payload["binding"] = ",".join(str(value) for value in sorted(set(by_chain[chain]["binding"])))
            if by_chain[chain]["not_binding"]:
                payload["not_binding"] = ",".join(str(value) for value in sorted(set(by_chain[chain]["not_binding"])))
            binding_types.append({"chain": payload})
        provenance = {"schema_version": "1.0", "policy": policy, "accepted": accepted, "rejected": rejected, "effective": {"positive": positives, "negative": negatives}}
        return binding_types, provenance

    def _materialize_job_binding_types(self, jobs: Sequence[DesignJob]) -> List[DesignJob]:
        for job in jobs:
            params = dict(job.params or {})
            policy = str(params.get("binding_site_policy") or "primary_expanded")
            binding_types, provenance = self._build_target_binding_types(
                primary=self.cfg.target.hotspots or job.hotspots,
                expanded=params.get("expanded_binding_residues") or params.get("auxiliary_hotspots") or [],
                negative=params.get("negative_binding_residues") or [],
                policy=policy,
            )
            params["target_binding_types"] = binding_types
            params["binding_residue_provenance"] = provenance
            job.hotspots = list(provenance["effective"]["positive"])
            job.params = params
        return list(jobs)

    def _current_config_snapshot(self) -> Dict[str, Any]:
        return {
            "task_name": self.cfg.task_name,
            "target": asdict(self.cfg.target),
            "binder_length_range": self.cfg.search_space.binder_length_range,
            "binder_lengths": list(self.cfg.search_space.binder_lengths),
            "hotspots": list(self.cfg.target.hotspots),
            "target_include": self.cfg.target.include,
            "target_binding_types": self.cfg.target.binding_types,
            "original_target_include": list(self._original_target_include),
            "original_target_binding_types": list(self._original_target_binding_types),
            "structure_groups": self.cfg.target.structure_groups,
            "candidate_rows_are_downstream_metrics": True,
            "top_k": self.cfg.active_learning.top_k,
            "exploration_ratio": self.cfg.active_learning.exploration_ratio,
            "max_rounds": self.cfg.active_learning.max_rounds,
            # Nested full copy is authoritative for rollback. The flattened keys
            # remain for compatibility with existing analysis/memory consumers.
            "boltzgen_config": copy.deepcopy(dict(self.cfg.search_space.boltzgen or {})),
            "rfd3_config": copy.deepcopy(dict(self.cfg.search_space.rfd3 or {})),
            "model_config": copy.deepcopy(dict(getattr(self.cfg.search_space, self._search_profile().search_space_attr) or {})),
            "search_profile_model": self._search_profile().model,
            **dict(getattr(self.cfg.search_space, self._search_profile().search_space_attr) or {}),
            "resource": asdict(self.cfg.resource),
            "num_designs": self._round_num_designs,
            "num_designs_per_round": self._round_num_designs,
            "requested_backbone_designs_per_round": self._round_num_designs,
            "max_binders_per_round": self._round_design_cap,
        }

    def _target_profile_context(self) -> Dict[str, Any]:
        chains = {str(self.cfg.target.chain_id)}
        for item in list(self.cfg.target.include or []) + list(self.cfg.target.binding_types or []):
            chain = (item.get("chain") or {}).get("id") if isinstance(item, Mapping) else None
            if chain:
                chains.add(str(chain))
        return {
            "target_name": self.cfg.task_name,
            "structure_path": self.cfg.target.structure_path,
            "primary_chain_id": self.cfg.target.chain_id,
            "target_chains": sorted(chains),
            "target_include": list(self.cfg.target.include or []),
            "target_binding_types": list(self.cfg.target.binding_types or []),
            "hotspots": list(self.cfg.target.hotspots or []),
            "notes": self.cfg.target.notes,
            "profile": dict(getattr(self.cfg.target, "profile", {}) or {}),
            "structure_groups": self.cfg.target.structure_groups,
            "source": "current_task_config",
        }

    def _hard_constraints(self) -> Dict[str, Any]:
        return {
            "max_binders_per_round": self._round_design_cap,
            "num_designs_per_round": self._round_num_designs,
            "requested_backbone_designs_per_round": self._round_num_designs,
            "binder_length_range": self.cfg.search_space.binder_length_range,
            "binder_lengths": list(self.cfg.search_space.binder_lengths),
            "target_include": list(self._original_target_include or []),
            "target_binding_types": list(self._original_target_binding_types or []),
            "current_target_include": list(self.cfg.target.include or []),
            "current_target_binding_types": list(self.cfg.target.binding_types or []),
            "hotspots": list(self.cfg.target.hotspots or []),
            "auxiliary_hotspots": self._effective_auxiliary_hotspots(),
            "effective_hotspots": self._effective_hotspots(),
            "structure_groups": self.cfg.target.structure_groups,
            "original_structure_groups": self._original_structure_groups,
            "freeze_target_definition": bool(getattr(self.cfg.task, "freeze_target_definition", True)),
            "freeze_binder_length_range": bool(getattr(self.cfg.task, "freeze_binder_length_range", True)),
            "freeze_round_budget": bool(getattr(self.cfg.task, "freeze_round_budget", True)),
            "epitope_crop_mode": str((self.cfg.search_space.boltzgen or {}).get("epitope_crop_mode", "disabled")),
            "epitope_crop_disabled_hard_constraint": self._epitope_crop_disabled_hard_constraint(),
        }

    def _allow_harness_target_crop(
        self,
        update: Mapping[str, Any],
        *,
        sources: Optional[Mapping[str, str]] = None,
    ) -> bool:
        if not (update.get("target_include") or update.get("target_binding_types")):
            return False
        if self._epitope_crop_disabled_hard_constraint():
            return False
        mode = str(update.get("epitope_crop_mode") or (self.cfg.search_space.boltzgen or {}).get("epitope_crop_mode", "disabled")).strip().lower()
        if self._crop_mode_disabled(mode):
            return False
        if sources is None:
            return True
        return any(
            sources.get(key) == "fragment_template_mining"
            for key in ("target_include", "target_binding_types", "structure_groups")
        )

    def _freeze_hard_constraints(self, update: Mapping[str, Any], *, allow_harness_target_crop: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        frozen = dict(update or {})
        notes: List[Dict[str, Any]] = []

        def drop_or_restore(key: str, kept_value: Any, reason: str) -> None:
            if key not in frozen:
                return
            proposed = frozen.get(key)
            if proposed == kept_value:
                frozen.pop(key, None)
                return
            notes.append({"key": key, "ignored_value": proposed, "kept_value": kept_value, "reason": reason})
            frozen.pop(key, None)

        if self._epitope_crop_disabled_hard_constraint():
            proposed_mode = str(frozen.get("epitope_crop_mode") or "").strip().lower()
            if proposed_mode and not self._crop_mode_disabled(proposed_mode):
                notes.append({
                    "key": "epitope_crop_mode",
                    "ignored_value": frozen.get("epitope_crop_mode"),
                    "kept_value": "disabled",
                    "reason": "epitope_crop_mode was disabled in the user YAML and is a hard constraint unless allow_agent_epitope_crop=true",
                })
                frozen["epitope_crop_mode"] = "disabled"
            allow_harness_target_crop = False

        if bool(getattr(self.cfg.task, "freeze_target_definition", True)):
            if not allow_harness_target_crop:
                drop_or_restore("target_include", list(self._original_target_include or []), "target definition is frozen by original user hard constraint")
                drop_or_restore("target_binding_types", list(self._original_target_binding_types or []), "target definition is frozen by original user hard constraint")
            else:
                for crop_key in ("target_include", "target_binding_types"):
                    if crop_key in frozen:
                        notes.append({"key": crop_key, "kept_value": frozen.get(crop_key), "reason": "harness-derived epitope crop allowed by epitope_crop_mode"})
            drop_or_restore("hotspots", list(self.cfg.target.hotspots or []), "target definition is frozen by user hard constraint")
            if not allow_harness_target_crop:
                drop_or_restore("structure_groups", self._original_structure_groups, "target definition is frozen by original user hard constraint")
        if bool(getattr(self.cfg.task, "freeze_binder_length_range", True)):
            drop_or_restore("binder_length_range", self.cfg.search_space.binder_length_range, "binder length range is frozen by user hard constraint")
        drop_or_restore("run_filtering", True, "BoltzGen filtering/final ranking is fixed on for closed-loop metrics and candidate ingestion")
        if "secondary_structure" in frozen:
            notes.append({"key": "secondary_structure", "ignored_value": frozen.get("secondary_structure"), "kept_value": None, "reason": "harness does not emit BoltzGen secondary_structure because the schema requires exact per-residue syntax"})
            frozen.pop("secondary_structure", None)
        if bool(getattr(self.cfg.task, "freeze_round_budget", True)):
            budget_kept_values = {
                "max_binders_per_round": self._round_design_cap,
                "num_designs_per_round": self._round_num_designs,
                "num_designs": self._round_num_designs,
            }
            for key, kept_value in budget_kept_values.items():
                if key not in frozen:
                    continue
                try:
                    proposed = int(frozen[key])
                except (TypeError, ValueError):
                    proposed = None
                if proposed != int(kept_value):
                    notes.append({"key": key, "ignored_value": frozen.get(key), "kept_value": kept_value, "reason": "round budget is frozen by user hard constraint; LLM and policy outputs cannot control round sample count"})
                    frozen.pop(key, None)
        resource_kept_values = {
            "GPUName": self.cfg.resource.gpu_name,
            "devices": max(1, int(getattr(self.cfg.resource, "host_gpu_num", 1) or 1)),
            "taiji_timeout": self.cfg.resource.timeout_seconds,
        }
        for key, kept_value in resource_kept_values.items():
            if key not in frozen:
                continue
            proposed = frozen.get(key)
            try:
                same_value = int(proposed) == int(kept_value)
            except (TypeError, ValueError):
                same_value = str(proposed) == str(kept_value)
            if same_value:
                frozen.pop(key, None)
                continue
            notes.append({"key": key, "ignored_value": proposed, "kept_value": kept_value, "reason": "runtime resource fields are frozen; only orchestrator retry degradation may change them per attempt"})
            frozen.pop(key, None)
        return frozen, notes

    def _apply_next_round_update(self, *updates: Mapping[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for update in updates:
            for key, value in dict(update or {}).items():
                if value is not None:
                    merged[key] = value
        merged, _ = self._freeze_hard_constraints(
            merged,
            allow_harness_target_crop=self._allow_harness_target_crop(merged),
        )
        if "auxiliary_hotspots" in merged:
            cleaned_aux = self._sanitize_auxiliary_hotspots(merged.get("auxiliary_hotspots"))
            if cleaned_aux:
                merged["auxiliary_hotspots"] = cleaned_aux
            else:
                merged.pop("auxiliary_hotspots", None)
        cap = self._round_design_cap
        if "max_binders_per_round" in merged and not bool(getattr(self.cfg.task, "freeze_round_budget", True)):
            cap = min(cap, max(1, int(merged["max_binders_per_round"])))
        if bool(getattr(self.cfg.task, "freeze_round_budget", True)):
            merged["max_binders_per_round"] = self._round_design_cap
            merged["num_designs"] = self._round_num_designs
            merged["num_designs_per_round"] = self._round_num_designs
        else:
            merged["max_binders_per_round"] = cap
            merged["num_designs"] = min(cap, max(1, int(merged.get("num_designs", self.cfg.search_space.num_designs_per_round))))
            merged["num_designs_per_round"] = min(cap, max(1, int(merged.get("num_designs_per_round", merged["num_designs"]))))
        merged["run_filtering"] = True
        merged.pop("secondary_structure", None)
        self.cfg.search_space.boltzgen.pop("secondary_structure", None)

        if "binder_lengths" in merged:
            self.cfg.search_space.binder_lengths = sorted({int(x) for x in merged["binder_lengths"]})
        if "binder_length_range" in merged:
            self.cfg.search_space.binder_length_range = merged["binder_length_range"]
            if "binder_lengths" not in merged:
                self.cfg.search_space.binder_lengths = _expand_length_range(merged["binder_length_range"], self.cfg.search_space.binder_length_step)
        if "hotspots" in merged:
            self.cfg.target.hotspots = list(merged["hotspots"])
        self.cfg.search_space.num_designs_per_round = self._round_num_designs if bool(getattr(self.cfg.task, "freeze_round_budget", True)) else merged["num_designs_per_round"]
        self.cfg.search_space.max_binders_per_round = self._round_design_cap if bool(getattr(self.cfg.task, "freeze_round_budget", True)) else cap

        for key in self._search_profile().restore_keys:
            if key in merged:
                space = getattr(self.cfg.search_space, self._search_profile().search_space_attr)
                space[key] = merged[key]
        if "target_include" in merged:
            self.cfg.target.include = list(merged["target_include"] or [])
        if "target_binding_types" in merged:
            self.cfg.target.binding_types = list(merged["target_binding_types"] or [])
        if "structure_groups" in merged:
            self.cfg.target.structure_groups = merged["structure_groups"]
        if self._epitope_crop_disabled_hard_constraint():
            self.cfg.search_space.boltzgen["epitope_crop_mode"] = "disabled"
        if self._crop_mode_disabled((self.cfg.search_space.boltzgen or {}).get("epitope_crop_mode", "disabled")):
            self._restore_original_target_definition()
        if "exploration_ratio" in merged:
            self.cfg.active_learning.exploration_ratio = float(merged["exploration_ratio"])
            self.learner.exploration_ratio = self.cfg.active_learning.exploration_ratio
        if "top_k" in merged:
            self.cfg.active_learning.top_k = max(1, int(merged["top_k"]))
        if "max_rounds" in merged:
            self.cfg.active_learning.max_rounds = max(1, int(merged["max_rounds"]))
        return supported_config_changes(merged, include_internal=True, allowed_keys=self._profile_allowed_keys(include_internal=True))

    def _restore_original_target_definition(self) -> None:
        self.cfg.target.include = list(self._original_target_include or [])
        self.cfg.target.binding_types = list(self._original_target_binding_types or [])
        self.cfg.target.structure_groups = self._original_structure_groups
        self.cfg.search_space.boltzgen["target_include"] = list(self._original_target_include or [])
        self.cfg.search_space.boltzgen["target_binding_types"] = list(self._original_target_binding_types or [])
        if self._original_structure_groups is not None:
            self.cfg.search_space.boltzgen["structure_groups"] = self._original_structure_groups
        else:
            self.cfg.search_space.boltzgen.pop("structure_groups", None)


    def _resolve_pressure_conflicts(self, merged: Mapping[str, Any], sources: Mapping[str, str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Prevent repeating hotspot/contact-pressure moves after core metric regression."""
        out = dict(merged or {})
        conflict = dict(getattr(self, "_latest_pressure_conflict", {}) or {})
        if not conflict:
            memory = self._active_memory
            if memory is None:
                memory = self.memory_store.load(target=asdict(self.cfg.target))
                self._active_memory = memory
            feedback = self._build_tuning_feedback(memory, self._current_config_snapshot())
            conflict = dict((feedback or {}).get("pressure_conflict") or {})
        if not conflict.get("active"):
            return out, []

        best_config = dict(conflict.get("best_round_config") or {})
        current = self._current_config_snapshot()
        notes: List[Dict[str, Any]] = []

        def note(key: str, proposed: Any, resolved: Any, action: str) -> None:
            notes.append({
                "key": key,
                "source": "pressure_conflict_resolver",
                "proposed": proposed,
                "resolved": resolved,
                "action": action,
                "reason": conflict.get("reason"),
                "regression_reasons": conflict.get("regression_reasons", []),
            })

        if "auxiliary_hotspots" in out:
            proposed_aux = list(out.get("auxiliary_hotspots") or [])
            current_aux = list(current.get("auxiliary_hotspots") or [])
            best_aux = list(best_config.get("auxiliary_hotspots") or current_aux)
            allowed = [item for item in proposed_aux if item in set(current_aux) or item in set(best_aux)]
            if len(allowed) < len(proposed_aux):
                if allowed:
                    out["auxiliary_hotspots"] = allowed
                else:
                    out.pop("auxiliary_hotspots", None)
                note("auxiliary_hotspots", proposed_aux, allowed, "dropped_new_auxiliary_hotspots")

        if "epitope_crop_mode" in out:
            proposed = str(out.get("epitope_crop_mode") or "disabled")
            current_mode = str(current.get("epitope_crop_mode") or "disabled")
            best_mode = str(best_config.get("epitope_crop_mode") or current_mode or "disabled")
            if self._crop_mode_disabled(current_mode) and not self._crop_mode_disabled(proposed):
                out["epitope_crop_mode"] = best_mode if self._crop_mode_disabled(best_mode) else "disabled"
                for key in ("target_include", "target_binding_types", "structure_groups"):
                    if sources.get(key) != "fragment_template_mining":
                        out.pop(key, None)
                note("epitope_crop_mode", proposed, out.get("epitope_crop_mode"), "prevented_crop_tightening")

        if self._has_filter_bindingsite(out) and not self._has_filter_bindingsite(best_config):
            proposed = list(out.get("config_overrides") or [])
            out["config_overrides"] = self._without_filter_bindingsite(proposed)
            note("config_overrides", proposed, out.get("config_overrides"), "removed_filter_bindingsite_pressure")

        if "template_conditioned_fraction" in out:
            proposed = _float_or_none(out.get("template_conditioned_fraction"))
            current_fraction = _float_or_none(current.get("template_conditioned_fraction"))
            best_fraction = _float_or_none(best_config.get("template_conditioned_fraction"))
            if proposed is not None and current_fraction is not None and proposed > current_fraction + 1e-9:
                resolved = min(current_fraction, best_fraction if best_fraction is not None else current_fraction)
                out["template_conditioned_fraction"] = resolved
                note("template_conditioned_fraction", proposed, resolved, "capped_template_pressure")

        if self._binder_lengths_narrowed(current.get("binder_lengths"), out.get("binder_lengths")):
            proposed = list(out.get("binder_lengths") or [])
            resolved = list(best_config.get("binder_lengths") or current.get("binder_lengths") or proposed)
            out["binder_lengths"] = sorted({int(x) for x in resolved})
            note("binder_lengths", proposed, out.get("binder_lengths"), "reverted_length_narrowing")

        if self._target_include_size(out.get("target_include")) < self._target_include_size(current.get("target_include")):
            proposed = out.get("target_include")
            out["target_include"] = list(best_config.get("target_include") or self._original_target_include or current.get("target_include") or [])
            note("target_include", proposed, out.get("target_include"), "reverted_target_crop_narrowing")

        if self._binding_residue_count(out.get("target_binding_types")) > self._binding_residue_count(current.get("target_binding_types")):
            proposed = out.get("target_binding_types")
            out["target_binding_types"] = list(best_config.get("target_binding_types") or self._original_target_binding_types or current.get("target_binding_types") or [])
            note("target_binding_types", proposed, out.get("target_binding_types"), "reverted_binding_residue_expansion")

        if notes:
            out["length_delta_hint"] = out.get("length_delta_hint") or 10
            out["exploration_ratio"] = max(float(current.get("exploration_ratio", self.cfg.active_learning.exploration_ratio) or 0.3), min(0.6, float(current.get("exploration_ratio", 0.3) or 0.3) + 0.1))
            notes.append({
                "key": "search_direction",
                "source": "pressure_conflict_resolver",
                "value": "alternative_patch_length_topology",
                "reason": "core metrics regressed after contact-pressure increase; explore a different branch instead of adding pressure",
            })
        return out, notes

    @staticmethod
    def _rollback_suppressed_merge_report(
        *,
        rollback_decision: Any,
        input_config: Mapping[str, Any],
        binder_length_update: Mapping[str, Any],
        fragment_template_update: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build the normal merge-report schema without applying degraded inputs."""
        return {
            "schema_version": "1.1",
            "merge_order": [],
            "inputs": {
                "input_configuration": dict(input_config or {}),
                "binder_length_policy": dict(binder_length_update or {}),
                "policy_proposal": {},
                "fragment_template_mining": dict(fragment_template_update or {}),
            },
            "ignored_unsupported_keys": {},
            "ignored_internal_only_keys": {},
            "ownership_conflicts": [],
            "experiment_arm_rejections": [],
            "decisions": [],
            "physical_guardrail_clamps": [],
            "pressure_conflict_notes": [],
            "hard_constraint_freeze_notes": [],
            "applied_update": {},
            "applied_sources": {},
            "rollback_policy_suppressed": True,
            "rollback_action": str(rollback_decision.action),
            "rollback_replay_source_round": int(rollback_decision.best_round),
            "current_round_inputs": "audit_only_suppressed",
            "normalization_notes": [
                "Quality recovery suppresses every current-round proposal before merge or config mutation.",
                "The next-jobs module replaces this empty update with the exact best-round state.",
            ],
        }

    def _resolve_probabilistic_sampler(self, input_config: InputConfiguration) -> Dict[str, Any]:
        """Resolve InputConfiguration label probabilities independently per sampler key."""
        raw = dict(getattr(input_config, "raw", {}) or {})
        proposals = getattr(input_config, "parameter_proposals", None)
        if not isinstance(proposals, Mapping):
            proposals = raw.get("parameter_proposals") if isinstance(raw.get("parameter_proposals"), Mapping) else {}
        endpoint = self._llm.resolved_endpoint if self._llm is not None else None
        mode = str(getattr(getattr(endpoint, "capabilities", None), "logprobs", "auto") or "auto")
        capability = raw.get("logprobs_capability") if isinstance(raw.get("logprobs_capability"), Mapping) else {}
        sampler_keys = set(self._sampler_keys())
        source = {key: dict(value) for key, value in dict(proposals or {}).items() if key in sampler_keys and isinstance(value, Mapping)}
        available_evidence = any(str(item.get("status") or "") == "available" for item in source.values())
        status = str(capability.get("status") or ("supported" if available_evidence else "indeterminate"))
        current = self._current_config_snapshot()
        owner = getattr(self.cfg, "owner", None)
        bounds_spec = getattr(owner, "sampler_bounds", None)
        spec = getattr(owner, "parameter_decision", None)
        proposed_state: Dict[str, Any] = {}
        guardrail: Dict[str, Any] = {}
        final: Dict[str, float] = {}
        if spec is None:
            return {"source_proposals": source, "proposed_state": {}, "guardrail_mapping": {}, "final_executable_state": {}, "catalog_digest": ""}
        profile_bounds = self._profile_param_bounds()
        for key in sorted(self._sampler_keys()):
            item = source.get(key, {})
            distribution = item.get("distribution") or item.get("probabilities") or {}
            label_map = item.get("labels_to_values") or item.get("candidate_values") or item.get("labels") or {}
            if isinstance(label_map, Sequence) and not isinstance(label_map, (str, bytes)):
                label_map = {str(index): value for index, value in enumerate(label_map)}
            if not isinstance(distribution, Mapping) or not isinstance(label_map, Mapping):
                distribution, label_map = {}, {}
            bounds = dict(getattr(bounds_spec, key, None) or profile_bounds.get(key, {}))
            bounds.update({name: profile_bounds.get(key, {}).get(name) for name in ("max_step_abs", "max_step_ratio") if profile_bounds.get(key, {}).get(name) is not None})
            result = decide_parameter_distribution(
                distribution, labels_to_values=label_map, candidates=parameter_axis(spec, key),
                current=current.get(key), thresholds=spec.thresholds, bounds=bounds,
                capability_status=status, capability_mode=mode,
            )
            proposed_state[key] = {"value": result.get("proposed"), "probabilities": result.get("probabilities", {})}
            guardrail[key] = result
            value = result.get("final")
            if value != HOLD_CURRENT:
                final[key] = float(value)
        return {"source_proposals": source, "proposed_state": proposed_state, "guardrail_mapping": guardrail, "final_executable_state": final, "catalog_digest": parameter_catalog_digest(spec), "capability": {"status": status, "mode": mode}}

    def _merge_next_round_updates(
        self,
        *updates: Tuple[str, Mapping[str, Any]],
        apply: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Merge next-round config proposals and record source/override provenance."""
        internal_only_keys = {
            "binder_template",
            "binder_templates",
            "binder_template_proximity",
            "exploit_fragment_modules",
            "module_guided_exploitation",
            "target_include",
            "target_binding_types",
            "structure_groups",
        }
        internal_source_keys = {
            "fragment_template_mining": {
                "binder_template",
                "binder_templates",
                "binder_template_proximity",
                "exploit_fragment_modules",
                "module_guided_exploitation",
                "target_include",
                "target_binding_types",
                "structure_groups",
            },
        }
        internal_sources = set(internal_source_keys)
        merged: Dict[str, Any] = {}
        sources: Dict[str, str] = {}
        decisions: List[Dict[str, Any]] = []
        inputs: Dict[str, Dict[str, Any]] = {}
        ignored_unsupported: Dict[str, List[str]] = {}
        ignored_internal_only: Dict[str, List[str]] = {}
        ownership_conflicts: List[Dict[str, Any]] = []
        merge_order: List[str] = []
        key_owners = {
            "binder_lengths": "binder_length_policy",
        }

        for source, update in updates:
            raw = dict(update or {})
            merge_order.append(source)
            inputs[source] = raw
            include_internal = source in internal_sources
            source_internal_keys = internal_source_keys.get(source, set())
            ignored = unsupported_config_keys(raw, include_internal=include_internal, allowed_keys=self._profile_allowed_keys(include_internal=include_internal))
            internal_ignored = sorted(
                k for k in raw
                if k in internal_only_keys and (not include_internal or k not in source_internal_keys)
            )
            if ignored:
                ignored_unsupported[source] = ignored
            if internal_ignored:
                ignored_internal_only[source] = internal_ignored
            for key, value in supported_config_changes(raw, include_internal=include_internal, allowed_keys=self._profile_allowed_keys(include_internal=include_internal)).items():
                if key in self._sampler_keys() and source != "probabilistic_sampler_final":
                    continue
                if value is None:
                    continue
                if key in internal_only_keys and key not in source_internal_keys:
                    continue
                owner = key_owners.get(key)
                if source == "policy_proposal" and sources.get(key) == "input_configuration":
                    owner = "input_configuration"
                if owner and source not in {owner, "strategy_conflict_resolution"} and sources.get(key) == owner and merged.get(key) != value:
                    ownership_conflicts.append({
                        "key": key,
                        "owner": owner,
                        "rejected_source": source,
                        "rejected_value": value,
                        "kept_value": merged[key],
                    })
                    continue
                decision: Dict[str, Any] = {"key": key, "source": source, "value": value}
                if key in merged and merged[key] != value:
                    decision["overrode"] = {"source": sources.get(key, "unknown"), "value": merged[key]}
                merged[key] = value
                sources[key] = source
                decisions.append(decision)

        proposal_conflicts: List[Dict[str, Any]] = []
        policy_meta = getattr(getattr(self, "_pending_policy_proposal", None), "analysis_metadata", {}) or {}
        for conflict in list(policy_meta.get("proposal_conflicts") or []):
            key = str(conflict.get("key") or "")
            rows = list(conflict.get("proposals") or [])
            if key and rows:
                proposal_conflicts.append({"key": key, "proposals": rows, "resolution": "hold_current", "reason": "diagnostic/quality/hypothesis values disagree"})
                merged.pop(key, None)
                sources.pop(key, None)
                decisions.append({"key": key, "source": "typed_proposal_arbitrator", "action": "hold_current", "proposals": rows})

        merged, freeze_notes = self._freeze_hard_constraints(
            merged,
            allow_harness_target_crop=self._allow_harness_target_crop(merged, sources=sources),
        )
        for note in freeze_notes:
            decisions.append({"key": note["key"], "source": "hard_constraint_guardrail", "value": note.get("kept_value"), "ignored_value": note.get("ignored_value"), "reason": note.get("reason")})

        # Compatibility pass-through: all normalized primary families and safety
        # controls coexist through central merge. Keep the rejection schema empty
        # unless a future explicit compatibility rule is introduced.
        merged, arm_rejections = enforce_single_primary_family(merged, sources=sources)
        for rejection in arm_rejections:
            decisions.append({**rejection, "source": "experiment_arm_guardrail", "rejected_source": rejection.get("source")})
            sources.pop(rejection["key"], None)

        # P0 physical guardrail: clamp numeric search knobs (alpha, exploration_ratio,
        # noise_scale, step_scale, ...) to their hard bounds AND limit per-round change
        # rate vs. the current config. This is the single chokepoint that stops an LLM
        # from pushing alpha 0.001 -> 0.7 (the v4 collapse). Runs AFTER source merge so
        # whichever source "won" a key is still bounded.
        current_snapshot = self._current_config_snapshot()
        sampler_keys = self._sampler_keys()
        profile_bounds = self._profile_param_bounds()
        sampler_exact = {key: merged.pop(key) for key in sampler_keys if key in merged}
        merged, clamp_notes = clamp_config_with_inertia(merged, current_config=current_snapshot, bounds=profile_bounds)
        merged.update(sampler_exact)
        for note in clamp_notes:
            decisions.append({
                "key": note["parameter"],
                "source": "physical_guardrail",
                "value": note["clamped_to"],
                "clamped_from": note["proposed"],
                "clamp_reasons": note["reasons"],
            })

        merged, conflict_notes = self._resolve_pressure_conflicts(merged, sources)
        for note in conflict_notes:
            decisions.append(note)
        if conflict_notes:
            sampler_exact = {key: merged.pop(key) for key in sampler_keys if key in merged}
            merged, second_clamp_notes = clamp_config_with_inertia(merged, current_config=current_snapshot, bounds=profile_bounds)
            merged.update(sampler_exact)
            clamp_notes.extend(second_clamp_notes)
            for note in second_clamp_notes:
                decisions.append({
                    "key": note["parameter"],
                    "source": "physical_guardrail_after_conflict_resolver",
                    "value": note["clamped_to"],
                    "clamped_from": note["proposed"],
                    "clamp_reasons": note["reasons"],
                })

        if apply:
            effective_config = self._apply_next_round_update(merged)
            # The merge result is a delta artifact, not a full runtime snapshot.
            # Executor-derived budget/filter defaults remain in live config and
            # lineage, but must not make an unsupported agent proposal look applied.
            applied = {key: effective_config[key] for key in merged if key in effective_config}
        else:
            applied = dict(merged)
        applied_sources = {key: sources.get(key, "orchestrator_normalization") for key in applied}
        if "binder_lengths" in applied:
            applied_sources.setdefault("binder_lengths", sources.get("binder_lengths", "orchestrator_length_guardrail"))
        return applied, {
            "schema_version": "1.1",
            "merge_order": merge_order,
            "inputs": inputs,
            "ignored_unsupported_keys": ignored_unsupported,
            "ignored_internal_only_keys": ignored_internal_only,
            "ownership_conflicts": ownership_conflicts,
            "typed_proposal_conflicts": proposal_conflicts,
            "experiment_arm_rejections": arm_rejections,
            "decisions": decisions,
            "physical_guardrail_clamps": clamp_notes,
            "pressure_conflict_notes": conflict_notes if 'conflict_notes' in locals() else [],
            "hard_constraint_freeze_notes": freeze_notes,
            "applied_update": dict(applied),
            "applied_sources": applied_sources,
            "normalization_notes": [
                "Later sources override earlier sources except where an explicit owner already resolved the key.",
                "Orchestrator clamps round budget fields to max_binders_per_round and current harness constraints.",
                "Physical guardrail clamps numeric knobs (alpha/exploration_ratio/noise_scale/step_scale) to hard bounds and per-round change-rate limits.",
                "All normalized primary families and safety controls coexist through central merge; experiment_arm_rejections remains empty absent a future explicit compatibility rule.",
                "Preview merge only; live config was not mutated." if not apply else "Applied merge mutated the live next-round config.",
            ],
        }

    def _write_next_round_config(self, path: Path, params_update: Mapping[str, Any]) -> None:
        data = {
            "task_name": self.cfg.task_name,
            "target": asdict(self.cfg.target),
            "search_space": asdict(self.cfg.search_space),
            "active_learning": asdict(self.cfg.active_learning),
            "runtime": asdict(self.cfg.runtime),
            "resource": asdict(self.cfg.resource),
            "applied_params_update": dict(params_update),
        }
        atomic_write_text(path, yaml.safe_dump(data, allow_unicode=True))
        self._artifact_digest_cache.invalidate(path)

    def _target_analysis(self) -> Dict[str, Any]:
        hotspots = tuple(str(value) for value in self._effective_hotspots())
        cache_key = (
            str(self.cfg.target.structure_path),
            str(self.cfg.target.chain_id),
            hotspots,
        )
        cached = self._target_analysis_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        try:
            payload = analyze_target_structure(
                self.cfg.target.structure_path,
                chain_id=self.cfg.target.chain_id,
                hotspots=hotspots,
            ).to_dict()
        except Exception as exc:
            payload = {"error": str(exc), "structure_file": self.cfg.target.structure_path}
        self._target_analysis_cache[cache_key] = dict(payload)
        return payload

    @staticmethod
    def _build_monitor_snapshot(execution_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        statuses: Dict[str, int] = {}
        failed_jobs: List[Dict[str, Any]] = []
        retried_jobs = 0
        for record in execution_records:
            status = str(record.get("status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
            if int(record.get("attempts") or 0) > 1:
                retried_jobs += 1
            if status in BinderDesignOrchestrator.FAILURE_STATUSES:
                failed_jobs.append({
                    "job_id": (record.get("job") or {}).get("job_id"),
                    "status": status,
                    "attempts": record.get("attempts"),
                    "error": sanitize_error_text(record.get("error")),
                })
        return {
            "state": "completed",
            "is_terminal": True,
            "is_success": not failed_jobs,
            "status_counts": statuses,
            "retried_jobs": retried_jobs,
            "failed_jobs": failed_jobs[:10],
            "missing_outputs": [],
            "failure_hints": [item.get("error") for item in failed_jobs[:5] if item.get("error")],
        }

    @staticmethod
    def _collect_structure_files(ingestions: Iterable[Dict[str, Any]]) -> List[str]:
        return list(dict.fromkeys(
            str(path) for item in ingestions for path in item.get("structure_files", [])
            if str(path).lower().endswith((".pdb", ".cif", ".mmcif"))
        ))

    def _analysis_candidates(self, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        filters = [str(item).strip() for item in (self.cfg.search_space.boltzgen or {}).get("additional_filters", []) or [] if str(item).strip()]
        meta: Dict[str, Any] = {
            "filters": filters,
            "input_candidate_count": len(candidates or []),
            "analysis_candidate_count": len(candidates or []),
            "rejected_candidate_count": 0,
            "filtering_applied": False,
            "analysis_scope": "all_candidates",
            "per_filter": [],
        }
        if not filters or not candidates:
            meta["filtering_reason"] = "no_additional_filters" if not filters else "no_candidates"
            return list(candidates or []), meta

        per_row_results: List[List[Optional[bool]]] = []
        filter_stats = [
            {"filter": expr, "metric": self._additional_filter_metric(expr), "pass_count": 0, "fail_count": 0, "unknown_count": 0}
            for expr in filters
        ]
        for row in candidates:
            row_results: List[Optional[bool]] = []
            for idx, expr in enumerate(filters):
                result = self._candidate_passes_additional_filter(row, expr)
                row_results.append(result)
                if result is True:
                    filter_stats[idx]["pass_count"] += 1
                elif result is False:
                    filter_stats[idx]["fail_count"] += 1
                else:
                    filter_stats[idx]["unknown_count"] += 1
            per_row_results.append(row_results)

        evaluable_filter_indexes = [
            idx for idx, stats in enumerate(filter_stats)
            if int(stats.get("pass_count") or 0) + int(stats.get("fail_count") or 0) > 0
        ]
        meta["per_filter"] = filter_stats
        if not evaluable_filter_indexes:
            meta["filtering_reason"] = "additional_filters_not_evaluable_from_candidate_rows"
            return list(candidates or []), meta

        filtered = [
            row for row, results in zip(candidates, per_row_results)
            if all(results[idx] is True for idx in evaluable_filter_indexes)
        ]
        meta.update({
            "analysis_candidate_count": len(filtered),
            "rejected_candidate_count": len(candidates) - len(filtered),
            "filtering_applied": True,
            "analysis_scope": "additional_filters_passed_candidates",
            "filtering_reason": "using_candidates_that_pass_user_additional_filters",
        })
        return filtered, meta

    @staticmethod
    def _additional_filter_metric(expr: str) -> str:
        match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*(?:>=|<=|==|!=|>|<)\s*", str(expr))
        return match.group(1) if match else str(expr)

    @classmethod
    def _candidate_passes_additional_filter(cls, row: Mapping[str, Any], expr: str) -> Optional[bool]:
        metric, op, threshold = cls._parse_additional_filter(expr)
        if not metric:
            return None
        for key in cls._additional_filter_pass_keys(metric):
            parsed = cls._parse_bool(row.get(key))
            if parsed is not None:
                return parsed
        value = cls._candidate_metric_value(row, metric)
        if value is None and metric == "iptm":
            value = cls._candidate_metric_value(row, "design_to_target_iptm")
        if value is None:
            return None
        if op == ">":
            return value > threshold
        if op == ">=":
            return value >= threshold
        if op == "<":
            return value < threshold
        if op == "<=":
            return value <= threshold
        if op == "==":
            return value == threshold
        if op == "!=":
            return value != threshold
        return None

    @staticmethod
    def _parse_additional_filter(expr: str) -> Tuple[str, str, float]:
        match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*(>=|<=|==|!=|>|<)\s*([-+]?\d+(?:\.\d+)?)\s*$", str(expr))
        if not match:
            return "", "", 0.0
        return match.group(1), match.group(2), float(match.group(3))

    @staticmethod
    def _additional_filter_pass_keys(metric: str) -> List[str]:
        clean = re.sub(r"[^A-Za-z0-9]+", "_", str(metric)).strip("_").lower()
        keys = [f"pass_{clean}_filter"] if clean else []
        if "iptm" in clean and "pass_iptm_filter" not in keys:
            keys.append("pass_iptm_filter")
        return keys

    @staticmethod
    def _parse_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return None

    @staticmethod
    def _candidate_metric_value(row: Mapping[str, Any], key: str) -> Optional[float]:
        value = row.get(key)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


    def _guess_binder_chain(self) -> str:
        # BoltzGen output structures are relabelled by entity order; for the
        # harness design spec the generated binder entity is first, usually A.
        return str((self.cfg.search_space.boltzgen or {}).get("output_binder_chain_hint", "A"))

    def _binder_length_hint(self) -> Optional[Union[int, List[int]]]:
        """Binder length hint(s) for BoltzGen output-chain auto-detection."""
        lengths = sorted({int(x) for x in (self.cfg.search_space.binder_lengths or [])})
        if not lengths:
            return None
        return lengths[0] if len(lengths) == 1 else lengths

    def _fragment_template_gate(self) -> str:
        """Metric used to gate which structures may seed reusable templates.

        Default ``interchain_pae`` (design-to-target PAE) captures local interface
        confidence; ``iptm`` is the legacy global complex gate, off by default.
        """
        return str((self.cfg.search_space.boltzgen or {}).get("fragment_template_gate", "interchain_pae"))

    def _fragment_interchain_pae_max(self) -> float:
        try:
            return float((self.cfg.search_space.boltzgen or {}).get("fragment_interchain_pae_max", FragmentTemplateMiningAgent.DEFAULT_INTERCHAIN_PAE_MAX))
        except (TypeError, ValueError):
            return FragmentTemplateMiningAgent.DEFAULT_INTERCHAIN_PAE_MAX

    def _fragment_templates_enabled(self) -> bool:
        value = (self.cfg.search_space.boltzgen or {}).get("fragment_templates_enabled", False)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}

    def _fragment_template_top_k(self) -> int:
        try:
            return max(1, int((self.cfg.search_space.boltzgen or {}).get("fragment_template_top_k", 1) or 1))
        except (TypeError, ValueError):
            return 1

    def _auto_binder_length_enabled(self) -> bool:
        value = (self.cfg.search_space.boltzgen or {}).get("auto_binder_length", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}

    def _binder_length_bounds(self) -> Tuple[Optional[int], Optional[int], int]:
        """Allowed [min, max] binder length envelope plus discretization step.

        When the user pinned a ``binder_length_range`` it is the hard outer bound
        (the agent only redistributes lengths within it). When no range is set
        the bounds are returned as ``None`` so the length policy derives an
        exploration envelope from the current lengths.
        """
        step = int(getattr(self.cfg.search_space, "binder_length_step", 10) or 10)
        rng = self.cfg.search_space.binder_length_range
        if rng is None:
            return None, None, step
        try:
            lengths = _expand_length_range(rng, step)
        except (TypeError, ValueError):
            return None, None, step
        if not lengths:
            return None, None, step
        return min(lengths), max(lengths), step




    def _build_pressure_conflict(self, memory: Any, current_candidates: List[Dict[str, Any]], current_config: Mapping[str, Any]) -> Dict[str, Any]:
        rounds = sorted(getattr(memory, "rounds", []) or [], key=lambda r: getattr(r, "round_id", 0))
        if not rounds:
            return {}
        previous = rounds[-1]
        previous_config = dict(getattr(previous, "config_snapshot", {}) or {})
        previous_eval = dict(getattr(previous, "evaluation", {}) or {})
        previous_rows = self._evaluation_rows(previous_eval)
        previous_stats = self._candidate_metric_stats(previous_rows)
        current_stats = self._candidate_metric_stats(current_candidates)
        regressed, regression_reasons = self._core_metrics_regressed(previous_stats, current_stats)
        if not regressed:
            return {}

        pressure_moves = self._pressure_moves_between(previous_config, current_config)
        if not any(bool(v) for v in pressure_moves.values()):
            return {}

        metrics = {int(m.get("round_id", -1)): m for m in (getattr(memory, "round_metrics", []) or [])}
        best_snapshot = previous_config
        best_key: tuple = ()
        for record in rounds:
            rid = getattr(record, "round_id", None)
            metric = metrics.get(rid) or {}
            key = _stored_round_decision_key(metric)
            if not best_key or key > best_key:
                best_key = key
                best_snapshot = dict(getattr(record, "config_snapshot", {}) or {})
        return {
            "active": True,
            "reason": "core metrics regressed after increasing hotspot/contact pressure",
            "regression_reasons": regression_reasons,
            "previous_stats": previous_stats,
            "last_stats": current_stats,
            "best_round_config": best_snapshot,
            "pressure_moves": {
                **pressure_moves,
            },
            "instruction": "Do not increase hotspot/contact/crop/template pressure next round; revert toward the best RoundRankKey round and explore alternative patch/length/topology.",
        }

    def _pressure_moves_between(self, previous_config: Mapping[str, Any], current_config: Mapping[str, Any]) -> Dict[str, bool]:
        old_fraction = _float_or_none(previous_config.get("template_conditioned_fraction"))
        new_fraction = _float_or_none(current_config.get("template_conditioned_fraction"))
        prev_crop = str(previous_config.get("epitope_crop_mode") or "disabled").strip().lower()
        cur_crop = str(current_config.get("epitope_crop_mode") or "disabled").strip().lower()
        return {
            "auxiliary_hotspots_grew": len(current_config.get("auxiliary_hotspots") or []) > len(previous_config.get("auxiliary_hotspots") or []),
            "crop_tightened": self._crop_mode_disabled(prev_crop) and not self._crop_mode_disabled(cur_crop),
            "filter_bindingsite_enabled": self._has_filter_bindingsite(current_config) and not self._has_filter_bindingsite(previous_config),
            "target_include_narrowed": self._target_include_size(current_config.get("target_include")) < self._target_include_size(previous_config.get("target_include")),
            "target_binding_types_expanded": self._binding_residue_count(current_config.get("target_binding_types")) > self._binding_residue_count(previous_config.get("target_binding_types")),
            "template_conditioned_fraction_increased": old_fraction is not None and new_fraction is not None and new_fraction > old_fraction + 1e-9,
            "binder_lengths_narrowed": self._binder_lengths_narrowed(previous_config.get("binder_lengths"), current_config.get("binder_lengths")),
        }

    @staticmethod
    def _crop_mode_disabled(mode: Any) -> bool:
        return str(mode or "disabled").strip().lower() in {"", "disabled", "off", "none", "false", "0"}

    def _epitope_crop_disabled_hard_constraint(self) -> bool:
        boltzgen = self.cfg.search_space.boltzgen or {}
        return self._crop_mode_disabled(self._initial_epitope_crop_mode) and not bool(boltzgen.get("allow_agent_epitope_crop", False))

    @staticmethod
    def _has_filter_bindingsite(config: Mapping[str, Any]) -> bool:
        truthy_values = {"true", "1", "yes", "on"}
        for item in config.get("config_overrides") or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            for token in item[1:]:
                text = str(token).strip()
                if "=" not in text:
                    continue
                key, value = text.split("=", 1)
                if key.strip().lower() == "filter_bindingsite" and value.strip().lower() in truthy_values:
                    return True
        return False

    @staticmethod
    def _without_filter_bindingsite(config_overrides: Any) -> List[Any]:
        """Remove only exact ``filter_bindingsite=<value>`` setting tokens."""
        cleaned: List[Any] = []
        for item in config_overrides or []:
            if not isinstance(item, (list, tuple)) or not item:
                cleaned.append(item)
                continue
            tokens = list(item)
            kept = [tokens[0]]
            for token in tokens[1:]:
                text = str(token).strip()
                key = text.split("=", 1)[0].strip().lower() if "=" in text else ""
                if key == "filter_bindingsite":
                    continue
                kept.append(token)
            if any("=" in str(token) and str(token).split("=", 1)[0].strip() for token in kept[1:]):
                cleaned.append(kept)
        return cleaned

    @staticmethod
    def _target_include_size(value: Any) -> int:
        total = 0
        for item in value or []:
            chain = (item.get("chain") or {}) if isinstance(item, Mapping) else {}
            res_index = str(chain.get("res_index") or "")
            if ".." in res_index:
                start, end = res_index.split("..", 1)
                try:
                    total += max(0, int(end) - int(start) + 1)
                    continue
                except ValueError:
                    pass
            if res_index:
                total += 1
        return total if total > 0 else 100000

    @staticmethod
    def _binding_residue_count(value: Any) -> int:
        residues = set()
        for item in value or []:
            chain = (item.get("chain") or {}) if isinstance(item, Mapping) else {}
            chain_id = str(chain.get("id") or "")
            for token in str(chain.get("binding") or "").split(","):
                token = token.strip()
                if token:
                    residues.add((chain_id, token))
        return len(residues)

    @staticmethod
    def _binder_lengths_narrowed(previous: Any, current: Any) -> bool:
        try:
            prev = {int(x) for x in (previous or [])}
            cur = {int(x) for x in (current or [])}
        except (TypeError, ValueError):
            return False
        if not prev or not cur:
            return False
        return len(cur) < len(prev) or (cur < prev)

    @staticmethod
    def _evaluation_rows(evaluation: Mapping[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in list(evaluation.get("top_candidates") or []) + list(evaluation.get("failed_examples") or []):
            if not isinstance(item, Mapping):
                continue
            raw = item.get("raw")
            rows.append(dict(raw if isinstance(raw, Mapping) and raw else item))
        return rows

    @staticmethod
    def _candidate_metric_stats(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        return core_metric_stats(candidates)

    def _core_metric_trends(self, memory: Any, current_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        rounds = sorted(getattr(memory, "rounds", []) or [], key=lambda r: getattr(r, "round_id", 0))
        current = self._candidate_metric_stats(current_candidates)
        if not rounds:
            return {"current": current}
        previous_eval = dict(getattr(rounds[-1], "evaluation", {}) or {})
        previous = self._candidate_metric_stats(self._evaluation_rows(previous_eval))
        deltas = {
            key: round(float(current.get(key, 0.0)) - float(previous.get(key, 0.0)), 6)
            for key in current
            if key in previous
        }
        return {"previous": previous, "current": current, "delta": deltas}

    @staticmethod
    def _core_metrics_regressed(prev_stats: Mapping[str, float], cur_stats: Mapping[str, float]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if float(cur_stats.get("best_iptm") or 0.0) < float(prev_stats.get("best_iptm") or 0.0) - 0.03:
            reasons.append("best_iptm_dropped")
        if float(cur_stats.get("mean_iptm") or 0.0) < float(prev_stats.get("mean_iptm") or 0.0) - 0.02:
            reasons.append("mean_iptm_dropped")
        if float(cur_stats.get("best_design_ptm") or 0.0) < float(prev_stats.get("best_design_ptm") or 0.0) - 0.03:
            reasons.append("best_design_ptm_dropped")
        if float(cur_stats.get("mean_design_ptm") or 0.0) < float(prev_stats.get("mean_design_ptm") or 0.0) - 0.03:
            reasons.append("mean_design_ptm_dropped")
        if float(cur_stats.get("best_min_pae") or 100000.0) > float(prev_stats.get("best_min_pae") or 100000.0) + 1.0:
            reasons.append("best_min_pae_worsened")
        if float(cur_stats.get("best_refold_rmsd") or 100000.0) > float(prev_stats.get("best_refold_rmsd") or 100000.0) + 0.75:
            reasons.append("best_refold_rmsd_worsened")
        return bool(reasons), reasons

    @staticmethod
    def _best_candidate_iptm(candidates: List[Dict[str, Any]]) -> float:
        values = []
        for c in candidates or []:
            try:
                values.append(float(c.get("design_to_target_iptm") or c.get("iptm") or 0.0))
            except (TypeError, ValueError):
                continue
        return max(values) if values else 0.0

    @staticmethod
    def _topk_candidate_iptm_median(candidates: List[Dict[str, Any]], top_k: int = 5) -> float:
        """Median iPTM over the top-k candidates of the round (robust round signal)."""
        values = []
        for c in candidates or []:
            try:
                values.append(float(c.get("design_to_target_iptm") or c.get("iptm") or 0.0))
            except (TypeError, ValueError):
                continue
        if not values:
            return 0.0
        top = sorted(values, reverse=True)[: max(1, int(top_k))]
        from binderloop.active_learning.rollback import median_of
        return median_of(top)

    def _load_prior_strategy_exposure(self, round_id: int) -> Dict[str, Any]:
        if int(round_id) <= 0:
            return {}
        path = self.out_dir / ("round_%02d" % (int(round_id) - 1)) / "next_strategy_exposure.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return dict(payload) if isinstance(payload, Mapping) else {}
        except Exception:
            return {}

    @staticmethod
    def _collect_learned_rule_ids(**payloads: Any) -> List[str]:
        found = set()

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if str(key) in {"learned_rule_ids", "selected_learned_rule_ids", "selected_rule_ids"}:
                        if isinstance(item, list):
                            found.update(str(token) for token in item if str(token))
                        elif item:
                            found.add(str(item))
                    else:
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(payloads)
        return sorted(found)

    @staticmethod
    def _collect_learned_nonuse_reasons(**payloads: Any) -> List[str]:
        found = set()

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if str(key) == "learned_skill_nonuse_reason" and item:
                        found.add(str(item))
                    else:
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(payloads)
        return sorted(found)

    def _build_self_improvement_evidence(
        self,
        *,
        round_id: int,
        memory: Any,
        current_jobs: Sequence[DesignJob],
        current_config: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        structural_analysis: Mapping[str, Any],
        outcome: Mapping[str, Any],
        rollback: Mapping[str, Any],
    ) -> Dict[str, Any]:
        structural = dict(structural_analysis or {})
        summaries = list(structural.get("summaries") or [])
        structure_phenotype = {
            "aggregate_tags": dict(structural.get("aggregate_tags") or {}),
            "reliable_seed_fraction": structural.get("reliable_seed_fraction"),
            "total_structures": structural.get("total_structures"),
            "interface_data_quality": dict(structural.get("interface_data_quality") or {}),
            "observed_binder_chain_count": len({
                str(item.get("binder_chain"))
                for item in summaries
                if item.get("binder_chain")
            }),
            "observed_target_chain_count": max(
                [len(item.get("target_chains") or []) for item in summaries] or [0]
            ),
        }
        recent_rounds = []
        for record in list(getattr(memory, "rounds", []) or [])[-max(
            1, int(getattr(self.cfg.self_improvement, "recent_round_window", 5) or 5)
        ):]:
            recent_rounds.append({
                "round_id": getattr(record, "round_id", None),
                "reward": getattr(record, "reward", None),
                "config_snapshot": dict(getattr(record, "config_snapshot", {}) or {}),
                "rollback": dict(getattr(record, "rollback_decision", {}) or {}),
            })
        return {
            "schema_version": "1.0",
            "round_id": int(round_id),
            "strategy_exposure": self._load_prior_strategy_exposure(round_id),
            "executed_arm_signature": self._round_arm_signature(list(current_jobs)),
            "current_config": dict(current_config or {}),
            "binder_length_range": self.cfg.search_space.binder_length_range,
            "outcome": dict(outcome or {}),
            "rollback": dict(rollback or {}),
            "evaluation": {
                "metric_facts": dict(evaluation.get("metric_facts") or {}),
                "core_metric_trends": dict(evaluation.get("core_metric_trends") or {}),
                "core_metric_stats": dict(evaluation.get("core_metric_stats") or {}),
                "tag_counts": dict(evaluation.get("tag_counts") or {}),
                "candidate_filtering": dict(evaluation.get("candidate_filtering") or {}),
            },
            "structure_phenotype": structure_phenotype,
            "tuning_feedback": self._build_tuning_feedback(memory, current_config),
            "pressure_conflict": dict(self._latest_pressure_conflict or {}),
            "recent_rounds": recent_rounds,
            "evidence_digest": stable_hash({
                "round_id": round_id,
                "outcome": outcome,
                "evaluation": evaluation.get("metric_facts"),
                "structure_phenotype": structure_phenotype,
                "prior_exposure": self._load_prior_strategy_exposure(round_id),
            }),
        }

    def _build_tuning_feedback(self, memory: Any, current_config: Mapping[str, Any]) -> Dict[str, Any]:
        """P1 closed-loop feedback: report how the LLM's previous numeric tuning
        moves affected the reward, so the LLM can avoid repeating harmful moves.

        Compares the two most recent rounds: for each guarded numeric knob, report
        the value change and whether the reward went up or down. If reward dropped
        after raising a knob, emit an explicit "do not repeat / revert" penalty.
        Also reports the best-reward round so the LLM can revert toward it.
        """
        from binderloop.agents.config_parameter_contract import PARAM_BOUNDS
        guarded_bounds = self._profile_param_bounds() or PARAM_BOUNDS
        rounds = sorted(getattr(memory, "rounds", []) or [], key=lambda r: getattr(r, "round_id", 0))
        metrics = {int(m.get("round_id", -1)): m for m in (getattr(memory, "round_metrics", []) or [])}
        if len(rounds) < 1 or not metrics:
            return {}

        # Build best-so-far reference by RoundRankKey (reward is legacy fallback).
        best_round_id, best_reward, best_snapshot = None, float("-inf"), {}
        best_key: tuple = ()
        for r in rounds:
            rid = getattr(r, "round_id", None)
            metric = metrics.get(rid) or {}
            rew = float(metric.get("reward") or 0.0)
            key = _stored_round_decision_key(metric)
            if not best_key or key > best_key:
                best_key = key
                best_reward, best_round_id = rew, rid
                best_snapshot = dict(getattr(r, "config_snapshot", {}) or {})

        prev = rounds[-1]
        prev_id = getattr(prev, "round_id", None)
        prev_reward = float((metrics.get(prev_id) or {}).get("reward") or 0.0)
        prev_snapshot = dict(getattr(prev, "config_snapshot", {}) or {})
        prev2_reward = None
        if len(rounds) >= 2:
            p2id = getattr(rounds[-2], "round_id", None)
            prev2_reward = float((metrics.get(p2id) or {}).get("reward") or 0.0)

        penalties: List[Dict[str, Any]] = []
        if prev2_reward is not None:
            reward_delta = prev_reward - prev2_reward
            prev2_snapshot = dict(getattr(rounds[-2], "config_snapshot", {}) or {})
            for key in guarded_bounds:
                old_v, new_v = prev2_snapshot.get(key), prev_snapshot.get(key)
                if old_v is None or new_v is None:
                    continue
                try:
                    old_f, new_f = float(old_v), float(new_v)
                except (TypeError, ValueError):
                    continue
                if abs(new_f - old_f) < 1e-9:
                    continue
                direction = "increased" if new_f > old_f else "decreased"
                if reward_delta < -1e-6:
                    penalties.append({
                        "parameter": key,
                        "previous_move": f"{direction} {old_f} -> {new_f}",
                        "reward_change": round(reward_delta, 6),
                        "instruction": f"This move was followed by a reward DROP. Do NOT repeat it; revert {key} toward {old_f} (or the best-round value).",
                    })

        pressure_conflict: Dict[str, Any] = {}
        if len(rounds) >= 2:
            prev2 = rounds[-2]
            prev2_snapshot = dict(getattr(prev2, "config_snapshot", {}) or {})
            prev_eval = dict(getattr(prev, "evaluation", {}) or {})
            prev2_eval = dict(getattr(prev2, "evaluation", {}) or {})
            prev_rows = self._evaluation_rows(prev_eval)
            prev2_rows = self._evaluation_rows(prev2_eval)
            prev_stats = self._candidate_metric_stats(prev_rows)
            prev2_stats = self._candidate_metric_stats(prev2_rows)
            regressed, regression_reasons = self._core_metrics_regressed(prev2_stats, prev_stats)
            pressure_moves = self._pressure_moves_between(prev2_snapshot, prev_snapshot)
            if regressed and any(bool(v) for v in pressure_moves.values()):
                pressure_conflict = {
                    "active": True,
                    "reason": "core metrics regressed after increasing hotspot/contact pressure",
                    "regression_reasons": regression_reasons,
                    "previous_stats": prev2_stats,
                    "last_stats": prev_stats,
                    "best_round_config": dict(best_snapshot),
                    "pressure_moves": pressure_moves,
                    "instruction": "Do not increase hotspot/contact/crop/template pressure next round; revert toward the best RoundRankKey round and explore alternative patch/length/topology.",
                }

        return {
            "best_round_id": best_round_id,
            "best_reward": round(best_reward, 6),
            "best_round_config": {k: best_snapshot.get(k) for k in guarded_bounds if k in best_snapshot},
            "last_round_id": prev_id,
            "last_round_reward": round(prev_reward, 6),
            "reward_trend": "down" if (prev2_reward is not None and prev_reward < prev2_reward) else "up_or_flat",
            "penalized_moves": penalties,
            "pressure_conflict": pressure_conflict,
            "note": "Penalized moves dropped reward last time. Prefer reverting guarded knobs toward best_round_config unless new evidence justifies otherwise.",
        }

    @staticmethod
    def _round_arm_signature(jobs: List[DesignJob]) -> str:
        arms = sorted({str(job.params.get("exploration_arm", "")) for job in jobs if (job.params or {}).get("exploration_arm")})
        return ";".join(arms)

    # Markers that indicate an infrastructure / configuration failure rather
    # than a genuine design-quality outcome.  When a round produced zero
    # candidates AND every executed job failed with one of these reasons, the
    # round must be excluded from reward/rollback quality accounting.
    EXECUTION_FAILURE_NEEDLES = (
        "boltzgen_config_error",
        "pre-submit config validation failed",
        "taiji_client start failed",
        "missing_ceph_mount_secret",
        "invalid config",
        "invalid simple config",
        "missing_boltzgen_cli",
        "missing_input_file",
        "conda_env_error",
        "taiji_resource_or_queue_issue",
        "pending timeout",
        "resource exhausted",
        "no available resource",
        "quota",
        "queued",
        "evicted",
        "permission denied",
        "unauthorized",
        "forbidden",
    )

    PRE_SUBMIT_SUMMARY_SCHEMA_VERSION = 1

    @staticmethod
    def _pre_submit_issue_is_blocking(issue: Mapping[str, Any]) -> bool:
        return str(issue.get("severity") or "").lower() == "error" and not bool(issue.get("resolved"))

    @classmethod
    def _build_pre_submit_summary(
        cls, round_id: int, jobs: List[DesignJob], execution_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ordered_records = cls._records_for_jobs(
            jobs, execution_records, label="pre_submit_summary_record", allow_legacy_order=True,
        )
        job_rows: List[Dict[str, Any]] = []
        backend_counts: Dict[str, int] = {}
        submit_status_counts: Dict[str, int] = {}
        diff_classification_counts = {"normalization": 0, "removal": 0, "addition": 0, "backend_override": 0, "unchanged": 0}
        total_normalizations = total_removals = total_additions = total_blocking = 0
        submittable_count = refinalization_count = 0
        artifact_count = 0
        artifact_paths: List[str] = []
        schema_version_counts: Dict[str, Dict[str, int]] = {
            "execution_record": {}, "pre_submit": {}, "validation": {}, "diff": {},
        }
        for job, record in zip(jobs, ordered_records):
            pre_submit = dict(record.get("pre_submit") or {})
            validation = dict(pre_submit.get("validation") or record.get("config_validation") or {})
            diff = dict(pre_submit.get("diff") or {})
            issues = [dict(item) for item in (validation.get("issues") or []) if isinstance(item, Mapping)]
            blocking_issues = [item for item in issues if cls._pre_submit_issue_is_blocking(item)]
            missing_required = sorted(str(value) for value in (validation.get("missing_required_keys") or []) if str(value))
            normalizations = dict(diff.get("normalization") or {})
            removals = list(diff.get("metadata_stripping") or validation.get("removals") or [])
            additions = dict(diff.get("validator_additions") or {})
            backend_overrides = dict(diff.get("backend_overrides") or pre_submit.get("backend_overrides") or {})
            classifications: List[str] = []
            for name, values in (("normalization", normalizations), ("removal", removals), ("addition", additions), ("backend_override", backend_overrides)):
                if values:
                    classifications.append(name)
                    diff_classification_counts[name] += 1
            if not classifications:
                classifications.append("unchanged")
                diff_classification_counts["unchanged"] += 1
            backend = str(record.get("backend") or pre_submit.get("backend") or "unknown")
            submit_status = str(record.get("submit_status") or record.get("status") or "unknown")
            is_submittable = bool(pre_submit.get("is_submittable", validation.get("is_submittable", validation.get("is_valid", False))))
            requires_refinalization = bool(pre_submit.get("requires_refinalization", validation.get("requires_refinalization", False)))
            artifact_path = str(pre_submit.get("artifact") or "")
            if artifact_path:
                artifact_count += 1
                artifact_paths.append(artifact_path)
            versions = {
                "execution_record": record.get("schema_version"),
                "pre_submit": pre_submit.get("schema_version"),
                "validation": validation.get("schema_version"),
                "diff": diff.get("schema_version"),
            }
            for contract, version in versions.items():
                label = "unversioned" if version is None else str(version)
                schema_version_counts[contract][label] = schema_version_counts[contract].get(label, 0) + 1
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            submit_status_counts[submit_status] = submit_status_counts.get(submit_status, 0) + 1
            submittable_count += int(is_submittable)
            refinalization_count += int(requires_refinalization)
            total_normalizations += len(normalizations)
            total_removals += len(removals)
            total_additions += len(additions)
            total_blocking += len(blocking_issues)
            params = dict(job.params or {})
            job_rows.append({
                "job_id": job.job_id,
                "arm_id": str(params.get("arm_id") or params.get("exploration_arm") or "legacy"),
                "backend": backend,
                "attempt": int(record.get("attempt") or 0),
                "is_submittable": is_submittable,
                "blocking_issues": blocking_issues,
                "blocking_issue_count": len(blocking_issues),
                "missing_required_keys": missing_required,
                "diff_classifications": classifications,
                "normalizations": normalizations,
                "removals": removals,
                "validator_additions": additions,
                "requires_refinalization": requires_refinalization,
                "submit_status": submit_status,
                "artifact": artifact_path,
                "schema_versions": versions,
            })
        return {
            "schema_version": cls.PRE_SUBMIT_SUMMARY_SCHEMA_VERSION,
            "round_id": int(round_id),
            "job_count": len(job_rows),
            "arm_count": len({row["arm_id"] for row in job_rows}),
            "backend_counts": backend_counts,
            "submit_status_counts": submit_status_counts,
            "is_submittable_count": submittable_count,
            "blocked_job_count": len(job_rows) - submittable_count,
            "blocking_issue_count": total_blocking,
            "missing_required_job_count": sum(bool(row["missing_required_keys"]) for row in job_rows),
            "diff_classification_counts": diff_classification_counts,
            "normalization_count": total_normalizations,
            "removal_count": total_removals,
            "validator_addition_count": total_additions,
            "requires_refinalization_count": refinalization_count,
            "validation_artifact_count": artifact_count,
            "validation_artifact_paths": artifact_paths,
            "schema_version_counts": schema_version_counts,
            "artifact": {"schema_version": 1, "path": "", "job_artifact_count": artifact_count, "job_artifact_paths": artifact_paths},
            "jobs": job_rows,
        }

    def _load_checkpoint_pre_submit_summary(self, checkpoint: Mapping[str, Any], round_dir: Path) -> Optional[Dict[str, Any]]:
        embedded = checkpoint.get("pre_submit_summary")
        if isinstance(embedded, Mapping):
            return dict(embedded)
        candidates = [checkpoint.get("pre_submit_summary_path"), round_dir / "pre_submit_summary.json"]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate))
            if not path.is_absolute() and not path.exists():
                path = round_dir / path
            if not path.exists():
                continue
            try:
                payload = self._load_json_path(path)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @classmethod
    def _classify_execution_state(cls, jobs: List[DesignJob], execution_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        ordered_records = cls._records_for_jobs(
            jobs, execution_records, label="execution_state_record", allow_legacy_order=True,
        )
        successful: List[str] = []
        failed: List[str] = []
        for job, record in zip(jobs, ordered_records):
            (failed if str(record.get("status") or "").lower() in cls.FAILURE_STATUSES else successful).append(job.job_id)
        total = len(jobs)
        return {"state": "complete" if total and not failed else "partial", "complete": bool(total and not failed), "quality_complete": False, "realized_fraction": len(successful) / total if total else 0.0, "successful_job_ids": successful, "failed_job_ids": failed, "successful_branch_ids": successful, "failed_branch_ids": failed, "expected_job_count": total, "realized_job_count": len(successful), "expected_branch_count": total, "realized_branch_count": len(successful)}

    def _observe_llm_fallback(self, round_dir: Path, round_id: int, agent_name: str, result: Any, checkpoint: Dict[str, Any]) -> None:
        if bool(getattr(result, "llm_used", False)):
            return
        raw = dict(getattr(result, "raw", {}) or {})
        reason = str(raw.get("fallback_reason") or raw.get("source") or raw.get("parse_error") or "deterministic_fallback")
        payload = {"event": "llm_deterministic_fallback", "severity": "WARNING", "round_id": round_id, "agent": agent_name, "llm_used": False, "reason": reason}
        artifact = round_dir / "llm_fallbacks.json"
        existing: List[Dict[str, Any]] = []
        if artifact.exists():
            try:
                loaded = json.loads(artifact.read_text(encoding="utf-8")); existing = loaded if isinstance(loaded, list) else []
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if not any(item.get("agent") == agent_name for item in existing):
            existing.append(payload); self._write_json(artifact, existing)
        self._append_artifact(checkpoint, artifact)
        warnings.warn(f"WARNING: {agent_name} used deterministic fallback: {reason}", RuntimeWarning, stacklevel=2)
        self.bus.publish(AgentMessage(agent_name, "all", "warning", payload, round_id=round_id, artifacts=[str(artifact)]))

    @classmethod
    def _execution_failure_reason(cls, error: str) -> str:
        lowered = str(error or "").lower()
        if any(needle in lowered for needle in cls.RESOURCE_SCHEDULING_FAILURE_NEEDLES):
            return "resource_scheduling_failure"
        for needle in cls.EXECUTION_FAILURE_NEEDLES:
            if needle in lowered:
                return needle
        return ""

    @classmethod
    def _detect_round_execution_failure(
        cls,
        *,
        total_candidates: int,
        execution_records: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Classify whether a round failed for infrastructure/config reasons.

        Returns ``(execution_failed, reason)``.  A round is flagged as an
        execution failure only when it produced **zero** usable candidates AND
        at least one executed job terminated with a recognised configuration /
        infrastructure error.  This deliberately avoids flagging a round that
        simply produced low-quality (but real) designs — that is a genuine
        quality signal and must still feed reward/rollback.
        """
        if int(total_candidates or 0) > 0:
            matched_reasons: List[str] = []
            for record in execution_records or []:
                status = str(record.get("status") or "").lower()
                if status not in cls.FAILURE_STATUSES:
                    continue
                if str(record.get("backend") or "").lower() != "taiji":
                    continue
                error = str(record.get("error") or "").lower()
                reason = cls._execution_failure_reason(error)
                if not reason:
                    reason = "taiji_task_failed"
                if reason:
                    matched_reasons.append(reason)
            # Successful branch evidence makes the round quality-evaluable;
            # partial failures are represented by execution_state.
            return False, ""
        matched_reasons: List[str] = []
        for record in execution_records or []:
            status = str(record.get("status") or "").lower()
            if status not in cls.FAILURE_STATUSES:
                continue
            error = str(record.get("error") or "").lower()
            reason = cls._execution_failure_reason(error)
            if reason:
                matched_reasons.append(reason)
        if matched_reasons:
            # De-duplicate while preserving order for a readable reason string.
            seen: List[str] = []
            for reason in matched_reasons:
                if reason not in seen:
                    seen.append(reason)
            return True, ";".join(seen)
        return False, ""

    def _seed_rollback_history(self, memory: Any) -> None:
        outcomes = []
        for metric in getattr(memory, "round_metrics", []) or []:
            # Execution/config-failure rounds were excluded from reward/rollback
            # accounting when first observed; keep them excluded on resume so the
            # best-reward baseline and regression counter stay consistent.
            if bool(metric.get("execution_failed")):
                continue
            try:
                outcomes.append(RoundOutcome(
                    round_id=int(metric.get("round_id")),
                    reward=float(metric.get("reward") or 0.0),
                    best_iptm=float(metric.get("best_iptm") or 0.0),
                    median_iptm=float(metric.get("median_iptm") or 0.0),
                    core_objective=float(metric.get("core_objective") or metric.get("reward") or 0.0),
                    core_metric_stats=dict(metric.get("core_metric_stats") or {}),
                    round_rank_key=[
                        float(value) for value in (metric.get("round_rank_key") or [])
                    ],
                    success_count=int(metric.get("success_count") or 0),
                    arm_signature=str(metric.get("arm_signature") or ""),
                    branch_id=str(metric.get("branch_id") or ""), config_digest=str(metric.get("config_digest") or ""),
                    intervention_digest=str(metric.get("intervention_digest") or ""), is_baseline=bool(metric.get("is_baseline")),
                    strict_successes=int(metric.get("strict_successes") or 0), strict_trials=int(metric.get("strict_trials") or 0),
                    raw_candidate_count=int(metric.get("raw_candidate_count") or metric.get("strict_trials") or 0),
                    analysis_candidate_count=int(metric.get("analysis_candidate_count") or 0),
                    raw_strict_yield=float(metric.get("raw_strict_yield") or 0.0),
                    conditional_strict_yield=float(metric.get("conditional_strict_yield") or 0.0),
                    execution_failed=bool(metric.get("execution_failed")),
                    execution_failure_reason=str(metric.get("execution_failure_reason") or ""),
                ))
            except (TypeError, ValueError):
                continue
        if outcomes:
            self.rollback.seed_history(outcomes, best_config_retests=getattr(memory.experiment_ledger, "best_config_retests", {}))

    @staticmethod
    def _record_round_metric(memory: Any, outcome: RoundOutcome) -> None:
        metrics = [m for m in (memory.round_metrics or []) if int(m.get("round_id", -1)) != outcome.round_id]
        metrics.append(outcome.to_dict())
        metrics.sort(key=lambda m: int(m.get("round_id", 0)))
        memory.round_metrics = metrics

    def _retrieve_memory_summary(
        self,
        memory: Any,
        *,
        fallback_summary: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        current_config: Mapping[str, Any],
        current_jobs: List[DesignJob],
    ) -> Dict[str, Any]:
        if not self.memory_retrieval_agent or not getattr(memory, "memory_items", None):
            return dict(fallback_summary)
        tag_counts = dict(evaluation.get("tag_counts") or {})
        failure_tags = [
            key for key, value in sorted(tag_counts.items(), key=lambda pair: float(pair[1] or 0), reverse=True)
            if value and not str(key).startswith("pass_")
        ][:5]
        previous_records = [
            record for record in (getattr(memory, "rounds", []) or [])
            if getattr(record, "config_snapshot", None)
        ]
        previous_config = dict(previous_records[-1].config_snapshot) if previous_records else {}
        executable_previous = supported_config_changes(previous_config, include_internal=True)
        executable_current = supported_config_changes(dict(current_config), include_internal=True)
        changed = parameter_diff(
            executable_previous,
            executable_current,
            allowed_keys=set(executable_previous) | set(executable_current),
        )
        query = MemoryRetrievalQuery(
            target=asdict(self.cfg.target),
            failure_tags=failure_tags,
            arm=self._round_arm_signature(current_jobs),
            parameter_names=sorted(changed),
            intent="Explain current failures and choose evidence-backed next-round parameter changes.",
        )
        result = self.memory_retrieval_agent.retrieve(memory.memory_items, query)
        summary = self.memory_store.summarize_for_agent(
            memory,
            extend_memory=bool(getattr(self.cfg.runtime, "extend_memory", False)),
            recalled_items=result.items,
        )
        summary["retrieval"] = {
            "structured_candidate_count": result.structured_candidate_count,
            "selected_count": len(result.items),
            "semantic_rerank_used": result.semantic_rerank_used,
            "cache_hit": result.cache_hit,
            "selected_scores": result.selected_scores,
            "rerank_reasons": result.rerank_reasons,
            "query": {
                "target_key": query.target_key,
                "failure_tags": query.failure_tags,
                "arm": query.arm,
                "parameter_names": query.parameter_names,
            },
        }
        return summary

    def _index_and_compress_round_memory(
        self,
        memory: Any,
        *,
        round_id: int,
        evaluation: Mapping[str, Any],
        outcome: Mapping[str, Any],
        current_config: Mapping[str, Any],
        artifact_refs: List[str],
    ) -> None:
        if not self.memory_index_enabled and not self.memory_compression_agent:
            return
        if self.memory_index_enabled:
            previous_records = sorted(
                [
                    record for record in (getattr(memory, "rounds", []) or [])
                    if int(getattr(record, "round_id", -1)) < int(round_id)
                    and getattr(record, "config_snapshot", None)
                ],
                key=lambda record: int(record.round_id),
            )
            previous_config = dict(previous_records[-1].config_snapshot) if previous_records else {}
            executable_previous = supported_config_changes(previous_config, include_internal=True)
            executable_current = supported_config_changes(dict(current_config), include_internal=True)
            config_diff = parameter_diff(
                executable_previous,
                executable_current,
                allowed_keys=set(executable_previous) | set(executable_current),
            )
            prior_metrics = sorted(
                [
                    metric for metric in (getattr(memory, "round_metrics", []) or [])
                    if int(metric.get("round_id", -1)) < int(round_id)
                    and not bool(metric.get("execution_failed"))
                ],
                key=lambda metric: int(metric.get("round_id", -1)),
            )
            previous_reward = float(prior_metrics[-1].get("reward")) if prior_metrics and prior_metrics[-1].get("reward") is not None else None
            failure_tags = [
                str(key) for key, value in (dict(evaluation.get("tag_counts") or {})).items()
                if value and not str(key).startswith("pass_")
            ]
            item = build_round_memory_item(
                round_id=round_id,
                target=asdict(self.cfg.target),
                failure_tags=failure_tags,
                config_diff=config_diff,
                arm=str(outcome.get("arm_signature") or ""),
                outcome=outcome,
                artifact_refs=artifact_refs,
                previous_reward=previous_reward,
            )
            self.memory_store.upsert_memory_item(memory, item)
        if self.memory_compression_agent:
            compression = self.memory_compression_agent.compress_to_budget(memory.memory_items)
            memory.memory_items = compression.items
            if compression.compressed_items:
                self.memory_store.append_event("memory_compressed", {
                    "round_id": round_id,
                    "before_active_count": compression.before_active_count,
                    "after_active_count": compression.after_active_count,
                    "compressed_item_ids": [value.item_id for value in compression.compressed_items],
                    "archived_item_ids": compression.archived_item_ids,
                    "llm_used": compression.llm_used,
                })

    def _backfill_indexed_memory(self, memory: Any) -> None:
        """Migrate durable v1 round records into v2 evidence cards on resume."""
        if not self.memory_index_enabled:
            return
        metrics_by_round = {
            int(metric.get("round_id", -1)): dict(metric)
            for metric in (getattr(memory, "round_metrics", []) or [])
        }
        previous_config: Dict[str, Any] = {}
        previous_reward: Optional[float] = None
        for record in sorted(
            getattr(memory, "rounds", []) or [],
            key=lambda value: int(getattr(value, "round_id", -1)),
        ):
            round_id = int(getattr(record, "round_id", -1))
            if round_id < 0 or not getattr(record, "evaluation", None):
                continue
            current_config = supported_config_changes(
                dict(getattr(record, "config_snapshot", {}) or {}),
                include_internal=True,
            )
            config_diff = parameter_diff(
                previous_config,
                current_config,
                allowed_keys=set(previous_config) | set(current_config),
            )
            outcome = dict(metrics_by_round.get(round_id) or {})
            if "reward" not in outcome:
                outcome["reward"] = getattr(record, "reward", None)
            evaluation = dict(getattr(record, "evaluation", {}) or {})
            failure_tags = [
                str(key) for key, value in dict(evaluation.get("tag_counts") or {}).items()
                if value and not str(key).startswith("pass_")
            ]
            arm = str(outcome.get("arm_signature") or "")
            if not arm:
                arm = self._round_arm_signature(self._jobs_from_dicts(getattr(record, "jobs", []) or []))
            item = build_round_memory_item(
                round_id=round_id,
                target=dict(getattr(memory, "target", {}) or asdict(self.cfg.target)),
                failure_tags=failure_tags,
                config_diff=config_diff,
                arm=arm,
                outcome=outcome,
                artifact_refs=list(getattr(record, "artifacts", []) or []),
                previous_reward=previous_reward,
            )
            self.memory_store.upsert_memory_item(memory, item)
            if not bool(outcome.get("execution_failed")) and outcome.get("reward") is not None:
                previous_reward = float(outcome["reward"])
            previous_config = current_config
        if self.memory_compression_agent:
            compression = self.memory_compression_agent.compress_to_budget(memory.memory_items)
            memory.memory_items = compression.items
            if compression.compressed_items:
                self.memory_store.append_event("memory_backfill_compressed", {
                    "before_active_count": compression.before_active_count,
                    "after_active_count": compression.after_active_count,
                    "compressed_item_ids": [value.item_id for value in compression.compressed_items],
                    "archived_item_ids": compression.archived_item_ids,
                    "llm_used": compression.llm_used,
                })

    def _restore_exact_config_snapshot(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        """Replace live generation state with a best-round snapshot.

        This intentionally replaces the complete BoltzGen dictionary instead of
        merging it, so optional keys introduced on a degraded branch disappear.
        User-frozen target/range/budget constraints remain authoritative.
        """
        source = copy.deepcopy(dict(snapshot or {}))
        profile = self._search_profile()
        nested_model = source.get("model_config") or source.get(f"{profile.search_space_attr}_config")
        nested_boltzgen = source.get("boltzgen_config")
        if profile.model == "rfd3" and isinstance(nested_model, Mapping):
            restored_model = copy.deepcopy(dict(nested_model))
        elif profile.model != "rfd3" and isinstance(nested_boltzgen, Mapping):
            restored_model = copy.deepcopy(dict(nested_boltzgen))
        elif isinstance(nested_boltzgen, Mapping):
            restored_model = copy.deepcopy(dict(nested_boltzgen))
        else:
            restored_model = {
                key: copy.deepcopy(source[key])
                for key in profile.restore_keys
                if key in source
            }
        restored_model.pop("secondary_structure", None)
        restored_model["run_filtering"] = True

        allowed_lengths = set(self._allowed_binder_lengths())
        requested_lengths = [int(value) for value in (source.get("binder_lengths") or [])]
        if allowed_lengths and any(value not in allowed_lengths for value in requested_lengths):
            raise RuntimeError("Best-round binder_lengths conflict with the current user hard range")
        if requested_lengths:
            self.cfg.search_space.binder_lengths = requested_lengths

        freeze_target = bool(getattr(self.cfg.task, "freeze_target_definition", True))
        target_snapshot = source.get("target") if isinstance(source.get("target"), Mapping) else {}
        if freeze_target:
            self.cfg.target.include = copy.deepcopy(list(self._original_target_include or []))
            self.cfg.target.binding_types = copy.deepcopy(list(self._original_target_binding_types or []))
            self.cfg.target.structure_groups = copy.deepcopy(self._original_structure_groups)
        else:
            self.cfg.target.include = copy.deepcopy(list(
                source.get("target_include", target_snapshot.get("include", self.cfg.target.include)) or []
            ))
            self.cfg.target.binding_types = copy.deepcopy(list(
                source.get("target_binding_types", target_snapshot.get("binding_types", self.cfg.target.binding_types)) or []
            ))
            self.cfg.target.structure_groups = copy.deepcopy(
                source.get("structure_groups", target_snapshot.get("structure_groups", self.cfg.target.structure_groups))
            )
            self.cfg.target.hotspots = copy.deepcopy(list(
                source.get("hotspots", target_snapshot.get("hotspots", self.cfg.target.hotspots)) or []
            ))

        self.cfg.search_space.boltzgen = restored_model if profile.model != "rfd3" else dict(self.cfg.search_space.boltzgen or {})
        if profile.model == "rfd3":
            self.cfg.search_space.rfd3 = restored_model
        elif profile.search_space_attr not in {"boltzgen", "rfd3"}:
            setattr(self.cfg.search_space, profile.search_space_attr, restored_model)
        else:
            self.cfg.search_space.boltzgen = restored_model
        self.cfg.search_space.boltzgen.setdefault("target_include", copy.deepcopy(list(self.cfg.target.include or [])))
        self.cfg.search_space.boltzgen.setdefault("target_binding_types", copy.deepcopy(list(self.cfg.target.binding_types or [])))
        if self.cfg.target.structure_groups is None:
            self.cfg.search_space.boltzgen.pop("structure_groups", None)
        else:
            self.cfg.search_space.boltzgen["structure_groups"] = copy.deepcopy(self.cfg.target.structure_groups)
        if self._epitope_crop_disabled_hard_constraint():
            self.cfg.search_space.boltzgen["epitope_crop_mode"] = "disabled"
        if self._crop_mode_disabled(self.cfg.search_space.boltzgen.get("epitope_crop_mode", "disabled")):
            self._restore_original_target_definition()

        if "exploration_ratio" in source:
            self.cfg.active_learning.exploration_ratio = float(source["exploration_ratio"])
            self.learner.exploration_ratio = self.cfg.active_learning.exploration_ratio
        if "top_k" in source:
            self.cfg.active_learning.top_k = max(1, int(source["top_k"]))
        if "max_rounds" in source:
            self.cfg.active_learning.max_rounds = max(1, int(source["max_rounds"]))

        # Exact user round budget always wins over historical or agent values.
        self.cfg.search_space.num_designs_per_round = self._round_num_designs
        self.cfg.search_space.max_binders_per_round = self._round_design_cap
        space = getattr(self.cfg.search_space, profile.search_space_attr)
        space["num_designs"] = self._round_num_designs
        space["num_designs_per_round"] = self._round_num_designs
        space["max_binders_per_round"] = self._round_design_cap
        self.cfg.search_space.boltzgen["num_designs"] = self._round_num_designs
        self.cfg.search_space.boltzgen["num_designs_per_round"] = self._round_num_designs
        self.cfg.search_space.boltzgen["max_binders_per_round"] = self._round_design_cap

        restored_snapshot = self._current_config_snapshot()
        return {
            **copy.deepcopy(dict(space or {})),
            "binder_lengths": copy.deepcopy(list(self.cfg.search_space.binder_lengths or [])),
            "binder_length_range": copy.deepcopy(self.cfg.search_space.binder_length_range),
            "target_include": copy.deepcopy(list(self.cfg.target.include or [])),
            "target_binding_types": copy.deepcopy(list(self.cfg.target.binding_types or [])),
            "structure_groups": copy.deepcopy(self.cfg.target.structure_groups),
            "exploration_ratio": self.cfg.active_learning.exploration_ratio,
            "top_k": self.cfg.active_learning.top_k,
            "max_rounds": self.cfg.active_learning.max_rounds,
            "boltzgen_config": copy.deepcopy(restored_snapshot["boltzgen_config"]),
        }

    def _prepare_exact_rollback_replay(
        self,
        memory: Any,
        decision: Any,
        *,
        next_round_id: int,
    ) -> Tuple[List[DesignJob], Dict[str, Any], Dict[str, Any], List[str]]:
        """Restore the best config; clone jobs for a capped retest or branch seed."""
        best_round = int(decision.best_round)
        record = next((r for r in memory.rounds if int(r.round_id) == best_round), None)
        if record is None or not record.jobs:
            raise RuntimeError(f"Cannot exactly replay best round {best_round}: round jobs are missing")
        best_snapshot = copy.deepcopy(dict(record.config_snapshot or {}))
        if not best_snapshot:
            raise RuntimeError(f"Cannot exactly replay best round {best_round}: config snapshot is missing")

        applied_update = self._restore_exact_config_snapshot(best_snapshot)
        source_jobs = self._jobs_from_dicts(record.jobs)
        source_job_ids = [job.job_id for job in source_jobs]
        replay_jobs: List[DesignJob] = []
        stale_execution_keys = {
            "execution_retry_source_job_id",
            "execution_retry_preserve_budget",
            "resource_retry_degradation",
            "multi_taiji_host_shard",
            "taiji_submit_host_num",
            "native_taiji_multi_host",
            "execution_result",
            "retry_metadata",
        }
        for index, source_job in enumerate(source_jobs):
            params = copy.deepcopy(dict(source_job.params or {}))
            if params.get("template_conditioned") or params.get("binder_template"):
                stored_identity = params.get("template_execution_identity")
                if not isinstance(stored_identity, Mapping):
                    stored_identity = build_template_execution_identity(
                        params,
                        target_structure=source_job.target_structure,
                        target_chain=source_job.chain_id,
                        output_dir=source_job.output_dir,
                    )
                stored_lineage = dict(stored_identity.get("lineage") or (stored_identity.get("semantic") or {}).get("lineage") or {})
                current_identity = build_template_execution_identity(
                    params,
                    target_structure=source_job.target_structure,
                    target_chain=source_job.chain_id,
                    output_dir=str(self.out_dir / f"round_{next_round_id:02d}"),
                    lineage_schema_version=stored_lineage.get("schema_version"),
                    lineage_manifest_digest=str(stored_lineage.get("manifest_digest") or ""),
                )
                replay = classify_template_replay(stored_identity, current_identity)
                params["template_execution_identity"] = current_identity
                params["template_replay_classification"] = replay
                if replay["status"] != "exact_replay":
                    raise RuntimeError(
                        f"Cannot exactly replay template job {source_job.job_id}: "
                        f"{replay['status']}:{replay['reason']}"
                    )
            for key in stale_execution_keys:
                params.pop(key, None)
            blocked = [
                name.strip()
                for name in str(getattr(decision, "blocked_arm_signature", "") or "").split(";")
                if name.strip()
            ]
            if blocked:
                params["blocked_strategy_arms"] = sorted(set(blocked))
            recovery_name = "retest_best_config" if decision.action in {"replay_best", "retest_best_config"} else "branch_seed"
            replay_id = f"round{next_round_id:02d}_{recovery_name}_{index:02d}"
            replay_jobs.append(DesignJob(
                job_id=replay_id,
                target_structure=source_job.target_structure,
                chain_id=source_job.chain_id,
                hotspots=copy.deepcopy(list(source_job.hotspots or [])),
                binder_length=int(source_job.binder_length),
                seed=int(source_job.seed),
                params=params,
                output_dir=str(self.out_dir / f"round_{next_round_id:02d}" / replay_id),
            ))

        replay_jobs = self._finalize_semantic_job_identities(
            replay_jobs, round_id=next_round_id,
        )
        for replay_job, source_job in zip(replay_jobs, source_jobs):
            replay_job.params["replay_source_job_id"] = source_job.job_id
            replay_job.params["replay_source_job_identity_digest"] = self._job_identity_digest(source_job)

        # Run the normal range guard on copies and reject incompatible durable
        # state rather than silently changing a best-round replay strategy.
        range_checked = self._enforce_binder_length_range(copy.deepcopy(replay_jobs))
        for stored, checked in zip(replay_jobs, range_checked):
            if (
                stored.binder_length != checked.binder_length
                or stored.params.get("binder_lengths") != checked.params.get("binder_lengths")
            ):
                raise RuntimeError(
                    f"Best-round replay job {stored.job_id} conflicts with the current "
                    "user binder-length hard range"
                )
        if len(replay_jobs) > self._round_design_cap:
            raise RuntimeError("Best-round replay has more logical jobs than the user round budget")
        requested_budget = sum(int(job.params.get("num_designs", 0) or 0) for job in replay_jobs)
        if requested_budget != self._round_design_cap:
            raise RuntimeError(
                f"Best-round replay budget mismatch: stored={requested_budget}, "
                f"required={self._round_design_cap}"
            )
        self._assert_normal_round_budget(replay_jobs)
        replay_jobs = self._split_multi_host_taiji_jobs(replay_jobs)
        restored_snapshot = self._current_config_snapshot()
        return replay_jobs, applied_update, restored_snapshot, source_job_ids

    def _prepare_rollback_seed(self, memory: Any, decision: Any, regressed_update: Mapping[str, Any]) -> Tuple[List[DesignJob], Dict[str, Any]]:
        """Compatibility helper returning the exact best baseline without divergence."""
        del regressed_update
        best_round = int(decision.best_round)
        record = next((r for r in memory.rounds if int(r.round_id) == best_round), None)
        if record is None or not record.jobs or not record.config_snapshot:
            raise RuntimeError(f"Cannot exactly replay best round {best_round}: durable state is incomplete")
        update = self._restore_exact_config_snapshot(record.config_snapshot)
        return self._jobs_from_dicts(record.jobs), update

    @staticmethod
    def _build_metrics_summary(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build compact metrics summary for the DiagnosticCoachAgent."""
        if not candidates:
            return {"count": 0}
        iptm_values = []
        plddt_values = []
        for c in candidates:
            iptm = _float_or_none(c.get("design_to_target_iptm") or c.get("iptm"))
            plddt = _float_or_none(c.get("design_ptm") or c.get("plddt"))
            if iptm is not None and math.isfinite(iptm):
                iptm_values.append(iptm)
            if plddt is not None and math.isfinite(plddt):
                plddt_values.append(plddt)

        def stats(vals):
            if not vals:
                return {"min": 0, "max": 0, "mean": 0}
            return {"min": round(min(vals), 4), "max": round(max(vals), 4), "mean": round(sum(vals) / len(vals), 4)}

        summary = {
            "count": len(candidates),
            "iptm": stats(iptm_values),
            "core_metric_stats": core_metric_stats(candidates),
            "any_iptm_above_0.3": any(v > 0.3 for v in iptm_values),
            "any_iptm_above_0.4": any(v > 0.4 for v in iptm_values),
        }
        if plddt_values:
            summary["plddt"] = stats(plddt_values)
            summary["plddt_above_0.7_fraction"] = sum(1 for v in plddt_values if v > 0.7) / len(plddt_values)
        return summary


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _stored_round_decision_key(metric: Mapping[str, Any]) -> tuple:
    rank = metric.get("round_rank_key")
    if isinstance(rank, (list, tuple)) and rank:
        try:
            return tuple(float(value) for value in rank)
        except (TypeError, ValueError):
            pass
    return (float(metric.get("reward") or 0.0),)
