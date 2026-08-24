
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import random

from binderloop.models.base import DesignJob
from binderloop.analysis.core_objective import core_rank_key
from binderloop.analysis.scoring import CandidateScore
from binderloop.templates.length_mapping import plan_length_transform
from binderloop.templates.residue_identity import ResidueIdentity


@dataclass
class StrategyProposal:
    round_id: int
    jobs: List[DesignJob]
    rationale: str
    requested_branch_width: int = 1
    realized_branch_width: int = 1
    excluded_arms: Dict[str, str] = None
    degraded: bool = False
    filtering_report: Dict[str, Any] = None


@dataclass(frozen=True)
class StrategyArmDefinition:
    name: str
    family: str
    branch_role: str
    intervention: Mapping[str, Any]
    requires_templates: bool = False


CANONICAL_STRATEGY_ARM_CATALOG: Dict[str, StrategyArmDefinition] = {
    "baseline_hold": StrategyArmDefinition("baseline_hold", "control", "control", {"kind": "hold"}),
    "site_primary_condition": StrategyArmDefinition("site_primary_condition", "binding_site", "repair", {"kind": "binding_site", "positive_scope": "primary"}),
    "site_expanded_condition": StrategyArmDefinition("site_expanded_condition", "binding_site", "repair", {"kind": "binding_site", "positive_scope": "primary_expanded"}),
    "site_negative_exclusion": StrategyArmDefinition("site_negative_exclusion", "binding_site", "repair", {"kind": "binding_site", "negative_scope": "off_patch"}),
    "target_context_focus": StrategyArmDefinition("target_context_focus", "target_context", "repair", {"kind": "target_context", "mode": "focus"}),
    "sampler_explore": StrategyArmDefinition("sampler_explore", "sampling", "exploration", {"kind": "sampling", "direction": "explore"}),
    "template_exploit": StrategyArmDefinition("template_exploit", "template", "exploitation", {"kind": "template", "mode": "structure_redesign"}, requires_templates=True),
    "sequence_repair": StrategyArmDefinition("sequence_repair", "sequence", "repair", {"kind": "sequence", "mode": "repair"}),
}

DEPRECATED_STRATEGY_KEYS = ("hotspot_weight", "prioritize_hotspots", "clash_filter", "module_guided_repair", "module_guided_exploitation", "exploit_fragment_modules")


def canonical_strategy_arm(name: str) -> Dict[str, Any]:
    definition = CANONICAL_STRATEGY_ARM_CATALOG.get(str(name))
    if definition is None or not definition.intervention:
        return {}
    return {"name": definition.name, "family": definition.family, "branch_role": definition.branch_role, "intervention": dict(definition.intervention), "requires_templates": bool(definition.requires_templates), "params": {}}


def effective_intervention_digest(arm: Mapping[str, Any], params: Mapping[str, Any], hotspots: Sequence[Any]) -> str:
    payload = {"arm": str(arm.get("name") or ""), "intervention": dict(arm.get("intervention") or {}), "binding_site_policy": params.get("binding_site_policy"), "target_context_policy": params.get("target_context_policy"), "selection_policy": params.get("selection_policy"), "template": params.get("binder_template"), "hotspots": [str(value) for value in hotspots or []]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StrategyLevelActiveLearner:
    """Strategy-level active learning over model/constraint choices, not weights."""

    def __init__(self, exploration_ratio: float = 0.3, seed: int = 0):
        self.exploration_ratio = exploration_ratio
        self.rng = random.Random(seed)

    def initial_jobs(self, target_structure: str, chain_id: str, hotspots: List[str], lengths: List[int], output_dir: str, base_params: Dict, *, branch_width: int = 1) -> List[DesignJob]:
        """Create a deterministic, closed-catalog multi-arm baseline round."""
        clean_lengths = sorted({int(x) for x in (lengths or [])}) or [int(base_params.get("binder_length", 100))]
        params = dict(base_params)
        params["binder_lengths"] = list(clean_lengths)
        parent = DesignJob(
            job_id="r0_seed", target_structure=target_structure, chain_id=chain_id,
            hotspots=list(hotspots), binder_length=clean_lengths[len(clean_lengths) // 2],
            params=params, output_dir=f"{output_dir}/r0/seed",
        )
        # These arms are executable without prior candidates/templates. Binding-site
        # and target-context intents are materialized by the normal resolver. A
        # sampler arm is intentionally not used before a legal final sampler state exists.
        initial_order = ["baseline_hold", "site_primary_condition", "target_context_focus", "site_expanded_condition"]
        return self.propose_next(
            0, [parent], [], output_dir, branch_width=max(1, int(branch_width)),
            hypotheses=[{"failure_modes": ["hotspot_miss", "binding_pose_failure"]}],
            selection_context={"strict_positive_count": 0, "failure_tag_counts": {"hotspot_miss": 1, "binding_pose_failure": 1}},
            ranked_arm_names=initial_order, enable_exploitation_arms=False,
        ).jobs

    def propose_next(self, round_id: int, previous_jobs: List[DesignJob], scores: Iterable[CandidateScore], output_dir: str, top_k: int = 8, *, policy_update: Optional[Mapping[str, Any]] = None, structural_summary: Optional[Any] = None, hypotheses: Optional[List[Mapping[str, Any]]] = None, quality_analysis: Optional[Mapping[str, Any]] = None, blocked_arms: Optional[Iterable[str]] = None, blocked_arm_combinations: Optional[Sequence[Sequence[str]]] = None, pressure_conflict: Optional[Mapping[str, Any]] = None, active_skills: Optional[Sequence[Mapping[str, Any]]] = None, branch_width: int = 1, enable_exploitation_arms: bool = True, selection_context: Optional[Mapping[str, Any]] = None, ranked_arm_names: Optional[Sequence[str]] = None, defer_branch_width: bool = False) -> StrategyProposal:
        del scores
        branch_width = max(1, int(branch_width))
        requested_branch_width = branch_width
        excluded_arms: Dict[str, str] = {}
        ranked: List[Any] = []
        parents = list(previous_jobs[:1])
        fallback_parent = previous_jobs[0] if previous_jobs else None
        arms = self.candidate_arms(
            structural_summary=structural_summary,
            hypotheses=list(hypotheses or []),
            quality_analysis=quality_analysis,
            pressure_conflict=pressure_conflict,
            active_skills=active_skills,
            enable_exploitation_arms=enable_exploitation_arms,
            selection_context=selection_context,
        )
        validated_parents = list(previous_jobs[:1])
        # Branch rollback: a rolled-back / failed round reports the arm signature
        # ("arm_a;arm_b") that produced the bad branch. We must NOT re-run those
        # same arms next round (that is the whole point of "回退分支" — switch the
        # branch, not just the round). Prune blocked arms; if every arm was
        # blocked, hold the safe baseline rather than inventing an unvalidated move.
        blocked = {name.strip() for name in (blocked_arms or []) if str(name).strip()}
        if blocked:
            excluded_arms.update({str(arm.get("name")): "soft_blocked_arm" for arm in arms if arm.get("name") in blocked})
            kept = [arm for arm in arms if arm["name"] not in blocked]
            if kept:
                arms = kept
            else:
                arms = []
        if fallback_parent is not None and any((fallback_parent.params or {}).get(key) for key in ("binder_template", "binder_templates")):
            if not any(arm.get("name") == "template_exploit" for arm in arms):
                template_arm = canonical_strategy_arm("template_exploit")
                template_arm.update({"deterministic_priority": 75, "trigger_evidence": ["effective template payload on parent"], "expected_effect": "run a real structure-redesign template branch", "risk": "template overfitting"})
                arms.insert(0, template_arm)
        if ranked_arm_names:
            by_name = {str(arm.get("name")): arm for arm in arms}
            reordered = [by_name[name] for name in ranked_arm_names if name in by_name]
            reordered.extend(arm for arm in arms if arm not in reordered)
            arms = reordered
        update = dict(policy_update or {})
        deprecated_update = {key: update.pop(key) for key in DEPRECATED_STRATEGY_KEYS if key in update}
        # Select executable arms, then supplement from the canonical catalog.
        # A rollback blocks an exact combination, not every member by implication.
        blocked_combinations = {tuple(sorted(str(v) for v in combo if str(v))) for combo in (blocked_arm_combinations or [])}
        selected_arms: List[Dict[str, Any]] = []
        seen_names = set()
        template_available = bool(fallback_parent and ((fallback_parent.params or {}).get("binder_template") or (fallback_parent.params or {}).get("binder_templates")))
        ordered_pool = list(arms)
        for catalog_name in CANONICAL_STRATEGY_ARM_CATALOG:
            if catalog_name in blocked or any(str(arm.get("name")) == catalog_name for arm in ordered_pool):
                continue
            supplement = canonical_strategy_arm(catalog_name)
            supplement.update({"deterministic_priority": -1, "trigger_evidence": ["canonical executable supplement"], "expected_effect": "restore a distinct legal comparison arm", "risk": "low evidence priority"})
            ordered_pool.append(supplement)
        if ranked_arm_names:
            rank_index = {str(name): index for index, name in enumerate(ranked_arm_names)}
            original_index = {id(arm): index for index, arm in enumerate(ordered_pool)}
            ordered_pool.sort(key=lambda arm: (rank_index.get(str(arm.get("name") or ""), len(rank_index) + original_index[id(arm)]), original_index[id(arm)]))
        for candidate_arm in ordered_pool:
            name = str(candidate_arm.get("name") or "")
            if not name or name in seen_names:
                continue
            if candidate_arm.get("requires_templates") and not template_available:
                excluded_arms[name] = "requires_templates"
                continue
            trial_names = tuple(sorted([*(str(item.get("name")) for item in selected_arms), name]))
            if len(trial_names) == requested_branch_width and trial_names in blocked_combinations:
                excluded_arms[name] = "blocked_arm_combination"
                continue
            selected_arms.append(candidate_arm); seen_names.add(name)
            if not defer_branch_width and len(selected_arms) == requested_branch_width:
                break
        if not selected_arms:
            # Last-resort, reproducible sampler state is materialized by the orchestrator.
            fallback = canonical_strategy_arm("sampler_explore")
            fallback.update({"params": {"random_sampler_fallback": True, "random_sampler_seed": int(round_id)}, "trigger_evidence": ["no executable strategy arms"], "expected_effect": "sample one legal non-parent sampler state", "risk": "unguided exploration"})
            selected_arms = [fallback]
        branch_width = len(selected_arms)
        jobs: List[DesignJob] = []
        rationale_bits: List[str] = []
        for arm_index, arm in enumerate(selected_arms):
            if arm["name"] == "template_exploit":
                base_parent = validated_parents[0] if validated_parents else fallback_parent
            elif parents:
                base_parent = parents[-1] if arm["name"] == "sampler_explore" and len(parents) > 1 else parents[0]
            else:
                base_parent = fallback_parent
            params = dict(base_parent.params) if base_parent else {}
            legacy_audit = dict(params.get("deprecated_strategy_audit") or {})
            for key in DEPRECATED_STRATEGY_KEYS:
                if key in params:
                    legacy_audit[key] = {"value": params.pop(key), "schema_version": "1.0", "status": "deprecated_audit_only"}
            for key, value in deprecated_update.items():
                legacy_audit[key] = {"value": value, "schema_version": "1.0", "status": "deprecated_audit_only"}
        # Drop stale per-job bookkeeping copied from the parent.
            for stale_key in (
                "round_budget_allocation",
                "round_budget_weight",
                "binder_length_guardrail",
                "search_arm",
                "template_conditioned",
                "binder_template_dropped",
                "branch_id",
                "multi_taiji_host_shard",
                "native_taiji_multi_host",
                "taiji_host_num_requested",
                "taiji_submit_host_num",
                "execution_retry_source_job_id",
                "execution_retry_preserve_budget",
                "blocked_strategy_arms",
            ):
                params.pop(stale_key, None)
            if arm["name"] != "baseline_hold":
                params.update(update)
            params.update(arm.get("params", {}))
            params["arm_id"] = arm["name"]
            params["exploration_arm"] = arm["name"]
            params["arm_rank"] = arm_index
            params["logical_branch_id"] = f"r{round_id}_{arm['name']}"
            params["strategy_intent"] = dict(arm.get("intervention") or {})
            params["branch_id"] = f"r{round_id}_{arm['name']}"
            if legacy_audit:
                params["deprecated_strategy_audit"] = legacy_audit
            self._apply_arm_intent(params, arm)
            template_specs = list(params.get("binder_templates") or [])
            if not template_specs and params.get("binder_template"):
                template_specs = [params.get("binder_template")]
            template_active = bool(template_specs)
            params["template_conditioned"] = template_active

            # Resolve the round's set of binder lengths (policy update first,
            # then strategy skill hint, then parent).
            avoid_lengths = {int(x) for x in update.get("avoid_binder_lengths", []) or []}
            allowed_lengths = sorted({int(x) for x in update.get("binder_lengths", []) or []})
            if not allowed_lengths:
                allowed_lengths = sorted({int(x) for x in arm.get("binder_lengths", []) or []})
            if not allowed_lengths and base_parent is not None:
                allowed_lengths = sorted({int(x) for x in (base_parent.params.get("binder_lengths") or [base_parent.binder_length])})
            if not allowed_lengths:
                allowed_lengths = [int(params.get("binder_length", base_parent.binder_length if base_parent else 100))]
            allowed_lengths = [length for length in allowed_lengths if length not in avoid_lengths] or allowed_lengths
            params["binder_lengths"] = allowed_lengths
            representative = allowed_lengths[len(allowed_lengths) // 2]

            job_hotspots = list(base_parent.hotspots if base_parent else [])
            if update.get("auxiliary_hotspots"):
                params["expanded_binding_residues"] = list(update.get("auxiliary_hotspots") or [])
            base_job_kwargs = {
                "target_structure": base_parent.target_structure if base_parent else "",
                "chain_id": base_parent.chain_id if base_parent else "",
                "hotspots": job_hotspots,
                "binder_length": representative,
            }
            if arm.get("requires_templates") and not template_active:
                continue
            params["effective_intervention_digest"] = effective_intervention_digest(arm, params, job_hotspots)
            if template_active and arm["name"] == "template_exploit":
                template = dict(template_specs[0])
                params["binder_template"] = template
                params.pop("binder_templates", None)
                params["template_conditioned"] = True
                params["template_count"] = 1
                params["round_budget_weight"] = 1.0
                jobs.append(DesignJob(
                    job_id=f"r{round_id}_{arm['name']}",
                    params=params,
                    output_dir=f"{output_dir}/r{round_id}/arms/pending_{arm_index:02d}_{arm['name']}",
                    **base_job_kwargs,
                ))
            else:
                for key in ("binder_template", "binder_templates", "binder_template_proximity"):
                    if arm["name"] != "template_exploit":
                        params.pop(key, None)
                params["template_conditioned"] = False
                params.setdefault("round_budget_weight", float(arm.get("round_budget_weight", 1.0)))
                jobs.append(DesignJob(
                    job_id=f"r{round_id}_{arm['name']}",
                    params=params,
                    output_dir=f"{output_dir}/r{round_id}/arms/pending_{arm_index:02d}_{arm['name']}",
                    **base_job_kwargs,
                ))
            rationale_bits.append(f"{arm['name']} lengths={allowed_lengths}")
        jobs = self._deduplicate_effective_jobs(jobs)
        if not jobs and fallback_parent is not None:
            hold = self._hold_arm("all effective interventions were unavailable or duplicate")
            hold_params = dict(fallback_parent.params or {})
            for key in DEPRECATED_STRATEGY_KEYS:
                hold_params.pop(key, None)
            hold_params.update({"exploration_arm": "baseline_hold", "strategy_intent": dict(hold["intervention"]), "branch_id": f"r{round_id}_baseline_hold", "template_conditioned": False})
            hold_params["effective_intervention_digest"] = effective_intervention_digest(hold, hold_params, fallback_parent.hotspots)
            jobs = [DesignJob(f"r{round_id}_round", fallback_parent.target_structure, fallback_parent.chain_id, list(fallback_parent.hotspots), fallback_parent.binder_length, params=hold_params, output_dir=f"{output_dir}/r{round_id}/round")]
            selected_arms = [hold]
            rationale_bits = ["baseline_hold fallback"]
        rationale = (
            "Strategy-level AL round: primary arm="
            + selected_arms[0]["name"]
            + f"; logical_arms={len(selected_arms)}"
            + "; arms=[" + "; ".join(rationale_bits) + "]"
            + (f"; branch rollback blocked arms={sorted(blocked)} (forced arm switch)" if blocked else "")
        )
        report = {"schema_version": 1, "round_id": int(round_id), "requested_branch_width": requested_branch_width, "ranked_arms": [str(arm.get("name") or "") for arm in ordered_pool], "pre_materialization_exclusions": dict(excluded_arms)}
        return StrategyProposal(round_id, jobs, rationale, requested_branch_width, len(jobs), excluded_arms, len(jobs) < requested_branch_width, report)

    def candidate_arms(
        self,
        *,
        structural_summary: Optional[Any],
        hypotheses: List[Mapping[str, Any]],
        quality_analysis: Optional[Mapping[str, Any]] = None,
        pressure_conflict: Optional[Mapping[str, Any]] = None,
        active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
        enable_exploitation_arms: bool = True,
        selection_context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build evidence-triggered arms before an LLM or fallback ranks them."""
        arms = self._strategy_arms(
            structural_summary=structural_summary,
            hypotheses=hypotheses,
            quality_analysis=quality_analysis,
            pressure_conflict=pressure_conflict,
            active_skills=active_skills,
            enable_exploitation_arms=enable_exploitation_arms,
            selection_context=selection_context,
        )
        return arms or [self._hold_arm("no validated intervention was triggered")]

    def _select_parents(self, previous_jobs: List[DesignJob], ranked: List[CandidateScore], *, top_k: int) -> List[DesignJob]:
        if not previous_jobs:
            return []
        n = max(1, min(top_k, len(previous_jobs)))
        exploit_n = max(1, int(round(n * (1.0 - self.exploration_ratio))))
        parents = self._parents_from_ranked_candidates(previous_jobs, ranked, limit=exploit_n)
        if len(parents) < exploit_n:
            parents.extend([j for j in previous_jobs if j not in parents][: exploit_n - len(parents)])
        remaining = [j for j in previous_jobs if j not in parents]
        self.rng.shuffle(remaining)
        parents.extend(remaining[: max(0, n - len(parents))])
        return parents

    @staticmethod
    def _parents_from_ranked_candidates(previous_jobs: List[DesignJob], ranked: List[CandidateScore], *, limit: int) -> List[DesignJob]:
        selected: List[DesignJob] = []
        for candidate in ranked:
            evidence = _candidate_evidence_text(candidate)
            if not evidence:
                continue
            for job in previous_jobs:
                if job in selected:
                    continue
                job_tokens = [job.job_id, job.output_dir]
                if any(token and str(token) in evidence for token in job_tokens):
                    selected.append(job)
                    break
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _rank_parent_candidates(scores: List[Any]) -> List[Any]:
        if not scores:
            return []

        def raw_metrics(candidate: Any) -> Mapping[str, Any]:
            raw = getattr(candidate, "raw", None)
            if isinstance(raw, Mapping):
                return raw
            metrics = getattr(candidate, "metrics", None)
            if isinstance(metrics, Mapping):
                return metrics
            if isinstance(candidate, Mapping):
                raw = candidate.get("raw")
                if isinstance(raw, Mapping):
                    return raw
                metrics = candidate.get("metrics")
                if isinstance(metrics, Mapping):
                    return metrics
                return candidate
            return {}

        return sorted(scores, key=lambda candidate: core_rank_key(raw_metrics(candidate)), reverse=True)

    def _strategy_arms(self, *, structural_summary: Optional[Any], hypotheses: List[Mapping[str, Any]], quality_analysis: Optional[Mapping[str, Any]] = None, pressure_conflict: Optional[Mapping[str, Any]] = None, active_skills: Optional[Sequence[Mapping[str, Any]]] = None, enable_exploitation_arms: bool = True, selection_context: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        failure_modes = {str(mode) for hypothesis in hypotheses for mode in (hypothesis.get("failure_modes") or [])}
        tags = getattr(structural_summary, "aggregate_tags", {}) or (structural_summary.get("aggregate_tags", {}) if isinstance(structural_summary, dict) else {})
        context = dict(selection_context or {})
        failure_counts = dict(context.get("failure_tag_counts") or {})
        conflict_active = bool((pressure_conflict or {}).get("active"))
        arms: List[Dict[str, Any]] = []

        def add(name: str, priority: int, evidence: str, expected: str, risk: str) -> None:
            arm = canonical_strategy_arm(name)
            if not arm:
                return
            arm.update({"deterministic_priority": priority, "trigger_evidence": [evidence], "expected_effect": expected, "risk": risk})
            arms.append(arm)

        hotspot_failure = "hotspot_miss" in failure_modes or bool(tags.get("hotspot_not_covered")) or int(failure_counts.get("hotspot_miss") or 0) > 0
        _clash_failure = "clash" in failure_modes or bool(tags.get("interface_clash_risk")) or int(failure_counts.get("clash") or 0) > 0
        pose_failure = "binding_pose_failure" in failure_modes or bool(tags.get("weak_or_tiny_interface")) or int(failure_counts.get("binding_pose_failure") or 0) > 0
        diversity_collapse = "diversity_collapse" in failure_modes or bool(tags.get("diversity_collapse")) or int(failure_counts.get("diversity_collapse") or 0) > 0
        folding_failure = "folding_failure" in failure_modes or bool(tags.get("binder_chain_break")) or int(failure_counts.get("folding_failure") or 0) > 0
        seq_stats = dict((context.get("core_metric_stats") or {}).get("sequence_designability") or {})
        low_seq = False
        try:
            mean_seq = seq_stats.get("mean")
            if mean_seq is not None and float(mean_seq) < 0.4:
                low_seq = True
        except (TypeError, ValueError):
            low_seq = False
        if int(failure_counts.get("sequence_designability") or 0) > 0:
            low_seq = True
        if hotspot_failure and not conflict_active:
            add("site_primary_condition", 90, "primary hotspot coverage failure", "condition on the user primary binding residues", "binary conditioning can reduce diversity")
            add("site_expanded_condition", 80, "evidence-supported nearby residues may recover coverage", "add validated expanded residues at equal BINDING strength", "expanded coverage must not mask primary misses")
        if pose_failure and not conflict_active:
            add("target_context_focus", 70, "weak or high-PAE binding pose", "focus the real target context when user crop constraints allow", "cropping can exclude productive context")
        strict_positive_count = int(context.get("strict_positive_count") or 0)
        min_positives = max(1, int(context.get("min_positives_for_exploit") or 2))
        template_available = bool(context.get("effective_templates_available"))
        if not template_available:
            # Direct/library callers may not provide the orchestrator selection
            # context; infer availability from the current quality/template payload.
            quality = dict(quality_analysis or {})
            template_available = bool(
                quality.get("binder_template")
                or quality.get("binder_templates")
                or quality.get("high_quality_modules")
                or "exploit_high_quality" in list(quality.get("guidance") or [])
            )
        if enable_exploitation_arms and template_available and (strict_positive_count >= min_positives or not selection_context) and not conflict_active:
            add("template_exploit", 75, "validated template provenance and strict positives", "run a real structure-redesign template branch", "template overfitting")
        if pose_failure or strict_positive_count == 0 or bool(context.get("plateau")) or diversity_collapse or not arms:
            add("sampler_explore", 50, "pose/diversity failure, plateau, or no strict positives", "explore with sparse resolver-owned sampler changes", "target adherence may fall")
        if folding_failure or low_seq:
            add("sequence_repair", 55, "folding or sequence designability failure", "repair inverse-fold settings with the bound sequence tool", "sequence diversity may drop")
        add("baseline_hold", 0, "closed-catalog control", "preserve the resolved parent without overrides", "uses budget without a directed intervention")
        base = {arm["name"]: arm for arm in arms}
        if diversity_collapse and "sampler_explore" in base:
            params = dict(base["sampler_explore"].get("params") or {})
            params["diversity_collapse"] = True
            base["sampler_explore"]["params"] = params
        skill_arms = self._strategy_skill_arms(active_skills or [], base_arms_by_name=base)
        ordered: List[Dict[str, Any]] = []
        for arm in skill_arms + sorted(arms, key=lambda item: (float(item.get("deterministic_priority") or 0), item["name"]), reverse=True):
            if arm["name"] not in [item["name"] for item in ordered]:
                ordered.append(arm)
        return ordered

    @staticmethod
    def _strategy_skill_arms(active_skills: Sequence[Mapping[str, Any]], *, base_arms_by_name: Optional[Mapping[str, Mapping[str, Any]]] = None) -> List[Dict[str, Any]]:
        arms: List[Dict[str, Any]] = []
        base_arms_by_name = dict(base_arms_by_name or {})
        for skill in active_skills or []:
            if str(skill.get("type")) != "strategy":
                continue
            params = dict(skill.get("params") or {})
            name = str(params.pop("arm_name", None) or skill.get("id") or "strategy_skill_arm")
            suggested_lengths = params.pop("suggested_binder_lengths", None)
            base_arm = dict(base_arms_by_name.get(name) or StrategyLevelActiveLearner._builtin_strategy_arm(name) or {})
            if not base_arm:
                # Strategy skills may rank/configure only the closed, executable
                # arm catalog. Unknown/deleted arm names remain advisory.
                continue
            if "params" in base_arm:
                base_arm["params"] = dict(base_arm.get("params") or {})
            del suggested_lengths, params
            arms.append({
                "name": name,
                "family": base_arm.get("family"),
                "branch_role": base_arm.get("branch_role"),
                "intervention": dict(base_arm.get("intervention") or {}),
                "requires_templates": bool(base_arm.get("requires_templates")),
                "deterministic_priority": int(skill.get("priority") or 0),
                "trigger_evidence": [str(skill.get("trigger_reason") or "strategy skill")],
                "expected_effect": str(skill.get("description") or ""),
                "risk": str(skill.get("risk") or ""),
                "params": dict(base_arm.get("params") or {}),
            })
        return arms

    @staticmethod
    def _builtin_strategy_arm(name: str) -> Dict[str, Any]:
        return canonical_strategy_arm(name)

    @staticmethod
    def _hold_arm(reason: str) -> Dict[str, Any]:
        arm = canonical_strategy_arm("baseline_hold")
        arm.update({"deterministic_priority": 0, "trigger_evidence": [reason], "expected_effect": "hold the resolved baseline", "risk": "uses a round without a directed intervention"})
        return arm

    @staticmethod
    def _range_to_tokens(value: str, chain: str) -> List[str]:
        tokens: List[str] = []
        for part in str(value or "").split(","):
            part = part.strip()
            if not part: continue
            try:
                if ".." in part:
                    lo, hi = (int(x) for x in part.split("..", 1))
                    tokens.extend(ResidueIdentity(chain, i).token for i in range(lo, hi + 1))
                else:
                    tokens.append(ResidueIdentity(chain, int(part)).token)
            except ValueError:
                continue
        return tokens

    @staticmethod
    def _tokens_to_residue_range(tokens: Sequence[str]) -> str:
        values = sorted(int(str(token).split(":", 1)[-1]) for token in tokens)
        if not values:
            return ""
        groups = []
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            groups.append(str(start) if start == previous else f"{start}..{previous}")
            start = previous = value
        groups.append(str(start) if start == previous else f"{start}..{previous}")
        return ",".join(groups)

    @staticmethod
    def _shift_residue_range(value: str, offset: int) -> str:
        shifted: List[str] = []
        for token in str(value or "").split(","):
            token = token.strip()
            if not token:
                continue
            if ".." in token:
                start, end = token.split("..", 1)
                shifted.append(f"{int(start) + int(offset)}..{int(end) + int(offset)}")
            else:
                shifted.append(str(int(token) + int(offset)))
        return ",".join(shifted)

    @staticmethod
    def _apply_arm_intent(params: Dict[str, Any], arm: Mapping[str, Any]) -> None:
        name = str(arm.get("name") or "")
        if name == "site_primary_condition":
            params["binding_site_policy"] = "primary"
        elif name == "site_expanded_condition":
            params["binding_site_policy"] = "primary_expanded"
        elif name == "site_negative_exclusion":
            params["binding_site_policy"] = "primary_negative"
        elif name == "target_context_focus":
            params["target_context_policy"] = "focus"
        elif name == "clash_select":
            params["selection_policy"] = {"type": "cross_chain_heavy_atom_clash", "gate": True, "rank": "ascending"}
        elif name == "sampler_explore":
            params["sampler_policy"] = "explore"
        elif name == "sequence_repair":
            params["sequence_policy"] = "repair"

    @staticmethod
    def _deduplicate_effective_jobs(jobs: Sequence[DesignJob]) -> List[DesignJob]:
        seen = set()
        out: List[DesignJob] = []
        for job in jobs:
            digest = str((job.params or {}).get("effective_intervention_digest") or "")
            if digest and digest in seen:
                continue
            if digest:
                seen.add(digest)
            out.append(job)
        return out



def _candidate_evidence_text(candidate: Any) -> str:
    fields = [
        getattr(candidate, "candidate_id", ""),
        getattr(candidate, "source", ""),
        getattr(candidate, "path", ""),
    ]
    raw = getattr(candidate, "raw", None)
    if isinstance(raw, Mapping):
        fields.extend(str(raw.get(key, "")) for key in ("job_id", "output_dir", "_metrics_file", "design", "name", "candidate_id", "id"))
    return " ".join(str(field) for field in fields if field)
