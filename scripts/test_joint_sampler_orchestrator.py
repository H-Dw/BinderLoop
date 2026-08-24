#!/usr/bin/env python3
"""Orchestrator policy-gate and evidence-provenance regression tests."""

from dataclasses import asdict
from types import SimpleNamespace

import binderloop.orchestration.orchestrator as orchestrator_module
from binderloop.config import TargetSpec
from binderloop.memory import ExperimentMemory
from binderloop.models.base import DesignJob
from binderloop.models.search_profile import get_model_search_profile
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.parameter_decision import (
    JointParameterEvidence,
    ParameterCandidate,
    ParameterDecisionSpec,
    parameter_catalog_digest,
)
from binderloop.strategy_governance import ArmApplicability, effective_semantic_digest


def _spec(mode: str) -> ParameterDecisionSpec:
    return ParameterDecisionSpec(
        alpha_candidates=(1.0, 2.0),
        noise_scale_candidates=(10.0,),
        sampler_axes=("alpha", "noise_scale"),
        joint_evidence_fallback_mode=mode,
    )


def _orchestrator(mode: str, *, memory_target=None, rounds=()):
    target = TargetSpec("current_target.cif", "A", ["A:1"])
    spec = _spec(mode)
    instance = BinderDesignOrchestrator.__new__(BinderDesignOrchestrator)
    instance.cfg = SimpleNamespace(
        target=target,
        search_space=SimpleNamespace(model_order=["boltzgen"]),
    )
    instance._target_identity_digest = "target-artifact-sha"
    instance._active_memory = ExperimentMemory(
        experiment_id="test",
        target=dict(memory_target if memory_target is not None else asdict(target)),
        rounds=list(rounds),
    )
    baseline = DesignJob(
        "baseline",
        target.structure_path,
        target.chain_id,
        list(target.hotspots),
        60,
        params={
            "alpha": 1.0,
            "noise_scale": 10.0,
            "search_profile_model": "boltzgen",
            "sequence_tool": "boltz_ifold",
            "refold_tool": "boltz2",
        },
    )
    return instance, baseline, spec


def _compatible_round(spec: ParameterDecisionSpec, *, model: str = "boltzgen"):
    common = {
        "target_identity_digest": "target-artifact-sha",
        "search_profile_model": model,
        "sequence_tool": "boltz_ifold",
        "refold_tool": "boltz2",
    }
    return {
        "round_id": 1,
        "jobs": [
            {"params": {**common, "arm_id": "baseline_hold", "alpha": 1.0, "noise_scale": 10.0}},
            {"params": {
                **common,
                "arm_id": "sampler_explore_fallback_00",
                "final_parameter_state": {"alpha": 2.0, "noise_scale": 10.0},
                "parameter_catalog_digest": parameter_catalog_digest(spec),
                "random_sampler_fallback": True,
            }},
        ],
        "arm_outcomes": [
            {"arm_id": "baseline_hold", "status": "closed", "requested_budget": 4, "completed_budget": 4, "trials": 4, "successes": 1, "is_baseline": True},
            {"arm_id": "sampler_explore_fallback_00", "status": "closed", "requested_budget": 4, "completed_budget": 4, "trials": 4, "successes": 3},
        ],
    }


def test_off_mode_does_not_read_active_memory():
    instance, baseline, spec = _orchestrator("off")
    instance._active_memory = object()  # would fail if the disabled path inspected it

    mode, evidence, report = instance._joint_sampler_evidence(
        baseline, spec=spec, catalog_digest=parameter_catalog_digest(spec),
    )

    assert mode == "off"
    assert evidence == ()
    assert report["activation_reason"] == "policy_disabled"


def test_wrong_target_memory_is_rejected_before_row_extraction():
    wrong_target = asdict(TargetSpec("different_target.cif", "A", ["A:1"]))
    instance, baseline, spec = _orchestrator("active", memory_target=wrong_target)

    mode, evidence, report = instance._joint_sampler_evidence(
        baseline, spec=spec, catalog_digest=parameter_catalog_digest(spec),
    )

    assert mode == "active"
    assert evidence == ()
    assert report["activation_reason"] == "memory_header_target_mismatch"


def test_same_target_stale_execution_context_is_rejected():
    _, _, initial_spec = _orchestrator("active")
    stale_round = _compatible_round(initial_spec, model="rfd3")
    instance, baseline, spec = _orchestrator("active", rounds=(stale_round,))
    baseline.params.pop("search_profile_model")
    baseline.params.pop("sequence_tool")
    baseline.params.pop("refold_tool")

    _, evidence, report = instance._joint_sampler_evidence(
        baseline, spec=spec, catalog_digest=parameter_catalog_digest(spec),
    )

    assert evidence == ()
    assert report["activation_reason"] == "no_compatible_evidence"


def test_shadow_mode_extracts_compatible_evidence_without_activating_policy():
    _, _, initial_spec = _orchestrator("shadow")
    compatible_round = _compatible_round(initial_spec)
    instance, baseline, spec = _orchestrator("shadow", rounds=(compatible_round,))

    mode, evidence, report = instance._joint_sampler_evidence(
        baseline, spec=spec, catalog_digest=parameter_catalog_digest(spec),
    )

    assert mode == "shadow"
    assert len(evidence) == 2
    assert report["activation_reason"] == "compatible_evidence_available"
    assert report["scope"] == "memory_header_and_job_provenance_match"


def test_shadow_mode_contains_malformed_memory_failure():
    malformed_round = {"round_id": "not-an-integer", "jobs": [], "arm_outcomes": []}
    instance, baseline, spec = _orchestrator("shadow", rounds=(malformed_round,))

    mode, evidence, report = instance._joint_sampler_evidence(
        baseline, spec=spec, catalog_digest=parameter_catalog_digest(spec),
    )

    assert mode == "shadow"
    assert evidence == ()
    assert report["activation_reason"] == "shadow_evidence_extraction_failed"
    assert report["shadow_error_type"] == "ValueError"


def test_shadow_mode_contains_corrupt_memory_header():
    instance, baseline, spec = _orchestrator("shadow")
    instance._active_memory.target = "not-a-mapping"

    _, evidence, report = instance._joint_sampler_evidence(
        baseline, spec=spec, catalog_digest=parameter_catalog_digest(spec),
    )

    assert evidence == ()
    assert report["activation_reason"] == "shadow_memory_header_validation_failed"
    assert report["shadow_error_type"] == "AttributeError"


def test_shadow_scoring_failure_still_materializes_seeded_job(monkeypatch, tmp_path):
    instance, baseline, spec = _orchestrator("shadow")
    instance.cfg.owner = SimpleNamespace(parameter_decision=spec, sampler_bounds=None)
    instance.out_dir = tmp_path
    instance._sampler_keys = spec.active_sampler_keys
    instance._materialize_job_binding_types = lambda jobs: jobs
    instance._materialize_sampler_and_context_intents = lambda jobs: jobs
    evidence = (
        JointParameterEvidence(ParameterCandidate(alpha=2.0, noise_scale=10.0), 3, 4, "row"),
    )
    report = {
        "mode": "shadow",
        "matched_control_groups": 0,
        "shadow_recommendations": [],
    }
    instance._joint_sampler_evidence = lambda *args, **kwargs: ("shadow", evidence, report)

    def deterministic_stub(_spec, *, evidence=(), **kwargs):
        if evidence:
            raise RuntimeError("shadow scorer unavailable")
        return (ParameterCandidate(alpha=2.0, noise_scale=10.0),)

    class EligiblePlan:
        applicability = ArmApplicability.ELIGIBLE
        reason = "test"

        @staticmethod
        def to_dict():
            return {"applicability": ArmApplicability.ELIGIBLE.value, "reason": "test"}

    monkeypatch.setattr(orchestrator_module, "deterministic_sampler_states", deterministic_stub)
    monkeypatch.setattr(
        orchestrator_module,
        "assess_candidate_intervention",
        lambda *args, **kwargs: EligiblePlan(),
    )

    jobs = instance._deterministic_sampler_fallback_jobs(
        baseline,
        round_id=2,
        count=1,
        blocked_digests=(),
        seen_digests=(),
    )

    assert len(jobs) == 1
    assert jobs[0].params["sampler_policy_status"] == "applied:deterministic_random_fallback"
    assert report["activation_reason"] == "shadow_selection_failed"
    assert report["shadow_error_type"] == "RuntimeError"


def test_evidence_audit_metadata_survives_filter_but_not_semantic_digest():
    base_params = {
        "alpha": 2.0,
        "noise_scale": 10.0,
        "sampler_policy": "explore",
    }
    audit_params = {
        "fallback_selection_policy": "joint_evidence",
        "joint_sampler_evidence_count": 4,
        "joint_sampler_matched_control_groups": 2,
        "joint_sampler_evidence_scope": "memory_header_and_job_provenance_match",
        "matched_control_semantics": "same_target_same_round_baseline",
        "current_round_control_owned_by_governance": True,
        "current_sampler_state_excluded": True,
    }
    filtered = get_model_search_profile("boltzgen").filter_params({**base_params, **audit_params}).params
    assert all(key in filtered for key in audit_params)

    plain = DesignJob("plain", "target.cif", "A", [], 60, params=base_params)
    audited = DesignJob("audited", "target.cif", "A", [], 60, params={**base_params, **audit_params})
    assert effective_semantic_digest(plain) == effective_semantic_digest(audited)
