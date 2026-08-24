#!/usr/bin/env python3
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.active_learning.strategy import CANONICAL_STRATEGY_ARM_CATALOG, StrategyLevelActiveLearner
from binderloop.analysis.structure_features import analyze_binder_structure, motif_retention_metrics
from binderloop.models.base import DesignJob
from binderloop.config import load_config
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.execution_governance import resolve_round_budget
from binderloop.strategy_governance import (
    ArmApplicability, CandidateIntervention, assess_candidate_intervention,
    attribution_identity_digest, deduplicate_effective_jobs, effective_semantic_digest,
    finalize_immutable_branch_plan, job_identity_semantic_digest, semantic_projection,
)


def atom(serial, name, chain, resid, x, y, z, element="C"):
    line = (
        "ATOM  " + ("%5d" % serial) + " " + ("%4s" % name) + " " + ("%3s" % "ALA") +
        " " + chain + ("%4d" % resid) + " " + "   " + ("%8.3f%8.3f%8.3f" % (x, y, z)) +
        "  1.00  0.00" + " " * 10 + ("%2s" % element)
    )
    return line



def _formal_orchestrator(tmp):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "sc2rbd_structured_task_iptm035_test.yaml")
    return BinderDesignOrchestrator(cfg, out_dir=Path(tmp) / "out", max_rounds=1)


def test_sampler_intent_without_final_state_does_not_create_values():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = _formal_orchestrator(tmp)
        job = _job("sampler_hold", params={"sampler_policy": "explore"})
        result = orchestrator._materialize_sampler_and_context_intents([job])[0]
        assert "alpha" not in result.params
        assert "noise_scale" not in result.params
        assert "step_scale" not in result.params
        assert result.params["sampler_policy_applied"] is False
        assert result.params["sampler_policy_status"] == "not_applicable:missing_final_probabilistic_state"
        assert result.params["strategy_intent"]["kind"] == "hold"


def test_sampler_intent_preserves_exact_final_state():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = _formal_orchestrator(tmp)
        final = {"alpha": 0.003, "noise_scale": 0.8, "step_scale": 1.0}
        job = _job("sampler_final", params={"sampler_policy": "explore", "final_parameter_state": dict(final), **final})
        result = orchestrator._materialize_sampler_and_context_intents([job])[0]
        assert result.params["final_parameter_state"] == final
        assert {key: result.params[key] for key in final} == final
        assert result.params["sampler_policy_status"] == "applied:final_probabilistic_state"


def test_catalog_and_hold():
    assert set(CANONICAL_STRATEGY_ARM_CATALOG) == {"baseline_hold", "site_primary_condition", "site_expanded_condition", "site_negative_exclusion", "target_context_focus", "sampler_explore", "template_exploit", "sequence_repair"}
    parent = DesignJob("p", "target.cif", "E", ["E:10"], 80, params={"binder_lengths": [80], "alpha": 0.003, "hotspot_weight": 2.0}, output_dir="out/p")
    proposal = StrategyLevelActiveLearner().propose_next(1, [parent], [], "out", policy_update={"alpha": 0.01}, ranked_arm_names=["baseline_hold"], selection_context={"strict_positive_count": 1})
    job = proposal.jobs[0]
    assert job.params["alpha"] == 0.003
    assert "hotspot_weight" not in job.params
    assert job.params["deprecated_strategy_audit"]["hotspot_weight"]["status"] == "deprecated_audit_only"
    assert job.params["effective_intervention_digest"]


def test_coverage_and_heavy_atom_clash():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "complex.pdb"
        path.write_text("\n".join([atom(1, "CA", "A", 1, 0, 0, 0), atom(2, "CB", "A", 1, 0, 0, 0.5), atom(3, "CA", "B", 10, 0, 0, 1.0), atom(4, "CB", "B", 10, 0, 0, 1.5), atom(5, "CA", "B", 11, 0, 4, 0)]) + "\n")
        result = analyze_binder_structure(path, binder_chain="A", target_chains=["B"], auto_detect_chains=False, primary_residues=["B:10"], expanded_residues=["B:11"], negative_residues=["B:12"])
        assert result.primary_coverage["coverage_fraction"] == 1.0
        assert result.expanded_coverage["coverage_fraction"] == 1.0
        assert result.negative_coverage["coverage_fraction"] == 0.0
        assert result.heavy_atom_clash_count >= 1
        assert result.clash_rank[1] == -float(result.heavy_atom_clash_count)


def test_motif_retention_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.pdb"
        cand = Path(tmp) / "cand.pdb"
        lines = [atom(1, "CA", "A", 1, 0, 0, 0), atom(2, "CA", "A", 2, 1, 0, 0), atom(3, "CA", "A", 3, 2, 0, 0), atom(4, "CA", "B", 10, 0, 0, 4)]
        ref.write_text("\n".join(lines) + "\n")
        cand.write_text("\n".join(lines) + "\n")
        metrics = motif_retention_metrics(ref, cand, reference_chain="A", candidate_chain="A", residue_ids=["A:1", "A:2", "A:3"], reference_sequence="AAA", reference_target_contacts=["B:10"])
        assert metrics["matched_ca_count"] == 3
        assert metrics["motif_rmsd"] < 1e-6
        assert metrics["sequence_identity"] == 1.0
        assert metrics["contact_retention"] == 1.0


def test_template_arm_is_one_round_job_without_control_attribution():
    learner = StrategyLevelActiveLearner()
    template = {"mode": "structure_redesign", "source_structure_file": "source.cif", "binder_chain": "A", "fixed_res_index": "31..38", "source_binder_length": 80}
    parent = DesignJob("p", "target.cif", "E", ["E:10"], 90, params={"binder_lengths": [90], "binder_templates": [template], "template_conditioned_fraction": 0.5}, output_dir="out/p")
    proposal = learner.propose_next(1, [parent], [], "out", ranked_arm_names=["template_exploit"], selection_context={"strict_positive_count": 2, "effective_templates_available": True})
    assert len(proposal.jobs) == 1
    job = proposal.jobs[0]
    assert job.params["template_conditioned"] is True
    assert job.params["template_count"] == 1
    assert "template_free_exploration" not in job.params
    assert "matched_group_id" not in job.params
    assert "matched_comparison" not in job.params


def test_global_conditioned_fraction_with_multiple_arms():
    result = resolve_round_budget(10, [
        {"id": "template_1", "bucket": "template_conditioned", "weight": 0.2},
        {"id": "template_2", "bucket": "template_conditioned", "weight": 0.2},
        {"id": "template_free", "bucket": "template_free", "weight": 0.6},
        {"id": "sampler", "bucket": "other", "weight": 1.0},
    ], requested_conditioned_fraction=0.4)
    conditioned = sum(item["num_designs"] for item in result.allocations if item["bucket"] == "template_conditioned")
    assert conditioned == 4, result.to_dict()
    assert sum(item["num_designs"] for item in result.allocations) == 10


def _job(job_id, *, lengths=(80,), hotspots=("A:10",), params=None):
    values = dict(params or {})
    values["binder_lengths"] = list(lengths)
    return DesignJob(job_id, "target.cif", "A", list(hotspots), lengths[len(lengths)//2], params=values, output_dir="out/" + job_id)


def test_strategy_applicability_five_states():
    base = _job("base", params={"alpha": 0.001})
    sampler = _job("sampler", params={"alpha": 0.003})
    candidate = CandidateIntervention("sampler_explore", "sampler", direction="increase")
    eligible = assess_candidate_intervention(candidate, base, sampler)
    assert eligible.applicability is ArmApplicability.ELIGIBLE
    noop = assess_candidate_intervention(candidate, base, _job("noop", params={"alpha": 0.001}))
    assert noop.applicability is ArmApplicability.NOT_APPLICABLE
    blocked = assess_candidate_intervention(candidate, base, sampler, blocked_digests=[eligible.effective_intervention_digest])
    assert blocked.applicability is ArmApplicability.BLOCKED
    unsupported_job = _job("unsupported", lengths=(70,), params={"alpha": 0.003, "selection_policy": {"metric": "clash"}})
    unsupported = assess_candidate_intervention(CandidateIntervention("mixed", "mixed"), base, unsupported_job)
    assert unsupported.applicability is ArmApplicability.UNSUPPORTED
    duplicate = assess_candidate_intervention(candidate, base, sampler, seen_effective_digests=[eligible.effective_intervention_digest])
    assert duplicate.applicability is ArmApplicability.DUPLICATE_EFFECTIVE_INTERVENTION


def test_target_context_and_sampler_noops_removed():
    base = _job("base", params={"alpha": 0.001, "target_context_policy": "focus"})
    context = assess_candidate_intervention(CandidateIntervention("target_context_focus", "target_context"), base, _job("same", params={"alpha": 0.001, "target_context_policy": "focus"}))
    sampler = assess_candidate_intervention(CandidateIntervention("sampler_explore", "sampler"), base, _job("same2", params={"alpha": 0.001, "sampler_policy": "explore"}))
    assert context.applicability is ArmApplicability.NOT_APPLICABLE
    assert sampler.applicability is ArmApplicability.NOT_APPLICABLE


def test_single_arm_preserves_one_combined_parameter_vector():
    parent = _job("parent", lengths=(80, 90, 100), hotspots=("A:10",), params={"alpha": 0.001, "noise_scale": 0.7})
    learner = StrategyLevelActiveLearner()
    proposal = learner.propose_next(2, [parent], [], "out", policy_update={"binder_lengths": [55, 60], "alpha": 0.003, "noise_scale": 0.8}, ranked_arm_names=["sampler_explore"])
    assert len(proposal.jobs) == 1
    job = proposal.jobs[0]
    assert job.params["exploration_arm"] == "sampler_explore"
    assert job.params["binder_lengths"] == [55, 60]
    assert job.params["alpha"] == 0.003 and job.params["noise_scale"] == 0.8
    assert "controlled_comparison" not in job.params


def test_semantic_digest_uses_config_contract_canonicalization():
    typed = _job("typed", params={"filter_biased": True, "devices": 3, "num_workers": 4})
    textual = _job("textual", params={"filter_biased": "true", "devices": "3", "num_workers": "4"})
    assert semantic_projection(typed) == semantic_projection(textual)
    assert effective_semantic_digest(typed) == effective_semantic_digest(textual)
    assert job_identity_semantic_digest(typed) == job_identity_semantic_digest(textual)


def test_effective_digest_dedup_and_immutable_budget_digest():
    first = _job("one", params={"alpha": 0.003, "exploration_arm": "sampler_explore"})
    second = _job("two", params={"alpha": 0.003, "exploration_arm": "different_label"})
    assert effective_semantic_digest(first) == effective_semantic_digest(second)
    assert len(deduplicate_effective_jobs([first, second])) == 2
    small = finalize_immutable_branch_plan(first, 2)
    large = finalize_immutable_branch_plan(first, 5)
    assert small.effective_intervention_digest != large.effective_intervention_digest
    assert small.plan_digest != large.plan_digest



def test_multiple_logical_jobs_receive_isolated_arm_identities():
    from binderloop.strategy_governance import materialize_deterministic_job_identities, resolved_within
    first = _job("candidate-a", params={"alpha": 0.003, "arm_id": "sampler_explore", "exploration_arm": "sampler_explore", "arm_rank": 0})
    second = _job("candidate-b", params={"alpha": 0.004, "arm_id": "baseline_hold", "exploration_arm": "baseline_hold", "arm_rank": 1})
    jobs = materialize_deterministic_job_identities([first, second], round_id=3, output_root="out")
    assert len({job.output_dir for job in jobs}) == 2
    assert all(resolved_within(job.output_dir, job.params["arm_root"]) for job in jobs)
    assert [job.params["arm_id"] for job in jobs] == ["sampler_explore", "baseline_hold"]


def test_duplicate_logical_job_identity_is_rejected_before_budget():
    from binderloop.strategy_governance import materialize_deterministic_job_identities
    first = _job("copy-a", params={
        "alpha": 0.003, "arm_id": "sampler_explore",
        "logical_branch_id": "shared-branch",
    })
    duplicate = _job("copy-b", params={
        "alpha": 0.003, "arm_id": "sampler_explore",
        "logical_branch_id": "shared-branch",
    })
    try:
        materialize_deterministic_job_identities([first, duplicate], round_id=3, output_root="out")
    except ValueError as exc:
        assert str(exc) == "duplicate_logical_job_identity"
    else:
        raise AssertionError("duplicate logical job identity was accepted")


def test_identity_finalization_replaces_stale_clone_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = _formal_orchestrator(tmp)
        source = _job("old", params={"alpha": 0.001, "exploration_arm": "baseline_hold"})
        source = orchestrator._finalize_semantic_job_identities([source], round_id=1)[0]
        stale = dict(source.params["job_identity"])
        source.params["alpha"] = 0.003
        finalized = orchestrator._finalize_semantic_job_identities([source], round_id=2)[0]
        assert finalized.job_id.startswith("r2_")
        assert finalized.params["job_identity"] != stale
        assert finalized.params["job_identity"]["execution_semantic_digest"] == effective_semantic_digest(finalized)


def test_final_sampler_is_part_of_identity_digest_before_budget():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = _formal_orchestrator(tmp)
        job = _job("sampler", params={
            "alpha": 0.003,
            "noise_scale": 0.8,
            "step_scale": 1.0,
            "final_parameter_state": {"alpha": 0.003, "noise_scale": 0.8, "step_scale": 1.0},
            "exploration_arm": "sampler_explore",
        })
        finalized = orchestrator._finalize_semantic_job_identities([job], round_id=5)[0]
        before_budget = finalized.params["job_identity"]["semantic_digest"]
        budgeted = orchestrator._enforce_round_cap([finalized])[0]
        assert budgeted.params["job_identity"]["execution_semantic_digest"] == effective_semantic_digest(budgeted)
        assert budgeted.params["immutable_branch_plan"]["semantic_projection"]["model_params"]["alpha"] == 0.003
        assert budgeted.params["immutable_branch_plan"]["semantic_projection"]["model_params"]["noise_scale"] == 0.8


def test_round3_round4_identity_is_finalized_after_budget_and_plan():
    # Reproduces pdl1 r4_round and sc2rbd r3_round: a pre-budget identity used
    # to survive num_designs/immutable-plan mutation and fail validation.
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = _formal_orchestrator(tmp)
        orchestrator._round_design_cap = 8
        job = _job("r3_round", params={
            "alpha": 0.003,
            "noise_scale": 0.8,
            "final_parameter_state": {"alpha": 0.003, "noise_scale": 0.8},
            "exploration_arm": "sampler_explore",
        })
        pre_budget = orchestrator._finalize_semantic_job_identities([job], round_id=3)[0]
        old_id = pre_budget.job_id
        finalized = orchestrator._enforce_round_cap([pre_budget], round_id=3)[0]
        assert finalized.job_id != old_id
        assert finalized.params["job_identity"]["finalized"] is True
        assert finalized.params["job_identity"]["semantic_digest"] == job_identity_semantic_digest(finalized)
        assert finalized.params["job_identity"]["job_id"] == finalized.job_id
        assert finalized.params["job_identity"]["purpose"] == "arm_scoped_execution_identity"
        assert finalized.params["immutable_branch_plan"]["allocated_designs"] == 8
        assert finalized.output_dir.endswith(finalized.job_id.split("_", 1)[1])
        orchestrator._validate_job_identities([finalized])


def test_immutable_plan_separates_execution_from_attribution():
    left = _job("left", params={"arm_id": "baseline_hold", "exploration_arm": "baseline_hold", "logical_branch_id": "r2_baseline", "strategy_intent": {"kind": "hold"}})
    right = _job("right", params={"arm_id": "site_primary_condition", "exploration_arm": "site_primary_condition", "logical_branch_id": "r2_primary", "strategy_intent": {"kind": "binding_site", "positive_scope": "primary"}})
    left_plan = finalize_immutable_branch_plan(left, 4)
    right_plan = finalize_immutable_branch_plan(right, 4)
    assert left_plan.effective_intervention_digest == right_plan.effective_intervention_digest
    assert left_plan.plan_digest != right_plan.plan_digest


def test_execution_projection_excludes_strategy_attribution_metadata():
    base = _job("base", params={"alpha": 0.003})
    attributed = _job("attributed", params={
        "alpha": 0.003,
        "arm_id": "sampler_explore",
        "exploration_arm": "sampler_explore",
        "logical_branch_id": "r2_sampler",
        "strategy_effect": {"arm": "sampler_explore", "reason": "attribution_only"},
        "strategy_intent": {"kind": "sampler"},
    })
    assert semantic_projection(base) == semantic_projection(attributed)
    assert effective_semantic_digest(base) == effective_semantic_digest(attributed)
    assert attribution_identity_digest(base) != attribution_identity_digest(attributed)


def test_identity_materialization_uses_current_attribution_fields():
    with tempfile.TemporaryDirectory() as tmp:
        job = _job("retry", params={"alpha": 0.003, "exploration_arm": "sampler_explore"})
        from binderloop.strategy_governance import materialize_deterministic_job_identities
        current = materialize_deterministic_job_identities([job], round_id=2, output_root=tmp)[0]
        expected_attribution = attribution_identity_digest(current)
        expected_logical = effective_semantic_digest(current)
        metadata = current.params["job_identity"]
        assert current.params["execution_semantic_digest"] == expected_logical
        assert current.params["attribution_identity_digest"] == expected_attribution
        assert metadata["execution_semantic_digest"] == expected_logical
        assert metadata["attribution_identity_digest"] == expected_attribution
        BinderDesignOrchestrator._validate_job_identities([current])


if __name__ == "__main__":
    test_immutable_plan_separates_execution_from_attribution()
    test_execution_projection_excludes_strategy_attribution_metadata()
    test_identity_materialization_uses_current_attribution_fields()
    test_sampler_intent_without_final_state_does_not_create_values()
    test_sampler_intent_preserves_exact_final_state()
    test_catalog_and_hold()
    test_coverage_and_heavy_atom_clash()
    test_motif_retention_metrics()
    test_template_arm_is_one_round_job_without_control_attribution()
    test_global_conditioned_fraction_with_multiple_arms()
    test_strategy_applicability_five_states()
    test_target_context_and_sampler_noops_removed()
    test_single_arm_preserves_one_combined_parameter_vector()
    test_effective_digest_dedup_and_immutable_budget_digest()
    test_multiple_logical_jobs_receive_isolated_arm_identities()
    test_identity_finalization_replaces_stale_clone_metadata()
    test_final_sampler_is_part_of_identity_digest_before_budget()
    test_round3_round4_identity_is_finalized_after_budget_and_plan()
    print("ALL STRATEGY GOVERNANCE TESTS PASSED")
