#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.execution_governance import (
    bind_template_application_budget,
    build_template_application_plan,
    resolve_execution_plan,
    resolve_round_budget,
    stable_digest,
    validate_template_application,
)
from binderloop.models.base import DesignJob
from binderloop.models.boltzgen_adapter import BoltzGenAdapter



def test_global_round_budget():
    candidates = [
        {"id": "t1", "bucket": "template_conditioned", "weight": 0.2},
        {"id": "t2", "bucket": "template_conditioned", "weight": 0.2},
        {"id": "free", "bucket": "template_free", "weight": 0.6},
        {"id": "other", "bucket": "other", "weight": 1.0},
    ]
    result = resolve_round_budget(11, candidates, requested_conditioned_fraction=0.4)
    assert sum(item["num_designs"] for item in result.allocations) == 11
    assert result.bucket_allocations["template_conditioned"] == 4
    assert sum(item["num_designs"] for item in result.allocations if item["bucket"] == "template_conditioned") == 4
    assert result.digest

    tiny = resolve_round_budget(2, candidates[:3], requested_conditioned_fraction=0.4)
    assert sum(item["num_designs"] for item in tiny.allocations) == 2
    assert tiny.bucket_allocations["template_conditioned"] == 1


def test_invalid_template_rematerialization():
    result = resolve_round_budget(5, [
        {"id": "bad", "bucket": "template_conditioned", "valid": False, "rejection_reason": "alignment_not_evaluable"},
        {"id": "free", "bucket": "template_free", "weight": 1.0},
    ], requested_conditioned_fraction=0.6)
    assert result.bucket_allocations == {"template_conditioned": 0, "template_free": 5, "other": 0}
    assert result.rejections[0]["id"] == "bad"
    assert result.rematerialization
    assert result.rematerialization[0]["destination_buckets"] == ["template_free"]
    assert not any(item["id"] == "bad" for item in result.allocations)


def test_template_application_plan_fail_closed_and_binding():
    legacy = build_template_application_plan(None, current_target="target.cif", round_fraction=0.0, allocated_num_designs=3)
    assert legacy.applicability["status"] == "not_applicable"
    alignment = {"status": "aligned", "digest": "align", "source_target_chain": "B", "current_target_chain": "E"}
    transform = {"status": "identity", "digest": "transform"}
    template = {
        "template_id": "t1", "source_digest": "source", "source_structure_file": "source.cif",
        "target_alignment": alignment, "source_to_effective_residue_map": {"A:1": "A:1"},
        "length_transform": transform,
    }
    plan = build_template_application_plan(template, current_target="target.cif", current_target_chain="E", round_fraction=0.4, allocated_num_designs=0)
    assert plan.applicability["applicable"] is True
    bound = bind_template_application_budget(plan, 4, receipt={"consumer": "orchestrator"})
    assert bound.allocated_num_designs == 4 and bound.digest != plan.digest
    assert bound.consumer_receipts == [{"consumer": "orchestrator"}]
    rejected = build_template_application_plan({"template_id": "bad"}, current_target="target.cif", round_fraction=0.4, allocated_num_designs=1)
    assert rejected.applicability["status"] == "rejected"



def test_budget_independent_template_validation():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.cif"
        source.write_text("template")
        digest = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
        template = {
            "template_id": "valid", "source_digest": digest, "source_structure_file": str(source),
            "target_alignment": {"status": "aligned", "digest": "align"},
            "source_to_effective_residue_map": {"A:1": "A:1"},
            "length_transform": {"status": "identity", "digest": "transform"},
        }
        valid = validate_template_application(template)
        assert valid.valid and valid.status == "validated"
        assert "allocated_num_designs" not in valid.to_dict()
        mismatched = validate_template_application({**template, "source_digest": "wrong"})
        assert not mismatched.valid
        assert "source_digest_mismatch" in mismatched.failures
        staged_failure = validate_template_application({**template, "staging_status": "failed"})
        assert not staged_failure.valid
        assert "template_staging_failed" in staged_failure.failures

def main():
    plan = resolve_execution_plan({"num_designs": 12, "inverse_fold_num_sequences": 2, "budget": 10, "devices": 4, "host_count": 1, "noise_scale": 1.2, "step_scale": 1.5}, operational_bounds={"noise_scale": {"min": 0.65, "max": 0.8}, "step_scale": {"min": 0.7, "max": 0.9}})
    assert plan.resolved_params["budget"] == 99999
    assert plan.resolved_params["noise_scale"] == 0.8
    assert plan.resolved_params["step_scale"] == 0.9
    assert "template_conditioned_fraction" not in plan.resolved_params
    exact = resolve_execution_plan(
        {"num_designs": 2, "budget": 2, "noise_scale": 0.9},
        final_parameter_state={"noise_scale": 0.9},
        parameter_catalog={"noise_scale": [0.6, 0.7, 0.8, 0.9]},
        parameter_catalog_digest="catalog",
    )
    assert exact.resolved_params["noise_scale"] == 0.9
    assert exact.final_parameter_state == {"noise_scale": 0.9}
    try:
        resolve_execution_plan(
            {"num_designs": 2, "budget": 2, "noise_scale": 0.85},
            final_parameter_state={"noise_scale": 0.85},
            parameter_catalog={"noise_scale": [0.8, 0.9]},
            parameter_catalog_digest="catalog",
        )
        raise AssertionError("non-catalog final state was accepted")
    except ValueError as exc:
        assert "exact catalog member" in str(exc)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "target.cif"; target.write_text("cif")
        params = dict(plan.resolved_params)
        params.update({"protocol": "protein-anything", "binder_lengths": [60]})
        job = DesignJob("j", str(target), "A", [], 60, params=params, output_dir=str(root / "out"))
        command = BoltzGenAdapter(root=str(root)).build_command(job)
        assert command[command.index("--num_designs") + 1] == "12"
        assert command[command.index("--budget") + 1] == "99999"
    test_global_round_budget()
    test_invalid_template_rematerialization()
    test_template_application_plan_fail_closed_and_binding()
    test_budget_independent_template_validation()
    print("OK execution governance")

if __name__ == "__main__": main()
