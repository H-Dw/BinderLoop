#!/usr/bin/env python3
import json
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.templates.outcome_ledger import (
    OutcomeLedger, assert_matched_pair, canonical_digest, compute_utility,
    lifecycle_key, matched_comparison_signature, matched_group_id, rank_templates,
    update_ledger_from_round,
)


def test_atomic_digest_stability_and_target_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.json"
        ledger = OutcomeLedger.open(path)
        ledger.record_failure("target-a", "t1", round_id=1, failure_type="runtime_failure")
        ledger.save(); first = json.loads(path.read_text()); first_digest = first["digest"]
        assert first_digest == canonical_digest(first)
        assert not ledger.entry("target-a", "t1")["blacklisted"]
        assert ledger.entry("target-b", "t1", create=False) is None
        reopened = OutcomeLedger.open(path); reopened.save()
        assert json.loads(path.read_text())["digest"] == first_digest
        assert not list(path.parent.glob("*.tmp"))


def test_hard_blacklist_and_cooldown_recovery():
    ledger = OutcomeLedger.open(Path("/does/not/exist/ledger.json"))
    ledger.record_failure("a", "hard", round_id=1, failure_type="digest_mismatch")
    assert not ledger.eligible("a", "hard", 100)
    bad = {"quality": .2, "primary_coverage": .2, "retention": .2, "clash": .3}
    good = {"quality": .5, "primary_coverage": .5, "retention": .5, "clash": .1}
    ledger.record_outcome("a", "soft", round_id=1, template_metrics=bad, control_metrics=good, confidence=1, matched_group_id="g", cooldown_failures=2)
    ledger.record_outcome("a", "soft", round_id=2, template_metrics=bad, control_metrics=good, confidence=1, matched_group_id="g", cooldown_failures=2)
    assert not ledger.eligible("a", "soft", 2)
    assert ledger.eligible("a", "soft", 3)
    ledger.record_outcome("a", "soft", round_id=3, template_metrics=good, control_metrics=bad, confidence=1, matched_group_id="g", cooldown_failures=2)
    assert ledger.entry("a", "soft")["cooldown_until"] is None


def test_matched_uplift_direction_and_time_decay():
    template = {"quality": .8, "primary_coverage": .7, "retention": .9, "clash": .1}
    control = {"quality": .6, "primary_coverage": .5, "retention": .8, "clash": .3}
    fresh = compute_utility(template, control, confidence=.8)
    old = compute_utility(template, control, confidence=.8, rounds_since_use=3, decay=.5)
    assert fresh["uplift"]["clash"] > 0 and fresh["adjusted"] > 0
    assert old["adjusted"] == fresh["adjusted"] * .125


def test_utility_aware_topk_and_cold_start():
    ledger = OutcomeLedger.open(Path("/does/not/exist/ledger.json"))
    ledger.record_outcome("a", "known", round_id=1, template_metrics={"quality": .9}, control_metrics={"quality": .1}, confidence=1, matched_group_id="g")
    snapshot = ledger.target_snapshot("a", round_id=1)
    templates = [{"template_id": "known", "target_compatibility": .9, "source_digest": "s1"}, {"template_id": "cold", "target_compatibility": .9, "source_digest": "s2"}]
    ranked = rank_templates(templates, snapshot, top_k=2, round_id=2)
    assert ranked[0]["template_id"] == "known"
    assert ranked[1]["ledger_state"]["uncertainty"] == 1.0


def test_pooled_partition_pairing_and_parity():
    params = {"binder_lengths": [80], "steps": 200, "additional_filters": ["x"], "inverse_fold_num_sequences": 2, "design_checkpoints": ["c"], "num_designs": 4}
    signature = matched_comparison_signature(params, target_structure="target", chain_id="E", binder_length=80)
    group = matched_group_id("target-digest", "t1", 2, signature)
    template = {"template_conditioned": True, "matched_group_id": group, "matched_comparison": signature}
    control = {"template_conditioned": False, "matched_group_id": group, "matched_comparison": signature}
    assert_matched_pair(template, control)
    assert group == matched_group_id("target-digest", "t1", 2, signature)
    broken = dict(control); broken["matched_comparison"] = {**signature, "effective_length": 81}
    try: assert_matched_pair(template, broken)
    except ValueError: pass
    else: raise AssertionError("mismatch must fail")




def test_final_matched_outcome_without_stage_attribution():
    class Job:
        def __init__(self, job_id, params):
            self.job_id = job_id
            self.params = params

    group = {"matched_group_id": "g"}
    control = Job("control", {"template_conditioned": False, "matched_group_ids": [group]})
    template = Job("template", {
        "template_conditioned": True,
        "matched_group_id": "g",
        "target_identity_digest": "target",
        "binder_template": {"template_id": "t1"},
    })
    ingestions = [
        {"candidates": [{"id": "c", "quality_score": .4, "hotspot_coverage": .3, "clash_density": .2}]},
        {"candidates": [{"id": "t", "quality_score": .8, "hotspot_coverage": .7, "clash_density": .1}]},
    ]
    ledger = OutcomeLedger.open(Path("/does/not/exist/ledger.json"))
    update_ledger_from_round(
        ledger, round_id=1, jobs=[control, template], ingestions=ingestions,
        execution_records=[{"job_id": "control", "status": "completed"}, {"job_id": "template", "status": "completed"}],
        attribution_documents=[],
    )
    entry = ledger.entry("target", "t1")
    assert entry["uses"] == 1 and entry["successes"] == 1
    assert entry["stage_attribution"] == {}
    assert entry["utility"]["uplift"]["quality"] > 0
    assert entry["matched_control"]["matched_group_id"] == "g"
    assert entry["matched_control"]["evidence_mode"] == "aggregate_only"
    assert entry["utility"]["confidence"] < 1.0


def test_evaluated_stage_attribution_is_optional_metadata():
    class Job:
        def __init__(self, job_id, params):
            self.job_id = job_id
            self.params = params

    control = Job("control", {"template_conditioned": False, "matched_group_ids": [{"matched_group_id": "g"}]})
    template = Job("template", {"template_conditioned": True, "matched_group_id": "g", "target_identity_digest": "target", "binder_template": {"template_id": "t1"}})
    ledger = OutcomeLedger.open(Path("/does/not/exist/ledger.json"))
    update_ledger_from_round(
        ledger, round_id=1, jobs=[control, template],
        ingestions=[{"candidates": [{"id": "c", "quality_score": .4}], "identity_capability": "validated_lineage", "exact_attribution": True}, {"candidates": [{"id": "t", "quality_score": .8}], "identity_capability": "validated_lineage", "exact_attribution": True}],
        execution_records=[{"job_id": "control", "status": "completed"}, {"job_id": "template", "status": "completed"}],
        attribution_documents=[{"job_id": "template", "comparisons": [{"from_stage": "source", "to_stage": "final_refold", "status": "evaluated", "metrics": {"motif": 1}}]}],
    )
    assert ledger.entry("target", "t1")["stage_attribution"]["source->final_refold"] == {"motif": 1}

def main():
    test_atomic_digest_stability_and_target_isolation()
    test_hard_blacklist_and_cooldown_recovery()
    test_matched_uplift_direction_and_time_decay()
    test_utility_aware_topk_and_cold_start()
    test_pooled_partition_pairing_and_parity()
    test_final_matched_outcome_without_stage_attribution()
    test_evaluated_stage_attribution_is_optional_metadata()
    print("OK template outcome ledger")


if __name__ == "__main__": main()
