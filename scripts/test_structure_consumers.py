#!/usr/bin/env python3
"""Focused regression tests for explicit candidate/structure bindings."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.structure_consumers import (
    structure_files_for_candidates,
    structure_interchain_pae,
    success_structure_files,
)
from binderloop.selection import apply_candidate_selection


def test_structure_filter_prefers_explicit_and_scoped_rows_fail_closed():
    one = "/job-a/rank1_boltzgen_design_spec_3_1.cif"
    ten = "/job-a/rank2_boltzgen_design_spec_3_10.cif"
    other_job = "/job-b/rank1_boltzgen_design_spec_3_1.cif"
    files = [one, ten, other_job]
    rows = [{
        "local_candidate_id": "boltzgen_design_spec_3_1",
        "global_candidate_id": "job-a:3_1",
        "identity_quality": "scoped_source",
        "structure_file": one,
    }]
    assert structure_files_for_candidates(rows, files) == [one]

    unbound_scoped = [{
        "id": "boltzgen_design_spec_3_1",
        "job_id": "job-a",
        "identity_quality": "scoped_source",
    }]
    assert structure_files_for_candidates(unbound_scoped, files) == []


def test_structure_filter_legacy_exact_suffix_avoids_3_1_3_10():
    one = "/run/rank1_boltzgen_design_spec_3_1.cif"
    ten = "/run/rank2_boltzgen_design_spec_3_10.cif"
    selected = structure_files_for_candidates(
        [{"id": "boltzgen_design_spec_3_1"}], [one, ten]
    )
    assert selected == [one]


def test_success_uses_candidate_evaluation_raw_structure_file():
    one = "/job-a/shared.cif"
    two = "/job-b/shared.cif"
    evaluation = SimpleNamespace(top_candidates=[
        SimpleNamespace(
            candidate_id="duplicate-local-id",
            source="/metrics/job-a.csv",
            tags=["pass_compute_gate"],
            raw={"structure_file": two},
        ),
        SimpleNamespace(
            candidate_id="duplicate-local-id",
            source="/metrics/job-b.csv",
            tags=["low_iptm"],
            raw={"structure_file": one},
        ),
        SimpleNamespace(
            candidate_id="shared",
            source=one,
            tags=["pass_compute_gate"],
            raw={},
        ),
    ])
    assert success_structure_files(evaluation, [one, two]) == [two]


def test_pae_binds_explicit_paths_with_duplicate_local_ids():
    one = "/job-a/rank1_shared.cif"
    two = "/job-b/rank1_shared.cif"
    candidates = [
        {"id": "shared", "job_id": "job-a", "structure_file": one, "min_design_to_target_pae": 7.0},
        {"id": "shared", "job_id": "job-b", "structure_file": two, "min_design_to_target_pae": 19.0},
        {"id": "shared", "job_id": "job-c", "min_design_to_target_pae": 2.0},
    ]
    assert structure_interchain_pae(candidates, [one, two]) == {
        one: 7.0, two: 19.0
    }


def test_pae_legacy_suffix_avoids_3_1_3_10():
    one = "/run/rank1_boltzgen_design_spec_3_1.cif"
    ten = "/run/rank2_boltzgen_design_spec_3_10.cif"
    mapping = structure_interchain_pae([
        {"id": "boltzgen_design_spec_3_1", "min_design_to_target_pae": 8.0},
        {"id": "boltzgen_design_spec_3_10", "min_design_to_target_pae": 20.0},
    ], [one, ten])
    assert mapping == {one: 8.0, ten: 20.0}


def test_clash_selection_uses_explicit_structure_file_only():
    safe = "/job-a/rank1_shared.cif"
    clash = "/job-b/rank1_shared.cif"
    candidates = [
        {"candidate_id": "shared", "job_id": "a", "structure_file": safe},
        {"candidate_id": "shared", "job_id": "b", "structure_file": clash},
        {"candidate_id": "shared", "job_id": "c"},
    ]
    policies = {key: {"type": "cross_chain_heavy_atom_clash", "gate": True} for key in ("a", "b", "c")}
    summaries = [
        {"structure_file": safe, "clash_gate_pass": True, "clash_rank": [1.0, 0.0, 0.0]},
        {"structure_file": clash, "clash_gate_pass": False, "heavy_atom_clash_count": 5},
    ]
    accepted, report = apply_candidate_selection(
        candidates, policies_by_job=policies, structure_summaries=summaries
    )
    assert {row["job_id"] for row in accepted} == {"a", "c"}
    assert [item["candidate"]["job_id"] for item in report["rejected"]] == ["b"]
    assert report["not_evaluable"] == ["shared"]
    assert "harness_clash_rank" not in next(row for row in accepted if row["job_id"] == "c")


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print("ALL STRUCTURE CONSUMER TESTS PASSED")




def test_trusted_structure_batch_reads_each_file_once(monkeypatch, tmp_path) -> None:
    from binderloop.agents.structure_evaluation_agent import StructureEvaluationAgent
    import binderloop.agents.structure_evaluation_agent as structure_module
    files = []
    for index in range(2):
        path = tmp_path / f"design_{index}.pdb"
        path.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
        files.append(path)
    calls = []
    original = structure_module.analyze_binder_structure
    def tracked(path, **kwargs):
        calls.append(str(path))
        return original(path, **kwargs)
    monkeypatch.setattr(structure_module, "analyze_binder_structure", tracked)
    batch = StructureEvaluationAgent().analyze_trusted_structures(files, binder_chain="A", target_chains=[], auto_detect_chains=False)
    assert batch.total_structures == 2
    assert calls == [str(path) for path in files]


if __name__ == "__main__":
    main()
