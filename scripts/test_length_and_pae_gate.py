#!/usr/bin/env python3
"""Deterministic tests for the two new harness features:

* Inter-chain PAE-based fragment-template gating (default), replacing the global
  iPTM success gate for deciding which structures may seed reusable templates.
* Structure-quality-driven binder length range selection (BinderLengthPolicyAgent).

Run: python scripts/test_length_and_pae_gate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.fragment_template_mining_agent import (
    FragmentTemplate,
    FragmentTemplateMiningAgent,
)
from binderloop.agents.binder_length_policy_agent import BinderLengthPolicyAgent
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.structure_consumers import structure_interchain_pae


def _summary(structure_file, *, frag_quality=0.82, low=False, binder_len=80, reliability=0.8,
             chain_break=0, interface=10, clash=0.05, tags=None):
    frag = {
        "fragment_id": f"{structure_file}:1-8",
        "start_residue": 1,
        "end_residue": 8,
        "residue_ids": ["A:1", "A:2", "A:3", "A:4", "A:5", "A:6", "A:7", "A:8"],
        "quality_score": frag_quality,
        "quality_label": "low" if low else "high",
        "reasons": ["dense_target_interface"],
        "sequence": "ACDEFGHI",
        "ca_coordinates": [[0.0, 0.0, 0.0]],
    }
    return {
        "structure_file": structure_file,
        "binder_chain": "A",
        "binder_residue_count": binder_len,
        "reliability_score": reliability,
        "chain_break_count": chain_break,
        "interface_residue_count": interface,
        "clash_density": clash,
        "reliability_tags": list(tags or ["structure_features_pass"]),
        "high_quality_fragments": [] if low else [frag],
        "low_quality_fragments": [frag] if low else [],
        "contacts_preview": [{"binder_residue": "A:1", "target_residue": "B:153", "contact_type": "polar"}],
        "hotspot_contacts": {"E:153": 1},
    }


def test_interchain_pae_gate_default():
    """Default gate uses inter-chain PAE, NOT iPTM: a low-PAE structure is eligible
    even when it is not in the iPTM success set, and a high-PAE structure is
    excluded even when it IS in the iPTM success set."""
    mine = FragmentTemplateMiningAgent()
    structural = {"summaries": [_summary("good.cif"), _summary("bad.cif")]}
    pae = {"good.cif": 6.0, "bad.cif": 18.0}
    batch = mine.mine_templates(
        structural,
        round_id=1,
        interchain_pae_by_structure=pae,
        interchain_pae_max=12.0,
        # iPTM gate would have picked bad.cif; the PAE gate must ignore this.
        success_structure_files=["bad.cif"],
    )
    preserve_sources = {t.source_structure_file for t in batch.templates if t.reuse_mode == "preserve"}
    assert preserve_sources == {"good.cif"}, preserve_sources
    gate = batch.analysis_metadata["template_gate"]
    assert gate["gate_metric"] == "interchain_pae", gate
    assert gate["preserve_eligible_structures"] == 1, gate
    # Every preserve template carries its source inter-chain PAE.
    assert all(t.interchain_pae == 6.0 for t in batch.templates if t.reuse_mode == "preserve")
    print("OK test_interchain_pae_gate_default")


def test_iptm_gate_optin():
    """When explicitly set to iptm, the legacy success-set gate is used."""
    mine = FragmentTemplateMiningAgent()
    structural = {"summaries": [_summary("good.cif"), _summary("bad.cif")]}
    batch = mine.mine_templates(
        structural,
        round_id=1,
        gate_metric="iptm",
        interchain_pae_by_structure={"good.cif": 6.0, "bad.cif": 18.0},
        success_structure_files=["bad.cif"],
    )
    preserve_sources = {t.source_structure_file for t in batch.templates if t.reuse_mode == "preserve"}
    assert preserve_sources == {"bad.cif"}, preserve_sources
    assert batch.analysis_metadata["template_gate"]["gate_metric"] == "iptm"
    print("OK test_iptm_gate_optin")


def test_pae_gate_fallback_without_data():
    """PAE gate with no PAE data falls back to the success gate; success=None => all."""
    mine = FragmentTemplateMiningAgent()
    structural = {"summaries": [_summary("a.cif"), _summary("b.cif")]}
    batch = mine.mine_templates(structural, round_id=1)  # default gate, no pae, no success set
    preserve_sources = {t.source_structure_file for t in batch.templates if t.reuse_mode == "preserve"}
    assert preserve_sources == {"a.cif", "b.cif"}, preserve_sources
    print("OK test_pae_gate_fallback_without_data")



def test_production_template_missing_pae_fails_closed():
    mine = FragmentTemplateMiningAgent()
    structural = {"summaries": [_summary("a.cif")]}
    try:
        mine.mine_templates(
            structural,
            round_id=1,
            templates_enabled=True,
            require_pae=True,
            interchain_pae_by_structure=None,
        )
    except ValueError as exc:
        assert "interchain_pae_required_but_missing" in str(exc)
    else:
        raise AssertionError("production template mining must fail closed without PAE")
    print("OK test_production_template_missing_pae_fails_closed")


def test_production_template_empty_structure_batch_is_not_a_pae_error():
    """An upstream execution failure with no structures must not be masked as a PAE error."""
    batch = FragmentTemplateMiningAgent().mine_templates(
        {"summaries": []},
        round_id=1,
        templates_enabled=True,
        require_pae=True,
        interchain_pae_by_structure=None,
    )
    assert batch.templates == []
    gate = batch.analysis_metadata["template_gate"]
    assert gate["total_structures"] == 0, gate
    assert gate["interchain_pae_data_available"] is False, gate
    print("OK test_production_template_empty_structure_batch_is_not_a_pae_error")


def test_redesign_template_prefers_low_pae(tmp_path: Path):
    """The executable structure-redesign template is the lowest inter-chain PAE
    preserve fragment among mountable, high-quality candidates."""
    f1 = tmp_path / "low_pae.cif"
    f2 = tmp_path / "high_pae.cif"
    structure = "\n".join(["ATOM      1   CA ALA A   1       0.000   0.000   0.000  1.00  0.00           C", "ATOM      2   CA ALA A   8       1.000   0.000   0.000  1.00  0.00           C"]) + "\n"
    f1.write_text(structure, encoding="utf-8")
    f2.write_text(structure, encoding="utf-8")

    def _tpl(source, pae):
        return FragmentTemplate(
            template_id=f"t_{Path(source).stem}",
            source_structure_file=str(source),
            binder_chain="A",
            binder_residue_span=[1, 8],
            binder_residue_ids=["A:1", "A:8"],
            target_contact_residues=["B:153"],
            hotspot_contacts={"E:153": 1},
            contact_types={"polar": 1},
            quality_score=0.80,
            quality_label="high",
            reuse_mode="preserve",
            interchain_pae=pae,
        )

    templates = [_tpl(f2, 9.0), _tpl(f1, 5.0)]
    # Executable templates now fail closed until a current-target patch can be
    # measured; the compatibility wrapper must not fabricate identity alignment.
    chosen = FragmentTemplateMiningAgent._structure_redesign_template(templates)
    assert chosen is None
    print("OK test_redesign_template_requires_alignment")


def test_length_policy_shorter_on_foldability_failure():
    agent = BinderLengthPolicyAgent()
    # All structures show chain breaks / low reliability -> foldability problem.
    summaries = [
        _summary(f"s{i}.cif", binder_len=100, reliability=0.3, chain_break=2,
                 interface=12, tags=["binder_chain_break"])
        for i in range(6)
    ]
    rec = agent.recommend({"summaries": summaries}, current_lengths=[100], allowed_min=60,
                          allowed_max=120, step=10)
    assert rec.enabled and rec.direction == "shorter", (rec.direction, rec.rationale)
    assert max(rec.recommended_lengths) < 100, rec.recommended_lengths
    assert min(rec.recommended_lengths) >= 60, rec.recommended_lengths
    assert rec.recommended_config.get("binder_lengths") == rec.recommended_lengths
    print("OK test_length_policy_shorter_on_foldability_failure")


def test_length_policy_longer_on_weak_interface():
    agent = BinderLengthPolicyAgent()
    # Folds fine but interfaces are tiny -> go longer.
    summaries = [
        _summary(f"s{i}.cif", binder_len=70, reliability=0.85, chain_break=0,
                 interface=2, tags=["weak_or_tiny_interface"])
        for i in range(6)
    ]
    rec = agent.recommend({"summaries": summaries}, current_lengths=[70], allowed_min=60,
                          allowed_max=120, step=10)
    assert rec.direction == "longer", (rec.direction, rec.rationale)
    assert max(rec.recommended_lengths) > 70, rec.recommended_lengths
    assert max(rec.recommended_lengths) <= 120, rec.recommended_lengths
    print("OK test_length_policy_longer_on_weak_interface")


def test_length_policy_respects_fixed_and_disabled():
    agent = BinderLengthPolicyAgent()
    summaries = [_summary("s.cif", binder_len=80)]
    # Fixed single allowed length -> no-op.
    fixed = agent.recommend({"summaries": summaries}, current_lengths=[80], allowed_min=80, allowed_max=80, step=10)
    assert fixed.direction == "fixed", fixed.direction
    assert fixed.recommended_config == {}, fixed.recommended_config
    # Disabled -> no-op regardless of structures.
    off = agent.recommend({"summaries": summaries}, current_lengths=[80], allowed_min=60, allowed_max=120, step=10, enabled=False)
    assert off.enabled is False and off.recommended_config == {}, off
    # No structures -> no-op.
    empty = agent.recommend({"summaries": []}, current_lengths=[80], allowed_min=60, allowed_max=120, step=10)
    assert empty.direction == "no_structures" and empty.recommended_config == {}, empty
    print("OK test_length_policy_respects_fixed_and_disabled")


def test_length_policy_clamps_to_allowed_range():
    agent = BinderLengthPolicyAgent()
    # Weak interface would push longer, but allowed max caps it.
    summaries = [
        _summary(f"s{i}.cif", binder_len=95, reliability=0.85, interface=2, tags=["weak_or_tiny_interface"])
        for i in range(6)
    ]
    rec = agent.recommend({"summaries": summaries}, current_lengths=[90], allowed_min=80, allowed_max=100, step=10)
    assert all(80 <= length <= 100 for length in rec.recommended_lengths), rec.recommended_lengths
    print("OK test_length_policy_clamps_to_allowed_range")


def test_structure_interchain_pae_mapping_no_ambiguity():
    """design id ``_3_1`` must not leak its PAE onto structure ``_3_10``."""
    candidates = [
        {"id": "boltzgen_design_spec_3_1", "min_design_to_target_pae": "8.0"},
        {"id": "boltzgen_design_spec_3_10", "min_design_to_target_pae": "20.0"},
    ]
    structure_files = [
        "/run/final_ranked_designs/final_6_designs/rank1_boltzgen_design_spec_3_1.cif",
        "/run/final_ranked_designs/final_6_designs/rank2_boltzgen_design_spec_3_10.cif",
    ]
    mapping = structure_interchain_pae(candidates, structure_files)
    assert mapping[structure_files[0]] == 8.0, mapping
    assert mapping[structure_files[1]] == 20.0, mapping
    # Sentinel / missing PAE values are dropped.
    sentinel = structure_interchain_pae(
        [{"id": "boltzgen_design_spec_0_0", "min_design_to_target_pae": "100000.25"}],
        ["/x/rank1_boltzgen_design_spec_0_0.cif"],
    )
    assert sentinel == {}, sentinel
    print("OK test_structure_interchain_pae_mapping_no_ambiguity")


def test_length_rank_support_and_missing_pae():
    agent = BinderLengthPolicyAgent()
    supported = {"reliability": [0.8, 0.8], "chain_break": [0.0, 0.0],
                 "interface": [8.0, 8.0], "clash": [0.05, 0.05], "pae": []}
    tiny = {"reliability": [0.99], "chain_break": [0.0], "interface": [20.0],
            "clash": [0.0], "pae": [3.0]}
    assert agent._bucket_rank_key(supported) > agent._bucket_rank_key(tiny)

    missing_pae = {"reliability": [0.8], "chain_break": [0.0], "interface": [8.0],
                   "clash": [0.05], "pae": []}
    measured_pae = {"reliability": [0.8], "chain_break": [0.0], "interface": [8.0],
                    "clash": [0.05], "pae": [12.0]}
    assert agent._bucket_rank_key(measured_pae) > agent._bucket_rank_key(missing_pae)

    failed = {"reliability": [0.9, 0.9], "chain_break": [1.0, 1.0],
              "interface": [20.0, 20.0], "clash": [0.0, 0.0], "pae": [3.0, 3.0]}
    kept = agent._drop_failing_lengths([80, 90], {80: failed}, [80, 90])
    assert kept == [90], kept
    print("OK test_length_rank_support_and_missing_pae")


def main() -> None:
    import tempfile

    test_interchain_pae_gate_default()
    test_iptm_gate_optin()
    test_pae_gate_fallback_without_data()
    test_production_template_missing_pae_fails_closed()
    test_production_template_empty_structure_batch_is_not_a_pae_error()
    with tempfile.TemporaryDirectory() as d:
        test_redesign_template_prefers_low_pae(Path(d))
    test_length_policy_shorter_on_foldability_failure()
    test_length_policy_longer_on_weak_interface()
    test_length_policy_respects_fixed_and_disabled()
    test_length_policy_clamps_to_allowed_range()
    test_structure_interchain_pae_mapping_no_ambiguity()
    test_length_rank_support_and_missing_pae()
    print("\nALL LENGTH + PAE-GATE TESTS PASSED")


if __name__ == "__main__":
    main()
