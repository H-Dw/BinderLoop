#!/usr/bin/env python3
"""Deterministic tests for the harness strategy improvements:

* binder/target chain auto-detection and chain-agnostic hotspot matching,
* data-driven epitope target cropping,
* cross-round fragment-template library merge,
* quality-aware backtracking (RollbackController).

Run: python scripts/test_strategy_improvements.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import asdict

from binderloop.analysis.structure_features import (
    analyze_binder_structure,
    detect_binder_target_chains,
    parse_structure,
    _contacts,
    _contact_type,
    _dist,
    _contact_matches_hotspot,
    _hotspot_residue_number,
)
from binderloop.analysis.epitope import propose_epitope_crop, aggregate_engaged_residues
from binderloop.agents.structure_evaluation_agent import StructureEvaluationAgent
from binderloop.agents.fragment_template_mining_agent import FragmentTemplateMiningAgent
from binderloop.agents.active_learning_policy_agent import ActiveLearningPolicyAgent
from binderloop.agents.evaluation_agent import CandidateEvaluation, EvaluationSummary
from binderloop.agents.strategy_arm_ranking_agent import StrategyArmRankingAgent
from binderloop.agents.hypothesis_agent import HypothesisAgent
from binderloop.active_learning.examples import build_active_learning_examples, extract_round_examples, hard_negative_iptm_min_from_additional_filters
from binderloop.active_learning.strategy import StrategyLevelActiveLearner
from binderloop.active_learning.rollback import RollbackController, RoundOutcome, round_reward
from binderloop.analysis.core_objective import core_rank_key, rank_by_core_objective, round_rank_key
from binderloop.models.base import DesignJob
from binderloop.skills import SkillRegistry


def _atom(serial: int, name: str, resname: str, chain: str, resseq: int, x: float, y: float, z: float, element: str = "C") -> str:
    line = (
        "ATOM  "
        + f"{serial:>5d}"
        + " "
        + f"{name:^4s}"
        + " "
        + f"{resname:>3s}"
        + " "
        + f"{chain}"
        + f"{resseq:>4d}"
        + " "
        + "   "
        + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        + "  1.00  0.00"
        + " " * 10
        + f"{element:>2s}"
    )
    assert len(line) >= 54, (len(line), line)
    # Sanity-check the columns the parser reads.
    assert line[17:20].strip() == resname
    assert line[21] == chain
    assert int(line[22:26]) == resseq
    return line


def _synthetic_complex_pdb() -> str:
    """Binder chain X (5 res) packed against target chain Y (res 10,11,12)."""
    lines = []
    serial = 1
    # Binder backbone along the x axis.
    for i in range(1, 6):
        lines.append(_atom(serial, "CA", "ALA", "X", i, (i - 1) * 3.8, 0.0, 0.0))
        serial += 1
    # Target residues placed ~4 A from selected binder residues -> real contacts.
    for resseq, x in [(10, 0.0), (11, 3.8), (12, 7.6)]:
        lines.append(_atom(serial, "CA", "GLU", "Y", resseq, x, 4.0, 0.0))
        serial += 1
    return "\n".join(lines) + "\n"


def test_chain_detection():
    # Binder length closest to chain count wins.
    counts = {"A": 80, "B": 194}
    binder, targets, note = detect_binder_target_chains(counts, binder_length=80, configured_binder_chain="D", configured_target_chains=["E"])
    assert binder == "A", (binder, note)
    assert targets == ["B"], targets
    assert "overrode_configured_binder=D" in note, note

    # No length hint -> smallest chain is the binder.
    binder2, targets2, _ = detect_binder_target_chains({"A": 200, "C": 60}, configured_binder_chain=None)
    assert binder2 == "C", binder2
    assert targets2 == ["A"], targets2

    # Empty structure falls back to configured.
    binder3, targets3, note3 = detect_binder_target_chains({}, configured_binder_chain="D", configured_target_chains=["E"])
    assert binder3 == "D" and targets3 == ["E"], (binder3, targets3)
    assert note3 == "no_chains_parsed"
    print("OK test_chain_detection")


def test_hotspot_number_matching():
    assert _hotspot_residue_number("E:153") == 153
    assert _hotspot_residue_number("153") == 153
    assert _hotspot_residue_number("A/77") == 77
    # Chain letters differ (config E vs output B) but residue number matches.
    assert _contact_matches_hotspot("B:153", "E:153") is True
    assert _contact_matches_hotspot("B:154", "E:153") is False
    assert _contact_matches_hotspot("E:153", "E:153") is True
    print("OK test_hotspot_number_matching")


def test_analyze_structure_autodetect(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(_synthetic_complex_pdb(), encoding="utf-8")
    # Configure WRONG chains on purpose; auto-detect must still find the interface.
    s = analyze_binder_structure(
        pdb,
        binder_chain="B",
        target_chains=["E"],
        hotspots=["Z:11"],  # different chain letter, residue number 11 exists in target Y
        binder_length=5,
    )
    assert s.binder_chain == "X", s.binder_chain
    assert s.target_chains == ["Y"], s.target_chains
    assert s.interface_contact_count >= 1, s.interface_contact_count
    assert s.hotspot_contacts.get("Z:11", 0) >= 1, s.hotspot_contacts
    assert s.binder_residue_count == 5, s.binder_residue_count
    print("OK test_analyze_structure_autodetect")

    # Batch agent surfaces a data-quality flag and binder length hint passes through.
    agent = StructureEvaluationAgent()
    batch = agent.analyze_structures([str(pdb)], binder_chain="B", target_chains=["E"], hotspots=["Z:11"], binder_length=5)
    assert batch.total_structures == 1
    assert batch.interface_data_quality.get("status") == "ok", batch.interface_data_quality
    assert batch.interface_data_quality.get("zero_contact_fraction") == 0.0
    print("OK test_structure_eval_agent_data_quality")


def test_boltzgen_multilength_chain_relabel_detection(tmp_path: Path):
    """BoltzGen output chain A is the generated binder; target chain is shifted."""
    pdb = tmp_path / "rank01_boltzgen_design_spec_len100_0_0.cif"
    lines = []
    serial = 1
    for i in range(1, 101):
        lines.append(_atom(serial, "CA", "ALA", "A", i, (i - 1) * 3.8, 0.0, 0.0))
        serial += 1
    for resseq, x in [(151, 0.0), (153, 3.8), (155, 7.6)]:
        lines.append(_atom(serial, "CA", "GLU", "B", resseq, x, 4.0, 0.0))
        serial += 1
    pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")

    agent = StructureEvaluationAgent()
    batch = agent.analyze_structures(
        [str(pdb)],
        binder_chain="B",
        target_chains=["E"],
        hotspots=["E:153"],
        binder_length=[80, 85, 90, 95, 100],
    )
    summary = batch.summaries[0]
    assert summary["binder_chain"] == "A", summary
    assert summary["target_chains"] == ["B"], summary
    assert "binder_by_length" in summary["chain_detection_note"], summary["chain_detection_note"]
    assert summary["hotspot_contacts"]["E:153"] >= 1, summary["hotspot_contacts"]
    print("OK test_boltzgen_multilength_chain_relabel_detection")


def test_additional_filters_define_analysis_candidates(tmp_path: Path):
    from binderloop.agents.context_compaction import (
        build_metric_facts,
        compact_context_for_input_config,
        compact_structural_analysis,
    )
    from binderloop.agents.evaluation_agent import EvaluationAgent
    from binderloop.config import load_config
    from binderloop.orchestration.orchestrator import BinderDesignOrchestrator

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "sc2rbd_structured_task.yaml")
    cfg.search_space.boltzgen["additional_filters"] = ["iptm>0.35"]
    orch = BinderDesignOrchestrator(cfg, out_dir=str(tmp_path / "run"), max_rounds=1)

    rows = [
        {"id": "design_low", "design_to_target_iptm": "0.20", "design_ptm": "0.8", "pass_iptm_filter": "False"},
        {"id": "design_high", "design_to_target_iptm": "0.42", "design_ptm": "0.8", "pass_iptm_filter": "True"},
    ]
    filtered, meta = orch._analysis_candidates(rows)
    assert [row["id"] for row in filtered] == ["design_high"], filtered
    assert meta["filtering_applied"] is True
    assert meta["input_candidate_count"] == 2
    assert meta["analysis_candidate_count"] == 1

    summary = EvaluationAgent().evaluate_candidates(filtered)
    summary.candidate_filtering = meta
    facts = build_metric_facts(asdict(summary), candidates=filtered)
    assert facts["total_candidates"] == 1, facts
    assert facts["input_candidate_count"] == 2, facts
    assert facts["additional_filter_pass"]["pass_count"] == 1, facts
    assert facts["additional_filter_pass"]["fail_count"] == 1, facts

    compact = compact_context_for_input_config(
        target_name="sc2rbd",
        current_config={"additional_filters": ["iptm>0.35"]},
        diagnostic_report={},
        evaluation_summary={**asdict(summary), "metric_facts": facts},
        round_id=1,
    )
    assert compact["evaluation_summary"]["candidate_filtering"]["analysis_candidate_count"] == 1, compact
    assert compact["evaluation_summary"]["metric_facts"]["input_candidate_count"] == 2, compact
    structural = compact_structural_analysis({
        "total_structures": 1,
        "summaries": [{
            "binder_chain": "A",
            "target_chains": ["B"],
            "chain_detection_note": "binder_by_length(len=100)",
        }],
    }, include_summaries=False)
    assert structural["output_chain_mapping"]["binder_chain"] == "A", structural
    assert "entity-order" in structural["output_chain_mapping"]["namespace_note"], structural
    print("OK test_additional_filters_define_analysis_candidates")


def test_contrastive_active_learning_examples_and_skills():
    rows = [
        {
            "id": "strong_positive",
            "design_to_target_iptm": "0.56",
            "min_design_to_target_pae": "7.5",
            "design_ptm": "0.82",
            "designfolding-filter_rmsd": "1.4",
            "pass_iptm_filter": "True",
        },
        {
            "id": "iptm_high_bad_geometry",
            "design_to_target_iptm": "0.53",
            "min_design_to_target_pae": "14.0",
            "design_ptm": "0.83",
            "designfolding-filter_rmsd": "1.2",
            "pass_iptm_filter": "True",
        },
        {
            "id": "near_miss",
            "design_to_target_iptm": "0.44",
            "min_design_to_target_pae": "8.0",
            "design_ptm": "0.80",
            "designfolding-filter_rmsd": "1.1",
            "pass_iptm_filter": "True",
        },
        {
            "id": "low_noise",
            "design_to_target_iptm": "0.20",
            "min_design_to_target_pae": "20.0",
            "design_ptm": "0.78",
            "designfolding-filter_rmsd": "1.0",
            "pass_iptm_filter": "False",
        },
    ]
    lower, source = hard_negative_iptm_min_from_additional_filters(["iptm>0.35"])
    assert lower == 0.35, (lower, source)
    assert hard_negative_iptm_min_from_additional_filters(["ALA_fraction<0.3"])[0] == 0.0

    round_examples = extract_round_examples(rows, round_id=3, hard_negative_iptm_min=lower)
    assert len(round_examples["strict_positive_examples"]) == 1, round_examples
    assert len(round_examples["near_miss_examples"]) == 2, round_examples
    assert len(round_examples["other_negative_examples"]) == 1, round_examples
    assert "candidate_id" not in str(round_examples)
    assert all(item["weight"] < item["confidence"] for item in round_examples["near_miss_examples"])
    assert round_examples["counts"]["strict_positive"] == 1, round_examples
    assert round_examples["counts"]["near_miss"] == 2, round_examples
    assert round_examples["counts"]["other_negative"] == 1, round_examples
    assert round_examples["counts"]["iptm_ge_0_5_before_quality_gates"] == 2, round_examples
    assert round_examples["counts"]["iptm_ge_0_5_rejected_by_quality_gates"] == 1, round_examples

    context_examples = build_active_learning_examples(
        round_id=4,
        current_candidates=rows[:1],
        prior_rounds=[{"current_round": round_examples}],
        additional_filters=["iptm>0.35"],
    )
    assert context_examples["current_round"]["round_id"] == 4, context_examples
    assert context_examples["prior_rounds"]["counts"]["strict_positive"] == 1, context_examples
    assert context_examples["prior_rounds"]["counts"]["near_miss"] == 2, context_examples
    assert context_examples["prior_rounds"]["counts"]["other_negative"] == 1, context_examples
    assert context_examples["cumulative"]["strict_positive_count"] == 2, context_examples
    assert context_examples["thresholds"]["near_miss_examples"]["design_to_target_iptm_min_exclusive"] == 0.35, context_examples

    default_examples = build_active_learning_examples(
        round_id=5,
        current_candidates=rows,
        prior_rounds=[],
        additional_filters=["ALA_fraction<0.3"],
    )
    assert sum(len(default_examples["current_round"][pool]) for pool in ("near_miss_examples", "other_negative_examples")) >= 2
    assert "candidate_id" not in str(default_examples)
    assert default_examples["thresholds"]["near_miss_examples"]["design_to_target_iptm_min_exclusive"] == 0.0, default_examples

    from binderloop.agents.context_compaction import compact_active_learning_examples
    legacy = compact_active_learning_examples({
        "current_round": {
            "positive_examples": [{"candidate_id": "legacy_p", "label": "strict_positive"}],
            "near_miss_examples": [{"candidate_id": "legacy_n", "label": "near_miss"}],
            "hard_negative_examples": [{"candidate_id": "legacy_o", "label": "hard_negative"}],
        }
    })
    compact_current = legacy["current_round"]
    assert "candidate_id" not in str(compact_current)
    assert compact_current["other_negative_examples"][0]["label"] == "other_negative"
    assert compact_current["counts"] == {
        "strict_positive": 1,
        "near_miss": 1,
        "other_negative": 1,
    }
    assert "positive_examples" not in compact_current
    assert "hard_negative_examples" not in compact_current

    from binderloop.agents.evaluation_agent import EvaluationAgent
    assert EvaluationAgent.PRIMARY_GATE_THRESHOLDS == {
        "design_to_target_iptm": 0.50,
        "min_design_to_target_pae": 10.0,
        "design_ptm": 0.70,
        "designfolding_filter_rmsd": 2.5,
    }

    repo = Path(__file__).resolve().parents[1]
    registry = SkillRegistry.from_yaml(repo / "configs" / "skills" / "binder_skills.yaml")
    llm_skills = registry.select(
        agent_name="HypothesisAgent",
        context={"active_learning_examples": context_examples},
        skill_types=["llm_reasoning"],
    )
    assert any(skill["id"] == "contrastive-active-learning-examples" for skill in llm_skills), llm_skills
    strategy_skills = registry.select(
        agent_name="StrategyLevelActiveLearner",
        context={"active_learning_examples": context_examples},
        skill_types=["strategy"],
    )
    assert any(skill["id"] == "strategy-contrastive-positive-exploit" for skill in strategy_skills), strategy_skills
    print("OK test_contrastive_active_learning_examples_and_skills")


def test_epitope_crop_focus_and_keep():
    # Designs engage the requested hotspot epitope -> crop tightens around it.
    summaries_focus = [
        {"reliability_score": 0.8, "contacts_preview": [{"target_residue": f"B:{n}"} for n in (150, 151, 152, 153, 154)]}
        for _ in range(5)
    ]
    counts = aggregate_engaged_residues(summaries_focus, min_reliability=0.5)
    assert counts.get(153, 0) == 5, counts
    crop = propose_epitope_crop(summaries_focus, target_chain="E", requested_hotspots=["E:153"], mode="auto", margin=4)
    assert crop.mode == "hotspot_focus", crop.mode
    assert crop.recommended_config.get("target_include"), crop.recommended_config
    window = crop.crop_window
    assert window[0] <= 153 <= window[1], window

    # Designs engage an off-target patch -> keep full chain, raise hotspot priority only.
    summaries_wrong = [
        {"reliability_score": 0.8, "contacts_preview": [{"target_residue": f"B:{n}"} for n in (40, 41, 42, 43)]}
        for _ in range(5)
    ]
    crop2 = propose_epitope_crop(summaries_wrong, target_chain="E", requested_hotspots=["E:153"], mode="auto")
    assert crop2.mode == "keep_full_chain_raise_hotspots", crop2.mode
    assert "target_include" not in crop2.recommended_config, crop2.recommended_config
    assert crop2.recommended_config.get("prioritize_hotspots") is True, crop2.recommended_config

    # Forced engaged_focus crops to the observed off-target patch.
    crop3 = propose_epitope_crop(summaries_wrong, target_chain="E", requested_hotspots=["E:153"], mode="engaged_focus", margin=3)
    assert crop3.recommended_config.get("target_include"), crop3.recommended_config
    assert crop3.crop_window[0] <= 40, crop3.crop_window
    print("OK test_epitope_crop_focus_and_keep")


def test_template_library_merge():
    mine = FragmentTemplateMiningAgent()
    source = Path(__file__).resolve()
    structural = {
        "summaries": [
            {
                "structure_file": str(source),
                "binder_chain": "A",
                "high_quality_fragments": [
                    {"fragment_id": "A:1-A:8", "start_residue": 1, "end_residue": 8, "residue_ids": ["A:1", "A:8"], "quality_score": 0.82, "quality_label": "high", "reasons": ["dense_target_interface"]}
                ],
                "low_quality_fragments": [],
                "contacts_preview": [{"binder_residue": "A:1", "target_residue": "B:153", "contact_type": "polar"}],
                "hotspot_contacts": {"E:153": 1},
            }
        ]
    }
    prior = [
        {"template_id": "frag_old", "reuse_mode": "preserve", "quality_score": 0.9, "binder_residue_span": [10, 18]},
    ]
    batch = mine.mine_templates(
        structural,
        round_id=1,
        prior_templates=prior,
        target_chain="E",
        requested_hotspots=["E:153"],
        interchain_pae_by_structure={str(source): 7.5},
        templates_enabled=False,
    )
    library_ids = {t.get("template_id") for t in batch.library}
    assert "frag_old" in library_ids, library_ids  # carried across rounds
    assert len(batch.templates) >= 1
    assert any(t.get("reuse_mode") == "preserve" for t in batch.library)
    assert "binder_template" not in batch.recommended_config, batch.recommended_config
    assert batch.analysis_metadata["template_insertion_decision"]["reason"] == "fragment_templates_disabled"
    print("OK test_template_library_merge")


def test_fragment_template_topk_and_strategy_allocation(tmp_path: Path):
    source = tmp_path / "template_source.cif"
    lines = []
    serial = 1
    for resid in range(1, 25):
        lines.append(_atom(serial, "CA", "ALA", "A", resid, float(resid), 0.0, 0.0)); serial += 1
    for resid, x in ((151, 0.0), (152, 4.0), (153, 8.0)):
        lines.append(_atom(serial, "CA", "GLU", "B", resid, x, 4.0, 0.0)); serial += 1
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summaries = []
    for idx, start in enumerate((1, 9, 17), 1):
        summaries.append({
            "structure_file": str(source),
            "binder_chain": "A",
            "high_quality_fragments": [{
                "fragment_id": f"A:{start}-A:{start + 7}",
                "start_residue": start,
                "end_residue": start + 7,
                "residue_ids": [f"A:{start}", f"A:{start + 7}"],
                "quality_score": 0.9 - idx * 0.01,
                "quality_label": "high",
                "reasons": ["dense_target_interface"],
            }],
            "low_quality_fragments": [],
            "contacts_preview": [
                {"binder_residue": f"A:{start}", "target_residue": f"B:{resid}", "contact_type": "polar"}
                for resid in (151, 152, 153)
            ],
            "hotspot_contacts": {"E:153": 1},
        })
    batch = FragmentTemplateMiningAgent().mine_templates(
        {"summaries": summaries},
        round_id=2,
        interchain_pae_by_structure={str(source): 5.0},
        templates_enabled=True,
        template_top_k=2,
        template_artifact_dir=tmp_path / "template_artifacts",
        current_target_structure=str(source),
        target_chain="B",
    )
    templates = batch.recommended_config.get("binder_templates")
    assert len(templates) == 2, batch.recommended_config

    parent = DesignJob("parent", "target.cif", "E", ["E:153"], 100, params={"binder_lengths": [100], "binder_templates": templates, "template_conditioned_fraction": 0.4}, output_dir=str(tmp_path / "parent"))
    proposal = StrategyLevelActiveLearner().propose_next(3, [parent], [], str(tmp_path / "run"), policy_update={})
    assert len(proposal.jobs) == 1, [job.job_id for job in proposal.jobs]
    assert proposal.jobs[0].params.get("template_conditioned") is True
    assert proposal.jobs[0].params.get("template_count") == 1
    assert abs(float(proposal.jobs[0].params["round_budget_weight"]) - 1.0) < 1e-9
    assert "template_free_exploration" not in proposal.jobs[0].params
    print("OK test_fragment_template_topk_and_strategy_allocation")


def test_template_foldability_guard_lowers_fraction():
    summary = EvaluationSummary(
        total_candidates=2,
        success_count=0,
        failure_count=2,
        tag_counts={"folding_failure": 1},
        top_candidates=[
            CandidateEvaluation("c1", 0.1, {"binder_plddt": 0.62}, ["folding_failure"], raw={"design_ptm": "0.62", "designfolding-filter_rmsd": "3.4"}),
            CandidateEvaluation("c2", 0.1, {"binder_plddt": 0.66}, ["binding_pose_failure"], raw={"design_ptm": "0.66", "designfolding-filter_rmsd": "3.1"}),
        ],
        failed_examples=[],
        observations=[],
    )
    proposal = ActiveLearningPolicyAgent().propose_next_boltzgen_params(
        summary,
        {
            "binder_template": {"mode": "structure_redesign"},
            "template_conditioned_fraction": 0.6,
            "binder_lengths": [85],
            "binder_length_range": [80, 120],
            "binder_length_step": 5,
            "hotspot_weight": 3.0,
            "epitope_crop_mode": "hotspot_focus",
        },
        round_id=2,
    )
    assert proposal.params_update["template_conditioned_fraction"] == 0.3, proposal.params_update
    assert proposal.params_update["epitope_crop_mode"] == "disabled", proposal.params_update
    assert proposal.params_update["binder_lengths"] == [80, 85, 90], proposal.params_update
    print("OK test_template_foldability_guard_lowers_fraction")


def _brute_force_contacts(binder, target, cutoff):
    best = {}
    for x in binder:
        if x.name.upper().startswith("H"):
            continue
        for y in target:
            if y.name.upper().startswith("H"):
                continue
            d = _dist(x.coord, y.coord)
            if d <= cutoff:
                key = (x.residue_id, y.residue_id)
                if key not in best or d < best[key][0]:
                    best[key] = (d, _contact_type(x, y))
    return best


def test_contacts_prescreen_equivalence(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(_synthetic_complex_pdb(), encoding="utf-8")
    atoms = parse_structure(pdb)
    binder = [a for a in atoms if a.chain == "X"]
    target = [a for a in atoms if a.chain == "Y"]
    for cutoff in (5.0, 2.0, 8.0):
        brute = _brute_force_contacts(binder, target, cutoff)
        opt = {(c.binder_residue, c.target_residue): (c.min_distance, c.contact_type) for c in _contacts(binder, target, cutoff)}
        assert set(brute) == set(opt), (cutoff, set(brute) ^ set(opt))
        assert all(abs(brute[k][0] - opt[k][0]) < 1e-9 for k in brute), cutoff
        assert all(brute[k][1] == opt[k][1] for k in brute), cutoff
    print("OK test_contacts_prescreen_equivalence")


def test_orchestrator_rollback_seed(tmp_path: Path):
    from binderloop.config import load_config
    from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
    from binderloop.memory import ExperimentMemory, RoundRecord
    from binderloop.active_learning.rollback import RollbackDecision

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "sc2rbd_structured_task.yaml")
    orch = BinderDesignOrchestrator(cfg, out_dir=str(tmp_path / "run"), max_rounds=5)

    jobs0 = orch._initial_jobs()
    jobs1 = [DesignJobClone(j, suffix="r1") for j in jobs0]
    replay_share = orch._round_design_cap // len(jobs1)
    for replay_job in jobs1:
        replay_job.params["num_designs"] = replay_share
    best_snapshot = orch._current_config_snapshot()
    best_snapshot["exploration_ratio"] = 0.3
    best_snapshot["binder_lengths"] = [80]
    best_boltzgen = dict(best_snapshot["boltzgen_config"])
    best_boltzgen.update({
        "alpha": 0.003,
        "noise_scale": 0.7,
        "step_scale": 0.8,
        "steps": 240,
        "config_overrides": [["filtering", "filter_bindingsite=false"]],
        "auxiliary_hotspots": ["E:154"],
    })
    best_snapshot["boltzgen_config"] = best_boltzgen
    best_snapshot.update(best_boltzgen)
    jobs1[0].params.update({
        "alpha": 0.003,
        "noise_scale": 0.7,
        "step_scale": 0.8,
        "steps": 240,
        "config_overrides": [["filtering", "filter_bindingsite=false"]],
        "auxiliary_hotspots": ["E:154"],
        "strategy_arm": "best_arm",
        "branch_id": "best_branch",
        "execution_retry_source_job_id": "stale_retry",
        "execution_retry_preserve_budget": True,
    })

    mem = ExperimentMemory(experiment_id="t", target={})
    mem.rounds = [
        RoundRecord(round_id=0, jobs=[asdict(j) for j in jobs0], config_snapshot=orch._current_config_snapshot(), reward=0.30),
        RoundRecord(round_id=1, jobs=[asdict(j) for j in jobs1], config_snapshot=best_snapshot, reward=0.90),
        RoundRecord(round_id=2, jobs=[asdict(j) for j in jobs0], config_snapshot=orch._current_config_snapshot(), reward=0.40),
    ]
    decision = RollbackDecision(
        action="replay_best", branch_from_round=1, best_round=1, best_reward=0.90,
        current_reward=0.40, is_regression=True, consecutive_regressions=1, relative_drop=0.55,
        rationale="test",
    )
    orch.cfg.search_space.boltzgen["degraded_only_stale_key"] = "must disappear"
    orch.cfg.search_space.boltzgen["alpha"] = 0.02
    orch.cfg.active_learning.exploration_ratio = 0.55
    parents, seed = orch._prepare_rollback_seed(mem, decision, regressed_update={"length_delta_hint": 20})
    parent_ids = {j.job_id for j in parents}
    assert parent_ids == {j.job_id for j in jobs1}, parent_ids
    assert seed.get("exploration_ratio") == 0.3, seed
    assert seed.get("binder_lengths") == [80], seed
    assert orch.cfg.search_space.boltzgen["alpha"] == 0.003
    assert orch.cfg.search_space.boltzgen["noise_scale"] == 0.7
    assert orch.cfg.search_space.boltzgen["step_scale"] == 0.8
    assert orch.cfg.search_space.boltzgen["steps"] == 240
    assert orch.cfg.search_space.boltzgen["config_overrides"] == [["filtering", "filter_bindingsite=false"]]
    assert orch.cfg.search_space.boltzgen["auxiliary_hotspots"] == ["E:154"]
    assert "degraded_only_stale_key" not in orch.cfg.search_space.boltzgen

    replay, applied, replay_snapshot, source_ids = orch._prepare_exact_rollback_replay(
        mem, decision, next_round_id=3,
    )
    replay_logical = orch._logical_jobs_for_memory(replay)
    assert source_ids == [job.job_id for job in jobs1]
    assert len(replay_logical) == len(jobs1)
    for source, clone in zip(jobs1, replay_logical):
        assert clone.job_id != source.job_id
        assert clone.output_dir != source.output_dir
        assert clone.target_structure == source.target_structure
        assert clone.chain_id == source.chain_id
        assert clone.hotspots == source.hotspots
        assert clone.binder_length == source.binder_length
        assert clone.seed == source.seed
        expected_params = dict(source.params)
        expected_params.pop("execution_retry_source_job_id", None)
        expected_params.pop("execution_retry_preserve_budget", None)
        clone_params = dict(clone.params)
        for generated_key in ("branch_id", "effective_intervention_digest", "execution_semantic_digest", "attribution_identity_digest", "job_identity", "replay_source_job_id", "replay_source_job_identity_digest", "arm_digest", "arm_root", "execution_job_id", "logical_job_id"):
            clone_params.pop(generated_key, None)
            expected_params.pop(generated_key, None)
        assert clone_params == expected_params, (clone_params, expected_params)
        from binderloop.strategy_governance import effective_semantic_digest
        assert clone.params["job_identity"]["execution_semantic_digest"] == effective_semantic_digest(clone)
        assert clone.params["branch_id"]
        assert clone.params["replay_source_job_id"] == source.job_id
    assert applied["exploration_ratio"] == 0.3
    assert replay_snapshot["boltzgen_config"] == orch.cfg.search_space.boltzgen
    assert all(not bool((job.params.get("job_identity") or {}).get("finalized")) for job in replay)

    compatibility = RollbackDecision(
        action="branch_from_best", branch_from_round=1, best_round=1, best_reward=0.90,
        current_reward=0.40, is_regression=True, consecutive_regressions=2,
        relative_drop=0.55, rationale="legacy checkpoint",
    )
    compat_replay, _, _, _ = orch._prepare_exact_rollback_replay(mem, compatibility, next_round_id=4)
    compat_params = dict(orch._logical_jobs_for_memory(compat_replay)[0].params)
    replay_params = dict(replay_logical[0].params)
    for generated_key in ("branch_id", "effective_intervention_digest", "execution_semantic_digest", "attribution_identity_digest", "job_identity", "replay_source_job_id", "replay_source_job_identity_digest", "arm_digest", "arm_root", "execution_job_id", "logical_job_id"):
        compat_params.pop(generated_key, None)
        replay_params.pop(generated_key, None)
    assert compat_params == replay_params

    executable = orch._bind_execution_identities_if_needed(replay, round_id=3)
    assert all(job.params["job_identity"]["finalized"] is True for job in executable)
    assert all(job.params["job_identity"]["execution_slot"] is not None for job in executable)
    assert len({job.job_id for job in executable}) == len(executable)
    assert all("/jobs/" in str(job.output_dir) for job in executable)

    suppressed = orch._rollback_suppressed_merge_report(
        rollback_decision=decision,
        input_config={"alpha": 0.02},
        binder_length_update={"binder_lengths": [100]},
        fragment_template_update={"binder_template": {"id": "degraded"}},
    )
    assert suppressed["rollback_policy_suppressed"] is True
    assert suppressed["applied_update"] == {}
    assert suppressed["current_round_inputs"] == "audit_only_suppressed"
    assert orch.cfg.search_space.boltzgen["alpha"] == 0.003
    print("OK test_orchestrator_rollback_seed")


def DesignJobClone(job, *, suffix):
    from binderloop.models.base import DesignJob
    return DesignJob(
        job_id=f"{job.job_id}_{suffix}",
        target_structure=job.target_structure,
        chain_id=job.chain_id,
        hotspots=list(job.hotspots),
        binder_length=job.binder_length,
        seed=job.seed,
        params=dict(job.params),
        output_dir=f"{job.output_dir}_{suffix}",
    )


def test_rollback_controller_trajectory():
    # Replicate the reference run: 0.26 -> 0.55 -> 0.57 -> 0.30 -> 0.18.
    ctrl = RollbackController(enabled=True, regression_tolerance=0.15, patience=1)
    rewards = [
        (0, 0.261, 0),
        (1, 0.547, 0),
        (2, 0.573, 3),
        (3, 0.302, 0),
        (4, 0.177, 0),
    ]
    decisions = []
    for rid, iptm, succ in rewards:
        d = ctrl.observe(RoundOutcome(round_id=rid, reward=round_reward(iptm, succ), best_iptm=iptm, success_count=succ, arm_signature=f"arm{rid}"))
        decisions.append(d)
    assert decisions[0].action == "advance"
    assert decisions[1].action == "advance"
    assert decisions[2].action == "advance" and decisions[2].best_round == 2
    # Round 3 regressed hard -> roll back to the peak (round 2).
    assert decisions[3].action == "retest_best_config", decisions[3].to_dict()
    assert decisions[3].branch_from_round == 2, decisions[3].branch_from_round
    # The best-config retest is capped; continued regression creates a fresh branch.
    assert decisions[4].action == "branch_from_best", decisions[4].to_dict()
    assert decisions[4].branch_from_round == 2

    # Disabled controller always advances.
    off = RollbackController(enabled=False)
    for rid, iptm, succ in rewards:
        d = off.observe(RoundOutcome(round_id=rid, reward=round_reward(iptm, succ), best_iptm=iptm, success_count=succ))
        assert d.action == "advance"
    print("OK test_rollback_controller_trajectory")




def test_zero_pass_quality_signal_and_exploration_governance(tmp_path: Path):
    from binderloop.config import load_config
    from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
    from binderloop.models.base import DesignJob

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "sc2rbd_structured_task.yaml")
    orch = BinderDesignOrchestrator(cfg, out_dir=str(tmp_path / "governance"), max_rounds=1)
    rows = [
        {"id": "near", "design_to_target_iptm": 0.47, "min_design_to_target_pae": 5.0, "design_ptm": 0.82, "designfolding-filter_rmsd": 0.8, "pass_iptm_filter": "False", "pass_filters": "False"},
        {"id": "negative", "design_to_target_iptm": 0.15, "min_design_to_target_pae": 20.0, "design_ptm": 0.75, "designfolding-filter_rmsd": 3.0, "pass_iptm_filter": "False", "pass_filters": "False"},
    ]
    examples = build_active_learning_examples(round_id=5, current_candidates=rows, additional_filters=["iptm>0.35"])
    assert examples["current_round"]["counts"]["strict_positive"] == 0
    assert examples["current_round"]["counts"]["near_miss"] >= 1
    assert examples["current_round"]["counts"]["other_negative"] >= 1
    assert not orch._detect_round_execution_failure(total_candidates=len(rows), execution_records=[])[0]

    base = DesignJob("base", cfg.target.structure_path, cfg.target.chain_id, [], 80, 0,
                     {"alpha": 0.003, "noise_scale": 0.7, "binder_lengths": [80, 85, 90, 95], "inverse_fold_avoid": "C"}, str(tmp_path / "base"))
    proposed = DesignJob("proposed", cfg.target.structure_path, cfg.target.chain_id, [], 95, 0,
                         {"alpha": 0.005, "noise_scale": 0.6, "binder_lengths": [85, 90, 95, 100], "inverse_fold_avoid": "CP", "auxiliary_hotspots": ["E:158"]}, str(tmp_path / "proposed"))
    governed = orch._govern_exploration_jobs([proposed], current_jobs=[base], next_round_id=6, strict_positive_count=5)
    assert len(governed) == orch.cfg.active_learning.branch_width
    round_job = governed[0]
    assert all(job.params.get("random_sampler_fallback") for job in governed[1:])
    assert round_job.params["binder_lengths"] == [85, 90, 95, 100]
    assert round_job.params["auxiliary_hotspots"] == ["E:158"]
    assert round_job.params["alpha"] == 0.005
    assert round_job.params["noise_scale"] == 0.6
    assert round_job.params["round_budget_weight"] == 1.0
    assert round_job.params["branch_id"] == "round_6"
    assert "controlled_comparison" not in round_job.params
    assert "probe_policy" not in round_job.params
    print("OK test_zero_pass_quality_signal_and_exploration_governance")


def test_rollback_controller_execution_failure_no_branch_switch():
    ctrl = RollbackController(enabled=True, regression_tolerance=0.15, patience=1)
    first = ctrl.observe(RoundOutcome(round_id=0, reward=0.4, best_iptm=0.4, success_count=0, arm_signature="exploit"))
    assert first.action == "advance"

    failed = ctrl.observe(
        RoundOutcome(
            round_id=1,
            reward=0.0,
            best_iptm=0.0,
            success_count=0,
            arm_signature="exploit",
            execution_failed=True,
            execution_failure_reason="pre-submit config validation failed",
        )
    )
    assert failed.action == "advance", failed.to_dict()
    assert failed.is_regression is False
    assert failed.blocked_arm_signature is None
    assert failed.branch_from_round == 1
    print("OK test_rollback_controller_execution_failure_no_branch_switch")


def test_round_rank_rollback_respects_noise_tolerance():
    ctrl = RollbackController(enabled=True, patience=2)
    first = ctrl.observe(RoundOutcome(
        round_id=0, reward=0.2, round_rank_key=[0.2, 0.1, 0.6, -5.0, -1.0],
    ))
    assert first.action == "advance"
    noisy = ctrl.observe(RoundOutcome(
        round_id=1, reward=0.2, round_rank_key=[0.2, 0.05, 0.59, -5.2, -1.1],
    ))
    assert noisy.action == "advance" and noisy.is_regression is False, noisy.to_dict()
    regressed = ctrl.observe(RoundOutcome(
        round_id=2, reward=0.1, round_rank_key=[0.15, 0.2, 0.7, -4.0, -0.8],
        arm_signature="hotspot_repair",
    ))
    assert regressed.action == "retest_best_config" and regressed.is_regression is True, regressed.to_dict()
    assert regressed.blocked_arm_signature == "hotspot_repair", regressed.to_dict()
    print("OK test_round_rank_rollback_respects_noise_tolerance")


def test_skill_registry_activation():
    repo = Path(__file__).resolve().parents[1]
    registry = SkillRegistry.from_yaml(repo / "configs" / "skills" / "binder_skills.yaml")
    directory_registry = SkillRegistry.from_yaml(repo / "configs" / "skills")
    summary = registry.audit_summary()
    assert summary["counts_by_type"]["llm_reasoning"] >= 2, summary
    assert summary["counts_by_type"]["strategy"] >= 3, summary
    assert summary["counts_by_type"]["deterministic_policy"] >= 3, summary
    assert directory_registry.audit_summary()["skill_count"] >= summary["skill_count"], directory_registry.audit_summary()

    context = {
        "evaluation": {"tag_counts": {"binding_pose_failure": 3, "hotspot_miss": 1}},
        "structural_analysis": {"aggregate_tags": {"hotspot_not_covered": 2}},
    }
    quality_skills = registry.select(
        agent_name="BinderQualityAnalysisAgent",
        context=context,
        skill_types=["llm_reasoning", "deterministic_policy"],
    )
    ids = {skill["id"] for skill in quality_skills}
    assert "quality-interface-hotspot-reasoning" in ids, ids
    assert all("trigger_reason" in skill for skill in quality_skills)

    rollback_skills = registry.select(
        agent_name="RollbackController",
        context=context,
        skill_types=["deterministic_policy"],
    )
    assert any(skill["deterministic_controls"].get("non_overridable") for skill in rollback_skills), rollback_skills
    print("OK test_skill_registry_activation")


def test_strategy_skills_materialize_branch_jobs():
    parent = DesignJob(
        job_id="r0_round",
        target_structure="target.cif",
        chain_id="E",
        hotspots=["E:153"],
        binder_length=90,
        params={"hotspot_weight": 2.0, "binder_lengths": [85, 90, 95], "diffusion_batch_size": 1},
        output_dir="out/r0/round",
    )
    active_skills = [
        {
            "id": "strategy-relaxed-pose-explore",
            "type": "strategy",
            "params": {
                "arm_name": "sampler_explore",
                "budget_policy": "equal_branch_share",
            },
        },
        {
            "id": "strategy-hotspot-focus-crop",
            "type": "strategy",
            "params": {
                "arm_name": "site_primary_condition",
                "budget_policy": "equal_branch_share",
            },
        },
    ]
    proposal = StrategyLevelActiveLearner().propose_next(
        1,
        [parent],
        [],
        "out",
        active_skills=active_skills,
        branch_width=2,
    )
    assert len(proposal.jobs) == 2, [job.job_id for job in proposal.jobs]
    by_arm = {job.params.get("exploration_arm"): job for job in proposal.jobs}
    assert set(by_arm) == {"sampler_explore", "site_primary_condition"}, by_arm
    sampler = by_arm["sampler_explore"]
    assert sampler.params["sampler_policy"] == "explore", sampler.params
    for job in proposal.jobs:
        assert "hotspot_weight" not in job.params
        assert job.params["binder_lengths"] == [85, 90, 95], job.params
        assert "length_policy" not in job.params
        assert "hotspot_weight_policy" not in job.params
        assert "budget_policy" not in job.params
    print("OK test_strategy_skills_materialize_branch_jobs")


def test_orchestrator_filters_strategy_skills_by_evidence():
    import tempfile
    from binderloop.config import HarnessConfig, TargetSpec
    from binderloop.orchestration.orchestrator import BinderDesignOrchestrator

    cfg = HarnessConfig(target=TargetSpec(structure_path="missing.cif"))
    orch = BinderDesignOrchestrator(cfg, out_dir=Path(tempfile.mkdtemp()), max_rounds=1)
    context = {
        "active_learning_examples": {
            "current_round": {"counts": {"strict_positive": 3}, "strict_positive_examples": [{"candidate_id": "p0"}]},
            "prior_rounds": {"strict_positive_examples": []},
            "cumulative": {"consecutive_zero_current_positive_rounds": 0},
        },
        "fragment_templates": {"library": [{"template_id": "t0"}], "recommended_config": {"binder_templates": [{"id": "t0"}]}},
        "evaluation": {"tag_counts": {}, "core_metric_trends": {"delta": {"best_core_objective": 0.02}}},
    }

    disabled = orch._select_agent_skills("StrategyLevelActiveLearner", context, ["strategy"])
    assert disabled == [], disabled

    cfg.active_learning.enable_strategy_skills = True
    cfg.active_learning.enable_exploitation_arms = True
    enabled = orch._select_agent_skills("StrategyLevelActiveLearner", context, ["strategy"])
    enabled_ids = {skill["id"] for skill in enabled}
    assert "strategy-contrastive-positive-exploit" in enabled_ids, enabled_ids
    assert "strategy-template-strict-exploit" in enabled_ids, enabled_ids

    weak_context = {
        "active_learning_examples": {
            "current_round": {"counts": {"strict_positive": 0}, "strict_positive_examples": []},
            "prior_rounds": {"strict_positive_examples": [{"candidate_id": "stale"}]},
            "cumulative": {"consecutive_zero_current_positive_rounds": 2},
        },
        "fragment_templates": {"library": [{"template_id": "t0"}], "recommended_config": {"binder_templates": [{"id": "t0"}]}},
        "evaluation": {"tag_counts": {}, "core_metric_trends": {"delta": {"best_core_objective": -0.01}}},
    }
    filtered = orch._select_agent_skills("StrategyLevelActiveLearner", weak_context, ["strategy"])
    filtered_ids = {skill["id"] for skill in filtered}
    assert "strategy-contrastive-positive-exploit" not in filtered_ids, filtered_ids
    assert "strategy-template-strict-exploit" not in filtered_ids, filtered_ids

    update = orch._filter_fragment_template_update_by_evidence(
        {"binder_templates": [{"id": "t0"}], "module_guided_exploitation": True, "template_conditioned_fraction": 0.3},
        weak_context,
    )
    assert "binder_templates" not in update
    assert "module_guided_exploitation" not in update
    assert "template_conditioned_fraction" not in update

    print("OK test_orchestrator_filters_strategy_skills_by_evidence")


def test_exploitation_arms_can_be_disabled():
    parent = DesignJob(
        job_id="r0_round",
        target_structure="target.cif",
        chain_id="E",
        hotspots=["E:153"],
        binder_length=90,
        params={"hotspot_weight": 2.0, "binder_lengths": [85, 90, 95], "diffusion_batch_size": 1},
        output_dir="out/r0/round",
    )
    proposal = StrategyLevelActiveLearner().propose_next(
        1,
        [parent],
        [],
        "out",
        branch_width=1,
        enable_exploitation_arms=False,
    )
    assert len(proposal.jobs) == 1
    assert proposal.jobs[0].params["exploration_arm"] == "sampler_explore", proposal.jobs[0].params
    assert "hotspot_weight" not in proposal.jobs[0].params
    assert proposal.jobs[0].params["deprecated_strategy_audit"]["hotspot_weight"]["status"] == "deprecated_audit_only"
    print("OK test_exploitation_arms_can_be_disabled")


def test_strategy_arms_are_evidence_triggered_and_deleted_arms_stay_deleted():
    learner = StrategyLevelActiveLearner(seed=7)
    context = {
        "strict_positive_count": 0,
        "min_positives_for_exploit": 2,
        "failure_tag_counts": {"hotspot_miss": 4},
        "core_delta": -0.02,
        "plateau": False,
    }
    arms = learner.candidate_arms(
        structural_summary={"aggregate_tags": {"hotspot_not_covered": 3}},
        hypotheses=[],
        quality_analysis={"low_quality_modules": [{"module_id": "bad"}]},
        enable_exploitation_arms=True,
        selection_context=context,
    )
    names = {arm["name"] for arm in arms}
    assert "site_primary_condition" in names, names
    assert "sampler_explore" in names, names
    assert "baseline_hold" in names, names
    assert "template_exploit" not in names, names
    assert not names.intersection({"foldability_repair", "module_repair", "forced_branch_switch_explore", "interface_pose_repair"}), names

    structured = learner.candidate_arms(
        structural_summary={"aggregate_tags": {}},
        hypotheses=[{"name": "arbitrary_descriptive_title", "failure_modes": ["hotspot_miss"]}],
        quality_analysis={},
        enable_exploitation_arms=False,
        selection_context={**context, "failure_tag_counts": {}},
    )
    assert "site_primary_condition" in {arm["name"] for arm in structured}, structured

    exploit = learner.candidate_arms(
        structural_summary={"aggregate_tags": {}},
        hypotheses=[],
        quality_analysis={},
        enable_exploitation_arms=True,
        selection_context={**context, "strict_positive_count": 2, "core_delta": 0.01},
    )
    assert "template_exploit" not in {arm["name"] for arm in exploit}, exploit  # no effective template payload
    print("OK test_strategy_arms_are_evidence_triggered_and_deleted_arms_stay_deleted")


def test_hypothesis_failure_modes_are_closed_and_name_independent():
    sanitized = HypothesisAgent._sanitize_hypotheses([
        {"name": "free-form title", "failure_modes": ["hotspot_miss", "invented"], "config_parameter_changes": {}},
        {"name": "missing mode", "config_parameter_changes": {}},
    ])
    assert sanitized[0]["failure_modes"] == ["hotspot_miss"], sanitized
    assert sanitized[1]["failure_modes"] == ["no_dominant_failure"], sanitized
    augmented = HypothesisAgent._augment_failure_coverage(
        [{"name": "missed obvious failure", "failure_modes": ["no_dominant_failure"], "config_parameter_changes": {}}],
        {"evaluation": {"total_candidates": 4, "tag_counts": {"hotspot_miss": 3}}, "structural_analysis": {}},
    )
    repair = [item for item in augmented if "hotspot_miss" in item.get("failure_modes", [])]
    assert repair and repair[0]["source"] == "deterministic_coverage_repair", augmented
    fallback = HypothesisAgent._fallback({
        "evaluation": {"total_candidates": 4, "tag_counts": {"binding_pose_failure": 3}},
        "structural_analysis": {},
    })
    assert fallback[0]["failure_modes"] == ["binding_pose_failure"], fallback
    print("OK test_hypothesis_failure_modes_are_closed_and_name_independent")


def test_strategy_arm_ranking_closed_catalog_and_parent_diversity():
    class FakeLLM:
        def available(self):
            return True

        def chat_json(self, **_kwargs):
            return {
                "ranked_arms": [
                    {"arm_name": "sampler_explore", "confidence": 0.9, "evidence_ids": ["N1"], "reason": "plateau"},
                    {"arm_name": "invented_arm", "confidence": 1.0, "reason": "invalid"},
                    {"arm_name": "site_primary_condition", "confidence": 0.8, "evidence_ids": ["S1"], "reason": "miss"},
                ]
            }

    arms = [
        {"name": "site_primary_condition", "deterministic_priority": 80},
        {"name": "sampler_explore", "deterministic_priority": 50},
    ]
    ranking = StrategyArmRankingAgent(FakeLLM()).rank(round_id=2, arms=arms, context={})
    assert ranking.llm_used is True
    assert ranking.ordered_arm_names == ["sampler_explore", "site_primary_condition"], ranking

    parents = [
        DesignJob("best", "target.cif", "E", ["E:153"], 90, params={"hotspot_weight": 2.0, "binder_lengths": [90]}, output_dir="out/best"),
        DesignJob("explore", "target.cif", "E", ["E:153"], 100, params={"hotspot_weight": 1.0, "binder_lengths": [100]}, output_dir="out/explore"),
    ]
    proposal = StrategyLevelActiveLearner(seed=1).propose_next(
        2,
        parents,
        [],
        "out",
        structural_summary={"aggregate_tags": {"hotspot_not_covered": 1}},
        branch_width=2,
        enable_exploitation_arms=False,
        selection_context={
            "strict_positive_count": 0,
            "failure_tag_counts": {"hotspot_miss": 1},
            "core_delta": 0.0,
            "plateau": True,
        },
        ranked_arm_names=ranking.ordered_arm_names,
    )
    assert len(proposal.jobs) == 2
    assert [job.params["exploration_arm"] for job in proposal.jobs] == [
        "sampler_explore", "site_primary_condition"
    ]
    assert proposal.jobs[0].binder_length == 90
    assert proposal.jobs[1].binder_length == 90
    print("OK test_strategy_arm_ranking_closed_catalog_and_parent_diversity")


def test_blocked_strategy_arms_fall_back_to_hold():
    parent = DesignJob("p", "target.cif", "E", ["E:153"], 90, params={"binder_lengths": [90]}, output_dir="out/p")
    proposal = StrategyLevelActiveLearner().propose_next(
        1,
        [parent],
        [],
        "out",
        blocked_arms=["sampler_explore"],
        enable_exploitation_arms=False,
        selection_context={"strict_positive_count": 0, "failure_tag_counts": {}, "plateau": True},
    )
    assert proposal.jobs[0].params["exploration_arm"] == "baseline_hold", proposal.jobs[0].params
    print("OK test_blocked_strategy_arms_fall_back_to_hold")


def test_core_rank_boundaries_and_no_compensation():
    passing = {"id": "pass", "design_to_target_iptm": 0.50, "min_design_to_target_pae": 10.0,
               "design_ptm": 0.70, "designfolding-filter_rmsd": 2.5}
    compensated_failure = {"id": "fail", "design_to_target_iptm": 0.49, "min_design_to_target_pae": 1.0,
                           "design_ptm": 0.99, "designfolding-filter_rmsd": 0.1}
    better_worst_margin = {"id": "margin", "design_to_target_iptm": 0.54, "min_design_to_target_pae": 8.0,
                           "design_ptm": 0.76, "designfolding-filter_rmsd": 2.0}
    ranked = rank_by_core_objective([compensated_failure, passing, better_worst_margin])
    assert [row["id"] for row in ranked] == ["margin", "pass", "fail"], ranked
    assert core_rank_key(passing)[0] == 1
    assert core_rank_key(compensated_failure)[0] == 0

    high_yield = round_rank_key([passing, better_worst_margin], top_k=2)
    flashy_low_yield = round_rank_key([better_worst_margin, compensated_failure], top_k=2)
    assert high_yield > flashy_low_yield, (high_yield, flashy_low_yield)
    print("OK test_core_rank_boundaries_and_no_compensation")


def test_fragment_quality_hard_gates(tmp_path: Path):
    good = tmp_path / "good_fragment.pdb"
    good.write_text(_synthetic_complex_pdb(), encoding="utf-8")
    summary = analyze_binder_structure(good, binder_chain="X", target_chains=["Y"],
                                       binder_length=5, auto_detect_chains=False)
    fragment = summary.fragment_qualities[0]
    assert fragment.quality_rank[0] == 1, fragment
    assert fragment.gate_failures == [], fragment

    broken_lines = _synthetic_complex_pdb().splitlines()
    # Move binder residue 3 far away, creating local chain breaks.
    broken_lines[2] = _atom(3, "CA", "ALA", "X", 3, 100.0, 0.0, 0.0)
    broken = tmp_path / "broken_fragment.pdb"
    broken.write_text("\n".join(broken_lines) + "\n", encoding="utf-8")
    bad_summary = analyze_binder_structure(broken, binder_chain="X", target_chains=["Y"],
                                           binder_length=5, auto_detect_chains=False)
    bad = bad_summary.fragment_qualities[0]
    assert bad.quality_rank[0] == 0, bad
    assert "local_chain_break" in bad.gate_failures, bad.gate_failures
    assert bad.quality_label == "low"
    print("OK test_fragment_quality_hard_gates")


def main() -> None:
    import tempfile

    test_chain_detection()
    test_hotspot_number_matching()
    with tempfile.TemporaryDirectory() as d:
        test_analyze_structure_autodetect(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_boltzgen_multilength_chain_relabel_detection(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_additional_filters_define_analysis_candidates(Path(d))
    test_contrastive_active_learning_examples_and_skills()
    with tempfile.TemporaryDirectory() as d:
        test_contacts_prescreen_equivalence(Path(d))
    test_epitope_crop_focus_and_keep()
    test_template_library_merge()
    with tempfile.TemporaryDirectory() as d:
        test_fragment_template_topk_and_strategy_allocation(Path(d))
    test_template_foldability_guard_lowers_fraction()
    with tempfile.TemporaryDirectory() as d:
        test_orchestrator_rollback_seed(Path(d))
    test_rollback_controller_trajectory()
    with tempfile.TemporaryDirectory() as tmp:
        test_zero_pass_quality_signal_and_exploration_governance(Path(tmp))
    test_rollback_controller_execution_failure_no_branch_switch()
    test_round_rank_rollback_respects_noise_tolerance()
    test_skill_registry_activation()
    test_strategy_skills_materialize_branch_jobs()
    test_orchestrator_filters_strategy_skills_by_evidence()
    test_exploitation_arms_can_be_disabled()
    test_strategy_arms_are_evidence_triggered_and_deleted_arms_stay_deleted()
    test_hypothesis_failure_modes_are_closed_and_name_independent()
    test_strategy_arm_ranking_closed_catalog_and_parent_diversity()
    test_blocked_strategy_arms_fall_back_to_hold()
    test_core_rank_boundaries_and_no_compensation()
    with tempfile.TemporaryDirectory() as d:
        test_fragment_quality_hard_gates(Path(d))
    print("\nALL STRATEGY IMPROVEMENT TESTS PASSED")



def test_partial_execution_remains_quality_evaluable(tmp_path: Path):
    from binderloop.config import load_config
    from binderloop.orchestration.orchestrator import BinderDesignOrchestrator

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "sc2rbd_structured_task.yaml")
    orch = BinderDesignOrchestrator(cfg, out_dir=str(tmp_path / "partial"), max_rounds=1)
    jobs = [
        DesignJob(job_id="a", target_structure="target.cif", chain_id="A", binder_length=80, hotspots=[], params={"branch_id": "branch-a"}, output_dir=str(tmp_path / "a")),
        DesignJob(job_id="b", target_structure="target.cif", chain_id="A", binder_length=80, hotspots=[], params={"branch_id": "branch-b"}, output_dir=str(tmp_path / "b")),
    ]
    records = [
        {"job_id": "a", "status": "completed", "output_dir": str(tmp_path / "a")},
        {"job_id": "b", "status": "failed", "backend": "taiji", "error": "task failed"},
    ]
    state = orch._classify_execution_state(jobs, records)
    assert state["state"] == "partial", state
    assert state["realized_fraction"] == 0.5, state
    assert state["successful_job_ids"] == ["a"], state
    assert state["failed_job_ids"] == ["b"], state
    failed, reason = orch._detect_round_execution_failure(total_candidates=1, execution_records=records)
    assert failed is False and reason == "", (failed, reason)

if __name__ == "__main__":
    main()
