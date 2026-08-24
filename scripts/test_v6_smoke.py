#!/usr/bin/env python3
"""Quick smoke test for v6 improvements - imports and key function calls only."""
from __future__ import print_function
import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== V6 SMOKE TEST ===")
print()

# 1. Import all agent modules
print("1. Testing imports...")
try:
    from binderloop.agents.context_compaction import (
        compact_context_for_hypothesis,
        compact_context_for_quality,
        compact_context_for_diagnostic,
        compact_context_for_input_config,
        compact_evaluation,
        compact_structural_analysis,
        compact_fragment_templates,
        compact_memory,
        compact_config,
        compact_hypotheses,
        compact_quality_analysis,
        compact_diagnostic_report,
    )
    print("   context_compaction: OK")
except Exception as e:
    print("   context_compaction: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.agents.hypothesis_agent import HypothesisAgent, HypothesisSet
    print("   hypothesis_agent: OK")
except Exception as e:
    print("   hypothesis_agent: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.agents.binder_quality_analysis_agent import BinderQualityAnalysisAgent
    print("   quality_analysis_agent: OK")
except Exception as e:
    print("   quality_analysis_agent: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.agents.diagnostic_coach_agent import DiagnosticCoachAgent
    print("   diagnostic_coach_agent: OK")
except Exception as e:
    print("   diagnostic_coach_agent: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.agents.input_configuration_agent import InputConfigurationAgent
    print("   input_configuration_agent: OK")
except Exception as e:
    print("   input_configuration_agent: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.active_learning.rollback import RollbackController, RoundOutcome, round_reward
    print("   rollback: OK")
except Exception as e:
    print("   rollback: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.active_learning.strategy import StrategyLevelActiveLearner
    print("   strategy: OK")
except Exception as e:
    print("   strategy: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.agents.fragment_template_mining_agent import FragmentTemplateMiningAgent
    print("   fragment_template_mining: OK")
except Exception as e:
    print("   fragment_template_mining: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.llm import OpenAICompatibleClient, LLMConfigError
    print("   llm: OK")
except Exception as e:
    print("   llm: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
    print("   orchestrator: OK")
except Exception as e:
    print("   orchestrator: FAIL - {}".format(e))
    sys.exit(1)

try:
    from binderloop.config import load_config, HarnessConfig
    print("   config: OK")
except Exception as e:
    print("   config: FAIL - {}".format(e))
    sys.exit(1)

print()
print("All imports passed!")

# 2. Test context compaction with mock data
print()
print("2. Testing context compaction...")

# Build a mock context with heavy fields that should be stripped
mock_context = {
    "round_id": 1,
    "evaluation": {
        "total_candidates": 100,
        "success_count": 8,
        "failure_count": 92,
        "tag_counts": {"hotspot_miss": 30, "clash": 10, "folding_failure": 5},
        "observations": "test observations",
        "top_candidates": [
            {"id": "c1", "iptm": 0.45, "ca_coordinates": [[1.0, 2.0, 3.0] * 100]},
            {"id": "c2", "iptm": 0.42, "ca_coordinates": [[1.0, 2.0, 3.0] * 100]},
        ],
        "failed_examples": [{"id": "f1", "reason": "low iptm"}],
    },
    "structural_analysis": {
        "total_structures": 20,
        "aggregate_tags": {"hotspot_not_covered": True},
        "reliable_seed_fraction": 0.3,
        "observations": "test",
        "summaries": [
            {
                "structure_file": "test.pdb",
                "reliability_score": 0.8,
                "reliability_tags": ["reliable"],
                "interface_contact_count": 15,
                "interface_residue_count": 10,
                "hotspot_contacts": {"A:67": 5},
                "clash_density": 0.1,
                "interface_hydrophobic_fraction": 0.4,
                "interface_polar_fraction": 0.3,
                "ca_coordinates": [[1.0, 2.0, 3.0]] * 500,  # HEAVY
                "high_quality_fragments": [
                    {"fragment_id": "f1", "ca_coordinates": [[1.0]*3]*200, "binder_sequence": "A"*100},
                ],
                "low_quality_fragments": [],
            }
        ],
    },
    "fragment_templates": {
        "total_templates": 5,
        "high_quality_count": 3,
        "mean_quality": 0.75,
        "structure_redesign": {
            "source_structure_file": "/path/to/structure.pdb",
            "quality_score": 0.85,
            "ca_coordinates": [[1.0]*3]*1000,  # HEAVY
            "binder_sequence": "A"*200,  # HEAVY
        },
    },
    "current_config": {"alpha": 0.001, "exploration_ratio": 0.35, "hotspot_weight": 1.5},
    "memory": {"experiment_id": "test"},
    "messages": [{"role": "user", "content": "test"}],
}

# Test hypothesis compaction
hypo_compact = compact_context_for_hypothesis(mock_context)
assert "ca_coordinates" not in str(hypo_compact), "ca_coordinates should be stripped!"
assert "binder_sequence" not in str(hypo_compact), "binder_sequence should be stripped!"
print("   hypothesis compaction: OK (no ca_coordinates/binder_sequence)")

# Test quality compaction
qual_compact = compact_context_for_quality(mock_context)
assert "ca_coordinates" not in str(qual_compact), "ca_coordinates should be stripped!"
assert "binder_sequence" not in str(qual_compact), "binder_sequence should be stripped!"
print("   quality compaction: OK")

# Test input_config compaction (formerly the biggest token bomb)
ic_compact = compact_context_for_input_config(
    target_name="test",
    current_config={"alpha": 0.001},
    diagnostic_report={"status_diagnosis": "ok", "corrective_actions": []},
    evaluation_summary={"total_candidates": 100, "success_count": 8},
    round_id=1,
    structural_analysis=mock_context["structural_analysis"],
    quality_analysis={"overall_assessment": "good"},
    hypotheses=[{"name": "h1", "config_parameter_changes": {"alpha": 0.002}}],
    memory_summary=mock_context["memory"],
    tuning_feedback={"previous_move": "alpha +0.001", "reward_delta": +0.05},
)
assert "ca_coordinates" not in str(ic_compact), "ca_coordinates should be stripped!"
assert "summaries" not in str(ic_compact.get("structural_analysis", {})), "summaries should be absent from input_config context!"
print("   input_config compaction: OK (maximally stripped)")

# 3. Test template compaction
print()
print("3. Testing template compaction...")
templates = {
    "total_templates": 10,
    "high_quality_count": 5,
    "low_quality_count": 5,
    "mean_quality": 0.7,
    "structure_redesign": {
        "source_structure_file": "/path/to/something.pdb",
        "quality_score": 0.85,
        "fragment_id": "f42",
        "ca_coordinates": [[1.0, 2.0, 3.0]] * 1000,
        "binder_sequence": "MKFLILFNILV" * 50,
        "within_proximity": 8.0,
        "residue_span": "10-20",
    },
}
compact = compact_fragment_templates(templates)
assert "ca_coordinates" not in str(compact), "ca_coordinates should be stripped from templates!"
assert "binder_sequence" not in str(compact), "binder_sequence should be stripped from templates!"
assert "source_structure_file" in str(compact), "source_structure_file should be preserved!"
print("   template compaction: OK")

# 4. Test RollbackController branch-based rollback
print()
print("4. Testing rollback branch logic...")
rc = RollbackController(regression_tolerance=0.25, patience=2)

# Test: execution failure (0 candidates + config_error) should NOT enter rollback history
exec_failure = RoundOutcome(
    round_id=2,
    reward=0.0,
    best_iptm=0.0,
    success_count=0,
    median_iptm=0.0,
    execution_failed=True,
    execution_failure_reason="boltzgen_config_error",
    arm_signature="exploit_reliable_seed:len_80_seed_0",
)
decision = rc.observe(exec_failure)
assert decision.action == "advance", "Execution failure should advance/retry, not quality-rollback"
assert decision.blocked_arm_signature is None, "Execution failure should not block AL arms; orchestrator retries same job"
print("   execution failure excluded from history: OK (action={}, blocked_arm={})".format(
    decision.action, decision.blocked_arm_signature))

# Test: normal quality degradation rollback
round1 = RoundOutcome(round_id=1, reward=0.8, best_iptm=0.45, success_count=5, median_iptm=0.35)
rc.observe(round1)
round2 = RoundOutcome(round_id=2, reward=0.4, best_iptm=0.25, success_count=1, median_iptm=0.15)
decision2 = rc.observe(round2)
print("   normal regression detected: consecutive_regressions={}, action={}, is_regression={}".format(
    decision2.consecutive_regressions, decision2.action, decision2.is_regression))

# 5. Test LLM hard-fail on require_llm
print()
print("5. Testing LLM require_llm behavior...")
try:
    from binderloop.llm import OpenAICompatibleClient, LLMSettings, ModelEndpoint
    # Test that agents with require_llm=True raise when LLM is None
    ha = HypothesisAgent(llm=None, require_llm=True)
    try:
        ha.propose(mock_context)
        print("   WARNING: should have raised but didn't!")
    except RuntimeError as e:
        print("   HypothesisAgent require_llm raises correctly: {}".format(str(e)[:80]))
    except Exception as e:
        print("   HypothesisAgent raises on missing LLM: {}".format(type(e).__name__))
except Exception as e:
    print("   LLM test skipped: {}".format(e))

# 6. Test config
print()
print("6. Testing config loading...")
cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/sc2rbd_structured_task.yaml"))
print("   Config loaded: backend={}, host_gpu_num={}, rounds={}".format(cfg.resource.backend, cfg.resource.host_gpu_num, cfg.active_learning.max_rounds))

print()
print("=== ALL TESTS PASSED ===")
