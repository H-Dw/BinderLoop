#!/usr/bin/env python3
"""Lightweight integration check for the P0/P1 guardrail + feedback logic.

Run with: /data/miniconda3/envs/bg/bin/python scripts/check_guardrail.py
"""
from types import SimpleNamespace
from binderloop.agents.config_parameter_contract import (
    clamp_config_with_inertia, PARAM_BOUNDS,
)
from binderloop.active_learning.rollback import (
    RollbackController, RoundOutcome, round_reward, median_of,
)
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator


def test_clamp_blocks_alpha_explosion():
    # The exact v4 failure: LLM proposes alpha=0.7 from baseline 0.001.
    merged = {"alpha": 0.7, "exploration_ratio": 0.6, "noise_scale": 0.7}
    current = {"alpha": 0.001, "exploration_ratio": 0.35, "noise_scale": 0.7}
    clamped, notes = clamp_config_with_inertia(merged, current_config=current)
    assert clamped["alpha"] <= 0.05, clamped
    assert clamped["alpha"] <= 0.001 * 3 + 1e-9, clamped  # change-rate cap
    assert clamped["exploration_ratio"] <= 0.5 + 1e-9, clamped
    print("[OK] clamp blocks alpha explosion:", clamped, "(%d notes)" % len(notes))


def test_reward_robust_to_single_outlier():
    # Round A: one lucky 0.62 but the rest low. Round B: a solid distribution.
    a_best, a_med = 0.62, median_of([0.62, 0.20, 0.18, 0.17, 0.15])
    b_best, b_med = 0.50, median_of([0.50, 0.48, 0.47, 0.46, 0.45])
    legacy_a, legacy_b = round_reward(a_best, 0), round_reward(b_best, 0)
    new_a = round_reward(a_best, 0, median_iptm=a_med)
    new_b = round_reward(b_best, 0, median_iptm=b_med)
    print("  legacy: A=%.3f B=%.3f (single-best driven)" % (legacy_a, legacy_b))
    print("  new   : A=%.3f B=%.3f (median driven)" % (new_a, new_b))
    assert legacy_a > legacy_b  # old reward prefers the lucky outlier round
    assert new_b > new_a        # new reward prefers the robust distribution
    print("[OK] median reward prefers robust round B over lucky-outlier round A")


def test_rollback_patience_tolerance_defaults():
    rc = RollbackController()
    assert rc.patience == 2 and rc.regression_tolerance == 0.25, (rc.patience, rc.regression_tolerance)
    print("[OK] rollback defaults patience=%d tolerance=%.2f" % (rc.patience, rc.regression_tolerance))


def test_rollback_less_trigger_happy():
    # Best at round0 (reward 0.6), then small dips that USED to trigger rollback at patience=1.
    rc = RollbackController(regression_tolerance=0.25, patience=2)
    rc.observe(RoundOutcome(0, reward=0.60, best_iptm=0.6))
    d1 = rc.observe(RoundOutcome(1, reward=0.50, best_iptm=0.5))  # 17% drop, within 25% tol -> advance
    d2 = rc.observe(RoundOutcome(2, reward=0.40, best_iptm=0.4))  # 33% drop but patience=2 buffer
    print("  r1 action=%s (drop=%.0f%%)  r2 action=%s (consec=%d)" % (d1.action, d1.relative_drop*100, d2.action, d2.consecutive_regressions))
    assert d1.action == "advance"  # was rollback under old 0.15 tol
    print("[OK] mild dips no longer trigger immediate rollback")


def test_build_tuning_feedback():
    # Mock a memory with 2 rounds: round0 alpha=0.001 reward 0.6, round1 alpha raised, reward dropped.
    memory = SimpleNamespace(
        rounds=[
            SimpleNamespace(round_id=0, config_snapshot={"alpha": 0.001, "exploration_ratio": 0.35}),
            SimpleNamespace(round_id=1, config_snapshot={"alpha": 0.003, "exploration_ratio": 0.50}),
        ],
        round_metrics=[
            {"round_id": 0, "reward": 0.60},
            {"round_id": 1, "reward": 0.40},
        ],
    )
    fb = BinderDesignOrchestrator._build_tuning_feedback.__wrapped__ if hasattr(BinderDesignOrchestrator._build_tuning_feedback, "__wrapped__") else BinderDesignOrchestrator._build_tuning_feedback
    # _build_tuning_feedback is an instance method; call unbound with a stub self.
    stub_self = SimpleNamespace()
    out = BinderDesignOrchestrator._build_tuning_feedback(stub_self, memory, {})
    print("  feedback:", out)
    assert out["best_round_id"] == 0
    assert out["reward_trend"] == "down"
    assert any(p["parameter"] in ("alpha", "exploration_ratio") for p in out["penalized_moves"]), out
    print("[OK] tuning feedback penalizes harmful moves and points to best round 0")


if __name__ == "__main__":
    test_clamp_blocks_alpha_explosion()
    test_reward_robust_to_single_outlier()
    test_rollback_patience_tolerance_defaults()
    test_rollback_less_trigger_happy()
    test_build_tuning_feedback()
    print("\nALL GUARDRAIL CHECKS PASSED")
