import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.active_learning.rollback import RollbackController, RoundOutcome
from binderloop.memory import ArmOutcome, ExperimentMemoryStore


class RollbackPolicyTest(unittest.TestCase):
    def test_significant_drop_is_immediate_and_patience_counts_non_best(self):
        ctl = RollbackController(regression_tolerance=0.2, patience=2)
        self.assertEqual(ctl.observe(RoundOutcome(0, 1.0)).action, "advance")
        first = ctl.observe(RoundOutcome(1, 0.95))
        self.assertEqual(first.action, "advance")
        second = ctl.observe(RoundOutcome(2, 0.94))
        self.assertEqual(second.action, "advance")
        self.assertEqual(second.consecutive_regressions, 0)
        ctl.observe(RoundOutcome(3, 1.1))
        self.assertEqual(ctl.observe(RoundOutcome(4, 0.7)).action, "retest_best_config")
        failed = ctl.observe(RoundOutcome(5, 0.0, execution_failed=True))
        self.assertEqual(failed.consecutive_regressions, 1)

    def test_new_records_choose_round_rank_not_reward(self):
        ctl = RollbackController(regression_tolerance=1.0, patience=1)
        first = RoundOutcome(0, reward=99.0, round_rank_key=[0.0, 0.9, 0.9, -1.0, -0.1])
        second = RoundOutcome(1, reward=0.01, round_rank_key=[0.5, 0.0, 0.5, -10.0, -2.5])
        self.assertEqual(ctl.observe(first).best_round, 0)
        decision = ctl.observe(second)
        self.assertEqual(decision.best_round, 1)
        self.assertEqual(decision.action, "advance")

    def test_retest_cap_then_branch_from_best_and_true_trailing_streak(self):
        ctl = RollbackController(regression_tolerance=0.2, patience=2, max_best_config_retests=1)
        ctl.observe(RoundOutcome(0, 1.0, config_digest="best"))
        self.assertEqual(ctl.observe(RoundOutcome(1, 0.5, intervention_digest="bad")).action, "retest_best_config")
        branched = ctl.observe(RoundOutcome(2, 0.5, intervention_digest="bad"))
        self.assertEqual(branched.action, "branch_from_best")
        self.assertEqual(branched.blocked_intervention_digest, "bad")
        recovered = ctl.observe(RoundOutcome(3, 0.85))
        self.assertEqual(recovered.consecutive_regressions, 0)

    def test_absolute_strict_count_selects_r3_style_round(self):
        ctl = RollbackController(regression_tolerance=0.25, patience=2)
        r2 = RoundOutcome(2, reward=10.0, round_rank_key=[10, 0.31, 0.57, -3.58, -0.94], strict_successes=10, strict_trials=64)
        r3 = RoundOutcome(3, reward=18.0, round_rank_key=[18, 0.42, 0.59, -3.25, -0.54], strict_successes=18, strict_trials=64)
        ctl.observe(r2)
        decision = ctl.observe(r3)
        self.assertEqual(decision.best_round, 3)
        self.assertEqual(decision.action, "advance")


class ExperimentLedgerTest(unittest.TestCase):
    def test_migration_idempotent_upsert_and_bounded_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            (root / "experiment_memory.json").write_text(json.dumps({
                "experiment_id": "old", "target": {}, "unknown_future_field": True,
                "rounds": [{"round_id": 0, "unknown": 1}],
            }))
            store = ExperimentMemoryStore(root)
            memory = store.load()
            for rid, reward in [(0, .5), (1, .4), (1, .45)]:
                store.upsert_ledger_round(memory, round_id=rid, outcome={"reward": reward},
                    policy_snapshot={"alpha": rid}, candidate_denominators={"generated": 10},
                    next_hypotheses=[{"id": rid}])
            self.assertEqual(len(memory.experiment_ledger.rounds), 2)
            self.assertEqual(memory.experiment_ledger.best_round_id, 0)
            self.assertEqual(store.ledger_prompt_view(memory, max_rounds=1)["recent_rounds"][0]["round_id"], 1)
            store.save(memory)
            self.assertEqual(store.load().experiment_ledger.schema_version, "2.0")

    def test_uncertainty_branch_cooldown_and_resume_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentMemoryStore(tmp)
            memory = store.load()
            for rid in (0, 1):
                store.upsert_ledger_round(
                    memory, round_id=rid, outcome={"reward": 0.5}, policy_snapshot={},
                    candidate_denominators={"generated": 10},
                )
                store.record_governance_outcome(
                    memory, round_id=rid, branch_id="bad", arm_id="arm", successes=1, trials=10,
                    intervention_digest="digest", regressed=True,
                )
            arm = memory.experiment_ledger.arm_outcomes["arm"]
            self.assertEqual((arm.successes, arm.trials, arm.uses), (2, 20, 2))
            # Re-observing the same round rebuilds rather than double-counting.
            store.record_governance_outcome(
                memory, round_id=1, branch_id="bad", arm_id="arm", successes=1, trials=10,
                intervention_digest="digest", regressed=True,
            )
            self.assertEqual(memory.experiment_ledger.arm_outcomes["arm"].trials, 20)
            self.assertIn("digest", store.blocked_interventions(memory, 1))
            self.assertTrue(store.uncertainty_overlaps(arm, ArmOutcome("new")))
            store.save(memory)
            loaded = store.load()
            self.assertEqual(loaded.experiment_ledger.branches["bad"].status, "cooldown")
            self.assertEqual(loaded.experiment_ledger.arm_outcomes["arm"].posterior_alpha, 3.0)

    def test_legacy_artifact_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiment_memory.json").write_text(json.dumps({
                "experiment_id": "legacy", "target": {}, "ledger": {
                    "schema_version": "1.0", "rounds": [{"round_id": 0, "outcome": {"reward": 0.1}}]
                }
            }))
            memory = ExperimentMemoryStore(root).load()
            self.assertEqual(memory.experiment_ledger.schema_version, "2.0")
            self.assertEqual(memory.experiment_ledger.rounds[0].round_id, 0)

    def test_ledger_best_uses_round_rank_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperimentMemoryStore(Path(tmp))
            memory = store.load()
            store.upsert_ledger_round(
                memory, round_id=0,
                outcome={"reward": 10.0, "round_rank_key": [0.0, 1.0, 1.0, -1.0, -1.0]},
                policy_snapshot={"alpha": 0},
            )
            store.upsert_ledger_round(
                memory, round_id=1,
                outcome={"reward": 0.1, "round_rank_key": [0.5, 0.0, 0.5, -10.0, -2.5]},
                policy_snapshot={"alpha": 1},
            )
            self.assertEqual(memory.experiment_ledger.best_round_id, 1)

    def test_corrected_best_round_recomputes_ledger_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ExperimentMemoryStore(tmp); memory=store.load()
            store.upsert_ledger_round(memory,round_id=0,outcome={"reward":1,"round_rank_key":[5,0,0,0,0]},policy_snapshot={"x":0})
            store.upsert_ledger_round(memory,round_id=1,outcome={"reward":2,"round_rank_key":[6,0,0,0,0]},policy_snapshot={"x":1})
            self.assertEqual(memory.experiment_ledger.best_round_id,1)
            store.upsert_ledger_round(memory,round_id=1,outcome={"reward":0,"round_rank_key":[4,0,0,0,0]},policy_snapshot={"x":2})
            self.assertEqual(memory.experiment_ledger.best_round_id,0)
            self.assertEqual(memory.experiment_ledger.best_policy_snapshot,{"x":0})


if __name__ == "__main__":
    unittest.main()
