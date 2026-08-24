#!/usr/bin/env python3
import unittest
from binderloop.agents.binder_quality_collaboration_agent import QualityCollaborationController
from binderloop.config import QualityCollaborationSpec
from binderloop.memory import ExperimentMemory

class AdaptiveQualityStateMachineTest(unittest.TestCase):
    def memory(self):
        return ExperimentMemory("x", {}, round_metrics=[{"round_id":0,"reward":1.0,"execution_failed":False}])

    def test_default_single_and_noise_band(self):
        m=self.memory(); s=QualityCollaborationSpec(enabled=True)
        d=QualityCollaborationController.decide(m,{"round_id":1,"reward":.99},s)
        self.assertEqual(d.mode,"single")

    def test_trigger_recovery_and_two_round_cap(self):
        m=self.memory(); s=QualityCollaborationSpec(enabled=True)
        d=QualityCollaborationController.decide(m,{"round_id":1,"reward":.90},s)
        self.assertEqual(d.mode,"multi"); self.assertAlmostEqual(d.recovery_target_reward,.97)
        m.round_metrics.append({"round_id":1,"reward":.90,"execution_failed":False})
        d=QualityCollaborationController.decide(m,{"round_id":2,"reward":.91},s)
        self.assertEqual(d.mode,"multi")
        m.round_metrics.append({"round_id":2,"reward":.91,"execution_failed":False})
        d=QualityCollaborationController.decide(m,{"round_id":3,"reward":.92},s)
        self.assertEqual(d.exit_reason,"maximum_consecutive_multi_rounds")

    def test_retrigger_and_execution_failure_does_not_pollute(self):
        m=self.memory(); s=QualityCollaborationSpec(enabled=True)
        QualityCollaborationController.decide(m,{"round_id":1,"reward":.8},s,signals={"failure_tags":["a"]})
        before=dict(m.quality_collaboration_state)
        d=QualityCollaborationController.decide(m,{"round_id":2,"reward":0,"execution_failed":True},s)
        self.assertEqual(d.mode,"multi"); self.assertEqual(m.quality_collaboration_state["consecutive_multi_rounds"],before["consecutive_multi_rounds"])
        d=QualityCollaborationController.decide(m,{"round_id":3,"reward":.81},s,signals={"failure_tags":["new"]})
        self.assertEqual(d.mode,"multi"); self.assertEqual(d.failure_signature, QualityCollaborationController._signature({}, {"failure_tags":["new"]}))

    def test_signal_triggers_and_audits(self):
        m=self.memory(); s=QualityCollaborationSpec(enabled=True)
        d=QualityCollaborationController.decide(m,{"round_id":1,"reward":1.0},s,signals={"compute_gate_yield":.4,"previous_compute_gate_yield":.8,"metric_conflict":"x","high_value_events":["rollback"]})
        self.assertEqual(d.mode,"multi")
        self.assertEqual({x["code"] for x in d.trigger_reasons},{"compute_gate_degradation","metric_conflict","high_value_decision"})

if __name__ == "__main__": unittest.main()
