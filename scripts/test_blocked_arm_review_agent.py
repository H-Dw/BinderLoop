import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binderloop.agents.blocked_arm_review_agent import BlockedArmReviewAgent
from binderloop.memory import ExperimentLedger, ExperimentMemoryStore, LedgerRound
from binderloop.llm import LLMDefinitiveError
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator

class FakeLLM:
    def __init__(self, result): self.result=result; self.calls=[]
    def available(self): return True
    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices":[{"message":{"content":json.dumps(self.result)},"finish_reason":"stop"}],"usage":{"completion_tokens":10}}

def test_valid_unfreeze_review():
    result={"reviews":[{"arm_id":"sampler_explore","recommendation":"eligible_for_unfreeze","accepted_evidence_ids":["E1"],"counterevidence_ids":[],"risk_codes":[],"reason":"new complete outcome"}]}
    decision=BlockedArmReviewAgent(FakeLLM(result)).review(round_id=3,blocked_arms=[{"arm_id":"sampler_explore"}],evidence=[{"evidence_id":"E1","arm_id":"sampler_explore","status":"closed","requested_budget":1,"completed_budget":1,"trials":1}],context={})
    assert decision.llm_used and decision.reviews[0]["recommendation"]=="eligible_for_unfreeze"

def test_invalid_ids_fail_closed():
    result={"reviews":[{"arm_id":"invented","recommendation":"eligible_for_unfreeze","accepted_evidence_ids":["BAD"],"counterevidence_ids":[],"risk_codes":[],"reason":"x"}]}
    decision=BlockedArmReviewAgent(FakeLLM(result)).review(round_id=3,blocked_arms=[{"arm_id":"sampler_explore"}],evidence=[{"evidence_id":"E1","arm_id":"sampler_explore","status":"closed","requested_budget":1,"completed_budget":1,"trials":1}],context={})
    assert not decision.llm_used and decision.reviews[0]["recommendation"]=="insufficient_evidence"

def test_unavailable_keeps_blocked():
    decision=BlockedArmReviewAgent(None).review(round_id=3,blocked_arms=[{"arm_id":"sampler_explore"}],evidence=[],context={})
    assert not decision.llm_used

def test_huge_payload_is_compacted_and_ids_retained():
    result={"reviews":[{"arm_id":"sampler_explore","recommendation":"keep_blocked","accepted_evidence_ids":["E1"],"counterevidence_ids":[],"risk_codes":[],"reason":"bounded"}]}
    llm=FakeLLM(result)
    decision=BlockedArmReviewAgent(llm).review(
        round_id=3, blocked_arms=[{"arm_id":"sampler_explore","reason":"x","junk":"Z"*2_000_000}],
        evidence=[{"evidence_id":"E1","arm_id":"sampler_explore","status":"closed","requested_budget":1,"completed_budget":1,"trials":1,"successes":0,"ca_coordinates":[[1,2,3]]*10000}],
        context={"selection_context":{"score":1,"huge":"Y"*2_000_000},"structural_summary":{"summaries":[{"ca_coordinates":[[1,2,3]]*10000}]}})
    payload=json.loads(llm.calls[0]["messages"][1]["content"])
    assert decision.llm_used and payload["blocked_arms"][0]["arm_id"]=="sampler_explore"
    assert payload["evidence"][0]["evidence_id"]=="E1"
    assert "junk" not in payload["blocked_arms"][0] and "ca_coordinates" not in json.dumps(payload)

def test_context_limit_required_fails_closed():
    class ContextLimit(FakeLLM):
        def create_chat_completion(self, **kwargs):
            self.calls.append(kwargs); raise LLMDefinitiveError(400,"context window",failure_class="context_limit")
    decision=BlockedArmReviewAgent(ContextLimit({}),require_llm=True).review(round_id=3,blocked_arms=[{"arm_id":"sampler_explore"}],evidence=[{"evidence_id":"E1","arm_id":"sampler_explore","status":"closed","requested_budget":1,"completed_budget":1,"trials":1}],context={})
    assert not decision.llm_used and decision.raw["fallback_reason"]=="context_limit"

def test_auth_and_quota_propagate():
    for failure in ("authentication","quota_exhausted"):
        class Definitive(FakeLLM):
            def create_chat_completion(self, **kwargs): raise LLMDefinitiveError(403,"denied",failure_class=failure)
        try:
            BlockedArmReviewAgent(Definitive({}),require_llm=True).review(round_id=3,blocked_arms=[{"arm_id":"sampler_explore"}],evidence=[{"evidence_id":"E1","arm_id":"sampler_explore","status":"closed","requested_budget":1,"completed_budget":1,"trials":1}],context={})
        except LLMDefinitiveError as exc:
            assert exc.failure_class==failure
        else:
            raise AssertionError("definitive auth/quota error did not propagate")

def test_orchestrator_skips_llm_without_exact_complete_evidence():
    class ReviewSpy:
        def __init__(self): self.calls=0
        def review(self, **kwargs): self.calls+=1; raise AssertionError("LLM review must be skipped")
    huge = "POLICY_SHOULD_NEVER_BE_COPIED_" + ("X" * 2_000_000)
    ledger = ExperimentLedger(
        arm_blocks={"sampler_explore":{"arm_id":"sampler_explore","status":"soft_blocked"}},
        rounds=[LedgerRound(
            round_id=2, policy_snapshot={"huge": huge},
            outcome={"irrelevant_evaluation":{"huge":huge}, "reward":0.1},
            per_arm_outcomes=[
                {"arm_id":"sampler_explore","evidence_id":"OLD","status":"closed","trials":1},
                {"arm_id":"other","evidence_id":"DROP","unknown":huge},
            ],
        )],
    )
    class Memory: experiment_ledger=ledger
    class Store:
        @staticmethod
        def ledger_prompt_view(*args, **kwargs): raise AssertionError("generic ledger_prompt_view must not be called")
    with tempfile.TemporaryDirectory() as tmp:
        orch=BinderDesignOrchestrator.__new__(BinderDesignOrchestrator)
        orch.blocked_arm_review_agent=ReviewSpy(); orch.memory_store=Store()
        orch._write_json=lambda path,payload: Path(path).write_text(json.dumps(payload)) or Path(path)
        remaining=orch._review_and_unfreeze_arms(
            memory=Memory(),round_dir=Path(tmp),next_round_id=3,blocked_arms={"sampler_explore"},
            arm_evidence_cards={"arms":[{"arm_id":"sampler_explore_fallback_01","status":"closed","requested_budget":1,"completed_budget":1,"trials":1}]},
            selection_context={},hypotheses=[],structural_summary={},quality_analysis={})
        assert remaining=={"sampler_explore"} and orch.blocked_arm_review_agent.calls==0
        saved=json.loads((Path(tmp)/"blocked_arm_review.json").read_text())
        assert saved["reviews"][0]["recommendation"]=="insufficient_evidence"

def test_orchestrator_passes_only_lightweight_blocked_ledger():
    class ReviewSpy:
        def __init__(self): self.context=None
        def review(self, **kwargs):
            self.context=kwargs["context"]
            return type("Decision", (), {"reviews":[], "to_dict":lambda self:{"round_id":3,"reviews":[],"llm_used":False,"raw":{}}})()
    huge = "POLICY_SHOULD_NEVER_REACH_REVIEW_" + ("X" * 2_000_000)
    ledger = ExperimentLedger(
        schema_version="2.0", best_round_id=1, best_reward=0.5,
        arm_blocks={"sampler_explore":{"arm_id":"sampler_explore","status":"soft_blocked"}},
        rounds=[LedgerRound(
            round_id=2, policy_snapshot={"huge":huge}, current_vs_best_diff={"huge":huge},
            outcome={
                "irrelevant_evaluation":{"huge":huge},
                "arm_comparison":{"winner_arm_id":"sampler_explore","verbose":huge},
                "rollback":{"is_regression":False}, "reward":0.2,
            },
            per_arm_outcomes=[
                {"arm_id":"sampler_explore","evidence_id":"OLD","status":"closed","trials":2,"unknown":huge},
                {"arm_id":"other","evidence_id":"DROP","unknown":huge},
            ],
        )],
    )
    class Memory: experiment_ledger=ledger
    class Store:
        @staticmethod
        def ledger_prompt_view(*args, **kwargs): raise AssertionError("generic ledger_prompt_view must not be called")
        @staticmethod
        def apply_arm_unfreeze(*args, **kwargs): raise AssertionError("unexpected unfreeze")
    with tempfile.TemporaryDirectory() as tmp:
        orch=BinderDesignOrchestrator.__new__(BinderDesignOrchestrator)
        spy=ReviewSpy(); orch.blocked_arm_review_agent=spy; orch.memory_store=Store()
        orch._write_json=lambda path,payload: Path(path).write_text(json.dumps(payload)) or Path(path)
        orch._review_and_unfreeze_arms(
            memory=Memory(),round_dir=Path(tmp),next_round_id=3,blocked_arms={"sampler_explore"},
            arm_evidence_cards={"arms":[{"evidence_id":"E1","arm_id":"sampler_explore","status":"closed","requested_budget":1,"completed_budget":1,"trials":1}]},
            selection_context={},hypotheses=[],structural_summary={},quality_analysis={})
        rendered=json.dumps(spy.context)
        assert "OLD" in rendered and "DROP" not in rendered
        assert "policy_snapshot" not in rendered and "irrelevant_evaluation" not in rendered
        assert "POLICY_SHOULD_NEVER_REACH_REVIEW" not in rendered
        assert len(rendered) < 10_000

def test_agent_skips_llm_without_complete_evidence():
    llm=FakeLLM({"reviews":[]})
    decision=BlockedArmReviewAgent(llm,require_llm=True).review(
        round_id=3, blocked_arms=[{"arm_id":"A"}],
        evidence=[{"evidence_id":"EA","arm_id":"A","status":"incomplete","requested_budget":2,"completed_budget":1,"trials":1}], context={})
    assert not decision.llm_used and not llm.calls
    assert decision.reviews[0]["arm_id"]=="A" and decision.reviews[0]["recommendation"]=="insufficient_evidence"

def test_agent_partitions_partial_complete_evidence():
    result={"reviews":[{"arm_id":"A","recommendation":"keep_blocked","accepted_evidence_ids":["EA"],"counterevidence_ids":[],"risk_codes":[],"reason":"reviewed"}]}
    llm=FakeLLM(result)
    decision=BlockedArmReviewAgent(llm).review(round_id=3,blocked_arms=[{"arm_id":"A"},{"arm_id":"B"}],evidence=[
        {"evidence_id":"EA","arm_id":"A","status":"closed","requested_budget":1,"completed_budget":1,"trials":1},
        {"evidence_id":"EB","arm_id":"B","status":"incomplete","requested_budget":2,"completed_budget":1,"trials":1},
    ],context={})
    payload=json.loads(llm.calls[0]["messages"][1]["content"])
    assert [row["arm_id"] for row in payload["blocked_arms"]]==["A"]
    assert [row["arm_id"] for row in decision.reviews]==["A","B"]
    assert decision.reviews[1]["recommendation"]=="insufficient_evidence"

def test_orchestrator_never_reads_structural_summaries():
    class StructuralTrap:
        total_structures=2; aggregate_tags={"ok":2}; reliable_seed_fraction=.5
        observations=["aggregate"]; interface_data_quality={"measured":True}
        @property
        def summaries(self): raise AssertionError("summaries must not be accessed")
    class ReviewSpy:
        def review(self, **kwargs):
            assert kwargs["context"]["structural_summary"]["total_structures"]==2
            return type("Decision", (), {"reviews":[{"arm_id":"A","recommendation":"keep_blocked","accepted_evidence_ids":[],"counterevidence_ids":[],"risk_codes":[],"reason":"x"}], "llm_used":True, "raw":{}, "to_dict":lambda self:{}})()
    ledger=ExperimentLedger(arm_blocks={"A":{"arm_id":"A","status":"soft_blocked"}})
    class Memory: experiment_ledger=ledger
    class Store:
        @staticmethod
        def apply_arm_unfreeze(*args,**kwargs): raise AssertionError("unexpected")
    with tempfile.TemporaryDirectory() as tmp:
        orch=BinderDesignOrchestrator.__new__(BinderDesignOrchestrator); orch.blocked_arm_review_agent=ReviewSpy(); orch.memory_store=Store()
        orch._write_json=lambda path,payload: Path(path).write_text(json.dumps(payload))
        orch._review_and_unfreeze_arms(memory=Memory(),round_dir=Path(tmp),next_round_id=3,blocked_arms={"A"},arm_evidence_cards={"arms":[{"evidence_id":"EA","arm_id":"A","status":"closed","requested_budget":1,"completed_budget":1,"trials":1}]},selection_context={},hypotheses=[],structural_summary=StructuralTrap(),quality_analysis={})

def test_orchestrator_isolates_arms_and_rejects_cross_arm_evidence():
    class ReviewSpy:
        def __init__(self): self.kwargs=None
        def review(self, **kwargs):
            self.kwargs=kwargs
            assert [item["arm_id"] for item in kwargs["blocked_arms"]]==["A"]
            assert {item["arm_id"] for item in kwargs["evidence"]}=={"A"}
            return type("Decision", (), {"reviews":[
                {"arm_id":"A","recommendation":"eligible_for_unfreeze","accepted_evidence_ids":["EB"],"counterevidence_ids":[],"risk_codes":[],"reason":"cross-arm"},
                {"arm_id":"B","recommendation":"eligible_for_unfreeze","accepted_evidence_ids":["EA"],"counterevidence_ids":[],"risk_codes":[],"reason":"invented"},
            ], "llm_used":True, "raw":{}, "to_dict":lambda self:{}})()
    ledger=ExperimentLedger(arm_blocks={arm:{"arm_id":arm,"status":"soft_blocked"} for arm in ("A","B")})
    class Memory: experiment_ledger=ledger
    class Store:
        unfrozen=[]
        @classmethod
        def apply_arm_unfreeze(cls,*args,**kwargs): cls.unfrozen.append(kwargs["arm_id"])
    with tempfile.TemporaryDirectory() as tmp:
        orch=BinderDesignOrchestrator.__new__(BinderDesignOrchestrator); spy=ReviewSpy(); orch.blocked_arm_review_agent=spy; orch.memory_store=Store()
        orch._write_json=lambda path,payload: Path(path).write_text(json.dumps(payload))
        remaining=orch._review_and_unfreeze_arms(memory=Memory(),round_dir=Path(tmp),next_round_id=3,blocked_arms={"A","B"},arm_evidence_cards={"arms":[
            {"evidence_id":"EA","arm_id":"A","status":"closed","requested_budget":1,"completed_budget":1,"trials":1},
            {"evidence_id":"EB","arm_id":"B","status":"incomplete","requested_budget":2,"completed_budget":1,"trials":1},
        ]},selection_context={},hypotheses=[],structural_summary={},quality_analysis={})
        saved=json.loads((Path(tmp)/"blocked_arm_review.json").read_text())
        assert remaining=={"A","B"} and not Store.unfrozen
        assert [row["arm_id"] for row in saved["reviews"]]==["A","B"]
        assert saved["reviews"][1]["recommendation"]=="insufficient_evidence"

def test_durable_unfreeze_audit():
    with tempfile.TemporaryDirectory() as tmp:
        store=ExperimentMemoryStore(tmp); memory=store.load()
        store.record_arm_block(memory,arm_id="sampler_explore",round_id=1,reason="regressed",cooldown_until_round=2)
        store.apply_arm_unfreeze(memory,arm_id="sampler_explore",round_id=3,evidence_ids=["E1"],reason="reviewed")
        assert memory.experiment_ledger.arm_blocks["sampler_explore"]["status"]=="unfrozen"
        assert memory.experiment_ledger.arm_unfreeze_audit[-1]["evidence_ids"]==["E1"]

if __name__=="__main__":
    test_valid_unfreeze_review(); test_invalid_ids_fail_closed(); test_unavailable_keeps_blocked(); test_huge_payload_is_compacted_and_ids_retained(); test_context_limit_required_fails_closed(); test_auth_and_quota_propagate(); test_orchestrator_skips_llm_without_exact_complete_evidence(); test_orchestrator_passes_only_lightweight_blocked_ledger(); test_agent_skips_llm_without_complete_evidence(); test_agent_partitions_partial_complete_evidence(); test_orchestrator_never_reads_structural_summaries(); test_orchestrator_isolates_arms_and_rejects_cross_arm_evidence(); test_durable_unfreeze_audit(); print("BLOCKED ARM REVIEW TESTS PASSED")
