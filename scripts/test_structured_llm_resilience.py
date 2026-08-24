#!/usr/bin/env python3
import json
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.llm import LLMSettings, ModelEndpoint, OpenAICompatibleClient, LLMDefinitiveError
from binderloop.structured_llm import call_structured_json
from binderloop.skills.composer import compose_agent_system
from binderloop.agents.strategy_conflict_resolution_agent import StrategyConflictResolutionAgent


class SequenceLLM:
    def __init__(self, values): self.values=list(values); self.calls=[]
    def available(self): return True
    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs); value=self.values.pop(0)
        return {"choices":[{"message":{"content":value},"finish_reason":"stop"}],"usage":{"completion_tokens":20}}


class StructuredResilienceTest(unittest.TestCase):
    def test_unknown_explanation_field_is_ignored(self):
        llm=SequenceLLM([json.dumps({"a":1,"explanation":"ok"})])
        out=call_structured_json(llm,system="json",user={},required_fields=["a"])
        self.assertEqual(out.value,{"a":1})
        self.assertEqual(out.attempts[0]["unknown_fields"],["explanation"])

    def test_missing_field_gets_one_targeted_repair(self):
        llm=SequenceLLM(['{"a":1}', '{"a":1,"b":2}'])
        out=call_structured_json(llm,system="json",user={},required_fields=["a","b"])
        self.assertTrue(out.repaired); self.assertEqual(out.value["b"],2)
        repair=json.loads(llm.calls[1]["messages"][1]["content"])
        self.assertEqual(repair["missing_fields"],["b"])
        self.assertEqual(repair["task"],"repair_json_only_do_not_reanalyze")

    def test_reasoning_only_retries_with_reasoning_off_and_larger_budget(self):
        class ReasoningSequence(SequenceLLM):
            def create_chat_completion(self, **kwargs):
                self.calls.append(kwargs)
                value=self.values.pop(0)
                if value is None:
                    return {"choices":[{"message":{"content":"","reasoning_content":"hidden"},"finish_reason":"length"}],"usage":{"completion_tokens":64,"completion_tokens_details":{"reasoning_tokens":64}}}
                return {"choices":[{"message":{"content":value},"finish_reason":"stop"}],"usage":{"completion_tokens":4}}
        llm=ReasoningSequence([None, json.dumps({"a":1})])
        out=call_structured_json(llm,system="json",user={},required_fields=["a"],visible_json_tokens=64,max_completion_tokens=1024,thinking="low",reasoning_budget_tokens=32)
        self.assertEqual(out.value,{"a":1})
        self.assertEqual([call["max_tokens"] for call in llm.calls],[96,1024])
        self.assertEqual(llm.calls[0]["reasoning_budget_tokens"],32)
        self.assertEqual(llm.calls[1]["thinking"],"low")
        self.assertEqual(llm.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(out.attempts[0]["retry_reason"],"length")
        self.assertEqual(out.attempts[0]["visible_json_tokens"],64)

    def test_initial_and_repair_prompts_are_budgeted(self):
        llm=SequenceLLM(['{"a":1}', json.dumps({"a":1,"b":2})])
        llm.resolved_endpoint=ModelEndpoint(name="x",base_url="https://x",api_key="k",max_prompt_bytes=1400)
        out=call_structured_json(llm,system="json",user={"blob":"X"*100000},required_fields=["a","b"])
        self.assertEqual(out.value["b"],2)
        self.assertTrue(out.attempts[0]["prompt_compacted"])
        self.assertLessEqual(out.attempts[0]["prompt_final_bytes"],1400)
        self.assertLessEqual(out.attempts[1]["prompt_final_bytes"],1400)
        self.assertNotIn("prompt", out.attempts[0])

    def test_context_limit_returns_structured_failure_but_auth_raises(self):
        class Definitive:
            def __init__(self, failure): self.failure=failure
            def available(self): return True
            def create_chat_completion(self, **kwargs): raise LLMDefinitiveError(400,"failed",failure_class=self.failure)
        out=call_structured_json(Definitive("context_limit"),system="json",user={},required_fields=["a"])
        self.assertIsNone(out.value); self.assertEqual(out.attempts[0]["failure_class"],"context_limit")
        self.assertIn("status",out.attempts[0]); self.assertIn("error",out.attempts[0])
        with self.assertRaises(LLMDefinitiveError):
            call_structured_json(Definitive("authentication"),system="json",user={},required_fields=["a"])

    def test_history_schema_failure_falls_back_even_when_required(self):
        llm=SequenceLLM(["bad", "still bad"])
        comparison={"status":"winner","winner_arm_id":"a","closed_arm_ids":["a","b"],"evidence_ids":["A"]}
        out=StrategyConflictResolutionAgent(llm,require_llm=True).resolve_arm_direction(round_id=1,arm_comparison=comparison,ledger_history={"recent_rounds":[]})
        self.assertFalse(out.llm_used); self.assertEqual(out.selected_arm_id,"a")
        self.assertEqual(len(out.raw["llm_attempts"]),2)

    def test_composer_does_not_require_schema_extra_fields(self):
        text=compose_agent_system("Return exactly {\"a\":1}",active_skills=[{"id":"s","guidance":["use evidence"]}])
        self.assertNotIn("Every LLM response must report",text)
        self.assertIn("must never add top-level response fields",text)

    def test_completion_budget_clamps_to_endpoint(self):
        client=OpenAICompatibleClient(LLMSettings(default_model="x",enabled=True,endpoints={"x":ModelEndpoint(name="x",base_url="https://x",api_key="k",max_output_tokens=8192)}))
        budget=client.effective_completion_budget(1_000_000)
        self.assertEqual(budget["effective_completion_tokens"],8192)
        self.assertEqual(budget["completion_clamp_reason"],"endpoint_max_output_tokens")


if __name__ == "__main__": unittest.main(verbosity=2)
