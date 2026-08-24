#!/usr/bin/env python3
"""Regression tests for indexed retrieval and performance-aware compression."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.context_compaction import compact_memory
from binderloop.agents.memory_compression_agent import MemoryCompressionAgent
from binderloop.agents.memory_retrieval_agent import (
    MemoryRetrievalAgent,
    MemoryRetrievalQuery,
)
from binderloop.memory import (
    ExperimentMemory,
    ExperimentMemoryStore,
    MemoryItem,
    RoundRecord,
    build_round_memory_item,
    parameter_diff,
    target_memory_key,
)


TARGET = {
    "structure_path": "examples/bg_example/SC2RBD.cif",
    "chain_id": "E",
    "hotspots": ["E:153", "E:157"],
}


class FakeRerankLLM:
    def __init__(self):
        self.calls = []

    def available(self):
        return True

    def chat_json(self, *, system, user, **kwargs):
        self.calls.append(user)
        if "candidates" in user:
            return {
                "ranked": [
                    {
                        "item_id": row["item_id"],
                        "relevance": 1.0 - index * 0.1,
                        "reason": "field and summary match",
                    }
                    for index, row in enumerate(user["candidates"])
                ]
            }
        return {"summary": "LLM-compressed evidence preserving supplied ranges."}


def item(
    item_id,
    round_id,
    reward,
    tags,
    *,
    target=TARGET,
    arm="baseline",
    params=None,
    summary=None,
):
    return MemoryItem(
        item_id=item_id,
        round_id=round_id,
        target=dict(target),
        target_key=target_memory_key(target),
        failure_tags=list(tags),
        parameter_diff=dict(params or {}),
        arm=arm,
        reward=reward,
        reward_delta=None,
        performance={"core_objective": reward},
        summary=summary or f"round {round_id} tags {' '.join(tags)}",
        source_round_ids=[round_id],
    )


class MemorySchemaTest(unittest.TestCase):
    def test_memory_optimization_defaults_are_opt_in_off(self):
        from binderloop.config import MemorySpec

        spec = MemorySpec()
        self.assertFalse(spec.enabled)
        self.assertFalse(spec.index_items)
        self.assertFalse(spec.retrieval)
        self.assertFalse(spec.semantic_rerank)
        self.assertFalse(spec.compression)
        self.assertFalse(spec.apply_prompt_budget)
        self.assertFalse(spec.wants_index_items())
        self.assertFalse(spec.wants_retrieval())
        self.assertFalse(spec.wants_semantic_rerank())
        self.assertFalse(spec.wants_compression())
        self.assertFalse(spec.wants_prompt_budget())
        self.assertFalse(spec.any_optimization_enabled())

    def test_enabled_master_does_not_imply_semantic_rerank(self):
        from binderloop.config import MemorySpec

        spec = MemorySpec(enabled=True)
        self.assertTrue(spec.wants_index_items())
        self.assertTrue(spec.wants_retrieval())
        self.assertTrue(spec.wants_compression())
        self.assertTrue(spec.wants_prompt_budget())
        self.assertFalse(spec.wants_semantic_rerank())

    def test_selective_flags_are_independent(self):
        from binderloop.config import HarnessConfig, MemorySpec, TargetSpec
        from binderloop.orchestration.orchestrator import BinderDesignOrchestrator

        with tempfile.TemporaryDirectory() as directory:
            cfg = HarnessConfig(target=TargetSpec(**TARGET))
            cfg.memory = MemorySpec(retrieval=True, semantic_rerank=False)
            orchestrator = BinderDesignOrchestrator(
                cfg,
                out_dir=Path(directory),
                max_rounds=1,
            )
            self.assertFalse(orchestrator.memory_index_enabled)
            self.assertIsNotNone(orchestrator.memory_retrieval_agent)
            self.assertIsNone(orchestrator.memory_compression_agent)
            self.assertIsNone(orchestrator.memory_retrieval_agent.llm)

    def test_old_memory_file_loads_with_new_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiment_memory.json").write_text(
                json.dumps({
                    "experiment_id": "old",
                    "target": TARGET,
                    "rounds": [],
                    "messages": [],
                    "template_library": [],
                    "round_metrics": [],
                }),
                encoding="utf-8",
            )
            loaded = ExperimentMemoryStore(root).load()
            self.assertEqual(loaded.memory_schema_version, "2.1")
            self.assertEqual(loaded.memory_items, [])

    def test_round_item_indexes_requested_fields(self):
        diff = parameter_diff(
            {"alpha": 0.001, "hotspot_weight": 1.0},
            {"alpha": 0.002, "hotspot_weight": 1.0},
            allowed_keys={"alpha", "hotspot_weight"},
        )
        built = build_round_memory_item(
            round_id=2,
            target=TARGET,
            failure_tags=["hotspot_miss", "binding_pose_failure"],
            config_diff=diff,
            arm="explore_patch",
            outcome={
                "reward": 0.4,
                "core_objective": 0.45,
                "success_count": 1,
                "execution_failed": False,
            },
            previous_reward=0.55,
        )
        self.assertEqual(built.arm, "explore_patch")
        self.assertEqual(built.reward_delta, -0.15)
        self.assertEqual(built.parameter_diff["alpha"]["delta"], 0.001)
        self.assertEqual(built.target_key, target_memory_key(TARGET))
        self.assertIn("hotspot_miss", built.failure_tags)

    def test_orchestrator_backfills_existing_round_records(self):
        from binderloop.config import HarnessConfig, TargetSpec
        from binderloop.orchestration.orchestrator import BinderDesignOrchestrator

        with tempfile.TemporaryDirectory() as directory:
            cfg = HarnessConfig(target=TargetSpec(**TARGET))
            cfg.memory.index_items = True
            cfg.memory.semantic_rerank = False
            orchestrator = BinderDesignOrchestrator(
                cfg,
                out_dir=Path(directory),
                max_rounds=1,
            )
            self.assertTrue(orchestrator.memory_index_enabled)
            self.assertIsNone(orchestrator.memory_retrieval_agent)
            self.assertIsNone(orchestrator.memory_compression_agent)
            memory = ExperimentMemory(
                experiment_id="old",
                target=TARGET,
                rounds=[
                    RoundRecord(
                        round_id=0,
                        evaluation={"tag_counts": {"hotspot_miss": 3}},
                        reward=0.25,
                        config_snapshot={"alpha": 0.001},
                    )
                ],
                round_metrics=[{
                    "round_id": 0,
                    "reward": 0.25,
                    "core_objective": 0.25,
                    "execution_failed": False,
                }],
            )
            orchestrator._backfill_indexed_memory(memory)
            self.assertEqual(len(memory.memory_items), 1)
            self.assertEqual(memory.memory_items[0].failure_tags, ["hotspot_miss"])


class MemoryRetrievalTest(unittest.TestCase):
    def test_structured_filter_precedes_llm_rerank_and_cache(self):
        llm = FakeRerankLLM()
        agent = MemoryRetrievalAgent(llm=llm, candidate_limit=10, top_k=4)
        other_target = {**TARGET, "chain_id": "A"}
        items = [
            item("match", 1, 0.3, ["hotspot_miss"]),
            item("wrong_tag", 2, 0.8, ["clash"]),
            item("wrong_target", 3, 0.9, ["hotspot_miss"], target=other_target),
        ]
        query = MemoryRetrievalQuery(
            target=TARGET,
            failure_tags=["hotspot_miss"],
            intent="repair hotspot failure",
        )
        result = agent.retrieve(items, query)
        self.assertEqual([row.item_id for row in result.items], ["match"])
        self.assertEqual(
            [row["item_id"] for row in llm.calls[0]["candidates"]],
            ["match"],
        )
        cached = agent.retrieve(items, query)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(len(llm.calls), 1)

    def test_clustering_and_mmr_reduce_repeated_memories(self):
        agent = MemoryRetrievalAgent(llm=None, top_k=3, mmr_lambda=0.55)
        items = [
            item("dup_old", 0, 0.2, ["hotspot_miss"], summary="same hotspot miss pattern"),
            item("dup_new", 1, 0.3, ["hotspot_miss"], summary="same hotspot miss pattern"),
            item("pose", 2, 0.4, ["binding_pose_failure"], summary="pose and interface failure"),
            item("clash", 3, 0.5, ["clash"], summary="packing clash at interface"),
        ]
        result = agent.retrieve(
            items,
            MemoryRetrievalQuery(target=TARGET, intent="review all failure patterns"),
        )
        selected = {row.item_id for row in result.items}
        self.assertEqual(len(selected & {"dup_old", "dup_new"}), 1)
        self.assertGreaterEqual(len({tuple(row.failure_tags) for row in result.items}), 2)


class MemoryCompressionTest(unittest.TestCase):
    def test_bad_performance_then_old_age_controls_compression(self):
        items = [
            item("strong_old", 0, 0.9, ["hotspot_miss"]),
            item("weak_new", 5, 0.1, ["hotspot_miss"]),
            item("weak_old", 1, 0.1, ["hotspot_miss"]),
            item("middle", 2, 0.5, ["hotspot_miss"]),
        ]
        result = MemoryCompressionAgent(max_active_items=3, batch_size=6).compress_to_budget(items)
        archived = set(result.archived_item_ids)
        self.assertEqual(archived, {"weak_old", "weak_new"})
        self.assertFalse(next(row for row in result.items if row.item_id == "strong_old").archived)
        self.assertEqual(result.after_active_count, 3)
        compressed = result.compressed_items[0]
        self.assertEqual(compressed.source_round_ids, [1, 5])
        self.assertEqual(set(compressed.source_item_ids), {"weak_old", "weak_new"})

    def test_llm_failure_is_not_required_for_compression(self):
        result = MemoryCompressionAgent(max_active_items=1).compress_to_budget([
            item("a", 0, 0.1, ["clash"]),
            item("b", 1, 0.2, ["clash"]),
        ])
        self.assertFalse(result.llm_used)
        self.assertIn("Compressed rounds", result.compressed_items[0].summary)

    def test_compacted_prompt_prefers_recalled_items(self):
        recalled = item("recall", 7, 0.7, ["hotspot_miss"])
        payload = compact_memory({
            "experiment_id": "x",
            "target": TARGET,
            "extend_memory": True,
            "recalled_items": [recalled.__dict__],
            "recent_rounds": [
                {"round_id": round_id, "evaluation": {}, "jobs": []}
                for round_id in range(8)
            ],
        })
        self.assertEqual(payload["recalled_items"][0]["item_id"], "recall")
        self.assertNotIn("recent_rounds", payload)


class MemoryCliOverrideTest(unittest.TestCase):
    def test_cli_flags_selectively_activate_features(self):
        from binderloop.config import HarnessConfig, TargetSpec, apply_memory_cli_overrides

        cfg = HarnessConfig(target=TargetSpec(**TARGET))
        apply_memory_cli_overrides(
            cfg.memory,
            enabled=False,
            index_items=True,
            retrieval=False,
            semantic_rerank=True,
            compression=False,
            apply_prompt_budget=True,
        )
        self.assertTrue(cfg.memory.index_items)
        self.assertTrue(cfg.memory.retrieval)  # implied by semantic_rerank
        self.assertTrue(cfg.memory.semantic_rerank)
        self.assertFalse(cfg.memory.compression)
        self.assertTrue(cfg.memory.apply_prompt_budget)
        self.assertFalse(cfg.memory.enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
