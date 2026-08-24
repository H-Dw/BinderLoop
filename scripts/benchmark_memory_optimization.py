#!/usr/bin/env python3
"""Measure indexed-memory retrieval and compression on a durable run."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.context_compaction import compact_memory
from binderloop.agents.memory_compression_agent import MemoryCompressionAgent
from binderloop.agents.memory_retrieval_agent import MemoryRetrievalAgent, MemoryRetrievalQuery
from binderloop.agents.config_parameter_contract import supported_config_changes
from binderloop.memory import (
    ExperimentMemoryStore,
    MemoryItem,
    build_round_memory_item,
    parameter_diff,
)


DEFAULT_MEMORY = (
    "outputs/sc2rbd_closed_loop_llm_np_160s_8r_v17_bug/"
    "sc2rbd_closed_loop_llm_np_160s_8r_v17/memory"
)


def _wire_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def build_items(memory: Any) -> List[MemoryItem]:
    metrics = {
        int(row.get("round_id", -1)): dict(row)
        for row in (memory.round_metrics or [])
    }
    previous_config: Dict[str, Any] = {}
    previous_reward = None
    items: List[MemoryItem] = []
    for record in sorted(memory.rounds, key=lambda row: row.round_id):
        if not record.evaluation:
            continue
        current_config = supported_config_changes(record.config_snapshot, include_internal=True)
        diff = parameter_diff(
            previous_config,
            current_config,
            allowed_keys=set(previous_config) | set(current_config),
        )
        outcome = metrics.get(record.round_id, {
            "round_id": record.round_id,
            "reward": record.reward,
            "execution_failed": False,
        })
        tags = [
            key for key, count in dict((record.evaluation or {}).get("tag_counts") or {}).items()
            if count and not str(key).startswith("pass_")
        ]
        items.append(build_round_memory_item(
            round_id=record.round_id,
            target=memory.target,
            failure_tags=tags,
            config_diff=diff,
            arm=str(outcome.get("arm_signature") or ""),
            outcome=outcome,
            artifact_refs=record.artifacts,
            previous_reward=previous_reward,
        ))
        if not outcome.get("execution_failed") and outcome.get("reward") is not None:
            previous_reward = float(outcome["reward"])
        previous_config = current_config
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY)
    parser.add_argument(
        "--out",
        default="outputs/gpt55_limit_probe_sc2rbd_round00/memory_benchmark.json",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    memory_dir = Path(args.memory_dir)
    if not memory_dir.is_absolute():
        memory_dir = root / memory_dir
    output = Path(args.out)
    if not output.is_absolute():
        output = root / output
    memory = ExperimentMemoryStore(memory_dir).load()
    items = build_items(memory)
    legacy = compact_memory(ExperimentMemoryStore(memory_dir).summarize_for_agent(
        memory,
        extend_memory=True,
    ))
    last = items[-1] if items else None
    query = MemoryRetrievalQuery(
        target=memory.target,
        failure_tags=list(last.failure_tags if last else []),
        arm=str(last.arm if last else ""),
        parameter_names=sorted(last.parameter_diff if last else {}),
        intent="Find diverse historical evidence for the latest round.",
    )
    retrieval = MemoryRetrievalAgent(
        llm=None,
        candidate_limit=24,
        top_k=8,
        mmr_lambda=0.7,
    ).retrieve(items, query)
    indexed = compact_memory(ExperimentMemoryStore(memory_dir).summarize_for_agent(
        memory,
        extend_memory=True,
        recalled_items=retrieval.items,
    ))
    simulation_limit = max(2, len(items) // 2) if items else 2
    compression = MemoryCompressionAgent(
        llm=None,
        max_active_items=simulation_limit,
    ).compress_to_budget(items)
    active_after = [item for item in compression.items if not item.archived]
    report = {
        "memory_dir": str(memory_dir),
        "round_count": len(memory.rounds),
        "indexed_item_count": len(items),
        "legacy_prompt_bytes": _wire_bytes(legacy),
        "retrieved_prompt_bytes": _wire_bytes(indexed),
        "retrieval": {
            "structured_candidate_count": retrieval.structured_candidate_count,
            "selected_count": len(retrieval.items),
            "selected_item_ids": [item.item_id for item in retrieval.items],
            "cluster_mmr_reduction_fraction": round(
                1.0 - len(retrieval.items) / max(1, retrieval.structured_candidate_count),
                6,
            ),
        },
        "compression_policy_simulation": {
            "active_item_limit": simulation_limit,
            "before_active_count": compression.before_active_count,
            "after_active_count": compression.after_active_count,
            "archived_item_ids": compression.archived_item_ids,
            "compressed_item_ids": [item.item_id for item in compression.compressed_items],
            "active_item_ids": [item.item_id for item in active_after],
            "compressed_source_rounds": [
                item.source_round_ids for item in compression.compressed_items
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
