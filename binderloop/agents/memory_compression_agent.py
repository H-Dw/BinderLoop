"""Performance-first, age-second compression for indexed experiment memory."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from binderloop.llm import OpenAICompatibleClient
from binderloop.memory import MemoryItem


@dataclass
class MemoryCompressionResult:
    items: List[MemoryItem]
    compressed_items: List[MemoryItem] = field(default_factory=list)
    archived_item_ids: List[str] = field(default_factory=list)
    llm_used: bool = False
    before_active_count: int = 0
    after_active_count: int = 0


class MemoryCompressionAgent:
    """Compress weak rounds first, breaking performance ties by oldest round."""

    SYSTEM = """You compress binder-design evidence without changing facts.
Return JSON only: {"summary":"one concise evidence summary"}.
Preserve failure modes, parameter directions, arms, reward range, and uncertainty.
Do not invent causal claims or numeric values."""

    def __init__(
        self,
        llm: Optional[OpenAICompatibleClient] = None,
        *,
        max_active_items: int = 24,
        batch_size: int = 6,
        max_summary_chars: int = 1200,
    ) -> None:
        self.llm = llm
        self.max_active_items = max(1, int(max_active_items))
        self.batch_size = max(2, int(batch_size))
        self.max_summary_chars = max(120, int(max_summary_chars))

    @staticmethod
    def compression_priority(item: MemoryItem) -> Tuple[float, int, float, str]:
        """Lexicographic order required by policy: poor performance, then age."""
        if item.execution_failed:
            performance = float("-inf")
        elif item.reward is not None:
            performance = float(item.reward)
        else:
            try:
                performance = float(item.performance.get("core_objective"))
            except (TypeError, ValueError):
                performance = float("-inf")
        return performance, int(item.round_id), float(item.created_at), item.item_id

    def compress_to_budget(
        self,
        items: Sequence[MemoryItem],
        *,
        max_active_items: Optional[int] = None,
    ) -> MemoryCompressionResult:
        all_items = list(items)
        active = [item for item in all_items if not item.archived]
        limit = max(1, int(max_active_items or self.max_active_items))
        result = MemoryCompressionResult(
            items=all_items,
            before_active_count=len(active),
            after_active_count=len(active),
        )
        llm_used = False
        while len(active) > limit:
            # Compress only as many sources as required to reach the budget;
            # this prevents a same-signature cluster from pulling a strong
            # round into an early low-performance compression batch.
            needed_reduction = len(active) - limit
            batch = self._next_batch(
                active,
                max_batch=min(self.batch_size, needed_reduction + 1),
            )
            if len(batch) < 2:
                break
            compressed, used = self._compress_batch(batch)
            llm_used = llm_used or used
            source_ids = {item.item_id for item in batch}
            for item in all_items:
                if item.item_id in source_ids:
                    item.archived = True
                    item.compressed_into = compressed.item_id
            all_items.append(compressed)
            result.compressed_items.append(compressed)
            result.archived_item_ids.extend(sorted(source_ids))
            active = [item for item in all_items if not item.archived]
        result.items = all_items
        result.after_active_count = len(active)
        result.llm_used = llm_used
        return result

    def _next_batch(
        self,
        active: Sequence[MemoryItem],
        *,
        max_batch: Optional[int] = None,
    ) -> List[MemoryItem]:
        ordered = sorted(active, key=self.compression_priority)
        if len(ordered) < 2:
            return ordered
        limit = max(2, min(self.batch_size, int(max_batch or self.batch_size)))
        seed = ordered[0]
        signature = (
            seed.target_key,
            tuple(sorted(seed.failure_tags)),
            seed.arm,
        )
        related = [
            item
            for item in ordered[1:]
            if (
                item.target_key,
                tuple(sorted(item.failure_tags)),
                item.arm,
            ) == signature
        ]
        batch = [seed] + related[: limit - 1]
        if len(batch) < 2:
            same_target = [
                item for item in ordered[1:]
                if item.target_key == seed.target_key
            ]
            batch.extend(same_target[: 2 - len(batch)])
        if len(batch) < 2:
            batch.append(ordered[1])
        return batch[:limit]

    def _compress_batch(self, batch: Sequence[MemoryItem]) -> Tuple[MemoryItem, bool]:
        source_ids = sorted(item.item_id for item in batch)
        source_rounds = sorted({
            round_id
            for item in batch
            for round_id in (item.source_round_ids or [item.round_id])
        })
        rewards = [float(item.reward) for item in batch if item.reward is not None]
        failure_tags = sorted({tag for item in batch for tag in item.failure_tags})
        parameter_diff = self._aggregate_parameter_diff(batch)
        arms = sorted({item.arm for item in batch if item.arm})
        payload = {
            "source_items": [
                {
                    "item_id": item.item_id,
                    "round_ids": item.source_round_ids or [item.round_id],
                    "failure_tags": item.failure_tags,
                    "parameter_diff": item.parameter_diff,
                    "arm": item.arm,
                    "reward": item.reward,
                    "reward_delta": item.reward_delta,
                    "performance": item.performance,
                    "summary": item.summary[: self.max_summary_chars],
                }
                for item in batch
            ],
            "immutable_aggregate": {
                "source_round_ids": source_rounds,
                "failure_tags": failure_tags,
                "parameter_diff": parameter_diff,
                "arms": arms,
                "reward_min": min(rewards) if rewards else None,
                "reward_max": max(rewards) if rewards else None,
            },
        }
        summary = self._deterministic_summary(payload["immutable_aggregate"])
        llm_used = False
        if self.llm and self.llm.available():
            try:
                response = self.llm.chat_json(
                    system=self.SYSTEM,
                    user=payload,
                    temperature=0.0,
                    max_tokens=600,
                    max_prompt_bytes=120_000,
                )
                candidate = str(response.get("summary") or "").strip()
                if candidate:
                    summary = candidate[: self.max_summary_chars]
                    llm_used = True
            except Exception:
                pass
        identity = hashlib.sha256(
            json.dumps(source_ids, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        reward = max(rewards) if rewards else None
        deltas = [float(item.reward_delta) for item in batch if item.reward_delta is not None]
        return MemoryItem(
            item_id=f"mem_comp_{identity}",
            round_id=max(source_rounds) if source_rounds else max(item.round_id for item in batch),
            item_type="compressed_rounds",
            target=dict(batch[0].target),
            target_key=batch[0].target_key,
            failure_tags=failure_tags,
            parameter_diff=parameter_diff,
            arm=arms[0] if len(arms) == 1 else ("mixed" if arms else ""),
            reward=reward,
            reward_delta=max(deltas) if deltas else None,
            performance={
                "reward_min": min(rewards) if rewards else None,
                "reward_max": max(rewards) if rewards else None,
                "source_count": len(batch),
            },
            execution_failed=all(item.execution_failed for item in batch),
            summary=summary,
            source_round_ids=source_rounds,
            source_item_ids=source_ids,
            artifact_refs=sorted({path for item in batch for path in item.artifact_refs}),
            compression_level=max(item.compression_level for item in batch) + 1,
        ), llm_used

    @staticmethod
    def _aggregate_parameter_diff(
        batch: Sequence[MemoryItem],
    ) -> Dict[str, Dict[str, Any]]:
        aggregate: Dict[str, Dict[str, Any]] = {}
        for key in sorted({key for item in batch for key in item.parameter_diff}):
            rows = [item.parameter_diff[key] for item in batch if key in item.parameter_diff]
            before_values = _unique([row.get("before") for row in rows])
            after_values = _unique([row.get("after") for row in rows])
            deltas = [row.get("delta") for row in rows if isinstance(row.get("delta"), (int, float))]
            directions = sorted({
                "increase" if float(delta) > 0 else "decrease" if float(delta) < 0 else "flat"
                for delta in deltas
            })
            aggregate[key] = {
                "before_values": before_values,
                "after_values": after_values,
                "directions": directions,
            }
        return aggregate

    @staticmethod
    def _deterministic_summary(aggregate: Mapping[str, Any]) -> str:
        rounds = aggregate.get("source_round_ids") or []
        tags = aggregate.get("failure_tags") or []
        params = sorted((aggregate.get("parameter_diff") or {}).keys())
        return (
            f"Compressed rounds {rounds}; reward range "
            f"{aggregate.get('reward_min')}..{aggregate.get('reward_max')}; "
            f"failure tags={tags or ['none']}; changed parameters={params or ['none']}; "
            f"arms={aggregate.get('arms') or ['baseline']}."
        )


def _unique(values: Sequence[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result
