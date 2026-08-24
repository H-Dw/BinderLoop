"""Two-stage structured + semantic memory retrieval with diversity control."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from binderloop.llm import OpenAICompatibleClient
from binderloop.memory import MemoryItem, target_memory_key


@dataclass
class MemoryRetrievalQuery:
    target: Dict[str, Any] = field(default_factory=dict)
    failure_tags: List[str] = field(default_factory=list)
    arm: str = ""
    parameter_names: List[str] = field(default_factory=list)
    item_types: List[str] = field(default_factory=list)
    intent: str = ""

    @property
    def target_key(self) -> str:
        return target_memory_key(self.target) if self.target else ""

    def semantic_text(self) -> str:
        return (
            f"intent={self.intent}; target={self.target_key}; "
            f"failure_tags={','.join(sorted(self.failure_tags))}; arm={self.arm}; "
            f"parameters={','.join(sorted(self.parameter_names))}; "
            f"item_types={','.join(sorted(self.item_types))}"
        )


@dataclass
class MemoryRetrievalResult:
    items: List[MemoryItem]
    structured_candidate_count: int
    semantic_rerank_used: bool
    cache_hit: bool
    selected_scores: Dict[str, float] = field(default_factory=dict)
    rerank_reasons: Dict[str, str] = field(default_factory=dict)
    retrieval_mode: str = "deterministic_structured_mmr"


class MemoryRetrievalAgent:
    """Recall by indexed fields, rerank semantically, then cluster and MMR."""

    SYSTEM = """You rerank compact binder-design memory evidence.
Return JSON only: {"ranked":[{"item_id":"...","relevance":0-1,"reason":"short evidence-grounded reason"}]}.
Judge relevance to the supplied query. Do not invent facts or alter item IDs."""

    def __init__(
        self,
        llm: Optional[OpenAICompatibleClient] = None,
        *,
        candidate_limit: int = 24,
        top_k: int = 8,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.llm = llm
        self.candidate_limit = max(1, int(candidate_limit))
        self.top_k = max(1, int(top_k))
        self.mmr_lambda = min(1.0, max(0.0, float(mmr_lambda)))
        self._cache: Dict[str, Tuple[Dict[str, float], Dict[str, str]]] = {}

    def retrieve(
        self,
        items: Sequence[MemoryItem],
        query: MemoryRetrievalQuery,
        *,
        top_k: Optional[int] = None,
    ) -> MemoryRetrievalResult:
        candidates = self.structured_recall(items, query)
        deterministic = {
            item.item_id: self._deterministic_relevance(item, query)
            for item in candidates
        }
        semantic_scores, reasons, semantic_used, cache_hit = self._semantic_rerank(
            candidates,
            query,
            deterministic,
        )
        clustered = self._cluster_representatives(candidates, semantic_scores)
        selected = self._mmr_select(
            clustered,
            semantic_scores,
            k=max(1, int(top_k or self.top_k)),
        )
        return MemoryRetrievalResult(
            items=selected,
            structured_candidate_count=len(candidates),
            semantic_rerank_used=semantic_used,
            cache_hit=cache_hit,
            selected_scores={item.item_id: round(semantic_scores.get(item.item_id, 0.0), 6) for item in selected},
            rerank_reasons={item.item_id: reasons.get(item.item_id, "") for item in selected if reasons.get(item.item_id)},
            retrieval_mode="semantic_opt_in" if semantic_used else "deterministic_structured_mmr",
        )

    def structured_recall(
        self,
        items: Sequence[MemoryItem],
        query: MemoryRetrievalQuery,
    ) -> List[MemoryItem]:
        """Apply exact indexed conditions before any LLM sees candidates."""
        active = [item for item in items if not item.archived]
        if query.target_key:
            active = [item for item in active if not item.target_key or item.target_key == query.target_key]
        if query.item_types:
            allowed = set(query.item_types)
            matching = [item for item in active if item.item_type in allowed]
            if matching:
                active = matching
        if query.failure_tags:
            wanted = set(query.failure_tags)
            matching = [item for item in active if wanted.intersection(item.failure_tags)]
            if matching:
                active = matching
        if query.arm:
            matching = [item for item in active if item.arm == query.arm]
            if matching:
                active = matching
        if query.parameter_names:
            wanted_params = set(query.parameter_names)
            matching = [item for item in active if wanted_params.intersection(item.parameter_diff)]
            if matching:
                active = matching
        # Explicit item_id tie-break makes recall reproducible across memory
        # serialization/input order. Semantic reranking remains opt-in via llm.
        active.sort(key=lambda item: item.item_id)
        active.sort(
            key=lambda item: (
                self._deterministic_relevance(item, query),
                item.round_id,
            ),
            reverse=True,
        )
        return active[: self.candidate_limit]

    def _semantic_rerank(
        self,
        candidates: Sequence[MemoryItem],
        query: MemoryRetrievalQuery,
        fallback_scores: Mapping[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, str], bool, bool]:
        if not candidates or not (self.llm and self.llm.available()):
            return dict(fallback_scores), {}, False, False
        cache_key = self._cache_key(candidates, query)
        if cache_key in self._cache:
            scores, reasons = self._cache[cache_key]
            return dict(scores), dict(reasons), True, True
        payload = {
            "query": query.semantic_text(),
            "candidates": [
                {
                    "item_id": item.item_id,
                    "round_id": item.round_id,
                    "item_type": item.item_type,
                    "failure_tags": item.failure_tags,
                    "parameter_names": sorted(item.parameter_diff),
                    "arm": item.arm,
                    "reward": item.reward,
                    "reward_delta": item.reward_delta,
                    "summary": item.summary[:800],
                }
                for item in candidates
            ],
        }
        try:
            result = self.llm.chat_json(
                system=self.SYSTEM,
                user=payload,
                temperature=0.0,
                max_tokens=1200,
                max_prompt_bytes=120_000,
            )
            ranked = list(result.get("ranked") or [])
            scores = dict(fallback_scores)
            reasons: Dict[str, str] = {}
            valid_ids = {item.item_id for item in candidates}
            for row in ranked:
                item_id = str((row or {}).get("item_id") or "")
                if item_id not in valid_ids:
                    continue
                try:
                    score = min(1.0, max(0.0, float((row or {}).get("relevance"))))
                except (TypeError, ValueError):
                    continue
                scores[item_id] = score
                reasons[item_id] = str((row or {}).get("reason") or "")[:400]
            self._cache[cache_key] = (dict(scores), dict(reasons))
            return scores, reasons, True, False
        except Exception:
            return dict(fallback_scores), {}, False, False

    def _cluster_representatives(
        self,
        candidates: Sequence[MemoryItem],
        scores: Mapping[str, float],
    ) -> List[MemoryItem]:
        clusters: Dict[Tuple[Any, ...], MemoryItem] = {}
        for item in candidates:
            signature = (
                item.target_key,
                item.item_type,
                tuple(sorted(item.failure_tags)),
                item.arm,
                tuple(sorted(item.parameter_diff)),
                item.compression_level,
            )
            existing = clusters.get(signature)
            if existing is None or (scores.get(item.item_id, 0.0), item.round_id) > (
                scores.get(existing.item_id, 0.0),
                existing.round_id,
            ):
                clusters[signature] = item
        return list(clusters.values())

    def _mmr_select(
        self,
        candidates: Sequence[MemoryItem],
        scores: Mapping[str, float],
        *,
        k: int,
    ) -> List[MemoryItem]:
        remaining = list(candidates)
        selected: List[MemoryItem] = []
        while remaining and len(selected) < k:
            def mmr(item: MemoryItem) -> Tuple[float, float, int, str]:
                relevance = float(scores.get(item.item_id, 0.0))
                redundancy = max((self._similarity(item, chosen) for chosen in selected), default=0.0)
                value = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * redundancy
                return value, relevance, item.round_id, item.item_id

            best = max(remaining, key=mmr)
            selected.append(best)
            remaining.remove(best)
        return selected

    @staticmethod
    def _deterministic_relevance(item: MemoryItem, query: MemoryRetrievalQuery) -> float:
        score = 0.0
        if query.target_key and item.target_key == query.target_key:
            score += 0.25
        wanted_tags = set(query.failure_tags)
        if wanted_tags:
            score += 0.35 * _jaccard(wanted_tags, set(item.failure_tags))
        if query.arm and item.arm == query.arm:
            score += 0.1
        wanted_params = set(query.parameter_names)
        if wanted_params:
            score += 0.2 * _jaccard(wanted_params, set(item.parameter_diff))
        if query.item_types and item.item_type in set(query.item_types):
            score += 0.1
        if not query.failure_tags and not query.parameter_names and item.reward is not None:
            score += 0.1 * max(0.0, min(1.0, float(item.reward)))
        return min(1.0, max(0.0, score))

    @staticmethod
    def _similarity(left: MemoryItem, right: MemoryItem) -> float:
        text = _jaccard(_tokens(left.summary), _tokens(right.summary))
        tags = _jaccard(set(left.failure_tags), set(right.failure_tags))
        params = _jaccard(set(left.parameter_diff), set(right.parameter_diff))
        arm = 1.0 if left.arm and left.arm == right.arm else 0.0
        return 0.45 * text + 0.3 * tags + 0.2 * params + 0.05 * arm

    @staticmethod
    def _cache_key(candidates: Sequence[MemoryItem], query: MemoryRetrievalQuery) -> str:
        raw = {
            "query": asdict(query),
            "candidates": [
                {
                    "item_id": item.item_id,
                    "summary": item.summary,
                    "failure_tags": item.failure_tags,
                    "parameter_diff": item.parameter_diff,
                    "arm": item.arm,
                }
                for item in candidates
            ],
        }
        return hashlib.sha256(
            json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()


def _tokens(value: str) -> set:
    return set(re.findall(r"[a-z0-9_:.+-]+", str(value).lower()))


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))
