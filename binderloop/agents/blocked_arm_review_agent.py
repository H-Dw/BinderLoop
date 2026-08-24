"""Evidence-bounded review of soft-blocked strategy arms."""
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from binderloop.llm import OpenAICompatibleClient
from binderloop.structured_llm import call_structured_json
from binderloop.agents.context_compaction import (
    MAX_PROMPT_BYTES, compact_context_for_blocked_arm_review, context_digest,
)

RECOMMENDATIONS = frozenset({"keep_blocked", "eligible_for_unfreeze", "insufficient_evidence"})

@dataclass
class BlockedArmReviewDecision:
    round_id: int
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    llm_used: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

class BlockedArmReviewAgent:
    SYSTEM = """You review previously soft-blocked protein-binder strategy arms. Return JSON only: {"reviews":[{"arm_id":"...","recommendation":"keep_blocked|eligible_for_unfreeze|insufficient_evidence","accepted_evidence_ids":[],"counterevidence_ids":[],"risk_codes":[],"reason":"..."}]}. Review every supplied arm exactly once. Use only supplied arm and evidence IDs. Infrastructure failures are not scientific evidence. Never emit parameter values or config changes. Recommend unfreeze only when new complete evidence contradicts the original arm-level block; otherwise keep blocked or report insufficient evidence."""
    def __init__(self, llm: Optional[OpenAICompatibleClient], *, require_llm: bool = False):
        self.llm=llm; self.require_llm=bool(require_llm)
    def review(self, *, round_id: int, blocked_arms: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> BlockedArmReviewDecision:
        all_arms = sorted({str(item.get("arm_id") or "") for item in blocked_arms if str(item.get("arm_id") or "")})
        complete_arms = {
            str(item.get("arm_id") or "")
            for item in evidence
            if str(item.get("arm_id") or "") in all_arms
            and str(item.get("status") or "").lower() == "closed"
            and int(item.get("completed_budget") or 0) >= int(item.get("requested_budget") or 0) > 0
            and int(item.get("trials") or 0) > 0
        }
        arms = sorted(complete_arms)
        blocked_arms = [item for item in blocked_arms if str(item.get("arm_id") or "") in complete_arms]
        evidence = [item for item in evidence if str(item.get("arm_id") or "") in complete_arms]
        evidence_ids = sorted({str(item.get("evidence_id") or "") for item in evidence if str(item.get("evidence_id") or "")})
        evidence_arm_by_id = {str(item.get("evidence_id")): str(item.get("arm_id") or "") for item in evidence if str(item.get("evidence_id") or "")}

        def deterministic(reason: str, *, attempts: Optional[List[Dict[str, Any]]] = None, payload: Optional[Mapping[str, Any]] = None) -> BlockedArmReviewDecision:
            return BlockedArmReviewDecision(
                round_id,
                [{
                    "arm_id": arm, "recommendation": "insufficient_evidence",
                    "accepted_evidence_ids": [], "counterevidence_ids": [],
                    "risk_codes": ["llm_or_evidence_unavailable"],
                    "reason": "Soft block retained by deterministic fallback.",
                } for arm in all_arms],
                False,
                {
                    "source": "deterministic_keep_blocked",
                    "fallback_reason": reason,
                    "llm_attempts": list(attempts or []),
                    "context_digest": context_digest(payload or {}),
                    "compaction": dict((payload or {}).get("_context_compaction") or {}),
                },
            )

        endpoint_budget = MAX_PROMPT_BYTES
        endpoint = getattr(self.llm, "resolved_endpoint", None)
        if endpoint is not None and getattr(endpoint, "max_prompt_bytes", None):
            endpoint_budget = int(endpoint.max_prompt_bytes)
        payload = compact_context_for_blocked_arm_review(
            round_id=round_id, blocked_arms=blocked_arms, evidence=evidence,
            context=context, max_bytes=endpoint_budget,
        )
        compaction_metadata = {
            "policy": "blocked_arm_review_allowlist",
            "max_bytes": endpoint_budget,
            "final_bytes": len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")),
            **dict(payload.get("_context_compaction") or {}),
        }
        fallback = deterministic("llm_or_evidence_unavailable", payload=payload)
        fallback.raw["compaction"] = compaction_metadata
        if not arms:
            return fallback
        if not (self.llm and self.llm.available()):
            if self.require_llm:
                raise RuntimeError("BlockedArmReviewAgent requires an available LLM")
            return fallback

        def validate(result):
            rows = result.get("reviews") if isinstance(result.get("reviews"), list) else []
            seen, illegal_arms, illegal_evidence, invalid = [], [], [], []
            for row in rows:
                arm = str((row or {}).get("arm_id") or ""); seen.append(arm)
                if arm not in arms: illegal_arms.append(arm)
                if str((row or {}).get("recommendation") or "") not in RECOMMENDATIONS: invalid.append("recommendation")
                for key in ("accepted_evidence_ids", "counterevidence_ids"):
                    ids = (row or {}).get(key)
                    if not isinstance(ids, list): invalid.append(key); continue
                    illegal_evidence.extend(
                        str(v) for v in ids
                        if str(v) not in evidence_ids or evidence_arm_by_id.get(str(v)) != arm
                    )
            if sorted(seen) != arms: invalid.append("reviews_complete_unique")
            return {"invalid_fields": sorted(set(invalid)), "illegal_arm_ids": sorted(set(illegal_arms)), "illegal_evidence_ids": sorted(set(illegal_evidence))}

        visible_contract = min(4096, max(1024, 512 * len(arms)))
        completion_budget = min(8192, max(4096, 1024 + 512 * len(arms), visible_contract))
        result = call_structured_json(
            self.llm, system=self.SYSTEM, user=payload, required_fields=("reviews",),
            field_validator=validate, temperature=.05,
            max_completion_tokens=completion_budget, visible_json_tokens=visible_contract,
            thinking="low", repair=True, valid_arm_ids=arms, valid_evidence_ids=evidence_ids,
        )
        if result.value is None:
            reason = "context_limit" if any(item.get("failure_class") == "context_limit" for item in result.attempts) else "invalid_structured_output"
            decision = deterministic(reason, attempts=result.attempts, payload=payload)
            decision.raw["compaction"] = compaction_metadata
            return decision
        reviews_by_arm = {}
        for raw in result.value.get("reviews") or []:
            row = dict(raw); arm_id = str(row["arm_id"])
            reviews_by_arm[arm_id] = {
                "arm_id": arm_id, "recommendation": str(row["recommendation"]),
                "accepted_evidence_ids": [str(v) for v in row.get("accepted_evidence_ids") or []],
                "counterevidence_ids": [str(v) for v in row.get("counterevidence_ids") or []],
                "risk_codes": [str(v) for v in row.get("risk_codes") or []],
                "reason": str(row.get("reason") or ""),
            }
        for arm_id in set(all_arms) - complete_arms:
            reviews_by_arm[arm_id] = {
                "arm_id": arm_id, "recommendation": "insufficient_evidence",
                "accepted_evidence_ids": [], "counterevidence_ids": [],
                "risk_codes": ["no_direct_complete_closed_evidence"],
                "reason": "No direct, complete, closed evidence for this blocked arm.",
            }
        reviews = [reviews_by_arm[arm_id] for arm_id in all_arms]
        return BlockedArmReviewDecision(round_id, reviews, True, {
            "source": "validated_llm_block_review", "fallback_reason": None,
            "context_digest": context_digest(payload), "compaction": compaction_metadata,
            "llm_repaired": result.repaired, "llm_attempts": result.attempts,
        })

