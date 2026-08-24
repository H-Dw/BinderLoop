"""Evidence-bounded ranking for executable strategy arms."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from binderloop.llm import LLMConfigError, LLMTransportError, OpenAICompatibleClient
from binderloop.resume import stable_hash


@dataclass
class StrategyArmRanking:
    round_id: int
    ordered_arm_names: List[str] = field(default_factory=list)
    llm_used: bool = False
    rationale: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompletedArmComparison:
    round_id: int
    status: str
    closed_arm_ids: List[str] = field(default_factory=list)
    winner_arm_id: Optional[str] = None
    endpoint_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    positive_differences: List[str] = field(default_factory=list)
    negative_differences: List[str] = field(default_factory=list)
    confounders: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    update_direction: str = "hold"
    llm_used: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]: return asdict(self)


class StrategyArmRankingAgent:
    """Rank a closed catalog of arms; never invent or mutate arm parameters."""

    SYSTEM = """You rank already-valid protein-binder strategy arms for one controlled round.
Return JSON only:
{"ranked_arms":[{"arm_name":"...","confidence":0.0,"evidence_ids":[],"reason":"..."}]}.
Use only supplied arm names and evidence. Prefer one-factor, testable interventions.
Do not invent parameter values, arm names, measurements, affinity, or causal claims.
Arms carry typed, value-free intent. Parameter values come only from sparse policy deltas and the deterministic resolver.
Strict positives are successes; near misses are boundary evidence; other negatives
support failure-mode prevalence. Prefer hold when evidence is conflicting."""

    def __init__(self, llm: Optional[OpenAICompatibleClient], *, require_llm: bool = False):
        self.llm = llm
        self.require_llm = bool(require_llm)

    @staticmethod
    def deterministic_order(arms: Sequence[Mapping[str, Any]]) -> List[str]:
        ordered = sorted(
            (dict(arm) for arm in arms or []),
            key=lambda arm: (
                float(arm.get("deterministic_priority") or 0.0),
                str(arm.get("name") or ""),
            ),
            reverse=True,
        )
        return [str(arm.get("name")) for arm in ordered if arm.get("name")]

    def compare_completed_arms(self, *, round_id: int, arm_evidence: Sequence[Mapping[str, Any]]) -> CompletedArmComparison:
        rows = [dict(row) for row in arm_evidence if str(row.get("status") or "closed") == "closed" and row.get("arm_id")]
        ids = [str(row["arm_id"]) for row in rows]
        evidence_ids = sorted({str(eid) for row in rows for eid in row.get("evidence_ids", []) or []})
        complete = [row for row in rows if int(row.get("completed_budget", row.get("trials", 0)) or 0) >= int(row.get("requested_budget", row.get("trials", 0)) or 0) and int(row.get("trials", 0) or 0) > 0]
        confounders = sorted({str(x) for row in rows for x in row.get("confounders", []) or []})
        if len(complete) != len(rows): confounders.append("incomplete_execution_or_budget_denominator")
        if len({str(row["arm_id"]) for row in complete}) < 2:
            return CompletedArmComparison(round_id, "insufficient", sorted(set(ids)), evidence_ids=evidence_ids, confounders=sorted(set(confounders)), raw={"source":"deterministic_fallback","validated_evidence_ids":evidence_ids})
        higher = ("strict_yield", "core_objective", "interface_confidence", "design_ptm", "positive_feature_score")
        lower = ("interface_pae", "refold_rmsd", "negative_feature_score")
        def endpoint_score(row):
            e=dict(row.get("endpoints") or {})
            return tuple(float(e.get(key) or 0) for key in higher) + tuple(-float(e.get(key) if e.get(key) is not None else 1e9) for key in lower)
        ordered=sorted(complete, key=lambda row:(endpoint_score(row), str(row["arm_id"])), reverse=True)
        top, second=ordered[0], ordered[1]; top_score, second_score=endpoint_score(top), endpoint_score(second)
        status="tie" if top_score==second_score else "winner"; winner=None if status=="tie" else str(top["arm_id"])
        comparisons=[]; positive=[]; negative=[]
        keys=sorted(set((top.get("endpoints") or {})) | set((second.get("endpoints") or {})))
        for key in keys:
            left=(top.get("endpoints") or {}).get(key); right=(second.get("endpoints") or {}).get(key)
            direction="lower_better" if key in lower else "higher_better"
            comparisons.append({"endpoint":key,"direction":direction,"winner_value":left,"runner_up_value":right})
            if left is not None and right is not None and left != right:
                improved = float(left) < float(right) if direction=="lower_better" else float(left) > float(right)
                (positive if improved else negative).append(f"{key}: {left} vs {right} ({direction})")
        positive.extend(str(x) for x in top.get("positive_features",[]) or [])
        negative.extend(str(x) for x in top.get("negative_features",[]) or [])
        fallback=CompletedArmComparison(round_id,status,sorted(set(ids)),winner,comparisons,positive,negative,sorted(set(confounders)),evidence_ids,
            str(top.get("update_direction") or ("hold" if not winner else "preserve_winner")),False,
            {"source":"deterministic_closed_arm_comparison","validated_evidence_ids":evidence_ids,"budget_denominators":{str(r["arm_id"]):{"requested":r.get("requested_budget"),"completed":r.get("completed_budget"),"trials":r.get("trials")} for r in rows}})
        if not (self.llm and self.llm.available()): return fallback
        payload={"round_id":round_id,"executed_arm_ids":sorted(set(ids)),"validated_evidence_ids":evidence_ids,"deterministic_comparison":fallback.to_dict(),"arm_evidence":complete}
        try:
            result=self.llm.chat_json(system="Compare only supplied executed arms and evidence IDs. Return JSON with status, winner_arm_id, update_direction, evidence_ids, confounders.",user=payload,temperature=.05,max_tokens=1200,thinking="low")
            proposed=str(result.get("winner_arm_id") or "") or None; cited=[str(x) for x in result.get("evidence_ids",[]) or []]
            if proposed not in set(ids) or not set(cited).issubset(evidence_ids): return fallback
            if str(result.get("status")) not in {"winner","tie","insufficient"}: return fallback
            fallback.status=str(result["status"]); fallback.winner_arm_id=proposed if fallback.status=="winner" else None
            fallback.update_direction=str(result.get("update_direction") or fallback.update_direction); fallback.llm_used=True
            fallback.confounders=sorted(set(fallback.confounders+[str(x) for x in result.get("confounders",[]) or []])); fallback.raw={"source":"validated_llm_completed_arm_comparison","context_digest":stable_hash(payload)}
            return fallback
        except Exception: return fallback

    def rank(
        self,
        *,
        round_id: int,
        arms: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> StrategyArmRanking:
        fallback = self.deterministic_order(arms)
        preferred = str((context or {}).get("preferred_arm_id") or "")
        if preferred in fallback: fallback = [preferred] + [name for name in fallback if name != preferred]
        if not fallback:
            return StrategyArmRanking(round_id, [], False, raw={"source": "no_candidate_arms"})
        if not (self.llm and self.llm.available()):
            if self.require_llm:
                raise RuntimeError("StrategyArmRankingAgent requires an available LLM")
            return StrategyArmRanking(
                round_id,
                fallback,
                False,
                raw={"source": "deterministic_fallback", "reason": "llm_unavailable"},
            )
        allowed = set(fallback)
        payload = {
            "round_id": int(round_id),
            "candidate_arms": [
                {
                    "arm_name": arm.get("name"),
                    "expected_effect": arm.get("expected_effect"),
                    "family": arm.get("family"),
                    "branch_role": arm.get("branch_role"),
                    "intervention": arm.get("intervention"),
                    "trigger_evidence": arm.get("trigger_evidence"),
                    "risk": arm.get("risk"),
                    "deterministic_priority": arm.get("deterministic_priority"),
                }
                for arm in arms
            ],
            "evidence": dict(context or {}),
        }
        try:
            result = self.llm.chat_json(
                system=self.SYSTEM,
                user=payload,
                temperature=0.05,
                max_tokens=1600,
                thinking="low",
            )
        except (LLMConfigError, LLMTransportError, Exception) as exc:
            if self.require_llm:
                raise
            return StrategyArmRanking(
                round_id,
                fallback,
                False,
                raw={"source": "deterministic_fallback", "reason": str(exc)},
            )
        ordered: List[str] = []
        rationale: List[Dict[str, Any]] = []
        for raw in result.get("ranked_arms") or []:
            item = dict(raw or {})
            name = str(item.get("arm_name") or "")
            if name not in allowed or name in ordered:
                continue
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence") or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            ordered.append(name)
            rationale.append({
                "arm_name": name,
                "confidence": confidence,
                "evidence_ids": [str(value) for value in item.get("evidence_ids", []) or []][:12],
                "reason": str(item.get("reason") or "")[:500],
            })
        ordered.extend(name for name in fallback if name not in ordered)
        if not rationale:
            return StrategyArmRanking(
                round_id,
                fallback,
                False,
                raw={"source": "deterministic_fallback", "reason": "llm_returned_no_valid_arms"},
            )
        return StrategyArmRanking(
            round_id,
            ordered,
            True,
            rationale,
            raw={"source": "llm_closed_catalog_ranking", "context_digest": stable_hash(payload)},
        )
