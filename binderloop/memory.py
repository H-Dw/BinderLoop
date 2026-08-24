
import json
import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from binderloop.communication import AgentMessage, compact_messages
from binderloop.models.base import DesignJob
from binderloop.strategy_governance import BindingSiteResolution, LengthPolicyState
from binderloop.resume import atomic_write_json


@dataclass
class RoundRecord:
    round_id: int
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    submissions: List[Dict[str, Any]] = field(default_factory=list)
    monitor_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    ingestion: List[Dict[str, Any]] = field(default_factory=list)
    evaluation: Optional[Dict[str, Any]] = None
    structural_analysis: List[Dict[str, Any]] = field(default_factory=list)
    active_learning_examples: Optional[Dict[str, Any]] = None
    quality_analysis: Optional[Dict[str, Any]] = None
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    retry_events: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    reward: Optional[float] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    rollback_decision: Optional[Dict[str, Any]] = None
    arm_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    arm_evidence_cards: Optional[Dict[str, Any]] = None
    arm_history_resolution: Optional[Dict[str, Any]] = None
    arm_comparison: Optional[Dict[str, Any]] = None
    final_strategy_decision: Optional[Dict[str, Any]] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


@dataclass
class MemoryItem:
    """Normalized, retrievable evidence card derived from one or more rounds."""

    item_id: str
    round_id: int
    item_type: str = "round_outcome"
    target: Dict[str, Any] = field(default_factory=dict)
    target_key: str = ""
    failure_tags: List[str] = field(default_factory=list)
    parameter_diff: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    arm: str = ""  # legacy storage only; omitted from aggregate LLM evidence
    reward: Optional[float] = None
    reward_delta: Optional[float] = None
    performance: Dict[str, Any] = field(default_factory=dict)
    execution_failed: bool = False
    summary: str = ""
    source_round_ids: List[int] = field(default_factory=list)
    source_item_ids: List[str] = field(default_factory=list)
    artifact_refs: List[str] = field(default_factory=list)
    compression_level: int = 0
    archived: bool = False
    compressed_into: Optional[str] = None
    created_at: float = field(default_factory=time.time)


LEDGER_SCHEMA_VERSION = "2.0"


@dataclass
class ArmOutcome:
    arm_id: str
    config_digest: str = ""
    intervention_digest: str = ""
    successes: int = 0
    trials: int = 0
    posterior_alpha: float = 1.0
    posterior_beta: float = 1.0
    wilson_low: float = 0.0
    wilson_high: float = 1.0
    uses: int = 0
    last_round_id: Optional[int] = None

    def observe(self, successes: int, trials: int, round_id: int) -> None:
        successes = max(0, min(int(successes), int(trials)))
        trials = max(0, int(trials))
        self.successes += successes
        self.trials += trials
        self.posterior_alpha = 1.0 + self.successes
        self.posterior_beta = 1.0 + self.trials - self.successes
        self.wilson_low, self.wilson_high = wilson_interval(self.successes, self.trials)
        self.uses += 1
        self.last_round_id = int(round_id)


@dataclass
class BranchState:
    branch_id: str
    parent_branch_id: Optional[str] = None
    parent_round_id: Optional[int] = None
    status: str = "probe"  # probe | promoted | cooldown | retired
    arm_ids: List[str] = field(default_factory=list)
    config_digest: str = ""
    intervention_digest: str = ""
    is_baseline: bool = False
    created_round_id: int = 0
    last_round_id: Optional[int] = None
    cooldown_until_round: Optional[int] = None
    outcome_round_ids: List[int] = field(default_factory=list)


@dataclass
class LedgerRound:
    round_id: int
    outcome: Dict[str, Any] = field(default_factory=dict)
    policy_snapshot: Dict[str, Any] = field(default_factory=dict)
    current_vs_best_diff: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failed_arms: List[str] = field(default_factory=list)
    candidate_denominators: Dict[str, int] = field(default_factory=dict)
    next_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    per_arm_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


@dataclass
class ExperimentLedger:
    schema_version: str = LEDGER_SCHEMA_VERSION
    best_round_id: Optional[int] = None
    best_reward: Optional[float] = None
    best_round_rank_key: List[float] = field(default_factory=list)
    best_policy_snapshot: Dict[str, Any] = field(default_factory=dict)
    rounds: List[LedgerRound] = field(default_factory=list)
    arm_outcomes: Dict[str, ArmOutcome] = field(default_factory=dict)
    branches: Dict[str, BranchState] = field(default_factory=dict)
    intervention_cooldowns: Dict[str, int] = field(default_factory=dict)
    best_config_retests: Dict[str, int] = field(default_factory=dict)
    blocked_arm_combinations: List[Dict[str, Any]] = field(default_factory=list)
    arm_blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    arm_unfreeze_audit: List[Dict[str, Any]] = field(default_factory=list)

    def upsert_round(self, entry: LedgerRound) -> LedgerRound:
        self.rounds = [row for row in self.rounds if row.round_id != entry.round_id]
        self.rounds.append(entry)
        self.rounds.sort(key=lambda row: row.round_id)
        return entry


@dataclass
class ExperimentMemory:
    experiment_id: str
    target: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    rounds: List[RoundRecord] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    template_library: List[Dict[str, Any]] = field(default_factory=list)
    round_metrics: List[Dict[str, Any]] = field(default_factory=list)
    memory_schema_version: str = "2.1"
    memory_items: List[MemoryItem] = field(default_factory=list)
    quality_collaboration_state: Dict[str, Any] = field(default_factory=dict)
    experiment_ledger: ExperimentLedger = field(default_factory=ExperimentLedger)
    length_policy_state: LengthPolicyState = field(default_factory=LengthPolicyState)
    binding_site_resolution: BindingSiteResolution = field(default_factory=BindingSiteResolution)


class ExperimentMemoryStore:
    """Durable cross-round trajectory store for downstream Agents."""

    def __init__(self, root: Union[str, Path], experiment_id: str = "binder_experiment"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.memory_path = self.root / "experiment_memory.json"
        self.events_path = self.root / "events.jsonl"

    def load(self, target: Optional[Dict[str, Any]] = None) -> ExperimentMemory:
        if self.memory_path.exists():
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            data["rounds"] = [_safe_dataclass(RoundRecord, r) for r in data.get("rounds", [])]
            data["memory_items"] = [
                item if isinstance(item, MemoryItem) else _safe_dataclass(MemoryItem, item)
                for item in data.get("memory_items", [])
            ]
            raw_ledger = data.get("experiment_ledger") or data.get("ledger") or {}
            ledger = _safe_dataclass(ExperimentLedger, raw_ledger)
            ledger.schema_version = LEDGER_SCHEMA_VERSION
            ledger.rounds = [
                row if isinstance(row, LedgerRound) else _safe_dataclass(LedgerRound, row)
                for row in (raw_ledger.get("rounds", []) if isinstance(raw_ledger, Mapping) else [])
            ]
            ledger.arm_outcomes = {
                str(key): value if isinstance(value, ArmOutcome) else _safe_dataclass(ArmOutcome, value)
                for key, value in dict(raw_ledger.get("arm_outcomes") or {}).items()
            } if isinstance(raw_ledger, Mapping) else {}
            ledger.branches = {
                str(key): value if isinstance(value, BranchState) else _safe_dataclass(BranchState, value)
                for key, value in dict(raw_ledger.get("branches") or {}).items()
            } if isinstance(raw_ledger, Mapping) else {}
            raw_length = raw_ledger.get("length_policy_state") or {} if isinstance(raw_ledger, Mapping) else {}
            ledger.length_policy_state = _safe_dataclass(LengthPolicyState, raw_length)
            ledger.length_policy_state.outcomes = {
                str(key): value if isinstance(value, ArmOutcome) else _safe_dataclass(ArmOutcome, value)
                for key, value in dict(raw_length.get("outcomes") or {}).items()
            } if isinstance(raw_length, Mapping) else {}
            data["experiment_ledger"] = ledger
            data["length_policy_state"] = _safe_dataclass(LengthPolicyState, data.get("length_policy_state") or {})
            data["binding_site_resolution"] = _safe_dataclass(BindingSiteResolution, data.get("binding_site_resolution") or {})
            data.pop("ledger", None)
            allowed = set(ExperimentMemory.__dataclass_fields__)
            return ExperimentMemory(**{key: value for key, value in data.items() if key in allowed})
        return ExperimentMemory(experiment_id=self.experiment_id, target=target or {})

    def save(self, memory: ExperimentMemory) -> Path:
        memory.updated_at = time.time()
        return atomic_write_json(self.memory_path, asdict(memory))

    def append_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), "type": event_type, "payload": payload}, ensure_ascii=False) + "\n")

    def upsert_round(self, memory: ExperimentMemory, round_id: int) -> RoundRecord:
        for record in memory.rounds:
            if record.round_id == round_id:
                return record
        record = RoundRecord(round_id=round_id)
        memory.rounds.append(record)
        memory.rounds.sort(key=lambda r: r.round_id)
        return record

    def upsert_ledger_round(
        self, memory: ExperimentMemory, *, round_id: int, outcome: Mapping[str, Any],
        policy_snapshot: Mapping[str, Any], failed_arms: Sequence[str] = (),
        candidate_denominators: Optional[Mapping[str, int]] = None,
        next_hypotheses: Sequence[Mapping[str, Any]] = (),
    ) -> LedgerRound:
        """Idempotently record one structured round and update the best snapshot."""
        ledger = memory.experiment_ledger
        entry = LedgerRound(
            round_id=int(round_id), outcome=dict(outcome), policy_snapshot=dict(policy_snapshot),
            current_vs_best_diff=parameter_diff(ledger.best_policy_snapshot, policy_snapshot),
            failed_arms=sorted({str(arm) for arm in failed_arms if str(arm)}),
            candidate_denominators={str(k): int(v or 0) for k, v in dict(candidate_denominators or {}).items()},
            next_hypotheses=[dict(value) for value in next_hypotheses],
        )
        ledger.upsert_round(entry)
        self._recompute_ledger_best(ledger)
        self.append_event("ledger_round_upserted", {"round_id": round_id, "best_round_id": ledger.best_round_id})
        return entry


    @staticmethod
    def _recompute_ledger_best(ledger: ExperimentLedger) -> None:
        """Rebuild best state after every idempotent insert or correction."""
        valid = [row for row in ledger.rounds if not bool((row.outcome or {}).get("execution_failed"))]
        ranked = [row for row in valid if _rank_key((row.outcome or {}).get("round_rank_key"))]
        if ranked:
            best = max(ranked, key=lambda row: _rank_key((row.outcome or {}).get("round_rank_key")))
        else:
            rewarded = [row for row in valid if _optional_float((row.outcome or {}).get("reward")) is not None]
            best = max(rewarded, key=lambda row: float((row.outcome or {}).get("reward"))) if rewarded else None
        if best is None:
            ledger.best_round_id = None; ledger.best_reward = None
            ledger.best_round_rank_key = []; ledger.best_policy_snapshot = {}
            return
        ledger.best_round_id = int(best.round_id)
        ledger.best_reward = _optional_float((best.outcome or {}).get("reward"))
        ledger.best_round_rank_key = list(_rank_key((best.outcome or {}).get("round_rank_key")))
        ledger.best_policy_snapshot = dict(best.policy_snapshot or {})

    def record_governance_outcome(
        self, memory: ExperimentMemory, *, round_id: int, branch_id: str, arm_id: str,
        successes: int, trials: int, config_digest: str = "", intervention_digest: str = "",
        parent_branch_id: Optional[str] = None, parent_round_id: Optional[int] = None,
        is_baseline: bool = False, cooldown_rounds: int = 2, regressed: bool = False,
    ) -> ArmOutcome:
        """Idempotently upsert one arm outcome without replacing sibling arms."""
        ledger = memory.experiment_ledger
        branch_id = str(branch_id or "baseline")
        arm_id = str(arm_id or "baseline_hold")
        is_baseline = bool(is_baseline or arm_id in {"baseline", "baseline_hold"})
        row = next((r for r in ledger.rounds if r.round_id == int(round_id)), None)
        if row is None:
            row = LedgerRound(round_id=int(round_id)); ledger.upsert_round(row)
        arm_row = {
            "arm_id": arm_id, "branch_id": branch_id, "strict_successes": int(successes),
            "strict_trials": int(trials), "config_digest": str(config_digest),
            "intervention_digest": str(intervention_digest), "is_baseline": is_baseline,
            "regressed": bool(regressed),
        }
        row.per_arm_outcomes = [item for item in row.per_arm_outcomes if not (str(item.get("arm_id")) == arm_id and str(item.get("branch_id")) == branch_id)]
        row.per_arm_outcomes.append(arm_row)
        row.per_arm_outcomes.sort(key=lambda item: (str(item.get("arm_id")), str(item.get("branch_id"))))
        row.outcome["per_arm_outcomes"] = [dict(item) for item in row.per_arm_outcomes]
        branch = ledger.branches.get(branch_id) or BranchState(
            branch_id=branch_id, parent_branch_id=parent_branch_id, parent_round_id=parent_round_id,
            config_digest=config_digest, intervention_digest=intervention_digest,
            is_baseline=is_baseline, created_round_id=int(round_id),
        )
        branch.last_round_id = int(round_id); branch.is_baseline = is_baseline
        branch.outcome_round_ids = sorted(set(branch.outcome_round_ids + [int(round_id)]))
        if arm_id not in branch.arm_ids: branch.arm_ids.append(arm_id)
        if regressed and not branch.is_baseline:
            branch.status = "cooldown"; branch.cooldown_until_round = int(round_id) + max(1, int(cooldown_rounds))
            if intervention_digest: ledger.intervention_cooldowns[intervention_digest] = branch.cooldown_until_round
            self.record_arm_block(memory, arm_id=arm_id, round_id=round_id, reason="per_arm_regression", cooldown_until_round=branch.cooldown_until_round, intervention_digest=intervention_digest)
        elif branch.status == "probe" and int(trials) > 0: branch.status = "promoted"
        ledger.branches[branch_id] = branch
        ledger.arm_outcomes = {}
        for evidence in ledger.rounds:
            outcomes = list(evidence.per_arm_outcomes or (evidence.outcome or {}).get("per_arm_outcomes") or [])
            # Legacy single-arm records remain readable.
            if not outcomes and (evidence.outcome or {}).get("arm_id"): outcomes = [dict(evidence.outcome)]
            for item in outcomes:
                aid = str(item.get("arm_id") or "")
                if not aid: continue
                accumulated = ledger.arm_outcomes.get(aid) or ArmOutcome(
                    arm_id=aid, config_digest=str(item.get("config_digest") or ""),
                    intervention_digest=str(item.get("intervention_digest") or ""),
                )
                accumulated.observe(int(item.get("strict_successes") or 0), int(item.get("strict_trials") or 0), evidence.round_id)
                ledger.arm_outcomes[aid] = accumulated
        return ledger.arm_outcomes.get(arm_id) or ArmOutcome(arm_id=arm_id)


    @staticmethod
    def record_blocked_combination(memory: ExperimentMemory, *, round_id: int, arm_ids: Sequence[str], reason: str, intervention_digest: str = "") -> None:
        normalized = sorted({str(value) for value in arm_ids if str(value)})
        if not normalized:
            return
        ledger = memory.experiment_ledger
        record = {"arm_ids": normalized, "source_round_id": int(round_id), "reason": str(reason), "intervention_digest": str(intervention_digest), "active": True}
        ledger.blocked_arm_combinations = [item for item in ledger.blocked_arm_combinations if list(item.get("arm_ids") or []) != normalized]
        ledger.blocked_arm_combinations.append(record)

    @staticmethod
    def active_blocked_combinations(memory: ExperimentMemory) -> List[List[str]]:
        return [list(item.get("arm_ids") or []) for item in memory.experiment_ledger.blocked_arm_combinations if bool(item.get("active", True))]

    @staticmethod
    def record_arm_block(memory: ExperimentMemory, *, arm_id: str, round_id: int, reason: str, cooldown_until_round: int, intervention_digest: str = "") -> None:
        memory.experiment_ledger.arm_blocks[str(arm_id)] = {"arm_id": str(arm_id), "source_round_id": int(round_id), "reason": str(reason), "cooldown_until_round": int(cooldown_until_round), "intervention_digest": str(intervention_digest), "status": "soft_blocked"}

    @staticmethod
    def soft_blocked_arms(memory: ExperimentMemory, round_id: int) -> List[str]:
        result = []
        for arm_id, state in memory.experiment_ledger.arm_blocks.items():
            if str(state.get("status")) != "soft_blocked":
                continue
            if int(state.get("cooldown_until_round") or 0) > int(round_id):
                result.append(str(arm_id))
            else:
                result.append(str(arm_id))
        return sorted(set(result))

    @staticmethod
    def apply_arm_unfreeze(memory: ExperimentMemory, *, arm_id: str, round_id: int, evidence_ids: Sequence[str], reason: str) -> None:
        state = dict(memory.experiment_ledger.arm_blocks.get(str(arm_id)) or {})
        state["status"] = "unfrozen"; state["unfrozen_round_id"] = int(round_id)
        memory.experiment_ledger.arm_blocks[str(arm_id)] = state
        memory.experiment_ledger.arm_unfreeze_audit.append({"arm_id": str(arm_id), "round_id": int(round_id), "evidence_ids": [str(v) for v in evidence_ids], "reason": str(reason)})

    @staticmethod
    def blocked_interventions(memory: ExperimentMemory, round_id: int) -> List[str]:
        return sorted(digest for digest, until in memory.experiment_ledger.intervention_cooldowns.items() if digest and int(until) > int(round_id))

    @staticmethod
    def uncertainty_overlaps(left: ArmOutcome, right: ArmOutcome) -> bool:
        return left.wilson_low <= right.wilson_high and right.wilson_low <= left.wilson_high

    @staticmethod
    def ledger_prompt_view(memory: ExperimentMemory, max_rounds: int = 4) -> Dict[str, Any]:
        ledger = memory.experiment_ledger
        rows = ledger.rounds[-max(1, int(max_rounds)):]
        return {
            "schema_version": ledger.schema_version,
            "best_round_id": ledger.best_round_id,
            "best_reward": ledger.best_reward,
            "best_round_rank_key": list(ledger.best_round_rank_key),
            "best_policy_snapshot": ledger.best_policy_snapshot,
            "recent_rounds": [asdict(row) for row in rows],
        }

    def record_jobs(self, memory: ExperimentMemory, round_id: int, jobs: List[DesignJob], *, extend_memory: bool = False) -> None:
        record = self.upsert_round(memory, round_id)
        record.jobs = [asdict(job) for job in jobs]
        self.append_event("jobs", {"round_id": round_id, "jobs": record.jobs, "extend_memory": bool(extend_memory)})

    def record_message_bus(self, memory: ExperimentMemory, messages: List[AgentMessage], max_items: int = 200) -> None:
        memory.messages = compact_messages(messages, max_items=max_items)
        self.append_event("messages_compacted", {"count": len(memory.messages)})

    def upsert_memory_item(self, memory: ExperimentMemory, item: MemoryItem) -> MemoryItem:
        """Idempotently insert or replace a normalized evidence card."""
        for index, existing in enumerate(memory.memory_items):
            if existing.item_id == item.item_id:
                memory.memory_items[index] = item
                return item
        memory.memory_items.append(item)
        memory.memory_items.sort(key=lambda value: (value.round_id, value.created_at, value.item_id))
        self.append_event("memory_item_upserted", {
            "item_id": item.item_id,
            "round_id": item.round_id,
            "item_type": item.item_type,
        })
        return item

    def summarize_for_agent(
        self,
        memory: ExperimentMemory,
        max_rounds: int = 5,
        *,
        extend_memory: bool = False,
        recalled_items: Optional[Sequence[Union[MemoryItem, Mapping[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        recent_rounds = [asdict(r) for r in memory.rounds[-max_rounds:]]
        if extend_memory:
            for record in recent_rounds:
                for job in record.get("jobs") or []:
                    job["params_summary"] = _job_params_summary(job)
        summary = {
            "experiment_id": memory.experiment_id,
            "target": memory.target,
            "round_count": len(memory.rounds),
            "extend_memory": bool(extend_memory),
            "recent_rounds": recent_rounds,
            "recent_messages": memory.messages[-50:],
            "experiment_ledger": self.ledger_prompt_view(memory, max_rounds=max_rounds),
            "length_policy_state": asdict(memory.length_policy_state),
            "binding_site_resolution": asdict(memory.binding_site_resolution),
        }
        if recalled_items:
            summary["recalled_items"] = [
                asdict(item) if isinstance(item, MemoryItem) else dict(item)
                for item in recalled_items
            ]
        return summary


def target_memory_key(target: Mapping[str, Any]) -> str:
    """Stable target identity resilient to absolute output-path differences."""
    structure = Path(str(target.get("structure_path") or target.get("target_structure") or "")).name
    chain = str(target.get("chain_id") or "")
    hotspots = sorted(str(value) for value in (target.get("hotspots") or []))
    raw = json.dumps(
        {"structure": structure, "chain": chain, "hotspots": hotspots},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parameter_diff(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    allowed_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return an auditable before/after diff for executable parameters."""
    keys = set(allowed_keys or (set(previous) | set(current)))
    changes: Dict[str, Dict[str, Any]] = {}
    for key in sorted(keys):
        old = previous.get(key)
        new = current.get(key)
        if old == new:
            continue
        row: Dict[str, Any] = {"before": old, "after": new}
        if isinstance(old, (int, float)) and not isinstance(old, bool) and isinstance(new, (int, float)) and not isinstance(new, bool):
            row["delta"] = round(float(new) - float(old), 9)
        changes[str(key)] = row
    return changes


def build_round_memory_item(
    *,
    round_id: int,
    target: Mapping[str, Any],
    failure_tags: Sequence[str],
    config_diff: Mapping[str, Mapping[str, Any]],
    arm: str,
    outcome: Mapping[str, Any],
    artifact_refs: Optional[Sequence[str]] = None,
    previous_reward: Optional[float] = None,
) -> MemoryItem:
    """Create a deterministic round evidence card for resume-safe upserts."""
    reward = _optional_float(outcome.get("reward"))
    reward_delta = None if reward is None or previous_reward is None else round(reward - float(previous_reward), 6)
    performance = {
        key: outcome.get(key)
        for key in (
            "best_iptm",
            "median_iptm",
            "core_objective",
            "round_rank_key",
            "core_metric_stats",
            "success_count",
        )
        if outcome.get(key) is not None
    }
    execution_failed = bool(outcome.get("execution_failed"))
    if execution_failed:
        performance["execution_failure_reason"] = outcome.get("execution_failure_reason")
    normalized_tags = sorted({str(tag) for tag in failure_tags if str(tag)})
    target_dict = dict(target)
    identity_payload = {
        "round_id": int(round_id),
        "target_key": target_memory_key(target_dict),
        "item_type": "round_outcome",
    }
    item_id = "mem_" + hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    changed = ", ".join(sorted(config_diff)) or "no executable parameter change"
    tag_text = ", ".join(normalized_tags) or "no dominant failure tag"
    if execution_failed:
        summary = (
            f"Round {round_id} infrastructure/configuration failure "
            f"({outcome.get('execution_failure_reason') or 'unknown'}); {changed}."
        )
    else:
        summary = (
            f"Round {round_id}: reward={reward if reward is not None else 'n/a'}, "
            f"reward_delta={reward_delta if reward_delta is not None else 'n/a'}, "
            f"failures={tag_text}, changes={changed}."
        )
    return MemoryItem(
        item_id=item_id,
        round_id=int(round_id),
        target=target_dict,
        target_key=identity_payload["target_key"],
        failure_tags=normalized_tags,
        parameter_diff={str(key): dict(value) for key, value in config_diff.items()},
        arm=str(arm or ""),
        reward=reward,
        reward_delta=reward_delta,
        performance=performance,
        execution_failed=execution_failed,
        summary=summary,
        source_round_ids=[int(round_id)],
        artifact_refs=[str(path) for path in (artifact_refs or [])],
    )


def _job_params_summary(job: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(job.get("params") or {})
    shard = dict(params.get("multi_taiji_host_shard") or {})
    keys = (
        "binder_lengths",
        "hotspot_weight",
        "prioritize_hotspots",
        "auxiliary_hotspots",
        "template_conditioned",
        "template_free_exploration",
        "diffusion_batch_size",
        "alpha",
        "noise_scale",
        "step_scale",
        "num_designs",
    )
    summary = {key: params.get(key) for key in keys if key in params and params.get(key) is not None}
    if shard.get("source_num_designs"):
        summary["num_designs"] = shard.get("source_num_designs")
    return summary


def _optional_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _rank_key(value: Any) -> tuple:
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return ()


def _safe_dataclass(cls: Any, value: Any) -> Any:
    """Load older/newer persisted records while ignoring unknown fields."""
    if isinstance(value, cls):
        return value
    raw = dict(value or {}) if isinstance(value, Mapping) else {}
    allowed = set(cls.__dataclass_fields__)
    return cls(**{key: val for key, val in raw.items() if key in allowed})


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple:
    trials = max(0, int(trials))
    successes = max(0, min(int(successes), trials))
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    margin = z * ((p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)
