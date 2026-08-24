"""Quality-aware backtracking for the closed-loop binder design search.

The orchestrator historically advanced strictly linearly: round ``N+1`` was
always seeded from round ``N``. When a round regressed (the observed
``sc2rbd_closed_loop_llm_5r_v2`` run peaked at round 2 with iPTM 0.573 then
collapsed to 0.302 and 0.177), the search kept riding the decline because there
was no way to discard the bad branch and return to the best round.

``RollbackController`` tracks a per-round reward and decides, before the next
round is prepared, whether to:

* ``advance``  – the current round is the best (or within tolerance); branch from it.
* ``replay_best`` – the current round crossed the configured recovery trigger;
                    discard its proposals and exactly replay the best round's
                    configuration and logical jobs.
* ``stop``     – optional early stop after sustained regression with no headroom.
"""


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RoundOutcome:
    round_id: int
    reward: float
    best_iptm: float = 0.0
    median_iptm: float = 0.0
    core_objective: float = 0.0
    core_metric_stats: Dict[str, float] = field(default_factory=dict)
    round_rank_key: List[float] = field(default_factory=list)
    success_count: int = 0
    arm_signature: str = ""
    # True when the round produced no usable candidates because of an
    # infrastructure / configuration failure (e.g. ``boltzgen_config_error``,
    # ``missing_ceph_mount_secret``) rather than genuine design-quality
    # regression.  Such rounds must NOT be treated as quality regressions: they
    # are excluded from the reward history and never trigger a rollback.
    execution_failed: bool = False
    execution_failure_reason: str = ""
    branch_id: str = ""
    config_digest: str = ""
    intervention_digest: str = ""
    is_baseline: bool = False
    strict_successes: int = 0
    strict_trials: int = 0
    raw_candidate_count: int = 0
    analysis_candidate_count: int = 0
    raw_strict_yield: float = 0.0
    conditional_strict_yield: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "round_id": self.round_id,
            "reward": round(float(self.reward), 6),
            "best_iptm": round(float(self.best_iptm), 6),
            "median_iptm": round(float(self.median_iptm), 6),
            "core_objective": round(float(self.core_objective), 6),
            "core_metric_stats": dict(self.core_metric_stats or {}),
            "round_rank_key": [round(float(value), 6) for value in self.round_rank_key],
            "success_count": int(self.success_count),
            "arm_signature": self.arm_signature,
            "execution_failed": bool(self.execution_failed),
            "execution_failure_reason": self.execution_failure_reason,
            "branch_id": self.branch_id,
            "config_digest": self.config_digest,
            "intervention_digest": self.intervention_digest,
            "is_baseline": bool(self.is_baseline),
            "strict_successes": int(self.strict_successes),
            "strict_trials": int(self.strict_trials),
            "raw_candidate_count": int(self.raw_candidate_count),
            "analysis_candidate_count": int(self.analysis_candidate_count),
            "raw_strict_yield": round(float(self.raw_strict_yield), 6),
            "conditional_strict_yield": round(float(self.conditional_strict_yield), 6),
        }


@dataclass
class RollbackDecision:
    action: str  # "advance" | "retest_best_config" | "branch_from_best" | "stop"
    branch_from_round: int
    best_round: int
    best_reward: float
    current_reward: float
    is_regression: bool
    consecutive_regressions: int
    relative_drop: float
    rationale: str
    blocked_arm_signature: Optional[str] = None
    best_round_rank_key: List[float] = field(default_factory=list)
    current_round_rank_key: List[float] = field(default_factory=list)
    blocked_intervention_digest: Optional[str] = None
    retest_number: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "branch_from_round": self.branch_from_round,
            "best_round": self.best_round,
            "best_reward": round(float(self.best_reward), 6),
            "current_reward": round(float(self.current_reward), 6),
            "is_regression": self.is_regression,
            "consecutive_regressions": self.consecutive_regressions,
            "relative_drop": round(float(self.relative_drop), 6),
            "rationale": self.rationale,
            "blocked_arm_signature": self.blocked_arm_signature,
            "best_round_rank_key": list(self.best_round_rank_key),
            "current_round_rank_key": list(self.current_round_rank_key),
            "blocked_intervention_digest": self.blocked_intervention_digest,
            "retest_number": int(self.retest_number),
        }


class RollbackController:
    """Decide whether to advance from the current round or roll back to the best."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        regression_tolerance: float = 0.25,
        patience: int = 2,
        min_round_for_rollback: int = 1,
        stop_after_regressions: int = 0,
        max_best_config_retests: int = 1,
    ) -> None:
        self.enabled = enabled
        # Relative reward drop (vs best so far) that counts as a regression.
        self.regression_tolerance = max(0.0, float(regression_tolerance))
        # How many consecutive regressed rounds to tolerate before rolling back.
        self.patience = max(1, int(patience))
        self.min_round_for_rollback = max(0, int(min_round_for_rollback))
        # 0 disables early stop; otherwise stop after this many consecutive regressions.
        self.stop_after_regressions = max(0, int(stop_after_regressions))
        self.max_best_config_retests = max(0, int(max_best_config_retests))
        self.history: List[RoundOutcome] = []
        self.best_config_retests: Dict[str, int] = {}

    def seed_history(self, outcomes: List[RoundOutcome], *, best_config_retests: Optional[Dict[str, int]] = None) -> None:
        self.history = list(sorted(outcomes, key=lambda o: o.round_id))
        self.best_config_retests = {str(k): max(0, int(v)) for k, v in dict(best_config_retests or {}).items()}

    def _best_outcome(self) -> Optional[RoundOutcome]:
        if not self.history:
            return None
        ranked = [outcome for outcome in self.history if outcome.round_rank_key]
        # Do not compare incompatible scalar and tuple schemas. Once a new
        # RoundRankKey record exists, legacy reward-only records are historical
        # display/resume context rather than selection candidates.
        return max(ranked or self.history, key=self._decision_key)

    @staticmethod
    def _decision_key(outcome: RoundOutcome) -> Tuple[float, ...]:
        if outcome.round_rank_key:
            return tuple(float(value) for value in outcome.round_rank_key)
        return (float(outcome.reward),)

    @staticmethod
    def _significant_rank_drop(current: List[float], best: List[float]) -> bool:
        """Apply noise tolerances in the same order as RoundRankKey."""
        if len(current) < 5 or len(best) < 5:
            return False
        tolerances = (0.0, 0.10, 0.02, 0.50, 0.25)
        for index, tolerance in enumerate(tolerances):
            delta = float(current[index]) - float(best[index])
            if abs(delta) <= tolerance:
                continue
            return delta < 0.0
        return False

    def _consecutive_regressions(self, best_round: int) -> int:
        """Count the true trailing streak of significant quality regressions."""
        best = next((row for row in self.history if row.round_id == best_round), None)
        if best is None:
            return 0
        streak = 0
        for row in reversed(self.history):
            if row.round_id <= best_round:
                break
            if row.round_rank_key and best.round_rank_key:
                regressed = self._significant_rank_drop(row.round_rank_key, best.round_rank_key)
            elif best.reward <= 1e-9:
                regressed = row.reward < best.reward
            else:
                regressed = ((best.reward - row.reward) / best.reward) > self.regression_tolerance
            if not regressed:
                break
            streak += 1
        return streak

    @staticmethod
    def _best_config_key(best: RoundOutcome) -> str:
        return best.config_digest or f"round:{best.round_id}"

    def observe(self, outcome: RoundOutcome) -> RollbackDecision:
        """Record an outcome and return the decision for seeding the next round."""
        # ------------------------------------------------------------------
        # Execution / configuration failure: NOT a quality regression.
        # A round that produced zero candidates because of an infrastructure
        # or config error (boltzgen_config_error, missing_ceph_mount_secret,
        # ...) carries no signal about design quality.  We must not append it
        # to the reward history (it would poison the best-reward baseline and
        # the consecutive-regression counter), and we must not let it trigger a
        # quality-driven rollback. The orchestrator will retry the same
        # normalized job/config rather than switch active-learning branches.
        # ------------------------------------------------------------------
        if outcome.execution_failed:
            best = self._best_outcome()
            best_round = best.round_id if best is not None else outcome.round_id
            best_reward = best.reward if best is not None else 0.0
            return RollbackDecision(
                action="advance",
                branch_from_round=outcome.round_id,
                best_round=best_round,
                best_reward=best_reward,
                current_reward=0.0,
                is_regression=False,
                consecutive_regressions=self._consecutive_regressions(best_round) if best is not None else 0,
                relative_drop=0.0,
                rationale=(
                    f"Round {outcome.round_id} failed to execute "
                    f"({outcome.execution_failure_reason or 'execution/config error'}); "
                    "this is an infrastructure failure, not a quality regression, so it is "
                    "excluded from reward/rollback accounting. Retrying the same normalized "
                    "job/config rather than generating a new active-learning branch."
                ),
                blocked_arm_signature=None,
            )

        # Replace any existing entry for this round (resume/idempotency).
        self.history = [o for o in self.history if o.round_id != outcome.round_id]
        self.history.append(outcome)
        self.history.sort(key=lambda o: o.round_id)

        best = self._best_outcome()
        assert best is not None
        best_reward = best.reward
        current_reward = outcome.reward
        best_rank = list(best.round_rank_key or [])
        current_rank = list(outcome.round_rank_key or [])
        relative_drop = 0.0
        if best_reward > 1e-9 and current_reward < best_reward:
            relative_drop = (best_reward - current_reward) / best_reward

        is_best = outcome.round_id == best.round_id
        is_regression = (not is_best) and (
            self._significant_rank_drop(current_rank, best_rank)
            if current_rank and best_rank
            else relative_drop > self.regression_tolerance
        )
        consecutive = self._consecutive_regressions(best.round_id)

        if not self.enabled:
            return RollbackDecision(
                action="advance",
                branch_from_round=outcome.round_id,
                best_round=best.round_id,
                best_reward=best_reward,
                current_reward=current_reward,
                is_regression=is_regression,
                consecutive_regressions=consecutive,
                relative_drop=relative_drop,
                rationale="Backtracking disabled; advancing linearly.",
            )

        if is_best:
            return RollbackDecision(
                action="advance",
                branch_from_round=outcome.round_id,
                best_round=best.round_id,
                best_reward=best_reward,
                current_reward=current_reward,
                is_regression=False,
                consecutive_regressions=0,
                relative_drop=0.0,
                rationale=f"Round {outcome.round_id} is best by round_rank_key={current_rank or 'legacy reward fallback'}; advancing from it.",
                best_round_rank_key=best_rank,
                current_round_rank_key=current_rank,
            )

        significant_drop = is_regression
        patience_exhausted = consecutive >= self.patience
        # Quality regressions use one comparison predicate and respect patience.
        # Execution/configuration failures are handled above and never enter this path.
        legacy_fraction_rank = bool(current_rank and best_rank and max(float(current_rank[0]), float(best_rank[0])) <= 1.0)
        legacy_immediate_drop = significant_drop and (not (current_rank and best_rank) or legacy_fraction_rank)
        if outcome.round_id >= self.min_round_for_rollback and (patience_exhausted or legacy_immediate_drop):
            if self.stop_after_regressions and consecutive >= self.stop_after_regressions:
                return RollbackDecision(
                    action="stop", branch_from_round=best.round_id, best_round=best.round_id,
                    best_reward=best_reward, current_reward=current_reward, is_regression=True,
                    consecutive_regressions=consecutive, relative_drop=relative_drop,
                    rationale=f"{consecutive} trailing regressions; stopping.",
                    blocked_arm_signature=outcome.arm_signature or None,
                    blocked_intervention_digest=None if outcome.is_baseline else (outcome.intervention_digest or None),
                )
            key = self._best_config_key(best)
            used = self.best_config_retests.get(key, 0)
            if used < self.max_best_config_retests:
                used += 1
                self.best_config_retests[key] = used
                return RollbackDecision(
                    action="retest_best_config", branch_from_round=best.round_id, best_round=best.round_id,
                    best_reward=best_reward, current_reward=current_reward, is_regression=significant_drop,
                    consecutive_regressions=consecutive, relative_drop=relative_drop,
                    rationale=f"Retesting best config once to separate noise from regression ({used}/{self.max_best_config_retests}).",
                    blocked_arm_signature=outcome.arm_signature or None,
                    best_round_rank_key=best_rank, current_round_rank_key=current_rank,
                    blocked_intervention_digest=None if outcome.is_baseline else (outcome.intervention_digest or None),
                    retest_number=used,
                )
            return RollbackDecision(
                action="branch_from_best", branch_from_round=best.round_id, best_round=best.round_id,
                best_reward=best_reward, current_reward=current_reward, is_regression=significant_drop,
                consecutive_regressions=consecutive, relative_drop=relative_drop,
                rationale="Best-config retest cap exhausted; preserve evidence and create a new branch from the best baseline.",
                blocked_arm_signature=outcome.arm_signature or None,
                best_round_rank_key=best_rank, current_round_rank_key=current_rank,
                blocked_intervention_digest=None if outcome.is_baseline else (outcome.intervention_digest or None),
                retest_number=used,
            )

        return RollbackDecision(
            action="advance",
            branch_from_round=outcome.round_id,
            best_round=best.round_id,
            best_reward=best_reward,
            current_reward=current_reward,
            is_regression=is_regression,
            consecutive_regressions=consecutive,
            relative_drop=relative_drop,
            rationale=(
                f"Round {outcome.round_id} is non-best (drop={relative_drop:.0%}) but within rollback "
                f"patience ({consecutive}/{self.patience}); advancing one more valid round."
            ),
            best_round_rank_key=best_rank,
            current_round_rank_key=current_rank,
        )


def median_of(values: List[float]) -> float:
    """Median of a numeric list; 0.0 when empty."""
    vals = sorted(float(v) for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def round_reward(
    best_iptm: float,
    success_count: int,
    *,
    median_iptm: Optional[float] = None,
    core_objective: Optional[float] = None,
    success_weight: float = 0.0,
    median_weight: float = 0.7,
) -> float:
    """Legacy/monitoring scalar retained for collaboration and old artifacts.

    New rollback decisions use ``RoundOutcome.round_rank_key``.
    """
    if core_objective is not None:
        core = float(core_objective or 0.0)
        return core + float(success_weight) * float(max(0, int(success_count or 0)))
    best = float(best_iptm or 0.0)
    if median_iptm is None:
        core = best
    else:
        mw = float(median_weight)
        core = mw * float(median_iptm or 0.0) + (1.0 - mw) * best
    return core + float(success_weight) * float(max(0, int(success_count or 0)))
