"""Declared LLM-wave graph for one closed-loop round.

Inspired by LangGraph StateGraph (typed state, named nodes, parallel waves)
without taking LangGraph as a runtime dependency. Taiji submit / resume stay
in BinderDesignOrchestrator.
"""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Protocol, Tuple, TypedDict

from binderloop.harness.contracts import HarnessEventType


class RoundState(TypedDict, total=False):
    round_id: int
    facts_metric: Dict[str, Any]
    examples_al_clusters: List[Dict[str, Any]]
    candidates_clusters: Dict[str, Any]
    structure_aggregates: Dict[str, Any]
    config_current: Dict[str, Any]
    constraints_hard: Dict[str, Any]
    execution_monitor: Dict[str, Any]
    memory_retrieved: Dict[str, Any]
    quality_analysis: Dict[str, Any]
    hypotheses: Dict[str, Any]
    diagnostic: Dict[str, Any]
    arm_comparison: Dict[str, Any]
    arm_history: Dict[str, Any]
    final_strategy: Dict[str, Any]
    input_configuration: Dict[str, Any]
    policy_update: Dict[str, Any]


@dataclass(frozen=True)
class GraphNode:
    name: str
    wave: str
    reads: Tuple[str, ...]
    writes: Tuple[str, ...]


WAVE_A_NODES: Tuple[GraphNode, ...] = (
    GraphNode(
        "hypotheses", "A",
        reads=("facts.metric", "examples.al_clusters", "candidates.clusters", "structure.aggregates", "config.current"),
        writes=("upstream.hypotheses",),
    ),
    GraphNode(
        "diagnostic", "A",
        reads=("execution.monitor", "facts.metric", "examples.al_clusters", "structure.aggregates", "config.current"),
        writes=("upstream.diagnostic",),
    ),
    GraphNode(
        "arm_comparison", "A",
        reads=("arms.evidence",),
        writes=("upstream.arm_comparison",),
    ),
    GraphNode(
        "quality", "A",
        reads=("facts.metric", "examples.al_clusters", "candidates.clusters", "structure.fragments_diverse", "config.current"),
        writes=("upstream.quality",),
    ),
)

WAVE_B_NODES: Tuple[GraphNode, ...] = (
    GraphNode(
        "quality_manager", "B",
        reads=("upstream.quality",),
        writes=("upstream.quality",),
    ),
    GraphNode(
        "arm_history", "B",
        reads=("upstream.arm_comparison", "ledger.compact"),
        writes=("arm_history",),
    ),
    GraphNode(
        "final_strategy", "B",
        reads=("upstream.arm_comparison", "arm_history"),
        writes=("final_strategy",),
    ),
)

WAVE_C_NODES: Tuple[GraphNode, ...] = (
    GraphNode(
        "input_config", "C",
        reads=("config.current", "constraints.hard", "facts.metric", "upstream.quality", "upstream.hypotheses", "upstream.diagnostic", "memory.retrieved"),
        writes=("input_configuration",),
    ),
    GraphNode(
        "policy", "C",
        reads=("input_configuration", "facts.metric", "upstream.quality"),
        writes=("policy_update",),
    ),
)


def nodes_for_wave(wave_name: str) -> Tuple[GraphNode, ...]:
    return {
        "A": WAVE_A_NODES,
        "B": WAVE_B_NODES,
        "C": WAVE_C_NODES,
    }[str(wave_name)]


@dataclass
class WaveResult:
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, Exception] = field(default_factory=dict)
    telemetry_errors: Dict[str, Exception] = field(default_factory=dict)


class EventRecorder(Protocol):
    """Minimal interface accepted by RoundGraph for optional durable telemetry."""

    def record(self, event_type: HarnessEventType, payload: Mapping[str, Any]) -> Any:
        ...


class RoundGraph:
    """Fan-out independent callables for one declared wave."""

    def __init__(
        self,
        nodes: Optional[Mapping[str, Tuple[GraphNode, ...]]] = None,
        *,
        event_recorder: Optional[EventRecorder] = None,
    ) -> None:
        self.nodes = dict(nodes or {"A": WAVE_A_NODES, "B": WAVE_B_NODES, "C": WAVE_C_NODES})
        self.event_recorder = event_recorder

    def declared_names(self, wave_name: str) -> List[str]:
        return [node.name for node in self.nodes.get(str(wave_name), ())]

    def run_wave(
        self,
        wave_name: str,
        tasks: Mapping[str, Callable[[], Any]],
        *,
        event_context: Optional[Mapping[str, Any]] = None,
    ) -> WaveResult:
        result = WaveResult()
        if not tasks:
            return result

        wave = str(wave_name)
        context = dict(event_context or {})
        telemetry_error_lock = threading.Lock()

        def record_event(
            name: str, phase: str, event_type: HarnessEventType, payload: Mapping[str, Any],
        ) -> None:
            if self.event_recorder is None:
                return
            try:
                self.event_recorder.record(event_type, payload)
            except Exception as exc:
                # The recorder is optional telemetry, not the authoritative
                # execution transaction.  Surface its failure separately while
                # preserving the node's output or original exception.
                with telemetry_error_lock:
                    result.telemetry_errors[f"{name}:{phase}"] = exc

        def invoke(name: str, function: Callable[[], Any]) -> Any:
            payload = {**context, "wave": wave, "node": name}
            record_event(name, "started", HarnessEventType.GRAPH_NODE_STARTED, payload)
            try:
                value = function()
            except Exception as exc:
                record_event(
                    name, "failed", HarnessEventType.GRAPH_NODE_FAILED,
                    {**payload, "error_type": type(exc).__name__},
                )
                raise
            record_event(name, "succeeded", HarnessEventType.GRAPH_NODE_SUCCEEDED, payload)
            return value

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
            futures = {pool.submit(invoke, name, func): name for name, func in tasks.items()}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result.results[name] = future.result()
                except Exception as exc:
                    result.errors[name] = exc
        return result

    def merge_writes(self, state: MutableMapping[str, Any], wave_name: str, results: Mapping[str, Any]) -> MutableMapping[str, Any]:
        """Copy successful node outputs onto RoundState using declared write tags."""
        by_name = {node.name: node for node in self.nodes.get(str(wave_name), ())}
        for name, value in results.items():
            node = by_name.get(name)
            writes = node.writes if node else (name,)
            for tag in writes:
                state[tag] = value
        return state
