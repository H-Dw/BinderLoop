"""Orchestrator-facing tool registry.

Tools are deterministic Python callables, not LLM function-calling. Agents do
not bind this registry unless a future role needs a read-only lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]
    args_schema: Dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError("unknown tool: %s" % name)
        return self._tools[name]

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name).func(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def specs(self) -> List[ToolSpec]:
        return [self._tools[name] for name in self.names()]


TOOLS = ToolRegistry()


def _ingest_results(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.result_ingestion_agent import ResultIngestionAgent
    return ResultIngestionAgent().ingest_boltzgen_output(*args, **kwargs)


def _evaluate_candidates(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.evaluation_agent import EvaluationAgent
    return EvaluationAgent().evaluate_candidates(*args, **kwargs)


def _analyze_structures(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.structure_evaluation_agent import StructureEvaluationAgent
    return StructureEvaluationAgent().analyze_structures(*args, **kwargs)


def _mine_fragment_templates(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.fragment_template_mining_agent import FragmentTemplateMiningAgent
    return FragmentTemplateMiningAgent().mine_templates(*args, **kwargs)


def _validate_config(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.config_validation_agent import ConfigValidationAgent
    return ConfigValidationAgent().validate_full_job_config(*args, **kwargs)


def _apply_config_contract(changes: Mapping[str, Any], **kwargs: Any) -> Any:
    from binderloop.agents.config_parameter_contract import supported_config_changes
    return supported_config_changes(changes, **kwargs)


def _fact_check_metric_facts(text: str, metric_facts: Optional[Mapping[str, Any]] = None) -> Any:
    from binderloop.agents.context_compaction import fact_check_text_against_metric_facts
    return fact_check_text_against_metric_facts(text, metric_facts)


def _retrieve_memory(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.memory_retrieval_agent import MemoryRetrievalAgent
    return MemoryRetrievalAgent().retrieve(*args, **kwargs)


def _compress_memory(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.memory_compression_agent import MemoryCompressionAgent
    return MemoryCompressionAgent().compress_to_budget(*args, **kwargs)


def _assemble_prompt(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.prompt_assembler import assemble
    return assemble(*args, **kwargs)


def _cluster_candidates(**kwargs: Any) -> Any:
    from binderloop.analysis.candidate_clusters import aggregate_candidate_phenotypes
    return aggregate_candidate_phenotypes(**kwargs)


def _submit_taiji(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.taiji_execution_agent import TaijiExecutionAgent
    return TaijiExecutionAgent().submit(*args, **kwargs)


def _poll_run(*args: Any, **kwargs: Any) -> Any:
    from binderloop.agents.run_monitor_agent import RunMonitorAgent
    return RunMonitorAgent().check_once(*args, **kwargs)


def _register_defaults() -> None:
    specs = (
        ToolSpec("ingest_results", "Scan a BoltzGen output directory into an ingested run.", _ingest_results, {"output_dir": "path"}),
        ToolSpec("evaluate_candidates", "Score candidates and assign failure tags.", _evaluate_candidates, {"candidates": "list"}),
        ToolSpec("analyze_structures", "Extract coordinate-level interface features.", _analyze_structures, {"structure_files": "list"}),
        ToolSpec("mine_fragment_templates", "Mine reusable fragment templates.", _mine_fragment_templates),
        ToolSpec("validate_config", "Deterministic + advisory full-job config validation.", _validate_config, {"config": "mapping"}),
        ToolSpec("apply_config_contract", "Whitelist executable config keys.", _apply_config_contract, {"changes": "mapping"}),
        ToolSpec("fact_check_metric_facts", "Check LLM text against immutable metric facts.", _fact_check_metric_facts, {"text": "str"}),
        ToolSpec("retrieve_memory", "Structured recall + optional semantic rerank.", _retrieve_memory),
        ToolSpec("compress_memory", "Compress indexed memory items to a budget.", _compress_memory),
        ToolSpec("assemble_prompt", "Assemble a tagged prompt for one agent role.", _assemble_prompt, {"role": "str"}),
        ToolSpec("cluster_candidates", "Deterministic phenotype clustering for prompt cards.", _cluster_candidates),
        ToolSpec("submit_taiji", "Submit a Taiji job spec.", _submit_taiji),
        ToolSpec("poll_run", "Snapshot Taiji / local run status.", _poll_run),
    )
    for spec in specs:
        TOOLS.register(spec)


_register_defaults()
