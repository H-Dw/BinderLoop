# Agent Architecture Quick Reference

## TL;DR Pattern

```python
# LLM-powered agent template
class MyAgent:
    SYSTEM = """Detailed instructions for LLM including JSON schema."""
    
    def __init__(self, llm: Optional[OpenAICompatibleClient] = None):
        self.llm = llm
    
    def compute(self, context: Mapping[str, Any]) -> MyResult:
        if self.llm and self.llm.available():
            result = self.llm.chat_json(system=self.SYSTEM, user=context, temperature=0.2)
            if valid_schema(result):
                return MyResult(**result, llm_used=True)
            return MyResult(fallback_data, llm_used=False, raw={"parse_error": ...})
        return MyResult(fallback_data, llm_used=False)
    
    @staticmethod
    def _fallback(context: Mapping[str, Any]) -> Dict:
        # Deterministic rules matching LLM output schema
        pass

@dataclass
class MyResult:
    main_output: List[Dict[str, Any]]
    llm_used: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)
```

---

## How Agents Are Used in Orchestrator

```python
# 1. Instantiate in __init__
orchestrator = BinderDesignOrchestrator(cfg, llm=llm)

# 2. Called in run() loop, per round
for round_id in range(max_rounds):
    # Get metrics
    evaluation = evaluator.evaluate_candidates(candidates)
    struct_eval = structure_agent.analyze_structures(structures)
    
    # Build context
    context = {
        "evaluation": asdict(evaluation),
        "structural_analysis": asdict(struct_eval),
        "memory": memory.summarize_for_agent(memory),
        "messages": bus.query(round_id=round_id)
    }
    
    # LLM agents use context
    quality = quality_agent.analyze(round_id=round_id, context=context)
    hypotheses = hypothesis_agent.propose(context)
    
    # Deterministic agents refine
    proposal = policy_agent.propose_next_boltzgen_params(
        evaluation,
        current_params,
        structural_summary=struct_eval,
        hypotheses=hypotheses.hypotheses,
        quality_analysis=asdict(quality)
    )
    
    # Persist
    quality_agent.write_analysis(quality, f"round_{round_id:02d}/binder_quality_analysis.json")
    (round_dir / "hypotheses.json").write_text(json.dumps(asdict(hypotheses)))
    policy_agent.write_proposal(proposal, f"round_{round_id:02d}/next_round_parameter_proposal.json")
```

---

## LLM Config Flow

### 1. JSON Config File (`llm_config.json`)
```json
{
  "enabled": true,
  "default_model": "openai_gpt4",
  "endpoints": {
    "openai_gpt4": {
      "name": "openai_gpt4",
      "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-4o",
      "provider": "openai",
      "temperature": 0.2
    }
  }
}
```

### 2. CLI Invocation
```bash
python scripts/run_closed_loop_orchestrator.py \
  --config configs/example_binder_task.yaml \
  --llm-config ./llm_config.json \
  --llm-model openai_gpt4 \
  --llm-thinking medium
```

### 3. Config Flow in Code
```python
# In run_closed_loop_orchestrator.py:main()
llm = OpenAICompatibleClient.from_json(Path(llm_config))
llm.configure_default(model_key="openai_gpt4", thinking="medium")
llm.available()  # Check: enabled + endpoint + API key present
orchestrator = BinderDesignOrchestrator(..., llm=llm)

# In orchestrator.__init__:
self.hypothesis_agent = HypothesisAgent(llm=llm)
self.quality_agent = BinderQualityAnalysisAgent(llm=llm)
```

---

## Agent Types Summary

### LLM-Powered (Optional LLM)
| Agent | Input | Output | Fallback |
|-------|-------|--------|----------|
| **HypothesisAgent** | context: {evaluation, structural_analysis, ...} | HypothesisSet | Metric-rule-based hypotheses |
| **BinderQualityAnalysisAgent** | round_id, context | BinderQualityAnalysis | Fragment-rule-based analysis |

### Deterministic (No LLM)
| Agent | Input | Output |
|-------|-------|--------|
| **StructureEvaluationAgent** | structure_files: Seq[Path] | StructureBatchEvaluation |
| **ActiveLearningPolicyAgent** | evaluation, params, metadata | NextRoundParameterProposal |
| **EvaluationAgent** | candidates: List[Dict] | EvaluationSummary |
| **ResultIngestionAgent** | job.output_dir: Path | IngestionResult |

---

## Key Methods to Implement

### For New LLM Agent

```python
class NewLLMAgent:
    SYSTEM = """Your system prompt with JSON schema instruction."""
    
    def __init__(self, llm: Optional[OpenAICompatibleClient] = None):
        self.llm = llm
    
    def main_method(self, param: Type, *, kwarg: Type) -> OutputDataclass:
        """Public method that agents/orchestrator will call."""
        if self.llm and self.llm.available():
            result = self.llm.chat_json(
                system=self.SYSTEM,
                user={"param": param, "kwarg": kwarg},
                temperature=0.2,
                max_tokens=2000
            )
            # Validate schema
            if valid(result):
                return OutputDataclass(..., llm_used=True, raw=result)
            # Parse error → fallback
            return OutputDataclass(self._fallback(...), llm_used=False, raw={"parse_error": result})
        # No LLM → fallback
        return OutputDataclass(self._fallback(...), llm_used=False)
    
    @staticmethod
    def _fallback(param: Type) -> Dict:
        """Deterministic fallback matching LLM output schema."""
        # Your deterministic logic here
        pass
    
    def write_result(self, result: OutputDataclass, path: str | Path) -> Path:
        """Optional: persist result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return path
```

### For New Deterministic Agent

```python
class NewDeterministicAgent:
    def main_method(self, param: Type) -> OutputDataclass:
        """Compute and return result."""
        # Your logic here
        return OutputDataclass(...)
    
    def write_result(self, result: OutputDataclass, path: str | Path) -> Path:
        """Optional: persist result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return path
```

---

## Test Pattern

```python
import pytest
from binderloop.llm import OpenAICompatibleClient
from binderloop.agents.my_agent import MyAgent, MyResult

def test_my_agent_deterministic():
    """Test without LLM (fallback mode)."""
    agent = MyAgent(llm=None)
    result = agent.compute({"context": {...}})
    
    assert isinstance(result, MyResult)
    assert result.llm_used is False
    assert isinstance(result.main_output, list)

def test_my_agent_with_mock_llm(mocker):
    """Test with mocked LLM."""
    llm = mocker.Mock(spec=OpenAICompatibleClient)
    llm.available.return_value = True
    llm.chat_json.return_value = {"main_output": [{"id": "test"}]}
    
    agent = MyAgent(llm=llm)
    result = agent.compute({"context": {...}})
    
    assert result.llm_used is True
    llm.chat_json.assert_called_once()

def test_my_agent_llm_parse_error(mocker):
    """Test LLM parse error → fallback."""
    llm = mocker.Mock(spec=OpenAICompatibleClient)
    llm.available.return_value = True
    llm.chat_json.return_value = {"invalid": "schema"}  # Missing "main_output"
    
    agent = MyAgent(llm=llm)
    result = agent.compute({"context": {...}})
    
    assert result.llm_used is False  # Fallback due to parse error
    assert "raw" in result.__dataclass_fields__
```

---

## Debugging Checklist

- [ ] Is `llm` being passed to agent constructor?
- [ ] Is `llm.available()` returning `True`?
  - Check: `enabled=true` in JSON config
  - Check: `default_model` matches an endpoint name
  - Check: API key env var is set (e.g., `echo $OPENAI_API_KEY`)
- [ ] Does LLM schema match `SYSTEM` prompt?
- [ ] Is fallback code matching LLM output schema?
- [ ] Is result dataclass using `@dataclass` and `field(default_factory=...)`?
- [ ] Are deterministic tests passing?

---

## Files to Reference

- **Existing LLM agent**: `binderloop/agents/hypothesis_agent.py`
- **Complex LLM agent**: `binderloop/agents/binder_quality_analysis_agent.py`
- **Deterministic agent**: `binderloop/agents/structure_evaluation_agent.py`
- **Orchestrator**: `binderloop/orchestration/orchestrator.py`
- **LLM client**: `binderloop/llm.py`
- **CLI entry**: `scripts/run_closed_loop_orchestrator.py`

---

## Common Pitfalls

1. **Forgetting `llm.available()` check**: Always check both `self.llm` and `self.llm.available()` before calling
2. **Schema mismatch**: Make sure fallback returns exactly same keys as LLM output
3. **No audit trail**: Always include `llm_used` flag and `raw` in result dataclass
4. **Hardcoding params**: Use `temperature` and `max_tokens` as parameters to allow tuning
5. **No persistence**: Add `write_*()` method for downstream analysis
6. **Missing imports**: Remember `from dataclasses import asdict, dataclass, field`

