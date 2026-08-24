# Agent Architecture Documentation

Welcome! This directory contains comprehensive documentation on the binder-harness agent architecture.

## Quick Start

👉 **Start here**: [Agent Quick Reference](AGENT_QUICK_REFERENCE.md) (5 min read)
- TL;DR pattern for LLM and deterministic agents
- Orchestrator usage examples
- LLM config flow
- Common pitfalls

## Comprehensive Reference

📚 **Deep dive**: [Agent Architecture](AGENT_ARCHITECTURE.md) (30 min read)
- Complete pattern documentation
- Full code examples for all agent types
- Orchestrator integration details
- LLM client documentation
- Design principles
- Architecture insights

## Related Documentation

- [Architecture Overview](architecture.md) - System-level architecture
- [Skills Documentation](skills/) - Agent communication and skills

## Quick Links to Source Code

### LLM-Powered Agents
- [HypothesisAgent](../binderloop/agents/hypothesis_agent.py) - Simple LLM agent (start here!)
- [BinderQualityAnalysisAgent](../binderloop/agents/binder_quality_analysis_agent.py) - Complex LLM agent with context compression

### Deterministic Agents
- [StructureEvaluationAgent](../binderloop/agents/structure_evaluation_agent.py) - Parse structures
- [ActiveLearningPolicyAgent](../binderloop/agents/active_learning_policy_agent.py) - Rule-based strategy

### Core Components
- [Orchestrator](../binderloop/orchestration/orchestrator.py) - Multi-round orchestration loop
- [LLM Client](../binderloop/llm.py) - OpenAI-compatible HTTP client
- [CLI Entry Point](../scripts/run_closed_loop_orchestrator.py) - Config flow from CLI

## Key Concepts

### Agent Class Pattern
```python
class MyAgent:
    SYSTEM = """LLM system prompt with JSON schema."""
    
    def __init__(self, llm: Optional[OpenAICompatibleClient] = None):
        self.llm = llm
    
    def main_method(self, context: Mapping[str, Any]) -> ResultDataclass:
        if self.llm and self.llm.available():
            result = self.llm.chat_json(system=self.SYSTEM, user=context, temperature=0.2)
            if valid_schema(result):
                return ResultDataclass(**result, llm_used=True)
            return ResultDataclass(self._fallback(context), llm_used=False)
        return ResultDataclass(self._fallback(context), llm_used=False)
    
    @staticmethod
    def _fallback(context: Mapping[str, Any]) -> Dict:
        # Deterministic rules
        pass
```

### LLM Config Flow
```
CLI args (--llm-config, --llm-model, --llm-thinking)
  ↓
OpenAICompatibleClient.from_json(path)
  ↓
llm.configure_default(model_key, thinking)
  ↓
llm.available()  [Check enabled + endpoint + API key]
  ↓
BinderDesignOrchestrator(cfg, llm=llm)
  ↓
HypothesisAgent(llm=llm)
BinderQualityAnalysisAgent(llm=llm)
  ↓
agent.llm.chat_json(system, user, temperature, max_tokens)
```

### Per-Round Agent Call Sequence
1. **EvaluationAgent** - Compute metrics from candidates
2. **StructureEvaluationAgent** - Extract coordinate-level features
3. **BinderQualityAnalysisAgent** - Analyze quality (LLM or fallback)
4. **HypothesisAgent** - Generate hypotheses (LLM or fallback)
5. **ActiveLearningPolicyAgent** - Propose next-round params (deterministic)

## Architecture Principles

✓ **Graceful Degradation** - LLM calls optional, fallback to deterministic rules
✓ **Audit Trail** - Each result includes `llm_used: bool` and `raw: Dict`
✓ **Schema Consistency** - Fallback output matches LLM schema exactly
✓ **Multi-Round Optimization** - Each round feeds into next via memory store
✓ **Testability** - Works offline, no LLM required for testing
✓ **Cost Efficiency** - LLM is enhancement, not requirement

## Common Tasks

### Run closed-loop orchestrator with LLM
```bash
python scripts/run_closed_loop_orchestrator.py \
  --config configs/example_binder_task.yaml \
  --llm-config ./llm_config.json \
  --llm-model openai_gpt4 \
  --llm-thinking medium
```

### Run without LLM (deterministic mode)
```bash
python scripts/run_closed_loop_orchestrator.py \
  --config configs/example_binder_task.yaml
```

### Create LLM config JSON
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
      "provider": "openai"
    }
  }
}
```

### Create new LLM agent
See template in [Agent Quick Reference - Key Methods to Implement](AGENT_QUICK_REFERENCE.md#key-methods-to-implement)

### Debug LLM availability
Checklist in [Agent Quick Reference - Debugging Checklist](AGENT_QUICK_REFERENCE.md#debugging-checklist)

## File Structure

```
docs/
├── README_AGENTS.md              [You are here]
├── AGENT_ARCHITECTURE.md         [Comprehensive 13-section guide]
├── AGENT_QUICK_REFERENCE.md      [Quick patterns and templates]
├── architecture.md               [System architecture overview]
└── skills/
    ├── agent_communication.skill.md
    ├── closed_loop_orchestrator.skill.md
    ├── hypothesis_generation.skill.md
    ├── strategy_active_learning.skill.md
    └── structure_evaluation.skill.md
```

## Summary Table: Agents at a Glance

| Agent | Type | LLM | Input | Output | Fallback |
|-------|------|-----|-------|--------|----------|
| HypothesisAgent | LLM | Yes | context | HypothesisSet | Metric-rules |
| BinderQualityAnalysisAgent | LLM | Yes | round_id, context | BinderQualityAnalysis | Fragment-rules |
| StructureEvaluationAgent | Det. | No | structures | StructureBatchEvaluation | N/A |
| ActiveLearningPolicyAgent | Det. | No | evaluation, params | NextRoundParameterProposal | N/A |
| EvaluationAgent | Det. | No | candidates | EvaluationSummary | N/A |
| ResultIngestionAgent | Det. | No | job_dir | IngestionResult | N/A |

## Need Help?

- **Beginner?** Start with [Agent Quick Reference](AGENT_QUICK_REFERENCE.md)
- **Advanced?** Read [Agent Architecture](AGENT_ARCHITECTURE.md)
- **Implementing new agent?** Use templates in Quick Reference
- **Debugging LLM issues?** Check Debugging Checklist
- **Understanding config flow?** See LLM Config Flow section

---

**Last Updated**: 2026-05-26  
**Status**: Complete (13 sections, 748 lines in AGENT_ARCHITECTURE.md)
