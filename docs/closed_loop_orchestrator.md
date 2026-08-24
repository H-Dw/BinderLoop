# Closed-loop Binder design orchestration

This iteration adds a closed-loop architecture around Binder design experiments:

- `BinderDesignOrchestrator`: global scheduler with `max_rounds`, bounded `max_parallel`, and `max_retries`.
- `ExperimentMemoryStore`: durable cross-round trajectory in `memory/experiment_memory.json` plus append-only `events.jsonl`.
- `MessageBus`: JSONL agent communication protocol (`agent_messages.jsonl`).
- `StructureEvaluationAgent`: coordinate-level Binder/interface analysis beyond scalar model metrics.
- `HypothesisAgent`: LLM-backed hypothesis generation through OpenAI-compatible endpoints, with deterministic fallback rules when API is unavailable.
- `StrategyLevelActiveLearner`: mixes exploitation, hotspot repair, foldability repair, pose/interface repair, clash repair, and diversity exploration arms.

## API key file and LLM mode

By default the orchestrator can run without an LLM API. In that case `HypothesisAgent` uses deterministic fallback rules. To force real LLM API usage:

```bash
cd /projects/design_harness/BinderLoop
cp configs/llm_endpoints.example.json configs/llm_endpoints.local.json
chmod 600 configs/llm_endpoints.local.json
```

Edit `configs/llm_endpoints.local.json`:

```json
{
  "enabled": true,
  "default_model": "openrouter_deepseek",
  "secrets": {
    "OPENROUTER_API_KEY": {
      "env": "OPENROUTER_API_KEY"
    }
  },
  "endpoints": {
    "openrouter_deepseek": {
      "provider": "openrouter",
      "base_url": "https://openrouter.ai/api/v1",
      "model": "deepseek/deepseek-v4-pro",
      "api_key_env": "OPENROUTER_API_KEY",
      "timeout_seconds": 120,
      "thinking": "enabled",
      "extra_body": {
        "reasoning": {
          "enabled": true
        }
      }
    },
    "gpt_default": {
      "provider": "openai-compatible",
      "base_url": "https://api.example.com/v1",
      "model": "gpt-4o-mini",
      "api_key_env": "BINDERLOOP_OPENAI_KEY",
      "timeout_seconds": 60,
      "thinking": "medium",
      "extra_body": {}
    },
    "claude_default": {
      "provider": "anthropic-compatible",
      "base_url": "https://api.anthropic-compatible.example.com/v1",
      "model": "claude-3-5-sonnet-latest",
      "api_key_env": "BINDERLOOP_CLAUDE_KEY",
      "timeout_seconds": 60,
      "thinking": "enabled",
      "thinking_budget_tokens": 2048,
      "extra_body": {}
    }
  }
}
```

Then export the key without committing it:

```bash
export BINDERLOOP_OPENAI_KEY='<your-api-key>'
export OPENROUTER_API_KEY='<your-openrouter-api-key>'
```

Important fields:

- `enabled: true` is required to enable LLM API mode.
- `default_model` selects one endpoint key under `endpoints`.
- `model` is the real provider model name sent in the request body.
- `thinking` controls reasoning/thinking. For `openrouter` endpoints it is sent as `reasoning`; for `deepseek` endpoints it is sent as `thinking: {type}` plus `reasoning_effort`; for `openai-compatible` endpoints it is sent as `reasoning_effort`; for `anthropic-compatible` endpoints it is sent as a `thinking` object with `thinking_budget_tokens`.
- `extra_body` is merged into the JSON request body for provider-specific knobs.
- OpenRouter may return `reasoning_details`; use `OpenAICompatibleClient.chat_messages()` when you need to preserve and pass the full assistant message back unmodified in a multi-turn conversation.

The local file is ignored by `.gitignore`. Keep real keys out of git.

## Commands

### Deterministic fallback smoke test

```bash
cd /projects/design_harness/BinderLoop
/usr/bin/python3 scripts/run_closed_loop_orchestrator.py \
  --config configs/example_binder_task.yaml \
  --out outputs/closed_loop_rules \
  --max-rounds 1 \
  --max-parallel 1 \
  --max-retries 1
```

### LLM API mode, fail if API is not available

```bash
cd /projects/design_harness/BinderLoop
export BINDERLOOP_OPENAI_KEY='<your-api-key>'
/usr/bin/python3 scripts/run_closed_loop_orchestrator.py \
  --config configs/example_binder_task.yaml \
  --out outputs/closed_loop_llm \
  --max-rounds 2 \
  --max-parallel 2 \
  --max-retries 3 \
  --llm-config configs/llm_endpoints.local.json \
  --llm-model gpt_default \
  --llm-thinking medium \
  --require-llm
```

`--require-llm` prevents a silent fallback to deterministic rules. Check `outputs/closed_loop_llm/round_00/hypotheses.json`; `llm_used: true` means the LLM API was used.

### YAML-based LLM config

```yaml
runtime:
  llm_config_path: configs/llm_endpoints.local.json
```

Then:

```bash
/usr/bin/python3 scripts/run_closed_loop_orchestrator.py \
  --config configs/example_binder_task.yaml \
  --out outputs/closed_loop_llm \
  --require-llm
```

## Full harness integration

The new modules are connected through the current Harness path:

```text
run_closed_loop_orchestrator.py
  -> OpenAICompatibleClient.from_json(...)
  -> BinderDesignOrchestrator(..., llm=client)
  -> ResultIngestionAgent.ingest_boltzgen_output(...)
  -> EvaluationAgent.evaluate_candidates(...)
  -> StructureEvaluationAgent.analyze_structures(...)
  -> HypothesisAgent.propose(...)
  -> ActiveLearningPolicyAgent.propose_next_boltzgen_params(... structural_summary, hypotheses ...)
  -> StrategyLevelActiveLearner.propose_next(... policy_update, structural_summary, hypotheses ...)
```

Outputs proving the integration:

- `agent_messages.jsonl` — agent events and retry/status messages.
- `memory/experiment_memory.json` and `memory/events.jsonl` — persisted cross-round state.
- `round_*/structure_evaluation.json` — coordinate-level tags and reliability summaries.
- `round_*/hypotheses.json` — LLM or deterministic hypotheses.
- `round_*/next_round_parameter_proposal.json` — policy changes that consume numeric, structural, and hypothesis evidence.
- `orchestrator_summary.json` — final multi-round summary.

The closed-loop command above is a scheduler/analysis loop. For production BoltzGen/Taiji execution, use the existing full-path command:

```bash
/usr/bin/python3 scripts/run_boltzgen_complete_path_test.py --out outputs/my_boltzgen_run --submit
```

## Per-round binder quality and module analysis

Each round now writes a detailed quality analysis artifact:

```text
outputs/<run>/round_XX/binder_quality_analysis.json
```

Inputs:

- candidate metrics from `EvaluationAgent`;
- coordinate summaries from `StructureEvaluationAgent`;
- fragment/module quality labels from `analysis/structure_features.py`;
- previous round memory and message-bus context.

`BinderQualityAnalysisAgent` uses the configured LLM API when available. It returns high-quality modules, low-quality modules, likely causal factors, and next-round guidance. Without an LLM, deterministic fallback preserves the same schema for tests.

The analysis is not just reported: it is consumed by both next-round policy and strategy:

```text
binder_quality_analysis.json
  -> ActiveLearningPolicyAgent(... quality_analysis ...)
  -> StrategyLevelActiveLearner(... quality_analysis ...)
```

This can add `exploit_fragment_modules`, `avoid_fragment_modules`, `module_guided_exploitation`, and `module_guided_repair` signals to the next round.
