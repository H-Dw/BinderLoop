# Binder-Harness Agent Architecture Analysis

## Executive Summary

The binder-harness project implements a sophisticated agent orchestration system where:
- **LLM-powered agents** (HypothesisAgent, BinderQualityAnalysisAgent) accept an optional `OpenAICompatibleClient`
- **Deterministic agents** (StructureEvaluation, ActiveLearningPolicy) compute results from metrics
- The **BinderDesignOrchestrator** instantiates all agents once and calls them per round
- **LLM configuration flows**: CLI args → OpenAICompatibleClient → orchestrator constructor → individual agents
- Each agent returns a **typed dataclass** result that is serialized to JSON for downstream consumption

---

## 1. Agent Class Pattern

### Constructor Signature Pattern

**LLM-Powered Agents:**
```python
class HypothesisAgent:
    SYSTEM = """System prompt..."""
    
    def __init__(self, llm: Optional[OpenAICompatibleClient] = None):
        self.llm = llm
```

**Deterministic Agents:**
```python
class StructureEvaluationAgent:
    def __init__(self):  # No LLM parameter
        pass
```

### Core Methods Pattern

**LLM-Powered Agents have:**
1. **Main public method** (e.g., `propose()`, `analyze()`)
   - Accepts context/data as `Mapping[str, Any]`
   - Returns typed `@dataclass` result
2. **LLM check**: `if self.llm and self.llm.available()`
3. **LLM call**: `self.llm.chat_json(system=self.SYSTEM, user={...}, temperature=T)`
4. **Parse fallback**: Handle JSON parse failures
5. **Deterministic fallback**: `self._fallback(context)` method
6. **Result wrapping**: Return dataclass with `llm_used=True/False` flag

**Deterministic Agents have:**
1. **Main public method** (e.g., `analyze_structures()`)
2. **No LLM dependency**
3. **Optional**: `write_*()` method for JSON serialization

### Result Dataclass Pattern

**With LLM flag:**
```python
@dataclass
class HypothesisSet:
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    llm_used: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)  # Raw LLM response or parse error info
```

**Extended with metadata:**
```python
@dataclass
class BinderQualityAnalysis:
    round_id: int
    llm_used: bool
    overall_assessment: str
    high_quality_modules: List[Dict[str, Any]] = field(default_factory=list)
    low_quality_modules: List[Dict[str, Any]] = field(default_factory=list)
    causal_factors: List[Dict[str, Any]] = field(default_factory=list)
    next_round_guidance: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
```

**Simple deterministic result:**
```python
@dataclass
class StructureBatchEvaluation:
    total_structures: int
    summaries: List[Dict]
    aggregate_tags: Dict[str, int]
    reliable_seed_fraction: float
    observations: List[str] = field(default_factory=list)
```

---

## 2. LLM-Powered Agent: HypothesisAgent (Example)

### Full Implementation

```python
class HypothesisAgent:
    """Generate failure hypotheses via LLM, with deterministic fallback rules."""

    SYSTEM = """You are a protein binder design research agent. 
Return JSON only: {
  "hypotheses":[{
    "name":...,
    "evidence":...,
    "confidence":0-1,
    "intervention":...,
    "expected_signal_next_round":...,
    "risk":...
  }]
}
Be cautious and do not invent unavailable measurements."""

    def __init__(self, llm: Optional[OpenAICompatibleClient] = None):
        self.llm = llm

    def propose(self, context: Mapping[str, Any]) -> HypothesisSet:
        if self.llm and self.llm.available():
            result = self.llm.chat_json(
                system=self.SYSTEM,
                user={"context": context},
                temperature=0.25
            )
            # Parse success: validate schema, return with llm_used=True
            if isinstance(result.get("hypotheses"), list):
                return HypothesisSet(result["hypotheses"], True, result)
            # Parse failure: use deterministic fallback, mark raw with error
            return HypothesisSet(self._fallback(context), False, {"llm_parse_failed": result})
        
        # LLM unavailable: use deterministic fallback only
        return HypothesisSet(self._fallback(context), False)

    @staticmethod
    def _fallback(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
        # Deterministic rules based on metric tags (e.g., hotspot_miss, folding_failure)
        # Returns list of hypothesis dicts matching LLM output schema
        evaluation = context.get("evaluation") or {}
        structure = context.get("structural_analysis") or {}
        tags = dict(evaluation.get("tag_counts") or {})
        struct_tags = dict(structure.get("aggregate_tags") or {})
        total = max(1, int(evaluation.get("total_candidates") or 1))
        hyps: List[Dict[str, Any]] = []
        
        if tags.get("hotspot_miss", 0) / total > 0.25 or struct_tags.get("hotspot_not_covered", 0):
            hyps.append({
                "name": "hotspot_conditioning_too_weak_or_patch_misaligned",
                "evidence": "hotspot misses are frequent",
                "confidence": 0.75,
                "intervention": "Increase hotspot conditioning...",
                "expected_signal_next_round": "Higher hotspot_contact...",
                "risk": "Over-constraining...",
                "source": "deterministic_fallback"
            })
        
        if not hyps:
            hyps.append({
                "name": "no_single_dominant_failure_mode",
                "evidence": "No dominant failure mode identified",
                "confidence": 0.45,
                ...
            })
        
        return hyps
```

### Key Patterns:
1. **SYSTEM constant**: Instructs LLM on output format (JSON schema)
2. **Two-layer availability check**: `if self.llm and self.llm.available()`
3. **Schema validation**: Check `isinstance(result.get("hypotheses"), list)` after parse
4. **Graceful degradation**: Parse error → fallback; unavailable → fallback
5. **Audit trail**: Include `raw` field so downstream can see what was used
6. **Deterministic rules**: Match LLM output schema exactly, include `"source"` marker

---

## 3. Quality Analysis Agent (Deep Dive)

### Full Signature

```python
def analyze(self, *, round_id: int, context: Mapping[str, Any]) -> BinderQualityAnalysis:
```

### Key Distinctions

1. **Context compression**: `_compact_context()` reduces token count
   - Limits to top-20 structures
   - Truncates candidate lists to top-10 / failed-10
   - Preserves last 50 messages
   
2. **Rich output**: Multiple fields (not just one result)
   ```python
   overall_assessment: str
   high_quality_modules: List[Dict[str, Any]]
   low_quality_modules: List[Dict[str, Any]]
   causal_factors: List[Dict[str, Any]]
   next_round_guidance: List[Dict[str, Any]]
   ```

3. **LLM call parameters**:
   ```python
   result = self.llm.chat_json(
       system=self.SYSTEM,
       user={"round_id": round_id, "context": compact},
       temperature=0.2,      # Lower than HypothesisAgent (0.25)
       max_tokens=2500       # Higher token budget
   )
   ```

4. **Deterministic fallback is sophisticated**:
   - Extracts high/low quality fragments from structural analysis
   - Maps failure reasons → human-readable causes
   - Generates guidance tuples matching LLM output schema
   - Falls back to "collect_richer_structure_evidence" if no data

5. **Persistence**: `write_analysis(analysis: BinderQualityAnalysis, path: Path) -> Path`
   - Uses `asdict()` to serialize dataclass
   - Ensures parent directories exist

---

## 4. Deterministic Agents

### StructureEvaluationAgent (No LLM)

```python
class StructureEvaluationAgent:
    def analyze_structures(
        self,
        structure_files: Sequence[str | Path],
        *,
        binder_chain: str = "B",
        target_chains: Sequence[str] | None = None,
        hotspots: Sequence[str] | None = None
    ) -> StructureBatchEvaluation:
        # Parse PDB/CIF files
        summaries = [analyze_binder_structure(...).to_dict() for path in structure_files]
        
        # Aggregate metrics
        tags: Dict[str, int] = {}
        reliable = 0
        for item in summaries:
            if float(item.get("reliability_score", 0.0)) >= 0.7:
                reliable += 1
            for tag in item.get("reliability_tags", []):
                tags[tag] = tags.get(tag, 0) + 1
        
        return StructureBatchEvaluation(
            len(summaries), summaries, tags, reliable / max(1, len(summaries))
        )
```

### ActiveLearningPolicyAgent (Deterministic Rule-Based)

```python
class ActiveLearningPolicyAgent:
    def propose_next_boltzgen_params(
        self,
        summary: EvaluationSummary,
        current_params: Mapping[str, Any],
        *,
        round_id: int = 1,
        structural_summary: Any | None = None,
        hypotheses: Sequence[Mapping[str, Any]] | None = None,
        quality_analysis: Mapping[str, Any] | None = None,
    ) -> NextRoundParameterProposal:
        params = dict(current_params)
        rationale: list[str] = []
        
        # Rule-based param adjustment based on metrics, hypotheses, quality analysis
        if summary.total_candidates == 0:
            params["num_designs"] = max(5, int(params.get("num_designs", 20)))
            params["run_filtering"] = False
            rationale.append("No metrics collected: keep run small, preserve intermediates...")
        
        if tags.get("hotspot_miss", 0) / total > 0.3:
            params["hotspot_weight"] = float(params.get("hotspot_weight", 1.0)) * 1.2
            rationale.append("Hotspot miss dominated: increase hotspot weight...")
        
        # Multi-round signal incorporation
        if high_modules:  # From quality_analysis
            params["exploit_fragment_modules"] = [m.get("module_id") for m in high_modules[:5]]
            rationale.append("Found reusable high-quality fragments...")
        
        return NextRoundParameterProposal(round_id=round_id, params_update=params, rationale=rationale)
```

---

## 5. Orchestrator Integration

### Orchestrator.__init__: Agent Instantiation

```python
class BinderDesignOrchestrator:
    def __init__(
        self,
        cfg: HarnessConfig,
        *,
        out_dir: str | Path | None = None,
        max_rounds: int | None = None,
        max_parallel: int | None = None,
        max_retries: int | None = None,
        llm=None  # OpenAICompatibleClient passed here
    ):
        self.cfg = cfg
        self.out_dir = Path(out_dir or cfg.runtime.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Instantiate all agents
        self.ingestor = ResultIngestionAgent()
        self.evaluator = EvaluationAgent()
        self.structure_agent = StructureEvaluationAgent()
        self.policy_agent = ActiveLearningPolicyAgent()
        
        # LLM-powered agents get the client
        self.hypothesis_agent = HypothesisAgent(llm=llm)
        self.quality_agent = BinderQualityAnalysisAgent(llm=llm)
```

### Orchestrator.run: Agent Call Sequence Per Round

```python
def run(self, execute_job: Optional[Callable] = None) -> Dict[str, Any]:
    memory = self.memory_store.load(target=asdict(self.cfg.target))
    current_jobs = self._initial_jobs()
    summary: Dict[str, Any] = {"out_dir": str(self.out_dir), "rounds": []}
    
    for round_id in range(self.max_rounds):
        round_dir = self.out_dir / f"round_{round_id:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Execute jobs (in parallel with retries)
        execution_records = self._run_jobs(current_jobs, round_id, execute_job)
        
        # 2. Ingest results → candidates
        ingestions = [asdict(self.ingestor.ingest_boltzgen_output(job.output_dir)) for job in current_jobs]
        candidates = [row for item in ingestions for row in item.get("candidates", [])]
        
        # 3. Evaluate candidates → metrics
        evaluation = self.evaluator.evaluate_candidates(candidates)
        self.evaluator.write_summary(evaluation, round_dir / "evaluation_summary.json")
        
        # 4. Analyze structures → coordinate-level features
        structures = self._collect_structure_files(ingestions)
        struct_eval = self.structure_agent.analyze_structures(
            structures,
            binder_chain=self._guess_binder_chain(),
            target_chains=[self.cfg.target.chain_id],
            hotspots=self.cfg.target.hotspots
        )
        self.structure_agent.write_batch(struct_eval, round_dir / "structure_evaluation.json")
        
        # 5. Build comprehensive context for LLM agents
        context = {
            "round_id": round_id,
            "evaluation": asdict(evaluation),
            "structural_analysis": asdict(struct_eval),
            "memory": self.memory_store.summarize_for_agent(memory),
            "messages": [m.to_dict() for m in self.bus.query(round_id=round_id)]
        }
        
        # 6. Quality analysis (LLM or deterministic)
        quality_analysis = self.quality_agent.analyze(round_id=round_id, context=context)
        self.quality_agent.write_analysis(quality_analysis, round_dir / "binder_quality_analysis.json")
        context["quality_analysis"] = asdict(quality_analysis)
        
        # 7. Hypothesis generation (LLM or deterministic)
        hypotheses = self.hypothesis_agent.propose(context)
        (round_dir / "hypotheses.json").write_text(
            json.dumps(asdict(hypotheses), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # 8. Policy agent proposes next round params (deterministic)
        proposal = self.policy_agent.propose_next_boltzgen_params(
            evaluation,
            self._base_params(),
            round_id=round_id + 1,
            structural_summary=struct_eval,
            hypotheses=hypotheses.hypotheses,
            quality_analysis=asdict(quality_analysis)
        )
        self.policy_agent.write_proposal(proposal, round_dir / "next_round_parameter_proposal.json")
        
        # 9. Store round in memory
        record = self.memory_store.upsert_round(memory, round_id)
        record.ingestion = ingestions
        record.evaluation = asdict(evaluation)
        record.structural_analysis = [asdict(struct_eval)]
        record.quality_analysis = asdict(quality_analysis)
        record.hypotheses = hypotheses.hypotheses
        record.decisions = [asdict(proposal)]
        record.finished_at = time.time()
        
        # 10. Update summary and propose next round jobs
        summary["rounds"].append({
            "round_id": round_id,
            "evaluation": asdict(evaluation),
            "structural_analysis": asdict(struct_eval),
            "quality_analysis": asdict(quality_analysis),
            "hypotheses": asdict(hypotheses),
            "proposal": asdict(proposal)
        })
        
        if round_id + 1 >= self.max_rounds:
            break
        
        # 11. Active learner proposes next set of jobs
        current_jobs = self.learner.propose_next(
            round_id + 1,
            current_jobs,
            evaluation.top_candidates,
            str(self.out_dir),
            top_k=self.cfg.active_learning.top_k,
            policy_update=proposal.params_update,
            structural_summary=struct_eval,
            hypotheses=hypotheses.hypotheses,
            quality_analysis=asdict(quality_analysis)
        ).jobs
    
    (self.out_dir / "orchestrator_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return summary
```

### Call Dependency Flow
```
ingestor → evaluation → structure_agent → quality_agent (LLM)
                                          hypothesis_agent (LLM)
                                          policy_agent (deterministic)
```

---

## 6. LLM Config Flow: CLI → Orchestrator → Agents

### Flow Diagram
```
CLI args (--llm-config, --llm-model, --llm-thinking, --require-llm)
    ↓
load_config(path)  [YAML config]
    ↓
llm_config_path = args.llm_config or cfg.runtime.llm_config_path
    ↓
OpenAICompatibleClient.from_json(llm_config_path)  [reads JSON]
    ↓
llm.configure_default(model_key, thinking)  [CLI overrides]
    ↓
llm.available()  [check enabled + endpoint + API key]
    ↓
BinderDesignOrchestrator(..., llm=llm)  [pass to constructor]
    ↓
HypothesisAgent(llm=llm)
BinderQualityAnalysisAgent(llm=llm)
    ↓
agent.llm.chat_json(system, user, temperature, max_tokens)
```

### Script: run_closed_loop_orchestrator.py

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Run closed-loop Binder design orchestrator")
    parser.add_argument("--config", default="configs/example_binder_task.yaml")
    parser.add_argument("--out", default="outputs/closed_loop_orchestrator")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--llm-config", help="Local JSON with OpenAI-compatible endpoints/API-key env names")
    parser.add_argument("--llm-model", help="Endpoint key from the LLM config to use as default_model")
    parser.add_argument("--llm-thinking", help="Reasoning/thinking level: low|medium|high|enabled")
    parser.add_argument("--require-llm", action="store_true", help="Fail if LLM not available")
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config)
    
    # 1. Load LLM config from JSON or YAML
    llm_config = args.llm_config or cfg.runtime.llm_config_path
    if llm_config:
        llm_config_path = Path(llm_config).expanduser()
        if not llm_config_path.is_absolute():
            llm_config_path = root / llm_config_path
        llm = OpenAICompatibleClient.from_json(llm_config_path)
    else:
        llm = None
    
    # 2. Apply CLI overrides
    if llm:
        llm.configure_default(model_key=args.llm_model, thinking=args.llm_thinking)
    
    # 3. Validate if required
    if args.require_llm and not (llm and llm.available()):
        raise SystemExit("--require-llm set but LLM not available")
    
    # 4. Pass to orchestrator
    summary = BinderDesignOrchestrator(
        cfg,
        out_dir=root / args.out,
        max_rounds=args.max_rounds,
        max_parallel=args.max_parallel,
        max_retries=args.max_retries,
        llm=llm  # <-- LLM flows here
    ).run()
    
    print(f"Closed-loop summary: {summary['out_dir']}/orchestrator_summary.json")
    return 0
```

### LLM Config JSON Structure

```json
{
  "enabled": true,
  "default_model": "openrouter_claude",
  "endpoints": {
    "openrouter_claude": {
      "name": "openrouter_claude",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY",
      "model": "anthropic/claude-3.5-sonnet",
      "provider": "openrouter",
      "timeout_seconds": 60,
      "thinking": "medium",
      "thinking_budget_tokens": 5000
    },
    "openai_gpt4": {
      "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-4o",
      "provider": "openai"
    }
  }
}
```

---

## 7. OpenAICompatibleClient Details

### Key Methods

#### `from_json(path: str | Path | None) -> OpenAICompatibleClient | None`
- Reads JSON with `endpoints` and `default_model`
- Creates `ModelEndpoint` objects
- Returns `None` if path is None
- Raises `FileNotFoundError` if file missing

#### `configure_default(model_key: str | None = None, thinking: str | None = None) -> None`
- Overrides `settings.default_model` from CLI
- Modifies endpoint's `thinking` level at runtime

#### `available() -> bool`
- Checks `enabled` flag
- Checks `default_model` in `endpoints`
- Checks API key availability (env var or secrets store)

#### `chat_json(...) -> Dict[str, Any]`
- Calls `chat_text()` to get response
- Parses JSON using `_extract_json_object()`
- Returns `{"parse_error": ..., "raw_text": ...}` on failure

#### `chat_text(*, system: str, user: str, model_key: str | None, temperature: float, max_tokens: int) -> str`
- Wraps `chat_messages()` with system/user roles
- Returns content string

#### `chat_messages(*, messages: Sequence[Mapping], model_key, temperature, max_tokens) -> Dict[str, Any]`
- Builds request payload with model, temperature, max_tokens
- Adds thinking config via `_thinking_payload(endpoint)`
- Adds provider-specific `extra_body`
- Makes HTTP POST to endpoint's base_url + "/chat/completions"
- Returns full assistant message dict (for multi-turn continuation)

### Thinking Configuration (`_thinking_payload`)

```
OpenRouter     → {"reasoning": {"enabled": true}} or {"reasoning": {"effort": "medium"}}
DeepSeek       → {"thinking": {"type": "enabled|disabled"}, "reasoning_effort": "low|high|max"}
Anthropic      → {"thinking": {"type": "enabled", "budget_tokens": 1024}}
OpenAI/Other   → {} or {"reasoning_effort": "medium"}
```

---

## 8. Agents Module Export Pattern

### agents/__init__.py

```python
from .design_parameter_agent import DesignParameterAgent
from .design_spec_agent import DesignSpecAgent
from .taiji_execution_agent import TaijiExecutionAgent
from .run_monitor_agent import RunMonitorAgent
from .result_ingestion_agent import ResultIngestionAgent
from .evaluation_agent import EvaluationAgent
from .active_learning_policy_agent import ActiveLearningPolicyAgent
from .structure_evaluation_agent import StructureEvaluationAgent
from .hypothesis_agent import HypothesisAgent
from .binder_quality_analysis_agent import BinderQualityAnalysisAgent

__all__ = [
    "DesignParameterAgent",
    "DesignSpecAgent",
    "TaijiExecutionAgent",
    "RunMonitorAgent",
    "ResultIngestionAgent",
    "EvaluationAgent",
    "ActiveLearningPolicyAgent",
    "StructureEvaluationAgent",
    "HypothesisAgent",
    "BinderQualityAnalysisAgent",
]
```

---

## 9. Skill Documentation Pattern

### File Format
- Location: `docs/skills/*.skill.md`
- YAML frontmatter + Markdown body

### Example: agent_communication.skill.md

```yaml
---
name: binder-agent-communication
version: 0.1.0
description: JSONL protocol for Binder harness agents to exchange observations, failures, hypotheses, proposals, and decisions.
---

# Binder Agent Communication Skill

`AgentMessage` envelope: sender, recipient, message_type, round_id, optional job_id, correlation_id, parent_id, content, confidence, artifacts. Messages are append-only JSONL. Store evidence and decisions, not hidden reasoning or secrets.
```

### Existing Skills
1. **agent_communication.skill.md** - Agent message protocol
2. **closed_loop_orchestrator.skill.md** - Main orchestration loop
3. **hypothesis_generation.skill.md** - Hypothesis generation (HypothesisAgent)
4. **strategy_active_learning.skill.md** - Active learning strategy
5. **structure_evaluation.skill.md** - Structure analysis and feature extraction

---

## 10. Summary Table: Agent Patterns

| Agent | Type | LLM | Input | Output Dataclass | Fallback |
|-------|------|-----|-------|------------------|----------|
| HypothesisAgent | LLM | Yes | context: Mapping | HypothesisSet | Metric-rule-based hypotheses |
| BinderQualityAnalysisAgent | LLM | Yes | round_id, context | BinderQualityAnalysis | Fragment-rule-based analysis |
| StructureEvaluationAgent | Deterministic | No | structure_files | StructureBatchEvaluation | N/A (no fallback) |
| ActiveLearningPolicyAgent | Deterministic | No | summary, params, metadata | NextRoundParameterProposal | N/A (no fallback) |
| EvaluationAgent | Deterministic | No | candidates | EvaluationSummary | N/A (no fallback) |
| ResultIngestionAgent | Deterministic | No | job.output_dir | IngestionResult | N/A (no fallback) |

---

## 11. Key Design Insights

### 1. **Graceful Degradation**
- All LLM calls are optional (can fall back to deterministic)
- Improves testability, reliability, and cost-efficiency

### 2. **Audit Trail**
- Each agent result includes `llm_used: bool` and `raw: Dict`
- Downstream can see what generated each output

### 3. **Schema Consistency**
- Deterministic fallback outputs match LLM schema exactly
- Enables downstream code to treat them identically

### 4. **Context Building**
- Orchestrator builds rich context dict for each agent
- Agents receive only what they need
- Allows context compression (quality agent compacts structures)

### 5. **Two-Layer Checks**
```python
if self.llm and self.llm.available():
    # Try LLM
else:
    # Use fallback
```
- First check: is client instantiated?
- Second check: is endpoint enabled + configured + API key present?

### 6. **JSON Serialization**
- All results are `@dataclass` for easy `asdict()` conversion
- Orchestrator writes to `round_XX/agent_output.json`
- Enables persistence and downstream analysis

### 7. **Multi-Round Signal Incorporation**
- Policy agent receives: evaluation metrics + structural tags + hypotheses + quality analysis
- Adapts strategy based on multi-source evidence
- Learns from round to round via memory store

---

## 12. Orchestrator Output Structure

```
outputs/closed_loop_orchestrator/
├── orchestrator_summary.json           # Top-level summary
├── agent_messages.jsonl                # Message bus log
├── memory/
│   └── [memory store]
├── round_00/
│   ├── evaluation_summary.json        # From EvaluationAgent
│   ├── structure_evaluation.json       # From StructureEvaluationAgent
│   ├── binder_quality_analysis.json   # From BinderQualityAnalysisAgent (LLM or fallback)
│   ├── hypotheses.json                # From HypothesisAgent (LLM or fallback)
│   └── next_round_parameter_proposal.json  # From ActiveLearningPolicyAgent
├── round_01/
│   └── [same structure]
└── ...
```

---

## 13. Example Agent Instantiation in Tests

```python
# With LLM
llm = OpenAICompatibleClient.from_json("llm_config.json")
llm.configure_default(model_key="openai_gpt4")
hypothesis_agent = HypothesisAgent(llm=llm)

# Without LLM (offline/deterministic mode)
hypothesis_agent = HypothesisAgent(llm=None)

# Both will work identically in terms of interface
result = hypothesis_agent.propose({"evaluation": {...}, "structural_analysis": {...}})
assert isinstance(result, HypothesisSet)
assert isinstance(result.hypotheses, list)
assert isinstance(result.llm_used, bool)
```

---

## Conclusion

The binder-harness agent architecture exemplifies **composable, deterministic-first design with optional LLM enhancement**:
- Clear separation of concerns (agents are stateless, deterministic where possible)
- Graceful fallbacks for robustness and testability
- Config flows cleanly from CLI → JSON → objects → agents
- Each agent returns typed dataclass for auditability and downstream processing
- Orchestrator orchestrates, agents process, results persist

This pattern is reusable for any multi-round decision-making system with optional LLM reasoning.
