# Self-Improving Binder Skill

## Scope

The evolving skill is run-local. An existing YAML can seed a new run, but the
source is never modified. Without a source, the harness creates a unique file
under `outputs/<run>/memory/self_improvement_skills/`. Resume reuses the same
working file recorded in `self_improvement_skill_state.json`.

## Document contract

The YAML has fixed modules:

- `successful_patterns`
- `failure_avoidance`
- `parameter_effects`
- `structural_context_rules`
- `exploration_exploitation`
- `rollback_recovery`
- `transfer_candidates`

Rules are maps keyed by stable `rule_id`; they are not an append-only prose
list. The LLM proposes typed operations and `SkillDocumentEditor` validates,
de-identifies and atomically rewrites the complete document.

Each rule includes a canonical signature: experience type, parameter families,
relative action directions, structural trigger phenotypes, expected/watch
signals and contraindications. Target names, file paths, chain/residue tokens,
candidate IDs and target-specific absolute lengths are forbidden in
prompt-visible rule text.

## Learning lifecycle

1. Static strategy YAMLs bootstrap `seed_active` rules.
2. The prior round's strategy exposure is joined with the evaluated outcome.
3. `SelfImprovementSkillAgent` proposes structured updates.
4. Deterministic signature overlap shortlists semantic candidates.
5. The LLM classifies rule relations as equivalent, subsuming, complementary,
   contradictory or distinct.
6. Schema, de-identification and evidence gates apply the update.
7. Candidates require support before becoming active; contradictions can make
   a rule contested or retired.
8. Only bounded Top-K active rules enter downstream prompts.

Infrastructure failures do not produce scientific lessons. Multi-parameter
rounds remain correlational unless a controlled comparison isolates a family.

## Instruction precedence

1. Immutable metric facts and deterministic controls.
2. Validated run-local learned rules, as the highest advisory layer.
3. Static strategy and reasoning skills.
4. Raw memory and unconsolidated suggestions.

Downstream agents must cite `learned_rule_ids` when using a learned rule.

## Conflict arbitration

Hard constraints reject invalid changes directly. Soft conflicts are marked
`contested` and routed to `StrategyConflictResolutionAgent` at most once per
round. It considers exact cross-round outcomes, historical best configuration,
interface confidence, PAE, foldability, refold RMSD, hotspot/geometry,
diversity and rollback risk.

The decision is one of `choose`, `blend`, `hold`, `revert_to_best`,
`controlled_compare`, or `insufficient_evidence`. It remains advisory until
supported-key filtering, ownership, inertia, pressure conflict, target bounds
and template provenance checks pass.

## Research basis

- ExpeL: incremental natural-language insight operations and persistent reuse.
- AutoGuide: contrastive, context-conditioned guidelines.
- Agent Workflow Memory: inducing reusable workflows from trajectories.
- Voyager and EvoSkill: validate before retaining evolving skills.
- SkillsBench: self-generated skills can harm performance without curation.
- Agentic Skills SoK: explicit lifecycle, composition and governance.
- Experience-faithfulness studies: presence in a prompt is not proof of use;
  citations and ablations are required.

## Main artifacts

- `round_XX/self_improvement_evidence.json`
- `round_XX/self_improvement_update.json`
- `round_XX/self_improvement_skill_snapshot.yaml`
- `round_XX/strategy_conflicts.json`
- `round_XX/strategy_conflict_resolution.json`
- `round_XX/next_strategy_exposure.json`
- `memory/self_improvement_skill_state.json`
- `memory/self_improvement_skills/*.yaml`

