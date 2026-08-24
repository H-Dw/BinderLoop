---
name: closed-loop-binder-orchestrator
version: 0.1.0
description: Global scheduler for automated multi-round Binder design with bounded parallelism, retry, memory, and dynamic re-planning.
---

# Closed-loop Binder Orchestrator Skill

Trigger when a Binder design experiment must run multiple iterations, persist trajectory, recover from failed jobs, or coordinate agents.

Procedure: create initial jobs; publish round status; execute independent jobs with bounded parallelism; retry failures up to `max_retries`; ingest outputs; run numeric and structure evaluation; generate hypotheses; propose next-round parameters; persist memory; re-plan until `max_rounds`.

Fallback: if no LLM Agent API is configured, call `HypothesisAgent` deterministic fallback rather than embedding ad-hoc rules in orchestration.
