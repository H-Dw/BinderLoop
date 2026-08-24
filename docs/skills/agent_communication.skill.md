---
name: binder-agent-communication
version: 0.1.0
description: JSONL protocol for Binder harness agents to exchange observations, failures, hypotheses, proposals, and decisions.
---

# Binder Agent Communication Skill

`AgentMessage` envelope: sender, recipient, message_type, round_id, optional job_id, correlation_id, parent_id, content, confidence, artifacts. Messages are append-only JSONL. Store evidence and decisions, not hidden reasoning or secrets.
