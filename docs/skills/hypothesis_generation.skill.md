---
name: binder-hypothesis-generation
version: 0.1.0
description: Generate failure-mode hypotheses and intervention options from multi-round Binder evidence.
---

# Binder Hypothesis Generation Skill

LLM mode uses OpenAI-compatible chat-completions and must return JSON hypotheses with `name`, `evidence`, `confidence`, `intervention`, `expected_signal_next_round`, and `risk`.

Fallback mode maps hotspot, folding, pose/interface, clash, and diversity-collapse evidence to deterministic interventions. Never log or store API keys.
