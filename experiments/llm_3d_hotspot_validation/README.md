# LLM 3D Hotspot Validation

This directory contains a preregistered, label-isolated benchmark of LLM hotspot
prediction from target-only protein structures. Human-readable execution records
live in `process/`; human-readable analyses live in `results/`.

Ground-truth labels are deliberately absent until every prediction artifact has
been validated, hashed, and frozen.

## Conditions

- `named_no_web`: anonymous structure plus an identity and residue-map card; no web.
- `anonymous_no_web`: anonymous structure only; no web.
- `anonymous_generic_packet`: anonymous structure plus one frozen, target-agnostic
  generic-method packet; no live web during prediction.

Each target-condition cell has three fresh `gpt-5.6-sol` / `xhigh` runs. Every run
returns exactly three primary and three alternate residues.

Anonymous local chains use opaque IDs `T1`, `T2`, `T3`, ... in first-seen chain
order. Residue tokens therefore have the form `T<positive-int>:<positive-int>`,
for example `T1:7`; `L*` chain IDs are not part of this experiment contract.
