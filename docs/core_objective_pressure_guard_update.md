# Core Objective and Pressure Guard Update

This update makes the closed-loop harness prioritize the core binder-design
metrics consistently across candidate ranking, rollback, LLM context, pressure
conflict handling, and crop semantics.

## Why

Previous runs could improve secondary interface proxies such as PLIP H-bonds and
refolded delta-SASA while the main success metrics still oscillated. The root
cause was that candidate-level gates emphasized core metrics, but later loop
decisions still reacted strongly to hotspot/contact/SASA signals through policy
agents, strategy arms, crop proposals, and binding-site filters.

## Main Changes

- Added a shared `core_objective` definition:
  `0.35*iPTM + 0.25*PAE_score + 0.25*pTM + 0.15*RMSD_score`.
- Changed candidate scoring so H-bonds and SASA are only a tiny tie-breaker
  (`0.04`) after the core objective.
- Changed rollback reward to use the unified core objective instead of only
  top-k/best iPTM plus success count. Secondary H-bond/SASA proxies are excluded
  from rollback and best-round selection.
- Added `top_by_core`, core metric stats, and core metric trends to compacted
  LLM contexts so agents see PAE, pTM, RMSD, and the composite core objective.
- Expanded pressure conflict detection to include:
  `prioritize_hotspots`, `filter_bindingsite=true`, target crop narrowing,
  target binding residue expansion, template fraction increases,
  `module_guided_repair + clash_filter`, and binder length narrowing.
- Expanded pressure conflict resolution to remove or revert those pressure
  moves toward the best core-objective round.
- Added job-level pressure conflict resolution after strategy-arm proposal so
  strategy arms cannot reintroduce hotspot/contact pressure after merge-level
  resolution.
- Updated strategy parent selection to rank exploitation parents by core
  objective first, then iPTM, with secondary contacts only as a final tie-breaker.
- Treated `epitope_crop_mode: disabled` in the user YAML as a default hard
  constraint unless `allow_agent_epitope_crop=true` is explicitly set.
- Preserved immutable original `target_include`, `target_binding_types`, and
  `structure_groups`, and restored them whenever crop mode is disabled.

## Expected Effect

The loop should now stop rewarding interface-size/contact improvements when
they come at the expense of ipTM, design-to-target PAE, pTM, or refolding RMSD.
Hotspot and SASA signals still help choose between otherwise similar core
candidates, but they no longer steer rollback, best-round selection, or conflict
recovery.
