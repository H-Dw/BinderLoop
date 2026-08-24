#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.config import load_config
from binderloop.pipeline import run_pipeline

cfg = load_config("configs/example_binder_task.yaml")
cfg.resource.backend = 'dry_run'
results = run_pipeline(cfg)
assert results, "No commands generated"
assert any("boltzgen" in r.name for r in results), "BoltzGen command missing"
if "odesign" in cfg.search_space.model_order:
    assert any("odesign" in r.name for r in results), "ODesign command missing"
print(f"OK: generated {len(results)} dry-run commands. See {cfg.runtime.output_dir}/commands.json")

rfd3_cfg = load_config("configs/example_rfd3_binder_task.yaml")
rfd3_cfg.resource.backend = "dry_run"
rfd3_results = run_pipeline(rfd3_cfg)
assert rfd3_results, "No RFD3 commands generated"
assert any("rfd3" in r.name for r in rfd3_results), "RFD3 command missing"
print(f"OK: generated {len(rfd3_results)} RFD3 dry-run commands. See {rfd3_cfg.runtime.output_dir}/commands.json")
