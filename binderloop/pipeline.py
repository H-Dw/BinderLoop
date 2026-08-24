
from pathlib import Path
from typing import Dict, List

from .config import HarnessConfig, binder_generation_cap, primary_design_model
from .models.base import DesignJob
from .models.boltzgen_adapter import BoltzGenAdapter
from .models.odesign_adapter import ODesignAdapter
from .models.rfd3_adapter import RFD3Adapter
from .models.rfd3_renderer import with_default_foundry_artifacts
from .models.search_profile import isolate_model_params
from .active_learning.strategy import StrategyLevelActiveLearner
from .orchestration.runner import Runner, CommandResult, conda_run_command


def _model_params_with_runtime(cfg: HarnessConfig, model_name: str, params: Dict) -> Dict:
    resolved = dict(params or {})
    model_runtime = cfg.runtime.model_runtime(model_name)
    if model_name == "boltzgen":
        weight_root = model_runtime.weights_path
        if model_runtime.checkpoint_dir or weight_root:
            resolved.setdefault("checkpoint_dir", model_runtime.checkpoint_dir or weight_root)
        if model_runtime.cache_dir or weight_root:
            resolved.setdefault("cache", model_runtime.cache_dir or weight_root)
        if model_runtime.moldir:
            resolved.setdefault("moldir", model_runtime.moldir)
    elif model_name == "rfd3":
        weight_root = model_runtime.weights_path or model_runtime.checkpoint_dir
        if weight_root:
            resolved.setdefault("weights_path", weight_root)
            resolved.setdefault("checkpoint_dir", model_runtime.checkpoint_dir or weight_root)
        resolved.setdefault("conda_env_name", model_runtime.conda_env)
        resolved.setdefault("conda_base", cfg.runtime.conda_base)
        resolved = with_default_foundry_artifacts(resolved, weights_path=weight_root)
    return resolved


def run_pipeline(cfg: HarnessConfig) -> List[CommandResult]:
    out = Path(cfg.runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    backend = str(cfg.resource.backend or "direct").lower()
    if backend == "taiji":
        raise ValueError("the strategy pipeline does not submit Taiji jobs; use scripts/run_closed_loop_orchestrator.py")
    if backend not in {"direct", "local", "dry_run"}:
        raise ValueError(f"unsupported resource backend: {backend}")
    runner = Runner(dry_run=backend == "dry_run")
    learner = StrategyLevelActiveLearner(exploration_ratio=cfg.active_learning.exploration_ratio)
    cap = binder_generation_cap(cfg)
    primary = primary_design_model(cfg)
    base_params: Dict = {"num_designs_per_round": min(cfg.search_space.num_designs_per_round, cap), "num_designs": min(cfg.search_space.num_designs_per_round, cap), "max_binders_per_round": cap}
    base_params.update(isolate_model_params(cfg, primary, {}))
    jobs = learner.initial_jobs(
        target_structure=cfg.target.structure_path,
        chain_id=cfg.target.chain_id,
        hotspots=cfg.target.hotspots,
        lengths=cfg.search_space.binder_lengths,
        output_dir=str(out),
        base_params=base_params,
    )
    adapters = {
        "boltzgen": BoltzGenAdapter(cfg.runtime.boltzgen_root, cfg.runtime.python_bin),
        "odesign": ODesignAdapter(cfg.runtime.odesign_root, cfg.runtime.python_bin),
        "rfd3": RFD3Adapter(getattr(cfg.runtime, "foundry_root", "models/foundry"), cfg.runtime.python_bin),
    }
    results: List[CommandResult] = []
    selected_jobs = jobs[: min(len(jobs), cap)]
    per_job_budget = max(1, cap // max(1, len(selected_jobs)))
    for job in selected_jobs:
        job.params["num_designs"] = min(per_job_budget, int(job.params.get("num_designs", per_job_budget)))
        job.params["num_designs_per_round"] = job.params["num_designs"]
        for model_name in cfg.search_space.model_order:
            adapter = adapters[model_name]
            model_params = _model_params_with_runtime(cfg, model_name, isolate_model_params(cfg, model_name, job.params))
            model_job = DesignJob(**{**job.__dict__, "params": model_params, "output_dir": f"{job.output_dir}/{model_name}"})
            cmd = adapter.build_command(model_job)
            if backend == "direct":
                model_runtime = cfg.runtime.model_runtime(model_name)
                cmd = conda_run_command(
                    cmd,
                    env_name=model_runtime.conda_env,
                    conda_executable=cfg.runtime.conda_executable,
                )
            results.append(runner.run(f"{model_name}:{model_job.job_id}", cmd, cwd=adapter.root))
    Runner.save_results(results, out / "commands.json")
    return results
