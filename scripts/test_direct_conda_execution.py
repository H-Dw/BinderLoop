#!/usr/bin/env python3
"""Direct Conda execution tests that never launch BoltzGen or RFD3."""

import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binderloop.config import ConfigError, ResourceSpec, load_config
from binderloop.models.base import DesignJob
from binderloop.orchestration.runner import CommandResult, conda_run_command
from binderloop.pipeline import run_pipeline
from scripts.run_closed_loop_orchestrator import _apply_execution_args, build_job_executor


CONFIG_PATH = ROOT / "configs" / "pdl1_structured_task_notemp_iptm035_simple.yaml"


def _args(**overrides) -> argparse.Namespace:
    values = {
        "backend": None,
        "submit": False,
        "conda_base": None,
        "conda_executable": None,
        "conda_env_name": None,
        "checkpoint_dir": None,
        "cache_dir": None,
        "moldir": None,
        "llm_model": None,
        "llm_thinking": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_direct_is_default_and_taiji_requires_an_explicit_selection() -> None:
    assert ResourceSpec().backend == "direct"
    cfg = load_config(CONFIG_PATH)
    args = _args(submit=True)
    _apply_execution_args(cfg, args)
    assert cfg.resource.backend == "direct"

    taiji_args = _args(backend="taiji")
    _apply_execution_args(cfg, taiji_args)
    assert cfg.resource.backend == "taiji"


def test_model_runtime_config_resolves_conda_environments_and_weight_roots() -> None:
    cfg = load_config(CONFIG_PATH)
    boltzgen = cfg.runtime.model_runtime("boltzgen")
    rfd3 = cfg.runtime.model_runtime("rfd3")
    assert (boltzgen.conda_env, boltzgen.weights_path) == (
        "bg",
        "/data1/dhuang/boltzgen/cache",
    )
    assert (rfd3.conda_env, rfd3.weights_path) == (
        "foundry",
        "/data1/dhuang/foundry/checkpoints",
    )
    assert conda_run_command(["rfd3", "design"], env_name=rfd3.conda_env) == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "foundry",
        "rfd3",
        "design",
    ]


def test_direct_rfd3_dry_run_uses_foundry_conda(tmp_path: Path) -> None:
    cfg = load_config(ROOT / "configs" / "example_rfd3_binder_task.yaml")
    args = _args(backend="direct")
    _apply_execution_args(cfg, args)
    assert args.conda_env_name == "foundry"
    assert args.target_model == "rfd3"
    executor = build_job_executor(cfg, root=ROOT, args=args, llm_config_path=None)
    assert executor is not None
    params = dict(cfg.search_space.rfd3)
    job = DesignJob(
        job_id="direct_rfd3_contract",
        target_structure=str(ROOT / "examples" / "bg_example" / "PD-L1.cif"),
        chain_id="A",
        hotspots=list(cfg.target.hotspots),
        binder_length=50,
        params=params,
        output_dir=str(tmp_path / "rfd3_job"),
    )
    record = executor(job, 1)
    assert record["backend"] == "direct"
    assert record["status"] == "dry_run"
    assert record["execution_command"][:6] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "foundry",
        "bash",
    ]
    script = Path(record["run_spec"]["run_script_path"]).read_text(encoding="utf-8")
    assert "rfd3 design" in script
    assert "protein_mpnn" in script
    assert "rf3 fold" in script
    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-foundry}"' in script


def test_direct_boltzgen_dry_run_contains_conda_and_weight_arguments(tmp_path: Path) -> None:
    cfg = load_config(CONFIG_PATH)
    args = _args()
    _apply_execution_args(cfg, args)
    executor = build_job_executor(cfg, root=ROOT, args=args, llm_config_path=None)
    assert executor is not None

    params = dict(cfg.search_space.boltzgen)
    params["devices"] = 1
    job = DesignJob(
        job_id="direct_config_contract",
        target_structure=str(ROOT / "examples" / "bg_example" / "PD-L1.cif"),
        chain_id="A",
        hotspots=[],
        binder_length=50,
        params=params,
        output_dir=str(tmp_path / "job"),
    )
    record = executor(job, 1)

    assert record["backend"] == "direct"
    assert record["status"] == "dry_run"
    assert record["execution_command"][:6] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "bg",
        "bash",
    ]
    command = record["run_spec"]["command"]
    command_string = record["run_spec"]["command_string"]
    weight_root = "/data1/dhuang/boltzgen/cache"
    assert command[command.index("--cache") + 1] == weight_root
    assert f"{weight_root}/boltzgen1_diverse.ckpt" in command_string
    assert f"{weight_root}/boltzgen1_ifold.ckpt" in command_string
    assert f"{weight_root}/boltz2_conf_final.ckpt" in command_string
    assert f"{weight_root}/boltz2_aff.ckpt" in command_string
    script = Path(record["run_spec"]["run_script_path"]).read_text(encoding="utf-8")
    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-bg}"' in script
    assert 'CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data1/dhuang/boltzgen/cache}"' in script


def test_model_runtime_rejects_unknown_fields(tmp_path: Path) -> None:
    config = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "          weights_path: /data1/dhuang/boltzgen/cache",
        "          weights_path: /data1/dhuang/boltzgen/cache\n          unsupported: true",
        1,
    )
    path = tmp_path / "bad.yaml"
    path.write_text(config, encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported"):
        load_config(path)


def test_strategy_pipeline_uses_the_same_direct_conda_contract(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config(CONFIG_PATH)
    cfg.runtime.output_dir = str(tmp_path / "pipeline")
    cfg.resource.backend = "direct"
    cfg.search_space.model_order = ["boltzgen"]

    def capture(_runner, name, command, cwd):
        return CommandResult(name=name, command=command, cwd=str(cwd), returncode=None, dry_run=True)

    monkeypatch.setattr("binderloop.pipeline.Runner.run", capture)
    results = run_pipeline(cfg)
    assert results
    assert results[0].command[:5] == ["conda", "run", "--no-capture-output", "-n", "bg"]
    assert "/data1/dhuang/boltzgen/cache" in results[0].command
