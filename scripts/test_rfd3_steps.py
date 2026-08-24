#!/usr/bin/env python3
"""CPU-only tests for the Foundry RFD3 / ProteinMPNN / RF3 harness modules."""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binderloop.agents.config_parameter_contract import PARAM_BOUNDS
from binderloop.agents.config_validation_agent import ConfigValidationAgent
from binderloop.agents.model_input_spec import get_model_input_spec
from binderloop.agents.rfd3_spec_agent import RFD3SpecAgent
from binderloop.analysis.parsers import parse_rfd3_scores
from binderloop.config import load_config, primary_design_model
from binderloop.models.base import DesignJob
from binderloop.models.rfd3_adapter import RFD3Adapter, hotspot_to_rfd3_residue
from binderloop.models.rfd3_renderer import render_mpnn_command, render_rf3_fold_command
from binderloop.models.rfd3_step_bridge import write_metrics_csv, write_mpnn_config
from binderloop.orchestration.runner import conda_run_command
from binderloop.pipeline import run_pipeline


EXAMPLE = ROOT / "configs" / "example_rfd3_binder_task.yaml"


def _job(tmp_path: Path, **overrides) -> DesignJob:
    params = {
        "num_designs": 2,
        "diffusion_batch_size": 1,
        "n_batches": 2,
        "target_res_index": "17-132",
        "target_chain": "A",
        "binder_chain": "A",
        "select_hotspots": {"A56": "CG,OH", "A115": "CG,SD", "A123": "CD2,OH"},
        "weights_path": "/data1/dhuang/foundry/checkpoints",
        "inverse_fold_num_sequences": 1,
        "step_scale": 3.0,
        "gamma_0": 0.2,
    }
    params.update(overrides.pop("params", {}))
    return DesignJob(
        job_id="rfd3_unit",
        target_structure=str(ROOT / "examples" / "bg_example" / "PD-L1.cif"),
        chain_id="A",
        hotspots=["A:56", "A:115", "A:123"],
        binder_length=50,
        params=params,
        output_dir=str(tmp_path / "job"),
        **overrides,
    )


def test_hotspot_token_to_rfd3_residue() -> None:
    assert hotspot_to_rfd3_residue("A:56", "A") == "A56"
    assert hotspot_to_rfd3_residue("A/56", "B") == "A56"
    assert hotspot_to_rfd3_residue("56", "A") == "A56"


def test_design_spec_and_command(tmp_path: Path) -> None:
    adapter = RFD3Adapter(str(ROOT / "models" / "foundry"))
    job = _job(tmp_path)
    spec_path = adapter.write_design_spec(job)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    example = next(iter(spec.values()))
    assert example["contig"] == "50,/0,A1-116"
    assert example["infer_ori_strategy"] == "hotspots"
    assert example["is_non_loopy"] is True
    assert example["select_hotspots"]["A40"] == "CG,OH"
    assert example["select_hotspots"]["A99"] == "CG,SD"
    assert example["select_hotspots"]["A107"] == "CD2,OH"
    assert "A56" not in example["select_hotspots"]
    command = adapter.build_step_command(job, "design")
    assert command[:2] == ["rfd3", "design"]
    assert any(token.startswith("inputs=") for token in command)
    assert "inference_sampler.step_scale=3.0" in command
    assert "inference_sampler.gamma_0=0.2" in command
    assert "skip_existing=false" in command
    wrapped = conda_run_command(command, env_name="foundry")
    assert wrapped[:5] == ["conda", "run", "--no-capture-output", "-n", "foundry"]


def test_inverse_folding_uses_proteinmpnn(tmp_path: Path) -> None:
    adapter = RFD3Adapter()
    job = _job(tmp_path)
    command = adapter.build_step_command(job, "inverse_folding")
    assert command[0] == "mpnn"
    assert "--model_type" in command and "protein_mpnn" in command
    assert "--is_legacy_weights" in command
    assert command[command.index("--is_legacy_weights") + 1] == "True"
    assert "--config_json" in command
    config = json.loads(Path(command[command.index("--config_json") + 1]).read_text(encoding="utf-8"))
    assert config["model_type"] == "protein_mpnn"
    assert config["is_legacy_weights"] is True
    single = render_mpnn_command(
        params={"mpnn_checkpoint": "/tmp/proteinmpnn_v_48_020.pt", "designed_chains": "A"},
        out_directory=tmp_path / "mpnn",
        structure_path=tmp_path / "design.cif",
    )
    assert "--designed_chains" in single and "A" in single


def test_folding_uses_rf3(tmp_path: Path) -> None:
    adapter = RFD3Adapter()
    job = _job(tmp_path)
    command = adapter.build_step_command(job, "folding")
    assert command[:2] == ["rf3", "fold"]
    assert any(token.startswith("inputs=") for token in command)
    assert any(token.startswith("out_dir=") for token in command)
    fold = render_rf3_fold_command(inputs=tmp_path / "inverse_fold", output_dir=tmp_path / "fold", params={"rf3_checkpoint": "rf3"})
    assert "ckpt_path=rf3" in fold


def test_pipeline_dry_run_writes_three_step_script(tmp_path: Path) -> None:
    cfg = load_config(EXAMPLE)
    cfg.resource.backend = "dry_run"
    cfg.runtime.output_dir = str(tmp_path / "pipeline")
    results = run_pipeline(cfg)
    assert results
    assert any("rfd3" in item.name for item in results)
    command = results[0].command
    assert command[:2] == ["bash", command[1]]
    script = Path(command[1]).read_text(encoding="utf-8")
    assert "rfd3 design" in script
    assert "mpnn" in script and "protein_mpnn" in script
    assert "rf3 fold" in script
    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-foundry}"' in script


def test_config_contract_keeps_official_ppi_step_scale() -> None:
    cfg = load_config(EXAMPLE)
    assert primary_design_model(cfg) == "rfd3"
    assert cfg.runtime.model_runtime("rfd3").conda_env == "foundry"
    assert cfg.search_space.rfd3["step_scale"] == 3.0
    assert cfg.search_space.rfd3["gamma_0"] == 0.2
    assert float(PARAM_BOUNDS["step_scale"]["max"]) < 3.0
    result = ConfigValidationAgent().validate_for_submission(
        {"step_scale": 3.0, "gamma_0": 0.2, "model_type": "protein_mpnn", "steps": ["design", "inverse_folding", "folding"]},
        target_model="rfd3",
    )
    assert result.corrected_config["step_scale"] == 3.0
    assert result.corrected_config["model_type"] == "protein_mpnn"
    spec = get_model_input_spec("rfd3")
    assert spec.allowed_keys is None


def test_rfd3_spec_agent_packages_run_script(tmp_path: Path) -> None:
    agent = RFD3SpecAgent(ROOT / "models" / "foundry", weights_path="/data1/dhuang/foundry/checkpoints")
    spec = agent.create_rfd3_run_spec(_job(tmp_path), conda_env_name="foundry")
    assert Path(spec.run_script_path).exists()
    assert Path(spec.design_spec_path).exists()
    script = Path(spec.run_script_path).read_text(encoding="utf-8")
    assert "conda activate" in script
    assert 'command -v "$cli"' in script
    assert "rfd3 design" in script
    assert "rf3 fold" in script


def test_parse_rfd3_scores_from_bridge_csv(tmp_path: Path) -> None:
    write_metrics_csv(
        [{"design": "demo", "path": "x.json", "iptm": 0.7, "ptm": 0.8, "plddt": 85.0, "ranking_score": 0.9}],
        tmp_path / "final_designs_metrics.csv",
    )
    scores = parse_rfd3_scores(tmp_path, {"interface_confidence": 1.0, "binder_plddt": 0.0, "hotspot_contact": 0.0, "clash_penalty": 0.0, "diversity": 0.0, "sequence_designability": 0.0})
    assert scores
    assert scores[0].model == "rfd3"
    assert scores[0].interface_confidence == 0.7


def test_mpnn_config_json_lists_designs(tmp_path: Path) -> None:
    design = tmp_path / "demo_model_0.cif"
    design.write_text("cif")
    out = write_mpnn_config(
        [design],
        out_path=tmp_path / "mpnn.json",
        out_directory=tmp_path / "mpnn",
        params={"designed_chains": "A", "mpnn_checkpoint": "/tmp/proteinmpnn_v_48_020.pt"},
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["inputs"][0]["designed_chains"] == ["A"]
    assert payload["inputs"][0]["structure_path"] == str(design)


def test_smoke_script_isolated_fallbacks(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_rfd3_step_smoke", ROOT / "scripts" / "run_rfd3_step_smoke.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    cfg = load_config(EXAMPLE)
    adapter = RFD3Adapter(str(ROOT / "models" / "foundry"))
    job = module._job(tmp_path / "job", cfg, n_batches=1, diffusion_batch_size=1, num_timesteps=None)
    design = adapter.build_step_command(job, "design")
    assert design[:2] == ["rfd3", "design"]
    assert "n_batches=1" in design
    assert "diffusion_batch_size=1" in design
    spec_token = next(token for token in design if token.startswith("inputs="))
    sample = next(iter(yaml.safe_load(Path(spec_token.split("=", 1)[1]).read_text(encoding="utf-8")).values()))
    assert sample["contig"] == "50,/0,A1-116"
    assert sample["input"].endswith("PD-L1.cif")
    mpnn = module._command_for_step(adapter, job, "inverse_folding")
    assert "--structure_path" in mpnn
    assert str(module.OFFICIAL_MPNN_PDB) in mpnn
    fold = module._command_for_step(adapter, job, "folding")
    assert any(token.startswith(f"inputs={module.OFFICIAL_RF3_JSON}") for token in fold)
