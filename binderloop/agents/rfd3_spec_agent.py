"""Package a DesignJob into a runnable Foundry RFD3 project directory."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml

from binderloop.models.base import DesignJob
from binderloop.models.rfd3_adapter import RFD3Adapter, render_rfd3_pipeline_script
from binderloop.models.rfd3_renderer import with_default_foundry_artifacts
from binderloop.package_layout import PROJECT_PACKAGE_DIRNAME


@dataclass
class RFD3RunSpec:
    """Concrete runnable Foundry RFD3 spec produced by RFD3SpecAgent."""

    task_id: str
    job_id: str
    design_spec_path: str
    run_script_path: str
    output_dir: str
    log_file: str
    command: List[str]
    command_string: str
    expected_outputs: Dict[str, str]
    params: Dict[str, Any] = field(default_factory=dict)
    package_dir: Optional[str] = None


class RFD3SpecAgent:
    """Translate DesignJob + selected parameters into a Foundry three-step run."""

    DEFAULT_STEPS = ["design", "inverse_folding", "folding"]

    def __init__(
        self,
        foundry_root: Union[str, Path] = "models/foundry",
        *,
        weights_path: Union[str, Optional[Path]] = None,
    ):
        self.foundry_root = Path(foundry_root)
        self.weights_path = Path(weights_path).expanduser() if weights_path else None
        self.adapter = RFD3Adapter(str(self.foundry_root))

    def create_rfd3_run_spec(
        self,
        job: DesignJob,
        *,
        params: Optional[Mapping[str, Any]] = None,
        params_mode: str = "overlay",
        execution_identity_context: Optional[Mapping[str, Any]] = None,
        conda_base: str = "/data/miniconda3",
        conda_env_name: str = "foundry",
    ) -> RFD3RunSpec:
        if params_mode not in {"overlay", "replacement"}:
            raise ValueError(f"unsupported params_mode: {params_mode}")
        supplied = dict(params or {})
        if params_mode == "replacement":
            merged = dict(supplied)
        else:
            merged = dict(job.params or {})
            merged.update(supplied)
        merged = with_default_foundry_artifacts(merged, weights_path=self.weights_path)
        from binderloop.models.search_profile import get_model_search_profile
        merged = get_model_search_profile("rfd3").filter_params(merged).params
        merged.setdefault("steps", list(self.DEFAULT_STEPS))
        merged.setdefault("conda_base", conda_base)
        merged.setdefault("conda_env_name", conda_env_name)

        run_root = Path(job.output_dir)
        package_dir = Path(merged.get("package_dir") or run_root / PROJECT_PACKAGE_DIRNAME)
        input_dir = package_dir / "inputs"
        config_dir = package_dir / "configs"
        script_dir = package_dir / "scripts"
        output_dir = package_dir / "outputs" / "rfd3_output"
        log_dir = package_dir / "logs"
        for path in (input_dir, config_dir, script_dir, output_dir, log_dir):
            path.mkdir(parents=True, exist_ok=True)

        source_target = Path(job.target_structure)
        packaged_target = input_dir / source_target.name
        if source_target.exists():
            shutil.copy2(source_target, packaged_target)
        else:
            packaged_target.write_text("", encoding="utf-8")

        packaged_job = DesignJob(
            job_id=job.job_id,
            target_structure=str(packaged_target.resolve()),
            chain_id=job.chain_id,
            hotspots=list(job.hotspots),
            binder_length=job.binder_length,
            seed=job.seed,
            params=dict(merged),
            output_dir=str(config_dir),
        )
        packaged_job.params["target_path_for_spec"] = str(packaged_target.resolve())
        spec_path = self.adapter.write_design_spec(packaged_job)
        # Keep the spec next to other configs even if adapter wrote it in config_dir.
        if spec_path.parent != config_dir:
            dest = config_dir / spec_path.name
            shutil.copy2(spec_path, dest)
            spec_path = dest

        design_dir = output_dir / "design"
        inverse_fold_dir = output_dir / "inverse_fold"
        fold_dir = output_dir / "fold"
        metrics_path = output_dir / "final_designs_metrics.csv"
        mpnn_config_path = config_dir / "mpnn_inputs.json"
        bridge_path = script_dir / "rfd3_step_bridge.py"
        shutil.copy2(Path(__file__).resolve().parents[1] / "models" / "rfd3_step_bridge.py", bridge_path)
        log_file = log_dir / "rfd3_full.log"
        script_path = script_dir / "run_rfd3_pipeline.sh"
        script_path.write_text(
            render_rfd3_pipeline_script(
                spec_path=spec_path,
                design_dir=design_dir,
                inverse_fold_dir=inverse_fold_dir,
                fold_dir=fold_dir,
                metrics_path=metrics_path,
                mpnn_config_path=mpnn_config_path,
                bridge_path=bridge_path,
                params=merged,
                conda_base=conda_base,
                conda_env_name=conda_env_name,
                log_file=log_file,
            ),
            encoding="utf-8",
        )
        script_path.chmod(script_path.stat().st_mode | 0o111)

        param_path = config_dir / "rfd3_parameter_plan.yaml"
        param_path.write_text(yaml.safe_dump(dict(merged), allow_unicode=True), encoding="utf-8")
        command = ["bash", str(script_path)]
        expected_outputs = {
            "package_dir": str(package_dir),
            "target_file": str(packaged_target),
            "rfd3_output_dir": str(output_dir),
            "design_dir": str(design_dir),
            "inverse_fold_dir": str(inverse_fold_dir),
            "fold_dir": str(fold_dir),
            "metrics": str(metrics_path),
            "log_file": str(log_file),
            "design_spec": str(spec_path),
        }
        spec = RFD3RunSpec(
            task_id=str(merged.get("task_id") or job.job_id),
            job_id=job.job_id,
            design_spec_path=str(spec_path),
            run_script_path=str(script_path),
            output_dir=str(output_dir),
            log_file=str(log_file),
            command=command,
            command_string=" ".join(command),
            expected_outputs=expected_outputs,
            params=dict(merged),
            package_dir=str(package_dir),
        )
        identity = dict(execution_identity_context or {})
        (package_dir / "rfd3_run_manifest.json").write_text(
            json.dumps({"run_spec": asdict(spec), "execution_identity": identity}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return spec
