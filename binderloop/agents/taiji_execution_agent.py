
import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from binderloop.secrets import redact_sensitive

from .design_spec_agent import BoltzGenRunSpec
from binderloop.package_layout import LEGACY_PROJECT_PACKAGE_DIRNAMES, PROJECT_PACKAGE_DIRNAME


_PLACEHOLDER = object()


def shlex_quote(value: str) -> str:
    return shlex.quote(value)


@dataclass
class TaijiSubmitSpec:
    task_id: str
    task_flag: str
    simple_config_path: str
    submit_command: str
    run_script_path: str
    output_dir: str
    full_config_path: Optional[str] = None
    simple_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaijiSubmissionRecord:
    task_id: str
    task_flag: str
    simple_config_path: str
    submit_command: str
    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    taiji_job_id: Optional[str] = None
    dry_run: bool = True


class TaijiExecutionAgent:
    """Create Taiji simple-config JSON and submit BoltzGen jobs via taiji_client.

    Submission command follows the project example:
    ``taiji_client start -scfg /path/to/json_file.json``.
    """

    def __init__(self, taiji_client_bin: str = "taiji_client", dry_run: bool = True):
        self.taiji_client_bin = taiji_client_bin
        self.dry_run = dry_run

    def create_boltzgen_taiji_spec(
        self,
        run_spec: BoltzGenRunSpec,
        *,
        template_json: Union[str, Optional[Path]] = None,
        output_json: Union[str, Optional[Path]] = None,
        task_flag: Optional[str] = None,
        taiji_options: Optional[Mapping[str, Any]] = None,
    ) -> TaijiSubmitSpec:
        taiji_options = dict(taiji_options or {})
        submit_with_full_config = bool(taiji_options.pop("submit_with_full_config", False))
        template = self._load_template(template_json)
        task_flag = task_flag or taiji_options.get("task_flag") or f"binder_boltzgen_{run_spec.task_id}"

        config = dict(template)
        config.update({k: v for k, v in taiji_options.items() if v is not None})
        config["task_flag"] = task_flag
        config.setdefault("readable_name", task_flag)
        config.setdefault("host_num", 1)
        config.setdefault("host_gpu_num", int(run_spec.params.get("devices") or 1))
        config.setdefault("GPUName", run_spec.params.get("GPUName", "V100"))
        config.setdefault("is_resource_waiting", True)
        config.setdefault("is_elasticity", False)
        config.setdefault("enable_evicted_end_task", True)
        config.setdefault("exec_start_in_all_mpi_pods", True)
        config.setdefault("priority_level", "HIGH")
        config.setdefault("quota_type", "public")
        config.setdefault("envs", {})
        config["envs"]["TAIJI_TIMEOUT"] = str(run_spec.params.get("taiji_timeout", 3600))
        config["envs"]["NUM_DESIGNS"] = str(run_spec.params.get("num_designs", ""))
        config["envs"].setdefault("HARNESS_HOST_COUNT", str(config.get("host_num", 1)))
        config["envs"].setdefault("HARNESS_GPUS_PER_HOST", str(config.get("host_gpu_num", 1)))
        config["envs"].setdefault(
            "HARNESS_MULTI_HOST_MODE",
            str(run_spec.params.get("taiji_multi_host_mode") or "native"),
        )
        config["envs"].setdefault("HARNESS_RUN_TOKEN", str(task_flag))

        prefix_cmd = config.pop("start_cmd_prefix", taiji_options.get("start_cmd_prefix", ""))
        remote_project_dir = config.pop("remote_project_dir", taiji_options.get("remote_project_dir"))
        package_dir = run_spec.package_dir or str(Path(run_spec.run_script_path).parents[1])
        run_script_rel = str(Path(run_spec.run_script_path).relative_to(package_dir))
        # The package is uploaded as the task code/model directory; run from package root
        # so inputs/configs/outputs are all project-local and reproducible.
        config["start_cmd"] = self._build_start_cmd(
            run_script_rel=run_script_rel,
            prefix_cmd=prefix_cmd,
            remote_project_dir=remote_project_dir,
        )

        if not config.get("model_local_file_path") and not config.get("model_id"):
            config["model_local_file_path"] = package_dir
        config.setdefault("model_local_path_exclude", ".git/*,__pycache__/*,*.pyc")

        # Taiji v2 simple config. If dataset_id/model_id are provided, use the stable
        # v2 image/code-path path. Otherwise keep model_local_file_path for local upload.
        if config.get("version") == "v2.0":
            if config.get("dataset_id") and config.get("model_id"):
                config.pop("model_local_file_path", None)
            else:
                config.setdefault("dataset_path", "/")
                config.setdefault("code_path", config.get("model_local_file_path", package_dir))
            config.pop("tensorboard_business_flag", None)
            config.pop("tensorboard_container_path", None)
            config.pop("tensorboard_custom_path", None)

        output_json = Path(output_json or Path(run_spec.run_script_path).with_name("taiji_simple_config.json"))
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(config, ensure_ascii=False, indent=4), encoding="utf-8")
        output_json.with_name(output_json.stem + ".redacted.json").write_text(
            json.dumps(redact_sensitive(config), ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

        full_config_path = None
        if submit_with_full_config:
            full_config_path = str(output_json.with_name("taiji_full_config.json"))
            self._write_full_config(Path(full_config_path), config, run_spec=run_spec, package_dir=package_dir)

        submit_target = full_config_path or str(output_json)
        submit_flag = "-cfg" if full_config_path else "-scfg"
        submit_command = f"{self.taiji_client_bin} start {submit_flag} {submit_target}"
        manifest = TaijiSubmitSpec(
            task_id=run_spec.task_id,
            task_flag=task_flag,
            simple_config_path=str(output_json),
            submit_command=submit_command,
            run_script_path=run_spec.run_script_path,
            output_dir=run_spec.output_dir,
            full_config_path=full_config_path,
            simple_config=config,
        )
        manifest_data = asdict(manifest)
        manifest_data["simple_config"] = redact_sensitive(manifest_data.get("simple_config", {}))
        Path(run_spec.run_script_path).with_name("taiji_submit_manifest.json").write_text(
            json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def submit(self, spec: TaijiSubmitSpec, *, dry_run: Optional[bool] = None) -> TaijiSubmissionRecord:
        dry_run = self.dry_run if dry_run is None else dry_run
        if spec.full_config_path:
            command = [self.taiji_client_bin, "start", "-cfg", spec.full_config_path]
        else:
            command = [self.taiji_client_bin, "start", "-scfg", spec.simple_config_path]
        if dry_run:
            record = TaijiSubmissionRecord(
                task_id=spec.task_id,
                task_flag=spec.task_flag,
                simple_config_path=spec.simple_config_path,
                submit_command=" ".join(command),
                returncode=None,
                dry_run=True,
            )
        else:
            proc = subprocess.run(
                command,
                universal_newlines=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            combined = f"{proc.stdout}\n{proc.stderr}"
            returncode = proc.returncode
            if returncode == 0 and self._looks_like_taiji_error(combined):
                returncode = 1
            record = TaijiSubmissionRecord(
                task_id=spec.task_id,
                task_flag=spec.task_flag,
                simple_config_path=spec.simple_config_path,
                submit_command=" ".join(command),
                returncode=returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                taiji_job_id=self._extract_job_id(combined),
                dry_run=False,
            )
        Path(spec.simple_config_path).with_name("taiji_submission_record.json").write_text(
            json.dumps(redact_sensitive(asdict(record)), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return record

    @staticmethod
    def _load_template(template_json: Union[str, Optional[Path]]) -> Dict[str, Any]:
        if template_json is None:
            return {}
        with open(template_json, "r", encoding="utf-8") as f:
            return TaijiExecutionAgent._drop_placeholder_values(json.load(f))

    @staticmethod
    def _drop_placeholder_values(obj: Any) -> Any:
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                value = TaijiExecutionAgent._drop_placeholder_values(value)
                if value is _PLACEHOLDER:
                    continue
                cleaned[key] = value
            return cleaned
        if isinstance(obj, list):
            return [
                value
                for item in obj
                for value in [TaijiExecutionAgent._drop_placeholder_values(item)]
                if value is not _PLACEHOLDER
            ]
        if isinstance(obj, str) and obj.startswith("<SET_") and obj.endswith(">"):
            return _PLACEHOLDER
        return obj

    @staticmethod
    def _looks_like_taiji_error(text: str) -> bool:
        lowered = text.lower()
        return "[error]" in lowered or "traceback" in lowered or "exception" in lowered

    @staticmethod
    def _build_start_cmd(
        *,
        run_script_rel: str,
        prefix_cmd: str = "",
        remote_project_dir: Optional[str] = None,
    ) -> str:
        workspace_expr = shlex_quote(remote_project_dir) if remote_project_dir else '"${JIZHI_WORKSPACE_PATH:-$(pwd)}"'
        script = f"""
set -euo pipefail
{prefix_cmd}
mkdir -p /aceph/daweihuang
if ! mountpoint -q /aceph/daweihuang 2>/dev/null; then
  if [[ -n "${{CEPH_SECRET:-}}" ]]; then
    mount -t ceph 11.18.83.17:6789,11.18.83.31:6789,11.18.83.32:6789:/fandiwu/buddy1/daweihuang \\
      /aceph/daweihuang -o name=fandiwubuddy1,secret="${{CEPH_SECRET}}"
  else
    echo "[HARNESS][ERROR] /aceph/daweihuang is not mounted and CEPH_SECRET is not set" >&2
    exit 16
  fi
fi
WORKSPACE={workspace_expr}
cd "$WORKSPACE"
if [[ ! -f {shlex_quote(run_script_rel)} ]]; then
  if [[ -f {shlex_quote(PROJECT_PACKAGE_DIRNAME + "/" + run_script_rel)} ]]; then
    cd {shlex_quote(PROJECT_PACKAGE_DIRNAME)}
  elif [[ -f {shlex_quote(LEGACY_PROJECT_PACKAGE_DIRNAMES[0] + "/" + run_script_rel)} ]]; then
    cd {shlex_quote(LEGACY_PROJECT_PACKAGE_DIRNAMES[0])}
  fi
fi
echo "[HARNESS] workspace=$(pwd)"
bash {shlex_quote(run_script_rel)}
"""
        return "bash -lc " + shlex_quote(script)

    @staticmethod
    def _write_full_config(path: Path, config: Dict[str, Any], *, run_spec: BoltzGenRunSpec, package_dir: str) -> Path:
        common: Dict[str, Any] = {
            "business_flag": config["business_flag"],
            "readable_name": config.get("readable_name", config["task_flag"]),
            "task_flag": config["task_flag"],
        }
        if config.get("dataset_id"):
            common["dataset_id"] = config["dataset_id"]
        else:
            common["dataset_params"] = {
                "dataset_name": config["task_flag"],
                "dataset_source": "plat_ceph",
                "path_info": {"path": config.get("dataset_path", "/")},
            }

        if config.get("model_id"):
            common["model_id"] = config["model_id"]
        else:
            common["model_params"] = {
                "model_name": config["task_flag"],
                "model_path_info": {
                    "model_source": "local_upload",
                    "path_info": {"file_path": config.get("model_local_file_path", package_dir)},
                },
            }

        designated_resource = {
            "strategy": {},
            "host_gpu_num": config.get("host_gpu_num", 1),
            "GPUName": config.get("GPUName", "V100"),
            "quota_type": config.get("quota_type", "public"),
            "host_num": config.get("host_num", 1),
            "gpu_configuration": 0,
            "is_resource_waiting": config.get("is_resource_waiting", True),
            "image_full_name": config["image_full_name"],
            "is_elasticity": config.get("is_elasticity", False),
            "is_enable_host_network": False,
            "is_enable_ssh_without_password": True,
            "is_enable_elastic_evicted_pulled_up": config.get("enable_evicted_pulled_up", False),
            "is_enable_fault_tolerance": True,
            "keep_running_after_trainer_finish": False,
            "is_store_core_file": False,
            "core_file_store_relative_path": "",
            "is_enable_node_evicted_end_task": config.get("enable_evicted_end_task", True),
            "cuda_version": config.get("cuda_version", "11.0"),
            "is_enable_rdma": False,
            "is_enable_ceph_metadata_cache": False,
            "rdma_in_same_module": False,
            "extra_plat_business": "",
            "mount_ceph_business_flag": "",
            "is_mount_dpd_ceph": False,
            "priority_level": config.get("priority_level", "HIGH"),
            "elastic_level": config.get("elastic_level", 1),
            "min_host_num": 0,
            "max_host_num": 0,
            "min_gpu_num": 0,
            "max_gpu_num": 0,
            "is_tidal_task": False,
            "is_mount_ceph": False,
            "wwfs_token": "",
            "wwfs_host_path": "",
            "wwfs_mount_path": "",
            "location": config.get("location", ""),
            "enable_mixing_offline": False,
        }
        task_params = {
            "task_type": "general_gpu_type",
            "common": common,
            "project_id": config["project_id"],
            "task_config": {"designated_resource": designated_resource},
            "job_config": {
                "start_cmd": config["start_cmd"],
                "exec_start_in_all_mpi_pods": config.get("exec_start_in_all_mpi_pods", True),
                "report_period": 60,
            },
            "job_type": "mpijob",
            "env_vars_dict": config.get("envs", {}),
        }
        token = config.get("Token", "")
        path.write_text(
            json.dumps({"Token": token, "task_params": task_params}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _redact_sensitive(obj: Any) -> Any:
        if isinstance(obj, dict):
            redacted = {}
            for key, value in obj.items():
                key_lower = str(key).lower()
                if any(s in key_lower for s in ["token", "secret", "password", "credential", "key"]):
                    redacted[key] = "<REDACTED>"
                else:
                    redacted[key] = TaijiExecutionAgent._redact_sensitive(value)
            return redacted
        if isinstance(obj, list):
            return [TaijiExecutionAgent._redact_sensitive(v) for v in obj]
        return obj

    @staticmethod
    def _extract_job_id(text: str) -> Optional[str]:
        patterns = [
            r"instance[_ -]?id[:=\s]+([A-Za-z0-9_.:-]+)",
            r"job[_ -]?id[:=\s]+([A-Za-z0-9_.:-]+)",
            # Keep task_id last: Taiji output also contains task_flag/task_id-like strings
            # that are not monitorable instance ids.
            r"task[_ -]?id[:=\s]+([A-Za-z0-9_.:-]+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return None
