
import glob
import fnmatch
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from binderloop.secrets import redact_sensitive, redact_sensitive_text


TERMINAL_SUCCESS = {"success", "succeeded", "finished", "complete", "completed", "done", "ok", "end"}
TERMINAL_FAILURE = {"fail", "failed", "error", "timeout", "killed", "cancelled", "canceled", "evicted"}
# Expected-output keys that may legitimately be absent on a successful run and
# therefore must not, on their own, flip a real success into a failure (which
# would trigger an unnecessary resubmit):
#   * analysis_metrics_candidates - CSVs may be empty/absent when nothing passes.
#   * candidate_manifest          - optional enhanced lineage for four-stage
#                                   template diagnostics. Final candidate scoring
#                                   and matched-control utility remain evaluable
#                                   without it.
# When filtering is in the executed steps, final_ranked_designs is required:
# BoltzGen still creates that directory (and all_designs_metrics.csv) on a
# valid 0-pass run. A missing directory after exit 0 means analysis/filtering
# never ran.
OPTIONAL_EXPECTED_OUTPUTS = {"analysis_metrics_candidates", "candidate_manifest"}
RUNNING_STATES = {
    "ready",
    "running",
    "training_running",
    "queue",
    "queued",
    "pending",
    "waiting",
    "starting",
}


@dataclass
class RunStatusSnapshot:
    task_flag: str
    instance_id: Optional[str] = None
    state: str = "unknown"
    is_terminal: bool = False
    is_success: bool = False
    detail_stdout: str = ""
    detail_stderr: str = ""
    log_tail: str = ""
    expected_outputs: Dict[str, str] = field(default_factory=dict)
    missing_outputs: List[str] = field(default_factory=list)
    manifest_status: Dict[str, Any] = field(default_factory=dict)
    failure_hints: List[str] = field(default_factory=list)
    needs_followup: bool = False
    recommended_followup_seconds: int = 300


class RunMonitorAgent:
    """Check Taiji status, logs and expected BoltzGen outputs.

    The class intentionally performs one status check per call. Scheduling or
    repeated polling should be handled by the caller/cron, not by a sleep loop.
    """

    def __init__(
        self,
        taiji_client_bin: str = "taiji_client",
        *,
        log_host_index: int = 0,
        command_timeout_seconds: int = 60,
    ):
        self.taiji_client_bin = taiji_client_bin
        self.log_host_index = max(0, int(log_host_index))
        self.command_timeout_seconds = max(1, int(command_timeout_seconds))

    def check_once(
        self,
        *,
        task_flag: str,
        instance_id: Optional[str] = None,
        expected_outputs: Optional[Dict[str, str]] = None,
        tail: int = 80,
        simple_config_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> RunStatusSnapshot:
        expected_outputs = dict(expected_outputs or {})
        detail = self._run_taiji_detail(
            task_flag=task_flag,
            instance_id=instance_id,
            simple_config_path=simple_config_path,
            config_path=config_path,
        )
        state = self._infer_state(detail["stdout"] + "\n" + detail["stderr"])

        if instance_id is None:
            instance_id = self._infer_instance_id(detail["stdout"] + "\n" + detail["stderr"])

        log_tail = ""
        if instance_id:
            log_tail = self._run_taiji_logs(
                task_flag=task_flag,
                instance_id=instance_id,
                tail=tail,
                simple_config_path=simple_config_path,
                config_path=config_path,
            )
            if state == "unknown":
                state = self._infer_state(log_tail)

        failure_text = (log_tail or "") + "\n" + detail["stderr"]
        if not failure_text.strip():
            failure_text = detail["stdout"] + "\n" + detail["stderr"]
        exit_code = self._infer_user_exit_code(log_tail)
        if state == "end" and exit_code not in (None, 0):
            state = "failed"
        is_terminal = state in TERMINAL_SUCCESS or state in TERMINAL_FAILURE
        # Ceph wildcard scans are useful only after the producer has stopped.
        # Running/queued checks rely solely on Taiji state and logs.
        if is_terminal:
            missing_outputs, manifest_status = self._evaluate_expected_outputs(expected_outputs)
        else:
            missing_outputs, manifest_status = [], {"state": "not_checked_non_terminal", "diagnostics": []}
        failure_hints = self._failure_hints(failure_text, missing_outputs)
        is_success = self._is_success(state, exit_code, missing_outputs)
        if is_success:
            failure_hints = []
        needs_followup = state in RUNNING_STATES or state == "unknown"
        recommended_followup_seconds = 300 if state in {"pending", "queued", "queue", "waiting"} else 120

        snapshot = RunStatusSnapshot(
            task_flag=task_flag,
            instance_id=instance_id,
            state=state,
            is_terminal=is_terminal,
            is_success=is_success,
            detail_stdout=redact_sensitive_text(detail["stdout"]),
            detail_stderr=redact_sensitive_text(detail["stderr"]),
            log_tail=redact_sensitive_text(log_tail),
            expected_outputs=expected_outputs,
            missing_outputs=missing_outputs,
            manifest_status=manifest_status,
            failure_hints=failure_hints,
            needs_followup=needs_followup,
            recommended_followup_seconds=recommended_followup_seconds,
        )
        return snapshot

    def write_snapshot(self, snapshot: RunStatusSnapshot, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(redact_sensitive(asdict(snapshot)), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _run_taiji_detail(
        self,
        *,
        task_flag: str,
        instance_id: Optional[str],
        simple_config_path: Optional[str],
        config_path: Optional[str],
    ) -> Dict[str, str]:
        if instance_id:
            command = [self.taiji_client_bin, "instance_detail"]
            if config_path:
                command += ["-cfg", config_path]
            elif simple_config_path:
                command += ["-scfg", simple_config_path]
            command += [task_flag, instance_id]
        else:
            command = [self.taiji_client_bin, "instance_list", "-vn", "5"]
            if config_path:
                command += ["-cfg", config_path]
            elif simple_config_path:
                command += ["-scfg", simple_config_path]
            command += [task_flag]
        try:
            proc = subprocess.run(
                command,
                universal_newlines=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.command_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": _timeout_output(exc.stdout),
                "stderr": _timeout_output(exc.stderr)
                + f"\n[HARNESS][TAIJI] instance_detail timed out after {self.command_timeout_seconds}s",
                "returncode": "124",
            }
        return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": str(proc.returncode)}

    def _run_taiji_logs(
        self,
        *,
        task_flag: str,
        instance_id: str,
        tail: int,
        simple_config_path: Optional[str],
        config_path: Optional[str],
    ) -> str:
        command = [self.taiji_client_bin, "logs", "--tail", str(tail)]
        if config_path:
            command += ["-cfg", config_path]
        elif simple_config_path:
            command += ["-scfg", simple_config_path]
        command += [task_flag, instance_id]
        try:
            # Multi-host Taiji jobs prompt for a host index even without --follow.
            # Supplying the launcher index keeps monitoring non-interactive and
            # avoids SIGTTIN when the orchestrator itself runs as a background job.
            proc = subprocess.run(
                command,
                input=f"{self.log_host_index}\n",
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.command_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = _timeout_output(exc.stdout)
            error = _timeout_output(exc.stderr)
            timeout_message = f"[HARNESS][TAIJI] logs timed out after {self.command_timeout_seconds}s"
            return output + ("\n" + error if error else "") + "\n" + timeout_message
        return proc.stdout + ("\n" + proc.stderr if proc.stderr else "")

    @staticmethod
    def _infer_state(text: str) -> str:
        lowered = text.lower()
        # Prefer explicit Taiji state fields over broad keyword matching; logs and
        # configs can contain words such as TAIJI_TIMEOUT that are not task states.
        import re

        for pattern in [r'"state"\s*:\s*"([^"]+)"', r"\bstate[:=\s]+([A-Za-z_]+)"]:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return m.group(1).lower()
        for state in TERMINAL_FAILURE:
            if state in lowered:
                return state
        for state in TERMINAL_SUCCESS:
            if state in lowered:
                return state
        # Match longer/more specific queue states before generic running words.
        for state in ["pending", "queued", "queue", "waiting", "starting", "running"]:
            if state in lowered:
                return state
        return "unknown"

    @staticmethod
    def _infer_instance_id(text: str) -> Optional[str]:
        # Keep this parser intentionally broad because taiji_client output format varies.
        for pattern in [r"instance[_ -]?id[:=\s]+([A-Za-z0-9_.:-]+)", r"\b(ins[-_A-Za-z0-9:.]+)\b"]:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _is_success(state: str, exit_code: Optional[int], missing_outputs: List[str]) -> bool:
        """Decide run success without letting optional artifacts cause false failures.

        Success requires a terminal-success state and no *required* output missing.
        When the runner explicitly reports ``user exit code: 0`` we trust it even if
        only optional artifacts (e.g. analysis_metrics_candidates when nothing
        passes) are absent, so a genuinely successful job is not re-submitted just
        because a glob came back empty. Missing ``final_ranked_designs`` after a
        filtering run is incomplete execution, not a valid 0-pass.
        """
        if state not in TERMINAL_SUCCESS:
            return False
        required_missing = [key for key in missing_outputs if key not in OPTIONAL_EXPECTED_OUTPUTS]
        if required_missing:
            return False
        if exit_code == 0:
            return True
        return not missing_outputs

    @staticmethod
    def _infer_user_exit_code(text: str) -> Optional[int]:
        m = re.search(r"\buser exit code:\s*(-?\d+)", text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    @staticmethod
    def _missing_expected_outputs(expected_outputs: Dict[str, str]) -> List[str]:
        return RunMonitorAgent._evaluate_expected_outputs(expected_outputs)[0]

    @staticmethod
    def _evaluate_expected_outputs(expected_outputs: Dict[str, str]) -> tuple[List[str], Dict[str, Any]]:
        """Validate authoritative manifest inventory and concrete filesystem entities."""
        missing: List[str] = []
        diagnostics: List[str] = []
        output_root_value = expected_outputs.get("boltzgen_output_dir")
        output_root = Path(output_root_value) if output_root_value else None
        manifest_value = expected_outputs.get("result_manifest")
        manifest_path = Path(manifest_value) if manifest_value else output_root / "result_manifest.json" if output_root else None
        manifest, inventory = RunMonitorAgent._read_manifest(manifest_path)
        state = "missing_or_invalid" if manifest is None else "legacy_inventory"
        authoritative = False
        if manifest is not None:
            authoritative = isinstance(manifest.get("authoritative"), dict) and manifest.get("authoritative", {}).get("inventory") == "files"
            state = "authoritative" if authoritative else "legacy_inventory"
            status = manifest.get("status") or {}
            status_code = status.get("code") if isinstance(status, dict) else None
            diagnostics.append(f"result_manifest_status:{manifest.get('execution_status', 'unknown')}:{status_code}")

        for key, value in expected_outputs.items():
            if key == "result_manifest":
                if manifest is None:
                    missing.append(key)
                continue
            if inventory is not None and output_root is not None and RunMonitorAgent._path_is_within_output_root(value, output_root):
                listed = RunMonitorAgent._manifest_contains(value, output_root, inventory)
                exists = RunMonitorAgent._manifest_entities_exist(value, output_root, inventory)
                if not listed:
                    # Compatibility is deliberately limited to the exact root steps file.
                    exact_steps = key == "steps_manifest" and Path(value) == output_root / "steps.yaml"
                    if exact_steps and exists and not authoritative:
                        diagnostics.append("legacy_steps_manifest_files_omission_exact_fallback")
                    else:
                        missing.append(key)
                        diagnostics.append(f"manifest_inventory_missing:{key}")
                elif not exists:
                    missing.append(key)
                    diagnostics.append(f"manifest_entity_missing:{key}")
            elif "*" in value or "?" in value or "[" in value:
                if not glob.glob(value, recursive=True):
                    missing.append(key)
            elif not Path(value).exists():
                missing.append(key)

        if manifest is not None and output_root is not None:
            required = manifest.get("required_artifacts") or []
            if isinstance(required, list):
                for relative_value in required:
                    relative = Path(str(relative_value))
                    if relative.is_absolute() or ".." in relative.parts:
                        diagnostics.append(f"required_artifact_unsafe:{relative_value}")
                        continue
                    required_path = output_root / relative
                    if inventory is None or relative.as_posix() not in inventory:
                        diagnostics.append(f"required_artifact_unlisted:{relative.as_posix()}")
                    if not required_path.exists():
                        diagnostics.append(f"required_artifact_entity_missing:{relative.as_posix()}")
        return list(dict.fromkeys(missing)), {
            "state": state,
            "path": str(manifest_path) if manifest_path else None,
            "schema_version": manifest.get("schema_version") if manifest else None,
            "authoritative": authoritative,
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _read_manifest(manifest_path: Optional[Path]) -> tuple[Optional[Dict[str, Any]], Optional[set]]:
        if manifest_path is None:
            return None, None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            listed_files = manifest.get("files")
            if not isinstance(manifest, dict) or not isinstance(listed_files, list):
                return None, None
        except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None, None
        inventory = {"."}
        for value in listed_files:
            relative = Path(str(value))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            inventory.add(relative.as_posix())
            inventory.update(parent.as_posix() for parent in relative.parents if parent.as_posix() != ".")
        return manifest, inventory

    @staticmethod
    def _read_manifest_inventory(manifest_path: Optional[Path]) -> Optional[set]:
        return RunMonitorAgent._read_manifest(manifest_path)[1]

    @staticmethod
    def _manifest_entities_exist(value: str, output_root: Path, inventory: set) -> bool:
        relative_pattern = os.path.relpath(str(value), str(output_root)).replace(os.sep, "/")
        if "*" in relative_pattern or "?" in relative_pattern or "[" in relative_pattern:
            patterns = [relative_pattern]
            if relative_pattern.startswith("**/"):
                patterns.append(relative_pattern[3:])
            matches = [item for item in inventory if any(fnmatch.fnmatchcase(item, pattern) for pattern in patterns)]
            return any((output_root / item).exists() for item in matches)
        return (output_root / relative_pattern).exists()

    @staticmethod
    def _path_is_within_output_root(value: str, output_root: Path) -> bool:
        root_text = os.path.abspath(str(output_root))
        value_text = os.path.abspath(str(value))
        return value_text == root_text or value_text.startswith(root_text + os.sep)

    @staticmethod
    def _manifest_contains(value: str, output_root: Path, inventory: set) -> bool:
        relative_pattern = os.path.relpath(str(value), str(output_root)).replace(os.sep, "/")
        if "*" in relative_pattern or "?" in relative_pattern or "[" in relative_pattern:
            patterns = [relative_pattern]
            # pathlib/glob treats a leading **/ as zero or more directories;
            # fnmatch alone requires at least one slash, so include the root form.
            if relative_pattern.startswith("**/"):
                patterns.append(relative_pattern[3:])
            return any(fnmatch.fnmatchcase(item, pattern) for item in inventory for pattern in patterns)
        return relative_pattern in inventory

    @staticmethod
    def _has_runtime_ceph_mount_error(text: str) -> bool:
        """Ignore mount-error text embedded in an echoed shell command."""
        needles = ("ceph_secret is not set", "/aceph/daweihuang is not mounted")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            lowered = line.lower()
            if not any(needle in lowered for needle in needles):
                continue
            if "[harness][error]" in lowered and not re.search(r"\b(?:echo|printf)\b", lowered):
                return True
            # Taiji can echo start_cmd shell source. Only accept an unadorned
            # runtime error line here; shell syntax such as echo/if/then is source.
            if re.fullmatch(
                r"(?:error:\s*)?(?:/aceph/daweihuang is not mounted(?: and)?\s*)?"
                r"ceph_secret is not set[.!]?",
                lowered,
            ):
                return True
        return False

    @staticmethod
    def _failure_hints(text: str, missing_outputs: List[str]) -> List[str]:
        lowered = text.lower()
        hints: List[str] = []
        resource_scheduling_needles = [
            "pending timeout",
            "state pending timeout",
            "resource exhausted",
            "no available resource",
            "quota",
            "queued",
            "evicted",
        ]
        resource_scheduling_failure = any(needle in lowered for needle in resource_scheduling_needles)
        checks = {
            "cuda_out_of_memory": ["out of memory", "cuda oom", "cublas", "cuda error"],
            "missing_boltzgen_cli": ["boltzgen/bg cli not found", "command not found"],
            "missing_input_file": ["no such file", "file not found", "不存在"],
            "conda_env_error": ["conda: command not found", "could not find conda environment", "activate"],
            "taiji_resource_or_queue_issue": ["resource exhausted", "no available resource", "quota", "queued", "evicted"],
            "generated_script_python_syntax_error": [
                "syntaxerror: unterminated string literal",
            ],
            # Use specific config-error signatures. A bare "yaml" match was too
            # broad: any traceback that merely references a *.yaml config path
            # (e.g. .../config/filtering.yaml) was mislabeled a config error, which
            # would wrongly mark an otherwise retryable infra/resource failure as
            # non-retryable.
            "boltzgen_config_error": [
                "invalid step",
                "omegaconf",
                "hydra.errors",
                "instantiationexception",
                "unexpected keyword argument",
                "keyerror",
                "invalid config",
            ],
        }
        if resource_scheduling_failure:
            hints.append("resource_scheduling_failure")
        for label, needles in checks.items():
            if any(n in lowered for n in needles):
                hints.append(label)
        if (
            "generated_script_python_syntax_error" not in hints
            and "syntaxerror" in lowered
            and 'file "<stdin>"' in lowered
        ):
            hints.append("generated_script_python_syntax_error")
        if not resource_scheduling_failure and RunMonitorAgent._has_runtime_ceph_mount_error(text):
            hints.append("missing_ceph_mount_secret")
        if missing_outputs:
            hints.append("missing_expected_outputs:" + ",".join(missing_outputs))
        return hints


def _timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
