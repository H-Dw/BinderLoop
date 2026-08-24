import csv
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Mapping

from binderloop.package_layout import is_project_package_name


class ResultPathSafetyError(ValueError):
    """Deterministic result path containment failure."""


class TransportBindingError(ResultPathSafetyError):
    """Harness transport provenance does not match the on-disk alias."""


@dataclass
class IngestedBoltzGenRun:
    """Round-level BoltzGen output inventory without candidate attribution."""

    output_dir: str
    job_id: str = ""
    arm_id: str = ""
    exploration_arm: str = ""
    logical_branch_id: str = ""
    execution_job_id: str = ""
    execution_slot: Optional[int] = None
    arm_root: str = ""
    log_file: Optional[str] = None
    metrics_files: List[str] = field(default_factory=list)
    selected_metrics_files: List[str] = field(default_factory=list)
    unfiltered_metrics_files: List[str] = field(default_factory=list)
    all_metrics_files: List[str] = field(default_factory=list)
    structure_files: List[str] = field(default_factory=list)
    candidate_scope: str = "selected_ranked"
    selected_metric_count: int = 0
    unfiltered_metric_count: int = 0
    filter_pass_count: int = 0
    selected_failed_filter_count: int = 0
    filter_pass_count_status: str = "unavailable"
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    log_tail: str = ""
    run_level_issues: List[str] = field(default_factory=list)
    core_ingestion_status: str = "unavailable"
    collection_mode: str = "round_aggregate"
    max_rows: int = 2000
    metrics_rows_read: int = 0
    metrics_rows_truncated: bool = False  # always false for new aggregate ingestion
    truncated_metrics_files: List[str] = field(default_factory=list)
    metrics_rows_over_limit: bool = False
    metrics_row_limit_notice: str = ""
    structure_file_count: int = 0
    population_metadata: Dict[str, Any] = field(default_factory=dict)
    native_inventory: Dict[str, Any] = field(default_factory=dict)


class ResultIngestionAgent:
    """Collect one round's native BoltzGen outputs as aggregate evidence."""

    FALLBACK_SCAN_MAX_DEPTH = 9
    FALLBACK_SCAN_MAX_ENTRIES = 100000
    STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}

    def ingest_boltzgen_output(
        self,
        output_dir: Union[str, Path],
        *,
        log_file: Union[str, Optional[Path]] = None,
        max_rows: int = 2000,
        identity_context: Optional[Dict[str, Any]] = None,
    ) -> IngestedBoltzGenRun:
        context = dict(identity_context or {})
        root = Path(output_dir)
        arm_root = Path(context.get("arm_root") or root)
        binding = context.get("transport_binding")
        path_policy = self._validate_transport_binding(root, arm_root, context, binding) if binding else None
        if identity_context and path_policy is None and not self._contained(root, arm_root):
            raise ResultPathSafetyError(f"output root is outside declared arm_root: {root}")
        log_path = Path(log_file) if log_file else self._guess_log_file(root, path_policy=path_policy)
        if log_path is not None:
            self._validate_log_path(log_path, arm_root, path_policy)
        index = self._build_file_index(root, arm_root=arm_root, path_policy=path_policy)
        selected_files = index["selected_metrics_files"]
        unfiltered_files = index["unfiltered_metrics_files"]

        row_cache: Dict[str, List[Dict[str, Any]]] = {}
        selected_rows, _ = self._read_metric_files(
            selected_files, root=root, max_rows=None, identity_context=context, row_cache=row_cache,
        )
        unfiltered_rows, _ = self._read_metric_files(
            unfiltered_files, root=root, max_rows=None, identity_context=context, row_cache=row_cache,
        )
        pass_values = [self._parse_bool(row.get("pass_filters")) for row in selected_rows]
        pass_flags_present = bool(selected_rows) and all(value is not None for value in pass_values)
        filter_pass_count = sum(value is True for value in pass_values)
        selected_failed = sum(value is False for value in pass_values)

        rows = selected_rows
        metrics_files = selected_files
        scope = "selected_ranked"
        if (not selected_rows or (pass_flags_present and filter_pass_count == 0)) and unfiltered_rows:
            rows = unfiltered_rows
            metrics_files = unfiltered_files
            scope = "unfiltered_zero_pass_recovery"
            index["index_issues"].append("zero_filter_pass_unfiltered_metrics_recovered")

        issues = list(index["index_issues"])
        over_limit = len(rows) > max_rows
        row_limit_notice = ""
        if over_limit:
            row_limit_notice = (
                f"metrics population contains {len(rows)} rows, above the {max_rows}-row "
                "LLM evidence budget; ingestion retained every row and downstream skill-guided "
                "evidence compaction will select representative high-value rows."
            )
            issues.append("metrics_rows_over_llm_evidence_budget")
            warnings.warn(row_limit_notice, RuntimeWarning, stacklevel=2)
        log_tail = self._read_tail(log_path) if log_path and log_path.exists() else ""
        issues.extend(self._run_level_issues(root, log_tail, metrics_files))
        structures = index["structure_files"]
        native_inventory = self._native_inventory(index["inventory_files"], root)
        population = {
            "mode": "arm_scoped" if identity_context else "round_aggregate",
            "metrics_scope": scope,
            "metrics_rows_observed": len(rows),
            "metrics_rows_truncated": False,
            "metrics_rows_over_limit": over_limit,
            "metrics_row_limit": max_rows,
            "metrics_selection_policy": "full_ingestion_then_skill_guided_compaction",
            "structure_files_observed": len(structures),
            "populations_are_not_row_aligned": True,
            "candidate_structure_attribution": bool(identity_context),
            "job_shard_arm_attribution": bool(identity_context),
            "native_hosts": len(native_inventory["hosts"]),
            "native_gpus": sum(len(host["gpus"]) for host in native_inventory["hosts"]),
            "native_shards": sum(len(gpu["shards"]) for host in native_inventory["hosts"] for gpu in host["gpus"]),
        }
        return IngestedBoltzGenRun(
            output_dir=str(root),
            job_id=str(context.get("job_id") or ""), arm_id=str(context.get("arm_id") or ""),
            exploration_arm=str(context.get("exploration_arm") or context.get("arm_id") or ""),
            logical_branch_id=str(context.get("logical_branch_id") or ""),
            execution_job_id=str(context.get("execution_job_id") or context.get("job_id") or ""),
            execution_slot=context.get("execution_slot"), arm_root=str(arm_root),
            log_file=str(log_path) if log_path else None,
            metrics_files=metrics_files,
            selected_metrics_files=selected_files,
            unfiltered_metrics_files=unfiltered_files,
            all_metrics_files=index["all_metrics_files"],
            structure_files=structures,
            candidate_scope=scope,
            selected_metric_count=len(selected_rows),
            unfiltered_metric_count=len(unfiltered_rows),
            filter_pass_count=filter_pass_count,
            selected_failed_filter_count=selected_failed,
            filter_pass_count_status="evaluated" if pass_flags_present else "unavailable",
            candidates=rows,
            log_tail=log_tail,
            run_level_issues=issues,
            core_ingestion_status="evaluated" if rows or metrics_files or structures else "unavailable",
            max_rows=max_rows,
            metrics_rows_read=len(rows),
            metrics_rows_truncated=False,
            truncated_metrics_files=[],
            metrics_rows_over_limit=over_limit,
            metrics_row_limit_notice=row_limit_notice,
            structure_file_count=len(structures),
            population_metadata=population,
            native_inventory=native_inventory,
        )

    def ingest_rfd3_output(
        self,
        output_dir: Union[str, Path],
        *,
        log_file: Union[str, Optional[Path]] = None,
        max_rows: int = 2000,
        identity_context: Optional[Dict[str, Any]] = None,
    ) -> IngestedBoltzGenRun:
        """Collect Foundry RF3 / RFD3 bridge metrics as the same ingestion payload."""
        from binderloop.analysis.parsers import parse_rfd3_scores

        context = dict(identity_context or {})
        root = Path(output_dir)
        arm_root = Path(context.get("arm_root") or root)
        if identity_context and not self._contained(root, arm_root):
            raise ResultPathSafetyError(f"output root is outside declared arm_root: {root}")
        weights = {
            "interface_confidence": 0.30,
            "hotspot_contact": 0.25,
            "binder_plddt": 0.15,
            "clash_penalty": 0.15,
            "diversity": 0.10,
            "sequence_designability": 0.05,
        }
        scores = parse_rfd3_scores(root, weights)
        rows: List[Dict[str, Any]] = []
        structures: List[str] = []
        metrics_files: List[str] = []
        for score in scores:
            path = str(score.path or "")
            if path:
                metrics_files.append(path)
                suffix = Path(path).suffix.lower()
                if suffix in self.STRUCTURE_SUFFIXES:
                    structures.append(path)
            rows.append({
                "design": score.candidate_id,
                "name": score.candidate_id,
                "model": "rfd3",
                "path": path,
                "structure_file": path,
                "iptm": score.interface_confidence,
                "design_to_target_iptm": score.interface_confidence,
                "plddt": score.binder_plddt,
                "binder_plddt": score.binder_plddt,
                "hotspot_contact": score.hotspot_contact,
                "sequence_designability": score.sequence_designability,
                "ranking_score": score.sequence_designability,
                "clash_penalty": score.clash_penalty,
                "diversity": score.diversity,
                "refold_tool": "rf3",
                "_metrics_file": path,
            })
        if not structures:
            for path in root.rglob("*"):
                if path.suffix.lower() in self.STRUCTURE_SUFFIXES:
                    structures.append(str(path))
        log_path = Path(log_file) if log_file else None
        log_tail = self._read_tail(log_path) if log_path and log_path.exists() else ""
        over_limit = len(rows) > max_rows
        return IngestedBoltzGenRun(
            output_dir=str(root),
            job_id=str(context.get("job_id") or ""),
            arm_id=str(context.get("arm_id") or ""),
            exploration_arm=str(context.get("exploration_arm") or context.get("arm_id") or ""),
            logical_branch_id=str(context.get("logical_branch_id") or ""),
            execution_job_id=str(context.get("execution_job_id") or context.get("job_id") or ""),
            execution_slot=context.get("execution_slot"),
            arm_root=str(arm_root),
            log_file=str(log_path) if log_path else None,
            metrics_files=list(dict.fromkeys(metrics_files)),
            selected_metrics_files=list(dict.fromkeys(metrics_files)),
            unfiltered_metrics_files=[],
            all_metrics_files=list(dict.fromkeys(metrics_files)),
            structure_files=list(dict.fromkeys(structures)),
            candidate_scope="rfd3_rf3",
            selected_metric_count=len(rows),
            candidates=rows,
            log_tail=log_tail,
            run_level_issues=[],
            core_ingestion_status="evaluated" if rows else "unavailable",
            max_rows=max_rows,
            metrics_rows_read=len(rows),
            metrics_rows_over_limit=over_limit,
            structure_file_count=len(structures),
            population_metadata={
                "mode": "arm_scoped" if identity_context else "round_aggregate",
                "metrics_scope": "rfd3_rf3",
                "metrics_rows_observed": len(rows),
                "refold_tool": "rf3",
                "ingest_model": "rfd3",
            },
        )

    @classmethod
    def _read_metric_files(
        cls, paths: List[str], *, root: Path, max_rows: Optional[int],
        identity_context: Optional[Mapping[str, Any]] = None,
        row_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        rows: List[Dict[str, Any]] = []
        truncated_files: List[str] = []
        for file_index, metrics_file in enumerate(paths):
            remaining = None if max_rows is None else max(0, max_rows - len(rows))
            if remaining == 0:
                truncated_files.extend(paths[file_index:])
                break
            cache_key = str(ResultIngestionAgent._lexical(Path(metrics_file)))
            try:
                if row_cache is not None and cache_key in row_cache:
                    cached = row_cache[cache_key]
                    page = [dict(row) for row in cached[:remaining]] if remaining is not None else [dict(row) for row in cached]
                    truncated = remaining is not None and len(cached) > remaining
                else:
                    page, truncated = cls._read_metrics_csv_page(
                        Path(metrics_file), max_rows=remaining, root=root, identity_context=identity_context
                    )
                    if row_cache is not None and max_rows is None:
                        row_cache[cache_key] = [dict(row) for row in page]
            except OSError:
                continue
            rows.extend(page)
            if truncated:
                truncated_files.append(metrics_file)
            if max_rows is not None and len(rows) >= max_rows and file_index + 1 < len(paths):
                truncated_files.extend(paths[file_index + 1 :])
                break
        return rows, list(dict.fromkeys(truncated_files))

    @staticmethod
    def _build_file_index(root: Path, *, arm_root: Optional[Path] = None, path_policy: Optional[Mapping[str, Path]] = None) -> Dict[str, Any]:
        arm_root = Path(arm_root or root)
        if path_policy is None and not ResultIngestionAgent._contained(root, arm_root):
            raise ResultPathSafetyError("unsafe root outside arm_root")
        manifest_path = root / "result_manifest.json"
        ResultIngestionAgent._validate_result_path(manifest_path, root, arm_root, path_policy, "result_manifest")
        files: List[Path] = []
        issues: List[str] = []
        loaded = False
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                listed = manifest.get("files")
                if not isinstance(listed, list):
                    raise TypeError("result manifest files must be a list")
                files.extend(ResultIngestionAgent._resolve_inventory_entries(root, arm_root, listed, issues, "result_manifest", path_policy))
                shard_refs = manifest.get("shard_manifests") or []
                if not isinstance(shard_refs, list):
                    raise TypeError("result manifest shard_manifests must be a list")
                for shard_ref in shard_refs:
                    resolved = ResultIngestionAgent._resolve_inventory_entries(root, arm_root, [shard_ref], issues, "shard_manifest_ref", path_policy)
                    if not resolved:
                        continue
                    shard_path = resolved[0]
                    if not shard_path.is_file():
                        issues.append(f"shard_manifest_entity_missing:{shard_ref}")
                        continue
                    try:
                        shard = json.loads(shard_path.read_text(encoding="utf-8"))
                        shard_files = shard.get("files")
                        if not isinstance(shard_files, list):
                            raise TypeError("shard files must be a list")
                        if shard.get("shard_manifests"):
                            issues.append(f"nested_shard_manifests_ignored:{shard_ref}")
                        files.extend(ResultIngestionAgent._resolve_inventory_entries(root, arm_root, shard_files, issues, f"shard_manifest:{shard_ref}", path_policy))
                    except (OSError, TypeError, json.JSONDecodeError):
                        issues.append(f"shard_manifest_invalid:{shard_ref}")
                loaded = True
            except (OSError, TypeError, json.JSONDecodeError):
                issues.append("result_manifest_invalid_bounded_scan")
        else:
            issues.append("result_manifest_missing_bounded_scan")
        if not loaded:
            files, scan_issues = ResultIngestionAgent._bounded_scan_known_result_dirs(root)
            issues.extend(scan_issues)

        unique = list(dict.fromkeys(files))
        concrete = []
        for candidate in unique:
            ResultIngestionAgent._validate_result_path(candidate, root, arm_root, path_policy, "inventory")
            if candidate.is_file():
                concrete.append(candidate)
            else:
                try:
                    relative = candidate.relative_to(root).as_posix()
                except ValueError:
                    relative = str(candidate)
                issues.append(f"result_manifest_entity_missing:{relative}")
        all_metrics = [p for p in concrete if p.suffix.lower() == ".csv" and "metrics" in p.name.lower()]
        selected = [p for p in all_metrics if "final_ranked_designs" in p.parts and (p.name.startswith("final_designs_metrics") or (p.name.startswith("final_") and "_metrics" in p.name))]
        unfiltered = [p for p in all_metrics if "final_ranked_designs" in p.parts and p.name == "all_designs_metrics.csv"]
        if not selected:
            selected = list(unfiltered)
        structures = [p for p in concrete if p.suffix.lower() in ResultIngestionAgent.STRUCTURE_SUFFIXES and "final_ranked_designs" in p.parts and "before_refolding" not in p.parts and not any(part.startswith("intermediate_ranked_") for part in p.parts) and any(part.startswith("final_") and part.endswith("_designs") for part in p.parts)]
        return {
            "selected_metrics_files": [str(p) for p in selected],
            "unfiltered_metrics_files": [str(p) for p in unfiltered],
            "all_metrics_files": [str(p) for p in all_metrics],
            "structure_files": [str(p) for p in structures],
            "inventory_files": [str(p) for p in concrete],
            "index_issues": list(dict.fromkeys(issues)),
        }

    @staticmethod
    def _resolve_inventory_entries(root: Path, arm_root: Path, values: Iterable[Any], issues: List[str], source: str, path_policy: Optional[Mapping[str, Path]] = None) -> List[Path]:
        resolved: List[Path] = []
        for value in values:
            relative = Path(str(value))
            if relative.is_absolute() or ".." in relative.parts:
                if path_policy is not None or arm_root.resolve(strict=False) != root.resolve(strict=False):
                    raise ResultPathSafetyError(f"result manifest path outside declared output root: {value}")
                issues.append(f"{source}_unsafe_path:{value}")
                continue
            candidate = root / relative
            ResultIngestionAgent._validate_result_path(candidate, root, arm_root, path_policy, source)
            resolved.append(candidate)
        return resolved

    @staticmethod
    def _validate_result_path(candidate: Path, root: Path, arm_root: Path, path_policy: Optional[Mapping[str, Path]], source: str) -> None:
        if path_policy is None:
            if not ResultIngestionAgent._contained(candidate, arm_root):
                raise ResultPathSafetyError(f"{source} path outside declared arm_root: {candidate}")
            return
        local_alias = path_policy["local_output_alias"]
        if not ResultIngestionAgent._lexically_contained(candidate, local_alias):
            raise TransportBindingError(f"transport binding mismatch: {source} lexical path outside local output alias")

    @staticmethod
    def _validate_transport_binding(root: Path, arm_root: Path, context: Mapping[str, Any], raw_binding: Any) -> Mapping[str, Path]:
        if not isinstance(raw_binding, Mapping):
            raise TransportBindingError("transport binding mismatch: binding must be an object")
        binding = dict(raw_binding)
        required = ("mode", "local_package_dir", "local_output_alias", "local_logs_alias", "remote_package_dir", "remote_output_root", "remote_logs_root", "link_text", "logs_link_text", "job_id", "attempt", "task_flag")
        missing = [key for key in required if binding.get(key) in (None, "")]
        if missing:
            raise TransportBindingError(f"transport binding mismatch: missing {','.join(missing)}")
        if str(binding["mode"]).lower() != "symlink":
            raise TransportBindingError("transport binding mismatch: trusted binding mode is not symlink")
        for key in ("job_id", "attempt", "task_flag"):
            if str(binding[key]) != str(context.get(key)):
                raise TransportBindingError(f"transport binding mismatch: {key}")
        local_package = Path(str(binding["local_package_dir"]))
        local_alias = Path(str(binding["local_output_alias"]))
        remote_package = Path(str(binding["remote_package_dir"]))
        remote_root = Path(str(binding["remote_output_root"]))
        local_logs = Path(str(binding["local_logs_alias"]))
        remote_logs = Path(str(binding["remote_logs_root"]))
        if ResultIngestionAgent._lexical(local_alias) != ResultIngestionAgent._lexical(root):
            raise TransportBindingError("transport binding mismatch: output root is not local output alias")
        expected_alias = local_package / "outputs" / "boltzgen_output"
        if ResultIngestionAgent._lexical(local_alias) != ResultIngestionAgent._lexical(expected_alias):
            raise TransportBindingError("transport binding mismatch: local output alias is not package/outputs/boltzgen_output")
        if not ResultIngestionAgent._lexically_contained(local_package, arm_root):
            raise TransportBindingError("transport binding mismatch: local package outside declared arm_root")
        attempt_root_value = context.get("attempt_root") or binding.get("attempt_root")
        if attempt_root_value and not ResultIngestionAgent._lexically_contained(local_package, Path(str(attempt_root_value))):
            raise TransportBindingError("transport binding mismatch: local package outside current attempt root")
        transport_link = local_package / "outputs"
        logs_link = local_package / "logs"
        if not transport_link.is_symlink():
            raise TransportBindingError("transport binding mismatch: package/outputs is not the transport symlink")
        if not logs_link.is_symlink():
            raise TransportBindingError("transport binding mismatch: package/logs is not the transport symlink")
        actual_link = os.readlink(str(transport_link))
        actual_logs_link = os.readlink(str(logs_link))
        if Path(actual_link).is_absolute() or actual_link != str(binding["link_text"]):
            raise TransportBindingError("transport binding mismatch: relative outputs link text")
        if Path(actual_logs_link).is_absolute() or actual_logs_link != str(binding["logs_link_text"]):
            raise TransportBindingError("transport binding mismatch: relative logs link text")
        if ResultIngestionAgent._lexical(local_logs) != ResultIngestionAgent._lexical(logs_link):
            raise TransportBindingError("transport binding mismatch: local logs alias is not package/logs")
        if local_alias.resolve(strict=False) != remote_root.resolve(strict=False):
            raise TransportBindingError("transport binding mismatch: resolved output root differs")
        if logs_link.resolve(strict=False) != remote_logs.resolve(strict=False):
            raise TransportBindingError("transport binding mismatch: resolved logs root differs")
        if not ResultIngestionAgent._contained(remote_root, remote_package):
            raise TransportBindingError("transport binding mismatch: remote output root outside remote package root")
        if not ResultIngestionAgent._contained(remote_logs, remote_package):
            raise TransportBindingError("transport binding mismatch: remote logs root outside remote package root")
        if not is_project_package_name(remote_package.name) or remote_package.parent.name != str(binding["task_flag"]):
            raise TransportBindingError("transport binding mismatch: remote package does not belong to current task_flag")
        expected_remote = remote_package / "outputs" / "boltzgen_output"
        if remote_root.resolve(strict=False) != expected_remote.resolve(strict=False):
            raise TransportBindingError("transport binding mismatch: remote output root is not package output root")
        if remote_logs.resolve(strict=False) != (remote_package / "logs").resolve(strict=False):
            raise TransportBindingError("transport binding mismatch: remote logs root is not package logs root")
        return {
            "local_output_alias": local_alias, "remote_output_root": remote_root,
            "local_logs_alias": local_logs, "remote_logs_root": remote_logs,
        }

    @staticmethod
    def _validate_log_path(path: Path, arm_root: Path, path_policy: Optional[Mapping[str, Path]]) -> None:
        if path_policy is None:
            if not ResultIngestionAgent._contained(path, arm_root):
                raise ResultPathSafetyError(f"log file is outside declared arm_root: {path}")
            return
        if not ResultIngestionAgent._lexically_contained(path, path_policy["local_logs_alias"]):
            raise TransportBindingError("transport binding mismatch: log lexical path outside local logs alias")

    @staticmethod
    def _lexical(path: Path) -> Path:
        return Path(os.path.abspath(os.path.normpath(str(path))))

    @staticmethod
    def _lexically_contained(path: Path, root: Path) -> bool:
        try:
            ResultIngestionAgent._lexical(path).relative_to(ResultIngestionAgent._lexical(root))
            return True
        except ValueError:
            return False

    @staticmethod
    def _native_inventory(paths: List[str], root: Path) -> Dict[str, Any]:
        tree: Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]] = {}
        lexical_root = ResultIngestionAgent._lexical(root)
        for value in paths:
            path = Path(value)
            try:
                relative = ResultIngestionAgent._lexical(path).relative_to(lexical_root)
            except ValueError:
                continue
            host = next((part for part in relative.parts if part.startswith("host_") and part[5:].isdigit()), None)
            gpu = next((part for part in relative.parts if part.startswith("gpu_") and part[4:].isdigit()), None)
            shard = next((part for part in relative.parts if part.startswith("shard_") and part[6:7].isdigit()), None)
            if gpu is None and shard is None:
                continue
            host = host or "host_00"
            gpu = gpu or "gpu_00"
            shard = shard or "shard_000"
            kind = "metrics" if path.suffix.lower() == ".csv" and "metrics" in path.name.lower() else "structures" if path.suffix.lower() in ResultIngestionAgent.STRUCTURE_SUFFIXES else "artifacts"
            tree.setdefault(host, {}).setdefault(gpu, {}).setdefault(shard, {"metrics": [], "structures": [], "artifacts": []})[kind].append(str(path))
        return {"hosts": [{"host": host, "gpus": [{"gpu": gpu, "shards": [{"shard": shard, **groups} for shard, groups in sorted(shards.items())]} for gpu, shards in sorted(gpus.items())]} for host, gpus in sorted(tree.items())]}

    @staticmethod
    def _bounded_scan_known_result_dirs(root: Path) -> Tuple[List[Path], List[str]]:
        files: List[Path] = []
        issues: List[str] = []
        pending: List[Tuple[Path, int]] = [(root, 0)]
        visited = 0
        while pending:
            directory, depth = pending.pop()
            try:
                with os.scandir(str(directory)) as entries:
                    for entry in entries:
                        visited += 1
                        if visited > ResultIngestionAgent.FALLBACK_SCAN_MAX_ENTRIES:
                            issues.append("result_fallback_scan_entry_limit_reached")
                            return files, issues
                        try:
                            if entry.is_file(follow_symlinks=False):
                                files.append(Path(entry.path))
                            elif depth < ResultIngestionAgent.FALLBACK_SCAN_MAX_DEPTH and entry.is_dir(follow_symlinks=False) and ResultIngestionAgent._is_known_result_dir(entry.name):
                                pending.append((Path(entry.path), depth + 1))
                        except OSError:
                            continue
            except OSError:
                continue
        return files, issues

    @staticmethod
    def _is_known_result_dir(name: str) -> bool:
        return (
            name in {"final_ranked_designs", "intermediate_designs", "intermediate_designs_inverse_folded", "before_refolding", "refold_cif", "refold_design_cif"}
            or name.startswith(("host_", "gpu_", "shard_"))
            or (name.startswith("final_") and name.endswith("_designs"))
            or (name.startswith("intermediate_ranked_") and name.endswith("_designs"))
        )

    @staticmethod
    def _contained(path: Path, root: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(root.resolve(strict=False)); return True
        except (OSError, ValueError): return False

    @staticmethod
    def _parse_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        text = str(value or "").strip().lower()
        if text in {"true", "1", "yes", "y", "pass", "passed"}:
            return True
        if text in {"false", "0", "no", "n", "fail", "failed"}:
            return False
        return None

    @staticmethod
    def _read_metrics_csv_page(path: Path, *, max_rows: Optional[int], root: Optional[Path] = None, identity_context: Optional[Mapping[str, Any]] = None) -> Tuple[List[Dict[str, Any]], bool]:
        if max_rows is not None and max_rows <= 0:
            return [], True
        rows: List[Dict[str, Any]] = []
        truncated = False
        relative_path: Optional[Path] = None
        if root is not None:
            try:
                relative_path = ResultIngestionAgent._lexical(path).relative_to(ResultIngestionAgent._lexical(root))
            except ValueError:
                pass
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            for ordinal, row in enumerate(csv.DictReader(handle)):
                if max_rows is not None and len(rows) >= max_rows:
                    truncated = True
                    break
                value = dict(row)
                if root is not None:
                    value["_metrics_file"] = str(path)
                    value["_metrics_relative_path"] = str(relative_path) if relative_path is not None else path.name
                    value["_metrics_row_ordinal"] = ordinal
                    parts = relative_path.parts if relative_path is not None else ()
                    value["native_host"] = next((part for part in parts if part.startswith("host_")), "host_00")
                    value["native_gpu"] = next((part for part in parts if part.startswith("gpu_")), "gpu_00")
                    value["native_shard"] = next((part for part in parts if part.startswith("shard_")), "shard_000")
                for key, item in dict(identity_context or {}).items():
                    if key in {"job_id", "arm_id", "exploration_arm", "logical_branch_id", "execution_job_id", "execution_slot", "host_shard", "arm_root", "output_root"}:
                        value[key] = item
                rows.append(value)
        return rows, truncated

    @staticmethod
    def _read_metrics_csv(path: Path, *, max_rows: int, root: Optional[Path] = None) -> List[Dict[str, Any]]:
        return ResultIngestionAgent._read_metrics_csv_page(path, max_rows=max_rows, root=root)[0]

    def write_manifest(self, run: IngestedBoltzGenRun, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(run), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @staticmethod
    def _guess_log_file(root: Path, *, path_policy: Optional[Mapping[str, Path]] = None) -> Optional[Path]:
        directories = (path_policy["local_logs_alias"],) if path_policy is not None else (root, root.parent, root.parent.parent / "logs")
        for directory in directories:
            try:
                logs = sorted(Path(entry.path) for entry in os.scandir(str(directory)) if entry.is_file(follow_symlinks=False) and entry.name.endswith(".log"))
            except OSError:
                logs = []
            if logs:
                return logs[0]
        return None

    @staticmethod
    def _read_tail(path: Path, max_chars: int = 8000) -> str:
        # Logs can be very large on Ceph. Read only a bounded byte window from
        # the end; UTF-8 replacement is acceptable for diagnostic tails.
        max_bytes = max(4096, int(max_chars) * 4)
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = handle.read(max_bytes)
        return data.decode("utf-8", errors="ignore")[-max_chars:]

    @staticmethod
    def _run_level_issues(root: Path, log_tail: str, metrics_files: List[str]) -> List[str]:
        issues: List[str] = []
        lowered = log_tail.lower()
        if not root.exists():
            issues.append("output_dir_missing")
        if not metrics_files:
            issues.append("metrics_missing")
        if "out of memory" in lowered or "cuda oom" in lowered:
            issues.append("cuda_out_of_memory")
        if "no such file" in lowered or "file not found" in lowered:
            issues.append("missing_input_file")
        if "error" in lowered or "traceback" in lowered:
            issues.append("runtime_error_in_log")
        return issues
