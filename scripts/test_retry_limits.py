#!/usr/bin/env python3

import sys
import tempfile
import json
import pytest
from dataclasses import asdict
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.config import HarnessConfig, TargetSpec
from binderloop.agents import ConfigValidationAgent, ConfigValidationResult
from binderloop.agents.config_parameter_contract import supported_config_changes
from binderloop.agents.run_monitor_agent import RunMonitorAgent
from binderloop.models.base import DesignJob
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator
from binderloop.memory import ExperimentMemoryStore
from scripts.run_closed_loop_orchestrator import _classify_taiji_failure


def _cfg() -> HarnessConfig:
    cfg = HarnessConfig(target=TargetSpec(structure_path="missing.cif"))
    cfg.resource.backend = 'local'
    cfg.resource.max_parallel_jobs = 1
    return cfg


def _job(out_dir: Path) -> DesignJob:
    return DesignJob(
        job_id="retry_limit_job",
        target_structure="missing.cif",
        chain_id="A",
        hotspots=[],
        binder_length=50,
        seed=0,
        params={},
        output_dir=str(out_dir / "job"),
    )


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def available(self) -> bool:
        return True

    def chat_json(self, **kwargs):
        return self.payload


def test_retry_cap_persists_across_reruns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        attempts_path = out_dir / "round_00" / "execution_attempts.json"
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=2)
        calls: List[int] = []

        def failing_executor(job: DesignJob, attempt: int):
            calls.append(attempt)
            return {"job_id": job.job_id, "status": "failed", "error": f"submit failed {attempt}"}

        records = orchestrator._run_jobs([_job(out_dir)], 0, failing_executor, attempts_path=attempts_path)
        assert calls == [1, 2], calls
        assert records[0]["status"] == "failed"
        assert records[0]["attempts"] == 2

        def should_not_submit(job: DesignJob, attempt: int):
            raise AssertionError("executor should not be called after persisted retry cap")

        rerun_records = orchestrator._run_jobs([_job(out_dir)], 0, should_not_submit, attempts_path=attempts_path)
        assert rerun_records[0]["status"] == "failed"
        assert rerun_records[0]["attempts"] == 2



def _assert_retry_terminal_context_consistent(record, attempts_path: Path, validation) -> None:
    ledger = json.loads(attempts_path.read_text(encoding="utf-8"))
    job_entry = ledger["jobs"]["retry_limit_job"]
    terminal = job_entry["terminal_record"]
    attempt = job_entry["attempts"][-1]
    assert record["pre_submit"]["validation"] == validation
    assert terminal["pre_submit"]["validation"] == validation
    assert terminal == record
    for key in ("failure_fingerprint", "semantic_failure_fingerprint_version", "semantic_failure_fingerprint", "semantic_failure_scope"):
        assert record[key]
        assert terminal[key] == record[key]
        assert attempt[key] == record[key]


def test_retry_cap_failed_executor_preserves_terminal_context() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        attempts_path = out_dir / "round_00" / "execution_attempts.json"
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
        validation = {"issues": [{"parameter": "budget", "severity": "error", "resolved": False, "problem": "budget is required"}], "missing_required_keys": ["budget"]}
        record = orchestrator._run_jobs(
            [_job(out_dir)], 0,
            lambda job, attempt: {
                "job_id": job.job_id, "status": "failed", "error": "pre-submit validation failed", "retryable": True,
                "pre_submit": {"validation": validation, "request_id": "submit-1"},
                "runtime": {"digest": "runtime-1", "image_digest": "image-1"},
            }, attempts_path=attempts_path,
        )[0]
        assert record["runtime"] == {"digest": "runtime-1", "image_digest": "image-1"}
        _assert_retry_terminal_context_consistent(record, attempts_path, validation)


def test_retry_cap_executor_exception_preserves_terminal_context(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        attempts_path = out_dir / "round_00" / "execution_attempts.json"
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
        validation = {"issues": [{"parameter": "num_designs", "severity": "error", "resolved": False, "problem": "num_designs is invalid"}], "missing_required_keys": ["num_designs"]}
        original_failure_fingerprint = orchestrator._failure_fingerprint
        original_semantic_fingerprint = orchestrator._semantic_failure_fingerprint

        def enrich_failure(record):
            record["pre_submit"] = {"validation": validation, "request_id": "exception-1"}
            record["runtime"] = {"digest": "runtime-exception"}
            return original_failure_fingerprint(record)

        monkeypatch.setattr(orchestrator, "_failure_fingerprint", enrich_failure)
        monkeypatch.setattr(orchestrator, "_semantic_failure_fingerprint", lambda record: original_semantic_fingerprint(record))

        def raising_executor(job: DesignJob, attempt: int):
            raise RuntimeError("executor exploded")

        record = orchestrator._run_jobs([_job(out_dir)], 0, raising_executor, attempts_path=attempts_path)[0]
        assert record["error"] == "executor exploded"
        assert record["runtime"] == {"digest": "runtime-exception"}
        _assert_retry_terminal_context_consistent(record, attempts_path, validation)


def test_semantic_failure_fingerprint_reads_actual_pre_submit_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=1)
        validation = {"issues": [{"parameter": "budget", "severity": "error", "resolved": False, "problem": "budget is required"}], "missing_required_keys": ["budget"]}
        actual = {"status": "failed", "error": "pre-submit validation failed", "pre_submit": {"validation": validation}}
        legacy_nested = {**actual, "pre_submit": {"config_validation": validation}}
        top_level = {**actual, "pre_submit": {}, "config_validation": validation}
        assert orchestrator._semantic_failure_fingerprint(actual) == orchestrator._semantic_failure_fingerprint(legacy_nested)
        assert orchestrator._semantic_failure_fingerprint(actual) == orchestrator._semantic_failure_fingerprint(top_level)
        changed = json.loads(json.dumps(actual))
        changed["pre_submit"]["validation"]["missing_required_keys"] = ["num_designs"]
        changed["pre_submit"]["validation"]["issues"][0]["parameter"] = "num_designs"
        assert orchestrator._semantic_failure_fingerprint(actual) != orchestrator._semantic_failure_fingerprint(changed)

def test_non_retryable_failure_stops_immediately() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=5)
        calls: List[int] = []

        def non_retryable_executor(job: DesignJob, attempt: int):
            calls.append(attempt)
            return {"job_id": job.job_id, "status": "failed", "error": "bad submit config", "retryable": False}

        records = orchestrator._run_jobs(
            [_job(out_dir)],
            0,
            non_retryable_executor,
            attempts_path=out_dir / "round_00" / "execution_attempts.json",
        )
        assert calls == [1], calls
        assert records[0]["status"] == "failed"
        assert records[0]["attempts"] == 1
        assert records[0]["retryable"] is False


def test_zero_retries_means_single_submit_attempt() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=0)
        calls: List[int] = []

        def failing_executor(job: DesignJob, attempt: int):
            calls.append(attempt)
            return {"job_id": job.job_id, "status": "failed", "error": "submit failed"}

        records = orchestrator._run_jobs(
            [_job(out_dir)],
            0,
            failing_executor,
            attempts_path=out_dir / "round_00" / "execution_attempts.json",
        )
        assert calls == [1], calls
        assert records[0]["status"] == "failed"
        assert records[0]["attempts"] == 1


def test_incomplete_attempt_blocks_resubmit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        attempts_path = out_dir / "round_00" / "execution_attempts.json"
        attempts_path.parent.mkdir(parents=True, exist_ok=True)
        attempts_path.write_text(
            json.dumps(
                {
                    "round_id": 0,
                    "max_attempts_per_job": 3,
                    "jobs": {
                        "retry_limit_job": {
                            "job": {},
                            "attempts": [{"attempt": 1, "status": "started"}],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=3)

        def should_not_submit(job: DesignJob, attempt: int):
            raise AssertionError("executor should not run while a previous attempt is incomplete")

        try:
            orchestrator._run_jobs([_job(out_dir)], 0, should_not_submit, attempts_path=attempts_path)
        except RuntimeError as exc:
            assert "Refusing to submit new jobs" in str(exc)
            assert "retry_limit_job#attempt1" in str(exc)
        else:
            raise AssertionError("incomplete attempt did not block resubmit")


def test_multigpu_round_cap_splits_budget_across_sequential_jobs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for gpu_count, cap, requested_parallel, candidate_jobs in [(2, 10, 2, 7), (4, 17, 4, 9), (8, 32, 5, 25)]:
            cfg = _cfg()
            cfg.resource.host_gpu_num = gpu_count
            cfg.resource.max_parallel_jobs = gpu_count
            cfg.search_space.max_binders_per_round = cap
            cfg.search_space.num_designs_per_round = cap
            orchestrator = BinderDesignOrchestrator(cfg, out_dir=out_dir / f"gpu_{gpu_count}", max_parallel=requested_parallel, max_retries=1)
            jobs = [
                DesignJob(
                    job_id=f"job_{index}",
                    target_structure="missing.cif",
                    chain_id="A",
                    hotspots=[],
                    binder_length=80,
                    seed=index,
                    params={
                        "num_designs": cap,
                        "devices": gpu_count,
                        "arm_id": f"arm_{index}",
                        "exploration_arm": f"arm_{index}",
                        "arm_rank": index,
                    },
                    output_dir=str(out_dir / f"gpu_{gpu_count}" / f"job_{index}"),
                )
                for index in range(candidate_jobs)
            ]

            selected = orchestrator._enforce_round_cap(jobs)
            assert orchestrator.max_parallel == 1
            assert len(selected) == min(candidate_jobs, cap)
            assert sum(job.params["num_designs"] for job in selected) == cap
            assert all(job.params["devices"] == gpu_count for job in selected)
            assert all(job.params["round_budget_allocation"]["strategy"] == "multi_job_round_multi_length_gpu_fanout" for job in selected)


def test_first_round_cap_preserves_multi_length_single_task_fanout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        cfg.resource.host_gpu_num = 8
        cfg.resource.max_parallel_jobs = 8
        cfg.search_space.binder_length_range = [80, 120]
        cfg.search_space.binder_length_step = 10
        cfg.search_space.binder_lengths = [80, 90, 100, 110, 120]
        cfg.search_space.max_binders_per_round = 15
        cfg.search_space.num_designs_per_round = 15
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=Path(tmp), max_parallel=8, max_retries=1)

        selected = orchestrator._enforce_round_cap(orchestrator._initial_jobs())

        assert orchestrator.max_parallel == 1
        assert len(selected) == 1
        job = selected[0]
        assert job.binder_length == 100
        assert job.params["binder_lengths"] == [80, 90, 100, 110, 120]
        assert job.params["num_designs"] == 15
        assert job.params["round_budget_allocation"]["binder_lengths"] == [80, 90, 100, 110, 120]
        assert job.params["round_budget_allocation"]["strategy"] == "multi_job_round_multi_length_gpu_fanout"
        assert "seed" not in job.params


def test_taiji_host_num_splits_round_into_single_host_shards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        cfg.resource.backend = "taiji"
        cfg.resource.host_num = 3
        cfg.resource.host_gpu_num = 8
        cfg.resource.taiji_multi_host_mode = "split_jobs"
        cfg.resource.max_parallel_jobs = 8
        cfg.search_space.binder_length_range = [80, 120]
        cfg.search_space.binder_length_step = 20
        cfg.search_space.binder_lengths = [80, 100, 120]
        cfg.search_space.max_binders_per_round = 10
        cfg.search_space.num_designs_per_round = 10
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=Path(tmp), max_retries=1)

        selected = orchestrator._enforce_round_cap(orchestrator._initial_jobs())

        assert orchestrator.max_parallel == 3
        assert len(selected) == 3
        assert [job.params["num_designs"] for job in selected] == [4, 3, 3]
        assert sum(job.params["num_designs"] for job in selected) == 10
        assert all(job.params["devices"] == 8 for job in selected)
        assert all(job.params["taiji_submit_host_num"] == 1 for job in selected)
        assert all(job.params["multi_taiji_host_shard"]["submitted_host_num"] == 1 for job in selected)
        assert all(job.params["multi_taiji_host_shard"]["requested_host_num"] == 3 for job in selected)
        assert [job.params["multi_taiji_host_shard"]["shard_id"] for job in selected] == [
            "1_of_3",
            "2_of_3",
            "3_of_3",
        ]
        assert all(job.params["round_budget_allocation"]["strategy"] == "multi_host_single_host_taiji_fanout" for job in selected)
        assert len({job.output_dir for job in selected}) == 3
        assert len({job.job_id for job in selected}) == 3


def test_multi_host_parent_budget_does_not_decay_next_round() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        cfg.resource.backend = "taiji"
        cfg.resource.host_num = 2
        cfg.resource.host_gpu_num = 8
        cfg.resource.taiji_multi_host_mode = "split_jobs"
        cfg.resource.max_parallel_jobs = 8
        cfg.search_space.max_binders_per_round = 160
        cfg.search_space.num_designs_per_round = 160
        cfg.search_space.binder_length_range = [80, 120]
        cfg.search_space.binder_length_step = 5
        cfg.search_space.binder_lengths = [80, 85, 90, 95, 100, 105, 110, 115, 120]
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=Path(tmp), max_retries=1)

        first_round = orchestrator._enforce_round_cap(orchestrator._initial_jobs())
        assert [job.params["num_designs"] for job in first_round] == [80, 80]
        logical_parents = orchestrator._logical_jobs_for_memory(first_round)
        assert len(logical_parents) == 1
        assert logical_parents[0].params["num_designs"] == 160

        proposal = orchestrator.learner.propose_next(
            1,
            logical_parents,
            [],
            str(orchestrator.out_dir),
            policy_update={"binder_lengths": [85]},
            enable_exploitation_arms=False,
        )
        second_round = orchestrator._enforce_round_cap(proposal.jobs)

        assert sum(job.params["num_designs"] for job in second_round) == 160
        assert [job.params["num_designs"] for job in second_round] == [80, 80]
        assert all(job.params["multi_taiji_host_shard"]["source_num_designs"] == 160 for job in second_round)


def test_taiji_native_multi_host_keeps_one_cluster_job_and_budget() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        cfg.resource.backend = "taiji"
        cfg.resource.host_num = 2
        cfg.resource.host_gpu_num = 8
        cfg.resource.taiji_multi_host_mode = "native"
        cfg.resource.max_parallel_jobs = 8
        cfg.search_space.max_binders_per_round = 160
        cfg.search_space.num_designs_per_round = 160
        cfg.search_space.binder_length_range = [80, 120]
        cfg.search_space.binder_length_step = 5
        cfg.search_space.binder_lengths = [80, 85, 90, 95, 100, 105, 110, 115, 120]
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=Path(tmp), max_parallel=8, max_retries=1)

        first_round = orchestrator._enforce_round_cap(orchestrator._initial_jobs())
        assert orchestrator.max_parallel == 1
        assert len(first_round) == 1
        assert first_round[0].params["num_designs"] == 160
        assert first_round[0].params["host_count"] == 2
        assert first_round[0].params["taiji_submit_host_num"] == 2
        assert first_round[0].params["taiji_multi_host_mode"] == "native"
        assert first_round[0].params["round_budget_allocation"]["strategy"] == "single_taiji_multi_host_gpu_fanout"
        assert "multi_taiji_host_shard" not in first_round[0].params

        proposal = orchestrator.learner.propose_next(
            1,
            first_round,
            [],
            str(orchestrator.out_dir),
            policy_update={"binder_lengths": [85]},
            enable_exploitation_arms=False,
        )
        second_round = orchestrator._enforce_round_cap(proposal.jobs)
        assert len(second_round) == 1
        assert second_round[0].params["num_designs"] == 160
        assert second_round[0].params["native_taiji_multi_host"]["host_count"] == 2



def test_malicious_count_proposals_are_ignored_and_normal_budget_is_exact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        cfg.search_space.max_binders_per_round = 11
        cfg.search_space.num_designs_per_round = 11
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=Path(tmp), max_retries=1)
        applied, report = orchestrator._merge_next_round_updates(
            ("policy_proposal", {
                "num_designs": 999,
                "num_designs_per_round": 998,
                "max_binders_per_round": 997,
                "alpha": 0.003,
            }),
        )
        assert "num_designs" not in applied
        assert "num_designs_per_round" not in applied
        assert "max_binders_per_round" not in applied
        ignored = set(report["ignored_unsupported_keys"].get("policy_proposal", []))
        assert {"num_designs", "num_designs_per_round", "max_binders_per_round"} <= ignored
        jobs = orchestrator._enforce_round_cap(orchestrator._initial_jobs())
        assert sum(job.params["num_designs"] for job in jobs) == 11

def test_policy_resolution_preserves_input_configuration_owner() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=1)
        applied, report = orchestrator._merge_next_round_updates(
            ("input_configuration", {"diffusion_batch_size": 2, "filter_biased": "false"}),
            ("policy_proposal", {"diffusion_batch_size": 1, "filter_biased": "true", "inverse_fold_avoid": "C"}),
            apply=False,
        )
        assert applied["diffusion_batch_size"] == 2
        assert applied["filter_biased"] == "false"
        assert applied["inverse_fold_avoid"] == "C"
        assert {item["key"] for item in report["ownership_conflicts"]} == {"diffusion_batch_size", "filter_biased"}


def test_jobs_from_dicts_ignores_memory_params_summary() -> None:
    payload = {
        "job_id": "r0_round",
        "target_structure": "missing.cif",
        "chain_id": "A",
        "hotspots": [],
        "binder_length": 80,
        "seed": 0,
        "params": {"num_designs": 160},
        "output_dir": "out/r0",
        "params_summary": {"num_designs": 160},
        "status": "completed",
    }
    jobs = BinderDesignOrchestrator._jobs_from_dicts([payload])
    assert len(jobs) == 1
    assert jobs[0].job_id == "r0_round"
    assert jobs[0].params["num_designs"] == 160


def test_extend_memory_params_summary_is_summary_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperimentMemoryStore(Path(tmp))
        memory = store.load(target={"structure_path": "missing.cif"})
        job = DesignJob(
            job_id="r0_round",
            target_structure="missing.cif",
            chain_id="A",
            hotspots=[],
            binder_length=80,
            params={"num_designs": 160, "binder_lengths": [80, 90], "hotspot_weight": 1.0},
            output_dir="out/r0",
        )
        store.record_jobs(memory, 0, [job], extend_memory=True)

        assert "params_summary" not in memory.rounds[0].jobs[0]
        summary = store.summarize_for_agent(memory, extend_memory=True)
        assert summary["recent_rounds"][0]["jobs"][0]["params_summary"]["num_designs"] == 160


def test_resource_fields_are_not_agent_adjustable_and_base_params_restore_defaults() -> None:
    changes = supported_config_changes({"GPUName": "A100", "devices": 4, "taiji_timeout": 14400, "budget": 80, "secondary_structure": "alpha"})
    assert changes == {}

    cfg = _cfg()
    cfg.resource.host_gpu_num = 8
    cfg.resource.gpu_name = "V100"
    cfg.resource.timeout_seconds = 7200
    cfg.search_space.boltzgen.update({"GPUName": "A100", "devices": 4, "taiji_timeout": 14400})
    orchestrator = BinderDesignOrchestrator(cfg, out_dir=Path(tempfile.mkdtemp()), max_parallel=1, max_retries=1)
    params = orchestrator._base_params()
    assert params["GPUName"] == "V100"
    assert params["devices"] == 8
    assert params["taiji_timeout"] == 7200
    assert params["run_filtering"] is True
    assert "secondary_structure" not in params


def test_next_round_update_forces_filtering_and_drops_secondary_structure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=Path(tmp), max_parallel=1, max_retries=1)

        applied = orchestrator._apply_next_round_update({"run_filtering": False, "secondary_structure": "alpha", "budget": 20})

        assert applied["run_filtering"] is True
        assert cfg.search_space.boltzgen["run_filtering"] is True
        assert "secondary_structure" not in applied
        assert "secondary_structure" not in cfg.search_space.boltzgen


def test_pending_timeout_is_resource_scheduling_failure_not_ceph_failure() -> None:
    hints = RunMonitorAgent._failure_hints(
        '"state": "END", "msg": "state PENDING timeout(1318.1/1260) rtx:user"',
        ["steps_manifest", "analysis_metrics_candidates"],
    )
    assert "resource_scheduling_failure" in hints
    assert "missing_ceph_mount_secret" not in hints
    execution_failed, reason = BinderDesignOrchestrator._detect_round_execution_failure(
        total_candidates=0,
        execution_records=[
            {
                "status": "failed",
                "error": "missing_ceph_mount_secret;taiji_resource_or_queue_issue;missing_expected_outputs:steps_manifest",
            }
        ],
    )
    assert execution_failed is True
    assert reason == "resource_scheduling_failure"


def test_partial_taiji_failure_retries_only_failed_host_shards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        cfg.resource.backend = "taiji"
        cfg.resource.host_num = 2
        cfg.resource.host_gpu_num = 8
        cfg.resource.taiji_multi_host_mode = "split_jobs"
        cfg.resource.max_parallel_jobs = 8
        cfg.search_space.max_binders_per_round = 9
        cfg.search_space.num_designs_per_round = 9
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=Path(tmp), max_retries=1)
        jobs = orchestrator._enforce_round_cap(orchestrator._initial_jobs())
        assert [job.params["num_designs"] for job in jobs] == [5, 4]

        records = [
            {
                "job": jobs[0].__dict__,
                "job_id": jobs[0].job_id,
                "backend": "taiji",
                "status": "completed",
                "output_dir": jobs[0].output_dir,
            },
            {
                "job": jobs[1].__dict__,
                "job_id": jobs[1].job_id,
                "backend": "taiji",
                "status": "failed",
                "error": "evicted",
                "retryable": False,
            },
        ]
        execution_failed, reason = BinderDesignOrchestrator._detect_round_execution_failure(
            total_candidates=5,
            execution_records=records,
        )
        assert execution_failed is False
        assert reason == ""

        execution_state = BinderDesignOrchestrator._classify_execution_state(jobs, records)
        assert execution_state["state"] in {"partial", "partial_execution_failure"}
        assert execution_state["complete"] is False
        assert execution_state["realized_fraction"] == 0.5
        assert execution_state["successful_branch_ids"] == [jobs[0].job_id]
        assert execution_state["failed_branch_ids"] == [jobs[1].job_id]

        retry_jobs = orchestrator._retry_jobs_after_execution_failure(jobs, records, next_round_id=1)
        assert len(retry_jobs) == 1
        retry_job = retry_jobs[0]
        assert retry_job.params["execution_retry_source_job_id"] == jobs[1].job_id
        assert retry_job.params["num_designs"] == jobs[1].params["num_designs"] == 4
        assert retry_job.params["multi_taiji_host_shard"] == jobs[1].params["multi_taiji_host_shard"]
        assert retry_job.params["multi_taiji_host_shard"]["shard_id"] == "2_of_2"


def test_execution_failure_retry_replaces_stale_job_identity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=1)
        from binderloop.strategy_governance import materialize_deterministic_job_identities, effective_semantic_digest
        source = materialize_deterministic_job_identities([_job(Path(tmp))], round_id=0, output_root=tmp)[0]
        stale = dict(source.params["job_identity"])
        record_job = DesignJob(**{**source.__dict__, "params": {**source.params, "alpha": 0.003}})
        records = [{"job_id": source.job_id, "status": "failed", "job": record_job.__dict__}]
        retry = orchestrator._retry_jobs_after_execution_failure([source], records, next_round_id=1)[0]
        assert retry.job_id.startswith("r1_")
        assert retry.params["job_identity"] != stale
        assert retry.params["job_identity"]["execution_semantic_digest"] == effective_semantic_digest(retry)
        orchestrator._validate_job_identities([retry])
        assert retry.params["execution_retry_source_job_id"] == source.job_id



def test_typed_retry_patch_applies_over_source_and_preserves_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_retries=1)
        job = _job(out_dir)
        job.params = {
            "alpha": 0.001,
            "config_overrides": [["filtering", "bad=true"]],
            "custom_metadata": {"owner": "user", "nested": {"kept": True}},
            "unrelated": "preserved",
        }
        record = {
            "job_id": job.job_id,
            "status": "failed",
            "error": "deterministic config failure",
            "retry_correction_proposal": {
                "requires_refinalization": True,
                "correction_patch": {
                    "version": 1,
                    "set": {"alpha": 0.003, "custom_metadata.reviewed": True},
                    "remove": ["config_overrides"],
                    "classification": "invalid_config",
                    "identity_effect": "semantic_change",
                    "source_validation_digest": "validation-abc",
                },
            },
        }
        retry = orchestrator._retry_jobs_after_execution_failure([job], [record], next_round_id=1)[0]
        assert retry.params["alpha"] == 0.003
        assert retry.params["unrelated"] == "preserved"
        assert "config_overrides" not in retry.params
        assert retry.params["custom_metadata"] == {
            "owner": "user", "nested": {"kept": True}, "reviewed": True,
        }
        audit = retry.params["retry_metadata"]["correction_patch"]
        assert audit["version"] == 1
        assert audit["classification"] == "invalid_config"
        assert audit["identity_effect"] == "semantic_change"
        assert audit["source_validation_digest"] == "validation-abc"
        assert retry.params["job_identity"]["semantic_digest"]


def test_legacy_corrected_params_remain_compatible_and_keep_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_retries=1)
        job = _job(out_dir)
        orchestration_metadata = {
            "round_budget_resolution": {"requested": 8, "allocated": 4},
            "arm_rank": 2,
            "arm_root": "/runs/r1/arm2",
            "template_application_plan": {"mode": "structure_redesign"},
            "lineage_identity": {"parent": "source-job"},
        }
        job.params = {
            "alpha": 0.001,
            "obsolete": True,
            "custom_metadata": {"trace": "keep"},
            "branch_id": "source-branch",
            **orchestration_metadata,
        }
        record = {
            "job_id": job.job_id,
            "status": "failed",
            "error": "legacy correction",
            "retry_correction_proposal": {
                "requires_refinalization": True,
                "corrected_params": {"alpha": 0.004},
            },
        }
        compatible, audit = orchestrator._apply_retry_correction_patch(
            job.params, record["retry_correction_proposal"],
        )
        assert compatible["alpha"] == 0.004
        assert compatible["obsolete"] is True
        assert compatible["custom_metadata"] == {"trace": "keep"}
        for key, value in orchestration_metadata.items():
            assert compatible[key] == value
        assert audit["legacy_corrected_params"] is True

        retry = orchestrator._retry_jobs_after_execution_failure([job], [record], next_round_id=1)[0]
        assert retry.params["custom_metadata"] == {"trace": "keep"}
        assert retry.params["round_budget_resolution"] == orchestration_metadata["round_budget_resolution"]
        assert retry.params["arm_rank"] == orchestration_metadata["arm_rank"]
        assert retry.params["template_application_plan"] == orchestration_metadata["template_application_plan"]
        assert retry.params["lineage_identity"] == orchestration_metadata["lineage_identity"]
        assert retry.params["arm_root"] != orchestration_metadata["arm_root"]
        assert retry.params["retry_metadata"]["correction_patch"]["legacy_corrected_params"] is True



@pytest.mark.parametrize(
    "metadata_key,value",
    [
        ("branch_id", "source-branch"),
        ("round_budget_resolution", {"requested": 8, "allocated": 4}),
        ("arm_rank", 2),
        ("template_application_plan", {"mode": "structure_redesign"}),
        ("lineage_identity", {"parent": "source-job"}),
        ("retry_metadata", {"existing": {"nested": [1, 2]}}),
    ],
)
def test_legacy_corrected_params_preserve_parameterized_contract_metadata(
    metadata_key, value,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=1)
        compatible, audit = orchestrator._apply_retry_correction_patch(
            {"alpha": 0.001, metadata_key: value},
            {"corrected_params": {"alpha": 0.004, metadata_key: "malicious"}},
        )
        assert compatible[metadata_key] == value
        assert any(item["path"] == metadata_key for item in audit["ignored_metadata_overrides"])


@pytest.mark.parametrize(
    "contract_key,initial,corrected",
    [
        ("num_designs", 8, 4),
        ("binder_lengths", [50], [60]),
        ("host_count", 2, 1),
    ],
)
def test_legacy_corrected_params_replace_each_execution_partition_only(
    contract_key, initial, corrected,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=1)
        original = {
            contract_key: initial,
            "branch_id": "source-branch",
            "arbitrary_payload": {"nested": [1, {"kept": True}]},
        }
        compatible, audit = orchestrator._apply_retry_correction_patch(
            original, {"corrected_params": {contract_key: corrected}},
        )
        assert compatible[contract_key] == corrected
        assert compatible["branch_id"] == "source-branch"
        assert compatible["arbitrary_payload"] == {"nested": [1, {"kept": True}]}
        assert audit["replaced_partitions"] == ["runner", "adapter", "runtime"]


@pytest.mark.parametrize(
    "unknown_key,value",
    [
        ("metadata_without_suffix", {"nested": {"items": [1, 2, {"ok": True}]}}),
        ("anything_at_all", ["a", {"deep": [3, 4]}]),
    ],
)
def test_legacy_corrected_params_preserve_arbitrarily_named_unknown_metadata(
    unknown_key, value,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=1)
        compatible, _ = orchestrator._apply_retry_correction_patch(
            {"alpha": 0.001, unknown_key: value},
            {"corrected_params": {"alpha": 0.004}},
        )
        assert compatible[unknown_key] == value


def test_retry_corrections_audit_and_ignore_orchestrator_owned_metadata_overrides() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=1)
        original = {
            "alpha": 0.001,
            "branch_id": "trusted-branch",
            "job_identity": {"job_id": "trusted-job"},
            "unknown_blob": {"kept": [1, 2]},
        }
        compatible, typed_audit = orchestrator._apply_retry_correction_patch(original, {
            "correction_patch": {
                "version": 1,
                "set": {"alpha": 0.004, "branch_id": "evil", "job_identity.job_id": "evil"},
                "remove": ["unknown_blob", "branch_id"],
            },
        })
        assert compatible["alpha"] == 0.004
        assert compatible["branch_id"] == "trusted-branch"
        assert compatible["job_identity"] == {"job_id": "trusted-job"}
        assert "unknown_blob" not in compatible
        assert {item["path"] for item in typed_audit["ignored_orchestrator_metadata_overrides"]} == {
            "branch_id", "job_identity.job_id",
        }

        legacy, legacy_audit = orchestrator._apply_retry_correction_patch(original, {
            "corrected_params": {"alpha": 0.005, "branch_id": "evil", "unknown_blob": "evil"},
        })
        assert legacy["alpha"] == 0.005
        assert legacy["branch_id"] == "trusted-branch"
        assert legacy["unknown_blob"] == {"kept": [1, 2]}
        assert {item["path"] for item in legacy_audit["ignored_metadata_overrides"]} == {
            "branch_id", "unknown_blob",
        }


def test_retry_refinalization_audit_records_prior_and_final_values() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_retries=1)
        source = orchestrator._finalize_semantic_job_identities([_job(out_dir)], round_id=0)[0]
        prior = {
            "job_id": source.job_id,
            "output_dir": source.output_dir,
            "job_identity": source.params["job_identity"],
        }
        record = {
            "job_id": source.job_id,
            "status": "failed",
            "retry_correction_proposal": {
                "requires_refinalization": True,
                "corrected_params": {"alpha": 0.004},
            },
        }
        retry = orchestrator._retry_jobs_after_execution_failure([source], [record], next_round_id=1)[0]
        audit = retry.params["retry_metadata"]["refinalization"]
        assert audit["required"] is True
        assert audit["prior_values"]["job_id"] == prior["job_id"]
        assert audit["prior_values"]["output_dir"] == prior["output_dir"]
        assert audit["prior_values"]["job_identity"] == prior["job_identity"]
        assert audit["final_values"]["job_id"] == retry.job_id
        assert audit["final_values"]["output_dir"] == retry.output_dir
        assert audit["final_values"]["job_identity"] == retry.params["job_identity"]
        assert audit["final_values"]["job_id"] != audit["prior_values"]["job_id"]

def test_retry_correction_patch_unknown_version_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=1)
        for version, expected in [(2, "unsupported_retry_correction_patch_version:2"), ("future", "invalid_retry_correction_patch_version:'future'")]:
            record = {
                "job_id": "retry_limit_job", "status": "failed", "error": "invalid config",
                "retry_correction_proposal": {
                    "requires_refinalization": True,
                    "correction_patch": {"version": version, "set": {"alpha": 0.004}},
                },
            }
            try:
                orchestrator._retry_jobs_after_execution_failure([_job(Path(tmp))], [record], next_round_id=1)
            except ValueError as exc:
                assert str(exc) == expected
            else:
                raise AssertionError(f"patch version {version!r} was accepted")


def test_cross_round_semantic_failure_breaker_ignores_refinalized_identity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_retries=1)
        first = orchestrator._finalize_semantic_job_identities([_job(out_dir)], round_id=0)[0]
        first_record = {
            "job": first.__dict__,
            "job_id": first.job_id,
            "job_identity_digest": first.params["job_identity"]["semantic_digest"],
            "status": "failed",
            "error": "ValueError: deterministic schema mismatch at /tmp/r0/job.json",
        }
        first_record["semantic_failure_fingerprint"] = orchestrator._semantic_failure_fingerprint(first_record)
        first_record["semantic_failure_scope"] = orchestrator._semantic_failure_scope(first)
        round_dir = out_dir / "round_00"
        round_dir.mkdir(parents=True)
        (round_dir / "execution_records.json").write_text(json.dumps([first_record]), encoding="utf-8")

        second = orchestrator._finalize_semantic_job_identities([_job(out_dir)], round_id=1)[0]
        assert second.job_id != first.job_id
        second_record = {
            "job": second.__dict__,
            "job_id": second.job_id,
            "job_identity_digest": second.params["job_identity"]["semantic_digest"],
            "status": "failed",
            "error": "ValueError: deterministic schema mismatch at /tmp/r1/job.json",
        }
        assert orchestrator._semantic_failure_fingerprint(second_record) == first_record["semantic_failure_fingerprint"]
        retries = orchestrator._retry_jobs_after_execution_failure([second], [second_record], next_round_id=2)
        assert len(retries) == 1
        assert retries[0].params["continuation_kind"] == "fresh_complementary_arm"
        assert retries[0].params["arm_id"] != second.params.get("arm_id")
        assert second_record["semantic_retry_circuit_breaker"] == "cross_round_identical_semantic_failure"



def test_semantic_failure_fingerprint_includes_blocking_config_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=1)
        base = {
            "status": "failed",
            "error": "pre-submit config validation failed at /tmp/r7/job.json",
            "config_validation": {
                "issues": [{
                    "parameter": "budget", "severity": "error", "resolved": False,
                    "problem": "Required submission readiness key is missing.",
                    "correction": "Provide a positive integer.",
                }],
                "missing_required_keys": ["budget"],
                "semantic_changes": [
                    {"parameter": "alpha", "before": 0.1, "after": 0.2, "change": "replacement", "partition": "runner"},
                    {"parameter": "filter_biased", "before": True, "after": "true", "change": "normalization", "partition": "runner"},
                    {"parameter": "branch_id", "before": "r7_a", "after": "r8_b", "change": "replacement", "partition": "orchestration"},
                ],
            },
        }
        same = json.loads(json.dumps(base))
        same["job_id"] = "different_identity"
        same["error"] = "pre-submit config validation failed at /var/tmp/r8/other.json"
        same["config_validation"]["semantic_changes"][2]["after"] = "r9_c"
        assert orchestrator._semantic_failure_fingerprint(base) == orchestrator._semantic_failure_fingerprint(same)

        nested_pre_submit = json.loads(json.dumps(base))
        nested_pre_submit["pre_submit"] = {"config_validation": nested_pre_submit.pop("config_validation")}
        assert orchestrator._semantic_failure_fingerprint(base) == orchestrator._semantic_failure_fingerprint(nested_pre_submit)

        different_blocker = json.loads(json.dumps(base))
        different_blocker["config_validation"]["issues"][0]["parameter"] = "num_designs"
        different_blocker["config_validation"]["missing_required_keys"] = ["num_designs"]
        assert orchestrator._semantic_failure_fingerprint(base) != orchestrator._semantic_failure_fingerprint(different_blocker)

        different_semantic_change = json.loads(json.dumps(base))
        different_semantic_change["config_validation"]["semantic_changes"][0]["after"] = 0.3
        assert orchestrator._semantic_failure_fingerprint(base) != orchestrator._semantic_failure_fingerprint(different_semantic_change)


def test_semantic_failure_fingerprint_version_persists_to_record_ledger_and_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
        records = orchestrator._run_jobs(
            [_job(out_dir)], 0,
            lambda job, attempt: {"job_id": job.job_id, "status": "failed", "error": "deterministic", "retryable": False},
            attempts_path=out_dir / "round_00" / "execution_attempts.json",
        )
        assert records[0]["semantic_failure_fingerprint_version"] == 1
        ledger = json.loads((out_dir / "round_00" / "execution_attempts.json").read_text())
        attempt = ledger["jobs"]["retry_limit_job"]["attempts"][0]
        assert attempt["semantic_failure_fingerprint_version"] == 1


def test_retryable_taiji_failure_resubmits_within_retry_cap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=2)
        calls: List[int] = []

        def taiji_executor(job: DesignJob, attempt: int):
            calls.append(attempt)
            if attempt == 1:
                return {
                    "job_id": job.job_id,
                    "backend": "taiji",
                    "status": "failed",
                    "error": "transient connection reset",
                    "retryable": True,
                }
            return {
                "job_id": job.job_id,
                "backend": "taiji",
                "status": "completed",
                "output_dir": job.output_dir,
            }

        records = orchestrator._run_jobs(
            [_job(out_dir)],
            0,
            taiji_executor,
            attempts_path=out_dir / "round_00" / "execution_attempts.json",
        )
        assert calls == [1, 2], calls
        assert records[0]["status"] == "completed"


def test_resource_scheduling_failure_returns_refinalized_next_round_proposal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        cfg = _cfg()
        cfg.resource.host_gpu_num = 8
        cfg.resource.gpu_name = "V100"
        cfg.resource.timeout_seconds = 7200
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=out_dir, max_parallel=1, max_retries=2)
        job = _job(out_dir)
        job.params.update({"GPUName": "A100", "devices": 8, "taiji_timeout": 7200})
        original = dict(job.params)
        calls: List[int] = []

        def taiji_executor(execution_job: DesignJob, attempt: int):
            calls.append(attempt)
            execution_job.params["executor_accidental_mutation"] = True
            return {
                "job_id": execution_job.job_id,
                "backend": "taiji",
                "status": "failed",
                "error": "taiji_resource_or_queue_issue",
                "retryable": False,
            }

        records = orchestrator._run_jobs(
            [job], 0, taiji_executor,
            attempts_path=out_dir / "round_00" / "execution_attempts.json",
        )
        assert calls == [1]
        assert job.params == original
        proposal = records[0]["retry_correction_proposal"]
        assert proposal["requires_refinalization"] is True
        assert proposal["corrected_params"]["devices"] == 4
        retry = orchestrator._retry_jobs_after_execution_failure([job], records, next_round_id=1)[0]
        assert retry.job_id != job.job_id
        assert retry.output_dir != job.output_dir
        assert retry.params["devices"] == 4
        assert retry.params["taiji_timeout"] == 14400
        assert "executor_accidental_mutation" not in retry.params


def test_identical_failure_fingerprint_opens_retry_circuit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=5)
        calls: List[int] = []

        def executor(job: DesignJob, attempt: int):
            calls.append(attempt)
            return {"job_id": job.job_id, "status": "failed", "error": "same deterministic failure"}

        records = orchestrator._run_jobs(
            [_job(out_dir)], 0, executor,
            attempts_path=out_dir / "round_00" / "execution_attempts.json",
        )
        assert calls == [1, 2]
        assert records[0]["retry_circuit_breaker"] == "identical_failure_fingerprint"
        ledger = json.loads((out_dir / "round_00" / "execution_attempts.json").read_text())
        assert ledger["jobs"]["retry_limit_job"]["attempts"][1]["failure_fingerprint"]


def test_persisted_retryable_taiji_terminal_record_can_resubmit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        attempts_path = out_dir / "round_00" / "execution_attempts.json"
        attempts_path.parent.mkdir(parents=True, exist_ok=True)
        attempts_path.write_text(
            json.dumps(
                {
                    "round_id": 0,
                    "max_attempts_per_job": 2,
                    "jobs": {
                        "retry_limit_job": {
                            "job": {},
                            "attempts": [{"attempt": 1, "status": "failed"}],
                            "terminal_record": {
                                "job_id": "retry_limit_job",
                                "backend": "taiji",
                                "status": "failed",
                                "attempts": 1,
                                "error": "evicted",
                                "retryable": False,
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=2)
        calls: List[int] = []

        def taiji_executor(job: DesignJob, attempt: int):
            calls.append(attempt)
            return {
                "job_id": job.job_id,
                "backend": "taiji",
                "status": "completed",
                "output_dir": job.output_dir,
            }

        records = orchestrator._run_jobs([_job(out_dir)], 0, taiji_executor, attempts_path=attempts_path)
        assert calls == [2], calls
        assert records[0]["status"] == "completed"


def test_non_retryable_taiji_environment_failure_does_not_resubmit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=3)
        calls: List[int] = []

        def taiji_executor(job: DesignJob, attempt: int):
            calls.append(attempt)
            return {
                "job_id": job.job_id,
                "backend": "taiji",
                "status": "failed",
                "error": "missing_input_file",
                "retryable": False,
            }

        records = orchestrator._run_jobs(
            [_job(out_dir)],
            0,
            taiji_executor,
            attempts_path=out_dir / "round_00" / "execution_attempts.json",
        )
        assert calls == [1], calls
        assert records[0]["status"] == "failed"
        assert records[0]["retryable"] is False


def test_all_failed_execution_records_are_valid_module_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        round_dir = out_dir / "round_00"
        round_dir.mkdir(parents=True, exist_ok=True)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=3)
        job = _job(out_dir)
        records = [
            {
                "job": job.__dict__,
                "job_id": job.job_id,
                "attempts": 1,
                "status": "failed",
                "error": "missing_expected_outputs",
                "retryable": False,
            }
        ]
        records_path = orchestrator._write_json(round_dir / "execution_records.json", records)

        orchestrator._validate_execution_module({"records": records, "path": str(records_path)}, [job])


def test_validated_module_uses_custom_attempt_limit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        round_dir = out_dir / "round_00"
        round_dir.mkdir(parents=True, exist_ok=True)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
        calls: List[int] = []

        def invalid_module():
            calls.append(len(calls) + 1)
            return {"bad": True}

        def always_invalid(result):
            raise RuntimeError("bad format")

        try:
            orchestrator._run_validated_module(
                module_name="results_ingested",
                round_id=0,
                round_dir=round_dir,
                checkpoint={},
                action=invalid_module,
                validator=always_invalid,
                max_attempts=4,
            )
        except RuntimeError as exc:
            assert "bad format" in str(exc)
        else:
            raise AssertionError("invalid module unexpectedly passed validation")
        assert calls == [1, 2, 3, 4], calls



def test_validated_module_stops_non_retryable_ingestion_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp); round_dir = out_dir / "round_00"; round_dir.mkdir(parents=True)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=5)
        calls = []
        def invalid_module():
            calls.append(1)
            raise ValueError("invalid v2 result manifest (fail closed): digest mismatch")
        try:
            orchestrator._run_validated_module(
                module_name="results_ingested", round_id=0, round_dir=round_dir, checkpoint={},
                action=invalid_module, validator=lambda result: None, max_attempts=5,
                retry_predicate=orchestrator._ingestion_error_retryable,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("deterministic ingestion error unexpectedly passed")
        assert calls == [1]


def test_config_validation_result_legacy_constructor_inherits_validity_fail_closed() -> None:
    assert ConfigValidationResult("boltzgen", "legacy", False, False).is_submittable is False
    assert ConfigValidationResult("boltzgen", "legacy", False, True).is_submittable is True
    assert ConfigValidationResult(
        "boltzgen", "v2", False, True, is_submittable=False,
    ).is_submittable is False


def test_config_validation_v2_partitions_and_readiness_fail_closed() -> None:
    from binderloop.agents.config_parameter_contract import parameter_contract_entry, partition_config_parameters

    partitioned = partition_config_parameters({
        "num_designs": 8,
        "binder_lengths": [80],
        "host_count": 2,
        "branch_id": "r1_probe",
        "template_requested": True,
        "mystery": 1,
    })
    assert partitioned["runner"] == {"num_designs": 8}
    assert partitioned["adapter"] == {"binder_lengths": [80]}
    assert partitioned["runtime"] == {"host_count": 2}
    assert partitioned["orchestration"]["branch_id"] == "r1_probe"
    assert partitioned["unknown"] == {"mystery": 1}
    assert parameter_contract_entry("branch_id")["policy_class"] == "identity_policy"
    assert parameter_contract_entry("template_requested")["policy_class"] == "template_policy"

    missing = ConfigValidationAgent().validate_for_submission({"num_designs": 8})
    assert missing.schema_version == 2
    assert missing.is_valid is True  # legacy shape validity remains compatible
    assert missing.is_submittable is False
    assert missing.missing_required_keys == ["budget"]

    ready = ConfigValidationAgent().validate_for_submission({"num_designs": "8", "budget": "20", "filter_biased": True})
    assert ready.is_submittable is True
    assert ready.corrected_config["filter_biased"] == "true"
    assert ready.validated_partition["runner"]["filter_biased"] == "true"
    assert {item["parameter"] for item in ready.normalizations} >= {"num_designs", "budget", "filter_biased"}
    assert ready.missing_required_keys == []


def test_config_validation_v2_tracks_semantic_removals_and_llm_stays_advisory() -> None:
    agent = ConfigValidationAgent(_FakeLLM({
        "is_valid": False,
        "corrected_config": {"num_designs": 999},
        "issues": [{"parameter": "budget", "severity": "error", "problem": "advisory veto", "correction": "remove"}],
        "recommendations": [],
    }))
    result = agent.validate_for_submission({"num_designs": 8, "budget": 20, "secondary_structure": "alpha"})
    assert result.is_submittable is True
    assert result.corrected_config["num_designs"] == 8
    assert result.corrected_config["budget"] == 20
    assert any(item["parameter"] == "secondary_structure" for item in result.removals)
    assert not any(item["parameter"] == "secondary_structure" for item in result.semantic_changes)
    assert result.requires_refinalization is False
    assert any(issue.get("advisory") for issue in result.issues if issue.get("parameter") == "budget")


def test_v24_full_job_metadata_is_safe_and_defaults_do_not_refinalize() -> None:
    params = {
        "num_designs": 24, "budget": 24, "protocol": "protein-anything",
        "job_identity": {"schema_version": 1, "job_id": "r24_job"},
        "arm_rank": 0, "arm_root": "/runs/r24/arm0", "arm_digest": "arm-digest",
        "logical_branch_id": "r24_baseline", "logical_job_id": "r24_logical",
        "execution_job_id": "r24_job", "execution_slot": 0,
        "target_identity_digest": "target-digest",
        "round_budget_resolution": {"requested": 24, "allocated": 24},
        "immutable_branch_plan": {"schema_version": 1, "allocated_designs": 24},
        "arm_gpu_allocation": {"devices": 1},
        "template_application_plan": {"mode": "free"},
        "lineage_identity": {"parent": "root"},
        "template_execution_identity": {"template_id": "none"},
        "round_budget_allocation": {"num_designs": 24},
        "native_taiji_multi_host": {"enabled": False},
        "parameter_catalog": {"alpha": [0.001]},
        "parameter_catalog_digest": "catalog-digest",
        "final_parameter_state": {"alpha": 0.001},
        "resolved_intervention_plan": {"intent_digest": "intent"},
        "secondary_structure": "not executable",
    }
    result = ConfigValidationAgent().validate_for_submission(params, target_model="boltzgen")
    assert result.is_valid is True
    assert result.is_submittable is True
    assert result.requires_refinalization is False
    assert result.missing_required_keys == []
    assert result.validated_partition["orchestration"]["job_identity"]["job_id"] == "r24_job"
    assert any(item["parameter"] == "secondary_structure" for item in result.removals)
    assert not any(item["parameter"] == "run_filtering" for item in result.semantic_changes)

def test_config_validation_normalizes_boltzgen_config_overrides() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {
            "config_overrides": ["filtering, filter_bindingsite=true"],
            "additional_filters": "iptm>0.35",
            "num_designs": "8",
        },
        target_model="boltzgen",
    )
    assert result.corrected_config["config_overrides"] == [["filtering", "filter_bindingsite=true"]]
    assert result.corrected_config["additional_filters"] == ["iptm>0.35"]
    assert result.corrected_config["num_designs"] == 8


def test_config_validation_forces_filtering_and_drops_secondary_structure() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {
            "run_filtering": False,
            "secondary_structure": "alpha",
            "num_designs": 8,
        },
        target_model="boltzgen",
    )

    assert result.is_valid
    assert result.corrected_config["run_filtering"] is True
    assert "secondary_structure" not in result.corrected_config


def test_config_validation_trusts_sanitized_config_over_llm_missing_key_conflict() -> None:
    agent = ConfigValidationAgent(
        _FakeLLM(
            {
                "is_valid": True,
                "corrected_config": {
                    "num_designs": 80,
                    "num_designs_per_round": 80,
                    "max_binders_per_round": 80,
                },
                "issues": [
                    {
                        "parameter": "num_designs_per_round",
                        "severity": "error",
                        "problem": "The supported key num_designs_per_round was missing.",
                        "correction": "Restored num_designs_per_round with value 80.",
                    }
                ],
                "recommendations": [],
            }
        )
    )

    result = agent.validate_for_submission({"num_designs": 80, "max_binders_per_round": 80}, target_model="boltzgen")

    assert result.is_valid
    assert result.corrected_config["num_designs"] == 80
    assert result.corrected_config["max_binders_per_round"] == 80
    assert "num_designs_per_round" not in result.corrected_config
    matching = [issue for issue in result.issues if issue.get("parameter") == "num_designs_per_round"]
    assert matching
    assert matching[-1]["resolved"] is True


def test_full_job_validation_preserves_user_fields_and_treats_llm_as_advisory() -> None:
    agent = ConfigValidationAgent(
        _FakeLLM(
            {
                "is_valid": False,
                "corrected_config": {
                    "additional_filters": [{"feature": "design_to_target_iptm", "threshold": 0.35, "lower_is_better": False}],
                    "binder_lengths": ["80", "95", "bad"],
                },
                "issues": [
                    {
                        "parameter": "additional_filters",
                        "severity": "error",
                        "problem": "additional_filters is user-owned and must not be emitted in executable config.",
                        "correction": "Remove it.",
                    },
                    {
                        "parameter": "binder_lengths",
                        "severity": "error",
                        "problem": "binder_lengths is internal-only and must not be submitted.",
                        "correction": "Remove it.",
                    },
                ],
                "recommendations": [],
            }
        )
    )

    result = agent.validate_for_submission(
        {
            "additional_filters": "design_to_target_iptm>0.35",
            "binder_lengths": [80, 90],
        },
        target_model="boltzgen",
    )

    assert result.is_valid
    assert result.corrected_config["additional_filters"] == ["design_to_target_iptm>0.35"]
    assert result.corrected_config["binder_lengths"] == [80, 90]
    unresolved = [
        issue for issue in result.issues
        if issue.get("parameter") in {"additional_filters", "binder_lengths"}
        and str(issue.get("severity", "")).lower() == "error"
        and not issue.get("resolved")
    ]
    assert unresolved == []
    assert any(issue.get("advisory") for issue in result.issues if issue.get("parameter") == "additional_filters")


def test_agent_delta_validation_rejects_user_owned_full_job_fields() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_agent_delta(
        {
            "hotspot_weight": 2.0,
            "additional_filters": "iptm>0.35",
            "run_filtering": True,
            "target_include": [{"chain": {"id": "E", "res_index": "1..194"}}],
            "binder_lengths": [80, 90],
        },
        target_model="boltzgen",
    )

    assert result.is_valid
    assert "hotspot_weight" not in result.corrected_config
    assert result.corrected_config["binder_lengths"] == [80, 90]
    assert "additional_filters" not in result.corrected_config
    assert "run_filtering" not in result.corrected_config
    assert "target_include" not in result.corrected_config


def test_config_validation_renders_dict_additional_filters_as_boltzgen_cli_tokens() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {
            "additional_filters": [
                {"feature": "ALA_fraction", "threshold": 0.3, "lower_is_better": True},
                {"feature": "design_to_target_iptm", "threshold": 0.35, "lower_is_better": False},
            ]
        },
        target_model="boltzgen",
    )

    assert result.is_valid
    assert result.corrected_config["additional_filters"] == ["ALA_fraction<0.3", "design_to_target_iptm>0.35"]


def test_config_validation_drops_designfolding_iptm_additional_filter() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {
            "additional_filters": [
                "iptm>0.35",
                "designfolding_iptm>0.15",
                {"feature": "filter_designfolding_iptm", "threshold": 0.3, "lower_is_better": False},
            ]
        },
        target_model="boltzgen",
    )

    assert result.is_valid
    assert result.corrected_config["additional_filters"] == ["iptm>0.35"]
    assert any("designfolding_iptm" in issue.get("problem", "") for issue in result.issues)


def test_config_validation_repairs_invalid_config_failure_context() -> None:
    agent = ConfigValidationAgent()
    result = agent.improve_after_failure(
        {"config_overrides": ["filtering, filter_bindingsite=true"]},
        target_model="boltzgen",
        error_context={
            "boltzgen_log_tail": "ValueError: Invalid config: ['filtering, filter_bindingsite=true']. Expected format: <step_name> <arg1>=<value1>",
        },
    )
    assert result.corrected_config["config_overrides"] == [["filtering", "filter_bindingsite=true"]]


def test_config_validation_normalizes_boltzgen_choice_flags_and_enums() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {
            "filter_biased": True,  # Python bool -> must become lowercase "true"
            "protocol": "Protein-Anything",
            "steps": ["design", "filter", "folding"],  # "filter" is invalid
            "exploit_fragment_modules": ["frag_Union[deadbeef00ff", "A:1-A:8"],
        },
        target_model="boltzgen",
    )
    assert result.corrected_config["filter_biased"] == "true"
    assert result.corrected_config["protocol"] == "protein-anything"
    assert result.corrected_config["steps"] == ["design", "folding"]
    assert "exploit_fragment_modules" not in result.corrected_config
    assert result.corrected_config["deprecated_strategy_audit"]["exploit_fragment_modules"]["status"] == "deprecated_audit_only"
    assert result.is_valid


def test_config_validation_is_model_aware_for_odesign() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {
            "use_msa": True,  # Hydra bool -> lowercase "true"
            "invfold_use_beam": False,
            "design_modality": "Protein",
            "center_method": "bad_center",  # invalid enum -> dropped
            "N_sample": "5",
            "seeds": [1, "x", 3],
            "some_future_key": 123,  # no boltzgen whitelist for odesign -> kept
        },
        target_model="odesign",
    )
    assert result.corrected_config["use_msa"] == "true"
    assert result.corrected_config["invfold_use_beam"] == "false"
    assert result.corrected_config["design_modality"] == "protein"
    assert "center_method" not in result.corrected_config
    assert result.corrected_config["N_sample"] == 5
    assert result.corrected_config["seeds"] == [1, 3]
    assert result.corrected_config["some_future_key"] == 123


def test_retry_intervention_correlates_runtime_error_to_parameter() -> None:
    agent = ConfigValidationAgent()
    # Reproduces the real v13 r1 failure: config value is still the Python bool.
    result = agent.improve_after_failure(
        {"filter_biased": True, "num_designs": 80},
        target_model="boltzgen",
        error_context={
            "boltzgen_log_tail": "boltzgen run: error: argument --filter_biased: invalid choice: 'True' (choose from 'true', 'false')",
        },
    )
    analysis = result.raw["runtime_error_analysis"]
    assert analysis["config_related"] is True
    assert analysis["findings"][0]["parameter"] == "filter_biased"
    assert result.corrected_config["filter_biased"] == "true"
    assert result.is_valid


def test_retry_intervention_removes_unrecognized_argument() -> None:
    agent = ConfigValidationAgent()
    result = agent.improve_after_failure(
        {"frobnicate": 5, "num_designs": 80},
        target_model="boltzgen",
        error_context={"boltzgen_log_tail": "boltzgen run: error: unrecognized arguments: --frobnicate 5"},
    )
    assert "frobnicate" not in result.corrected_config
    assert result.raw["runtime_error_analysis"]["config_related"] is True


def test_config_validation_repairs_strategy_keys() -> None:
    agent = ConfigValidationAgent()
    # Valid values normalize (case folding, bool/float coercion).
    ok = agent.validate_for_submission(
        {
            "epitope_crop_mode": "Hotspot_Focus",
            "fragment_template_gate": "IPTM",
            "fragment_interchain_pae_max": "12.5",
            "auto_binder_length": "yes",
        },
        target_model="boltzgen",
    )
    assert ok.corrected_config["epitope_crop_mode"] == "hotspot_focus"
    assert ok.corrected_config["fragment_template_gate"] == "iptm"
    assert ok.corrected_config["fragment_interchain_pae_max"] == 12.5
    assert ok.corrected_config["auto_binder_length"] is True
    assert ok.is_valid

    # Invalid values are dropped (resolved) so the round still proceeds on defaults.
    bad = agent.validate_for_submission(
        {
            "epitope_crop_mode": "crop_everything",
            "fragment_template_gate": "magic",
            "fragment_interchain_pae_max": "NaN",
            "auto_binder_length": "maybe",
        },
        target_model="boltzgen",
    )
    assert "epitope_crop_mode" not in bad.corrected_config
    assert "fragment_template_gate" not in bad.corrected_config
    assert "fragment_interchain_pae_max" not in bad.corrected_config
    assert "auto_binder_length" not in bad.corrected_config
    assert bad.is_valid


def test_config_validation_keeps_prioritize_hotspots_boolean() -> None:
    # Deprecated hotspot intent is audit-only and never reaches a new execution plan.
    agent = ConfigValidationAgent()
    ok = agent.validate_for_submission({"prioritize_hotspots": "true"}, target_model="boltzgen")
    assert "prioritize_hotspots" not in ok.corrected_config
    assert ok.is_valid

    bad_shape = agent.validate_for_submission({"prioritize_hotspots": [True]}, target_model="boltzgen")
    assert "prioritize_hotspots" not in bad_shape.corrected_config
    assert bad_shape.is_valid


def test_retry_intervention_ignores_non_config_runtime_failure() -> None:
    agent = ConfigValidationAgent()
    result = agent.improve_after_failure(
        {"num_designs": 80},
        target_model="boltzgen",
        error_context={"boltzgen_log_tail": "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"},
    )
    assert result.raw["runtime_error_analysis"]["config_related"] is False


def test_config_validation_coalesces_flat_config_overrides() -> None:
    # The v15 failure source: a single intended --config group accidentally split
    # into sibling string items. It must be coalesced into one nested override,
    # not silently dropped item-by-item.
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {"config_overrides": ["filtering", "filter_bindingsite=true"]},
        target_model="boltzgen",
    )
    assert result.corrected_config["config_overrides"] == [["filtering", "filter_bindingsite=true"]]
    assert result.is_valid


def test_config_validation_drops_invalid_boltzgen_filter_override_key() -> None:
    # filter_rmsd_threshold crashes BoltzGen's Filter with an unexpected-keyword
    # argument; it must be stripped before submission so no GPU job is wasted.
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {"config_overrides": ["filtering", "filter_rmsd_threshold=5.0"]},
        target_model="boltzgen",
    )
    assert result.corrected_config["config_overrides"] == []
    assert result.is_valid


def test_config_validation_keeps_valid_override_drops_only_invalid_token() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {"config_overrides": [["filtering", "filter_rmsd_threshold=5.0", "filter_bindingsite=true"]]},
        target_model="boltzgen",
    )
    assert result.corrected_config["config_overrides"] == [["filtering", "filter_bindingsite=true"]]
    assert result.is_valid


def test_config_validation_preserves_valid_tokens_regardless_of_order() -> None:
    agent = ConfigValidationAgent()
    first_invalid = agent.validate_for_submission(
        {"config_overrides": [["filtering", "malformed", "keep=1"]]},
        target_model="boltzgen",
    )
    first_valid = agent.validate_for_submission(
        {"config_overrides": [["filtering", "keep=1", "malformed"]]},
        target_model="boltzgen",
    )
    expected = [["filtering", "keep=1"]]
    assert first_invalid.corrected_config["config_overrides"] == expected
    assert first_valid.corrected_config["config_overrides"] == expected
    assert first_invalid.is_valid and first_valid.is_valid


def test_config_validation_preserves_multiple_groups_and_valid_siblings() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {
            "config_overrides": [
                ["filtering", "filter_rmsd_threshold=5.0", "keep_filter=true"],
                ["analysis", "score=iptm", "broken"],
            ]
        },
        target_model="boltzgen",
    )
    assert result.corrected_config["config_overrides"] == [
        ["filtering", "keep_filter=true"],
        ["analysis", "score=iptm"],
    ]
    assert result.is_valid


def test_config_validation_reports_when_all_overrides_removed() -> None:
    agent = ConfigValidationAgent()
    result = agent.validate_for_submission(
        {"config_overrides": [["not_a_step", "broken"], ["filtering", "filter_rmsd_threshold=5.0"]]},
        target_model="boltzgen",
    )
    assert result.corrected_config["config_overrides"] == []
    assert result.is_valid
    assert any("all nonempty config_overrides were removed" in issue.get("problem", "").lower() for issue in result.issues)


def test_retry_intervention_removes_unexpected_kwarg_override() -> None:
    # Reproduces the v15 r2/r3 Taiji crash log so a single post-failure repair
    # removes the offending override deterministically.
    agent = ConfigValidationAgent()
    result = agent.improve_after_failure(
        {"config_overrides": [["filtering", "filter_rmsd_threshold=5.0"]]},
        target_model="boltzgen",
        error_context={
            "boltzgen_log_tail": (
                "hydra.errors.InstantiationException: Error in call to target "
                "'boltzgen.task.filter.filter.Filter':\n"
                "TypeError(\"Filter.__init__() got an unexpected keyword argument 'filter_rmsd_threshold'\")"
            ),
        },
    )
    assert result.corrected_config["config_overrides"] == []
    assert result.raw["runtime_error_analysis"]["config_related"] is True


def test_taiji_presubmit_validation_failure_is_not_resubmitted() -> None:
    # Core v15 fix: a non-retryable pre-submit validation failure must terminate
    # the job, not spawn a wasted try2 via the old error-string re-promotion.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=3)
        calls: List[int] = []

        def taiji_executor(job: DesignJob, attempt: int):
            calls.append(attempt)
            return {
                "job_id": job.job_id,
                "backend": "taiji",
                "status": "failed",
                "error": "pre-submit config validation failed",
                "retryable": False,
            }

        records = orchestrator._run_jobs(
            [_job(out_dir)],
            0,
            taiji_executor,
            attempts_path=out_dir / "round_00" / "execution_attempts.json",
        )
        assert calls == [1], calls
        assert records[0]["status"] == "failed"
        assert records[0]["retryable"] is False


def test_taiji_config_error_resubmits_only_when_repair_changed_params() -> None:
    # A boltzgen_config_error is non-retryable by itself, but becomes retryable
    # when the executor produced changed corrected params for the next attempt.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=3)
        calls: List[int] = []

        def taiji_executor(job: DesignJob, attempt: int):
            calls.append(attempt)
            if attempt == 1:
                return {
                    "job_id": job.job_id,
                    "backend": "taiji",
                    "status": "failed",
                    "error": "boltzgen_config_error;missing_expected_outputs:final_ranked_designs",
                    "retryable": True,
                    "corrected_params_for_retry": {"config_overrides": []},
                }
            return {"job_id": job.job_id, "backend": "taiji", "status": "completed", "output_dir": job.output_dir}

        records = orchestrator._run_jobs(
            [_job(out_dir)],
            0,
            taiji_executor,
            attempts_path=out_dir / "round_00" / "execution_attempts.json",
        )
        assert calls == [1, 2], calls
        assert records[0]["status"] == "completed"



def test_image_filename_incompatibility_is_deterministic_and_non_configurable() -> None:
    traceback = (
        "File /opt/boltzgen/task/predict/data_from_generated.py, line 713\n"
        "target_id = re.search(self.cfg.target_id_regex, p.stem).group(1)\n"
        "AttributeError: 'NoneType' object has no attribute 'group'"
    )
    record = {"status": "failed", "error": "boltzgen_config_error"}
    assert _classify_taiji_failure(record, {"boltzgen_log_tail": traceback}) == "boltzgen_image_filename_contract_incompatible"
    assert _classify_taiji_failure(record, {"boltzgen_log_tail": "AttributeError elsewhere"}) is None
    # The architecture classifies the immutable image contract; it must never
    # propose injecting the image-internal target_id_regex into job parameters.
    proposal_text = json.dumps(record)
    assert "target_id_regex" not in proposal_text

def test_monitor_success_trusts_exit_code_with_optional_output_missing() -> None:
    # exit code 0 + only an optional artifact missing -> success (no false retry).
    assert RunMonitorAgent._is_success("end", 0, ["analysis_metrics_candidates"]) is True
    assert RunMonitorAgent._is_success("completed", 0, ["candidate_manifest"]) is True
    # Filtering ran: missing final_ranked_designs is incomplete even with exit 0.
    assert RunMonitorAgent._is_success("end", 0, ["final_ranked_designs"]) is False
    # A required output missing is still a failure even with exit code 0.
    assert RunMonitorAgent._is_success("end", 0, ["steps_manifest"]) is False
    # Without an explicit success exit code, a missing output stays conservative.
    assert RunMonitorAgent._is_success("end", None, ["analysis_metrics_candidates"]) is False
    # Non-terminal-success states are never success.
    assert RunMonitorAgent._is_success("failed", 0, []) is False


def test_monitor_config_error_hint_not_triggered_by_yaml_path() -> None:
    # A traceback that only references a *.yaml config path must NOT be labeled a
    # boltzgen_config_error (which would wrongly mark a retryable failure as not).
    benign = "loading outputs/boltzgen_output/gpu_0/config/filtering.yaml ... evicted by scheduler"
    hints = RunMonitorAgent._failure_hints(benign, [])
    assert "boltzgen_config_error" not in hints
    # A real Hydra unexpected-keyword crash is still detected.
    real = "TypeError: Filter.__init__() got an unexpected keyword argument 'filter_rmsd_threshold'"
    assert "boltzgen_config_error" in RunMonitorAgent._failure_hints(real, [])



def test_multiple_logical_jobs_are_safe_before_budget():
    with tempfile.TemporaryDirectory() as tmp:
        jobs = []
        for index in range(3):
            job = _job(Path(tmp)); job.job_id = "same"; job.output_dir = str(Path(tmp) / f"job_{index}")
            job.params.update({
                "exploration_arm": f"arm_{index}", "arm_rank": index,
                "logical_branch_id": f"branch_{index}", "alpha": 0.001 + index * .001,
            })
            jobs.append(job)
        from binderloop.strategy_governance import materialize_deterministic_job_identities
        materialized = materialize_deterministic_job_identities(jobs, round_id=2, output_root=str(tmp))
        assert len(materialized) == 3
        assert len({job.job_id for job in materialized}) == 3
        assert len({job.output_dir for job in materialized}) == 3


def test_identity_finalization_before_budget_preserves_host_shards():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg()
        cfg.resource.backend = "taiji"
        cfg.resource.host_num = 2
        cfg.resource.taiji_multi_host_mode = "split_jobs"
        cfg.search_space.max_binders_per_round = 8
        cfg.search_space.num_designs_per_round = 8
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=Path(tmp), max_retries=1)
        logical = orchestrator._finalize_semantic_job_identities([_job(Path(tmp))], round_id=1)
        shards = orchestrator._enforce_round_cap(logical)
        assert len(shards) == 2
        assert [job.params["num_designs"] for job in shards] == [4, 4]
        assert len({job.job_id for job in shards}) == 2


def test_duplicate_output_identity_is_non_retryable():
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=3)
        a = _job(Path(tmp)); b = _job(Path(tmp)); a.job_id = "a"; b.job_id = "b"
        a.params["branch_id"] = "a"; b.params["branch_id"] = "b"; b.output_dir = a.output_dir
        try:
            orchestrator._run_jobs([a, b], 0, lambda job, attempt: {})
        except ValueError as exc:
            assert "duplicate_output_dir" in str(exc)
            assert not orchestrator._identity_or_budget_error_retryable(exc)
        else:
            raise AssertionError("duplicate output directory was accepted")

def test_attempt_ledger_rejects_same_id_identity_change_but_accepts_refinalized_id() -> None:
    # Reproduces sc2rbd r3_control_... attempt_ledger_identity_mismatch.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
        job = orchestrator._finalize_semantic_job_identities([_job(out_dir)], round_id=3)[0]
        job = orchestrator._enforce_round_cap([job], round_id=3)[0]
        ledger = {"round_id": 3, "jobs": {}}
        entry = orchestrator._job_attempt_entry(ledger, job)
        mutated = DesignJob(**{**job.__dict__, "params": {**job.params, "alpha": 0.003}})
        try:
            orchestrator._job_attempt_entry(ledger, mutated)
        except ValueError as exc:
            assert str(exc) == f"attempt_ledger_identity_mismatch:{job.job_id}"
        else:
            raise AssertionError("same job ID silently changed identity")
        refinalized = orchestrator._enforce_round_cap(
            orchestrator._finalize_semantic_job_identities([mutated], round_id=3), round_id=3,
        )[0]
        assert refinalized.job_id != job.job_id
        assert orchestrator._job_attempt_entry(ledger, refinalized) is not entry
        assert set(ledger["jobs"]) == {job.job_id, refinalized.job_id}



def test_partial_arm_failure_continuation_is_execution_finalized() -> None:
    """A failed arm and successful sibling must both be safe to submit next round."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        cfg = _cfg()
        cfg.active_learning.branch_width = 2
        orchestrator = BinderDesignOrchestrator(cfg, out_dir=out_dir, max_parallel=1, max_retries=1)
        logical = orchestrator._finalize_semantic_job_identities(
            [_job(out_dir), DesignJob(**{**_job(out_dir).__dict__, "job_id": "probe", "params": {**_job(out_dir).params, "arm_id": "probe", "exploration_arm": "probe", "branch_id": "r0_probe", "logical_branch_id": "r0_probe", "alpha": 0.003}})],
            round_id=0,
        )
        current = orchestrator._enforce_round_cap(logical, round_id=0)
        records = [
            {"job_id": current[0].job_id, "job": current[0].__dict__, "status": "completed", "output_dir": current[0].output_dir},
            {"job_id": current[1].job_id, "job": current[1].__dict__, "status": "failed", "error": "missing_expected_outputs:final_ranked_designs", "retryable": False},
        ]

        continuations = orchestrator._retry_jobs_after_execution_failure(current, records, next_round_id=1)
        finalized = orchestrator._enforce_round_cap(
            orchestrator._enforce_binder_length_range(continuations), round_id=1,
        )

        assert {job.params["continuation_kind"] for job in finalized} == {"successful_arm_retest", "failed_arm_retry"}
        assert all(job.params["job_identity"]["finalized"] is True for job in finalized)
        assert all(job.params["job_identity"]["execution_slot"] is not None for job in finalized)
        assert len({job.job_id for job in finalized}) == 2
        assert len({job.output_dir for job in finalized}) == 2
        assert [job.params["num_designs"] for job in finalized] == [job.params["immutable_branch_plan"]["allocated_designs"] for job in finalized]
        ledger = {"round_id": 1, "jobs": {}}
        for job in finalized:
            orchestrator._job_attempt_entry(ledger, job)
        assert set(ledger["jobs"]) == {job.job_id for job in finalized}


def test_attempt_ledger_identity_mismatch_is_not_module_retryable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_retries=3)
        assert not orchestrator._identity_or_budget_error_retryable(
            ValueError("attempt_ledger_identity_mismatch:r4_arm00")
        )
        assert not orchestrator._identity_or_budget_error_retryable(
            ValueError("execution_job_identity_not_finalized:r4_arm00")
        )
        assert not orchestrator._identity_or_budget_error_retryable(
            ValueError("next_jobs_execution_identity_not_finalized:r5_arm00")
        )


def test_exact_replay_bind_makes_jobs_executable() -> None:
    """Best-config retest jobs stay logical until execution identity is bound."""
    from binderloop.orchestration.orchestrator import ModuleOutputValidationError

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=out_dir, max_parallel=1, max_retries=1)
        logical = orchestrator._finalize_semantic_job_identities(
            [
                DesignJob(**{**_job(out_dir).__dict__, "job_id": "hold", "params": {
                    "arm_id": "baseline_hold", "exploration_arm": "baseline_hold", "num_designs": 1,
                }}),
                DesignJob(**{**_job(out_dir).__dict__, "job_id": "probe", "params": {
                    "arm_id": "probe", "exploration_arm": "probe", "branch_id": "r2_probe",
                    "logical_branch_id": "r2_probe", "alpha": 0.003, "num_designs": 1,
                }}),
            ],
            round_id=2,
        )
        assert all(job.params["job_identity"]["finalized"] is False for job in logical)
        try:
            orchestrator._run_jobs(
                logical, 5,
                lambda job, attempt: {"job_id": job.job_id, "status": "completed", "output_dir": job.output_dir},
            )
        except ValueError as exc:
            assert str(exc).startswith("execution_job_identity_not_finalized:")
        else:
            raise AssertionError("logical replay jobs reached the executor")

        path = out_dir / "next_jobs.json"
        path.write_text(json.dumps([asdict(job) for job in logical], indent=2) + "\n", encoding="utf-8")
        try:
            orchestrator._validate_next_jobs_module({"next_jobs": logical, "path": str(path)}, expect_jobs=True)
        except ModuleOutputValidationError as exc:
            assert str(exc).startswith("next_jobs_execution_identity_not_finalized:")
        else:
            raise AssertionError("unfinalized next_jobs passed validation")

        executable = orchestrator._bind_execution_identities_if_needed(logical, round_id=5)
        assert all(job.params["job_identity"]["finalized"] is True for job in executable)
        assert all(job.params["job_identity"]["execution_slot"] is not None for job in executable)
        assert all("/jobs/" in str(job.output_dir).replace("\\", "/") for job in executable)
        assert len({job.job_id for job in executable}) == 2
        assert orchestrator._bind_execution_identities_if_needed(executable, round_id=5) == executable
        path.write_text(json.dumps([asdict(job) for job in executable], indent=2) + "\n", encoding="utf-8")
        orchestrator._validate_next_jobs_module({"next_jobs": executable, "path": str(path)}, expect_jobs=True)
        records = orchestrator._run_jobs(
            executable, 5,
            lambda job, attempt: {"job_id": job.job_id, "status": "completed", "output_dir": job.output_dir},
        )
        assert [record["status"] for record in records] == ["completed", "completed"]


def test_run_jobs_rejects_unfinalized_logical_identity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = BinderDesignOrchestrator(_cfg(), out_dir=Path(tmp), max_parallel=1, max_retries=1)
        logical = orchestrator._finalize_semantic_job_identities([_job(Path(tmp))], round_id=0)
        try:
            orchestrator._run_jobs(logical, 0, lambda job, attempt: {"job_id": job.job_id, "status": "completed"})
        except ValueError as exc:
            assert str(exc).startswith("execution_job_identity_not_finalized:")
        else:
            raise AssertionError("unfinalized logical identity reached the executor")

def main() -> int:
    import pytest
    return int(pytest.main([str(Path(__file__).resolve()), "-q"]))

if __name__ == "__main__":
    raise SystemExit(main())
