#!/usr/bin/env python3
"""Reconstruct binder-harness prompts and probe an OpenAI-compatible limit.

The live mode deliberately bypasses ``chat_json``'s byte guard so it can find
the provider boundary. It never writes credentials or full long completions.
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.binder_quality_analysis_agent import BinderQualityAnalysisAgent
from binderloop.agents.config_validation_agent import ConfigValidationAgent
from binderloop.agents.context_compaction import (
    build_metric_facts,
    compact_context_for_config_validation,
    compact_context_for_diagnostic,
    compact_context_for_hypothesis,
    compact_context_for_input_config,
    compact_context_for_quality,
    compact_context_for_target_config,
)
from binderloop.agents.diagnostic_coach_agent import DiagnosticCoachAgent
from binderloop.agents.hypothesis_agent import HypothesisAgent
from binderloop.agents.input_configuration_agent import InputConfigurationAgent
from binderloop.agents.model_input_spec import get_model_input_spec
from binderloop.llm import (
    LLMConfigError,
    LLMHTTPError,
    LLMTransportError,
    OpenAICompatibleClient,
)


DEFAULT_ROUND_DIR = (
    "outputs/sc2rbd_closed_loop_llm_np_160s_8r_v17_bug/round_00"
)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: Any, *, indent: Optional[int] = None) -> int:
    return len(json.dumps(value, ensure_ascii=False, indent=indent).encode("utf-8"))


def _wire_messages(system: str, user: Mapping[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def _agent_skills(skills: Mapping[str, Any], name: str) -> List[Dict[str, Any]]:
    return list((skills.get("activations_by_agent") or {}).get(name) or [])


def _memory_summary(round_dir: Path, target: Mapping[str, Any]) -> Dict[str, Any]:
    memory = _read_json(round_dir.parent / "memory" / "experiment_memory.json", {}) or {}
    rounds = list(memory.get("rounds") or [])
    return {
        "experiment_id": memory.get("experiment_id", "reconstructed_round_00"),
        "target": memory.get("target") or dict(target),
        "round_count": len(rounds),
        "extend_memory": True,
        "recent_rounds": rounds[-5:],
        "recent_messages": list(memory.get("messages") or [])[-50:],
    }


def build_prompt_corpus(round_dir: Path) -> List[Dict[str, Any]]:
    """Build the eight prompt kinds from one round's durable artifacts."""
    evaluation = dict(_read_json(round_dir / "evaluation_summary.json", {}) or {})
    structural = dict(_read_json(round_dir / "structure_evaluation.json", {}) or {})
    active_examples = dict(_read_json(round_dir / "active_learning_examples.json", {}) or {})
    checkpoint = dict(_read_json(round_dir / "round_checkpoint.json", {}) or {})
    rollback = dict(_read_json(round_dir / "rollback_decision.json", {}) or {})
    skills = dict(_read_json(round_dir / "active_skills.json", {}) or {})
    execution_records = list(_read_json(round_dir / "execution_records.json", []) or [])
    jobs = list(checkpoint.get("current_jobs") or [])
    if not jobs and execution_records:
        jobs = [dict(item.get("job") or {}) for item in execution_records if item.get("job")]
    job = dict(jobs[0] if jobs else {})
    current_config = dict(job.get("params") or {})
    current_config.update({
        "task_name": current_config.get("task_id") or "sc2rbd",
        "target": {
            "structure_path": job.get("target_structure"),
            "chain_id": job.get("chain_id"),
            "hotspots": job.get("hotspots") or [],
        },
        "binder_lengths": current_config.get("binder_lengths")
        or current_config.get("binder_length_range")
        or ([job.get("binder_length")] if job.get("binder_length") else []),
    })
    target = dict(current_config.get("target") or {})
    evaluation["metric_facts"] = evaluation.get("metric_facts") or build_metric_facts(evaluation)
    evaluation["active_learning_examples"] = active_examples
    memory = _memory_summary(round_dir, target)
    round_id = int(checkpoint.get("round_id") or 0)
    constraints = {
        "max_binders_per_round": current_config.get("max_binders_per_round"),
        "binder_length_range": current_config.get("binder_length_range"),
        "epitope_crop_disabled_hard_constraint": current_config.get("epitope_crop_mode") == "disabled",
    }
    context = {
        "round_id": round_id,
        "evaluation": evaluation,
        "metric_facts": evaluation["metric_facts"],
        "active_learning_examples": active_examples,
        "structural_analysis": structural,
        "memory": memory,
        "target_analysis": target,
        "current_config": current_config,
        "constraints": constraints,
        "execution_failure": {"failed": False, "reason": ""},
        "messages": [],
        "rollback": rollback.get("decision") or {},
        "reward": rollback.get("outcome") or {},
    }

    quality_user = {"round_id": round_id, "context": compact_context_for_quality(context)}
    hypothesis_user = {"context": compact_context_for_hypothesis(context)}
    quality = BinderQualityAnalysisAgent().analyze(round_id=round_id, context=context)
    hypotheses = HypothesisAgent().propose(context)

    monitor = {
        "state": "completed",
        "is_terminal": True,
        "is_success": True,
        "status_counts": {"completed": len(execution_records)},
        "failed_jobs": [],
    }
    diagnostic_context = compact_context_for_diagnostic(
        round_id=round_id,
        monitor_snapshot=monitor,
        metrics_summary=evaluation["metric_facts"],
        evaluation_summary=evaluation,
        structural_analysis=structural,
        job_history=memory.get("recent_rounds"),
        config=current_config,
        active_skills=_agent_skills(skills, "DiagnosticCoachAgent"),
    )
    diagnostic_user = {"round_id": round_id, "pipeline_state": diagnostic_context}
    diagnostic = DiagnosticCoachAgent().diagnose(
        round_id=round_id,
        monitor_snapshot=monitor,
        metrics_summary=evaluation["metric_facts"],
        evaluation_summary=evaluation,
        structural_analysis=structural,
        job_history=memory.get("recent_rounds"),
        config=current_config,
        active_skills=_agent_skills(skills, "DiagnosticCoachAgent"),
    )
    input_user = compact_context_for_input_config(
        target_name=str(current_config.get("task_name") or "sc2rbd"),
        current_config=current_config,
        diagnostic_report=asdict(diagnostic),
        evaluation_summary=evaluation,
        round_id=round_id + 1,
        target_profile=target,
        structural_analysis=structural,
        quality_analysis=asdict(quality),
        hypotheses=hypotheses.hypotheses,
        memory_summary=memory,
        constraints=constraints,
        tuning_feedback={},
        active_skills=_agent_skills(skills, "InputConfigurationAgent"),
    )
    initial_user = compact_context_for_target_config({
        "target_name": current_config.get("task_name") or "sc2rbd",
        "target_info": target,
        "target_profile": target,
        "previous_results": evaluation,
        "constraints": constraints,
    })

    validator = ConfigValidationAgent()
    spec = get_model_input_spec("boltzgen")
    pre = validator.validate_full_job_config(current_config, target_model="boltzgen")
    delta_config = {
        key: current_config[key]
        for key in ("hotspot_weight", "diffusion_batch_size", "alpha", "filter_biased")
        if key in current_config
    }
    delta = validator.validate_agent_delta(delta_config, target_model="boltzgen")
    failure_context = {
        "error_type": "runner_error",
        "message": str((checkpoint.get("error") or {}).get("message") or "representative runner failure"),
        "exit_code": 1,
        "step": "taiji_submission",
    }
    failed = validator.improve_after_failure(
        current_config,
        error_context=failure_context,
        target_model="boltzgen",
    )
    validation_system = validator._system_prompt(spec)
    corpus = [
        ("quality.round_analysis", BinderQualityAnalysisAgent.SYSTEM, quality_user, "reconstructed_from_round_artifacts"),
        ("hypothesis.round_proposal", HypothesisAgent.SYSTEM, hypothesis_user, "reconstructed_from_round_artifacts"),
        ("diagnostic.round_coach", DiagnosticCoachAgent.SYSTEM, diagnostic_user, "reconstructed_with_deterministic_dependencies"),
        ("input_config.next_round", InputConfigurationAgent.SYSTEM, input_user, "reconstructed_with_deterministic_dependencies"),
        (
            "config_validation.pre_submit",
            validation_system,
            compact_context_for_config_validation(
                target_model="boltzgen",
                activation="pre_submit",
                config=current_config,
                deterministic_prefilter=asdict(pre),
            ),
            "reconstructed_from_recorded_job_config",
        ),
        (
            "config_validation.agent_delta",
            validation_system,
            compact_context_for_config_validation(
                target_model="boltzgen",
                activation="agent_delta",
                config=delta_config,
                deterministic_prefilter=asdict(delta),
            ),
            "synthetic_activation_using_round_config",
        ),
        (
            "config_validation.taiji_failure",
            validation_system,
            compact_context_for_config_validation(
                target_model="boltzgen",
                activation="taiji_failure",
                config=current_config,
                deterministic_prefilter=asdict(failed),
                context={"error_context": failure_context},
            ),
            "synthetic_activation_using_round_error",
        ),
        ("input_config.initial_target", InputConfigurationAgent.SYSTEM, initial_user, "reconstructed_from_round_target"),
    ]
    result: List[Dict[str, Any]] = []
    for kind, system, user, provenance in corpus:
        messages = _wire_messages(system, user)
        result.append({
            "kind": kind,
            "system": system,
            "user": user,
            "provenance": provenance,
            "system_bytes": len(system.encode("utf-8")),
            "user_wire_bytes": _json_bytes(user, indent=2),
            "request_message_bytes": _json_bytes(messages),
            "digest": hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
        })
    return result


def _response_summary(data: Mapping[str, Any], elapsed: float) -> Dict[str, Any]:
    choices = list(data.get("choices") or [])
    choice = dict(choices[0] if choices else {})
    message = dict(choice.get("message") or {})
    content = str(message.get("content") or "")
    usage = dict(data.get("usage") or {})
    return {
        "ok": True,
        "elapsed_seconds": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "content_bytes": len(content.encode("utf-8")),
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_preview_start": content[:160],
        "content_preview_end": content[-160:] if content else "",
        "model": data.get("model"),
        "provider_id": data.get("id"),
    }


def _call(
    client: OpenAICompatibleClient,
    *,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    started = time.time()
    try:
        data = client.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            max_retries=1,
        )
        return _response_summary(data, time.time() - started)
    except LLMHTTPError as exc:
        detail = exc.detail
        lowered = detail.lower()
        return {
            "ok": False,
            "error_class": type(exc).__name__,
            "status_code": exc.status_code,
            "is_context_limit": any(
                term in lowered for term in ("context", "token", "maximum", "max_tokens", "length")
            ),
            "detail": detail[-1000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except (LLMTransportError, LLMConfigError) as exc:
        return {
            "ok": False,
            "error_class": type(exc).__name__,
            "is_context_limit": False,
            "detail": str(exc)[-1000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_class": type(exc).__name__,
            "is_context_limit": False,
            "detail": str(exc)[-1000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }


def _attempt_record(
    *,
    phase: str,
    kind: str,
    messages: Sequence[Mapping[str, Any]],
    max_tokens: int,
    result: Mapping[str, Any],
    units: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "phase": phase,
        "kind": kind,
        "request_message_bytes": _json_bytes(messages),
        "request_chars": sum(len(str(message.get("content") or "")) for message in messages),
        "max_tokens": max_tokens,
        "padding_units": units,
        **dict(result),
    }


def probe_live(
    client: OpenAICompatibleClient,
    corpus: Sequence[Mapping[str, Any]],
    *,
    max_probe_bytes: int,
    max_output_probe_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    attempts: List[Dict[str, Any]] = []
    baseline: Dict[str, Any] = {}
    for item in corpus:
        messages = _wire_messages(str(item["system"]), dict(item["user"]))
        result = _call(client, messages=messages, max_tokens=32)
        record = _attempt_record(
            phase="representative_prompt",
            kind=str(item["kind"]),
            messages=messages,
            max_tokens=32,
            result=result,
        )
        attempts.append(record)
        baseline[str(item["kind"])] = record

    successful = [item for item in corpus if baseline[str(item["kind"])].get("ok")]
    boundary: Dict[str, Any] = {"status": "not_run"}
    if successful:
        base = max(successful, key=lambda item: int(item["request_message_bytes"]))
        unit = json.dumps(base["user"], ensure_ascii=False, separators=(",", ":"))

        def messages_for(units: int) -> List[Dict[str, str]]:
            return [
                {"role": "system", "content": str(base["system"])},
                {"role": "user", "content": json.dumps(base["user"], ensure_ascii=False, indent=2)},
                {
                    "role": "user",
                    "content": (
                        "Additional archived harness evidence follows. Acknowledge with JSON only.\n"
                        + "\n".join(unit for _ in range(max(0, units - 1)))
                    ),
                },
            ]

        last_success: Optional[Dict[str, Any]] = None
        first_reject: Optional[Dict[str, Any]] = None
        units = 1
        while True:
            messages = messages_for(units)
            if _json_bytes(messages) > max_probe_bytes:
                break
            result = _call(client, messages=messages, max_tokens=16)
            record = _attempt_record(
                phase="input_exponential",
                kind=str(base["kind"]),
                messages=messages,
                max_tokens=16,
                result=result,
                units=units,
            )
            attempts.append(record)
            if result.get("ok"):
                last_success = record
                units *= 2
                continue
            if result.get("is_context_limit"):
                first_reject = record
            break
        if last_success and first_reject:
            lo = int(last_success["padding_units"])
            hi = int(first_reject["padding_units"])
            while hi - lo > 1:
                mid = (lo + hi) // 2
                messages = messages_for(mid)
                result = _call(client, messages=messages, max_tokens=16)
                record = _attempt_record(
                    phase="input_binary",
                    kind=str(base["kind"]),
                    messages=messages,
                    max_tokens=16,
                    result=result,
                    units=mid,
                )
                attempts.append(record)
                if result.get("ok"):
                    lo = mid
                    last_success = record
                elif result.get("is_context_limit"):
                    hi = mid
                    first_reject = record
                else:
                    break
            boundary = {
                "status": "bounded",
                "base_kind": base["kind"],
                "last_success": last_success,
                "first_context_reject": first_reject,
            }
        elif last_success:
            boundary = {
                "status": "lower_bound_only",
                "base_kind": base["kind"],
                "last_success": last_success,
                "max_probe_bytes": max_probe_bytes,
            }
        else:
            boundary = {"status": "unavailable", "reason": "no successful expanded request"}

    output: Dict[str, Any] = {"status": "not_run"}
    request_tokens = 4096
    last_output_success: Optional[Dict[str, Any]] = None
    while request_tokens <= max_output_probe_tokens:
        messages = [
            {
                "role": "system",
                "content": "You are an output-limit probe. Follow the user literally and do not summarize.",
            },
            {
                "role": "user",
                "content": (
                    "Output the token x followed by one space repeatedly. Continue until the server "
                    "stops generation. Do not explain, conclude, or use punctuation."
                ),
            },
        ]
        result = _call(client, messages=messages, max_tokens=request_tokens)
        record = _attempt_record(
            phase="output_full",
            kind="output_limit_stream",
            messages=messages,
            max_tokens=request_tokens,
            result=result,
        )
        attempts.append(record)
        if result.get("ok"):
            last_output_success = record
            output = {"status": "lower_bound", "last_success": record}
            if result.get("finish_reason") != "length":
                output = {
                    "status": "model_stopped_before_limit",
                    "last_success": record,
                    "note": "The model did not sustain generation to the requested cap.",
                }
                break
            request_tokens *= 2
            continue
        output = {
            "status": "bounded" if result.get("is_context_limit") else "interrupted",
            "last_success": last_output_success,
            "first_reject": record,
        }
        break
    return attempts, {"representative": baseline, "input_boundary": boundary, "output_boundary": output}


def compute_safe_budget(
    corpus: Sequence[Mapping[str, Any]],
    live_summary: Mapping[str, Any],
    *,
    safety_factor: float = 0.85,
    default_prompt_bytes: int = 750_000,
    output_reserve_tokens: int = 8192,
) -> Dict[str, Any]:
    """Derive production byte/token budgets from probe measurements."""
    max_representative_bytes = max(
        int(item.get("request_message_bytes") or 0) for item in corpus
    ) if corpus else 0
    max_user_bytes = max(int(item.get("user_wire_bytes") or 0) for item in corpus) if corpus else 0
    max_system_bytes = max(int(item.get("system_bytes") or 0) for item in corpus) if corpus else 0

    representative = dict(live_summary.get("representative") or {})
    prompt_tokens = [
        int((record.get("usage") or {}).get("prompt_tokens") or 0)
        for record in representative.values()
        if record.get("ok")
    ]
    baseline_prompt_tokens = max(prompt_tokens) if prompt_tokens else None

    input_boundary = dict(live_summary.get("input_boundary") or {})
    last_success = dict(input_boundary.get("last_success") or {})
    upper_request_bytes = int(last_success.get("request_message_bytes") or 0)
    if upper_request_bytes <= 0:
        upper_request_bytes = max_representative_bytes

    recommended_prompt_bytes = int(min(
        default_prompt_bytes,
        max(32_000, upper_request_bytes * safety_factor),
    ))
    if not last_success and max_representative_bytes:
        recommended_prompt_bytes = int(min(
            default_prompt_bytes,
            max(32_000, max_representative_bytes * 1.5),
        ))

    output_boundary = dict(live_summary.get("output_boundary") or {})
    output_success = dict(output_boundary.get("last_success") or {})
    completion_tokens = int((output_success.get("usage") or {}).get("completion_tokens") or 0)
    total_tokens = int((output_success.get("usage") or {}).get("total_tokens") or 0)
    reasoning_tokens = None
    usage = dict(output_success.get("usage") or {})
    for key in ("reasoning_tokens", "completion_tokens_details"):
        if isinstance(usage.get(key), dict):
            reasoning_tokens = usage[key].get("reasoning_tokens")
        elif usage.get(key) is not None:
            reasoning_tokens = usage.get(key)

    context_window_tokens = None
    if last_success:
        success_tokens = int((last_success.get("usage") or {}).get("prompt_tokens") or 0)
        if success_tokens:
            context_window_tokens = success_tokens + int(output_reserve_tokens)
    if baseline_prompt_tokens and total_tokens and not context_window_tokens:
        context_window_tokens = baseline_prompt_tokens + completion_tokens + int(output_reserve_tokens)

    return {
        "max_representative_request_bytes": max_representative_bytes,
        "max_representative_user_bytes": max_user_bytes,
        "max_representative_system_bytes": max_system_bytes,
        "baseline_prompt_tokens": baseline_prompt_tokens,
        "input_boundary_status": input_boundary.get("status"),
        "input_upper_request_bytes": upper_request_bytes or None,
        "recommended_prompt_max_bytes": recommended_prompt_bytes,
        "recommended_max_output_tokens": int(
            max(completion_tokens, output_reserve_tokens) if completion_tokens else output_reserve_tokens
        ),
        "observed_completion_tokens": completion_tokens or None,
        "observed_reasoning_tokens": reasoning_tokens,
        "estimated_context_window_tokens": context_window_tokens,
        "output_boundary_status": output_boundary.get("status"),
        "notes": (
            "recommended_prompt_max_bytes applies to serialized user JSON only; "
            "system prompts and completion reserve are separate."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", default=DEFAULT_ROUND_DIR)
    parser.add_argument("--llm-config", default="configs/llm_endpoints.gpt.json")
    parser.add_argument("--llm-model", default="gpt-5.5")
    parser.add_argument("--out-dir", default="outputs/gpt55_limit_probe_sc2rbd_round00")
    parser.add_argument("--live", action="store_true", help="Perform paid live API probes.")
    parser.add_argument("--max-probe-bytes", type=int, default=8_000_000)
    parser.add_argument("--max-output-probe-tokens", type=int, default=262_144)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    round_dir = Path(args.round_dir)
    if not round_dir.is_absolute():
        round_dir = root / round_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = build_prompt_corpus(round_dir)
    serializable_corpus = [
        {key: value for key, value in item.items() if key not in {"system", "user"}}
        for item in corpus
    ]
    (out_dir / "prompt_corpus.json").write_text(
        json.dumps(serializable_corpus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    existing_summary = _read_json(out_dir / "summary.json", {}) if isinstance(_read_json(out_dir / "summary.json", {}), dict) else {}
    summary: Dict[str, Any] = {
        "round_dir": str(round_dir),
        "prompt_count": len(corpus),
        "live": bool(args.live),
        "corpus": serializable_corpus,
    }
    if args.live:
        client = OpenAICompatibleClient.from_json(root / args.llm_config)
        if client is None:
            raise RuntimeError("LLM client was not created")
        client.configure_default(model_key=args.llm_model)
        if not client.available():
            raise LLMConfigError("Configured gpt-5.5 endpoint is unavailable")
        attempts, live_summary = probe_live(
            client,
            corpus,
            max_probe_bytes=max(1, int(args.max_probe_bytes)),
            max_output_probe_tokens=max(1, int(args.max_output_probe_tokens)),
        )
        with (out_dir / "attempts.jsonl").open("w", encoding="utf-8") as handle:
            for attempt in attempts:
                handle.write(json.dumps(attempt, ensure_ascii=False) + "\n")
        summary.update(live_summary)
        summary["safe_budget"] = compute_safe_budget(corpus, live_summary)
    elif existing_summary.get("live"):
        summary.update({
            key: existing_summary[key]
            for key in ("representative", "input_boundary", "output_boundary", "safe_budget")
            if key in existing_summary
        })
        summary["live"] = True
        summary["safe_budget"] = existing_summary.get("safe_budget") or compute_safe_budget(corpus, existing_summary)
    else:
        summary["safe_budget"] = compute_safe_budget(corpus, {})
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# gpt-5.5 LLM limit probe (round_00 corpus)",
        "",
        f"- Round dir: `{summary['round_dir']}`",
        f"- Prompt kinds measured: {summary['prompt_count']}",
        f"- Live probe: {'yes' if summary.get('live') else 'offline corpus only'}",
        "",
        "## Representative prompt sizes",
        "",
    ]
    for item in summary.get("corpus") or []:
        report_lines.append(
            f"- `{item['kind']}`: request={item['request_message_bytes']:,} bytes "
            f"(user={item['user_wire_bytes']:,}, system={item['system_bytes']:,}), "
            f"provenance={item['provenance']}"
        )
    budget = dict(summary.get("safe_budget") or {})
    prompt_bytes = budget.get("recommended_prompt_max_bytes")
    output_tokens = budget.get("recommended_max_output_tokens")
    report_lines.extend([
        "",
        "## Recommended production budget",
        "",
        f"- `prompt_max_bytes`: {prompt_bytes:,}" if prompt_bytes is not None else "- `prompt_max_bytes`: pending live probe",
        f"- `max_output_tokens`: {output_tokens:,}" if output_tokens is not None else "- `max_output_tokens`: pending live probe",
        f"- Input boundary: {budget.get('input_boundary_status')}",
        f"- Output boundary: {budget.get('output_boundary_status')}",
        f"- Baseline prompt tokens (successful representative): {budget.get('baseline_prompt_tokens')}",
        "",
        budget.get("notes", ""),
    ])
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
