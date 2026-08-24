#!/usr/bin/env python3
"""Compare round_04 quality prompt strategies against an OpenAI-compatible API.

Strategies:
1. faithful_single_high: reconstructed single request, current high reasoning.
2. compact_single_high: deduplicated/capped context, current high reasoning.
3. compact_single_low: same compact context, low reasoning.
4. modular_low: serialized metric + structure-batch analysis followed by synthesis.

The whole run holds an inter-process file lock. Every request is serialized;
only modular stages retry transient CDN/SSL failures once after a long backoff.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.binder_quality_analysis_agent import (
    BinderQualityAnalysisAgent,
)
from binderloop.agents.context_compaction import (
    MAX_STRUCTURE_SUMMARIES,
    compact_active_learning_examples,
    compact_context_for_quality,
    compact_evaluation,
    compact_messages,
    compact_structural_analysis,
    fact_check_text_against_metric_facts,
)
from binderloop.llm import (
    LLMConfigError,
    LLMHTTPError,
    LLMTransportError,
    OpenAICompatibleClient,
)
from scripts.probe_llm_limits import build_prompt_corpus


DEFAULT_ROUND_DIR = "outputs/sc2rbd_closed_loop_llm_np_160s_8r_v18/round_04"

METRIC_SYSTEM = """You are the metric specialist in a binder-quality pipeline.
Return JSON only with keys:
{
  "metric_findings": [{"finding":"...", "evidence":["..."], "confidence":0-1}],
  "failure_modes": [{"name":"...", "evidence":["..."], "severity":"low|medium|high"}],
  "historical_contrasts": [{"comparison":"...", "evidence":["..."]}],
  "recommended_changes": [{"action":"...", "evidence":["..."], "risk":"..."}]
}
Use only supplied evidence. Keep at most 6 metric findings, 5 failure modes,
4 historical contrasts, and 5 recommendations. Distinguish additional-filter
passes, BoltzGen pass_filters, and harness success_count."""

STRUCTURE_SYSTEM = """You are the structural-module specialist in a
binder-quality pipeline. Return JSON only with keys:
{
  "high_quality_modules": [{"module_id":"...", "evidence":["..."], "confidence":0-1}],
  "low_quality_modules": [{"module_id":"...", "evidence":["..."], "confidence":0-1}],
  "structure_findings": [{"finding":"...", "evidence":["..."], "confidence":0-1}]
}
Use only supplied structures. Keep at most 6 items in each list. Distinguish
local fragment quality from whole-candidate binding quality. Do not infer a
chain mismatch merely because output chain labels differ from input labels."""

SYNTHESIS_SUFFIX = """
This is a synthesis step. The supplied metric and structure specialist outputs
are evidence summaries, not new measurements. Return at most 6 high-quality
modules, 6 low-quality modules, 6 causal factors, and 6 next-round actions.
Do not repeat the same module or recommendation."""


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def _messages(system: str, user: Mapping[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def _classify_error(detail: str) -> str:
    lowered = str(detail or "").lower()
    if "context window" in lowered or "input exceeds" in lowered:
        return "context_limit"
    if "concurr" in lowered or "并发" in lowered or "too many requests" in lowered:
        return "concurrency_limit"
    if "cdn" in lowered or "防火墙" in lowered or "源服务器" in lowered:
        return "cdn_origin"
    if "ssl" in lowered:
        return "ssl_transport"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    return "other_transport"


def _extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    try:
        value = json.loads(stripped)
        return (dict(value), None) if isinstance(value, dict) else (None, "not_object")
    except Exception as exc:
        return None, str(exc)


def _configured_client(
    config_path: Path,
    model_key: str,
    *,
    reasoning: str,
) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient.from_json(config_path)
    if client is None:
        raise RuntimeError("LLM client was not created")
    client.configure_default(model_key=model_key)
    endpoint = client.settings.endpoints[client.settings.default_model]
    endpoint.thinking = reasoning
    if reasoning == "low":
        # The existing profile redundantly enables both reasoning_effort=high
        # and a generic reasoning object. Use one low-effort control here so the
        # experiment measures processing-time reduction rather than prompt only.
        endpoint.extra_body = {
            key: value
            for key, value in endpoint.extra_body.items()
            if key != "reasoning"
        }
    return client


def _call_once(
    client: OpenAICompatibleClient,
    *,
    strategy: str,
    stage: str,
    system: str,
    user: Mapping[str, Any],
    max_tokens: int,
    attempt: int,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    messages = _messages(system, user)
    started = time.time()
    base = {
        "strategy": strategy,
        "stage": stage,
        "attempt": attempt,
        "request_message_bytes": _json_bytes(messages),
        "system_bytes": len(system.encode("utf-8")),
        "user_bytes": _json_bytes(user),
        "max_tokens": max_tokens,
    }
    try:
        data = client.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
            max_retries=1,
        )
        choice = dict((data.get("choices") or [{}])[0] or {})
        message = dict(choice.get("message") or {})
        content = str(message.get("content") or "")
        parsed, parse_error = _extract_json(content)
        record = {
            **base,
            "ok": True,
            "elapsed_seconds": round(time.time() - started, 3),
            "finish_reason": choice.get("finish_reason"),
            "usage": dict(data.get("usage") or {}),
            "content_bytes": len(content.encode("utf-8")),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "json_parsed": parsed is not None,
            "parse_error": parse_error,
        }
        return record, parsed
    except LLMHTTPError as exc:
        detail = exc.detail
        return {
            **base,
            "ok": False,
            "status_code": exc.status_code,
            "error_type": _classify_error(detail),
            "detail": detail[-1000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }, None
    except (LLMTransportError, LLMConfigError) as exc:
        detail = str(exc)
        return {
            **base,
            "ok": False,
            "error_type": _classify_error(detail),
            "detail": detail[-1000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }, None
    except Exception as exc:
        detail = str(exc)
        return {
            **base,
            "ok": False,
            "error_type": _classify_error(detail),
            "detail": detail[-1000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }, None


def _call_controlled(
    client: OpenAICompatibleClient,
    *,
    strategy: str,
    stage: str,
    system: str,
    user: Mapping[str, Any],
    max_tokens: int,
    attempts: List[Dict[str, Any]],
    max_attempts: int,
    backoff_seconds: float,
) -> Optional[Dict[str, Any]]:
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        record, parsed = _call_once(
            client,
            strategy=strategy,
            stage=stage,
            system=system,
            user=user,
            max_tokens=max_tokens,
            attempt=attempt,
        )
        attempts.append(record)
        if record.get("ok") and parsed is not None:
            return parsed
        terminal = record.get("error_type") in {
            "context_limit",
            "concurrency_limit",
        }
        if terminal or attempt >= max_attempts:
            return None
        delay = max(0.0, backoff_seconds) * (2 ** (attempt - 1))
        delay += random.uniform(0.0, min(10.0, delay * 0.2))
        time.sleep(delay)
    return None


def _round_messages_before_quality(round_dir: Path) -> List[Dict[str, Any]]:
    path = round_dir.parent / "agent_messages.jsonl"
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: List[Dict[str, Any]] = []
    round_id = int(round_dir.name.split("_")[-1])
    for row in rows:
        if int(row.get("round_id", -1)) != round_id:
            continue
        # Quality runs after rollback observation and before DiagnosticCoach.
        if row.get("sender") == "DiagnosticCoach":
            break
        selected.append(row)
        if (row.get("content") or {}).get("event") == "rollback_decision":
            break
    return selected


def _compact_status_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for message in messages:
        content = dict(message.get("content") or {})
        compacted.append({
            "sender": message.get("sender"),
            "message_type": message.get("message_type"),
            "round_id": message.get("round_id"),
            "job_id": message.get("job_id") or content.get("job_id"),
            "content": {
                key: content.get(key)
                for key in (
                    "event",
                    "status",
                    "attempts",
                    "error",
                    "action",
                    "best_round",
                    "best_reward",
                    "current_reward",
                    "relative_drop",
                )
                if content.get(key) is not None
            },
        })
    return compacted


def _tokens_for_structure(item: Mapping[str, Any]) -> set:
    tokens = set(str(value) for value in (item.get("reliability_tags") or []))
    tokens.update(
        str(key)
        for key, value in dict(item.get("hotspot_contacts") or {}).items()
        if value
    )
    for key in ("high_quality_fragments", "low_quality_fragments"):
        for fragment in item.get(key) or []:
            tokens.update(str(reason) for reason in fragment.get("reasons") or [])
    return tokens


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def _structure_score(item: Mapping[str, Any]) -> float:
    reliability = float(item.get("reliability_score") or 0.0)
    fragments = list(item.get("high_quality_fragments") or [])
    fragments += list(item.get("low_quality_fragments") or [])
    best_fragment = max(
        (float(fragment.get("quality_score") or 0.0) for fragment in fragments),
        default=0.0,
    )
    contacts = min(1.0, float(item.get("interface_contact_count") or 0.0) / 30.0)
    return 0.45 * reliability + 0.35 * best_fragment + 0.2 * contacts


def _select_diverse_structures(
    summaries: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    remaining = [dict(item) for item in summaries]
    selected: List[Dict[str, Any]] = []
    while remaining and len(selected) < limit:
        def rank(item: Mapping[str, Any]) -> Tuple[float, float]:
            relevance = _structure_score(item)
            redundancy = max(
                (
                    _jaccard(_tokens_for_structure(item), _tokens_for_structure(chosen))
                    for chosen in selected
                ),
                default=0.0,
            )
            return 0.7 * relevance - 0.3 * redundancy, relevance

        chosen = max(remaining, key=rank)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _prune_structure(item: Mapping[str, Any]) -> Dict[str, Any]:
    row = copy.deepcopy(dict(item))
    structure_file = str(row.get("structure_file") or "")
    if structure_file:
        row["structure_file"] = Path(structure_file).name
    row["high_quality_fragments"] = sorted(
        list(row.get("high_quality_fragments") or []),
        key=lambda fragment: float(fragment.get("quality_score") or 0.0),
        reverse=True,
    )[:2]
    row["low_quality_fragments"] = sorted(
        list(row.get("low_quality_fragments") or []),
        key=lambda fragment: float(fragment.get("quality_score") or 0.0),
    )[:2]
    row["target_contact_residues"] = list(
        row.get("target_contact_residues") or []
    )[:6]
    return row


def _compact_active_examples(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(value))
    current = dict(result.get("current_round") or {})
    current["strict_positive_examples"] = list(
        current.get("strict_positive_examples") or current.get("positive_examples") or []
    )[:4]
    current["near_miss_examples"] = list(
        current.get("near_miss_examples") or []
    )[:4]
    current["other_negative_examples"] = list(
        current.get("other_negative_examples") or current.get("hard_negative_examples") or []
    )[:5]
    current.pop("positive_examples", None)
    current.pop("hard_negative_examples", None)
    if current:
        result["current_round"] = current
    prior = dict(result.get("prior_rounds") or {})
    prior["by_round"] = list(prior.get("by_round") or [])[-3:]
    prior["strict_positive_examples"] = list(
        prior.get("strict_positive_examples") or prior.get("positive_examples") or []
    )[:4]
    prior["near_miss_examples"] = list(prior.get("near_miss_examples") or [])[:4]
    prior["other_negative_examples"] = list(
        prior.get("other_negative_examples") or prior.get("hard_negative_examples") or []
    )[:4]
    prior.pop("positive_examples", None)
    prior.pop("hard_negative_examples", None)
    if prior:
        result["prior_rounds"] = prior
    return result


def _compact_evaluation(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(value))
    # Active examples are already a top-level quality section.
    result.pop("active_learning_examples", None)
    preferred = (
        result.get("top_candidates")
        or result.get("top_by_core")
        or result.get("top_by_iptm")
        or result.get("top_by_score")
        or []
    )
    for key in ("top_by_score", "top_by_core", "top_by_iptm"):
        result.pop(key, None)
    result["top_candidates"] = list(preferred)[:5]
    result["failed_examples"] = list(result.get("failed_examples") or [])[:4]
    return result


def _compact_skills(value: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "guidance": list(item.get("guidance") or [])[:3],
            "allowed_config_keys": list(item.get("allowed_config_keys") or []),
            "required_inputs": list(item.get("required_inputs") or [])[:6],
        }
        for item in value
    ]


def _build_contexts(round_dir: Path) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    quality = next(
        item
        for item in build_prompt_corpus(round_dir)
        if item["kind"] == "quality.round_analysis"
    )
    faithful = copy.deepcopy(dict(quality["user"]))
    seed_context = faithful["context"]
    raw_evaluation = json.loads(
        (round_dir / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    raw_active_examples = json.loads(
        (round_dir / "active_learning_examples.json").read_text(encoding="utf-8")
    )
    raw_structural = json.loads(
        (round_dir / "structure_evaluation.json").read_text(encoding="utf-8")
    )
    raw_evaluation["metric_facts"] = (
        (seed_context.get("evaluation") or {}).get("metric_facts") or {}
    )
    raw_evaluation["active_learning_examples"] = raw_active_examples
    # Rebuild the pre-fix projection explicitly so this experiment remains
    # reproducible after production compaction is tightened.
    faithful["context"]["evaluation"] = compact_evaluation(raw_evaluation)
    faithful["context"]["active_learning_examples"] = (
        compact_active_learning_examples(raw_active_examples)
    )
    faithful["context"]["structural_analysis"] = compact_structural_analysis(
        raw_structural,
        include_summaries=True,
        max_summaries=MAX_STRUCTURE_SUMMARIES,
    )
    skills_payload = json.loads(
        (round_dir / "active_skills.json").read_text(encoding="utf-8")
    )
    skills = list(
        (skills_payload.get("activations_by_agent") or {}).get(
            "BinderQualityAnalysisAgent"
        )
        or []
    )
    faithful["context"]["active_skills"] = skills
    faithful["context"]["messages"] = compact_messages(
        _round_messages_before_quality(round_dir)
    )

    compact = {
        "round_id": faithful.get("round_id"),
        "context": compact_context_for_quality(faithful["context"]),
    }
    return str(quality["system"]), faithful, compact


def _validate_final(
    result: Optional[Mapping[str, Any]],
    metric_facts: Mapping[str, Any],
) -> Dict[str, Any]:
    if result is None:
        return {"valid": False, "reason": "missing_or_unparseable_result"}
    expected = {
        "overall_assessment",
        "high_quality_modules",
        "low_quality_modules",
        "causal_factors",
        "next_round_guidance",
    }
    missing = sorted(expected - set(result))
    fact_issues = fact_check_text_against_metric_facts(
        json.dumps(result, ensure_ascii=False),
        metric_facts,
    )
    return {
        "valid": not missing and not fact_issues,
        "missing_keys": missing,
        "fact_check_issues": fact_issues,
    }


def _run_modular(
    client: OpenAICompatibleClient,
    *,
    compact_user: Mapping[str, Any],
    attempts: List[Dict[str, Any]],
    backoff_seconds: float,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    context = copy.deepcopy(dict(compact_user.get("context") or {}))
    metric_user = {
        "evaluation": context.get("evaluation"),
        "active_learning_examples": context.get("active_learning_examples"),
        "current_config": context.get("current_config"),
        "constraints": context.get("constraints"),
        "memory": context.get("memory"),
    }
    metric_result = _call_controlled(
        client,
        strategy="modular_low",
        stage="metrics",
        system=METRIC_SYSTEM,
        user=metric_user,
        max_tokens=1400,
        attempts=attempts,
        max_attempts=2,
        backoff_seconds=backoff_seconds,
    )
    if metric_result is None:
        return None, []

    summaries = list(
        (context.get("structural_analysis") or {}).get("summaries") or []
    )
    structure_results: List[Dict[str, Any]] = []
    for index in range(0, len(summaries), 3):
        batch = summaries[index : index + 3]
        structure_user = {
            "batch_index": index // 3,
            "batch_count": (len(summaries) + 2) // 3,
            "aggregate": {
                key: value
                for key, value in dict(context.get("structural_analysis") or {}).items()
                if key != "summaries"
            },
            "summaries": batch,
            "target_analysis": context.get("target_analysis"),
        }
        result = _call_controlled(
            client,
            strategy="modular_low",
            stage=f"structures_{index // 3 + 1}",
            system=STRUCTURE_SYSTEM,
            user=structure_user,
            max_tokens=1400,
            attempts=attempts,
            max_attempts=2,
            backoff_seconds=backoff_seconds,
        )
        if result is None:
            return None, structure_results
        structure_results.append(result)

    synthesis_user = {
        "round_id": compact_user.get("round_id"),
        "metric_specialist": metric_result,
        "structure_specialists": structure_results,
        "target_analysis": context.get("target_analysis"),
        "current_config": context.get("current_config"),
        "constraints": context.get("constraints"),
        "memory": context.get("memory"),
        "active_skills": context.get("active_skills"),
        "immutable_metric_facts": (
            (context.get("evaluation") or {}).get("metric_facts") or {}
        ),
    }
    final = _call_controlled(
        client,
        strategy="modular_low",
        stage="synthesis",
        system=BinderQualityAnalysisAgent.SYSTEM + SYNTHESIS_SUFFIX,
        user=synthesis_user,
        max_tokens=3000,
        attempts=attempts,
        max_attempts=2,
        backoff_seconds=backoff_seconds,
    )
    return final, structure_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", default=DEFAULT_ROUND_DIR)
    parser.add_argument("--llm-config", default="configs/llm_endpoints.gpt.json")
    parser.add_argument("--llm-model", default="gpt-5.5")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--backoff-seconds", type=float, default=45.0)
    parser.add_argument(
        "--lock-file",
        default="/tmp/binderloop_suixiang_api.lock",
    )
    parser.add_argument(
        "--out",
        default=(
            "outputs/gpt55_limit_probe_sc2rbd_round04/"
            "quality_strategy_test.json"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    round_dir = resolve(args.round_dir)
    system, faithful_user, compact_user = _build_contexts(round_dir)
    measurements = {
        "faithful_single_high": {
            "user_bytes": _json_bytes(faithful_user),
            "request_message_bytes": _json_bytes(_messages(system, faithful_user)),
            "section_bytes": {
                key: _json_bytes(value)
                for key, value in faithful_user["context"].items()
            },
        },
        "compact_single_high": {
            "user_bytes": _json_bytes(compact_user),
            "request_message_bytes": _json_bytes(_messages(system, compact_user)),
            "section_bytes": {
                key: _json_bytes(value)
                for key, value in compact_user["context"].items()
            },
        },
        "compact_single_low": {
            "user_bytes": _json_bytes(compact_user),
            "request_message_bytes": _json_bytes(_messages(system, compact_user)),
            "section_bytes": {
                key: _json_bytes(value)
                for key, value in compact_user["context"].items()
            },
        },
    }
    attempts: List[Dict[str, Any]] = []
    finals: Dict[str, Optional[Dict[str, Any]]] = {}

    if args.live:
        lock_path = Path(args.lock_file)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            high_client = _configured_client(
                resolve(args.llm_config),
                args.llm_model,
                reasoning="high",
            )
            low_client = _configured_client(
                resolve(args.llm_config),
                args.llm_model,
                reasoning="low",
            )
            for strategy, client, user in (
                ("faithful_single_high", high_client, faithful_user),
                ("compact_single_high", high_client, compact_user),
                ("compact_single_low", low_client, compact_user),
            ):
                finals[strategy] = _call_controlled(
                    client,
                    strategy=strategy,
                    stage="quality",
                    system=system,
                    user=user,
                    max_tokens=3000,
                    attempts=attempts,
                    max_attempts=1,
                    backoff_seconds=args.backoff_seconds,
                )
                time.sleep(20.0)
            modular_final, structure_results = _run_modular(
                low_client,
                compact_user=compact_user,
                attempts=attempts,
                backoff_seconds=args.backoff_seconds,
            )
            finals["modular_low"] = modular_final
            measurements["modular_low"] = {
                "stage_count": (
                    2
                    + len(
                        list(
                            (compact_user["context"].get("structural_analysis") or {})
                            .get("summaries")
                            or []
                        )
                    )
                    // 3
                ),
                "structure_result_count": len(structure_results),
            }
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    metric_facts = (
        (compact_user.get("context") or {}).get("evaluation") or {}
    ).get("metric_facts") or {}
    validation = {
        strategy: _validate_final(value, metric_facts)
        for strategy, value in finals.items()
    }
    result = {
        "round_dir": str(round_dir),
        "live": bool(args.live),
        "concurrency_control": {
            "max_outstanding_requests": 1,
            "interprocess_lock": args.lock_file,
            "single_call_http_retries": 1,
            "modular_transient_attempts": 2,
            "modular_backoff_seconds": args.backoff_seconds,
        },
        "measurements": measurements,
        "attempts": attempts,
        "validation": validation,
        "successful_strategies": [
            strategy
            for strategy, outcome in validation.items()
            if outcome.get("valid")
        ],
    }
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
