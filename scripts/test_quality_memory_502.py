#!/usr/bin/env python3
"""Run a sequential quality-prompt A/B test against the configured LLM API.

The baseline is reconstructed from the requested round. The indexed-memory
variant keeps every other quality-prompt section identical and replaces only
the memory block with structured-retrieval evidence cards from a durable
memory fixture. Requests are deliberately serialized and use one HTTP attempt
so concurrency and retry amplification cannot explain the result.
"""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.context_compaction import compact_memory
from binderloop.agents.memory_retrieval_agent import (
    MemoryRetrievalAgent,
    MemoryRetrievalQuery,
)
from binderloop.llm import (
    LLMConfigError,
    LLMHTTPError,
    LLMTransportError,
    OpenAICompatibleClient,
)
from binderloop.memory import ExperimentMemoryStore
from scripts.benchmark_memory_optimization import build_items
from scripts.probe_llm_limits import build_prompt_corpus


DEFAULT_ROUND_DIR = (
    "outputs/sc2rbd_closed_loop_llm_np_160s_8r_v17_bug/round_00"
)
DEFAULT_MEMORY_DIR = (
    "outputs/sc2rbd_closed_loop_llm_np_160s_8r_v17_bug/"
    "sc2rbd_closed_loop_llm_np_160s_8r_v17/memory"
)


def _wire_bytes(value: Any) -> int:
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
    if "cdn" in lowered or "防火墙" in lowered or "源服务器" in lowered:
        return "cdn_origin"
    if "ssl" in lowered:
        return "ssl_transport"
    if "concurr" in lowered or "并发" in lowered or "too many requests" in lowered:
        return "concurrency_limit"
    return "other_transport"


def _call(
    client: OpenAICompatibleClient,
    *,
    messages: List[Dict[str, str]],
    variant: str,
    trial: int,
) -> Dict[str, Any]:
    started = time.time()
    base = {
        "variant": variant,
        "trial": trial,
        "request_message_bytes": _wire_bytes(messages),
    }
    try:
        data = client.create_chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=32,
            max_retries=1,
        )
        choice = dict((data.get("choices") or [{}])[0] or {})
        message = dict(choice.get("message") or {})
        content = str(message.get("content") or "")
        return {
            **base,
            "ok": True,
            "elapsed_seconds": round(time.time() - started, 3),
            "finish_reason": choice.get("finish_reason"),
            "usage": dict(data.get("usage") or {}),
            "content_bytes": len(content.encode("utf-8")),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    except LLMHTTPError as exc:
        detail = exc.detail
        return {
            **base,
            "ok": False,
            "status_code": exc.status_code,
            "error_type": _classify_error(detail),
            "detail": detail[-1000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except (LLMTransportError, LLMConfigError) as exc:
        detail = str(exc)
        return {
            **base,
            "ok": False,
            "error_type": _classify_error(detail),
            "detail": detail[-1000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        detail = str(exc)
        return {
            **base,
            "ok": False,
            "error_type": _classify_error(detail),
            "detail": detail[-1000:],
            "elapsed_seconds": round(time.time() - started, 3),
        }


def _build_variants(
    *,
    round_dir: Path,
    memory_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    corpus = build_prompt_corpus(round_dir)
    quality = next(
        item for item in corpus if item["kind"] == "quality.round_analysis"
    )
    current_user = copy.deepcopy(quality["user"])
    current_context = dict(current_user.get("context") or {})
    evaluation = dict(current_context.get("evaluation") or {})

    store = ExperimentMemoryStore(memory_dir)
    memory = store.load()
    items = build_items(memory)
    failure_tags = [
        str(key)
        for key, value in dict(evaluation.get("tag_counts") or {}).items()
        if value and not str(key).startswith("pass_")
    ]
    retrieval = MemoryRetrievalAgent(
        llm=None,
        candidate_limit=24,
        top_k=8,
        mmr_lambda=0.7,
    ).retrieve(
        items,
        MemoryRetrievalQuery(
            target=memory.target,
            failure_tags=failure_tags,
            intent=(
                "Explain current failures and choose evidence-backed "
                "next-round parameter changes."
            ),
        ),
    )
    indexed_summary = store.summarize_for_agent(
        memory,
        extend_memory=True,
        recalled_items=retrieval.items,
    )
    optimized_user = copy.deepcopy(current_user)
    optimized_user["context"]["memory"] = compact_memory(indexed_summary)

    memory_free_user = copy.deepcopy(current_user)
    memory_free_user["context"]["memory"] = {}
    return {
        "current_memory": {
            "system": quality["system"],
            "user": current_user,
            "provenance": quality["provenance"],
            "selected_memory_items": [],
        },
        "indexed_memory": {
            "system": quality["system"],
            "user": optimized_user,
            "provenance": "v17 durable-memory fixture; deterministic retrieval fallback",
            "selected_memory_items": [item.item_id for item in retrieval.items],
        },
        "memory_removed": {
            "system": quality["system"],
            "user": memory_free_user,
            "provenance": "negative control; only the memory block is removed",
            "selected_memory_items": [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", default=DEFAULT_ROUND_DIR)
    parser.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--llm-config", default="configs/llm_endpoints.gpt.json")
    parser.add_argument("--llm-model", default="gpt-5.5")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=15.0)
    parser.add_argument(
        "--out",
        default="outputs/gpt55_limit_probe_sc2rbd_round00/quality_memory_502_test.json",
    )
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    variants = _build_variants(
        round_dir=resolve(args.round_dir),
        memory_dir=resolve(args.memory_dir),
    )
    measurements: Dict[str, Any] = {}
    for name, variant in variants.items():
        user = dict(variant["user"])
        context = dict(user.get("context") or {})
        measurements[name] = {
            "provenance": variant["provenance"],
            "system_bytes": len(str(variant["system"]).encode("utf-8")),
            "user_bytes": _wire_bytes(user),
            "memory_bytes": _wire_bytes(context.get("memory") or {}),
            "context_section_bytes": {
                key: _wire_bytes(value) for key, value in context.items()
            },
            "selected_memory_items": variant["selected_memory_items"],
        }

    attempts: List[Dict[str, Any]] = []
    if args.live:
        client = OpenAICompatibleClient.from_json(resolve(args.llm_config))
        if client is None:
            raise RuntimeError("LLM client was not created")
        client.configure_default(model_key=args.llm_model)
        sequence = ["current_memory", "indexed_memory"]
        for trial in range(1, max(1, int(args.trials)) + 1):
            if trial % 2 == 0:
                sequence = list(reversed(sequence))
            for variant_name in sequence:
                variant = variants[variant_name]
                attempts.append(
                    _call(
                        client,
                        messages=_messages(variant["system"], variant["user"]),
                        variant=variant_name,
                        trial=trial,
                    )
                )
                time.sleep(max(0.0, float(args.delay_seconds)))

    summary: Dict[str, Dict[str, Any]] = {}
    for name in ("current_memory", "indexed_memory"):
        rows = [row for row in attempts if row["variant"] == name]
        summary[name] = {
            "attempt_count": len(rows),
            "success_count": sum(1 for row in rows if row.get("ok")),
            "error_counts": {
                error_type: sum(
                    1 for row in rows if row.get("error_type") == error_type
                )
                for error_type in sorted(
                    {str(row.get("error_type")) for row in rows if row.get("error_type")}
                )
            },
            "mean_elapsed_seconds": (
                round(
                    sum(float(row.get("elapsed_seconds") or 0.0) for row in rows)
                    / len(rows),
                    3,
                )
                if rows
                else None
            ),
        }

    result = {
        "live": bool(args.live),
        "serialized_requests": True,
        "http_retries_per_request": 1,
        "measurements": measurements,
        "attempts": attempts,
        "summary": summary,
        "interpretation_guardrail": (
            "A success-rate difference in this small A/B sample is not evidence "
            "that memory caused or prevented a CDN failure. Compare payload size "
            "and classified error bodies as well."
        ),
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
