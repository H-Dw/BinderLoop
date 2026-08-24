"""Validated, observable JSON calls with one targeted repair attempt."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

from binderloop.llm import LLMDefinitiveError, OpenAICompatibleClient

MAX_RAW_TEXT_CHARS = 16_000


@dataclass
class StructuredValidation:
    valid: bool
    projected: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    unknown_fields: List[str] = field(default_factory=list)
    invalid_fields: List[str] = field(default_factory=list)
    illegal_evidence_ids: List[str] = field(default_factory=list)
    illegal_arm_ids: List[str] = field(default_factory=list)


@dataclass
class StructuredCallResult:
    value: Optional[Dict[str, Any]]
    attempts: List[Dict[str, Any]]
    repaired: bool = False
    error: Optional[str] = None


def validate_mapping(
    value: Any,
    *,
    required_fields: Sequence[str],
    optional_fields: Sequence[str] = (),
    field_validator: Optional[Callable[[Mapping[str, Any]], Mapping[str, Sequence[str]]]] = None,
) -> StructuredValidation:
    required, optional = set(required_fields), set(optional_fields)
    if not isinstance(value, Mapping):
        return StructuredValidation(False, invalid_fields=["$:not_an_object"])
    keys = set(map(str, value.keys()))
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    projected = {key: value.get(key) for key in required | optional if key in value}
    details = dict(field_validator(projected) or {}) if field_validator else {}
    invalid = sorted(map(str, details.get("invalid_fields", [])))
    evidence = sorted(map(str, details.get("illegal_evidence_ids", [])))
    arms = sorted(map(str, details.get("illegal_arm_ids", [])))
    return StructuredValidation(
        not (missing or invalid or evidence or arms), projected, missing, unknown,
        invalid, evidence, arms,
    )


def call_structured_json(
    llm: OpenAICompatibleClient,
    *,
    system: str,
    user: Mapping[str, Any],
    required_fields: Sequence[str],
    optional_fields: Sequence[str] = (),
    field_validator: Optional[Callable[[Mapping[str, Any]], Mapping[str, Sequence[str]]]] = None,
    temperature: float = 0.0,
    max_completion_tokens: int = 65_536,
    visible_json_tokens: int = 4096,
    thinking: Optional[str] = "low",
    reasoning_budget_tokens: Optional[int] = None,
    repair: bool = True,
    valid_arm_ids: Sequence[str] = (),
    valid_evidence_ids: Sequence[str] = (),
) -> StructuredCallResult:
    """Request validated JSON under an explicit visible-output contract.

    The first request budgets ``visible_json_tokens`` plus a thinking reserve.
    Empty ``content``, ``finish_reason=length``, or reasoning tokens consuming
    the completion budget retry at a larger max_tokens while keeping thinking.
    Chain-of-thought fields are never parsed as the JSON answer. A final
    targeted schema repair remains available for visible-but-invalid JSON.
    """
    from binderloop.agents.context_compaction import MAX_PROMPT_BYTES, enforce_byte_budget
    from binderloop.llm import _ensure_json_instruction, _json_output_retry_reason, reasoning_reserve_tokens

    attempts: List[Dict[str, Any]] = []
    visible_contract = max(1, int(visible_json_tokens))
    endpoint_ceiling = max(1, int(max_completion_tokens))
    if reasoning_budget_tokens is not None:
        reasoning_reserve = max(0, int(reasoning_budget_tokens))
    else:
        reasoning_reserve = reasoning_reserve_tokens(thinking, getattr(llm, "resolved_endpoint", None))
    primary_budget = min(endpoint_ceiling, visible_contract + reasoning_reserve)
    request_plan = [(primary_budget, thinking, reasoning_budget_tokens, False)]
    scaled_budget = min(endpoint_ceiling, max(primary_budget * 2, visible_contract + (2 * reasoning_reserve), visible_contract + 16_384))
    if scaled_budget > primary_budget:
        request_plan.append((scaled_budget, thinking, reasoning_budget_tokens, False))
    request_plan.append((*request_plan[-1][:3], False))
    current_system, current_user = _ensure_json_instruction(system), dict(user)
    prompt_budget = MAX_PROMPT_BYTES
    endpoint = getattr(llm, "resolved_endpoint", None)
    if endpoint is not None and getattr(endpoint, "max_prompt_bytes", None):
        prompt_budget = int(endpoint.max_prompt_bytes)
    repair_used = False
    plan_index = 0
    while plan_index < len(request_plan) + (1 if repair else 0):
        if plan_index < len(request_plan):
            completion_budget, request_thinking, request_reasoning_budget, repair_attempt = request_plan[plan_index]
        else:
            repair_attempt = True
            repair_used = True
            completion_budget = min(endpoint_ceiling, max(visible_contract, 1024))
            request_thinking, request_reasoning_budget = "off", 0
        prompt_original_text = json.dumps(current_user, ensure_ascii=False, separators=(",", ":"), default=str)
        safe_user = enforce_byte_budget(current_user, max_bytes=prompt_budget)
        prompt_final_text = json.dumps(safe_user, ensure_ascii=False, separators=(",", ":"), default=str)
        prompt_audit = {
            "prompt_original_bytes": len(prompt_original_text.encode("utf-8")),
            "prompt_final_bytes": len(prompt_final_text.encode("utf-8")),
            "prompt_compacted": prompt_final_text != prompt_original_text,
            "prompt_sha256": hashlib.sha256(prompt_final_text.encode("utf-8")).hexdigest(),
        }
        try:
            if hasattr(llm, "create_chat_completion"):
                response = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": current_system},
                        {"role": "user", "content": prompt_final_text},
                    ],
                    temperature=0.0 if repair_attempt else temperature,
                    max_tokens=completion_budget, thinking=request_thinking,
                    reasoning_budget_tokens=request_reasoning_budget,
                    response_format={"type": "json_object"},
                )
            else:
                legacy = llm.chat_json(system=current_system, user=safe_user,
                                       temperature=0.0 if repair_attempt else temperature,
                                       max_tokens=completion_budget, thinking=request_thinking)
                response = {"choices": [{"message": {"content": json.dumps(legacy, ensure_ascii=False)},
                                           "finish_reason": "legacy_chat_json"}], "usage": {}}
        except LLMDefinitiveError as exc:
            if getattr(exc, "failure_class", None) != "context_limit":
                raise
            attempts.append({
                "attempt": len(attempts) + 1, "repair": repair_attempt, "ok": False,
                "failure_kind": "definitive", "failure_class": "context_limit",
                "status": getattr(exc, "status_code", None), "error_type": type(exc).__name__,
                "error": str(exc)[-2000:], "requested_completion_tokens": completion_budget,
                "visible_json_tokens": visible_contract, "thinking": request_thinking,
                "reasoning_budget_tokens": request_reasoning_budget, **prompt_audit,
            })
            return StructuredCallResult(None, attempts, repair_used, str(exc))
        except Exception as exc:
            attempts.append({"attempt": len(attempts)+1, "repair": repair_attempt, "ok": False,
                             "failure_kind": "transport", "error_type": type(exc).__name__, "error": str(exc)[-2000:],
                             "requested_completion_tokens": completion_budget, "visible_json_tokens": visible_contract,
                             "thinking": request_thinking, "reasoning_budget_tokens": request_reasoning_budget,
                             **prompt_audit})
            return StructuredCallResult(None, attempts, repair_used, str(exc))
        choice = dict((response.get("choices") or [{}])[0] or {})
        message = dict(choice.get("message") or {})
        # Never parse reasoning_content / reasoning_details as the JSON answer.
        raw = str(message.get("content") or "")
        parsed: Any = None
        parse_error = None
        try:
            parsed = json.loads(_extract_json_object(raw))
        except Exception as exc:
            parse_error = str(exc)
        validation = validate_mapping(parsed, required_fields=required_fields,
                                      optional_fields=optional_fields, field_validator=field_validator)
        usage = dict(response.get("usage") or {})
        details = dict(usage.get("completion_tokens_details") or {})
        reasoning_tokens = int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        finish_reason = choice.get("finish_reason")
        retry_reason = None
        if not validation.valid:
            retry_reason = _json_output_retry_reason(
                message, finish_reason, raw, usage,
                requested_max_tokens=completion_budget, parse_error=parse_error,
                parse_ok=parse_error is None and str(finish_reason or "").lower() != "length",
            )
        provider_meta = dict(response.get("_binder_harness") or {})
        audit = {
            "attempt": len(attempts)+1, "repair": repair_attempt, "ok": validation.valid,
            "parse_error": parse_error, "finish_reason": finish_reason, "retry_reason": retry_reason,
            "missing_fields": validation.missing_fields, "unknown_fields": validation.unknown_fields,
            "invalid_fields": validation.invalid_fields, "illegal_evidence_ids": validation.illegal_evidence_ids,
            "illegal_arm_ids": validation.illegal_arm_ids, "usage": usage,
            "reasoning_tokens": reasoning_tokens, "visible_output_tokens": max(0, completion_tokens-reasoning_tokens),
            "visible_json_tokens": visible_contract, "requested_completion_tokens": completion_budget,
            "effective_completion_tokens": provider_meta.get("effective_completion_tokens", completion_budget),
            "completion_clamp_reason": provider_meta.get("completion_clamp_reason"),
            "thinking": request_thinking, "reasoning_budget_tokens": request_reasoning_budget,
            "transport_attempts": provider_meta.get("transport_attempts", 1),
            "transport_retry_count": provider_meta.get("retry_count", 0),
            "raw_text": raw[:MAX_RAW_TEXT_CHARS], "raw_text_truncated": len(raw) > MAX_RAW_TEXT_CHARS,
            "raw_text_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "response_fields": sorted(response.keys()), "message_fields": sorted(message.keys()),
            **prompt_audit,
        }
        attempts.append(audit)
        if validation.valid:
            return StructuredCallResult(validation.projected, attempts, repair_used)
        if retry_reason and plan_index + 1 < len(request_plan):
            plan_index += 1
            continue
        if repair and not repair_attempt and raw.strip():
            current_system = (
                "Repair the previous response into one JSON object only. Do not re-analyze or change the decision. "
                "Use exactly the required semantic fields; explanatory/unknown fields may be omitted."
            )
            current_user = {
                "task": "repair_json_only_do_not_reanalyze",
                "required_fields": list(required_fields), "optional_fields": list(optional_fields),
                "missing_fields": validation.missing_fields, "unknown_fields": validation.unknown_fields,
                "invalid_fields": validation.invalid_fields, "illegal_evidence_ids": validation.illegal_evidence_ids,
                "illegal_arm_ids": validation.illegal_arm_ids, "valid_arm_ids": list(valid_arm_ids),
                "valid_evidence_ids": list(valid_evidence_ids), "previous_raw_response": raw[:MAX_RAW_TEXT_CHARS],
            }
            plan_index = len(request_plan)
            continue
        break
    return StructuredCallResult(None, attempts, repair_used, "structured output remained invalid after visible-output retries and targeted repair")


def _extract_json_object(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return text[start:end + 1]


def _response_retry_reason(message: Mapping[str, Any], finish_reason: Any) -> Optional[str]:
    if str(finish_reason or "").lower() == "length":
        return "length"
    content = str(message.get("content") or "")
    if not content.strip():
        if any(message.get(key) for key in ("reasoning", "reasoning_content", "reasoning_details")):
            return "reasoning_only"
        return "empty_visible_content"
    return None
