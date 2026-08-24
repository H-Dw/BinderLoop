
import json
import os
import http.client
import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

ENDPOINT_OUTPUT_CEILING_TOKENS = 65_536
JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}
REASONING_RESERVE_TOKENS = {
    "low": 4_096,
    "medium": 8_192,
    "enabled": 8_192,
    "on": 8_192,
    "true": 8_192,
    "high": 16_384,
    "xhigh": 16_384,
    "max": 32_768,
}

from binderloop.secrets import SecretStore
from binderloop.file_lock import exclusive_file_lock


@dataclass
class ModelEndpointCapabilities:
    """Optional endpoint features which are not uniformly provider-supported."""

    logprobs: str = "auto"
    top_logprobs_max: Optional[int] = None


@dataclass
class ModelEndpoint:
    name: str
    base_url: str
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    provider: str = "openai-compatible"
    timeout_seconds: int = 60
    default_headers: Dict[str, str] = field(default_factory=dict)
    # Optional reasoning control.  OpenRouter uses `extra_body.reasoning`,
    # OpenAI-compatible endpoints commonly use `reasoning_effort`, and
    # Anthropic-compatible endpoints use a `thinking` object.
    thinking: Optional[str] = None
    thinking_budget_tokens: Optional[int] = None
    # Endpoint-specific production guard populated from live limit probes.
    # This caps the serialized user payload; system/output reserves are
    # accounted for when choosing the value.
    max_prompt_bytes: Optional[int] = None
    context_window_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    # Optional cross-process exclusive request lock. This is intentionally
    # endpoint-specific so independent providers do not block one another.
    request_lock_path: Optional[str] = None
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    # Provider-specific request body additions.  This lets users keep the client
    # small while still passing knobs such as `top_p`, `reasoning`, etc.
    extra_body: Dict[str, Any] = field(default_factory=dict)
    capabilities: ModelEndpointCapabilities = field(default_factory=ModelEndpointCapabilities)

    def __post_init__(self) -> None:
        # Keep direct construction convenient for tests and downstream callers.
        if isinstance(self.capabilities, Mapping):
            self.capabilities = ModelEndpointCapabilities(**dict(self.capabilities))


@dataclass
class LLMSettings:
    default_model: str
    endpoints: Dict[str, ModelEndpoint]
    enabled: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMConfigError(RuntimeError):
    pass


class LLMTransportError(RuntimeError):
    """Recoverable transport failure safe to resume from a checkpoint."""

    def __init__(self, message: str, *, failure_class: str = "transport", retry_after_seconds: Optional[float] = None):
        self.failure_class = failure_class
        self.retryable = True
        self.recoverable_stop = True
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class LLMHTTPError(LLMConfigError):
    """Non-retryable HTTP response with machine-readable status/detail."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = int(status_code)
        self.detail = str(detail or "")
        super().__init__(f"LLM HTTP error {self.status_code}: {self.detail}")


class LLMDefinitiveError(LLMHTTPError):
    """Terminal provider failure that must bypass every retry layer."""

    def __init__(self, status_code: int, detail: str, *, failure_class: str):
        self.failure_class = failure_class
        self.retryable = False
        self.recoverable_stop = False
        self.definitive_stop = True
        super().__init__(status_code, detail)


WEB_SEARCH_PAYLOAD_KEYS = frozenset({
    "tools",
    "tool_choice",
    "plugins",
    "web_search",
    "web_search_options",
    "file_search",
    "file_search_options",
    "functions",
    "function_call",
    "builtin_tools",
    "extra_tools",
})
WEB_PLUGIN_IDS = frozenset({"web", "web_search", "online", "openaisearch", "file_search"})
NO_WEB_SEARCH_SYSTEM_INSTRUCTION = (
    "Do not search the web, call tools, or retrieve external documents. "
    "Use only the supplied structured payload."
)


def is_online_web_search_model(model: Optional[str]) -> bool:
    token = str(model or "").strip().lower()
    if not token:
        return False
    return token.endswith(":online") or ":online" in token or token.endswith("-online")


def reject_online_web_search_model(model: Optional[str]) -> None:
    if is_online_web_search_model(model):
        raise LLMConfigError(
            "web search is disabled but the endpoint model %r is an online/search variant" % (model,)
        )


def _plugin_id(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("id") or item.get("name") or "").strip().lower()
    return str(item or "").strip().lower()


def strip_web_search_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Drop provider tool/plugin fields that can retrieve external data."""
    cleaned: Dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        token = str(key)
        if token in WEB_SEARCH_PAYLOAD_KEYS:
            continue
        if token.endswith("_plugins"):
            continue
        if isinstance(value, list) and any(_plugin_id(item) in WEB_PLUGIN_IDS for item in value):
            filtered = [item for item in value if _plugin_id(item) not in WEB_PLUGIN_IDS]
            if filtered:
                cleaned[token] = filtered
            continue
        cleaned[token] = value
    return cleaned


def ensure_no_web_search_instruction(system: str) -> str:
    text = str(system or "").strip()
    if NO_WEB_SEARCH_SYSTEM_INSTRUCTION.lower() in text.lower():
        return text
    if not text:
        return NO_WEB_SEARCH_SYSTEM_INSTRUCTION
    return text + "\n\n" + NO_WEB_SEARCH_SYSTEM_INSTRUCTION


def _validate_endpoint_config(path: str, value: Mapping[str, Any]) -> None:
    for key in ("base_url", "model"):
        if not isinstance(value.get(key), str) or not str(value.get(key)).strip():
            raise LLMConfigError(f"{path}.{key} must be a nonempty string")
    for key in ("provider", "api_key_env", "api_key", "thinking", "request_lock_path"):
        if key in value and value[key] is not None and (not isinstance(value[key], str) or not value[key].strip()):
            raise LLMConfigError(f"{path}.{key} must be a nonempty string when set")
    for key in ("timeout_seconds", "max_retries", "thinking_budget_tokens", "max_prompt_bytes", "context_window_tokens", "max_output_tokens"):
        if key in value and value[key] is not None:
            item = value[key]
            if isinstance(item, bool) or not isinstance(item, (int, float)) or item <= 0:
                raise LLMConfigError(f"{path}.{key} must be a positive number")
    if "retry_backoff_seconds" in value:
        item = value["retry_backoff_seconds"]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0:
            raise LLMConfigError(f"{path}.retry_backoff_seconds must be a nonnegative number")
    for key in ("default_headers", "extra_body", "capabilities"):
        if key in value and not isinstance(value[key], dict):
            raise LLMConfigError(f"{path}.{key} must be an object")
    capabilities = value.get("capabilities", {})
    if isinstance(capabilities, dict):
        unknown = set(capabilities) - {"logprobs", "top_logprobs_max"}
        if unknown:
            raise LLMConfigError(f"{path}.capabilities has unknown fields: {sorted(unknown)}")
        mode = capabilities.get("logprobs", "auto")
        if mode not in {"auto", "required", "disabled"}:
            raise LLMConfigError(f"{path}.capabilities.logprobs must be auto, required, or disabled")
        maximum = capabilities.get("top_logprobs_max")
        if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0):
            raise LLMConfigError(f"{path}.capabilities.top_logprobs_max must be a positive integer")
    headers = value.get("default_headers", {})
    if isinstance(headers, dict) and not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise LLMConfigError(f"{path}.default_headers keys and values must be strings")


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat-completions client.

    Store keys in local JSON or environment variables; never commit the real file.
    """

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.secrets = SecretStore(settings.raw)
        self._logprobs_probe_cache: Dict[str, Dict[str, Any]] = {}
        self.last_json_call: Optional[Dict[str, Any]] = None

    @classmethod
    def from_json(cls, path: Union[str, Optional[Path]]) -> "Optional[OpenAICompatibleClient]":
        if path is None:
            return None
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LLM config not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LLMConfigError(f"LLM config is not valid JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise LLMConfigError("LLM config root must be an object")
        enabled = data.get("enabled", False)
        if not isinstance(enabled, bool):
            raise LLMConfigError("LLM config enabled must be a JSON boolean")
        endpoint_data = data.get("endpoints", {})
        if not isinstance(endpoint_data, dict):
            raise LLMConfigError("LLM config endpoints must be an object")
        endpoints: Dict[str, ModelEndpoint] = {}
        for name, value in endpoint_data.items():
            endpoint_path = f"endpoints.{name}" if isinstance(name, str) and name else "endpoints.<key>"
            if not isinstance(name, str) or not name.strip():
                raise LLMConfigError("LLM config endpoint keys must be nonempty strings")
            if not isinstance(value, dict):
                raise LLMConfigError(f"{endpoint_path} must be an object")
            _validate_endpoint_config(endpoint_path, value)
            try:
                endpoints[name] = ModelEndpoint(name=name, **value)
            except TypeError as exc:
                raise LLMConfigError(f"{endpoint_path}: {exc}") from exc
        default_model = data.get("default_model", "")
        if default_model is None:
            default_model = ""
        if not isinstance(default_model, str):
            raise LLMConfigError("LLM config default_model must be a string")
        default_model = default_model.strip()
        if enabled and not default_model:
            raise LLMConfigError("enabled LLM config requires an explicit nonempty default_model")
        if enabled and default_model not in endpoints:
            raise LLMConfigError(f"LLM config default_model references unknown endpoint: {default_model}")
        settings = LLMSettings(default_model=default_model, endpoints=endpoints, enabled=enabled, raw=data)
        return cls(settings)

    @property
    def resolved_endpoint_key(self) -> str:
        """Return the validated endpoint key selected for this client."""
        key = self.settings.default_model
        if key not in self.settings.endpoints:
            raise LLMConfigError(f"unknown model endpoint: {key}")
        return key

    @property
    def resolved_endpoint(self) -> ModelEndpoint:
        return self.settings.endpoints[self.resolved_endpoint_key]

    def configure_default(self, *, model_key: Optional[str] = None, thinking: Optional[str] = None) -> None:
        """Override endpoint selection/reasoning from CLI without editing key files."""
        if model_key is not None:
            if model_key not in self.settings.endpoints:
                raise LLMConfigError(f"unknown model endpoint: {model_key}")
            self.settings.default_model = model_key
        if thinking:
            key = self.settings.default_model
            if key not in self.settings.endpoints:
                raise LLMConfigError(f"unknown model endpoint: {key}")
            self.settings.endpoints[key].thinking = thinking

    def available(self) -> bool:
        if not (self.settings.enabled and self.settings.default_model in self.settings.endpoints):
            return False
        return bool(self._api_key_for(self.settings.endpoints[self.settings.default_model]))

    def preflight(self, *, max_tokens: int = 512, timeout_seconds: Optional[int] = 30) -> Dict[str, Any]:
        """Lightweight live API check before a long harness run.

        Sends a tiny JSON/nonce chat request against the configured default
        endpoint. For endpoints configured to reason, the probe uses the lowest
        reasoning level and a sufficient output budget so reasoning-only models
        have room to emit the requested JSON answer.

        Raises ``LLMConfigError`` / ``LLMTransportError`` /
        ``LLMHTTPError`` / ``LLMConfigError`` when the endpoint is unavailable
        or the response is not usable.
        """
        if not self.available():
            raise LLMConfigError(
                "Configured LLM endpoint is not available. Ensure enabled=true, "
                "the endpoint exists, and its api_key_env/api_key is set."
            )
        endpoint_key = self.settings.default_model
        endpoint = self.settings.endpoints[endpoint_key]
        previous_timeout = endpoint.timeout_seconds
        if timeout_seconds is not None:
            endpoint.timeout_seconds = int(timeout_seconds)
        nonce = f"binder-harness-llm-preflight-{int(time.time())}"
        started = time.time()
        has_reasoning = bool(
            endpoint.thinking
            or any(key in endpoint.extra_body for key in ("reasoning", "reasoning_effort", "thinking"))
        )
        probe_thinking = "low" if has_reasoning else "none"
        visible = max(1, int(max_tokens))
        plan = json_completion_plan(visible_tokens=visible, thinking=probe_thinking, endpoint=endpoint)
        last_error: Optional[BaseException] = None
        data: Dict[str, Any] = {}
        choice: Mapping[str, Any] = {}
        message: Mapping[str, Any] = {}
        text = ""
        try:
            for budget, _plan_thinking in plan:
                data = self.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a smoke-test endpoint for binder-harness. Return JSON only. "
                                "Do not include markdown."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps({
                                "task": "Return the exact nonce and confirm this was a live LLM call.",
                                "required_schema": {"ok": True, "nonce": nonce, "message": "short string"},
                                "nonce": nonce,
                            }),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=budget,
                    thinking=probe_thinking,
                    response_format=JSON_OBJECT_RESPONSE_FORMAT,
                )
                choices = data.get("choices") or []
                if not choices or not isinstance(choices[0], Mapping):
                    raise LLMConfigError(f"LLM preflight response had no usable choice: {data!r}")
                choice = choices[0]
                message = choice.get("message") or {} if isinstance(choice.get("message"), Mapping) else {}
                text = str(message.get("content") or "")
                try:
                    response = json.loads(_extract_json_object(text))
                except Exception as exc:
                    last_error = exc
                    retry_reason = _json_output_retry_reason(
                        message, choice.get("finish_reason"), text, data.get("usage") or {},
                        requested_max_tokens=budget, parse_error=str(exc), parse_ok=False,
                    )
                    if retry_reason:
                        continue
                    break
                if response.get("nonce") != nonce:
                    raise LLMConfigError(f"LLM preflight did not echo nonce {nonce!r}: {response!r}")
                if response.get("ok") is not True:
                    raise LLMConfigError(f"LLM preflight did not set ok=true: {response!r}")
                elapsed = time.time() - started
                return {
                    "llm_used": True,
                    "endpoint_key": endpoint_key,
                    "provider": endpoint.provider,
                    "base_url": endpoint.base_url,
                    "model": endpoint.model,
                    "thinking": endpoint.thinking,
                    "elapsed_seconds": round(elapsed, 3),
                    "response": response,
                }
        finally:
            endpoint.timeout_seconds = previous_timeout
        finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        message_fields = sorted(message.keys()) if isinstance(message, Mapping) else []
        raise LLMConfigError(
            "LLM preflight did not return valid JSON content "
            f"(finish_reason={finish_reason!r}, content_length={len(text)}, "
            f"message_fields={message_fields}, max_tokens={max_tokens}, "
            f"reasoning={probe_thinking!r}): {last_error}"
        )

    def _api_key_for(self, endpoint: ModelEndpoint) -> Optional[str]:
        if endpoint.api_key:
            return endpoint.api_key
        if endpoint.api_key_env and os.environ.get(endpoint.api_key_env):
            return os.environ[endpoint.api_key_env]
        if endpoint.api_key_env:
            return self.secrets.get(endpoint.api_key_env)
        return None

    def chat_json(self, *, system: str, user: Mapping[str, Any], model_key: Optional[str] = None,
                  temperature: float = 0.2, max_tokens: int = 8000,
                  max_prompt_bytes: Optional[int] = None,
                  thinking: Optional[str] = None,
                  json_object: bool = True,
                  allow_web_search: Optional[bool] = None) -> Dict[str, Any]:
        # Last-resort byte-budget guard.  The per-agent compactors usually keep
        # the payload small, but this *guarantees* the serialised user payload
        # never exceeds the hard ceiling (default 1 MB) before it hits the wire,
        # so a pathological round can no longer trip the provider's HTTP 400
        # context-length error and silently force a deterministic fallback.
        # Imported lazily to avoid a circular import (agents import llm).
        from binderloop.agents.context_compaction import enforce_byte_budget, MAX_PROMPT_BYTES
        key = model_key or self.settings.default_model
        if key not in self.settings.endpoints:
            raise LLMConfigError(f"unknown model endpoint: {key}")
        endpoint = self.settings.endpoints[key]
        endpoint_budget = endpoint.max_prompt_bytes
        prompt_budget = int(max_prompt_bytes or endpoint_budget or MAX_PROMPT_BYTES)
        safe_user = enforce_byte_budget(user, max_bytes=prompt_budget)
        visible_tokens = max(1, int(max_tokens))
        effective_thinking = thinking if thinking is not None else endpoint.thinking
        plan = json_completion_plan(visible_tokens=visible_tokens, thinking=effective_thinking, endpoint=endpoint)
        system_text = _ensure_json_instruction(system) if json_object else system
        if allow_web_search is False:
            system_text = ensure_no_web_search_instruction(system_text)
        user_text = json.dumps(safe_user, ensure_ascii=False, separators=(",", ":"))
        attempts: List[Dict[str, Any]] = []
        last_text = ""
        last_error = "no JSON completion attempt ran"
        response_format = JSON_OBJECT_RESPONSE_FORMAT if json_object else None
        for budget, _plan_thinking in plan:
            data = self.create_chat_completion(
                messages=[{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
                model_key=model_key,
                temperature=temperature,
                max_tokens=budget,
                thinking=thinking,
                response_format=response_format,
                allow_web_search=allow_web_search,
            )
            meta = dict(data.get("_binder_harness") or {})
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") if isinstance(choice, Mapping) else {}
            if not isinstance(message, Mapping):
                message = {}
            # Official DeepSeek semantics: reasoning_content is the chain of
            # thought and must not be parsed as the JSON answer. Empty content
            # is a failed visible completion and is retried.
            text = str(message.get("content") or "")
            last_text = text
            finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
            usage = dict(data.get("usage") or {})
            parse_error = None
            parsed: Any = None
            try:
                parsed = json.loads(_extract_json_object(text))
            except Exception as exc:
                parse_error = str(exc)
                last_error = parse_error
            retry_reason = _json_output_retry_reason(
                message, finish_reason, text, usage,
                requested_max_tokens=budget, parse_error=parse_error, parse_ok=parsed is not None,
            )
            attempts.append({
                "max_tokens": budget,
                "thinking": thinking if thinking is not None else endpoint.thinking,
                "finish_reason": finish_reason,
                "reasoning_tokens": meta.get("reasoning_tokens", 0),
                "visible_output_tokens": meta.get("visible_output_tokens"),
                "content_length": len(text),
                "parse_error": parse_error,
                "retry_reason": retry_reason,
                "used_reasoning_content": False,
                "message_fields": sorted(message.keys()),
            })
            if parsed is not None:
                self.last_json_call = {"ok": True, "attempts": attempts, "parse_error": None}
                return parsed
            if retry_reason is None:
                break
        self.last_json_call = {"ok": False, "attempts": attempts, "parse_error": last_error}
        return {"parse_error": last_error, "raw_text": last_text, "attempts": attempts}

    def chat_text(self, *, system: str, user: str, model_key: Optional[str] = None,
                  temperature: float = 0.2, max_tokens: int = 8000,
                  thinking: Optional[str] = None) -> str:
        message = self.chat_messages(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            model_key=model_key,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        return str(message.get("content") or "")

    def chat_messages(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        thinking: Optional[str] = None,
        reasoning_budget_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call chat-completions with caller-supplied messages.

        This returns the full assistant message dict, including provider-specific
        fields such as OpenRouter `reasoning_details`.  Callers that implement
        multi-turn reasoning should pass that assistant message back unchanged.
        """
        data = self.create_chat_completion(
            messages=messages,
            model_key=model_key,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_budget_tokens=reasoning_budget_tokens,
        )
        return dict(data["choices"][0]["message"])

    def probe_logprobs(self, *, model_key: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """Probe logprobs support, returning supported/unsupported/indeterminate.

        ``disabled`` is resolved statically and never sends a request. Probe
        outcomes are cached per endpoint. ``required`` remains visible in the
        result so callers can decide whether an indeterminate result is fatal.
        """
        key = model_key or self.settings.default_model
        if key not in self.settings.endpoints:
            raise LLMConfigError(f"unknown model endpoint: {key}")
        endpoint = self.settings.endpoints[key]
        mode = endpoint.capabilities.logprobs
        if mode == "disabled":
            return {"status": "unsupported", "source": "configuration", "mode": mode, "endpoint_key": key}
        if not force and key in self._logprobs_probe_cache:
            return dict(self._logprobs_probe_cache[key])
        try:
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                model_key=key, temperature=0.0, max_tokens=2, max_retries=1,
                extra_body={"logprobs": True, "top_logprobs": 1},
            )
            normalized = normalize_logprobs(response)
            if normalized["tokens"]:
                result = {"status": "supported", "source": "probe", "mode": mode,
                          "endpoint_key": key, "evidence": normalized}
            else:
                result = {"status": "indeterminate", "source": "probe", "mode": mode,
                          "endpoint_key": key, "reason": "successful response omitted usable logprobs"}
        except LLMHTTPError as exc:
            if _is_explicit_logprobs_rejection(exc.status_code, exc.detail):
                result = {"status": "unsupported", "source": "probe", "mode": mode,
                          "endpoint_key": key, "status_code": exc.status_code,
                          "reason": "provider explicitly rejected logprobs"}
            else:
                result = {"status": "indeterminate", "source": "probe", "mode": mode,
                          "endpoint_key": key, "status_code": exc.status_code,
                          "reason": "request failed without a definitive capability signal"}
        except (LLMTransportError, LLMConfigError) as exc:
            result = {"status": "indeterminate", "source": "probe", "mode": mode,
                      "endpoint_key": key, "reason": str(exc)}
        self._logprobs_probe_cache[key] = dict(result)
        return result

    def chat_label_distribution(
        self, *, system: str, user: Mapping[str, Any], labels: Sequence[str],
        model_key: Optional[str] = None, top_logprobs: Optional[int] = None,
        temperature: float = 0.0, max_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Return closed-label evidence, retrying responses with no visible label.

        Closed-label calls deliberately disable reasoning first so a tiny output
        allowance cannot be consumed entirely by hidden reasoning tokens. Providers
        that still return reasoning-only, empty, or length-truncated responses get
        two bounded retries before evidence is marked unavailable.
        """
        if not labels or any(not isinstance(label, str) or not label for label in labels):
            raise LLMConfigError("labels must contain nonempty strings")
        key = model_key or self.settings.default_model
        if key not in self.settings.endpoints:
            raise LLMConfigError(f"unknown model endpoint: {key}")
        endpoint = self.settings.endpoints[key]
        if endpoint.capabilities.logprobs == "disabled":
            raise LLMConfigError(f"logprobs are disabled for endpoint {key}")
        requested = int(top_logprobs or len(labels))
        if requested <= 0:
            raise LLMConfigError("top_logprobs must be positive")
        maximum = endpoint.capabilities.top_logprobs_max
        if maximum is not None:
            requested = min(requested, maximum)
        safe_user = dict(user)
        safe_user["allowed_labels"] = list(labels)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(safe_user, ensure_ascii=False)}]
        budgets = [max(64, int(max_tokens)), 512, 1024]
        attempts: List[Dict[str, Any]] = []
        last_response: Dict[str, Any] = {}
        last_evidence: Dict[str, Any] = {"format": "none", "tokens": [], "raw": None}
        for retry_index, completion_tokens in enumerate(dict.fromkeys(budgets)):
            response = self.create_chat_completion(
                messages=messages, model_key=key, temperature=temperature,
                max_tokens=completion_tokens, thinking="off", reasoning_budget_tokens=0,
                extra_body={"logprobs": True, "top_logprobs": requested},
            )
            last_response = response
            normalized = normalize_logprobs(response)
            last_evidence = normalized
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") if isinstance(choice, Mapping) else {}
            text = str(message.get("content") or "") if isinstance(message, Mapping) else ""
            finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
            distribution = _label_probabilities(labels, normalized)
            selected = text.strip() if text.strip() in labels else (max(distribution, key=distribution.get) if distribution else None)
            retry_reason = _visible_response_retry_reason(message, finish_reason, selected)
            attempts.append({
                "attempt": retry_index + 1, "retry": retry_index > 0,
                "requested_completion_tokens": completion_tokens, "thinking": "off",
                "reasoning_budget_tokens": 0, "finish_reason": finish_reason,
                "usage": dict(response.get("usage") or {}), "visible_text_length": len(text),
                "retry_reason": retry_reason, "available": selected in labels and bool(distribution),
            })
            if selected in labels and distribution:
                return {"status": "available", "label": selected, "distribution": distribution,
                        "text": text, "evidence": normalized, "response": response,
                        "request_artifact": {"attempts": attempts, "retry_count": retry_index}}
            if retry_reason is None:
                break
        return {"status": "unavailable", "label": None, "distribution": {}, "text": "",
                "evidence": last_evidence, "response": last_response,
                "unavailable_reason": attempts[-1].get("retry_reason") or "no_usable_closed_label_evidence",
                "request_artifact": {"attempts": attempts, "retry_count": max(0, len(attempts)-1)}}

    def effective_completion_budget(self, requested_tokens: int, *, model_key: Optional[str] = None) -> Dict[str, Any]:
        key = model_key or self.settings.default_model
        if key not in self.settings.endpoints:
            raise LLMConfigError(f"unknown model endpoint: {key}")
        requested = max(1, int(requested_tokens))
        endpoint_limit = self.settings.endpoints[key].max_output_tokens
        configured_limit = int(endpoint_limit) if endpoint_limit is not None else ENDPOINT_OUTPUT_CEILING_TOKENS
        effective_limit = min(configured_limit, ENDPOINT_OUTPUT_CEILING_TOKENS)
        effective = min(requested, effective_limit)
        if effective < requested:
            reason = "endpoint_output_ceiling" if configured_limit > ENDPOINT_OUTPUT_CEILING_TOKENS else (
                "endpoint_max_output_tokens" if endpoint_limit is not None else "endpoint_max_output_tokens_unspecified_safe_default"
            )
        else:
            reason = None if endpoint_limit is not None else "endpoint_limit_unspecified"
        return {"requested_completion_tokens": requested, "effective_completion_tokens": effective,
                "completion_clamp_reason": reason, "endpoint_max_output_tokens": endpoint_limit,
                "endpoint_output_ceiling_tokens": ENDPOINT_OUTPUT_CEILING_TOKENS}

    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        max_retries: Optional[int] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        thinking: Optional[str] = None,
        reasoning_budget_tokens: Optional[int] = None,
        response_format: Optional[Mapping[str, Any]] = None,
        allow_web_search: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Return the complete OpenAI-compatible response envelope.

        Agent-facing helpers intentionally return only the assistant message.
        Limit probes and observability tooling need ``usage``, ``finish_reason``
        and provider metadata, so this lower-level method preserves the full
        response without changing the established helpers.
        """
        key = model_key or self.settings.default_model
        if key not in self.settings.endpoints:
            raise LLMConfigError(f"unknown model endpoint: {key}")
        endpoint = self.settings.endpoints[key]
        budget_meta = self.effective_completion_budget(max_tokens, model_key=key)
        max_tokens = int(budget_meta["effective_completion_tokens"])
        api_key = self._api_key_for(endpoint)
        if not api_key:
            raise LLMConfigError(f"missing API key for endpoint {key}; set api_key_env or api_key")
        url = endpoint.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": endpoint.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        combined_extra_body = dict(endpoint.extra_body)
        if extra_body:
            combined_extra_body.update(dict(extra_body))
        if allow_web_search is False:
            reject_online_web_search_model(endpoint.model)
            combined_extra_body = strip_web_search_payload(combined_extra_body)
        thinking_endpoint = endpoint
        if thinking is not None:
            # A per-call override is authoritative: remove every conflicting
            # provider dialect before adding the selected dialect back.
            for key_to_remove in ("reasoning", "reasoning_effort", "thinking"):
                combined_extra_body.pop(key_to_remove, None)
            thinking_endpoint = replace(
                endpoint, thinking=thinking,
                thinking_budget_tokens=(int(reasoning_budget_tokens) if reasoning_budget_tokens is not None else endpoint.thinking_budget_tokens),
                extra_body={},
            )
        elif reasoning_budget_tokens is not None:
            thinking_endpoint = replace(endpoint, thinking_budget_tokens=int(reasoning_budget_tokens))
        payload.update(combined_extra_body)
        payload.update(_thinking_payload(thinking_endpoint))
        _drop_incompatible_reasoning_keys(payload, thinking_endpoint)
        if allow_web_search is False:
            payload = strip_web_search_payload(payload)
        if response_format:
            payload["response_format"] = dict(response_format)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", **endpoint.default_headers}
        body = json.dumps(payload).encode("utf-8")
        attempts = max(1, int(endpoint.max_retries if max_retries is None else max_retries))
        backoff = max(0.0, float(endpoint.retry_backoff_seconds or 0.0))
        last_error: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with _endpoint_request_lock(endpoint):
                    with urllib.request.urlopen(
                        req,
                        timeout=int(timeout_seconds or endpoint.timeout_seconds),
                    ) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")[-1000:]
                definitive_class = _definitive_failure_class(exc.code, detail)
                if definitive_class is not None:
                    raise LLMDefinitiveError(exc.code, detail, failure_class=definitive_class) from exc
                if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                    raise LLMHTTPError(exc.code, detail) from exc
                retry_after = _retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
                failure_class = "rate_limit" if exc.code == 429 else "transient_http"
                last_error = LLMTransportError(
                    f"LLM transient HTTP error {exc.code}: {detail}",
                    failure_class=failure_class, retry_after_seconds=retry_after,
                )
            except _TRANSIENT_TRANSPORT_ERRORS as exc:
                failure_class = "ssl" if _is_ssl_error(exc) else "transport"
                last_error = LLMTransportError(str(exc), failure_class=failure_class)
            except json.JSONDecodeError as exc:
                last_error = exc

            if attempt < attempts:
                delay = backoff * (2 ** (attempt - 1))
                delay += random.uniform(0.0, min(10.0, delay * 0.2))
                time.sleep(delay)
        else:
            if isinstance(last_error, LLMTransportError):
                raise last_error
            raise LLMTransportError(f"LLM request failed after {attempts} attempts: {last_error}") from last_error
        result = dict(data)
        choice = (result.get("choices") or [{}])[0]
        usage = dict(result.get("usage") or {})
        details = dict(usage.get("completion_tokens_details") or {})
        reasoning_tokens = int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        result["_binder_harness"] = {
            **budget_meta, "thinking": thinking if thinking is not None else endpoint.thinking,
            "reasoning_budget_tokens": reasoning_budget_tokens, "transport_attempts": attempt,
            "retry_count": max(0, attempt - 1), "usage": usage,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, Mapping) else None,
            "reasoning_tokens": reasoning_tokens,
            "visible_output_tokens": max(0, completion_tokens - reasoning_tokens),
        }
        return result


def _visible_response_retry_reason(message: Any, finish_reason: Any, selected: Optional[str]) -> Optional[str]:
    if selected:
        return None
    content = str(message.get("content") or "") if isinstance(message, Mapping) else ""
    if str(finish_reason or "").lower() == "length":
        return "length"
    if not content.strip():
        if isinstance(message, Mapping) and any(message.get(key) for key in ("reasoning", "reasoning_content", "reasoning_details")):
            return "reasoning_only"
        return "empty_visible_content"
    return None


def normalize_logprobs(response: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize OpenAI chat content and legacy parallel-array logprobs."""
    choices = response.get("choices") if isinstance(response, Mapping) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
    raw = choice.get("logprobs") if isinstance(choice, Mapping) else None
    tokens: List[Dict[str, Any]] = []
    source = "none"
    if isinstance(raw, Mapping) and isinstance(raw.get("content"), list):
        source = "openai_chat_content"
        for item in raw["content"]:
            if not isinstance(item, Mapping):
                continue
            tops = item.get("top_logprobs") if isinstance(item.get("top_logprobs"), list) else []
            tokens.append({"token": item.get("token"), "logprob": item.get("logprob"),
                           "top_logprobs": [{"token": x.get("token"), "logprob": x.get("logprob")}
                                             for x in tops if isinstance(x, Mapping)]})
    elif isinstance(raw, Mapping) and isinstance(raw.get("tokens"), list):
        source = "legacy_arrays"
        values = raw.get("token_logprobs") if isinstance(raw.get("token_logprobs"), list) else []
        tops = raw.get("top_logprobs") if isinstance(raw.get("top_logprobs"), list) else []
        for index, token in enumerate(raw["tokens"]):
            top = tops[index] if index < len(tops) and isinstance(tops[index], Mapping) else {}
            tokens.append({"token": token, "logprob": values[index] if index < len(values) else None,
                           "top_logprobs": [{"token": key, "logprob": value} for key, value in top.items()]})
    return {"format": source, "tokens": tokens, "raw": raw}


def _label_probabilities(labels: Sequence[str], evidence: Mapping[str, Any]) -> Dict[str, float]:
    import math
    scores: Dict[str, float] = {}
    for item in evidence.get("tokens", []):
        candidates = list(item.get("top_logprobs", []))
        candidates.append({"token": item.get("token"), "logprob": item.get("logprob")})
        for candidate in candidates:
            token = str(candidate.get("token") or "").strip()
            value = candidate.get("logprob")
            if token in labels and isinstance(value, (int, float)):
                scores[token] = max(scores.get(token, float("-inf")), float(value))
        if scores:
            break
    if not scores:
        return {}
    weights = {label: math.exp(value) for label, value in scores.items()}
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def _is_explicit_logprobs_rejection(status_code: int, detail: str) -> bool:
    if int(status_code) not in {400, 404, 422}:
        return False
    lowered = str(detail or "").lower()
    feature = "logprob" in lowered or "top_logprobs" in lowered
    rejection = any(marker in lowered for marker in ("unsupported", "not supported", "unknown", "unrecognized", "not allowed", "invalid parameter"))
    return feature and rejection


@contextmanager
def _endpoint_request_lock(endpoint: ModelEndpoint):
    """Optionally serialize endpoint requests across local harness processes."""
    if not endpoint.request_lock_path:
        yield
        return
    with exclusive_file_lock(endpoint.request_lock_path):
        yield


def _definitive_failure_class(status_code: int, detail: str) -> Optional[str]:
    if int(status_code) in {401, 403}:
        return "authentication"
    if _is_context_limit_detail(detail):
        return "context_limit"
    lowered = str(detail or "").lower()
    quota_markers = ("quota exhausted", "quota_exhausted", "insufficient_quota", "billing quota", "credit balance")
    if any(marker in lowered for marker in quota_markers):
        return "quota_exhausted"
    return None


def _is_context_limit_detail(detail: str) -> bool:
    lowered = str(detail or "").lower()
    return any(
        marker in lowered
        for marker in (
            "context window",
            "input exceeds",
            "maximum context",
            "context length",
            "too many tokens",
        )
    )


def _provider_family(endpoint: ModelEndpoint) -> str:
    provider = str(endpoint.provider or "").lower()
    base = str(endpoint.base_url or "").lower()
    if "openrouter" in provider or "openrouter.ai" in base:
        return "openrouter"
    if "anthropic" in provider:
        return "anthropic"
    if "deepseek" in provider or "api.deepseek.com" in base:
        return "deepseek"
    return "openai-compatible"


def reasoning_reserve_tokens(thinking: Optional[str], endpoint: Optional[ModelEndpoint] = None) -> int:
    """Token allowance reserved for hidden chain-of-thought before visible JSON."""
    value = str(thinking if thinking is not None else (endpoint.thinking if endpoint is not None else "") or "").lower()
    if value in {"", "none", "disabled", "off", "false"}:
        return 0
    if endpoint is not None and endpoint.thinking_budget_tokens:
        return max(0, int(endpoint.thinking_budget_tokens))
    return int(REASONING_RESERVE_TOKENS.get(value, 8_192))


def json_completion_plan(
    *,
    visible_tokens: int,
    thinking: Optional[str],
    endpoint: ModelEndpoint,
    reasoning_budget_tokens: Optional[int] = None,
) -> List[Tuple[int, Optional[str]]]:
    """Budget attempts as thinking+JSON, then a larger ceiling, then an empty-content retry."""
    ceiling = min(int(endpoint.max_output_tokens or ENDPOINT_OUTPUT_CEILING_TOKENS), ENDPOINT_OUTPUT_CEILING_TOKENS)
    visible = max(1, int(visible_tokens))
    reserve = (
        max(0, int(reasoning_budget_tokens))
        if reasoning_budget_tokens is not None
        else reasoning_reserve_tokens(thinking, endpoint)
    )
    primary = min(ceiling, visible + reserve)
    scaled = min(ceiling, max(primary * 2, visible + (2 * reserve), visible + 16_384))
    plan: List[Tuple[int, Optional[str]]] = [(primary, thinking)]
    if scaled > primary:
        plan.append((scaled, thinking))
    plan.append(plan[-1])
    return plan


def _deepseek_reasoning_effort(thinking: str) -> Optional[str]:
    value = thinking.lower()
    if value in {"none", "disabled", "off", "false"}:
        return None
    if value == "low":
        return "low"
    if value == "max":
        return "max"
    return "high"


def _thinking_payload(endpoint: ModelEndpoint) -> Dict[str, Any]:
    if not endpoint.thinking:
        return {}
    family = _provider_family(endpoint)
    thinking = endpoint.thinking.lower()
    # OpenAI-compatible APIs do not share a standard value for disabling
    # reasoning. Omitting provider-specific controls is the portable way to
    # disable it (some endpoints reject reasoning_effort="none").
    if family != "deepseek" and thinking in {"none", "disabled", "off", "false"}:
        return {}
    if family == "openrouter":
        if thinking in {"enabled", "on", "true"}:
            reasoning: Dict[str, Any] = {"enabled": True}
        else:
            reasoning = {"effort": endpoint.thinking}
        if endpoint.thinking_budget_tokens:
            reasoning["max_tokens"] = endpoint.thinking_budget_tokens
        return {"reasoning": reasoning}
    if family == "anthropic":
        budget = endpoint.thinking_budget_tokens or 1024
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    if family == "deepseek":
        if thinking in {"none", "disabled", "off", "false"}:
            return {"thinking": {"type": "disabled"}}
        payload: Dict[str, Any] = {"thinking": {"type": "enabled"}}
        effort = _deepseek_reasoning_effort(thinking)
        if effort:
            payload["reasoning_effort"] = effort
        return payload
    if thinking in {"enabled", "on", "true"}:
        return {}
    return {"reasoning_effort": endpoint.thinking}


def _drop_incompatible_reasoning_keys(payload: Dict[str, Any], endpoint: ModelEndpoint) -> None:
    """Keep OpenRouter `reasoning` and DeepSeek `thinking` dialects from mixing."""
    family = _provider_family(endpoint)
    thinking_obj = payload.get("thinking")
    is_deepseek_switch = isinstance(thinking_obj, dict) and "type" in thinking_obj
    if family == "deepseek":
        payload.pop("reasoning", None)
    elif family == "openrouter" and is_deepseek_switch:
        payload.pop("thinking", None)
    elif family == "anthropic":
        payload.pop("reasoning", None)
        payload.pop("reasoning_effort", None)


def _ensure_json_instruction(system: str) -> str:
    if "json" in str(system or "").lower():
        return system
    return str(system or "").rstrip() + "\nReturn a single json object."


def _usage_reasoning_tokens(usage: Mapping[str, Any]) -> Tuple[int, int]:
    details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), Mapping) else {}
    reasoning_tokens = int(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return reasoning_tokens, completion_tokens


def _looks_like_truncated_json(text: str, parse_error: Optional[str]) -> bool:
    stripped = str(text or "").strip()
    if not stripped.startswith("{"):
        return False
    err = str(parse_error or "").lower()
    if "unterminated" in err or "delimiter" in err:
        return True
    return not stripped.rstrip().endswith("}")


def _json_output_retry_reason(
    message: Mapping[str, Any],
    finish_reason: Any,
    content: str,
    usage: Mapping[str, Any],
    *,
    requested_max_tokens: int,
    parse_error: Optional[str],
    parse_ok: bool,
) -> Optional[str]:
    if parse_ok:
        return None
    if str(finish_reason or "").lower() == "length":
        return "length"
    visible = str(content or "")
    if not visible.strip():
        if any(message.get(key) for key in ("reasoning", "reasoning_content", "reasoning_details")):
            return "reasoning_only"
        return "empty_visible_content"
    reasoning_tokens, completion_tokens = _usage_reasoning_tokens(usage)
    visible_tokens = max(0, completion_tokens - reasoning_tokens) if completion_tokens else 0
    if reasoning_tokens and requested_max_tokens and reasoning_tokens >= int(0.85 * requested_max_tokens):
        return "reasoning_tokens_consumed_output"
    if reasoning_tokens and visible_tokens < 32:
        return "reasoning_tokens_consumed_output"
    if _looks_like_truncated_json(visible, parse_error):
        return "truncated_json"
    return None


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_ssl_error(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", None)
    return isinstance(exc, ssl.SSLError) or isinstance(reason, ssl.SSLError)


_TRANSIENT_TRANSPORT_ERRORS = (
    TimeoutError,
    ssl.SSLError,
    ConnectionError,
    ConnectionResetError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    socket.timeout,
    urllib.error.URLError,
)


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text
