"""Minimal, dependency-free DeepSeek Chat Completions client.

The client deliberately exposes no web-search or function tools.  It sends only
the per-run document bundle assembled by :mod:`deepseek_runner` and never logs
the API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import socket
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error, parse, request


class DeepSeekAPIError(RuntimeError):
    """A classified provider or transport failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        retryable: bool = False,
        content_filter: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable
        self.content_filter = content_filter


@dataclass(frozen=True)
class APIConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    thinking: bool = True
    reasoning_effort: str = "high"
    max_tokens: int = 32_768
    timeout_seconds: float = 900.0
    transport_retries: int = 3
    backoff_base_seconds: float = 2.0
    json_mode: bool = True
    credential_source: str = "environment"
    endpoint_key: str | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("DeepSeek API key is empty")
        parsed = parse.urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not embed credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.reasoning_effort not in {
            "none", "minimal", "low", "medium", "high", "xhigh", "max"
        }:
            raise ValueError("unsupported reasoning_effort")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.transport_retries < 0:
            raise ValueError("transport_retries must be non-negative")
        if self.backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds must be non-negative")

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    @property
    def public_endpoint(self) -> str:
        parsed = parse.urlparse(self.endpoint)
        return parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


@dataclass(frozen=True)
class ChatResponse:
    content: str
    response_id: str | None
    model: str | None
    finish_reason: str | None
    usage: Mapping[str, Any]
    public_response: Mapping[str, Any]
    transport_attempts: int


def _error_details(raw: bytes) -> tuple[str | None, str]:
    text = raw.decode("utf-8", errors="replace")[:4000]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, text.strip() or "empty provider error"
    if isinstance(payload, Mapping):
        nested = payload.get("error")
        if isinstance(nested, Mapping):
            code = nested.get("code", nested.get("type"))
            message = nested.get("message", text)
            return (None if code is None else str(code), str(message))
        code = payload.get("code")
        message = payload.get("message", text)
        return (None if code is None else str(code), str(message))
    return None, text


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                pieces.append(str(item["text"]))
        return "".join(pieces)
    return ""


class DeepSeekClient:
    """OpenAI-compatible `/chat/completions` client with bounded retries."""

    def __init__(
        self,
        config: APIConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._sleep = sleep

    def _payload(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "stream": False,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.config.thinking:
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = self.config.reasoning_effort
        else:
            payload["thinking"] = {"type": "disabled"}
        return payload

    def _once(self, messages: Sequence[Mapping[str, Any]]) -> ChatResponse:
        encoded = json.dumps(
            self._payload(messages), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        req = request.Request(
            self.config.endpoint,
            data=encoded,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "binder-harness-deepseek-hotspot/1.0",
            },
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except error.HTTPError as exc:
            raw = exc.read()
            code, message = _error_details(raw)
            normalized = f"{code or ''} {message}".lower()
            is_filter = any(
                marker in normalized
                for marker in ("content_filter", "content filter", "safety", "policy")
            )
            raise DeepSeekAPIError(
                f"DeepSeek HTTP {exc.code}: {message}",
                status_code=exc.code,
                code=code,
                retryable=exc.code in {408, 409, 425, 429} or 500 <= exc.code <= 599,
                content_filter=is_filter,
            ) from exc
        except (error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            raise DeepSeekAPIError(
                f"DeepSeek transport error: {exc}", retryable=True
            ) from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeepSeekAPIError(
                "DeepSeek returned a non-JSON HTTP response", retryable=True
            ) from exc
        if not isinstance(payload, Mapping):
            raise DeepSeekAPIError("DeepSeek response must be a JSON object", retryable=True)
        if isinstance(payload.get("error"), Mapping):
            nested = payload["error"]
            code = None if nested.get("code") is None else str(nested.get("code"))
            message = str(nested.get("message", "provider error"))
            normalized = f"{code or ''} {message}".lower()
            raise DeepSeekAPIError(
                f"DeepSeek API error: {message}",
                code=code,
                retryable=any(x in normalized for x in ("rate", "timeout", "overload")),
                content_filter=any(
                    x in normalized for x in ("content_filter", "safety", "policy")
                ),
            )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise DeepSeekAPIError("DeepSeek response has no choices", retryable=True)
        choice = choices[0]
        finish_reason = (
            None if choice.get("finish_reason") is None else str(choice.get("finish_reason"))
        )
        if finish_reason == "content_filter":
            raise DeepSeekAPIError(
                "DeepSeek response was blocked by content_filter",
                code="content_filter",
                content_filter=True,
            )
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise DeepSeekAPIError("DeepSeek choice has no message", retryable=True)
        content = _content_text(message.get("content"))
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        public_response = {
            "id": payload.get("id"),
            "model": payload.get("model"),
            "finish_reason": finish_reason,
            "content": content,
            "usage": dict(usage),
        }
        return ChatResponse(
            content=content,
            response_id=None if payload.get("id") is None else str(payload.get("id")),
            model=None if payload.get("model") is None else str(payload.get("model")),
            finish_reason=finish_reason,
            usage=dict(usage),
            public_response=public_response,
            transport_attempts=1,
        )

    def chat(self, messages: Sequence[Mapping[str, Any]]) -> ChatResponse:
        last_error: DeepSeekAPIError | None = None
        for attempt in range(self.config.transport_retries + 1):
            try:
                response = self._once(messages)
                return ChatResponse(
                    **{
                        **response.__dict__,
                        "transport_attempts": attempt + 1,
                    }
                )
            except DeepSeekAPIError as exc:
                last_error = exc
                if (
                    exc.content_filter
                    or not exc.retryable
                    or attempt >= self.config.transport_retries
                ):
                    raise
                self._sleep(self.config.backoff_base_seconds * (2**attempt))
        assert last_error is not None
        raise last_error
