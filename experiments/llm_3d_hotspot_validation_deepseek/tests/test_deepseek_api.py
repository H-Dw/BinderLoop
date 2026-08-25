from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from experiments.llm_3d_hotspot_validation_deepseek.src.deepseek_api import (
    APIConfig,
    DeepSeekAPIError,
    DeepSeekClient,
)


class _Server:
    def __init__(self, responses: list[tuple[int, dict[str, object]]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                size = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(size))
                owner.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": body,
                    }
                )
                status, payload = owner.responses.pop(0)
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_args: object) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _success(content: str = '{"prediction":{},"process_markdown":"ok"}') -> dict[str, object]:
    return {
        "id": "response-1",
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content, "reasoning_content": "never persist me"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_chat_payload_is_tool_free_and_reasoning_is_not_persisted() -> None:
    with _Server([(200, _success())]) as server:
        config = APIConfig(
            api_key="secret-test-key",
            base_url=server.base_url,
            transport_retries=0,
            timeout_seconds=5,
        )
        response = DeepSeekClient(config).chat(
            [{"role": "user", "content": "Return valid JSON."}]
        )

    sent = server.requests[0]
    body = sent["body"]
    assert sent["path"] == "/chat/completions"
    assert sent["authorization"] == "Bearer secret-test-key"
    assert body["model"] == "deepseek-v4-pro"
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert body["response_format"] == {"type": "json_object"}
    assert "tools" not in body
    assert "reasoning_content" not in response.public_response
    assert "secret-test-key" not in json.dumps(response.public_response)
    assert "secret-test-key" not in repr(config)


def test_retry_is_bounded_and_counted() -> None:
    with _Server(
        [
            (429, {"error": {"code": "rate_limit", "message": "slow down"}}),
            (200, _success()),
        ]
    ) as server:
        sleeps: list[float] = []
        config = APIConfig(
            api_key="x",
            base_url=server.base_url,
            transport_retries=1,
            backoff_base_seconds=0.25,
            timeout_seconds=5,
        )
        response = DeepSeekClient(config, sleep=sleeps.append).chat(
            [{"role": "user", "content": "JSON"}]
        )

    assert len(server.requests) == 2
    assert sleeps == [0.25]
    assert response.transport_attempts == 2


def test_content_filter_is_terminal_even_if_http_status_is_retryable() -> None:
    with _Server(
        [(429, {"error": {"code": "content_filter", "message": "policy block"}})]
    ) as server:
        config = APIConfig(
            api_key="x",
            base_url=server.base_url,
            transport_retries=3,
            timeout_seconds=5,
        )
        with pytest.raises(DeepSeekAPIError) as caught:
            DeepSeekClient(config, sleep=lambda _seconds: None).chat(
                [{"role": "user", "content": "JSON"}]
            )

    assert caught.value.content_filter is True
    assert len(server.requests) == 1


def test_base_url_cannot_embed_credentials() -> None:
    with pytest.raises(ValueError, match="must not embed credentials"):
        APIConfig(api_key="x", base_url="https://user:password@api.deepseek.com")
