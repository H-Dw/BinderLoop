#!/usr/bin/env python3
"""Regression tests for per-call reasoning and terminal context errors."""

import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.llm import (
    LLMConfigError,
    LLMHTTPError,
    LLMDefinitiveError,
    LLMSettings,
    LLMTransportError,
    ModelEndpoint,
    ModelEndpointCapabilities,
    OpenAICompatibleClient,
    _thinking_payload,
    json_completion_plan,
    normalize_logprobs,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _client(*, lock_path=None, retries=2):
    endpoint = ModelEndpoint(
        name="test",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        provider="sui-xiang",
        thinking="high",
        extra_body={"reasoning": {"enabled": True}},
        request_lock_path=lock_path,
        max_retries=retries,
        retry_backoff_seconds=0,
    )
    return OpenAICompatibleClient(
        LLMSettings(
            default_model="test",
            endpoints={"test": endpoint},
            enabled=True,
        )
    )


class LLMRequestControlTest(unittest.TestCase):
    def test_low_reasoning_replaces_generic_reasoning_object(self):
        with tempfile.TemporaryDirectory() as directory:
            client = _client(lock_path=str(Path(directory) / "endpoint.lock"))
            response = {
                "choices": [{"message": {"content": "{\"ok\": true}"}}],
                "usage": {},
            }
            with patch(
                "binderloop.llm.urllib.request.urlopen",
                return_value=_Response(response),
            ) as mocked:
                result = client.chat_json(
                    system="Return JSON.",
                    user={"task": "test"},
                    thinking="low",
                    max_tokens=32,
                )
            self.assertTrue(result["ok"])
            request = mocked.call_args.args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["reasoning_effort"], "low")
            self.assertNotIn("reasoning", payload)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(payload["max_tokens"], 32 + 4096)

    def test_reasoning_override_cleans_conflicts_and_forwards_budget(self):
        client = _client(retries=1)
        response = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}
        with patch("binderloop.llm.urllib.request.urlopen", return_value=_Response(response)) as mocked:
            client.create_chat_completion(
                messages=[{"role": "user", "content": "x"}], thinking="low",
                reasoning_budget_tokens=321,
                extra_body={"reasoning": {"enabled": True}, "thinking": {"type": "enabled"}},
            )
        payload = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertNotIn("reasoning", payload)
        self.assertNotIn("thinking", payload)

    def test_closed_label_retries_reasoning_only_at_512(self):
        client = _client(retries=1)
        responses = [
            _Response({"choices": [{"message": {"content": "", "reasoning_content": "hidden"}, "finish_reason": "length"}], "usage": {"completion_tokens": 64}}),
            _Response({"choices": [{"message": {"content": "A"}, "finish_reason": "stop", "logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": [{"token": "A", "logprob": -0.1}]}]}}], "usage": {"completion_tokens": 1}}),
        ]
        with patch("binderloop.llm.urllib.request.urlopen", side_effect=responses) as mocked:
            result = client.chat_label_distribution(system="choose", user={}, labels=["A", "B"] )
        self.assertEqual(result["status"], "available")
        self.assertEqual(mocked.call_count, 2)
        payloads = [json.loads(call.args[0].data) for call in mocked.call_args_list]
        self.assertEqual([item["max_tokens"] for item in payloads], [64, 512])
        self.assertNotIn("reasoning_effort", payloads[0])
        self.assertEqual(result["request_artifact"]["retry_count"], 1)

    def test_closed_label_returns_unavailable_after_1024(self):
        client = _client(retries=1)
        response = _Response({"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {}})
        with patch("binderloop.llm.urllib.request.urlopen", return_value=response) as mocked:
            result = client.chat_label_distribution(system="choose", user={}, labels=["A", "B"] )
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(result["request_artifact"]["attempts"][-1]["requested_completion_tokens"], 1024)

    def test_endpoint_output_ceiling_is_65536(self):
        client = _client(retries=1)
        client.resolved_endpoint.max_output_tokens = 100_000
        budget = client.effective_completion_budget(100_000)
        self.assertEqual(budget["effective_completion_tokens"], 65_536)
        self.assertEqual(budget["completion_clamp_reason"], "endpoint_output_ceiling")

    def test_context_limit_502_is_not_retried(self):
        client = _client(retries=3)
        detail = json.dumps({
            "error": {
                "message": "Your input exceeds the context window of this model."
            }
        }).encode("utf-8")
        error = urllib.error.HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=502,
            msg="Bad Gateway",
            hdrs=None,
            fp=io.BytesIO(detail),
        )
        with patch(
            "binderloop.llm.urllib.request.urlopen",
            side_effect=error,
        ) as mocked:
            with self.assertRaises(LLMHTTPError):
                client.create_chat_completion(
                    messages=[{"role": "user", "content": "large"}],
                )
        self.assertEqual(mocked.call_count, 1)

    def test_quota_exhausted_429_is_definitive_without_retry(self):
        client = _client(retries=3)
        error = urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions", 429, "Too Many Requests", None,
            io.BytesIO(b'{"error":{"message":"quota exhausted"}}'),
        )
        with patch("binderloop.llm.urllib.request.urlopen", side_effect=error) as mocked:
            with self.assertRaises(LLMDefinitiveError) as raised:
                client.create_chat_completion(messages=[{"role": "user", "content": "x"}])
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(raised.exception.failure_class, "quota_exhausted")
        self.assertFalse(raised.exception.retryable)

    def test_auth_403_is_definitive_without_retry(self):
        client = _client(retries=3)
        error = urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions", 403, "Forbidden", None,
            io.BytesIO(b'{"error":{"message":"forbidden"}}'),
        )
        with patch("binderloop.llm.urllib.request.urlopen", side_effect=error) as mocked:
            with self.assertRaises(LLMDefinitiveError) as raised:
                client.create_chat_completion(messages=[{"role": "user", "content": "x"}])
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(raised.exception.failure_class, "authentication")

    def test_ordinary_rate_limit_remains_recoverable(self):
        client = _client(retries=2)
        error = urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions", 429, "Too Many Requests", None,
            io.BytesIO(b'{"error":{"message":"requests per minute exceeded"}}'),
        )
        with patch("binderloop.llm.urllib.request.urlopen", side_effect=error) as mocked:
            with self.assertRaises(LLMTransportError) as raised:
                client.create_chat_completion(messages=[{"role": "user", "content": "x"}])
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(raised.exception.failure_class, "rate_limit")
        self.assertTrue(raised.exception.recoverable_stop)

    def test_preflight_uses_low_reasoning_for_reasoning_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            client = _client(lock_path=str(Path(directory) / "endpoint.lock"))

            def _urlopen(request, timeout=None):
                payload = json.loads(request.data.decode("utf-8"))
                user = json.loads(payload["messages"][1]["content"])
                nonce = user["nonce"]
                return _Response({
                    "choices": [{"message": {"content": json.dumps({"ok": True, "nonce": nonce, "message": "live"})}}],
                })

            with patch("binderloop.llm.urllib.request.urlopen", side_effect=_urlopen) as mocked:
                client.preflight(max_tokens=32, timeout_seconds=5)
            request = mocked.call_args.args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(payload["reasoning_effort"], "low")
            self.assertNotIn("reasoning", payload)
            self.assertNotIn("thinking", payload)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertGreater(payload["max_tokens"], 32)

    def test_preflight_succeeds_with_nonce_echo(self):
        client = _client(retries=1)

        def _urlopen(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            user = json.loads(payload["messages"][1]["content"])
            nonce = user["nonce"]
            return _Response({
                "choices": [{"message": {"content": json.dumps({"ok": True, "nonce": nonce, "message": "live"})}}],
            })

        with patch("binderloop.llm.urllib.request.urlopen", side_effect=_urlopen):
            result = client.preflight(max_tokens=32, timeout_seconds=5)
        self.assertTrue(result["llm_used"])
        self.assertEqual(result["endpoint_key"], "test")
        self.assertEqual(result["response"]["ok"], True)

    def test_preflight_empty_content_reports_provider_diagnostics(self):
        client = _client(retries=1)
        response = {
            "choices": [{
                "message": {"content": "", "reasoning_content": "internal"},
                "finish_reason": "length",
            }],
        }
        with patch("binderloop.llm.urllib.request.urlopen", return_value=_Response(response)):
            with self.assertRaisesRegex(
                LLMConfigError,
                r"finish_reason='length'.*content_length=0.*reasoning_content.*max_tokens=32",
            ):
                client.preflight(max_tokens=32, timeout_seconds=5)

    def test_preflight_rejects_unavailable_endpoint(self):
        client = OpenAICompatibleClient(
            LLMSettings(default_model="missing", endpoints={}, enabled=True)
        )
        with self.assertRaises(LLMConfigError):
            client.preflight()

    def test_preflight_rejects_bad_json_response(self):
        client = _client(retries=1)
        with patch(
            "binderloop.llm.urllib.request.urlopen",
            return_value=_Response({"choices": [{"message": {"content": "not-json"}}]}),
        ):
            with self.assertRaises(LLMConfigError):
                client.preflight()

    def test_preflight_propagates_transport_failure(self):
        client = _client(retries=1)
        with patch(
            "binderloop.llm.urllib.request.urlopen",
            side_effect=urllib.error.URLError("network down"),
        ):
            with self.assertRaises(LLMTransportError):
                client.preflight()

    def _write_config(self, directory, payload):
        path = Path(directory) / "llm.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_from_json_requires_explicit_default_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, {
                "enabled": True,
                "endpoints": {"one": {"base_url": "https://example.invalid/v1", "model": "provider-model"}},
            })
            with self.assertRaisesRegex(LLMConfigError, "explicit nonempty default_model"):
                OpenAICompatibleClient.from_json(path)

    def test_from_json_default_and_validated_override_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(directory, {
                "enabled": True,
                "default_model": "one",
                "endpoints": {
                    "one": {"base_url": "https://one.invalid/v1", "model": "model-one", "provider": "p1"},
                    "two": {"base_url": "https://two.invalid/v1", "model": "model-two", "provider": "p2", "thinking": "high"},
                },
            })
            client = OpenAICompatibleClient.from_json(path)
            self.assertEqual(client.resolved_endpoint_key, "one")
            self.assertEqual(client.resolved_endpoint.model, "model-one")
            client.configure_default(model_key="two", thinking="low")
            self.assertEqual(client.resolved_endpoint_key, "two")
            self.assertEqual(client.resolved_endpoint.model, "model-two")
            self.assertEqual(client.resolved_endpoint.thinking, "low")
            with self.assertRaisesRegex(LLMConfigError, "unknown model endpoint"):
                client.configure_default(model_key="missing")

    def test_deepseek_payload_uses_official_thinking_switch(self):
        endpoint = ModelEndpoint(
            name="deepseek",
            base_url="https://api.deepseek.com",
            api_key="test-key",
            provider="deepseek",
            model="deepseek-v4-pro",
            thinking="high",
            extra_body={"reasoning": {"enabled": True}},
        )
        client = OpenAICompatibleClient(LLMSettings(default_model="deepseek", endpoints={"deepseek": endpoint}, enabled=True))
        response = {"choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}], "usage": {}}
        with patch("binderloop.llm.urllib.request.urlopen", return_value=_Response(response)) as mocked:
            client.create_chat_completion(messages=[{"role": "user", "content": "x"}])
        payload = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("reasoning", payload)

    def test_deepseek_thinking_off_sends_disabled_switch(self):
        endpoint = ModelEndpoint(
            name="deepseek",
            base_url="https://api.deepseek.com",
            api_key="test-key",
            provider="deepseek",
            thinking="high",
        )
        self.assertEqual(_thinking_payload(endpoint), {"thinking": {"type": "enabled"}, "reasoning_effort": "high"})
        off = ModelEndpoint(name="deepseek", base_url="https://api.deepseek.com", api_key="k", provider="deepseek", thinking="off")
        self.assertEqual(_thinking_payload(off), {"thinking": {"type": "disabled"}})

    def test_openrouter_reasoning_format_is_unchanged(self):
        endpoint = ModelEndpoint(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            provider="openrouter",
            thinking="enabled",
            extra_body={"reasoning": {"enabled": True}},
        )
        client = OpenAICompatibleClient(LLMSettings(default_model="openrouter", endpoints={"openrouter": endpoint}, enabled=True))
        response = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}
        with patch("binderloop.llm.urllib.request.urlopen", return_value=_Response(response)) as mocked:
            client.create_chat_completion(messages=[{"role": "user", "content": "x"}])
        payload = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(payload["reasoning"], {"enabled": True})
        self.assertNotIn("reasoning_effort", payload)

    def test_chat_json_retries_empty_content_and_never_parses_reasoning(self):
        client = _client(retries=1)
        responses = [
            _Response({"choices": [{"message": {"content": "", "reasoning_content": "hidden chain"}, "finish_reason": "length"}],
                       "usage": {"completion_tokens": 64, "completion_tokens_details": {"reasoning_tokens": 64}}}),
            _Response({"choices": [{"message": {"content": "{\"ok\": true, \"answer\": 1}"}, "finish_reason": "stop"}],
                       "usage": {"completion_tokens": 12, "completion_tokens_details": {"reasoning_tokens": 4}}}),
        ]
        with patch("binderloop.llm.urllib.request.urlopen", side_effect=responses) as mocked:
            result = client.chat_json(system="Return JSON.", user={"task": "t"}, thinking="high", max_tokens=32)
        self.assertEqual(result, {"ok": True, "answer": 1})
        self.assertEqual(mocked.call_count, 2)
        payloads = [json.loads(call.args[0].data) for call in mocked.call_args_list]
        self.assertEqual(payloads[0]["response_format"], {"type": "json_object"})
        self.assertGreater(payloads[1]["max_tokens"], payloads[0]["max_tokens"])
        self.assertFalse(client.last_json_call["attempts"][0]["used_reasoning_content"])
        self.assertEqual(client.last_json_call["attempts"][0]["retry_reason"], "length")

    def test_json_completion_plan_reserves_thinking_plus_visible_tokens(self):
        endpoint = ModelEndpoint(name="x", base_url="https://x", api_key="k", thinking="high", max_output_tokens=65536)
        plan = json_completion_plan(visible_tokens=8000, thinking="high", endpoint=endpoint)
        self.assertEqual(plan[0][0], 8000 + 16384)
        self.assertGreater(plan[1][0], plan[0][0])
        self.assertEqual(plan[-1], plan[-2])

    def test_from_json_rejects_shapes_unknown_fields_and_bad_controls(self):
        bad_payloads = [
            [],
            {"enabled": "true", "endpoints": {}},
            {"enabled": False, "endpoints": []},
            {"enabled": False, "endpoints": {"": {}}},
            {"enabled": False, "endpoints": {"one": []}},
            {"enabled": False, "endpoints": {"one": {"base_url": "", "model": "m"}}},
            {"enabled": False, "endpoints": {"one": {"base_url": "https://x", "model": "m", "timeout_seconds": 0}}},
            {"enabled": False, "endpoints": {"one": {"base_url": "https://x", "model": "m", "retry_backoff_seconds": -1}}},
            {"enabled": False, "endpoints": {"one": {"base_url": "https://x", "model": "m", "default_headers": []}}},
            {"enabled": False, "endpoints": {"one": {"base_url": "https://x", "model": "m", "unknown": 1}}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, payload in enumerate(bad_payloads):
                path = Path(directory) / f"bad-{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(LLMConfigError):
                    OpenAICompatibleClient.from_json(path)


class LLMLogprobsTest(unittest.TestCase):
    def test_config_validates_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.json"
            good.write_text(json.dumps({
                "enabled": True, "default_model": "one",
                "endpoints": {"one": {"base_url": "https://x", "model": "m",
                    "capabilities": {"logprobs": "required", "top_logprobs_max": 5}}},
            }))
            client = OpenAICompatibleClient.from_json(good)
            self.assertEqual(client.resolved_endpoint.capabilities.logprobs, "required")
            for index, capabilities in enumerate((
                {"logprobs": "sometimes"}, {"top_logprobs_max": 0},
                {"top_logprobs_max": True}, {"unknown": 1},
            )):
                bad = Path(directory) / f"bad-cap-{index}.json"
                bad.write_text(json.dumps({"enabled": False, "endpoints": {
                    "one": {"base_url": "https://x", "model": "m", "capabilities": capabilities}}}))
                with self.subTest(capabilities=capabilities), self.assertRaises(LLMConfigError):
                    OpenAICompatibleClient.from_json(bad)

    def test_normalize_openai_chat_content(self):
        result = normalize_logprobs({"choices": [{"logprobs": {"content": [{
            "token": "yes", "logprob": -0.1,
            "top_logprobs": [{"token": "yes", "logprob": -0.1}, {"token": "no", "logprob": -2.0}],
        }]}}]})
        self.assertEqual(result["format"], "openai_chat_content")
        self.assertEqual(result["tokens"][0]["top_logprobs"][1]["token"], "no")

    def test_normalize_legacy_arrays(self):
        result = normalize_logprobs({"choices": [{"logprobs": {
            "tokens": ["A"], "token_logprobs": [-0.2], "top_logprobs": [{"A": -0.2, "B": -1.7}],
        }}]})
        self.assertEqual(result["format"], "legacy_arrays")
        self.assertEqual(result["tokens"][0]["logprob"], -0.2)

    def test_disabled_probe_never_requests_and_is_not_cached_as_live(self):
        client = _client(retries=1)
        client.resolved_endpoint.capabilities = ModelEndpointCapabilities(logprobs="disabled")
        with patch("binderloop.llm.urllib.request.urlopen") as mocked:
            result = client.probe_logprobs()
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["source"], "configuration")
        mocked.assert_not_called()

    def test_probe_supported_and_cached(self):
        client = _client(retries=1)
        response = {"choices": [{"message": {"content": "OK"}, "logprobs": {"content": [
            {"token": "OK", "logprob": -0.01, "top_logprobs": [{"token": "OK", "logprob": -0.01}]}
        ]}}]}
        with patch("binderloop.llm.urllib.request.urlopen", return_value=_Response(response)) as mocked:
            first = client.probe_logprobs()
            second = client.probe_logprobs()
        self.assertEqual(first["status"], "supported")
        self.assertEqual(second["status"], "supported")
        self.assertEqual(mocked.call_count, 1)

    def test_probe_classification_is_conservative(self):
        for detail, expected in (("unknown parameter logprobs", "unsupported"), ("bad request", "indeterminate")):
            client = _client(retries=1)
            error = urllib.error.HTTPError("https://x", 400, "bad", None, io.BytesIO(detail.encode()))
            with patch("binderloop.llm.urllib.request.urlopen", side_effect=error):
                result = client.probe_logprobs()
            self.assertEqual(result["status"], expected)

    def test_required_mode_is_exposed_for_caller_decision(self):
        client = _client(retries=1)
        client.resolved_endpoint.capabilities = ModelEndpointCapabilities(logprobs="required")
        response = {"choices": [{"message": {"content": "OK"}}]}
        with patch("binderloop.llm.urllib.request.urlopen", return_value=_Response(response)):
            result = client.probe_logprobs()
        self.assertEqual(result["status"], "indeterminate")
        self.assertEqual(result["mode"], "required")

    def test_label_distribution_returns_full_evidence_and_caps_request(self):
        client = _client(retries=1)
        client.resolved_endpoint.capabilities = ModelEndpointCapabilities(logprobs="auto", top_logprobs_max=2)
        response = {"id": "complete-envelope", "choices": [{"message": {"content": "A"}, "logprobs": {
            "content": [{"token": "A", "logprob": -0.1, "top_logprobs": [
                {"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -1.1}]}]}}]}
        with patch("binderloop.llm.urllib.request.urlopen", return_value=_Response(response)) as mocked:
            result = client.chat_label_distribution(system="Choose.", user={"x": 1}, labels=["A", "B", "C"])
        payload = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(payload["top_logprobs"], 2)
        self.assertEqual(result["label"], "A")
        self.assertAlmostEqual(sum(result["distribution"].values()), 1.0)
        self.assertEqual(result["response"]["id"], "complete-envelope")
        self.assertEqual(result["evidence"]["format"], "openai_chat_content")


if __name__ == "__main__":
    unittest.main(verbosity=2)
