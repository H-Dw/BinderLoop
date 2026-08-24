#!/usr/bin/env python3

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.communication import AgentMessage, MessageBus, project_execution_message
from binderloop.resume import (
    ArtifactDigestCache,
    artifact_record,
    artifacts_match,
    atomic_write_text,
    file_sha256,
)


def _message(index: int) -> AgentMessage:
    return AgentMessage(
        sender="writer",
        recipient="reader",
        message_type="status",
        content={"index": index},
        correlation_id=f"message-{index}",
    )


def _line(message: AgentMessage) -> str:
    return json.dumps(message.to_dict(), ensure_ascii=False) + "\n"


class MessageBusCacheTests(unittest.TestCase):
    def test_publish_initializes_and_synchronizes_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "messages.jsonl"
            path.write_text(_line(_message(1)), encoding="utf-8")
            bus = MessageBus(path)

            with mock.patch(
                "binderloop.communication.json.loads",
                wraps=json.loads,
            ) as loads:
                bus.publish(_message(2))
                self.assertEqual(loads.call_count, 1)
                self.assertEqual([item.content["index"] for item in bus.read_all()], [1, 2])
                self.assertEqual(loads.call_count, 1)

    def test_reads_once_then_parses_only_external_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "messages.jsonl"
            path.write_text(_line(_message(1)), encoding="utf-8")
            bus = MessageBus(path)
            original_open = Path.open
            read_opens = {"count": 0}

            def counting_open(open_path, *args, **kwargs):
                mode = args[0] if args else kwargs.get("mode", "r")
                if Path(open_path) == path and mode == "rb":
                    read_opens["count"] += 1
                return original_open(open_path, *args, **kwargs)

            original_loads = json.loads
            with (
                mock.patch.object(Path, "open", new=counting_open),
                mock.patch("binderloop.communication.json.loads", wraps=original_loads) as loads,
            ):
                self.assertEqual([item.content["index"] for item in bus.read_all()], [1])
                self.assertEqual([item.content["index"] for item in bus.read_all()], [1])
                self.assertEqual(len(bus.query(sender="writer")), 1)
                self.assertEqual(read_opens["count"], 1)
                self.assertEqual(loads.call_count, 1)

                with original_open(path, "a", encoding="utf-8") as handle:
                    handle.write(_line(_message(2)))
                self.assertEqual([item.content["index"] for item in bus.read_all()], [1, 2])
                self.assertEqual(read_opens["count"], 2)
                self.assertEqual(loads.call_count, 2)

                bus.publish(_message(3))
                self.assertEqual([item.content["index"] for item in bus.read_all()], [1, 2, 3])
                self.assertEqual(loads.call_count, 2)

    def test_truncate_and_replace_rebuild_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "messages.jsonl"
            path.write_text(_line(_message(1)) + _line(_message(2)), encoding="utf-8")
            bus = MessageBus(path)
            self.assertEqual([item.content["index"] for item in bus.read_all()], [1, 2])

            path.write_text(_line(_message(3)), encoding="utf-8")
            self.assertEqual([item.content["index"] for item in bus.read_all()], [3])

            replacement = path.with_suffix(".replacement")
            replacement.write_text(_line(_message(4)), encoding="utf-8")
            os.replace(replacement, path)
            self.assertEqual([item.content["index"] for item in bus.read_all()], [4])

    def test_duplicate_publish_is_suppressed_across_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "messages.jsonl"
            first = MessageBus(path, run_id="run-a")
            message = AgentMessage(
                sender="worker", recipient="all", message_type="status",
                content={"event": "completed", "value": 1}, round_id=2, module="evaluate",
                input_digest="input-digest",
            )
            published = first.publish(message)
            resumed = MessageBus(path, run_id="run-a")
            duplicate = resumed.publish(AgentMessage(
                sender="worker", recipient="all", message_type="status",
                content={"event": "completed", "value": 1}, round_id=2, module="evaluate",
                input_digest="input-digest",
            ))
            self.assertEqual(duplicate.idempotency_key, published.idempotency_key)
            self.assertEqual(len(resumed.read_all()), 1)

    def test_event_type_changes_idempotency_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bus = MessageBus(Path(tmp) / "messages.jsonl", run_id="run-a")
            for event in ("started", "completed"):
                bus.publish(AgentMessage(
                    sender="worker", recipient="all", message_type="status",
                    content={"event": event}, round_id=0, module="evaluate",
                    input_digest="same-input",
                ))
            self.assertEqual(len(bus.read_all()), 2)

    def test_concurrent_publish_keeps_cache_and_file_in_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bus = MessageBus(Path(tmp) / "messages.jsonl")
            self.assertEqual(bus.read_all(), [])
            threads = [
                threading.Thread(target=bus.publish, args=(_message(index),))
                for index in range(20)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            cached = bus.read_all()
            fresh = MessageBus(bus.path).read_all()
            self.assertEqual(len(cached), 20)
            self.assertEqual(
                {item.correlation_id for item in cached},
                {item.correlation_id for item in fresh},
            )


class ExecutionMessageProjectionTests(unittest.TestCase):
    def test_projection_omits_private_execution_payloads_and_paths(self) -> None:
        record = {
            "job_id": "r1_round",
            "backend": "taiji",
            "attempt": 1,
            "status": "failed",
            "error": "failed while reading /secret/local/run.log",
            "run_spec": {"command": "secret"},
            "submit_spec": {"config": "secret"},
            "submission": {"stdout": "secret"},
            "stdout_tail": "secret",
            "stderr_tail": "secret",
            "output_dir": "/secret/output",
            "log_file": "/secret/run.log",
            "taiji_job_id": "123",
        }
        projected = project_execution_message(record)
        self.assertEqual(projected["job_id"], "r1_round")
        self.assertEqual(projected["taiji_job_id"], "123")
        self.assertEqual(projected["error"], "[local execution detail omitted]")
        for key in ("run_spec", "submit_spec", "submission", "stdout_tail", "stderr_tail", "output_dir", "log_file"):
            self.assertNotIn(key, projected)


class ArtifactDigestCacheTests(unittest.TestCase):
    def test_reuses_hash_within_one_cache_but_not_another(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.txt"
            path.write_text("content", encoding="utf-8")
            cache = ArtifactDigestCache()

            with mock.patch("binderloop.resume.file_sha256", wraps=file_sha256) as digest:
                expected = artifact_record(path, cache=cache)
                self.assertEqual(artifact_record(path, cache=cache), expected)
                self.assertTrue(artifacts_match([expected], cache=cache))
                self.assertEqual(digest.call_count, 1)

                other_process_cache = ArtifactDigestCache()
                self.assertEqual(artifact_record(path, cache=other_process_cache), expected)
                self.assertEqual(digest.call_count, 2)

    def test_default_calls_remain_uncached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.txt"
            path.write_text("content", encoding="utf-8")

            with mock.patch("binderloop.resume.file_sha256", wraps=file_sha256) as digest:
                artifact_record(path)
                artifact_record(path)
                self.assertEqual(digest.call_count, 2)

    def test_atomic_write_requires_explicit_invalidate_or_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.txt"
            path.write_text("before", encoding="utf-8")
            cache = ArtifactDigestCache()
            expected = artifact_record(path, cache=cache)

            atomic_write_text(path, "after!")
            self.assertTrue(artifacts_match([expected], cache=cache))

            cache.invalidate(path)
            self.assertFalse(artifacts_match([expected], cache=cache))

            current = cache.update(path)
            self.assertEqual(artifact_record(path, cache=cache), current)
            self.assertTrue(artifacts_match([current], cache=cache))


if __name__ == "__main__":
    unittest.main()
