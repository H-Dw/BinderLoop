#!/usr/bin/env python3
"""Tests for the durable Harness event journal and optional graph telemetry."""

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.harness import (
    EventJournal,
    HarnessEventType,
    JournalCorruptionError,
    JournalTailError,
)
from binderloop.orchestration.round_graph import RoundGraph


class EventJournalTests(unittest.TestCase):
    def test_append_replay_builds_contiguous_hash_chain(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            journal = EventJournal(Path(temporary_directory) / "events.jsonl", run_id="run-001")
            first = journal.record(HarnessEventType.RUN_STARTED, {"mode": "test"})
            second = journal.record("round.started", {"round_id": 1})

            replayed = journal.replay()

            self.assertEqual([event.sequence for event in replayed.events], [1, 2])
            self.assertEqual(replayed.events[0].event_hash, first.event_hash)
            self.assertEqual(second.previous_hash, first.event_hash)
            self.assertFalse(replayed.truncated_tail_ignored)
            self.assertGreater(replayed.valid_bytes, 0)

    def test_payload_is_detached_and_later_mutation_is_hash_detectable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            journal = EventJournal(Path(temporary_directory) / "events.jsonl", run_id="run-payload")
            source = {"nested": [1]}
            event = journal.record("test.payload", source)
            source["nested"].append(2)

            self.assertEqual({"nested": [1]}, event.payload)
            event.payload["nested"].append(3)
            with self.assertRaisesRegex(ValueError, "event_hash"):
                event.verify_hash()

    def test_concurrent_writers_keep_sequence_contiguous(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            journals = [EventJournal(path, run_id="run-concurrent") for _ in range(2)]
            barrier = threading.Barrier(8)
            errors = []

            def write(index):
                try:
                    barrier.wait(timeout=2)
                    journals[index % 2].record("test.concurrent", {"index": index})
                except Exception as exc:  # pragma: no cover - assertion reports details
                    errors.append(exc)

            threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

            self.assertFalse(errors)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            replayed = journals[0].replay()
            self.assertEqual([event.sequence for event in replayed.events], list(range(1, 9)))
            self.assertEqual({event.payload["index"] for event in replayed.events}, set(range(8)))

    def test_multiple_processes_share_the_journal_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            project_root = Path(__file__).resolve().parents[1]
            writer = (
                "import sys\n"
                "from binderloop.harness import EventJournal\n"
                "journal = EventJournal(sys.argv[1], run_id='run-process')\n"
                "for index in range(4):\n"
                "    journal.record('test.process', {'writer': sys.argv[2], 'index': index})\n"
            )
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", writer, str(path), name],
                    cwd=str(project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for name in ("first", "second")
            ]
            failures = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=15)
                if process.returncode:
                    failures.append((process.returncode, stdout, stderr))

            self.assertFalse(failures)
            replayed = EventJournal(path, run_id="run-process").replay()
            self.assertEqual([event.sequence for event in replayed.events], list(range(1, 9)))
            self.assertEqual({event.payload["writer"] for event in replayed.events}, {"first", "second"})

    def test_content_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            journal = EventJournal(path, run_id="run-tamper")
            journal.record("test.original", {"value": 1})
            value = json.loads(path.read_text(encoding="utf-8"))
            value["payload"]["value"] = 2
            path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(JournalCorruptionError, "event_hash"):
                journal.replay()

    def test_torn_tail_is_reported_and_explicitly_repairable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            journal = EventJournal(path, run_id="run-torn")
            journal.record("test.complete", {"value": 1})
            with path.open("ab") as handle:
                handle.write(b'{"partial":')

            with self.assertRaises(JournalTailError):
                journal.replay()
            tolerant = journal.replay(allow_truncated_tail=True)
            self.assertTrue(tolerant.truncated_tail_ignored)
            self.assertEqual(len(tolerant.events), 1)
            with self.assertRaises(JournalTailError):
                journal.record("test.blocked", {})

            self.assertGreater(journal.repair_truncated_tail(), 0)
            resumed = journal.record("test.resumed", {})
            self.assertEqual(resumed.sequence, 2)


class RoundGraphEventTests(unittest.TestCase):
    def test_graph_records_success_and_failure_without_changing_results(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            journal = EventJournal(
                Path(temporary_directory) / "events.jsonl",
                run_id="run-graph",
            )
            graph = RoundGraph(event_recorder=journal)

            def fail():
                raise LookupError("expected test failure")

            result = graph.run_wave(
                "A",
                {"ok": lambda: 7, "bad": fail},
                event_context={"round_id": 3},
            )

            self.assertEqual(result.results, {"ok": 7})
            self.assertIsInstance(result.errors["bad"], LookupError)
            replayed = journal.replay()
            by_node = {}
            for event in replayed.events:
                by_node.setdefault(event.payload["node"], []).append(event.event_type)
                self.assertEqual(event.payload["round_id"], 3)
                self.assertEqual(event.payload["wave"], "A")
            self.assertEqual(
                by_node["ok"],
                ["graph.node.started", "graph.node.succeeded"],
            )
            self.assertEqual(
                by_node["bad"],
                ["graph.node.started", "graph.node.failed"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
