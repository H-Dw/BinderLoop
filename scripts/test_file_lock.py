#!/usr/bin/env python3
"""Regression tests for the harness's cross-process request lock."""

import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.file_lock import exclusive_file_lock
from binderloop.llm import ModelEndpoint, _endpoint_request_lock


def _lock_worker(lock_path, entered, release):
    with exclusive_file_lock(lock_path, poll_interval_seconds=0.01):
        entered.set()
        release.wait(10)


class FileLockTest(unittest.TestCase):
    def test_lock_releases_when_context_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "endpoint.lock"
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                with exclusive_file_lock(lock_path):
                    raise RuntimeError("sentinel")
            with exclusive_file_lock(lock_path):
                self.assertTrue(lock_path.exists())

    def test_endpoint_without_lock_path_is_a_noop(self):
        endpoint = ModelEndpoint(name="test", base_url="https://example.invalid")
        with _endpoint_request_lock(endpoint):
            pass

    def test_competing_process_waits_until_owner_releases(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            lock_path = str(Path(directory) / "endpoint.lock")
            first_entered = context.Event()
            first_release = context.Event()
            second_entered = context.Event()
            second_release = context.Event()
            first = context.Process(
                target=_lock_worker,
                args=(lock_path, first_entered, first_release),
            )
            second = context.Process(
                target=_lock_worker,
                args=(lock_path, second_entered, second_release),
            )
            try:
                first.start()
                self.assertTrue(first_entered.wait(5), "first process did not acquire the lock")
                second.start()
                self.assertFalse(
                    second_entered.wait(0.3),
                    "second process entered while the first still held the lock",
                )
                first_release.set()
                self.assertTrue(second_entered.wait(5), "second process never acquired the released lock")
                second_release.set()
                first.join(5)
                second.join(5)
                self.assertEqual(first.exitcode, 0)
                self.assertEqual(second.exitcode, 0)
            finally:
                first_release.set()
                second_release.set()
                for process in (first, second):
                    if process.is_alive():
                        process.terminate()
                    process.join(5)


if __name__ == "__main__":
    unittest.main()
