"""Append-only JSONL event journal with replay and integrity validation."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

from binderloop.file_lock import exclusive_file_lock

from .contracts import GENESIS_HASH, HarnessEvent, HarnessEventType


class JournalError(RuntimeError):
    """Base class for journal read/write failures."""


class JournalCorruptionError(JournalError):
    """A complete journal record failed parsing or chain validation."""


class JournalTailError(JournalError):
    """The journal ends in a partial record, normally after an interrupted write."""


@dataclass(frozen=True)
class ReplayResult:
    events: Tuple[HarnessEvent, ...]
    truncated_tail_ignored: bool = False
    valid_bytes: int = 0


_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: Dict[str, threading.RLock] = {}


def _process_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.resolve()))
    with _LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _sync_directory(path: Path) -> None:
    """Persist a newly-created directory entry where the platform permits it."""

    if os.name == "nt":
        return
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


class EventJournal:
    """Run-scoped durable event writer and deterministic replay reader."""

    def __init__(self, path: Union[str, Path], *, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        self.path = Path(path)
        self.run_id = run_id
        self._lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._process_lock = _process_lock(self.path)

    def record(
        self,
        event_type: Union[HarnessEventType, str],
        payload: Mapping[str, Any],
    ) -> HarnessEvent:
        """Validate the current chain, append one record, flush, and fsync it."""

        with self._process_lock, exclusive_file_lock(self._lock_path):
            replayed = self._replay(allow_truncated_tail=False)
            previous = replayed.events[-1] if replayed.events else None
            event = HarnessEvent.create(
                run_id=self.run_id,
                sequence=(previous.sequence + 1) if previous else 1,
                event_type=event_type,
                payload=payload,
                previous_hash=previous.event_hash if previous else GENESIS_HASH,
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            was_missing = not self.path.exists()
            encoded = f"{event.to_json()}\n".encode("utf-8")
            with self.path.open("ab") as handle:
                written = handle.write(encoded)
                if written != len(encoded):
                    raise JournalError(f"short journal write: {written} of {len(encoded)} bytes")
                handle.flush()
                os.fsync(handle.fileno())
            if was_missing:
                _sync_directory(self.path.parent)
            return event

    append = record

    def replay(self, *, allow_truncated_tail: bool = False) -> ReplayResult:
        """Replay and validate sequence, run identity, hashes, and hash links."""

        with self._process_lock, exclusive_file_lock(self._lock_path):
            return self._replay(allow_truncated_tail=allow_truncated_tail)

    def _replay(self, *, allow_truncated_tail: bool) -> ReplayResult:
        if not self.path.exists():
            return ReplayResult(events=(), valid_bytes=0)
        raw = self.path.read_bytes()
        truncated = bool(raw) and not raw.endswith(b"\n")
        if truncated:
            boundary = raw.rfind(b"\n") + 1
            if not allow_truncated_tail:
                raise JournalTailError(
                    f"journal has an incomplete final record beginning at byte {boundary}"
                )
            complete = raw[:boundary]
        else:
            complete = raw

        events = []
        expected_sequence = 1
        expected_previous_hash = GENESIS_HASH
        offset = 0
        for line_number, encoded_line in enumerate(complete.splitlines(keepends=True), start=1):
            line_offset = offset
            offset += len(encoded_line)
            stripped = encoded_line.rstrip(b"\r\n")
            if not stripped:
                raise JournalCorruptionError(f"blank journal record at line {line_number}")
            try:
                decoded = stripped.decode("utf-8")
                value = json.loads(decoded)
                if not isinstance(value, dict):
                    raise ValueError("record must be a JSON object")
                event = HarnessEvent.from_dict(value)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise JournalCorruptionError(
                    f"invalid journal record at line {line_number}, byte {line_offset}: {exc}"
                ) from exc
            if event.run_id != self.run_id:
                raise JournalCorruptionError(
                    f"run_id mismatch at line {line_number}: {event.run_id!r} != {self.run_id!r}"
                )
            if event.sequence != expected_sequence:
                raise JournalCorruptionError(
                    f"non-contiguous sequence at line {line_number}: "
                    f"{event.sequence} != {expected_sequence}"
                )
            if event.previous_hash != expected_previous_hash:
                raise JournalCorruptionError(f"broken hash chain at line {line_number}")
            events.append(event)
            expected_sequence += 1
            expected_previous_hash = event.event_hash
        return ReplayResult(
            events=tuple(events),
            truncated_tail_ignored=truncated,
            valid_bytes=len(complete),
        )

    def repair_truncated_tail(self) -> int:
        """Discard only a detected partial final record and fsync the repair.

        Complete but corrupt records are never changed automatically.
        Returns the number of discarded bytes.
        """

        with self._process_lock, exclusive_file_lock(self._lock_path):
            if not self.path.exists():
                return 0
            raw = self.path.read_bytes()
            if not raw or raw.endswith(b"\n"):
                self._replay(allow_truncated_tail=False)
                return 0
            boundary = raw.rfind(b"\n") + 1
            self._replay(allow_truncated_tail=True)
            removed = len(raw) - boundary
            with self.path.open("r+b") as handle:
                handle.truncate(boundary)
                handle.flush()
                os.fsync(handle.fileno())
            return removed
