"""Typed, hashable event contracts for the Harness control plane.

The contract intentionally contains only JSON values.  This keeps journals
portable across Python versions and makes every event independently
verifiable without importing application-specific result classes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Union


SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

JsonScalar = Union[None, bool, int, float, str]
JsonValue = Union[JsonScalar, list["JsonValue"], Dict[str, "JsonValue"]]


class HarnessEventType(str, Enum):
    """Stable event names understood by the initial Harness runtime."""

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    ROUND_STARTED = "round.started"
    ROUND_COMPLETED = "round.completed"
    GRAPH_NODE_STARTED = "graph.node.started"
    GRAPH_NODE_SUCCEEDED = "graph.node.succeeded"
    GRAPH_NODE_FAILED = "graph.node.failed"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_json(value: Any, path: str = "payload") -> JsonValue:
    """Return a detached JSON value and reject lossy/non-portable inputs."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: Dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            normalized[key] = _normalize_json(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class HarnessEvent:
    """One frozen event envelope in a run-scoped, append-only hash chain.

    Payloads are detached JSON values at construction time but remain ordinary
    Python dict/list objects after replay.  ``verify_hash`` detects later content
    mutation; callers should not treat the nested payload as recursively frozen.
    """

    schema_version: int
    event_id: str
    run_id: str
    sequence: int
    event_type: str
    occurred_at: str
    payload: JsonValue
    previous_hash: str
    event_hash: str

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        sequence: int,
        event_type: Union[HarnessEventType, str],
        payload: Mapping[str, Any],
        previous_hash: str,
        event_id: Optional[str] = None,
        occurred_at: Optional[str] = None,
    ) -> "HarnessEvent":
        type_value = event_type.value if isinstance(event_type, HarnessEventType) else str(event_type)
        body: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id or str(uuid.uuid4()),
            "run_id": run_id,
            "sequence": sequence,
            "event_type": type_value,
            "occurred_at": occurred_at or _utc_now(),
            "payload": _normalize_json(payload),
            "previous_hash": previous_hash,
        }
        cls._validate_body(body)
        event_hash = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
        return cls(event_hash=event_hash, **body)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HarnessEvent":
        required = {
            "schema_version",
            "event_id",
            "run_id",
            "sequence",
            "event_type",
            "occurred_at",
            "payload",
            "previous_hash",
            "event_hash",
        }
        missing = required.difference(value)
        unexpected = set(value).difference(required)
        if missing:
            raise ValueError(f"event is missing fields: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"event has unexpected fields: {sorted(unexpected)}")

        body = {key: value[key] for key in required if key != "event_hash"}
        body["payload"] = _normalize_json(body["payload"])
        cls._validate_body(body)
        event_hash = value["event_hash"]
        if not isinstance(event_hash, str) or not _SHA256_RE.fullmatch(event_hash):
            raise ValueError("event_hash must be a lowercase SHA-256 digest")
        event = cls(event_hash=event_hash, **body)
        event.verify_hash()
        return event

    @staticmethod
    def _validate_body(body: Mapping[str, Any]) -> None:
        if body["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported event schema_version {body['schema_version']!r}")
        for name in ("event_id", "run_id", "event_type", "occurred_at"):
            if not isinstance(body[name], str) or not body[name].strip():
                raise ValueError(f"{name} must be a non-empty string")
        try:
            uuid.UUID(body["event_id"])
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("event_id must be a UUID") from exc
        sequence = body["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("sequence must be a positive integer")
        previous_hash = body["previous_hash"]
        if not isinstance(previous_hash, str) or not _SHA256_RE.fullmatch(previous_hash):
            raise ValueError("previous_hash must be a lowercase SHA-256 digest")
        if not isinstance(body["payload"], dict):
            raise TypeError("payload must be a JSON object")

    def body_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> Dict[str, Any]:
        value = self.body_dict()
        value["event_hash"] = self.event_hash
        return value

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    def verify_hash(self) -> None:
        expected = hashlib.sha256(_canonical_json(self.body_dict()).encode("utf-8")).hexdigest()
        if self.event_hash != expected:
            raise ValueError("event_hash does not match event content")
