
import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

# Literal is Python 3.8+; use str for Python 3.6 compatibility
MessageType = str  # type: Literal["observation", "hypothesis", "proposal", "decision", "failure", "retry", "status", "artifact"]


_EXECUTION_MESSAGE_FIELDS = {
    "job_id", "backend", "attempt", "attempts", "status", "error", "retryable",
    "returncode", "taiji_job_id", "task_flag", "failure_class",
    "resource_retry_degradation", "wait_override",
}
_EXECUTION_PRIVATE_FIELDS = {
    "run_spec", "submit_spec", "submission", "stdout", "stderr", "stdout_tail",
    "stderr_tail", "log_tail", "boltzgen_log_tail", "gpu_log_tails",
}


def project_execution_message(content: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the bounded, path-free projection safe for the message bus.

    Execution records remain complete in their local JSON artifacts; only the
    inter-agent envelope is projected.
    """
    projected: Dict[str, Any] = {}
    for key in _EXECUTION_MESSAGE_FIELDS:
        if key in content and key not in _EXECUTION_PRIVATE_FIELDS:
            value = _safe_execution_value(content[key])
            if value is not None:
                projected[key] = value
    return projected


def _safe_execution_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        # Do not publish absolute filesystem paths, including paths embedded in
        # error strings. Keep a stable generic marker instead.
        normalized = value.replace("\\", "/")
        if re.search(r"(?:^|[\s=\"'(:])/(?:[^\s,;]+)", normalized):
            return "[local execution detail omitted]"
        return value[:1000]
    if isinstance(value, Mapping):
        return {
            str(k): safe
            for k, item in value.items()
            if str(k) not in _EXECUTION_PRIVATE_FIELDS
            for safe in [_safe_execution_value(item)]
            if safe is not None
        }
    if isinstance(value, (list, tuple)):
        return [safe for item in value for safe in [_safe_execution_value(item)] if safe is not None]
    return str(value)[:1000]


@dataclass
class AgentMessage:
    """JSON-serializable envelope for agent-to-agent communication."""

    sender: str
    recipient: str
    message_type: MessageType
    content: Dict[str, Any]
    round_id: int = 0
    job_id: Optional[str] = None
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    confidence: Optional[float] = None
    requires_response: bool = False
    artifacts: List[str] = field(default_factory=list)
    idempotency_key: Optional[str] = None
    run_id: Optional[str] = None
    module: Optional[str] = None
    input_digest: Optional[str] = None
    event_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MessageBus:
    """Append-only JSONL message bus shared by deterministic and LLM agents."""

    def __init__(self, path: Union[str, Path], *, run_id: Optional[str] = None):
        self.path = Path(path)
        self.run_id = str(run_id or self.path.parent.resolve())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._messages: List[AgentMessage] = []
        self._idempotency_index: Dict[str, AgentMessage] = {}
        self._initialized = False
        self._file_identity: Optional[tuple] = None
        self._offset = 0
        self._mtime_ns: Optional[int] = None
        # A small prefix check distinguishes a normal external append from a
        # truncate-and-rewrite that grew past the previous offset.
        self._tail = b""

    def publish(self, message: AgentMessage) -> AgentMessage:
        with self._lock:
            # Refresh before deriving/checking the key so resumes and independent
            # bus instances observe already-published events.
            self._refresh_locked()
            self._ensure_idempotency_key(message)
            existing = self._idempotency_index.get(str(message.idempotency_key))
            if existing is not None:
                return existing
            line = (json.dumps(message.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")

            cached_offset = self._offset
            cached_identity = self._file_identity
            with self.path.open("ab") as handle:
                before = os.fstat(handle.fileno())
                handle.write(line)
                handle.flush()
                end_offset = handle.tell()
                after = os.fstat(handle.fileno())

            identity = (after.st_dev, after.st_ino)
            can_update_directly = (
                self._initialized
                and before.st_size == cached_offset
                and end_offset == cached_offset + len(line)
                and (cached_identity is None or cached_identity == identity)
            )
            if can_update_directly:
                self._messages.append(message)
                self._idempotency_index[str(message.idempotency_key)] = message
                self._file_identity = identity
                self._offset = end_offset
                self._mtime_ns = after.st_mtime_ns
                self._tail = (self._tail + line)[-128:]
            else:
                # A replacement or an external writer raced with publish.
                # Refreshing from the previous offset preserves file order.
                self._refresh_locked()
        return message

    def read_all(self) -> List[AgentMessage]:
        with self._lock:
            self._refresh_locked()
            return list(self._messages)

    def _refresh_locked(self) -> None:
        try:
            current = self.path.stat()
        except FileNotFoundError:
            self._reset_empty_locked()
            return

        identity = (current.st_dev, current.st_ino)
        if (
            self._initialized
            and identity == self._file_identity
            and current.st_size == self._offset
            and current.st_mtime_ns == self._mtime_ns
        ):
            return

        with self.path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            opened_identity = (opened.st_dev, opened.st_ino)
            incremental = (
                self._initialized
                and opened_identity == self._file_identity
                and opened.st_size > self._offset
            )

            if incremental and self._tail:
                handle.seek(self._offset - len(self._tail))
                incremental = handle.read(len(self._tail)) == self._tail

            if incremental:
                handle.seek(self._offset)
                data = handle.read()
                self._messages.extend(self._decode_messages(data))
                self._rebuild_idempotency_index_locked()
                self._offset += len(data)
                self._tail = (self._tail + data)[-128:]
            else:
                handle.seek(0)
                data = handle.read()
                self._messages = self._decode_messages(data)
                self._rebuild_idempotency_index_locked()
                self._offset = len(data)
                self._tail = data[-128:]

            final = os.fstat(handle.fileno())
            self._file_identity = (final.st_dev, final.st_ino)
            self._mtime_ns = final.st_mtime_ns
            self._initialized = True

    def _ensure_idempotency_key(self, message: AgentMessage) -> None:
        event_type = str(message.event_type or message.content.get("event") or message.message_type)
        module = str(message.module or message.content.get("module") or message.sender)
        input_digest = str(message.input_digest or _stable_digest(message.content))
        run_id = str(message.run_id or self.run_id)
        message.event_type = event_type
        message.module = module
        message.input_digest = input_digest
        message.run_id = run_id
        if not message.idempotency_key:
            message.idempotency_key = _stable_digest({
                "run": run_id, "round": int(message.round_id), "module": module,
                "input_digest": input_digest, "event_type": event_type,
            })

    def _rebuild_idempotency_index_locked(self) -> None:
        self._idempotency_index = {}
        for message in self._messages:
            self._ensure_idempotency_key(message)
            self._idempotency_index.setdefault(str(message.idempotency_key), message)

    @staticmethod
    def _decode_messages(data: bytes) -> List[AgentMessage]:
        messages: List[AgentMessage] = []
        for raw_line in data.splitlines():
            if raw_line.strip():
                messages.append(AgentMessage(**json.loads(raw_line.decode("utf-8"))))
        return messages

    def _reset_empty_locked(self) -> None:
        self._messages = []
        self._idempotency_index = {}
        self._initialized = True
        self._file_identity = None
        self._offset = 0
        self._mtime_ns = None
        self._tail = b""

    def query(self, *, round_id: Optional[int] = None, job_id: Optional[str] = None, sender: Optional[str] = None,
              recipient: Optional[str] = None, message_type: Optional[str] = None) -> List[AgentMessage]:
        items = self.read_all()
        if round_id is not None:
            items = [m for m in items if m.round_id == round_id]
        if job_id is not None:
            items = [m for m in items if m.job_id == job_id]
        if sender is not None:
            items = [m for m in items if m.sender == sender]
        if recipient is not None:
            items = [m for m in items if m.recipient == recipient]
        if message_type is not None:
            items = [m for m in items if m.message_type == message_type]
        return items


def compact_messages(messages: List[AgentMessage], max_items: int = 50) -> List[Dict[str, Any]]:
    return [
        {
            "round_id": m.round_id,
            "job_id": m.job_id,
            "sender": m.sender,
            "recipient": m.recipient,
            "message_type": m.message_type,
            "content": m.content,
            "confidence": m.confidence,
        }
        for m in messages[-max_items:]
    ]


def _stable_digest(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
