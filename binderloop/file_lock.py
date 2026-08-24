"""Small cross-platform advisory file-lock primitive.

The harness uses file locks to serialize access shared by independent local
processes.  Import platform-specific modules lazily so importing the package is
safe on both Windows and POSIX systems.
"""

from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Union


PathLike = Union[str, os.PathLike[str]]


@contextmanager
def exclusive_file_lock(
    path: PathLike,
    *,
    poll_interval_seconds: float = 0.05,
) -> Iterator[None]:
    """Hold an exclusive advisory lock for ``path`` until the context exits.

    On POSIX this delegates to blocking ``flock``.  Windows byte-range locks do
    not expose an indefinitely blocking operation, so a non-blocking attempt is
    retried until it succeeds.  The one-byte sentinel is only lock metadata;
    callers should use a dedicated ``*.lock`` file rather than a data file.

    The primitive is intentionally non-reentrant and has no timeout: callers
    must not acquire the same path recursively and should keep critical sections
    bounded.  A release failure is raised when the body succeeded; when the body
    already failed it is attached as a note so the original exception survives.
    """

    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")

    lock_path = Path(path).expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b", buffering=0) as handle:
        if os.name == "nt":
            _prepare_windows_lock_byte(handle)
            _acquire_windows_lock(handle, poll_interval_seconds)
            release = _release_windows_lock
        elif os.name == "posix":
            _acquire_posix_lock(handle)
            release = _release_posix_lock
        else:  # pragma: no cover - Python's supported desktop OSes use nt/posix.
            raise OSError(f"unsupported file-lock platform: os.name={os.name!r}")

        body_error = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                release(handle)
            except Exception as release_error:
                if body_error is None:
                    raise
                body_error.add_note(f"file lock release also failed: {release_error!r}")


def _prepare_windows_lock_byte(handle: BinaryIO) -> None:
    """Ensure the byte range used by ``msvcrt.locking`` exists."""

    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)


def _acquire_windows_lock(handle: BinaryIO, poll_interval_seconds: float) -> None:
    import msvcrt

    retryable_errnos = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
    while True:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in retryable_errnos:
                raise
            time.sleep(poll_interval_seconds)


def _release_windows_lock(handle: BinaryIO) -> None:
    import msvcrt

    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _acquire_posix_lock(handle: BinaryIO) -> None:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_posix_lock(handle: BinaryIO) -> None:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
