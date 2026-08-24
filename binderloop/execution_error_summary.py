"""Bounded, path-redacted execution error summaries."""

import re
from typing import Any, Dict, Mapping, Optional, Sequence

EXECUTION_ERROR_SUMMARY_SCHEMA_VERSION = "1.0"
_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|/)[^\s\]\[(){}<>\"']+")

def sanitize_error_text(value: Any, *, limit: int = 500) -> Optional[str]:
    if value in (None, ""):
        return None
    text = _PATH.sub("<path>", str(value)).replace("\x00", "")
    return text if len(text) <= limit else text[:max(0, limit-14)] + "...[truncated]"

def build_execution_error_summary(records: Optional[Sequence[Mapping[str, Any]]], *, max_errors: int = 10, text_limit: int = 500) -> Dict[str, Any]:
    statuses: Dict[str, int] = {}
    failures = []
    retried = 0
    for record in records or []:
        status = str(record.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if int(record.get("attempts") or 0) > 1: retried += 1
        if status in {"failed", "error", "not_executed", "timeout", "cancelled", "canceled"}:
            job = record.get("job") if isinstance(record.get("job"), Mapping) else {}
            failures.append({"job_id": job.get("job_id"), "status": status, "attempts": record.get("attempts"), "error": sanitize_error_text(record.get("error"), limit=text_limit)})
    failures = failures[:max_errors]
    return {"schema": "binder_harness.execution_error_summary", "schema_version": EXECUTION_ERROR_SUMMARY_SCHEMA_VERSION, "record_count": len(records or []), "status_counts": statuses, "retried_jobs": retried, "failed_jobs": failures, "failure_hints": [x["error"] for x in failures if x.get("error")] }
