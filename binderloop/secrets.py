
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union


DEFAULT_SECRET_CONFIG = Path("configs/llm_endpoints.local.json")


@dataclass
class SecretStore:
    """Local-only secret resolver.

    Secrets may be referenced by environment variable name or stored directly in
    an ignored local JSON file.  Callers must use resolved values only at the
    side-effect boundary that needs them, e.g. HTTP headers or Taiji submission.
    """

    data: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Union[str, Optional[Path]]) -> "SecretStore":
        if path is None:
            return cls()
        path = Path(path).expanduser()
        if not path.exists():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_default(cls, *, root: Union[str, Optional[Path]] = None) -> "SecretStore":
        path = DEFAULT_SECRET_CONFIG
        if root is not None and not path.is_absolute():
            path = Path(root) / path
        return cls.from_json(path)

    def get(self, name: str, *, env: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
        if env and os.environ.get(env):
            return os.environ[env]
        if os.environ.get(name):
            return os.environ[name]

        secrets = self.data.get("secrets", {}) if isinstance(self.data, Mapping) else {}
        item = secrets.get(name, {}) if isinstance(secrets, Mapping) else {}
        if isinstance(item, str):
            return item
        if isinstance(item, Mapping):
            env_name = item.get("env") or item.get("api_key_env")
            if env_name and os.environ.get(str(env_name)):
                return os.environ[str(env_name)]
            value = item.get("value")
            if value:
                return str(value)
        return default

    def ceph_secret(self) -> Optional[str]:
        return self.get("CEPH_SECRET", env="CEPH_SECRET")


def redact_sensitive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: ("<REDACTED>" if _is_sensitive_key(str(key)) else redact_sensitive(value))
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [redact_sensitive(value) for value in obj]
    if isinstance(obj, str):
        return redact_sensitive_text(obj)
    return obj


def redact_sensitive_text(text: str) -> str:
    text = re.sub(r"(?<!\{)(CEPH_SECRET[\"']?\s*[:=](?!-)\s*[\"']?)([^\"'\s,}]+)", r"\1<REDACTED>", text)
    text = re.sub(r"(secret=)([^,\s\"']+)", r"\1<REDACTED>", text, flags=re.IGNORECASE)
    text = re.sub(r"(Authorization:\s*Bearer\s+)([A-Za-z0-9._~+/=-]+)", r"\1<REDACTED>", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\{)([A-Za-z0-9_]*SECRET[\"']?\s*[:=](?!-)\s*[\"']?)([^\"'\s,}]+)", r"\1<REDACTED>", text)
    return text


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("token", "secret", "password", "credential", "api_key", "apikey", "key"))
