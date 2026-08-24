"""Canonical typed residue identities."""
from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Optional

@dataclass(frozen=True, order=True)
class ResidueIdentity:
    chain: str
    author_residue_number: int
    insertion_code: str = ""
    label: Optional[str] = None
    name: Optional[str] = None
    def __post_init__(self):
        object.__setattr__(self, "chain", str(self.chain).strip())
        object.__setattr__(self, "insertion_code", str(self.insertion_code or "").strip())
        if not self.chain or len(self.insertion_code) > 1: raise ValueError("invalid residue identity")
    @property
    def token(self): return f"{self.chain}:{self.author_residue_number}{self.insertion_code}"
    @property
    def digest(self): return hashlib.sha256(self.token.encode()).hexdigest()
    def to_dict(self): return asdict(self)

_RX = re.compile(r"^\s*([^:/\s]+)\s*[:/]\s*(-?\d+)\s*([A-Za-z]?)\s*$")
def parse_residue_identity(value: Any, *, default_chain: str = "") -> ResidueIdentity:
    if isinstance(value, ResidueIdentity): return value
    if isinstance(value, Mapping):
        return ResidueIdentity(str(value.get("chain") or default_chain), int(value.get("author_residue_number", value.get("resseq", value.get("residue_number")))), str(value.get("insertion_code", value.get("icode", "")) or ""), value.get("label"), value.get("name"))
    match = _RX.match(str(value))
    if match: return ResidueIdentity(match.group(1), int(match.group(2)), match.group(3))
    match = re.match(r"^(-?\d+)([A-Za-z]?)$", str(value).strip())
    if match and default_chain: return ResidueIdentity(default_chain, int(match.group(1)), match.group(2))
    raise ValueError(f"invalid residue identity: {value!r}")

def canonical_residue_digest(values: Iterable[Any]) -> str:
    payload = json.dumps([parse_residue_identity(x).token for x in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
