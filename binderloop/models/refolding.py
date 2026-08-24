"""Independent refolding tools used by closed-loop search profiles.

Boltz-2 and RF3 are separate backends under this module. RMSD thresholds share a
name but not a pose model; callers must record which tool produced the metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Union
from pathlib import Path


class RefoldToolError(ValueError):
    """Raised when a refold tool or backbone/tool pairing is invalid."""


@dataclass(frozen=True)
class RefoldTool:
    """One structure-refold backend with an isolated parameter contract."""

    name: str
    allowed_keys: FrozenSet[str]
    forbidden_keys: FrozenSet[str]
    ingest_model: str

    def ingest(self, output_dir: Union[str, Path], **kwargs: Any) -> Any:
        raise NotImplementedError


class Boltz2RefoldTool(RefoldTool):
    def ingest(self, output_dir: Union[str, Path], **kwargs: Any) -> Any:
        ingestor = kwargs.get("ingestor")
        if ingestor is None:
            from binderloop.agents.result_ingestion_agent import ResultIngestionAgent
            ingestor = ResultIngestionAgent()
        return ingestor.ingest_boltzgen_output(
            output_dir,
            log_file=kwargs.get("log_file"),
            identity_context=kwargs.get("identity_context"),
            max_rows=int(kwargs.get("max_rows") or 2000),
        )


class RF3RefoldTool(RefoldTool):
    def ingest(self, output_dir: Union[str, Path], **kwargs: Any) -> Any:
        ingestor = kwargs.get("ingestor")
        if ingestor is None:
            from binderloop.agents.result_ingestion_agent import ResultIngestionAgent
            ingestor = ResultIngestionAgent()
        return ingestor.ingest_rfd3_output(
            output_dir,
            log_file=kwargs.get("log_file"),
            identity_context=kwargs.get("identity_context"),
            max_rows=int(kwargs.get("max_rows") or 2000),
        )


BOLTZ2 = Boltz2RefoldTool(
    name="boltz2",
    allowed_keys=frozenset({
        "refolding_rmsd_threshold", "folding_checkpoint", "affinity_checkpoint",
        "refold_tool",
    }),
    forbidden_keys=frozenset({
        "n_recycles", "rf3_checkpoint", "early_stopping_plddt_threshold",
        "num_steps", "ckpt_path",
    }),
    ingest_model="boltzgen",
)

RF3 = RF3RefoldTool(
    name="rf3",
    allowed_keys=frozenset({
        "refolding_rmsd_threshold", "n_recycles", "rf3_checkpoint", "folding_checkpoint",
        "early_stopping_plddt_threshold", "num_steps", "refold_tool",
    }),
    forbidden_keys=frozenset({
        "affinity_checkpoint",
    }),
    ingest_model="rfd3",
)

REFOLD_TOOLS: Dict[str, RefoldTool] = {
    BOLTZ2.name: BOLTZ2,
    RF3.name: RF3,
}

_REFOLD_ALIASES = {
    "boltz-2": "boltz2",
    "boltz_2": "boltz2",
    "boltzgen": "boltz2",
    "rf3_fold": "rf3",
    "rf3-fold": "rf3",
}


def normalize_refold_tool_name(name: Optional[str]) -> str:
    token = str(name or "").strip().lower()
    return _REFOLD_ALIASES.get(token, token)


def get_refold_tool(name: str) -> RefoldTool:
    key = normalize_refold_tool_name(name)
    tool = REFOLD_TOOLS.get(key)
    if tool is None:
        raise RefoldToolError(f"unknown refold tool {name!r}; available={sorted(REFOLD_TOOLS)}")
    return tool
