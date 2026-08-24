"""Independent sequence (inverse-fold) tools used by closed-loop search profiles.

GPU pipelines can later extract CLI renderers from these tools. This module owns
the executable key contracts so Boltz ifold flags never reach ProteinMPNN and
vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Optional


class SequenceToolError(ValueError):
    """Raised when a sequence tool or backbone/tool pairing is invalid."""


@dataclass(frozen=True)
class SequenceTool:
    """One inverse-fold backend with an isolated parameter contract."""

    name: str
    allowed_keys: FrozenSet[str]
    forbidden_keys: FrozenSet[str]
    conda_env: str

    def materialize(self, policy: str, current: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class BoltzIFoldTool(SequenceTool):
    def materialize(self, policy: str, current: Mapping[str, Any]) -> Dict[str, Any]:
        if str(policy or "").strip().lower() != "repair":
            return {}
        update: Dict[str, Any] = {"filter_biased": "true", "sequence_tool": self.name}
        avoid = current.get("inverse_fold_avoid")
        if not avoid:
            update["inverse_fold_avoid"] = "C"
        return update


class ProteinMPNNTool(SequenceTool):
    def materialize(self, policy: str, current: Mapping[str, Any]) -> Dict[str, Any]:
        if str(policy or "").strip().lower() != "repair":
            return {}
        try:
            temperature = float(current.get("temperature", 0.1) or 0.1)
        except (TypeError, ValueError):
            temperature = 0.1
        next_temperature = min(0.3, round(temperature + 0.1, 6))
        if next_temperature <= temperature:
            next_temperature = 0.2 if temperature < 0.2 else temperature
        return {
            "temperature": next_temperature,
            "model_type": str(current.get("model_type") or "protein_mpnn"),
            "is_legacy_weights": True,
            "sequence_tool": self.name,
        }


BOLTZ_IFOLD = BoltzIFoldTool(
    name="boltz_ifold",
    allowed_keys=frozenset({
        "filter_biased", "inverse_fold_avoid", "inverse_fold_num_sequences",
        "inverse_fold_checkpoint", "skip_inverse_folding", "only_inverse_fold",
        "sequence_tool",
    }),
    forbidden_keys=frozenset({
        "temperature", "is_legacy_weights", "model_type", "designed_chains",
        "mpnn_checkpoint", "checkpoint_path", "batch_size", "number_of_batches",
        "write_fasta", "write_structures",
    }),
    conda_env="bg",
)

PROTEIN_MPNN = ProteinMPNNTool(
    name="protein_mpnn",
    allowed_keys=frozenset({
        "temperature", "is_legacy_weights", "model_type", "designed_chains",
        "mpnn_checkpoint", "checkpoint_path", "batch_size", "number_of_batches",
        "write_fasta", "write_structures", "inverse_fold_num_sequences",
        "sequence_tool",
    }),
    forbidden_keys=frozenset({
        "filter_biased", "inverse_fold_avoid", "inverse_fold_checkpoint",
        "skip_inverse_folding", "only_inverse_fold", "alpha",
    }),
    conda_env="foundry",
)

SEQUENCE_TOOLS: Dict[str, SequenceTool] = {
    BOLTZ_IFOLD.name: BOLTZ_IFOLD,
    PROTEIN_MPNN.name: PROTEIN_MPNN,
}

_SEQUENCE_ALIASES = {
    "boltz-if": "boltz_ifold",
    "boltz-ifold": "boltz_ifold",
    "boltz_if": "boltz_ifold",
    "boltzgen_ifold": "boltz_ifold",
    "proteinmpnn": "protein_mpnn",
    "protein-mpnn": "protein_mpnn",
    "mpnn": "protein_mpnn",
}


def normalize_sequence_tool_name(name: Optional[str]) -> str:
    token = str(name or "").strip().lower()
    return _SEQUENCE_ALIASES.get(token, token)


def get_sequence_tool(name: str) -> SequenceTool:
    key = normalize_sequence_tool_name(name)
    tool = SEQUENCE_TOOLS.get(key)
    if tool is None:
        raise SequenceToolError(f"unknown sequence tool {name!r}; available={sorted(SEQUENCE_TOOLS)}")
    return tool
