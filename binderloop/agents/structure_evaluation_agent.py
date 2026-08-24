
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Optional, Union

from binderloop.analysis.structure_features import analyze_binder_structure
from binderloop.resume import atomic_write_json


@dataclass
class StructureBatchEvaluation:
    total_structures: int
    summaries: List[Dict]
    aggregate_tags: Dict[str, int]
    reliable_seed_fraction: float
    observations: List[str] = field(default_factory=list)
    interface_data_quality: Dict[str, Any] = field(default_factory=dict)


class StructureEvaluationAgent:
    """Extract coordinate-level binder/interface features and seed reliability."""

    def analyze_structures(self, structure_files: Sequence[Union[str, Path]], *, binder_chain: str = "B", target_chains: Optional[Sequence[str]] = None, hotspots: Optional[Sequence[str]] = None, primary_residues: Optional[Sequence[str]] = None, expanded_residues: Optional[Sequence[str]] = None, negative_residues: Optional[Sequence[str]] = None, binder_length: Optional[Union[int, Sequence[int], Mapping[str, int]]] = None, auto_detect_chains: bool = True) -> StructureBatchEvaluation:
        summaries = []
        for path in structure_files:
            p = Path(path)
            if p.exists() and p.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
                length_hint = self._binder_length_for_structure(p, binder_length)
                summaries.append(analyze_binder_structure(p, binder_chain=binder_chain, target_chains=target_chains, hotspots=hotspots, primary_residues=primary_residues, expanded_residues=expanded_residues, negative_residues=negative_residues, binder_length=length_hint, auto_detect_chains=auto_detect_chains).to_dict())
        return self.aggregate_summaries(summaries)

    def analyze_trusted_structures(self, structure_files: Sequence[Union[str, Path]], **kwargs: Any) -> StructureBatchEvaluation:
        """Analyze inventory-validated paths without one metadata lookup per file."""
        summaries = []
        binder_length = kwargs.pop("binder_length", None)
        for path in structure_files:
            p = Path(path)
            if p.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
                continue
            length_hint = self._binder_length_for_structure(p, binder_length)
            summaries.append(analyze_binder_structure(p, binder_length=length_hint, **kwargs).to_dict())
        return self.aggregate_summaries(summaries)

    def aggregate_summaries(self, summaries: Sequence[Mapping[str, Any]]) -> StructureBatchEvaluation:
        rows = [dict(item) for item in summaries]
        tags: Dict[str, int] = {}
        reliable = 0
        for item in rows:
            if float(item.get("reliability_score", 0.0)) >= 0.7:
                reliable += 1
            for tag in item.get("reliability_tags", []):
                tags[tag] = tags.get(tag, 0) + 1
        data_quality = self._interface_data_quality(rows)
        return StructureBatchEvaluation(len(rows), rows, tags, reliable / max(1, len(rows)), self._observations(tags, reliable, len(rows), data_quality), data_quality)

    @staticmethod
    def _binder_length_for_structure(path: Path, binder_length: Optional[Union[int, Sequence[int], Mapping[str, int]]]) -> Optional[int]:
        """Resolve a per-structure binder length hint for BoltzGen output chains.

        BoltzGen output CIFs relabel chains by entity order, while filenames
        preserve the design spec token such as ``len100``. In multi-length
        rounds the orchestrator cannot pass a single length, so use the file
        token to keep chain auto-detection on the generated binder.
        """
        if isinstance(binder_length, int):
            return int(binder_length) if binder_length > 0 else None
        if isinstance(binder_length, Mapping):
            for key in (str(path), path.name, path.stem):
                value = binder_length.get(key)
                if value:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        pass
        match = re.search(r"(?:^|[_-])len(\d+)(?:[_-]|$)", path.stem)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        if isinstance(binder_length, Sequence) and not isinstance(binder_length, (str, bytes)):
            lengths = sorted({int(x) for x in binder_length if int(x) > 0})
            if len(lengths) == 1:
                return lengths[0]
        return None

    @staticmethod
    def _interface_data_quality(summaries: List[Dict]) -> Dict[str, Any]:
        if not summaries:
            return {"status": "no_structures", "zero_contact_fraction": 0.0, "zero_binder_residue_fraction": 0.0}
        zero_contact = sum(1 for s in summaries if int(s.get("interface_contact_count") or 0) == 0)
        zero_binder = sum(1 for s in summaries if int(s.get("binder_residue_count") or 0) == 0)
        notes = sorted({str(s.get("chain_detection_note") or "") for s in summaries if s.get("chain_detection_note")})
        binder_chains = sorted({str(s.get("binder_chain")) for s in summaries if s.get("binder_chain")})
        target_chain_sets = sorted({",".join(str(c) for c in (s.get("target_chains") or [])) for s in summaries})
        n = len(summaries)
        zero_contact_fraction = zero_contact / n
        # When (almost) every structure shows zero interface contacts, the coordinate
        # analysis is most likely mis-resolving the binder/target chains rather than
        # genuinely producing non-binding designs. Flag it so downstream policy does
        # not "repair" phantom weak-interface/hotspot-miss failures.
        suspicious = zero_contact_fraction >= 0.95 or (zero_binder / n) >= 0.5
        return {
            "status": "suspect_chain_mapping" if suspicious else "ok",
            "zero_contact_fraction": round(zero_contact_fraction, 4),
            "zero_binder_residue_fraction": round(zero_binder / n, 4),
            "chain_detection_notes": notes,
            "detected_binder_chains": binder_chains,
            "detected_target_chain_sets": target_chain_sets,
        }

    def write_batch(self, batch: StructureBatchEvaluation, path: Union[str, Path]) -> Path:
        return atomic_write_json(path, asdict(batch))

    @staticmethod
    def _observations(tags: Dict[str, int], reliable: int, total: int, data_quality: Optional[Dict[str, Any]] = None) -> List[str]:
        if total == 0:
            return ["No structure files were available; rely on metric/log analysis and defer coordinate-level conclusions."]
        obs = [f"{reliable}/{total} structures have reliability_score >= 0.7."]
        if (data_quality or {}).get("status") == "suspect_chain_mapping":
            obs.append(
                "WARNING: ~all structures show zero interface contacts; binder/target chain resolution is likely wrong. "
                "Treat weak_or_tiny_interface/hotspot_not_covered tags as unreliable this round and verify chain detection."
            )
        if tags.get("hotspot_not_covered", 0) > total * 0.3:
            obs.append("Many structures miss hotspots; strengthen hotspot conditioning or diversify hotspot subsets.")
        if tags.get("interface_clash_risk", 0) > total * 0.2:
            obs.append("Interface clashes are common; add clash-aware filters or soften packing constraints.")
        if tags.get("weak_or_tiny_interface", 0) > total * 0.3:
            obs.append("Interfaces are small; increase interface-contact constraints or explore longer/scaffolded binders.")
        if tags.get("binder_chain_break", 0):
            obs.append("Some binders show chain breaks; avoid exploiting those seeds.")
        return obs
