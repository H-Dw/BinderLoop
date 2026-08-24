"""Data-driven target epitope cropping from real interface evidence.

The harness historically passed a static full-chain ``target_include`` (e.g.
``chain E res_index 1..194``) to BoltzGen every round, so the "dynamic target
cropping" knob was never actually exercised. Once binder/target chains are
resolved correctly (see ``structure_features.detect_binder_target_chains``), the
per-structure contacts reveal which target residues the designs actually engage.

This module aggregates that evidence into an executable, tunable crop proposal:

* ``target_include``        – a focused residue window around the engaged epitope
                              (and/or the requested hotspots), instead of the
                              whole chain.
* ``target_binding_types``  – binding-site residues for that focused region.
* ``prioritize_hotspots``   – hotspots that are actually contactable.

The proposal is deliberately conservative: it only narrows the crop when there
is consistent, reasonably high-quality interface evidence, and it never drops
the user's requested hotspot epitope when ``mode`` keeps it in scope.
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


@dataclass
class EpitopeCropProposal:
    mode: str
    engaged_residues: List[int] = field(default_factory=list)
    requested_hotspot_residues: List[int] = field(default_factory=list)
    crop_window: Optional[List[int]] = None
    target_include: List[Dict[str, Any]] = field(default_factory=list)
    target_binding_types: List[Dict[str, Any]] = field(default_factory=list)
    prioritize_hotspots: List[str] = field(default_factory=list)
    recommended_config: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "engaged_residues": self.engaged_residues,
            "requested_hotspot_residues": self.requested_hotspot_residues,
            "crop_window": self.crop_window,
            "target_include": self.target_include,
            "target_binding_types": self.target_binding_types,
            "prioritize_hotspots": self.prioritize_hotspots,
            "recommended_config": self.recommended_config,
            "observations": self.observations,
        }


def _residue_number(token: Any) -> Optional[int]:
    text = str(token)
    if ":" in text:
        text = text.split(":", 1)[1]
    elif "/" in text:
        text = text.split("/", 1)[1]
    digits = "".join(ch for ch in text if ch.isdigit() or ch == "-")
    try:
        return int(digits)
    except ValueError:
        return None


def aggregate_engaged_residues(
    summaries: Sequence[Mapping[str, Any]],
    *,
    min_reliability: float = 0.0,
    top_n: Optional[int] = None,
) -> Dict[int, int]:
    """Count, per target residue number, how many structures contact it.

    Only structures with ``reliability_score >= min_reliability`` are counted, so
    the consensus epitope reflects the better designs rather than every failure.
    """
    ranked = sorted(
        (s for s in summaries if float(s.get("reliability_score") or 0.0) >= min_reliability),
        key=lambda s: float(s.get("reliability_score") or 0.0),
        reverse=True,
    )
    if top_n is not None:
        ranked = ranked[: max(1, top_n)]
    counts: Dict[int, int] = {}
    for summary in ranked:
        residues_this_structure: Set[int] = set()
        for contact in summary.get("contacts_preview") or []:
            number = _residue_number(contact.get("target_residue"))
            if number is not None:
                residues_this_structure.add(number)
        for number in residues_this_structure:
            counts[number] = counts.get(number, 0) + 1
    return counts


def propose_epitope_crop(
    summaries: Sequence[Mapping[str, Any]],
    *,
    target_chain: str,
    requested_hotspots: Optional[Sequence[str]] = None,
    mode: str = "auto",
    margin: int = 6,
    min_structures: int = 3,
    consensus_fraction: float = 0.25,
    min_reliability: float = 0.5,
    structure_groups: Optional[str] = None,
) -> EpitopeCropProposal:
    """Build a data-driven target crop from observed interface contacts.

    Modes:
      * ``hotspot_focus``  – crop tightly to the requested hotspot epitope ± margin.
      * ``engaged_focus``  – crop to the consensus engaged residues ± margin.
      * ``union``          – crop to cover both engaged residues and hotspots.
      * ``auto``           – pick ``hotspot_focus`` when designs already engage the
                             hotspot epitope; otherwise keep the full chain and only
                             raise hotspot priority (the designs bind the wrong patch,
                             so narrowing the crop would be premature).
    """
    summaries = list(summaries or [])
    requested_hotspots = list(requested_hotspots or [])
    hotspot_numbers = sorted({n for n in (_residue_number(h) for h in requested_hotspots) if n is not None})

    counts = aggregate_engaged_residues(summaries, min_reliability=min_reliability)
    if not counts:
        counts = aggregate_engaged_residues(summaries, min_reliability=0.0)
    structures_considered = sum(1 for s in summaries if float(s.get("reliability_score") or 0.0) >= min_reliability) or len(summaries)
    threshold = max(1, int(round(structures_considered * consensus_fraction)))
    engaged = sorted(n for n, c in counts.items() if c >= threshold)
    if not engaged and counts:
        # Fall back to the most frequently contacted residues.
        engaged = sorted(sorted(counts, key=lambda n: counts[n], reverse=True)[: max(1, margin)])

    observations: List[str] = []
    hotspot_engagement = [n for n in hotspot_numbers if counts.get(n, 0) >= max(1, threshold // 2)]
    engages_hotspots = bool(hotspot_numbers) and len(hotspot_engagement) >= max(1, len(hotspot_numbers) // 3)

    resolved_mode = mode
    if mode == "auto":
        if engages_hotspots:
            resolved_mode = "hotspot_focus"
            observations.append("Designs already engage the requested hotspot epitope; focusing the crop tightens capacity there.")
        elif engaged:
            resolved_mode = "keep_full_chain_raise_hotspots"
            observations.append("Designs engage an off-target patch; keeping the full target chain and raising hotspot priority instead of cropping to the wrong region.")
        else:
            resolved_mode = "keep_full_chain_raise_hotspots"
            observations.append("No consistent engaged epitope detected; keeping the full target chain.")

    if len(summaries) < min_structures:
        observations.append(f"Only {len(summaries)} structures available (<{min_structures}); deferring an aggressive crop.")
        resolved_mode = "keep_full_chain_raise_hotspots" if resolved_mode not in {"hotspot_focus"} else resolved_mode

    crop_residues: List[int] = []
    if resolved_mode == "hotspot_focus":
        crop_residues = sorted(set(hotspot_numbers) | {n for n in engaged if hotspot_numbers and min(hotspot_numbers) - 2 * margin <= n <= max(hotspot_numbers) + 2 * margin})
    elif resolved_mode == "engaged_focus":
        crop_residues = list(engaged)
    elif resolved_mode == "union":
        crop_residues = sorted(set(engaged) | set(hotspot_numbers))

    recommended: Dict[str, Any] = {}
    crop_window: Optional[List[int]] = None
    target_include: List[Dict[str, Any]] = []
    target_binding_types: List[Dict[str, Any]] = []

    if crop_residues:
        lo = max(1, min(crop_residues) - margin)
        hi = max(crop_residues) + margin
        crop_window = [lo, hi]
        target_include = [{"chain": {"id": target_chain, "res_index": f"{lo}..{hi}"}}]
        binding_numbers = sorted(set(hotspot_numbers) | set(n for n in crop_residues if n in engaged)) or sorted(set(crop_residues))
        target_binding_types = [{"chain": {"id": target_chain, "binding": ",".join(str(n) for n in binding_numbers)}}]
        recommended["target_include"] = target_include
        recommended["target_binding_types"] = target_binding_types
        if structure_groups:
            recommended["structure_groups"] = structure_groups
        observations.append(f"Proposed target crop {lo}..{hi} on chain {target_chain} from {len(crop_residues)} epitope residues.")

    prioritize = [h for h, n in zip(requested_hotspots, (_residue_number(h) for h in requested_hotspots)) if n is not None and counts.get(n, 0) > 0]
    if not prioritize and requested_hotspots:
        # Still surface the requested hotspots so policy can keep pressure on them.
        prioritize = list(requested_hotspots)
    if prioritize:
        # Specific residues remain in the proposal payload for auditability; the
        # executable config uses only a boolean hint so downstream schemas do not
        # confuse a residue list with a toggle.
        recommended["prioritize_hotspots"] = True

    return EpitopeCropProposal(
        mode=resolved_mode,
        engaged_residues=engaged,
        requested_hotspot_residues=hotspot_numbers,
        crop_window=crop_window,
        target_include=target_include,
        target_binding_types=target_binding_types,
        prioritize_hotspots=prioritize,
        recommended_config=recommended,
        observations=observations,
    )
