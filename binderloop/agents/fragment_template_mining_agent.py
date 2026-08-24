
import hashlib
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Union

from binderloop.agents.config_parameter_contract import supported_config_changes
from binderloop.analysis.epitope import propose_epitope_crop
from binderloop.resume import atomic_write_json
from binderloop.execution_governance import stable_digest
from binderloop.templates.outcome_ledger import rank_templates


@dataclass
class FragmentTemplate:
    """Portable local binder fragment evidence for next-round conditioning."""

    template_id: str
    source_structure_file: str
    binder_chain: str
    binder_residue_span: List[int]
    binder_residue_ids: List[str]
    target_contact_residues: List[str]
    hotspot_contacts: Dict[str, int]
    contact_types: Dict[str, int]
    quality_score: float
    quality_label: str
    quality_rank: List[float] = field(default_factory=list)
    gate_failures: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    suggested_action: str = ""
    reuse_mode: str = "neutral"
    compatible_target_patch: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    # Design-level inter-chain PAE (``min_design_to_target_pae``) of the source
    # structure. This is the local interaction-confidence signal used to gate
    # which structures may seed reusable ``preserve`` templates. ``None`` when no
    # PAE data was available for the source.
    interchain_pae: Optional[float] = None
    # Structure-level template payload (used for BoltzGen structure_redesign).
    # ``source_structure_file`` already points at the real cif/pdb that BoltzGen
    # re-tokenizes; the fields below make the fragment a portable artifact.
    binder_sequence: str = ""
    ca_coordinates: List[List[float]] = field(default_factory=list)
    original_source_structure_file: str = ""
    staged_source_structure_file: str = ""
    source_digest: str = ""
    staging_status: str = "not_requested"
    staging_reason: str = ""


@dataclass
class FragmentTemplateBatch:
    round_id: int
    templates: List[FragmentTemplate] = field(default_factory=list)
    template_clusters: List[Dict[str, Any]] = field(default_factory=list)
    recommended_config: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    epitope_crop: Dict[str, Any] = field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    library: List[Dict[str, Any]] = field(default_factory=list)


class FragmentTemplateMiningAgent:
    """Convert structure-level fragment quality signals into reusable templates.

    Beyond per-round mining, the agent maintains a cross-round library of the
    best ``preserve`` templates (the historically best interface regions). It
    can propose an executable target crop (``target_include`` /
    ``target_binding_types``) only when crop_mode is explicitly enabled; the
    harness default is disabled so user target definitions stay consistent.

    Template *eligibility* (which structures may seed reusable ``preserve``
    templates) is controlled by ``gate_metric``:

    * ``"interchain_pae"`` (default): a structure is eligible when its
      design-to-target (inter-chain) PAE is at or below
      ``interchain_pae_max``. Inter-chain PAE measures the *local* confidence of
      the binder-vs-target geometric relationship, so it captures local
      interface quality that the global complex iPTM cannot. This is the
      recommended gate.
    * ``"iptm"``: legacy gate that relies on ``success_structure_files`` (the
      candidate-level iPTM ``pass_compute_gate``). iPTM is a global complex
      score that saturates near zero on hard targets, so it frequently admits
      *no* structures and silently disables the whole exploitation/repair
      feature. Kept for comparison but disabled by default.
    """

    # Inter-chain (design->target) PAE gate, in Angstroms, applied to the
    # *minimum* design-to-target PAE of a structure. Grounded in the canonical
    # de novo binder-design success line ``pae_interaction < 10`` (Bennett et al.
    # 2023; BindCraft uses an equivalent ~11A interface-PAE cutoff). The literature
    # line is defined on the *mean* interface PAE; applied here to the *minimum*
    # inter-chain PAE it conservatively means "the model is confident about at
    # least one local binder<->target docking region" -- exactly the condition for
    # a fragment being worth preserving as a template.
    DEFAULT_INTERCHAIN_PAE_MAX = 10.0

    def mine_templates(
        self,
        structural_analysis: Any,
        *,
        round_id: int,
        max_templates: int = 20,
        prior_templates: Optional[List[Mapping[str, Any]]] = None,
        target_chain: Optional[str] = None,
        requested_hotspots: Optional[List[str]] = None,
        structure_groups: Optional[str] = None,
        crop_mode: str = "disabled",
        library_size: int = 30,
        success_structure_files: Optional[List[str]] = None,
        gate_metric: str = "interchain_pae",
        interchain_pae_by_structure: Optional[Mapping[str, float]] = None,
        interchain_pae_max: float = DEFAULT_INTERCHAIN_PAE_MAX,
        templates_enabled: bool = False,
        template_top_k: int = 1,
        template_artifact_dir: Optional[Union[str, Path]] = None,
        min_template_quality: float = 0.70,
        current_target_structure: Optional[str] = None,
        min_alignment_coverage: float = 0.75,
        max_target_patch_rmsd: float = 2.5,
        require_pae: bool = True,
        max_fixed_fraction: float = 0.5,
        min_designable_residues: int = 8,
        within_proximity: float = 8.0,
        outcome_ledger_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> FragmentTemplateBatch:
        """Mine reusable fragment templates from structure-level quality signals.

        ``gate_metric`` selects the eligibility signal for *preserve* templates:

        * ``"interchain_pae"`` (default): eligibility is decided by the
          per-structure inter-chain PAE supplied via
          ``interchain_pae_by_structure`` (keyed by structure file path);
          a structure is eligible when its PAE ``<= interchain_pae_max``.
        * ``"iptm"``: eligibility falls back to ``success_structure_files`` (the
          candidate-level iPTM success gate).

        ``avoid`` templates are still mined from every structure since negative
        evidence is always useful. When the chosen gate has no usable data
        (e.g. inter-chain PAE requested but ``interchain_pae_by_structure`` is
        empty), the agent falls back to the ``success_structure_files`` set, and
        when that is ``None`` it preserves the legacy behaviour (collect from all
        structures) so offline tests keep working.
        """
        structural = _as_mapping(structural_analysis)
        summaries = [_as_mapping(s) for s in list(structural.get("summaries") or [])]
        success_set = _normalize_structure_keys(success_structure_files) if success_structure_files is not None else None
        gate = str(gate_metric or "interchain_pae").strip().lower()
        pae_map = dict(interchain_pae_by_structure or {})
        # An empty structure batch means the upstream execution/ingestion stage
        # produced nothing to evaluate. Return an empty template batch and let the
        # orchestrator report that execution failure; describing it as missing PAE
        # hides the actionable upstream error. Fail closed only when structures
        # actually exist and would otherwise be considered for template reuse.
        if templates_enabled and summaries and gate == "interchain_pae" and require_pae and not pae_map:
            raise ValueError("fragment_template_not_evaluable:interchain_pae_required_but_missing")
        templates: List[FragmentTemplate] = []
        preserve_eligible_sources: List[str] = []
        for summary in summaries:
            source = str(summary.get("structure_file") or "")
            source_pae = _lookup_interchain_pae(pae_map, source)
            is_eligible = self._preserve_eligible(
                gate=gate,
                source=source,
                source_pae=source_pae,
                interchain_pae_max=interchain_pae_max,
                success_set=success_set,
                has_pae_data=bool(pae_map),
            )
            if is_eligible:
                preserve_eligible_sources.append(source)
                for fragment in list(summary.get("high_quality_fragments") or []):
                    templates.append(self._template_from_fragment(summary, _as_mapping(fragment), reuse_mode="preserve", interchain_pae=source_pae))
            for fragment in list(summary.get("low_quality_fragments") or []):
                templates.append(self._template_from_fragment(summary, _as_mapping(fragment), reuse_mode="avoid", interchain_pae=source_pae))

        templates = sorted(templates, key=_template_quality_key, reverse=True)[:max_templates]
        if templates_enabled and template_artifact_dir is not None:
            self._stage_template_sources(templates, Path(template_artifact_dir), round_id=round_id)
        library = self._merge_library(prior_templates, templates, library_size=library_size)
        executable_pool = list(templates)
        if templates_enabled:
            known_ids = {item.template_id for item in executable_pool}
            for item in library:
                if str(item.get("template_id") or "") in known_ids:
                    continue
                try:
                    executable_pool.append(FragmentTemplate(**dict(item)))
                except (TypeError, ValueError):
                    continue
        clusters = self._cluster_templates(templates)
        recommended_config, analysis_metadata = self._recommended_config(
            executable_pool,
            templates_enabled=templates_enabled,
            template_top_k=template_top_k,
            min_quality=min_template_quality,
            current_target_structure=current_target_structure,
            current_target_chain=target_chain,
            min_alignment_coverage=min_alignment_coverage,
            max_target_patch_rmsd=max_target_patch_rmsd,
            max_fixed_fraction=max_fixed_fraction,
            min_designable_residues=min_designable_residues,
            within_proximity=within_proximity,
            outcome_ledger_snapshot=outcome_ledger_snapshot,
            round_id=round_id,
        )
        analysis_metadata["template_staging"] = {
            "requested": bool(templates_enabled and template_artifact_dir is not None),
            "staged": sum(1 for item in templates if item.staging_status == "staged"),
            "failed": sum(1 for item in templates if item.staging_status == "failed"),
            "artifact_dir": str(template_artifact_dir or ""),
        }
        analysis_metadata["template_gate"] = {
            "gate_metric": gate,
            "interchain_pae_max": float(interchain_pae_max),
            "preserve_eligible_structures": len(preserve_eligible_sources),
            "total_structures": len(summaries),
            "interchain_pae_data_available": bool(pae_map),
            "templates_enabled": bool(templates_enabled),
            "template_top_k": max(1, int(template_top_k or 1)),
        }

        epitope = {}
        crop_mode_normalized = str(crop_mode or "disabled").strip().lower()
        if target_chain and summaries and crop_mode_normalized not in {"disabled", "off", "none", "false", "0"}:
            proposal = propose_epitope_crop(
                summaries,
                target_chain=str(target_chain),
                requested_hotspots=list(requested_hotspots or []),
                mode=crop_mode,
                structure_groups=structure_groups,
            )
            epitope = proposal.to_dict()
            for key, value in proposal.recommended_config.items():
                recommended_config.setdefault(key, value)

        observations = self._observations(templates)
        observations.append(self._gate_observation(analysis_metadata["template_gate"]))
        if epitope.get("observations"):
            observations.extend(epitope["observations"])
        return FragmentTemplateBatch(
            round_id=round_id,
            templates=templates,
            template_clusters=clusters,
            recommended_config=supported_config_changes(recommended_config, include_internal=True),
            observations=observations,
            epitope_crop=epitope,
            analysis_metadata=analysis_metadata,
            library=library,
        )

    @staticmethod
    def _stage_template_sources(templates: List[FragmentTemplate], artifact_dir: Path, *, round_id: int) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for item in templates:
            if item.reuse_mode != "preserve":
                continue
            original = str(item.original_source_structure_file or item.source_structure_file or "")
            item.original_source_structure_file = original
            valid, reason = _packagable_source(original, binder_chain=item.binder_chain, residue_ids=item.binder_residue_ids)
            if not valid:
                item.staging_status = "failed"
                item.staging_reason = reason
                continue
            source = Path(original)
            try:
                digest = _file_digest(source)
                suffix = source.suffix.lower() or ".cif"
                target = artifact_dir / f"{digest}{suffix}"
                if not target.exists():
                    fd, tmp_name = tempfile.mkstemp(prefix=".template_", suffix=suffix, dir=str(artifact_dir))
                    os.close(fd)
                    try:
                        shutil.copy2(str(source), tmp_name)
                        os.replace(tmp_name, str(target))
                    finally:
                        if os.path.exists(tmp_name):
                            os.unlink(tmp_name)
                item.source_digest = digest
                item.staged_source_structure_file = str(target)
                item.source_structure_file = str(target)
                item.staging_status = "staged"
                item.staging_reason = ""
            except (OSError, ValueError) as exc:
                item.staging_status = "failed"
                item.staging_reason = f"staging_error:{exc}"

    @staticmethod
    def _merge_library(prior_templates: Optional[List[Mapping[str, Any]]], templates: List[FragmentTemplate], *, library_size: int) -> List[Dict[str, Any]]:
        """Keep the globally best ``preserve`` templates across rounds, deduped by id."""
        merged: Dict[str, Dict[str, Any]] = {}
        for item in list(prior_templates or []):
            item = dict(_as_mapping(item))
            tid = str(item.get("template_id") or "")
            if tid and str(item.get("reuse_mode")) == "preserve":
                merged[tid] = item
        for template in templates:
            if template.reuse_mode != "preserve":
                continue
            existing = merged.get(template.template_id)
            if existing is None or _template_quality_key(template) > _template_quality_key(existing):
                merged[template.template_id] = asdict(template)
        ordered = sorted(merged.values(), key=_template_quality_key, reverse=True)
        return ordered[: max(1, library_size)]

    def write_templates(self, batch: FragmentTemplateBatch, path: Union[str, Path]) -> Path:
        return atomic_write_json(path, asdict(batch))

    @staticmethod
    def _preserve_eligible(
        *,
        gate: str,
        source: str,
        source_pae: Optional[float],
        interchain_pae_max: float,
        success_set: Optional[Set[str]],
        has_pae_data: bool,
    ) -> bool:
        """Decide whether ``source`` may seed a reusable ``preserve`` template.

        The default ``interchain_pae`` gate uses the design-to-target PAE: a
        structure is eligible only when its inter-chain PAE is confident
        (``<= interchain_pae_max``). When inter-chain PAE was requested but no
        PAE data is available the gate degrades to the legacy success-set
        behaviour so offline callers/tests still mine templates.
        """
        if gate in {"none", "all", "off", "disabled"}:
            return True
        if gate == "iptm":
            return success_set is None or _structure_in_success_set(source, success_set)
        # Default: inter-chain PAE gate. Preserve eligibility is controlled only
        # by the configured local interaction-confidence threshold; foldability
        # regressions are handled later by the template allocation guard.
        if has_pae_data:
            return source_pae is not None and float(source_pae) <= float(interchain_pae_max)
        # No PAE data: fall back to the success gate (None => all eligible).
        return success_set is None or _structure_in_success_set(source, success_set)

    @staticmethod
    def _gate_observation(gate_info: Mapping[str, Any]) -> str:
        metric = str(gate_info.get("gate_metric"))
        eligible = int(gate_info.get("preserve_eligible_structures") or 0)
        total = int(gate_info.get("total_structures") or 0)
        if metric == "interchain_pae":
            return (
                f"Preserve-template eligibility gated by inter-chain PAE "
                f"(min_design_to_target_pae <= {gate_info.get('interchain_pae_max')}); "
                f"{eligible}/{total} structures eligible. iPTM gate is disabled because the global "
                f"complex iPTM does not reflect local interface confidence."
            )
        return f"Preserve-template eligibility gated by iPTM success gate; {eligible}/{total} structures eligible."

    def _template_from_fragment(self, summary: Mapping[str, Any], fragment: Mapping[str, Any], *, reuse_mode: str, interchain_pae: Optional[float] = None) -> FragmentTemplate:
        residue_ids = [str(item) for item in fragment.get("residue_ids") or []]
        contacts = _contacts_for_fragment(summary, residue_ids)
        target_contacts = sorted({str(contact.get("target_residue")) for contact in contacts if contact.get("target_residue")})
        hotspot_contacts = _hotspot_contacts_for_fragment(summary, target_contacts)
        contact_types: Dict[str, int] = {}
        for contact in contacts:
            ctype = str(contact.get("contact_type") or "unknown")
            contact_types[ctype] = contact_types.get(ctype, 0) + 1
        quality_label = str(fragment.get("quality_label") or ("high" if reuse_mode == "preserve" else "low"))
        evidence = [str(item) for item in fragment.get("reasons") or []]
        risk_flags = [item for item in evidence if "clash" in item or "break" in item or "weak" in item]
        source = str(summary.get("structure_file") or "")
        fragment_id = str(fragment.get("fragment_id") or "-".join(residue_ids))
        return FragmentTemplate(
            template_id=_template_id(source, fragment_id, quality_label, reuse_mode),
            source_structure_file=source,
            binder_chain=str(summary.get("binder_chain") or ""),
            binder_residue_span=[
                int(fragment.get("start_residue") or 0),
                int(fragment.get("end_residue") or 0),
            ],
            binder_residue_ids=residue_ids,
            target_contact_residues=target_contacts,
            hotspot_contacts=hotspot_contacts,
            contact_types=contact_types,
            quality_score=float(fragment.get("quality_score") or 0.0),
            quality_label=quality_label,
            quality_rank=[float(value) for value in (fragment.get("quality_rank") or [])],
            gate_failures=[str(value) for value in (fragment.get("gate_failures") or [])],
            evidence=evidence,
            suggested_action=str(fragment.get("suggested_action") or ""),
            reuse_mode=reuse_mode,
            compatible_target_patch=target_contacts,
            risk_flags=risk_flags,
            binder_sequence=str(fragment.get("sequence") or ""),
            ca_coordinates=[list(c) for c in (fragment.get("ca_coordinates") or [])],
            interchain_pae=(float(interchain_pae) if interchain_pae is not None else None),
        )

    @staticmethod
    def _cluster_templates(templates: List[FragmentTemplate]) -> List[Dict[str, Any]]:
        clusters: Dict[str, Dict[str, Any]] = {}
        for template in templates:
            patch_key = ",".join(template.compatible_target_patch[:5]) or "no_target_contacts"
            key = f"{template.reuse_mode}:{patch_key}"
            cluster = clusters.setdefault(
                key,
                {
                    "cluster_id": _short_hash(key),
                    "reuse_mode": template.reuse_mode,
                    "compatible_target_patch": template.compatible_target_patch[:5],
                    "template_ids": [],
                    "best_quality_score": 0.0,
                },
            )
            cluster["template_ids"].append(template.template_id)
            cluster["best_quality_score"] = max(float(cluster["best_quality_score"]), template.quality_score)
        return sorted(clusters.values(), key=lambda item: float(item.get("best_quality_score") or 0.0), reverse=True)

    @staticmethod
    def _recommended_config(
        templates: List[FragmentTemplate],
        *,
        templates_enabled: bool = False,
        template_top_k: int = 1,
        min_quality: float = 0.70,
        current_target_structure: Optional[str] = None,
        current_target_chain: Optional[str] = None,
        min_alignment_coverage: float = 0.75,
        max_target_patch_rmsd: float = 2.5,
        max_fixed_fraction: float = 0.5,
        min_designable_residues: int = 8,
        within_proximity: float = 8.0,
        outcome_ledger_snapshot: Optional[Mapping[str, Any]] = None,
        round_id: int = 0,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        high = [template.template_id for template in templates if template.reuse_mode == "preserve"][:5]
        low = [template.template_id for template in templates if template.reuse_mode == "avoid"][:5]
        config: Dict[str, Any] = {}
        analysis_metadata: Dict[str, Any] = {}
        if high:
            analysis_metadata["pae_gated_preserve_fragment_count"] = len(high)
        if low:
            analysis_metadata["avoid_fragment_modules"] = low
        binder_templates = FragmentTemplateMiningAgent._structure_redesign_templates(
            templates,
            top_k=max(1, int(template_top_k or 1)),
            min_quality=float(min_quality),
            within_proximity=float(within_proximity),
            current_target_structure=current_target_structure,
            current_target_chain=current_target_chain,
            min_alignment_coverage=float(min_alignment_coverage),
            max_target_patch_rmsd=float(max_target_patch_rmsd),
            max_fixed_fraction=float(max_fixed_fraction),
            min_designable_residues=int(min_designable_residues),
            outcome_ledger_snapshot=outcome_ledger_snapshot,
            round_id=round_id,
        ) if templates_enabled else []
        analysis_metadata["template_evaluation_policy"] = {
            "current_target_structure": str(current_target_structure or ""),
            "current_target_chain": str(current_target_chain or ""),
            "min_alignment_coverage": float(min_alignment_coverage),
            "max_target_patch_rmsd": float(max_target_patch_rmsd),
            "max_fixed_fraction": float(max_fixed_fraction),
            "min_designable_residues": int(min_designable_residues),
            "within_proximity": float(within_proximity),
            "alignment_status": "evaluated" if binder_templates else ("not_evaluable" if templates_enabled else "not_applicable"),
        }
        if binder_templates:
            config["binder_templates"] = binder_templates
            # Backward-compatible single-template field; the active learner will
            # prefer binder_templates when present.
            config["binder_template"] = binder_templates[0]
            analysis_metadata["template_insertion_decision"] = {
                "insert_template": True,
                "template_id": binder_templates[0].get("template_id"),
                "template_ids": [item.get("template_id") for item in binder_templates],
                "policy": "top_k_packagable_pae_gated_preserve_fragments",
                "template_free_control_required": True,
            }
        else:
            analysis_metadata["template_insertion_decision"] = {
                "insert_template": False,
                "reason": (
                    "fragment_templates_disabled"
                    if not templates_enabled
                    else "no packagable PAE-gated preserve fragment passed template quality/span checks"
                ),
            }
        return supported_config_changes(config, include_internal=True), analysis_metadata

    @staticmethod
    def _structure_redesign_templates(
        templates: List[FragmentTemplate], *, top_k: int = 1, min_quality: float = 0.70, within_proximity: float = 8.0,
        current_target_structure: Optional[str] = None, current_target_chain: Optional[str] = None,
        min_alignment_coverage: float = 0.75, max_target_patch_rmsd: float = 2.5,
        max_fixed_fraction: float = 0.5, min_designable_residues: int = 8,
        outcome_ledger_snapshot: Optional[Mapping[str, Any]] = None, round_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """Pick the top-k usable preserve fragments and convert each into a
        BoltzGen structure-redesign template description.

        Sources are accepted by content and packagability, not by directory name.
        Standard prior-round Taiji output paths are valid when they still exist on
        the submitting host and can be staged into the current run artifact store."""
        candidates = [
            t
            for t in templates
            if t.reuse_mode == "preserve"
            and not t.gate_failures
            and (_template_quality_key(t)[0] > 0)
            and float(t.quality_score) >= float(min_quality)
            and t.source_structure_file
            and _is_mountable_source(t.source_structure_file, binder_chain=t.binder_chain, residue_ids=t.binder_residue_ids)
            and t.binder_chain
            and len(t.binder_residue_span) == 2
            and t.binder_residue_span[1] >= t.binder_residue_span[0] > 0
        ]
        if not candidates:
            return []
        # Prefer the most locally confident interface (lowest inter-chain PAE),
        # then the highest geometric quality. When no PAE is available the PAE
        # key is +inf for every candidate, so selection reduces to quality.
        def _selection_key(t: FragmentTemplate) -> tuple:
            pae = float(t.interchain_pae) if t.interchain_pae is not None else float("inf")
            return (pae, tuple(-value for value in _template_quality_key(t)))

        ledger_order = rank_templates(
            [{**asdict(item), "target_compatibility": max(0.0, 1.0 - (float(item.interchain_pae) / 10.0 if item.interchain_pae is not None else 0.5))} for item in candidates],
            outcome_ledger_snapshot, top_k=len(candidates), round_id=round_id,
        )
        by_id = {item.template_id: item for item in candidates}
        ordered_candidates = [by_id[str(item.get("template_id"))] for item in ledger_order if str(item.get("template_id")) in by_id]
        selected: List[FragmentTemplate] = []
        seen_sources: Set[str] = set()
        seen_patches: Set[str] = set()
        for candidate in ordered_candidates:
            source_key = str(candidate.source_digest or candidate.source_structure_file)
            patch_key = ",".join(candidate.compatible_target_patch[:5])
            # Prefer source and target-patch diversity. If this would leave fewer
            # than Top-K entries, a second pass below fills from the remaining rank.
            if source_key in seen_sources or (patch_key and patch_key in seen_patches):
                continue
            selected.append(candidate)
            seen_sources.add(source_key)
            if patch_key:
                seen_patches.add(patch_key)
            if len(selected) >= max(1, int(top_k or 1)):
                break
        if len(selected) < max(1, int(top_k or 1)):
            for candidate in ordered_candidates:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= max(1, int(top_k or 1)):
                    break
        out: List[Dict[str, Any]] = []
        for item in selected:
            lo, hi = int(item.binder_residue_span[0]), int(item.binder_residue_span[1])
            exact_res_index = _residue_ids_to_range(item.binder_residue_ids, chain=item.binder_chain) or f"{lo}..{hi}"
            source_length = _chain_residue_count(item.source_structure_file, item.binder_chain)
            fixed_count = len(_residue_numbers(item.binder_residue_ids)) or max(1, hi - lo + 1)
            fixed_fraction = float(fixed_count) / max(1, source_length)
            designable_count = max(0, source_length - fixed_count)
            # A redesign template that fixes most of the binder is not an
            # optimization branch; it is effectively a replay with no freedom.
            if item.staging_status == "staged" and (source_length <= 0 or fixed_fraction > float(max_fixed_fraction) or designable_count < int(min_designable_residues)):
                continue
            # Production template execution requires a measured source-target to
            # current-target transform. Missing target structures/patches are
            # explicitly not evaluable; identity alignment is never fabricated.
            if not current_target_structure or not current_target_chain or not item.target_contact_residues:
                continue
            source_target_chains = {str(value).split(":", 1)[0] for value in item.target_contact_residues if ":" in str(value)}
            if len(source_target_chains) != 1:
                continue
            try:
                from binderloop.analysis.template_alignment import align_target_patch, write_aligned_binder_template
                alignment = align_target_patch(
                    item.source_structure_file, str(current_target_structure),
                    source_target_chain=next(iter(source_target_chains)),
                    current_target_chain=str(current_target_chain),
                    residue_ids=item.target_contact_residues,
                    min_coverage=float(min_alignment_coverage),
                    max_rmsd=float(max_target_patch_rmsd),
                ).to_dict()
            except (ImportError, OSError, ValueError, RuntimeError):
                continue
            if alignment.get("status") != "aligned":
                continue
            source_binder_residue_ids = _chain_residue_ids(item.source_structure_file, item.binder_chain)
            aligned_source = item.source_structure_file
            try:
                aligned_path = Path(item.source_structure_file).with_name(f"{Path(item.source_structure_file).stem}.{alignment['digest'][:12]}.aligned.pdb")
                aligned_source = write_aligned_binder_template(item.source_structure_file, str(aligned_path), binder_chain=item.binder_chain, alignment=alignment)
            except (OSError, ValueError):
                continue
            template: Dict[str, Any] = {
                "mode": "structure_redesign",
                "template_id": item.template_id,
                "source_structure_file": aligned_source,
                "coherent_frame_source_structure_file": aligned_source,
                "unaligned_source_structure_file": item.source_structure_file,
                "binder_chain": item.binder_chain,
                "fixed_res_index": exact_res_index,
                "binder_residue_ids": list(item.binder_residue_ids),
                "within_proximity": within_proximity,
                "source_binder_length": source_length,
                "source_binder_residue_ids": source_binder_residue_ids,
                "max_fixed_fraction": float(max_fixed_fraction),
                "min_designable_residues": int(min_designable_residues),
                "fixed_residue_count": fixed_count,
                "designable_residue_count": designable_count,
                "fixed_fraction": round(fixed_fraction, 4),
                "binder_structure_groups": [{"group": {"visibility": 1, "id": item.binder_chain, "res_index": exact_res_index}}],
                "structure_conditioning_semantics": "soft_exact_motif_shape",
                "original_source_structure_file": item.original_source_structure_file or item.source_structure_file,
                "staged_source_structure_file": item.staged_source_structure_file or item.source_structure_file,
                "source_digest": _file_digest(Path(aligned_source)),
                "unaligned_source_digest": item.source_digest,
                "target_alignment": alignment,
                "alignment_digest": alignment.get("digest"),
                "plan_input_digest": stable_digest({"template_id": item.template_id, "source_digest": item.source_digest, "alignment_digest": alignment.get("digest"), "binder_residue_ids": item.binder_residue_ids}),
                "source_target_identity": {"structure": item.source_structure_file, "chain": next(iter(source_target_chains)), "digest": alignment.get("source_target_identity_digest")},
                "current_target_identity": {"structure": str(current_target_structure), "chain": str(current_target_chain), "digest": alignment.get("current_target_identity_digest")},
                "staging_status": item.staging_status,
                "quality_score": round(float(item.quality_score), 3),
                "quality_rank": list(item.quality_rank),
                "gate_failures": list(item.gate_failures),
            }
            if item.interchain_pae is not None:
                template["interchain_pae"] = round(float(item.interchain_pae), 3)
            out.append(template)
        return out

    @staticmethod
    def _structure_redesign_template(
        templates: List[FragmentTemplate], *, min_quality: float = 0.70, within_proximity: float = 8.0
    ) -> Optional[Dict[str, Any]]:
        """Compatibility wrapper for callers expecting one template."""
        selected = FragmentTemplateMiningAgent._structure_redesign_templates(
            templates, top_k=1, min_quality=min_quality, within_proximity=within_proximity
        )
        return selected[0] if selected else None

    @staticmethod
    def _observations(templates: List[FragmentTemplate]) -> List[str]:
        if not templates:
            return ["No reusable fragment templates were mined from structure evaluation outputs."]
        high = sum(1 for template in templates if template.reuse_mode == "preserve")
        low = sum(1 for template in templates if template.reuse_mode == "avoid")
        return [f"Mined {len(templates)} fragment templates: preserve={high}, avoid={low}."]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _template_quality_key(value: Any) -> tuple:
    """Read canonical fragment rank, with old quality fields as a fallback."""
    item = _as_mapping(value)
    rank = item.get("quality_rank")
    if isinstance(rank, (list, tuple)) and rank:
        try:
            return tuple(float(part) for part in rank)
        except (TypeError, ValueError):
            pass
    label = str(item.get("quality_label") or "")
    score = float(item.get("quality_score") or 0.0)
    return (1.0 if label == "high" and score >= 0.70 else 0.0, score)


def _contacts_for_fragment(summary: Mapping[str, Any], residue_ids: List[str]) -> List[Mapping[str, Any]]:
    residue_set = set(residue_ids)
    contacts = []
    for contact in list(summary.get("contacts_preview") or []):
        item = _as_mapping(contact)
        if str(item.get("binder_residue")) in residue_set:
            contacts.append(item)
    return contacts


def _hotspot_contacts_for_fragment(summary: Mapping[str, Any], target_contacts: List[str]) -> Dict[str, int]:
    target_set = set(target_contacts)
    out: Dict[str, int] = {}
    for hotspot, count in dict(summary.get("hotspot_contacts") or {}).items():
        if hotspot in target_set or int(count or 0) > 0:
            out[str(hotspot)] = int(count or 0)
    return out


def _template_id(source: str, fragment_id: str, quality_label: str, reuse_mode: str) -> str:
    return "frag_" + _short_hash(", ".join([source, fragment_id, quality_label, reuse_mode]))


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


_SUPPORTED_TEMPLATE_SUFFIXES = {".cif", ".mmcif", ".pdb"}


def _packagable_source(source: str, *, binder_chain: str = "", residue_ids: Optional[List[str]] = None) -> tuple[bool, str]:
    if not source:
        return False, "missing_source"
    path = Path(str(source))
    try:
        if not path.exists() or not path.is_file():
            return False, "source_missing_or_not_file"
        if path.suffix.lower() not in _SUPPORTED_TEMPLATE_SUFFIXES:
            return False, "unsupported_structure_format"
        if path.stat().st_size <= 0:
            return False, "empty_structure_file"
        from binderloop.analysis.structure_features import parse_structure
        atoms = parse_structure(path)
        if not atoms:
            return False, "unparseable_structure"
        if binder_chain and binder_chain not in {atom.chain for atom in atoms}:
            return False, "binder_chain_missing"
        if residue_ids:
            present = {atom.residue_id for atom in atoms if not binder_chain or atom.chain == binder_chain}
            missing = [value for value in residue_ids if str(value) not in present]
            if missing:
                return False, "binder_residues_missing:" + ",".join(map(str, missing[:5]))
        return True, ""
    except OSError as exc:
        return False, f"source_io_error:{exc}"


def _is_mountable_source(source: str, *, binder_chain: str = "", residue_ids: Optional[List[str]] = None) -> bool:
    """Compatibility eligibility check used before optional stable staging.

    Content/chain/residue validation is authoritative when staging is requested.
    Direct library/test callers without an artifact store retain path-level
    eligibility and are revalidated by DesignSpecAgent before submission.
    """
    if not source:
        return False
    try:
        path = Path(str(source))
        return path.exists() and path.is_file() and path.suffix.lower() in _SUPPORTED_TEMPLATE_SUFFIXES and path.stat().st_size > 0
    except OSError:
        return False


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _residue_numbers(residue_ids: List[str]) -> List[int]:
    values: List[int] = []
    for token in residue_ids or []:
        text = str(token).split(":", 1)[-1]
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            values.append(int(digits))
    return sorted(set(values))


def _residue_ids_to_range(residue_ids: List[str], *, chain: str = "") -> str:
    values = _residue_numbers([token for token in residue_ids or [] if not chain or str(token).startswith(chain + ":")])
    if not values:
        return ""
    groups: List[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append(str(start) if start == prev else f"{start}..{prev}")
        start = prev = value
    groups.append(str(start) if start == prev else f"{start}..{prev}")
    return ",".join(groups)



def _chain_residue_ids(source: str, chain: str) -> List[str]:
    try:
        from binderloop.analysis.structure_features import parse_structure
        seen = []
        for atom in parse_structure(source):
            if atom.chain == chain and atom.residue_id not in seen:
                seen.append(atom.residue_id)
        return seen
    except (OSError, ValueError):
        return []


def _chain_residue_count(source: str, chain: str) -> int:
    try:
        from binderloop.analysis.structure_features import parse_structure
        return len({atom.residue_id for atom in parse_structure(source) if atom.chain == chain})
    except (OSError, ValueError):
        return 0


def _normalize_structure_keys(structure_files: Optional[List[str]]) -> Set[str]:
    """Build a lookup set of normalized keys (basename + stem) for matching."""
    keys: Set[str] = set()
    for item in structure_files or []:
        text = str(item or "")
        if not text:
            continue
        name = Path(text).name
        keys.add(name)
        keys.add(Path(text).stem)
        keys.add(text)
    return keys


def _lookup_interchain_pae(pae_map: Mapping[str, float], source: str) -> Optional[float]:
    """Resolve the inter-chain PAE for ``source`` from a path-keyed PAE map.

    The map is normally keyed by the exact structure file path, but we also try
    basename/stem and embedded-id substring matches so the lookup is robust to
    rank-prefixed file names (e.g. ``rank1_<design_id>.cif``)."""
    if not source or not pae_map:
        return None
    if source in pae_map:
        return _coerce_pae(pae_map[source])
    name = Path(source).name
    stem = Path(source).stem
    for key in (name, stem):
        if key in pae_map:
            return _coerce_pae(pae_map[key])
    # Suffix match (boundary-safe): the structure stem ends with the keyed design
    # id, or vice-versa. Avoids ``_3_1`` falsely matching ``_3_10``.
    for key, value in pae_map.items():
        key_stem = Path(str(key)).stem
        if not key_stem:
            continue
        if stem.endswith(key_stem) or key_stem.endswith(stem):
            return _coerce_pae(value)
    return None


def _coerce_pae(value: Any) -> Optional[float]:
    """Coerce a PAE value to float, dropping missing/sentinel (>=1000) values."""
    try:
        pae = float(value)
    except (TypeError, ValueError):
        return None
    if pae <= 0.0 or pae >= 1000.0:
        # Non-physical / sentinel values (BoltzGen emits ~100000 when an
        # interaction is absent); treat as "no usable inter-chain PAE".
        return None
    return pae


def _structure_in_success_set(source: str, success_set: Set[str]) -> bool:
    """Match a structure summary's source path against the success key set."""
    if not source:
        return False
    if source in success_set:
        return True
    name = Path(source).name
    stem = Path(source).stem
    if name in success_set or stem in success_set:
        return True
    # Fall back to substring matching against any success key (design id may be
    # embedded in the structure file name, e.g. ``final_20_d3_seed42.cif``).
    for key in success_set:
        if key and (key in source or source in key):
            return True
    return False
