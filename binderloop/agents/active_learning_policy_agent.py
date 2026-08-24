from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Optional, List, Union

from binderloop.agents.config_parameter_contract import supported_config_changes
from binderloop.parameter_decision import PROBABILISTIC_SAMPLER_KEYS

from .evaluation_agent import EvaluationSummary
from binderloop.resume import atomic_write_json


@dataclass
class NextRoundParameterProposal:
    round_id: int
    params_update: Dict[str, Any]
    rationale: List[str] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    final_params_update: Dict[str, Any] = field(default_factory=dict)
    applied_params_update: Dict[str, Any] = field(default_factory=dict)


class ActiveLearningPolicyAgent:
    """Failure-aware rule policy for the next design round."""

    def propose_next_params(
        self,
        summary: EvaluationSummary,
        current_params: Mapping[str, Any],
        *,
        round_id: int = 1,
        model: Optional[str] = None,
        structural_summary: Optional[Any] = None,
        hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
        quality_analysis: Optional[Mapping[str, Any]] = None,
        diagnostic_report: Optional[Mapping[str, Any]] = None,
        memory_summary: Optional[Mapping[str, Any]] = None,
        max_binders_per_round: Optional[int] = None,
        active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> NextRoundParameterProposal:
        model_name = str(model or current_params.get("search_profile_model") or "boltzgen").strip().lower()
        proposal = self.propose_next_boltzgen_params(
            summary,
            current_params,
            round_id=round_id,
            structural_summary=structural_summary,
            hypotheses=hypotheses,
            quality_analysis=quality_analysis,
            diagnostic_report=diagnostic_report,
            memory_summary=memory_summary,
            max_binders_per_round=max_binders_per_round,
            active_skills=active_skills,
        )
        try:
            from binderloop.models.search_profile import get_model_search_profile
            profile = get_model_search_profile(model_name)
        except Exception:
            return proposal
        allowed = frozenset(profile.adjustable_parameters) | frozenset({"binder_lengths"})
        sampler = frozenset(profile.sampler_axes)
        proposal.params_update = {
            key: value
            for key, value in dict(proposal.params_update or {}).items()
            if key in allowed and key not in sampler and key not in profile.forbidden_keys
        }
        metadata = dict(proposal.analysis_metadata or {})
        directions = dict(metadata.get("probabilistic_sampler_directions") or {})
        if directions:
            kept = {key: value for key, value in directions.items() if key in sampler}
            if model_name == "rfd3" and "alpha" in directions and "gamma_0" not in kept:
                kept["gamma_0"] = directions["alpha"]
            metadata["probabilistic_sampler_directions"] = kept
            proposal.analysis_metadata = metadata
        return proposal

    def propose_next_boltzgen_params(
        self,
        summary: EvaluationSummary,
        current_params: Mapping[str, Any],
        *,
        round_id: int = 1,
        structural_summary: Optional[Any] = None,
        hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
        quality_analysis: Optional[Mapping[str, Any]] = None,
        diagnostic_report: Optional[Mapping[str, Any]] = None,
        memory_summary: Optional[Mapping[str, Any]] = None,
        max_binders_per_round: Optional[int] = None,
        active_skills: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> NextRoundParameterProposal:
        current = dict(current_params)
        params = dict(current)
        tags = dict(summary.tag_counts)
        total = max(1, summary.total_candidates)
        cap = max(1, int(max_binders_per_round or params.get("max_binders_per_round") or params.get("num_designs") or 20))
        params["max_binders_per_round"] = cap
        rationale: List[str] = []
        analysis_metadata: Dict[str, Any] = {}
        typed_proposals: Dict[str, List[Dict[str, Any]]] = {}
        active_skills = list(active_skills or [])
        if active_skills:
            analysis_metadata["activated_skills"] = [
                {"id": skill.get("id"), "type": skill.get("type"), "trigger_reason": skill.get("trigger_reason")}
                for skill in active_skills
            ]

        structural_tags = getattr(structural_summary, "aggregate_tags", {}) if structural_summary is not None else {}
        structural_items = getattr(structural_summary, "summaries", []) if structural_summary is not None else []
        hypothesis_names = " ".join(str(h.get("name", "")) for h in (hypotheses or [])).lower()
        quality_analysis = dict(quality_analysis or {})
        diagnostic_report = dict(diagnostic_report or {})
        memory_summary = dict(memory_summary or {})
        high_modules = list(quality_analysis.get("high_quality_modules") or [])
        low_modules = list(quality_analysis.get("low_quality_modules") or [])
        guidance_items = list(quality_analysis.get("next_round_guidance", []) or [])
        guidance_text = " ".join(str(g.get("action", "")) + " " + str(g.get("parameter_or_constraint_change", "")) for g in guidance_items).lower()
        foldability = _foldability_metrics(summary)
        foldability_risk = _foldability_risk(foldability, tags, total)
        crop_locked_disabled = _crop_locked_disabled(params)

        if summary.total_candidates == 0:
            params["diffusion_batch_size"] = 1
            if structural_tags.get("hotspot_not_covered", 0) or "hotspot" in hypothesis_names:
                analysis_metadata["binding_site_intent"] = "site_primary_condition"
            if structural_tags.get("interface_clash_risk", 0) or "clash" in hypothesis_names:
                analysis_metadata["selection_intent"] = "clash_select"
            rationale.append("No metrics collected: keep run small, preserve intermediates, and use structure/hypothesis fallback signals if available.")
            return NextRoundParameterProposal(round_id=round_id, params_update=_sparse_executable_delta(current, params), rationale=rationale, analysis_metadata=analysis_metadata)

        for source_type, items, change_field, label_field in (
            ("diagnostic", diagnostic_report.get("corrective_actions", []) or [], "parameter_changes", "action"),
            ("quality", guidance_items, "config_parameter_changes", "action"),
            ("hypothesis", hypotheses or [], "config_parameter_changes", "name"),
        ):
            for source_index, item in enumerate(items):
                changes = supported_config_changes(item.get(change_field) or {})
                for key, value in changes.items():
                    if key in PROBABILISTIC_SAMPLER_KEYS:
                        continue
                    typed_proposals.setdefault(key, []).append({
                        "proposal_type": source_type, "source_index": source_index,
                        "source_label": str(item.get(label_field) or source_type),
                        "key": key, "value": value,
                        "evidence_finding_ids": list(item.get("evidence_finding_ids") or item.get("evidence_ids") or []),
                    })
        if typed_proposals:
            analysis_metadata["typed_parameter_proposals"] = typed_proposals
            analysis_metadata["proposal_conflicts"] = [
                {"key": key, "proposals": rows}
                for key, rows in typed_proposals.items()
                if len({repr(row.get("value")) for row in rows}) > 1
            ]
            for key, rows in typed_proposals.items():
                values = {repr(row.get("value")) for row in rows}
                if len(values) == 1:
                    params[key] = rows[0]["value"]
                    rationale.append(f"Concordant typed proposals selected {key} from {', '.join(sorted({r['proposal_type'] for r in rows}))}.")
                else:
                    rationale.append(f"Conflicting typed proposals for {key} were held for orchestrator arbitration; no last-writer overwrite occurred.")

        if tags.get("hotspot_miss", 0) / total > 0.3:
            params["config_overrides"] = supported_config_changes(
                {"config_overrides": params.get("config_overrides", [])}
            ).get("config_overrides", [])
            if ["filtering", "filter_bindingsite=true"] not in params["config_overrides"]:
                params["config_overrides"].append(["filtering", "filter_bindingsite=true"])
            rationale.append("Hotspot miss dominated: enforce binding-site-aware filtering without changing user-owned hard filter thresholds.")

        if tags.get("folding_failure", 0) / total > 0.3:
            analysis_metadata.setdefault("probabilistic_sampler_directions", {})["noise_scale"] = "decrease"
            rationale.append("Folding failure dominated: record a lower-noise direction for probabilistic resolution; do not mutate the numeric sampler value.")

        if tags.get("binding_pose_failure", 0) / total > 0.3:
            params["diffusion_batch_size"] = 1
            rationale.append("Binding pose failure dominated: keep sampling batches small for pose diversity without changing user-owned budget.")

        if tags.get("diversity_collapse", 0) / total > 0.3:
            analysis_metadata.setdefault("probabilistic_sampler_directions", {})["alpha"] = "increase"
            params["diffusion_batch_size"] = 1
            rationale.append("Diversity collapse dominated: record an alpha-increase direction for probabilistic resolution and use batch size 1.")


        if structural_tags.get("hotspot_not_covered", 0) or "hotspot" in hypothesis_names:
            if not crop_locked_disabled:
                params["epitope_crop_mode"] = "hotspot_focus"
            contacted_hotspots = _contacted_hotspots(structural_items)
            if contacted_hotspots:
                analysis_metadata["contacted_primary_residues"] = contacted_hotspots
            auxiliary = _auxiliary_hotspots(structural_items, current.get("hotspots") or [])
            if auxiliary:
                params["auxiliary_hotspots"] = auxiliary
            analysis_metadata["binding_site_intent"] = "site_expanded_condition" if auxiliary else "site_primary_condition"
            rationale.append("Structure/hypothesis evidence suggests binding-site coverage risk: propose only validated nearby residues; the typed arm and resolver own materialization.")

        if foldability_risk:
            if str(params.get("epitope_crop_mode", "")).strip().lower() in {"hotspot_focus", "engaged_focus"}:
                params["epitope_crop_mode"] = "disabled"
            broadened = _broaden_binder_lengths(params)
            if broadened:
                params["binder_lengths"] = broadened
            if params.get("binder_template") or params.get("binder_templates"):
                current_fraction = _float_param(params.get("template_conditioned_fraction"), 0.5)
                params["template_conditioned_fraction"] = max(0.1, round(current_fraction * 0.5, 3))
                analysis_metadata["template_allocation_guard"] = {
                    "reason": "foldability_or_refolding_regression",
                    "previous_fraction": current_fraction,
                    "next_fraction": params["template_conditioned_fraction"],
                    "foldability_metrics": foldability,
                }
            rationale.append(
                "Foldability/refolding risk detected: reduce hotspot/crop pressure, broaden allowed binder lengths when possible, "
                "and lower template-conditioned allocation if a template was active."
            )

        if structural_tags.get("interface_clash_risk", 0) or "clash" in hypothesis_names:
            analysis_metadata["selection_intent"] = "clash_select"
            rationale.append("Coordinate-level clash risk detected: request measured heavy-atom clash selection, not a model repair flag.")

        if structural_tags.get("weak_or_tiny_interface", 0) or "pose" in hypothesis_names:
            params["diffusion_batch_size"] = 1
            rationale.append("Weak interface/pose hypothesis detected: keep sampling batches small without changing user-owned length or budget.")

        if structural_tags.get("binder_chain_break", 0) or tags.get("folding_failure", 0) / total > 0.3:
            rationale.append("Coordinate/folding evidence suggests foldability risk; inverse-fold/refolding settings are user-owned and left unchanged.")

        if structural_tags.get("over_hydrophobic_interface", 0):
            params["inverse_fold_avoid"] = params.get("inverse_fold_avoid") or "C"
            rationale.append("Over-hydrophobic interface risk detected: keep sequence design conservative without changing hard filter thresholds.")

        if high_modules:
            rationale.append("Per-round quality analysis found reusable high-quality fragments; FragmentTemplateMiningAgent will decide template use from PAE-gated fragments.")

        if low_modules or "repair_low_quality" in guidance_text:
            low_module_ids = [m.get("module_id") for m in low_modules[:5] if m.get("module_id")]
            analysis_metadata["avoid_fragment_modules"] = low_module_ids
            if "clash" in guidance_text:
                analysis_metadata["selection_intent"] = "clash_select"
            rationale.append("Per-round quality analysis found low-quality modules; recorded them as analysis metadata and applied repair-oriented filters without emitting avoid_fragment_modules as executable config.")

        failed_lengths = _failed_lengths(memory_summary)
        if failed_lengths:
            if any(count >= 2 for count in failed_lengths.values()):
                rationale.append("ExperimentMemoryStore shows repeated zero-success rounds for some lengths; keep user-owned length policy unchanged.")

        if summary.success_count > 0:
            rationale.append("At least one pass candidate found: keep user-owned sample budget unchanged and tune only BoltzGen knobs.")
        else:
            params["diffusion_batch_size"] = 1
            rationale.append("No pass candidate found: keep user-owned sample budget unchanged and use small sampling batches for diagnostics.")

        if crop_locked_disabled:
            params["epitope_crop_mode"] = "disabled"
        return NextRoundParameterProposal(round_id=round_id, params_update=_sparse_executable_delta(current, params), rationale=rationale, analysis_metadata=analysis_metadata)

    def write_proposal(self, proposal: NextRoundParameterProposal, path: Union[str, Path]) -> Path:
        return atomic_write_json(path, asdict(proposal))


def _contacted_hotspots(structural_items: Sequence[Mapping[str, Any]]) -> List[str]:
    counts: Dict[str, int] = {}
    for item in structural_items or []:
        for hotspot, count in (item.get("hotspot_contacts") or {}).items():
            if int(count or 0) > 0:
                counts[hotspot] = counts.get(hotspot, 0) + int(count)
    return sorted(counts, key=lambda h: counts[h], reverse=True)[:5]


def _auxiliary_hotspots(structural_items: Sequence[Mapping[str, Any]], user_hotspots: Sequence[Any], *, max_items: int = 3, max_distance: int = 15) -> List[str]:
    user = [_parse_hotspot(h) for h in user_hotspots]
    user = [(c, r) for c, r in user if c and r is not None]
    if not user:
        return []
    existing = {f"{c}:{r}" for c, r in user}
    counts: Dict[str, int] = {}
    for item in structural_items or []:
        for contact in list(item.get("contacts_preview") or [])[:50]:
            chain, resid = _parse_hotspot(contact.get("target_residue"))
            if not chain or resid is None:
                continue
            token = f"{chain}:{resid}"
            if token in existing:
                continue
            if any(chain == uc and abs(int(resid) - int(ur)) <= int(max_distance) for uc, ur in user):
                counts[token] = counts.get(token, 0) + 1
    return sorted(counts, key=lambda h: counts[h], reverse=True)[:max_items]


def _parse_hotspot(value: Any) -> tuple:
    text = str(value or "").strip()
    if not text:
        return "", None
    if ":" in text:
        chain, raw = text.split(":", 1)
    else:
        chain, raw = "", text
    digits = "".join(ch for ch in raw if ch.isdigit() or ch == "-")
    try:
        return chain.strip(), int(digits)
    except ValueError:
        return chain.strip(), None


def _failed_lengths(memory_summary: Mapping[str, Any]) -> Dict[int, int]:
    failed: Dict[int, int] = {}
    for record in memory_summary.get("recent_rounds", []) or []:
        evaluation = record.get("evaluation") or {}
        if int(evaluation.get("success_count") or 0) > 0:
            continue
        for job in record.get("jobs", []) or []:
            try:
                length = int(job.get("binder_length"))
            except (TypeError, ValueError):
                continue
            failed[length] = failed.get(length, 0) + 1
    return failed


def _crop_locked_disabled(params: Mapping[str, Any]) -> bool:
    mode = str(params.get("epitope_crop_mode") or "disabled").strip().lower()
    return mode in {"", "disabled", "off", "none", "false", "0"} and not bool(params.get("allow_agent_epitope_crop", False))


def _foldability_metrics(summary: EvaluationSummary) -> Dict[str, float]:
    rows = list(summary.top_candidates or []) + list(summary.failed_examples or [])
    design_ptm = []
    refold_rmsd = []
    for cand in rows:
        raw = getattr(cand, "raw", {}) or {}
        metrics = getattr(cand, "metrics", {}) or {}
        ptm = _float_param(raw.get("design_ptm"), _float_param(metrics.get("binder_plddt"), None))
        rmsd = _float_param(raw.get("designfolding-filter_rmsd"), _float_param(raw.get("filter_rmsd_design"), None))
        if ptm is not None:
            design_ptm.append(float(ptm))
        if rmsd is not None:
            refold_rmsd.append(float(rmsd))
    out: Dict[str, float] = {}
    if design_ptm:
        out["mean_design_ptm"] = sum(design_ptm) / len(design_ptm)
        out["best_design_ptm"] = max(design_ptm)
    if refold_rmsd:
        out["mean_designfolding_rmsd"] = sum(refold_rmsd) / len(refold_rmsd)
        out["best_designfolding_rmsd"] = min(refold_rmsd)
    return out


def _foldability_risk(metrics: Mapping[str, float], tags: Mapping[str, int], total: int) -> bool:
    if int(tags.get("folding_failure", 0) or 0) / max(1, int(total or 1)) > 0.3:
        return True
    mean_ptm = metrics.get("mean_design_ptm")
    mean_rmsd = metrics.get("mean_designfolding_rmsd")
    return (mean_ptm is not None and float(mean_ptm) < 0.70) or (mean_rmsd is not None and float(mean_rmsd) > 2.5)


def _broaden_binder_lengths(params: Mapping[str, Any]) -> List[int]:
    current = sorted({int(x) for x in (params.get("binder_lengths") or []) if int(x) > 0})
    allowed = _allowed_lengths_from_params(params)
    if not current or not allowed:
        return current
    widened = set(current)
    for length in current:
        idx = min(range(len(allowed)), key=lambda i: (abs(allowed[i] - length), allowed[i]))
        for neighbor in (idx - 1, idx + 1):
            if 0 <= neighbor < len(allowed):
                widened.add(allowed[neighbor])
    return sorted(widened)


def _allowed_lengths_from_params(params: Mapping[str, Any]) -> List[int]:
    rng = params.get("binder_length_range")
    step = int(params.get("binder_length_step") or 10)
    if not rng:
        return []
    try:
        if isinstance(rng, (list, tuple)):
            lo, hi = int(rng[0]), int(rng[1] if len(rng) > 1 else rng[0])
        elif isinstance(rng, dict):
            lo, hi = int(rng.get("min") or rng.get("start")), int(rng.get("max") or rng.get("end"))
        else:
            token = str(rng).replace("..", "-").replace(":", "-")
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
    except Exception:
        return []
    if lo > hi:
        lo, hi = hi, lo
    values = list(range(lo, hi + 1, max(1, step)))
    if values and values[-1] != hi:
        values.append(hi)
    return sorted({int(x) for x in values})


def _float_param(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def _sparse_executable_delta(current: Mapping[str, Any], proposed: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only changed, executable fields; legacy strategy flags stay audit-only."""
    sanitized = {k: v for k, v in supported_config_changes(proposed).items() if k not in PROBABILISTIC_SAMPLER_KEYS}
    baseline = {k: v for k, v in supported_config_changes(current).items() if k not in PROBABILISTIC_SAMPLER_KEYS}
    return {key: value for key, value in sanitized.items() if baseline.get(key) != value}
