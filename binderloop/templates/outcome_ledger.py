"""Target-isolated lifecycle ledger for fragment-template interventions."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from binderloop.resume import atomic_write_json, stable_hash

SCHEMA_VERSION = 1
HARD_FAILURES = frozenset({
    "parse_error", "chain_mismatch", "mapping_mismatch", "mapping_error",
    "digest_mismatch", "lineage_identity_mismatch", "lineage_digest_mismatch",
})
TRANSIENT_FAILURES = frozenset({"runtime_failure", "package_failure", "alignment_failure", "map_failure"})
DIRECTIONS = {"quality": 1.0, "primary_coverage": 1.0, "retention": 1.0, "clash": -1.0}
DEFAULT_WEIGHTS = {"quality": 0.35, "primary_coverage": 0.30, "retention": 0.25, "clash": 0.10}


def lifecycle_key(target_identity_digest: str, template_id: str) -> str:
    if not target_identity_digest or not template_id:
        raise ValueError("target_identity_digest and template_id are required")
    return stable_hash({"target_identity_digest": str(target_identity_digest), "template_id": str(template_id)})


def canonical_digest(payload: Mapping[str, Any]) -> str:
    body = copy.deepcopy(dict(payload))
    body.pop("digest", None)
    return stable_hash(body)


def matched_group_id(target_identity_digest: str, template_id: str, round_id: int, comparison: Mapping[str, Any]) -> str:
    return "matched_" + stable_hash({
        "target_identity_digest": target_identity_digest,
        "template_id": template_id,
        "round_id": int(round_id),
        "comparison": dict(comparison),
    })[:20]


def matched_comparison_signature(params: Mapping[str, Any], *, target_structure: str = "", chain_id: str = "", binder_length: Optional[int] = None) -> Dict[str, Any]:
    """Canonical fields that must be identical on intervention and control."""
    return {
        "target_structure": str(target_structure), "target_chain": str(chain_id),
        "target_identity_digest": str(params.get("target_identity_digest") or ""),
        "effective_length": int(binder_length if binder_length is not None else params.get("binder_length", 0) or 0),
        "binder_lengths": [int(x) for x in params.get("binder_lengths") or []],
        "sampler": {k: params.get(k) for k in ("steps", "step_scale", "noise_scale", "diffusion_batch_size")},
        "filters": {k: params.get(k) for k in ("run_filtering", "filter_biased", "additional_filters", "refolding_rmsd_threshold")},
        "folding": {k: params.get(k) for k in ("inverse_fold_num_sequences", "skip_inverse_folding", "only_inverse_fold", "inverse_fold_checkpoint", "folding_checkpoint", "affinity_checkpoint")},
        "checkpoint_seed_policy": {k: params.get(k) for k in ("design_checkpoints", "checkpoint_dir", "seed", "seed_policy")},
        "backbone_budget": int(params.get("num_designs", 0) or params.get("num_designs_per_round", 0) or 0),
    }


def assert_matched_pair(template: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    left, right = dict(template.get("matched_comparison") or {}), dict(control.get("matched_comparison") or {})
    if not left or left != right:
        differing = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        raise ValueError("matched-control parity failure: " + ",".join(differing or ["missing_comparison"]))
    if template.get("matched_group_id") != control.get("matched_group_id"):
        raise ValueError("matched-control group mismatch")
    if not template.get("template_conditioned") or control.get("template_conditioned"):
        raise ValueError("matched pair must differ only by template intervention")


def compute_utility(template_metrics: Mapping[str, Any], control_metrics: Mapping[str, Any], *, confidence: float, rounds_since_use: int = 0, decay: float = 0.90, weights: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    weights = {**DEFAULT_WEIGHTS, **dict(weights or {})}
    raw, uplift = {}, {}
    for name, direction in DIRECTIONS.items():
        t = float(template_metrics.get(name, 0.0) or 0.0); c = float(control_metrics.get(name, 0.0) or 0.0)
        raw[name] = {"template": t, "control": c}
        uplift[name] = direction * (t - c)
    weighted = sum(float(weights[name]) * uplift[name] for name in DIRECTIONS)
    confidence = min(1.0, max(0.0, float(confidence)))
    time_decay = min(1.0, max(0.0, float(decay))) ** max(0, int(rounds_since_use))
    adjusted = weighted * confidence * time_decay
    uncertainty = max(0.0, 1.0 - confidence)
    return {"raw": raw, "uplift": uplift, "weighted_uplift": weighted, "confidence": confidence, "time_decay": time_decay, "adjusted": adjusted, "uncertainty": uncertainty}


def _new_entry(target: str, template: str) -> Dict[str, Any]:
    return {"target_identity_digest": target, "template_id": template, "uses": 0, "successes": 0,
            "package_failures": 0, "runtime_failures": 0, "stage_attribution": {},
            "primary_coverage": None, "clash": None, "final_quality": None,
            "matched_control": {}, "last_round": None, "utility": None, "uncertainty": 1.0,
            "consecutive_no_gain": 0, "retention_regressions": 0,
            "cooldown_until": None, "blacklisted": False, "blacklist_reason": "",
            "failure_taxonomy": {}, "events": []}


@dataclass
class OutcomeLedger:
    path: Path
    document: Dict[str, Any]

    @classmethod
    def open(cls, path: Path | str) -> "OutcomeLedger":
        path = Path(path)
        if path.is_file():
            import json
            document = json.loads(path.read_text(encoding="utf-8"))
            if int(document.get("schema_version", 0)) != SCHEMA_VERSION or canonical_digest(document) != document.get("digest"):
                raise ValueError("template outcome ledger schema/digest mismatch")
        else:
            document = {"schema_version": SCHEMA_VERSION, "record_type": "template_outcome_ledger", "entries": {}, "event_count": 0}
            document["digest"] = canonical_digest(document)
        return cls(path, document)

    def entry(self, target_identity_digest: str, template_id: str, *, create: bool = True) -> Optional[Dict[str, Any]]:
        key = lifecycle_key(target_identity_digest, template_id)
        entries = self.document.setdefault("entries", {})
        if key not in entries and create:
            entries[key] = _new_entry(str(target_identity_digest), str(template_id))
        return entries.get(key)

    def record_failure(self, target_identity_digest: str, template_id: str, *, round_id: int, failure_type: str, detail: str = "") -> Dict[str, Any]:
        entry = self.entry(target_identity_digest, template_id)
        kind = str(failure_type)
        entry["failure_taxonomy"][kind] = int(entry["failure_taxonomy"].get(kind, 0)) + 1
        if kind == "package_failure": entry["package_failures"] += 1
        if kind == "runtime_failure": entry["runtime_failures"] += 1
        if kind in HARD_FAILURES:
            entry["blacklisted"], entry["blacklist_reason"] = True, kind
        self._event(entry, round_id, "failure", {"failure_type": kind, "detail": detail, "recoverable": kind not in HARD_FAILURES})
        return entry

    def record_outcome(self, target_identity_digest: str, template_id: str, *, round_id: int,
                       template_metrics: Mapping[str, Any], control_metrics: Mapping[str, Any],
                       confidence: float, matched_group_id: str, stage_attribution: Optional[Mapping[str, Any]] = None,
                       evidence_mode: str = "validated_lineage", decay: float = 0.90,
                       cooldown_failures: int = 2, cooldown_rounds: int = 1) -> Dict[str, Any]:
        entry = self.entry(target_identity_digest, template_id)
        entry["uses"] += 1; entry["successes"] += 1; entry["last_round"] = int(round_id)
        utility = compute_utility(template_metrics, control_metrics, confidence=confidence, decay=decay)
        entry.update({"stage_attribution": dict(stage_attribution or {}), "primary_coverage": template_metrics.get("primary_coverage"),
                      "clash": template_metrics.get("clash"), "final_quality": template_metrics.get("quality"),
                      "matched_control": {"matched_group_id": matched_group_id, "control_metrics": dict(control_metrics), "uplift": utility["uplift"], "confidence": utility["confidence"], "evidence_mode": evidence_mode},
                      "utility": utility, "uncertainty": utility["uncertainty"]})
        retention_regression = utility["uplift"]["retention"] < 0
        no_gain = utility["weighted_uplift"] <= 0
        entry["consecutive_no_gain"] = entry["consecutive_no_gain"] + 1 if no_gain else 0
        entry["retention_regressions"] = entry["retention_regressions"] + 1 if retention_regression else 0
        if max(entry["consecutive_no_gain"], entry["retention_regressions"]) >= max(1, int(cooldown_failures)):
            entry["cooldown_until"] = int(round_id) + max(1, int(cooldown_rounds))
        elif entry.get("cooldown_until") is not None and int(round_id) >= int(entry["cooldown_until"]):
            entry["cooldown_until"] = None
        self._event(entry, round_id, "outcome", {"matched_group_id": matched_group_id, "utility": utility, "evidence_mode": evidence_mode})
        return entry

    def eligible(self, target_identity_digest: str, template_id: str, round_id: int) -> bool:
        entry = self.entry(target_identity_digest, template_id, create=False)
        if not entry: return True
        if entry.get("blacklisted"): return False
        until = entry.get("cooldown_until")
        return until is None or int(round_id) >= int(until)

    def target_snapshot(self, target_identity_digest: str, *, round_id: Optional[int] = None, decay: float = 0.90) -> Dict[str, Any]:
        entries = {}
        for key, raw in self.document.get("entries", {}).items():
            if raw.get("target_identity_digest") != target_identity_digest: continue
            item = copy.deepcopy(raw)
            if item.get("utility") and round_id is not None and item.get("last_round") is not None:
                age = max(0, int(round_id) - int(item["last_round"]))
                item["utility"]["time_decay"] = float(decay) ** age
                item["utility"]["adjusted"] = item["utility"]["weighted_uplift"] * item["utility"]["confidence"] * item["utility"]["time_decay"]
            entries[key] = item
        body = {"schema_version": SCHEMA_VERSION, "target_identity_digest": target_identity_digest, "entries": entries}
        body["digest"] = canonical_digest(body)
        return body

    def save(self) -> Path:
        self.document["digest"] = canonical_digest(self.document)
        return atomic_write_json(self.path, self.document)

    def _event(self, entry: Dict[str, Any], round_id: int, event_type: str, payload: Mapping[str, Any]) -> None:
        event = {"sequence": int(self.document.get("event_count", 0)) + 1, "round_id": int(round_id), "event_type": event_type, "payload": dict(payload)}
        event["digest"] = canonical_digest(event)
        entry["events"].append(event); self.document["event_count"] = event["sequence"]


def rank_templates(templates: Sequence[Mapping[str, Any]], ledger_snapshot: Optional[Mapping[str, Any]], *, top_k: int, round_id: int = 0) -> List[Dict[str, Any]]:
    """Eligibility -> compatibility -> utility/UCB -> deterministic diversity."""
    by_template = {str(v.get("template_id")): v for v in (ledger_snapshot or {}).get("entries", {}).values()}
    candidates = []
    for raw in templates:
        item = dict(raw); tid = str(item.get("template_id") or ""); state = by_template.get(tid)
        eligible = bool(item.get("eligible", True)) and not bool((state or {}).get("blacklisted"))
        until = (state or {}).get("cooldown_until")
        eligible = eligible and (until is None or int(round_id) >= int(until))
        if not eligible: continue
        compatibility = float(item.get("target_compatibility", item.get("quality_score", 0.0)) or 0.0)
        unknown = state is None or state.get("utility") is None
        adjusted = float(((state or {}).get("utility") or {}).get("adjusted", 0.0) or 0.0)
        uncertainty = float((state or {}).get("uncertainty", 1.0) if state else 1.0)
        item["ledger_state"] = {"known": not unknown, "utility": adjusted, "uncertainty": uncertainty}
        candidates.append((item, compatibility, adjusted + 0.25 * uncertainty, unknown))
    candidates.sort(key=lambda row: (-row[1], -row[2], not row[3], str(row[0].get("template_id"))))
    selected, deferred, sources, patches, clusters = [], [], set(), set(), set()
    for row in candidates:
        item = row[0]; source = str(item.get("source_digest") or item.get("source_structure_file") or "")
        patch = ",".join(map(str, item.get("compatible_target_patch") or item.get("target_contact_residues") or []))
        cluster = str(item.get("cluster_id") or "")
        if source in sources or (patch and patch in patches) or (cluster and cluster in clusters): deferred.append(item); continue
        selected.append(item); sources.add(source); patches.add(patch); clusters.add(cluster)
        if len(selected) >= top_k: break
    for item in deferred + [row[0] for row in candidates if row[0] not in selected and row[0] not in deferred]:
        if len(selected) >= top_k: break
        if item not in selected: selected.append(item)
    return selected


def _candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows: return {name: 0.0 for name in DIRECTIONS}
    def value(row: Mapping[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
        for key in keys:
            raw = row.get(key)
            if isinstance(raw, Mapping): raw = raw.get("value")
            try:
                if raw is not None: return float(raw)
            except (TypeError, ValueError): pass
        return default
    totals = {name: 0.0 for name in DIRECTIONS}
    for row in rows:
        totals["quality"] += value(row, ("final_quality", "quality_score", "iptm", "confidence"))
        totals["primary_coverage"] += value(row, ("primary_coverage", "primary_contact_coverage", "hotspot_coverage"))
        totals["retention"] += value(row, ("retention", "contact_retention", "motif_retention"))
        totals["clash"] += value(row, ("clash", "clash_density", "heavy_atom_clash_count"))
    return {key: val / len(rows) for key, val in totals.items()}


def update_ledger_from_round(ledger: OutcomeLedger, *, round_id: int, jobs: Sequence[Any], ingestions: Sequence[Mapping[str, Any]], execution_records: Sequence[Mapping[str, Any]], attribution_documents: Sequence[Mapping[str, Any]], decay: float = 0.90, cooldown_failures: int = 2) -> None:
    """Update outcomes and failures from one fully evaluated round."""
    controls = []
    control_groups = []
    for job, ingestion in zip(jobs, ingestions):
        params = dict(getattr(job, "params", {}) or {})
        if not params.get("template_conditioned") and params.get("matched_group_ids"):
            controls.extend(list(ingestion.get("candidates") or [])); control_groups = list(params.get("matched_group_ids") or [])
    partitioned = {str(g.get("matched_group_id")): [] for g in control_groups}
    ordered_groups = sorted(partitioned)
    for row in controls:
        candidate_id = str(row.get("global_candidate_id") or row.get("candidate_id") or row.get("design_id") or stable_hash(row))
        if ordered_groups: partitioned[ordered_groups[int(stable_hash(candidate_id), 16) % len(ordered_groups)]].append(row)
    def unique_by_job(rows: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Mapping[str, Any]]:
        indexed: Dict[str, Mapping[str, Any]] = {}
        for row in rows:
            job_id = str(row.get("job_id") or "")
            if not job_id:
                raise ValueError(f"{label}_job_id_missing")
            if job_id in indexed:
                raise ValueError(f"duplicate_{label}_job_id:{job_id}")
            indexed[job_id] = row
        return indexed
    record_by_job = unique_by_job(execution_records, "template_execution_record")
    attr_by_job = unique_by_job(attribution_documents, "template_attribution")
    for job, ingestion in zip(jobs, ingestions):
        params = dict(getattr(job, "params", {}) or {})
        if not params.get("template_conditioned"): continue
        template = dict(params.get("binder_template") or {}); tid = str(template.get("template_id") or "")
        target = str(params.get("target_identity_digest") or ""); group = str(params.get("matched_group_id") or "")
        if not target or not tid: continue
        record = record_by_job.get(str(getattr(job, "job_id", "")), {})
        status = str(record.get("status") or "").lower()
        if status in {"failed", "error", "timeout", "cancelled", "canceled", "not_executed"}:
            ledger.record_failure(target, tid, round_id=round_id, failure_type="runtime_failure", detail=str(record.get("error") or record.get("reason") or status)); continue
        template_rows = list(ingestion.get("candidates") or [])
        if not template_rows:
            ledger.record_failure(target, tid, round_id=round_id, failure_type="package_failure", detail="no_ingested_template_candidates"); continue
        control_rows = partitioned.get(group) or controls
        if not control_rows:
            ledger.record_failure(target, tid, round_id=round_id, failure_type="runtime_failure", detail="matched_control_unavailable"); continue
        attribution = attr_by_job.get(str(getattr(job, "job_id", "")), {})
        comparisons = list(attribution.get("comparisons") or [])
        exact = ingestion.get("identity_capability") == "validated_lineage" and ingestion.get("exact_attribution") is True
        stage_summary = {f"{r.get('from_stage')}->{r.get('to_stage')}": r.get("metrics") for r in comparisons if exact and r.get("status") == "evaluated"}
        confidence = min(1.0, math.sqrt(min(len(template_rows), len(control_rows))) / 3.0)
        evidence_mode = "validated_lineage" if exact else "aggregate_only"
        if not exact:
            confidence *= 0.5
        ledger.record_outcome(target, tid, round_id=round_id, template_metrics=_candidate_metrics(template_rows), control_metrics=_candidate_metrics(control_rows), confidence=confidence, matched_group_id=group, stage_attribution=stage_summary, evidence_mode=evidence_mode, decay=decay, cooldown_failures=cooldown_failures)
