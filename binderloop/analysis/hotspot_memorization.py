"""Direct LLM hotspot probes for training-data memorization (never used in-loop)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from binderloop.analysis.hotspot_compare import compare_hotspot_sets, normalize_hotspot_set
from binderloop.analysis.hotspot_descriptors import TargetResidueTable, sanitize_hotspot_tokens
from binderloop.llm import NO_WEB_SEARCH_SYSTEM_INSTRUCTION, OpenAICompatibleClient

MEMORIZATION_KEYWORDS = (
    "pdb",
    "uniprot",
    "literature",
    "known epitope",
    "known hotspot",
    "crystal",
    "complex with",
    "published",
    "paper",
    "alphafold db",
)
IDENTITY_ONLY_SYSTEM = (
    "You propose protein-binder hotspot residues as JSON. "
    + NO_WEB_SEARCH_SYSTEM_INSTRUCTION
    + " Return {\"hotspots\":[\"A:12\"],\"rationale\":\"...\"}."
)
STRUCTURE_ONLY_SYSTEM = (
    "You propose binder hotspot residues from an anonymous residue table. "
    + NO_WEB_SEARCH_SYSTEM_INSTRUCTION
    + " Do not guess a protein name or PDB ID. "
    "Return {\"hotspots\":[\"A:12\"],\"rationale\":\"...\"}."
)


def literature_keyword_hits(text: str) -> List[str]:
    lowered = str(text or "").lower()
    hits = [token for token in MEMORIZATION_KEYWORDS if token in lowered]
    if re.search(r"\bpdb\s*[0-9][a-z0-9]{3}\b", lowered):
        if "pdb" not in hits:
            hits.append("pdb")
    return hits


def classify_memorization(
    *,
    identity_jaccard: float,
    structure_jaccard: float,
    identity_keyword_hits: Sequence[str],
    identity_overlap: int,
) -> str:
    if identity_jaccard >= 0.5 and identity_overlap >= 2 and (identity_keyword_hits or identity_jaccard >= 0.67):
        return "likely_memorized"
    if structure_jaccard >= 0.5 and identity_jaccard < 0.34:
        return "likely_structure_reasoned"
    if identity_jaccard >= 0.34 and structure_jaccard >= 0.34:
        return "ambiguous_identity_and_structure"
    if identity_jaccard < 0.2 and structure_jaccard < 0.2:
        return "no_prior_match"
    return "weak_overlap"


def score_probe_response(
    hotspots: Sequence[str],
    prior: Sequence[str],
    *,
    rationale: str = "",
    condition: str,
) -> Dict[str, Any]:
    comparison = compare_hotspot_sets(hotspots, prior, label=condition)
    hits = literature_keyword_hits(rationale)
    return {
        **comparison,
        "rationale": rationale,
        "literature_keyword_hits": hits,
    }


def build_identity_prompt(*, protein_name: str, pdb_id: Optional[str] = None, sequence: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "protein_name": str(protein_name or "").strip(),
        "task": "Propose the commonly used binder-design hotspot residues for this named target.",
    }
    if pdb_id:
        payload["pdb_id"] = str(pdb_id).strip()
    if sequence:
        payload["sequence"] = str(sequence).strip()
    return payload


def build_anonymized_structure_prompt(table: TargetResidueTable) -> Dict[str, Any]:
    return {
        "task": "Select a compact surface hotspot patch from this anonymous residue table.",
        "residue_table": table.prompt_payload(),
    }


def run_named_probe(
    llm: OpenAICompatibleClient,
    *,
    protein_name: str,
    pdb_id: Optional[str] = None,
    sequence: Optional[str] = None,
    chain_id: str = "A",
    allowed_tokens: Optional[Sequence[str]] = None,
    model_key: Optional[str] = None,
) -> Dict[str, Any]:
    user = build_identity_prompt(protein_name=protein_name, pdb_id=pdb_id, sequence=sequence)
    raw = llm.chat_json(
        system=IDENTITY_ONLY_SYSTEM if not sequence else IDENTITY_ONLY_SYSTEM + " A sequence is supplied; still do not use web search.",
        user=user,
        model_key=model_key,
        temperature=0.0,
        allow_web_search=False,
    )
    hotspots = list(raw.get("hotspots") or []) if isinstance(raw, Mapping) else []
    if allowed_tokens:
        hotspots, notes = sanitize_hotspot_tokens(
            hotspots,
            allowed_tokens=allowed_tokens,
            chain_id=chain_id,
            min_hotspots=1,
            max_hotspots=max(1, len(allowed_tokens)),
            max_change_per_round=10**6,
        )
    else:
        notes = []
        hotspots = normalize_hotspot_set(hotspots)
    return {
        "raw": raw,
        "hotspots": hotspots,
        "rationale": str((raw or {}).get("rationale") or "") if isinstance(raw, Mapping) else "",
        "sanitize_notes": notes,
        "allow_web_search": False,
    }


def run_anonymized_probe(
    llm: OpenAICompatibleClient,
    *,
    table: TargetResidueTable,
    chain_id: str,
    min_hotspots: int,
    max_hotspots: int,
    model_key: Optional[str] = None,
) -> Dict[str, Any]:
    raw = llm.chat_json(
        system=STRUCTURE_ONLY_SYSTEM,
        user=build_anonymized_structure_prompt(table),
        model_key=model_key,
        temperature=0.0,
        allow_web_search=False,
    )
    hotspots, notes = sanitize_hotspot_tokens(
        (raw or {}).get("hotspots") if isinstance(raw, Mapping) else [],
        allowed_tokens=table.tokens(),
        chain_id=chain_id,
        min_hotspots=min_hotspots,
        max_hotspots=max_hotspots,
        fallback=table.tokens()[:max_hotspots],
        max_change_per_round=10**6,
    )
    return {
        "raw": raw,
        "hotspots": hotspots,
        "rationale": str((raw or {}).get("rationale") or "") if isinstance(raw, Mapping) else "",
        "sanitize_notes": notes,
        "allow_web_search": False,
        "identity_hidden": True,
    }
