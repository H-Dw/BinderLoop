#!/usr/bin/env python3
"""Probe whether a named LLM remembers literature hotspots without web search.

This script is independent of the closed-loop harness. It may load target identity
and a user-provided prior hotspot file. Those priors never enter a harness run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.analysis.hotspot_compare import load_prior_hotspots  # noqa: E402
from binderloop.analysis.hotspot_descriptors import build_target_residue_table  # noqa: E402
from binderloop.analysis.hotspot_memorization import (  # noqa: E402
    classify_memorization,
    run_anonymized_probe,
    run_named_probe,
    score_probe_response,
)
from binderloop.llm import LLMConfigError, OpenAICompatibleClient  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-config", default="configs/llm_endpoints.local.json")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--protein-name", required=True)
    parser.add_argument("--pdb-id", default=None)
    parser.add_argument("--sequence", default=None, help="Optional one-letter sequence for identity_plus_sequence.")
    parser.add_argument("--target-structure", default=None, help="Optional CIF/PDB for the anonymized-structure control.")
    parser.add_argument("--chain-id", default="A")
    parser.add_argument("--prior-hotspots", required=True, help="YAML/JSON literature hotspots loaded only by this script.")
    parser.add_argument("--min-hotspots", type=int, default=3)
    parser.add_argument("--max-hotspots", type=int, default=6)
    parser.add_argument("--out", default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    client = OpenAICompatibleClient.from_json(Path(args.llm_config))
    if client is None:
        raise RuntimeError("LLM config was not loaded: %s" % args.llm_config)
    client.configure_default(model_key=args.llm_model)
    prior = load_prior_hotspots(args.prior_hotspots)
    conditions = {}

    identity = run_named_probe(
        client,
        protein_name=args.protein_name,
        pdb_id=args.pdb_id,
        chain_id=args.chain_id,
        model_key=args.llm_model,
    )
    conditions["identity_only"] = {
        **identity,
        "score": score_probe_response(identity["hotspots"], prior, rationale=identity["rationale"], condition="identity_only"),
    }

    if args.sequence:
        identity_seq = run_named_probe(
            client,
            protein_name=args.protein_name,
            pdb_id=args.pdb_id,
            sequence=args.sequence,
            chain_id=args.chain_id,
            model_key=args.llm_model,
        )
        conditions["identity_plus_sequence"] = {
            **identity_seq,
            "score": score_probe_response(
                identity_seq["hotspots"], prior, rationale=identity_seq["rationale"], condition="identity_plus_sequence",
            ),
        }

    structure_score = {"jaccard_residue_numbers": 0.0}
    if args.target_structure:
        table = build_target_residue_table(args.target_structure, chain_id=args.chain_id)
        anonymized = run_anonymized_probe(
            client,
            table=table,
            chain_id=args.chain_id,
            min_hotspots=args.min_hotspots,
            max_hotspots=args.max_hotspots,
            model_key=args.llm_model,
        )
        structure_score = score_probe_response(
            anonymized["hotspots"], prior, rationale=anonymized["rationale"], condition="anonymized_structure",
        )
        conditions["anonymized_structure"] = {**anonymized, "score": structure_score}

    identity_score = conditions["identity_only"]["score"]
    verdict = classify_memorization(
        identity_jaccard=float(identity_score.get("jaccard_residue_numbers") or 0.0),
        structure_jaccard=float(structure_score.get("jaccard_residue_numbers") or 0.0),
        identity_keyword_hits=list(identity_score.get("literature_keyword_hits") or []),
        identity_overlap=len(identity_score.get("overlap_residue_numbers") or []),
    )
    payload = {
        "allow_web_search": False,
        "protein_name": args.protein_name,
        "pdb_id": args.pdb_id,
        "prior_hotspots": prior,
        "conditions": conditions,
        "verdict": verdict,
        "note": "Priors and protein identity are used only by this probe, not by binder-harness closed-loop runs.",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LLMConfigError as exc:
        raise SystemExit(str(exc)) from exc
