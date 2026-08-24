#!/usr/bin/env python3
"""Dump tagged {system, user} prompts for each LLM role from a round directory.

This is the machine-readable backup of what each agent would see. It does not
require a live LLM. Default round is the PD-L1 8r v1 round_00 when present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.binder_quality_analysis_agent import BinderQualityAnalysisAgent
from binderloop.agents.diagnostic_coach_agent import DiagnosticCoachAgent
from binderloop.agents.hypothesis_agent import HypothesisAgent
from binderloop.agents.input_configuration_agent import InputConfigurationAgent
from binderloop.agents.prompt_assembler import assemble, build_store, load_round_artifacts
from binderloop.agents.prompt_catalog import AGENT_PROMPT_SPECS


DEFAULT_ROUND_DIR = Path("outputs/pdl1_closed_loop_llm_notemp_100s_8r_v1/round_00")
LIVE_SYSTEM = {
    "HypothesisAgent": HypothesisAgent.SYSTEM,
    "BinderQualityAnalysisAgent": BinderQualityAnalysisAgent.SYSTEM,
    "DiagnosticCoachAgent": DiagnosticCoachAgent.SYSTEM,
    "InputConfigurationAgent": InputConfigurationAgent.SYSTEM,
}


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))


def dump_round_prompts(
    round_dir: Path,
    *,
    roles: Optional[Iterable[str]] = None,
    out_dir: Optional[Path] = None,
    tagged: bool = True,
) -> List[Dict[str, Any]]:
    artifacts = load_round_artifacts(round_dir)
    store = build_store(artifacts)
    selected = list(roles or AGENT_PROMPT_SPECS)
    packets: List[Dict[str, Any]] = []
    for role in selected:
        if role not in AGENT_PROMPT_SPECS:
            raise KeyError("unknown role: %s" % role)
        packet = assemble(role, store, tagged=tagged)
        live_system = LIVE_SYSTEM.get(role)
        if live_system:
            packet["system"] = live_system
            packet["system_bytes"] = len(live_system.encode("utf-8"))
        packet["user_bytes"] = _json_bytes(packet.get("user"))
        packet["round_dir"] = str(round_dir)
        packets.append(packet)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / ("%s.json" % role)
            path.write_text(json.dumps(packet, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return packets


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round-dir",
        type=Path,
        default=DEFAULT_ROUND_DIR if DEFAULT_ROUND_DIR.exists() else None,
        help="Path to a round_XX directory with evaluation/structure/AL artifacts.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Write one JSON file per role.")
    parser.add_argument("--roles", nargs="*", default=None, help="Subset of AGENT_PROMPT_SPECS keys.")
    parser.add_argument("--legacy", action="store_true", help="Dump compact_context_for_* user dicts instead of tagged slices.")
    args = parser.parse_args(argv)
    if args.round_dir is None:
        parser.error("round directory not found; pass --round-dir")
    packets = dump_round_prompts(
        Path(args.round_dir),
        roles=args.roles,
        out_dir=args.out_dir,
        tagged=not args.legacy,
    )
    summary = [
        {
            "role": item["role"],
            "system_bytes": item["system_bytes"],
            "user_bytes": item["user_bytes"],
            "required_tags": item.get("required_tags"),
        }
        for item in packets
    ]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
