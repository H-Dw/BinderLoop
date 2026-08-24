#!/usr/bin/env python3
"""Replay the R2 InputConfigurationAgent prompt against the live LLM endpoint.

Reconstructs the compact next-round prompt from a failed closed-loop round
directory and calls ``chat_json`` with the same system/user payload the agent
used when JSON parse failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.context_compaction import compact_context_for_input_config
from binderloop.agents.input_configuration_agent import InputConfigurationAgent
from binderloop.llm import OpenAICompatibleClient
from binderloop.skills import compose_agent_system


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_prompt(round_dir: Path, *, target_name: str, next_round_id: int) -> Dict[str, Any]:
    checkpoint = _load(round_dir / "round_checkpoint.json")
    jobs = list(checkpoint.get("current_jobs") or [])
    current_config = dict((jobs[0].get("params") or {}) if jobs else {})
    diagnostic = _load(round_dir / "diagnostic_report.json")
    evaluation = _load(round_dir / "evaluation_summary.json")
    quality = _load(round_dir / "binder_quality_analysis.json")
    hypotheses = _load(round_dir / "hypotheses.json")
    structural = _load(round_dir / "structure_evaluation.json")
    skills_blob = _load(round_dir / "active_skills.json") if (round_dir / "active_skills.json").exists() else {}
    active_skills = list((skills_blob.get("activations_by_agent") or {}).get("InputConfigurationAgent") or [])
    target_profile = {
        "target_name": target_name,
        "primary_chain_id": current_config.get("chain_id") or (jobs[0].get("chain_id") if jobs else "A"),
        "hotspots": list(jobs[0].get("hotspots") or current_config.get("hotspots") or []) if jobs else list(current_config.get("hotspots") or []),
        "source": "replay_from_failed_round_artifacts",
    }
    llm_context = compact_context_for_input_config(
        target_name=target_name,
        current_config=current_config,
        diagnostic_report=diagnostic,
        evaluation_summary=evaluation,
        round_id=next_round_id,
        target_profile=target_profile,
        structural_analysis=structural,
        quality_analysis=quality,
        hypotheses=list(hypotheses.get("hypotheses") or hypotheses) if isinstance(hypotheses, dict) else list(hypotheses or []),
        active_skills=active_skills,
    )
    prompt_context = dict(llm_context)
    prompt_context.pop("active_skills", None)
    system = compose_agent_system(InputConfigurationAgent.SYSTEM, active_skills=llm_context.get("active_skills"))
    return {"system": system, "user": prompt_context, "current_config": current_config}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-config", default="configs/llm_endpoints.ds.json")
    parser.add_argument("--round-dir", default="outputs/pdl1_direct/round_02")
    parser.add_argument("--target-name", default="PD-L1_len50_120_iptm035")
    parser.add_argument("--out", default="outputs/pdl1_direct/replay_input_config_r2.json")
    args = parser.parse_args()
    round_dir = Path(args.round_dir)
    prompt = reconstruct_prompt(round_dir, target_name=args.target_name, next_round_id=3)
    user_bytes = len(json.dumps(prompt["user"], ensure_ascii=False).encode("utf-8"))
    client = OpenAICompatibleClient.from_json(args.llm_config)
    if client is None or not client.available():
        raise SystemExit(f"LLM endpoint is not available from {args.llm_config}")
    result = client.chat_json(
        system=prompt["system"],
        user=prompt["user"],
        temperature=0.2,
        max_tokens=8000,
    )
    last = dict(client.last_json_call or {})
    complete = isinstance(result, dict) and ("parameter_delta" in result or "recommended_config" in result) and not result.get("parse_error")
    artifact = {
        "ok": bool(complete),
        "prompt_user_bytes": user_bytes,
        "prompt_system_bytes": len(prompt["system"].encode("utf-8")),
        "result_keys": sorted(result.keys()) if isinstance(result, dict) else [],
        "has_parameter_delta": bool(isinstance(result, dict) and result.get("parameter_delta") is not None),
        "has_recommended_config": bool(isinstance(result, dict) and result.get("recommended_config") is not None),
        "parse_error": result.get("parse_error") if isinstance(result, dict) else None,
        "raw_text_length": len(str(result.get("raw_text") or "")) if isinstance(result, dict) else 0,
        "last_json_call": last,
        "result": result,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = {
        "ok": artifact["ok"],
        "prompt_user_bytes": user_bytes,
        "result_keys": artifact["result_keys"],
        "has_parameter_delta": artifact["has_parameter_delta"],
        "has_recommended_config": artifact["has_recommended_config"],
        "parse_error": artifact["parse_error"],
        "attempts": last.get("attempts"),
        "out": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
