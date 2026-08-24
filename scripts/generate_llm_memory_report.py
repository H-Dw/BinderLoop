#!/usr/bin/env python3
"""Merge limit-probe and memory-benchmark artifacts into one report bundle."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-dir",
        default="outputs/gpt55_limit_probe_sc2rbd_round00",
    )
    parser.add_argument(
        "--benchmark",
        default="outputs/gpt55_limit_probe_sc2rbd_round00/memory_benchmark.json",
    )
    parser.add_argument(
        "--out",
        default="outputs/gpt55_limit_probe_sc2rbd_round00/consolidated_report.json",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    probe_dir = resolve(args.probe_dir)
    benchmark = _read_json(resolve(args.benchmark))
    summary = _read_json(probe_dir / "summary.json")
    corpus = _read_json(probe_dir / "prompt_corpus.json")
    if isinstance(corpus, list):
        prompt_corpus = corpus
    else:
        prompt_corpus = list(summary.get("corpus") or [])

    safe_budget = dict(summary.get("safe_budget") or {})
    legacy_bytes = int(benchmark.get("legacy_prompt_bytes") or 0)
    retrieved_bytes = int(benchmark.get("retrieved_prompt_bytes") or 0)
    reduction = 0.0
    if legacy_bytes:
        reduction = round(1.0 - retrieved_bytes / legacy_bytes, 6)

    prompt_bytes = safe_budget.get("recommended_prompt_max_bytes")
    output_tokens = safe_budget.get("recommended_max_output_tokens")

    consolidated = {
        "probe_summary": summary,
        "prompt_corpus": prompt_corpus,
        "safe_budget": safe_budget,
        "memory_benchmark": benchmark,
        "optimization": {
            "legacy_memory_prompt_bytes": legacy_bytes,
            "retrieved_memory_prompt_bytes": retrieved_bytes,
            "memory_prompt_reduction_fraction": reduction,
            "retrieval_selected_count": (benchmark.get("retrieval") or {}).get("selected_count"),
            "cluster_mmr_reduction_fraction": (benchmark.get("retrieval") or {}).get(
                "cluster_mmr_reduction_fraction"
            ),
        },
    }
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(consolidated, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Consolidated LLM + memory optimization report",
        "",
        "## Prompt corpus (round_00)",
        "",
    ]
    for item in prompt_corpus:
        md_lines.append(
            f"- `{item.get('kind')}`: {item.get('request_message_bytes', 0):,} bytes "
            f"({item.get('provenance')})"
        )
    md_lines.extend([
        "",
        "## Safe production budget",
        "",
        f"- recommended `prompt_max_bytes`: {prompt_bytes:,}" if prompt_bytes is not None else "- recommended `prompt_max_bytes`: pending live probe",
        f"- recommended `max_output_tokens`: {output_tokens:,}" if output_tokens is not None else "- recommended `max_output_tokens`: pending live probe",
        f"- input boundary: {safe_budget.get('input_boundary_status')}",
        f"- output boundary: {safe_budget.get('output_boundary_status')}",
        "",
        "## Memory prompt optimization",
        "",
        f"- legacy memory block: {legacy_bytes:,} bytes",
        f"- retrieved memory block: {retrieved_bytes:,} bytes",
        f"- reduction: {reduction:.1%}" if legacy_bytes else "- reduction: n/a",
        "",
        f"Full JSON: `{out_path}`",
    ])
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(consolidated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
