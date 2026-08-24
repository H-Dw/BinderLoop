#!/usr/bin/env python3
"""Compare LLM-selected hotspots from a completed harness run to user-provided priors.

The prior file is loaded only by this script. It is never read by the closed-loop harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.analysis.hotspot_compare import (  # noqa: E402
    choose_best_round,
    collect_run_hotspot_records,
    compare_run_to_prior,
    load_prior_hotspots,
)


def compare_run(
    run_dir: Path,
    prior_hotspots: List[str],
    *,
    baseline_run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = compare_run_to_prior(run_dir, prior_hotspots)
    payload.update({
        "run_dir": str(run_dir),
        "prior_hotspots": list(prior_hotspots),
        "note": "Prior hotspots were loaded only by this verification script, not by the harness.",
    })
    if baseline_run_dir is not None:
        baseline_records = collect_run_hotspot_records(baseline_run_dir)
        baseline_best = choose_best_round(baseline_records)
        payload["baseline_run"] = {
            "run_dir": str(baseline_run_dir),
            "rounds": [
                {
                    "round_id": item["round_id"],
                    "success_rate": item["success_rate"],
                    "success_count": item["success_count"],
                    "round_rank_key": item["round_rank_key"],
                }
                for item in baseline_records
            ],
            "best_round": baseline_best,
        }
        payload["success_rate_delta_vs_baseline"] = None
        best = payload.get("best_round") or {}
        if best and baseline_best:
            payload["success_rate_delta_vs_baseline"] = float(best.get("success_rate") or 0.0) - float(
                baseline_best.get("success_rate") or 0.0
            )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Harness output directory containing round_XX folders.")
    parser.add_argument("--prior-hotspots", required=True, help="YAML/JSON with literature/common hotspots. Never used by the harness.")
    parser.add_argument("--baseline-run-dir", default=None, help="Optional previous literature-hotspot harness run for success-rate comparison.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_dir = Path(args.run_dir)
    prior = load_prior_hotspots(args.prior_hotspots)
    baseline = Path(args.baseline_run_dir) if args.baseline_run_dir else None
    payload = compare_run(run_dir, prior, baseline_run_dir=baseline)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
