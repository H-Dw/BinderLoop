#!/usr/bin/env python3
"""Command-line entry point for the label-blind benchmark stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src import evaluate as evaluation
from src import pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, validate, or terminal-freeze the blinded benchmark."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="experiment directory (defaults to this script's directory)",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prepare = subparsers.add_parser("prepare", help="create blinded run inputs")
    prepare.add_argument(
        "--sphere-points",
        type=int,
        default=960,
        help="deterministic Shrake-Rupley sphere point count",
    )
    subparsers.add_parser("validate", help="check all 72 terminal run outcomes")
    subparsers.add_parser("freeze", help="hash validated outcomes, inputs, and code")
    evaluate = subparsers.add_parser(
        "evaluate", help="unseal labels only after freeze verification and write reports"
    )
    evaluate.add_argument(
        "--labels",
        type=Path,
        default=Path("process/hotspot_labels.json"),
        help="post-freeze author-chain label JSON (relative paths use experiment root)",
    )
    evaluate.add_argument("--per-run-mc-draws", type=int, default=9_999)
    evaluate.add_argument("--joint-mc-draws", type=int, default=99_999)
    evaluate.add_argument("--bootstrap-draws", type=int, default=10_000)
    evaluate.add_argument("--seed", type=int, default=20260824)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.stage == "prepare":
            result = pipeline.prepare(args.experiment_root, sphere_points=args.sphere_points)
            summary = {
                "stage": "prepare",
                "run_count": result.run_count,
                "run_plan": str(result.run_plan_path),
                "qc": str(result.qc_path),
                "checksums": str(result.checksums_path),
            }
            success = True
        elif args.stage == "validate":
            report = pipeline.validate(args.experiment_root)
            summary = {
                "stage": "validate",
                "expected_runs": report["expected_runs"],
                "terminal_outcomes": report["terminal_outcomes"],
                "valid_predictions": report["valid_predictions"],
                "eligible_predictions": report["eligible_predictions"],
                "excluded_predictions": report["excluded_predictions"],
                "terminal_failures": report["terminal_failures"],
                "unaccounted": report["unaccounted"],
                "dual_outcome": report["dual_outcome"],
                "all_valid_predictions": report["all_valid_predictions"],
                "all_terminal": report["all_terminal"],
                # Compatibility fields for callers of the original CLI.
                "successful_artifacts": report["successful_artifacts"],
                "expected_artifacts": report["expected_artifacts"],
                "all_valid": report["all_valid"],
            }
            success = bool(report["all_terminal"])
        elif args.stage == "freeze":
            manifest = pipeline.freeze(args.experiment_root)
            summary = {
                "stage": "freeze",
                "expected_runs": manifest["expected_runs"],
                "validated_predictions": manifest["validated_predictions"],
                "eligible_predictions": manifest["eligible_predictions"],
                "excluded_predictions": manifest["excluded_predictions"],
                "terminal_failures": manifest["terminal_failures"],
                "all_terminal": manifest["all_terminal"],
                "artifact_count": len(manifest["artifacts"]),
                "labels_absent": manifest["labels_absent"],
            }
            success = True
        else:
            root = args.experiment_root.resolve()
            labels = args.labels if args.labels.is_absolute() else root / args.labels
            result = evaluation.evaluate_benchmark(
                root / "process" / "prediction_freeze_manifest.json",
                root / "process" / "run_plan.json",
                labels,
                target_manifest_path=root / "process" / "target_manifest.json",
                per_run_mc_draws=args.per_run_mc_draws,
                joint_mc_draws=args.joint_mc_draws,
                bootstrap_draws=args.bootstrap_draws,
                seed=args.seed,
            )
            evaluation.write_reports(result, root / "results")
            summary = {
                "stage": "evaluate",
                "verified_predictions": result["freeze"]["verified_run_count"],
                "primary_decision": result["primary"]["decision"],
                "primary_p": result["primary"]["p_greater_equal"],
                "results": str(root / "results"),
            }
            success = True
    except (
        OSError,
        ValueError,
        pipeline.BenchmarkStateError,
        evaluation.EvaluationError,
    ) as exc:
        print(json.dumps({"stage": args.stage, "ok": False, "error": str(exc)}))
        return 2
    print(json.dumps({"ok": success, **summary}, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
