#!/usr/bin/env python3
"""Run the frozen hotspot benchmark through the DeepSeek Chat API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.deepseek_api import DeepSeekAPIError
from src.deepseek_runner import (
    CONDITIONS,
    DeepSeekBenchmarkError,
    audit_prepared_inputs,
    benchmark_status,
    load_api_config,
    prepare_experiment,
    run_benchmark,
    unseal_labels,
    verify_source_dependency,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_SOURCE = HERE.parent / "llm_3d_hotspot_validation"
DEFAULT_LLM_CONFIG = REPO_ROOT / "configs" / "llm_endpoints.ds.json"


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=DEFAULT_LLM_CONFIG,
        help="llm_endpoints.*.json containing the selected endpoint and secret reference",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="endpoint key inside --llm-config (defaults to default_model)",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="legacy environment-only fallback when load_api_config is used without a JSON config",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--json-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--transport-retries", type=int, default=None)
    parser.add_argument("--backoff-base-seconds", type=float, default=None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and execute the label-blind 72-run hotspot benchmark with "
            "DeepSeek-V4-Pro."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=HERE,
        help="DeepSeek experiment directory (defaults to this directory)",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    prepare = stages.add_parser("prepare", help="copy verified blind inputs into a new backend run")
    prepare.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--backend-id", default="deepseek-v4-pro")

    run = stages.add_parser("run", help="execute or dry-run selected API waves")
    _add_api_arguments(run)
    run.add_argument("--workers", type=int, default=3)
    run.add_argument("--requests-per-minute", type=float, default=0)
    run.add_argument("--max-input-bytes", type=int, default=3_500_000)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--run-id")
    run.add_argument("--case-id")
    run.add_argument("--condition", choices=CONDITIONS)
    run.add_argument("--replicate", type=int, choices=(1, 2, 3))
    run.add_argument("--max-waves", type=int)

    stages.add_parser("audit-inputs", help="scan exact API input bundles for leakage")
    stages.add_parser("status", help="summarize outcomes without changing predictions")
    stages.add_parser("validate", help="validate all 72 terminal outcomes")
    stages.add_parser("freeze", help="freeze predictions, process logs, inputs, and code")

    unseal = stages.add_parser(
        "unseal-labels", help="copy the user labels only after prediction freeze verifies"
    )
    unseal.add_argument(
        "--source-labels",
        type=Path,
        default=DEFAULT_SOURCE / "process" / "hotspot_labels.json",
    )

    evaluate = stages.add_parser("evaluate", help="score an already frozen and unsealed run")
    evaluate.add_argument(
        "--labels",
        type=Path,
        default=Path("process/hotspot_labels.json"),
    )
    evaluate.add_argument("--per-run-mc-draws", type=int, default=9_999)
    evaluate.add_argument("--joint-mc-draws", type=int, default=99_999)
    evaluate.add_argument("--bootstrap-draws", type=int, default=10_000)
    evaluate.add_argument("--seed", type=int, default=20260824)
    return parser


def _run_stage(args: argparse.Namespace) -> tuple[bool, dict[str, object]]:
    root = args.experiment_root.resolve()
    if args.stage == "prepare":
        result = prepare_experiment(root, args.source_root, backend_id=args.backend_id)
        return True, result
    if args.stage == "run":
        config = load_api_config(
            llm_config=args.llm_config,
            endpoint_key=args.llm_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            thinking=args.thinking,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            transport_retries=args.transport_retries,
            backoff_base_seconds=args.backoff_base_seconds,
            json_mode=args.json_mode,
            allow_missing_key=args.dry_run,
        )
        result = run_benchmark(
            root,
            config,
            workers=args.workers,
            requests_per_minute=args.requests_per_minute,
            max_input_bytes=args.max_input_bytes,
            dry_run=args.dry_run,
            filters={
                "run_id": args.run_id,
                "case_id": args.case_id,
                "condition": args.condition,
                "replicate": args.replicate,
            },
            max_waves=args.max_waves,
        )
        return bool(result["ok"]), result
    if args.stage == "status":
        verify_source_dependency(root)
        return True, benchmark_status(root)
    if args.stage == "audit-inputs":
        verify_source_dependency(root)
        result = audit_prepared_inputs(root)
        return bool(result["passed"]), result

    # These stages deliberately reuse the exact frozen validator/evaluator from
    # the GPT benchmark, so backend comparisons cannot silently change metrics.
    from experiments.llm_3d_hotspot_validation.src import evaluate as evaluation
    from experiments.llm_3d_hotspot_validation.src import pipeline

    if args.stage == "validate":
        verify_source_dependency(root)
        report = pipeline.validate(root)
        return bool(report["all_terminal"]), {
            "stage": "validate",
            "expected_runs": report["expected_runs"],
            "terminal_outcomes": report["terminal_outcomes"],
            "valid_predictions": report["valid_predictions"],
            "terminal_failures": report["terminal_failures"],
            "unaccounted": report["unaccounted"],
            "all_terminal": report["all_terminal"],
        }
    if args.stage == "freeze":
        verify_source_dependency(root)
        manifest = pipeline.freeze(root)
        return True, {
            "stage": "freeze",
            "expected_runs": manifest["expected_runs"],
            "validated_predictions": manifest["validated_predictions"],
            "terminal_failures": manifest["terminal_failures"],
            "artifact_count": len(manifest["artifacts"]),
            "labels_absent": manifest["labels_absent"],
        }
    if args.stage == "unseal-labels":
        return True, unseal_labels(root, args.source_labels)

    verify_source_dependency(root)
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
    return True, {
        "stage": "evaluate",
        "verified_predictions": result["freeze"]["verified_run_count"],
        "primary_decision": result["primary"]["decision"],
        "primary_p": result["primary"]["p_greater_equal"],
        "results": str(root / "results"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        success, summary = _run_stage(args)
    except (
        DeepSeekAPIError,
        DeepSeekBenchmarkError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"stage": args.stage, "ok": False, "error": str(exc)}))
        return 2
    except Exception as exc:
        # Import lazily so CLI startup and dry-run do not need evaluator internals.
        from experiments.llm_3d_hotspot_validation.src import evaluate as evaluation
        from experiments.llm_3d_hotspot_validation.src import pipeline

        if isinstance(exc, (pipeline.BenchmarkStateError, evaluation.EvaluationError)):
            print(json.dumps({"stage": args.stage, "ok": False, "error": str(exc)}))
            return 2
        raise
    print(json.dumps({"ok": success, **summary}, sort_keys=True, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
