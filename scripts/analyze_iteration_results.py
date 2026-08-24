#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.visualization import analyze_iteration_quality, plot_iteration_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze closed-loop binder generation quality per round")
    parser.add_argument("--out", required=True, help="Closed-loop output directory containing round_XX folders")
    parser.add_argument("--quality-json", help="Output quality summary JSON (default: <out>/iteration_quality_summary.json)")
    parser.add_argument("--quality-csv", help="Output quality summary CSV (default: <out>/iteration_quality_summary.csv)")
    parser.add_argument("--report-md", help="Output Markdown report (default: <out>/iteration_quality_report.md)")
    parser.add_argument("--quality-plot", help="Output quality trends PNG (default: <out>/iteration_quality_trends.png)")
    parser.add_argument("--metric-plot", help="Output metric trends PNG (default: <out>/iteration_metrics_trends.png)")
    parser.add_argument("--metric-stats-json", help="Output metric stats JSON (default: <out>/iteration_metrics_stats.json)")
    parser.add_argument("--no-plots", action="store_true", help="Only write JSON/CSV/Markdown summaries; skip PNG plots")
    args = parser.parse_args()

    quality_artifacts = analyze_iteration_quality(
        args.out,
        summary_json_path=args.quality_json,
        summary_csv_path=args.quality_csv,
        report_md_path=args.report_md,
        plot_path=args.quality_plot,
        write_plot=not args.no_plots,
    )
    print(f"Wrote quality JSON: {quality_artifacts['summary_json']}")
    print(f"Wrote quality CSV: {quality_artifacts['summary_csv']}")
    print(f"Wrote quality report: {quality_artifacts['report_md']}")
    if "plot_png" in quality_artifacts:
        print(f"Wrote quality plot: {quality_artifacts['plot_png']}")

    if not args.no_plots:
        metric_artifacts = plot_iteration_metrics(args.out, output_path=args.metric_plot, stats_json_path=args.metric_stats_json)
        print(f"Wrote metric stats: {metric_artifacts['stats_json']}")
        print(f"Wrote metric plot: {metric_artifacts['plot_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
