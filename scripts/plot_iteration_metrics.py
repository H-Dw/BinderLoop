#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.visualization import plot_iteration_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot closed-loop iteration metrics and success binder counts per round")
    parser.add_argument("--out", required=True, help="Closed-loop output directory containing round_XX folders")
    parser.add_argument("--plot", help="Output PNG path (default: <out>/iteration_metrics_trends.png)")
    parser.add_argument("--stats-json", help="Output stats JSON path (default: <out>/iteration_metrics_stats.json)")
    args = parser.parse_args()

    artifacts = plot_iteration_metrics(args.out, output_path=args.plot, stats_json_path=args.stats_json)
    print(f"Wrote stats: {artifacts['stats_json']}")
    print(f"Wrote plot: {artifacts['plot_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
