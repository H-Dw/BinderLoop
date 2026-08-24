#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from dataclasses import asdict

from binderloop.config import load_config
from binderloop.pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Run BinderLoop strategy-level active learning pipeline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="Force dry-run regardless of config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.resource.backend = "dry_run"
    results = run_pipeline(cfg)
    print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
