"""Command-line entry point.

    python -m weather_pricing.cli build --cities NY CHI --start 2025-01-01
    python -m weather_pricing.cli evaluate
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from .config import DEFAULT_CITIES, PROCESSED_DIR, RESULTS_DIR


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="wp")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="collect data and build point-in-time snapshots")
    b.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    b.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 1))
    b.add_argument("--end", type=date.fromisoformat, default=None)

    e = sub.add_parser("evaluate", help="fit models A/B/C, score them and value the instrument")
    e.add_argument("--test-frac", type=float, default=0.3)
    e.add_argument("--n-boot", type=int, default=2000)
    e.add_argument("--rate", type=float, default=0.04, help="annual risk-free rate (continuous)")
    e.add_argument("--wang-lambda", type=float, default=0.0, help="risk loading for valuation")
    e.add_argument("--save-artifacts", action="store_true", help="also write models, predictions and results to app_artifacts/")

    args = p.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.cmd == "build":
        from .dataset import build, quality_report

        df = build(args.cities, args.start, args.end)
        report = quality_report(df)
        (RESULTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))

    elif args.cmd == "evaluate":
        from .experiment import run

        df = pd.read_parquet(PROCESSED_DIR / "snapshots.parquet")
        summary = run(
            df,
            test_frac=args.test_frac,
            n_boot=args.n_boot,
            rate=args.rate,
            wang_lambda=args.wang_lambda,
            save_artifacts=args.save_artifacts,
        )
        print(summary)


if __name__ == "__main__":
    main()
