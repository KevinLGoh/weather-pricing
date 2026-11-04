"""Command-line entry point.

    python -m weather_pricing.cli build --cities NY CHI --start 2025-01-01
    python -m weather_pricing.cli evaluate
    python -m weather_pricing.cli weather-baseline
    python -m weather_pricing.cli rolling --data snapshots snapshots_chi
    python -m weather_pricing.cli exact-runs
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
    b.add_argument("--name", default="snapshots", help="file name under data/processed, so another city can sit beside the main dataset")

    e = sub.add_parser("evaluate", help="fit models A/B/C, score them and value the instrument")
    e.add_argument("--test-frac", type=float, default=0.3)
    e.add_argument("--n-boot", type=int, default=2000)
    e.add_argument("--rate", type=float, default=0.04, help="annual risk-free rate (continuous)")
    e.add_argument("--wang-lambda", type=float, default=0.0, help="risk loading for valuation")
    e.add_argument("--save-artifacts", action="store_true", help="also write models, predictions and results to app_artifacts/")

    w = sub.add_parser("weather-baseline", help="compare the market with a stronger weather model")
    w.add_argument("--test-frac", type=float, default=0.3)
    w.add_argument("--n-boot", type=int, default=2000)
    w.add_argument("--lag-scale", type=float, default=1.0, help="multiply the assumed guidance release delays")
    w.add_argument("--data", default="snapshots", help="dataset name under data/processed")

    r = sub.add_parser("rolling", help="refit on earlier days and score the next window, per city")
    r.add_argument("--data", nargs="+", default=["snapshots"], help="dataset names under data/processed")
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--report-only", action="store_true", help="rebuild the tables from the saved predictions")

    x = sub.add_parser("exact-runs", help="test individually archived model runs on the rolling predictions")
    x.add_argument("--n-boot", type=int, default=2000)

    args = p.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.cmd == "build":
        from .dataset import build, quality_report

        df = build(args.cities, args.start, args.end, args.name)
        report = quality_report(df)
        # The app artifacts copy data_quality.json, so another dataset must not replace it.
        suffix = "" if args.name == "snapshots" else f"_{args.name}"
        (RESULTS_DIR / f"data_quality{suffix}.json").write_text(json.dumps(report, indent=2, default=str))
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

    elif args.cmd == "weather-baseline":
        from .guidance import enrich
        from .weather_experiment import run

        name = "weather_baseline" if args.lag_scale == 1.0 else f"weather_baseline_lag{args.lag_scale:g}"
        if args.data != "snapshots":
            name = f"{name}_{args.data.removeprefix('snapshots_')}"
        df = enrich(pd.read_parquet(PROCESSED_DIR / f"{args.data}.parquet"), lag_scale=args.lag_scale)
        print(run(df, RESULTS_DIR / name, test_frac=args.test_frac, n_boot=args.n_boot))

    elif args.cmd == "rolling":
        from .guidance import enrich
        from .guidance_plus import enrich_extended
        from .rolling import report, run

        if args.report_only:
            saved = pd.read_parquet(RESULTS_DIR / "rolling" / "oos_predictions.parquet")
            print(report(saved, RESULTS_DIR / "rolling", n_boot=args.n_boot))
            return
        df = pd.concat([pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in args.data], ignore_index=True)
        print(run(enrich_extended(enrich(df)), RESULTS_DIR / "rolling", n_boot=args.n_boot))

    elif args.cmd == "exact-runs":
        from .exact_runs import run
        from .guidance_plus import enrich_exact

        oos = pd.read_parquet(RESULTS_DIR / "rolling" / "oos_predictions.parquet")
        features = enrich_exact(oos[["ticker", "snapshot", "city", "target_date", "snapshot_ts"]])
        print(run(oos, features, RESULTS_DIR / "exact_runs", n_boot=args.n_boot))


if __name__ == "__main__":
    main()
