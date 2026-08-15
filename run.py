"""
run.py: single entry point for the IEEE-CIS Fraud Detection pipeline.

Usage:
    python run.py --step ingestion
    python run.py --step ingestion --split train
    python run.py --step ingestion --nrows 5000     # quick smoke test
    python run.py --step eda
    python run.py --step all

Stages available so far:
    ingestion   Load raw CSVs, join transaction to identity, save Parquet
    eda         Profile the joined training data, write reports and charts
    all         Every stage above, in order

More stages are added in Steps 3 to 6.
"""

from __future__ import annotations

import argparse
import time


def run_ingestion_stage(args: argparse.Namespace) -> dict:
    # Imported inside the function rather than at the top of the file so
    # that starting the script is fast even when a stage is not being used.
    # pandas alone takes about a second to import.
    from src.pipelines.ingestion import run_ingestion

    splits = ["train", "test"] if args.split == "both" else [args.split]
    return run_ingestion(splits=splits, nrows=args.nrows)


def run_eda_stage(args: argparse.Namespace) -> dict:
    from src.pipelines.eda import run_eda

    return run_eda()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IEEE-CIS Fraud Detection pipeline runner",
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["ingestion", "eda", "all"],
        help="Which pipeline stage to run.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Which data split to ingest. Only affects the ingestion stage.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Read only this many rows. For quick testing, not real runs.",
    )
    args = parser.parse_args()

    started_at = time.time()

    if args.step == "ingestion":
        run_ingestion_stage(args)
    elif args.step == "eda":
        run_eda_stage(args)
    elif args.step == "all":
        run_ingestion_stage(args)
        run_eda_stage(args)

    elapsed = time.time() - started_at
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\nFinished in {minutes}m {seconds}s.")


# Only run when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()
