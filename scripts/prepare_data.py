#!/usr/bin/env python3
"""
Data preparation script.

Downloads, preprocesses, deduplicates, and saves the training corpus.

Usage:
    python scripts/prepare_data.py \
        --datasets taco apps code_contests codeforces \
        --output-dir ./data/processed \
        --cache-dir ./data/raw \
        --dedup-threshold 0.85 \
        --val-ratio 0.02 \
        --test-ratio 0.01
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.builder import DatasetBuilder
from src.data.filters import CPFilter
from src.utils.helpers import set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CP training data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["taco", "apps", "code_contests", "codeforces"],
        help="Dataset names to include (space-separated)",
    )
    parser.add_argument("--output-dir", default="./data/processed")
    parser.add_argument("--cache-dir", default="./data/raw")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--max-solutions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cp-signal", action="store_true",
                        help="Require positive CP keyword signal (stricter filter)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info("Datasets to download: %s", args.datasets)

    cp_filter = CPFilter(require_positive_signal=args.require_cp_signal)

    builder = DatasetBuilder(
        dataset_names=args.datasets,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dedup_threshold=args.dedup_threshold,
        max_solutions_per_problem=args.max_solutions,
        cp_filter=cp_filter,
        seed=args.seed,
    )

    output_paths = builder.run()

    logger.info("Data preparation complete!")
    logger.info("Output files:")
    for name, path in output_paths.items():
        logger.info("  %-20s %s", name, path)


if __name__ == "__main__":
    main()
