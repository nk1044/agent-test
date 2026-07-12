#!/usr/bin/env python3
"""
Evaluate a trained model on the CP test set.

Usage:
    python scripts/evaluate.py \
        --model ./outputs/sft/best_model \
        --test-file ./data/processed/test.jsonl \
        --n-samples 10 \
        --k 1 5 10 \
        --output-dir ./outputs/eval \
        --max-problems 200
"""

import os, sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_AGENT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _AGENT_DIR)

import argparse
import logging

from evaluation.cp_evaluator import CPEvaluator, load_test_records
from model.model_utils import load_model_and_tokenizer
from utils.helpers import set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CP model")
    parser.add_argument("--model", required=True, help="Path to trained model / HF model ID")
    parser.add_argument("--test-file", default="./data/processed/test.jsonl")
    parser.add_argument("--n-samples", type=int, default=10, help="Candidate solutions per problem")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10], help="k values for pass@k")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--output-dir", default="./outputs/eval")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-sandbox", action="store_true", help="Disable code execution")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)
    logger = logging.getLogger(__name__)

    logger.info("Loading model from: %s", args.model)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        cache_dir=args.cache_dir,
        gradient_checkpointing=False,
    )
    model.eval()

    # Move to GPU if available
    import torch
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model on GPU")
    else:
        logger.info("Model on CPU (evaluation will be slow)")

    logger.info("Loading test records from: %s", args.test_file)
    test_records = load_test_records(args.test_file)
    logger.info("Test records: %d", len(test_records))

    evaluator = CPEvaluator(
        model=model,
        tokenizer=tokenizer,
        max_problems=args.max_problems,
        use_sandbox=not args.no_sandbox,
    )

    results = evaluator.evaluate(
        test_records,
        n_samples=args.n_samples,
        k_values=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
    )

    print()
    print(results.summary())
    print()
    logger.info("Results saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
