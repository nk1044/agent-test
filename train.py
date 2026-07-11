#!/usr/bin/env python3
"""
Main entry point for the CP-LLM training pipeline.

Runs the full pipeline end-to-end:
  1. Download + preprocess datasets
  2. Continued pretraining (CPT)
  3. Supervised fine-tuning (SFT)
  4. Evaluation on test set
  (5. Export to GGUF is done separately via scripts/export_gguf.sh)

Configuration:
  - Dataset names are set via CODING_DATASETS (env var or --datasets flag)
  - Base model is set via BASE_MODEL (env var or --base-model flag)
  - Stages can be run selectively with --stages flag

Quick start (single GPU, minimal config):
    BASE_MODEL=Qwen/Qwen2.5-Coder-7B \\
    CODING_DATASETS="taco apps code_contests" \\
    python train.py

Full run with distributed training:
    torchrun --nproc_per_node=4 train.py \\
        --base-model Qwen/Qwen2.5-Coder-7B \\
        --datasets taco apps code_contests codeforces \\
        --stages data pretrain sft eval \\
        --pretrain-config config/pretrain.yaml \\
        --sft-config config/sft.yaml \\
        --output-dir ./outputs
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from src.utils.helpers import load_yaml, save_yaml, set_seed, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CP-LLM: Competitive Programming Language Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core settings (also readable from environment variables)
    parser.add_argument(
        "--base-model",
        default=None,
        help="HuggingFace model ID or local path (overrides BASE_MODEL env var). "
             "Default: Qwen/Qwen2.5-Coder-7B",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names (overrides CODING_DATASETS env var). "
             "Supported: taco, apps, code_contests, codeforces, codeforces_cots, leetcode",
    )

    # Pipeline stages
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["data", "pretrain", "sft", "eval"],
        choices=["data", "pretrain", "sft", "eval"],
        help="Which pipeline stages to run (default: all)",
    )

    # Config files
    parser.add_argument("--pretrain-config", default="config/pretrain.yaml")
    parser.add_argument("--sft-config", default="config/sft.yaml")

    # Output
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--data-dir", default="./data/processed")
    parser.add_argument("--cache-dir", default="./data/raw")

    # Data options
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)

    # Training overrides
    parser.add_argument("--epochs-pretrain", type=int, default=None)
    parser.add_argument("--epochs-sft", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr-pretrain", type=float, default=None)
    parser.add_argument("--lr-sft", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)

    # DeepSpeed
    parser.add_argument("--deepspeed", default=None, help="DeepSpeed config JSON path")

    # Evaluation
    parser.add_argument("--eval-n-samples", type=int, default=10)
    parser.add_argument("--eval-max-problems", type=int, default=200)
    parser.add_argument("--no-sandbox", action="store_true")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="cp-llm")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--skip-existing-data", action="store_true",
                        help="Skip data download if processed files already exist")
    parser.add_argument(
        "--sft-from-pretrain",
        action="store_true",
        default=True,
        help="Use pretrain checkpoint as starting point for SFT (default: True)",
    )
    parser.add_argument("--no-sft-from-pretrain", dest="sft_from_pretrain", action="store_false")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_data_stage(args, datasets: list, data_dir: str) -> dict:
    """Download, preprocess, deduplicate, split, and save training data."""
    from src.data.builder import DatasetBuilder
    from src.data.filters import CPFilter

    logger.info("=== STAGE: Data Preparation ===")

    # Check if we can skip
    if args.skip_existing_data:
        pretrain_train = Path(data_dir) / "pretrain_train.jsonl"
        if pretrain_train.exists():
            logger.info("Processed data found at %s — skipping download", data_dir)
            return {
                "pretrain_train": str(Path(data_dir) / "pretrain_train.jsonl"),
                "pretrain_val":   str(Path(data_dir) / "pretrain_val.jsonl"),
                "sft_train":      str(Path(data_dir) / "sft_train.jsonl"),
                "sft_val":        str(Path(data_dir) / "sft_val.jsonl"),
                "test":           str(Path(data_dir) / "test.jsonl"),
            }

    builder = DatasetBuilder(
        dataset_names=datasets,
        output_dir=data_dir,
        cache_dir=args.cache_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dedup_threshold=args.dedup_threshold,
        cp_filter=CPFilter(),
        seed=args.seed,
    )
    return builder.run()


def run_pretrain_stage(args, base_model: str, data_paths: dict, output_dir: str) -> str:
    """Continued pretraining on CP corpus."""
    from src.training.pretrain_trainer import PretrainConfig, run_pretraining

    logger.info("=== STAGE: Continued Pretraining ===")

    config_file = args.pretrain_config
    if Path(config_file).exists():
        config = load_yaml(config_file)
    else:
        logger.warning("Config %s not found, using defaults", config_file)
        config = {}

    pretrain_output = os.path.join(output_dir, "pretrain")

    cfg = PretrainConfig(
        base_model=base_model,
        train_file=data_paths.get("pretrain_train", "./data/processed/pretrain_train.jsonl"),
        val_file=data_paths.get("pretrain_val", "./data/processed/pretrain_val.jsonl"),
        output_dir=pretrain_output,
        max_seq_length=args.max_seq_length or config.get("data", {}).get("max_seq_length", 4096),
        packing=config.get("packing", True),
        use_flash_attention=args.flash_attention,
        num_train_epochs=args.epochs_pretrain or config.get("training", {}).get("num_train_epochs", 3),
        per_device_train_batch_size=args.batch_size or config.get("training", {}).get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=args.grad_accum or config.get("training", {}).get("gradient_accumulation_steps", 16),
        learning_rate=args.lr_pretrain or config.get("training", {}).get("learning_rate", 2e-5),
        seed=args.seed,
        deepspeed_config=args.deepspeed or config.get("deepspeed_config"),
        report_to=["tensorboard"] if args.no_wandb else ["wandb", "tensorboard"],
        run_name="cp-llm-pretrain",
    )

    return run_pretraining(cfg)


def run_sft_stage(args, base_model: str, data_paths: dict, output_dir: str) -> str:
    """Supervised fine-tuning on problem/solution pairs."""
    from src.training.sft_trainer import SFTConfig, run_sft

    logger.info("=== STAGE: Supervised Fine-Tuning ===")

    config_file = args.sft_config
    if Path(config_file).exists():
        config = load_yaml(config_file)
    else:
        config = {}

    sft_output = os.path.join(output_dir, "sft")

    cfg = SFTConfig(
        base_model=base_model,
        train_file=data_paths.get("sft_train", "./data/processed/sft_train.jsonl"),
        val_file=data_paths.get("sft_val", "./data/processed/sft_val.jsonl"),
        output_dir=sft_output,
        max_seq_length=args.max_seq_length or config.get("data", {}).get("max_seq_length", 4096),
        use_flash_attention=args.flash_attention,
        num_train_epochs=args.epochs_sft or config.get("training", {}).get("num_train_epochs", 5),
        per_device_train_batch_size=args.batch_size or config.get("training", {}).get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=args.grad_accum or config.get("training", {}).get("gradient_accumulation_steps", 16),
        learning_rate=args.lr_sft or config.get("training", {}).get("learning_rate", 1e-5),
        seed=args.seed,
        deepspeed_config=args.deepspeed or config.get("deepspeed_config"),
        report_to=["tensorboard"] if args.no_wandb else ["wandb", "tensorboard"],
        run_name="cp-llm-sft",
    )

    return run_sft(cfg)


def run_eval_stage(args, model_path: str, data_paths: dict, output_dir: str):
    """Evaluate the final model on the test set."""
    from src.evaluation.cp_evaluator import CPEvaluator, load_test_records
    from src.model.model_utils import load_model_and_tokenizer

    logger.info("=== STAGE: Evaluation ===")

    import torch

    model, tokenizer = load_model_and_tokenizer(
        model_path, gradient_checkpointing=False
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    test_file = data_paths.get("test", "./data/processed/test.jsonl")
    test_records = load_test_records(test_file)

    eval_output = os.path.join(output_dir, "eval")
    evaluator = CPEvaluator(
        model=model,
        tokenizer=tokenizer,
        max_problems=args.eval_max_problems,
        use_sandbox=not args.no_sandbox,
    )
    results = evaluator.evaluate(
        test_records,
        n_samples=args.eval_n_samples,
        k_values=[1, 5, 10],
        output_dir=eval_output,
    )
    print()
    print(results.summary())
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging(
        args.log_level,
        log_file=os.path.join(args.output_dir, "training.log"),
    )
    set_seed(args.seed)

    # ── Resolve BASE_MODEL ─────────────────────────────────────────────────
    base_model = (
        args.base_model
        or os.environ.get("BASE_MODEL")
        or "Qwen/Qwen2.5-Coder-7B"
    )
    logger.info("Base model: %s", base_model)

    # ── Resolve CODING_DATASETS ────────────────────────────────────────────
    if args.datasets:
        datasets = args.datasets
    else:
        env_datasets = os.environ.get("CODING_DATASETS", "")
        datasets = env_datasets.split() if env_datasets else ["taco", "apps", "code_contests", "codeforces"]
    logger.info("Datasets: %s", datasets)

    stages = args.stages
    logger.info("Stages to run: %s", stages)

    # ── W&B setup ─────────────────────────────────────────────────────────
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args), resume="allow")
        except Exception as exc:
            logger.warning("W&B init failed: %s (continuing without W&B)", exc)
            args.no_wandb = True

    # ── Paths ─────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    data_paths = {}

    # Save run config
    run_config = {
        "base_model": base_model,
        "datasets": datasets,
        "stages": stages,
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
        "seed": args.seed,
    }
    save_yaml(run_config, os.path.join(args.output_dir, "run_config.yaml"))

    # ── Stage: Data ────────────────────────────────────────────────────────
    if "data" in stages:
        data_paths = run_data_stage(args, datasets, args.data_dir)
    else:
        # Use existing files
        data_paths = {
            "pretrain_train": os.path.join(args.data_dir, "pretrain_train.jsonl"),
            "pretrain_val":   os.path.join(args.data_dir, "pretrain_val.jsonl"),
            "sft_train":      os.path.join(args.data_dir, "sft_train.jsonl"),
            "sft_val":        os.path.join(args.data_dir, "sft_val.jsonl"),
            "test":           os.path.join(args.data_dir, "test.jsonl"),
        }

    # ── Stage: Pretraining ────────────────────────────────────────────────
    pretrained_model_path = base_model
    if "pretrain" in stages:
        pretrained_model_path = run_pretrain_stage(
            args, base_model, data_paths, args.output_dir
        )

    # ── Stage: SFT ────────────────────────────────────────────────────────
    final_model_path = pretrained_model_path
    if "sft" in stages:
        # Start SFT from pretrain output if available
        sft_start_model = pretrained_model_path if args.sft_from_pretrain else base_model
        logger.info("SFT starting from: %s", sft_start_model)
        final_model_path = run_sft_stage(
            args, sft_start_model, data_paths, args.output_dir
        )

    # ── Stage: Evaluation ─────────────────────────────────────────────────
    if "eval" in stages and Path(data_paths.get("test", "")).exists():
        run_eval_stage(args, final_model_path, data_paths, args.output_dir)

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== Training Pipeline Complete ===")
    logger.info("Final model: %s", final_model_path)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Evaluate: python scripts/evaluate.py --model %s", final_model_path)
    logger.info("  2. Export:   bash scripts/export_gguf.sh --model %s", final_model_path)
    logger.info("  3. Run:      ollama run cp-coder")

    if not args.no_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
