"""
SE-Coder: Software Engineering LLM Training Pipeline.

Quick start (single GPU):
    BASE_MODEL=Qwen/Qwen2.5-Coder-7B \\
    SE_DATASETS="stack_exchange magicoder_oss magicoder_evol code_feedback" \\
    python agents/se-coder/train.py

Full run with distributed training:
    torchrun --nproc_per_node=4 agents/se-coder/train.py \\
        --base-model Qwen/Qwen2.5-Coder-7B \\
        --datasets stack_exchange magicoder_oss magicoder_evol code_feedback \\
                   evol_codealpaca glaive_code text_to_sql self_oss_instruct \\
        --stages data pretrain sft eval \\
        --output-dir ./outputs/se \\
        --deepspeed config/ds_zero2.json

For The Stack v2 (very large — use --max-samples to limit):
    python agents/se-coder/train.py \\
        --datasets the_stack_python the_stack_js the_stack_ts stack_exchange magicoder_oss \\
        --max-samples 500000 \\
        --stages data pretrain sft eval
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Path setup — must be before any project imports
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_AGENT_DIR))
sys.path.insert(0, _PROJECT_ROOT)   # enables: from shared.X import ...
sys.path.insert(0, _AGENT_DIR)      # enables: from data.X import ..., from evaluation.X import ...

from utils.helpers import load_yaml, save_yaml, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SE-Coder: Software Engineering LLM Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-model", default=None,
                        help="HuggingFace model ID or local path. Default: Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names (overrides SE_DATASETS env var)")
    parser.add_argument("--stages", nargs="+",
                        default=["data", "pretrain", "sft", "eval"],
                        choices=["data", "pretrain", "sft", "eval"],
                        help="Pipeline stages to run")
    parser.add_argument("--pretrain-config", default="config/pretrain.yaml")
    parser.add_argument("--sft-config", default="config/sft.yaml")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--data-dir", default="./data/processed")
    parser.add_argument("--cache-dir", default="./data/raw")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (use for large datasets like The Stack)")
    parser.add_argument("--epochs-pretrain", type=int, default=None)
    parser.add_argument("--epochs-sft", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr-pretrain", type=float, default=None)
    parser.add_argument("--lr-sft", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--eval-max-problems", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="se-llm")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--skip-existing-data", action="store_true")
    return parser.parse_args()


def run_data_stage(args, datasets: list, data_dir: str) -> dict:
    from data.builder import DatasetBuilder
    from data.filters import SEFilter

    logger.info("=== STAGE: Data Preparation ===")

    if args.skip_existing_data:
        pretrain_train = Path(data_dir) / "pretrain_train.jsonl"
        if pretrain_train.exists():
            logger.info("Processed data found — skipping download")
            return {k: str(Path(data_dir) / f"{k}.jsonl") for k in
                    ["pretrain_train", "pretrain_val", "sft_train", "sft_val", "test"]}

    builder = DatasetBuilder(
        dataset_names=datasets,
        output_dir=data_dir,
        cache_dir=args.cache_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dedup_threshold=args.dedup_threshold,
        se_filter=SEFilter(),
        seed=args.seed,
        max_samples_per_dataset=args.max_samples,
    )
    return builder.run()


def run_pretrain_stage(args, base_model: str, data_paths: dict, output_dir: str) -> str:
    from shared.training.pretrain_trainer import PretrainConfig, run_pretraining

    logger.info("=== STAGE: Continued Pretraining ===")

    config_file = args.pretrain_config
    config = load_yaml(config_file) if Path(config_file).exists() else {}
    pretrain_output = os.path.join(output_dir, "pretrain")

    cfg = PretrainConfig(
        base_model=base_model,
        train_file=data_paths.get("pretrain_train", "./data/processed/pretrain_train.jsonl"),
        val_file=data_paths.get("pretrain_val", "./data/processed/pretrain_val.jsonl"),
        output_dir=pretrain_output,
        max_seq_length=args.max_seq_length or config.get("data", {}).get("max_seq_length", 8192),
        packing=config.get("packing", True),
        use_flash_attention=args.flash_attention,
        num_train_epochs=args.epochs_pretrain or config.get("training", {}).get("num_train_epochs", 1),
        per_device_train_batch_size=args.batch_size or config.get("training", {}).get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=args.grad_accum or config.get("training", {}).get("gradient_accumulation_steps", 16),
        learning_rate=args.lr_pretrain or config.get("training", {}).get("learning_rate", 1e-5),
        seed=args.seed,
        deepspeed_config=args.deepspeed or config.get("deepspeed_config"),
        report_to=["tensorboard"] if args.no_wandb else ["wandb", "tensorboard"],
        run_name="se-llm-pretrain",
    )
    return run_pretraining(cfg)


def run_sft_stage(args, base_model: str, data_paths: dict, output_dir: str) -> str:
    from shared.training.sft_trainer import SFTConfig, run_sft

    logger.info("=== STAGE: Supervised Fine-Tuning ===")

    config_file = args.sft_config
    config = load_yaml(config_file) if Path(config_file).exists() else {}
    sft_output = os.path.join(output_dir, "sft")

    cfg = SFTConfig(
        base_model=base_model,
        train_file=data_paths.get("sft_train", "./data/processed/sft_train.jsonl"),
        val_file=data_paths.get("sft_val", "./data/processed/sft_val.jsonl"),
        output_dir=sft_output,
        max_seq_length=args.max_seq_length or config.get("data", {}).get("max_seq_length", 8192),
        use_flash_attention=args.flash_attention,
        num_train_epochs=args.epochs_sft or config.get("training", {}).get("num_train_epochs", 3),
        per_device_train_batch_size=args.batch_size or config.get("training", {}).get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=args.grad_accum or config.get("training", {}).get("gradient_accumulation_steps", 16),
        learning_rate=args.lr_sft or config.get("training", {}).get("learning_rate", 5e-6),
        seed=args.seed,
        deepspeed_config=args.deepspeed or config.get("deepspeed_config"),
        report_to=["tensorboard"] if args.no_wandb else ["wandb", "tensorboard"],
        run_name="se-llm-sft",
    )
    return run_sft(cfg)


def run_eval_stage(args, model_path: str, data_paths: dict, output_dir: str):
    from evaluation.se_evaluator import SEEvaluator, load_test_records
    from shared.model.model_utils import load_model_and_tokenizer
    import torch

    logger.info("=== STAGE: Evaluation ===")

    model, tokenizer = load_model_and_tokenizer(model_path, gradient_checkpointing=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    test_file = data_paths.get("test", "./data/processed/test.jsonl")
    test_records = load_test_records(test_file)

    eval_output = os.path.join(output_dir, "eval")
    evaluator = SEEvaluator(model=model, tokenizer=tokenizer, max_problems=args.eval_max_problems)
    results = evaluator.evaluate(test_records, output_dir=eval_output)
    print()
    print(results.summary())
    return results


def main():
    args = parse_args()
    setup_logging(args.log_level, log_file=os.path.join(args.output_dir, "training.log"))
    set_seed(args.seed)

    base_model = args.base_model or os.environ.get("BASE_MODEL") or "Qwen/Qwen2.5-Coder-7B"
    logger.info("Base model: %s", base_model)

    if args.datasets:
        datasets = args.datasets
    else:
        env_datasets = os.environ.get("SE_DATASETS", "")
        datasets = env_datasets.split() if env_datasets else [
            "stack_exchange", "magicoder_oss", "magicoder_evol",
            "code_feedback", "evol_codealpaca", "self_oss_instruct",
        ]
    logger.info("Datasets: %s", datasets)

    stages = args.stages
    logger.info("Stages: %s", stages)

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args), resume="allow")
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)
            args.no_wandb = True

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    data_paths = {}

    save_yaml({"base_model": base_model, "datasets": datasets, "stages": stages},
              os.path.join(args.output_dir, "run_config.yaml"))

    if "data" in stages:
        data_paths = run_data_stage(args, datasets, args.data_dir)
    else:
        data_paths = {
            "pretrain_train": os.path.join(args.data_dir, "pretrain_train.jsonl"),
            "pretrain_val":   os.path.join(args.data_dir, "pretrain_val.jsonl"),
            "sft_train":      os.path.join(args.data_dir, "sft_train.jsonl"),
            "sft_val":        os.path.join(args.data_dir, "sft_val.jsonl"),
            "test":           os.path.join(args.data_dir, "test.jsonl"),
        }

    pretrained_model_path = base_model
    if "pretrain" in stages:
        pretrained_model_path = run_pretrain_stage(args, base_model, data_paths, args.output_dir)

    final_model_path = pretrained_model_path
    if "sft" in stages:
        final_model_path = run_sft_stage(args, pretrained_model_path, data_paths, args.output_dir)

    if "eval" in stages and Path(data_paths.get("test", "")).exists():
        run_eval_stage(args, final_model_path, data_paths, args.output_dir)

    logger.info("=== SE Training Pipeline Complete ===")
    logger.info("Final model: %s", final_model_path)
    logger.info("Next: python scripts/export_gguf.sh --model %s", final_model_path)


if __name__ == "__main__":
    main()
