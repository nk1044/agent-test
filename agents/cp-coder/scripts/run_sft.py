#!/usr/bin/env python3
"""
Supervised fine-tuning script.

Single GPU:
    python scripts/run_sft.py --config config/sft.yaml

Multi-GPU:
    torchrun --nproc_per_node=4 scripts/run_sft.py --config config/sft.yaml

From pretrained checkpoint:
    python scripts/run_sft.py \
        --config config/sft.yaml \
        --base-model ./outputs/pretrain/best_model
"""

import os, sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_AGENT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _AGENT_DIR)

import argparse
import logging

from training.sft_trainer import run_sft_from_config
from utils.helpers import load_yaml, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning")
    parser.add_argument("--config", default="config/sft.yaml")
    parser.add_argument("--base-model", help="Override base_model (use pretrain checkpoint)")
    parser.add_argument("--output-dir", help="Override output_dir")
    parser.add_argument("--resume", help="Resume from checkpoint path")
    parser.add_argument("--deepspeed", help="DeepSpeed config JSON path")
    parser.add_argument("--wandb-project", default="cp-llm")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = load_yaml(args.config)

    if args.base_model:
        config["base_model"] = args.base_model
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
    if args.resume:
        config.setdefault("training", {})["resume_from_checkpoint"] = args.resume
    if args.deepspeed:
        config["deepspeed_config"] = args.deepspeed

    base_model_env = os.environ.get("BASE_MODEL")
    if base_model_env and not args.base_model:
        config["base_model"] = base_model_env

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        config.setdefault("training", {}).setdefault("report_to", ["wandb", "tensorboard"])
    else:
        config.setdefault("training", {})["report_to"] = ["tensorboard"]

    set_seed(config.get("training", {}).get("seed", 42))

    logger.info("Starting SFT with base model: %s", config.get("base_model"))
    best_model_dir = run_sft_from_config(config)
    logger.info("SFT done. Best model: %s", best_model_dir)


if __name__ == "__main__":
    main()
