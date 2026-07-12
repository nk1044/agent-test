#!/usr/bin/env python3
"""
Continued pretraining script.

Single GPU:
    python scripts/run_pretrain.py --config config/pretrain.yaml

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 scripts/run_pretrain.py \
        --config config/pretrain.yaml

With DeepSpeed:
    deepspeed --num_gpus=4 scripts/run_pretrain.py \
        --config config/pretrain.yaml \
        --deepspeed config/ds_zero2.json
"""

import os, sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_AGENT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _AGENT_DIR)

import argparse
import logging

from training.pretrain_trainer import run_pretraining_from_config
from utils.helpers import load_yaml, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Run continued pretraining")
    parser.add_argument("--config", default="config/pretrain.yaml")
    parser.add_argument("--base-model", help="Override base_model in config")
    parser.add_argument("--output-dir", help="Override output_dir in config")
    parser.add_argument("--resume", help="Resume from checkpoint path")
    parser.add_argument("--deepspeed", help="DeepSpeed config JSON path")
    parser.add_argument("--wandb-project", default="cp-llm", help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    # Allow passing extra training_args as --train.key=value
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    config = load_yaml(args.config)

    # Apply CLI overrides
    if args.base_model:
        config["base_model"] = args.base_model
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
    if args.resume:
        config.setdefault("training", {})["resume_from_checkpoint"] = args.resume
    if args.deepspeed:
        config["deepspeed_config"] = args.deepspeed

    # Also respect BASE_MODEL env var
    base_model_env = os.environ.get("BASE_MODEL")
    if base_model_env and not args.base_model:
        config["base_model"] = base_model_env
        logger.info("Using BASE_MODEL from environment: %s", base_model_env)

    # Set W&B project
    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        config.setdefault("training", {}).setdefault("report_to", ["wandb", "tensorboard"])
    else:
        config.setdefault("training", {})["report_to"] = ["tensorboard"]

    set_seed(config.get("training", {}).get("seed", 42))

    logger.info("Starting pretraining with base model: %s", config.get("base_model"))
    best_model_dir = run_pretraining_from_config(config)
    logger.info("Pretraining done. Best model: %s", best_model_dir)


if __name__ == "__main__":
    main()
