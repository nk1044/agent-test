"""
Rejection Sampling Fine-Tuning (RFT).

Why this matters: SFT data may contain incorrect or suboptimal solutions.
RFT generates many candidate solutions from the current model, keeps only
those that pass all test cases (execution-verified), then fine-tunes again
on that higher-signal dataset. This is the self-improvement loop.

Pipeline:
  1. Load a model checkpoint (typically the SFT model).
  2. For each problem in the training set, sample N candidates.
  3. Execute every candidate; keep those that pass all I/O examples or unit tests.
  4. Write verified (problem, solution) pairs to a new JSONL file.
  5. Run one SFT epoch on the verified data → stronger model.

Repeating this loop (RFT → RLVR → RFT) progressively raises the ceiling.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solution generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Think step by step, then write a complete, runnable Python solution. "
    "Wrap your code in ```python ... ``` fences."
)

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{problem}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _format_prompt(record: Dict) -> str:
    problem = record.get("problem", "")
    examples = record.get("examples", [])
    ex_block = ""
    if examples:
        parts = []
        for ex in examples[:3]:
            parts.append(f"Input:\n{ex.get('input','').strip()}\nOutput:\n{ex.get('output','').strip()}")
        ex_block = "\n\n### Examples\n" + "\n\n".join(parts)
    return PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, problem=problem + ex_block)


@torch.no_grad()
def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    n_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int = 4,
) -> List[str]:
    """Generate `n_candidates` solutions for a single prompt."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    candidates = []
    remaining = n_candidates

    while remaining > 0:
        this_batch = min(batch_size, remaining)
        input_ids = inputs["input_ids"].expand(this_batch, -1)
        attention_mask = inputs["attention_mask"].expand(this_batch, -1)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

        for seq in output_ids:
            new_tokens = seq[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            candidates.append(text)

        remaining -= this_batch

    return candidates


# ---------------------------------------------------------------------------
# Execution verification (reuse from rl_trainer)
# ---------------------------------------------------------------------------

from .rl_trainer import (
    extract_code,
    score_against_examples,
    score_against_unittests,
)


def is_solution_correct(
    completion: str,
    examples: List[Dict],
    test_code: str,
    exec_timeout: float,
) -> bool:
    code = extract_code(completion)

    if examples:
        pass_rate, _ = score_against_examples(code, examples, timeout=exec_timeout)
        return pass_rate >= 1.0

    if test_code and test_code.strip():
        pass_rate, _ = score_against_unittests(code, test_code, timeout=exec_timeout + 5)
        return pass_rate >= 1.0

    return False


# ---------------------------------------------------------------------------
# Config & main runner
# ---------------------------------------------------------------------------

@dataclass
class RFTConfig:
    model_path: str
    train_file: str = "./data/processed/sft_train.jsonl"
    output_dir: str = "./outputs/rft"
    verified_data_path: str = "./data/processed/rft_verified.jsonl"
    cache_dir: Optional[str] = None

    # Generation
    n_candidates: int = 16
    gen_batch_size: int = 4
    max_new_tokens: int = 2048
    temperature: float = 0.9
    top_p: float = 0.95

    # Execution
    exec_timeout: float = 10.0

    # SFT on verified data
    run_sft_on_verified: bool = True
    sft_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    report_to: List[str] = field(default_factory=lambda: ["wandb", "tensorboard"])
    run_name: str = "cp-llm-rft"
    deepspeed_config: Optional[str] = None


def run_rejection_sampling(cfg: RFTConfig) -> str:
    """
    Run RFT: generate → verify → save → SFT.
    Returns path to the best model after SFT on verified data.
    """
    from model.model_utils import load_model_and_tokenizer, save_model
    from training.sft_trainer import SFTConfig, run_sft

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=== Rejection Sampling Fine-Tuning ===")
    logger.info("Loading model from: %s", cfg.model_path)

    model, tokenizer = load_model_and_tokenizer(
        cfg.model_path,
        cache_dir=cfg.cache_dir,
        gradient_checkpointing=False,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Load training problems
    problems = []
    with open(cfg.train_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                examples = rec.get("examples") or []
                test_code = rec.get("test_code") or ""
                if examples or test_code:
                    problems.append(rec)
            except json.JSONDecodeError:
                continue

    logger.info("%d verifiable problems loaded for RFT", len(problems))

    verified_records = []
    total_candidates = 0
    total_verified = 0

    with open(cfg.verified_data_path, "w") as out_f:
        for rec in tqdm(problems, desc="Rejection sampling"):
            prompt = _format_prompt(rec)
            examples = rec.get("examples") or []
            test_code = rec.get("test_code") or ""

            candidates = generate_candidates(
                model, tokenizer, prompt,
                n_candidates=cfg.n_candidates,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                batch_size=cfg.gen_batch_size,
            )
            total_candidates += len(candidates)

            for completion in candidates:
                if is_solution_correct(completion, examples, test_code, cfg.exec_timeout):
                    code = extract_code(completion)
                    verified_rec = {
                        "problem": rec.get("problem", ""),
                        "solutions": [code],
                        "examples": examples,
                        "test_code": test_code,
                        "difficulty": rec.get("difficulty", "unknown"),
                        "tags": rec.get("tags") or [],
                        "source": f"rft/{rec.get('source', '')}",
                        "language": "python",
                    }
                    out_f.write(json.dumps(verified_rec) + "\n")
                    verified_records.append(verified_rec)
                    total_verified += 1
                    break  # one verified solution per problem is enough

    pass_rate = total_verified / max(len(problems), 1)
    logger.info(
        "RFT sampling: %d/%d problems solved (%.1f%% pass rate) from %d total candidates",
        total_verified, len(problems), pass_rate * 100, total_candidates,
    )

    if total_verified == 0:
        logger.warning("No solutions verified — skipping SFT on verified data.")
        return cfg.model_path

    stats_path = os.path.join(cfg.output_dir, "rft_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "n_problems": len(problems),
            "n_candidates_per_problem": cfg.n_candidates,
            "n_verified": total_verified,
            "pass_rate": pass_rate,
            "verified_data": cfg.verified_data_path,
        }, f, indent=2)
    logger.info("RFT stats saved to %s", stats_path)

    if not cfg.run_sft_on_verified:
        return cfg.model_path

    # Free model memory before SFT
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Running SFT on %d verified solutions ...", total_verified)

    sft_cfg = SFTConfig(
        base_model=cfg.model_path,
        train_file=cfg.verified_data_path,
        val_file=cfg.verified_data_path,   # small dataset — reuse as val
        output_dir=os.path.join(cfg.output_dir, "sft"),
        num_train_epochs=cfg.sft_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        seed=cfg.seed,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        deepspeed_config=cfg.deepspeed_config,
    )

    from training.sft_trainer import run_sft
    best_model = run_sft(sft_cfg)
    logger.info("RFT complete. Best model: %s", best_model)
    return best_model


def run_rft_from_config(config: Dict, model_path: str) -> str:
    train_cfg = config.get("training", {})
    gen_cfg = config.get("generation", {})
    sandbox_cfg = config.get("sandbox", {})

    cfg = RFTConfig(
        model_path=model_path,
        train_file=config.get("train_file", "./data/processed/sft_train.jsonl"),
        output_dir=train_cfg.get("output_dir", "./outputs/rft"),
        verified_data_path=config.get("verified_data_path", "./data/processed/rft_verified.jsonl"),
        n_candidates=gen_cfg.get("n_candidates", 16),
        gen_batch_size=gen_cfg.get("batch_size", 4),
        max_new_tokens=gen_cfg.get("max_new_tokens", 2048),
        temperature=gen_cfg.get("temperature", 0.9),
        top_p=gen_cfg.get("top_p", 0.95),
        exec_timeout=sandbox_cfg.get("timeout_seconds", 10.0),
        run_sft_on_verified=config.get("run_sft_on_verified", True),
        sft_epochs=train_cfg.get("sft_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", ["wandb", "tensorboard"]),
        run_name=train_cfg.get("run_name", "cp-llm-rft"),
        deepspeed_config=config.get("deepspeed_config"),
    )
    return run_rejection_sampling(cfg)
