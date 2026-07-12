"""
GRPO-based RLVR (Reinforcement Learning with Verifiable Rewards) trainer.

Training loop:
  1. Sample a batch of problems from the SFT dataset.
  2. Generate G solutions per problem using the current policy (temperature sampling).
  3. Execute each solution against I/O examples or unit tests to get a reward.
  4. GRPO update: advantage-normalized policy gradient with KL penalty.

Reward functions (all combined additively, clipped to [0, 1]):
  - execution_reward : fraction of test cases passed         (weight 1.0)
  - format_reward    : +bonus for syntactically valid Python (weight 0.1)
  - efficiency_reward: +bonus for finishing under 1s         (weight 0.05)

The reward design deliberately makes execution dominant — format/efficiency
bonuses are too small to be gamed without also solving the problem.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"```(?:python|py|cpp|c\+\+|java)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code(text: str) -> str:
    """Extract the first fenced code block, or fall back to the whole text."""
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        return matches[0].strip()
    # fallback: strip prose lines that look like natural language
    lines = text.splitlines()
    code_lines = [l for l in lines if l.startswith((" ", "\t")) or re.match(r"^[a-zA-Z_]", l)]
    return "\n".join(code_lines).strip() or text.strip()


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Execution-based reward helpers (reuse evaluator's subprocess runner)
# ---------------------------------------------------------------------------

def _run_in_subprocess(code: str, stdin: str = "", timeout: float = 10.0):
    """Run code in an isolated subprocess. Returns (stdout, stderr, timed_out, elapsed)."""
    import subprocess

    MAX_OUT = 65536

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmpfile = f.name

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, tmpfile],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": ""},
        )
        elapsed = time.monotonic() - t0
        return proc.stdout[:MAX_OUT], proc.stderr[:MAX_OUT], False, elapsed
    except subprocess.TimeoutExpired:
        return "", "Timeout", True, timeout
    except Exception as exc:
        return "", str(exc), False, time.monotonic() - t0
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


def _normalize(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def score_against_examples(
    code: str,
    examples: List[Dict],
    timeout: float = 10.0,
) -> tuple[float, float]:
    """
    Returns (pass_rate, avg_elapsed).
    pass_rate = fraction of examples where stdout matches expected output.
    """
    if not examples:
        return 0.0, 0.0

    passed = 0
    total_elapsed = 0.0
    for ex in examples:
        stdin = ex.get("input", "")
        expected = _normalize(ex.get("output", ""))
        stdout, stderr, timed_out, elapsed = _run_in_subprocess(code, stdin, timeout)
        total_elapsed += elapsed
        if not timed_out and not (stderr and not stdout):
            if _normalize(stdout) == expected:
                passed += 1

    return passed / len(examples), total_elapsed / len(examples)


def score_against_unittests(
    solution: str,
    test_code: str,
    timeout: float = 15.0,
) -> tuple[float, float]:
    """
    Append test_code after solution, run with unittest discovery.
    Returns (pass_rate, elapsed).
    """
    if not test_code or not test_code.strip():
        return 0.0, 0.0

    combined = f"{solution}\n\n{test_code}\n\nimport unittest\nif __name__ == '__main__':\n    unittest.main(verbosity=2)\n"
    stdout, stderr, timed_out, elapsed = _run_in_subprocess(combined, timeout=timeout)

    if timed_out:
        return 0.0, elapsed

    # Parse unittest output: "Ran N tests" and look for failures/errors
    ran_match = re.search(r"Ran (\d+) test", stderr + stdout)
    fail_match = re.search(r"failures=(\d+)", stderr + stdout)
    err_match = re.search(r"errors=(\d+)", stderr + stdout)

    if not ran_match:
        return 0.0, elapsed

    n_ran = int(ran_match.group(1))
    if n_ran == 0:
        return 0.0, elapsed

    n_fail = int(fail_match.group(1)) if fail_match else 0
    n_err = int(err_match.group(1)) if err_match else 0
    n_passed = n_ran - n_fail - n_err
    return max(0.0, n_passed / n_ran), elapsed


_SELF_TEST_RE = re.compile(r"<tests>(.*?)</tests>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_self_tests(completion: str) -> List[Dict]:
    """Parse <tests> block. Expects lines like: INPUT -> OUTPUT"""
    match = _SELF_TEST_RE.search(completion)
    if not match:
        return []
    examples = []
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if "->" in line:
            parts = line.split("->", 1)
            examples.append({"input": parts[0].strip(), "output": parts[1].strip()})
    return examples[:5]


def has_reasoning(completion: str) -> bool:
    """Check if the model produced a non-trivial <think> block."""
    match = _THINK_RE.search(completion)
    if not match:
        return False
    return len(match.group(1).strip()) > 50


# ---------------------------------------------------------------------------
# Reward functions (GRPO-compatible signatures)
# ---------------------------------------------------------------------------

_EXEC_WEIGHT = 1.0
_FORMAT_WEIGHT = 0.1
_EFFICIENCY_WEIGHT = 0.05
_FAST_THRESHOLD_S = 1.0   # solutions finishing under this get efficiency bonus


def make_reward_fn(
    exec_timeout: float = 10.0,
    exec_weight: float = _EXEC_WEIGHT,
    format_weight: float = _FORMAT_WEIGHT,
    efficiency_weight: float = _EFFICIENCY_WEIGHT,
):
    """
    Returns a GRPO-compatible reward function.

    The reward function receives `completions` (model outputs as strings) and
    dataset columns forwarded as kwargs.
    """

    def reward_fn(
        completions: List[str],
        examples: Optional[List] = None,
        test_code: Optional[List] = None,
        **kwargs,
    ) -> List[float]:
        rewards = []

        for i, completion in enumerate(completions):
            code = extract_code(completion)

            # --- format reward (small, prevents degenerate outputs) ---
            fmt_bonus = format_weight if is_valid_python(code) else 0.0

            # --- reasoning reward (incentivise <think> blocks) ---
            reasoning_bonus = 0.08 if has_reasoning(completion) else 0.0

            # --- execution reward against provided tests ---
            ex_list = (examples[i] if examples and i < len(examples) else None) or []
            tc = (test_code[i] if test_code and i < len(test_code) else None) or ""

            exec_score = 0.0
            avg_elapsed = exec_timeout

            if ex_list:
                exec_score, avg_elapsed = score_against_examples(code, ex_list, timeout=exec_timeout)
            elif tc:
                exec_score, avg_elapsed = score_against_unittests(code, tc, timeout=exec_timeout + 5)

            exec_r = exec_weight * exec_score

            # --- self-test reward: model's OWN generated test cases ---
            # Only awarded when the model already passes provided tests — prevents
            # the model from gaming the reward by writing trivially easy self-tests.
            self_test_bonus = 0.0
            if exec_score >= 1.0 or not ex_list:
                self_tests = extract_self_tests(completion)
                if self_tests and is_valid_python(code):
                    st_rate, _ = score_against_examples(code, self_tests, timeout=exec_timeout)
                    self_test_bonus = 0.15 * st_rate

            # --- efficiency reward (only on full pass) ---
            eff_bonus = 0.0
            if exec_score >= 1.0 and avg_elapsed < _FAST_THRESHOLD_S:
                eff_bonus = efficiency_weight

            total = min(1.0, exec_r + fmt_bonus + reasoning_bonus + self_test_bonus + eff_bonus)
            rewards.append(total)

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a world-class competitive programmer. Your ONLY task is solving coding problems. "
    "You refuse all non-coding requests. "
    "For every problem: (1) think step-by-step inside <think>...</think> tags, "
    "(2) identify the algorithm and edge cases, "
    "(3) write a complete runnable Python solution in ```python...``` fences, "
    "(4) write 3–5 of your own test cases inside <tests>...</tests> tags to verify your solution."
)

SFT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{problem}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)


def _build_prompt(record: Dict) -> str:
    problem = record.get("problem", "")
    examples = record.get("examples", [])
    ex_block = ""
    if examples:
        lines = []
        for ex in examples[:3]:
            lines.append(f"Input:\n{ex.get('input', '').strip()}\nOutput:\n{ex.get('output', '').strip()}")
        ex_block = "\n\n### Examples\n" + "\n\n".join(lines)
    return SFT_TEMPLATE.format(system=SYSTEM_PROMPT, problem=problem + ex_block)


def build_rl_dataset(jsonl_path: str, difficulty_filter: Optional[str] = None) -> Dataset:
    """Load SFT JSONL and reformat for GRPO (adds 'prompt' field).

    difficulty_filter: None = all, 'medium' = medium+hard only, 'hard' = hard only
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # difficulty curriculum filter
            if difficulty_filter:
                diff = rec.get("difficulty", "unknown")
                if difficulty_filter == "hard" and diff not in ("hard",):
                    continue
                if difficulty_filter == "medium" and diff not in ("medium", "hard"):
                    continue

            examples = rec.get("examples") or []
            test_code = rec.get("test_code") or ""

            # Only keep records that have some verifiable signal
            if not examples and not test_code:
                continue

            records.append({
                "prompt": _build_prompt(rec),
                "examples": examples,
                "test_code": test_code,
                "source": rec.get("source", ""),
            })

    logger.info("RL dataset: %d verifiable problems loaded from %s", len(records), jsonl_path)
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    sft_model_path: str
    train_file: str = "./data/processed/sft_train.jsonl"
    output_dir: str = "./outputs/rl"
    cache_dir: Optional[str] = None

    # GRPO hyperparameters
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-7
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    report_to: List[str] = field(default_factory=lambda: ["wandb", "tensorboard"])
    run_name: str = "cp-llm-rl"

    # GRPO-specific
    num_generations: int = 8
    max_new_tokens: int = 2048
    temperature: float = 0.9
    top_p: float = 0.95
    beta: float = 0.04   # KL penalty weight

    # Reward
    exec_timeout: float = 10.0
    exec_weight: float = 1.0
    format_weight: float = 0.1
    efficiency_weight: float = 0.05

    deepspeed_config: Optional[str] = None
    difficulty_filter: Optional[str] = None  # None | 'medium' | 'hard'
    rl_round: int = 1  # current round number (for logging)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_rl_training(cfg: RLConfig) -> str:
    """Run GRPO-based RLVR. Returns path to best RL checkpoint."""
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError("trl >= 0.9.0 required for GRPOTrainer. Run: pip install trl>=0.9.0")

    from shared.model.model_utils import load_model_and_tokenizer, save_model
    from .callbacks import CheckpointMetadataCallback, RichProgressCallback

    logger.info("=== RLVR Training (GRPO) ===")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Load SFT model as starting policy
    model, tokenizer = load_model_and_tokenizer(
        cfg.sft_model_path,
        cache_dir=cfg.cache_dir,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    # Dataset
    train_dataset = build_rl_dataset(cfg.train_file, difficulty_filter=cfg.difficulty_filter)
    if len(train_dataset) == 0:
        raise RuntimeError(
            "RL dataset is empty — no problems with verifiable test cases found. "
            "Make sure sft_train.jsonl contains records with 'examples' or 'test_code' fields."
        )

    # Reward function
    reward_fn = make_reward_fn(
        exec_timeout=cfg.exec_timeout,
        exec_weight=cfg.exec_weight,
        format_weight=cfg.format_weight,
        efficiency_weight=cfg.efficiency_weight,
    )

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16 and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        deepspeed=cfg.deepspeed_config,
        # GRPO-specific
        num_generations=cfg.num_generations,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        beta=cfg.beta,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
    )
    trainer.add_callback(RichProgressCallback())
    trainer.add_callback(CheckpointMetadataCallback())

    logger.info("Starting RLVR training on %d verifiable problems ...", len(train_dataset))
    trainer.train()

    best_dir = os.path.join(cfg.output_dir, "best_model")
    save_model(trainer.model, tokenizer, best_dir)
    logger.info("RLVR complete. Best model at: %s", best_dir)
    return best_dir


def run_rl_from_config(config: Dict, sft_model_path: str) -> str:
    train_cfg = config.get("training", {})
    grpo_cfg = config.get("grpo", {})
    reward_cfg = config.get("reward", {})
    sandbox_cfg = config.get("sandbox", {})

    cfg = RLConfig(
        sft_model_path=sft_model_path,
        output_dir=train_cfg.get("output_dir", "./outputs/rl"),
        num_train_epochs=train_cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 5e-7),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", ["wandb", "tensorboard"]),
        run_name=train_cfg.get("run_name", "cp-llm-rl"),
        deepspeed_config=config.get("deepspeed_config"),
        num_generations=grpo_cfg.get("num_generations", 8),
        max_new_tokens=grpo_cfg.get("max_new_tokens", 2048),
        temperature=grpo_cfg.get("temperature", 0.9),
        top_p=grpo_cfg.get("top_p", 0.95),
        beta=grpo_cfg.get("kl_coeff", grpo_cfg.get("beta", 0.04)),
        exec_timeout=sandbox_cfg.get("timeout_seconds", 10.0),
        exec_weight=reward_cfg.get("execution_weight", 1.0),
        format_weight=reward_cfg.get("format_weight", 0.1),
        efficiency_weight=reward_cfg.get("efficiency_weight", 0.05),
        difficulty_filter=config.get("difficulty_filter"),
        rl_round=config.get("rl_round", 1),
    )
    return run_rl_training(cfg)
