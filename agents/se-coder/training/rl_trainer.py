"""
GRPO-based RLVR for software engineering tasks.

Unlike CP-RLVR (binary pass/fail on algorithmic test cases), SE-RLVR uses a
multi-dimensional reward that reflects real software engineering quality:

  execution_reward : Python code runs without error           (weight 1.0)
  sql_reward       : SQL executes against SQLite               (weight 1.0)
  syntax_reward    : code is syntactically valid               (weight 0.2)
  format_reward    : proper ```lang...``` fences present       (weight 0.1)
  reasoning_bonus  : <think>...</think> block present          (weight 0.08)
  self_test_bonus  : model writes + passes its own <tests>     (weight 0.15)

Reward is execution-dominant by design — format/reasoning bonuses are too
small to be gamed without also producing working code.

Datasets used for RLVR (records with executable/verifiable code):
  - self_oss_instruct  : Python functions with execution filter
  - code_feedback      : code Q&A with runnable examples
  - magicoder_oss      : OSS-seeded Python code
  - text_to_sql        : SQL generation with schema context
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SE system prompt — structured reasoning for software engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a world-class software engineer. Your ONLY task is solving software engineering problems. "
    "You refuse all non-SE requests. "
    "For every task: (1) think through the design and edge cases inside <think>...</think> tags, "
    "(2) write complete, production-ready code in ```language...``` fences, "
    "(3) write 2-3 of your own test cases inside <tests>...</tests> tags to verify your implementation."
)

SFT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{task}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```(?P<lang>\w*)\s*\n(?P<code>.*?)```", re.DOTALL)
_THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TESTS_TAG = re.compile(r"<tests>(.*?)</tests>", re.DOTALL)


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Return [(language, code), ...] for all fenced blocks."""
    return [(m.group("lang").lower() or "python", m.group("code")) for m in _CODE_FENCE.finditer(text)]


def has_reasoning(text: str) -> bool:
    m = _THINK_TAG.search(text)
    return bool(m and len(m.group(1).strip()) > 30)


def extract_self_tests(text: str) -> str:
    m = _TESTS_TAG.search(text)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------

def _check_python_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _run_python(code: str, timeout: float = 5.0) -> Tuple[bool, str]:
    """Execute Python code. Returns (success, stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_sql(sql: str, schema_ddl: str = "") -> Tuple[bool, str]:
    """Execute SQL against an in-memory SQLite database. Returns (success, error)."""
    try:
        conn = sqlite3.connect(":memory:")
        if schema_ddl:
            try:
                conn.executescript(schema_ddl)
            except sqlite3.Error:
                pass
        conn.execute(sql)
        conn.close()
        return True, ""
    except sqlite3.Error as exc:
        return False, str(exc)


def score_python_code(code: str, timeout: float = 5.0) -> float:
    """
    0.0 = syntax error
    0.3 = valid syntax but runtime error
    0.5 = timeout (may be correct but slow)
    1.0 = runs successfully
    """
    if not _check_python_syntax(code):
        return 0.0
    success, err = _run_python(code, timeout=timeout)
    if success:
        return 1.0
    if "timeout" in err:
        return 0.5
    return 0.3


def score_sql_code(sql: str, context: str = "") -> float:
    """
    0.0 = not valid SQL at all
    0.5 = valid syntax but execution error (e.g., missing table)
    1.0 = executes successfully
    """
    sql = sql.strip()
    if not sql or not re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|WITH)\b", sql, re.IGNORECASE):
        return 0.0
    success, _ = _run_sql(sql, schema_ddl=context)
    return 1.0 if success else 0.5


def score_self_tests(primary_code: str, self_tests: str, timeout: float = 5.0) -> float:
    """Run self-generated tests against the primary code. Returns 0.0 or 0.15."""
    if not self_tests or not primary_code:
        return 0.0
    combined = primary_code + "\n\n" + self_tests
    if not _check_python_syntax(combined):
        return 0.0
    success, _ = _run_python(combined, timeout=timeout)
    return 0.15 if success else 0.0


def compute_reward(response: str, record: Dict, timeout: float = 5.0) -> float:
    """
    Compute total reward for an SE model response.
    Returns a float in [0.0, 1.0].
    """
    rec_type = record.get("type", "code")
    code_blocks = extract_code_blocks(response)
    reasoning_bonus = 0.08 if has_reasoning(response) else 0.0

    # Format: has at least one fenced code block
    format_bonus = 0.1 if code_blocks else 0.0

    self_tests = extract_self_tests(response)

    if rec_type == "sql":
        context = record.get("context", "")
        sql_blocks = [code for lang, code in code_blocks if "sql" in lang or lang == ""]
        if not sql_blocks and code_blocks:
            sql_blocks = [code for _, code in code_blocks]
        exec_reward = max((score_sql_code(c, context) for c in sql_blocks), default=0.0)
        return min(1.0, exec_reward + format_bonus + reasoning_bonus)

    elif rec_type in ("code", "debug"):
        py_blocks = [code for lang, code in code_blocks if lang in ("python", "py", "")]
        if not py_blocks and code_blocks:
            # Try any block that looks like Python
            py_blocks = [code for _, code in code_blocks if _check_python_syntax(code)]

        exec_reward = max((score_python_code(c, timeout) for c in py_blocks), default=0.0)

        # Self-test bonus only when execution already succeeded (prevents easy gaming)
        self_test_bonus = 0.0
        if exec_reward >= 0.9 and self_tests and py_blocks:
            self_test_bonus = score_self_tests(py_blocks[0], self_tests, timeout)

        return min(1.0, exec_reward + format_bonus + reasoning_bonus + self_test_bonus)

    else:
        # QA / design / pretrain — reward based on structure quality only
        syntax_bonus = 0.2 if any(_check_python_syntax(c) for _, c in code_blocks) else 0.0
        return min(1.0, format_bonus + reasoning_bonus + syntax_bonus)


# ---------------------------------------------------------------------------
# GRPO config + dataset
# ---------------------------------------------------------------------------

@dataclass
class SEGRPOConfig:
    # Model
    base_model: str = "./outputs/sft/best_model"
    use_flash_attention: bool = False
    cache_dir: Optional[str] = None

    # Data
    sft_train_file: str = "./data/processed/sft_train.jsonl"
    rl_output_dir: str = "./outputs/rl"
    code_types: List[str] = field(default_factory=lambda: ["code", "debug", "sql"])

    # GRPO
    num_generations: int = 6
    max_new_tokens: int = 3072
    temperature: float = 0.85
    top_p: float = 0.95
    kl_coeff: float = 0.04

    # Training
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
    report_to: List[str] = None
    run_name: str = "se-llm-rl"
    deepspeed_config: Optional[str] = None

    # Sandbox
    timeout_seconds: float = 8.0


def build_rl_dataset(sft_train_file: str, code_types: List[str]) -> List[Dict]:
    """
    Load SE SFT training data and filter to records suitable for RLVR:
    - type must be in code_types (e.g., "code", "debug", "sql")
    - must have a non-trivial instruction
    - Python records: instruction implies a function/implementation task
    - SQL records: has some SQL-like instruction
    """
    records = []
    try:
        with open(sft_train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") not in code_types:
                    continue
                instruction = rec.get("instruction", "").strip()
                if len(instruction) < 20:
                    continue
                records.append(rec)
    except FileNotFoundError:
        logger.warning("SFT train file not found: %s", sft_train_file)

    logger.info("RL dataset: %d records (types: %s)", len(records), code_types)
    return records


def make_prompt(record: Dict) -> str:
    instruction = record.get("instruction", "")
    context = record.get("context", "")
    full_task = f"{context}\n\n{instruction}".strip() if context else instruction
    return SFT_TEMPLATE.format(system=SYSTEM_PROMPT, task=full_task)


def make_reward_fn(timeout: float = 8.0):
    """Return a reward function compatible with TRL GRPOTrainer."""
    def reward_fn(completions: List[str], prompts: List[str] = None, batch: List[Dict] = None, **kwargs) -> List[float]:
        rewards = []
        for i, completion in enumerate(completions):
            record = (batch[i] if batch else {}) if isinstance(batch, list) else {}
            r = compute_reward(completion, record, timeout=timeout)
            rewards.append(r)
        return rewards
    return reward_fn


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_se_rl_training(cfg: SEGRPOConfig) -> str:
    """Run SE GRPO RLVR. Returns path to best model."""
    try:
        from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError("trl>=0.12.0 required: pip install trl>=0.12.0")

    Path(cfg.rl_output_dir).mkdir(parents=True, exist_ok=True)

    from model.model_utils import load_model_and_tokenizer, save_model

    logger.info("=== SE Reinforcement Learning (GRPO) ===")
    model, tokenizer = load_model_and_tokenizer(
        cfg.base_model,
        use_flash_attention=cfg.use_flash_attention,
        cache_dir=cfg.cache_dir,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    records = build_rl_dataset(cfg.sft_train_file, cfg.code_types)
    if not records:
        logger.error("No RL training records found. Make sure sft_train.jsonl has type=code/debug/sql records.")
        return cfg.base_model

    from datasets import Dataset
    prompts = [make_prompt(r) for r in records]
    hf_dataset = Dataset.from_dict({
        "prompt": prompts,
        "record": [json.dumps(r) for r in records],
    })

    reward_fn = make_reward_fn(timeout=cfg.timeout_seconds)

    def wrapped_reward(completions, prompts=None, **kwargs):
        batch_records = []
        if "record" in kwargs:
            for r_str in kwargs["record"]:
                try:
                    batch_records.append(json.loads(r_str))
                except Exception:
                    batch_records.append({})
        else:
            batch_records = [{}] * len(completions)
        return reward_fn(completions, prompts=prompts, batch=batch_records)

    report_to = cfg.report_to or ["none"]
    grpo_config = TRLGRPOConfig(
        output_dir=cfg.rl_output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=report_to,
        run_name=cfg.run_name,
        deepspeed=cfg.deepspeed_config,
        num_generations=cfg.num_generations,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        kl_coeff=cfg.kl_coeff,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=wrapped_reward,
        args=grpo_config,
        train_dataset=hf_dataset,
    )

    logger.info("Starting SE GRPO training on %d records ...", len(records))
    trainer.train()

    best_model_dir = os.path.join(cfg.rl_output_dir, "best_model")
    save_model(trainer.model, tokenizer, best_model_dir)
    logger.info("SE RL training complete. Best model: %s", best_model_dir)
    return best_model_dir


def run_se_rl_from_config(config: Dict) -> str:
    train_cfg = config.get("training", {})
    grpo_cfg = config.get("grpo", {})
    cfg = SEGRPOConfig(
        base_model=config.get("base_model", SEGRPOConfig.base_model),
        rl_output_dir=train_cfg.get("output_dir", "./outputs/rl"),
        num_generations=grpo_cfg.get("num_generations", 6),
        max_new_tokens=grpo_cfg.get("max_new_tokens", 3072),
        temperature=grpo_cfg.get("temperature", 0.85),
        top_p=grpo_cfg.get("top_p", 0.95),
        kl_coeff=grpo_cfg.get("kl_coeff", 0.04),
        num_train_epochs=train_cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 5e-7),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", ["none"]),
        run_name=train_cfg.get("run_name", "se-llm-rl"),
        deepspeed_config=config.get("deepspeed_config"),
        timeout_seconds=config.get("sandbox", {}).get("timeout_seconds", 8.0),
    )
    return run_se_rl_training(cfg)
