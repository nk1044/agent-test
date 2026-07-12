"""
Rejection Sampling Fine-Tuning (RFT) for software engineering tasks.

Pipeline:
  1. Load the RL-trained (or SFT-trained) model.
  2. For each SE problem in the training set, sample N candidate solutions.
  3. Execute / validate each candidate:
       - Python: runs without error → kept
       - SQL: executes in SQLite → kept
       - Other: syntactically valid code block → kept
  4. Write verified (prompt, response) pairs to rft_verified.jsonl.
  5. Run one SFT epoch on the verified data → stronger model.

The self-improvement loop: SFT → RL → RFT → RL → RFT ...
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


@dataclass
class SERFTConfig:
    # Model
    base_model: str = "./outputs/rl/best_model"
    cache_dir: Optional[str] = None
    use_flash_attention: bool = False

    # Data
    sft_train_file: str = "./data/processed/sft_train.jsonl"
    output_dir: str = "./outputs/rft"
    verified_file: str = "./outputs/rft/rft_verified.jsonl"

    # Sampling
    num_candidates: int = 12
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    batch_size: int = 2

    # Execution
    timeout_seconds: float = 8.0

    # SFT on verified data
    sft_epochs: int = 1
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    seed: int = 42
    deepspeed_config: Optional[str] = None
    report_to: List[str] = None


def _load_sft_records(path: str, max_records: Optional[int] = None) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
            if max_records and len(records) >= max_records:
                break
    return records


def _verify_response(response: str, record: Dict, timeout: float = 8.0) -> bool:
    """Return True if the response passes execution checks for this record type."""
    from .rl_trainer import extract_code_blocks, score_python_code, score_sql_code, _check_python_syntax

    rec_type = record.get("type", "code")
    code_blocks = extract_code_blocks(response)

    if not code_blocks:
        return False

    if rec_type == "sql":
        context = record.get("context", "")
        sql_blocks = [code for lang, code in code_blocks if "sql" in lang or lang == ""]
        if not sql_blocks:
            sql_blocks = [code for _, code in code_blocks]
        return any(score_sql_code(c, context) >= 0.9 for c in sql_blocks)

    elif rec_type in ("code", "debug"):
        py_blocks = [code for lang, code in code_blocks if lang in ("python", "py", "")]
        if not py_blocks:
            py_blocks = [code for _, code in code_blocks if _check_python_syntax(code)]
        return any(score_python_code(c, timeout) >= 0.9 for c in py_blocks)

    else:
        # For QA/design, accept if there's any valid code block
        return any(_check_python_syntax(code) for _, code in code_blocks)


def run_rejection_sampling(cfg: SERFTConfig) -> str:
    """Generate candidates, keep verified ones, SFT on them. Returns best model path."""
    from model.model_utils import load_model_and_tokenizer, save_model
    from .rl_trainer import SYSTEM_PROMPT, SFT_TEMPLATE

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("=== SE Rejection Sampling FT ===")

    # Load records
    records = _load_sft_records(cfg.sft_train_file)
    # Only use verifiable types
    records = [r for r in records if r.get("type") in ("code", "debug", "sql")]
    logger.info("RFT: %d verifiable records", len(records))

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        cfg.base_model,
        use_flash_attention=getattr(cfg, "use_flash_attention", False),
        cache_dir=cfg.cache_dir,
        gradient_checkpointing=False,  # inference mode
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Generate and verify
    verified: List[Dict] = []
    skipped = 0

    for rec in tqdm(records, desc="Generating candidates"):
        instruction = rec.get("instruction", "")
        context = rec.get("context", "")
        full_task = f"{context}\n\n{instruction}".strip() if context else instruction
        prompt = SFT_TEMPLATE.format(system=SYSTEM_PROMPT, task=full_task)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        accepted = None
        try:
            with torch.no_grad():
                for _ in range(cfg.num_candidates):
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=True,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                    response = tokenizer.decode(new_ids, skip_special_tokens=True)

                    if _verify_response(response, rec, timeout=cfg.timeout_seconds):
                        accepted = response
                        break
        except Exception as exc:
            logger.debug("Generation error: %s", exc)
            skipped += 1
            continue

        if accepted:
            verified.append({"prompt": prompt, "response": accepted})

    logger.info(
        "RFT sampling: %d verified / %d total (%.1f%% accept rate, %d errors)",
        len(verified), len(records), 100 * len(verified) / max(1, len(records)), skipped,
    )

    if not verified:
        logger.warning("No verified solutions found. Returning base model.")
        return cfg.base_model

    # Save verified pairs
    with open(cfg.verified_file, "w") as f:
        for pair in verified:
            f.write(json.dumps(pair) + "\n")
    logger.info("Saved %d verified pairs to %s", len(verified), cfg.verified_file)

    # Free model memory before SFT
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # SFT on verified data
    from training.sft_trainer import SFTConfig, run_sft

    sft_cfg = SFTConfig(
        base_model=cfg.base_model,
        train_file=cfg.verified_file,
        val_file=cfg.verified_file,  # small set — use same for val
        output_dir=os.path.join(cfg.output_dir, "sft_on_verified"),
        max_seq_length=4096,
        num_train_epochs=cfg.sft_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        deepspeed_config=cfg.deepspeed_config,
        report_to=cfg.report_to or ["none"],
        run_name="se-llm-rft",
    )

    best_model = run_sft(sft_cfg)
    logger.info("SE RFT complete. Best model: %s", best_model)
    return best_model


def run_se_rft_from_config(config: Dict) -> str:
    train_cfg = config.get("training", {})
    cfg = SERFTConfig(
        base_model=config.get("base_model", SERFTConfig.base_model),
        sft_train_file=config.get("sft_train_file", "./data/processed/sft_train.jsonl"),
        output_dir=train_cfg.get("output_dir", "./outputs/rft"),
        verified_file=config.get("verified_file", "./outputs/rft/rft_verified.jsonl"),
        num_candidates=config.get("num_candidates", 12),
        max_new_tokens=config.get("max_new_tokens", 2048),
        temperature=config.get("temperature", 0.8),
        timeout_seconds=config.get("sandbox", {}).get("timeout_seconds", 8.0),
        sft_epochs=train_cfg.get("sft_epochs", 1),
        learning_rate=train_cfg.get("learning_rate", 5e-6),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        seed=train_cfg.get("seed", 42),
        deepspeed_config=config.get("deepspeed_config"),
        report_to=train_cfg.get("report_to", ["none"]),
    )
    return run_rejection_sampling(cfg)
