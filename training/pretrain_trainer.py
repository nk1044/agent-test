"""
Continued pretraining stage.

Trains the base model on the corpus using a causal language-modeling objective
(next-token prediction on packed sequences).

Supports:
  - Full-parameter training (no LoRA)
  - Mixed precision (bf16 / fp16)
  - Gradient checkpointing
  - DeepSpeed ZeRO-2 / ZeRO-3
  - Multi-GPU via Accelerate / torchrun
  - Sequence packing for efficiency
  - Resumable from checkpoint
  - WandB + TensorBoard logging
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from data.builder import load_jsonl_as_hf_dataset
from model.model_utils import load_model_and_tokenizer, save_model
from .callbacks import (
    CheckpointMetadataCallback,
    EarlyStoppingOnNaN,
    RichProgressCallback,
    TokenThroughputCallback,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence packing dataset
# ---------------------------------------------------------------------------

class PackedDataset(torch.utils.data.Dataset):
    """
    Concatenates tokenized documents and splits into fixed-length chunks.
    This fully utilizes every token without padding waste.
    """

    def __init__(
        self,
        raw_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        text_field: str = "text",
        num_proc: int = 4,
    ):
        self.max_seq_length = max_seq_length
        logger.info("Tokenizing and packing %d documents ...", len(raw_dataset))

        def tokenize(batch):
            return tokenizer(
                batch[text_field],
                add_special_tokens=True,
                truncation=False,
            )

        tokenized = raw_dataset.map(
            tokenize,
            batched=True,
            num_proc=num_proc,
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing",
        )

        all_ids: List[int] = []
        eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        for item in tokenized:
            all_ids.extend(item["input_ids"])
            all_ids.append(eos_id)

        self.chunks: List[List[int]] = [
            all_ids[i : i + max_seq_length]
            for i in range(0, len(all_ids) - max_seq_length, max_seq_length)
        ]
        logger.info(
            "Packing done: %d documents → %d chunks of length %d",
            len(raw_dataset), len(self.chunks), max_seq_length,
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


class SimpleTokenizedDataset(torch.utils.data.Dataset):
    """Non-packed tokenized dataset (used when packing=False)."""

    def __init__(
        self,
        raw_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        text_field: str = "text",
        num_proc: int = 4,
    ):
        def tokenize(batch):
            return tokenizer(
                batch[text_field],
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_length,
                padding=False,
            )

        tokenized = raw_dataset.map(
            tokenize,
            batched=True,
            num_proc=num_proc,
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing",
        )
        tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 10)
        self._data = tokenized

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        ids = torch.tensor(item["input_ids"], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": ids.clone(),
        }


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    # Paths
    train_file: str = "./data/processed/pretrain_train.jsonl"
    val_file: str = "./data/processed/pretrain_val.jsonl"
    output_dir: str = "./outputs/pretrain"
    cache_dir: Optional[str] = None

    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-7B"
    use_flash_attention: bool = False

    # Data
    max_seq_length: int = 4096
    text_field: str = "text"
    packing: bool = True
    dataloader_num_workers: int = 4

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 20
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    report_to: List[str] = None
    run_name: str = "cp-llm-pretrain"
    deepspeed_config: Optional[str] = None


def build_training_args(cfg: PretrainConfig) -> TrainingArguments:
    report_to = cfg.report_to or ["none"]

    return TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16 and torch.cuda.is_bf16_supported(),
        fp16=cfg.fp16 and not (cfg.bf16 and torch.cuda.is_bf16_supported()),
        tf32=cfg.tf32,
        logging_steps=cfg.logging_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        group_by_length=not cfg.packing,
        seed=cfg.seed,
        report_to=report_to,
        run_name=cfg.run_name,
        deepspeed=cfg.deepspeed_config,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        torch_compile=False,
    )


def run_pretraining(cfg: PretrainConfig) -> str:
    """Run continued pretraining. Returns path to the best model checkpoint."""
    logger.info("=== Continued Pretraining ===")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        cfg.base_model,
        cache_dir=cfg.cache_dir,
        use_flash_attention=cfg.use_flash_attention,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    logger.info("Loading training data from %s", cfg.train_file)
    raw_train = load_jsonl_as_hf_dataset(cfg.train_file, text_field=cfg.text_field)
    raw_val = load_jsonl_as_hf_dataset(cfg.val_file, text_field=cfg.text_field)

    DatasetClass = PackedDataset if cfg.packing else SimpleTokenizedDataset
    train_dataset = DatasetClass(
        raw_train, tokenizer, cfg.max_seq_length,
        text_field=cfg.text_field,
        num_proc=cfg.dataloader_num_workers,
    )
    val_dataset = DatasetClass(
        raw_val, tokenizer, cfg.max_seq_length,
        text_field=cfg.text_field,
        num_proc=cfg.dataloader_num_workers,
    )

    logger.info("Train size: %d chunks | Val size: %d chunks", len(train_dataset), len(val_dataset))

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = build_training_args(cfg)

    callbacks = [
        RichProgressCallback(),
        EarlyStoppingOnNaN(),
        CheckpointMetadataCallback(),
        TokenThroughputCallback(),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    logger.info("Starting training ...")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    best_model_dir = os.path.join(cfg.output_dir, "best_model")
    save_model(trainer.model, tokenizer, best_model_dir)

    logger.info("Running final evaluation ...")
    metrics = trainer.evaluate()
    logger.info("Final val metrics: %s", metrics)

    import json
    with open(os.path.join(cfg.output_dir, "pretrain_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Pretraining complete. Best model at: %s", best_model_dir)
    return best_model_dir


def run_pretraining_from_config(config: Dict) -> str:
    """Build PretrainConfig from a dict and run pretraining."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    cfg = PretrainConfig(
        base_model=config.get("base_model", PretrainConfig.base_model),
        train_file=os.path.join(data_cfg.get("processed_dir", "./data/processed"), data_cfg.get("train_file", "pretrain_train.jsonl")),
        val_file=os.path.join(data_cfg.get("processed_dir", "./data/processed"), data_cfg.get("val_file", "pretrain_val.jsonl")),
        output_dir=train_cfg.get("output_dir", "./outputs/pretrain"),
        max_seq_length=data_cfg.get("max_seq_length", 4096),
        text_field=data_cfg.get("text_field", "text"),
        packing=config.get("packing", True),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        tf32=train_cfg.get("tf32", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 20),
        eval_steps=train_cfg.get("eval_steps", 500),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        seed=train_cfg.get("seed", 42),
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"),
        report_to=train_cfg.get("report_to", ["none"]),
        run_name=train_cfg.get("run_name", "cp-llm-pretrain"),
        deepspeed_config=config.get("deepspeed_config"),
    )
    return run_pretraining(cfg)
