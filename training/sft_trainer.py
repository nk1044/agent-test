"""
Supervised Fine-Tuning (SFT) stage.

Trains on (prompt, solution) pairs where the loss is applied only to the
response tokens (the solution), not the prompt.

Supports:
  - Full-parameter training (no LoRA)
  - Prompt masking (loss only on response)
  - Mixed precision, gradient checkpointing, DeepSpeed
  - Resumable from checkpoint
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from data.builder import load_jsonl_as_hf_dataset
from model.model_utils import load_model_and_tokenizer, save_model
from .callbacks import (
    CheckpointMetadataCallback,
    EarlyStoppingOnNaN,
    RichProgressCallback,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset with prompt masking
# ---------------------------------------------------------------------------

class SFTDataset(torch.utils.data.Dataset):
    """
    Tokenizes (prompt, response) pairs and masks the prompt tokens in labels
    so the loss is computed only on the response (solution) tokens.
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        raw_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        prompt_field: str = "prompt",
        response_field: str = "response",
        train_on_prompt: bool = False,
        num_proc: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_on_prompt = train_on_prompt

        logger.info("Building SFT dataset from %d raw examples ...", len(raw_dataset))
        self._items = []

        for record in raw_dataset:
            prompt = record.get(prompt_field, "")
            response = record.get(response_field, "")
            if not prompt or not response:
                continue
            item = self._encode(prompt, response)
            if item is not None:
                self._items.append(item)

        logger.info("SFT dataset: %d valid pairs (from %d raw)", len(self._items), len(raw_dataset))

    def _encode(self, prompt: str, response: str) -> Optional[Dict]:
        full_text = prompt + response + self.tokenizer.eos_token

        full_enc = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors=None,
        )
        input_ids = full_enc["input_ids"]

        if len(input_ids) < 10:
            return None

        labels = list(input_ids)

        if not self.train_on_prompt:
            prompt_enc = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors=None,
            )
            prompt_len = len(prompt_enc["input_ids"])
            for i in range(min(prompt_len, len(labels))):
                labels[i] = self.IGNORE_INDEX

        if all(l == self.IGNORE_INDEX for l in labels):
            return None

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(full_enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class SFTCollator:
    """Pad a batch of SFT examples to the same length."""

    IGNORE_INDEX = -100

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.pad_id = tokenizer.pad_token_id or 0

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids_list, attn_mask_list, labels_list = [], [], []
        for item in batch:
            n = item["input_ids"].shape[0]
            pad_len = max_len - n
            input_ids_list.append(
                torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_id, dtype=torch.long)])
            )
            attn_mask_list.append(
                torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([item["labels"], torch.full((pad_len,), self.IGNORE_INDEX, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_mask_list),
            "labels": torch.stack(labels_list),
        }


# ---------------------------------------------------------------------------
# Config + runner
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    # Paths
    train_file: str = "./data/processed/sft_train.jsonl"
    val_file: str = "./data/processed/sft_val.jsonl"
    output_dir: str = "./outputs/sft"
    cache_dir: Optional[str] = None

    # Model
    base_model: str = "./outputs/pretrain/best_model"
    use_flash_attention: bool = False

    # Data
    max_seq_length: int = 4096
    prompt_field: str = "prompt"
    response_field: str = "response"
    train_on_prompt: bool = False
    dataloader_num_workers: int = 4

    # Training
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    report_to: List[str] = None
    run_name: str = "cp-llm-sft"
    deepspeed_config: Optional[str] = None


def run_sft(cfg: SFTConfig) -> str:
    """Run supervised fine-tuning. Returns path to the best model."""
    logger.info("=== Supervised Fine-Tuning ===")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        cfg.base_model,
        cache_dir=cfg.cache_dir,
        use_flash_attention=cfg.use_flash_attention,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    raw_train = load_jsonl_as_hf_dataset(cfg.train_file)
    raw_val = load_jsonl_as_hf_dataset(cfg.val_file)

    train_dataset = SFTDataset(
        raw_train, tokenizer, cfg.max_seq_length,
        prompt_field=cfg.prompt_field,
        response_field=cfg.response_field,
        train_on_prompt=cfg.train_on_prompt,
        num_proc=cfg.dataloader_num_workers,
    )
    val_dataset = SFTDataset(
        raw_val, tokenizer, cfg.max_seq_length,
        prompt_field=cfg.prompt_field,
        response_field=cfg.response_field,
        train_on_prompt=cfg.train_on_prompt,
        num_proc=cfg.dataloader_num_workers,
    )

    logger.info("SFT train: %d | val: %d examples", len(train_dataset), len(val_dataset))

    collator = SFTCollator(tokenizer)
    report_to = cfg.report_to or ["none"]

    training_args = TrainingArguments(
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
        seed=cfg.seed,
        report_to=report_to,
        run_name=cfg.run_name,
        deepspeed=cfg.deepspeed_config,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    callbacks = [
        RichProgressCallback(),
        EarlyStoppingOnNaN(),
        CheckpointMetadataCallback(),
    ]

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    logger.info("Starting SFT ...")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    best_model_dir = os.path.join(cfg.output_dir, "best_model")
    save_model(trainer.model, tokenizer, best_model_dir)

    metrics = trainer.evaluate()
    logger.info("Final SFT val metrics: %s", metrics)

    with open(os.path.join(cfg.output_dir, "sft_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("SFT complete. Best model at: %s", best_model_dir)
    return best_model_dir


def run_sft_from_config(config: Dict) -> str:
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})

    cfg = SFTConfig(
        base_model=config.get("base_model", SFTConfig.base_model),
        train_file=os.path.join(data_cfg.get("processed_dir", "./data/processed"), data_cfg.get("train_file", "sft_train.jsonl")),
        val_file=os.path.join(data_cfg.get("processed_dir", "./data/processed"), data_cfg.get("val_file", "sft_val.jsonl")),
        output_dir=train_cfg.get("output_dir", "./outputs/sft"),
        max_seq_length=data_cfg.get("max_seq_length", 4096),
        prompt_field=data_cfg.get("prompt_field", "prompt"),
        response_field=data_cfg.get("response_field", "response"),
        train_on_prompt=data_cfg.get("train_on_prompt", False),
        num_train_epochs=train_cfg.get("num_train_epochs", 5),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        tf32=train_cfg.get("tf32", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_steps=train_cfg.get("eval_steps", 200),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        seed=train_cfg.get("seed", 42),
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"),
        report_to=train_cfg.get("report_to", ["none"]),
        run_name=train_cfg.get("run_name", "cp-llm-sft"),
        deepspeed_config=config.get("deepspeed_config"),
    )
    return run_sft(cfg)
