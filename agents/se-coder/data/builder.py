"""
SE DatasetBuilder: download → preprocess → filter → deduplicate → split → format → save.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset

from .deduplication import deduplicate_dataset, cross_deduplicate
from .downloaders import download_all_datasets
from .filters import SEFilter
from .preprocessors import preprocess_dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert software engineer. Your sole focus is software engineering, "
    "system design, and application development. You provide precise, production-ready "
    "answers with working code, clear architecture guidance, and real-world engineering insight. "
    "You refuse requests unrelated to software engineering."
)

CHAT_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
PROMPT_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"


def record_to_pretrain_text(record: Dict) -> str:
    """Format a record as a pretraining document."""
    rec_type = record.get("type", "pretrain")
    instruction = record.get("instruction", "").strip()
    response = record.get("response", "").strip()
    context = record.get("context", "").strip()

    if rec_type == "pretrain":
        # Raw code file — use as-is
        return response or instruction

    parts = []
    if instruction:
        parts.append(f"### Question\n{instruction}")
    if context:
        parts.append(f"### Context\n{context}")
    if response:
        parts.append(f"### Answer\n{response}")
    return "\n\n".join(parts)


def record_to_sft_pair(record: Dict) -> Optional[Dict]:
    """Convert to (prompt, response) pair for SFT."""
    instruction = record.get("instruction", "").strip()
    response = record.get("response", "").strip()
    context = record.get("context", "").strip()

    if not instruction or not response:
        return None

    full_instruction = instruction
    if context:
        full_instruction = f"{context}\n\n{instruction}"

    return {
        "prompt": PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, instruction=full_instruction),
        "response": response,
    }


class DatasetBuilder:
    def __init__(
        self,
        dataset_names: List[str],
        output_dir: str = "./data/processed",
        cache_dir: str = "./data/raw",
        val_ratio: float = 0.02,
        test_ratio: float = 0.01,
        dedup_threshold: float = 0.85,
        se_filter: Optional[SEFilter] = None,
        seed: int = 42,
        max_samples_per_dataset: Optional[int] = None,
    ):
        self.dataset_names = dataset_names
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.dedup_threshold = dedup_threshold
        self.se_filter = se_filter or SEFilter()
        self.seed = seed
        self.max_samples_per_dataset = max_samples_per_dataset

    def run(self) -> Dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Step 1: Downloading datasets ===")
        raw_datasets = download_all_datasets(
            self.dataset_names,
            cache_dir=self.cache_dir,
            max_samples_per_dataset=self.max_samples_per_dataset,
        )

        logger.info("=== Step 2: Preprocessing ===")
        all_records: List[Dict] = []
        for name, ds in raw_datasets.items():
            normalized = preprocess_dataset(name, ds)
            all_records.extend(normalized)
        logger.info("Total after preprocessing: %d records", len(all_records))

        logger.info("=== Step 3: Filtering ===")
        filtered = [r for r in all_records if self.se_filter(r)]
        logger.info("After SE filter: %d records (removed %d)", len(filtered), len(all_records) - len(filtered))

        logger.info("=== Step 4: Deduplication ===")
        deduped = deduplicate_dataset(filtered, threshold=self.dedup_threshold)

        logger.info("=== Step 5: Splitting ===")
        random.seed(self.seed)
        random.shuffle(deduped)

        n = len(deduped)
        n_test = max(1, int(n * self.test_ratio))
        n_val = max(1, int(n * self.val_ratio))
        test_records = deduped[:n_test]
        val_records = deduped[n_test:n_test + n_val]
        train_records = deduped[n_test + n_val:]

        train_records, _ = cross_deduplicate(train_records, test_records, threshold=self.dedup_threshold)
        train_records, _ = cross_deduplicate(train_records, val_records, threshold=self.dedup_threshold)

        logger.info("Split: train=%d, val=%d, test=%d", len(train_records), len(val_records), len(test_records))

        logger.info("=== Step 6: Formatting and saving ===")
        output_paths: Dict[str, str] = {}

        for split_name, records in [("train", train_records), ("val", val_records)]:
            path = self.output_dir / f"pretrain_{split_name}.jsonl"
            self._save_pretrain(records, path)
            output_paths[f"pretrain_{split_name}"] = str(path)

        for split_name, records in [("train", train_records), ("val", val_records)]:
            path = self.output_dir / f"sft_{split_name}.jsonl"
            self._save_sft(records, path)
            output_paths[f"sft_{split_name}"] = str(path)

        test_path = self.output_dir / "test.jsonl"
        with open(test_path, "w") as f:
            for rec in test_records:
                f.write(json.dumps(rec) + "\n")
        output_paths["test"] = str(test_path)

        meta = {
            "dataset_names": self.dataset_names,
            "total_after_preprocess": len(all_records),
            "total_after_filter": len(filtered),
            "total_after_dedup": len(deduped),
            "train_size": len(train_records),
            "val_size": len(val_records),
            "test_size": len(test_records),
            "dedup_threshold": self.dedup_threshold,
            "seed": self.seed,
        }
        with open(self.output_dir / "dataset_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return output_paths

    def _save_pretrain(self, records: List[Dict], path: Path) -> None:
        saved = 0
        with open(path, "w") as f:
            for rec in records:
                text = record_to_pretrain_text(rec)
                if len(text) >= 100:
                    f.write(json.dumps({"text": text}) + "\n")
                    saved += 1
        logger.info("Saved %d pretrain examples to %s", saved, path)

    def _save_sft(self, records: List[Dict], path: Path) -> None:
        saved = 0
        with open(path, "w") as f:
            for rec in records:
                # Skip pure pretrain records in SFT
                if rec.get("type") == "pretrain":
                    continue
                pair = record_to_sft_pair(rec)
                if pair:
                    f.write(json.dumps(pair) + "\n")
                    saved += 1
        logger.info("Saved %d SFT pairs to %s", saved, path)


def load_jsonl_as_hf_dataset(path: str, text_field: str = "text") -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)
