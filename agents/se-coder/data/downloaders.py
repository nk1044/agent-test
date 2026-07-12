"""
Dataset downloaders for software engineering training.

Supported sources:
  - the_stack_python        : bigcode/the-stack-v2-dedup (Python subset)
  - the_stack_js            : bigcode/the-stack-v2-dedup (JavaScript subset)
  - the_stack_ts            : bigcode/the-stack-v2-dedup (TypeScript subset)
  - the_stack_go            : bigcode/the-stack-v2-dedup (Go subset)
  - the_stack_java          : bigcode/the-stack-v2-dedup (Java subset)
  - the_stack_rust          : bigcode/the-stack-v2-dedup (Rust subset)
  - commitpackft            : bigcode/commitpackft (702K filtered commits with messages)
  - stack_exchange          : ArmelR/stack-exchange-instruction (SO/SE Q&A)
  - magicoder_oss           : ise-uiuc/Magicoder-OSS-Instruct-75K
  - magicoder_evol          : ise-uiuc/Magicoder-Evol-Instruct-110K
  - code_feedback           : m-a-p/CodeFeedback-Filtered-Instruction
  - evol_codealpaca         : theblackcat102/evol-codealpaca-v1
  - glaive_code             : glaiveai/glaive-code-assistant-v3
  - text_to_sql             : gretelai/synthetic-text-to-sql
  - ultrachat               : HuggingFaceH4/ultrachat_200k
  - self_oss_instruct       : bigcode/self-oss-instruct-sc2-exec-filter-50k
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

# Registry: name -> HuggingFace dataset id + kwargs
DATASET_REGISTRY: Dict[str, Dict] = {
    # The Stack v2 — by language (streaming recommended for large subsets)
    "the_stack_python": {
        "path": "bigcode/the-stack-v2-dedup",
        "split": "train",
        "config_name": "Python",
        "trust_remote_code": False,
    },
    "the_stack_js": {
        "path": "bigcode/the-stack-v2-dedup",
        "split": "train",
        "config_name": "JavaScript",
        "trust_remote_code": False,
    },
    "the_stack_ts": {
        "path": "bigcode/the-stack-v2-dedup",
        "split": "train",
        "config_name": "TypeScript",
        "trust_remote_code": False,
    },
    "the_stack_go": {
        "path": "bigcode/the-stack-v2-dedup",
        "split": "train",
        "config_name": "Go",
        "trust_remote_code": False,
    },
    "the_stack_java": {
        "path": "bigcode/the-stack-v2-dedup",
        "split": "train",
        "config_name": "Java",
        "trust_remote_code": False,
    },
    "the_stack_rust": {
        "path": "bigcode/the-stack-v2-dedup",
        "split": "train",
        "config_name": "Rust",
        "trust_remote_code": False,
    },
    "the_stack_sql": {
        "path": "bigcode/the-stack-v2-dedup",
        "split": "train",
        "config_name": "SQL",
        "trust_remote_code": False,
    },
    "the_stack_shell": {
        "path": "bigcode/the-stack-v2-dedup",
        "split": "train",
        "config_name": "Shell",
        "trust_remote_code": False,
    },
    # Commits — real-world code changes with natural language descriptions
    "commitpackft": {
        "path": "bigcode/commitpackft",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    # StackExchange Q&A (SO, Server Fault, Unix, etc.)
    "stack_exchange": {
        "path": "ArmelR/stack-exchange-instruction",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    # Instruction pairs — real OSS code as seeds
    "magicoder_oss": {
        "path": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    # Instruction pairs — evolved diverse coding instructions
    "magicoder_evol": {
        "path": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    # Instruction pairs — filtered code feedback
    "code_feedback": {
        "path": "m-a-p/CodeFeedback-Filtered-Instruction",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    # Evolved code alpaca pairs
    "evol_codealpaca": {
        "path": "theblackcat102/evol-codealpaca-v1",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    # Conversational code assistant
    "glaive_code": {
        "path": "glaiveai/glaive-code-assistant-v3",
        "split": "train",
        "config_name": None,
        "trust_remote_code": True,
    },
    # Text-to-SQL
    "text_to_sql": {
        "path": "gretelai/synthetic-text-to-sql",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    # High-quality multi-turn technical chat
    "ultrachat": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "config_name": None,
        "trust_remote_code": False,
    },
    # OSS code instruction pairs with execution filter
    "self_oss_instruct": {
        "path": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
}


def download_dataset(
    name: str,
    cache_dir: str = "./data/raw",
    streaming: bool = False,
    max_samples: Optional[int] = None,
) -> Optional[Dataset]:
    """Download a single dataset by registry name. Returns None on failure.

    For large datasets like the_stack_*, use streaming=True or set max_samples
    to avoid downloading terabytes of data.
    """
    if name not in DATASET_REGISTRY:
        logger.warning("Unknown dataset '%s'. Available: %s", name, list(DATASET_REGISTRY))
        return None

    reg = DATASET_REGISTRY[name]
    os.makedirs(cache_dir, exist_ok=True)

    logger.info("Downloading '%s' from '%s' ...", name, reg["path"])
    try:
        ds = load_dataset(
            reg["path"],
            reg.get("config_name"),
            split=reg["split"],
            cache_dir=cache_dir,
            trust_remote_code=reg.get("trust_remote_code", False),
            streaming=streaming,
        )
        if not streaming and max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        if not streaming:
            logger.info("Downloaded '%s': %d examples", name, len(ds))
        else:
            logger.info("Opened '%s' as streaming dataset", name)
        return ds
    except Exception as exc:
        logger.error("Failed to download '%s': %s", name, exc)
        return None


def download_all_datasets(
    dataset_names: List[str],
    cache_dir: str = "./data/raw",
    streaming: bool = False,
    max_samples_per_dataset: Optional[int] = None,
) -> Dict[str, Dataset]:
    """Download all named datasets. Skips any that fail."""
    os.makedirs(cache_dir, exist_ok=True)
    result: Dict[str, Dataset] = {}

    for name in dataset_names:
        ds = download_dataset(name, cache_dir=cache_dir, streaming=streaming, max_samples=max_samples_per_dataset)
        if ds is not None:
            result[name] = ds

    if not result:
        raise RuntimeError(f"None of the requested datasets could be downloaded: {dataset_names}")

    logger.info("Successfully downloaded %d/%d datasets", len(result), len(dataset_names))
    return result


def list_available_datasets() -> List[str]:
    return list(DATASET_REGISTRY.keys())
