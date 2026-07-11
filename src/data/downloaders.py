"""
Dataset downloaders for competitive programming datasets.

Supported sources:
  - taco                  : BAAI/TACO
  - apps                  : codeparrot/apps
  - code_contests         : deepmind/code_contests
  - codeforces            : open-r1/codeforces (Codeforces problems with solutions)
  - codeforces_cots       : open-r1/codeforces-cots (chain-of-thought)
  - leetcode              : greengerong/leetcode filtered to Hard/Medium only
  - amc_aime              : math-ai/amc_aime
  - project_euler         : ajibawa-2023/project_euler
  - opencode_reasoning    : nvidia/OpenCodeReasoning (736K CP problems + reasoning traces)
  - opencode_reasoning2   : nvidia/OpenCodeReasoning-2 (~1.5M, higher quality)
  - kodcode               : KodCode/KodCode (447K verified problems with unit tests)
  - code_feedback         : m-a-p/CodeFeedback-Filtered-Instruction (157K curated OSS pairs)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

# Registry: name -> HuggingFace dataset id + kwargs
DATASET_REGISTRY: Dict[str, Dict] = {
    "taco": {
        "path": "BAAI/TACO",
        "split": "train",
        "config_name": None,
        "trust_remote_code": True,
    },
    "apps": {
        "path": "codeparrot/apps",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    "code_contests": {
        "path": "deepmind/code_contests",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    "codeforces": {
        "path": "open-r1/codeforces",
        "split": "train",
        "config_name": None,
        "trust_remote_code": True,
    },
    "codeforces_cots": {
        "path": "open-r1/codeforces-cots",
        "split": "train",
        "config_name": "all",
        "trust_remote_code": True,
    },
    "leetcode": {
        "path": "greengerong/leetcode",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    "amc_aime": {
        "path": "math-ai/amc_aime",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    "project_euler": {
        "path": "ajibawa-2023/project_euler",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    # High-quality additions
    "opencode_reasoning": {
        "path": "nvidia/OpenCodeReasoning",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    "opencode_reasoning2": {
        "path": "nvidia/OpenCodeReasoning-2",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    "kodcode": {
        "path": "KodCode/KodCode",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
    "code_feedback": {
        "path": "m-a-p/CodeFeedback-Filtered-Instruction",
        "split": "train",
        "config_name": None,
        "trust_remote_code": False,
    },
}


def download_dataset(
    name: str,
    cache_dir: str = "./data/raw",
    streaming: bool = False,
) -> Optional[Dataset]:
    """Download a single dataset by registry name. Returns None on failure."""
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
) -> Dict[str, Dataset]:
    """Download all named datasets. Skips any that fail."""
    os.makedirs(cache_dir, exist_ok=True)
    result: Dict[str, Dataset] = {}

    for name in dataset_names:
        ds = download_dataset(name, cache_dir=cache_dir, streaming=streaming)
        if ds is not None:
            result[name] = ds

    if not result:
        raise RuntimeError(
            f"None of the requested datasets could be downloaded: {dataset_names}"
        )

    logger.info("Successfully downloaded %d/%d datasets", len(result), len(dataset_names))
    return result


def list_available_datasets() -> List[str]:
    return list(DATASET_REGISTRY.keys())
