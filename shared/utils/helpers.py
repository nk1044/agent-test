import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """Configure root logger with optional file output. Only rank 0 writes to file."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file and rank == 0:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Quiet noisy libraries
    for lib in ("urllib3", "filelock", "git", "PIL"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_size(n: int) -> str:
    """Format a parameter count as a human-readable string."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def get_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except ImportError:
        pass
    return int(os.environ.get("RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str) -> None:
    import yaml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def bytes_to_gb(b: int) -> float:
    return b / (1024 ** 3)


def print_model_info(model, name: str = "Model") -> None:
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    print(f"{name}: {format_size(total)} total params, {format_size(trainable)} trainable")
