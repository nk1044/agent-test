"""
Model and tokenizer loading utilities for full-parameter training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recommended base models (shorthand aliases → HuggingFace model IDs)
#
# Primary recommendation: Qwen/Qwen2.5-Coder-7B
#   - Best coding 7B dense model; strongest ≤10B choice for full fine-tuning
#
# Alternative: deepseek-ai/DeepSeek-Coder-V2-Lite-Base
#   - 16B MoE, 2.4B active params. Inference cost ≈ 2.4B dense but model
#     knowledge ≈ 16B. Excellent on hard algorithmic problems.
#   - Use if you have enough VRAM to load 16B total weights (~32GB).
# ---------------------------------------------------------------------------

RECOMMENDED_MODELS = {
    # Primary — best coding 7B dense model
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    # MoE alternative — 16B total / 2.4B active, stronger on hard problems
    "deepseek-coder-v2-lite": "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
    "deepseek-coder-v2-lite-instruct": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    # Smaller fallbacks
    "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
}


def get_model_id(base_model: str) -> str:
    """Resolve shorthand aliases to full HuggingFace model IDs."""
    return RECOMMENDED_MODELS.get(base_model, base_model)


def load_tokenizer(model_id: str, cache_dir: Optional[str] = None) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )

    # Ensure pad token exists (many base models lack one)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    logger.info("Tokenizer loaded: vocab_size=%d, pad='%s'", tokenizer.vocab_size, tokenizer.pad_token)
    return tokenizer


def load_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_flash_attention: bool = False,
    gradient_checkpointing: bool = True,
    resize_token_embeddings: int = 0,
    low_cpu_mem_usage: bool = True,
) -> PreTrainedModel:
    """Load a causal LM for full-parameter training."""

    if torch_dtype is None:
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

    attn_impl = "flash_attention_2" if use_flash_attention else "eager"

    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Enable sliding window / RoPE scaling if needed for long contexts
    if hasattr(config, "max_position_embeddings") and config.max_position_embeddings < 4096:
        logger.warning(
            "Model max_position_embeddings=%d < 4096. Consider a model with longer context.",
            config.max_position_embeddings,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=True,
    )

    if resize_token_embeddings > 0 and resize_token_embeddings != model.config.vocab_size:
        model.resize_token_embeddings(resize_token_embeddings)
        logger.info("Resized token embeddings: %d → %d", model.config.vocab_size, resize_token_embeddings)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled")

    # Log parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: %s | params=%.2fB | trainable=%.2fB | dtype=%s",
        model_id, total / 1e9, trainable / 1e9, torch_dtype,
    )

    return model


def load_model_and_tokenizer(
    base_model: str,
    cache_dir: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_flash_attention: bool = False,
    gradient_checkpointing: bool = True,
    low_cpu_mem_usage: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Convenience function: load both model and tokenizer, syncing embeddings."""
    model_id = get_model_id(base_model)
    logger.info("Loading model from: %s", model_id)

    tokenizer = load_tokenizer(model_id, cache_dir=cache_dir)

    model = load_model(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        use_flash_attention=use_flash_attention,
        gradient_checkpointing=gradient_checkpointing,
        resize_token_embeddings=len(tokenizer),
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    return model, tokenizer


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    safe_serialization: bool = True,
) -> None:
    """Save model + tokenizer in HuggingFace format."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # If model is wrapped (DeepSpeed, DDP), unwrap first
    unwrapped = model
    if hasattr(model, "module"):
        unwrapped = model.module

    unwrapped.save_pretrained(output_dir, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model + tokenizer saved to: %s", output_dir)


def load_checkpoint(
    checkpoint_dir: str,
    cache_dir: Optional[str] = None,
    gradient_checkpointing: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a previously saved checkpoint."""
    return load_model_and_tokenizer(
        checkpoint_dir,
        cache_dir=cache_dir,
        gradient_checkpointing=gradient_checkpointing,
    )
