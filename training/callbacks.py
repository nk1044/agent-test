"""
Custom training callbacks.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class RichProgressCallback(TrainerCallback):
    """Pretty progress using rich if available, otherwise plain logging."""

    def __init__(self):
        self._start_time: Optional[float] = None
        try:
            from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
            self._rich = True
        except ImportError:
            self._rich = False

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._start_time = time.time()
        logger.info("Training started — max_steps=%s", state.max_steps)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        if logs and state.is_local_process_zero:
            elapsed = time.time() - (self._start_time or time.time())
            step = state.global_step
            loss = logs.get("loss", logs.get("train_loss", "?"))
            lr = logs.get("learning_rate", "?")
            logger.info("[step %d/%d | %.0fs] loss=%.4f lr=%s", step, state.max_steps, elapsed, loss if isinstance(loss, float) else 0, lr)


class EarlyStoppingOnNaN(TrainerCallback):
    """Stop training if loss becomes NaN or inf."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss") or logs.get("train_loss")
            if loss is not None:
                import math
                if math.isnan(loss) or math.isinf(loss):
                    logger.error("Loss is %s at step %d — stopping training!", loss, state.global_step)
                    control.should_training_stop = True


class CheckpointMetadataCallback(TrainerCallback):
    """Write a JSON file alongside each checkpoint with training metadata."""

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        import json
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        meta = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "log_history": state.log_history[-5:],
        }
        meta_path = os.path.join(ckpt_dir, "training_meta.json")
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:
            logger.warning("Could not write checkpoint metadata: %s", exc)


class WandbArtifactCallback(TrainerCallback):
    """Log best model checkpoint as a W&B artifact at end of training."""

    def __init__(self, artifact_name: str = "cp-llm-checkpoint"):
        self.artifact_name = artifact_name

    def on_train_end(self, args, state, control, **kwargs):
        try:
            import wandb
            if wandb.run is None:
                return
            artifact = wandb.Artifact(self.artifact_name, type="model")
            artifact.add_dir(args.output_dir)
            wandb.run.log_artifact(artifact)
            logger.info("W&B artifact logged: %s", self.artifact_name)
        except Exception as exc:
            logger.warning("W&B artifact logging failed: %s", exc)


class TokenThroughputCallback(TrainerCallback):
    """Log tokens-per-second throughput."""

    def __init__(self):
        self._last_step = 0
        self._last_time = time.time()
        self._last_tokens = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            now = time.time()
            dt = now - self._last_time
            if dt > 0 and "num_input_tokens_seen" in (logs or {}):
                tokens = logs["num_input_tokens_seen"]
                tps = (tokens - self._last_tokens) / dt
                logger.info("Throughput: %.0f tokens/sec", tps)
                self._last_tokens = tokens
                self._last_time = now
