"""Thin re-export from shared — keeps agent self-contained if shared/ moves."""
from shared.model.model_utils import (  # noqa: F401
    RECOMMENDED_MODELS,
    get_model_id,
    load_tokenizer,
    load_model,
    load_model_and_tokenizer,
    save_model,
)
