"""SE-coder model utilities — re-exports shared model loading with SE-optimized defaults."""
import os, sys
_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_AGENT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from shared.model.model_utils import (  # noqa: F401
    RECOMMENDED_MODELS,
    get_model_id,
    load_tokenizer,
    load_model,
    load_model_and_tokenizer,
    save_model,
)

# SE-recommended base models
SE_RECOMMENDED_MODELS = {
    # Primary: best coding+language balance for SE tasks
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    # 8B alternative with strong instruction following
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    # MoE — 16B total / 2.4B active, excellent at broad SE tasks
    "deepseek-coder-v2-lite": "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
}
