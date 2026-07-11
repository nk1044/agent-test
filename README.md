# CP-LLM: Competitive Programming Language Model

Full-parameter training pipeline for a 1–2B parameter language model specialized exclusively in competitive programming and algorithmic problem solving.

## Features

- **Full-parameter training** (no LoRA/adapters) on 1–2B base models
- **Two-stage pipeline**: continued pretraining (CPT) → supervised fine-tuning (SFT)
- **Multi-dataset support**: TACO, APPS, DeepMind CodeContests, Codeforces, and more
- **Automatic**: download → preprocess → deduplicate → filter → split → train → eval → export
- **CP-focused filtering**: keeps only algorithmic/mathematical problems; rejects web dev, ML, DevOps
- **Efficient training**: mixed precision (bf16), gradient checkpointing, sequence packing, DeepSpeed ZeRO
- **Distributed training**: multi-GPU via `torchrun` or `deepspeed`
- **Resumable**: automatic checkpoint saving and resumption
- **Evaluation**: pass@k metric with sandboxed code execution
- **GGUF export**: ready for deployment in [Ollama](https://ollama.ai)

## Recommended Base Models (1–2B)

| Model | Size | HF ID |
|-------|------|-------|
| DeepSeek-Coder 1.3B | 1.3B | `deepseek-ai/deepseek-coder-1.3b-base` |
| Qwen2.5-Coder 1.5B | 1.5B | `Qwen/Qwen2.5-Coder-1.5B` |
| TinyLlama 1.1B | 1.1B | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| StableLM 2 1.6B | 1.6B | `stabilityai/stablelm-2-1_6b` |
| Gemma-2 2B | 2B | `google/gemma-2-2b` |

## Supported Datasets

| Key | HuggingFace ID | Description |
|-----|----------------|-------------|
| `taco` | BAAI/TACO | Train your Algorithms for Competitive Programming |
| `apps` | codeparrot/apps | Automated Programming Progress Standard |
| `code_contests` | deepmind/code_contests | DeepMind CodeContests (CF, CC, AtCoder) |
| `codeforces` | open-r1/codeforces | Codeforces problems + solutions |
| `codeforces_cots` | open-r1/codeforces-cots | Codeforces with chain-of-thought |
| `leetcode` | greengerong/leetcode | LeetCode (Medium/Hard only) |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# For distributed training:
pip install deepspeed
# For Flash Attention (Linux/CUDA):
pip install flash-attn --no-build-isolation
```

### 2. Run full pipeline (single GPU)

```bash
# Simplest — uses env vars
BASE_MODEL=deepseek-ai/deepseek-coder-1.3b-base \
CODING_DATASETS="taco apps code_contests codeforces" \
python train.py

# Or with explicit flags
python train.py \
    --base-model deepseek-ai/deepseek-coder-1.3b-base \
    --datasets taco apps code_contests codeforces \
    --output-dir ./outputs \
    --stages data pretrain sft eval
```

### 3. Multi-GPU training (torchrun)

```bash
torchrun --nproc_per_node=4 train.py \
    --base-model deepseek-ai/deepseek-coder-1.3b-base \
    --datasets taco apps code_contests codeforces \
    --deepspeed config/ds_zero2.json
```

### 4. Run individual stages

```bash
# Data preparation only
python scripts/prepare_data.py \
    --datasets taco apps code_contests codeforces

# Pretraining only
python scripts/run_pretrain.py --config config/pretrain.yaml

# SFT from pretrain checkpoint
python scripts/run_sft.py \
    --config config/sft.yaml \
    --base-model ./outputs/pretrain/best_model

# Evaluation
python scripts/evaluate.py \
    --model ./outputs/sft/best_model \
    --n-samples 10 \
    --k 1 5 10
```

### 5. Export to GGUF and load in Ollama

```bash
# Prerequisites: clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j$(nproc) && pip install -r requirements.txt
cd ..

# Export
bash scripts/export_gguf.sh \
    --model ./outputs/sft/best_model \
    --output ./outputs/gguf \
    --quantize q4_k_m \
    --llama-cpp ./llama.cpp \
    --name cp-coder

# Run
ollama run cp-coder
```

## Project Structure

```
.
├── train.py                    # Main entry point (full pipeline)
├── config/
│   ├── pretrain.yaml           # Pretraining hyperparameters
│   ├── sft.yaml                # SFT hyperparameters
│   ├── ds_zero2.json           # DeepSpeed ZeRO-2 config
│   └── ds_zero3.json           # DeepSpeed ZeRO-3 config (CPU offload)
├── src/
│   ├── data/
│   │   ├── downloaders.py      # HuggingFace dataset downloaders
│   │   ├── preprocessors.py    # Per-dataset normalization
│   │   ├── filters.py          # CP relevance filter
│   │   ├── deduplication.py    # MinHash LSH deduplication
│   │   └── builder.py          # End-to-end data pipeline
│   ├── model/
│   │   └── model_utils.py      # Model/tokenizer loading and saving
│   ├── training/
│   │   ├── pretrain_trainer.py # Continued pretraining (packed CLM)
│   │   ├── sft_trainer.py      # SFT with prompt masking
│   │   └── callbacks.py        # Custom HF Trainer callbacks
│   ├── evaluation/
│   │   └── cp_evaluator.py     # pass@k evaluation with sandboxed execution
│   └── utils/
│       └── helpers.py          # Logging, seeding, formatting
└── scripts/
    ├── prepare_data.py          # Standalone data preparation
    ├── run_pretrain.py          # Standalone pretraining
    ├── run_sft.py               # Standalone SFT
    ├── evaluate.py              # Standalone evaluation
    └── export_gguf.sh           # GGUF conversion + Ollama setup
```

## Training Pipeline Details

### Stage 1: Data Preparation

1. **Download**: Fetches datasets from HuggingFace
2. **Normalize**: Converts each dataset's schema to a unified format:
   `{problem, solutions, examples, difficulty, tags, source}`
3. **Filter**: Removes non-CP content (web dev, ML, DevOps, documentation) using keyword/tag heuristics
4. **Deduplicate**: MinHash LSH (Jaccard ≥ 0.85) removes near-duplicate problems
5. **Split**: Creates train/val/test with cross-deduplication to prevent leakage
6. **Format**: Produces two file types:
   - `pretrain_*.jsonl`: `{"text": "### Problem\n...\n\n### Solution\n..."}` (packed CLM)
   - `sft_*.jsonl`: `{"prompt": "...", "response": "..."}` (instruction tuning)

### Stage 2: Continued Pretraining

- Causal language modeling (next-token prediction) on the CP corpus
- Sequence packing: short documents are concatenated to fill `max_seq_length` windows
- Trains all parameters of the base model
- Learning rate: cosine decay with warmup (lr=2e-5)

### Stage 3: Supervised Fine-Tuning

- Instruction tuning on `(problem → solution)` pairs
- Loss is masked on prompt tokens — only solution tokens contribute to the loss
- Lower learning rate (1e-5) to preserve pretraining knowledge

### Stage 4: Evaluation

- Generates `n` candidate solutions per problem (temperature sampling)
- Executes each solution in a sandboxed subprocess with a 5-second timeout
- Checks stdout against expected output for each example
- Reports pass@1, pass@5, pass@10 using the unbiased Codex estimator

## Configuration

Edit `config/pretrain.yaml` and `config/sft.yaml` to change hyperparameters.
Key parameters:

| Parameter | Pretrain | SFT | Notes |
|-----------|----------|-----|-------|
| `learning_rate` | 2e-5 | 1e-5 | Lower for SFT to avoid forgetting |
| `num_train_epochs` | 3 | 5 | More SFT epochs for instruction following |
| `per_device_train_batch_size` | 2 | 2 | Increase if GPU VRAM allows |
| `gradient_accumulation_steps` | 16 | 16 | Effective batch = bs × accum × n_gpus |
| `max_seq_length` | 4096 | 4096 | Reduce to 2048 for less VRAM |
| `packing` | true | false | Packing not used in SFT (masked loss) |

## Hardware Requirements

| GPU | Batch size | Grad accum | Effective batch | Notes |
|-----|-----------|------------|----------------|-------|
| 1× A100 80GB | 4 | 8 | 32 | Comfortable for 1.3B |
| 1× A100 40GB | 2 | 16 | 32 | Use bf16 + grad ckpt |
| 1× RTX 3090 24GB | 1 | 32 | 32 | Use ZeRO-2 or ZeRO-3 |
| 4× A100 80GB | 4 | 4 | 64 | Ideal setup |

For machines with limited VRAM, use ZeRO-3 with CPU offload:
```bash
python train.py --deepspeed config/ds_zero3.json
```

## Quantization Options (GGUF)

| Quantization | Size (1.3B) | Quality | Speed |
|-------------|-------------|---------|-------|
| `f16` | ~2.6 GB | Lossless | Slow |
| `q8_0` | ~1.4 GB | Near-lossless | Fast |
| `q4_k_m` | ~800 MB | Excellent | Very fast |
| `q5_k_m` | ~950 MB | Very good | Fast |
| `q2_k` | ~500 MB | Moderate | Fastest |
