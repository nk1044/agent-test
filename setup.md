# Setup Guide

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 40–80 GB GPU VRAM per node (A100 40GB minimum; A100 80GB recommended for 7B full-parameter)
- 200+ GB disk space for datasets and checkpoints
- Linux (recommended) or macOS (CPU-only / testing)

---

## 1. Clone and install

#### Clone repo
```bash
git clone https://github.com/nk1044/agent-test.git

cd agent-test
```
#### Setup environment:-
```bash
uv venv

source .venv/bin/activate 

pip install -r requirements.txt
```

### Optional: Flash Attention 2 (strongly recommended for A100/H100)

```bash
pip install flash-attn --no-build-isolation
```

### Optional: DeepSpeed (required for multi-GPU / ZeRO training)

```bash
pip install deepspeed
ds_report  # verify installation
```

---

## 2. HuggingFace access

Some datasets and models require HuggingFace authentication:

```bash
pip install huggingface_hub
huggingface-cli login
# paste your token from https://huggingface.co/settings/tokens
```

Models that require access requests:
- `meta-llama/Llama-3.1-8B` — request access at huggingface.co/meta-llama
- All other recommended models (Qwen, DeepSeek) are open-weight

---

## 3. Weights & Biases (optional but recommended)

```bash
pip install wandb
wandb login
# paste your API key from https://wandb.ai/settings
```

Pass `--no-wandb` to any training command to skip W&B logging.

---

## 4. Project structure

```
project/
├── model/          # Model loading/saving utilities (shared by all agents)
├── training/       # Training logic: pretrain, SFT, RL, RFT (shared by all agents)
├── utils/          # Logging, seeding, YAML helpers (shared by all agents)
├── agents/
│   ├── cp-coder/   # Competitive programming specialist
│   │   ├── train.py        — main entry point
│   │   ├── README.md       — command reference
│   │   ├── config/         — hyperparameter YAMLs + DeepSpeed configs
│   │   ├── data/           — CP dataset downloaders, preprocessors, filters
│   │   ├── evaluation/     — pass@k evaluation with sandboxed execution
│   │   └── scripts/        — standalone stage scripts
│   └── se-coder/   # Software engineering specialist
│       ├── train.py        — main entry point
│       ├── README.md       — command reference
│       ├── config/         — hyperparameter YAMLs + DeepSpeed configs
│       ├── data/           — SE dataset downloaders, preprocessors, filters
│       └── evaluation/     — code syntax + SQL validity checks
├── requirements.txt
├── setup.md        — this file
└── README.md       — project overview
```

Each agent is self-contained. Root `model/`, `training/`, and `utils/` are automatically resolved at runtime via `sys.path` — no installation step needed.

---

## 5. Choose your agent

| Goal | Agent | Entry point |
|------|-------|-------------|
| Train a competitive programming model | cp-coder | `agents/cp-coder/train.py` |
| Train a software engineering model | se-coder | `agents/se-coder/train.py` |

---

## 6. Data preparation (run this first, separately)

Data preparation downloads raw datasets from HuggingFace, normalizes schemas, filters for domain relevance, deduplicates, and writes processed JSONL files to `data/processed/`. This is a CPU-only step — do it before starting GPU training.

**What it produces:**
- `data/processed/pretrain_train.jsonl` — packed text for continued pretraining
- `data/processed/pretrain_val.jsonl`
- `data/processed/sft_train.jsonl` — (prompt, response) pairs for SFT
- `data/processed/sft_val.jsonl`
- `data/processed/test.jsonl` — held-out evaluation set
- `data/processed/dataset_meta.json` — statistics (counts, dedup threshold, sources)

### CP-Coder — fast dataset (recommended for first run)

```bash
cd agents/cp-coder
python train.py --stages data \
  --datasets taco apps code_contests codeforces opencode_reasoning kodcode
```

### CP-Coder — full dataset (production run)

```bash
cd agents/cp-coder
python train.py --stages data \
  --datasets taco apps code_contests codeforces codeforces_cots \
             opencode_reasoning opencode_reasoning2 kodcode bigcodebench \
             aizu competitive_programming_v4 magicoder_evol code_feedback
```

### SE-Coder — fast dataset (recommended for first run)

```bash
cd agents/se-coder
python train.py --stages data \
  --datasets stack_exchange magicoder_oss magicoder_evol \
             code_feedback evol_codealpaca self_oss_instruct
```

### SE-Coder — full dataset (includes The Stack v2, very large)

```bash
cd agents/se-coder
python train.py --stages data \
  --datasets the_stack_python the_stack_js the_stack_ts the_stack_go \
             the_stack_java the_stack_rust the_stack_sql commitpackft \
             stack_exchange magicoder_oss magicoder_evol code_feedback \
             evol_codealpaca glaive_code text_to_sql self_oss_instruct \
  --max-samples 500000
# --max-samples caps each Stack subset to 500K examples to avoid downloading ~100GB+
```

> Data preparation can take 30 minutes to several hours depending on dataset size and network speed. Run it once; subsequent training runs reuse the processed files with `--skip-existing-data`.

---

## 7. Training (after data preparation)

Once `data/processed/` is populated, run training with `--skip-existing-data` to skip re-downloading.

### Single GPU — SFT only (quick test)

```bash
# CP-Coder
cd agents/cp-coder
python train.py \
  --base-model Qwen/Qwen2.5-Coder-7B \
  --stages sft eval \
  --skip-existing-data \
  --max-seq-length 4096 \
  --epochs-sft 1

# SE-Coder
cd agents/se-coder
python train.py \
  --base-model Qwen/Qwen2.5-Coder-7B \
  --stages sft eval \
  --skip-existing-data \
  --max-seq-length 4096 \
  --epochs-sft 1
```

### Single GPU — full pipeline (pretrain → SFT → eval)

```bash
cd agents/cp-coder
python train.py \
  --base-model Qwen/Qwen2.5-Coder-7B \
  --stages pretrain sft eval \
  --skip-existing-data
```

### CP-Coder — full pipeline with RL (recommended)

```bash
cd agents/cp-coder
python train.py \
  --base-model Qwen/Qwen2.5-Coder-7B \
  --stages pretrain sft rl eval \
  --rl-rounds 3 \
  --skip-existing-data
```

---

## 8. Multi-GPU training

Use `torchrun` for distributed training. Each agent README has the full command set.

```bash
# Step 1: prepare data (CPU, single process)
cd agents/cp-coder
python train.py --stages data \
  --datasets taco apps code_contests codeforces opencode_reasoning kodcode

# Step 2: train on multiple GPUs
torchrun --nproc_per_node=4 train.py \
  --base-model Qwen/Qwen2.5-Coder-7B \
  --stages pretrain sft rl eval \
  --rl-rounds 3 \
  --skip-existing-data \
  --deepspeed config/ds_zero2.json \
  --flash-attention
```

---

## 8. GGUF export (Ollama)

After training completes, export to GGUF for local inference with Ollama:

```bash
bash agents/cp-coder/scripts/export_gguf.sh --model ./outputs/sft/best_model
```

---

## 9. Hardware requirements

| Setup | VRAM | Throughput |
|-------|------|-----------|
| 1× A100 80GB | 80 GB | ~2,000 tokens/sec |
| 2× A100 80GB | 160 GB | ~4,000 tokens/sec |
| 4× A100 80GB | 320 GB | ~8,000 tokens/sec |
| 1× A100 40GB | 40 GB | ~800 tokens/sec (use ZeRO-3) |
| 1× RTX 3090 (24GB) | 24 GB | SFT only with batch_size=1, grad_accum=32 |

For 7B full-parameter training with bf16: minimum ~28 GB VRAM. Use `--deepspeed ds_zero3.json` to shard optimizer states across GPUs when VRAM is limited.

---

## 10. Troubleshooting

**CUDA out of memory**
- Reduce `--batch-size` to 1 and increase `--grad-accum` to 32+
- Switch to ZeRO-3: `--deepspeed agents/cp-coder/config/ds_zero3.json`
- Reduce `--max-seq-length` to 4096

**Slow tokenization / data download**
- Increase `--dataloader-num-workers` (default 4)
- Run data prep separately first (see step 6), then use `--skip-existing-data` for all training runs
- For The Stack v2, always pass `--max-samples 500000` to avoid multi-hundred-GB downloads

**W&B not logging**
- Run `wandb login` or pass `--no-wandb` to disable

**HuggingFace 401 error**
- Run `huggingface-cli login` with a token that has read access to gated models
