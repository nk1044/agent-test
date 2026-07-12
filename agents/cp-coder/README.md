# CP-Coder: Competitive Programming LLM

A 7-10B parameter model specialized exclusively in competitive programming — optimized to solve hard algorithmic problems comparable to 100B+ general models.

**Focus areas**: algorithms, data structures, dynamic programming, graph theory, number theory, combinatorics, string processing, geometry, CP contest problems (Codeforces, AtCoder, ICPC, IOI, LeetCode Hard).

---

## Quick Start

```bash
cd /path/to/project

# Full pipeline (single GPU)
python agents/cp-coder/train.py \
  --stages data pretrain sft rl eval \
  --datasets taco apps code_contests codeforces opencode_reasoning kodcode

# Multi-GPU with DeepSpeed
torchrun --nproc_per_node=4 agents/cp-coder/train.py \
  --stages data pretrain sft rl eval \
  --datasets taco apps code_contests codeforces codeforces_cots opencode_reasoning \
             opencode_reasoning2 kodcode bigcodebench aizu competitive_programming_v4 \
  --deepspeed agents/cp-coder/config/ds_zero2.json \
  --flash-attention
```

## Training Commands

### Data preparation only
```bash
python agents/cp-coder/train.py --stages data \
  --datasets taco apps code_contests codeforces opencode_reasoning kodcode
```

### Pretraining only
```bash
python agents/cp-coder/train.py --stages pretrain \
  --base-model Qwen/Qwen2.5-Coder-7B --skip-existing-data
```

### SFT only
```bash
python agents/cp-coder/train.py --stages sft \
  --base-model Qwen/Qwen2.5-Coder-7B --skip-existing-data
```

### SFT with execution-verified data
```bash
python agents/cp-coder/train.py --stages data sft \
  --datasets kodcode bigcodebench --verify-sft
```

### RL (GRPO) only — single round
```bash
python agents/cp-coder/train.py --stages rl \
  --base-model ./outputs/sft/best_model --skip-existing-data
```

### RL — multi-round curriculum (recommended)
```bash
python agents/cp-coder/train.py --stages rl \
  --base-model ./outputs/sft/best_model \
  --rl-rounds 3 --skip-existing-data
# Round 1: all difficulties → Round 2: medium+hard → Round 3: hard only
```

### Rejection Sampling Fine-Tuning (RFT)
```bash
python agents/cp-coder/train.py --stages rft \
  --base-model ./outputs/rl/round3/best_model --skip-existing-data
```

### Full pipeline with 3-round curriculum RL
```bash
torchrun --nproc_per_node=4 agents/cp-coder/train.py \
  --stages data pretrain sft rl rft eval \
  --datasets taco apps code_contests codeforces codeforces_cots opencode_reasoning \
             opencode_reasoning2 kodcode bigcodebench aizu competitive_programming_v4 \
  --rl-rounds 3 \
  --deepspeed agents/cp-coder/config/ds_zero2.json \
  --flash-attention
```

### Evaluation only
```bash
python agents/cp-coder/train.py --stages eval \
  --base-model ./outputs/sft/best_model --skip-existing-data
```

### DeepSpeed ZeRO-3 (limited VRAM)
```bash
torchrun --nproc_per_node=4 agents/cp-coder/train.py \
  --deepspeed agents/cp-coder/config/ds_zero3.json \
  --datasets taco apps code_contests --stages data pretrain sft
```

## Script Commands

### Prepare data only
```bash
python agents/cp-coder/scripts/prepare_data.py \
  --datasets taco apps code_contests opencode_reasoning kodcode \
  --output-dir ./data/processed \
  --cache-dir ./data/raw
```

### Run pretraining directly
```bash
python agents/cp-coder/scripts/run_pretrain.py \
  --config agents/cp-coder/config/pretrain.yaml \
  --base-model Qwen/Qwen2.5-Coder-7B
```

### Run SFT directly
```bash
python agents/cp-coder/scripts/run_sft.py \
  --config agents/cp-coder/config/sft.yaml \
  --base-model ./outputs/pretrain/best_model
```

### Evaluate checkpoint
```bash
python agents/cp-coder/scripts/evaluate.py \
  --model ./outputs/sft/best_model \
  --test-file ./data/processed/test.jsonl \
  --max-problems 500
```

## Available Datasets

| Key | Source | Size | Best for |
|-----|--------|------|----------|
| `taco` | BAAI/TACO | ~26K | Pretraining + SFT |
| `apps` | codeparrot/apps | ~10K | Pretraining + SFT |
| `code_contests` | deepmind/code_contests | ~13K | Pretraining + SFT |
| `codeforces` | open-r1/codeforces | ~60K | SFT |
| `codeforces_cots` | open-r1/codeforces-cots | ~60K | SFT (with CoT) |
| `leetcode` | greengerong/leetcode | ~2.4K | SFT (Hard/Medium) |
| `opencode_reasoning` | nvidia/OpenCodeReasoning | ~736K | SFT + RL |
| `opencode_reasoning2` | nvidia/OpenCodeReasoning-2 | ~1.5M | SFT + RL |
| `kodcode` | KodCode/KodCode | ~447K | SFT + RL (unit tests) |
| `bigcodebench` | bigcode/bigcodebench | ~1.1K | RL eval |
| `aizu` | BAAI/AIZU-OJ | ~24K | SFT + RL |
| `competitive_programming_v4` | deepmind/competitive_programming | ~13K | SFT + RL |
| `magicoder_evol` | ise-uiuc/Magicoder-Evol-Instruct-110K | 110K | SFT |
| `code_feedback` | m-a-p/CodeFeedback-Filtered-Instruction | 157K | SFT |

## Export to GGUF (Ollama)

```bash
bash agents/cp-coder/scripts/export_gguf.sh --model ./outputs/sft/best_model
```
