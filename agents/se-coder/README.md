# SE-Coder: Software Engineering LLM

A 7-10B parameter model specialized exclusively in software engineering and application development. Target: match or exceed 100-150B general models on SE tasks.

**Focus areas**: frontend, backend, APIs, databases, DevOps, cloud, mobile, microservices, system design, debugging, testing, security, performance, architecture.

**Training pipeline**: continued pretraining → SFT → GRPO RLVR → rejection sampling FT (identical philosophy to CP-Coder, adapted for SE verification).

**SE reward signal**: Python code runs without error → 1.0 | SQL executes in SQLite → 1.0 | partial credit for syntax-valid code | reasoning + self-test bonuses.

---

## Quick Start

```bash
cd /path/to/project

# Step 1: data (CPU, run once)
python agents/se-coder/train.py --stages data \
  --datasets stack_exchange magicoder_oss magicoder_evol code_feedback evol_codealpaca self_oss_instruct

# Step 2: full training (GPU)
python agents/se-coder/train.py \
  --stages pretrain sft rl rft eval \
  --skip-existing-data \
  --rl-rounds 3
```

---

## Training Commands

### Data preparation only
```bash
python agents/se-coder/train.py --stages data \
  --datasets stack_exchange magicoder_oss magicoder_evol code_feedback evol_codealpaca self_oss_instruct
```

### Data — full dataset (includes The Stack v2)
```bash
python agents/se-coder/train.py --stages data \
  --datasets the_stack_python the_stack_js the_stack_ts the_stack_go \
             the_stack_java the_stack_rust the_stack_sql commitpackft \
             stack_exchange magicoder_oss magicoder_evol code_feedback \
             evol_codealpaca glaive_code text_to_sql self_oss_instruct \
  --max-samples 500000
```

### SFT only (fastest, good baseline)
```bash
python agents/se-coder/train.py --stages sft eval \
  --base-model Qwen/Qwen2.5-Coder-7B --skip-existing-data
```

### Pretrain + SFT
```bash
python agents/se-coder/train.py --stages pretrain sft eval \
  --base-model Qwen/Qwen2.5-Coder-7B --skip-existing-data
```

### RL only (from SFT checkpoint)
```bash
python agents/se-coder/train.py --stages rl \
  --base-model ./outputs/sft/best_model --skip-existing-data
```

### RL — 3-round curriculum (recommended)
```bash
python agents/se-coder/train.py --stages rl \
  --base-model ./outputs/sft/best_model \
  --rl-rounds 3 --skip-existing-data
# Round 1: code + debug + sql  (all verifiable types)
# Round 2: code + debug        (drop SQL, focus on implementation)
# Round 3: code only           (hardest: full implementation tasks)
```

### RFT only (from RL checkpoint)
```bash
python agents/se-coder/train.py --stages rft \
  --base-model ./outputs/rl/round3/best_model --skip-existing-data
```

### Full pipeline — single GPU
```bash
python agents/se-coder/train.py \
  --base-model Qwen/Qwen2.5-Coder-7B \
  --stages pretrain sft rl rft eval \
  --rl-rounds 3 \
  --skip-existing-data
```

### Full pipeline — multi-GPU with DeepSpeed
```bash
# Step 1: data (CPU)
python agents/se-coder/train.py --stages data \
  --datasets stack_exchange magicoder_oss magicoder_evol code_feedback \
             evol_codealpaca glaive_code text_to_sql self_oss_instruct

# Step 2: training (GPU)
cd agents/se-coder
torchrun --nproc_per_node=4 train.py \
  --base-model Qwen/Qwen2.5-Coder-7B \
  --stages pretrain sft rl rft eval \
  --rl-rounds 3 \
  --skip-existing-data \
  --deepspeed config/ds_zero2.json \
  --flash-attention
```

### DeepSpeed ZeRO-3 (limited VRAM)
```bash
torchrun --nproc_per_node=4 agents/se-coder/train.py \
  --deepspeed agents/se-coder/config/ds_zero3.json \
  --stages pretrain sft rl eval \
  --skip-existing-data
```

### Evaluation only
```bash
python agents/se-coder/train.py --stages eval \
  --base-model ./outputs/rft/sft_on_verified/best_model --skip-existing-data
```

### Custom base model
```bash
python agents/se-coder/train.py --base-model meta-llama/Llama-3.1-8B-Instruct \
  --stages sft rl rft eval --skip-existing-data
```

---

## Available Datasets

| Key | Source | Size | Best for |
|-----|--------|------|----------|
| `the_stack_python` | bigcode/the-stack-v2-dedup | ~100GB | Pretraining |
| `the_stack_js` | bigcode/the-stack-v2-dedup | ~50GB | Pretraining |
| `the_stack_ts` | bigcode/the-stack-v2-dedup | ~20GB | Pretraining |
| `the_stack_go` | bigcode/the-stack-v2-dedup | ~10GB | Pretraining |
| `the_stack_java` | bigcode/the-stack-v2-dedup | ~30GB | Pretraining |
| `the_stack_rust` | bigcode/the-stack-v2-dedup | ~5GB | Pretraining |
| `commitpackft` | bigcode/commitpackft | ~702K commits | Pretraining + SFT |
| `stack_exchange` | ArmelR/stack-exchange-instruction | ~1M Q&A | SFT |
| `magicoder_oss` | ise-uiuc/Magicoder-OSS-Instruct-75K | 75K | SFT + RL |
| `magicoder_evol` | ise-uiuc/Magicoder-Evol-Instruct-110K | 110K | SFT + RL |
| `code_feedback` | m-a-p/CodeFeedback-Filtered-Instruction | 157K | SFT + RL |
| `evol_codealpaca` | theblackcat102/evol-codealpaca-v1 | 111K | SFT + RL |
| `glaive_code` | glaiveai/glaive-code-assistant-v3 | ~136K | SFT + RL |
| `text_to_sql` | gretelai/synthetic-text-to-sql | ~105K | SFT + RL (SQL) |
| `self_oss_instruct` | bigcode/self-oss-instruct-sc2-exec-filter-50k | 50K | SFT + RL |
| `ultrachat` | HuggingFaceH4/ultrachat_200k | 200K | SFT |

---

## Export to GGUF (Ollama)

```bash
bash agents/cp-coder/scripts/export_gguf.sh --model ./outputs/rft/sft_on_verified/best_model
```
