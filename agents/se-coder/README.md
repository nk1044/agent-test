# SE-Coder: Software Engineering LLM

A 7-10B parameter model specialized exclusively in software engineering, system design, and application development. Target: match or exceed 100-150B general models on SE tasks.

**Focus areas**: frontend, backend, APIs, databases, DevOps, cloud, mobile, microservices, system design, debugging, testing, security, performance, architecture.

---

## Quick Start

```bash
cd /path/to/project

# Minimal run (SFT-only on curated datasets, single GPU)
python agents/se-coder/train.py \
  --stages data sft eval \
  --datasets stack_exchange magicoder_oss magicoder_evol code_feedback evol_codealpaca

# Full run (pretrain + SFT, multi-GPU)
torchrun --nproc_per_node=4 agents/se-coder/train.py \
  --stages data pretrain sft eval \
  --datasets stack_exchange magicoder_oss magicoder_evol code_feedback \
             evol_codealpaca glaive_code text_to_sql self_oss_instruct \
  --deepspeed agents/se-coder/config/ds_zero2.json \
  --flash-attention
```

## Recommended Dataset Sets

### Fast (SFT only, ~2M examples, hours not days)
```bash
--datasets stack_exchange magicoder_oss magicoder_evol code_feedback \
           evol_codealpaca glaive_code text_to_sql self_oss_instruct
```

### Complete (pretrain + SFT, includes The Stack v2, days of training)
```bash
--datasets the_stack_python the_stack_js the_stack_ts the_stack_go \
           the_stack_java the_stack_rust the_stack_sql commitpackft \
           stack_exchange magicoder_oss magicoder_evol code_feedback \
           evol_codealpaca glaive_code text_to_sql self_oss_instruct \
  --max-samples 500000  # limit The Stack subsets to 500K each
```

## Training Commands

### Data preparation only
```bash
python agents/se-coder/train.py --stages data --datasets stack_exchange magicoder_oss magicoder_evol
```

### Pretraining only (from a base model)
```bash
python agents/se-coder/train.py --stages pretrain --base-model Qwen/Qwen2.5-Coder-7B --skip-existing-data
```

### SFT only (skip pretrain, start from base or pretrained checkpoint)
```bash
python agents/se-coder/train.py --stages sft --base-model Qwen/Qwen2.5-Coder-7B --skip-existing-data
```

### Evaluation only
```bash
python agents/se-coder/train.py --stages eval --base-model ./outputs/sft/best_model --skip-existing-data
```

### Resume from checkpoint
```bash
python agents/se-coder/train.py --stages pretrain sft --base-model Qwen/Qwen2.5-Coder-7B \
  --skip-existing-data  # use already-prepared data
```

### DeepSpeed ZeRO-3 (for large model or limited VRAM)
```bash
torchrun --nproc_per_node=4 agents/se-coder/train.py \
  --deepspeed agents/se-coder/config/ds_zero3.json \
  --datasets stack_exchange magicoder_oss magicoder_evol \
  --stages data pretrain sft eval
```

### Custom base model
```bash
python agents/se-coder/train.py --base-model meta-llama/Llama-3.1-8B-Instruct \
  --stages data sft eval
```

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
| `magicoder_oss` | ise-uiuc/Magicoder-OSS-Instruct-75K | 75K | SFT |
| `magicoder_evol` | ise-uiuc/Magicoder-Evol-Instruct-110K | 110K | SFT |
| `code_feedback` | m-a-p/CodeFeedback-Filtered-Instruction | 157K | SFT |
| `evol_codealpaca` | theblackcat102/evol-codealpaca-v1 | 111K | SFT |
| `glaive_code` | glaiveai/glaive-code-assistant-v3 | ~136K | SFT |
| `text_to_sql` | gretelai/synthetic-text-to-sql | ~105K | SFT |
| `ultrachat` | HuggingFaceH4/ultrachat_200k | 200K | SFT |
| `self_oss_instruct` | bigcode/self-oss-instruct-sc2-exec-filter-50k | 50K | SFT |

## Export to GGUF (Ollama)

```bash
bash agents/cp-coder/scripts/export_gguf.sh --model ./outputs/sft/best_model
```
