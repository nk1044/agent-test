# LLM Training Agents

A production-ready, full-parameter LLM training framework with two specialized agents: one for competitive programming and one for software engineering. Both target performance comparable to 100-150B+ general-purpose models on their respective domains.

## What this is

This framework fine-tunes 7-10B parameter base models (primarily Qwen2.5-Coder-7B) using a multi-stage training pipeline: continued pretraining on domain-specific corpora, supervised fine-tuning on high-quality instruction pairs, and — for the competitive programming agent — reinforcement learning with verifiable rewards (RLVR) and rejection sampling fine-tuning (RFT).

Training is full-parameter only. No LoRA, no adapters, no quantization during training. The resulting models can be exported to GGUF for Ollama deployment.

---

## Agents

### CP-Coder — Competitive Programming

Specializes exclusively in algorithmic problem solving: dynamic programming, graph algorithms, number theory, combinatorics, geometry, string processing, and contest-level problem decomposition (Codeforces, AtCoder, ICPC, IOI).

Training pipeline: continued pretraining → SFT → multi-round GRPO RLVR → rejection sampling FT.

Datasets include TACO, APPS, DeepMind CodeContests, Codeforces problems with chain-of-thought traces, OpenCodeReasoning (736K problems), OpenCodeReasoning-2 (1.5M), KodCode (447K execution-verified), BigCodeBench, and AIZU-OJ.

See [agents/cp-coder/README.md](agents/cp-coder/README.md) for training commands.

### SE-Coder — Software Engineering

Specializes exclusively in software engineering and application development: frontend, backend, APIs, databases, DevOps, cloud infrastructure, mobile, microservices, system design, debugging, testing, security, performance optimization, and code architecture.

Explicitly excludes competitive programming, puzzle-style reasoning, and data science/ML research.

Training pipeline: continued pretraining → SFT.

Datasets include The Stack v2 (Python, JavaScript, TypeScript, Go, Java, Rust, SQL), CommitPackFT (702K real commits), StackExchange Q&A, Magicoder OSS/Evol, CodeFeedback, Glaive Code Assistant, synthetic text-to-SQL, UltraChat (technical), and self-OSS-instruct execution-filtered pairs.

See [agents/se-coder/README.md](agents/se-coder/README.md) for training commands.

---

## Architecture

```
project/
├── model/          # Model loading and saving (AutoModelForCausalLM, tokenizer setup)
├── training/       # Training stages shared across agents
│   ├── pretrain_trainer.py   — continued pretraining with sequence packing
│   ├── sft_trainer.py        — SFT with prompt masking
│   ├── rl_trainer.py         — GRPO-based RLVR with code execution rewards
│   ├── rft_trainer.py        — rejection sampling fine-tuning
│   └── callbacks.py          — progress, NaN detection, checkpoint metadata
├── utils/          # Logging, seeding, YAML I/O
├── agents/
│   ├── cp-coder/   — competitive programming agent
│   └── se-coder/   — software engineering agent
├── requirements.txt
├── setup.md        — installation and setup guide
└── README.md       — this file
```

Root `model/`, `training/`, and `utils/` packages are resolved at runtime via `sys.path` — no package installation required.

---

## Design principles

**Full-parameter training only.** Every weight is updated during training. LoRA and adapters are explicitly excluded. This maximizes the model's capacity to internalize domain knowledge rather than patching a frozen base.

**Domain exclusivity.** Each agent is trained on data from exactly one domain with explicit filtering to reject off-topic content. A competitive programmer never sees web dev tutorials; a software engineer never sees Codeforces contest problems. Domain purity improves specialization.

**Execution-verified training signal.** For the CP agent, training data quality is enforced through code execution: solutions that do not pass provided test cases are excluded from SFT and used as negative examples in RLVR. Rewards are derived from actual test case pass rates, not surface-level heuristics.

**Structured reasoning.** The CP agent is trained with a `<think>...</think>` + `<tests>...</tests>` output format, encouraging step-by-step problem decomposition and self-verification before committing to a final solution.

---

## Getting started

See [setup.md](setup.md) for installation instructions, hardware requirements, and first-run commands.
