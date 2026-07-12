from setuptools import find_packages, setup

setup(
    name="agent-trainer",
    version="0.2.0",
    description="Full-parameter LLM training pipeline for CP-Coder and SE-Coder agents",
    packages=find_packages(where=".", include=["model*", "training*", "utils*", "agents*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "datasets>=2.20.0",
        "tokenizers>=0.19.0",
        "trl>=0.12.0",
        "evaluate>=0.4.0",
        "datasketch>=1.6.4",
        "wandb>=0.17.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "rich>=13.7.0",
        "psutil>=6.0.0",
        "sentencepiece>=0.2.0",
    ],
    # No console_scripts: agent dirs use hyphens (cp-coder, se-coder) which are
    # invalid in Python module paths. Run agents directly:
    #   python agents/cp-coder/train.py ...
    #   python agents/se-coder/train.py ...
)
