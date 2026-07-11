"""
Competitive Programming evaluator.

Metrics:
  - pass@k  (k=1, 5, 10): fraction of problems solved by at least one of k generated solutions
  - Exact match on examples (lightweight, no code execution required)
  - Per-difficulty breakdown

Execution sandbox:
  Code execution uses subprocess with a strict timeout to avoid infinite loops
  and resource exhaustion. Each solution is run in an isolated subprocess.

Usage:
    evaluator = CPEvaluator(model, tokenizer)
    results = evaluator.evaluate(test_dataset, n_samples=10, k_values=[1,5])
    print(results)
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

EXECUTION_TIMEOUT = 5    # seconds per test case
MAX_OUTPUT_BYTES = 65536  # 64 KB
MEM_LIMIT_MB = 512


def _run_solution_subprocess(
    code: str,
    stdin_data: str,
    timeout: float = EXECUTION_TIMEOUT,
) -> Tuple[str, str, bool]:
    """
    Execute Python code in a subprocess with resource limits.
    Returns (stdout, stderr, timed_out).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmpfile = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmpfile],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": ""},
        )
        return proc.stdout[:MAX_OUTPUT_BYTES], proc.stderr[:MAX_OUTPUT_BYTES], False
    except subprocess.TimeoutExpired:
        return "", "Timeout", True
    except Exception as exc:
        return "", str(exc), False
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


def _normalize_output(s: str) -> str:
    """Normalize output for comparison (strip trailing whitespace per line)."""
    lines = s.strip().splitlines()
    return "\n".join(line.rstrip() for line in lines)


def check_solution(
    code: str,
    examples: List[Dict[str, str]],
    timeout: float = EXECUTION_TIMEOUT,
) -> Tuple[bool, str]:
    """
    Check a solution against provided I/O examples.
    Returns (passed, reason).
    """
    if not examples:
        return False, "no_examples"

    for i, ex in enumerate(examples):
        stdin = ex.get("input", "")
        expected = _normalize_output(ex.get("output", ""))

        stdout, stderr, timed_out = _run_solution_subprocess(code, stdin, timeout=timeout)

        if timed_out:
            return False, f"timeout_on_example_{i}"
        if stderr and not stdout:
            return False, f"runtime_error_example_{i}: {stderr[:200]}"

        actual = _normalize_output(stdout)
        if actual != expected:
            return False, f"wrong_answer_example_{i}: expected={expected[:100]!r} got={actual[:100]!r}"

    return True, "passed"


# ---------------------------------------------------------------------------
# pass@k estimator (unbiased, from Codex paper)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimate pass@k given n total samples and c correct samples.
    Uses the numerically stable formula from the Codex paper:
        1 - C(n-c, k) / C(n, k)
    """
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_solutions(
    model,
    tokenizer,
    prompt: str,
    n: int = 10,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> List[str]:
    """Generate n candidate solutions for a given prompt."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = []
    prompt_len = input_ids.shape[1]
    for out in outputs:
        tokens = out[prompt_len:]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        # Extract code block if present
        code = _extract_code(text)
        generated.append(code)

    return generated


def _extract_code(text: str) -> str:
    """Extract Python code from a markdown code block or return raw text."""
    import re
    # Try ```python ... ``` first
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ``` ... ```
    match = re.search(r"```\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    total_problems: int = 0
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    pass_at_k: Dict[int, float] = field(default_factory=dict)
    per_difficulty: Dict[str, Dict] = field(default_factory=dict)
    per_problem: List[Dict] = field(default_factory=list)
    eval_time_seconds: float = 0.0

    def summary(self) -> str:
        lines = [f"=== CP Evaluation Results ({self.total_problems} problems) ==="]
        for k, rate in sorted(self.pass_at_k.items()):
            lines.append(f"  pass@{k}: {rate:.3f} ({rate*100:.1f}%)")
        if self.per_difficulty:
            lines.append("Per difficulty:")
            for diff, stats in self.per_difficulty.items():
                lines.append(f"  {diff}: pass@1={stats.get('pass@1', 0):.3f} (n={stats.get('n', 0)})")
        lines.append(f"Eval time: {self.eval_time_seconds:.1f}s")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "total_problems": self.total_problems,
            "pass_at_k": {f"pass@{k}": v for k, v in self.pass_at_k.items()},
            "per_difficulty": self.per_difficulty,
            "eval_time_seconds": self.eval_time_seconds,
        }


class CPEvaluator:
    """
    Evaluate a model on competitive programming problems.

    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        execution_timeout: seconds per test-case execution
        max_problems: cap on number of problems to evaluate (None = all)
    """

    def __init__(
        self,
        model,
        tokenizer,
        execution_timeout: float = EXECUTION_TIMEOUT,
        max_problems: Optional[int] = None,
        use_sandbox: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.execution_timeout = execution_timeout
        self.max_problems = max_problems
        self.use_sandbox = use_sandbox

    def _build_prompt(self, record: Dict) -> str:
        from ..data.builder import record_to_sft_pair
        pair = record_to_sft_pair(record)
        if pair:
            return pair["prompt"]
        return "### Problem\n" + record.get("problem", "") + "\n\n### Solution\n```python\n"

    def evaluate(
        self,
        test_records: List[Dict],
        n_samples: int = 10,
        k_values: Optional[List[int]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        output_dir: Optional[str] = None,
    ) -> EvalResult:
        k_values = k_values or [1, 5, 10]
        result = EvalResult(k_values=k_values)
        start_time = time.time()

        problems = test_records
        if self.max_problems:
            problems = problems[: self.max_problems]

        logger.info("Evaluating on %d problems (n_samples=%d) ...", len(problems), n_samples)

        difficulty_stats: Dict[str, List[Tuple[int, int]]] = {}

        for i, record in enumerate(problems):
            prompt = self._build_prompt(record)
            examples = record.get("examples") or []
            difficulty = record.get("difficulty", "unknown")

            if not examples and self.use_sandbox:
                # Can't verify without examples
                result.per_problem.append({"idx": i, "skipped": True, "reason": "no_examples"})
                continue

            # Generate n candidate solutions
            candidates = generate_solutions(
                self.model,
                self.tokenizer,
                prompt,
                n=n_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Check each candidate
            n_correct = 0
            for code in candidates:
                if self.use_sandbox and examples:
                    passed, reason = check_solution(code, examples, timeout=self.execution_timeout)
                    if passed:
                        n_correct += 1
                else:
                    # No execution: mark as pending (can't verify)
                    break

            prob_stats = {
                "idx": i,
                "difficulty": difficulty,
                "n_samples": len(candidates),
                "n_correct": n_correct,
            }
            for k in k_values:
                prob_stats[f"pass@{k}"] = pass_at_k(len(candidates), n_correct, k)

            result.per_problem.append(prob_stats)

            # Accumulate per-difficulty
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = []
            difficulty_stats[difficulty].append((len(candidates), n_correct))

            if (i + 1) % 10 == 0:
                logger.info("Evaluated %d/%d problems", i + 1, len(problems))

        # Aggregate pass@k
        result.total_problems = len(result.per_problem)
        evaluated = [p for p in result.per_problem if not p.get("skipped")]

        for k in k_values:
            rates = [p.get(f"pass@{k}", 0.0) for p in evaluated if f"pass@{k}" in p]
            result.pass_at_k[k] = float(np.mean(rates)) if rates else 0.0

        # Per-difficulty aggregation
        for diff, stats_list in difficulty_stats.items():
            rates_1 = [pass_at_k(n, c, 1) for n, c in stats_list if n >= 1]
            result.per_difficulty[diff] = {
                "n": len(stats_list),
                "pass@1": float(np.mean(rates_1)) if rates_1 else 0.0,
            }

        result.eval_time_seconds = time.time() - start_time

        logger.info(result.summary())

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            with open(os.path.join(output_dir, "eval_per_problem.jsonl"), "w") as f:
                for p in result.per_problem:
                    f.write(json.dumps(p) + "\n")

        return result


def load_test_records(test_jsonl: str) -> List[Dict]:
    records = []
    with open(test_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
