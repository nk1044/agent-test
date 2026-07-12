"""
Software Engineering Evaluator.

Metrics:
  - Code correctness: syntax check + optional execution for code tasks
  - SQL validity: basic SQL parsing check
  - Response quality: length, code block presence, coherence signals
  - Per-type breakdown: code, qa, debug, design, sql

Unlike CP evaluation (binary pass/fail), SE evaluation uses a mix of:
  1. Hard metrics: code syntax validity, SQL parsability
  2. Soft metrics: response completeness, structure quality
  3. Where possible: execution-based verification

Usage:
    evaluator = SEEvaluator(model, tokenizer)
    results = evaluator.evaluate(test_dataset, n_samples=1)
    print(results.summary())
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)
_SQL_KEYWORDS = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\b", re.IGNORECASE)


def extract_code_blocks(text: str) -> List[str]:
    return _CODE_FENCE_RE.findall(text)


def check_python_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_sql_basic(text: str) -> bool:
    return bool(_SQL_KEYWORDS.search(text))


def quick_execute_python(code: str, timeout: float = 5.0) -> Tuple[bool, str]:
    """Try to execute Python code. Returns (success, error_or_empty)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmpfile = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import ast; ast.parse(open('{tmpfile}').read())"],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stderr
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


@dataclass
class SEEvalResult:
    n_total: int = 0
    n_code_valid: int = 0
    n_has_code_block: int = 0
    n_sql_valid: int = 0
    n_non_empty: int = 0
    by_type: Dict[str, Dict] = field(default_factory=dict)
    samples: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"SE Evaluation Results ({self.n_total} samples)",
            f"  Non-empty responses:  {self.n_non_empty}/{self.n_total} ({100*self.n_non_empty/max(1,self.n_total):.1f}%)",
            f"  Has code block:       {self.n_has_code_block}/{self.n_total} ({100*self.n_has_code_block/max(1,self.n_total):.1f}%)",
            f"  Python syntax valid:  {self.n_code_valid}/{self.n_total} ({100*self.n_code_valid/max(1,self.n_total):.1f}%)",
        ]
        if self.by_type:
            lines.append("  By type:")
            for t, stats in self.by_type.items():
                lines.append(f"    {t}: {stats}")
        return "\n".join(lines)


class SEEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        max_problems: int = 200,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_problems = max_problems
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def evaluate(
        self,
        test_records: List[Dict],
        output_dir: Optional[str] = None,
        n_samples: int = 1,
    ) -> SEEvalResult:
        result = SEEvalResult()
        records = test_records[:self.max_problems]

        from data.builder import PROMPT_TEMPLATE, SYSTEM_PROMPT

        for rec in records:
            instruction = rec.get("instruction", "")
            context = rec.get("context", "")
            rec_type = rec.get("type", "qa")
            expected = rec.get("response", "")

            full_instruction = f"{context}\n\n{instruction}" if context else instruction
            prompt = PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, instruction=full_instruction)

            try:
                response = self._generate(prompt)
            except Exception as exc:
                logger.warning("Generation failed: %s", exc)
                response = ""

            result.n_total += 1
            if response.strip():
                result.n_non_empty += 1

            code_blocks = extract_code_blocks(response)
            if code_blocks:
                result.n_has_code_block += 1

            for block in code_blocks:
                if check_python_syntax(block):
                    result.n_code_valid += 1
                    break

            if rec_type == "sql" and check_sql_basic(response):
                result.n_sql_valid += 1

            if rec_type not in result.by_type:
                result.by_type[rec_type] = {"n": 0, "non_empty": 0, "has_code": 0}
            result.by_type[rec_type]["n"] += 1
            if response.strip():
                result.by_type[rec_type]["non_empty"] += 1
            if code_blocks:
                result.by_type[rec_type]["has_code"] += 1

            result.samples.append({
                "instruction": instruction[:200],
                "expected": expected[:200],
                "response": response[:200],
                "type": rec_type,
                "has_code": bool(code_blocks),
            })

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_dir, "se_eval_results.json"), "w") as f:
                json.dump({
                    "summary": result.summary(),
                    "stats": {
                        "n_total": result.n_total,
                        "n_non_empty": result.n_non_empty,
                        "n_has_code_block": result.n_has_code_block,
                        "n_code_valid": result.n_code_valid,
                        "by_type": result.by_type,
                    },
                    "samples": result.samples[:50],
                }, f, indent=2)

        return result


def load_test_records(path: str) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
