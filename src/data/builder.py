"""
DatasetBuilder: end-to-end pipeline from raw downloads to tokenized HuggingFace datasets.

Pipeline:
  download → preprocess → filter → deduplicate → split → format → save
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset

from .deduplication import cross_deduplicate, deduplicate_dataset
from .downloaders import download_all_datasets
from .filters import CPFilter
from .preprocessors import preprocess_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text formatters (problem + solutions → plain text or instruction pairs)
# ---------------------------------------------------------------------------

def _format_examples(examples: List[Dict]) -> str:
    if not examples:
        return ""
    lines = []
    for i, ex in enumerate(examples[:3], 1):
        inp = ex.get("input", "").strip()
        out = ex.get("output", "").strip()
        if inp or out:
            lines.append(f"Example {i}:")
            if inp:
                lines.append(f"  Input:\n    {inp}")
            if out:
                lines.append(f"  Output:\n    {out}")
    return "\n".join(lines)


def record_to_pretrain_text(record: Dict) -> str:
    """
    Concatenate problem + solutions into a single pretraining document.
    Used for the continued-pretraining stage (causal LM objective).
    """
    parts: List[str] = []

    problem = record.get("problem", "").strip()
    if problem:
        parts.append("### Problem\n" + problem)

    examples_str = _format_examples(record.get("examples") or [])
    if examples_str:
        parts.append("### Examples\n" + examples_str)

    for sol in (record.get("solutions") or [])[:3]:
        sol = sol.strip()
        if sol:
            parts.append("### Solution\n```python\n" + sol + "\n```")

    return "\n\n".join(parts)


def record_to_sft_pair(record: Dict) -> Optional[Dict]:
    """
    Convert a record into a prompt/response pair for SFT.
    Returns None if no valid solution exists.
    """
    solutions = [s.strip() for s in (record.get("solutions") or []) if s.strip()]
    if not solutions:
        return None

    problem = record.get("problem", "").strip()
    if not problem:
        return None

    examples_str = _format_examples(record.get("examples") or [])
    tags = record.get("tags") or []
    difficulty = record.get("difficulty", "unknown")

    prompt_parts = ["### Competitive Programming Problem\n"]
    if difficulty != "unknown":
        prompt_parts.append(f"**Difficulty:** {difficulty.capitalize()}")
    if tags:
        prompt_parts.append(f"**Topics:** {', '.join(tags[:8])}")
    prompt_parts.append("\n" + problem)
    if examples_str:
        prompt_parts.append("\n### Examples\n" + examples_str)
    prompt_parts.append("\n### Solution\n```python")

    prompt = "\n".join(prompt_parts)
    response = solutions[0] + "\n```"

    return {"prompt": prompt, "response": response}


# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class DatasetBuilder:
    def __init__(
        self,
        dataset_names: List[str],
        output_dir: str = "./data/processed",
        cache_dir: str = "./data/raw",
        val_ratio: float = 0.02,
        test_ratio: float = 0.01,
        dedup_threshold: float = 0.85,
        max_solutions_per_problem: int = 3,
        cp_filter: Optional[CPFilter] = None,
        seed: int = 42,
        verify_sft: bool = False,
    ):
        self.dataset_names = dataset_names
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.dedup_threshold = dedup_threshold
        self.max_solutions_per_problem = max_solutions_per_problem
        self.cp_filter = cp_filter or CPFilter()
        self.seed = seed
        self.verify_sft = verify_sft

    def run(self) -> Dict[str, str]:
        """
        Execute the full pipeline. Returns dict of output file paths.
        {
            "pretrain_train": "...",
            "pretrain_val": "...",
            "sft_train": "...",
            "sft_val": "...",
            "test": "...",
        }
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Download ────────────────────────────────────────────────
        logger.info("=== Step 1: Downloading datasets ===")
        raw_datasets = download_all_datasets(self.dataset_names, cache_dir=self.cache_dir)

        # ── Step 2: Preprocess + normalize ─────────────────────────────────
        logger.info("=== Step 2: Preprocessing ===")
        all_records: List[Dict] = []
        for name, ds in raw_datasets.items():
            normalized = preprocess_dataset(name, ds)
            all_records.extend(normalized)
        logger.info("Total after preprocessing: %d records", len(all_records))

        # ── Step 3: CP content filter ──────────────────────────────────────
        logger.info("=== Step 3: Filtering ===")
        filtered = [r for r in all_records if self.cp_filter(r)]
        logger.info("After CP filter: %d records (removed %d)", len(filtered), len(all_records) - len(filtered))

        # ── Step 4: Cap solutions per problem ─────────────────────────────
        for r in filtered:
            r["solutions"] = r.get("solutions", [])[:self.max_solutions_per_problem]

        # ── Step 5: Deduplicate ────────────────────────────────────────────
        logger.info("=== Step 4: Deduplication ===")
        deduped = deduplicate_dataset(filtered, threshold=self.dedup_threshold)

        # ── Step 6: Shuffle + split ────────────────────────────────────────
        logger.info("=== Step 5: Splitting ===")
        random.seed(self.seed)
        random.shuffle(deduped)

        n = len(deduped)
        n_test = max(1, int(n * self.test_ratio))
        n_val = max(1, int(n * self.val_ratio))

        test_records = deduped[:n_test]
        val_records = deduped[n_test : n_test + n_val]
        train_records = deduped[n_test + n_val :]

        # Cross-dedup: ensure train doesn't leak into test/val
        train_records, _ = cross_deduplicate(train_records, test_records, threshold=self.dedup_threshold)
        train_records, _ = cross_deduplicate(train_records, val_records, threshold=self.dedup_threshold)

        logger.info(
            "Split: train=%d, val=%d, test=%d", len(train_records), len(val_records), len(test_records)
        )

        # ── Step 7: Format and save ────────────────────────────────────────
        logger.info("=== Step 6: Formatting and saving ===")

        output_paths: Dict[str, str] = {}

        # Pretraining files (plain text documents)
        for split_name, records in [("train", train_records), ("val", val_records)]:
            path = self.output_dir / f"pretrain_{split_name}.jsonl"
            self._save_pretrain(records, path)
            output_paths[f"pretrain_{split_name}"] = str(path)

        # SFT files (prompt/response pairs) — optionally execution-verified
        for split_name, records in [("train", train_records), ("val", val_records)]:
            path = self.output_dir / f"sft_{split_name}.jsonl"
            if self.verify_sft and split_name == "train":
                logger.info("Running execution verification on SFT train set (slow but high quality)...")
                self._save_sft_verified(records, path)
            else:
                self._save_sft(records, path)
            output_paths[f"sft_{split_name}"] = str(path)

        # Test set (raw normalized records)
        test_path = self.output_dir / "test.jsonl"
        self._save_jsonl(test_records, test_path)
        output_paths["test"] = str(test_path)

        # Save metadata
        meta = {
            "dataset_names": self.dataset_names,
            "total_after_preprocess": len(all_records),
            "total_after_filter": len(filtered),
            "total_after_dedup": len(deduped),
            "train_size": len(train_records),
            "val_size": len(val_records),
            "test_size": len(test_records),
            "dedup_threshold": self.dedup_threshold,
            "seed": self.seed,
        }
        meta_path = self.output_dir / "dataset_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Metadata saved to %s", meta_path)

        return output_paths

    def _save_pretrain(self, records: List[Dict], path: Path) -> None:
        texts_saved = 0
        with open(path, "w") as f:
            for record in records:
                text = record_to_pretrain_text(record)
                if len(text) >= 100:
                    f.write(json.dumps({"text": text}) + "\n")
                    texts_saved += 1
        logger.info("Saved %d pretrain examples to %s", texts_saved, path)

    def _save_sft_verified(self, records: List[Dict], path: Path, timeout: float = 8.0) -> None:
        """Save SFT pairs, but only those whose solutions pass execution against examples."""
        pairs_saved = 0
        pairs_skipped = 0
        with open(path, "w") as f:
            for record in records:
                solutions = [s.strip() for s in (record.get("solutions") or []) if s.strip()]
                examples = record.get("examples") or []
                test_code = record.get("test_code") or ""
                problem = record.get("problem", "").strip()

                if not problem or not solutions:
                    pairs_skipped += 1
                    continue

                # If no verifiable test cases, fall back to unverified (keep it)
                if not examples and not test_code:
                    pair = record_to_sft_pair(record)
                    if pair:
                        f.write(json.dumps(pair) + "\n")
                        pairs_saved += 1
                    continue

                # Find first solution that passes
                verified_solution = None
                for sol in solutions:
                    passed = self._quick_verify(sol, examples, test_code, timeout)
                    if passed:
                        verified_solution = sol
                        break

                if verified_solution is None:
                    pairs_skipped += 1
                    continue

                # Build pair using the verified solution
                modified_record = dict(record)
                modified_record["solutions"] = [verified_solution]
                pair = record_to_sft_pair(modified_record)
                if pair:
                    f.write(json.dumps(pair) + "\n")
                    pairs_saved += 1

        logger.info(
            "Verified SFT: saved %d, skipped %d (no passing solution) to %s",
            pairs_saved, pairs_skipped, path,
        )

    def _quick_verify(self, code: str, examples: List[Dict], test_code: str, timeout: float) -> bool:
        """Run code against examples or unit tests. Returns True if it passes."""
        MAX_OUT = 32768

        def _normalize(s: str) -> str:
            return "\n".join(line.rstrip() for line in s.strip().splitlines())

        if examples:
            for ex in examples[:3]:  # check first 3 examples only for speed
                stdin = ex.get("input", "")
                expected = _normalize(ex.get("output", ""))
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(code)
                    tmpfile = f.name
                try:
                    proc = subprocess.run(
                        [sys.executable, tmpfile],
                        input=stdin, capture_output=True, text=True,
                        timeout=timeout,
                        env={**os.environ, "PYTHONPATH": ""},
                    )
                    actual = _normalize(proc.stdout[:MAX_OUT])
                    if actual != expected:
                        return False
                except subprocess.TimeoutExpired:
                    return False
                except Exception:
                    return False
                finally:
                    try:
                        os.unlink(tmpfile)
                    except OSError:
                        pass
            return True

        if test_code and test_code.strip():
            combined = f"{code}\n\n{test_code}\n\nimport unittest\nif __name__ == '__main__':\n    unittest.main(verbosity=0)\n"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(combined)
                tmpfile = f.name
            try:
                proc = subprocess.run(
                    [sys.executable, tmpfile],
                    capture_output=True, text=True,
                    timeout=timeout + 5,
                    env={**os.environ, "PYTHONPATH": ""},
                )
                return proc.returncode == 0
            except Exception:
                return False
            finally:
                try:
                    os.unlink(tmpfile)
                except OSError:
                    pass

        return False

    def _save_sft(self, records: List[Dict], path: Path) -> None:
        pairs_saved = 0
        with open(path, "w") as f:
            for record in records:
                pair = record_to_sft_pair(record)
                if pair is not None:
                    f.write(json.dumps(pair) + "\n")
                    pairs_saved += 1
        logger.info("Saved %d SFT pairs to %s", pairs_saved, path)

    def _save_jsonl(self, records: List[Dict], path: Path) -> None:
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        logger.info("Saved %d records to %s", len(records), path)


def load_jsonl_as_hf_dataset(path: str, text_field: str = "text") -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)
