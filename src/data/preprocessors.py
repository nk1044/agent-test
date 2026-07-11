"""
Per-dataset normalization into a unified schema.

Unified record schema:
{
    "problem":   str,          # Problem statement (cleaned)
    "solutions": List[str],    # Accepted Python/C++ solutions
    "examples":  List[{"input": str, "output": str}],
    "difficulty": str,         # "easy" | "medium" | "hard" | "unknown"
    "tags":      List[str],    # Algorithm / topic tags
    "source":    str,          # Dataset name
    "language":  str,          # Primary solution language hint
}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Strip excessive whitespace while preserving code blocks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _parse_json_safe(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _difficulty_normalize(raw: Any) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).lower()
    if s in ("easy", "0", "beginner", "1"):
        return "easy"
    if s in ("medium", "1", "2", "intermediate"):
        return "medium"
    if s in ("hard", "3", "4", "5", "difficult", "expert", "2", "competition"):
        return "hard"
    # numeric Codeforces-style rating
    try:
        rating = int(s)
        if rating <= 1200:
            return "easy"
        if rating <= 2000:
            return "medium"
        return "hard"
    except ValueError:
        pass
    return "unknown"


def _extract_examples(io_data: Any) -> List[Dict[str, str]]:
    """Parse input/output examples from various formats."""
    examples: List[Dict[str, str]] = []
    if io_data is None:
        return examples

    parsed = _parse_json_safe(io_data)
    if parsed is None:
        return examples

    # Format: {"inputs": [...], "outputs": [...]}
    if isinstance(parsed, dict):
        inputs = parsed.get("inputs") or parsed.get("input") or []
        outputs = parsed.get("outputs") or parsed.get("output") or []
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]
        for inp, out in zip(inputs, outputs):
            examples.append({"input": str(inp).strip(), "output": str(out).strip()})

    # Format: [{"input": ..., "output": ...}, ...]
    elif isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                inp = item.get("input") or item.get("stdin") or ""
                out = item.get("output") or item.get("stdout") or ""
                examples.append({"input": str(inp).strip(), "output": str(out).strip()})

    return examples[:5]  # cap at 5 examples per problem


def _pick_solutions(solutions: Any, languages: Optional[List[str]] = None) -> List[str]:
    """Flatten and return a list of solution strings."""
    if solutions is None:
        return []
    if isinstance(solutions, str):
        try:
            solutions = json.loads(solutions)
        except Exception:
            return [solutions] if solutions.strip() else []
    if isinstance(solutions, dict):
        # CodeContests format: {"language": [...], "solution": [...]}
        langs = solutions.get("language", [])
        sols = solutions.get("solution", [])
        result = []
        for lang, sol in zip(langs, sols):
            if languages and lang not in languages:
                continue
            result.append(sol)
        return result
    if isinstance(solutions, list):
        result = []
        for item in solutions:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                sol = item.get("solution") or item.get("code") or item.get("content") or ""
                if sol:
                    result.append(sol)
        return result
    return []


# ---------------------------------------------------------------------------
# Per-dataset normalizers
# ---------------------------------------------------------------------------

def normalize_taco(record: Dict) -> Optional[Dict]:
    """Normalize BAAI/TACO record."""
    problem = _clean_text(record.get("question") or record.get("problem") or "")
    if not problem:
        return None

    solutions = _pick_solutions(record.get("solutions"))
    examples = _extract_examples(record.get("input_output"))
    tags_raw = record.get("tags") or record.get("raw_tags") or []
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in tags_raw.split(",")]
    difficulty = _difficulty_normalize(record.get("difficulty"))

    return {
        "problem": problem,
        "solutions": solutions,
        "examples": examples,
        "difficulty": difficulty,
        "tags": list(tags_raw),
        "source": "taco",
        "language": "python",
    }


def normalize_apps(record: Dict) -> Optional[Dict]:
    """Normalize codeparrot/apps record."""
    problem = _clean_text(record.get("question") or "")
    if not problem:
        return None

    solutions = _pick_solutions(record.get("solutions"))
    examples = _extract_examples(record.get("input_output"))

    diff_map = {"introductory": "easy", "interview": "medium", "competition": "hard"}
    difficulty = diff_map.get(str(record.get("difficulty", "")).lower(), "unknown")

    return {
        "problem": problem,
        "solutions": solutions,
        "examples": examples,
        "difficulty": difficulty,
        "tags": [],
        "source": "apps",
        "language": "python",
    }


def normalize_code_contests(record: Dict) -> Optional[Dict]:
    """Normalize deepmind/code_contests record."""
    problem = _clean_text(record.get("description") or "")
    if not problem:
        return None

    # Solutions: keep Python3 and C++ only for CP relevance
    solutions_raw = record.get("solutions") or {}
    CP_LANGS = {3, 4}  # 3=Python3, 4=Python2 in CodeContests enum; also check strings
    lang_list = solutions_raw.get("language", [])
    sol_list = solutions_raw.get("solution", [])
    solutions = []
    for lang, sol in zip(lang_list, sol_list):
        # language is an int in code_contests: 1=C++, 2=Java, 3=Python3, 4=Python2
        if lang in (1, 3, 4) or str(lang) in ("1", "3", "4"):
            solutions.append(sol)

    if not solutions:
        solutions = _pick_solutions(solutions_raw)

    # Examples from public_tests
    examples: List[Dict[str, str]] = []
    public = record.get("public_tests") or {}
    pub_inputs = public.get("input", [])
    pub_outputs = public.get("output", [])
    for inp, out in zip(pub_inputs, pub_outputs):
        examples.append({"input": str(inp).strip(), "output": str(out).strip()})

    tags = record.get("cf_tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    difficulty = _difficulty_normalize(record.get("cf_rating") or record.get("difficulty"))
    source = record.get("source", "")
    if isinstance(source, int):
        source_map = {1: "codeforces", 2: "codechef", 3: "atcoder", 4: "aizu", 5: "hackerearth", 6: "codecraft"}
        source = source_map.get(source, "code_contests")

    return {
        "problem": problem,
        "solutions": solutions,
        "examples": examples[:5],
        "difficulty": difficulty,
        "tags": list(tags),
        "source": f"code_contests/{source}",
        "language": "python",
    }


def normalize_codeforces(record: Dict) -> Optional[Dict]:
    """Normalize open-r1/codeforces record."""
    problem = _clean_text(
        record.get("description") or record.get("problem") or record.get("statement") or ""
    )
    if not problem:
        return None

    solutions = _pick_solutions(
        record.get("solutions") or record.get("accepted_solutions") or []
    )

    examples: List[Dict[str, str]] = []
    for ex in (record.get("examples") or []):
        if isinstance(ex, dict):
            examples.append({
                "input": str(ex.get("input", "")).strip(),
                "output": str(ex.get("output", "")).strip(),
            })

    tags = record.get("tags") or []
    difficulty = _difficulty_normalize(record.get("rating") or record.get("difficulty"))

    return {
        "problem": problem,
        "solutions": solutions,
        "examples": examples[:5],
        "difficulty": difficulty,
        "tags": list(tags),
        "source": "codeforces",
        "language": "python",
    }


def normalize_codeforces_cots(record: Dict) -> Optional[Dict]:
    """Normalize open-r1/codeforces-cots (chain-of-thought) record."""
    problem = _clean_text(record.get("problem") or record.get("description") or "")
    if not problem:
        return None

    # This dataset has reasoning chains; treat as solution
    solution_text = record.get("solution") or record.get("code") or ""
    reasoning = record.get("reasoning") or record.get("chain_of_thought") or ""
    solutions = []
    if solution_text:
        solutions.append(solution_text)

    return {
        "problem": problem,
        "solutions": solutions,
        "examples": [],
        "difficulty": _difficulty_normalize(record.get("rating")),
        "tags": record.get("tags") or [],
        "source": "codeforces_cots",
        "language": "python",
        "reasoning": reasoning,
    }


def normalize_leetcode(record: Dict) -> Optional[Dict]:
    """Normalize greengerong/leetcode record (keep Hard/Medium only)."""
    difficulty = str(record.get("difficulty") or "").lower()
    if difficulty not in ("hard", "medium"):
        return None  # skip Easy LeetCode — too basic for CP focus

    problem = _clean_text(record.get("content") or record.get("problem") or "")
    if not problem:
        return None

    solutions = []
    for key in ("python", "python3", "c++", "java"):
        sol = record.get(key) or record.get(f"{key}_solution")
        if sol and isinstance(sol, str) and sol.strip():
            solutions.append(sol.strip())

    tags_raw = record.get("related_topics") or record.get("tags") or []
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in tags_raw.split(",")]

    return {
        "problem": problem,
        "solutions": solutions,
        "examples": [],
        "difficulty": difficulty,
        "tags": list(tags_raw),
        "source": "leetcode",
        "language": "python",
    }


def normalize_generic(record: Dict, source: str = "unknown") -> Optional[Dict]:
    """Best-effort normalization for unrecognized datasets."""
    problem = (
        record.get("problem")
        or record.get("question")
        or record.get("description")
        or record.get("statement")
        or record.get("text")
        or ""
    )
    problem = _clean_text(str(problem))
    if not problem:
        return None

    solutions = _pick_solutions(
        record.get("solutions")
        or record.get("solution")
        or record.get("code")
        or record.get("answer")
        or []
    )

    return {
        "problem": problem,
        "solutions": solutions,
        "examples": _extract_examples(record.get("examples") or record.get("input_output")),
        "difficulty": _difficulty_normalize(record.get("difficulty")),
        "tags": record.get("tags") or [],
        "source": source,
        "language": "python",
    }


# Map dataset name -> normalizer function
NORMALIZER_MAP = {
    "taco": normalize_taco,
    "apps": normalize_apps,
    "code_contests": normalize_code_contests,
    "codeforces": normalize_codeforces,
    "codeforces_cots": normalize_codeforces_cots,
    "leetcode": normalize_leetcode,
}


def preprocess_dataset(name: str, dataset) -> List[Dict]:
    """Normalize an entire dataset into unified records. Returns list of dicts."""
    normalizer = NORMALIZER_MAP.get(name, lambda r: normalize_generic(r, source=name))
    results = []
    errors = 0

    for record in dataset:
        try:
            normalized = normalizer(dict(record))
            if normalized is not None:
                results.append(normalized)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                logger.debug("Normalization error in '%s': %s", name, exc)

    logger.info(
        "Preprocessed '%s': %d kept, %d errors from %d total",
        name, len(results), errors, len(dataset),
    )
    return results
