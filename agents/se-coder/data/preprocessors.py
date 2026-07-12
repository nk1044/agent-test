"""
Per-dataset normalization into a unified SE schema.

Unified record schema:
{
    "instruction": str,   # The task / question
    "response":    str,   # The answer / solution
    "context":     str,   # Optional context (file content, error, etc.)
    "type":        str,   # "code" | "qa" | "debug" | "design" | "sql" | "pretrain"
    "language":    str,   # Primary language
    "source":      str,   # Dataset name
    "tags":        List[str],
}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _clean(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text) if text else ""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _parse_json_safe(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if isinstance(s, str) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The Stack v2 normalizers
# ---------------------------------------------------------------------------

def normalize_the_stack(record: Dict, language: str = "python") -> Optional[Dict]:
    """Normalize bigcode/the-stack-v2-dedup record."""
    content = _clean(record.get("content") or record.get("text") or "")
    if len(content) < 100:
        return None

    # Skip test files, generated files, vendor files
    path = str(record.get("path") or record.get("filename") or "")
    skip_patterns = ["/vendor/", "/node_modules/", ".min.js", ".generated.", "test_", "_test.", ".spec."]
    if any(p in path.lower() for p in skip_patterns):
        return None

    return {
        "instruction": f"Complete the following {language} code or explain what it does:",
        "response": content[:8000],  # cap to avoid extremely long files
        "context": path,
        "type": "pretrain",
        "language": language.lower(),
        "source": f"the_stack_{language.lower()}",
        "tags": [language.lower()],
    }


def normalize_the_stack_python(record: Dict) -> Optional[Dict]:
    return normalize_the_stack(record, "python")

def normalize_the_stack_js(record: Dict) -> Optional[Dict]:
    return normalize_the_stack(record, "javascript")

def normalize_the_stack_ts(record: Dict) -> Optional[Dict]:
    return normalize_the_stack(record, "typescript")

def normalize_the_stack_go(record: Dict) -> Optional[Dict]:
    return normalize_the_stack(record, "go")

def normalize_the_stack_java(record: Dict) -> Optional[Dict]:
    return normalize_the_stack(record, "java")

def normalize_the_stack_rust(record: Dict) -> Optional[Dict]:
    return normalize_the_stack(record, "rust")

def normalize_the_stack_sql(record: Dict) -> Optional[Dict]:
    return normalize_the_stack(record, "sql")

def normalize_the_stack_shell(record: Dict) -> Optional[Dict]:
    return normalize_the_stack(record, "shell")


# ---------------------------------------------------------------------------
# CommitPackFT normalizer
# ---------------------------------------------------------------------------

def normalize_commitpackft(record: Dict) -> Optional[Dict]:
    """Normalize bigcode/commitpackft — git commits with messages."""
    message = _clean(record.get("message") or "")
    old_content = _clean(record.get("old_contents") or record.get("old_content") or "")
    new_content = _clean(record.get("new_contents") or record.get("new_content") or "")
    lang = _clean(record.get("lang") or record.get("language") or "unknown")

    if not message or not new_content:
        return None

    if old_content:
        instruction = f"Refactor or update the following {lang} code as described:\n\n{message}\n\nOriginal code:\n```{lang}\n{old_content[:2000]}\n```"
        response = f"Updated code:\n```{lang}\n{new_content[:4000]}\n```"
    else:
        instruction = f"Write {lang} code that: {message}"
        response = f"```{lang}\n{new_content[:4000]}\n```"

    return {
        "instruction": instruction,
        "response": response,
        "context": "",
        "type": "code",
        "language": lang.lower(),
        "source": "commitpackft",
        "tags": ["git", "refactoring", lang.lower()],
    }


# ---------------------------------------------------------------------------
# StackExchange normalizer
# ---------------------------------------------------------------------------

def normalize_stack_exchange(record: Dict) -> Optional[Dict]:
    """Normalize ArmelR/stack-exchange-instruction."""
    instruction = _clean(record.get("instruction") or record.get("question") or record.get("title") or "")
    response = _clean(record.get("response") or record.get("answer") or "")

    if not instruction or not response or len(response) < 50:
        return None

    # Detect type from content
    rec_type = "qa"
    if any(kw in instruction.lower() for kw in ["debug", "error", "exception", "traceback", "fix"]):
        rec_type = "debug"
    elif any(kw in instruction.lower() for kw in ["design", "architecture", "scale", "system"]):
        rec_type = "design"
    elif any(kw in instruction.lower() for kw in ["sql", "query", "database", "select", "table"]):
        rec_type = "sql"

    return {
        "instruction": instruction,
        "response": response,
        "context": "",
        "type": rec_type,
        "language": "mixed",
        "source": "stack_exchange",
        "tags": list(record.get("tags") or []),
    }


# ---------------------------------------------------------------------------
# Magicoder normalizers
# ---------------------------------------------------------------------------

def normalize_magicoder_oss(record: Dict) -> Optional[Dict]:
    instruction = _clean(record.get("problem") or record.get("instruction") or "")
    response = _clean(record.get("solution") or record.get("response") or "")
    if not instruction or not response:
        return None
    return {
        "instruction": instruction,
        "response": response,
        "context": _clean(record.get("seed_code") or ""),
        "type": "code",
        "language": _clean(record.get("lang") or "python").lower(),
        "source": "magicoder_oss",
        "tags": ["code generation"],
    }


def normalize_magicoder_evol(record: Dict) -> Optional[Dict]:
    instruction = _clean(record.get("instruction") or "")
    response = _clean(record.get("response") or "")
    if not instruction or not response:
        return None
    return {
        "instruction": instruction,
        "response": response,
        "context": "",
        "type": "code",
        "language": "python",
        "source": "magicoder_evol",
        "tags": ["code generation"],
    }


# ---------------------------------------------------------------------------
# CodeFeedback normalizer
# ---------------------------------------------------------------------------

def normalize_code_feedback(record: Dict) -> Optional[Dict]:
    instruction = _clean(record.get("query") or record.get("instruction") or record.get("question") or "")
    response = _clean(record.get("answer") or record.get("response") or "")
    if not instruction or not response:
        return None
    return {
        "instruction": instruction,
        "response": response,
        "context": "",
        "type": "code",
        "language": _clean(record.get("lang") or "python").lower(),
        "source": "code_feedback",
        "tags": [],
    }


# ---------------------------------------------------------------------------
# evol-codealpaca normalizer
# ---------------------------------------------------------------------------

def normalize_evol_codealpaca(record: Dict) -> Optional[Dict]:
    instruction = _clean(record.get("instruction") or "")
    response = _clean(record.get("output") or record.get("response") or "")
    if not instruction or not response:
        return None
    return {
        "instruction": instruction,
        "response": response,
        "context": _clean(record.get("input") or ""),
        "type": "code",
        "language": "python",
        "source": "evol_codealpaca",
        "tags": [],
    }


# ---------------------------------------------------------------------------
# Glaive Code Assistant normalizer
# ---------------------------------------------------------------------------

def normalize_glaive_code(record: Dict) -> Optional[Dict]:
    """Normalize glaiveai/glaive-code-assistant-v3 — chat format."""
    chat = record.get("chat") or ""
    system = _clean(record.get("system") or "")

    # Extract first user message and assistant response
    user_match = re.search(r"USER:\s*(.*?)(?:ASSISTANT:|$)", chat, re.DOTALL)
    asst_match = re.search(r"ASSISTANT:\s*(.*?)(?:USER:|$)", chat, re.DOTALL)

    if not user_match or not asst_match:
        # Try structured format
        instruction = _clean(record.get("instruction") or record.get("question") or "")
        response = _clean(record.get("response") or record.get("answer") or "")
        if not instruction or not response:
            return None
    else:
        instruction = _clean(user_match.group(1))
        response = _clean(asst_match.group(1))

    if not instruction or not response or len(response) < 20:
        return None

    return {
        "instruction": instruction,
        "response": response,
        "context": system,
        "type": "code",
        "language": _clean(record.get("language") or "python").lower(),
        "source": "glaive_code",
        "tags": [],
    }


# ---------------------------------------------------------------------------
# Text-to-SQL normalizer
# ---------------------------------------------------------------------------

def normalize_text_to_sql(record: Dict) -> Optional[Dict]:
    instruction = _clean(
        record.get("nl_prompt") or record.get("instruction") or record.get("question") or ""
    )
    sql = _clean(
        record.get("sql") or record.get("query") or record.get("response") or ""
    )
    schema = _clean(record.get("sql_schema") or record.get("schema") or record.get("context") or "")

    if not instruction or not sql:
        return None

    context = f"Schema:\n{schema}" if schema else ""

    return {
        "instruction": f"Write a SQL query to: {instruction}",
        "response": f"```sql\n{sql}\n```",
        "context": context,
        "type": "sql",
        "language": "sql",
        "source": "text_to_sql",
        "tags": ["sql", "database"],
    }


# ---------------------------------------------------------------------------
# UltraChat normalizer
# ---------------------------------------------------------------------------

def normalize_ultrachat(record: Dict) -> Optional[Dict]:
    """Normalize HuggingFaceH4/ultrachat_200k — keep only technical messages."""
    messages = record.get("messages") or []
    if len(messages) < 2:
        return None

    # Find first user and assistant pair
    user_msg = ""
    asst_msg = ""
    for msg in messages:
        role = msg.get("role", "")
        content = _clean(msg.get("content") or "")
        if role == "user" and not user_msg:
            user_msg = content
        elif role == "assistant" and not asst_msg:
            asst_msg = content

    if not user_msg or not asst_msg:
        return None

    # Quick check: must be technical
    tech_signals = ["code", "api", "database", "server", "deploy", "docker",
                    "python", "javascript", "sql", "git", "debug", "error",
                    "function", "class", "import", "framework", "library"]
    combined = (user_msg + asst_msg).lower()
    if not any(s in combined for s in tech_signals):
        return None

    return {
        "instruction": user_msg,
        "response": asst_msg,
        "context": "",
        "type": "qa",
        "language": "mixed",
        "source": "ultrachat",
        "tags": [],
    }


# ---------------------------------------------------------------------------
# Self-OSS-Instruct normalizer
# ---------------------------------------------------------------------------

def normalize_self_oss_instruct(record: Dict) -> Optional[Dict]:
    instruction = _clean(record.get("prompt") or record.get("instruction") or "")
    response = _clean(record.get("response") or record.get("completion") or "")
    if not instruction or not response:
        return None
    return {
        "instruction": instruction,
        "response": response,
        "context": "",
        "type": "code",
        "language": _clean(record.get("language") or "python").lower(),
        "source": "self_oss_instruct",
        "tags": [],
    }


# ---------------------------------------------------------------------------
# Normalizer registry
# ---------------------------------------------------------------------------

NORMALIZER_MAP = {
    "the_stack_python": normalize_the_stack_python,
    "the_stack_js": normalize_the_stack_js,
    "the_stack_ts": normalize_the_stack_ts,
    "the_stack_go": normalize_the_stack_go,
    "the_stack_java": normalize_the_stack_java,
    "the_stack_rust": normalize_the_stack_rust,
    "the_stack_sql": normalize_the_stack_sql,
    "the_stack_shell": normalize_the_stack_shell,
    "commitpackft": normalize_commitpackft,
    "stack_exchange": normalize_stack_exchange,
    "magicoder_oss": normalize_magicoder_oss,
    "magicoder_evol": normalize_magicoder_evol,
    "code_feedback": normalize_code_feedback,
    "evol_codealpaca": normalize_evol_codealpaca,
    "glaive_code": normalize_glaive_code,
    "text_to_sql": normalize_text_to_sql,
    "ultrachat": normalize_ultrachat,
    "self_oss_instruct": normalize_self_oss_instruct,
}


def preprocess_dataset(name: str, dataset) -> List[Dict]:
    """Normalize an entire dataset into unified SE records."""
    normalizer = NORMALIZER_MAP.get(name)
    if not normalizer:
        logger.warning("No normalizer for '%s' — skipping", name)
        return []

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

    logger.info("Preprocessed '%s': %d kept, %d errors", name, len(results), errors)
    return results
