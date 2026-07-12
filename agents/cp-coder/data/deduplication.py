"""
Near-duplicate removal using MinHash LSH.

Strategy:
  1. Tokenize problem text into character 3-grams (robust to minor edits).
  2. Compute MinHash signature.
  3. Use LSH to find near-duplicate pairs (Jaccard >= threshold).
  4. Keep one representative from each duplicate cluster.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, Iterable, List, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from datasketch import MinHash, MinHashLSH
    _MINHASH_AVAILABLE = True
except ImportError:
    logger.warning("datasketch not installed; deduplication will use exact-hash fallback.")
    _MINHASH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Text normalization for hashing
# ---------------------------------------------------------------------------

def _normalize_for_hash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _char_ngrams(text: str, n: int = 3) -> Set[bytes]:
    return {text[i : i + n].encode("utf-8") for i in range(len(text) - n + 1)}


def _word_ngrams(text: str, n: int = 1) -> Set[bytes]:
    words = text.split()
    return {(" ".join(words[i : i + n])).encode("utf-8") for i in range(len(words) - n + 1)}


def _fingerprint(text: str) -> Set[bytes]:
    normalized = _normalize_for_hash(text)
    return _char_ngrams(normalized, n=3) | _word_ngrams(normalized, n=2)


# ---------------------------------------------------------------------------
# MinHash-based deduplication
# ---------------------------------------------------------------------------

def _build_minhash(shingles: Set[bytes], num_perm: int = 128) -> "MinHash":
    mh = MinHash(num_perm=num_perm)
    for s in shingles:
        mh.update(s)
    return mh


def deduplicate_minhash(
    records: List[Dict],
    threshold: float = 0.85,
    num_perm: int = 128,
    text_key: str = "problem",
) -> List[Dict]:
    """Remove near-duplicates using MinHash LSH. O(n) expected time."""
    if not records:
        return records

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept_indices: List[int] = []
    duplicate_count = 0

    for idx, record in enumerate(records):
        text = record.get(text_key, "")
        shingles = _fingerprint(text)
        if not shingles:
            kept_indices.append(idx)
            continue

        mh = _build_minhash(shingles, num_perm=num_perm)
        key = f"doc_{idx}"

        result = lsh.query(mh)
        if result:
            # Near-duplicate found — skip this record
            duplicate_count += 1
        else:
            lsh.insert(key, mh)
            kept_indices.append(idx)

    deduped = [records[i] for i in kept_indices]
    logger.info(
        "MinHash deduplication: %d → %d records (removed %d duplicates, threshold=%.2f)",
        len(records), len(deduped), duplicate_count, threshold,
    )
    return deduped


# ---------------------------------------------------------------------------
# Exact-hash fallback (no datasketch dependency)
# ---------------------------------------------------------------------------

def deduplicate_exact(
    records: List[Dict],
    text_key: str = "problem",
) -> List[Dict]:
    """Exact deduplication by normalized text hash."""
    seen: Set[str] = set()
    deduped: List[Dict] = []
    for record in records:
        text = _normalize_for_hash(record.get(text_key, ""))
        if text not in seen:
            seen.add(text)
            deduped.append(record)

    removed = len(records) - len(deduped)
    logger.info("Exact deduplication: %d → %d records (removed %d)", len(records), len(deduped), removed)
    return deduped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deduplicate_dataset(
    records: List[Dict],
    threshold: float = 0.85,
    num_perm: int = 128,
    text_key: str = "problem",
) -> List[Dict]:
    """
    Deduplicate records.
    Uses MinHash LSH if datasketch is available, otherwise exact hash.
    """
    if _MINHASH_AVAILABLE:
        return deduplicate_minhash(records, threshold=threshold, num_perm=num_perm, text_key=text_key)
    return deduplicate_exact(records, text_key=text_key)


def cross_deduplicate(
    train: List[Dict],
    test: List[Dict],
    threshold: float = 0.85,
    num_perm: int = 128,
    text_key: str = "problem",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Remove train records that are near-duplicates of test records.
    The test set is unchanged; the train set is cleaned.
    """
    if not test:
        return train, test

    if _MINHASH_AVAILABLE:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        for idx, record in enumerate(test):
            shingles = _fingerprint(record.get(text_key, ""))
            if shingles:
                mh = _build_minhash(shingles, num_perm=num_perm)
                lsh.insert(f"test_{idx}", mh)

        clean_train: List[Dict] = []
        leaked = 0
        for record in train:
            shingles = _fingerprint(record.get(text_key, ""))
            if not shingles:
                clean_train.append(record)
                continue
            mh = _build_minhash(shingles, num_perm=num_perm)
            if lsh.query(mh):
                leaked += 1
            else:
                clean_train.append(record)

        logger.info(
            "Cross-dedup: removed %d train records that leaked into test set", leaked
        )
        return clean_train, test

    # Exact fallback
    test_hashes = {_normalize_for_hash(r.get(text_key, "")) for r in test}
    clean_train = [
        r for r in train
        if _normalize_for_hash(r.get(text_key, "")) not in test_hashes
    ]
    logger.info(
        "Cross-dedup (exact): removed %d train records", len(train) - len(clean_train)
    )
    return clean_train, test
