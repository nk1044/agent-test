"""
Content filter for competitive programming relevance.

Keeps problems that are about algorithmic/mathematical problem solving.
Rejects software engineering, web dev, mobile, DevOps, UI, and documentation.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Competitive programming positive signals
# ---------------------------------------------------------------------------

CP_TAGS: Set[str] = {
    # Algorithms
    "dynamic programming", "dp", "greedy", "divide and conquer",
    "binary search", "two pointers", "sliding window", "sorting",
    "backtracking", "recursion", "memoization", "brute force",
    # Data structures
    "array", "linked list", "stack", "queue", "deque", "heap", "priority queue",
    "hash map", "hash set", "trie", "segment tree", "fenwick tree", "bit",
    "binary indexed tree", "disjoint set", "union find", "sparse table",
    # Graphs
    "graph", "tree", "bfs", "dfs", "shortest path", "dijkstra", "bellman-ford",
    "floyd-warshall", "minimum spanning tree", "kruskal", "prim",
    "topological sort", "strongly connected components", "tarjan", "kosaraju",
    "bipartite", "matching", "network flow", "max flow", "articulation point",
    "bridge", "euler path", "hamiltonian", "dag",
    # Math / Number theory
    "number theory", "prime", "sieve", "gcd", "lcm", "modular arithmetic",
    "modular inverse", "euler totient", "chinese remainder theorem", "crt",
    "combinatorics", "permutation", "combination", "binomial coefficient",
    "matrix exponentiation", "fast fourier transform", "fft", "ntt",
    "probability", "expected value", "game theory", "nim",
    # Geometry
    "geometry", "convex hull", "line intersection", "polygon",
    "computational geometry", "coordinate geometry",
    # Strings
    "string", "kmp", "z-algorithm", "rabin-karp", "suffix array",
    "suffix automaton", "aho-corasick", "palindrome", "manacher",
    "string matching", "pattern matching",
    # Problem types
    "competitive programming", "codeforces", "atcoder", "leetcode",
    "algorithm", "algorithmic", "contest", "olympiad", "icpc", "ioi",
    "data structure", "optimization", "construction", "counting",
    "interactive", "output the answer", "modulo", "mod 1e9+7",
}

CP_KEYWORDS: List[str] = [
    r"\bT\s*=\s*int\(input\(\)\)",           # typical CP input pattern
    r"for.*in range.*int.*input",
    r"sys\.stdin",
    r"scanf\(",
    r"printf\(",
    r"cin\s*>>",
    r"cout\s*<<",
    r"\bmod\s*=\s*10\*\*9",
    r"\bmod\s*=\s*1e9\+7",
    r"\bMOD\s*=\s*10\*\*9",
    r"1000000007",
    r"998244353",
    r"\bINF\s*=\s*float\('inf'\)",
    r"\bpq\s*=\s*\[\]",                      # heapq usage
    r"heappush|heappop",
    r"bisect_left|bisect_right",
    r"defaultdict|Counter\(",
    r"adjacency list|adj\[",
    r"dp\[",
    r"\bfib\b|\bfibonacci\b",
    r"prefix sum|suffix sum",
    r"binary search|bisect",
    r"n, m\s*=\s*map|n,m\s*=\s*map",
    r"constraints?:\s*\d",
    r"\d\s*≤\s*[Nn]\s*≤\s*\d",
    r"\d+\s*<=\s*[Nn]\s*<=\s*\d+",
    r"input format|output format",
    r"sample input|sample output",
    r"time limit|memory limit",
    r"queries?|operations?",
]

CP_KEYWORD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CP_KEYWORDS]


# ---------------------------------------------------------------------------
# Negative signals (reject these)
# ---------------------------------------------------------------------------

REJECT_TAGS: Set[str] = {
    "web development", "frontend", "backend", "react", "angular", "vue",
    "html", "css", "javascript", "typescript", "nodejs", "django", "flask",
    "spring", "rest api", "graphql", "http", "sql", "database", "orm",
    "docker", "kubernetes", "devops", "ci/cd", "deployment", "cloud",
    "aws", "gcp", "azure", "mobile", "android", "ios", "flutter", "react native",
    "ui", "ux", "design", "documentation", "readme", "tutorial", "blog",
    "machine learning", "deep learning", "neural network", "nlp",
    "data science", "pandas", "numpy", "matplotlib",
    "scraping", "selenium", "playwright",
    "game development", "unity", "unreal",
    "testing", "unit test", "integration test", "pytest",
    "logging", "monitoring", "observability",
}

REJECT_KEYWORDS: List[str] = [
    r"\bimport\s+React\b",
    r"\bimport\s+flask\b",
    r"\bimport\s+django\b",
    r"\bfrom\s+django\b",
    r"\bimport\s+express\b",
    r"\bimport\s+fastapi\b",
    r"<!DOCTYPE html>",
    r"<html",
    r"SELECT\s+\w+\s+FROM\b",
    r"CREATE\s+TABLE\b",
    r"@app\.route",
    r"@router\.",
    r"app\.get\(|app\.post\(",
    r"useEffect|useState\b",
    r"import\s+tensorflow|import\s+torch\b",
    r"import\s+pandas\b",
    r"import\s+sklearn",
    r"\.fit\(|\.predict\(",
    r"docker-compose",
    r"kubernetes|kubectl",
]

REJECT_KEYWORD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REJECT_KEYWORDS]


class CPFilter:
    """Classify whether a problem/solution is competitive-programming relevant."""

    def __init__(
        self,
        min_problem_length: int = 50,
        max_problem_length: int = 8000,
        min_solution_length: int = 20,
        max_solution_length: int = 10000,
        require_positive_signal: bool = False,
    ):
        self.min_problem_length = min_problem_length
        self.max_problem_length = max_problem_length
        self.min_solution_length = min_solution_length
        self.max_solution_length = max_solution_length
        self.require_positive_signal = require_positive_signal

    def _has_cp_tag(self, tags: List[str]) -> bool:
        normalized = {t.lower().strip() for t in tags}
        return bool(normalized & CP_TAGS)

    def _has_reject_tag(self, tags: List[str]) -> bool:
        normalized = {t.lower().strip() for t in tags}
        return bool(normalized & REJECT_TAGS)

    def _has_cp_keyword(self, text: str) -> bool:
        return any(p.search(text) for p in CP_KEYWORD_PATTERNS)

    def _has_reject_keyword(self, text: str) -> bool:
        return any(p.search(text) for p in REJECT_KEYWORD_PATTERNS)

    def is_cp_relevant(
        self,
        problem: str,
        solutions: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> bool:
        """Return True if the item is competitive-programming relevant."""
        tags = tags or []
        solutions = solutions or []

        # Always trust certain authoritative sources
        trusted_sources = {"taco", "code_contests", "codeforces", "codeforces_cots"}
        if source and any(s in (source or "").lower() for s in trusted_sources):
            # Still apply reject keyword filter on solutions
            for sol in solutions:
                if self._has_reject_keyword(sol):
                    return False
            return True

        # Reject by tag
        if self._has_reject_tag(tags):
            return False

        # Reject by keyword in problem statement
        if self._has_reject_keyword(problem):
            return False

        # Reject by keyword in solutions
        for sol in solutions:
            if self._has_reject_keyword(sol):
                return False

        # Length checks on problem
        if not (self.min_problem_length <= len(problem) <= self.max_problem_length):
            return False

        # Positive signal check
        if self.require_positive_signal:
            positive = self._has_cp_tag(tags) or self._has_cp_keyword(problem)
            if not positive:
                return False

        return True

    def filter_dict(self, record: Dict) -> bool:
        """Callable suitable for datasets.filter()."""
        return self.is_cp_relevant(
            problem=record.get("problem", ""),
            solutions=record.get("solutions", []),
            tags=record.get("tags", []),
            source=record.get("source", ""),
        )

    def __call__(self, record: Dict) -> bool:
        return self.filter_dict(record)
