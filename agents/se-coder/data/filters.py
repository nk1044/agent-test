"""
Content filter for software engineering relevance.

Accepts: web dev, backend, APIs, databases, DevOps, cloud, mobile, system design,
         debugging, testing, security, networking, architecture, microservices.
Rejects: competitive programming, olympiad math, puzzle algorithms,
         pure data science / ML research.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SE positive signals
# ---------------------------------------------------------------------------

SE_TAGS: Set[str] = {
    # Web & frontend
    "react", "vue", "angular", "svelte", "nextjs", "nuxtjs", "html", "css",
    "javascript", "typescript", "webpack", "vite", "tailwind", "bootstrap",
    "redux", "graphql", "rest", "api", "fetch", "ajax", "dom",
    # Backend
    "django", "flask", "fastapi", "express", "nestjs", "spring", "rails",
    "laravel", "gin", "fiber", "actix", "asp.net", "node.js", "nodejs",
    "python", "java", "go", "rust", "c#", "ruby", "php", "kotlin",
    # Databases
    "sql", "postgresql", "mysql", "sqlite", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "firebase", "supabase", "orm", "prisma",
    "sqlalchemy", "database", "migration", "schema", "query", "index",
    # DevOps & cloud
    "docker", "kubernetes", "k8s", "terraform", "ansible", "helm", "ci/cd",
    "github actions", "gitlab ci", "jenkins", "aws", "gcp", "azure", "s3",
    "ec2", "lambda", "serverless", "nginx", "apache", "load balancer",
    "monitoring", "prometheus", "grafana", "elk", "logging", "observability",
    # Mobile
    "android", "ios", "flutter", "react native", "swift", "kotlin", "xcode",
    "android studio", "mobile", "app store", "play store",
    # Architecture & design
    "microservices", "monolith", "event-driven", "cqrs", "event sourcing",
    "design pattern", "solid", "dry", "clean architecture", "hexagonal",
    "domain driven design", "ddd", "api gateway", "service mesh",
    "message queue", "kafka", "rabbitmq", "pub/sub", "websocket",
    # System design
    "scalability", "availability", "reliability", "consistency", "cap theorem",
    "distributed system", "sharding", "replication", "caching", "cdn",
    "rate limiting", "circuit breaker", "saga pattern", "two-phase commit",
    # Testing & quality
    "unit test", "integration test", "e2e test", "pytest", "jest", "mocha",
    "cypress", "playwright", "tdd", "bdd", "code review", "refactoring",
    "technical debt", "code quality", "linting", "formatting",
    # Security
    "authentication", "authorization", "oauth", "jwt", "ssl", "tls",
    "encryption", "hashing", "xss", "csrf", "sql injection", "owasp",
    "penetration testing", "vulnerability", "security audit",
    # Networking
    "http", "https", "tcp", "udp", "dns", "grpc", "websocket", "sse",
    "proxy", "reverse proxy", "firewall", "vpn", "network",
    # Performance
    "performance", "optimization", "profiling", "latency", "throughput",
    "bottleneck", "memory leak", "garbage collection", "caching strategy",
    # Debugging
    "debugging", "bug", "error", "exception", "stack trace", "logging",
    "breakpoint", "root cause", "postmortem", "incident",
    # General SE
    "software engineering", "software development", "programming", "coding",
    "git", "version control", "agile", "scrum", "sprint", "deployment",
    "production", "staging", "environment", "configuration",
}

SE_KEYWORDS: List[str] = [
    r"\bimport\s+React\b",
    r"\bimport\s+(express|flask|django|fastapi|gin|fiber)\b",
    r"\bfrom\s+(django|flask|fastapi|express|spring)\b",
    r"\bapp\.(get|post|put|delete|patch)\s*\(",
    r"\brouter\.(get|post|put|delete)\s*\(",
    r"<!DOCTYPE html>",
    r"<html",
    r"\bSELECT\s+\w.*?\bFROM\b",
    r"\bCREATE\s+TABLE\b",
    r"\bINSERT\s+INTO\b",
    r"\bUPDATE\s+\w+\s+SET\b",
    r"docker-compose",
    r"FROM\s+\w+:\w+",  # Dockerfile FROM
    r"kubernetes|kubectl|helm\s+install",
    r"\.github/workflows",
    r"pipeline\s*\{",   # Jenkins/GitLab CI
    r"\baws\.\w+\(",
    r"boto3\.",
    r"@app\.route",
    r"@router\.",
    r"useEffect|useState|useContext",
    r"\bsetup\.py\b|pyproject\.toml",
    r"package\.json",
    r"npm\s+(install|run|start)",
    r"yarn\s+(add|install|start)",
    r"git\s+(commit|push|pull|clone|merge)",
    r"def\s+test_\w+",
    r"class\s+Test\w+",
    r"@pytest\.",
    r"describe\s*\(",  # Jest
    r"it\s*\(\s*['\"]",  # Mocha/Jest
    r"@controller|@service|@injectable",  # Spring/NestJS
    r"microservice|api\s+gateway",
    r"redis\.set|redis\.get",
    r"\.find\(\{|\.aggregate\(\[",  # MongoDB
    r"async\s+def\s+\w+|await\s+\w+",
    r"try\s*\{.*?catch\s*\(",
    r"JWT|Bearer\s+token",
    r"CORS|cors\(",
]

SE_KEYWORD_PATTERNS = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in SE_KEYWORDS]

# ---------------------------------------------------------------------------
# Negative signals (reject pure CP / math olympiad content)
# ---------------------------------------------------------------------------

REJECT_TAGS: Set[str] = {
    "competitive programming", "codeforces", "atcoder", "leetcode",
    "dynamic programming", "greedy algorithm", "binary search",
    "segment tree", "fenwick tree", "graph theory", "number theory",
    "olympiad", "icpc", "ioi", "contest", "puzzle",
    "data science", "machine learning", "deep learning", "neural network",
    "nlp", "computer vision", "reinforcement learning",
}

REJECT_KEYWORDS: List[str] = [
    r"1000000007|998244353",
    r"\bMOD\s*=\s*10\*\*9",
    r"T\s*=\s*int\(input\(\)\)",
    r"sys\.stdin\.readline",
    r"cin\s*>>\s*\w+\s*>>\s*\w+",   # CP-style multi-read
    r"scanf\(\s*['\"]%[dls]",
    r"\bheapq\b.*\bpush\b|bisect_left|bisect_right",
    r"Codeforces|AtCoder|HackerRank|HackerEarth",
    r"import\s+tensorflow|import\s+torch\b",
    r"import\s+sklearn|from\s+sklearn",
    r"\.fit\(.*\)|\.predict\(",
    r"plt\.plot|plt\.show|matplotlib",
    r"import\s+pandas\b",
    r"pd\.DataFrame|pd\.read_csv",
]

REJECT_KEYWORD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REJECT_KEYWORDS]


class SEFilter:
    """Classify whether content is software engineering relevant."""

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 32000,
        require_positive_signal: bool = False,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.require_positive_signal = require_positive_signal

    def _has_se_tag(self, tags: List[str]) -> bool:
        normalized = {t.lower().strip() for t in tags}
        return bool(normalized & SE_TAGS)

    def _has_reject_tag(self, tags: List[str]) -> bool:
        normalized = {t.lower().strip() for t in tags}
        return bool(normalized & REJECT_TAGS)

    def _has_se_keyword(self, text: str) -> bool:
        return any(p.search(text) for p in SE_KEYWORD_PATTERNS)

    def _has_reject_keyword(self, text: str) -> bool:
        return any(p.search(text) for p in REJECT_KEYWORD_PATTERNS)

    def is_se_relevant(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> bool:
        tags = tags or []

        # Always trust SE-specific sources
        trusted_sources = {
            "stack_exchange", "magicoder_oss", "magicoder_evol",
            "code_feedback", "evol_codealpaca", "glaive_code",
            "text_to_sql", "self_oss_instruct", "commitpackft",
            "ultrachat",
        }
        if source and any(s in (source or "").lower() for s in trusted_sources):
            return not self._has_reject_keyword(text)

        if self._has_reject_tag(tags) or self._has_reject_keyword(text):
            return False

        if not (self.min_length <= len(text) <= self.max_length):
            return False

        if self.require_positive_signal:
            return self._has_se_tag(tags) or self._has_se_keyword(text)

        return True

    def filter_dict(self, record: Dict) -> bool:
        text = record.get("text") or record.get("content") or record.get("problem") or ""
        if not text:
            # Try to build text from instruction+response for SFT records
            text = (record.get("instruction") or record.get("prompt") or "") + " " + \
                   (record.get("response") or record.get("answer") or "")
        return self.is_se_relevant(
            text=text,
            tags=record.get("tags") or [],
            source=record.get("source") or "",
        )

    def __call__(self, record: Dict) -> bool:
        return self.filter_dict(record)
