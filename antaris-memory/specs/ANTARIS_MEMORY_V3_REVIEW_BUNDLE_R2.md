# antaris-memory v3.2 + v3.3 — Review Bundle (R2)

**Package:** antaris-memory  
**Version:** 3.3.0  
**Review Round:** R2 (post-GPT fixes applied)  
**Tests:** 381/381 passing (2.66s)  
**Reviewer:** Gemini

---

## What Changed Since R1

GPT o3 reviewed R1 and identified 8 issues. All have been applied:

### Must-Fix (Correctness)
1. **PPMI marginals now computed in pair space** — `similar_words()` was mixing token counts (`_word_counts`) with pair counts (`_total_pairs`). Now uses `_get_pair_marginal()` which sums row values from the co-occurrence matrix, keeping joint and marginals on the same denominator.
2. **Single-word false positives guarded** — `_has_significant_overlap()` now requires `len(word) >= min_len` for single-token content, preventing short words like "api" or "done" from false-matching.
3. **`_check_conflict` indentation fixed** — `return` block under `if has_negation` was over-indented (4 extra spaces).

### Should-Fix (Quality/Safety)
4. **Shared pool reinforcement scoped to final results** — `read()` was reinforcing all `limit*2` BM25 results. Now only reinforces the top `limit` returned, preventing "rich get richer" for near-cutoff docs.
5. **Audit logs anonymize queries** — `read()` now runs queries through `anonymize_query()` before logging, preventing PII leaks to disk.
6. **`mark_used_cross_agent` guards unknown provenance** — Added `writer is not None` check so memories without an `agent:` tag don't get incorrectly boosted.
7. **`boost_score()` per-call cache** — Added `cache` dict so repeated query tokens don't recompute `similar_words()`.
8. **`save()` fsync for durability** — CooccurrenceIndex atomic write now calls `f.flush()` + `os.fsync()` before `tmp.replace()`.

### Nice-to-Have
9. **Dedup used IDs** — `extract_memory_references()` now uses a `seen` set to prevent duplicate IDs in output.

---

## Architecture Overview

### v3.2 — Instrumentation + Co-occurrence (Tier 1 Semantic)

**instrumentation.py** — Tracks which retrieved memories are actually used in agent responses.
- `SearchContext` dataclass: tracks request lifecycle (retrieved → used)
- `extract_memory_references()`: heuristic substring matching, no LLM calls
- `anonymize_query()`: strips emails, phones, SSNs, IPs, credit cards
- `build_usage_signal()`: structured event for opt-in aggregation

**cooccurrence.py** — Per-user word-pair frequency index for personalized semantic search.
- 5-word sliding window, stopword-filtered tokens
- PPMI scoring with pair-space marginals (R2 fix)
- `boost_score()`: additive re-ranking layer on top of BM25
- Atomic JSON persistence with fsync (R2 fix)
- Lazy pair-marginal cache, invalidated on updates

### v3.3 — Shared Multi-Agent Memory Pool

**shared.py** — Multi-agent memory with access controls, conflict detection, and cross-agent learning.
- Role-based access: read / write / admin
- Namespace isolation with shared default
- BM25 search (same SearchEngine as private memory)
- Negation-based conflict detection
- Cross-agent confidence boosting via `mark_used_cross_agent()`
- Audit trail with FileLock, PII-anonymized queries (R2 fix)

### Core Integration (core_v4.py)

- `search_with_context()`: BM25 + co-occurrence boost + instrumentation
- `mark_used()` / `mark_retrieved()` / `boost_relevance()`: usage signal pipeline
- `enable_shared_pool()` / `shared_write()` / `shared_search()` / `shared_mark_used()`: pool bridge
- Co-occurrence updated at ingest time, saved at `flush()` time

---

## Full Source Code

### antaris_memory/instrumentation.py

```python
"""
Instrumentation layer for antaris-memory v3.2.

Tracks which retrieved memories are actually used in agent responses,
enabling usage-based relevance signals, cross-agent learning (v3.3),
and high-quality crowdsourced training data (future release).

Zero dependencies. Zero overhead when disabled (callback=None).

Usage:
    from antaris_memory.instrumentation import SearchContext, extract_memory_references

    context = SearchContext(query="car maintenance", context_id="session_123")
    results = mem.search("car maintenance", instrumentation_context=context)

    # After agent generates response...
    used_ids = extract_memory_references(agent_response, context.retrieved_memory_ids)
    mem.mark_used(used_ids, context)
"""

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ─── SearchContext ────────────────────────────────────────────────────────────

@dataclass
class SearchContext:
    """Tracks a single search request through the retrieval → response cycle.

    Created at search time, updated after the agent responds. Passed to
    mark_used() to signal which memories actually appeared in the output.

    Attributes:
        request_id: Unique identifier for this search request (uuid4).
        query: The original search query string.
        retrieved_memory_ids: IDs of memories returned by search().
        used_memory_ids: IDs of memories that appeared in the agent response.
        timestamp: Unix timestamp when the search was initiated.
        context_id: Optional session/conversation identifier.
        session_id: Optional session key for multi-session tracking.
    """
    query: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retrieved_memory_ids: List[str] = field(default_factory=list)
    used_memory_ids: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    context_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize to dict for logging."""
        return {
            "request_id": self.request_id,
            "query": self.query,
            "retrieved_memory_ids": self.retrieved_memory_ids,
            "used_memory_ids": self.used_memory_ids,
            "timestamp": self.timestamp,
            "context_id": self.context_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SearchContext":
        """Deserialize from dict."""
        ctx = cls(query=data["query"])
        ctx.request_id = data.get("request_id", str(uuid.uuid4()))
        ctx.retrieved_memory_ids = data.get("retrieved_memory_ids", [])
        ctx.used_memory_ids = data.get("used_memory_ids", [])
        ctx.timestamp = data.get("timestamp", time.time())
        ctx.context_id = data.get("context_id")
        ctx.session_id = data.get("session_id")
        return ctx


# ─── Memory Reference Extraction ─────────────────────────────────────────────

def extract_memory_references(
    text: str,
    retrieved_entries: list,
    min_overlap: int = 6,
) -> List[str]:
    """Find which retrieved memories were actually referenced in the agent response.

    Uses substring matching: if a significant phrase from a memory appears in
    the output text, that memory is considered "used." No LLM calls required.

    Args:
        text: The agent's response text to check.
        retrieved_entries: List of SearchResult or MemoryEntry objects from search().
        min_overlap: Minimum number of characters of overlap to count as a match.

    Returns:
        List of memory IDs (entry.hash) that appear to be used in the text.
    """
    if not text or not retrieved_entries:
        return []

    text_lower = text.lower()
    used_ids = []
    seen = set()

    for item in retrieved_entries:
        # Support both SearchResult and MemoryEntry
        entry = getattr(item, "entry", item)
        memory_id = getattr(entry, "hash", None)
        content = getattr(entry, "content", "")

        if not memory_id or not content:
            continue

        # Strategy 1: Direct substring match (longest common phrase ≥ min_overlap)
        if _has_significant_overlap(content.lower(), text_lower, min_overlap):
            if memory_id not in seen:
                used_ids.append(memory_id)
                seen.add(memory_id)
            continue

        # Strategy 2: Key phrase extraction (first N words of memory)
        key_phrase = _extract_key_phrase(content)
        if key_phrase and key_phrase in text_lower:
            if memory_id not in seen:
                used_ids.append(memory_id)
                seen.add(memory_id)

    return used_ids


def _has_significant_overlap(source: str, target: str, min_len: int) -> bool:
    """Check if any phrase of length >= min_len from source appears in target."""
    # Sliding window over source words
    words = source.split()
    if len(words) < 2:
        # Single-word content: only count if it meets min_len to avoid noisy matches
        s = source.strip()
        return len(s) >= min_len and s in target

    # Try progressively shorter phrases (start from full, trim ends)
    for start in range(len(words)):
        for end in range(len(words), start, -1):
            phrase = " ".join(words[start:end])
            if len(phrase) >= min_len and phrase in target:
                return True
            if len(phrase) < min_len:
                break  # Remaining phrases too short

    return False


# Stopwords for key phrase extraction (module-level to avoid per-call allocation)
_KEYPHRASE_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'and',
    'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'it', 'its', 'this', 'that', 'by', 'as', 'from', 'have',
    'has', 'had', 'not', 'no', 'can', 'will', 'do', 'did',
})


def _extract_key_phrase(content: str, max_words: int = 6) -> str:
    """Extract a key phrase from memory content for matching."""
    # Remove punctuation, take first N meaningful words
    words = re.findall(r'\b\w{3,}\b', content.lower())
    filtered = [w for w in words if w not in _KEYPHRASE_STOPWORDS]
    return " ".join(filtered[:max_words])


# ─── Query Anonymization ──────────────────────────────────────────────────────

# PII patterns to strip before any aggregation reporting
_PII_PATTERNS = [
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # email
    re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),                       # phone US
    re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),             # credit card
    re.compile(r'\b(?:SSN|ssn)[\s:-]?\d{3}-\d{2}-\d{4}\b'),                # SSN with prefix
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),                                   # bare SSN (no prefix)
    re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),                 # IP address
]

def anonymize_query(query: str) -> str:
    """Strip PII patterns from a query before aggregation reporting.

    Removes emails, phone numbers, credit card numbers, SSNs, IPs.
    Returns lowercased, PII-stripped query string.

    Args:
        query: Raw query string from search().

    Returns:
        Anonymized query string safe for aggregation reporting.
    """
    result = query
    for pattern in _PII_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result.lower().strip()


# ─── Usage Signal Schema ──────────────────────────────────────────────────────

def build_usage_signal(
    context: SearchContext,
    anonymize: bool = True,
) -> Dict:
    """Build an anonymized usage signal dict for aggregation reporting.

    The signal contains only statistical data — never actual memory content.

    Args:
        context: SearchContext after mark_used() has been called.
        anonymize: Whether to anonymize the query (default True).

    Returns:
        Dict safe for opt-in aggregation reporting.
    """
    query = anonymize_query(context.query) if anonymize else context.query
    return {
        "request_id": context.request_id,
        "query_terms": query.split(),
        "retrieved_count": len(context.retrieved_memory_ids),
        "used_count": len(context.used_memory_ids),
        "use_rate": (
            len(context.used_memory_ids) / max(len(context.retrieved_memory_ids), 1)
        ),
        "timestamp": context.timestamp,
        "context_id": context.context_id,
    }
```

### antaris_memory/cooccurrence.py

```python
"""
Local co-occurrence collection for antaris-memory v3.2 — Tier 1 Semantic.

Builds a per-user word-pair frequency index from stored memories.
No data upload, zero API calls, zero dependencies.

Every time a memory is stored, word pairs within a 5-word window are
recorded. Over time this builds a personalized semantic clustering index:
  - User talks about "liver function" → index clusters "hepatic", "ALT", "enzymes"
  - Search for "liver" → results containing "hepatic" get a boost

The index is stored as a simple JSON file alongside memories.
Completely independent of BM25 — it's an additive re-ranking layer.

Usage:
    from antaris_memory.cooccurrence import CooccurrenceIndex

    idx = CooccurrenceIndex("./workspace")
    idx.load()
    idx.update_from_memory("Patient shows elevated ALT and hepatic inflammation.")
    similar = idx.similar_words("liver", top_k=10)
    # → ["hepatic", "ALT", "inflammation", "patient", ...]
    idx.save()
"""

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Stopwords to exclude from co-occurrence pairs
_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'and',
    'or', 'but', 'if', 'that', 'this', 'it', 'its', 'he', 'she', 'they',
    'we', 'you', 'i', 'me', 'my', 'our', 'their', 'what', 'which', 'who',
    'not', 'no', 'so', 'then', 'just', 'now', 'up', 'out', 'very',
    'about', 'after', 'before', 'between', 'over', 'under', 'more',
})

# Window size for co-occurrence (words within N positions are considered co-occurring)
DEFAULT_WINDOW = 5
DEFAULT_TOP_K = 20


class CooccurrenceIndex:
    """Per-user word-pair frequency index for personalized semantic search.

    Tracks which words appear near each other across all stored memories.
    Used to boost search results that contain words semantically related
    to the query terms — even when there's no exact keyword match.

    Args:
        workspace: Path to the memory workspace directory.
        window: Co-occurrence window size in words (default 5).
        filename: Index filename within workspace (default .cooccurrence.json).
    """

    def __init__(
        self,
        workspace: str,
        window: int = DEFAULT_WINDOW,
        filename: str = ".cooccurrence.json",
    ):
        self.workspace = Path(workspace)
        self.window = window
        self._path = self.workspace / filename
        # word → {co_word: count}
        self._counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # word → total occurrences (for PPMI normalization)
        self._word_counts: Dict[str, int] = defaultdict(int)
        self._total_pairs: int = 0
        self._dirty: bool = False
        # Cache for pair-marginals used in PPMI scoring (invalidated on updates)
        self._pair_marginals: Dict[str, int] = {}
        self._pair_marginals_valid: bool = False

    # ── Load / Save ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load index from disk. No-op if index doesn't exist yet."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_counts = data.get("counts", {})
            self._counts = defaultdict(lambda: defaultdict(int))
            for word, co_dict in raw_counts.items():
                self._counts[word] = defaultdict(int, co_dict)
            self._word_counts = defaultdict(int, data.get("word_counts", {}))
            self._total_pairs = data.get("total_pairs", 0)
        except (json.JSONDecodeError, KeyError):
            # Corrupted index — start fresh
            self._counts = defaultdict(lambda: defaultdict(int))
            self._word_counts = defaultdict(int)
            self._total_pairs = 0
        self._dirty = False

    def save(self) -> None:
        """Save index to disk. No-op if nothing changed."""
        if not self._dirty:
            return
        self.workspace.mkdir(parents=True, exist_ok=True)
        data = {
            "counts": {w: dict(co) for w, co in self._counts.items()},
            "word_counts": dict(self._word_counts),
            "total_pairs": self._total_pairs,
            "version": "3.2",
        }
        # Atomic write with fsync for durability
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        tmp.replace(self._path)
        self._dirty = False

    # ── Indexing ──────────────────────────────────────────────────────────────

    def update_from_memory(self, content: str) -> int:
        """Extract word pairs from memory content and update the index.

        Processes a 5-word sliding window across the content, recording
        all non-stopword word pairs. Call this whenever a memory is stored.

        Args:
            content: Memory content string to index.

        Returns:
            Number of word pairs added.
        """
        tokens = self._tokenize(content)
        pairs_added = 0

        for i, word in enumerate(tokens):
            self._word_counts[word] += 1
            # Look ahead within window
            window_end = min(i + self.window + 1, len(tokens))
            for j in range(i + 1, window_end):
                co_word = tokens[j]
                if co_word != word:
                    self._counts[word][co_word] += 1
                    self._counts[co_word][word] += 1
                    self._total_pairs += 2
                    pairs_added += 2

        if pairs_added:
            self._dirty = True
            self._pair_marginals_valid = False
        return pairs_added

    def update_from_entries(self, entries: list) -> int:
        """Rebuild index from a list of MemoryEntry objects.

        Args:
            entries: List of MemoryEntry objects.

        Returns:
            Total word pairs indexed.
        """
        total = 0
        for entry in entries:
            content = getattr(entry, "content", "")
            if content:
                total += self.update_from_memory(content)
        return total

    def _get_pair_marginal(self, word: str) -> int:
        """Return the marginal (row sum) for a word in co-occurrence-pair space."""
        if not self._pair_marginals_valid:
            self._pair_marginals = {w: sum(co.values()) for w, co in self._counts.items()}
            self._pair_marginals_valid = True
        return self._pair_marginals.get(word, 0)

    # ── Query ─────────────────────────────────────────────────────────────────

    def similar_words(self, word: str, top_k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        """Find words most similar to the given word by co-occurrence.

        Uses Positive PMI (PPMI) scoring with pair-space marginals:
        P(x,y) = c(x,y) / total_pairs
        P(x) = sum_y c(x,y) / total_pairs
        P(y) = sum_x c(x,y) / total_pairs

        This keeps joint and marginals on the same denominator.

        Args:
            word: Query word to find similar words for.
            top_k: Maximum number of results to return.

        Returns:
            List of (word, ppmi_score) tuples, sorted by score descending.
        """
        word = word.lower()
        if word not in self._counts or self._total_pairs == 0:
            return []

        co_counts = self._counts[word]
        # Compute marginals in pair space (consistent denominator for PPMI)
        row_total = self._get_pair_marginal(word)
        if row_total <= 0:
            return []
        p_word = row_total / self._total_pairs

        scored = []
        for co_word, co_count in co_counts.items():
            col_total = self._get_pair_marginal(co_word)
            if col_total <= 0 or co_count <= 0:
                continue
            p_co = col_total / self._total_pairs
            p_joint = co_count / self._total_pairs

            if p_joint <= 0 or p_word <= 0 or p_co <= 0:
                continue

            # PMI = log(P(x,y) / (P(x) * P(y)))
            pmi = math.log2(p_joint / (p_word * p_co))
            # PPMI: clip negative values (reduces noise from rare pairs)
            ppmi = max(0.0, pmi)

            if ppmi > 0:
                scored.append((co_word, ppmi))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def boost_score(
        self,
        query: str,
        entry_content: str,
        boost_factor: float = 0.3,
    ) -> float:
        """Compute a semantic boost score for an entry given a query.

        Checks whether any query terms co-occur strongly with terms in the
        entry content. Returns an additive boost (0.0 to boost_factor).

        Args:
            query: The search query string.
            entry_content: The memory entry content to check.
            boost_factor: Maximum boost to apply (default 0.3 = 30% extra).

        Returns:
            Float boost value in [0.0, boost_factor].
        """
        query_tokens = self._tokenize(query)
        entry_tokens = set(self._tokenize(entry_content))

        if not query_tokens or not entry_tokens:
            return 0.0

        total_boost = 0.0
        hits = 0
        cache = {}

        for q_token in query_tokens:
            if q_token not in cache:
                cache[q_token] = dict(self.similar_words(q_token, top_k=20))
            similar = cache[q_token]
            for e_token in entry_tokens:
                if e_token in similar:
                    # Scale similarity score to [0, 1] range (PPMI can be large)
                    raw_score = similar[e_token]
                    scaled = min(raw_score / 5.0, 1.0)  # Cap at PPMI=5
                    total_boost += scaled
                    hits += 1

        if hits == 0:
            return 0.0

        # Average boost, scaled to boost_factor
        avg = total_boost / hits
        return min(avg * boost_factor, boost_factor)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Number of unique words in the index."""
        return len(self._counts)

    @property
    def total_pairs(self) -> int:
        """Total co-occurrence pairs recorded."""
        return self._total_pairs

    def stats(self) -> Dict:
        """Return index statistics."""
        return {
            "vocab_size": self.vocab_size,
            "total_pairs": self._total_pairs,
            "top_words": sorted(
                self._word_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Extract meaningful tokens from text, excluding stopwords."""
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [t for t in tokens if t not in _STOPWORDS]
```

### antaris_memory/shared.py

```python
"""
Shared Memory — Multi-agent memory pools with access controls.

Enables multiple agents to share a memory space with:
- Role-based access (read, write, admin)
- Per-agent namespaces with optional sharing
- Conflict resolution when agents disagree
- Knowledge propagation between agents
- Audit trail of who wrote/modified what

All data stored as JSON files. Zero dependencies.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .entry import MemoryEntry
from .decay import DecayEngine
from .search import SearchEngine
from .sentiment import SentimentTagger
from .utils import atomic_write_json


class AgentPermission:
    """Access control for a single agent."""

    __slots__ = ("agent_id", "role", "namespaces", "created")

    ROLES = ("read", "write", "admin")

    def __init__(self, agent_id: str, role: str = "write",
                 namespaces: List[str] = None):
        if role not in self.ROLES:
            raise ValueError(f"Role must be one of {self.ROLES}")
        self.agent_id = agent_id
        self.role = role
        self.namespaces = namespaces or ["shared"]
        self.created = datetime.now().isoformat()

    def can_read(self) -> bool:
        return self.role in ("read", "write", "admin")

    def can_write(self) -> bool:
        return self.role in ("write", "admin")

    def can_admin(self) -> bool:
        return self.role == "admin"

    def can_access_namespace(self, namespace: str) -> bool:
        if self.role == "admin":
            return True
        return namespace in self.namespaces or namespace == "shared"

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "namespaces": self.namespaces,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AgentPermission":
        perm = cls(d["agent_id"], d.get("role", "write"),
                   d.get("namespaces", ["shared"]))
        perm.created = d.get("created", perm.created)
        return perm


class SharedMemoryPool:
    """Multi-agent shared memory with access controls and conflict resolution."""

    def __init__(self, pool_dir: str, pool_name: str = "default"):
        self.pool_dir = os.path.abspath(pool_dir)
        self.pool_name = pool_name
        self.state_path = os.path.join(self.pool_dir, f"pool_{pool_name}.json")
        self.audit_path = os.path.join(self.pool_dir, f"pool_{pool_name}_audit.json")

        # Internal state
        self.memories: List[MemoryEntry] = []
        self.permissions: Dict[str, AgentPermission] = {}
        self.conflicts: List[Dict] = []
        self._hashes: set = set()
        self._memory_by_hash: Dict[str, MemoryEntry] = {}  # O(1) lookup by hash
        self._decay = DecayEngine(half_life=7.0)
        self._sentiment = SentimentTagger()
        self._search_engine = SearchEngine()

        os.makedirs(self.pool_dir, exist_ok=True)

    # ── agent management ────────────────────────────────────────────────

    def register_agent(self, agent_id: str, role: str = "write",
                       namespaces: List[str] = None) -> AgentPermission:
        """Register an agent with the pool."""
        perm = AgentPermission(agent_id, role, namespaces)
        self.permissions[agent_id] = perm
        self._audit("register", agent_id, f"role={role}")
        return perm

    def remove_agent(self, agent_id: str, requester: str = None) -> bool:
        """Remove an agent from the pool. Requires admin."""
        if requester and not self._check_admin(requester):
            return False
        if agent_id in self.permissions:
            del self.permissions[agent_id]
            self._audit("remove_agent", requester or "system", f"removed={agent_id}")
            return True
        return False

    def list_agents(self) -> List[Dict]:
        """List all registered agents and their roles."""
        return [p.to_dict() for p in self.permissions.values()]

    # ── memory operations ───────────────────────────────────────────────

    def write(self, agent_id: str, content: str, namespace: str = "shared",
              category: str = "general", metadata: Dict = None) -> Optional[MemoryEntry]:
        """Write a memory to the pool. Checks permissions."""
        perm = self.permissions.get(agent_id)
        if not perm or not perm.can_write():
            return None
        if not perm.can_access_namespace(namespace):
            return None

        entry = MemoryEntry(content, source=f"agent:{agent_id}", category=category)
        entry.sentiment = self._sentiment.analyze(content)
        entry.tags.append(f"ns:{namespace}")
        entry.tags.append(f"agent:{agent_id}")
        if metadata:
            for k, v in metadata.items():
                entry.tags.append(f"{k}:{v}")

        conflict = self._check_conflict(entry, agent_id)
        if conflict:
            self.conflicts.append(conflict)

        if entry.hash not in self._hashes:
            self.memories.append(entry)
            self._hashes.add(entry.hash)
            self._memory_by_hash[entry.hash] = entry
            self._audit("write", agent_id, f"ns={namespace} hash={entry.hash}")
            return entry
        return None

    def read(self, agent_id: str, query: str, namespace: str = None,
             limit: int = 20) -> List[MemoryEntry]:
        """Search shared memories using BM25. Respects namespace permissions."""
        perm = self.permissions.get(agent_id)
        if not perm or not perm.can_read():
            return []

        accessible = [
            m for m in self.memories
            if perm.can_access_namespace(self._get_namespace(m))
            and (not namespace or self._get_namespace(m) == namespace)
        ]
        if not accessible:
            return []

        bm25_results = self._search_engine.search(query, accessible, limit=limit * 2)
        scored = []
        for r in bm25_results:
            decay_score = self._decay.score(r.entry)
            final_score = r.relevance * (0.5 + decay_score)
            if final_score > 0:
                scored.append((r.entry, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [m for m, _ in scored[:limit]]

        # Reinforce only the final returned results (avoid reinforcing near-cutoff docs)
        for m in top:
            self._decay.reinforce(m)

        # Avoid logging raw PII in queries
        try:
            from .instrumentation import anonymize_query as _anonymize
            qlog = _anonymize(query)
        except Exception:
            qlog = query
        self._audit("read", agent_id, f"query={qlog[:50]} results={len(top)}")
        return top

    def propagate(self, from_agent: str, to_namespace: str,
                  query: str = None, limit: int = 10) -> int:
        """Propagate an agent's memories to another namespace."""
        perm = self.permissions.get(from_agent)
        if not perm or not perm.can_write():
            return 0

        agent_memories = [m for m in self.memories if f"agent:{from_agent}" in m.tags]

        if query:
            bm25 = SearchEngine()
            bm25_results = bm25.search(query, agent_memories, limit=limit)
            agent_memories = [r.entry for r in bm25_results]
        else:
            agent_memories = agent_memories[:limit]

        count = 0
        for m in agent_memories:
            new_entry = MemoryEntry(m.content, source=f"propagated:{from_agent}",
                                    category=m.category)
            new_entry.tags = list(m.tags)
            new_entry.tags = [t for t in new_entry.tags if not t.startswith("ns:")]
            new_entry.tags.append(f"ns:{to_namespace}")
            new_entry.tags.append(f"propagated_from:{from_agent}")
            new_entry.sentiment = dict(m.sentiment)
            new_entry.confidence = m.confidence

            if new_entry.hash not in self._hashes:
                self.memories.append(new_entry)
                self._hashes.add(new_entry.hash)
                self._memory_by_hash[new_entry.hash] = new_entry
                count += 1

        self._audit("propagate", from_agent, f"to={to_namespace} count={count}")
        return count

    # ── conflict resolution ─────────────────────────────────────────────

    def get_conflicts(self) -> List[Dict]:
        return [c for c in self.conflicts if not c.get("resolved")]

    def resolve_conflict(self, conflict_index: int, resolution: str,
                         resolver: str = "system") -> bool:
        if conflict_index >= len(self.conflicts):
            return False
        conflict = self.conflicts[conflict_index]
        conflict["resolved"] = True
        conflict["resolution"] = resolution
        conflict["resolved_by"] = resolver
        conflict["resolved_at"] = datetime.now().isoformat()
        self._audit("resolve_conflict", resolver, f"index={conflict_index} resolution={resolution}")
        return True

    # ── persistence ─────────────────────────────────────────────────────

    def save(self) -> str:
        from . import __version__ as _pkg_version
        data = {
            "schema_version": "0.3.0",
            "package_version": _pkg_version,
            "pool_name": self.pool_name,
            "saved_at": datetime.now().isoformat(),
            "agent_count": len(self.permissions),
            "memory_count": len(self.memories),
            "conflict_count": len(self.conflicts),
            "permissions": {k: v.to_dict() for k, v in self.permissions.items()},
            "memories": [m.to_dict() for m in self.memories],
            "conflicts": self.conflicts,
        }
        atomic_write_json(self.state_path, data)
        return self.state_path

    def load(self) -> int:
        if not os.path.exists(self.state_path):
            return 0
        with open(self.state_path) as f:
            data = json.load(f)
        self.pool_name = data.get("pool_name", self.pool_name)
        self.permissions = {k: AgentPermission.from_dict(v)
                            for k, v in data.get("permissions", {}).items()}
        self.memories = [MemoryEntry.from_dict(d) for d in data.get("memories", [])]
        self._hashes = {m.hash for m in self.memories}
        self._memory_by_hash = {m.hash: m for m in self.memories}
        self.conflicts = data.get("conflicts", [])
        return len(self.memories)

    # ── stats ───────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        namespaces = {}
        agents = {}
        for m in self.memories:
            ns = self._get_namespace(m)
            namespaces[ns] = namespaces.get(ns, 0) + 1
            agent = self._get_agent(m)
            if agent:
                agents[agent] = agents.get(agent, 0) + 1
        return {
            "pool_name": self.pool_name,
            "total_memories": len(self.memories),
            "registered_agents": len(self.permissions),
            "namespaces": namespaces,
            "memories_by_agent": agents,
            "unresolved_conflicts": len(self.get_conflicts()),
            "total_conflicts": len(self.conflicts),
        }

    # ── cross-agent instrumentation signals ────────────────────────────

    def mark_used_cross_agent(self, requesting_agent: str, memory_id: str,
                               boost_factor: float = 1.3) -> bool:
        """Signal that a retrieved memory was actually used in an agent response.

        If the memory was written by a DIFFERENT agent (cross-agent use),
        boost its confidence. Only boosts when writer is known and different.
        """
        m = self._memory_by_hash.get(memory_id)
        if m is None:
            return False
        writer = self._get_agent(m)
        is_cross_agent = (writer is not None and writer != requesting_agent)
        self._decay.reinforce(m)
        if is_cross_agent:
            m.confidence = min(1.0, m.confidence * boost_factor)
            self._audit("cross_agent_boost", requesting_agent,
                        f"memory={memory_id[:16]} writer={writer} confidence={m.confidence:.3f}")
        else:
            self._audit("self_reinforce", requesting_agent, f"memory={memory_id[:16]}")
        return True

    def mark_used_batch(self, requesting_agent: str, memory_ids: List[str],
                        boost_factor: float = 1.3) -> int:
        return sum(self.mark_used_cross_agent(requesting_agent, mid, boost_factor)
                   for mid in memory_ids)

    def search_with_context(self, agent_id: str, query: str,
                            instrumentation_context=None, limit: int = 20,
                            namespace: str = None) -> Tuple:
        results = self.read(agent_id, query, namespace, limit)
        if instrumentation_context is not None:
            instrumentation_context.retrieved_memory_ids = [m.hash for m in results]
        return (results, instrumentation_context)

    # ── private ─────────────────────────────────────────────────────────

    def _check_admin(self, agent_id: str) -> bool:
        perm = self.permissions.get(agent_id)
        return perm and perm.can_admin()

    def _get_namespace(self, entry: MemoryEntry) -> str:
        for tag in entry.tags:
            if tag.startswith("ns:"):
                return tag[3:]
        return "shared"

    def _get_agent(self, entry: MemoryEntry) -> Optional[str]:
        for tag in entry.tags:
            if tag.startswith("agent:"):
                return tag[6:]
        return None

    def _check_conflict(self, new_entry: MemoryEntry, agent_id: str) -> Optional[Dict]:
        import re
        new_words = set(re.findall(r"\w{4,}", new_entry.content.lower()))
        for existing in self.memories:
            existing_agent = self._get_agent(existing)
            if existing_agent == agent_id:
                continue
            existing_words = set(re.findall(r"\w{4,}", existing.content.lower()))
            overlap = len(new_words & existing_words)
            if overlap < 3:
                continue
            new_lower = new_entry.content.lower()
            exist_lower = existing.content.lower()
            negation_signals = [
                ("don't ", "do "), ("won't ", "will "),
                ("can't ", "can "), ("shouldn't ", "should "),
                ("isn't ", "is "), ("aren't ", "are "),
                ("reject", "accept"), ("disagree", "agree"),
                ("false", "true"), ("wrong", "right"),
                ("failed", "succeeded"), ("no ", "yes "),
            ]
            has_negation = False
            for neg, pos in negation_signals:
                if ((neg in new_lower and pos in exist_lower) or
                        (pos in new_lower and neg in exist_lower)):
                    has_negation = True
                    break
            if not has_negation and overlap >= 3:
                new_has_not = (" not " in new_lower or new_lower.startswith("not "))
                exist_has_not = (" not " in exist_lower or exist_lower.startswith("not "))
                if new_has_not != exist_has_not:
                    has_negation = True
            if has_negation:
                return {
                    "type": "contradiction",
                    "new_memory": new_entry.hash,
                    "new_agent": agent_id,
                    "new_content": new_entry.content[:200],
                    "existing_memory": existing.hash,
                    "existing_agent": existing_agent,
                    "existing_content": existing.content[:200],
                    "overlap_words": list(new_words & existing_words)[:10],
                    "detected_at": datetime.now().isoformat(),
                    "resolved": False,
                }
        return None

    def _audit(self, action: str, agent_id: str, detail: str = "") -> None:
        from .locking import FileLock
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "agent_id": agent_id,
            "detail": detail,
        }
        with FileLock(self.audit_path, timeout=10.0):
            log = []
            if os.path.exists(self.audit_path):
                try:
                    with open(self.audit_path) as f:
                        log = json.load(f)
                except (json.JSONDecodeError, IOError):
                    log = []
            log.append(entry)
            atomic_write_json(self.audit_path, log[-500:], lock=False)
```

### antaris_memory/core_v4.py — v3.2/v3.3 Integration Points

(Full file is 1644 lines. Key integration methods shown below — full file available on request.)

**search_with_context()** — BM25 + co-occurrence boost + instrumentation:
```python
def search_with_context(self, query, limit=20, instrumentation_context=None,
                        cooccurrence_boost=True, **kwargs):
    from .instrumentation import SearchContext
    ctx = instrumentation_context or SearchContext(query=query)
    fetch_limit = limit * 2 if cooccurrence_boost else limit
    results = self.search(query, limit=fetch_limit, explain=True, **kwargs)
    if cooccurrence_boost and self._cooccurrence.vocab_size > 0:
        for r in results:
            boost = self._cooccurrence.boost_score(query, r.entry.content)
            r.score += boost
        results.sort(key=lambda r: r.score, reverse=True)
    results = results[:limit]
    retrieved_ids = [r.entry.hash for r in results]
    self.mark_retrieved(retrieved_ids, ctx)
    if kwargs.get("explain"):
        return results, ctx
    entries = []
    for r in results:
        entry_copy = copy.copy(r.entry)
        entry_copy.confidence = r.relevance
        entries.append(entry_copy)
    return entries, ctx
```

**Shared pool bridge methods:**
- `enable_shared_pool(pool_dir, pool_name, agent_id, role, load_existing)` → SharedMemoryPool
- `shared_write(content, namespace, category, metadata)` → MemoryEntry
- `shared_search(query, namespace, limit, instrumentation_context)` → List[MemoryEntry]
- `shared_mark_used(memory_ids, boost_factor)` → int

**Co-occurrence update at ingest time (deferred save to flush):**
```python
def _update_cooccurrence(self, content: str) -> None:
    self._cooccurrence.update_from_memory(content)
```

---

## Test Summary

**381 tests passing in 2.66s**

| Test File | Tests | Focus |
|-----------|-------|-------|
| test_instrumentation.py | 28 | SearchContext, extract_memory_references, anonymize_query, integration |
| test_cooccurrence.py | 45 | PPMI accuracy, vocab divergence, scale/stress, integration |
| test_shared.py | 25 | Permissions, namespace isolation, conflict detection, cross-agent signals |
| test_core_shared_integration.py | 12 | MemorySystemV4 ↔ SharedMemoryPool bridge |
| (existing tests) | 271 | Core memory, search, decay, sharding, WAL, etc. |

---

## Design Principles

1. **Zero new dependencies** — stdlib only (json, math, re, os, collections)
2. **Zero overhead when disabled** — instrumentation callbacks are None by default
3. **Opt-in aggregation** — `opt_in_to_aggregation` must be explicitly True before signals are written
4. **No LLM calls** — `extract_memory_references` uses pure string matching
5. **File-based persistence** — JSON files, atomic writes, no external DB
6. **Backward compatible** — existing `search()` unchanged, `search_with_context()` is additive
