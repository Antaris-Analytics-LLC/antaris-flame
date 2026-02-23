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

import heapq
import json
import math
import os
import re
import threading
import time
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
        vocab_cap: int = 50_000,
    ):
        self.workspace = Path(workspace)
        self.window = window
        self.vocab_cap = vocab_cap
        self._path = self.workspace / filename
        # word → {co_word: count}
        self._counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # word → total occurrences (for PPMI normalization)
        self._word_counts: Dict[str, int] = defaultdict(int)
        # word → last access time (for LRU eviction)
        self._access_times: Dict[str, float] = {}
        self._total_pairs: int = 0
        self._dirty: bool = False
        self._lock = threading.Lock()
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
            self._access_times = data.get("access_times", {})
            # Migration: existing words without access times get a fair baseline
            if self._counts and not self._access_times:
                now = time.time()
                for word in self._counts:
                    self._access_times[word] = now
            self._total_pairs = data.get("total_pairs", 0)
        except (json.JSONDecodeError, KeyError):
            # Corrupted index — start fresh
            self._counts = defaultdict(lambda: defaultdict(int))
            self._word_counts = defaultdict(int)
            self._access_times = {}
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
            "access_times": dict(self._access_times),
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

    def _evict_lru(self) -> int:
        """Evict least-recently-used words when vocab exceeds cap.
        
        Removes the oldest 10% of vocabulary by access time.
        Thread-safe: caller must hold self._lock or call from a locked context.
        Returns number of words evicted.
        """
        if self.vocab_size <= self.vocab_cap:
            return 0
            
        # Calculate how many words to evict (10% of current vocab)
        evict_count = max(1, int(self.vocab_size * 0.1))
        
        # Get words sorted by access time (oldest first) using heapq O(N log k) vs O(N log N)
        # Words without access_times get assigned time 0.0 (oldest)
        # Tuple order is (time, word) so heapq sorts on time
        words_by_access = [
            (self._access_times.get(word, 0.0), word)
            for word in self._counts.keys()
        ]
        words_to_evict = [word for _, word in heapq.nsmallest(evict_count, words_by_access)]
        
        # Evict the oldest words
        evicted = 0
        
        for word in words_to_evict:
            if word in self._counts:
                # Remove all co-occurrence pairs involving this word
                for co_word in list(self._counts[word].keys()):
                    if co_word in self._counts and word in self._counts[co_word]:
                        del self._counts[co_word][word]
                
                # Remove the word's own co-occurrence dict
                del self._counts[word]
                evicted += 1
            
            # Remove from word counts and access times
            if word in self._word_counts:
                del self._word_counts[word]
            if word in self._access_times:
                del self._access_times[word]
        
        if evicted > 0:
            self._dirty = True
            self._pair_marginals_valid = False
            
        return evicted

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
        current_time = time.time()

        with self._lock:
            for i, word in enumerate(tokens):
                self._word_counts[word] += 1
                self._access_times[word] = current_time
                
                window_end = min(i + self.window + 1, len(tokens))
                for j in range(i + 1, window_end):
                    co_word = tokens[j]
                    if co_word != word:
                        self._access_times[co_word] = current_time
                        self._counts[word][co_word] += 1
                        self._counts[co_word][word] += 1
                        self._total_pairs += 2
                        pairs_added += 2

            if pairs_added:
                self._dirty = True
                self._pair_marginals_valid = False
                
                while self.vocab_size > self.vocab_cap:
                    evicted = self._evict_lru()
                    if evicted == 0:
                        break
                    
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

        Uses Positive PMI (PPMI) scoring: penalizes common word pairs,
        rewards rare but meaningful co-occurrences.

        Args:
            word: Query word to find similar words for.
            top_k: Maximum number of results to return.

        Returns:
            List of (word, ppmi_score) tuples, sorted by score descending.
        """
        word = word.lower()
        if word not in self._counts or self._total_pairs == 0:
            return []

        with self._lock:
            # Update access time for queried word (thread-safe)
            self._access_times[word] = time.time()
            self._dirty = True

            co_counts = dict(self._counts[word])  # snapshot under lock
            total_pairs = self._total_pairs
            # Compute marginals in pair space (consistent denominator for PPMI)
            row_total = self._get_pair_marginal(word)
            if row_total <= 0:
                return []
            p_word = row_total / total_pairs

        scored = []
        for co_word, co_count in co_counts.items():
            col_total = self._get_pair_marginal(co_word)
            if col_total <= 0 or co_count <= 0:
                continue
            p_co = col_total / total_pairs
            p_joint = co_count / total_pairs

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

        # Update access times for query tokens (thread-safe)
        current_time = time.time()
        with self._lock:
            for token in query_tokens:
                if token in self._counts:
                    self._access_times[token] = current_time
                    self._dirty = True

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

    def detect_paraphrase(self, text_a: str, text_b: str, threshold: float = 0.6) -> Tuple[bool, float]:
        """Detect if two texts are paraphrases using co-occurrence neighbor overlap.
        
        Strategy: For each content word in text_a, get its top-K similar words.
        Check how many of those appear in text_b (and vice versa).
        High bidirectional overlap = likely paraphrase.
        
        Args:
            text_a: First text to compare.
            text_b: Second text to compare.
            threshold: Similarity threshold (0.6 = 60% overlap required).
            
        Returns:
            Tuple of (is_paraphrase: bool, similarity_score: float)
        """
        tokens_a = set(self._tokenize(text_a))
        tokens_b = set(self._tokenize(text_b))
        
        if not tokens_a or not tokens_b:
            return False, 0.0
            
        # Direct overlap baseline (Jaccard similarity — prevents short-text inflation)
        direct_overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        if direct_overlap >= threshold:
            return True, direct_overlap
            
        # Semantic overlap via co-occurrence neighbors
        neighbors_a = set()
        neighbors_b = set()
        
        # Get similar words for each token in text_a
        for token in tokens_a:
            similar = self.similar_words(token, top_k=10)
            neighbors_a.update(word for word, score in similar if score > 0.5)
            
        # Get similar words for each token in text_b  
        for token in tokens_b:
            similar = self.similar_words(token, top_k=10)
            neighbors_b.update(word for word, score in similar if score > 0.5)
            
        # Check bidirectional semantic overlap
        # How many neighbors of A appear in B's tokens or neighbors?
        a_to_b_overlap = len(neighbors_a & (tokens_b | neighbors_b))
        b_to_a_overlap = len(neighbors_b & (tokens_a | neighbors_a))
        
        if not neighbors_a or not neighbors_b:
            return direct_overlap >= threshold, direct_overlap
            
        # Bidirectional similarity score
        semantic_sim_a = a_to_b_overlap / len(neighbors_a) if neighbors_a else 0.0
        semantic_sim_b = b_to_a_overlap / len(neighbors_b) if neighbors_b else 0.0
        semantic_overlap = (semantic_sim_a + semantic_sim_b) / 2
        
        # Combined score: blend direct and semantic overlap
        combined_score = 0.4 * direct_overlap + 0.6 * semantic_overlap
        
        is_paraphrase = combined_score >= threshold
        return is_paraphrase, combined_score

    def find_paraphrases(self, query: str, entries: list, threshold: float = 0.6) -> List[Tuple[object, float]]:
        """Find memory entries that are paraphrases of the query.
        
        Args:
            query: Query text to find paraphrases for.
            entries: List of MemoryEntry objects to search.
            threshold: Paraphrase similarity threshold.
            
        Returns:
            List of (MemoryEntry, similarity_score) tuples for detected paraphrases.
        """
        paraphrases = []
        
        for entry in entries:
            content = getattr(entry, 'content', '')
            if content:
                is_paraphrase, similarity = self.detect_paraphrase(query, content, threshold)
                if is_paraphrase:
                    paraphrases.append((entry, similarity))
                    
        # Sort by similarity descending
        paraphrases.sort(key=lambda x: x[1], reverse=True)
        return paraphrases

    def inject_synthetic_pair(self, word_a: str, word_b: str, weight: float = 1.0) -> None:
        """Inject a synthetic co-occurrence pair from an external signal (e.g. semantic engine).

        Adds ``weight`` counts to both directions of the pair. Thread-safe.
        Obeys vocab_cap — words are only added if they're already in the index
        or if there's room under the cap.

        Args:
            word_a: First word/token in the pair.
            word_b: Second word/token in the pair.
            weight: Number of synthetic co-occurrences to add (default 1.0).
        """
        with self._lock:
            # Only inject if at least one word is already in the index.
            # This prevents polluting the vocab with arbitrary external tokens.
            if word_a not in self._counts and word_b not in self._counts:
                return

            current_time = time.time()

            # Add word_a if not present (only if under cap)
            if word_a not in self._counts:
                if self.vocab_size >= self.vocab_cap:
                    return  # Cap reached — skip entirely
                self._counts[word_a] = defaultdict(int)
                self._word_counts[word_a] = 0

            # Add word_b if not present (only if under cap)
            if word_b not in self._counts:
                if self.vocab_size >= self.vocab_cap:
                    return  # Cap reached — skip entirely
                self._counts[word_b] = defaultdict(int)
                self._word_counts[word_b] = 0

            # Inject co-occurrence in both directions
            self._counts[word_a][word_b] = self._counts[word_a].get(word_b, 0) + weight
            self._counts[word_b][word_a] = self._counts[word_b].get(word_a, 0) + weight

            # Update totals and word marginals
            self._total_pairs += weight * 2
            self._word_counts[word_a] = self._word_counts.get(word_a, 0) + weight
            self._word_counts[word_b] = self._word_counts.get(word_b, 0) + weight

            # Refresh access timestamps
            self._access_times[word_a] = current_time
            self._access_times[word_b] = current_time

            self._dirty = True
            self._pair_marginals_valid = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Extract meaningful tokens from text, excluding stopwords.

        Captures alphanumeric tokens (≥2 chars) so that domain-specific
        codes like "404", "gpt4", "100mg", and version strings are indexed.
        """
        tokens = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())
        return [t for t in tokens if t not in _STOPWORDS]
