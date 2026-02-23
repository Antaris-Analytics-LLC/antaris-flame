"""
Search engine for antaris-memory v1.0.

Provides ranked search with BM25-inspired scoring, combining:
- Term frequency (how often query terms appear in the memory)
- Inverse document frequency (rare terms score higher)
- Field boosting (tags, source, category get extra weight)
- Decay weighting (recent/accessed memories score higher)
- Length normalization (don't penalize short memories)

All deterministic, zero dependencies.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SearchResult:
    """A search result with relevance score and explanation."""
    entry: object  # MemoryEntry
    score: float
    relevance: float  # 0.0-1.0 normalized
    matched_terms: List[str]
    explanation: str
    
    @property
    def content(self):
        return self.entry.content
    
    @property
    def source(self):
        return self.entry.source
    
    @property
    def confidence(self):
        return self.relevance
    
    @property
    def category(self):
        return self.entry.category
    
    def __repr__(self):
        return f"<SearchResult score={self.score:.3f} relevance={self.relevance:.2f} '{self.content[:50]}...'>"


class SearchEngine:
    """BM25-inspired search engine for memory entries.
    
    Computes relevance using term frequency, inverse document frequency,
    field boosting, and decay weighting. All operations are deterministic
    and use only the Python standard library.
    
    Args:
        k1: Term frequency saturation parameter (default 1.5, higher = more weight on frequency)
        b: Length normalization parameter (default 0.75, 0 = no normalization, 1 = full)
    """
    
    # Stopwords to exclude from scoring (common English words)
    STOPWORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'don', 'now', 'and', 'but', 'or', 'if', 'while', 'that', 'this',
        'it', 'its', 'he', 'she', 'they', 'them', 'his', 'her', 'their',
        'what', 'which', 'who', 'whom', 'these', 'those', 'am', 'about',
        'up', 'down', 'we', 'our', 'you', 'your', 'my', 'me', 'i',
    })
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._idf_cache: Dict[str, float] = {}
        self._avg_doc_len: float = 0
        self._doc_count: int = 0
        self._doc_freqs: Counter = Counter()  # term → number of docs containing it
        self._corpus_version: int = 0
        self._indexed_version: int = -1
        # Content fingerprint: lightweight hash of all memory content at index time.
        # Detects in-place content mutations that don't change the memory count
        # (e.g. tag updates, content edits) — fixes search index drift (Gemini review).
        self._content_fingerprint: int = 0

    @staticmethod
    def _compute_fingerprint(memories: list) -> int:
        """XOR hash of all memory content hashes — cheap O(n) mutation detector."""
        fp = 0
        for m in memories:
            fp ^= hash(m.content)
        return fp

    def build_index(self, memories: list) -> None:
        """Build IDF statistics from the memory corpus.
        
        Call this after loading/ingesting memories to enable proper ranking.
        """
        self._indexed_version = self._corpus_version
        self._content_fingerprint = self._compute_fingerprint(memories)
        self._doc_count = len(memories)
        self._doc_freqs.clear()
        self._idf_cache.clear()
        
        total_len = 0
        for mem in memories:
            tokens = self._tokenize(mem.content)
            total_len += len(tokens)
            # Count unique terms per document
            for term in set(tokens):
                self._doc_freqs[term] += 1
        
        self._avg_doc_len = total_len / max(self._doc_count, 1)
        
        # Pre-compute IDF for all terms
        for term, df in self._doc_freqs.items():
            # BM25 IDF formula with smoothing
            self._idf_cache[term] = math.log(
                (self._doc_count - df + 0.5) / (df + 0.5) + 1.0
            )
    
    def search(self, query: str, memories: list, limit: int = 20,
               category: str = None, min_score: float = 0.01,
               decay_fn=None) -> List[SearchResult]:
        """Search memories with BM25-inspired ranking.
        
        Args:
            query: Search query string
            memories: List of MemoryEntry objects to search
            limit: Maximum results to return
            category: Filter by category (optional)
            min_score: Minimum relevance score threshold
            decay_fn: Optional decay scoring function (entry → float)
            
        Returns:
            List of SearchResult objects, sorted by relevance descending.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Rebuild index if stale: version mismatch, count change, or content mutation.
        # Content fingerprint catches in-place edits that don't change memory count.
        current_fp = self._compute_fingerprint(memories)
        if (self._indexed_version != self._corpus_version
                or self._doc_count != len(memories)
                or self._content_fingerprint != current_fp):
            self.build_index(memories)
        
        results = []
        max_score = 0.0
        
        for mem in memories:
            if category and mem.category != category:
                continue
            
            score, matched = self._score_entry(mem, query_tokens, query.lower())
            
            if score <= 0:
                continue
            
            # Apply decay weighting if available
            if decay_fn:
                decay_score = decay_fn(mem)
                score *= (0.3 + 0.7 * decay_score)  # Decay modulates 30-100% of score
            
            max_score = max(max_score, score)
            results.append((mem, score, matched))
        
        # Normalize scores to 0-1 range
        if max_score > 0:
            normalized = []
            for mem, score, matched in results:
                relevance = score / max_score
                if relevance < min_score:
                    continue
                
                explanation = self._explain(matched, score, relevance)
                normalized.append(SearchResult(
                    entry=mem,
                    score=score,
                    relevance=round(relevance, 4),
                    matched_terms=matched,
                    explanation=explanation,
                ))
            results = normalized
        else:
            return []
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
    
    def _score_entry(self, mem, query_tokens: List[str], query_lower: str) -> Tuple[float, List[str]]:
        """Score a single memory against query tokens."""
        content_lower = mem.content.lower()
        content_tokens = self._tokenize(mem.content)
        doc_len = len(content_tokens)
        
        # Count term frequencies in this document
        tf_counter = Counter(content_tokens)
        
        score = 0.0
        matched = []
        
        # BM25 scoring per query term
        for term in query_tokens:
            tf = tf_counter.get(term, 0)
            if tf == 0:
                continue
            
            matched.append(term)
            idf = self._idf_cache.get(term, 1.0)
            
            # BM25 TF component with length normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1))
            )
            
            score += idf * tf_norm
        
        # Exact phrase bonus (query tokens appear consecutively in content tokens)
        if len(query_tokens) > 1:
            content_token_str = " ".join(content_tokens)
            query_token_str = " ".join(query_tokens)
            if query_token_str in content_token_str:
                score *= 1.5
        
        # Field boosting: check tags (at most one 1.2x boost per entry)
        if hasattr(mem, 'tags') and mem.tags:
            tag_text = " ".join(mem.tags).lower()
            tag_hit = False
            for term in query_tokens:
                if term in tag_text:
                    if not tag_hit:
                        score *= 1.2
                        tag_hit = True
                    if term not in matched:
                        matched.append(f"tag:{term}")
        
        # Field boosting: check source (at most one 1.1x boost per entry)
        if hasattr(mem, 'source') and mem.source:
            source_lower = mem.source.lower()
            for term in query_tokens:
                if term in source_lower:
                    score *= 1.1
                    break
        
        return score, matched
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase terms, filtering stopwords."""
        tokens = re.findall(r'\w{2,}', text.lower())
        return [t for t in tokens if t not in self.STOPWORDS and not (t.isdigit() and len(t) < 3)]
    
    def _explain(self, matched: List[str], score: float, relevance: float) -> str:
        """Generate a human-readable explanation of the score."""
        parts = [f"matched: {', '.join(matched)}"]
        parts.append(f"raw={score:.3f}")
        parts.append(f"relevance={relevance:.2f}")
        return " | ".join(parts)
    
    def mark_dirty(self) -> None:
        """Mark the corpus as changed. Next search() will rebuild the index."""
        self._corpus_version += 1
    
    def reindex(self, memories: list) -> None:
        """Force a full index rebuild."""
        self._corpus_version += 1
        self.build_index(memories)
    
    def stats(self) -> Dict:
        """Return index statistics."""
        return {
            "doc_count": self._doc_count,
            "avg_doc_len": round(self._avg_doc_len, 1),
            "vocab_size": len(self._doc_freqs),
            "top_terms": self._doc_freqs.most_common(20),
        }
