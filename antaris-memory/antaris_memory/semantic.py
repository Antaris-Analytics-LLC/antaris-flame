"""
Semantic search engine for antaris-memory v3.9.0 — Optional Tier 2 feature.

Provides semantic embeddings using sentence-transformers/all-MiniLM-L6-v2.
Only loads when configured with semantic_backend='transformer'.
Zero mandatory dependencies — this is pro tier only.

Features:
- Lazy loading: model only loads when needed
- Local storage: vectors stored alongside memories in .vectors.json
- Two-stage pipeline: BM25 first pass → semantic re-rank
- Graceful fallback if sentence-transformers not available
- 384-dimensional embeddings from all-MiniLM-L6-v2 (22MB model)

Usage:
    from antaris_memory.semantic import SemanticEngine

    engine = SemanticEngine("./workspace")
    vector = engine.embed("Patient shows elevated liver enzymes")
    engine.store_vector("memory_123", vector)
    
    # Re-rank search results
    reranked = engine.rerank(query, bm25_candidates, top_k=10)
"""

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from antaris_memory.search import SearchResult

# Optional dependency - only imported when needed
# Per-model cache: model_name → SentenceTransformer instance
_MODEL_CACHE: Dict[str, object] = {}
_MODEL_LOCK = threading.Lock()

# Current package version — used in vector file metadata
_VECTORS_VERSION = "1.0.0"


class SemanticEngineError(Exception):
    """Base exception for semantic engine errors."""
    pass


class SemanticEngine:
    """Semantic search engine using sentence-transformers.
    
    Provides dense vector embeddings for semantic similarity matching.
    Uses all-MiniLM-L6-v2 model (22MB, 384-dim vectors, Apache 2.0).
    
    Thread safety: All public methods are thread-safe. Concurrent reads are
    allowed; writes (store/remove/gc) are serialized via _vectors_lock.
    
    Args:
        workspace: Path to the memory workspace directory.
        model_name: Transformer model name (default: all-MiniLM-L6-v2).
        vectors_file: Filename for stored vectors (default: .vectors.json).
        bootstrap_interval: Minimum seconds between bootstrap_cooccurrence calls
            in rerank() (default 60.0). Uses monotonic clock — immune to NTP jumps.
    """
    
    def __init__(
        self,
        workspace: str,
        model_name: str = "all-MiniLM-L6-v2",
        vectors_file: str = ".vectors.json",
        bootstrap_interval: float = 60.0,
    ):
        self.workspace = Path(workspace)
        self.model_name = model_name
        self.vectors_path = self.workspace / vectors_file
        
        # In-memory vector storage: memory_id -> List[float]
        self._vectors: Dict[str, List[float]] = {}
        self._dirty = False

        # A1: Dimensionality validation — set on first store, enforced thereafter
        self._dim: Optional[int] = None

        # B2: Cached L2 norms: memory_id -> float (avoids recomputing every search)
        self._norms: Dict[str, float] = {}

        # GPT-C: Thread safety for _dim/_vectors/_norms mutations
        # Concurrent reads are lock-free; writes acquire this lock.
        self._vectors_lock = threading.Lock()
        
        # Model will be loaded lazily on first use
        self._model = None
        self._model_loaded = False

        # C-1 + Gemini: Bootstrap debounce — monotonic clock (immune to wall-clock skew).
        # _bootstrap_lock ensures only one concurrent caller triggers bootstrap per interval.
        # Timestamp is set BEFORE the call so concurrent callers don't pile up.
        self._bootstrap_lock = threading.Lock()
        self._last_bootstrap_mono: float = 0.0
        self._bootstrap_interval: float = bootstrap_interval

        # Telemetry: lightweight counters so operators can verify the feedback loop
        # is doing real work (or diagnose why it isn't). Not persisted — reset on restart.
        self._bootstrap_attempts: int = 0    # times bootstrap was triggered in rerank()
        self._bootstrap_inject_total: int = 0  # cumulative inject_synthetic_pair() calls

    def _ensure_model(self):
        """Thread-safe lazy model loading. Per-model cache keyed by model_name."""
        global _MODEL_CACHE
        
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            self._model_loaded = True
            return
        
        with _MODEL_LOCK:
            if self.model_name in _MODEL_CACHE:
                self._model = _MODEL_CACHE[self.model_name]
                self._model_loaded = True
                return
                
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                _MODEL_CACHE[self.model_name] = self._model
                self._model_loaded = True
            except ImportError:
                raise SemanticEngineError(
                    "sentence-transformers not installed. "
                    "Install with: pip install 'antaris-memory[semantic]'"
                )
            except Exception as e:
                raise SemanticEngineError(f"Failed to load transformer model: {e}")

    # ── Vector Storage ────────────────────────────────────────────────────────

    def load_vectors(self) -> None:
        """Load stored vectors from disk."""
        if not self.vectors_path.exists():
            self._vectors = {}
            self._norms = {}
            return
            
        try:
            with open(self.vectors_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stored_model = data.get("model", "")
            if stored_model and stored_model != self.model_name:
                raise SemanticEngineError(
                    f"Vector file was created with model '{stored_model}' but "
                    f"engine is configured for '{self.model_name}'. Delete "
                    f"{self.vectors_path} to rebuild vectors with the new model."
                )
            self._vectors = data.get("vectors", {})

            # A1: Load stored dimensionality and validate against engine config
            stored_dim = data.get("dim")
            if stored_dim is not None and self._dim is not None and stored_dim != self._dim:
                raise SemanticEngineError(
                    f"Vector file has dim={stored_dim} but engine is configured for "
                    f"dim={self._dim}. Delete {self.vectors_path} to rebuild."
                )
            if self._dim is None and stored_dim is not None:
                self._dim = stored_dim

            # Gem-4: If dim not stored (older file format), infer from first vector.
            # This restores global dim validation for files created before A1 was added,
            # so per-vector guards aren't the only protection on migration.
            if self._dim is None and self._vectors:
                first_vec = next(iter(self._vectors.values()))
                self._dim = len(first_vec)

            # Non-blocking dim health check: warn on vectors that don't match _dim.
            # Per-vector guard in search() already skips them — this surfaces
            # corruption at load time rather than silently at query time.
            if self._dim is not None:
                mismatched = [
                    mid for mid, vec in self._vectors.items()
                    if len(vec) != self._dim
                ]
                if mismatched:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "load_vectors: %d vector(s) have unexpected dim (expected %d); "
                        "they will be skipped in search(). First affected: %s. "
                        "Delete %s to rebuild.",
                        len(mismatched), self._dim, mismatched[:3], self.vectors_path,
                    )

        except (json.JSONDecodeError, KeyError, OSError):
            self._vectors = {}

        # B2: Precompute norms for all loaded vectors
        self._norms = {
            mid: math.sqrt(sum(v * v for v in vec))
            for mid, vec in self._vectors.items()
        }
        self._dirty = False

    def save_vectors(self) -> None:
        """Save vectors to disk if dirty."""
        if not self._dirty:
            return
            
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        data = {
            "model": self.model_name,
            "vectors": self._vectors,
            "version": _VECTORS_VERSION,  # GPT-A: was hardcoded "3.4" — now tracks package version
        }

        # A1: Persist dimensionality if known
        if self._dim is not None:
            data["dim"] = self._dim
        
        # Atomic write
        tmp_path = self.vectors_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
                
        tmp_path.replace(self.vectors_path)
        self._dirty = False

    def store_vector(self, memory_id: str, vector: List[float]) -> None:
        """Store a vector for a memory entry.
        
        Thread-safe: serialized via _vectors_lock.
        
        Args:
            memory_id: Unique identifier for the memory.
            vector: 384-dimensional embedding vector.
        """
        if not isinstance(vector, list) or not vector:
            raise ValueError("Vector must be a non-empty list")

        with self._vectors_lock:
            # A1 + GPT-C: Validate / lock in dimensionality under lock to prevent
            # concurrent writers racing on _dim initialization.
            if self._dim is None:
                self._dim = len(vector)
            elif len(vector) != self._dim:
                raise ValueError(f"Expected {self._dim}-dim vector, got {len(vector)}")

            self._vectors[memory_id] = vector
            # B2: Cache the L2 norm for this vector
            self._norms[memory_id] = math.sqrt(sum(v * v for v in vector))
            self._dirty = True

    def get_vector(self, memory_id: str) -> Optional[List[float]]:
        """Get stored vector for a memory entry.

        Thread-safe: acquires _vectors_lock for consistent reads (Gem-A).

        Args:
            memory_id: Memory identifier.
            
        Returns:
            384-dim vector list, or None if not found.
        """
        with self._vectors_lock:
            return self._vectors.get(memory_id)

    def remove_vector(self, memory_id: str) -> bool:
        """Remove vector for a memory entry.
        
        Thread-safe: serialized via _vectors_lock.
        
        Args:
            memory_id: Memory identifier.
            
        Returns:
            True if vector was removed, False if not found.
        """
        with self._vectors_lock:
            if memory_id in self._vectors:
                del self._vectors[memory_id]
                self._norms.pop(memory_id, None)  # B2: evict cached norm
                self._dirty = True
                return True
        return False

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """Embed a single text string.
        
        Args:
            text: Text to embed.
            
        Returns:
            384-dimensional embedding vector.
        """
        if not text.strip():
            # Gem-B: use actual _dim when known so empty-text zero vector is always
            # the correct dimension regardless of model_name override.
            return [0.0] * (self._dim or 384)
            
        self._ensure_model()
        
        try:
            # sentence-transformers returns numpy arrays by default
            embedding = self._model.encode([text], convert_to_numpy=True)[0]
            # Convert to Python list for JSON serialization
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                # Fallback for mocked tests or other array types
                return list(embedding)
        except Exception as e:
            raise SemanticEngineError(f"Embedding failed: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of 384-dimensional embedding vectors.
        """
        if not texts:
            return []
            
        self._ensure_model()
        
        try:
            # Process in batch for efficiency
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            # Convert to Python lists for JSON serialization
            # Use .tolist() on the whole array when possible (faster than per-row)
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            else:
                # Fallback for mocked tests or other array types
                return [list(emb) for emb in embeddings]
        except Exception as e:
            raise SemanticEngineError(f"Batch embedding failed: {e}")

    # ── Search ────────────────────────────────────────────────────────────────

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            vec_a: First vector.
            vec_b: Second vector.
            
        Returns:
            Cosine similarity in range [-1, 1].
        """
        if len(vec_a) != len(vec_b):
            return 0.0
            
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot / (norm_a * norm_b)

    def search(self, query_vector: List[float], top_k: int = 20) -> List[Tuple[str, float]]:
        """Find most similar memory IDs by cosine similarity.

        GPT-B: Enforces dimensionality before scoring. Previously, a short query
        vector would silently truncate the dot product via zip() while dividing by
        the full stored norm — producing plausible-looking but meaningless scores.
        
        Args:
            query_vector: 384-dimensional query embedding.
            top_k: Maximum number of results to return.
            
        Returns:
            List of (memory_id, similarity_score) tuples, sorted by similarity descending.
        """
        if not self._vectors:
            return []
        if not isinstance(query_vector, list) or not query_vector:
            return []

        # GPT-B: Dimensionality enforcement — reject mismatched queries outright
        if self._dim is not None and len(query_vector) != self._dim:
            return []

        # B2: Compute query norm once, reuse cached stored norms
        query_norm = math.sqrt(sum(v * v for v in query_vector))
        if query_norm == 0:
            return []

        # Gem-A: Snapshot _vectors and _norms under lock so concurrent writes
        # (store/remove/gc) don't mutate the dict while we iterate. The snapshot
        # is a shallow copy — O(N) but avoids the risk of RuntimeError or torn reads.
        with self._vectors_lock:
            items_snapshot = list(self._vectors.items())
            norms_snapshot = dict(self._norms)

        scored = []
        for memory_id, vector in items_snapshot:
            # GPT-B: Per-vector guard for vectors stored before _dim was locked in
            if len(vector) != len(query_vector):
                continue
            dot = sum(a * b for a, b in zip(query_vector, vector))
            stored_norm = norms_snapshot.get(memory_id)
            if stored_norm is None:
                stored_norm = math.sqrt(sum(v * v for v in vector))
            if stored_norm == 0:
                continue
            similarity = dot / (query_norm * stored_norm)
            if similarity > 0:
                scored.append((memory_id, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def bootstrap_cooccurrence(
        self,
        cooccurrence_index,
        similarity_threshold: float = 0.85,
        max_pairs: int = 100,
        weight: float = 0.1,
        content_map: Optional[Dict[str, str]] = None,
        max_inject_calls: int = 500,
    ) -> int:
        """Bootstrap a co-occurrence index using high-confidence semantic matches.

        Finds pairs of stored memories with cosine similarity > *similarity_threshold*,
        then injects their shared tokens as synthetic co-occurrence pairs. This lets
        Tier 1 (PPMI) benefit from Tier 2 (transformer) signals even after the
        transformer model has been unloaded.

        When *content_map* is provided the method tokenises the actual memory content
        and injects word-token pairs (richer signal). Without it, memory IDs are used
        as proxy tokens — note that the CooccurrenceIndex's "at least one word already
        in index" guard means the ID-path often no-ops in practice. Pass content_map
        from callers whenever possible for meaningful signal injection.

        Args:
            cooccurrence_index: CooccurrenceIndex to inject pairs into.
            similarity_threshold: Min cosine similarity to consider a match (default 0.85).
            max_pairs: Maximum number of semantic pairs to process per call (default 100).
            weight: Synthetic co-occurrence weight per pair (default 0.1 — intentionally
                lower than organic signal weight=1.0 to prevent synthetic data from
                dominating PPMI scores).
            content_map: Optional mapping of memory_id → content string. When
                supplied, word tokens extracted from both memories are cross-injected
                (richer signal than ID-based injection).
            max_inject_calls: Hard cap on total inject_synthetic_pair() calls per run
                (default 500). Prevents O(T_a × T_b) cross-product explosion in the
                content_map path from overwhelming organic co-occurrence signal.

        Returns:
            Number of inject_synthetic_pair() calls performed (actual injection work done).
            Use this to verify the feedback loop is doing real work; zero means the
            ID-path guard dropped all pairs or no pairs exceeded the threshold.
        """
        if not self._vectors:
            return 0

        # C-3: Use CooccurrenceIndex's own tokenizer — ensures token space is identical
        # to what organic co-occurrence ingestion produces. Prevents silent accuracy loss
        # from divergent stopword lists or regex differences.
        tokenize = cooccurrence_index._tokenize

        total_inject_calls = 0
        semantic_pairs_processed = 0

        # Shiro S1: Snapshot _vectors under lock so concurrent remove/gc calls can't
        # delete a key between sorted(keys()) and the [id_a]/[id_b] lookups below.
        # Without the snapshot, a vector deleted mid-iteration raises KeyError.
        # The snapshot is a shallow dict copy — O(N), same as the search() pattern.
        with self._vectors_lock:
            vectors_snapshot = dict(self._vectors)

        # GPT-3: Deterministic ordering — identical corpora always produce identical
        # PPMI mutations regardless of ingest order. Without sort(), insertion order
        # of dict keys determines which pairs are processed before max_pairs/max_inject_calls
        # cuts off, causing retrieval differences for semantically equivalent stores.
        memory_ids = sorted(vectors_snapshot.keys())

        # Upper-triangle only — no self-pairs, no duplicates
        for i in range(len(memory_ids)):
            if semantic_pairs_processed >= max_pairs or total_inject_calls >= max_inject_calls:
                break
            for j in range(i + 1, len(memory_ids)):
                if semantic_pairs_processed >= max_pairs or total_inject_calls >= max_inject_calls:
                    break

                id_a, id_b = memory_ids[i], memory_ids[j]

                similarity = self.cosine_similarity(vectors_snapshot[id_a], vectors_snapshot[id_b])
                if similarity < similarity_threshold:
                    continue

                if content_map:
                    # Rich path: cross-inject content tokens (capped by max_inject_calls)
                    tokens_a = list(dict.fromkeys(tokenize(content_map.get(id_a, ""))))
                    tokens_b = list(dict.fromkeys(tokenize(content_map.get(id_b, ""))))
                    for ta in tokens_a:
                        if total_inject_calls >= max_inject_calls:
                            break
                        for tb in tokens_b:
                            if total_inject_calls >= max_inject_calls:
                                break
                            if ta != tb:
                                cooccurrence_index.inject_synthetic_pair(ta, tb, weight=weight)
                                total_inject_calls += 1
                    semantic_pairs_processed += 1
                else:
                    # Proxy path: treat memory IDs themselves as tokens.
                    # NOTE: inject_synthetic_pair() requires at least one token already
                    # in the organic index — this path often no-ops in practice.
                    # Pass content_map from callers for meaningful signal injection.
                    cooccurrence_index.inject_synthetic_pair(id_a, id_b, weight=weight)
                    total_inject_calls += 1
                    semantic_pairs_processed += 1

        return total_inject_calls  # GPT-3: return actual work done, not pairs examined

    def rerank(
        self,
        query: str,
        candidates: List,
        top_k: int = 10,
        cooccurrence_index=None,
    ) -> List:
        """Re-rank BM25 candidates using semantic similarity.

        Two-stage pipeline:
        1. BM25 provides fast, broad recall (candidates)
        2. Semantic similarity provides precise re-ranking

        Optionally bootstraps a CooccurrenceIndex with high-confidence pairs
        found during re-ranking (Tier 2 → Tier 1 feedback loop).

        C-1 + Gemini: Bootstrap is debounced using monotonic clock — immune to
        NTP/wall-clock jumps. _bootstrap_lock ensures only one concurrent caller
        triggers bootstrap per interval; the timestamp is set before the call so
        concurrent callers skip rather than pile up.

        Args:
            query: Original search query.
            candidates: List of SearchResult objects from BM25.
            top_k: Number of results to return after re-ranking.
            cooccurrence_index: Optional CooccurrenceIndex to receive synthetic
                co-occurrence pairs from high-confidence semantic matches.
                When supplied, ``bootstrap_cooccurrence()`` is called after
                re-ranking so that Tier 1 benefits from Tier 2 signals.

        Returns:
            List of SearchResult objects, re-ranked by semantic similarity.
        """
        if not candidates:
            return []

        # Embed the query
        try:
            query_vector = self.embed(query)
        except SemanticEngineError:
            # Fallback: return original candidates if embedding fails
            return candidates[:top_k]

        # Re-score candidates using semantic similarity
        from antaris_memory.search import SearchResult
        reranked = []
        for candidate in candidates:
            memory_id = getattr(candidate.entry, 'hash', str(id(candidate.entry)))
            stored_vector = self.get_vector(memory_id)

            if stored_vector:
                semantic_sim = self.cosine_similarity(query_vector, stored_vector)
                blended_score = 0.7 * semantic_sim + 0.3 * candidate.relevance
            else:
                blended_score = candidate.relevance

            reranked_result = SearchResult(
                entry=candidate.entry,
                score=blended_score,
                relevance=blended_score,
                matched_terms=candidate.matched_terms,
                explanation=f"{candidate.explanation} | semantic={blended_score:.3f}"
            )
            reranked.append(reranked_result)

        # Sort by new blended score
        reranked.sort(key=lambda x: x.score, reverse=True)
        result = reranked[:top_k]

        # C-1 + Gemini: Tier 2 → Tier 1 feedback — debounced via monotonic clock + lock.
        # Lock is checked and timestamp set BEFORE bootstrap call so concurrent rerank()
        # callers see the updated timestamp and skip rather than queuing up.
        if cooccurrence_index is not None and self._vectors:
            now = time.monotonic()
            with self._bootstrap_lock:
                if now - self._last_bootstrap_mono < self._bootstrap_interval:
                    return result  # debounce: not yet time for another bootstrap
                self._last_bootstrap_mono = now  # claim the slot before releasing lock

            # Run bootstrap outside the lock — it's expensive and doesn't need
            # to block other callers. They will see the updated timestamp and skip.
            inject_calls = self.bootstrap_cooccurrence(cooccurrence_index)
            self._bootstrap_attempts += 1
            self._bootstrap_inject_total += inject_calls

        return result

    # ── Memory Integration ────────────────────────────────────────────────────

    def embed_and_store(self, memory_id: str, content: str) -> List[float]:
        """Embed content and store the vector.
        
        Convenience method for write-time embedding.
        Skips storage for empty/whitespace content (zero vectors are unsearchable).
        
        Args:
            memory_id: Unique identifier for the memory.
            content: Memory content to embed.
            
        Returns:
            The computed embedding vector (zero vector for empty content).
        """
        vector = self.embed(content)
        # Don't store zero vectors — they're unsearchable (cosine sim = 0 with everything)
        if any(v != 0.0 for v in vector):
            self.store_vector(memory_id, vector)
        return vector

    def embed_batch_and_store(self, entries: List[Tuple[str, str]]) -> int:
        """Embed multiple entries and store their vectors.
        
        Args:
            entries: List of (memory_id, content) tuples.
            
        Returns:
            Number of vectors stored.
        """
        if not entries:
            return 0
            
        memory_ids = [entry[0] for entry in entries]
        contents = [entry[1] for entry in entries]
        
        vectors = self.embed_batch(contents)
        
        stored = 0
        for memory_id, vector in zip(memory_ids, vectors):
            if any(v != 0.0 for v in vector):
                self.store_vector(memory_id, vector)
                stored += 1
            
        return stored

    # ── Stats ─────────────────────────────────────────────────────────────────

    def gc_vectors(self, active_memory_ids: set) -> int:
        """Garbage-collect vectors for deleted memories.
        
        Removes any stored vector whose memory_id is not in the active set.
        Call periodically to prevent unbounded vector storage growth.
        
        Args:
            active_memory_ids: Set of memory IDs that still exist.
            
        Returns:
            Number of orphaned vectors removed.
        """
        with self._vectors_lock:
            orphans = [mid for mid in self._vectors if mid not in active_memory_ids]
            for mid in orphans:
                del self._vectors[mid]
                self._norms.pop(mid, None)  # B2: evict cached norm for orphan
            if orphans:
                self._dirty = True
        return len(orphans)

    def stats(self) -> Dict:
        """Return semantic engine statistics including bootstrap telemetry.
        
        bootstrap_attempts and bootstrap_inject_total are in-process counters
        (reset on restart). Use them to verify the Tier 2 → Tier 1 feedback loop
        is doing real work. Zero inject_total with non-zero attempts means the
        ID-path guard is dropping all pairs — pass content_map to rerank() or
        bootstrap_cooccurrence() for meaningful signal injection.

        seconds_since_bootstrap is an elapsed-seconds delta (time.monotonic()
        base), not a wall-clock timestamp. It reflects how long ago the last
        bootstrap ran. On a fresh engine with no bootstrap yet, this equals the
        total uptime of the engine instance.
        """
        return {
            "model": self.model_name,
            "vector_count": len(self._vectors),
            "dim": self._dim,
            "model_loaded": self._model_loaded,
            "vectors_file": str(self.vectors_path),
            "vectors_exist": self.vectors_path.exists(),
            # Bootstrap telemetry (in-process, not persisted)
            "bootstrap_attempts": self._bootstrap_attempts,
            "bootstrap_inject_total": self._bootstrap_inject_total,
            "seconds_since_bootstrap": time.monotonic() - self._last_bootstrap_mono,
        }

    @property
    def vector_count(self) -> int:
        """Number of stored vectors."""
        return len(self._vectors)

    @property
    def is_model_loaded(self) -> bool:
        """Whether the transformer model is loaded."""
        return self._model_loaded
