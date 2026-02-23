"""
MemorySystem — Production-ready memory system with sharding, BM25 search, and concurrency safety.

Features:
- Sharded storage for scalability (10K+ memories)
- BM25-inspired search with IDF weighting and field boosting
- File locking and optimistic conflict detection
- Fast indexes (full-text, tags, dates)
- Schema migration with backward compatibility

Usage:
    from antaris_memory import MemorySystem

    mem = MemorySystem("./workspace")
    mem.load()
    mem.ingest("Key decision", source="meeting", category="strategic")
    results = mem.search("decision")
    mem.save()  # Saves to sharded format
"""

import json
import os
import re
import time
from collections import defaultdict
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_MAX_MEMORIES: int = 5000  # Tier cap for antaris-flame

from .entry import MemoryEntry
from .decay import DecayEngine
from .sentiment import SentimentTagger
from .temporal import TemporalEngine
from .confidence import ConfidenceEngine
from .compression import CompressionEngine
from .forgetting import ForgettingEngine
from .consolidation import ConsolidationEngine
from .gating import InputGate
from .synthesis import KnowledgeSynthesizer
from .sharding import ShardManager
from .migration import MigrationManager
from .indexing import IndexManager
from .search import SearchEngine, SearchResult
from .context_packet import ContextPacket, ContextPacketBuilder
from .namespace import NamespaceManager
from .memory_types import (
    MEMORY_TYPE_CONFIGS, DEFAULT_TYPE, get_type_config,
    format_mistake_content, SEVERITY_LEVELS,
)
from .utils import atomic_write_json
from .performance import ReadCache, WALManager, PerformanceMonitor, AccessTracker
from .shared import SharedMemoryPool

# Default tags to auto-extract
_DEFAULT_TAG_TERMS = [
    "web3", "ethereum", "postgresql", "optimization", "cost",
    "revenue", "security", "deployment", "production", "testing",
]


class MemorySystemV4(NamespaceManager):
    """Production-ready memory system with sharding and indexing.

    Parameters
    ----------
    workspace : str
        Root directory. Shards and indexes are stored here.
    half_life : float
        Decay half-life in days (default 7).
    tag_terms : list[str] | None
        Custom terms to auto-tag. Merged with built-in defaults.
    use_sharding : bool
        Whether to use sharded storage (default True for v0.4).
    use_indexing : bool
        Whether to build search indexes (default True for v0.4).
    """

    def __init__(
        self,
        workspace: str = ".",
        half_life: float = 7.0,
        tag_terms: List[str] = None,
        use_sharding: bool = True,
        use_indexing: bool = True,
        enable_read_cache: bool = True,
        cache_max_entries: int = 1000,
        max_memories: Optional[int] = DEFAULT_MAX_MEMORIES,
        enable_semantic: bool = False,
        cooccurrence_boost_weight: float = 0.3,
        collection_endpoint: Optional[str] = None,
        collection_api_key: Optional[str] = None,
        collection_opt_in: bool = False,
    ):
        self.workspace = os.path.abspath(workspace)
        self.use_sharding = use_sharding
        self.use_indexing = use_indexing

        # Backward compatibility paths
        self.legacy_metadata_path = os.path.join(self.workspace, "memory_metadata.json")
        self.audit_path = os.path.join(self.workspace, "memory_audit.json")

        # v0.4 managers
        self.migration_manager = MigrationManager(workspace)
        self.shard_manager = ShardManager(workspace) if use_sharding else None
        self.index_manager = IndexManager(workspace) if use_indexing else None

        # Engines (same as v0.3)
        self.decay = DecayEngine(half_life=half_life)
        self.sentiment = SentimentTagger()
        self.temporal = TemporalEngine()
        self.confidence = ConfidenceEngine()
        self.compression = CompressionEngine()
        self.forgetting = ForgettingEngine()
        self.consolidation = ConsolidationEngine(decay=self.decay)
        self.gating = InputGate()
        self.synthesis = KnowledgeSynthesizer()
        self.search_engine = SearchEngine()

        # Tag terms
        self._tag_terms = list(set(_DEFAULT_TAG_TERMS + (tag_terms or [])))

        # Memory store (in-memory cache)
        self.memories: List[MemoryEntry] = []
        self._hashes: set = set()

        # Sprint 11 — performance subsystem
        self._read_cache: Optional[ReadCache] = (
            ReadCache(cache_max_entries) if enable_read_cache else None
        )
        self._enable_read_cache = enable_read_cache
        self._wal = WALManager(self.workspace)
        self._perf = PerformanceMonitor()
        self._access_tracker = AccessTracker(self.workspace)

        # Sprint 3 — pluggable embedding for hybrid semantic search
        self._embedding_fn: Optional[Callable[[str], List[float]]] = None
        self._embedding_cache: Dict[str, List[float]] = {}  # key -> vector, bounded to 1000

        # v3.2 — Instrumentation + Co-occurrence (Tier 1 Semantic)
        from .instrumentation import SearchContext
        from .cooccurrence import CooccurrenceIndex
        self._cooccurrence = CooccurrenceIndex(self.workspace)
        self._cooccurrence.load()
        self._instrumentation_log_path = os.path.join(self.workspace, ".instrumentation_log.jsonl")
        self._usage_signals_path = os.path.join(self.workspace, ".usage_signals.jsonl")
        self.opt_in_to_aggregation: bool = False  # Must explicitly opt in
        self._shared_pool: Optional["SharedMemoryPool"] = None
        self._shared_agent_id: Optional[str] = None

        # v4.1 — Tiered semantic system + data collection pipeline
        self.max_memories = max_memories
        self.enable_semantic = enable_semantic
        self.cooccurrence_boost_weight = cooccurrence_boost_weight

        # Tier 2: Optional semantic engine (requires sentence-transformers)
        self._semantic_engine: Optional[object] = None
        if enable_semantic:
            try:
                from .semantic import SemanticEngine
                self._semantic_engine = SemanticEngine(workspace=self.workspace)
                self._semantic_engine.load()
            except Exception:
                self._semantic_engine = None  # Graceful degradation

        # Tier 3: Crowdsourced PPMI index (loads ppmi.json if present)
        from .ppmi import StaticPPMILookup
        self._ppmi = StaticPPMILookup(workspace=self.workspace)
        self._ppmi.load()

        # Data collection: consent + collection agent
        from .consent import ConsentStore
        from .collection import CollectionAgent
        self.consent = ConsentStore(workspace=self.workspace)
        self.consent.load()
        if collection_opt_in and not self.consent.is_opted_in():
            self.consent.opt_in()
        self._collection_agent = CollectionAgent(
            workspace=self.workspace,
            cooccurrence=self._cooccurrence,
            consent=self.consent,
            endpoint_url=collection_endpoint,
            api_key=collection_api_key,
        )
        self._collection_agent.load_state()

        # Initialize system
        self._initialize()

    def _initialize(self):
        """Initialize the memory system, handling migrations if needed."""
        import logging  # Bug fix: moved to top of method — was inside success branch only
        _log = logging.getLogger("antaris_memory")
        # Check if migration is needed
        if self.migration_manager.needs_migration():
            migration_result = self.migration_manager.migrate()
            if migration_result["status"] == "success":
                _log.info(
                    f"Migrated from {migration_result['from_version']} to {migration_result['to_version']}")
            elif migration_result["status"] == "error":
                _log.error(
                    f"Migration failed: {migration_result['message']}")
                # Fall back to legacy format
                self.use_sharding = False
                self.use_indexing = False

        # Sprint 8 — namespace manager
        self._init_namespace_manager()

        # Load existing data
        self.load()

    # ── persistence ─────────────────────────────────────────────────────

    def save(self) -> str:
        """Save memory state to disk using the appropriate format."""
        if self.use_sharding and self.shard_manager:
            return self._save_sharded()
        else:
            return self._save_legacy()
    
    def _save_sharded(self) -> str:
        """Save using v0.4 sharded format."""
        if not self.shard_manager:
            return self._save_legacy()
        
        # Group memories by shard
        shard_groups = self.shard_manager.shard_memories(self.memories)
        
        # Save each shard
        for shard_key, shard_memories in shard_groups.items():
            self.shard_manager.save_shard(shard_key, shard_memories)
            self.shard_manager.index.add_shard(shard_key, shard_memories)
        
        # Save shard index
        self.shard_manager.index.save_index()
        
        # Update search indexes
        if self.use_indexing and self.index_manager:
            self.index_manager.rebuild_indexes(self.memories)
            self.index_manager.save_all_indexes()
        
        shard_dir = os.path.join(self.workspace, "shards")
        return shard_dir
    
    def _save_legacy(self) -> str:
        """Save using v0.2/v0.3 legacy single-file format."""
        from . import __version__ as _pkg_version  # Bug fix: distinguish format vs pkg version
        data = {
            "version": "0.4.0",           # storage schema version (used by migration logic)
            "package_version": _pkg_version,  # antaris-memory package version
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self.memories),
            "format": "legacy",
            "memories": [m.to_dict() for m in self.memories],
        }
        atomic_write_json(self.legacy_metadata_path, data)
        return self.legacy_metadata_path

    # ── WAL flush / close ─────────────────────────────────────────────────

    def flush(self) -> Dict:
        """Compact the WAL into permanent shard/legacy storage.

        Called automatically after ``WALManager.flush_interval`` writes or
        when the WAL exceeds 1 MB.  Also called by ``close()``.

        Returns:
            ``{"flushed_entries": N, "wal_cleared": True}``
        """
        pending_before = self._wal.pending_count()

        # Persist current in-memory state (includes WAL-replayed entries)
        self.save()

        # Persist access counts alongside the flush
        self._access_tracker.save()

        # Persist co-occurrence index (deferred from ingest for bulk-write perf)
        self._cooccurrence.save()

        # WAL has been safely persisted to shards — clear it
        self._wal.clear()

        # Invalidate read-cache (data on disk changed)
        if self._read_cache is not None:
            self._read_cache.invalidate()

        self._perf.record_compaction()

        return {"flushed_entries": pending_before, "wal_cleared": True}

    def close(self) -> None:
        """Flush WAL and release resources.  Call before process exit."""
        self.flush()

    def load(self) -> int:
        """Load memory state from disk, auto-detecting format.

        Sprint 11: after loading from shards/legacy, replays any pending WAL
        entries so that a crash between ingest and flush loses no data.
        """
        if self.use_sharding and self.shard_manager:
            count = self._load_sharded()
            if count > 0:
                self._replay_wal()
                return len(self.memories)

        # Fall back to legacy format
        result = self._load_legacy()
        self._replay_wal()
        return len(self.memories)

    def _replay_wal(self) -> int:
        """Replay pending WAL entries that were not yet compacted to shards.

        This is called automatically on ``load()`` and is safe to call even
        when the WAL is empty or doesn't exist.

        Returns:
            Number of entries replayed.
        """
        pending = self._wal.load_pending()
        if not pending:
            return 0

        replayed = 0
        for entry_dict in pending:
            try:
                entry = MemoryEntry.from_dict(entry_dict)
            except Exception:
                continue   # corrupted entry — skip

            if entry.hash in self._hashes:
                continue   # already in memory (e.g. flushed before crash)

            self.memories.append(entry)
            self._hashes.add(entry.hash)
            replayed += 1

        if replayed:
            self.search_engine.mark_dirty()

        return replayed

    def _load_sharded(self) -> int:
        """Load from v0.4 sharded format."""
        if not self.shard_manager:
            return 0

        # Load all memories from shards (careful with memory usage)
        _LOAD_LIMIT = 10000
        self.memories = self.shard_manager.get_all_memories(limit=_LOAD_LIMIT)
        # Bug fix: warn when the safety limit is hit so users know data was truncated
        if len(self.memories) >= _LOAD_LIMIT:
            import warnings
            warnings.warn(
                f"antaris-memory: loaded {_LOAD_LIMIT} memories (safety limit). "
                "Some older memories may not be in the active set. "
                "Consider calling forget() or archiving old shards.",
                UserWarning,
                stacklevel=3,
            )
        self._hashes = {m.hash for m in self.memories}
        if self.memories:
            self.search_engine.build_index(self.memories)
        return len(self.memories)

    def _load_legacy(self) -> int:
        """Load from v0.2/v0.3 legacy single-file format."""
        if not os.path.exists(self.legacy_metadata_path):
            return 0

        with open(self.legacy_metadata_path) as f:
            data = json.load(f)

        self.memories = [MemoryEntry.from_dict(d) for d in data.get("memories", [])]
        self._hashes = {m.hash for m in self.memories}
        if self.memories:
            self.search_engine.build_index(self.memories)
        return len(self.memories)

    # ── Sprint 3: embedding interface ────────────────────────────────────

    def set_embedding_fn(self, fn: Callable[[str], List[float]]) -> None:
        """Set a custom embedding function for semantic search.

        When set, search() uses hybrid BM25 + cosine similarity scoring.
        When not set, pure BM25 is used (default, zero-dependency behavior).

        Example::

            import openai
            client = openai.Client()
            def embed(text):
                return client.embeddings.create(
                    input=text, model="text-embedding-3-small"
                ).data[0].embedding
            mem.set_embedding_fn(embed)
        """
        self._embedding_fn = fn
        self._embedding_cache.clear()  # invalidate embedding cache on fn change
        if self._read_cache is not None:
            self._read_cache.invalidate()  # invalidate search result cache

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding with caching. Returns None if no embedding fn set."""
        if self._embedding_fn is None:
            return None
        key = text[:200]  # cache key
        if key not in self._embedding_cache:
            if len(self._embedding_cache) >= 1000:
                # Evict oldest (first inserted key in Python 3.7+ dict)
                del self._embedding_cache[next(iter(self._embedding_cache))]
            try:
                self._embedding_cache[key] = self._embedding_fn(text)
            except Exception:
                return None
        return self._embedding_cache.get(key)

    # ── search with indexing ───────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 20,
        tags: Optional[List[str]] = None,
        tag_mode: str = "any",
        date_range: Optional[Tuple[str, str]] = None,
        use_decay: bool = True,
        category: str = None,
        min_confidence: float = 0.0,
        sentiment_filter: str = None,
        memory_type: Optional[str] = None,    # Sprint 2
        explain: bool = False,
    ) -> list:
        """Search memories using BM25-inspired ranking with optional fast indexes.

        Sprint 11: results are served from the LRU read-cache when the query
        signature matches a previously seen search.  On a cache miss the full
        BM25 pipeline runs and the result-set is cached for future calls.
        Access counts are incremented for every returned entry so that hot
        memories receive a relevance boost in future searches.

        Bug fix: BM25 IDF is always computed over the full corpus (self.memories)
        so that type/sentiment filters do not corrupt IDF weights.  Filters are
        applied AFTER scoring.

        Feature: recall_priority from MEMORY_TYPE_CONFIGS is applied to each
        scored entry so that e.g. mistake memories (recall_priority=1.0) float
        to the top.

        Args:
            query: Search query string.
            limit: Maximum results.
            memory_type: Optional filter — only return entries of this type.
            explain: If True, return SearchResult objects with score explanations.
        """
        _t0 = time.monotonic()

        # ── cache key ───────────────────────────────────────────────────
        cache_key = json.dumps(
            [query, limit, tags, tag_mode, date_range, use_decay,
             category, min_confidence, sentiment_filter, memory_type, explain],
            sort_keys=True, default=str,
        )

        if self._read_cache is not None:
            cached = self._read_cache.get(cache_key)
            if cached is not None:
                self._perf.record_search((time.monotonic() - _t0) * 1000)
                return cached

        # ── full search pipeline ─────────────────────────────────────────
        # Bug fix (Bug 2): always build the BM25 index from the FULL corpus
        # so that IDF weights are never corrupted by a filtered subset.
        if self.search_engine._doc_count != len(self.memories):
            self.search_engine.build_index(self.memories)

        decay_fn = self.decay.score if use_decay else None

        # Fetch a generous pool from the full corpus, then filter & trim.
        # Using limit * 10 (min 200) ensures filters don't starve results.
        fetch_limit = max(limit * 10, 200)
        search_results = self.search_engine.search(
            query=query,
            memories=self.memories,
            limit=fetch_limit,
            category=category,
            min_score=min_confidence,
            decay_fn=decay_fn,
        )

        # ── post-scoring filters (Bug 2 fix: applied AFTER BM25 scoring) ──
        if sentiment_filter:
            search_results = [
                r for r in search_results
                if hasattr(r.entry, "sentiment")
                and r.entry.sentiment
                and sentiment_filter in r.entry.sentiment
            ]
        if memory_type is not None:
            search_results = [
                r for r in search_results
                if getattr(r.entry, "memory_type", "episodic") == memory_type
            ]

        # ── Sprint 3: hybrid BM25 + semantic scoring ──────────────────────
        if self._embedding_fn is not None:
            query_vec = self._get_embedding(query)
            if query_vec is not None:
                from .utils import cosine_similarity
                bm25_weight = 0.4
                semantic_weight = 0.6
                hybrid_results = []
                for result in search_results:
                    entry_vec = self._get_embedding(result.entry.content[:500])
                    if entry_vec is not None:
                        sem_score = cosine_similarity(query_vec, entry_vec)
                        hybrid_score = bm25_weight * result.relevance + semantic_weight * sem_score
                        hybrid_results.append(SearchResult(
                            entry=result.entry,
                            score=hybrid_score,
                            relevance=hybrid_score,
                            matched_terms=result.matched_terms,
                            explanation=result.explanation + f" [hybrid: bm25={result.relevance:.3f} sem={sem_score:.3f}]",
                        ))
                    else:
                        hybrid_results.append(result)
                search_results = sorted(hybrid_results, key=lambda r: r.score, reverse=True)

        # ── Feature 1: recall_priority boost ─────────────────────────────
        for r in search_results:
            priority = MEMORY_TYPE_CONFIGS.get(
                getattr(r.entry, "memory_type", "episodic"), {}
            ).get("recall_priority", 1.0)
            r.score *= priority

        # Re-sort after recall_priority adjustment
        search_results.sort(key=lambda r: r.score, reverse=True)

        # ── v4.1 Tier 1: Co-occurrence semantic boost ────────────────────
        if self.cooccurrence_boost_weight > 0 and self._cooccurrence.vocab_size > 0:
            for r in search_results:
                cooc_boost = self._cooccurrence.boost_score(
                    query=query,
                    entry_content=r.entry.content,
                    boost_factor=self.cooccurrence_boost_weight,
                )
                # Tier 3: PPMI additive boost (no-op if ppmi.json not present)
                ppmi_boost = self._ppmi.boost_score(
                    query=query,
                    entry_content=r.entry.content,
                    boost_factor=0.15,
                )
                r.relevance = min(r.relevance + cooc_boost + ppmi_boost, 1.0)
                r.score = r.relevance
            search_results.sort(key=lambda r: r.score, reverse=True)

        # ── v4.1 Tier 2: Optional semantic rerank ────────────────────────
        if self._semantic_engine is not None and search_results:
            try:
                search_results = self._semantic_engine.rerank(
                    query=query,
                    candidates=search_results,
                    top_k=limit,
                    cooccurrence_index=self._cooccurrence,
                )
            except Exception:
                search_results = search_results[:limit]  # Fallback: boosted BM25
        else:
            search_results = search_results[:limit]

        # Reinforce accessed memories & record access counts
        for r in search_results:
            self.decay.reinforce(r.entry)
            self._record_access(r.entry.hash)   # Sprint 11 access tracking

        # Apply access-count boost and re-sort (Sprint 11)
        if search_results:
            boosted = []
            for r in search_results:
                boost = self._access_tracker.boost_score(r.entry.hash)
                boosted.append((r, r.score * boost))
            boosted.sort(key=lambda x: x[1], reverse=True)
            search_results = [r for r, _ in boosted]

        if explain:
            self._perf.record_search((time.monotonic() - _t0) * 1000)
            if self._read_cache is not None:
                self._read_cache.put(cache_key, search_results)
            return search_results

        # Backward compat: return MemoryEntry with search-time confidence
        # Uses copy to avoid mutating the canonical entry in the index
        entries = []
        for r in search_results:
            entry_copy = copy.copy(r.entry)
            entry_copy.confidence = r.relevance
            entries.append(entry_copy)

        elapsed_ms = (time.monotonic() - _t0) * 1000
        self._perf.record_search(elapsed_ms)
        if self._read_cache is not None:
            self._read_cache.put(cache_key, entries)
        return entries
    
    def _search_indexed(self, 
                       query: str,
                       limit: int,
                       tags: Optional[List[str]],
                       tag_mode: str,
                       date_range: Optional[Tuple[str, str]],
                       use_decay: bool) -> List[MemoryEntry]:
        """Fast search using indexes."""
        
        # Get results from indexes
        indexed_results = self.index_manager.search(
            query=query,
            tags=tags,
            tag_mode=tag_mode,
            date_range=date_range,
            limit=limit * 2  # Get more for decay scoring
        )
        
        if not indexed_results:
            return []
        
        # Convert hash results to memory objects
        hash_to_memory = {m.hash: m for m in self.memories}
        results = []
        
        for memory_hash, base_score in indexed_results:
            if memory_hash in hash_to_memory:
                memory = hash_to_memory[memory_hash]
                
                # Apply decay scoring if requested
                if use_decay:
                    decay_score = self.decay.score(memory)
                    final_score = base_score * (0.5 + decay_score)
                    # Reinforce memory (simulate access)
                    self.decay.reinforce(memory)
                else:
                    final_score = base_score
                
                results.append((memory, final_score))
        
        # Sort by final score and return
        results.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, score in results[:limit]]
    
    def _search_legacy(self, query: str, limit: int, use_decay: bool) -> List[MemoryEntry]:
        """Fallback to legacy search method."""
        import re
        
        query_lower = query.lower()
        results = []
        
        for memory in self.memories:
            content_lower = memory.content.lower()
            
            # Simple text matching
            if query_lower in content_lower:
                if use_decay:
                    decay_score = self.decay.score(memory)
                    self.decay.reinforce(memory)
                    score = decay_score
                else:
                    score = 1.0
                
                results.append((memory, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, score in results[:limit]]

    # ── ingestion ───────────────────────────────────────────────────────

    def ingest(
        self,
        content: str,
        source: str = "inline",
        category: str = "general",
        memory_type: str = DEFAULT_TYPE,
        type_config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Ingest raw text. Returns count of new memories added.

        Sprint 2: accepts optional *memory_type* and *type_config* to attach
        structured type information to each entry.  Backward-compatible —
        existing callers that don't pass these get episodic defaults.

        Args:
            content: Raw text to ingest (multi-line supported).
            source: Source label (file path, "inline", etc.).
            category: Category label (general, strategic, etc.).
            memory_type: One of the canonical types or a custom string.
                Defaults to "episodic".
            type_config: For custom types, a dict with optional keys
                decay_multiplier, importance_boost, recall_priority.
            tags: Optional explicit tags to merge with auto-extracted tags.
        """
        cfg = get_type_config(memory_type, type_config)
        count = 0
        for i, line in enumerate(content.split("\n")):
            stripped = line.strip()
            if len(stripped) < 15 or stripped.startswith("```") or stripped == "---":
                continue

            # v4.1: enforce memory cap (Spark/Flame/Forge tier limits)
            if self.max_memories is not None and len(self.memories) >= self.max_memories:
                break

            entry = MemoryEntry(stripped, source, i + 1, category,
                                memory_type=memory_type)
            if entry.hash in self._hashes:
                continue

            # Process through gating system
            gate_priority = self.gating.classify(stripped)
            if gate_priority == "P3":  # Skip noise
                continue

            entry.tags = self._extract_tags(stripped)
            # Merge caller-supplied tags with auto-extracted tags (Bug fix #12)
            if tags:
                entry.tags = sorted(set(entry.tags) | set(tags))
            entry.sentiment = self.sentiment.analyze(stripped)

            # Apply type importance boost
            boost = cfg.get("importance_boost", 1.0)
            if boost != 1.0:
                entry.importance = min(entry.importance * boost, self.decay.max_score)

            # Store type_config for custom types
            if type_config:
                entry.type_metadata["_type_config"] = type_config

            self.memories.append(entry)
            self._hashes.add(entry.hash)
            self.search_engine.mark_dirty()

            if self.use_indexing and self.index_manager:
                self.index_manager.add_memory(entry)

            # Sprint 11 — WAL append
            self._wal.append(entry.to_dict())

            # v3.2 — Update local co-occurrence index (Tier 1 semantic)
            self._update_cooccurrence(stripped)

            count += 1

        # Invalidate read-cache on write
        if count and self._read_cache is not None:
            self._read_cache.invalidate()

        # Auto-flush if WAL threshold is reached
        if self._wal.should_flush():
            self.flush()

        # v4.1: trigger collection cycle if due and consented (non-blocking)
        if count:
            try:
                self._collection_agent.run_cycle()
            except Exception:
                pass  # Collection errors never block ingestion

        return count

    def ingest_file(self, file_path: str, category: str = "tactical") -> int:
        """Ingest a single file."""
        if not os.path.exists(file_path):
            return 0
        
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return self.ingest(content, source=file_path, category=category)
        except (OSError, UnicodeDecodeError):
            return 0

    def ingest_directory(self, dir_path: str, category: str = "tactical",
                        pattern: str = "*.md") -> int:
        """Ingest all matching files in a directory."""
        if not os.path.exists(dir_path):
            return 0
        
        total = 0
        for path in Path(dir_path).glob(pattern):
            if path.is_file():
                total += self.ingest_file(str(path), category)
        return total

    # ── Sprint 2: typed ingest methods ──────────────────────────────────

    def ingest_mistake(
        self,
        what_happened: str,
        correction: str,
        root_cause: Optional[str] = None,
        severity: str = "medium",
        tags: Optional[List[str]] = None,
        source: str = "mistake",
    ) -> Optional[MemoryEntry]:
        """Ingest a structured mistake memory.

        Mistakes receive 2× importance, 10× half-life, and are surfaced
        proactively in context packets via the "Known Pitfalls" section.

        Args:
            what_happened: Brief description of what went wrong.
            correction: What should be done instead.
            root_cause: Why it happened (optional but strongly recommended).
            severity: "high", "medium", or "low" (default "medium").
            tags: Extra tags to attach.
            source: Source label (default "mistake").

        Returns:
            The created MemoryEntry, or None if duplicate.
        """
        if severity not in SEVERITY_LEVELS:
            severity = "medium"

        content = format_mistake_content(what_happened, correction, root_cause, severity)
        cfg = MEMORY_TYPE_CONFIGS["mistake"]

        entry = MemoryEntry(content, source, 0, "mistake", memory_type="mistake")
        if entry.hash in self._hashes:
            return None

        entry.importance = min(1.0 * cfg["importance_boost"], self.decay.max_score)
        entry.tags = list(set(["mistake", f"severity:{severity}"] + (tags or [])))
        entry.sentiment = self.sentiment.analyze(content)
        entry.type_metadata = {
            "what_happened": what_happened,
            "correction": correction,
            "root_cause": root_cause,
            "severity": severity,
        }

        self.memories.append(entry)
        self._hashes.add(entry.hash)
        self.search_engine.mark_dirty()

        if self.use_indexing and self.index_manager:
            self.index_manager.add_memory(entry)

        # Sprint 11 — WAL
        self._wal.append(entry.to_dict())
        if self._read_cache is not None:
            self._read_cache.invalidate()
        if self._wal.should_flush():
            self.flush()

        return entry

    def ingest_fact(
        self,
        content: str,
        source: str = "fact",
        tags: Optional[List[str]] = None,
        category: str = "general",
    ) -> int:
        """Ingest a fact-type memory (normal decay, high recall priority)."""
        return self._ingest_typed(
            content, source, category, "fact", tags=tags
        )

    def ingest_preference(
        self,
        content: str,
        source: str = "preference",
        tags: Optional[List[str]] = None,
        category: str = "general",
    ) -> int:
        """Ingest a preference-type memory (3× slower decay, high context-matched recall)."""
        return self._ingest_typed(
            content, source, category, "preference", tags=tags
        )

    def ingest_procedure(
        self,
        content: str,
        source: str = "procedure",
        tags: Optional[List[str]] = None,
        category: str = "general",
    ) -> int:
        """Ingest a procedure-type memory (3× slower decay, high task-matched recall)."""
        return self._ingest_typed(
            content, source, category, "procedure", tags=tags
        )

    def _ingest_typed(
        self,
        content: str,
        source: str,
        category: str,
        memory_type: str,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Helper: ingest one or more lines with a specific memory type.

        Feature 2 fix: applies the same P0-P3 input gating that ``ingest()``
        uses so that P3 (ephemeral) content is never silently stored via the
        typed-ingest helpers.
        """
        cfg = get_type_config(memory_type)
        count = 0
        for i, line in enumerate(content.split("\n")):
            stripped = line.strip()
            if len(stripped) < 15 or stripped.startswith("```") or stripped == "---":
                continue

            # Feature 2: apply gating — drop P3 noise
            if not self.gating.should_store(stripped):
                continue

            entry = MemoryEntry(stripped, source, i + 1, category,
                                memory_type=memory_type)
            if entry.hash in self._hashes:
                continue

            entry.importance = min(1.0 * cfg.get("importance_boost", 1.0),
                                   self.decay.max_score)
            entry.tags = list(set([memory_type] + self._extract_tags(stripped) + (tags or [])))
            entry.sentiment = self.sentiment.analyze(stripped)

            self.memories.append(entry)
            self._hashes.add(entry.hash)
            self.search_engine.mark_dirty()

            if self.use_indexing and self.index_manager:
                self.index_manager.add_memory(entry)

            # Sprint 11 — WAL
            self._wal.append(entry.to_dict())
            count += 1

        if count:
            if self._read_cache is not None:
                self._read_cache.invalidate()
            if self._wal.should_flush():
                self.flush()

        return count

    # ── analysis & synthesis ────────────────────────────────────────────

    def analyze(self, query: str, limit: int = 10) -> Dict:
        """Analyze memories related to a query."""
        memories = self.search(query, limit=limit)
        if not memories:
            return {"status": "no_results", "query": query}
        
        # Basic analysis (can be enhanced)
        categories = defaultdict(int)
        sentiment_scores = []
        date_range = []
        
        for memory in memories:
            if memory.category:
                categories[memory.category] += 1
            if memory.sentiment and 'compound' in memory.sentiment:
                sentiment_scores.append(memory.sentiment['compound'])
            date_range.append(memory.created[:10])
        
        return {
            "status": "success",
            "query": query,
            "total_memories": len(memories),
            "categories": dict(categories),
            "avg_sentiment": sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            "date_range": [min(date_range), max(date_range)] if date_range else [],
            "sample_memories": [m.content[:100] for m in memories[:3]]
        }

    def synthesize_knowledge(self, topic: str, limit: int = 20) -> Optional[str]:
        """Generate knowledge synthesis using the KnowledgeSynthesizer."""
        memories = self.search(topic, limit=limit)
        if not memories:
            return None
        
        memory_texts = [m.content for m in memories]
        return self.synthesis.synthesize(memory_texts, topic)

    # ── context packets ──────────────────────────────────────────────────

    def build_context_packet(
        self,
        task: str,
        tags: Optional[List[str]] = None,
        category: str = None,
        environment: Optional[Dict[str, str]] = None,
        instructions: Optional[List[str]] = None,
        max_memories: int = 15,
        max_tokens: int = 4000,
        min_relevance: float = 0.1,
        include_mistakes: bool = True,     # Sprint 2
        max_pitfalls: int = 5,             # Sprint 2
    ) -> "ContextPacket":
        """Build a context packet for sub-agent spawning.

        Searches the memory store for relevant context and packages it
        into a structured ContextPacket that can be injected into a
        sub-agent's prompt at spawn time.

        Solves the "cold spawn" problem: sub-agents start with zero
        context, leading to confident mistakes based on incomplete
        information. A context packet gives them relevant onboarding.

        Args:
            task: Description of the sub-agent's task (used as search query).
            tags: Optional tag filters to narrow search.
            category: Optional category filter.
            environment: Key-value pairs (e.g. {"venv": "venv-svi", "python": "3.11"}).
            instructions: Explicit constraints for the sub-agent.
            max_memories: Maximum memories to include (default 15).
            max_tokens: Token budget for rendered output (default 4000).
            min_relevance: Minimum relevance score (default 0.1).

        Returns:
            ContextPacket with .render(), .to_dict(), .trim() methods.

        Example::

            packet = mem.build_context_packet(
                task="Verify antaris-guard installation",
                environment={"venv": "venv-svi"},
                instructions=["Check the venv, not global pip"],
            )
            prompt = f"{packet.render()}\\n\\nYOUR TASK: verify installation"
        """
        builder = ContextPacketBuilder(self)
        return builder.build(
            task=task,
            tags=tags,
            category=category,
            environment=environment,
            instructions=instructions,
            max_memories=max_memories,
            max_tokens=max_tokens,
            min_relevance=min_relevance,
            include_mistakes=include_mistakes,
            max_pitfalls=max_pitfalls,
        )

    def build_context_packet_multi(
        self,
        task: str,
        queries: List[str],
        environment: Optional[Dict[str, str]] = None,
        instructions: Optional[List[str]] = None,
        max_memories: int = 15,
        max_tokens: int = 4000,
        min_relevance: float = 0.1,
    ) -> "ContextPacket":
        """Build a context packet from multiple search queries.

        Useful when a task spans multiple topics. Deduplicates results
        across queries and merges into a single packet.

        Args:
            task: Overall task description.
            queries: List of search queries to run.
            environment: Optional environment context.
            instructions: Optional constraints.
            max_memories: Max total memories.
            max_tokens: Token budget.
            min_relevance: Minimum relevance threshold.

        Returns:
            Merged ContextPacket with deduplicated results.
        """
        builder = ContextPacketBuilder(self)
        return builder.build_multi(
            task=task,
            queries=queries,
            environment=environment,
            instructions=instructions,
            max_memories=max_memories,
            max_tokens=max_tokens,
            min_relevance=min_relevance,
        )

    # ── utility methods ─────────────────────────────────────────────────

    # ── Sprint 11: access pattern learning ──────────────────────────────

    def _record_access(self, entry_id: str) -> None:
        """Increment the access count for a memory entry.

        Called automatically by ``search()`` for every returned result.
        Persisted to ``access_counts.json`` on the next ``flush()``.
        """
        self._access_tracker.record_access(entry_id)

    def get_hot_entries(self, top_n: int = 10) -> List[MemoryEntry]:
        """Return the *top_n* most frequently accessed memory entries.

        Useful for pre-loading important context into packets.

        Args:
            top_n: Number of hot entries to return.

        Returns:
            List of MemoryEntry objects, ordered by access count descending.
        """
        top_ids = self._access_tracker.get_top(top_n)
        hash_to_entry = {m.hash: m for m in self.memories}
        result = []
        for entry_id, _count in top_ids:
            if entry_id in hash_to_entry:
                result.append(hash_to_entry[entry_id])
        return result

    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from content."""
        tags = []
        content_lower = content.lower()
        
        # Extract explicit tags (@tag format)
        explicit_tags = re.findall(r'@([a-zA-Z][a-zA-Z0-9_-]*)', content)
        tags.extend(explicit_tags)
        
        # Auto-tag based on terms
        for term in self._tag_terms:
            if term.lower() in content_lower:
                tags.append(term)
        
        return list(set(tags))  # Remove duplicates

    def compact(self) -> Dict:
        """Compact memory storage: remove duplicates, expire stale entries, save.

        Sprint 11: returns a richer result dict with shard counts and timing.

        Returns::

            {
                "shards_before": int,
                "shards_after": int,
                "entries_before": int,
                "entries_after": int,
                "space_freed_mb": float,
                "duration_ms": float,
                # backward-compat keys still present:
                "original_count": int,
                "final_count": int,
                "removed_count": int,
            }
        """
        _t0 = time.monotonic()

        # Snapshot shard count and disk usage before compaction
        shards_before = 0
        disk_before = 0.0
        if self.use_sharding and self.shard_manager:
            shards_dir = os.path.join(self.workspace, "shards")
            if os.path.isdir(shards_dir):
                shards_before = len([
                    f for f in os.listdir(shards_dir) if f.endswith(".json")
                ])
                disk_before = sum(
                    os.path.getsize(os.path.join(shards_dir, f))
                    for f in os.listdir(shards_dir) if f.endswith(".json")
                ) / (1024 * 1024)

        entries_before = len(self.memories)

        # Remove exact duplicates (by hash)
        seen_hashes: set = set()
        unique_memories = []
        for memory in self.memories:
            if memory.hash not in seen_hashes:
                unique_memories.append(memory)
                seen_hashes.add(memory.hash)

        # Apply decay-based forgetting: drop entries whose decay score is 0
        # (effectively dead entries that will never be recalled)
        retained_memories = [
            m for m in unique_memories
            if self.decay.score(m) > 0.0
        ]

        # Use consolidation to detect near-duplicate pairs; keep the
        # higher-confidence entry from each pair.
        dup_pairs = self.consolidation.find_duplicates(retained_memories)
        hashes_to_drop: set = set()
        for mem_a, mem_b in dup_pairs:
            # Drop whichever has lower confidence (or the second if tied)
            loser = mem_b if mem_a.confidence >= mem_b.confidence else mem_a
            hashes_to_drop.add(loser.hash)
        consolidated_memories = [
            m for m in retained_memories if m.hash not in hashes_to_drop
        ]

        self.memories = consolidated_memories
        self._hashes = {m.hash for m in self.memories}
        self.search_engine.mark_dirty()

        # Invalidate cache — data has changed
        if self._read_cache is not None:
            self._read_cache.invalidate()

        # Rebuild indexes after compaction
        if self.use_indexing and self.index_manager:
            self.index_manager.rebuild_indexes(self.memories)

        # Flush to disk so shards reflect compacted state
        self.save()
        self._wal.clear()   # WAL is also stale now
        self._perf.record_compaction()

        # Snapshot shard count and disk usage after compaction
        shards_after = 0
        disk_after = 0.0
        if self.use_sharding and self.shard_manager:
            shards_dir = os.path.join(self.workspace, "shards")
            if os.path.isdir(shards_dir):
                shards_after = len([
                    f for f in os.listdir(shards_dir) if f.endswith(".json")
                ])
                disk_after = sum(
                    os.path.getsize(os.path.join(shards_dir, f))
                    for f in os.listdir(shards_dir) if f.endswith(".json")
                ) / (1024 * 1024)

        entries_after = len(self.memories)
        duration_ms = round((time.monotonic() - _t0) * 1000, 1)

        return {
            # Sprint 11 spec keys
            "shards_before": shards_before,
            "shards_after": shards_after,
            "entries_before": entries_before,
            "entries_after": entries_after,
            "space_freed_mb": round(max(disk_before - disk_after, 0.0), 3),
            "duration_ms": duration_ms,
            # Backward-compatible keys
            "original_count": entries_before,
            "final_count": entries_after,
            "removed_count": entries_before - entries_after,
        }

    # ── system management ───────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get comprehensive system statistics."""
        from . import __version__ as _pkg_version  # Bug fix: add package version to stats
        stats = {
            "schema_version": "0.4.0",    # storage format version (used by migration logic)
            "package_version": _pkg_version,  # antaris-memory package version
            "total_memories": len(self.memories),
            "workspace": self.workspace,
            "features": {
                "sharding": self.use_sharding,
                "indexing": self.use_indexing
            }
        }
        
        if self.use_sharding and self.shard_manager:
            stats["sharding"] = self.shard_manager.index.get_stats()
        
        if self.use_indexing and self.index_manager:
            stats["indexing"] = self.index_manager.get_combined_stats()
        
        # Memory distribution by category
        categories = defaultdict(int)
        for memory in self.memories:
            categories[memory.category or "uncategorized"] += 1
        stats["categories"] = dict(categories)
        
        return stats

    def migrate_to_v4(self) -> Dict:
        """Manually trigger migration to v0.4 format."""
        return self.migration_manager.migrate("0.4.0")
    
    def rollback_migration(self) -> Dict:
        """Rollback the most recent migration."""
        return self.migration_manager.rollback()
    
    def validate_data(self) -> Dict:
        """Validate the current data schema."""
        return self.migration_manager.validate_schema()
    
    def rebuild_indexes(self):
        """Rebuild all search indexes from current memories."""
        if self.use_indexing and self.index_manager:
            self.index_manager.rebuild_indexes(self.memories)
            self.index_manager.save_all_indexes()
    
    def get_migration_history(self) -> List[Dict]:
        """Get history of applied migrations."""
        return self.migration_manager.get_migration_history()

    # ── Backward-compatible methods from v0.3 ──────────────────────

    def stats(self) -> Dict:
        """System statistics including Sprint 11 performance metrics.

        Backward-compatible — all v0.3/v0.4 keys are preserved.  New Sprint 11
        keys are added under their own names.

        Returns::

            {
                # v0.3 compat keys
                "total": int,
                "avg_score": float,
                "archive_candidates": int,
                "sentiments": dict,
                "avg_confidence": float,
                "categories": dict,
                # Sprint 11 performance keys
                "total_entries": int,
                "total_shards": int,
                "search_count": int,
                "avg_search_time_ms": float,
                "cache_hit_rate": float,
                "wal_pending_entries": int,
                "last_compaction": str | None,
                "disk_usage_mb": float,
                "memory_usage_mb": float,
            }
        """
        s = self.get_stats()
        now = datetime.now(timezone.utc)
        scores = [self.decay.score(m, now) for m in self.memories]
        sentiments: Dict[str, int] = defaultdict(int)
        for m in self.memories:
            dom = self.sentiment.dominant(m.sentiment)
            if dom:
                sentiments[dom] += 1
        # v0.3 compat
        s["total"] = len(self.memories)
        s["avg_score"] = round(sum(scores) / max(len(scores), 1), 4)
        s["archive_candidates"] = sum(1 for sc in scores if sc < self.decay.archive_threshold)
        s["sentiments"] = dict(sentiments)
        s["avg_confidence"] = round(
            sum(m.confidence for m in self.memories) / max(len(self.memories), 1), 4
        )

        # Sprint 11 performance metrics
        shards_count = 0
        disk_mb = 0.0
        if self.use_sharding and self.shard_manager:
            shards_dir = os.path.join(self.workspace, "shards")
            if os.path.isdir(shards_dir):
                shard_files = [
                    os.path.join(shards_dir, f)
                    for f in os.listdir(shards_dir) if f.endswith(".json")
                ]
                shards_count = len(shard_files)
                disk_mb = sum(os.path.getsize(p) for p in shard_files) / (1024 * 1024)

        import sys
        mem_mb = sys.getsizeof(self.memories) / (1024 * 1024)

        s["total_entries"] = len(self.memories)
        s["total_shards"] = shards_count
        s["search_count"] = self._perf.search_count
        s["avg_search_time_ms"] = self._perf.avg_search_time_ms
        s["cache_hit_rate"] = (
            round(self._read_cache.hit_rate, 4) if self._read_cache is not None else 0.0
        )
        s["wal_pending_entries"] = self._wal.pending_count()
        s["last_compaction"] = self._perf.last_compaction
        s["disk_usage_mb"] = round(disk_mb, 3)
        s["memory_usage_mb"] = round(mem_mb, 3)
        return s

    def ingest_with_gating(self, content: str, source: str = "inline",
                           context: Dict = None) -> int:
        """Ingest with P0-P3 input classification. P3 content is dropped.
        
        Multi-line content is split and each line classified independently.
        """
        count = 0
        for i, line in enumerate(content.split("\n")):
            stripped = line.strip()
            # Bug fix: align minimum length with ingest() (was 5, now 15)
            if len(stripped) < 15:
                continue
            gate_context = context or {}
            gate_context.update({"source": source, "line": i + 1})
            routing = self.gating.route(stripped, gate_context)
            if not routing["store"]:
                continue
            category = routing.get("category", "tactical")
            count += self.ingest(stripped, source=source, category=category)
        return count

    def on_date(self, date: str) -> List[MemoryEntry]:
        """Get memories from a specific date."""
        return self.temporal.on_date(self.memories, date)

    def between(self, start: str, end: str) -> List[MemoryEntry]:
        """Get memories between two dates."""
        return self.temporal.between(self.memories, start, end)

    def narrative(self, topic: str = None) -> str:
        """Build a narrative from memories, optionally filtered by topic."""
        return self.temporal.narrative(self.memories, topic=topic)

    def forget(self, topic: str = None, entity: str = None,
               before_date: str = None) -> Dict:
        """Selectively forget memories with audit trail.

        Bug fix (Bug 1): ForgettingEngine has no ``forget()`` method; instead
        it exposes ``forget_topic()``, ``forget_entity()``, and
        ``forget_before()``.  This method now delegates correctly and supports
        combining multiple criteria (applied in sequence).

        Args:
            topic: Remove all memories containing this topic string.
            entity: Remove all memories mentioning this entity.
            before_date: Remove all memories created before this date (YYYY-MM-DD).

        Returns:
            Dict with keys ``removed`` (list[MemoryEntry]) and ``audit`` (dict).
        """
        if topic is None and entity is None and before_date is None:
            return {"removed": [], "audit": self.forgetting.audit_log([])}

        current = list(self.memories)
        all_forgotten: List = []

        if topic is not None:
            current, forgotten = self.forgetting.forget_topic(current, topic)
            all_forgotten.extend(forgotten)

        if entity is not None:
            current, forgotten = self.forgetting.forget_entity(current, entity)
            all_forgotten.extend(forgotten)

        if before_date is not None:
            current, forgotten = self.forgetting.forget_before(current, before_date)
            all_forgotten.extend(forgotten)

        # Deduplicate (a memory might match multiple criteria)
        seen: set = set()
        unique_forgotten = []
        for m in all_forgotten:
            if m.hash not in seen:
                unique_forgotten.append(m)
                seen.add(m.hash)

        result = {
            "removed": unique_forgotten,
            "audit": self.forgetting.audit_log(unique_forgotten),
        }

        if unique_forgotten:
            removed_set = {m.hash for m in unique_forgotten}
            # Bug fix: `current` is already the post-forget filtered list from
            # forget_topic/forget_entity/forget_before — no need to re-filter.
            self.memories = current
            self._hashes -= removed_set
            self.search_engine.mark_dirty()
            if self._read_cache is not None:
                self._read_cache.invalidate()
            # Log to audit
            for m in unique_forgotten:
                self._append_audit({"action": "forget", "hash": m.hash,
                                    "content_preview": m.content[:50],
                                    "timestamp": datetime.now(timezone.utc).isoformat()})

        return result

    def reindex(self) -> None:
        """Force a full search index rebuild."""
        self.search_engine.reindex(self.memories)

    def consolidate(self) -> Dict:
        """Run consolidation: dedup, clustering, contradiction detection."""
        return self.consolidation.run(self.memories)

    def compress_old(self, days: int = 7) -> list:
        """Compress memories older than N days."""
        mem_dir = os.path.join(self.workspace, "memory")
        return self.compression.compress_old_files(mem_dir, days)

    def synthesize(self, research_results: Dict = None) -> Dict:
        """Integrate research findings into memory."""
        report = self.synthesis.run_cycle(self.memories, research_results=research_results)
        # Add synthesized entries to memory store
        if report.get("synthesized_entries", 0) > 0 and "new_entries" in report:
            for entry_dict in report["new_entries"]:
                entry = MemoryEntry.from_dict(entry_dict)
                if "synthesis" not in entry.tags:
                    entry.tags.append("synthesis")
                self.memories.append(entry)
                if self.use_indexing and self.index_manager:
                    self.index_manager.add_memory(entry)
        return report

    def research_suggestions(self, limit: int = 5) -> List[Dict]:
        """Get suggestions for knowledge gaps."""
        return self.synthesis.suggest_research_topics(self.memories, limit=limit)

    # ── Feature 3: Export / Import API ─────────────────────────────────

    def export(self, path: str, fmt: str = "json") -> None:
        """Export all memories to a JSON file.

        Args:
            path: Destination file path.
            fmt: Format (currently only "json" is supported).
        """
        data = [entry.to_dict() for entry in self.memories]
        atomic_write_json(path, data)

    def import_memories(self, path: str) -> int:
        """Import memories from a previously exported JSON file.

        Deduplicates against existing entries by hash so re-importing the
        same file is safe (idempotent).

        Args:
            path: Path to the JSON array file produced by ``export()``.

        Returns:
            Number of new entries added.
        """
        with open(path, encoding="utf-8") as fh:
            entries_data = json.load(fh)

        count = 0
        for entry_dict in entries_data:
            try:
                entry = MemoryEntry.from_dict(entry_dict)
            except Exception:
                continue  # skip malformed entries
            if entry.hash in self._hashes:
                continue
            self.memories.append(entry)
            self._hashes.add(entry.hash)
            count += 1

        if count:
            self.search_engine.mark_dirty()
            if self._read_cache is not None:
                self._read_cache.invalidate()
            self.save()

        return count

    # ── shared pool integration ──────────────────────────────────────────

    def enable_shared_pool(
        self,
        pool_dir: str,
        pool_name: str = "default",
        agent_id: str = None,
        role: str = "write",
        load_existing: bool = True,
    ) -> "SharedMemoryPool":
        """Attach a SharedMemoryPool to this MemorySystem instance.

        Enables multi-agent memory sharing. The pool is created (or loaded)
        at pool_dir. This agent is registered with the given role.

        Args:
            pool_dir: Directory for the shared pool files.
            pool_name: Pool identifier (default "default").
            agent_id: This agent's ID in the pool. Defaults to the workspace basename.
            role: Access role — "read", "write", or "admin" (default "write").
            load_existing: If True, load an existing pool from disk (default True).

        Returns:
            The attached SharedMemoryPool instance.
        """
        if agent_id is None:
            agent_id = os.path.basename(self.workspace.rstrip("/"))

        pool = SharedMemoryPool(pool_dir, pool_name)
        if load_existing:
            pool.load()

        # Register if not already registered
        if agent_id not in pool.permissions:
            pool.register_agent(agent_id, role)

        self._shared_pool = pool
        self._shared_agent_id = agent_id
        return pool

    def get_shared_pool(self) -> Optional["SharedMemoryPool"]:
        """Return the attached SharedMemoryPool, or None if not enabled."""
        return self._shared_pool

    def shared_write(
        self,
        content: str,
        namespace: str = "shared",
        category: str = "general",
        metadata: dict = None,
    ) -> Optional[object]:
        """Write a memory to the attached shared pool.

        Raises RuntimeError if no shared pool is attached (call enable_shared_pool first).
        """
        if self._shared_pool is None:
            raise RuntimeError("No shared pool attached. Call enable_shared_pool() first.")
        return self._shared_pool.write(
            self._shared_agent_id, content, namespace, category, metadata
        )

    def shared_search(
        self,
        query: str,
        namespace: str = None,
        limit: int = 20,
        instrumentation_context=None,
    ) -> list:
        """Search the attached shared pool.

        If instrumentation_context is provided (a SearchContext), populates
        retrieved_memory_ids for later mark_used() calls.

        Raises RuntimeError if no shared pool is attached.
        """
        if self._shared_pool is None:
            raise RuntimeError("No shared pool attached. Call enable_shared_pool() first.")

        if hasattr(self._shared_pool, 'search_with_context') and instrumentation_context is not None:
            results, _ = self._shared_pool.search_with_context(
                self._shared_agent_id, query, instrumentation_context, limit, namespace
            )
        else:
            results = self._shared_pool.read(
                self._shared_agent_id, query, namespace, limit
            )
        return results

    def shared_mark_used(
        self,
        memory_ids: list,
        boost_factor: float = 1.3,
    ) -> int:
        """Signal that retrieved shared memories were used in a response.

        Triggers cross-agent relevance boosting for memories written by other agents.

        Returns:
            Number of memories boosted.
        """
        if self._shared_pool is None:
            raise RuntimeError("No shared pool attached.")
        return self._shared_pool.mark_used_batch(
            self._shared_agent_id, memory_ids, boost_factor
        )

    def _append_audit(self, entry: Dict) -> None:
        """Append an entry to the audit log."""
        audit = []
        if os.path.exists(self.audit_path):
            with open(self.audit_path) as f:
                audit = json.load(f)
        audit.append(entry)
        atomic_write_json(self.audit_path, audit)

    # ── v3.2 Instrumentation ──────────────────────────────────────────────────

    def mark_retrieved(self, memory_ids: List[str], context) -> None:
        """Record that a set of memories were retrieved for a query.

        Called automatically when instrumentation_context is passed to search().
        Logs the retrieval event to the instrumentation log for later analysis.

        Args:
            memory_ids: List of memory hashes that were retrieved.
            context: SearchContext object for this search request.
        """
        context.retrieved_memory_ids = list(memory_ids)
        event = {
            "event": "retrieved",
            "request_id": context.request_id,
            "query": context.query,
            "memory_ids": memory_ids,
            "count": len(memory_ids),
            "timestamp": context.timestamp,
            "context_id": context.context_id,
        }
        try:
            with open(self._instrumentation_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass  # Instrumentation failures are non-fatal

    def mark_used(self, memory_ids: List[str], context=None) -> int:
        """Signal that specific memories were used in the agent's response.

        Boosts the relevance of used memories and resets their decay age.
        Optionally logs a usage signal if opt_in_to_aggregation is True.

        Args:
            memory_ids: List of memory hashes that appeared in the response.
            context: Optional SearchContext for linking to the retrieval event.

        Returns:
            Number of memories successfully boosted.
        """
        if context is not None:
            context.used_memory_ids = list(memory_ids)

        boosted = 0
        for memory_id in memory_ids:
            if self.boost_relevance(memory_id):
                boosted += 1

        # Log usage event
        event = {
            "event": "used",
            "memory_ids": memory_ids,
            "count": len(memory_ids),
            "timestamp": time.time(),
            "request_id": context.request_id if context else None,
        }
        try:
            with open(self._instrumentation_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass

        # Log to audit logger if available
        if context is not None and hasattr(context, 'audit_logger') and context.audit_logger is not None:
            try:
                for memory_id in memory_ids:
                    context.audit_logger.log("recall", {
                        "memory_id": memory_id,
                        "context_id": context.context_id,
                        "request_id": context.request_id,
                        "session_id": context.session_id,
                    })
            except Exception:
                # Audit logging failures are non-fatal
                pass

        # Report anonymized signal if user opted in
        if self.opt_in_to_aggregation and context is not None:
            from .instrumentation import build_usage_signal
            self.report_usage_signal(build_usage_signal(context))

        # Invalidate cache so boosted memories rank higher on next search
        if boosted and self._read_cache is not None:
            self._read_cache.invalidate()

        return boosted

    def boost_relevance(self, memory_id: str, multiplier: float = 1.5) -> bool:
        """Boost the importance/relevance of a memory by ID.

        Increases the memory's importance score and resets its decay age,
        making it surface higher in future searches.

        Args:
            memory_id: The memory hash to boost.
            multiplier: Importance multiplier (default 1.5 = 50% boost).

        Returns:
            True if the memory was found and boosted, False otherwise.
        """
        for entry in self.memories:
            if entry.hash == memory_id:
                entry.importance = min(entry.importance * multiplier, 10.0)
                # Reset decay: freshen the memory as if it was just accessed
                entry.last_accessed = datetime.now(timezone.utc).isoformat()
                entry.access_count += 1
                self.search_engine.mark_dirty()
                return True
        return False

    def report_usage_signal(self, data: Dict) -> None:
        """Append an anonymized usage signal for crowdsourced PPMI training.

        Only writes when opt_in_to_aggregation is True. Data contains only
        statistical information — never actual memory content.

        Args:
            data: Anonymized signal dict from build_usage_signal().
        """
        if not self.opt_in_to_aggregation:
            return
        try:
            with open(self._usage_signals_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except OSError:
            pass

    # ── v3.2 Co-occurrence Semantic Boost ──────────────────────────────────────

    def search_with_context(
        self,
        query: str,
        limit: int = 20,
        instrumentation_context=None,
        cooccurrence_boost: bool = True,
        **kwargs,
    ) -> tuple:
        """Search memories with optional instrumentation and co-occurrence boost.

        Returns (results, SearchContext) for use with mark_used().

        Args:
            query: Search query string.
            limit: Maximum results to return.
            instrumentation_context: Optional pre-built SearchContext.
                If None, one is created automatically.
            cooccurrence_boost: Whether to apply local co-occurrence re-ranking.
            **kwargs: Additional args forwarded to search() (explain, category, etc.)

        Returns:
            Tuple of (results, SearchContext).
        """
        from .instrumentation import SearchContext

        ctx = instrumentation_context or SearchContext(query=query)

        # Run BM25 search (fetch extra for re-ranking)
        fetch_limit = limit * 2 if cooccurrence_boost else limit
        results = self.search(query, limit=fetch_limit, explain=True, **kwargs)

        # Co-occurrence re-ranking (Tier 1 semantic boost)
        if cooccurrence_boost and self._cooccurrence.vocab_size > 0:
            for r in results:
                boost = self._cooccurrence.boost_score(query, r.entry.content)
                r.score += boost

            results.sort(key=lambda r: r.score, reverse=True)

        results = results[:limit]

        # Record retrieval for instrumentation
        retrieved_ids = [r.entry.hash for r in results]
        self.mark_retrieved(retrieved_ids, ctx)

        # Return plain entries by default (backward compat), explain if needed
        if kwargs.get("explain"):
            return results, ctx

        entries = []
        for r in results:
            entry_copy = copy.copy(r.entry)
            entry_copy.confidence = r.relevance
            entries.append(entry_copy)
        return entries, ctx

    def _update_cooccurrence(self, content: str) -> None:
        """Update the local co-occurrence index when a memory is stored.

        Note: does NOT save to disk on every call — deferred to flush() to
        avoid O(n) disk writes during bulk ingestion (e.g. ingest_directory).
        """
        self._cooccurrence.update_from_memory(content)


# Backward compatibility alias
MemorySystem = MemorySystemV4