# Changelog

## [3.3.2] - 2026-02-22

### Changed — Recovery Preset Refinement
- **Removed "full" preset** — pulling 7 days of all memories at startup is economically unsustainable (~$1000+ in tokens per compaction). Older memories are recalled naturally via search/decay when needed.
- **Updated presets:**
  - `"smart"` (default): 50 memories, 24h, cache injection (~5-10K tokens) — recommended for most use cases
  - `"minimal"`: 10 memories, session only, manual injection (~1-2K tokens) — for constrained environments
- Updated documentation and help text to explain natural recall for older memories
- Token cost estimates made explicit in all presets

## [3.3.1] - 2026-02-22

### Added — Recovery Configuration
- **RecoveryConfig** — user-configurable memory recovery after compaction/restart
  - Presets: `"smart"` (50 memories, 24h, all channels, cache), `"minimal"` (10, session, current, manual)
  - Options: `recovery_search_limit`, `recovery_time_window`, `recovery_channels`, `recovery_inject`
  - Full help text with examples: `RecoveryConfig.help()`
  - Serialization support: `.to_dict()`
- **RecoveryManager** — orchestrates memory recovery
  - `recover_memories()` — gather memories based on config
  - `get_cached_memories()` — access recovered memories
  - `inject_into_context()` — generate [MEMORY RESTORED] block
- 18 new tests for recovery

### Why
After compaction/gateway restart, agent context window is empty. Recovery bridges the gap by automatically gathering recent memories back into working cache, with configurable injection strategy. Smart mode balances context availability with token efficiency.

## [3.3.0] - 2026-02-22

### Added — Instrumentation & Shared Memory Pool
- **Instrumentation layer** (`instrumentation.py`) — collect metrics on ingest/search/decay
- **Co-occurrence Tier 1** (`cooccurrence.py`) — semantic clustering via word co-occurrence
- **Shared Memory Pool** (`shared.py` updates) — multi-agent memory coordination
- 388 tests (up from 289)

## [2.0.0] - 2026-02-18

### Added — Antaris Suite 2.0 GA
- **MCP Server** (`mcp_server.py`) — expose memory as MCP resources/tools via `create_server` (aliased as `create_mcp_server` in `__init__.py`)
- **Hybrid Semantic Search** — `set_embedding_fn(fn)` pluggable interface; BM25+cosine blending when embedding fn is set
- **Memory Types** — `MEMORY_TYPE_CONFIGS`, `get_type_config()`, per-type recall priority boost in BM25 scoring
- **Namespace Isolation** — `NamespacedMemory`, `NamespaceManager` for multi-tenant agent memory
- **Context Packets** — `ContextPacket`, `ContextPacketBuilder` for structured sub-agent context injection
- **LRU read cache** — Sprint 11 search caching with access-count boosting
- **Sprint 11 search** — BM25 IDF always computed over full corpus (filter-after-score; fixes IDF corruption bug)
- 289 unit tests (all passing)

## [1.1.0] - 2026-02-17

### Added
- **Context Packets** — structured memory injection for sub-agent spawning
  - `ContextPacket` data class with markdown/XML/JSON rendering
  - `ContextPacketBuilder` builds packets from BM25 search results
  - `mem.build_context_packet(task=...)` — single-query convenience method
  - `mem.build_context_packet_multi(task=..., queries=[...])` — multi-query with deduplication
  - Token budgeting via `.trim(max_tokens)` 
  - Full serialization roundtrip (`.to_dict()` / `.from_dict()`)
  - Environment and instruction injection for sub-agent context
- 26 new tests (104 total, up from 78)

### Why
Sub-agents spawn cold with zero context, leading to confident mistakes
based on incomplete information (e.g., checking global pip instead of a venv).
Context packets solve this by searching relevant memories and packaging them
as structured onboarding for the sub-agent's prompt.

## [1.0.1] - 2026-02-16

### Fixed (Review Polish)
- **Phrase matching** now token-sequence based (no substring false positives from punctuation)
- **Boost capping**: max one tag boost (1.2x) and one source boost (1.1x) per entry
- **Numeric tokens** kept when ≥ 3 chars ("500", "2026" are searchable)
- **Index staleness**: `mark_dirty()` wired into ingest, forget, and compact; `reindex()` on public API
- **Atomic writes** for legacy save and audit log (zero non-atomic JSON writes remain)
- **Corpus versioning**: search index rebuilds on version mismatch, not just count mismatch
- **Consolidate bug**: `core_v4.py` was calling nonexistent `.consolidate()` instead of `.run()`
- Clean `__init__.py` docstring and group headers (no stale version references)
- Locking scope clarified in README
- Benchmarks updated with Consolidate and Disk columns

## [1.0.0] - 2026-02-16

### Added (Search Engine)
- **BM25-inspired search** (`SearchEngine`): Proper relevance ranking with term frequency, inverse document frequency, field boosting, and length normalization
- **SearchResult** dataclass with `relevance` (0.0-1.0 normalized), `matched_terms`, and `explanation`
- **Stopword filtering**: 100+ English stopwords excluded from scoring for cleaner results
- **Exact phrase boost**: Token-sequence phrase match gets 1.5x score multiplier
- **Field boosting**: Tag matches (1.2x) and source matches (1.1x) improve ranking
- **Decay-weighted ranking**: Recent/frequently-accessed memories score higher
- **`explain=True` mode**: Returns SearchResult objects with full scoring breakdown
- **Index statistics**: `search_engine.stats()` returns vocab size, avg doc length, top terms
- 18 new search tests, 78 total

### Changed
- **Search confidence now varies by relevance** — top result scores 1.0, others normalized below. No more wall of 0.50 scores.
- Search engine auto-builds BM25 index on `load()` and rebuilds when corpus changes
- Both `core.py` (legacy) and `core_v4.py` (default) use the new search engine

### Why v1.0
This release marks production-readiness across all core features:
- Proper search ranking (v1.0: BM25)
- Concurrent writer safety (v0.5: locking + versioning)
- Production storage (v0.4: sharding + indexes)
- Multi-agent support (v0.3: shared pools)
- Input classification (v0.2: P0-P3 gating)
- Core primitives (v0.1: decay, sentiment, temporal, consolidation)

## [0.5.0] - 2026-02-16

### Added (Concurrency & Safety)
- **File Locking** (`FileLock`): Cross-platform directory-based lock using `os.mkdir()` — portable across POSIX and Windows, works on network filesystems, zero dependencies
  - Blocking and non-blocking acquisition modes
  - Configurable timeout with `LockTimeout` exception
  - Holder metadata (PID, timestamp) for debugging
  - Automatic stale lock detection and breaking (by age or dead PID)
  - Context manager support (`with FileLock(path): ...`)
- **Optimistic Conflict Detection** (`VersionTracker`): Detect when another process modifies a file between your read and write
  - `snapshot()` captures file mtime, size, and optional SHA-256 content hash
  - `check()` raises `ConflictError` if file changed since snapshot
  - `safe_update()` — read-modify-write with automatic retry on conflict
  - `FileVersion.is_current()` for quick staleness checks
- **Locked atomic writes**: `atomic_write_json()` now acquires a file lock by default (opt-out with `lock=False`)
- **`locked_read_json()`**: Read JSON files under lock to prevent torn reads
- 20 new tests (11 locking + 9 versioning), 60 total

### Enhanced
- **All JSON writes are now locked by default** — prevents lost updates from concurrent writers
- **Atomic writes + locking + versioning** form a complete concurrency safety stack:
  - Atomic writes prevent torn files
  - Locks prevent lost updates
  - Version tracking detects conflicts in read-heavy workloads

### Technical
- 3 new modules: `locking.py`, `versioning.py`, updated `utils.py`
- 4 new classes: `FileLock`, `LockTimeout`, `VersionTracker`, `ConflictError`, `FileVersion`
- Lock strategy: `os.mkdir()` (atomic on all platforms) with `holder.json` metadata
- Stale lock detection: age threshold (default 5 min) + dead PID check (POSIX `os.kill(pid, 0)`)
- Concurrent writer test: 4 threads × 50 iterations = 200 increments, zero lost updates

## [0.4.0] - 2026-02-15

### Added (Production Features)
- **Sharded Storage** (`ShardManager`): Split memories across multiple files by date/topic for better performance and scalability
- **Fast Search Indexes** (`IndexManager`): Full-text, tag-based, and date range indexes for sub-second search
- **Schema Migration System** (`MigrationManager`): Seamless migration from v0.2/v0.3 single-file format to v0.4 sharded format with rollback capability
- **Lightweight File-Based Indexes**: Text search index, tag index, and date index - all stored as transparent JSON files
- **Automatic Migration**: v0.2/v0.3 data is automatically migrated to v0.4 format on first load
- **Backward Compatibility**: Can still read and work with legacy single-file format
- **Production-Ready MemorySystem**: New `MemorySystemV4` class with enterprise features while maintaining same API
- **Index Rebuilding**: Rebuild corrupted indexes from existing data
- **Migration History**: Track all applied migrations with rollback information
- **Schema Validation**: Validate data integrity across format versions
- **Shard Compaction**: Merge small shards and split large ones for optimal performance

### Enhanced
- **Search Performance**: 10-100x faster search using pre-built indexes instead of scanning all memories
- **Storage Scalability**: Handle 10,000+ memories efficiently through sharding
- **Memory Usage**: Only load relevant shards into memory, not entire dataset
- **Concurrent Access**: Multiple processes can work on different shards simultaneously

### Technical
- 4 new core modules: `sharding.py`, `migration.py`, `indexing.py`, `core_v4.py`
- 11 new classes: `ShardManager`, `ShardKey`, `ShardIndex`, `MigrationManager`, `Migration`, `V2ToV4Migration`, `IndexManager`, `SearchIndex`, `TagIndex`, `DateIndex`  
- 34 new tests covering all v0.4 features
- Full API compatibility with v0.3 - existing code works unchanged

### Migration Notes
- First load after upgrading will automatically migrate v0.2/v0.3 data to v0.4 format
- Original data is backed up during migration
- Use `migration_manager.rollback()` to revert if needed
- New installs start directly in v0.4 format

## [0.3.0] - 2026-02-15

### Added
- **Multi-Agent Shared Memory** (`SharedMemoryPool`): Multiple agents share a memory space with role-based access controls
- **Agent Permissions** (`AgentPermission`): Read, write, admin roles with namespace-level isolation
- **Namespace Isolation**: Agents only see memories in namespaces they have access to
- **Cross-Agent Conflict Detection**: Flags contradictions when agents write conflicting memories
- **Conflict Resolution**: Resolve disputes with audit trail
- **Knowledge Propagation**: Share discoveries from one agent's namespace to another
- **Pool Statistics**: Per-agent, per-namespace memory counts and conflict tracking
- 18 new tests for shared memory (40 total)

## [0.2.0] - 2026-02-15

### Added
- **Input Gating (P0-P3)**: Classify content at intake as Critical, Operational, Contextual, or Ephemeral. P3 content is automatically filtered out.
- **Autonomous Knowledge Synthesis**: Identify knowledge gaps, suggest research topics, and integrate new findings into memory.
- **`ingest_with_gating()`**: Smart ingestion that auto-classifies and routes content.
- **`research_suggestions()`**: Get prioritized research topics based on memory gaps.
- **`synthesize()`**: Integrate external research with existing memories.
- **OpenClaw integration example** (`examples/openclaw_integration.py`)
- **LangChain integration example** (`examples/langchain_integration.py`)
- 13 new tests (22 total)

## [0.1.1] - 2026-02-15

### Changed
- GitHub org renamed to Antaris-Analytics
- Updated project URLs
- Cleaned up README comparison table for accuracy
- Removed build artifacts from repo, added .gitignore

## [0.1.0] - 2026-02-15

### Added
- Initial release
- `MemorySystem` with ingest, search, save/load
- Ebbinghaus decay curves with reinforcement on access
- Sentiment tagging (positive, negative, urgent, strategic, financial)
- Temporal reasoning (date queries, range queries, narratives)
- Confidence scoring and contradiction detection
- Memory compression for old files
- Selective forgetting with GDPR-ready audit trail
- Dream-state consolidation (duplicate detection, topic clustering)
- Zero external dependencies
