# Antaris Memory

**Production-ready file-based persistent memory for AI agents. Zero dependencies (core).**

Store, search, decay, and consolidate agent memories using only the Python standard library. Sharded storage for scalability, fast search indexes, namespace isolation, MCP server support, and automatic schema migration. No vector databases, no infrastructure, no API keys.

[![PyPI](https://img.shields.io/pypi/v/antaris-memory)](https://pypi.org/project/antaris-memory/)
[![Tests](https://github.com/Antaris-Analytics/antaris-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/Antaris-Analytics/antaris-memory/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

## What's New in v3.3.1

- **Recovery Configuration** — after compaction/restart, automatically gather recent memories back into working cache
  - Presets: `"smart"` (50 memories, 24h, all channels, cache injection), `"full"` (all memories, 7d, startup injection), `"minimal"` (10, session only, manual injection)
  - User-configurable: `recovery_search_limit`, `recovery_time_window`, `recovery_channels`, `recovery_inject`
  - Full help: `RecoveryConfig.help()`
  - Bridges context window gap after compaction without token bloat
- 18 new recovery tests (388 total)

## Earlier Versions

**v3.3.0** — Instrumentation layer, co-occurrence semantic clustering, shared memory pool
**v2.0.0** — MCP Server, hybrid semantic search, memory types, namespace isolation, context packets, LRU cache

See [CHANGELOG.md](CHANGELOG.md) for full version history.


## OpenClaw Integration

antaris-memory ships as a **native OpenClaw plugin** (`antaris-memory`). Install it once and your agent automatically recalls relevant memories before each turn and ingests new context after each turn — no code changes required.

```bash
# Install the OpenClaw plugin
cp -r antaris-memory-plugin/ ~/.openclaw/extensions/antaris-memory/
# Restart OpenClaw — hooks fire automatically
```

**What the plugin does:**
- `before_agent_start` — searches memory for relevant context, injects into agent prompt
- `agent_end` — ingests the turn into persistent memory

Also ships with an **MCP server** for integration with any MCP-compatible host:

```python
from antaris_memory import create_mcp_server  # pip install mcp
server = create_mcp_server(workspace="./memory")
```

## What It Does

- **Sharded storage** for production scalability (10,000+ memories, sub-second search)
- **Fast search indexes** (full-text, tags, dates) stored as transparent JSON files
- **Automatic schema migration** from single-file to sharded format with rollback
- **Multi-agent shared memory** pools with namespace isolation and access controls
- Retrieval weighted by **recency × importance × access frequency** ([Ebbinghaus-inspired](https://en.wikipedia.org/wiki/Forgetting_curve) decay)
- **Input gating** classifies incoming content by priority (P0–P3) and drops ephemeral noise at intake
- Detects contradictions between stored memories using deterministic rule-based comparison
- Runs fully offline — zero network calls, zero tokens, zero API keys

## What It Doesn't Do

- **Not a vector database** — no embeddings by default. Core search uses BM25 keyword ranking. Semantic search requires you to supply an embedding function (`set_embedding_fn(fn)`) — we never make that call for you.
- **Not a knowledge graph** — flat memory store with metadata indexing. No entity relationships or graph traversal.
- **Not semantic by default** — contradiction detection compares normalized statements using explicit conflict rules, not inference. It will not catch contradictions phrased differently.
- **Not LLM-dependent** — all operations are deterministic. No model calls, no prompt engineering.
- **Not infinitely scalable** — JSON file storage works well up to ~50,000 memories per workspace. Beyond that, you'll want a database. We're honest about this so you don't discover limits in production.

## Design Goals

| Goal | Rationale |
|------|-----------|
| Deterministic | Same input → same output. No model variance. |
| Offline | No network, no API keys, no phoning home. |
| Minimal surface area | One class (`MemorySystem`), obvious method names. |
| No hidden processes | Consolidation and synthesis run only when called. |
| Transparent storage | Plain JSON files. Inspect with any text editor. |

## Install

```bash
pip install antaris-memory
```

## Quick Start

```python
from antaris_memory import MemorySystem

mem = MemorySystem("./workspace", half_life=7.0)
mem.load()  # No-op on first run

# Store memories (typed ingestion)
mem.ingest("Decided to use PostgreSQL for the database.",
           source="meeting-notes", category="strategic", memory_type="episodic")
mem.ingest("API costs $500/month — too expensive.",
           source="review", category="operational")

# Typed helpers
mem.ingest_fact("PostgreSQL supports JSON natively")
mem.ingest_preference("User prefers concise explanations")
mem.ingest_mistake("Forgot to close DB connections in worker threads")
mem.ingest_procedure("Deploy: push to main → CI runs → auto-deploy to staging")

# Input gating — drops ephemeral noise (P3) before storage
mem.ingest_with_gating("Decided to switch to Redis for caching", source="chat")
mem.ingest_with_gating("thanks for the update!", source="chat")  # → dropped (P3)

# Search (BM25; hybrid BM25+cosine if embedding fn set)
for r in mem.search("database decision"):
    print(f"[{r.confidence:.2f}] {r.content}")
# → [1.00] Decided to use PostgreSQL for the database.

# Detailed search with score explanations
for r in mem.search("database decision", explain=True):
    print(f"[{r.relevance:.2f}] {r.content[:60]}  ({r.explanation})")

# Temporal queries
mem.on_date("2026-02-14")
mem.narrative(topic="database migration")

# Build a context packet for sub-agent injection
packet = mem.build_context_packet(
    task="Optimize database performance",
    max_memories=10,
    max_tokens=2000,
)
print(packet.render("markdown"))

# Selective deletion (GDPR-ready with audit trail)
mem.forget(entity="John Doe")
mem.forget(before_date="2025-01-01")

# Background consolidation
report = mem.consolidate()

mem.save()
```

## Hybrid Semantic Search

Plug in any embedding function to blend BM25 with cosine similarity:

```python
import openai

def my_embed(text: str) -> list[float]:
    resp = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

mem = MemorySystem("./workspace")
mem.load()
mem.set_embedding_fn(my_embed)  # BM25+cosine hybrid activates automatically

# Or use a local model — anything that returns a list of floats works
import ollama

def local_embed(text: str) -> list[float]:
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

mem.set_embedding_fn(local_embed)
```

When no embedding function is set, search uses BM25 only (zero API calls).

## Recovery (v3.3.1)

After compaction or gateway restart, automatically gather recent memories back into working cache:

```python
from antaris_memory import MemorySystem, RecoveryConfig

# Default smart recovery: 50 memories from last 24h, cache injection
mem = MemorySystem("./workspace", recovery_config=RecoveryConfig())
mem.load()
mem.recover_memories()  # Triggered automatically on init

# Or choose minimal mode for token efficiency
mem = MemorySystem(
    "./workspace",
    recovery_config=RecoveryConfig(recovery_mode="minimal")
)

# Or customize
mem = MemorySystem(
    "./workspace",
    recovery_config=RecoveryConfig(
        recovery_search_limit=100,
        recovery_time_window="24h",
        recovery_channels="all",
        recovery_inject="cache"  # Smart: store, don't inject
    )
)

# Access recovered memories
recovered = mem.recovery_manager.get_cached_memories()
context_block = mem.recovery_manager.inject_into_context()  # [MEMORY RESTORED] block
```

**Presets:**

| Mode | Search Limit | Time Window | Channels | Injection | Use Case |
|------|-------------|-------------|----------|-----------|----------|
| `smart` (default) | 50 | 24h | all | cache | Balanced: recent context + token efficiency (~5-10K tokens) |
| `minimal` | 10 | session | current | manual | Token-constrained environments (~1-2K tokens) |

Note: Older memories beyond the recovery window are recalled naturally via search and decay when needed — you don't need to load them all on startup.

For full options and help, call `RecoveryConfig.help()`.

## Memory Types

Store memories with type-specific recall boosts:

```python
mem.ingest("Deploy: push to main, CI runs, auto-deploy to staging",
           memory_type="procedural")   # High recall boost for how-to queries
mem.ingest_fact("PostgreSQL supports JSONB indexing")  # Semantic memory
mem.ingest_preference("User prefers Python examples")  # Preference memory
mem.ingest_mistake("Forgot to handle connection timeout")  # Mistake memory
mem.ingest_procedure("Run pytest from venv, not global pip")  # Procedure memory
```

| Type | Use for | Recall boost |
|------|---------|-------------|
| `episodic` | Events, decisions, meeting notes | Normal |
| `semantic` | Facts, concepts, general knowledge | Medium |
| `procedural` | How-to steps, runbooks | High |
| `preference` | User preferences, style notes | High |
| `mistake` | Errors to avoid, lessons learned | High |

## Namespace Isolation

Multi-tenant or multi-agent workspaces with hard boundaries:

```python
from antaris_memory import NamespacedMemory, NamespaceManager

manager = NamespaceManager("./workspace")

# Each agent gets its own isolated memory
agent_a = manager.create_namespace("agent-a")
agent_b = manager.create_namespace("agent-b")

# Or use NamespacedMemory directly
ns = NamespacedMemory("project-alpha", "./workspace")
ns.load()
ns.ingest("Alpha-specific decision")
results = ns.search("decision")
```

## Context Packets (Sub-Agent Injection)

Package relevant memories for sub-agents spawning cold:

```python
# Single-query context packet
packet = mem.build_context_packet(
    task="Debug the authentication flow",
    tags=["auth", "security"],
    max_memories=10,
    max_tokens=2000,
    include_mistakes=True,
)
print(packet.render("markdown"))  # → structured markdown for prompt injection
print(packet.render("xml"))       # → XML format

# Multi-query with deduplication
packet = mem.build_context_packet_multi(
    task="Fix performance issues",
    queries=["database bottleneck", "slow queries", "caching strategy"],
    max_tokens=3000,
)
# Trim to token budget
packet.trim(max_tokens=1500)
```

## MCP Server

Expose your memory workspace to any MCP-compatible host:

```python
from antaris_memory import create_mcp_server  # pip install mcp

server = create_mcp_server("./workspace")
server.run()  # Stdio transport — connect from Claude Desktop, Cursor, etc.
```

MCP tools exposed: `memory_search`, `memory_ingest`, `memory_consolidate`, `memory_stats`.

## Shared Memory Pools

Multiple agents sharing a common memory workspace:

```python
from antaris_memory import SharedMemoryPool, AgentPermission

pool = SharedMemoryPool("./shared", pool_name="team-alpha")
pool.grant("agent-1", AgentPermission.READ_WRITE)
pool.grant("agent-2", AgentPermission.READ_ONLY)

# Agent 1 writes
mem_1 = pool.open("agent-1")
mem_1.ingest("Deployed new API endpoint")

# Agent 2 reads
mem_2 = pool.open("agent-2")
results = mem_2.search("API deployment")
```

## Input Gating (P0–P3)

Classifies content at intake. Low-value data never enters storage:

```python
mem.ingest_with_gating("CRITICAL: API key compromised", source="alerts")
# → P0 (critical) → stored

mem.ingest_with_gating("Decided to switch to PostgreSQL", source="meeting")
# → P1 (operational) → stored

mem.ingest_with_gating("thanks for the update!", source="chat")
# → P3 (ephemeral) → dropped
```

| Level | Category | Stored | Examples |
|-------|----------|--------|----------|
| P0 | Strategic | ✅ | Security alerts, errors, deadlines, financial commitments |
| P1 | Operational | ✅ | Decisions, assignments, technical choices |
| P2 | Tactical | ✅ | Background info, research, general discussion |
| P3 | — | ❌ | Greetings, acknowledgments, filler |

Classification: keyword and pattern matching — no LLM calls. 0.177ms avg per input.

## Concurrency

Multiple processes can safely read and write to the same memory workspace.

```python
from antaris_memory import FileLock, VersionTracker

# Exclusive write access
with FileLock("/path/to/shard.json", timeout=10.0):
    data = load(shard)
    modify(data)
    save(shard, data)

# Optimistic conflict detection for read-heavy workloads
tracker = VersionTracker()
version = tracker.snapshot("/path/to/data.json")
data = load(data_path)
modify(data)
tracker.check(version)  # raises ConflictError if modified by another process
save(data_path, data)
```

Locks use `os.mkdir()` — atomic on all platforms including network filesystems.

## Benchmarks

Measured on Apple M4, Python 3.14.

| Memories | Ingest | Search (avg) | Search (p99) | Consolidate | Disk |
|----------|--------|-------------|-------------|-------------|------|
| 100 | 5.3ms (0.053ms/entry) | 0.40ms | 0.65ms | 4.2ms | 117KB |
| 500 | 16.8ms (0.034ms/entry) | 1.70ms | 2.51ms | 84.3ms | 575KB |
| 1,000 | 33.2ms (0.033ms/entry) | 3.43ms | 5.14ms | 343.3ms | 1.1MB |
| 5,000 | 173.7ms (0.035ms/entry) | 17.10ms | 25.70ms | 4.3s | 5.6MB |

Input gating (P0–P3 classification): **0.177ms avg** per input.

## Storage Format

```
workspace/
├── shards/
│   ├── 2026-02-strategic.json
│   ├── 2026-02-operational.json
│   └── 2026-01-tactical.json
├── indexes/
│   ├── search_index.json
│   ├── tag_index.json
│   └── date_index.json
├── migrations/
│   └── history.json
└── memory_audit.json           # Deletion audit trail (GDPR)
```

Plain JSON files. Inspect or edit with any text editor.

## Architecture

```
MemorySystem (v2.0)
├── ShardManager         — Date/topic sharding
├── IndexManager         — Full-text, tag, and date indexes
│   ├── SearchIndex      — BM25 inverted index
│   ├── TagIndex         — Tag → hash mapping
│   └── DateIndex        — Date range queries
├── SearchEngine         — BM25 + optional cosine hybrid
├── MigrationManager     — Schema versioning with rollback
├── InputGate            — P0-P3 classification at intake
├── DecayEngine          — Ebbinghaus forgetting curves
├── ConsolidationEngine  — Dedup, clustering, contradiction detection
├── ForgettingEngine     — Selective deletion with audit
├── KnowledgeSynthesizer — Gap identification and research integration
├── SharedMemoryPool     — Multi-agent coordination
├── NamespaceManager     — Multi-tenant isolation
├── ContextPacketBuilder — Sub-agent context injection
├── FileLock             — Cross-platform directory-based locking
└── VersionTracker       — Optimistic conflict detection
```

## Running Tests

```bash
git clone https://github.com/Antaris-Analytics/antaris-memory.git
cd antaris-memory
python -m pytest tests/ -v
```

All 293 tests pass with zero external dependencies.

## Migrating from v1.x

```python
pip install antaris-memory==2.0.0

# Existing workspaces load automatically — no changes required
mem = MemorySystem("./existing_workspace")
mem.load()  # Auto-detects format, migrates if needed

# New in v2.0 — opt into typed ingestion
mem.ingest_fact("PostgreSQL supports JSONB")
mem.set_embedding_fn(my_embed_fn)  # Enable semantic search
```

Legacy `LegacyMemorySystem` (single-file format) still importable: `from antaris_memory.core import MemorySystem as LegacyMemorySystem`.

## Zero Dependencies (Core)

The core package uses only the Python standard library. Optional extras:

- `pip install mcp` — enables `create_mcp_server()`
- Supply your own embedding function to `set_embedding_fn()` — any callable returning `list[float]` works (OpenAI, Ollama, sentence-transformers, etc.)

## Why Not a Vector DB?

LangChain Memory, Mem0, and Zep offer embedding-based semantic search and graph relationships. They require cloud APIs or self-hosted infrastructure (Redis, PostgreSQL, vector databases).

Antaris Memory stores everything in plain JSON files, runs fully offline, needs no API keys, and has no vector DB. It uses BM25 keyword ranking by default — good enough for most agent workloads. When you need semantic search, plug in your own embedding function. When you need scale, you already know it.

| | Antaris Memory | LangChain Memory | Mem0 | Zep |
|---|---|---|---|---|
| Search ranking | BM25 + optional cosine | Exact match | Embeddings | Embeddings |
| Input gating | ✅ P0-P3 | ❌ | ❌ | ❌ |
| Memory types | ✅ 5 types | ❌ | ❌ | ❌ |
| Namespace isolation | ✅ | ❌ | ❌ | ⚠️ |
| Context packets | ✅ | ❌ | ❌ | ❌ |
| MCP server | ✅ | ❌ | ❌ | ❌ |
| No database required | ✅ | ❌ | ❌ | ❌ |
| No API keys required | ✅ | ❌ | ❌ | ❌ |
| Memory decay | ✅ Ebbinghaus | ❌ | ❌ | ⚠️ |
| Contradiction detection | ✅ Rule-based | ❌ | ❌ | ⚠️ |
| Selective forgetting | ✅ With audit | ❌ | ⚠️ | ⚠️ |
| Infrastructure needed | None | Redis/PG | Vector + KV + Graph | PostgreSQL + Vector |

## Part of the Antaris Analytics Suite

- **antaris-memory** — Persistent memory for AI agents (this package)
- **[antaris-router](https://pypi.org/project/antaris-router/)** — Adaptive model routing with SLA enforcement
- **[antaris-guard](https://pypi.org/project/antaris-guard/)** — Security and prompt injection detection
- **[antaris-context](https://pypi.org/project/antaris-context/)** — Context window optimization

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
---

## 🔗 Related Packages

- [**antaris-router**](https://pypi.org/project/antaris-router/) - Adaptive model routing
- [**antaris-guard**](https://pypi.org/project/antaris-guard/) - Security and safety
- [**antaris-context**](https://pypi.org/project/antaris-context/) - Context window optimization
- [**antaris-pipeline**](https://pypi.org/project/antaris-pipeline/) - Agent orchestration pipeline

---

## 📞 Support

- **Documentation:** [docs.antarisanalytics.ai](https://docs.antarisanalytics.ai)
- **Email:** [dev@antarisanalytics.com](mailto:dev@antarisanalytics.com)
- **Website:** [antarisanalytics.ai](https://antarisanalytics.ai)

---

**Built with ❤️ by Antaris Analytics**  
*Deterministic infrastructure for AI agents*
