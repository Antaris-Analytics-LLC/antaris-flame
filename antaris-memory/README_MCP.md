# Antaris Memory — MCP Server

**Expose `antaris-memory` as an MCP (Model Context Protocol) server so any Claude or MCP-enabled agent can recall and store persistent memories.**

---

## What This Enables

Any Claude agent or MCP client can:

- **Recall past context** with semantic search: *"What did we decide about the database migration?"*
- **Store new memories** as conversations happen: important decisions, facts, preferences
- **Inspect the memory store** for diagnostics and stats

This turns `antaris-memory` into a memory backend that Claude Desktop, Cline, Continue, and other MCP clients can plug into directly — no custom code required.

---

## Quick Start

### Install

```bash
pip install antaris-memory mcp
```

### Run the server

```bash
# stdio transport (for Claude Desktop / MCP clients)
python -m antaris_memory.mcp_server --memory-path ./my_memory_store

# SSE transport (for HTTP-based MCP clients)
python -m antaris_memory.mcp_server --transport sse --port 8765
```

Or use the installed script:

```bash
antaris-memory-mcp --memory-path ./my_memory_store
```

---

## Add to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "antaris-memory": {
      "command": "python",
      "args": [
        "-m",
        "antaris_memory.mcp_server",
        "--memory-path",
        "/path/to/your/memory_store"
      ]
    }
  }
}
```

Restart Claude Desktop. You'll see `antaris-memory` appear in the MCP tools panel.

---

## Configuration

| Option | CLI Flag | Env Variable | Default |
|---|---|---|---|
| Memory store path | `--memory-path` | `ANTARIS_MEMORY_PATH` | `./antaris_memory_store` |
| Transport | `--transport` | — | `stdio` |
| SSE host | `--host` | — | `127.0.0.1` |
| SSE port | `--port` | — | `8765` |

### Environment variable

```bash
export ANTARIS_MEMORY_PATH=/home/user/.agent_memory
python -m antaris_memory.mcp_server
```

---

## Available Tools

### `recall_memories(query, limit=5)`

Search persistent memory and return the most relevant results.

**Parameters:**
- `query` (str) — Natural-language search query
- `limit` (int, default 5) — Max results to return

**Returns:** List of memory objects:
```json
[
  {
    "content": "We decided to migrate to PostgreSQL in Q3",
    "category": "strategic",
    "relevance": 0.87,
    "source": "inline",
    "memory_type": "episodic"
  }
]
```

---

### `store_memory(content, category="general", tags=None)`

Ingest new information into persistent memory.

**Parameters:**
- `content` (str) — Text to store (multi-line supported)
- `category` (str) — Category label: `general`, `strategic`, `tactical`, `personal`, etc.
- `tags` (list, optional) — Tag strings for filtering

**Returns:**
```json
{
  "stored": true,
  "entry_count": 42
}
```

---

### `memory_stats()`

Return storage statistics.

**Returns:**
```json
{
  "total_memories": 247,
  "categories": {"strategic": 12, "general": 198, "tactical": 37},
  "disk_usage_mb": 0.045,
  "avg_score": 0.73,
  "avg_confidence": 0.61,
  "cache_hit_rate": 0.82,
  "wal_pending": 0,
  "search_count": 156
}
```

---

## Available Resources

### `antaris://memory/{query}`

Returns a formatted memory context block for the query. Suitable for injecting directly into an LLM prompt.

```
[antaris-memory] 3 memories for query: 'database migration'

1. [strategic] We decided to migrate to PostgreSQL in Q3 this year  (relevance=0.87, type=episodic)
2. [tactical] Database migration scripts are in /infra/migrations  (relevance=0.71, type=procedural)
3. [general] The DBA team lead is Sarah, reachable at sarah@example.com  (relevance=0.54, type=episodic)
```

---

## Example: Claude Answering "What Did We Discuss Last Week?"

With antaris-memory MCP enabled in Claude Desktop:

> **You:** What did we discuss about the API redesign last week?
>
> **Claude:** *[calls recall_memories("API redesign")]*
>
> I found 3 relevant memories from your store:
>
> 1. **"REST API v3 will drop XML support and use JSON-only responses"** (strategic, relevance 0.91)
> 2. **"The /users endpoint will be renamed to /accounts for consistency"** (tactical, relevance 0.78)
> 3. **"Breaking changes require a 90-day deprecation notice per SLA"** (strategic, relevance 0.62)
>
> It looks like you decided on a JSON-only v3 API with an endpoint rename. Want me to store any follow-up decisions?

---

## Programmatic Usage

```python
from antaris_memory.mcp_server import create_server

# Create server with a specific memory path
mcp = create_server(memory_path="./project_memory")

# Run via stdio (for subprocess-based MCP clients)
mcp.run(transport="stdio")
```

---

## Graceful Degradation

The `mcp` package is **optional**. If it's not installed, `antaris_memory` imports cleanly and `create_mcp_server` returns a stub that raises `ImportError` with a helpful message. The rest of the `antaris_memory` API is unaffected.

```python
from antaris_memory import MCP_AVAILABLE, create_mcp_server

if MCP_AVAILABLE:
    server = create_mcp_server()
else:
    print("MCP not available — install with: pip install mcp")
```

---

## Memory Categories

Use categories to organise memories:

| Category | Use for |
|---|---|
| `general` | Default, miscellaneous facts |
| `strategic` | High-level decisions, goals, direction |
| `tactical` | How-to, procedures, implementation details |
| `personal` | Preferences, personal context |
| `mistake` | Errors and lessons learned (boosted recall priority) |
| `fact` | Factual knowledge, reference data |
| `preference` | User preferences and settings |

---

## See Also

- [antaris-memory README](./README.md) — Full package documentation
- [MCP Specification](https://modelcontextprotocol.io) — Model Context Protocol
- [Claude Desktop MCP Guide](https://docs.anthropic.com/claude/docs/mcp) — Adding MCP servers to Claude Desktop
