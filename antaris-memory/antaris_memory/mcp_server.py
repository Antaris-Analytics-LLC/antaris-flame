"""
Antaris Memory MCP Server

Exposes antaris-memory as MCP resources and tools for any MCP-enabled agent.

Resources:
  - antaris://memory/{query} → relevant memories for query

Tools:
  - recall_memories(query, limit) → search and return memories
  - store_memory(content, category, tags) → ingest a new memory
  - memory_stats() → storage statistics

Usage:
    python -m antaris_memory.mcp_server --memory-path ./my_memory_store
    # or
    from antaris_memory.mcp_server import create_server
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Graceful MCP availability check
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None  # type: ignore

from antaris_memory import MemorySystem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_memory_path(memory_path: Optional[str] = None) -> str:
    """Resolve memory path from arg, env, or default."""
    return (
        memory_path
        or os.environ.get("ANTARIS_MEMORY_PATH")
        or "./antaris_memory_store"
    )


def _load_memory(memory_path: str) -> MemorySystem:
    """Create and load a MemorySystem from the given workspace path."""
    mem = MemorySystem(memory_path)
    mem.load()
    return mem


def _entry_to_dict(entry: Any) -> Dict[str, Any]:
    """Convert a MemoryEntry to a plain dict for JSON serialisation."""
    return {
        "content": getattr(entry, "content", ""),
        "category": getattr(entry, "category", "general"),
        "relevance": round(float(getattr(entry, "importance", 1.0)), 4),
        "source": getattr(entry, "source", ""),
        "memory_type": getattr(entry, "memory_type", "episodic"),
    }


def _format_memory_block(entries: list, query: str) -> str:
    """Format search results as a readable memory context block."""
    if not entries:
        return f"[antaris-memory] No memories found for query: {query!r}"

    lines = [f"[antaris-memory] {len(entries)} memories for query: {query!r}", ""]
    for i, entry in enumerate(entries, 1):
        d = _entry_to_dict(entry)
        lines.append(
            f"{i}. [{d['category']}] {d['content']}"
            f"  (relevance={d['relevance']}, type={d['memory_type']})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server(memory_path: Optional[str] = None) -> "FastMCP":
    """Create and return the FastMCP server with antaris-memory tools/resources.

    Args:
        memory_path: Path to the memory workspace directory.  Falls back to
            ``$ANTARIS_MEMORY_PATH`` env var, then ``./antaris_memory_store``.

    Returns:
        A configured ``FastMCP`` instance ready to run.

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required to run the antaris-memory MCP server. "
            "Install it with: pip install mcp"
        )

    resolved_path = _get_memory_path(memory_path)
    mcp = FastMCP(
        name="antaris-memory",
        instructions=(
            "Antaris Memory — persistent agent memory. "
            "Use recall_memories to search past context, store_memory to save new information, "
            "and memory_stats to inspect the store."
        ),
    )

    # ------------------------------------------------------------------
    # Tool: recall_memories
    # ------------------------------------------------------------------
    @mcp.tool()
    def recall_memories(query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search agent memories and return the most relevant results.

        Args:
            query: Natural-language search query.
            limit: Maximum number of memories to return (default 5).

        Returns:
            List of memory dicts with keys: content, category, relevance,
            source, memory_type.  Empty list if nothing matches.
        """
        mem = _load_memory(resolved_path)
        results = mem.search(query=query, limit=limit, use_decay=True)
        mem.save()
        if not results:
            return []
        return [_entry_to_dict(e) for e in results]

    # ------------------------------------------------------------------
    # Tool: store_memory
    # ------------------------------------------------------------------
    @mcp.tool()
    def store_memory(
        content: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Ingest new information into persistent agent memory.

        Args:
            content: Text to store (multi-line supported).
            category: Category label, e.g. "strategic", "tactical", "general".
            tags: Optional list of tag strings to associate with this memory.

        Returns:
            Dict with keys: stored (bool), entry_count (int).
        """
        mem = _load_memory(resolved_path)
        # Bug fix: pass caller-supplied tags to ingest() so they are applied
        added = mem.ingest(content=content, category=category, tags=tags or None)
        mem.save()
        return {
            "stored": added > 0,
            "entry_count": len(mem.memories),
        }

    # ------------------------------------------------------------------
    # Tool: memory_stats
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_stats() -> Dict[str, Any]:
        """Return statistics about the memory store.

        Returns:
            Dict with keys: total_memories, categories, disk_usage_mb,
            avg_score, avg_confidence, cache_hit_rate, plus additional
            performance metrics from the Sprint 11 engine.
        """
        mem = _load_memory(resolved_path)
        raw = mem.stats()
        return {
            "total_memories": raw.get("total_entries", raw.get("total", 0)),
            "categories": raw.get("categories", {}),
            "disk_usage_mb": raw.get("disk_usage_mb", 0.0),
            "avg_score": raw.get("avg_score", 0.0),
            "avg_confidence": raw.get("avg_confidence", 0.0),
            "cache_hit_rate": raw.get("cache_hit_rate", 0.0),
            "wal_pending": raw.get("wal_pending_entries", 0),
            "search_count": raw.get("search_count", 0),
        }

    # ------------------------------------------------------------------
    # Tool: memory_store  (v3.9 canonical name)
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_store(
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: str = "general",
    ) -> Dict[str, Any]:
        """Store a new memory and return its memory_id.

        Args:
            content:  Text content to persist.
            metadata: Optional dict with extra fields (tags, source, etc.).
            category: Category label (default "general").

        Returns:
            {memory_id (str), stored (bool), entry_count (int)}
        """
        mem = _load_memory(resolved_path)
        tags: List[str] = []
        source: str = "mcp"
        if metadata:
            tags = metadata.get("tags", []) or []
            source = metadata.get("source", "mcp")

        before_count = len(mem.memories)
        mem.ingest(content=content, source=source, category=category)
        # Tags must be applied post-ingest (ingest parses lines)
        added_entries = mem.memories[before_count:]
        if tags and added_entries:
            for e in added_entries:
                e.tags = list(set(e.tags + tags))
        mem.save()

        memory_id = added_entries[0].hash if added_entries else ""
        return {
            "memory_id": memory_id,
            "stored": len(added_entries) > 0,
            "entry_count": len(mem.memories),
        }

    # ------------------------------------------------------------------
    # Tool: memory_search  (v3.9 canonical name)
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories and return ranked results.

        Args:
            query: Natural-language search string.
            limit: Maximum results to return (default 10).

        Returns:
            List of {id, content, score} dicts, ordered by relevance.
        """
        mem = _load_memory(resolved_path)
        results = mem.search(query=query, limit=limit, use_decay=True)
        mem.save()
        return [
            {
                "id": getattr(e, "hash", ""),
                "content": getattr(e, "content", ""),
                "score": round(float(getattr(e, "importance", 1.0)), 4),
                "category": getattr(e, "category", "general"),
                "memory_type": getattr(e, "memory_type", "episodic"),
            }
            for e in results
        ]

    # ------------------------------------------------------------------
    # Tool: memory_recall  (v3.9 canonical name)
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_recall(memory_id: str) -> Dict[str, Any]:
        """Retrieve a specific memory by its ID (hash).

        Args:
            memory_id: The unique memory hash returned by memory_store/memory_search.

        Returns:
            The memory dict, or {error: "not found"} if missing.
        """
        mem = _load_memory(resolved_path)
        for entry in mem.memories:
            if getattr(entry, "hash", "") == memory_id:
                return {
                    "id": entry.hash,
                    "content": entry.content,
                    "category": entry.category,
                    "created": entry.created,
                    "last_accessed": entry.last_accessed,
                    "importance": round(entry.importance, 4),
                    "confidence": round(entry.confidence, 4),
                    "tags": entry.tags,
                    "memory_type": entry.memory_type,
                }
        return {"error": "not found", "memory_id": memory_id}

    # ------------------------------------------------------------------
    # Tool: memory_delete  (v3.9 canonical name)
    # ------------------------------------------------------------------
    @mcp.tool()
    def memory_delete(memory_id: str) -> Dict[str, Any]:
        """Delete a memory by its ID.

        Args:
            memory_id: The unique memory hash to remove.

        Returns:
            {deleted (bool), memory_id (str), entry_count (int)}
        """
        mem = _load_memory(resolved_path)
        before = len(mem.memories)
        mem.memories = [e for e in mem.memories if getattr(e, "hash", "") != memory_id]
        mem._hashes = {e.hash for e in mem.memories}
        deleted = len(mem.memories) < before
        if deleted:
            mem.save()
        return {
            "deleted": deleted,
            "memory_id": memory_id,
            "entry_count": len(mem.memories),
        }

    # ------------------------------------------------------------------
    # Resource: antaris://memory/{query}
    # ------------------------------------------------------------------
    @mcp.resource("antaris://memory/{query}")
    def memory_resource(query: str) -> str:
        """Retrieve memories matching *query* as a formatted context block.

        URI: ``antaris://memory/<url-encoded-query>``

        Returns a human-readable block of the top matching memories, suitable
        for injecting directly into an LLM context window.
        """
        mem = _load_memory(resolved_path)
        results = mem.search(query=query, limit=10, use_decay=True)
        mem.save()
        return _format_memory_block(results, query)

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the antaris-memory MCP server (stdio transport by default)."""
    parser = argparse.ArgumentParser(
        description="Antaris Memory MCP Server — expose agent memory over MCP."
    )
    parser.add_argument(
        "--memory-path",
        default=None,
        help=(
            "Path to the memory workspace directory. "
            "Defaults to $ANTARIS_MEMORY_PATH or ./antaris_memory_store"
        ),
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for SSE transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for SSE transport (default: 8765).",
    )
    args = parser.parse_args()

    if not MCP_AVAILABLE:
        print(
            "ERROR: The 'mcp' package is not installed.\n"
            "Install it with: pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    server = create_server(memory_path=args.memory_path)

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        # SSE transport
        server.settings.host = args.host
        server.settings.port = args.port
        server.run(transport="sse")


if __name__ == "__main__":
    main()
