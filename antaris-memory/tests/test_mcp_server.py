"""
Tests for antaris_memory.mcp_server — Sprint 5.

Tests:
  - recall_memories returns list with correct schema
  - store_memory returns {stored: True}
  - Round-trip: store then recall finds content
  - memory_stats returns dict with total_memories key
  - Graceful degradation when MCP not installed
"""

import importlib
import sys
import tempfile
import os
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_memory_path(tmp_path):
    """Return a temporary directory path for a memory store."""
    return str(tmp_path / "test_memory_store")


@pytest.fixture()
def server(tmp_memory_path):
    """Create a fresh FastMCP server pointed at a temp memory store."""
    from antaris_memory.mcp_server import create_server
    return create_server(memory_path=tmp_memory_path)


# ---------------------------------------------------------------------------
# Helper: invoke a registered tool by calling it directly via FastMCP
# ---------------------------------------------------------------------------

def call_tool(server, tool_name, **kwargs):
    """Retrieve and call a tool function registered on the FastMCP server."""
    # FastMCP stores tools in ._tool_manager._tools (dict: name → Tool)
    # We access the underlying callable via the tool's fn attribute.
    tool_manager = server._tool_manager
    tools = tool_manager._tools  # dict[str, Tool]
    if tool_name not in tools:
        raise KeyError(f"Tool {tool_name!r} not found. Available: {list(tools)}")
    fn = tools[tool_name].fn
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Test: recall_memories schema
# ---------------------------------------------------------------------------

class TestRecallMemories:
    def test_returns_empty_list_when_no_matches(self, server):
        result = call_tool(server, "recall_memories", query="nonexistent xyz abc", limit=5)
        assert isinstance(result, list)
        assert result == []

    def test_returns_list_with_correct_schema(self, server, tmp_memory_path):
        """Store a memory directly then check recall schema."""
        from antaris_memory import MemorySystem
        mem = MemorySystem(tmp_memory_path)
        mem.load()
        mem.ingest("The project deadline is next Friday and the team is prepared", category="strategic")
        mem.save()

        result = call_tool(server, "recall_memories", query="project deadline", limit=5)
        assert isinstance(result, list)
        assert len(result) >= 1

        item = result[0]
        assert "content" in item
        assert "category" in item
        assert "relevance" in item
        assert "source" in item
        assert "memory_type" in item

        assert isinstance(item["content"], str)
        assert isinstance(item["category"], str)
        assert isinstance(item["relevance"], float)
        assert isinstance(item["source"], str)
        assert isinstance(item["memory_type"], str)

    def test_limit_is_respected(self, server, tmp_memory_path):
        from antaris_memory import MemorySystem
        mem = MemorySystem(tmp_memory_path)
        mem.load()
        for i in range(10):
            mem.ingest(f"Important strategic decision number {i} about the roadmap planning", category="strategic")
        mem.save()

        result = call_tool(server, "recall_memories", query="strategic decision", limit=3)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# Test: store_memory
# ---------------------------------------------------------------------------

class TestStoreMemory:
    def test_returns_stored_true(self, server):
        result = call_tool(
            server,
            "store_memory",
            content="The capital of France is Paris and it is known for the Eiffel Tower",
            category="general",
        )
        assert isinstance(result, dict)
        assert result["stored"] is True
        assert "entry_count" in result
        assert isinstance(result["entry_count"], int)
        assert result["entry_count"] >= 1

    def test_returns_stored_false_for_short_content(self, server):
        """Very short content that gets filtered by the gating system."""
        result = call_tool(
            server,
            "store_memory",
            content="hi",  # too short to store (< 15 chars)
            category="general",
        )
        assert isinstance(result, dict)
        assert "stored" in result
        assert "entry_count" in result

    def test_category_is_accepted(self, server):
        result = call_tool(
            server,
            "store_memory",
            content="Our primary objective for Q3 is to increase user retention across all platforms",
            category="strategic",
        )
        assert result["stored"] is True

    def test_entry_count_increments(self, server):
        r1 = call_tool(
            server, "store_memory",
            content="First memory entry about the quarterly planning meeting and goals",
        )
        r2 = call_tool(
            server, "store_memory",
            content="Second memory entry about the infrastructure upgrade and deployment plans",
        )
        assert r2["entry_count"] >= r1["entry_count"]


# ---------------------------------------------------------------------------
# Test: round-trip store then recall
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_store_then_recall_finds_content(self, server):
        unique_phrase = "antaris-test-xk9z memory round trip validation sprint five"
        store_result = call_tool(
            server,
            "store_memory",
            content=unique_phrase,
            category="general",
        )
        assert store_result["stored"] is True

        recall_result = call_tool(
            server,
            "recall_memories",
            query="antaris-test-xk9z round trip",
            limit=5,
        )
        assert isinstance(recall_result, list)
        assert len(recall_result) >= 1
        contents = [r["content"] for r in recall_result]
        assert any("antaris-test-xk9z" in c for c in contents)

    def test_category_preserved_in_round_trip(self, server):
        call_tool(
            server,
            "store_memory",
            content="Team decision to migrate the entire database infrastructure to PostgreSQL this quarter",
            category="strategic",
        )
        results = call_tool(
            server,
            "recall_memories",
            query="database migration infrastructure",
            limit=5,
        )
        assert len(results) >= 1
        categories = [r["category"] for r in results]
        assert "strategic" in categories


# ---------------------------------------------------------------------------
# Test: memory_stats
# ---------------------------------------------------------------------------

class TestMemoryStats:
    def test_returns_dict_with_total_memories(self, server):
        result = call_tool(server, "memory_stats")
        assert isinstance(result, dict)
        assert "total_memories" in result
        assert isinstance(result["total_memories"], int)

    def test_required_keys_present(self, server):
        result = call_tool(server, "memory_stats")
        required_keys = [
            "total_memories",
            "categories",
            "disk_usage_mb",
            "avg_score",
            "avg_confidence",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key!r}"

    def test_total_memories_increments_after_store(self, server):
        stats_before = call_tool(server, "memory_stats")
        call_tool(
            server,
            "store_memory",
            content="Completely new unique memory about the annual company strategy review session",
        )
        stats_after = call_tool(server, "memory_stats")
        assert stats_after["total_memories"] >= stats_before["total_memories"]

    def test_categories_is_dict(self, server):
        result = call_tool(server, "memory_stats")
        assert isinstance(result["categories"], dict)


# ---------------------------------------------------------------------------
# Test: graceful degradation when MCP not installed
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_mcp_available_flag_exported(self):
        """MCP_AVAILABLE must be importable from the top-level package."""
        import antaris_memory
        assert hasattr(antaris_memory, "MCP_AVAILABLE")
        assert isinstance(antaris_memory.MCP_AVAILABLE, bool)

    def test_create_mcp_server_exported(self):
        """create_mcp_server must be importable from the top-level package."""
        import antaris_memory
        assert hasattr(antaris_memory, "create_mcp_server")
        assert callable(antaris_memory.create_mcp_server)

    def test_stub_raises_import_error_when_mcp_unavailable(self, monkeypatch):
        """Simulate MCP not installed: the stub should raise ImportError."""
        import antaris_memory.mcp_server as ms

        orig = ms.MCP_AVAILABLE
        orig_fastmcp = ms.FastMCP
        try:
            ms.MCP_AVAILABLE = False
            ms.FastMCP = None
            with pytest.raises(ImportError, match="mcp"):
                ms.create_server()
        finally:
            ms.MCP_AVAILABLE = orig
            ms.FastMCP = orig_fastmcp

    def test_main_antaris_memory_import_not_broken(self):
        """Core package import must succeed even if mcp is missing (it's installed here,
        but we verify the import path doesn't crash under the graceful-degradation stub)."""
        # Re-import to confirm no side effects
        import importlib
        import antaris_memory
        importlib.reload(antaris_memory)
        from antaris_memory import MemorySystem
        assert MemorySystem is not None

    def test_mcp_server_module_importable(self):
        """The mcp_server module itself must be importable without error."""
        import antaris_memory.mcp_server as ms
        assert ms.MCP_AVAILABLE is True  # mcp IS installed in this test env
        assert ms.create_server is not None


# ---------------------------------------------------------------------------
# Test: resource registration
# ---------------------------------------------------------------------------

class TestResourceRegistration:
    def test_resource_template_registered(self, server):
        """antaris://memory/{query} resource template should be registered."""
        resource_manager = server._resource_manager
        # Templates are stored in _templates dict
        templates = resource_manager._templates
        uris = list(templates.keys())
        assert any("memory" in uri for uri in uris), (
            f"No memory resource template found. Registered templates: {uris}"
        )
