"""Tests for MemorySystemV4 ↔ SharedMemoryPool integration."""
import os
import pytest
from antaris_memory.core_v4 import MemorySystemV4
from antaris_memory.shared import SharedMemoryPool


class TestEnableSharedPool:
    def test_enable_shared_pool_returns_pool(self, tmp_path):
        store = str(tmp_path / "store")
        pool_dir = str(tmp_path / "pool")
        mem = MemorySystemV4(store)
        pool = mem.enable_shared_pool(pool_dir)
        assert isinstance(pool, SharedMemoryPool)
        assert mem.get_shared_pool() is pool

    def test_get_shared_pool_none_by_default(self, tmp_path):
        mem = MemorySystemV4(str(tmp_path / "store"))
        assert mem.get_shared_pool() is None

    def test_enable_shared_pool_registers_agent(self, tmp_path):
        mem = MemorySystemV4(str(tmp_path / "store"))
        pool = mem.enable_shared_pool(str(tmp_path / "pool"), agent_id="myagent")
        assert "myagent" in pool.permissions

    def test_enable_shared_pool_custom_role(self, tmp_path):
        mem = MemorySystemV4(str(tmp_path / "store"))
        pool = mem.enable_shared_pool(str(tmp_path / "pool"), agent_id="reader", role="read")
        assert pool.permissions["reader"].role == "read"


class TestSharedWrite:
    def test_shared_write_adds_to_pool(self, tmp_path):
        mem = MemorySystemV4(str(tmp_path / "store"))
        mem.enable_shared_pool(str(tmp_path / "pool"), agent_id="agent_a")
        entry = mem.shared_write("Important shared knowledge about Python")
        assert entry is not None
        assert len(mem.get_shared_pool().memories) == 1

    def test_shared_write_without_pool_raises(self, tmp_path):
        mem = MemorySystemV4(str(tmp_path / "store"))
        with pytest.raises(RuntimeError):
            mem.shared_write("This should fail")

    def test_shared_write_with_namespace(self, tmp_path):
        mem = MemorySystemV4(str(tmp_path / "store"))
        # Use admin role so we can write to any namespace
        mem.enable_shared_pool(str(tmp_path / "pool"), agent_id="agent_a", role="admin")
        entry = mem.shared_write("Namespace specific content", namespace="team_a")
        assert entry is not None
        ns_tag = f"ns:team_a"
        assert any(ns_tag in t for t in entry.tags)


class TestSharedSearch:
    def test_shared_search_finds_written_memory(self, tmp_path):
        pool_dir = str(tmp_path / "pool")

        # Two agents sharing a pool
        mem_a = MemorySystemV4(str(tmp_path / "store_a"))
        mem_a.enable_shared_pool(pool_dir, agent_id="agent_a")
        mem_a.shared_write("The transformer architecture uses attention mechanisms")
        mem_a.get_shared_pool().save()

        mem_b = MemorySystemV4(str(tmp_path / "store_b"))
        mem_b.enable_shared_pool(pool_dir, agent_id="agent_b", load_existing=True)

        results = mem_b.shared_search("transformer attention")
        assert len(results) >= 1

    def test_shared_search_without_pool_raises(self, tmp_path):
        mem = MemorySystemV4(str(tmp_path / "store"))
        with pytest.raises(RuntimeError):
            mem.shared_search("something")


class TestSharedMarkUsed:
    def test_shared_mark_used_without_pool_raises(self, tmp_path):
        mem = MemorySystemV4(str(tmp_path / "store"))
        with pytest.raises(RuntimeError):
            mem.shared_mark_used(["hash1"])

    def test_shared_mark_used_returns_count(self, tmp_path):
        pool_dir = str(tmp_path / "pool")

        mem_a = MemorySystemV4(str(tmp_path / "store_a"))
        mem_a.enable_shared_pool(pool_dir, agent_id="agent_a")
        entry = mem_a.shared_write("Knowledge about transformers and attention")

        mem_b = MemorySystemV4(str(tmp_path / "store_b"))
        mem_b.enable_shared_pool(pool_dir, agent_id="agent_b")
        # Pool is shared in memory (same object in this test)
        mem_b._shared_pool = mem_a.get_shared_pool()

        count = mem_b.shared_mark_used([entry.hash])
        assert count == 1


class TestFullMultiAgentFlow:
    def test_two_agents_share_and_search(self, tmp_path):
        """Agent A writes, Agent B finds it via shared search."""
        pool_dir = str(tmp_path / "pool")

        mem_a = MemorySystemV4(str(tmp_path / "store_a"))
        mem_a.enable_shared_pool(pool_dir, agent_id="agent_a")
        mem_a.shared_write("The optimal batch size for BERT is 32")
        mem_a.get_shared_pool().save()

        mem_b = MemorySystemV4(str(tmp_path / "store_b"))
        mem_b.enable_shared_pool(pool_dir, agent_id="agent_b", load_existing=True)

        results = mem_b.shared_search("BERT batch size")
        assert any("BERT" in r.content or "batch" in r.content.lower() for r in results)
