# tests/test_shared_decay_reinforce_behavior.py
"""Regression tests for shared pool reinforcement and cross-agent boost behavior."""

from antaris_memory.shared import SharedMemoryPool
from antaris_memory.entry import MemoryEntry


def test_shared_read_only_reinforces_returned_limit(tmp_path):
    """Regression guard: read() should not reinforce more docs than it returns."""
    pool = SharedMemoryPool(str(tmp_path), pool_name="t")
    pool.register_agent("a1", role="admin")
    pool.register_agent("a2", role="write")

    for i in range(40):
        pool.write("a2", f"shared memory item {i} about deployment systemd restart policy",
                    namespace="shared")

    first = pool.read("a1", "deployment systemd restart policy", limit=10)
    assert len(first) == 10
    second = pool.read("a1", "deployment systemd restart policy", limit=10)
    assert len(second) == 10
    # Stable ordering = no runaway reinforcement of near-cutoff docs
    assert [m.hash for m in first] == [m.hash for m in second]


def test_mark_used_cross_agent_no_writer_tag_does_not_boost(tmp_path):
    """Memories without agent: tag should not get cross-agent confidence boost."""
    pool = SharedMemoryPool(str(tmp_path), pool_name="t2")
    pool.register_agent("a1", role="admin")

    # Create an entry without agent tag (simulate legacy/foreign entry)
    m = MemoryEntry("legacy memory without writer tag", source="legacy")
    m.tags = [t for t in m.tags if not t.startswith("agent:")]
    m.confidence = 0.5
    pool.memories.append(m)
    pool._hashes.add(m.hash)
    pool._memory_by_hash[m.hash] = m

    before = m.confidence
    ok = pool.mark_used_cross_agent("a1", m.hash, boost_factor=1.3)
    assert ok is True
    assert m.confidence == before  # no boost for unknown writer


def test_mark_used_cross_agent_boosts_only_for_other_agent(tmp_path):
    """Cross-agent use boosts confidence; self-use does not."""
    pool = SharedMemoryPool(str(tmp_path), pool_name="t3")
    pool.register_agent("a1", role="admin")
    pool.register_agent("a2", role="write")

    m = pool.write("a2", "deployment uses systemd; restart always", namespace="shared")
    assert m is not None
    m.confidence = 0.4

    # Used by a1 (cross-agent) => boosted
    before = m.confidence
    ok = pool.mark_used_cross_agent("a1", m.hash, boost_factor=1.3)
    assert ok is True
    assert m.confidence > before

    # Used by a2 (self) => should not boost further
    before2 = m.confidence
    ok2 = pool.mark_used_cross_agent("a2", m.hash, boost_factor=1.3)
    assert ok2 is True
    assert m.confidence == before2
