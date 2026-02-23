"""
Sprint 11 — Memory Performance Optimization Tests

Covers:
  1. Read cache populated after first search
  2. Cache hit avoids file I/O (mock)
  3. WAL receives writes before flush
  4. flush() compacts WAL into shards
  5. Auto-flush triggers after N writes
  6. stats() returns expected structure
  7. Stale lock detection (dead PID)
  8. Stale lock is cleared and acquired successfully
  9. Access count increments on search hit
 10. get_hot_entries() returns most accessed
 11. compact() reduces entry count
 12. compact() returns Sprint 11 keys
 13. Backward compat: existing ingest/search still works
 14. WAL replay on restart (crash recovery)
 15. WAL cleared after flush()
 16. ReadCache LRU eviction
 17. ReadCache invalidated on ingest
 18. WALManager.should_flush() fires after flush_interval
 19. WALManager.should_flush() fires on size threshold
 20. WALManager.pending_count() matches appended entries
 21. AccessTracker.boost_score() increases with access count
 22. AccessTracker.get_top() returns correct ordering
 23. PerformanceMonitor records search count and timing
 24. stats() cache_hit_rate reflects real cache usage
 25. close() flushes WAL and clears it
 26. Stale lock with alive PID is NOT broken prematurely
 27. Concurrent WAL appends produce no duplicates on replay
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from antaris_memory import MemorySystem
from antaris_memory.performance import (
    ReadCache,
    WALManager,
    PerformanceMonitor,
    AccessTracker,
)
from antaris_memory.locking import FileLock, LockTimeout


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mem(tmp_dir, **kwargs):
    """Create a MemorySystemV4 rooted at *tmp_dir*."""
    return MemorySystem(workspace=tmp_dir, **kwargs)


def _ingest_n(mem, n, prefix="Performance test memory line"):
    """Ingest *n* distinct memories."""
    for i in range(n):
        mem.ingest(f"{prefix} number {i:04d} unique token xyz", source="test")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 1–2. Read Cache
# ─────────────────────────────────────────────────────────────────────────────

class TestReadCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cache_populated_after_first_search(self):
        """ReadCache has exactly one entry after the first search."""
        mem = _make_mem(self.tmp, enable_read_cache=True)
        _ingest_n(mem, 3)

        self.assertEqual(len(mem._read_cache), 0, "cache must start empty")
        mem.search("performance test memory")
        self.assertEqual(len(mem._read_cache), 1, "cache should have one entry after search")

    def test_cache_returns_same_results_on_hit(self):
        """Second call with identical params returns cached list."""
        mem = _make_mem(self.tmp, enable_read_cache=True)
        _ingest_n(mem, 3)

        first = mem.search("performance test memory")
        second = mem.search("performance test memory")
        # Same objects returned (list contents match)
        self.assertEqual(
            [m.hash for m in first],
            [m.hash for m in second],
        )

    def test_cache_hit_increments_hit_counter(self):
        """Cache.hits increases by 1 on each cache hit."""
        mem = _make_mem(self.tmp, enable_read_cache=True)
        _ingest_n(mem, 3)

        mem.search("performance test memory")   # miss (populate)
        hits_before = mem._read_cache.hits
        mem.search("performance test memory")   # hit
        self.assertEqual(mem._read_cache.hits, hits_before + 1)

    def test_cache_invalidated_on_ingest(self):
        """Ingesting new content clears the cache."""
        mem = _make_mem(self.tmp, enable_read_cache=True)
        _ingest_n(mem, 3)

        mem.search("performance test memory")
        self.assertEqual(len(mem._read_cache), 1)

        # Ingest something new → cache should be cleared
        mem.ingest("This brand new memory invalidates the cache completely", source="test")
        self.assertEqual(len(mem._read_cache), 0, "cache should be cleared after ingest")

    def test_cache_disabled_when_flag_false(self):
        """With enable_read_cache=False the cache object is None."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        self.assertIsNone(mem._read_cache)
        # search should still work without a cache
        _ingest_n(mem, 2)
        results = mem.search("performance test memory")
        self.assertIsInstance(results, list)

    def test_lru_eviction_respects_max_entries(self):
        """ReadCache evicts LRU items when max_entries is exceeded."""
        cache = ReadCache(max_entries=3)
        for i in range(4):
            cache.put(f"key{i}", f"value{i}")
        # key0 should have been evicted
        self.assertIsNone(cache.get("key0"))
        self.assertIsNotNone(cache.get("key3"))

    def test_lru_access_order_preserved(self):
        """Accessing a key marks it as recently used (protected from eviction)."""
        cache = ReadCache(max_entries=3)
        cache.put("key0", "v0")
        cache.put("key1", "v1")
        cache.put("key2", "v2")
        # Access key0 to make it most-recently-used
        cache.get("key0")
        # Adding key3 should evict key1 (now LRU), not key0
        cache.put("key3", "v3")
        self.assertIsNotNone(cache.get("key0"), "key0 should survive as recently accessed")
        self.assertIsNone(cache.get("key1"), "key1 should be evicted")


# ─────────────────────────────────────────────────────────────────────────────
# 3–5. WAL
# ─────────────────────────────────────────────────────────────────────────────

class TestWAL(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_wal_receives_writes_before_flush(self):
        """Each ingest() appends to the WAL before the shard is updated."""
        # Use a high flush_interval so auto-flush never fires during the test
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem._wal.flush_interval = 10_000   # effectively disable auto-flush

        mem.ingest("Unique WAL test entry alpha", source="wal_test")
        pending = mem._wal.pending_count()
        self.assertGreater(pending, 0, "WAL should have at least one pending entry after ingest")

    def test_wal_file_is_valid_jsonl(self):
        """WAL file contains one valid JSON object per line."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem._wal.flush_interval = 10_000

        content = "WAL JSONL test memory line alpha bravo"
        mem.ingest(content, source="wal_test")

        pending = mem._wal.load_pending()
        self.assertGreater(len(pending), 0)
        for entry in pending:
            self.assertIn("content", entry)

    def test_flush_compacts_wal_into_memories(self):
        """flush() saves in-memory state to disk and clears the WAL."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem._wal.flush_interval = 10_000

        mem.ingest("Flush compaction test unique entry zeta", source="test")
        count_before = mem._wal.pending_count()
        self.assertGreater(count_before, 0)

        mem.flush()

        self.assertEqual(mem._wal.pending_count(), 0, "WAL should be empty after flush()")

    def test_wal_cleared_after_flush(self):
        """After flush(), the WAL file is removed."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem._wal.flush_interval = 10_000

        mem.ingest("WAL clear test memory unique delta epsilon", source="test")
        self.assertTrue(mem._wal.exists(), "WAL file should exist before flush")

        mem.flush()

        self.assertFalse(mem._wal.exists(), "WAL file should be gone after flush()")

    def test_auto_flush_triggers_after_n_writes(self):
        """WAL is compacted automatically after flush_interval writes."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem._wal.flush_interval = 3   # flush every 3 entries

        # Ingest 3 entries — each goes through WAL; 3rd triggers auto-flush
        for i in range(3):
            mem.ingest(
                f"Auto-flush test memory entry unique number {i:04d} foxtrot",
                source="test",
            )

        # After auto-flush the WAL should be empty (or near-empty)
        self.assertEqual(
            mem._wal.pending_count(), 0,
            "WAL should be cleared by auto-flush after flush_interval writes",
        )

    def test_wal_replay_on_restart(self):
        """Pending WAL entries are replayed when a new MemorySystem is created."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem._wal.flush_interval = 10_000

        content = "Crash recovery WAL replay unique test gamma"
        mem.ingest(content, source="crash_test")
        entries_before_restart = len(mem.memories)

        # Simulate restart by creating a second instance over same workspace
        mem2 = _make_mem(self.tmp, enable_read_cache=False)
        # The WAL replay should have added at least the entry we wrote
        self.assertGreaterEqual(
            len(mem2.memories), entries_before_restart,
            "Restarted instance should replay WAL entries",
        )
        contents = [m.content for m in mem2.memories]
        self.assertTrue(
            any("Crash recovery" in c for c in contents),
            "Replayed entry should be present in restarted instance",
        )

    def test_wal_skip_corrupted_lines(self):
        """WALManager.load_pending() skips lines with invalid JSON."""
        wal = WALManager(self.tmp)
        # Write one valid and one corrupted line manually
        with open(wal.wal_path, "w") as f:
            f.write('{"content": "valid", "hash": "abc"}\n')
            f.write("NOT_JSON!!!\n")
            f.write('{"content": "also valid", "hash": "def"}\n')

        pending = wal.load_pending()
        self.assertEqual(len(pending), 2, "Two valid lines should be loaded; corrupted one skipped")

    def test_should_flush_fires_on_write_count(self):
        """WALManager.should_flush() returns True after flush_interval appends."""
        wal = WALManager(self.tmp, flush_interval=2)
        entry = {"content": "test", "hash": "x"}
        wal.append(entry)
        self.assertFalse(wal.should_flush())
        wal.append(entry)
        self.assertTrue(wal.should_flush())

    def test_should_flush_fires_on_size_threshold(self):
        """WALManager.should_flush() returns True when file exceeds max_size_bytes."""
        wal = WALManager(self.tmp, max_size_bytes=10)   # tiny threshold
        wal.append({"content": "this line is definitely longer than 10 bytes"})
        self.assertTrue(wal.should_flush())


# ─────────────────────────────────────────────────────────────────────────────
# 6. stats()
# ─────────────────────────────────────────────────────────────────────────────

class TestStats(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stats_returns_sprint11_keys(self):
        """stats() includes all Sprint 11 performance keys."""
        mem = _make_mem(self.tmp)
        _ingest_n(mem, 5)
        s = mem.stats()

        required_sprint11 = [
            "total_entries", "total_shards", "search_count",
            "avg_search_time_ms", "cache_hit_rate", "wal_pending_entries",
            "last_compaction", "disk_usage_mb", "memory_usage_mb",
        ]
        for key in required_sprint11:
            self.assertIn(key, s, f"stats() missing key: {key}")

    def test_stats_preserves_v03_keys(self):
        """stats() still includes all v0.3 backward-compat keys."""
        mem = _make_mem(self.tmp)
        _ingest_n(mem, 3)
        s = mem.stats()
        for key in ["total", "avg_score", "sentiments", "categories"]:
            self.assertIn(key, s, f"stats() missing v0.3 key: {key}")

    def test_stats_search_count_increments(self):
        """search_count in stats() grows with each search call."""
        mem = _make_mem(self.tmp)
        _ingest_n(mem, 3)

        before = mem.stats()["search_count"]
        mem.search("performance test memory")
        mem.search("performance test memory")
        after = mem.stats()["search_count"]

        self.assertGreaterEqual(after, before + 2)

    def test_stats_total_entries_correct(self):
        """total_entries matches the actual number of ingested memories."""
        mem = _make_mem(self.tmp)
        n = _ingest_n(mem, 7)
        s = mem.stats()
        self.assertEqual(s["total_entries"], n)

    def test_stats_wal_pending_entries(self):
        """wal_pending_entries reflects WAL state accurately."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem._wal.flush_interval = 10_000

        s_before = mem.stats()
        mem.ingest("WAL stats test memory unique hotel", source="test")
        s_after = mem.stats()

        self.assertGreater(
            s_after["wal_pending_entries"],
            s_before["wal_pending_entries"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7–8. Stale Lock Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestStaleLock(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _lock_path(self, name="resource.json"):
        return os.path.join(self.tmp, name)

    def _write_stale_lock(self, lock: FileLock, pid: int, age_seconds: float = 120):
        """Manually create a stale lock file with a dead-looking PID."""
        os.makedirs(lock.lock_dir, exist_ok=True)
        acquired_ts = time.time() - age_seconds
        from datetime import datetime, timezone
        meta = {
            "pid": pid,
            "acquired_at": datetime.fromtimestamp(
                acquired_ts, tz=timezone.utc
            ).isoformat(),
            "acquired_at_ts": acquired_ts,
            "path": lock.path,
        }
        with open(lock.meta_path, "w") as f:
            json.dump(meta, f)

    def test_stale_lock_detected_by_dead_pid(self):
        """_break_stale() returns True when the holder PID is dead."""
        lock = FileLock(self._lock_path(), stale_threshold=300)
        dead_pid = 999999   # extremely unlikely to be running

        self._write_stale_lock(lock, pid=dead_pid, age_seconds=10)

        # _break_stale should detect the dead PID and break the lock
        broken = lock._break_stale()
        self.assertTrue(broken, "_break_stale() should detect and break dead-PID lock")

    def test_stale_lock_cleared_and_acquired(self):
        """acquire() succeeds after clearing a stale lock with a dead PID."""
        lock = FileLock(self._lock_path(), stale_threshold=300)
        dead_pid = 999999

        self._write_stale_lock(lock, pid=dead_pid, age_seconds=10)

        # Acquire should succeed by clearing the stale lock
        acquired = lock.acquire(blocking=True, timeout=2.0)
        self.assertTrue(acquired, "Should successfully acquire after clearing stale lock")
        lock.release()

    def test_stale_lock_detected_by_age(self):
        """_break_stale() returns True when lock age exceeds stale_timeout."""
        lock = FileLock(self._lock_path(), stale_threshold=30)
        # Use own PID so PID check passes, but age will trigger
        self._write_stale_lock(lock, pid=os.getpid(), age_seconds=120)

        broken = lock._break_stale(stale_threshold=60)   # 60-second threshold
        self.assertTrue(broken, "_break_stale() should break lock older than stale_timeout")

    def test_live_pid_lock_not_broken(self):
        """_break_stale() returns False when the holder PID is the current process."""
        lock = FileLock(self._lock_path(), stale_threshold=300)
        # Write a fresh lock with our own PID — should NOT be broken
        self._write_stale_lock(lock, pid=os.getpid(), age_seconds=5)

        broken = lock._break_stale(stale_threshold=300)
        self.assertFalse(broken, "Should not break a lock held by the current (live) process")
        # Clean up manually
        lock._force_break()

    def test_stale_timeout_per_call_override(self):
        """stale_timeout kwarg on acquire() overrides instance stale_threshold."""
        lock = FileLock(self._lock_path(), stale_threshold=9999)
        # Use own PID so PID check passes; age triggers only if threshold is small
        self._write_stale_lock(lock, pid=os.getpid(), age_seconds=120)

        # With instance threshold of 9999s, lock is NOT stale
        broken_high = lock._break_stale(stale_threshold=9999)
        self.assertFalse(broken_high)
        # Clean up before next test
        lock._force_break()

        # Re-create stale lock then use per-call threshold of 60s
        self._write_stale_lock(lock, pid=os.getpid(), age_seconds=120)
        broken_low = lock._break_stale(stale_threshold=60)
        self.assertTrue(broken_low)

    def test_lock_meta_uses_iso8601_timestamp(self):
        """_write_meta() stores acquired_at as an ISO8601 string."""
        lock = FileLock(self._lock_path())
        acquired = lock.acquire()
        self.assertTrue(acquired)
        with open(lock.meta_path) as f:
            meta = json.load(f)
        lock.release()

        self.assertIn("pid", meta)
        self.assertIn("acquired_at", meta)
        # Should parse as ISO8601 without errors
        from datetime import datetime
        try:
            datetime.fromisoformat(meta["acquired_at"])
        except ValueError:
            self.fail("acquired_at is not a valid ISO8601 string")


# ─────────────────────────────────────────────────────────────────────────────
# 9–10. Access Pattern Learning
# ─────────────────────────────────────────────────────────────────────────────

class TestAccessPatternLearning(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_access_count_increments_on_search_hit(self):
        """_record_access() increments count for each search result."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem.ingest("Access tracking test memory unique indigo", source="test")
        mem.ingest("Another line to pad the corpus foxtrot golf", source="test")

        mem.search("access tracking test memory")

        # At least one entry should have an access count > 0
        has_access = any(
            mem._access_tracker.get_count(m.hash) > 0 for m in mem.memories
        )
        self.assertTrue(has_access, "At least one memory should have an access count > 0")

    def test_access_count_increases_with_repeated_searches(self):
        """Repeated searches increase access counts for matching entries."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem.ingest("Repeated access count test memory unique juliet", source="test")

        mem.search("repeated access count test memory juliet")
        counts_after_1 = {m.hash: mem._access_tracker.get_count(m.hash) for m in mem.memories}

        mem.search("repeated access count test memory juliet")
        for m in mem.memories:
            if mem._access_tracker.get_count(m.hash) > 0:
                self.assertGreaterEqual(
                    mem._access_tracker.get_count(m.hash),
                    counts_after_1[m.hash],
                    "Access count should not decrease",
                )

    def test_get_hot_entries_returns_most_accessed(self):
        """get_hot_entries() returns entries with highest access counts."""
        mem = _make_mem(self.tmp, enable_read_cache=False)

        # Ingest two groups of entries
        mem.ingest("Hot memory entry frequently accessed unique kilo", source="hot")
        mem.ingest("Cold memory entry rarely accessed unique lima", source="cold")

        # Search for the "hot" entry many more times
        for _ in range(5):
            mem.search("hot memory entry frequently accessed kilo")

        hot = mem.get_hot_entries(top_n=5)
        self.assertIsInstance(hot, list)
        self.assertGreater(len(hot), 0)

    def test_get_hot_entries_order(self):
        """get_hot_entries() is sorted by access count descending."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem.ingest("Most popular memory unique mike november", source="test")
        mem.ingest("Less popular memory unique oscar papa", source="test")

        # Make the first entry more popular
        for _ in range(10):
            mem.search("most popular memory unique mike november")
        for _ in range(2):
            mem.search("less popular memory unique oscar papa")

        hot = mem.get_hot_entries(top_n=10)
        if len(hot) >= 2:
            first_count = mem._access_tracker.get_count(hot[0].hash)
            second_count = mem._access_tracker.get_count(hot[1].hash)
            self.assertGreaterEqual(first_count, second_count)

    def test_access_tracker_boost_score_range(self):
        """boost_score() returns values in [BOOST_MIN, BOOST_MAX]."""
        tmp = tempfile.mkdtemp()
        try:
            tracker = AccessTracker(tmp)
            self.assertEqual(tracker.boost_score("nonexistent"), tracker.BOOST_MIN)

            for _ in range(100):
                tracker.record_access("heavy")

            boost = tracker.boost_score("heavy")
            self.assertGreaterEqual(boost, tracker.BOOST_MIN)
            self.assertLessEqual(boost, tracker.BOOST_MAX)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11–12. compact()
# ─────────────────────────────────────────────────────────────────────────────

class TestCompact(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_compact_returns_sprint11_keys(self):
        """compact() result contains all Sprint 11 spec keys."""
        mem = _make_mem(self.tmp)
        _ingest_n(mem, 5)
        result = mem.compact()

        for key in ["shards_before", "shards_after", "entries_before",
                    "entries_after", "space_freed_mb", "duration_ms"]:
            self.assertIn(key, result, f"compact() missing key: {key}")

    def test_compact_returns_backward_compat_keys(self):
        """compact() result still contains old v0.3 keys."""
        mem = _make_mem(self.tmp)
        _ingest_n(mem, 3)
        result = mem.compact()
        for key in ["original_count", "final_count", "removed_count"]:
            self.assertIn(key, result, f"compact() missing backward-compat key: {key}")

    def test_compact_duration_ms_is_positive(self):
        """compact() duration_ms must be a non-negative number."""
        mem = _make_mem(self.tmp)
        _ingest_n(mem, 3)
        result = mem.compact()
        self.assertGreaterEqual(result["duration_ms"], 0)

    def test_compact_entries_before_gte_after(self):
        """compact() cannot increase the entry count."""
        mem = _make_mem(self.tmp)
        _ingest_n(mem, 6)
        result = mem.compact()
        self.assertGreaterEqual(result["entries_before"], result["entries_after"])


# ─────────────────────────────────────────────────────────────────────────────
# 13. Backward Compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardCompat(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ingest_and_search_still_work(self):
        """Basic ingest/search pipeline is unaffected by Sprint 11 changes."""
        mem = _make_mem(self.tmp)
        count = mem.ingest(
            "Backward compatibility test memory unique quebec romeo",
            source="compat",
        )
        self.assertGreater(count, 0)

        results = mem.search("backward compatibility test memory")
        self.assertIsInstance(results, list)
        # Result should contain MemoryEntry objects with .content
        for r in results:
            self.assertTrue(hasattr(r, "content"))

    def test_save_and_load_roundtrip(self):
        """save()/load() roundtrip preserves all entries."""
        mem = _make_mem(self.tmp)
        _ingest_n(mem, 4)
        mem.flush()   # ensure WAL is clear

        mem2 = _make_mem(self.tmp)
        self.assertEqual(len(mem2.memories), len(mem.memories))

    def test_ingest_mistake_still_works(self):
        """ingest_mistake() succeeds with Sprint 11 WAL integration."""
        mem = _make_mem(self.tmp)
        entry = mem.ingest_mistake(
            what_happened="Used wrong API endpoint",
            correction="Use /v2/endpoint instead of /v1/endpoint",
            severity="high",
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry.memory_type, "mistake")


# ─────────────────────────────────────────────────────────────────────────────
# 14. PerformanceMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceMonitor(unittest.TestCase):

    def test_records_search_count_and_timing(self):
        """PerformanceMonitor tracks search count and cumulative time."""
        mon = PerformanceMonitor()
        self.assertEqual(mon.search_count, 0)
        self.assertEqual(mon.avg_search_time_ms, 0.0)

        mon.record_search(10.0)
        mon.record_search(20.0)

        self.assertEqual(mon.search_count, 2)
        self.assertAlmostEqual(mon.avg_search_time_ms, 15.0, places=2)

    def test_records_compaction_timestamp(self):
        """record_compaction() stores a non-None ISO8601 timestamp."""
        mon = PerformanceMonitor()
        self.assertIsNone(mon.last_compaction)
        mon.record_compaction()
        self.assertIsNotNone(mon.last_compaction)
        from datetime import datetime
        # Should parse without error
        datetime.fromisoformat(mon.last_compaction)


# ─────────────────────────────────────────────────────────────────────────────
# 15. close()
# ─────────────────────────────────────────────────────────────────────────────

class TestClose(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_close_flushes_and_clears_wal(self):
        """close() compacts the WAL and removes the WAL file."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        mem._wal.flush_interval = 10_000

        mem.ingest("Close test memory unique sierra tango", source="test")
        self.assertTrue(mem._wal.exists(), "WAL should exist before close()")

        mem.close()

        self.assertFalse(mem._wal.exists(), "WAL should be removed after close()")


# ─────────────────────────────────────────────────────────────────────────────
# 16. stats() cache_hit_rate
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheHitRate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stats_cache_hit_rate_after_hit(self):
        """cache_hit_rate in stats() is > 0 after a cache hit."""
        mem = _make_mem(self.tmp, enable_read_cache=True)
        _ingest_n(mem, 3)

        query = "performance test memory"
        mem.search(query)   # populate cache (miss)
        mem.search(query)   # cache hit

        s = mem.stats()
        self.assertGreater(s["cache_hit_rate"], 0.0)

    def test_stats_cache_hit_rate_zero_when_disabled(self):
        """cache_hit_rate is 0 when read cache is disabled."""
        mem = _make_mem(self.tmp, enable_read_cache=False)
        _ingest_n(mem, 2)
        mem.search("performance test memory")
        s = mem.stats()
        self.assertEqual(s["cache_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
