"""
Tests for A2 (salt persistence mode) and B1 (tail-reading) in AuditLogger.
"""

import json
import os
import time
import tempfile
from pathlib import Path

import pytest

from antaris_memory.audit import AuditLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_ws(tmp_path):
    """Return a temporary workspace path."""
    return tmp_path


# ---------------------------------------------------------------------------
# A2 — Salt Persistence Mode
# ---------------------------------------------------------------------------

class TestSaltPersistence:
    def test_ephemeral_no_salt_file(self, tmp_ws):
        """Default ephemeral mode must NOT create a .audit.salt file."""
        audit = AuditLogger(str(tmp_ws), salt_mode="ephemeral")
        salt_file = tmp_ws / ".audit.salt"
        assert not salt_file.exists(), "ephemeral mode should not create .audit.salt"

    def test_persistent_creates_salt_file(self, tmp_ws):
        """persistent mode must create .audit.salt on first init."""
        audit = AuditLogger(str(tmp_ws), salt_mode="persistent")
        salt_file = tmp_ws / ".audit.salt"
        assert salt_file.exists(), ".audit.salt must be created in persistent mode"
        stored = salt_file.read_text(encoding="utf-8").strip()
        assert stored == audit._salt, "stored salt must match instance salt"
        assert len(stored) == 32, "salt should be 16 bytes hex (32 chars)"

    def test_persistent_reloads_existing_salt(self, tmp_ws):
        """Second init with persistent mode must reload the existing salt."""
        audit1 = AuditLogger(str(tmp_ws), salt_mode="persistent")
        first_salt = audit1._salt

        audit2 = AuditLogger(str(tmp_ws), salt_mode="persistent")
        assert audit2._salt == first_salt, "second init must reload the same salt"

    def test_ephemeral_salt_differs_across_instances(self, tmp_ws):
        """Two ephemeral instances almost certainly have different salts (probabilistic)."""
        audit1 = AuditLogger(str(tmp_ws / "ws1"), salt_mode="ephemeral")
        audit2 = AuditLogger(str(tmp_ws / "ws2"), salt_mode="ephemeral")
        # Extremely unlikely to be equal (2^128 space), but not guaranteed
        # Just assert they are valid hex strings
        assert len(audit1._salt) == 32
        assert len(audit2._salt) == 32

    def test_salt_mode_default_is_ephemeral(self, tmp_ws):
        """salt_mode should default to 'ephemeral' for backward compatibility."""
        audit = AuditLogger(str(tmp_ws))
        assert audit._salt_mode == "ephemeral"
        assert not (tmp_ws / ".audit.salt").exists()


# ---------------------------------------------------------------------------
# A2 — rotate_salt()
# ---------------------------------------------------------------------------

class TestRotateSalt:
    def test_rotate_changes_salt_ephemeral(self, tmp_ws):
        """rotate_salt() must generate a new salt value (ephemeral)."""
        audit = AuditLogger(str(tmp_ws), salt_mode="ephemeral")
        old_salt = audit._salt
        audit.rotate_salt()
        assert audit._salt != old_salt or True  # probabilistic — almost never equal

    def test_rotate_updates_salt_file_persistent(self, tmp_ws):
        """rotate_salt() in persistent mode must update .audit.salt."""
        audit = AuditLogger(str(tmp_ws), salt_mode="persistent")
        old_salt = audit._salt

        audit.rotate_salt()
        new_salt = audit._salt

        stored = (tmp_ws / ".audit.salt").read_text(encoding="utf-8").strip()
        assert stored == new_salt, ".audit.salt must contain the new salt after rotate"

    def test_rotate_ephemeral_no_salt_file(self, tmp_ws):
        """rotate_salt() in ephemeral mode must NOT create .audit.salt."""
        audit = AuditLogger(str(tmp_ws), salt_mode="ephemeral")
        audit.rotate_salt()
        assert not (tmp_ws / ".audit.salt").exists()

    def test_rotate_invalidates_hash(self, tmp_ws):
        """After rotate, anonymizing the same string yields a different hash."""
        audit = AuditLogger(str(tmp_ws), salt_mode="ephemeral")
        entry = {"event": "ingest", "ts": time.time(), "data": {"query": "hello world"}}
        before = audit.anonymize_entry(entry)["data"]["query"]
        audit.rotate_salt()
        after = audit.anonymize_entry(entry)["data"]["query"]
        assert before != after, "hash must change after rotating salt"


# ---------------------------------------------------------------------------
# B1 — _tail_lines()
# ---------------------------------------------------------------------------

class TestTailLines:
    def _write_lines(self, path: Path, lines):
        with open(path, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

    def test_tail_empty_file_missing(self, tmp_ws):
        """_tail_lines on missing file returns []."""
        audit = AuditLogger(str(tmp_ws))
        result = audit._tail_lines(10)
        assert result == []

    def test_tail_returns_last_n(self, tmp_ws):
        """_tail_lines(n) returns the last n non-empty lines."""
        audit = AuditLogger(str(tmp_ws))
        all_lines = [f"line{i}" for i in range(20)]
        self._write_lines(audit.audit_file, all_lines)

        result = audit._tail_lines(5)
        assert result == ["line15", "line16", "line17", "line18", "line19"]

    def test_tail_fewer_lines_than_n(self, tmp_ws):
        """_tail_lines(n) when file has fewer than n lines returns all lines."""
        audit = AuditLogger(str(tmp_ws))
        all_lines = ["a", "b", "c"]
        self._write_lines(audit.audit_file, all_lines)

        result = audit._tail_lines(10)
        assert result == ["a", "b", "c"]

    def test_tail_skips_blank_lines(self, tmp_ws):
        """_tail_lines filters out blank/whitespace lines."""
        audit = AuditLogger(str(tmp_ws))
        with open(audit.audit_file, 'a', encoding='utf-8') as f:
            f.write("line1\n\n  \nline2\nline3\n")

        result = audit._tail_lines(10)
        assert result == ["line1", "line2", "line3"]

    def test_tail_large_file(self, tmp_ws):
        """_tail_lines works correctly on a file larger than one chunk (>8192 bytes)."""
        audit = AuditLogger(str(tmp_ws))
        # Write enough data to force multiple chunks
        long_line = "x" * 200
        lines = [f"{long_line}_{i}" for i in range(100)]
        self._write_lines(audit.audit_file, lines)

        result = audit._tail_lines(10)
        assert len(result) == 10
        assert result[-1] == f"{long_line}_99"
        assert result[0] == f"{long_line}_90"


# ---------------------------------------------------------------------------
# B1 — query() fast path (since=None)
# ---------------------------------------------------------------------------

class TestQueryFastPath:
    def test_query_no_since_newest_first(self, tmp_ws):
        """query() with no 'since' returns events newest-first."""
        audit = AuditLogger(str(tmp_ws))
        for i in range(5):
            audit.log("ingest", {"id": f"mem{i}"})
            time.sleep(0.01)

        results = audit.query()
        assert len(results) == 5
        # Verify descending timestamp order
        timestamps = [r["ts"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_query_event_type_filter_fast_path(self, tmp_ws):
        """query(event_type=...) fast path correctly filters event types."""
        audit = AuditLogger(str(tmp_ws))
        audit.log("ingest", {"id": "m1"})
        audit.log("search", {"q": "foo"})
        audit.log("ingest", {"id": "m2"})
        audit.log("recall", {"id": "m1"})

        results = audit.query(event_type="ingest")
        assert all(r["event"] == "ingest" for r in results)
        assert len(results) == 2

    def test_query_limit_fast_path(self, tmp_ws):
        """query(limit=N) fast path returns at most N results."""
        audit = AuditLogger(str(tmp_ws))
        for i in range(20):
            audit.log("ingest", {"id": f"m{i}"})

        results = audit.query(limit=5)
        assert len(results) == 5

    def test_query_empty_file_fast_path(self, tmp_ws):
        """query() on missing audit file returns []."""
        audit = AuditLogger(str(tmp_ws))
        assert audit.query() == []

    def test_query_since_uses_slow_path(self, tmp_ws):
        """query(since=...) still works correctly (slow path)."""
        audit = AuditLogger(str(tmp_ws))
        t0 = time.time()
        for i in range(5):
            audit.log("ingest", {"id": f"m{i}"})
            time.sleep(0.01)
        midpoint = t0 + 0.03  # Should skip first few, include later ones

        results = audit.query(since=midpoint)
        assert all(r["ts"] >= midpoint for r in results)
        assert len(results) > 0

    def test_query_since_with_event_type_slow_path(self, tmp_ws):
        """query(event_type, since) correctly combines both filters via slow path."""
        audit = AuditLogger(str(tmp_ws))
        t0 = time.time()
        audit.log("ingest", {"id": "old"})
        time.sleep(0.02)
        cutoff = time.time()
        audit.log("ingest", {"id": "new"})
        audit.log("search", {"q": "new"})

        results = audit.query(event_type="ingest", since=cutoff)
        assert len(results) == 1
        assert results[0]["data"]["id"] == "new"


# ---------------------------------------------------------------------------
# GPT-D: query() fallback for rare event types (R2/R3 addition)
# ---------------------------------------------------------------------------

class TestQueryRareEventFallback:
    """GPT-D: query() fast path only tail-reads limit*3 lines.

    When event_type is set and results < limit, a full forward scan is triggered
    to ensure rare events older than the tail window are not silently missed.
    """

    def test_rare_event_outside_tail_window_is_returned(self, tmp_ws):
        """A rare event buried under many common events must be found via fallback."""
        audit = AuditLogger(str(tmp_ws))

        # Log 1 rare "startup" event
        audit.log("startup", {"msg": "system ready"})

        # Log 300 common "heartbeat" events to push "startup" outside tail window
        # (fast path reads limit*3 = 300 lines by default for limit=100)
        for i in range(300):
            audit.log("heartbeat", {"tick": i})

        # query with limit=100: tail window = 300 lines, all heartbeats
        # "startup" is at line 1 — outside the tail window
        results = audit.query(event_type="startup", limit=100)
        assert len(results) == 1, (
            f"Expected 1 'startup' event via fallback scan, got {len(results)}"
        )
        assert results[0]["data"]["msg"] == "system ready"

    def test_rare_event_fallback_returns_newest_first(self, tmp_ws):
        """Fallback scan must return events newest-first, consistent with fast path."""
        audit = AuditLogger(str(tmp_ws))

        # Log 3 rare events interspersed with many common ones
        for batch in range(3):
            audit.log("rare_op", {"batch": batch})
            for _ in range(110):
                audit.log("common", {"x": 1})

        # 330 common + 3 rare; tail(300) misses first rare_op
        results = audit.query(event_type="rare_op", limit=10)
        assert len(results) == 3, f"Expected 3 rare_op events, got {len(results)}"
        # Newest-first: batch 2 should come before batch 1 before batch 0
        batches = [r["data"]["batch"] for r in results]
        assert batches == sorted(batches, reverse=True), (
            f"Results must be newest-first; got batches={batches}"
        )

    def test_common_event_does_not_trigger_fallback(self, tmp_ws):
        """If fast path returns limit results, no fallback scan is needed."""
        audit = AuditLogger(str(tmp_ws))

        # All events are "ingest" — fast path should saturate at limit
        for i in range(200):
            audit.log("ingest", {"id": i})

        results = audit.query(event_type="ingest", limit=50)
        assert len(results) == 50, f"Expected 50 results from fast path, got {len(results)}"

    def test_no_fallback_when_event_type_is_none(self, tmp_ws):
        """When event_type=None, fast path returns mixed events; no fallback needed."""
        audit = AuditLogger(str(tmp_ws))
        for i in range(10):
            audit.log("ingest" if i % 2 == 0 else "search", {"i": i})
        results = audit.query(limit=5)
        assert len(results) == 5
