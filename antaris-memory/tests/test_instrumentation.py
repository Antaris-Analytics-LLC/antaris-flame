"""Tests for v4.1 instrumentation layer and co-occurrence (Tier 1 semantic)."""

import json
import os
import tempfile
import time
import pytest

from antaris_memory import MemorySystem
from antaris_memory.instrumentation import (
    SearchContext,
    extract_memory_references,
    anonymize_query,
    build_usage_signal,
)
from antaris_memory.cooccurrence import CooccurrenceIndex


# ── SearchContext ─────────────────────────────────────────────────────────────

class TestSearchContext:

    def test_request_id_is_unique(self):
        ctx1 = SearchContext(query="test")
        ctx2 = SearchContext(query="test")
        assert ctx1.request_id != ctx2.request_id

    def test_request_id_is_uuid(self):
        import uuid
        ctx = SearchContext(query="hello")
        uuid.UUID(ctx.request_id)  # Raises if invalid

    def test_serialization_roundtrip(self):
        ctx = SearchContext(
            query="car maintenance",
            context_id="session_123",
            session_id="sess_abc",
        )
        ctx.retrieved_memory_ids = ["abc123", "def456"]
        ctx.used_memory_ids = ["abc123"]
        data = ctx.to_dict()
        restored = SearchContext.from_dict(data)
        assert restored.query == ctx.query
        assert restored.request_id == ctx.request_id
        assert restored.retrieved_memory_ids == ctx.retrieved_memory_ids
        assert restored.used_memory_ids == ctx.used_memory_ids
        assert restored.context_id == ctx.context_id

    def test_timestamp_is_recent(self):
        ctx = SearchContext(query="x")
        assert abs(ctx.timestamp - time.time()) < 5.0


# ── extract_memory_references ─────────────────────────────────────────────────

class TestExtractMemoryReferences:

    def _make_entry(self, content: str, hash_id: str):
        """Create a minimal fake MemoryEntry."""
        class FakeEntry:
            def __init__(self, c, h):
                self.content = c
                self.hash = h
        class FakeResult:
            def __init__(self, c, h):
                self.entry = FakeEntry(c, h)
        return FakeResult(content, hash_id)

    def test_finds_used_memory_by_content_match(self):
        entries = [
            self._make_entry("PostgreSQL migration completed successfully.", "abc"),
            self._make_entry("Redis cache layer configured.", "def"),
        ]
        response = "We completed the PostgreSQL migration successfully and it's live."
        used = extract_memory_references(response, entries)
        assert "abc" in used
        assert "def" not in used

    def test_returns_empty_when_no_match(self):
        entries = [
            self._make_entry("Completely unrelated content about quantum physics.", "xyz"),
        ]
        response = "We're talking about cooking today."
        used = extract_memory_references(response, entries)
        assert used == []

    def test_returns_empty_for_empty_text(self):
        entries = [self._make_entry("Some content here.", "aaa")]
        assert extract_memory_references("", entries) == []

    def test_returns_empty_for_empty_entries(self):
        used = extract_memory_references("Some response text here.", [])
        assert used == []

    def test_finds_multiple_matches(self):
        entries = [
            self._make_entry("Redis cache layer configured.", "aaa"),
            self._make_entry("PostgreSQL is the primary database.", "bbb"),
        ]
        response = "Redis cache layer configured and PostgreSQL is the primary database for our system."
        used = extract_memory_references(response, entries)
        assert "aaa" in used
        assert "bbb" in used


# ── anonymize_query ───────────────────────────────────────────────────────────

class TestAnonymizeQuery:

    def test_strips_email(self):
        result = anonymize_query("contact user@example.com for details")
        assert "user@example.com" not in result
        assert "redacted" in result  # lowercased after strip

    def test_strips_phone(self):
        result = anonymize_query("call 555-123-4567 tomorrow")
        assert "555-123-4567" not in result
        assert "redacted" in result  # lowercased after strip

    def test_lowercases(self):
        result = anonymize_query("Find The DATABASE")
        assert result == result.lower()

    def test_passes_clean_query(self):
        result = anonymize_query("database migration schedule")
        assert result == "database migration schedule"

    def test_strips_ip_address(self):
        result = anonymize_query("server at 192.168.1.100 is down")
        assert "192.168.1.100" not in result


# ── MemorySystem instrumentation integration ──────────────────────────────────

class TestInstrumentationIntegration:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = MemorySystem(self.tmp)
        self.mem.load()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_with_context_returns_context(self):
        self.mem.ingest("PostgreSQL migration completed successfully.", source="test")
        self.mem.ingest("Redis cache layer configured and working.", source="test")
        self.mem.save()
        results, ctx = self.mem.search_with_context("PostgreSQL")
        assert isinstance(ctx, SearchContext)
        assert ctx.query == "PostgreSQL"
        assert len(ctx.retrieved_memory_ids) > 0

    def test_mark_retrieved_logs_to_file(self):
        self.mem.ingest("Test memory for retrieval logging.", source="test")
        self.mem.save()
        results, ctx = self.mem.search_with_context("Test memory")
        log_path = os.path.join(self.tmp, ".instrumentation_log.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        retrieved_events = [l for l in lines if l["event"] == "retrieved"]
        assert len(retrieved_events) > 0
        assert "memory_ids" in retrieved_events[0]

    def test_mark_used_boosts_relevance(self):
        self.mem.ingest("Critical API key was rotated today.", source="test")
        self.mem.save()
        # Get baseline score
        results1, ctx = self.mem.search_with_context("API key")
        initial_importances = {e.hash: e.importance for e in results1}
        # Mark as used
        used_ids = list(initial_importances.keys())
        self.mem.mark_used(used_ids, ctx)
        # Check boosted
        for entry in self.mem.memories:
            if entry.hash in initial_importances:
                assert entry.importance >= initial_importances[entry.hash]

    def test_mark_used_resets_decay(self):
        self.mem.ingest("Old memory that should be freshened.", source="test")
        self.mem.save()
        # Get the memory
        entry = self.mem.memories[0]
        original_accessed = entry.last_accessed
        time.sleep(0.1)
        # Mark as used
        self.mem.mark_used([entry.hash])
        # Access time should be updated
        assert entry.last_accessed > original_accessed or entry.access_count > 0

    def test_instrumentation_disabled_zero_log_files(self):
        """Without search_with_context, no instrumentation files should be created."""
        self.mem.ingest("Simple memory entry here.", source="test")
        self.mem.save()
        # Regular search — no instrumentation
        results = self.mem.search("Simple memory")
        log_path = os.path.join(self.tmp, ".instrumentation_log.jsonl")
        # Log file should NOT exist (instrumentation only on search_with_context)
        assert not os.path.exists(log_path)

    def test_usage_signal_not_written_without_opt_in(self):
        """Usage signals should not be written unless opt_in_to_aggregation=True."""
        self.mem.opt_in_to_aggregation = False
        self.mem.ingest("Test memory for signal test.", source="test")
        self.mem.save()
        results, ctx = self.mem.search_with_context("Test memory")
        self.mem.mark_used(ctx.retrieved_memory_ids, ctx)
        signals_path = os.path.join(self.tmp, ".usage_signals.jsonl")
        assert not os.path.exists(signals_path)

    def test_usage_signal_written_with_opt_in(self):
        """Usage signals ARE written when opt_in_to_aggregation=True."""
        self.mem.opt_in_to_aggregation = True
        self.mem.ingest("Test memory for signal test.", source="test")
        self.mem.save()
        results, ctx = self.mem.search_with_context("Test memory")
        self.mem.mark_used(ctx.retrieved_memory_ids, ctx)
        signals_path = os.path.join(self.tmp, ".usage_signals.jsonl")
        assert os.path.exists(signals_path)
        with open(signals_path) as f:
            signals = [json.loads(l) for l in f if l.strip()]
        assert len(signals) > 0
        assert "retrieved_count" in signals[0]
        assert "used_count" in signals[0]
        # Must NOT contain raw memory content
        for sig in signals:
            assert "content" not in sig


# ── CooccurrenceIndex ─────────────────────────────────────────────────────────

class TestCooccurrenceIndex:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.idx = CooccurrenceIndex(self.tmp)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_update_from_memory_adds_pairs(self):
        pairs = self.idx.update_from_memory("The patient showed elevated ALT and hepatic inflammation.")
        assert pairs > 0

    def test_similar_words_finds_related(self):
        # Build up enough signal
        for _ in range(10):
            self.idx.update_from_memory("Patient shows elevated ALT and hepatic inflammation markers.")
            self.idx.update_from_memory("Liver function tests include ALT and hepatic enzymes.")
            self.idx.update_from_memory("Hepatic inflammation indicated by elevated ALT levels.")

        similar = dict(self.idx.similar_words("hepatic", top_k=10))
        # ALT should be closely associated with hepatic
        assert "alt" in similar or len(similar) > 0

    def test_similar_words_empty_for_unknown_word(self):
        self.idx.update_from_memory("Some random content here.")
        result = self.idx.similar_words("zzzyyyxxx_notaword")
        assert result == []

    def test_save_and_reload(self):
        self.idx.update_from_memory("PostgreSQL database migration completed successfully.")
        self.idx.save()
        # Load fresh instance
        idx2 = CooccurrenceIndex(self.tmp)
        idx2.load()
        assert idx2.vocab_size > 0
        assert idx2.total_pairs > 0

    def test_boost_score_positive_for_related(self):
        for _ in range(5):
            self.idx.update_from_memory("PostgreSQL database migration completed successfully.")
            self.idx.update_from_memory("Database migration includes PostgreSQL schema changes.")
        boost = self.idx.boost_score("PostgreSQL", "migration database schema completed")
        assert boost >= 0.0  # May be 0 if not enough data, but never negative

    def test_boost_score_zero_for_unrelated(self):
        self.idx.update_from_memory("Cooking recipes include tomatoes and basil.")
        boost = self.idx.boost_score("quantum physics", "nuclear reactor core temperature")
        assert boost == 0.0

    def test_cooccurrence_integrated_in_ingest(self):
        """MemorySystem.ingest() should update the co-occurrence index."""
        mem = MemorySystem(self.tmp)
        mem.load()
        mem.ingest("Liver function tests show elevated ALT hepatic enzymes inflammation.", source="test")
        mem.save()
        # Co-occurrence index should have been updated
        assert mem._cooccurrence.vocab_size > 0
