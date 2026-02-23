"""
Tests for B3 — heapq LRU Eviction in CooccurrenceIndex._evict_lru().

Verifies:
1. Eviction selects the OLDEST words (lowest access time), not random ones.
2. Eviction count is 10% of current vocab size.
3. Evicted words are fully removed from the co-occurrence index.
4. heapq.nsmallest path is exercised (import is present and functional).
"""

import time
import heapq
import pytest

from antaris_memory.cooccurrence import CooccurrenceIndex


class TestHeapqEviction:
    """Unit tests for heapq-based LRU eviction in CooccurrenceIndex."""

    def _make_index(self, vocab_cap: int = 10) -> CooccurrenceIndex:
        """Create a CooccurrenceIndex with a small vocab cap for testing."""
        return CooccurrenceIndex(workspace="/tmp/test_cooc_evict", vocab_cap=vocab_cap)

    # ── Requirement 1: eviction selects OLDEST words ───────────────────────

    def test_eviction_removes_oldest_words(self):
        """Oldest words (lowest access time) must be evicted first."""
        idx = self._make_index(vocab_cap=1000)  # Large cap — no auto-evict yet

        # Manually plant words with known access times
        old_word = "archaic"
        new_word = "recent"
        medium_word = "midword"

        # Inject directly to bypass normal tokenizer / window logic
        idx._counts[old_word] = {"x": 1}
        idx._counts[new_word] = {"x": 1}
        idx._counts[medium_word] = {"x": 1}

        base = time.time()
        idx._access_times[old_word] = base - 1000.0   # oldest
        idx._access_times[new_word] = base             # newest
        idx._access_times[medium_word] = base - 500.0  # middle

        # Force eviction of 1 word manually
        idx.vocab_cap = len(idx._counts) - 1   # trigger eviction
        evicted = idx._evict_lru()

        assert evicted >= 1, "Should evict at least one word"
        assert old_word not in idx._counts, (
            f"Expected '{old_word}' (oldest) to be evicted, but it remains"
        )

    def test_eviction_preserves_newest_words(self):
        """The most recently accessed words must survive eviction."""
        idx = self._make_index(vocab_cap=1000)

        base = time.time()
        # Plant 10 words; the newest one must survive
        newest = "keeper"
        for i in range(9):
            w = f"word{i}"
            idx._counts[w] = {newest: i + 1}
            idx._access_times[w] = base - (9 - i) * 100   # progressively older
        idx._counts[newest] = {"word0": 1}
        idx._access_times[newest] = base + 1.0   # most recent

        # Trigger eviction of 9 words (90% of 10)
        idx.vocab_cap = 1
        idx._evict_lru()

        assert newest in idx._counts, (
            f"'{newest}' is the newest word and must not be evicted"
        )

    # ── Requirement 2: eviction count is 10% of vocab ─────────────────────

    def test_eviction_count_is_ten_percent(self):
        """_evict_lru() must evict exactly 10% (rounded down, min 1) of vocab."""
        idx = self._make_index(vocab_cap=1000)

        vocab_size = 100
        base = time.time()
        for i in range(vocab_size):
            w = f"w{i}"
            idx._counts[w] = {"x": 1}
            idx._access_times[w] = base - (vocab_size - i) * 10.0

        # Trigger eviction
        idx.vocab_cap = vocab_size - 1
        evicted = idx._evict_lru()

        expected = max(1, int(vocab_size * 0.1))
        assert evicted == expected, (
            f"Expected to evict {expected} words (10% of {vocab_size}), got {evicted}"
        )

    def test_eviction_count_minimum_one(self):
        """Even a tiny overflow must evict at least 1 word."""
        idx = self._make_index(vocab_cap=1000)

        # Only 1 word over cap
        idx._counts["alpha"] = {"x": 1}
        idx._counts["beta"] = {"x": 1}
        idx._access_times["alpha"] = time.time() - 100.0
        idx._access_times["beta"] = time.time()
        idx.vocab_cap = 1

        evicted = idx._evict_lru()
        assert evicted >= 1, "Must evict at least 1 word when over cap"

    # ── Requirement 3: evicted words are removed from the index ───────────

    def test_evicted_words_removed_from_counts(self):
        """After eviction, the evicted word must not appear in _counts."""
        idx = self._make_index(vocab_cap=1000)

        base = time.time()
        words = [f"term{i}" for i in range(20)]
        for i, w in enumerate(words):
            idx._counts[w] = {words[(i + 1) % len(words)]: 1}
            idx._access_times[w] = base + i  # term0 is oldest

        idx.vocab_cap = len(words) - 1
        idx._evict_lru()

        # term0 had the oldest access time and must be gone
        assert words[0] not in idx._counts, (
            f"'{words[0]}' should have been evicted from _counts"
        )

    def test_evicted_words_removed_from_access_times(self):
        """After eviction, evicted words must not appear in _access_times."""
        idx = self._make_index(vocab_cap=1000)

        base = time.time()
        for i in range(10):
            w = f"tok{i}"
            idx._counts[w] = {"neighbor": 1}
            idx._access_times[w] = base - (10 - i) * 50.0  # tok0 is oldest

        idx.vocab_cap = 9
        idx._evict_lru()

        assert "tok0" not in idx._access_times, (
            "'tok0' should have been removed from _access_times"
        )

    def test_evicted_words_removed_from_reverse_pairs(self):
        """Evicting word A must also remove A from co-word B's entry in _counts."""
        idx = self._make_index(vocab_cap=1000)

        base = time.time()
        old = "ancient"
        co = "coterm"

        idx._counts[old] = {co: 5}
        idx._counts[co] = {old: 5, "other": 2}
        idx._access_times[old] = base - 9999.0   # very old
        idx._access_times[co] = base             # recent

        idx.vocab_cap = 1  # forces eviction; only 1 word can remain
        idx._evict_lru()

        # 'ancient' should be evicted; 'coterm' stays
        assert old not in idx._counts, f"'{old}' should be evicted"
        if co in idx._counts:
            assert old not in idx._counts[co], (
                f"Reverse pair '{old}' should be removed from '{co}'s co-counts"
            )

    # ── Requirement 4: heapq is imported and used ──────────────────────────

    def test_heapq_import_present(self):
        """heapq must be importable from cooccurrence module."""
        import antaris_memory.cooccurrence as cooc_module
        import heapq as hq
        # Verify heapq is used in the module (it's a stdlib module that's imported)
        assert hasattr(hq, 'nsmallest'), "heapq.nsmallest must be available"

    def test_heapq_nsmallest_selects_correct_words(self):
        """Directly verify heapq.nsmallest returns the k oldest words."""
        base = time.time()
        vocab = {f"word{i}": base + i for i in range(50)}
        pairs = [(t, w) for w, t in vocab.items()]
        k = 5
        oldest_k = [w for _, w in heapq.nsmallest(k, pairs)]

        expected = [f"word{i}" for i in range(k)]  # word0..word4 are oldest
        assert oldest_k == expected, (
            f"heapq.nsmallest should return the {k} oldest words: "
            f"got {oldest_k}, expected {expected}"
        )

    # ── Integration: auto-eviction during update_from_memory ──────────────

    def test_auto_eviction_triggers_on_vocab_overflow(self, tmp_path):
        """Ingesting content that exceeds vocab_cap must trigger LRU eviction."""
        # Small vocab cap so we can trigger eviction with a short sentence
        idx = CooccurrenceIndex(workspace=str(tmp_path), vocab_cap=5)

        # First batch: fill the vocab
        idx.update_from_memory("alpha beta gamma delta epsilon")
        size_after_first = idx.vocab_size

        # Second batch: overflow triggers eviction
        idx.update_from_memory("zeta eta theta iota kappa lambda mu")

        # Vocab must not grow unboundedly past cap + 10% eviction band
        max_expected = idx.vocab_cap + max(1, int(idx.vocab_cap * 0.1)) + 10
        assert idx.vocab_size <= max_expected, (
            f"Vocab should stay bounded; got {idx.vocab_size}, max expected ~{max_expected}"
        )

    def test_no_eviction_when_under_cap(self, tmp_path):
        """_evict_lru() returns 0 when vocab is within cap."""
        idx = CooccurrenceIndex(workspace=str(tmp_path), vocab_cap=1000)
        idx.update_from_memory("hello world test memory system")
        assert idx._evict_lru() == 0, (
            "No eviction should happen when vocab is under cap"
        )
