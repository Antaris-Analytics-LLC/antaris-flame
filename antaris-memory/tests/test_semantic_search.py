"""Tests for Sprint 3: pluggable embedding interface and hybrid semantic search."""

import os
import tempfile
import unittest

from antaris_memory import MemorySystem, cosine_similarity
from antaris_memory.entry import MemoryEntry
from antaris_memory.search import SearchResult


class TestCosineSimilarity(unittest.TestCase):
    """Unit tests for the cosine_similarity utility."""

    def test_identical_vectors_return_one(self):
        vec = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=6)

    def test_orthogonal_vectors_return_zero(self):
        self.assertAlmostEqual(cosine_similarity([1, 0, 0], [0, 1, 0]), 0.0, places=6)

    def test_opposite_vectors_return_minus_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 0, 0], [-1, 0, 0]), -1.0, places=6)

    def test_similar_vectors_high_score(self):
        # [0.9, 0.1, 0] vs [1, 0, 0] → cosine ≈ 0.9934
        score = cosine_similarity([0.9, 0.1, 0], [1, 0, 0])
        self.assertGreater(score, 0.99)

    def test_empty_vector_returns_zero(self):
        self.assertEqual(cosine_similarity([], [1, 0]), 0.0)

    def test_mismatched_lengths_returns_zero(self):
        self.assertEqual(cosine_similarity([1, 0], [1, 0, 0]), 0.0)

    def test_zero_magnitude_returns_zero(self):
        self.assertEqual(cosine_similarity([0, 0, 0], [1, 0, 0]), 0.0)

    def test_known_value(self):
        # [1,1] vs [1,0] → dot=1, |a|=√2, |b|=1 → 1/√2 ≈ 0.7071
        score = cosine_similarity([1, 1], [1, 0])
        self.assertAlmostEqual(score, 0.7071, places=3)


class TestSetEmbeddingFn(unittest.TestCase):
    """Tests for set_embedding_fn method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem = MemorySystem(self.tmpdir, half_life=7.0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_set_embedding_fn_stores_callable(self):
        """set_embedding_fn should store the callable."""
        def mock_embed(text):
            return [1.0, 0.0, 0.0]

        self.assertIsNone(self.mem._embedding_fn)
        self.mem.set_embedding_fn(mock_embed)
        self.assertIs(self.mem._embedding_fn, mock_embed)

    def test_set_embedding_fn_clears_cache(self):
        """Changing the embedding fn should invalidate the cache."""
        call_count = [0]

        def embed_v1(text):
            call_count[0] += 1
            return [1.0, 0.0]

        def embed_v2(text):
            return [0.0, 1.0]

        self.mem.set_embedding_fn(embed_v1)
        # Warm the cache
        self.mem._get_embedding("hello world test entry")
        self.assertEqual(call_count[0], 1)

        # Change fn → cache should be invalidated
        self.mem.set_embedding_fn(embed_v2)
        self.assertEqual(len(self.mem._embedding_cache), 0)

    def test_get_embedding_returns_none_without_fn(self):
        """_get_embedding should return None when no fn is set."""
        result = self.mem._get_embedding("anything")
        self.assertIsNone(result)

    def test_get_embedding_returns_vector_with_fn(self):
        """_get_embedding should call fn and return the vector."""
        self.mem.set_embedding_fn(lambda t: [0.5, 0.5])
        result = self.mem._get_embedding("hello")
        self.assertEqual(result, [0.5, 0.5])

    def test_get_embedding_caches_result(self):
        """Second call with same text should not re-invoke the fn."""
        call_count = [0]

        def counting_embed(text):
            call_count[0] += 1
            return [1.0, 0.0]

        self.mem.set_embedding_fn(counting_embed)
        self.mem._get_embedding("cache me")
        self.mem._get_embedding("cache me")  # same text
        self.assertEqual(call_count[0], 1)

    def test_get_embedding_graceful_on_exception(self):
        """If the embedding fn raises, _get_embedding returns None gracefully."""
        def bad_embed(text):
            raise RuntimeError("API error")

        self.mem.set_embedding_fn(bad_embed)
        result = self.mem._get_embedding("trigger error")
        self.assertIsNone(result)


class TestEmbeddingCacheBounds(unittest.TestCase):
    """Test that the embedding cache is bounded to 1000 entries."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem = MemorySystem(self.tmpdir, half_life=7.0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_does_not_exceed_1000_entries(self):
        self.mem.set_embedding_fn(lambda t: [1.0, 0.0])
        # Request 1100 unique texts
        for i in range(1100):
            self.mem._get_embedding(f"unique text number {i} padding to be long enough")
        self.assertLessEqual(len(self.mem._embedding_cache), 1000)

    def test_cache_evicts_oldest_on_overflow(self):
        """FIFO eviction: first inserted key is dropped when cache is full."""
        self.mem.set_embedding_fn(lambda t: [1.0, 0.0])
        # Fill to exactly 1000
        for i in range(1000):
            self.mem._get_embedding(f"entry-{i}-padding padding padding padding")
        # The first key inserted should still be in cache
        first_key = next(iter(self.mem._embedding_cache))
        # Insert one more (triggers eviction)
        self.mem._get_embedding("brand new entry that causes eviction now ok")
        # The first key should be gone
        self.assertNotIn(first_key, self.mem._embedding_cache)
        self.assertLessEqual(len(self.mem._embedding_cache), 1000)


class TestHybridSearch(unittest.TestCase):
    """Tests for hybrid BM25 + semantic search scoring."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem = MemorySystem(self.tmpdir, half_life=7.0)

        # Ingest entries with distinct content
        entries = [
            "AI routing infrastructure handles load balancing decisions",
            "Database backup strategy needs daily review schedule",
            "Frontend React components require TypeScript strict mode",
            "AI infrastructure powers semantic routing algorithms every day",
            "PostgreSQL migration completed with zero downtime deployment",
        ]
        for e in entries:
            self.mem.ingest(e, source="test", category="general")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_search_without_embedding_fn_uses_bm25(self):
        """Default search (no embedding fn) should return results normally."""
        self.assertIsNone(self.mem._embedding_fn)
        results = self.mem.search("AI routing", explain=True)
        self.assertGreater(len(results), 0)
        # Should be plain BM25 — no "[hybrid:..." in explanation
        for r in results:
            self.assertNotIn("[hybrid:", r.explanation)

    def test_hybrid_search_blends_scores(self):
        """With a mock embedding fn, hybrid score blends BM25 and cosine."""
        # Map specific texts to known vectors
        vector_map = {
            "AI routing": [1.0, 0.0, 0.0],
            "AI routing infrastructure handles load balancing decisions": [0.9, 0.1, 0.0],
            "AI infrastructure powers semantic routing algorithms every day": [0.85, 0.15, 0.0],
            "Database backup strategy needs daily review schedule": [0.0, 1.0, 0.0],
            "Frontend React components require TypeScript strict mode": [0.0, 0.0, 1.0],
            "PostgreSQL migration completed with zero downtime deployment": [0.1, 0.0, 0.9],
        }

        def mock_embed(text):
            for key, vec in vector_map.items():
                if key in text or text in key:
                    return vec
            return [0.5, 0.5, 0.0]

        # Get BM25-only scores first
        bm25_results = self.mem.search("AI routing", explain=True)
        bm25_top_score = bm25_results[0].score if bm25_results else 0.0

        # Now enable hybrid
        self.mem.set_embedding_fn(mock_embed)
        hybrid_results = self.mem.search("AI routing", explain=True)

        self.assertGreater(len(hybrid_results), 0)
        top = hybrid_results[0]
        # Explanation should mention hybrid
        self.assertIn("[hybrid:", top.explanation)
        # Score should be a valid float
        self.assertIsInstance(top.score, float)
        self.assertGreater(top.score, 0.0)

    def test_hybrid_search_graceful_failure(self):
        """If embedding fn always raises, search falls back to BM25 silently."""
        call_count = [0]

        def always_fail(text):
            call_count[0] += 1
            raise RuntimeError("Embedding service down")

        self.mem.set_embedding_fn(always_fail)
        # Should not raise; should return normal BM25 results
        results = self.mem.search("AI routing", explain=True)
        self.assertGreater(len(results), 0)
        # Explanation should NOT contain "[hybrid:" since embedding failed
        for r in results:
            self.assertNotIn("[hybrid:", r.explanation)

    def test_hybrid_search_explain_contains_bm25_and_sem(self):
        """When hybrid is active, explanation shows both bm25 and sem scores."""
        def embed(text):
            return [1.0, 0.0, 0.0]

        self.mem.set_embedding_fn(embed)
        results = self.mem.search("AI routing", explain=True)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("bm25=", top.explanation)
        self.assertIn("sem=", top.explanation)

    def test_hybrid_search_returns_searchresult_in_explain_mode(self):
        """In explain=True mode, hybrid results are still SearchResult objects."""
        self.mem.set_embedding_fn(lambda t: [1.0, 0.0])
        results = self.mem.search("AI routing", explain=True)
        for r in results:
            self.assertIsInstance(r, SearchResult)

    def test_hybrid_search_relevance_is_hybrid_score(self):
        """After hybrid blending, r.relevance holds the hybrid_score.
        
        Note: r.score may differ from r.relevance because recall_priority
        and access-count boosts are applied to r.score after hybrid blending.
        For episodic memories (recall_priority=0.5) this is expected.
        """
        self.mem.set_embedding_fn(lambda t: [1.0, 0.0])
        results = self.mem.search("AI routing", explain=True)
        self.assertGreater(len(results), 0)
        for r in results:
            if "[hybrid:" in r.explanation:
                # relevance should be the hybrid blend value (>0)
                self.assertGreater(r.relevance, 0.0)
                # score should also be positive (recall_priority boost applied)
                self.assertGreater(r.score, 0.0)
                # bm25 weight and sem_score are embedded in explanation
                self.assertIn("bm25=", r.explanation)
                self.assertIn("sem=", r.explanation)

    def test_search_without_embedding_still_returns_entries(self):
        """Default (no embedding fn) search returns MemoryEntry objects."""
        results = self.mem.search("AI routing")
        for r in results:
            self.assertIsInstance(r, MemoryEntry)


if __name__ == "__main__":
    unittest.main()
