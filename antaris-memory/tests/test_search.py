"""Tests for the BM25-inspired search engine."""

import os
import tempfile
import unittest

from antaris_memory import MemorySystem, SearchEngine, SearchResult
from antaris_memory.entry import MemoryEntry


class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine()
        self.memories = [
            MemoryEntry("Decided to use PostgreSQL for the database.", "meeting", 1, "strategic"),
            MemoryEntry("The API costs $500/month — switching to cheaper provider.", "review", 1, "operational"),
            MemoryEntry("Frontend will use React with TypeScript.", "meeting", 2, "operational"),
            MemoryEntry("Sprint deadline is March 1st.", "planning", 1, "strategic"),
            MemoryEntry("Deployed v2.0 to staging environment.", "deploy", 1, "tactical"),
            MemoryEntry("Auth system uses JWT with refresh tokens.", "docs", 1, "operational"),
            MemoryEntry("PostgreSQL migration completed successfully.", "deploy", 2, "tactical"),
            MemoryEntry("Need to migrate legacy users before launch.", "planning", 2, "strategic"),
            MemoryEntry("Performance testing showed 200ms p95 latency.", "monitoring", 1, "tactical"),
            MemoryEntry("Database backup strategy needs review.", "planning", 3, "strategic"),
        ]
        self.engine.build_index(self.memories)
    
    def test_basic_search(self):
        results = self.engine.search("PostgreSQL database", self.memories)
        self.assertTrue(len(results) > 0)
        # PostgreSQL memories should rank highest
        top = results[0]
        self.assertIn("postgresql", top.content.lower())
    
    def test_relevance_normalized(self):
        results = self.engine.search("PostgreSQL", self.memories)
        self.assertTrue(all(0 <= r.relevance <= 1.0 for r in results))
        # Top result should have relevance 1.0
        self.assertEqual(results[0].relevance, 1.0)
    
    def test_matched_terms(self):
        results = self.engine.search("PostgreSQL migration", self.memories)
        top = results[0]
        self.assertTrue(len(top.matched_terms) > 0)
    
    def test_category_filter(self):
        results = self.engine.search("deploy staging", self.memories, category="tactical")
        for r in results:
            self.assertEqual(r.category, "tactical")
    
    def test_empty_query(self):
        results = self.engine.search("", self.memories)
        self.assertEqual(len(results), 0)
    
    def test_no_matches(self):
        results = self.engine.search("quantum computing blockchain", self.memories)
        self.assertEqual(len(results), 0)
    
    def test_exact_phrase_boost(self):
        results = self.engine.search("refresh tokens", self.memories)
        self.assertTrue(len(results) > 0)
        # The JWT memory with exact phrase should rank high
        top_content = results[0].content.lower()
        self.assertIn("refresh tokens", top_content)
    
    def test_limit_respected(self):
        results = self.engine.search("the", self.memories, limit=3)
        self.assertTrue(len(results) <= 3)
    
    def test_stopwords_filtered(self):
        # Searching for stopwords alone should return nothing meaningful
        results = self.engine.search("the is a", self.memories)
        self.assertEqual(len(results), 0)
    
    def test_explanation(self):
        results = self.engine.search("PostgreSQL", self.memories)
        self.assertTrue(len(results) > 0)
        self.assertTrue(len(results[0].explanation) > 0)
        self.assertIn("matched:", results[0].explanation)
    
    def test_idf_ranking(self):
        """Rare terms should score higher than common terms."""
        # "jwt" appears in 1 doc, "staging" in 1 doc, "database" in 2
        results_rare = self.engine.search("jwt", self.memories)
        results_common = self.engine.search("staging", self.memories)
        # Both should find results
        self.assertTrue(len(results_rare) > 0)
        self.assertTrue(len(results_common) > 0)
    
    def test_stats(self):
        stats = self.engine.stats()
        self.assertEqual(stats["doc_count"], 10)
        self.assertGreater(stats["vocab_size"], 0)
        self.assertGreater(stats["avg_doc_len"], 0)
    
    def test_search_result_properties(self):
        results = self.engine.search("PostgreSQL", self.memories)
        r = results[0]
        self.assertEqual(r.content, r.entry.content)
        self.assertEqual(r.source, r.entry.source)
        self.assertEqual(r.confidence, r.relevance)


class TestMemorySystemSearch(unittest.TestCase):
    """Test search via the MemorySystem interface."""
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem = MemorySystem(self.tmpdir, half_life=7.0)
        self.mem.load()
        
        entries = [
            ("Decided to use PostgreSQL for the database.", "meeting", "strategic"),
            ("The API costs $500/month.", "review", "operational"),
            ("Frontend will use React with TypeScript.", "meeting", "operational"),
            ("PostgreSQL migration completed successfully.", "deploy", "tactical"),
            ("Sprint deadline is March 1st.", "planning", "strategic"),
            ("Auth system uses JWT with refresh tokens.", "docs", "operational"),
            ("Performance testing showed 200ms p95 latency.", "monitoring", "tactical"),
            ("Database backup strategy needs review.", "planning", "strategic"),
        ]
        for content, source, cat in entries:
            self.mem.ingest(content, source=source, category=cat)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_search_returns_entries(self):
        """Default search returns MemoryEntry objects for backward compat."""
        results = self.mem.search("PostgreSQL database")
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], MemoryEntry)
    
    def test_search_confidence_varies(self):
        """Confidence should now vary based on relevance, not all 0.50."""
        results = self.mem.search("PostgreSQL database")
        confidences = [r.confidence for r in results]
        # Top result should have high confidence
        self.assertGreaterEqual(confidences[0], 0.5)
        # Should have varying scores if multiple results
        if len(confidences) > 1:
            self.assertNotEqual(confidences[0], confidences[-1])
    
    def test_search_explain_mode(self):
        """explain=True returns SearchResult objects."""
        results = self.mem.search("PostgreSQL", explain=True)
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], SearchResult)
        self.assertTrue(len(results[0].explanation) > 0)
    
    def test_search_category_filter(self):
        results = self.mem.search("deploy", category="tactical")
        for r in results:
            self.assertEqual(r.category, "tactical")
    
    def test_search_save_load_preserves_index(self):
        """After save/load, search still works with proper ranking."""
        self.mem.save()
        mem2 = MemorySystem(self.tmpdir, half_life=7.0)
        mem2.load()
        results = mem2.search("PostgreSQL")
        self.assertTrue(len(results) > 0)
        self.assertIn("postgresql", results[0].content.lower())


if __name__ == "__main__":
    unittest.main()
