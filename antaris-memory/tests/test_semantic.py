"""
Tests for semantic search engine (antaris-memory v3.4).

Tests the SemanticEngine class API without requiring actual sentence-transformers.
Uses mocks for unit testing to avoid the dependency in CI/test environments.
"""

import json
import os
import tempfile
import unittest.mock
from pathlib import Path

import pytest

from antaris_memory.semantic import SemanticEngine, SemanticEngineError
from antaris_memory.search import SearchResult
from antaris_memory.entry import MemoryEntry


class TestSemanticEngine:
    """Test the SemanticEngine API."""

    def test_init(self, tmp_path):
        """Test SemanticEngine initialization."""
        engine = SemanticEngine(str(tmp_path))
        assert engine.workspace == tmp_path
        assert engine.model_name == "all-MiniLM-L6-v2"
        assert engine.vectors_path == tmp_path / ".vectors.json"
        assert engine.vector_count == 0

    def test_vectors_file_custom(self, tmp_path):
        """Test custom vectors filename."""
        engine = SemanticEngine(str(tmp_path), vectors_file="custom.json")
        assert engine.vectors_path == tmp_path / "custom.json"

    def test_store_and_get_vector(self, tmp_path):
        """Test vector storage and retrieval."""
        engine = SemanticEngine(str(tmp_path))
        
        # Store a vector
        test_vector = [0.1, 0.2, 0.3] * 128  # 384-dim mock vector
        engine.store_vector("mem_123", test_vector)
        
        assert engine.vector_count == 1
        retrieved = engine.get_vector("mem_123")
        assert retrieved == test_vector
        
        # Test non-existent vector
        assert engine.get_vector("nonexistent") is None

    def test_remove_vector(self, tmp_path):
        """Test vector removal."""
        engine = SemanticEngine(str(tmp_path))
        
        test_vector = [0.1] * 384
        engine.store_vector("mem_123", test_vector)
        assert engine.vector_count == 1
        
        # Remove existing vector
        result = engine.remove_vector("mem_123")
        assert result is True
        assert engine.vector_count == 0
        assert engine.get_vector("mem_123") is None
        
        # Remove non-existent vector
        result = engine.remove_vector("nonexistent")
        assert result is False

    def test_store_invalid_vector(self, tmp_path):
        """Test storing invalid vectors."""
        engine = SemanticEngine(str(tmp_path))
        
        with pytest.raises(ValueError, match="Vector must be a non-empty list"):
            engine.store_vector("mem_123", [])
            
        with pytest.raises(ValueError, match="Vector must be a non-empty list"):
            engine.store_vector("mem_123", None)

    def test_save_and_load_vectors(self, tmp_path):
        """Test vector persistence."""
        engine = SemanticEngine(str(tmp_path))
        
        # Store some vectors
        vectors = {
            "mem_1": [0.1] * 384,
            "mem_2": [0.2] * 384,
            "mem_3": [0.3] * 384,
        }
        
        for mem_id, vector in vectors.items():
            engine.store_vector(mem_id, vector)
        
        # Save to disk
        engine.save_vectors()
        assert engine.vectors_path.exists()
        
        # Create new engine and load
        engine2 = SemanticEngine(str(tmp_path))
        engine2.load_vectors()
        
        assert engine2.vector_count == 3
        for mem_id, expected_vector in vectors.items():
            assert engine2.get_vector(mem_id) == expected_vector

    def test_load_corrupted_vectors(self, tmp_path):
        """Test loading corrupted vectors file."""
        engine = SemanticEngine(str(tmp_path))
        
        # Write corrupted JSON
        with open(engine.vectors_path, "w") as f:
            f.write("invalid json{")
            
        # Should not crash, should start with empty vectors
        engine.load_vectors()
        assert engine.vector_count == 0

    def test_load_nonexistent_vectors(self, tmp_path):
        """Test loading when vectors file doesn't exist."""
        engine = SemanticEngine(str(tmp_path))
        engine.load_vectors()  # Should not crash
        assert engine.vector_count == 0

    def test_cosine_similarity(self, tmp_path):
        """Test cosine similarity calculation."""
        engine = SemanticEngine(str(tmp_path))
        
        # Identical vectors
        vec = [1.0, 0.0, 0.0]
        assert engine.cosine_similarity(vec, vec) == pytest.approx(1.0)
        
        # Orthogonal vectors
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        assert engine.cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)
        
        # Opposite vectors
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        assert engine.cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)
        
        # Different length vectors
        vec_a = [1.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        assert engine.cosine_similarity(vec_a, vec_b) == 0.0
        
        # Zero vectors
        vec_zero = [0.0, 0.0, 0.0]
        assert engine.cosine_similarity(vec_zero, vec) == 0.0

    def test_search_no_vectors(self, tmp_path):
        """Test search with no stored vectors."""
        engine = SemanticEngine(str(tmp_path))
        
        query_vector = [0.5] * 384
        results = engine.search(query_vector)
        assert results == []

    def test_search_with_vectors(self, tmp_path):
        """Test vector search."""
        engine = SemanticEngine(str(tmp_path))
        
        # Store some vectors
        engine.store_vector("mem_1", [1.0, 0.0, 0.0])
        engine.store_vector("mem_2", [0.0, 1.0, 0.0]) 
        engine.store_vector("mem_3", [0.8, 0.6, 0.0])  # Similar to query
        
        # Query similar to mem_3
        query_vector = [1.0, 0.5, 0.0]
        results = engine.search(query_vector, top_k=5)
        
        # Should return results sorted by similarity
        assert len(results) > 0
        memory_ids = [r[0] for r in results]
        similarities = [r[1] for r in results]
        
        # Check that similarities are sorted descending
        assert similarities == sorted(similarities, reverse=True)
        # mem_3 should be most similar
        assert results[0][0] == "mem_3"

    def test_embed_success(self, tmp_path):
        """Test successful text embedding (mocked)."""
        engine = SemanticEngine(str(tmp_path))
        
        # Mock the _ensure_model method to avoid importing sentence-transformers
        mock_model = unittest.mock.Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3] * 128]  # 384-dim mock
        
        with unittest.mock.patch.object(engine, '_ensure_model') as mock_ensure:
            engine._model = mock_model
            vector = engine.embed("test text")
        
        assert len(vector) == 384
        assert vector == [0.1, 0.2, 0.3] * 128
        mock_model.encode.assert_called_once()

    def test_embed_batch_success(self, tmp_path):
        """Test successful batch embedding (mocked)."""
        engine = SemanticEngine(str(tmp_path))
        
        # Mock the _ensure_model method to avoid importing sentence-transformers
        mock_model = unittest.mock.Mock()
        mock_model.encode.return_value = [
            [0.1, 0.2, 0.3] * 128,  # 384-dim mock
            [0.4, 0.5, 0.6] * 128,  # 384-dim mock
        ]
        
        with unittest.mock.patch.object(engine, '_ensure_model') as mock_ensure:
            engine._model = mock_model
            vectors = engine.embed_batch(["text1", "text2"])
        
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        assert len(vectors[1]) == 384
        mock_model.encode.assert_called_once()

    def test_embed_missing_dependency(self, tmp_path):
        """Test embedding when sentence-transformers not available."""
        # Don't mock - let import fail naturally
        with pytest.raises(SemanticEngineError, match="sentence-transformers not installed"):
            engine = SemanticEngine(str(tmp_path))
            engine.embed("test text")

    def test_embed_and_store(self, tmp_path):
        """Test embed_and_store convenience method."""
        engine = SemanticEngine(str(tmp_path))
        
        # Mock the _ensure_model method to avoid importing sentence-transformers
        mock_model = unittest.mock.Mock()
        mock_vector = [0.1, 0.2, 0.3] * 128  # 384-dim mock
        mock_model.encode.return_value = [mock_vector]
        
        with unittest.mock.patch.object(engine, '_ensure_model') as mock_ensure:
            engine._model = mock_model
            result_vector = engine.embed_and_store("mem_123", "test content")
        
        assert result_vector == mock_vector
        assert engine.get_vector("mem_123") == mock_vector

    def test_embed_batch_and_store(self, tmp_path):
        """Test embed_batch_and_store convenience method."""
        engine = SemanticEngine(str(tmp_path))
        
        # Mock the _ensure_model method to avoid importing sentence-transformers
        mock_model = unittest.mock.Mock()
        mock_vectors = [
            [0.1, 0.2, 0.3] * 128,  # 384-dim mock
            [0.4, 0.5, 0.6] * 128,  # 384-dim mock
        ]
        mock_model.encode.return_value = mock_vectors
        
        with unittest.mock.patch.object(engine, '_ensure_model') as mock_ensure:
            engine._model = mock_model
            entries = [("mem_1", "content 1"), ("mem_2", "content 2")]
            count = engine.embed_batch_and_store(entries)
        
        assert count == 2
        assert engine.get_vector("mem_1") == mock_vectors[0]
        assert engine.get_vector("mem_2") == mock_vectors[1]

    def test_rerank_no_candidates(self, tmp_path):
        """Test re-ranking with no candidates."""
        engine = SemanticEngine(str(tmp_path))
        results = engine.rerank("query", [], top_k=10)
        assert results == []

    def test_rerank_with_candidates(self, tmp_path):
        """Test re-ranking search results."""
        engine = SemanticEngine(str(tmp_path))
        
        # Mock the _ensure_model method to avoid importing sentence-transformers
        mock_model = unittest.mock.Mock()
        mock_model.encode.return_value = [[0.5] * 384]  # Query vector
        
        # Store vectors for some memories
        engine.store_vector("hash_1", [0.6] * 384)  # Similar to query
        engine.store_vector("hash_2", [0.1] * 384)  # Less similar
        
        # Create mock search results
        entry_1 = MemoryEntry("content 1")
        entry_1.hash = "hash_1"
        entry_2 = MemoryEntry("content 2") 
        entry_2.hash = "hash_2"
        
        candidates = [
            SearchResult(entry_1, 0.8, 0.8, ["term1"], "explanation1"),
            SearchResult(entry_2, 0.6, 0.6, ["term2"], "explanation2"),
        ]
        
        with unittest.mock.patch.object(engine, '_ensure_model') as mock_ensure:
            engine._model = mock_model
            results = engine.rerank("test query", candidates, top_k=5)
        
        # Should return re-ranked results
        assert len(results) == 2
        # Results should be sorted by blended score
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'entry') for r in results)

    def test_rerank_fallback_on_error(self, tmp_path):
        """Test re-ranking falls back gracefully on embedding errors."""
        engine = SemanticEngine(str(tmp_path))
        
        # Create mock search results
        entry = MemoryEntry("content")
        candidates = [SearchResult(entry, 0.8, 0.8, ["term"], "explanation")]
        
        # Should not crash and return original candidates
        results = engine.rerank("query", candidates, top_k=5)
        assert len(results) == 1

    def test_stats(self, tmp_path):
        """Test stats method."""
        engine = SemanticEngine(str(tmp_path))
        
        stats = engine.stats()
        assert "model" in stats
        assert "vector_count" in stats
        assert "model_loaded" in stats
        assert "vectors_file" in stats
        assert "vectors_exist" in stats
        
        assert stats["model"] == "all-MiniLM-L6-v2"
        assert stats["vector_count"] == 0
        assert stats["model_loaded"] is False

    def test_empty_text_embedding(self, tmp_path):
        """Test embedding empty or whitespace text."""
        engine = SemanticEngine(str(tmp_path))
        
        # Should return zero vector without trying to load model
        vector = engine.embed("")
        assert vector == [0.0] * 384
        
        vector = engine.embed("   ")
        assert vector == [0.0] * 384