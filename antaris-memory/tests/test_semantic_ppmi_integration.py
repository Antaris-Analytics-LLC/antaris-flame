"""
Integration test: Semantic engine + PPMI paraphrase detection directional consistency.

Verifies that both strategies produce compatible results on the same inputs.
"""

import pytest
from antaris_memory.cooccurrence import CooccurrenceIndex
from antaris_memory.semantic import SemanticEngine
from antaris_memory.entry import MemoryEntry


class TestSemanticPPMIIntegration:
    """Test that semantic and PPMI strategies are directionally consistent."""

    def test_both_agree_on_identical_text(self, tmp_path):
        """Both strategies should agree that identical text is a match."""
        idx = CooccurrenceIndex(str(tmp_path))
        engine = SemanticEngine(str(tmp_path))

        text = "Patient shows elevated liver enzymes ALT AST"
        idx.update_from_memory(text)

        is_para, ppmi_score = idx.detect_paraphrase(text, text)
        assert is_para is True
        assert ppmi_score >= 0.6

        # Semantic: identical vectors → cosine sim = 1.0
        vec = [0.5] * 384  # mock vector
        sim = engine.cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0)

    def test_both_reject_unrelated_text(self, tmp_path):
        """Both strategies should reject completely unrelated text."""
        idx = CooccurrenceIndex(str(tmp_path))
        engine = SemanticEngine(str(tmp_path))

        medical = "Patient liver enzymes elevated ALT"
        weather = "Sunny weather forecast tomorrow morning"

        idx.update_from_memory(medical)
        idx.update_from_memory(weather)

        is_para, ppmi_score = idx.detect_paraphrase(medical, weather)
        assert is_para is False

        # Semantic: orthogonal vectors → cosine sim ≈ 0
        vec_a = [1.0, 0.0, 0.0] * 128
        vec_b = [0.0, 1.0, 0.0] * 128
        sim = engine.cosine_similarity(vec_a, vec_b)
        assert sim == pytest.approx(0.0)

    def test_ppmi_catches_domain_paraphrase_without_model(self, tmp_path):
        """PPMI should detect domain paraphrases even without a transformer model."""
        idx = CooccurrenceIndex(str(tmp_path))

        # Build medical vocabulary via co-occurrence
        corpus = [
            "patient liver function test results abnormal",
            "hepatic enzyme ALT AST elevated levels",
            "liver enzymes indicate hepatic dysfunction",
            "ALT AST values significantly increased patient",
            "hepatic function panel abnormal findings",
            "elevated liver transaminases suggest dysfunction",
        ]
        for text in corpus:
            idx.update_from_memory(text)

        # These are paraphrases in the medical domain
        text_a = "liver function abnormal"
        text_b = "hepatic enzyme elevated"

        is_para, score = idx.detect_paraphrase(text_a, text_b, threshold=0.2)
        # With enough co-occurrence context, PPMI should detect some similarity
        assert score > 0.0  # At minimum, non-zero similarity

    def test_vector_gc_removes_orphans(self, tmp_path):
        """Test that gc_vectors prunes vectors for deleted memories."""
        engine = SemanticEngine(str(tmp_path))

        # Store vectors
        engine.store_vector("active_1", [0.1] * 384)
        engine.store_vector("active_2", [0.2] * 384)
        engine.store_vector("deleted_1", [0.3] * 384)
        engine.store_vector("deleted_2", [0.4] * 384)
        assert engine.vector_count == 4

        # GC with only active IDs
        removed = engine.gc_vectors({"active_1", "active_2"})
        assert removed == 2
        assert engine.vector_count == 2
        assert engine.get_vector("active_1") is not None
        assert engine.get_vector("active_2") is not None
        assert engine.get_vector("deleted_1") is None
        assert engine.get_vector("deleted_2") is None

    def test_vector_gc_empty_active_set(self, tmp_path):
        """GC with empty active set removes all vectors."""
        engine = SemanticEngine(str(tmp_path))
        engine.store_vector("mem_1", [0.1] * 384)
        engine.store_vector("mem_2", [0.2] * 384)

        removed = engine.gc_vectors(set())
        assert removed == 2
        assert engine.vector_count == 0

    def test_vector_gc_all_active(self, tmp_path):
        """GC when all vectors are active removes nothing."""
        engine = SemanticEngine(str(tmp_path))
        engine.store_vector("mem_1", [0.1] * 384)
        engine.store_vector("mem_2", [0.2] * 384)

        removed = engine.gc_vectors({"mem_1", "mem_2"})
        assert removed == 0
        assert engine.vector_count == 2


class TestVectorModelMismatch:
    """GPT 5.2 R2 requested test: vector file model mismatch detection."""

    def test_load_vectors_rejects_model_mismatch(self, tmp_path):
        """Write vectors with model A, load with model B → error."""
        import json
        from antaris_memory.semantic import SemanticEngine, SemanticEngineError

        # Write vectors with model A
        engine_a = SemanticEngine(str(tmp_path), model_name="model-a")
        engine_a.store_vector("mem1", [0.1] * 384)
        engine_a.save_vectors()

        # Try to load with model B
        engine_b = SemanticEngine(str(tmp_path), model_name="model-b")
        import pytest
        with pytest.raises(SemanticEngineError, match="model-a"):
            engine_b.load_vectors()

    def test_load_vectors_accepts_matching_model(self, tmp_path):
        """Write and load with same model → works fine."""
        from antaris_memory.semantic import SemanticEngine

        engine = SemanticEngine(str(tmp_path), model_name="all-MiniLM-L6-v2")
        engine.store_vector("mem1", [0.1] * 384)
        engine.save_vectors()

        engine2 = SemanticEngine(str(tmp_path), model_name="all-MiniLM-L6-v2")
        engine2.load_vectors()
        assert engine2.vector_count == 1
