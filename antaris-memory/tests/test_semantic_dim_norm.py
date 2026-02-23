"""
Tests for A1 (Vector Dimensionality Validation) and B2 (Cached Vector Norms)
added in antaris-memory v3.9.0.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from antaris_memory.semantic import SemanticEngine, SemanticEngineError


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_engine(tmp_path) -> SemanticEngine:
    return SemanticEngine(str(tmp_path))


def vec(n: int, value: float = 1.0) -> list:
    """Return an n-dimensional vector with all components = value."""
    return [value] * n


# ── A1: Vector Dimensionality Validation ──────────────────────────────────────

class TestDimensionalityValidation:
    def test_first_store_sets_dim(self, tmp_path):
        engine = make_engine(tmp_path)
        assert engine._dim is None
        engine.store_vector("m1", vec(4))
        assert engine._dim == 4

    def test_consistent_dim_accepted(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("m1", vec(4))
        engine.store_vector("m2", vec(4))  # same dim — should not raise
        assert engine._dim == 4

    def test_wrong_dim_raises_value_error(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("m1", vec(4))
        with pytest.raises(ValueError, match="Expected 4-dim vector, got 3"):
            engine.store_vector("m2", vec(3))

    def test_save_includes_dim(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("m1", vec(4))
        engine.save_vectors()
        data = json.loads((tmp_path / ".vectors.json").read_text())
        assert data["dim"] == 4

    def test_save_omits_dim_when_none(self, tmp_path):
        """If no vectors stored, dim should not appear in saved file."""
        engine = make_engine(tmp_path)
        engine._dirty = True  # force save without any vectors
        engine.save_vectors()
        data = json.loads((tmp_path / ".vectors.json").read_text())
        assert "dim" not in data

    def test_load_restores_dim(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("m1", vec(8))
        engine.save_vectors()

        engine2 = make_engine(tmp_path)
        engine2.load_vectors()
        assert engine2._dim == 8

    def test_load_dim_mismatch_raises(self, tmp_path):
        """Loading a file whose dim differs from the engine's pre-set dim should raise."""
        engine = make_engine(tmp_path)
        engine.store_vector("m1", vec(8))
        engine.save_vectors()

        engine2 = make_engine(tmp_path)
        engine2._dim = 4  # pre-configure with wrong dim
        with pytest.raises(SemanticEngineError, match="dim=8"):
            engine2.load_vectors()

    def test_load_no_dim_field_still_works(self, tmp_path):
        """Old vector files without 'dim' key load without error."""
        old_data = {
            "model": "all-MiniLM-L6-v2",
            "vectors": {"m1": [0.5, 0.5]},
            "version": "3.4",
        }
        (tmp_path / ".vectors.json").write_text(json.dumps(old_data))
        engine = make_engine(tmp_path)
        engine.load_vectors()
        # Gem-4: _dim is now inferred from the first vector when not stored in file.
        # Old behavior was None (backward compat); new behavior is inference for
        # global dim validation to work on migration from pre-A1 files.
        assert engine._dim == 2, (
            "Gem-4: _dim should be inferred from first vector when not stored in file"
        )

    def test_dim_after_remove_still_enforced(self, tmp_path):
        """Removing the only vector does NOT reset _dim."""
        engine = make_engine(tmp_path)
        engine.store_vector("m1", vec(4))
        engine.remove_vector("m1")
        assert engine._dim == 4
        with pytest.raises(ValueError, match="Expected 4-dim"):
            engine.store_vector("m2", vec(3))


# ── B2: Cached Vector Norms ───────────────────────────────────────────────────

class TestCachedNorms:
    def test_store_caches_norm(self, tmp_path):
        engine = make_engine(tmp_path)
        v = [3.0, 4.0]  # norm = 5.0
        engine.store_vector("m1", v)
        assert "m1" in engine._norms
        assert abs(engine._norms["m1"] - 5.0) < 1e-9

    def test_remove_evicts_norm(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("m1", [1.0, 0.0])
        engine.remove_vector("m1")
        assert "m1" not in engine._norms

    def test_remove_nonexistent_doesnt_crash(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.remove_vector("ghost")  # should not raise

    def test_gc_evicts_orphan_norms(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("keep", [1.0, 0.0])
        engine.store_vector("orphan", [0.0, 1.0])
        removed = engine.gc_vectors({"keep"})
        assert removed == 1
        assert "orphan" not in engine._norms
        assert "keep" in engine._norms

    def test_load_precomputes_norms(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("m1", [3.0, 4.0])  # norm 5
        engine.store_vector("m2", [0.0, 2.0])  # norm 2
        engine.save_vectors()

        engine2 = make_engine(tmp_path)
        engine2.load_vectors()
        assert abs(engine2._norms["m1"] - 5.0) < 1e-9
        assert abs(engine2._norms["m2"] - 2.0) < 1e-9

    def test_search_uses_cached_norms(self, tmp_path):
        """Search results should be identical whether norms are cached or not."""
        engine = make_engine(tmp_path)
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        engine.store_vector("m_a", a)
        engine.store_vector("m_b", b)

        query = [1.0, 0.0, 0.0]
        results = engine.search(query, top_k=5)
        ids = [r[0] for r in results]
        assert ids[0] == "m_a"
        assert "m_b" not in ids or results[0][1] > results[1][1]

    def test_search_zero_query_returns_empty(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("m1", [1.0, 0.0])
        results = engine.search([0.0, 0.0])
        assert results == []

    def test_search_zero_stored_vector_skipped(self, tmp_path):
        """A stored zero vector should be skipped gracefully."""
        engine = make_engine(tmp_path)
        # Bypass store_vector to inject a zero vector (wouldn't happen normally
        # because embed_and_store skips zero vecs, but test defensively)
        engine._vectors["zero"] = [0.0, 0.0]
        engine._norms["zero"] = 0.0
        engine.store_vector("real", [1.0, 0.0])
        results = engine.search([1.0, 0.0])
        ids = [r[0] for r in results]
        assert "zero" not in ids
        assert "real" in ids

    def test_cosine_similarity_unchanged(self, tmp_path):
        """cosine_similarity() must still work for external callers (no change)."""
        engine = make_engine(tmp_path)
        sim = engine.cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-9
        sim2 = engine.cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim2) < 1e-9


# ── Integration: A1 + B2 together ─────────────────────────────────────────────

class TestDimNormIntegration:
    def test_roundtrip_save_load_preserves_search(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("apple", [1.0, 0.0, 0.0, 0.0])
        engine.store_vector("banana", [0.0, 1.0, 0.0, 0.0])
        engine.save_vectors()

        engine2 = make_engine(tmp_path)
        engine2.load_vectors()
        assert engine2._dim == 4
        results = engine2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0][0] == "apple"

    def test_dim_mismatch_error_on_new_store_after_load(self, tmp_path):
        engine = make_engine(tmp_path)
        engine.store_vector("m1", vec(4))
        engine.save_vectors()

        engine2 = make_engine(tmp_path)
        engine2.load_vectors()
        with pytest.raises(ValueError, match="Expected 4-dim"):
            engine2.store_vector("m2", vec(8))

    # ── GPT-B: search() dim mismatch safety (R2/R3 additions) ─────────────────

    def test_search_returns_empty_on_dim_mismatch(self, tmp_path):
        """GPT-B: search() must return [] when query vector length != stored dim.

        Previously, zip() silently truncated the dot product and divided by the
        full stored norm — producing plausible but meaningless scores.
        """
        engine = make_engine(tmp_path)
        engine.store_vector("a", [1.0, 0.0, 0.0, 0.0])
        engine.store_vector("b", [0.0, 1.0, 0.0, 0.0])
        assert engine._dim == 4

        # Query with wrong dimension — must return empty, not garbage scores
        result = engine.search([1.0, 0.0], top_k=5)  # 2-dim query, 4-dim stored
        assert result == [], f"Expected [] for dim-mismatch query, got {result}"

    def test_search_per_vector_guard_for_legacy_vectors(self, tmp_path):
        """GPT-B: Per-vector dim guard handles vectors stored before _dim lock-in.

        Simulates a file written without dim metadata — _dim starts None and the
        global check is skipped, but the per-vector guard still protects.
        """
        engine = make_engine(tmp_path)
        # Manually inject a wrong-dim vector to simulate legacy data
        engine._vectors["legacy"] = [1.0, 0.0]         # 2-dim
        engine._vectors["good"] = [1.0, 0.0, 0.0, 0.0]  # 4-dim
        engine._norms["legacy"] = math.sqrt(1.0)
        engine._norms["good"] = 1.0
        # _dim intentionally left as None to test the per-vector path

        result = engine.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        ids_returned = [r[0] for r in result]
        assert "legacy" not in ids_returned, "Per-vector guard must skip wrong-dim vectors"
        assert "good" in ids_returned

    def test_load_vectors_infers_dim_from_first_vector(self, tmp_path):
        """Gem-4: load_vectors() infers _dim from first vector when dim not in file.

        Older vector files don't have a 'dim' key. We now infer it so global
        dim validation is restored for migration scenarios.
        """
        import json
        # Write a legacy-format vector file without 'dim' key
        vectors_path = tmp_path / ".vectors.json"
        data = {
            "model": "all-MiniLM-L6-v2",
            "vectors": {"m1": [1.0, 0.0, 0.0], "m2": [0.0, 1.0, 0.0]},
            "version": "3.4",  # old format, no 'dim'
        }
        vectors_path.write_text(json.dumps(data))

        engine = make_engine(tmp_path)
        engine.load_vectors()
        assert engine._dim == 3, f"Expected inferred dim=3, got {engine._dim}"
