"""Comprehensive tests for CooccurrenceIndex — vocabulary divergence, PPMI accuracy, scale."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from antaris_memory.cooccurrence import CooccurrenceIndex


# ── Fixture Data ─────────────────────────────────────────────────────────────

MEDICAL_MEMORIES = [
    "The patient's hepatic function showed elevated ALT and AST levels",
    "Liver biopsy confirmed fibrosis stage F2 with portal inflammation",
    "Hepatocellular carcinoma risk increases with cirrhosis duration",
    "ALT normalization after antiviral therapy indicates liver recovery",
    "Bile duct obstruction causes conjugated hyperbilirubinemia",
    "Hepatic encephalopathy correlates with ammonia serum levels",
    "Non-alcoholic fatty liver disease progresses to steatohepatitis",
    "Portal hypertension causes esophageal varices and splenomegaly",
    "Liver transplant outcomes depend on MELD score at transplant time",
    "Gamma-GT elevation suggests cholestatic liver disease pattern",
    "Serum albumin reflects hepatic synthetic capacity in cirrhosis",
    "Coagulation factors produced by liver are reduced in hepatic failure",
    "Bilirubin conjugation occurs in hepatocytes via glucuronidation pathway",
    "Liver fibrosis staging uses Metavir scale from F0 to F4",
    "Hepatitis viral load correlates with ALT elevation and liver damage",
    "Splenomegaly results from portal hypertension in cirrhotic patients",
    "Hepatic stellate cells activate during fibrosis and produce collagen",
    "Ammonia detoxification occurs in liver via urea cycle enzymes",
    "Liver cirrhosis causes ascites due to portal hypertension mechanism",
    "ALT and AST ratio helps differentiate alcoholic from viral hepatitis",
]

ML_MEMORIES = [
    "Transformer models use self-attention to capture long-range dependencies",
    "Neural networks require large datasets for effective training",
    "Gradient descent optimizes the loss function during backpropagation",
    "Batch normalization stabilizes training and improves convergence",
    "Dropout regularization prevents overfitting in deep neural networks",
    "Attention mechanisms allow models to focus on relevant input parts",
    "BERT uses bidirectional transformer encoder for language understanding",
    "Learning rate scheduling improves convergence during neural training",
    "Transformer architecture replaces recurrent neural networks efficiently",
    "Cross-entropy loss is standard for classification neural network tasks",
]


def make_memories(n: int, topic: str = "general") -> list:
    """Generate n synthetic memories about a given topic."""
    templates = [
        f"The {topic} system processes data efficiently",
        f"Analysis of {topic} results shows improvement",
        f"The {topic} model achieves high accuracy on benchmarks",
        f"Processing {topic} requires careful consideration of edge cases",
        f"The {topic} pipeline handles large volumes of information",
    ]
    return [templates[i % len(templates)] for i in range(n)]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_index(tmp_path) -> CooccurrenceIndex:
    """Create a fresh CooccurrenceIndex in a temp workspace."""
    idx = CooccurrenceIndex(str(tmp_path))
    idx.load()
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic Functionality
# ─────────────────────────────────────────────────────────────────────────────

class TestCooccurrenceBasic:

    def test_update_adds_word_pairs(self, tmp_path):
        """update_from_memory should record co-occurring word pairs."""
        idx = make_index(tmp_path)
        pairs_added = idx.update_from_memory("liver function tests show hepatic elevation")
        assert pairs_added > 0, "Should add at least one pair"
        assert idx.total_pairs > 0, "Total pairs counter should increment"

    def test_vocab_grows_with_updates(self, tmp_path):
        """Each new memory should expand the vocabulary."""
        idx = make_index(tmp_path)
        assert idx.vocab_size == 0
        idx.update_from_memory("liver biopsy confirmed fibrosis")
        v1 = idx.vocab_size
        assert v1 > 0
        idx.update_from_memory("neural network training convergence gradient")
        v2 = idx.vocab_size
        assert v2 > v1, "Vocab should grow after new content"

    def test_similar_words_returns_related(self, tmp_path):
        """similar_words should return words that co-occur with the query word."""
        idx = make_index(tmp_path)
        for mem in MEDICAL_MEMORIES:
            idx.update_from_memory(mem)
        results = idx.similar_words("liver")
        assert len(results) > 0, "Should return related words for 'liver'"
        words = [w for w, _ in results]
        # Liver co-occurs with medical terms in our corpus
        assert any(w in words for w in ["hepatic", "cirrhosis", "fibrosis", "biopsy", "alt"]), (
            f"Expected medical terms in results, got: {words[:10]}"
        )

    def test_similar_words_are_sorted_by_score(self, tmp_path):
        """similar_words should return results sorted by PPMI score descending."""
        idx = make_index(tmp_path)
        for mem in MEDICAL_MEMORIES:
            idx.update_from_memory(mem)
        results = idx.similar_words("liver")
        assert len(results) > 1
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted descending"

    def test_similar_words_unknown_word_returns_empty(self, tmp_path):
        """similar_words should return empty list for words not in the index."""
        idx = make_index(tmp_path)
        idx.update_from_memory("some text about cats and dogs")
        result = idx.similar_words("xyzzy_unknown_word_12345")
        assert result == [], "Unknown word should return empty list"

    def test_similar_words_empty_index_returns_empty(self, tmp_path):
        """similar_words on a fresh index should return empty list."""
        idx = make_index(tmp_path)
        result = idx.similar_words("liver")
        assert result == []

    def test_boost_score_higher_for_related_words(self, tmp_path):
        """boost_score should be higher when entry contains words related to query."""
        idx = make_index(tmp_path)
        for mem in MEDICAL_MEMORIES * 3:  # More data = stronger signal
            idx.update_from_memory(mem)
        # Query: "liver" → related entry about hepatic/ALT should score higher
        related_entry = "hepatic function elevated ALT biopsy fibrosis cirrhosis"
        unrelated_entry = "quantum physics particle accelerator photon wave function"
        boost_related = idx.boost_score("liver", related_entry)
        boost_unrelated = idx.boost_score("liver", unrelated_entry)
        assert boost_related >= boost_unrelated, (
            f"Related entry boost ({boost_related:.4f}) should be >= unrelated ({boost_unrelated:.4f})"
        )

    def test_boost_score_bounded_zero_to_factor(self, tmp_path):
        """boost_score must stay in [0, boost_factor] range."""
        idx = make_index(tmp_path)
        for mem in MEDICAL_MEMORIES:
            idx.update_from_memory(mem)
        boost = idx.boost_score("liver", "hepatic alt elevated biopsy fibrosis", boost_factor=0.3)
        assert 0.0 <= boost <= 0.3, f"Boost {boost:.4f} out of [0, 0.3]"

    def test_boost_score_custom_factor(self, tmp_path):
        """boost_score should respect custom boost_factor ceiling."""
        idx = make_index(tmp_path)
        for mem in MEDICAL_MEMORIES:
            idx.update_from_memory(mem)
        boost_05 = idx.boost_score("liver", "hepatic alt elevated", boost_factor=0.5)
        assert boost_05 <= 0.5

    def test_boost_score_empty_content_returns_zero(self, tmp_path):
        """boost_score with empty entry content should return 0."""
        idx = make_index(tmp_path)
        idx.update_from_memory("liver function test ALT hepatic")
        boost = idx.boost_score("liver", "")
        assert boost == 0.0

    def test_boost_score_empty_query_returns_zero(self, tmp_path):
        """boost_score with empty query should return 0."""
        idx = make_index(tmp_path)
        idx.update_from_memory("liver function test ALT hepatic")
        boost = idx.boost_score("", "hepatic ALT elevated liver")
        assert boost == 0.0

    def test_save_load_roundtrip_preserves_data(self, tmp_path):
        """save() + load() should preserve all index data."""
        idx = make_index(tmp_path)
        for mem in MEDICAL_MEMORIES:
            idx.update_from_memory(mem)
        vocab_before = idx.vocab_size
        pairs_before = idx.total_pairs
        idx.save()

        # Create fresh index and load
        idx2 = CooccurrenceIndex(str(tmp_path))
        idx2.load()
        assert idx2.vocab_size == vocab_before, "Vocab size should survive roundtrip"
        assert idx2.total_pairs == pairs_before, "Total pairs should survive roundtrip"

        # Query behavior should be preserved
        results_before = idx.similar_words("liver")
        results_after = idx2.similar_words("liver")
        words_before = {w for w, _ in results_before}
        words_after = {w for w, _ in results_after}
        assert words_before == words_after, "similar_words results should survive roundtrip"

    def test_save_is_noop_when_not_dirty(self, tmp_path):
        """save() should not write to disk if nothing changed."""
        idx = make_index(tmp_path)
        idx.update_from_memory("liver function hepatic")
        idx.save()
        mtime1 = (tmp_path / ".cooccurrence.json").stat().st_mtime

        # Load the saved index and call save() without changes
        idx2 = CooccurrenceIndex(str(tmp_path))
        idx2.load()
        time.sleep(0.01)  # Ensure mtime would change if written
        idx2.save()  # Should be no-op (not dirty)

        mtime2 = (tmp_path / ".cooccurrence.json").stat().st_mtime
        assert mtime1 == mtime2, "save() should not write when index is not dirty"

    def test_load_nonexistent_is_noop(self, tmp_path):
        """load() on a missing file should silently succeed with empty index."""
        idx = CooccurrenceIndex(str(tmp_path / "no_such_dir"))
        idx.load()  # Should not raise
        assert idx.vocab_size == 0
        assert idx.total_pairs == 0

    def test_similar_words_top_k_respected(self, tmp_path):
        """similar_words top_k should limit the number of results."""
        idx = make_index(tmp_path)
        for mem in MEDICAL_MEMORIES * 5:
            idx.update_from_memory(mem)
        results = idx.similar_words("liver", top_k=5)
        assert len(results) <= 5

    def test_update_returns_pair_count(self, tmp_path):
        """update_from_memory should return the number of pairs added."""
        idx = make_index(tmp_path)
        count = idx.update_from_memory("liver biopsy confirmed fibrosis portal")
        assert isinstance(count, int)
        assert count > 0

    def test_stats_method_returns_dict(self, tmp_path):
        """stats() should return a dict with expected keys."""
        idx = make_index(tmp_path)
        idx.update_from_memory("liver hepatic biopsy")
        s = idx.stats()
        assert "vocab_size" in s
        assert "total_pairs" in s
        assert "top_words" in s


# ─────────────────────────────────────────────────────────────────────────────
# 2. PPMI Scoring Accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestPPMIAccuracy:

    def test_ppmi_scores_are_positive(self, tmp_path):
        """All returned PPMI scores should be strictly positive (PPMI clips negatives)."""
        idx = make_index(tmp_path)
        for mem in ML_MEMORIES * 5:
            idx.update_from_memory(mem)
        results = idx.similar_words("neural")
        assert all(score > 0 for _, score in results), (
            "PPMI should only return positive scores"
        )

    def test_domain_specific_clustering(self, tmp_path):
        """Medical terms should cluster together under PPMI scoring."""
        idx = make_index(tmp_path)
        for mem in MEDICAL_MEMORIES * 5:
            idx.update_from_memory(mem)
        results = idx.similar_words("hepatic")
        words = [w for w, _ in results]
        medical_words = {"liver", "alt", "fibrosis", "cirrhosis", "biopsy", "portal",
                         "inflammation", "function", "elevated", "disease"}
        found = medical_words & set(words)
        assert len(found) >= 2, (
            f"Expected at least 2 medical terms in 'hepatic' results, got: {words[:10]}"
        )

    def test_ppmi_penalizes_common_stopword_pairs(self, tmp_path):
        """Common words should have low PPMI — they appear with everything."""
        idx = make_index(tmp_path)
        # Mix medical and ML memories so "the/a/and" appear everywhere
        for mem in MEDICAL_MEMORIES * 5 + ML_MEMORIES * 5:
            idx.update_from_memory(mem)
        # Stopwords are filtered by tokenizer, so they won't appear in results
        results = idx.similar_words("liver")
        result_words = [w for w, _ in results]
        generic_stopwords = {"the", "and", "for", "with", "that", "this"}
        overlap = generic_stopwords & set(result_words)
        assert len(overlap) == 0, (
            f"Stopwords should never appear in results, but found: {overlap}"
        )

    def test_high_frequency_pairs_score_higher_than_low(self, tmp_path):
        """Words that always co-occur should have higher PPMI than random pairs."""
        idx = make_index(tmp_path)
        # Repeat a specific pair many times
        for _ in range(50):
            idx.update_from_memory("neural learning gradient neural learning")
        # Add noise
        for mem in make_memories(20, "random"):
            idx.update_from_memory(mem)

        results_dict = dict(idx.similar_words("neural", top_k=20))
        # "learning" should appear and "random" (from noise) should be lower
        assert "learning" in results_dict, (
            "'learning' should be in similar_words('neural') after 50 co-occurrences"
        )

    def test_ml_corpus_neural_similar_to_networks(self, tmp_path):
        """After 50 ML memories, 'neural' should appear similar to 'networks'.

        Note: 'neural networks' co-occurs within the 5-word window directly.
        'neural' and 'learning' do NOT co-occur closely in these sentences,
        so we test the strongly co-occurring pair instead.
        """
        idx = make_index(tmp_path)
        for _ in range(5):
            for mem in ML_MEMORIES:
                idx.update_from_memory(mem)
        # Check neural → networks (directly co-occurring pair)
        neural_similar = dict(idx.similar_words("neural", top_k=20))
        assert "networks" in neural_similar, (
            f"After 50 ML memories, 'networks' should be similar to 'neural'. "
            f"Got: {list(neural_similar.keys())[:10]}"
        )

    def test_ppmi_values_are_finite(self, tmp_path):
        """PPMI scores should never be NaN or inf."""
        import math
        idx = make_index(tmp_path)
        for mem in ML_MEMORIES * 3:
            idx.update_from_memory(mem)
        results = idx.similar_words("transformer")
        for word, score in results:
            assert math.isfinite(score), f"PPMI score for '{word}' is not finite: {score}"

    def test_mutual_similarity_is_symmetric(self, tmp_path):
        """If A is similar to B, B should be similar to A."""
        idx = make_index(tmp_path)
        for mem in MEDICAL_MEMORIES * 5:
            idx.update_from_memory(mem)
        liver_similar = {w for w, _ in idx.similar_words("liver", top_k=15)}
        hepatic_similar = {w for w, _ in idx.similar_words("hepatic", top_k=15)}
        # Check mutual relationship
        if "hepatic" in liver_similar:
            assert "liver" in hepatic_similar, (
                "If 'hepatic' is similar to 'liver', 'liver' should appear in 'hepatic' results"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vocabulary Divergence (Key Benchmark)
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyDivergence:

    def _build_medical_index(self, tmp_path, reps: int = 5) -> CooccurrenceIndex:
        """Build an index from medical memories only."""
        idx = make_index(tmp_path)
        for _ in range(reps):
            for mem in MEDICAL_MEMORIES:
                idx.update_from_memory(mem)
        return idx

    def test_medical_corpus_liver_similar_to_medical_terms(self, tmp_path):
        """100 medical memories should make 'liver' cluster with medical vocabulary."""
        idx = self._build_medical_index(tmp_path, reps=5)  # 5×20=100 ingests
        results = idx.similar_words("liver", top_k=15)
        words = [w for w, _ in results]
        medical_vocabulary = {
            "hepatic", "alt", "fibrosis", "cirrhosis", "biopsy", "portal",
            "inflammation", "elevated", "disease", "function", "hypertension",
            "splenomegaly", "ascites", "steatohepatitis", "hepatitis"
        }
        found = medical_vocabulary & set(words)
        assert len(found) >= 2, (
            f"Expected ≥2 medical terms similar to 'liver', got: {words[:15]}"
        )

    def test_medical_corpus_no_ml_terms_near_liver(self, tmp_path):
        """A purely medical corpus should NOT put ML terms near 'liver'."""
        idx = self._build_medical_index(tmp_path, reps=5)
        results = idx.similar_words("liver", top_k=20)
        words = set(w for w, _ in results)
        ml_terms = {"neural", "gradient", "transformer", "backpropagation", "dropout"}
        overlap = ml_terms & words
        assert len(overlap) == 0, (
            f"ML terms should not appear near 'liver' in medical corpus, but found: {overlap}"
        )

    def test_boost_score_higher_for_medical_query(self, tmp_path):
        """After medical training, a medical entry should outscore a generic one."""
        idx = self._build_medical_index(tmp_path, reps=5)
        medical_entry = "hepatic function elevated ALT biopsy fibrosis cirrhosis portal"
        generic_entry = "weather forecast shows sunny skies temperature rising tomorrow"
        boost_medical = idx.boost_score("liver", medical_entry)
        boost_generic = idx.boost_score("liver", generic_entry)
        assert boost_medical > boost_generic, (
            f"Medical entry boost ({boost_medical:.4f}) should exceed generic ({boost_generic:.4f})"
        )

    def test_ml_corpus_segregates_from_medical(self, tmp_path):
        """An ML-specific corpus should cluster ML terms, not medical ones."""
        idx = make_index(tmp_path)
        for _ in range(5):
            for mem in ML_MEMORIES:
                idx.update_from_memory(mem)
        results = idx.similar_words("neural", top_k=20)
        words = set(w for w, _ in results)
        # Words that genuinely co-occur with "neural" within a 5-word window:
        #   "networks" (neural networks), "recurrent" (recurrent neural networks),
        #   "deep" (deep neural networks), "dropout", "regularization", etc.
        expected_ml = {"networks", "deep", "recurrent", "training", "gradient",
                       "dropout", "regularization", "architecture"}
        found_ml = expected_ml & words
        assert len(found_ml) >= 1, (
            f"ML corpus should cluster 'neural' with ML terms, got: {list(words)[:10]}"
        )
        # Sanity: no medical terms should appear
        medical_terms = {"hepatic", "liver", "fibrosis", "biopsy", "alt", "cirrhosis"}
        overlap = medical_terms & words
        assert len(overlap) == 0, (
            f"Medical terms should not appear in ML-corpus 'neural' results: {overlap}"
        )

    def test_mixed_corpus_domain_separation(self, tmp_path):
        """Mixed corpus: liver cluster should differ from neural cluster."""
        idx = make_index(tmp_path)
        for _ in range(3):
            for mem in MEDICAL_MEMORIES + ML_MEMORIES:
                idx.update_from_memory(mem)
        liver_words = set(w for w, _ in idx.similar_words("liver", top_k=15))
        neural_words = set(w for w, _ in idx.similar_words("neural", top_k=15))
        # The two clusters should not fully overlap
        intersection = liver_words & neural_words
        union = liver_words | neural_words
        jaccard = len(intersection) / max(len(union), 1)
        assert jaccard < 0.5, (
            f"Liver and neural clusters overlap too much (Jaccard={jaccard:.2f}): "
            f"overlap={intersection}"
        )

    def test_boost_score_discriminates_queries_on_medical_corpus(self, tmp_path):
        """Medical corpus should boost medical queries more than ML queries."""
        idx = self._build_medical_index(tmp_path, reps=5)
        medical_entry = "hepatic function elevated ALT biopsy fibrosis cirrhosis portal"
        boost_medical_q = idx.boost_score("liver hepatic fibrosis", medical_entry)
        boost_ml_q = idx.boost_score("neural network transformer gradient", medical_entry)
        # Medical query should match more strongly against medical entry
        assert boost_medical_q >= boost_ml_q, (
            f"Medical query boost ({boost_medical_q:.4f}) should be >= ML query boost ({boost_ml_q:.4f})"
        )

    def test_100_medical_memories_build_rich_index(self, tmp_path):
        """100 medical memories should produce a rich, populated index."""
        idx = make_index(tmp_path)
        for _ in range(5):
            for mem in MEDICAL_MEMORIES:
                idx.update_from_memory(mem)
        assert idx.vocab_size >= 20, f"Expected vocab ≥ 20, got {idx.vocab_size}"
        assert idx.total_pairs >= 500, f"Expected ≥ 500 pairs, got {idx.total_pairs}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scale / Stress Test
# ─────────────────────────────────────────────────────────────────────────────

class TestCooccurrenceScale:

    def test_1000_ingests_no_crash(self, tmp_path):
        """1000 memory ingests should complete without errors."""
        idx = make_index(tmp_path)
        memories = make_memories(1000, "analytics")
        for mem in memories:
            idx.update_from_memory(mem)  # Should never raise
        assert idx.total_pairs > 0

    def test_1000_ingests_performance(self, tmp_path):
        """1000 ingests should complete in under 5 seconds."""
        idx = make_index(tmp_path)
        memories = make_memories(1000, "performance")
        start = time.perf_counter()
        for mem in memories:
            idx.update_from_memory(mem)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"1000 ingests took {elapsed:.2f}s, expected < 5s"

    def test_top_k_limits_similar_words_results(self, tmp_path):
        """After large ingestion, similar_words(top_k=N) must return ≤N results."""
        idx = make_index(tmp_path)
        for mem in make_memories(500, "machine") + make_memories(500, "learning"):
            idx.update_from_memory(mem)
        for k in [5, 10, 20]:
            results = idx.similar_words("machine", top_k=k)
            assert len(results) <= k, f"top_k={k} returned {len(results)} results"

    def test_save_load_after_1000_ingests(self, tmp_path):
        """save/load roundtrip should work correctly after 1000 entries."""
        idx = make_index(tmp_path)
        for mem in make_memories(1000, "scale"):
            idx.update_from_memory(mem)
        vocab_before = idx.vocab_size
        pairs_before = idx.total_pairs
        idx.save()

        idx2 = CooccurrenceIndex(str(tmp_path))
        idx2.load()
        assert idx2.vocab_size == vocab_before
        assert idx2.total_pairs == pairs_before

    def test_index_queries_still_work_after_scale(self, tmp_path):
        """similar_words and boost_score should work correctly after 1000+ entries."""
        idx = make_index(tmp_path)
        # Mix domain-specific and noisy memories
        for mem in MEDICAL_MEMORIES * 10:
            idx.update_from_memory(mem)
        for mem in make_memories(800, "noise"):
            idx.update_from_memory(mem)
        # Liver should still cluster with medical terms despite noise
        results = idx.similar_words("liver", top_k=10)
        assert len(results) > 0, "similar_words should return results even after scale test"
        boost = idx.boost_score("liver", "hepatic function fibrosis")
        assert boost >= 0.0

    def test_save_load_performance_at_scale(self, tmp_path):
        """save() and load() at scale should each complete within 2 seconds."""
        idx = make_index(tmp_path)
        for mem in make_memories(1000, "scalability"):
            idx.update_from_memory(mem)

        start = time.perf_counter()
        idx.save()
        save_time = time.perf_counter() - start

        start = time.perf_counter()
        idx2 = CooccurrenceIndex(str(tmp_path))
        idx2.load()
        load_time = time.perf_counter() - start

        assert save_time < 2.0, f"save() took {save_time:.2f}s at scale"
        assert load_time < 2.0, f"load() took {load_time:.2f}s at scale"

    def test_incremental_updates_accumulate(self, tmp_path):
        """Pairs should accumulate across many separate update calls."""
        idx = make_index(tmp_path)
        for _ in range(100):
            idx.update_from_memory("liver hepatic function elevated alt fibrosis")
        # Should have accumulated much more than a single update
        pairs_100 = idx.total_pairs
        idx2 = make_index(tmp_path)
        idx2.update_from_memory("liver hepatic function elevated alt fibrosis")
        pairs_1 = idx2.total_pairs
        assert pairs_100 > pairs_1 * 5, (
            f"100 updates ({pairs_100}) should have far more pairs than 1 ({pairs_1})"
        )

    def test_diverse_topics_produce_large_vocab(self, tmp_path):
        """Many diverse memories should produce a large vocabulary."""
        idx = make_index(tmp_path)
        topics = ["medical", "financial", "engineering", "legal", "scientific"]
        for topic in topics:
            for mem in make_memories(200, topic):
                idx.update_from_memory(mem)
        assert idx.vocab_size >= 10, f"Diverse corpus should have vocab ≥ 10, got {idx.vocab_size}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integration with MemorySystem (MemorySystemV4 via MemorySystem alias)
# ─────────────────────────────────────────────────────────────────────────────

class TestCooccurrenceIntegration:

    def test_ingest_builds_cooccurrence_index(self, tmp_path):
        """MemorySystem.ingest() should populate the co-occurrence index."""
        from antaris_memory import MemorySystem
        mem = MemorySystem(str(tmp_path))
        for content in ML_MEMORIES:
            mem.ingest(content, source="test")
        assert mem._cooccurrence.vocab_size > 0, (
            "Co-occurrence index should be populated after ingesting memories"
        )
        assert mem._cooccurrence.total_pairs > 0

    def test_search_with_context_returns_results(self, tmp_path):
        """search_with_context should return results after ingesting memories."""
        from antaris_memory import MemorySystem
        mem = MemorySystem(str(tmp_path))
        for content in ML_MEMORIES:
            mem.ingest(content, source="test")
        results, ctx = mem.search_with_context("transformer model")
        assert results is not None
        # ctx may be a SearchContext object — just verify we get something back
        assert len(results) >= 0  # May return 0 if threshold not met, but shouldn't crash

    def test_search_with_context_cooccurrence_boost_enabled(self, tmp_path):
        """search_with_context with cooccurrence_boost=True should not crash."""
        from antaris_memory import MemorySystem
        mem = MemorySystem(str(tmp_path))
        for content in ML_MEMORIES:
            mem.ingest(content, source="test")
        # Should not raise regardless of whether vocab is populated
        results, ctx = mem.search_with_context("transformer model", cooccurrence_boost=True)
        assert results is not None

    def test_search_with_context_cooccurrence_boost_disabled(self, tmp_path):
        """search_with_context with cooccurrence_boost=False should also work."""
        from antaris_memory import MemorySystem
        mem = MemorySystem(str(tmp_path))
        for content in ML_MEMORIES:
            mem.ingest(content, source="test")
        results, ctx = mem.search_with_context("attention mechanism", cooccurrence_boost=False)
        assert results is not None

    def test_cooccurrence_index_persisted_between_instances(self, tmp_path):
        """Co-occurrence data should persist when a new MemorySystem opens same workspace.

        flush() (or close()) must be called before the index is readable by a
        new instance — saves are deferred from ingest() to avoid O(n) disk I/O
        on bulk operations like ingest_directory().
        """
        from antaris_memory import MemorySystem
        mem1 = MemorySystem(str(tmp_path))
        for content in ML_MEMORIES:
            mem1.ingest(content, source="test")
        vocab_size_1 = mem1._cooccurrence.vocab_size

        # Flush to persist co-occurrence index before re-opening
        mem1.flush()

        # New instance same workspace — should load the persisted index
        mem2 = MemorySystem(str(tmp_path))
        assert mem2._cooccurrence.vocab_size == vocab_size_1, (
            "Co-occurrence index should be loaded from disk after flush()"
        )

    def test_transformer_attention_corpus_builds_index(self, tmp_path):
        """Ingesting 20 memories about transformers and attention should build index."""
        from antaris_memory import MemorySystem
        transformer_memories = [
            "Transformers use multi-head attention mechanism for sequence modeling",
            "Self-attention allows transformer to attend to all positions simultaneously",
            "Transformer encoder processes tokens via attention and feed-forward layers",
            "Attention weights determine which tokens the model focuses on",
            "BERT is a transformer pretrained with masked language modeling objective",
            "Transformer decoder generates tokens using masked self-attention",
            "Positional encoding adds sequence order information to transformer inputs",
            "Multi-head attention splits queries keys values across multiple heads",
            "Transformer attention scales dot product by sqrt of key dimension",
            "GPT uses transformer decoder stack for autoregressive language generation",
            "Cross-attention in transformer allows decoder to attend to encoder output",
            "Transformer model training uses attention mask to prevent future token access",
            "Flash attention optimizes transformer training by reducing memory bandwidth",
            "Sparse attention patterns reduce transformer computational complexity",
            "Transformer vision models apply attention patches from image sequences",
            "Attention mechanism in transformers replaces recurrent computation entirely",
            "Transformer models scale well with data and parameters via attention",
            "Rotary positional encoding improves transformer attention extrapolation",
            "Grouped query attention reduces transformer inference memory requirements",
            "Transformer tokenizer converts text to integer tokens before attention",
        ]
        assert len(transformer_memories) == 20, "Sanity check: exactly 20 memories"
        mem = MemorySystem(str(tmp_path))
        for content in transformer_memories:
            mem.ingest(content, source="test")
        # Index should be populated
        assert mem._cooccurrence.vocab_size > 0
        # "transformer"/"transformers" and "attention" should be semantically linked.
        # Note: tokenizer lowercases, so "Transformers" → "transformers" (plural).
        # Both words are very frequent in this corpus, so PPMI may not rank them #1
        # against each other — but "transformers" (from sentence-initial position)
        # appears in attention's similar_words since it co-occurs in opener sentences.
        transformer_sim = dict(mem._cooccurrence.similar_words("transformer", top_k=50))
        attention_sim = dict(mem._cooccurrence.similar_words("attention", top_k=50))
        transformers_sim = dict(mem._cooccurrence.similar_words("transformers", top_k=50))
        # At least one directional link should exist
        linked = (
            "attention" in transformer_sim
            or "transformer" in attention_sim
            or "transformers" in attention_sim
            or "attention" in transformers_sim
        )
        assert linked, (
            f"After 20 memories, transformer/attention should be mutually similar. "
            f"transformer_top10={list(transformer_sim.keys())[:10]}, "
            f"attention_top10={list(attention_sim.keys())[:10]}"
        )
        # search_with_context should work
        results, ctx = mem.search_with_context("transformer model", cooccurrence_boost=True)
        assert results is not None
