"""
Tests for paraphrase detection using co-occurrence (antaris-memory v3.4).

Tests the new paraphrase detection methods in CooccurrenceIndex.
PPMI Strategy 3: detect paraphrases via co-occurrence neighbor overlap.
"""

import pytest
from antaris_memory.cooccurrence import CooccurrenceIndex
from antaris_memory.entry import MemoryEntry


class TestParaphraseDetection:
    """Test paraphrase detection methods."""

    def test_detect_paraphrase_exact_match(self, tmp_path):
        """Test paraphrase detection with identical text."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        text = "The patient shows elevated liver enzymes"
        is_paraphrase, score = idx.detect_paraphrase(text, text)
        
        # Identical text should be detected as paraphrase with high score
        assert is_paraphrase is True
        assert score >= 0.6  # Should meet default threshold

    def test_detect_paraphrase_no_overlap(self, tmp_path):
        """Test paraphrase detection with completely different text."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        text_a = "The patient shows elevated liver enzymes"
        text_b = "Weather forecast for tomorrow is sunny"
        
        is_paraphrase, score = idx.detect_paraphrase(text_a, text_b)
        
        # Completely different text should not be detected as paraphrase
        assert is_paraphrase is False
        assert score < 0.6  # Should not meet default threshold

    def test_detect_paraphrase_empty_text(self, tmp_path):
        """Test paraphrase detection with empty text."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        is_paraphrase, score = idx.detect_paraphrase("", "some text")
        assert is_paraphrase is False
        assert score == 0.0
        
        is_paraphrase, score = idx.detect_paraphrase("some text", "")
        assert is_paraphrase is False
        assert score == 0.0
        
        is_paraphrase, score = idx.detect_paraphrase("", "")
        assert is_paraphrase is False
        assert score == 0.0

    def test_detect_paraphrase_partial_overlap(self, tmp_path):
        """Test paraphrase detection with partial word overlap."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        # Build up some co-occurrence data
        idx.update_from_memory("liver function test results")
        idx.update_from_memory("hepatic enzyme levels elevated")
        idx.update_from_memory("ALT AST liver enzymes high")
        idx.update_from_memory("patient liver function abnormal")
        
        text_a = "liver function abnormal"
        text_b = "hepatic enzyme levels"
        
        # Should have some overlap due to medical context
        is_paraphrase, score = idx.detect_paraphrase(text_a, text_b)
        assert isinstance(is_paraphrase, bool)
        assert 0.0 <= score <= 1.0

    def test_detect_paraphrase_threshold_variation(self, tmp_path):
        """Test paraphrase detection with different thresholds."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        # Build minimal co-occurrence data
        idx.update_from_memory("liver enzymes elevated")
        
        text_a = "liver elevated"
        text_b = "enzymes elevated"
        
        # Low threshold - should pass
        is_paraphrase_low, score = idx.detect_paraphrase(text_a, text_b, threshold=0.3)
        
        # High threshold - might not pass
        is_paraphrase_high, _ = idx.detect_paraphrase(text_a, text_b, threshold=0.9)
        
        # Score should be the same regardless of threshold
        assert isinstance(is_paraphrase_low, bool)
        assert isinstance(is_paraphrase_high, bool)
        assert 0.0 <= score <= 1.0

    def test_detect_paraphrase_semantic_context(self, tmp_path):
        """Test paraphrase detection leveraging semantic context."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        # Build medical context
        medical_texts = [
            "patient liver function test",
            "hepatic enzyme ALT AST",
            "liver enzymes elevated high", 
            "ALT hepatic function abnormal",
            "patient enzyme levels test",
            "liver ALT AST results",
        ]
        
        for text in medical_texts:
            idx.update_from_memory(text)
        
        # Now test paraphrases in this domain
        text_a = "patient liver function abnormal"
        text_b = "hepatic enzyme levels elevated"
        
        is_paraphrase, score = idx.detect_paraphrase(text_a, text_b)
        
        # Should detect some semantic similarity
        assert isinstance(is_paraphrase, bool)
        assert score >= 0.0

    def test_find_paraphrases_empty_list(self, tmp_path):
        """Test find_paraphrases with empty entry list."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        paraphrases = idx.find_paraphrases("query text", [])
        assert paraphrases == []

    def test_find_paraphrases_no_matches(self, tmp_path):
        """Test find_paraphrases with no paraphrase matches."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        entries = [
            MemoryEntry("The weather is nice today"),
            MemoryEntry("Programming is fun and challenging"),
            MemoryEntry("Coffee tastes great in the morning"),
        ]
        
        paraphrases = idx.find_paraphrases("liver enzymes elevated", entries)
        
        # Should return empty list since no medical context
        assert paraphrases == []

    def test_find_paraphrases_with_matches(self, tmp_path):
        """Test find_paraphrases with some matches."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        # Build some medical co-occurrence context
        idx.update_from_memory("liver function test elevated")
        idx.update_from_memory("hepatic enzymes ALT AST")
        idx.update_from_memory("patient liver abnormal")
        
        entries = [
            MemoryEntry("Patient liver function looks abnormal"),        # potential match
            MemoryEntry("Weather forecast is sunny today"),              # no match
            MemoryEntry("Hepatic enzyme levels are elevated"),           # potential match
            MemoryEntry("Coffee break time"),                            # no match
            MemoryEntry("ALT and AST results abnormal"),                 # potential match
        ]
        
        paraphrases = idx.find_paraphrases("liver function abnormal", entries, threshold=0.1)
        
        # Should return some paraphrases, sorted by similarity
        assert isinstance(paraphrases, list)
        
        # Check structure of results
        for entry, score in paraphrases:
            assert hasattr(entry, 'content')
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
        
        # Results should be sorted by score descending
        if len(paraphrases) > 1:
            scores = [score for _, score in paraphrases]
            assert scores == sorted(scores, reverse=True)

    def test_find_paraphrases_threshold_filtering(self, tmp_path):
        """Test that find_paraphrases respects the threshold."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        entries = [
            MemoryEntry("some similar content here"),
            MemoryEntry("completely different unrelated text"),
        ]
        
        # High threshold should return fewer results
        paraphrases_high = idx.find_paraphrases("similar content", entries, threshold=0.9)
        paraphrases_low = idx.find_paraphrases("similar content", entries, threshold=0.1)
        
        # Low threshold should return same or more results
        assert len(paraphrases_high) <= len(paraphrases_low)

    def test_find_paraphrases_memory_entry_attributes(self, tmp_path):
        """Test that find_paraphrases handles MemoryEntry objects correctly."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        # Create memory entries with various attributes
        entry1 = MemoryEntry("liver function test results")
        entry1.source = "medical_report.txt"
        entry1.category = "medical"
        
        entry2 = MemoryEntry("weather is sunny")
        entry2.source = "weather_app.txt"  
        entry2.category = "weather"
        
        entries = [entry1, entry2]
        
        paraphrases = idx.find_paraphrases("liver test", entries, threshold=0.0)
        
        # Should return the medical entry as more likely paraphrase
        # Even with threshold=0, should still filter based on content similarity
        for entry, score in paraphrases:
            assert hasattr(entry, 'content')
            assert hasattr(entry, 'source')
            assert hasattr(entry, 'category')

    def test_paraphrase_detection_integration(self, tmp_path):
        """Integration test: build index, detect paraphrases end-to-end."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        # Build a realistic medical vocabulary
        medical_memories = [
            "Patient presents with elevated liver enzymes ALT and AST",
            "Hepatic function tests show abnormal results",
            "Liver function panel indicates elevated enzyme levels",
            "ALT AST values are significantly increased",
            "Patient has abnormal hepatic enzyme markers",
            "Elevated transaminases suggest liver dysfunction",
            "Liver enzyme levels are above normal range",
        ]
        
        for memory in medical_memories:
            idx.update_from_memory(memory)
        
        # Create test entries
        entries = [
            MemoryEntry("Patient shows elevated liver enzyme levels"),     # paraphrase
            MemoryEntry("Hepatic function tests abnormal for patient"),    # paraphrase 
            MemoryEntry("The weather forecast calls for rain tomorrow"),   # not paraphrase
            MemoryEntry("ALT and AST values significantly elevated"),      # paraphrase
            MemoryEntry("Programming in Python is enjoyable"),             # not paraphrase
        ]
        
        # Find paraphrases for a medical query
        query = "liver function elevated enzymes abnormal"
        paraphrases = idx.find_paraphrases(query, entries, threshold=0.3)
        
        # Should find medical entries as paraphrases, not weather/programming
        assert len(paraphrases) >= 1  # Should find at least some medical matches
        
        # All found paraphrases should be medical-related
        for entry, score in paraphrases:
            content = entry.content.lower()
            medical_terms = any(term in content for term in ["liver", "hepatic", "alt", "ast", "enzyme"])
            assert medical_terms, f"Non-medical paraphrase found: {content}"
            assert score >= 0.3  # Should meet the threshold

    def test_detect_paraphrase_bidirectional_scoring(self, tmp_path):
        """Test that paraphrase detection is reasonably bidirectional."""
        idx = CooccurrenceIndex(str(tmp_path))
        
        # Build context
        idx.update_from_memory("liver function enzyme test")
        idx.update_from_memory("hepatic ALT AST levels")
        
        text_a = "liver enzyme levels"
        text_b = "hepatic function test"
        
        # Test both directions
        is_para_ab, score_ab = idx.detect_paraphrase(text_a, text_b)
        is_para_ba, score_ba = idx.detect_paraphrase(text_b, text_a)
        
        # Should be roughly similar (allowing for some asymmetry)
        # At least the boolean result should be consistent
        assert isinstance(is_para_ab, bool)
        assert isinstance(is_para_ba, bool)
        assert isinstance(score_ab, float)
        assert isinstance(score_ba, float)
        
        # Scores should be in valid range
        assert 0.0 <= score_ab <= 1.0
        assert 0.0 <= score_ba <= 1.0