"""
Test vocab cap + LRU eviction for CooccurrenceIndex.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

from antaris_memory.cooccurrence import CooccurrenceIndex


class TestVocabCap:
    """Test vocabulary capacity and LRU eviction functionality."""

    def test_vocab_cap_parameter(self):
        """Test that vocab_cap parameter is properly set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CooccurrenceIndex(tmpdir, vocab_cap=1000)
            assert idx.vocab_cap == 1000

    def test_no_eviction_under_cap(self):
        """Test that no eviction occurs when vocab size is under cap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CooccurrenceIndex(tmpdir, vocab_cap=100)
            
            # Add some memories, staying well under cap
            idx.update_from_memory("this is a test")
            idx.update_from_memory("another test memory")
            
            # Should not trigger eviction
            vocab_before = idx.vocab_size
            evicted = idx._evict_lru()
            
            assert evicted == 0
            assert idx.vocab_size == vocab_before

    def test_eviction_triggers_at_cap(self):
        """Test that eviction is triggered when vocab exceeds cap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very small cap for testing
            idx = CooccurrenceIndex(tmpdir, vocab_cap=5)
            
            # Add enough content to exceed cap
            words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
            for i, word in enumerate(words):
                # Add slight delay to ensure different access times
                time.sleep(0.01)
                idx.update_from_memory(f"{word} is a fruit number {i}")
                
            # Should have triggered eviction
            assert idx.vocab_size <= idx.vocab_cap
            
            # Should still have some words
            assert idx.vocab_size > 0

    def test_lru_ordering(self):
        """Test that least recently used words are evicted first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use larger cap so words don't get evicted immediately
            idx = CooccurrenceIndex(tmpdir, vocab_cap=6)
            
            # Add words with specific access pattern
            time.sleep(0.01)
            idx.update_from_memory("old unique")  # Oldest
            
            time.sleep(0.01)
            idx.update_from_memory("middle unique")  # Middle
            
            time.sleep(0.01)
            idx.update_from_memory("new unique")  # Newest
            
            # Should have 6 words now: old, unique (2x), middle, new
            # All should still be present before eviction
            assert "old" in idx._counts
            assert "middle" in idx._counts
            assert "new" in idx._counts
            
            # Access the "old" word to make it recent
            time.sleep(0.01)
            idx.similar_words("old")
            
            # Add more words to trigger eviction beyond the cap
            time.sleep(0.01)
            idx.update_from_memory("trigger more eviction test")
            
            # After eviction, vocab should be at or below cap
            assert idx.vocab_size <= idx.vocab_cap
            
            # "old" should still be present because it was recently accessed
            # "middle" is more likely to be evicted as it's older and unaccessed
            if "old" not in idx._counts and "middle" in idx._counts:
                # If old was evicted but middle wasn't, something's wrong with LRU
                assert False, "LRU ordering failed: old was evicted but middle wasn't"

    def test_access_time_tracking_on_query(self):
        """Test that access times are updated when words are queried."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CooccurrenceIndex(tmpdir)
            
            # Add some content
            idx.update_from_memory("test word for access tracking")
            
            # Get initial access time
            initial_time = idx._access_times.get("test", 0)
            
            # Wait and query
            time.sleep(0.01)
            idx.similar_words("test")
            
            # Access time should be updated
            updated_time = idx._access_times.get("test", 0)
            assert updated_time > initial_time

    def test_access_time_tracking_on_boost_score(self):
        """Test that access times are updated when boost_score is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CooccurrenceIndex(tmpdir)
            
            # Add some content
            idx.update_from_memory("test word for boost tracking")
            
            # Get initial access time
            initial_time = idx._access_times.get("test", 0)
            
            # Wait and call boost_score
            time.sleep(0.01)
            idx.boost_score("test", "some content with test")
            
            # Access time should be updated
            updated_time = idx._access_times.get("test", 0)
            assert updated_time > initial_time

    def test_persistence_of_access_times(self):
        """Test that access times are saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index and add content
            idx1 = CooccurrenceIndex(tmpdir)
            idx1.update_from_memory("persistent test word")
            
            # Query to update access time
            time.sleep(0.01)
            idx1.similar_words("persistent")
            access_time = idx1._access_times.get("persistent")
            
            # Save
            idx1.save()
            
            # Create new index and load
            idx2 = CooccurrenceIndex(tmpdir)
            idx2.load()
            
            # Access time should be preserved
            assert idx2._access_times.get("persistent") == access_time

    def test_eviction_removes_co_occurrences(self):
        """Test that evicted words are removed from co-occurrence data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CooccurrenceIndex(tmpdir, vocab_cap=3)
            
            # Add content that creates co-occurrences
            idx.update_from_memory("apple banana")
            idx.update_from_memory("banana cherry")
            
            # Force eviction by adding more words
            idx.update_from_memory("date elderberry fig grape")
            
            # Check that evicted words are completely removed
            for word in idx._counts:
                # No evicted word should appear in co-occurrence data
                for co_word in idx._counts[word]:
                    assert co_word in idx._word_counts
                    assert co_word in idx._counts

    def test_eviction_count_calculation(self):
        """Test that eviction removes approximately 10% of vocabulary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = CooccurrenceIndex(tmpdir, vocab_cap=10)
            
            # Fill vocab to exactly the cap
            words = [f"word{i}" for i in range(15)]  # More than cap
            for word in words:
                time.sleep(0.001)  # Ensure different access times
                idx.update_from_memory(f"{word} unique content")
            
            # Should have evicted automatically to stay at or below cap
            assert idx.vocab_size <= idx.vocab_cap
            
            # Test manual eviction on an over-capacity vocab
            # First, fill beyond capacity manually
            for i in range(5):  # Add more words
                word = f"extra{i}"
                idx._word_counts[word] += 1
                idx._access_times[word] = time.time() - i  # Different access times
                idx._counts[word] = {}  # Add empty co-occurrence dict
            
            # Now vocab should be over capacity
            pre_evict_size = idx.vocab_size
            assert pre_evict_size > idx.vocab_cap
            
            # Manual eviction should remove about 10% of current vocab
            evicted = idx._evict_lru()
            expected_evict = max(1, int(pre_evict_size * 0.1))
            assert evicted >= expected_evict