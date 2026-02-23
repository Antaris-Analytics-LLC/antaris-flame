"""
Tests for antaris_memory.audit module.

Covers audit logging, rotation, anonymization, thread safety, and querying.
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest import TestCase

from antaris_memory.audit import AuditLogger


class TestAuditLogger(TestCase):
    """Test the AuditLogger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.audit = AuditLogger(self.temp_dir, max_size_mb=0.001, max_files=3)  # Very small for testing rotation

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_logging(self):
        """Test basic logging functionality."""
        # Log some events
        self.audit.log("ingest", {
            "id": "mem123", 
            "source": "chat", 
            "category": "technical",
            "timestamp": time.time()
        })
        
        self.audit.log("search", {
            "query": "Python tips",
            "results_count": 5,
            "duration_ms": 23
        })
        
        self.audit.log("recall", {
            "memory_id": "mem123",
            "context_id": "session_456"
        })

        # Check that file was created
        audit_file = Path(self.temp_dir) / ".audit.jsonl"
        self.assertTrue(audit_file.exists())

        # Check file contents
        with open(audit_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 3)
        
        # Parse first entry
        entry1 = json.loads(lines[0].strip())
        self.assertEqual(entry1["event"], "ingest")
        self.assertIn("ts", entry1)
        self.assertEqual(entry1["data"]["id"], "mem123")

    def test_query_functionality(self):
        """Test querying audit events."""
        start_time = time.time()
        
        # Log events
        self.audit.log("ingest", {"id": "mem1"})
        time.sleep(0.01)  # Small delay to ensure timestamp difference
        self.audit.log("search", {"query": "test"})
        time.sleep(0.01)
        self.audit.log("recall", {"memory_id": "mem1"})
        mid_time = time.time()
        time.sleep(0.01)
        self.audit.log("ingest", {"id": "mem2"})

        # Query all events
        all_events = self.audit.query()
        self.assertEqual(len(all_events), 4)
        
        # Query by event type
        ingest_events = self.audit.query(event_type="ingest")
        self.assertEqual(len(ingest_events), 2)
        self.assertEqual(ingest_events[0]["event"], "ingest")  # Newest first
        
        # Query by timestamp
        recent_events = self.audit.query(since=mid_time)
        self.assertEqual(len(recent_events), 1)
        self.assertEqual(recent_events[0]["data"]["id"], "mem2")
        
        # Query with limit
        limited_events = self.audit.query(limit=2)
        self.assertEqual(len(limited_events), 2)

    def test_log_rotation(self):
        """Test log rotation when file exceeds size limit."""
        audit_file = Path(self.temp_dir) / ".audit.jsonl"
        
        # Write many entries to exceed size limit
        large_data = "x" * 200  # Large data to quickly exceed 0.001 MB limit
        for i in range(20):
            self.audit.log("test", {"data": large_data, "iteration": i})

        # Check that rotation occurred
        rotated_file = Path(self.temp_dir) / ".audit.1.jsonl"
        self.assertTrue(rotated_file.exists())
        self.assertTrue(audit_file.exists())
        
        # Current file should have recent entries
        current_entries = self.audit.query()
        self.assertGreater(len(current_entries), 0)
        
        # Check that we can still read from current file
        recent_entry = current_entries[0]
        self.assertEqual(recent_entry["event"], "test")

    def test_rotation_manual(self):
        """Test manual log rotation."""
        # Add some entries
        for i in range(5):
            self.audit.log("test", {"iteration": i})
        
        # Manual rotation
        self.audit.rotate()
        
        # Check files exist
        audit_file = Path(self.temp_dir) / ".audit.jsonl"
        rotated_file = Path(self.temp_dir) / ".audit.1.jsonl"
        
        self.assertTrue(rotated_file.exists())
        self.assertFalse(audit_file.exists())  # New file not created until next log

    def test_anonymization(self):
        """Test PII anonymization functionality."""
        # Create entry with PII
        original_entry = {
            "ts": time.time(),
            "event": "search",
            "data": {
                "query": "Find emails from john@example.com",
                "content": "Personal information here",
                "source": "private chat log",
                "text": "Sensitive user data",
                "safe_field": "This should not be hashed",
                "nested": {
                    "content": "Nested sensitive content",
                    "other": "Non-sensitive nested data"
                }
            }
        }
        
        # Anonymize
        anonymized = self.audit.anonymize_entry(original_entry)
        
        # Check that PII fields are hashed
        self.assertTrue(anonymized["data"]["query"].startswith("hmac:"))
        self.assertTrue(anonymized["data"]["content"].startswith("hmac:"))
        self.assertTrue(anonymized["data"]["source"].startswith("hmac:"))
        self.assertTrue(anonymized["data"]["text"].startswith("hmac:"))
        self.assertTrue(anonymized["data"]["nested"]["content"].startswith("hmac:"))
        
        # Check that non-PII fields are preserved
        self.assertEqual(anonymized["data"]["safe_field"], "This should not be hashed")
        self.assertEqual(anonymized["data"]["nested"]["other"], "Non-sensitive nested data")
        self.assertEqual(anonymized["event"], "search")
        
        # Original should be unchanged
        self.assertEqual(original_entry["data"]["query"], "Find emails from john@example.com")

    def test_stats(self):
        """Test statistics functionality."""
        # Test empty stats
        empty_stats = self.audit.stats()
        self.assertEqual(empty_stats["total_entries"], 0)
        self.assertEqual(empty_stats["file_size_bytes"], 0)
        self.assertEqual(empty_stats["event_counts"], {})
        
        # Log some events
        self.audit.log("ingest", {"id": "mem1"})
        self.audit.log("ingest", {"id": "mem2"})
        self.audit.log("search", {"query": "test"})
        self.audit.log("recall", {"memory_id": "mem1"})
        
        # Get stats
        stats = self.audit.stats()
        
        self.assertEqual(stats["total_entries"], 4)
        self.assertGreater(stats["file_size_bytes"], 0)
        self.assertEqual(stats["event_counts"]["ingest"], 2)
        self.assertEqual(stats["event_counts"]["search"], 1)
        self.assertEqual(stats["event_counts"]["recall"], 1)
        self.assertIsNotNone(stats["oldest_entry_ts"])
        self.assertIsNotNone(stats["newest_entry_ts"])

    def test_thread_safety(self):
        """Test concurrent writes from multiple threads."""
        # Create a separate audit logger with larger size limit to avoid rotation during test
        thread_audit = AuditLogger(self.temp_dir, max_size_mb=10.0, max_files=3)
        
        num_threads = 5
        entries_per_thread = 10
        results = {}
        
        def write_entries(thread_id):
            """Write entries from a thread."""
            try:
                for i in range(entries_per_thread):
                    thread_audit.log("thread_test", {
                        "thread_id": thread_id,
                        "iteration": i
                    })
                results[thread_id] = "success"
            except Exception as e:
                results[thread_id] = f"error: {e}"
        
        # Start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=write_entries, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all threads succeeded
        for thread_id, result in results.items():
            self.assertEqual(result, "success", f"Thread {thread_id} failed: {result}")
        
        # Check that all entries were written
        all_events = thread_audit.query(event_type="thread_test")
        self.assertEqual(len(all_events), num_threads * entries_per_thread)
        
        # Check that we have entries from all threads
        thread_ids = set()
        for event in all_events:
            thread_ids.add(event["data"]["thread_id"])
        self.assertEqual(len(thread_ids), num_threads)

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON lines in log file."""
        audit_file = Path(self.temp_dir) / ".audit.jsonl"
        
        # Write some valid entries
        self.audit.log("valid", {"data": "good"})
        
        # Manually add invalid JSON line
        with open(audit_file, 'a', encoding='utf-8') as f:
            f.write("invalid json line\n")
            f.write('{"incomplete": json\n')
        
        # Add another valid entry
        self.audit.log("valid", {"data": "good2"})
        
        # Query should skip invalid lines and return valid ones
        events = self.audit.query()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["data"]["data"], "good2")  # Newest first
        self.assertEqual(events[1]["data"]["data"], "good")

    def test_max_files_rotation(self):
        """Test that rotation respects max_files limit."""
        # Create audit logger with max_files=2 for easier testing
        audit = AuditLogger(self.temp_dir, max_size_mb=0.001, max_files=2)
        
        large_data = "x" * 100
        
        # Write enough to cause multiple rotations
        for i in range(50):
            audit.log("rotation_test", {"data": large_data, "iteration": i})
        
        # Check that we have at most max_files + 1 files (current + max_files rotated)
        audit_files = list(Path(self.temp_dir).glob(".audit*.jsonl"))
        self.assertLessEqual(len(audit_files), 3)  # .audit.jsonl + .audit.1.jsonl + .audit.2.jsonl
        
        # Check that .audit.3.jsonl doesn't exist (should have been deleted)
        self.assertFalse((Path(self.temp_dir) / ".audit.3.jsonl").exists())

    def test_empty_workspace_creation(self):
        """Test that audit logger creates workspace directory if it doesn't exist."""
        nonexistent_dir = os.path.join(self.temp_dir, "new", "nested", "dir")
        audit = AuditLogger(nonexistent_dir)
        
        audit.log("test", {"data": "value"})
        
        # Check that directory was created and file exists
        audit_file = Path(nonexistent_dir) / ".audit.jsonl"
        self.assertTrue(audit_file.exists())

    def test_case_insensitive_pii_fields(self):
        """Test that PII field matching is case-insensitive."""
        entry = {
            "ts": time.time(),
            "event": "test",
            "data": {
                "Query": "Uppercase query field",
                "CONTENT": "Uppercase content field",
                "Text": "Mixed case text field",
                "normal_field": "Should not be hashed"
            }
        }
        
        anonymized = self.audit.anonymize_entry(entry)
        
        # Case-insensitive matches should be hashed
        self.assertTrue(anonymized["data"]["Query"].startswith("hmac:"))
        self.assertTrue(anonymized["data"]["CONTENT"].startswith("hmac:"))
        self.assertTrue(anonymized["data"]["Text"].startswith("hmac:"))
        
        # Non-PII field should be unchanged
        self.assertEqual(anonymized["data"]["normal_field"], "Should not be hashed")


if __name__ == "__main__":
    import unittest
    unittest.main()

class TestAuditQueryLatestN:
    """GPT 5.2 R2 requested test: verify query returns latest N, not oldest N."""

    def test_query_returns_latest_not_oldest(self, tmp_path):
        """Append 200 events, query limit=50, assert we get the last 50."""
        from antaris_memory.audit import AuditLogger
        import time

        audit = AuditLogger(str(tmp_path))
        for i in range(200):
            audit.log("test", {"seq": i})

        results = audit.query(limit=50)
        # Results should be newest-first
        assert len(results) == 50
        # The newest entry (seq=199) should be first
        assert results[0]["data"]["seq"] == 199
        # The 50th entry (seq=150) should be last
        assert results[49]["data"]["seq"] == 150
