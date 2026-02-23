"""Tests for optimistic conflict detection."""

import json
import os
import tempfile
import time
import unittest

from antaris_memory.versioning import VersionTracker, FileVersion, ConflictError


class TestVersionTracker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "data.json")
        with open(self.path, "w") as f:
            json.dump({"count": 0}, f)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_snapshot_captures_state(self):
        tracker = VersionTracker()
        version = tracker.snapshot(self.path)
        self.assertEqual(version.path, self.path)
        self.assertGreater(version.mtime, 0)
        self.assertGreater(version.size, 0)
    
    def test_check_passes_when_unchanged(self):
        tracker = VersionTracker()
        version = tracker.snapshot(self.path)
        # Should not raise
        tracker.check(version)
    
    def test_check_fails_when_modified(self):
        tracker = VersionTracker()
        version = tracker.snapshot(self.path)
        
        # Modify the file
        time.sleep(0.01)  # Ensure mtime changes
        with open(self.path, "w") as f:
            json.dump({"count": 1}, f)
        
        with self.assertRaises(ConflictError):
            tracker.check(version)
    
    def test_check_fails_when_deleted(self):
        tracker = VersionTracker()
        version = tracker.snapshot(self.path)
        os.unlink(self.path)
        
        with self.assertRaises(ConflictError):
            tracker.check(version)
    
    def test_content_hash_mode(self):
        tracker = VersionTracker(use_content_hash=True)
        version = tracker.snapshot(self.path)
        self.assertTrue(len(version.content_hash) > 0)
        tracker.check(version)  # Should pass
    
    def test_file_version_is_current(self):
        tracker = VersionTracker()
        version = tracker.snapshot(self.path)
        self.assertTrue(version.is_current())
        
        time.sleep(0.01)
        with open(self.path, "w") as f:
            json.dump({"count": 99}, f)
        
        self.assertFalse(version.is_current())
    
    def test_safe_update_succeeds(self):
        tracker = VersionTracker()
        result = tracker.safe_update(
            self.path,
            lambda data: {**data, "count": data["count"] + 1}
        )
        self.assertEqual(result["count"], 1)
        
        with open(self.path) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["count"], 1)
    
    def test_safe_update_retries_on_conflict(self):
        tracker = VersionTracker()
        call_count = [0]
        
        def modifier(data):
            call_count[0] += 1
            if call_count[0] == 1:
                # Simulate another process modifying the file during our read-modify
                time.sleep(0.01)
                with open(self.path, "w") as f:
                    json.dump({"count": 100}, f)
            return {**data, "count": data["count"] + 1}
        
        result = tracker.safe_update(self.path, modifier, max_retries=3)
        # Should have retried and succeeded on the second attempt
        self.assertEqual(call_count[0], 2)
        self.assertEqual(result["count"], 101)  # 100 + 1
    
    def test_conflict_error_message(self):
        err = ConflictError("/tmp/test.json", 1000.0, 2000.0)
        self.assertIn("test.json", str(err))
        self.assertIn("1000.0", str(err))


if __name__ == "__main__":
    unittest.main()
