"""Tests for the file locking system."""

import os
import json
import time
import tempfile
import threading
import unittest

from antaris_memory.locking import FileLock, LockTimeout


class TestFileLock(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.resource = os.path.join(self.tmpdir, "test_resource.json")
        # Create the resource file
        with open(self.resource, "w") as f:
            json.dump({"value": 0}, f)
    
    def tearDown(self):
        # Clean up any leftover locks
        lock_dir = self.resource + ".lock"
        if os.path.exists(lock_dir):
            meta = os.path.join(lock_dir, "holder.json")
            if os.path.exists(meta):
                os.unlink(meta)
            os.rmdir(lock_dir)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_basic_acquire_release(self):
        lock = FileLock(self.resource)
        self.assertTrue(lock.acquire())
        self.assertTrue(lock._held)
        self.assertTrue(os.path.isdir(self.resource + ".lock"))
        lock.release()
        self.assertFalse(lock._held)
        self.assertFalse(os.path.isdir(self.resource + ".lock"))
    
    def test_context_manager(self):
        with FileLock(self.resource) as lock:
            self.assertTrue(lock._held)
            self.assertTrue(os.path.isdir(self.resource + ".lock"))
        self.assertFalse(os.path.isdir(self.resource + ".lock"))
    
    def test_non_blocking_fails_when_held(self):
        lock1 = FileLock(self.resource)
        lock1.acquire()
        try:
            lock2 = FileLock(self.resource)
            self.assertFalse(lock2.acquire(blocking=False))
        finally:
            lock1.release()
    
    def test_timeout_raises(self):
        lock1 = FileLock(self.resource)
        lock1.acquire()
        try:
            lock2 = FileLock(self.resource, timeout=0.2)
            with self.assertRaises(LockTimeout):
                lock2.acquire()
        finally:
            lock1.release()
    
    def test_holder_metadata(self):
        lock = FileLock(self.resource)
        lock.acquire()
        try:
            meta_path = os.path.join(self.resource + ".lock", "holder.json")
            self.assertTrue(os.path.exists(meta_path))
            with open(meta_path) as f:
                meta = json.load(f)
            self.assertEqual(meta["pid"], os.getpid())
            self.assertIn("acquired_at", meta)
        finally:
            lock.release()
    
    def test_reentrant_fails(self):
        """Same process can't acquire the same lock twice."""
        lock = FileLock(self.resource)
        lock.acquire()
        try:
            lock2 = FileLock(self.resource, timeout=0.2)
            # Should timeout because we already hold it
            with self.assertRaises(LockTimeout):
                lock2.acquire()
        finally:
            lock.release()
    
    def test_stale_lock_broken(self):
        """Stale locks from crashed processes are broken."""
        # Simulate a stale lock
        lock_dir = self.resource + ".lock"
        os.mkdir(lock_dir)
        meta_path = os.path.join(lock_dir, "holder.json")
        with open(meta_path, "w") as f:
            json.dump({
                "pid": 99999999,  # Non-existent PID
                "acquired_at": time.time() - 600,  # 10 minutes ago
            }, f)
        
        # Should break the stale lock and acquire
        lock = FileLock(self.resource, stale_threshold=60)
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()
    
    def test_orphaned_lock_broken_by_pid(self):
        """Lock from a dead process is broken even if not timed out."""
        lock_dir = self.resource + ".lock"
        os.mkdir(lock_dir)
        meta_path = os.path.join(lock_dir, "holder.json")
        with open(meta_path, "w") as f:
            json.dump({
                "pid": 99999999,  # Non-existent PID
                "acquired_at": time.time(),  # Recent
            }, f)
        
        lock = FileLock(self.resource)
        self.assertTrue(lock.acquire(blocking=False))
        lock.release()
    
    def test_concurrent_writers_no_corruption(self):
        """Multiple threads writing through locks don't corrupt data."""
        counter_path = os.path.join(self.tmpdir, "counter.json")
        with open(counter_path, "w") as f:
            json.dump({"count": 0}, f)
        
        errors = []
        iterations = 50
        
        def increment():
            for _ in range(iterations):
                try:
                    lock = FileLock(counter_path, timeout=10.0)
                    with lock:
                        with open(counter_path) as f:
                            data = json.load(f)
                        data["count"] += 1
                        with open(counter_path, "w") as f:
                            json.dump(data, f)
                except Exception as e:
                    errors.append(str(e))
        
        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        
        with open(counter_path) as f:
            final = json.load(f)
        
        self.assertEqual(errors, [])
        self.assertEqual(final["count"], 4 * iterations,
                        f"Expected {4 * iterations}, got {final['count']} — lost updates detected")
    
    def test_atomic_write_with_lock(self):
        """atomic_write_json uses locking by default."""
        from antaris_memory.utils import atomic_write_json
        
        path = os.path.join(self.tmpdir, "locked_write.json")
        atomic_write_json(path, {"key": "value"})
        
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["key"], "value")
        # Lock should be released
        self.assertFalse(os.path.isdir(path + ".lock"))
    
    def test_atomic_write_without_lock(self):
        """atomic_write_json can skip locking."""
        from antaris_memory.utils import atomic_write_json
        
        path = os.path.join(self.tmpdir, "unlocked_write.json")
        atomic_write_json(path, {"key": "value"}, lock=False)
        
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["key"], "value")


if __name__ == "__main__":
    unittest.main()
