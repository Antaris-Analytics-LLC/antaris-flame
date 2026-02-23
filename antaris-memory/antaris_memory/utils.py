"""Shared utilities for antaris-memory."""


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two vectors. Pure stdlib, no numpy."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

import json
import os
import tempfile
import logging

logger = logging.getLogger("antaris_memory")


def atomic_write_json(path: str, data: dict, indent: int = 2, lock: bool = True) -> None:
    """Write JSON atomically with optional file locking.
    
    Prevents torn/partial writes from crashes or interrupted I/O.
    When lock=True (default), also prevents lost updates from concurrent writers
    using a directory-based lock.
    
    Args:
        path: File path to write
        data: JSON-serializable data
        indent: JSON indentation
        lock: If True, acquire a file lock before writing (default: True)
    """
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    
    if lock:
        from .locking import FileLock
        with FileLock(path, timeout=30.0):
            _do_atomic_write(path, data, indent, dir_path)
    else:
        _do_atomic_write(path, data, indent, dir_path)


def _do_atomic_write(path: str, data, indent: int, dir_path: str) -> None:
    """Internal: perform the actual atomic write."""
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        # Best-effort directory fsync for crash-consistent rename on POSIX
        try:
            dir_fd = os.open(dir_path, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass  # Windows or unsupported — rename is still atomic
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def locked_read_json(path: str, default=None):
    """Read a JSON file with a shared lock to prevent torn reads.
    
    Args:
        path: File path to read
        default: Value to return if file doesn't exist
        
    Returns:
        Parsed JSON data, or default if file doesn't exist.
    """
    if not os.path.exists(path):
        return default
    
    from .locking import FileLock
    with FileLock(path, timeout=10.0):
        with open(path) as f:
            return json.load(f)
