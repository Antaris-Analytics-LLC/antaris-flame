# tests/test_shared_multiprocess_stress.py
"""Multi-process contention stress test for SharedMemoryPool audit log.

Skipped by default. Run with: RUN_STRESS=1 python3 -m pytest tests/test_shared_multiprocess_stress.py -q
"""

import json
import os
import time
import pytest
import multiprocessing as mp

from antaris_memory.shared import SharedMemoryPool


def _worker(pool_dir: str, pool_name: str, proc_id: int):
    """Worker that hammers shared pool with writes + reads."""
    p = SharedMemoryPool(pool_dir, pool_name=pool_name)
    p.load()
    # Ensure permissions exist
    if "admin" not in p.permissions:
        p.register_agent("admin", role="admin")
    if "writer" not in p.permissions:
        p.register_agent("writer", role="write")

    for i in range(50):
        p.write("writer", f"[p{proc_id}] item {i} about deployment systemd restart",
                namespace="shared")
        _ = p.read("admin", "deployment systemd restart", limit=5)
        if i % 10 == 0:
            p.save()


@pytest.mark.skipif(
    os.getenv("RUN_STRESS") != "1",
    reason="Set RUN_STRESS=1 to run multiprocess stress test",
)
def test_shared_audit_is_not_corrupted_under_contention(tmp_path):
    """Stress audit log with 6 concurrent processes, verify JSON stays valid."""
    pool_dir = str(tmp_path)
    pool_name = "stress"
    audit_path = os.path.join(pool_dir, f"pool_{pool_name}_audit.json")

    # Initialize pool + agents in main process
    pool = SharedMemoryPool(pool_dir, pool_name=pool_name)
    pool.register_agent("admin", role="admin")
    pool.register_agent("writer", role="write")
    pool.save()

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(pool_dir, pool_name, k)) for k in range(6)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()

    time.sleep(0.1)
    assert os.path.exists(audit_path), "Audit file should exist after contention"

    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 50  # Should have non-trivial audit entries
