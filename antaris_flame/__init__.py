"""
antaris-flame v1.0.1

Memory, safety, and context for Antaris Flame bots.
Tier: Flame -- up to 5,000 memories.

Packages:
    antaris_memory  -- BM25 + semantic search, audit logging, co-occurrence PPMI
    antaris_guard   -- Prompt safety screening and injection detection
    antaris_context -- Token budget management and context windowing

Install:
    pip install antaris-flame            # core
    pip install antaris-flame[semantic]  # + transformer embeddings
"""

__version__ = "1.0.5"

MEMORY_CAP = 5000


def create_memory(workspace: str = ".", **kwargs):
    """Create a MemorySystem with Flame tier cap (5,000 entries)."""
    from antaris_memory import MemorySystem
    return MemorySystem(workspace, max_entries=MEMORY_CAP, **kwargs)


__all__ = ["__version__", "MEMORY_CAP", "create_memory"]
