"""
antaris-flame v1.0.0

Memory, safety, and context for Antaris Flame bots.
Tier: Flame — up to 5,000 memories.

Packages:
    antaris_memory  — BM25 + semantic search, audit logging, co-occurrence PPMI
    antaris_guard   — Prompt safety screening and injection detection
    antaris_context — Token budget management and context windowing

Install:
    pip install antaris-flame            # core
    pip install antaris-flame[semantic]  # + transformer embeddings
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
