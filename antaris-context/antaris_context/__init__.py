"""
Antaris Context - Zero-dependency context window optimization for AI agents.

A lightweight Python library for managing context windows, token budgets,
and message compression in AI agent applications.
"""

from .manager import ContextManager, MemoryClient
from .window import ContextWindow
from .compressor import MessageCompressor
from .strategy import (
    ContextStrategy,
    RecencyStrategy,
    RelevanceStrategy,
    HybridStrategy,
    BudgetStrategy
)
from .profiler import ContextProfiler
from .utils import atomic_write_json
from .importance import (
    CompressionResult,
    ImportanceWeightedCompressor,
    SemanticChunker,
    SemanticChunk,
)

__version__ = "1.0.0"
__author__ = "Antaris Analytics"
__license__ = "Apache 2.0"

__all__ = [
    "ContextManager",
    "MemoryClient",
    "ContextWindow",
    "MessageCompressor",
    "ContextStrategy",
    "RecencyStrategy",
    "RelevanceStrategy",
    "HybridStrategy",
    "BudgetStrategy",
    "ContextProfiler",
    "atomic_write_json",
    # Sprint 6
    "CompressionResult",
    "ImportanceWeightedCompressor",
    "SemanticChunker",
    "SemanticChunk",
]