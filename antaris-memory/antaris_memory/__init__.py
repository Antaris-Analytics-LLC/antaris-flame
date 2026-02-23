"""
Antaris Memory — File-based persistent memory for AI agents.

Store, search, decay, and consolidate agent memories using only the
Python standard library. Zero dependencies, deterministic operations,
transparent JSON storage.

Usage:
    from antaris_memory import MemorySystem

    mem = MemorySystem("./workspace")
    mem.load()
    mem.ingest("Key decision made", source="meeting", category="strategic")
    results = mem.search("decision")
    mem.save()
"""

__version__ = "1.0.0"

# Core
from antaris_memory.core_v4 import MemorySystemV4 as MemorySystem
from antaris_memory.entry import MemoryEntry
from antaris_memory.decay import DecayEngine
from antaris_memory.sentiment import SentimentTagger
from antaris_memory.temporal import TemporalEngine
from antaris_memory.confidence import ConfidenceEngine
from antaris_memory.compression import CompressionEngine
from antaris_memory.forgetting import ForgettingEngine
from antaris_memory.consolidation import ConsolidationEngine
from antaris_memory.gating import InputGate
from antaris_memory.synthesis import KnowledgeSynthesizer

# Multi-agent
from antaris_memory.shared import SharedMemoryPool, AgentPermission

# Storage
from antaris_memory.sharding import ShardManager, ShardKey
from antaris_memory.migration import MigrationManager, Migration
from antaris_memory.indexing import IndexManager, SearchIndex, TagIndex, DateIndex

# Concurrency
from antaris_memory.locking import FileLock, LockTimeout
from antaris_memory.versioning import VersionTracker, ConflictError

# Search
from antaris_memory.search import SearchEngine, SearchResult

# Context Packets (v1.1)
from antaris_memory.context_packet import ContextPacket, ContextPacketBuilder

# Recovery (v3.3.1)
from antaris_memory.recovery import RecoveryConfig, RecoveryManager

# Memory Types + Namespace Isolation (Sprint 2 + Sprint 8)
from antaris_memory.memory_types import MEMORY_TYPE_CONFIGS, get_type_config
from antaris_memory.namespace import NamespacedMemory, NamespaceManager

# Sprint 3 — Semantic utilities
from antaris_memory.utils import cosine_similarity

# v3.2 — Instrumentation + Co-occurrence (Tier 1 Semantic)
from antaris_memory.instrumentation import (
    SearchContext,
    extract_memory_references,
    anonymize_query,
    build_usage_signal,
)
from antaris_memory.cooccurrence import CooccurrenceIndex

# v4.1 — Tiered Semantic System + Data Collection
from antaris_memory.consent import ConsentStore
from antaris_memory.collection import CollectionAgent
from antaris_memory.ppmi import PPMILookup, StaticPPMILookup

# Backward compatibility - import legacy core if needed
try:
    from antaris_memory.core import MemorySystem as LegacyMemorySystem
except ImportError:
    LegacyMemorySystem = None

# MCP Server (optional — requires 'mcp' package)
try:
    from antaris_memory.mcp_server import create_server as create_mcp_server
    MCP_AVAILABLE = True
except ImportError:
    # mcp package not installed — provide a helpful stub
    MCP_AVAILABLE = False

    def create_mcp_server(*args, **kwargs):  # type: ignore[misc]
        """Stub: install 'mcp' to use the MCP server. ``pip install mcp``"""
        raise ImportError(
            "The 'mcp' package is required. Install with: pip install mcp"
        )

__all__ = [
    "MemorySystem",
    "MemoryEntry",
    
    # Core engines
    "DecayEngine",
    "SentimentTagger", 
    "TemporalEngine",
    "ConfidenceEngine",
    "CompressionEngine",
    "ForgettingEngine",
    "ConsolidationEngine",
    "InputGate",
    "KnowledgeSynthesizer",
    
    # Multi-agent (v0.3)
    "SharedMemoryPool",
    "AgentPermission",
    
    # Production features (v0.4)
    "ShardManager",
    "ShardKey", 
    "MigrationManager",
    "Migration",
    "IndexManager",
    "SearchIndex",
    "TagIndex",
    "DateIndex",
    
    # Concurrency (v0.5)
    "FileLock",
    "LockTimeout",
    "VersionTracker",
    "ConflictError",
    
    # Search (v1.0)
    "SearchEngine",
    "SearchResult",
    
    # Context Packets (v1.1)
    "ContextPacket",
    "ContextPacketBuilder",

    # Sprint 2 — Memory Types
    "MEMORY_TYPE_CONFIGS",
    "get_type_config",

    # Sprint 8 — Namespace Isolation
    "NamespacedMemory",
    "NamespaceManager",

    # Sprint 3 — Hybrid Semantic Search
    "cosine_similarity",   # pure-stdlib cosine similarity utility
    # MemorySystem.set_embedding_fn(fn)  — plug in an embedding callable
    # MemorySystem.search(...)           — uses hybrid BM25+cosine when fn is set

    # Sprint 5 — MCP Server
    "create_mcp_server",
    "MCP_AVAILABLE",

    # v3.2 — Instrumentation + Co-occurrence (Tier 1 Semantic)
    "SearchContext",
    "extract_memory_references",
    "anonymize_query",
    "build_usage_signal",
    "CooccurrenceIndex",

    # v4.1 — Tiered Semantic System + Data Collection
    "ConsentStore",
    "CollectionAgent",
    "PPMILookup",
    "StaticPPMILookup",
]
