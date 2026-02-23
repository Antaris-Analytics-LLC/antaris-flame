"""
Recovery and context restoration for antaris-memory.

After compaction or restart, automatically gather memories from current session
and recent history. User-configurable recovery strategies.
"""

from typing import Literal, Optional, List
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

RecoveryMode = Literal["smart", "full", "minimal"]
RecoveryTimeWindow = Literal["session", "24h", "7d"]
RecoveryInject = Literal["auto", "cache", "manual"]


class RecoveryConfig:
    """User-facing configuration for recovery behavior."""
    
    RECOVERY_MODES = {
        "smart": {
            "description": "Balanced recovery (default). Pulls top 50 memories from last 24h, tags by channel, stores in cache. Smart injection only when relevant. ~5-10K tokens. Covers recent conversation context.",
            "search_limit": 50,
            "time_window": "24h",
            "channels": "all",
            "inject": "cache"
        },
        "minimal": {
            "description": "Lean recovery. Pulls only top 10 memories from current session. Manual injection on demand (most token-efficient). ~1-2K tokens. For constrained environments.",
            "search_limit": 10,
            "time_window": "session",
            "channels": None,
            "inject": "manual"
        }
    }
    
    def __init__(
        self,
        recovery_mode: RecoveryMode = "smart",
        recovery_search_limit: Optional[int] = None,
        recovery_time_window: Optional[RecoveryTimeWindow] = None,
        recovery_channels: Optional[List[str]] = None,
        recovery_inject: Optional[RecoveryInject] = None
    ):
        """
        Initialize recovery configuration.
        
        Args:
            recovery_mode: Predefined strategy ("smart", "full", "minimal"). Default: "smart"
            recovery_search_limit: Max memories to pull (overrides preset). Default: preset-specific
            recovery_time_window: How far back to search (overrides preset). Default: preset-specific
            recovery_channels: Which channels to include (overrides preset). Default: preset-specific
            recovery_inject: When to inject context (overrides preset). Default: preset-specific
        """
        
        # Start with preset mode
        preset = self.RECOVERY_MODES.get(recovery_mode, self.RECOVERY_MODES["smart"])
        
        self.mode = recovery_mode
        self.search_limit = recovery_search_limit if recovery_search_limit is not None else preset["search_limit"]
        self.time_window = recovery_time_window if recovery_time_window is not None else preset["time_window"]
        self.channels = recovery_channels if recovery_channels is not None else (preset["channels"] if preset["channels"] == "all" else None)
        self.inject = recovery_inject if recovery_inject is not None else preset["inject"]
        
        # Validation
        if not (10 <= self.search_limit <= 999):
            logger.warning(f"recovery_search_limit {self.search_limit} out of range [10, 999], clamping")
            self.search_limit = max(10, min(999, self.search_limit))
        
        if self.time_window not in ["session", "24h", "7d"]:
            logger.warning(f"recovery_time_window {self.time_window} invalid, defaulting to '24h'")
            self.time_window = "24h"
    
    def to_dict(self) -> dict:
        """Serialize config for storage."""
        return {
            "recovery_mode": self.mode,
            "recovery_search_limit": self.search_limit,
            "recovery_time_window": self.time_window,
            "recovery_channels": self.channels,
            "recovery_inject": self.inject
        }
    
    @staticmethod
    def help():
        """Print recovery configuration help."""
        help_text = """
RECOVERY CONFIGURATION — Help

After compaction or restart, antaris-memory can automatically gather context
from recent memories to bridge the context window gap. Choose a preset or customize.

Older memories (beyond 24h) are recalled naturally via search and decay when needed —
you don't need to load them all on startup.

PRESETS:
  "smart"   (default) — Balanced: 50 memories, 24h window, all channels, cache injection (~5-10K tokens)
  "minimal"          — Lean: 10 memories, session only, current channel, manual injection (~1-2K tokens)

OPTIONS:

  recovery_mode: "smart" | "full" | "minimal"
    Predefined strategy. Overrides other options if specified.
    Default: "smart"

  recovery_search_limit: 10-999
    Number of memories to pull on recovery.
    Trade-off: More memories = more context, higher token cost.
    Default: 50 (smart), 999 (full), 10 (minimal)

  recovery_time_window: "session" | "24h" | "7d"
    How far back to search for memories.
    - "session": Only current session (fresh start, minimal context)
    - "24h": Last 24 hours (balanced, catches recent context)
    - "7d": Last 7 days (maximum context, highest cost)
    Default: "24h" (smart), "7d" (full), "session" (minimal)

  recovery_channels: None | "all" | ["ch1", "ch2", ...]
    Which channels/sources to include.
    - None (default): Current channel only
    - "all": Search across all channels (multi-channel awareness)
    - ["ch1", "ch2"]: Specific channels by name/ID
    Default: "all" (smart/full), None (minimal)

  recovery_inject: "auto" | "cache" | "manual"
    When to inject recovered context.
    - "auto": Inject all on startup (uses 20-30K tokens upfront)
    - "cache": Store in memory, inject only when needed (smart default)
    - "manual": User decides when to recall (most token-efficient)
    Default: "cache" (smart), "auto" (full), "manual" (minimal)

EXAMPLES:

  # Default smart recovery (recommended)
  memory = MemoryStore(recovery_config=RecoveryConfig())

  # Minimal for token-constrained environments
  memory = MemoryStore(recovery_config=RecoveryConfig(recovery_mode="minimal"))

  # Custom: 75 memories from 24h, all channels, cache injection
  memory = MemoryStore(
    recovery_config=RecoveryConfig(
      recovery_search_limit=75,
      recovery_time_window="24h",
      recovery_channels="all",
      recovery_inject="cache"
    )
  )

PERFORMANCE NOTES:

  Token Cost Estimate (approximate):
    - smart (50 memories, 24h):  5-10K tokens (recommended for most use cases)
    - minimal (10 memories, session):  1-2K tokens (for constrained environments)

  Search Time Estimate:
    - "session":  <50ms
    - "24h":      50-150ms
    - "7d":       150-400ms

BEST PRACTICES:

  - Use "smart" (default) for balanced context recovery + token efficiency
  - Use "minimal" in token-constrained environments (e.g., mobile, edge agents)
  - Use "cache" injection to avoid upfront token cost while keeping memories available
  - Remember: older memories beyond recovery window are recalled naturally via search when needed
"""
        return help_text


class RecoveryManager:
    """Manage memory recovery after compaction/restart."""
    
    def __init__(self, store, config: RecoveryConfig):
        self.store = store
        self.config = config
        self.recovered_cache = {}
    
    def recover_memories(self) -> dict:
        """
        Recover memories based on configuration.
        
        Returns:
            {
                "memories": [list of MemoryEntry objects],
                "tags": {channel/source -> count},
                "timestamp": ISO timestamp,
                "config_used": config dict
            }
        """
        logger.info(f"Starting recovery: mode={self.config.mode}, limit={self.config.search_limit}, window={self.config.time_window}")
        
        # Calculate time window
        now = datetime.utcnow()
        if self.config.time_window == "session":
            # Only current session memories (simplified for MVP)
            min_time = now - timedelta(hours=1)
        elif self.config.time_window == "24h":
            min_time = now - timedelta(hours=24)
        else:  # "7d"
            min_time = now - timedelta(days=7)
        
        # Search for memories
        try:
            results = self.store.search_all(
                limit=self.config.search_limit,
                min_timestamp=min_time.isoformat()
            )
        except Exception as e:
            logger.error(f"Recovery search failed: {e}")
            return {"memories": [], "error": str(e), "config_used": self.config.to_dict()}
        
        # Tag by channel if multi-channel
        tags = {}
        for memory in results:
            source = memory.get("source", "unknown")
            tags[source] = tags.get(source, 0) + 1
        
        # Cache for smart/manual injection
        self.recovered_cache = {
            "memories": results,
            "tags": tags,
            "timestamp": now.isoformat(),
            "config": self.config.to_dict()
        }
        
        logger.info(f"Recovered {len(results)} memories from {len(tags)} sources")
        
        return self.recovered_cache
    
    def get_cached_memories(self) -> list:
        """Get recovered memories from cache (for cache/manual injection)."""
        return self.recovered_cache.get("memories", [])
    
    def inject_into_context(self) -> str:
        """Generate [MEMORY RESTORED] block for context injection."""
        if not self.recovered_cache:
            return ""
        
        memories = self.recovered_cache.get("memories", [])
        tags = self.recovered_cache.get("tags", {})
        
        block = "[MEMORY RESTORED]\n"
        block += f"Recovered {len(memories)} memories from {len(tags)} sources:\n"
        for source, count in sorted(tags.items()):
            block += f"  - {source}: {count} memories\n"
        block += "Available in context. Use as needed.\n"
        
        return block
