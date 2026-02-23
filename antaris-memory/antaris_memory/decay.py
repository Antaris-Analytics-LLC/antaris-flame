"""Claim 16: Memory decay & reinforcement using Ebbinghaus-inspired curves.

Sprint 2: type-aware half-life via MEMORY_TYPE_CONFIGS decay_multiplier.
"""

import math
from datetime import datetime
from .entry import MemoryEntry

# Defaults — users can override via MemorySystem config
DEFAULT_HALF_LIFE = 7.0  # days
ARCHIVE_THRESHOLD = 0.15
REINFORCEMENT_BOOST = 0.25
MAX_SCORE = 2.0


class DecayEngine:
    """Ebbinghaus-inspired memory decay with reinforcement on access.

    Sprint 2 addition: if an entry has a ``memory_type`` attribute, the
    effective half-life is scaled by the type's ``decay_multiplier``.
    This makes mistakes decay 10× slower and preferences/procedures 3× slower.
    """

    def __init__(self, half_life: float = DEFAULT_HALF_LIFE,
                 archive_threshold: float = ARCHIVE_THRESHOLD,
                 reinforcement_boost: float = REINFORCEMENT_BOOST,
                 max_score: float = MAX_SCORE):
        self.half_life = half_life
        self.archive_threshold = archive_threshold
        self.reinforcement_boost = reinforcement_boost
        self.max_score = max_score

    # -- type config lookup (lazy import to avoid circular deps) ---------------

    @staticmethod
    def _type_multiplier(entry: MemoryEntry) -> float:
        """Return the decay multiplier for an entry's memory_type."""
        memory_type = getattr(entry, "memory_type", "episodic") or "episodic"
        # Lazy import to avoid circular dependency at module load time
        from .memory_types import MEMORY_TYPE_CONFIGS
        cfg = MEMORY_TYPE_CONFIGS.get(memory_type)
        if cfg is None:
            # Custom type — check type_metadata for an override
            type_metadata = getattr(entry, "type_metadata", {}) or {}
            return type_metadata.get("decay_multiplier", 1.0)
        return cfg["decay_multiplier"]

    # -- public API ------------------------------------------------------------

    def score(self, entry: MemoryEntry, now: datetime = None) -> float:
        """Calculate current memory strength after decay + reinforcement.

        Sprint 2: effective half-life = base_half_life × type_multiplier.
        """
        now = now or datetime.now()
        created = datetime.fromisoformat(entry.created)
        # Normalize timezone: if now is tz-aware but created is naive, strip tz from now
        if created.tzinfo is None and now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        elif created.tzinfo is not None and now.tzinfo is None:
            from datetime import timezone as _tz
            now = now.replace(tzinfo=_tz.utc)
        age_days = max((now - created).total_seconds() / 86400, 0.001)

        effective_hl = self.half_life * self._type_multiplier(entry)
        base_decay = entry.importance * math.pow(2, -age_days / effective_hl)
        reinforcement = (
            entry.access_count * self.reinforcement_boost
            * math.pow(2, -age_days / (effective_hl * 2))
        )
        return round(min(base_decay + reinforcement, self.max_score), 4)

    def reinforce(self, entry: MemoryEntry) -> None:
        """Boost a memory when it's accessed."""
        entry.access_count += 1
        entry.last_accessed = datetime.now().isoformat()
        entry.importance = min(
            entry.importance + 0.1 / (1 + entry.access_count * 0.1),
            self.max_score,
        )

    def should_archive(self, entry: MemoryEntry, now: datetime = None) -> bool:
        """True if this memory has decayed below the archive threshold."""
        return self.score(entry, now) < self.archive_threshold

    def effective_half_life(self, entry: MemoryEntry) -> float:
        """Return the effective half-life (days) for *entry*, including type scaling."""
        return self.half_life * self._type_multiplier(entry)
