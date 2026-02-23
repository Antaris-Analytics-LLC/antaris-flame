"""
Memory type configurations for antaris-memory Sprint 2.

Five canonical memory types with distinct decay and recall behaviour:

  episodic   — default, normal decay, normal recall priority
  fact       — normal decay, high recall priority
  preference — slow decay (3× half-life), high recall when context-matched
  procedure  — slow decay (3× half-life), high recall when task-matched
  mistake    — very slow decay (10× half-life), always surfaced first, 2× importance

Custom types are supported via type_config dicts with the same keys.
"""

from typing import Dict, Optional

# ── canonical type definitions ────────────────────────────────────────────────

MEMORY_TYPE_CONFIGS: Dict[str, Dict] = {
    "episodic": {
        "decay_multiplier": 1.0,   # Normal half-life
        "importance_boost": 1.0,   # No boost
        "recall_priority": 0.5,    # Normal
        "label": "Episodic",
    },
    "fact": {
        "decay_multiplier": 1.0,   # Normal half-life
        "importance_boost": 1.2,   # Slight boost
        "recall_priority": 0.7,    # High
        "label": "Fact",
    },
    "preference": {
        "decay_multiplier": 3.0,   # 3× half-life = slower decay
        "importance_boost": 1.2,
        "recall_priority": 0.7,    # High when context-matched
        "label": "Preference",
    },
    "procedure": {
        "decay_multiplier": 3.0,   # 3× half-life = slower decay
        "importance_boost": 1.3,
        "recall_priority": 0.75,   # High when task-matched
        "label": "Procedure",
    },
    "mistake": {
        "decay_multiplier": 10.0,  # 10× half-life = very slow decay
        "importance_boost": 2.0,   # 2× importance boost
        "recall_priority": 1.0,    # Highest — always first
        "label": "Mistake",
    },
}

DEFAULT_TYPE = "episodic"

# Severity levels for mistakes
SEVERITY_LEVELS = ("high", "medium", "low")


def get_type_config(memory_type: str, type_config: Optional[Dict] = None) -> Dict:
    """Return the config dict for a memory type.

    If *memory_type* is not in the canonical set and *type_config* is provided,
    the custom config is merged with the episodic defaults and returned.
    Unknown types with no override fall back to episodic defaults.
    """
    if memory_type in MEMORY_TYPE_CONFIGS:
        return MEMORY_TYPE_CONFIGS[memory_type]

    # Custom type — merge with episodic defaults
    base = dict(MEMORY_TYPE_CONFIGS["episodic"])
    if type_config:
        base.update(type_config)
    base.setdefault("label", memory_type.title())
    return base


def format_mistake_content(
    what_happened: str,
    correction: str,
    root_cause: Optional[str] = None,
    severity: str = "medium",
) -> str:
    """Format a mistake entry's content string."""
    parts = [
        f"MISTAKE: {what_happened}",
        f"CORRECTION: {correction}",
    ]
    if root_cause:
        parts.append(f"ROOT CAUSE: {root_cause}")
    parts.append(f"SEVERITY: {severity}")
    return " | ".join(parts)


def format_pitfall_line(type_metadata: Dict) -> str:
    """Format the '⚠️ Known Pitfall' line for context packets."""
    what = type_metadata.get("what_happened", "unknown issue")
    correction = type_metadata.get("correction", "no correction recorded")
    return f"⚠️ Known Pitfall: Last time this was attempted, {what}. Correction: {correction}"
