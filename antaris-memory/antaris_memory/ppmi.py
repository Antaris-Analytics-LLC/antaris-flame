"""
PPMILookup — stub interface for Tier 3 crowdsourced PPMI semantic index.

This module defines the abstract interface and a static JSON-based
implementation for the crowdsourced PPMI lookup table.

The PPMI (Pointwise Mutual Information) index is built by aggregating
anonymous word-pair co-occurrence statistics from opted-in users, then
computing PMI scores and shipping as a static JSON file bundled with
antaris-memory releases.

Tier 1 (local co-occurrence) provides personalized semantic boost.
Tier 3 (crowdsourced PPMI) provides domain-specific boost for cold-start
users and adds a network-effect layer on top of Tier 1.

Usage:
    from antaris_memory.ppmi import StaticPPMILookup

    ppmi = StaticPPMILookup("/path/to/workspace")
    ppmi.load()  # loads ppmi.json if present, no-op if absent

    related = ppmi.lookup("meeting")
    # {"reschedule": 3.2, "zoom": 2.8, "calendar": 2.1}
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

_PPMI_FILENAME = "ppmi.json"


class PPMILookup(ABC):
    """Abstract base class for PPMI lookup implementations.

    All implementations must support lookup(word) → Dict[str, float].
    This interface exists so future backends (cloud, embedded model, etc.)
    can be swapped in without touching the search integration layer.
    """

    @abstractmethod
    def lookup(self, word: str) -> Dict[str, float]:
        """Return related words and their PPMI scores for the given word.

        Args:
            word: The query word (lowercased, single token).

        Returns:
            Dict mapping related words to PPMI scores (0.0 if word unknown).
            Empty dict if word has no known associations.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this lookup has data loaded and is ready to use."""
        ...

    @abstractmethod
    def load(self) -> bool:
        """Load the lookup data. Returns True if data was found and loaded."""
        ...

    def boost_score(
        self,
        query: str,
        entry_content: str,
        boost_factor: float = 0.2,
    ) -> float:
        """Compute additive boost score for an entry given a query.

        Args:
            query: The search query string.
            entry_content: Memory entry content to score.
            boost_factor: Maximum additive boost (default 0.2).

        Returns:
            Float in [0.0, boost_factor].
        """
        if not self.is_available():
            return 0.0

        query_tokens = [t.lower() for t in query.split() if len(t) >= 2]
        entry_tokens = set(t.lower() for t in entry_content.split() if len(t) >= 2)

        if not query_tokens or not entry_tokens:
            return 0.0

        total = 0.0
        hits = 0
        for q_tok in query_tokens:
            related = self.lookup(q_tok)
            for e_tok in entry_tokens:
                if e_tok in related:
                    total += min(related[e_tok] / 5.0, 1.0)
                    hits += 1

        if hits == 0:
            return 0.0

        avg = total / len(query_tokens)
        return min(avg * boost_factor, boost_factor)


class StaticPPMILookup(PPMILookup):
    """PPMI lookup backed by a static JSON file bundled with the package.

    The JSON file format:
        {"word": {"related_word": score, ...}, ...}

    If the file does not exist, all lookups return empty dicts.
    This is the graceful-degradation path for users who don't have the
    crowdsourced index yet (e.g., fresh install, offline).

    The file is placed in the workspace directory and is loaded on demand.
    Future releases will ship an updated ppmi.json as the community index
    grows.
    """

    def __init__(
        self,
        workspace: str,
        filename: str = _PPMI_FILENAME,
    ):
        self.workspace = Path(workspace)
        self._path = self.workspace / filename
        self._data: Dict[str, Dict[str, float]] = {}
        self._loaded: bool = False
        self._available: bool = False

    def load(self) -> bool:
        """Load PPMI data from disk. No-op and returns False if file absent."""
        if not self._path.exists():
            self._loaded = True
            self._available = False
            return False
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Validate structure
            if not isinstance(raw, dict):
                self._loaded = True
                self._available = False
                return False
            self._data = {
                str(k): {str(rk): float(rv) for rk, rv in v.items()}
                for k, v in raw.items()
                if isinstance(v, dict)
            }
            self._loaded = True
            self._available = bool(self._data)
            return self._available
        except (json.JSONDecodeError, OSError, ValueError):
            self._loaded = True
            self._available = False
            return False

    def lookup(self, word: str) -> Dict[str, float]:
        """Return related words and PPMI scores for the given word."""
        if not self._available:
            return {}
        return self._data.get(word.lower(), {})

    def is_available(self) -> bool:
        """Return True if PPMI data is loaded and non-empty."""
        return self._available

    @property
    def vocab_size(self) -> int:
        """Number of words in the PPMI index."""
        return len(self._data)

    def __repr__(self) -> str:
        return (
            f"StaticPPMILookup(available={self._available}, "
            f"vocab_size={self.vocab_size})"
        )
