"""Memory entry — the atomic unit of memory."""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional


class MemoryEntry:
    """Single memory unit with metadata, decay, sentiment, and confidence.

    Sprint 2 additions
    ------------------
    memory_type : str
        One of "episodic", "fact", "preference", "procedure", "mistake"
        or any custom string.  Defaults to "episodic".
    type_metadata : dict
        Type-specific structured data.  For mistakes this holds
        {"what_happened": ..., "correction": ..., "root_cause": ..., "severity": ...}.
        Empty for other types unless the caller populates it.
    """

    __slots__ = (
        "content", "source", "line", "category", "created",
        "last_accessed", "access_count", "importance", "confidence",
        "sentiment", "tags", "related", "hash",
        # Sprint 2
        "memory_type", "type_metadata",
    )

    def __init__(
        self,
        content: str,
        source: str = "",
        line: int = 0,
        category: str = "general",
        created: str = None,
        memory_type: str = "episodic",
    ):
        self.content = content
        self.source = source
        self.line = line
        self.category = category
        self.created = created or datetime.now().isoformat()
        self.last_accessed = self.created
        self.access_count: int = 0
        self.importance: float = 1.0
        self.confidence: float = 0.5
        self.sentiment: Dict[str, float] = {}
        self.tags: List[str] = []
        self.related: List[str] = []
        self.hash = hashlib.md5(
            f"{source}:{line}:{content[:100]}".encode()
        ).hexdigest()[:12]
        # Sprint 2
        self.memory_type: str = memory_type
        self.type_metadata: Dict = {}

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> Dict:
        d = {
            "hash": self.hash,
            "content": self.content,
            "source": self.source,
            "line": self.line,
            "category": self.category,
            "created": self.created,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "importance": round(self.importance, 4),
            "confidence": round(self.confidence, 4),
            "sentiment": self.sentiment,
            "tags": self.tags,
            "related": self.related,
        }
        # Sprint 2 — only write if non-default to keep file size stable
        if self.memory_type and self.memory_type != "episodic":
            d["memory_type"] = self.memory_type
        if self.type_metadata:
            d["type_metadata"] = self.type_metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "MemoryEntry":
        m = cls(
            d["content"], d.get("source", ""), d.get("line", 0),
            d.get("category", "general"), d.get("created"),
            memory_type=d.get("memory_type", "episodic"),
        )
        m.last_accessed = d.get("last_accessed", m.created)
        m.access_count = d.get("access_count", 0)
        m.importance = d.get("importance", 1.0)
        m.confidence = d.get("confidence", 0.5)
        m.sentiment = d.get("sentiment", {})
        m.tags = d.get("tags", [])
        m.related = d.get("related", [])
        m.hash = d.get("hash", m.hash)
        m.type_metadata = d.get("type_metadata", {})
        return m

    def __repr__(self) -> str:
        return (
            f"<Memory {self.hash} type={self.memory_type} "
            f"score={self.importance:.2f} conf={self.confidence:.2f}>"
        )
