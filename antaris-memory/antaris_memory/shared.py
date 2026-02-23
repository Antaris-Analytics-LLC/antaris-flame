"""
Shared Memory — Multi-agent memory pools with access controls.

Enables multiple agents to share a memory space with:
- Role-based access (read, write, admin)
- Per-agent namespaces with optional sharing
- Conflict resolution when agents disagree
- Knowledge propagation between agents
- Audit trail of who wrote/modified what

All data stored as JSON files. Zero dependencies.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .entry import MemoryEntry
from .decay import DecayEngine
from .search import SearchEngine
from .sentiment import SentimentTagger
from .utils import atomic_write_json


class AgentPermission:
    """Access control for a single agent."""

    __slots__ = ("agent_id", "role", "namespaces", "created")

    ROLES = ("read", "write", "admin")

    def __init__(self, agent_id: str, role: str = "write",
                 namespaces: List[str] = None):
        if role not in self.ROLES:
            raise ValueError(f"Role must be one of {self.ROLES}")
        self.agent_id = agent_id
        self.role = role
        self.namespaces = namespaces or ["shared"]
        self.created = datetime.now().isoformat()

    def can_read(self) -> bool:
        return self.role in ("read", "write", "admin")

    def can_write(self) -> bool:
        return self.role in ("write", "admin")

    def can_admin(self) -> bool:
        return self.role == "admin"

    def can_access_namespace(self, namespace: str) -> bool:
        if self.role == "admin":
            return True
        return namespace in self.namespaces or namespace == "shared"

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "namespaces": self.namespaces,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AgentPermission":
        perm = cls(d["agent_id"], d.get("role", "write"),
                   d.get("namespaces", ["shared"]))
        perm.created = d.get("created", perm.created)
        return perm


class SharedMemoryPool:
    """Multi-agent shared memory with access controls and conflict resolution.

    Parameters
    ----------
    pool_dir : str
        Directory for the shared memory pool.
    pool_name : str
        Name of the pool (used in filenames).
    """

    def __init__(self, pool_dir: str, pool_name: str = "default"):
        self.pool_dir = os.path.abspath(pool_dir)
        self.pool_name = pool_name
        self.state_path = os.path.join(self.pool_dir,
                                        f"pool_{pool_name}.json")
        self.audit_path = os.path.join(self.pool_dir,
                                        f"pool_{pool_name}_audit.json")

        # Internal state
        self.memories: List[MemoryEntry] = []
        self.permissions: Dict[str, AgentPermission] = {}
        self.conflicts: List[Dict] = []
        self._hashes: set = set()
        self._memory_by_hash: Dict[str, MemoryEntry] = {}  # O(1) lookup by hash
        self._decay = DecayEngine(half_life=7.0)
        self._sentiment = SentimentTagger()
        # Bug fix: use BM25 SearchEngine for shared pool reads instead of
        # primitive word-overlap, closing the search quality gap vs private memory.
        # Bug fix: BM25 SearchEngine for shared pool reads (auto-rebuilds on corpus change)
        self._search_engine = SearchEngine()

        os.makedirs(self.pool_dir, exist_ok=True)

    # ── agent management ────────────────────────────────────────────────

    def register_agent(self, agent_id: str, role: str = "write",
                       namespaces: List[str] = None) -> AgentPermission:
        """Register an agent with the pool."""
        perm = AgentPermission(agent_id, role, namespaces)
        self.permissions[agent_id] = perm
        self._audit("register", agent_id, f"role={role}")
        return perm

    def remove_agent(self, agent_id: str, requester: str = None) -> bool:
        """Remove an agent from the pool. Requires admin."""
        if requester and not self._check_admin(requester):
            return False
        if agent_id in self.permissions:
            del self.permissions[agent_id]
            self._audit("remove_agent", requester or "system",
                        f"removed={agent_id}")
            return True
        return False

    def list_agents(self) -> List[Dict]:
        """List all registered agents and their roles."""
        return [p.to_dict() for p in self.permissions.values()]

    # ── memory operations ───────────────────────────────────────────────

    def write(self, agent_id: str, content: str, namespace: str = "shared",
              category: str = "general", metadata: Dict = None) -> Optional[MemoryEntry]:
        """Write a memory to the pool. Checks permissions."""
        perm = self.permissions.get(agent_id)
        if not perm:
            return None
        if not perm.can_write():
            return None
        if not perm.can_access_namespace(namespace):
            return None

        entry = MemoryEntry(content, source=f"agent:{agent_id}",
                            category=category)
        entry.sentiment = self._sentiment.analyze(content)

        # Add agent metadata tags
        entry.tags.append(f"ns:{namespace}")
        entry.tags.append(f"agent:{agent_id}")
        if metadata:
            for k, v in metadata.items():
                entry.tags.append(f"{k}:{v}")

        # Check for conflicts with existing memories
        conflict = self._check_conflict(entry, agent_id)
        if conflict:
            self.conflicts.append(conflict)

        if entry.hash not in self._hashes:
            self.memories.append(entry)
            self._hashes.add(entry.hash)
            self._memory_by_hash[entry.hash] = entry
            self._audit("write", agent_id,
                        f"ns={namespace} hash={entry.hash}")
            return entry
        return None

    def read(self, agent_id: str, query: str, namespace: str = None,
             limit: int = 20, recency_bias: float = 0.0) -> List[MemoryEntry]:
        """Search shared memories using BM25. Respects namespace permissions.

        Bug fix: replaced primitive word-overlap scorer with the same BM25
        SearchEngine used by private MemorySystem, closing the search quality
        gap in multi-agent shared-pool scenarios.
        
        Args:
            agent_id: ID of the agent making the request.
            query: Search query string.
            namespace: Optional namespace filter.
            limit: Maximum number of results to return.
            recency_bias: float 0.0-1.0. When > 0, boosts recent memories.
                Score adjustment: score *= (1 + recency_bias * recency_factor)
                where recency_factor = 1.0 for memories from today, decaying 
                to 0.0 for memories > 7 days old.
                At recency_bias=1.0, today's memories get up to 2x score boost.
                Recommended range: 0.0-0.5 for balanced results. Values > 0.5
                heavily favor recency over relevance.
        """
        perm = self.permissions.get(agent_id)
        if not perm or not perm.can_read():
            return []

        # Filter to accessible memories first
        accessible = [
            m for m in self.memories
            if perm.can_access_namespace(self._get_namespace(m))
            and (not namespace or self._get_namespace(m) == namespace)
        ]

        if not accessible:
            return []

        # BM25 search with decay re-scoring.
        # SearchEngine.search() auto-rebuilds its index when the corpus changes
        # (tracks doc count + corpus version internally), so no manual dirty flag needed.
        bm25_results = self._search_engine.search(query, accessible, limit=limit * 2)
        scored = []
        now = datetime.now()
        
        for r in bm25_results:
            decay_score = self._decay.score(r.entry)
            base_score = r.relevance * (0.5 + decay_score)
            
            # Apply recency bias if requested
            if recency_bias > 0.0:
                # Calculate age in days from memory timestamp
                try:
                    memory_time = datetime.fromisoformat(r.entry.created.replace('Z', '+00:00'))
                    # Handle timezone-naive timestamps
                    if memory_time.tzinfo is None:
                        memory_time = memory_time.replace(tzinfo=now.tzinfo) if now.tzinfo else memory_time
                        
                    age_days = (now - memory_time).total_seconds() / 86400.0  # Convert to days
                    # Linear decay over 7 days: 1.0 for today, 0.0 for 7+ days old
                    recency_factor = max(0.0, 1.0 - (age_days / 7.0))
                    
                    # Apply recency boost
                    final_score = base_score * (1.0 + recency_bias * recency_factor)
                except (ValueError, AttributeError, TypeError):
                    # If timestamp parsing fails, use base score without recency bias
                    final_score = base_score
            else:
                final_score = base_score
            
            if final_score > 0:
                scored.append((r.entry, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [m for m, _ in scored[:limit]]

        # Reinforce only the final returned results (avoid reinforcing near-cutoff docs)
        for m in top:
            self._decay.reinforce(m)

        # Avoid logging raw PII in queries
        try:
            from .instrumentation import anonymize_query as _anonymize
            qlog = _anonymize(query)
        except Exception:
            qlog = query
        self._audit("read", agent_id, f"query={qlog[:50]} results={len(top)}")
        return top

    def propagate(self, from_agent: str, to_namespace: str,
                  query: str = None, limit: int = 10) -> int:
        """Propagate an agent's memories to another namespace.

        Useful for sharing discoveries across agent teams.
        """
        perm = self.permissions.get(from_agent)
        if not perm or not perm.can_write():
            return 0

        # Find memories written by this agent
        agent_memories = [m for m in self.memories
                          if f"agent:{from_agent}" in m.tags]

        if query:
            # Use BM25 SearchEngine for consistent search quality
            bm25 = SearchEngine()
            bm25_results = bm25.search(query, agent_memories, limit=limit)
            agent_memories = [r.entry for r in bm25_results]
        else:
            agent_memories = agent_memories[:limit]

        count = 0
        for m in agent_memories:
            # Create a copy in the target namespace
            new_entry = MemoryEntry(
                m.content,
                source=f"propagated:{from_agent}",
                category=m.category,
            )
            new_entry.tags = list(m.tags)
            # Replace namespace tag
            new_entry.tags = [t for t in new_entry.tags
                              if not t.startswith("ns:")]
            new_entry.tags.append(f"ns:{to_namespace}")
            new_entry.tags.append(f"propagated_from:{from_agent}")
            new_entry.sentiment = dict(m.sentiment)
            new_entry.confidence = m.confidence

            if new_entry.hash not in self._hashes:
                self.memories.append(new_entry)
                self._hashes.add(new_entry.hash)
                self._memory_by_hash[new_entry.hash] = new_entry
                count += 1

        self._audit("propagate", from_agent,
                    f"to={to_namespace} count={count}")
        return count

    # ── conflict resolution ─────────────────────────────────────────────

    def get_conflicts(self) -> List[Dict]:
        """Return unresolved conflicts."""
        return [c for c in self.conflicts if not c.get("resolved")]

    def resolve_conflict(self, conflict_index: int, resolution: str,
                         resolver: str = "system") -> bool:
        """Resolve a conflict by choosing which memory wins."""
        if conflict_index >= len(self.conflicts):
            return False
        conflict = self.conflicts[conflict_index]
        conflict["resolved"] = True
        conflict["resolution"] = resolution
        conflict["resolved_by"] = resolver
        conflict["resolved_at"] = datetime.now().isoformat()
        self._audit("resolve_conflict", resolver,
                    f"index={conflict_index} resolution={resolution}")
        return True

    # ── persistence ─────────────────────────────────────────────────────

    def save(self) -> str:
        """Save pool state to disk (Bug 3 fix: uses atomic write with FileLock)."""
        from . import __version__ as _pkg_version  # Bug fix: add package version
        data = {
            "schema_version": "0.3.0",       # storage format version
            "package_version": _pkg_version,  # antaris-memory package version
            "pool_name": self.pool_name,
            "saved_at": datetime.now().isoformat(),
            "agent_count": len(self.permissions),
            "memory_count": len(self.memories),
            "conflict_count": len(self.conflicts),
            "permissions": {k: v.to_dict()
                            for k, v in self.permissions.items()},
            "memories": [m.to_dict() for m in self.memories],
            "conflicts": self.conflicts,
        }
        atomic_write_json(self.state_path, data)
        return self.state_path

    def load(self) -> int:
        """Load pool state from disk."""
        if not os.path.exists(self.state_path):
            return 0
        with open(self.state_path) as f:
            data = json.load(f)
        self.pool_name = data.get("pool_name", self.pool_name)
        self.permissions = {
            k: AgentPermission.from_dict(v)
            for k, v in data.get("permissions", {}).items()
        }
        self.memories = [MemoryEntry.from_dict(d)
                         for d in data.get("memories", [])]
        self._hashes = {m.hash for m in self.memories}
        self._memory_by_hash = {m.hash: m for m in self.memories}
        self.conflicts = data.get("conflicts", [])
        return len(self.memories)

    # ── stats ───────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Pool statistics."""
        namespaces = {}
        agents = {}
        for m in self.memories:
            ns = self._get_namespace(m)
            namespaces[ns] = namespaces.get(ns, 0) + 1
            agent = self._get_agent(m)
            if agent:
                agents[agent] = agents.get(agent, 0) + 1

        return {
            "pool_name": self.pool_name,
            "total_memories": len(self.memories),
            "registered_agents": len(self.permissions),
            "namespaces": namespaces,
            "memories_by_agent": agents,
            "unresolved_conflicts": len(self.get_conflicts()),
            "total_conflicts": len(self.conflicts),
        }

    # ── cross-agent instrumentation signals ────────────────────────────

    def mark_used_cross_agent(
        self,
        requesting_agent: str,
        memory_id: str,
        boost_factor: float = 1.3,
    ) -> bool:
        """Signal that a retrieved memory was actually used in an agent response.

        If the memory was written by a DIFFERENT agent (cross-agent use),
        boost its confidence and reset its decay. This is the core cross-agent
        learning signal: good memories that help other agents get promoted.

        Args:
            requesting_agent: The agent that retrieved and used the memory.
            memory_id: Hash of the memory entry that was used.
            boost_factor: Confidence multiplier for cross-agent boost (default 1.3).

        Returns:
            True if the memory was found and boosted, False if not found.
        """
        m = self._memory_by_hash.get(memory_id)
        if m is None:
            return False
        writer = self._get_agent(m)
        is_cross_agent = (writer is not None and writer != requesting_agent)
        # Always reinforce decay (reset freshness)
        self._decay.reinforce(m)
        if is_cross_agent:
            # Cross-agent use: boost confidence
            m.confidence = min(1.0, m.confidence * boost_factor)
            self._audit(
                "cross_agent_boost",
                requesting_agent,
                f"memory={memory_id[:16]} writer={writer} "
                f"confidence={m.confidence:.3f}",
            )
        else:
            self._audit(
                "self_reinforce",
                requesting_agent,
                f"memory={memory_id[:16]}",
            )
        return True

    def mark_used_batch(
        self,
        requesting_agent: str,
        memory_ids: List[str],
        boost_factor: float = 1.3,
    ) -> int:
        """Mark multiple memories as used. Returns count of memories boosted."""
        return sum(
            self.mark_used_cross_agent(requesting_agent, mid, boost_factor)
            for mid in memory_ids
        )

    def search_with_context(
        self,
        agent_id: str,
        query: str,
        instrumentation_context=None,
        limit: int = 20,
        namespace: str = None,
        recency_bias: float = 0.0,
    ) -> Tuple:
        """Search and populate an instrumentation SearchContext if provided.

        Returns:
            (results: List[MemoryEntry], context: Optional[SearchContext])
        """
        results = self.read(agent_id, query, namespace, limit, recency_bias)
        if instrumentation_context is not None:
            instrumentation_context.retrieved_memory_ids = [
                m.hash for m in results
            ]
        return (results, instrumentation_context)

    # ── private ─────────────────────────────────────────────────────────

    def _check_admin(self, agent_id: str) -> bool:
        perm = self.permissions.get(agent_id)
        return perm and perm.can_admin()

    def _get_namespace(self, entry: MemoryEntry) -> str:
        for tag in entry.tags:
            if tag.startswith("ns:"):
                return tag[3:]
        return "shared"

    def _get_agent(self, entry: MemoryEntry) -> Optional[str]:
        for tag in entry.tags:
            if tag.startswith("agent:"):
                return tag[6:]
        return None

    def _check_conflict(self, new_entry: MemoryEntry,
                        agent_id: str) -> Optional[Dict]:
        """Check if new memory conflicts with existing ones from other agents."""
        import re
        new_words = set(re.findall(r"\w{4,}", new_entry.content.lower()))

        for existing in self.memories:
            existing_agent = self._get_agent(existing)
            if existing_agent == agent_id:
                continue  # Same agent, no conflict

            existing_words = set(
                re.findall(r"\w{4,}", existing.content.lower()))
            overlap = len(new_words & existing_words)

            if overlap < 3:
                continue

            # Check for negation patterns suggesting contradiction
            new_lower = new_entry.content.lower()
            exist_lower = existing.content.lower()
            negation_signals = [
                ("don't ", "do "), ("won't ", "will "),
                ("can't ", "can "), ("shouldn't ", "should "),
                ("isn't ", "is "), ("aren't ", "are "),
                ("reject", "accept"), ("disagree", "agree"),
                ("false", "true"), ("wrong", "right"),
                ("failed", "succeeded"), ("no ", "yes "),
            ]

            # Check for explicit negation pairs
            has_negation = False
            for neg, pos in negation_signals:
                if ((neg in new_lower and pos in exist_lower) or
                        (pos in new_lower and neg in exist_lower)):
                    has_negation = True
                    break

            # Also detect general negation asymmetry: one text
            # contains "not" while the other doesn't, with high overlap
            if not has_negation and overlap >= 3:
                new_has_not = (" not " in new_lower or
                               new_lower.startswith("not "))
                exist_has_not = (" not " in exist_lower or
                                 exist_lower.startswith("not "))
                if new_has_not != exist_has_not:
                    has_negation = True

            if has_negation:
                return {
                        "type": "contradiction",
                        "new_memory": new_entry.hash,
                        "new_agent": agent_id,
                        "new_content": new_entry.content[:200],
                        "existing_memory": existing.hash,
                        "existing_agent": existing_agent,
                        "existing_content": existing.content[:200],
                        "overlap_words": list(new_words & existing_words)[:10],
                        "detected_at": datetime.now().isoformat(),
                        "resolved": False,
                    }
        return None

    def _audit(self, action: str, agent_id: str, detail: str = "") -> None:
        """Append to audit log with full read-modify-write locking.

        Uses FileLock around the entire read→append→write cycle to prevent
        lost updates when multiple agents write concurrently.
        """
        from .locking import FileLock

        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "agent_id": agent_id,
            "detail": detail,
        }
        with FileLock(self.audit_path, timeout=10.0):
            log = []
            if os.path.exists(self.audit_path):
                try:
                    with open(self.audit_path) as f:
                        log = json.load(f)
                except (json.JSONDecodeError, IOError):
                    log = []
            log.append(entry)
            # Keep last 500 entries
            atomic_write_json(self.audit_path, log[-500:], lock=False)
