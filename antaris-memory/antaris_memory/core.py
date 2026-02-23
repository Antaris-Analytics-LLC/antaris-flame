"""
MemorySystem — the main interface to antaris-memory.

Usage:
    from antaris_memory import MemorySystem

    mem = MemorySystem("./workspace")
    mem.ingest_file("notes.md", category="tactical")
    mem.ingest_directory("./memory", category="tactical")
    results = mem.search("patent filing")
    mem.save()
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .entry import MemoryEntry
from .decay import DecayEngine
from .sentiment import SentimentTagger
from .temporal import TemporalEngine
from .confidence import ConfidenceEngine
from .compression import CompressionEngine
from .forgetting import ForgettingEngine
from .consolidation import ConsolidationEngine
from .gating import InputGate
from .search import SearchEngine, SearchResult
from .synthesis import KnowledgeSynthesizer
from .cooccurrence import CooccurrenceIndex
from .consent import ConsentStore
from .collection import CollectionAgent
from .ppmi import StaticPPMILookup

# Default tags to auto-extract
_DEFAULT_TAG_TERMS = [
    "web3", "ethereum", "postgresql", "optimization", "cost",
    "revenue", "security", "deployment", "production", "testing",
]


class MemorySystem:
    """Complete memory system with decay, sentiment, gating, and synthesis.

    Parameters
    ----------
    workspace : str
        Root directory. Metadata files are stored here.
    half_life : float
        Decay half-life in days (default 7).
    tag_terms : list[str] | None
        Custom terms to auto-tag. Merged with built-in defaults.
    """

    def __init__(
        self,
        workspace: str = ".",
        half_life: float = 7.0,
        tag_terms: List[str] = None,
        max_memories: Optional[int] = None,
        enable_semantic: bool = False,
        cooccurrence_boost_weight: float = 0.3,
        collection_endpoint: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        workspace : str
            Root directory. Metadata and index files are stored here.
        half_life : float
            Decay half-life in days (default 7).
        tag_terms : list[str] | None
            Custom terms to auto-tag. Merged with built-in defaults.
        max_memories : int | None
            Maximum number of memories to retain. When the cap is reached,
            new ingests are rejected until old memories are pruned.
            None = unlimited (default).
        enable_semantic : bool
            If True, SemanticEngine.rerank() is called during search
            (requires sentence-transformers installed). Default False.
        cooccurrence_boost_weight : float
            Additive boost factor for co-occurrence semantic signals (0.0–1.0).
            Default 0.3.
        collection_endpoint : str | None
            HTTPS POST URL for anonymous word-pair collection (Supabase).
            None = stubbed (not yet wired). Default None.
        """
        self.workspace = os.path.abspath(workspace)
        self.metadata_path = os.path.join(self.workspace, "memory_metadata.json")
        self.audit_path = os.path.join(self.workspace, "memory_audit.json")
        self.max_memories = max_memories
        self.enable_semantic = enable_semantic
        self.cooccurrence_boost_weight = cooccurrence_boost_weight

        # Engines
        self.decay = DecayEngine(half_life=half_life)
        self.sentiment = SentimentTagger()
        self.temporal = TemporalEngine()
        self.confidence = ConfidenceEngine()
        self.compression = CompressionEngine()
        self.forgetting = ForgettingEngine()
        self.consolidation = ConsolidationEngine(decay=self.decay)
        self.gating = InputGate()
        self.synthesis = KnowledgeSynthesizer()
        self.search_engine = SearchEngine()

        # Tag terms
        self._tag_terms = list(set(_DEFAULT_TAG_TERMS + (tag_terms or [])))

        # Memory store
        self.memories: List[MemoryEntry] = []
        self._hashes: set = set()

        # ── Tier 1: Co-occurrence index (always-on) ───────────────────────────
        self.cooccurrence = CooccurrenceIndex(workspace=self.workspace)
        self.cooccurrence.load()

        # ── Tier 2: Semantic engine (opt-in) ─────────────────────────────────
        self.semantic_engine: Optional[object] = None
        if enable_semantic:
            try:
                from .semantic import SemanticEngine
                self.semantic_engine = SemanticEngine(workspace=self.workspace)
                self.semantic_engine.load()
            except Exception:
                # Graceful degradation — semantic unavailable (missing dep etc.)
                self.semantic_engine = None

        # ── Tier 3: Crowdsourced PPMI (loads if ppmi.json present) ───────────
        self.ppmi = StaticPPMILookup(workspace=self.workspace)
        self.ppmi.load()

        # ── Data collection pipeline ─────────────────────────────────────────
        self.consent = ConsentStore(workspace=self.workspace)
        self.consent.load()
        self.collection_agent = CollectionAgent(
            workspace=self.workspace,
            cooccurrence=self.cooccurrence,
            consent=self.consent,
            endpoint_url=collection_endpoint,
        )
        self.collection_agent.load_state()

    # ── persistence ─────────────────────────────────────────────────────

    def save(self) -> str:
        """Write memory state and co-occurrence index to disk. Returns path."""
        data = {
            "version": "0.2.0",
            "saved_at": datetime.now().isoformat(),
            "count": len(self.memories),
            "memories": [m.to_dict() for m in self.memories],
        }
        from .utils import atomic_write_json
        atomic_write_json(self.metadata_path, data)

        # Persist co-occurrence index if it has been updated
        try:
            if self.cooccurrence._dirty:
                self.cooccurrence.save()
        except Exception:
            pass  # Index save errors never block memory saves

        return self.metadata_path

    def load(self) -> int:
        """Load memory state from disk. Returns count loaded."""
        if not os.path.exists(self.metadata_path):
            return 0
        with open(self.metadata_path) as f:
            data = json.load(f)
        self.memories = [MemoryEntry.from_dict(d) for d in data.get("memories", [])]
        self._hashes = {m.hash for m in self.memories}
        # Build search index for BM25 ranking
        if self.memories:
            self.search_engine.build_index(self.memories)
        return len(self.memories)

    # ── ingestion ───────────────────────────────────────────────────────

    def ingest(self, content: str, source: str = "inline",
               category: str = "general") -> int:
        """Ingest raw text. Returns count of new memories added.

        If max_memories is set and the cap is reached, new memories are
        silently rejected. Accepted memories are fed into the CooccurrenceIndex
        to power Tier 1 semantic search boost.
        """
        count = 0
        for i, line in enumerate(content.split("\n")):
            stripped = line.strip()
            if len(stripped) < 15 or stripped.startswith("```") or stripped == "---":
                continue

            # Enforce memory cap
            if self.max_memories is not None and len(self.memories) >= self.max_memories:
                break

            entry = MemoryEntry(stripped, source, i + 1, category)
            if entry.hash in self._hashes:
                continue
            entry.tags = self._extract_tags(stripped)
            entry.sentiment = self.sentiment.analyze(stripped)
            if category == "strategic":
                entry.confidence = 0.8
            elif category in ("operational", "business"):
                entry.confidence = 0.6
            self.memories.append(entry)
            self._hashes.add(entry.hash)

            # ── Tier 1: Feed co-occurrence index ─────────────────────────────
            self.cooccurrence.update_from_memory(stripped)

            count += 1

        # Trigger collection cycle if due (non-blocking check)
        try:
            self.collection_agent.run_cycle()
        except Exception:
            pass  # Collection errors never block ingestion

        return count

    def ingest_file(self, path: str, category: str = "general") -> int:
        """Ingest a single file."""
        try:
            content = Path(path).read_text(errors="replace")
        except FileNotFoundError:
            return 0
        return self.ingest(content, source=path, category=category)

    def ingest_directory(self, directory: str, pattern: str = "*.md",
                         category: str = "general") -> int:
        """Ingest all matching files in a directory."""
        d = Path(directory)
        if not d.exists():
            return 0
        count = 0
        for f in sorted(d.glob(pattern)):
            count += self.ingest_file(str(f), category=category)
        return count

    def ingest_with_gating(self, content: str, source: str = "inline", context: Dict = None) -> int:
        """Ingest content with automatic priority-based routing and filtering.
        
        P3 (ephemeral) content is silently dropped. P0-P2 content is stored
        with appropriate category assignment.
        
        Returns count of new memories added (P3 content doesn't count).
        """
        count = 0
        for i, line in enumerate(content.split("\n")):
            stripped = line.strip()
            if len(stripped) < 5:  # Skip very short lines
                continue
                
            # Use gating system to classify and route
            gate_context = context or {}
            gate_context.update({"source": source, "line": i + 1})
            routing = self.gating.route(stripped, gate_context)
            
            # Skip P3 ephemeral content
            if not routing["store"]:
                continue
                
            # Check for duplicates
            entry = MemoryEntry(stripped, source, i + 1, routing["category"])
            if entry.hash in self._hashes:
                continue
                
            # Apply standard processing
            entry.tags = self._extract_tags(stripped)
            entry.sentiment = self.sentiment.analyze(stripped)
            
            # Set confidence based on priority
            if routing["priority"] == "P0":
                entry.confidence = 0.9  # High confidence for critical items
            elif routing["priority"] == "P1":
                entry.confidence = 0.7  # Good confidence for operational items
            else:  # P2
                entry.confidence = 0.5  # Standard confidence for contextual items
                
            entry.tags.append(routing["priority"])  # Add priority as tag
            
            self.memories.append(entry)
            self._hashes.add(entry.hash)
            count += 1
            
        return count

    # ── search ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 20,
        category: str = None,
        min_confidence: float = 0.0,
        sentiment_filter: str = None,
        explain: bool = False,
    ) -> list:
        """Search memories using BM25-inspired ranking.
        
        Results are ranked by term relevance × inverse document frequency × decay.
        Scores are normalized to 0.0-1.0 range.
        
        Args:
            query: Search query string
            limit: Maximum results
            category: Filter by category
            min_confidence: Minimum relevance score (0.0-1.0)
            sentiment_filter: Filter by sentiment tag
            explain: If True, return SearchResult objects with explanations.
                     If False (default), return MemoryEntry objects for backward compat.
        
        Returns:
            List of MemoryEntry (default) or SearchResult (if explain=True),
            sorted by relevance descending.
        """
        # Filter by sentiment first if needed
        memories = self.memories
        if sentiment_filter:
            memories = [m for m in memories if sentiment_filter in m.sentiment]

        search_results = self.search_engine.search(
            query=query,
            memories=memories,
            limit=limit * 2,  # Fetch extra candidates for re-ranking
            category=category,
            min_score=min_confidence,
            decay_fn=self.decay.score,
        )

        # ── Tier 1: Apply co-occurrence semantic boost ────────────────────────
        if search_results and self.cooccurrence_boost_weight > 0:
            for r in search_results:
                boost = self.cooccurrence.boost_score(
                    query=query,
                    entry_content=r.entry.content,
                    boost_factor=self.cooccurrence_boost_weight,
                )
                # Inject PPMI boost on top (Tier 3, if available)
                ppmi_boost = self.ppmi.boost_score(
                    query=query,
                    entry_content=r.entry.content,
                    boost_factor=0.15,
                )
                r.relevance = min(r.relevance + boost + ppmi_boost, 1.0)
            # Re-sort after boosting
            search_results.sort(key=lambda r: r.relevance, reverse=True)

        # ── Tier 2: Optional semantic rerank ─────────────────────────────────
        if self.semantic_engine is not None and search_results:
            try:
                search_results = self.semantic_engine.rerank(
                    query=query,
                    candidates=search_results,
                    top_k=limit,
                    cooccurrence_index=self.cooccurrence,
                )
            except Exception:
                # Graceful fallback: use co-occurrence boosted BM25 results
                search_results = search_results[:limit]
        else:
            search_results = search_results[:limit]

        # Reinforce accessed memories
        for r in search_results:
            self.decay.reinforce(r.entry)

        if explain:
            return search_results

        # Backward compat: update entry confidence with search relevance and return entries
        entries = []
        for r in search_results:
            r.entry.confidence = r.relevance
            entries.append(r.entry)
        return entries

    # ── temporal queries ────────────────────────────────────────────────

    def on_date(self, date: str) -> List[MemoryEntry]:
        return self.temporal.on_date(self.memories, date)

    def between(self, start: str, end: str) -> List[MemoryEntry]:
        return self.temporal.between(self.memories, start, end)

    def narrative(self, topic: str = None) -> str:
        return self.temporal.narrative(self.memories, topic)

    # ── forgetting ──────────────────────────────────────────────────────

    def forget(self, topic: str = None, entity: str = None,
               before_date: str = None) -> Dict:
        """Selectively forget memories. Returns audit log."""
        if topic:
            self.memories, gone = self.forgetting.forget_topic(self.memories, topic)
        elif entity:
            self.memories, gone = self.forgetting.forget_entity(self.memories, entity)
        elif before_date:
            self.memories, gone = self.forgetting.forget_before(self.memories, before_date)
        else:
            return {"error": "Specify topic, entity, or before_date"}

        self._hashes = {m.hash for m in self.memories}
        audit = self.forgetting.audit_log(gone)
        self._append_audit(audit)
        return audit

    # ── consolidation ───────────────────────────────────────────────────

    def consolidate(self) -> Dict:
        """Run dream-state consolidation. Returns report."""
        return self.consolidation.run(self.memories)

    def compress_old(self, days: int = 7) -> list:
        """Compress daily files older than *days*."""
        mem_dir = os.path.join(self.workspace, "memory")
        return self.compression.compress_old_files(mem_dir, days)

    # ── synthesis ───────────────────────────────────────────────────────

    def synthesize(self, research_results: Dict = None) -> Dict:
        """Run autonomous knowledge synthesis cycle.
        
        Args:
            research_results: Optional dict of {source: research_info} to integrate
            
        Returns:
            Synthesis report with gaps, suggestions, and integration results
        """
        report = self.synthesis.run_cycle(self.memories, research_results)
        
        # Add any new synthesized entries to our memory store
        if research_results and "new_entries" in report:
            for entry_data in report["new_entries"]:
                entry = MemoryEntry.from_dict(entry_data)
                if entry.hash not in self._hashes:
                    self.memories.append(entry)
                    self._hashes.add(entry.hash)
        
        return report

    def research_suggestions(self, limit: int = 5) -> List[Dict]:
        """Get research topic suggestions based on knowledge gaps.
        
        Args:
            limit: Maximum number of suggestions to return
            
        Returns:
            List of research suggestions with topic, reason, and priority
        """
        return self.synthesis.suggest_research_topics(self.memories, limit)

    # ── stats ───────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        now = datetime.now()
        scores = [self.decay.score(m, now) for m in self.memories]
        sentiments = defaultdict(int)
        for m in self.memories:
            dom = self.sentiment.dominant(m.sentiment)
            if dom:
                sentiments[dom] += 1
        categories = defaultdict(int)
        for m in self.memories:
            categories[m.category] += 1

        return {
            "total": len(self.memories),
            "avg_score": round(sum(scores) / max(len(scores), 1), 4),
            "archive_candidates": sum(1 for s in scores if s < self.decay.archive_threshold),
            "sentiments": dict(sentiments),
            "categories": dict(categories),
            "avg_confidence": round(
                sum(m.confidence for m in self.memories) / max(len(self.memories), 1), 4
            ),
        }

    # ── private ─────────────────────────────────────────────────────────

    def _extract_tags(self, text: str) -> List[str]:
        tags = []
        t = text.lower()
        for term in self._tag_terms:
            if term in t:
                tags.append(term)
        for m in re.findall(r"\$[\d,]+(?:\.\d{2})?", text):
            tags.append(m)
        return tags[:10]

    def _append_audit(self, entry: Dict) -> None:
        log = []
        if os.path.exists(self.audit_path):
            with open(self.audit_path) as f:
                log = json.load(f)
        log.append(entry)
        from .utils import atomic_write_json
        atomic_write_json(self.audit_path, log[-100:])
