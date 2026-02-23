"""Claim 24: Dream-state consolidation — background memory processing."""

import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

from .entry import MemoryEntry
from .decay import DecayEngine
from .confidence import ConfidenceEngine


class ConsolidationEngine:
    """Background memory processing: duplicates, clusters, insights."""

    def __init__(self, decay: DecayEngine = None):
        self.decay = decay or DecayEngine()

    def find_duplicates(
        self, memories: List[MemoryEntry], threshold: float = 0.7,
        max_comparisons: int = 50000,
    ) -> List[Tuple[MemoryEntry, MemoryEntry]]:
        """Find near-duplicate memories using word overlap.
        
        Optimized: pre-compute word sets, limit comparisons for large sets,
        and use hash-based bucketing for efficiency.
        """
        dupes = []
        
        # Pre-compute word sets
        word_sets = []
        for m in memories:
            ws = set(re.findall(r"\w{4,}", m.content.lower()))
            word_sets.append(ws)

        # For large memory sets, use bucketed comparison
        # (only compare memories that share at least one word)
        if len(memories) > 2000:
            from collections import defaultdict
            buckets = defaultdict(list)
            for idx, ws in enumerate(word_sets):
                for w in list(ws)[:5]:  # Use first 5 words as bucket keys
                    buckets[w].append(idx)
            
            seen_pairs = set()
            comparisons = 0
            for indices in buckets.values():
                if len(indices) < 2:
                    continue
                for i_pos, i in enumerate(indices):
                    for j in indices[i_pos + 1:]:
                        if comparisons >= max_comparisons:
                            return dupes
                        pair = (min(i, j), max(i, j))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        comparisons += 1
                        
                        w1, w2 = word_sets[i], word_sets[j]
                        if not w1 or not w2:
                            continue
                        overlap = len(w1 & w2) / max(len(w1 | w2), 1)
                        if overlap >= threshold:
                            dupes.append((memories[i], memories[j]))
            return dupes

        # Small sets: brute force
        for i, m1 in enumerate(memories):
            w1 = word_sets[i]
            if not w1:
                continue
            for j in range(i + 1, len(memories)):
                w2 = word_sets[j]
                if not w2:
                    continue
                overlap = len(w1 & w2) / max(len(w1 | w2), 1)
                if overlap >= threshold:
                    dupes.append((memories[i], memories[j]))
        return dupes

    def topic_clusters(self, memories: List[MemoryEntry]) -> Dict[str, List[str]]:
        """Group memories by shared tags."""
        clusters: Dict[str, List[str]] = defaultdict(list)
        for m in memories:
            for tag in m.tags:
                clusters[tag.lower()].append(m.hash)
        return {k: v for k, v in clusters.items() if len(v) >= 2}

    def run(self, memories: List[MemoryEntry]) -> Dict:
        """Full consolidation pass — returns a report."""
        now = datetime.now()
        scored = [(m, self.decay.score(m, now)) for m in memories]
        archive = [(m, s) for m, s in scored if self.decay.should_archive(m, now)]
        active = [(m, s) for m, s in scored if not self.decay.should_archive(m, now)]

        dupes = self.find_duplicates(memories)
        clusters = self.topic_clusters(memories)
        contradictions = ConfidenceEngine.find_contradictions(memories)

        return {
            "timestamp": now.isoformat(),
            "total": len(memories),
            "active": len(active),
            "archive_candidates": len(archive),
            "duplicates": len(dupes),
            "clusters": len(clusters),
            "contradictions": len(contradictions),
            "top_clusters": dict(
                sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            ),
            "archive_suggestions": [
                {"hash": m.hash, "preview": m.content[:100], "score": s}
                for m, s in archive[:20]
            ],
            "contradiction_details": [
                {"m1": m1.content[:100], "m2": m2.content[:100], "reason": r}
                for m1, m2, r in contradictions[:10]
            ],
        }
