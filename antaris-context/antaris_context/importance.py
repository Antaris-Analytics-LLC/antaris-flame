"""
Sprint 6: Importance-weighted compression, semantic chunking, and compression result reporting.

Zero-dependency, pure-Python implementation.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SemanticChunk:
    """A semantically bounded chunk of text with metadata."""
    content: str
    chunk_type: str          # 'paragraph', 'sentence', 'code_block', 'heading'
    importance_score: float = 0.0
    tokens: int = 0


class CompressionResult:
    """Result of a context optimization / compression operation.

    Supports both attribute access (``result.compression_ratio``) and
    dict-style access (``result['success']``) so that existing code that
    treats ``optimize_context()`` as returning a plain dict still works.
    """

    def __init__(
        self,
        *,
        compression_ratio: float = 1.0,
        sections_dropped: int = 0,
        sections_compressed: int = 0,
        tokens_saved: int = 0,
        original_tokens: int = 0,
        final_tokens: int = 0,
        actions_taken: Optional[List[str]] = None,
        success: bool = True,
        initial_state: Optional[Dict] = None,
        final_state: Optional[Dict] = None,
    ) -> None:
        self.compression_ratio = compression_ratio
        self.sections_dropped = sections_dropped
        self.sections_compressed = sections_compressed
        self.tokens_saved = tokens_saved
        self.original_tokens = original_tokens
        self.final_tokens = final_tokens
        self.actions_taken: List[str] = actions_taken if actions_taken is not None else []
        self.success = success
        self.initial_state: Dict = initial_state if initial_state is not None else {}
        self.final_state: Dict = final_state if final_state is not None else {}

    # ------------------------------------------------------------------
    # Backward-compat dict interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CompressionResult(ratio={self.compression_ratio:.3f}, "
            f"tokens_saved={self.tokens_saved}, "
            f"sections_dropped={self.sections_dropped}, "
            f"sections_compressed={self.sections_compressed})"
        )


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """Split content at semantic boundaries before compressing.

    Splitting is purely rule-based (regex), so no ML/NLP libraries are
    required.

    Args:
        min_chunk_size: Minimum chunk size in characters.
        max_chunk_size: Maximum chunk size in characters before a paragraph
            is further split at sentence boundaries.
        split_on: Ordered list of boundary types to honour.
            Supported values: ``"paragraph"``, ``"sentence"``,
            ``"code_block"``.
    """

    # Fenced code-block regex (``` or ~~~, optionally with language tag)
    _CODE_FENCE_RE = re.compile(r'(```[\s\S]*?```|~~~[\s\S]*?~~~)', re.MULTILINE)
    # Sentence-ending punctuation followed by whitespace
    _SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')
    # Common abbreviations that contain a period but are NOT sentence endings
    _ABBREV_RE = re.compile(
        r'\b(Mr|Mrs|Ms|Dr|Prof|etc|vs|Inc|Ltd|St|Ave|Fig|No)\.',
        re.IGNORECASE,
    )

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 500,
        split_on: Optional[List[str]] = None,
    ) -> None:
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.split_on: List[str] = split_on if split_on is not None else [
            "paragraph", "sentence", "code_block"
        ]

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split *text* into semantic chunks.

        Args:
            text: Input text.

        Returns:
            List of :class:`SemanticChunk` objects (never empty for non-empty
            input — at minimum returns the whole text as a single chunk).
        """
        if not text or not text.strip():
            return []

        # Step 1 – isolate fenced code blocks
        if "code_block" in self.split_on:
            segments = self._split_code_blocks(text)
        else:
            segments = [("text", text)]

        final_chunks: List[SemanticChunk] = []

        for seg_type, seg_content in segments:
            if not seg_content.strip():
                continue

            if seg_type == "code_block":
                chunk = SemanticChunk(
                    content=seg_content,
                    chunk_type="code_block",
                    importance_score=self._score("code_block", seg_content),
                    tokens=max(1, len(seg_content) // 4),
                )
                final_chunks.append(chunk)
                continue

            # Step 2 – split non-code text into paragraphs
            if "paragraph" in self.split_on:
                paragraphs = self._split_paragraphs(seg_content)
            else:
                paragraphs = [seg_content]

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Step 3 – split large paragraphs into sentences
                if len(para) > self.max_chunk_size and "sentence" in self.split_on:
                    sentences = self._split_sentences(para)
                    buf = ""
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        candidate = (buf + " " + sent).strip() if buf else sent
                        if len(candidate) >= self.min_chunk_size:
                            final_chunks.append(SemanticChunk(
                                content=candidate,
                                chunk_type="sentence",
                                importance_score=self._score("sentence", candidate),
                                tokens=max(1, len(candidate) // 4),
                            ))
                            buf = ""
                        else:
                            buf = candidate
                    # Flush leftover buffer
                    if buf:
                        final_chunks.append(SemanticChunk(
                            content=buf,
                            chunk_type="sentence",
                            importance_score=self._score("sentence", buf),
                            tokens=max(1, len(buf) // 4),
                        ))
                else:
                    final_chunks.append(SemanticChunk(
                        content=para,
                        chunk_type="paragraph",
                        importance_score=self._score("paragraph", para),
                        tokens=max(1, len(para) // 4),
                    ))

        # Fallback: never return an empty list for non-empty input
        if not final_chunks:
            final_chunks.append(SemanticChunk(
                content=text.strip(),
                chunk_type="paragraph",
                importance_score=self._score("paragraph", text),
                tokens=max(1, len(text) // 4),
            ))

        return final_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_code_blocks(self, text: str) -> List[tuple]:
        """Separate fenced code blocks from surrounding prose."""
        segments: List[tuple] = []
        last_end = 0
        for match in self._CODE_FENCE_RE.finditer(text):
            before = text[last_end:match.start()]
            if before.strip():
                segments.append(("text", before))
            segments.append(("code_block", match.group()))
            last_end = match.end()
        remainder = text[last_end:]
        if remainder.strip():
            segments.append(("text", remainder))
        return segments

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split on blank lines (two or more newlines)."""
        return re.split(r'\n\s*\n', text)

    def _split_sentences(self, text: str) -> List[str]:
        """Split on sentence boundaries, respecting common abbreviations."""
        # Temporarily hide abbreviation dots
        protected = self._ABBREV_RE.sub(lambda m: m.group().replace('.', '<DOT>'), text)
        sentences = self._SENTENCE_END_RE.split(protected)
        return [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]

    @staticmethod
    def _score(chunk_type: str, content: str) -> float:
        """Heuristic importance score for a chunk (0.0–1.0)."""
        score = 0.5

        if chunk_type == "code_block":
            score += 0.3

        # Bullet lists / structured content
        if re.search(r'^\s*[-*•]\s', content, re.MULTILINE):
            score += 0.1

        # Numbers / data
        if re.search(r'\d+', content):
            score += 0.05

        # Error / warning keywords
        if re.search(r'\b(error|warning|exception|critical|fail|important)\b',
                     content, re.IGNORECASE):
            score += 0.15

        # Headings
        if re.match(r'^#{1,6}\s', content):
            score += 0.1

        # Very short chunks are less informative
        if len(content) < 50:
            score -= 0.15

        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Importance-weighted compressor
# ---------------------------------------------------------------------------

class ImportanceWeightedCompressor:
    """Compress a list of context items using importance scoring.

    Each item is scored on several axes (recency, priority, message type,
    content density).  The top-N items by importance are kept verbatim;
    middle-importance items are optionally summarised via
    :class:`SemanticChunker`; low-importance items below
    *drop_threshold* are discarded.

    Args:
        keep_top_n: Always keep this many highest-scored items verbatim.
        compress_middle: Whether to compress (rather than keep verbatim)
            items that rank below *keep_top_n* but above *drop_threshold*.
        drop_threshold: Items with an importance score below this value are
            dropped entirely.
    """

    def __init__(
        self,
        keep_top_n: int = 5,
        compress_middle: bool = True,
        drop_threshold: float = 0.1,
    ) -> None:
        self.keep_top_n = keep_top_n
        self.compress_middle = compress_middle
        self.drop_threshold = drop_threshold
        self._chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=300)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_item(self, item: Dict, total_items: int, item_index: int) -> float:
        """Compute an importance score in [0, 1] for a single item.

        Factors (weighted):

        * **Recency** (25 %) — normalised position in the list.
        * **Priority label** (25 %) — 'critical' > 'important' > 'normal' > 'optional'.
        * **Explicit importance_score** (15 %) — pass-through if already set.
        * **Message role** (15 %) — system / error messages are more important.
        * **Content density** (20 %) — code blocks, structured data, error keywords.

        Args:
            item: Dict with at minimum a ``'content'`` key.
            total_items: Total number of items being scored together.
            item_index: 0-based position of *item* in the original list
                (0 = oldest, ``total_items-1`` = newest).

        Returns:
            Float in [0.0, 1.0].
        """
        score = 0.0

        # 1. Recency
        if total_items > 1:
            recency = item_index / (total_items - 1)
        else:
            recency = 1.0
        score += recency * 0.25

        # 2. String priority label
        _priority_scores = {
            "critical": 1.0,
            "important": 0.75,
            "normal": 0.5,
            "optional": 0.25,
        }
        priority_str = item.get("priority", "normal")
        score += _priority_scores.get(str(priority_str), 0.5) * 0.25

        # 3. Explicit importance_score already attached to item
        if "importance_score" in item:
            score += float(item["importance_score"]) * 0.15

        # 4. Message role
        role = item.get("role", "")
        if role == "system":
            score += 0.15
        elif role in ("error", "tool"):
            score += 0.075

        # 5. Content density
        content: str = item.get("content", "") or ""
        if content:
            if re.search(r'```[\s\S]*?```', content) or re.search(r'^    \S', content, re.MULTILINE):
                score += 0.1
            if re.search(r'^\s*[-*]\s|\{.*\}|\[.*\]', content, re.MULTILINE | re.DOTALL):
                score += 0.05
            if re.search(r'\b(error|exception|warning|critical|fail)\b', content, re.IGNORECASE):
                score += 0.05

        return max(0.0, min(1.0, score))

    def compress_items(self, items: List[Dict]) -> Dict:
        """Apply importance-weighted retention to *items*.

        Args:
            items: List of content item dicts (as stored in
                ``ContextWindow.sections[x]['content']``).

        Returns:
            Dict with keys:

            * ``'kept'`` – items kept verbatim (highest importance).
            * ``'compressed'`` – items kept but content condensed.
            * ``'dropped'`` – items removed entirely.
            * ``'tokens_saved'`` – total tokens reclaimed.
        """
        if not items:
            return {"kept": [], "compressed": [], "dropped": [], "tokens_saved": 0}

        total = len(items)
        scored = [
            (self.score_item(item, total, idx), idx, item)
            for idx, item in enumerate(items)
        ]
        # Sort by score descending for top-N selection
        scored_by_rank = sorted(scored, key=lambda x: x[0], reverse=True)

        kept: List[Dict] = []
        compressed: List[Dict] = []
        dropped: List[Dict] = []
        tokens_saved = 0

        for rank, (score, _original_idx, item) in enumerate(scored_by_rank):
            if score < self.drop_threshold:
                dropped.append(item)
                tokens_saved += item.get("tokens", 0)
            elif rank < self.keep_top_n:
                copy = item.copy()
                copy["importance_score"] = score
                copy["_keep_reason"] = "top_n"
                kept.append(copy)
            elif self.compress_middle:
                copy = item.copy()
                copy["importance_score"] = score
                copy["_keep_reason"] = "compressed"
                content: str = item.get("content", "") or ""
                if len(content) > 200:
                    copy["content"], saved = self._condense(content, item.get("tokens", 0))
                    copy["tokens"] = max(1, len(copy["content"]) // 4)
                    tokens_saved += saved
                compressed.append(copy)
            else:
                copy = item.copy()
                copy["importance_score"] = score
                kept.append(copy)

        return {
            "kept": kept,
            "compressed": compressed,
            "dropped": dropped,
            "tokens_saved": tokens_saved,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _condense(self, content: str, original_tokens: int) -> tuple:
        """Return (condensed_content, tokens_saved)."""
        chunks = self._chunker.chunk(content)
        # Keep top-3 chunks by importance score
        high_importance = sorted(chunks, key=lambda c: c.importance_score, reverse=True)[:3]
        # Re-sort to preserve original reading order (by position in text)
        ordered = sorted(high_importance, key=lambda c: content.find(c.content))
        condensed = " ".join(c.content for c in ordered)
        new_tokens = max(1, len(condensed) // 4)
        tokens_saved = max(0, original_tokens - new_tokens)
        return condensed, tokens_saved
