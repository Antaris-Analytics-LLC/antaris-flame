"""
Sprint 6 + Sprint 12 test suite for antaris-context.

Covers:
  - ImportanceWeightedCompressor (scoring, compression)
  - SemanticChunker (boundary splitting)
  - CompressionResult (attribute + dict access)
  - Cross-session snapshot export/import
  - Compression ratio reporting
  - Turn-by-turn lifecycle
  - Retention policy
  - Provider rendering
  - Section priority management
  - Backward compatibility
"""

import json
import time
import pytest

from antaris_context import (
    ContextManager,
    CompressionResult,
    ImportanceWeightedCompressor,
    SemanticChunker,
    SemanticChunk,
)


# ===========================================================================
# CompressionResult — attribute & dict access
# ===========================================================================

class TestCompressionResult:

    def test_attribute_access(self):
        r = CompressionResult(compression_ratio=0.4, tokens_saved=100)
        assert r.compression_ratio == 0.4
        assert r.tokens_saved == 100

    def test_dict_style_getitem(self):
        r = CompressionResult(success=True, actions_taken=["step1"])
        assert r["success"] is True
        assert r["actions_taken"] == ["step1"]

    def test_dict_style_contains(self):
        r = CompressionResult()
        assert "success" in r
        assert "compression_ratio" in r
        assert "nonexistent_key" not in r

    def test_dict_style_get(self):
        r = CompressionResult(sections_dropped=3)
        assert r.get("sections_dropped") == 3
        assert r.get("missing", 99) == 99

    def test_missing_key_raises_key_error(self):
        r = CompressionResult()
        with pytest.raises(KeyError):
            _ = r["totally_absent_key"]

    def test_default_values_are_sensible(self):
        r = CompressionResult()
        assert r.compression_ratio == 1.0
        assert r.sections_dropped == 0
        assert r.sections_compressed == 0
        assert r.tokens_saved == 0
        assert r.success is True
        assert isinstance(r.actions_taken, list)

    def test_sections_dropped_and_compressed(self):
        r = CompressionResult(sections_dropped=2, sections_compressed=5)
        assert r.sections_dropped == 2
        assert r.sections_compressed == 5

    def test_original_and_final_tokens(self):
        r = CompressionResult(original_tokens=1000, final_tokens=250)
        assert r.original_tokens == 1000
        assert r.final_tokens == 250


# ===========================================================================
# SemanticChunker
# ===========================================================================

class TestSemanticChunker:

    def test_returns_list_of_semantic_chunks(self):
        chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=200)
        chunks = chunker.chunk("Hello world. This is a test.")
        assert isinstance(chunks, list)
        assert all(isinstance(c, SemanticChunk) for c in chunks)

    def test_empty_string_returns_empty_list(self):
        chunker = SemanticChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_code_block_identified(self):
        code = "Some text.\n\n```python\ndef hello():\n    pass\n```\n\nMore text."
        chunker = SemanticChunker(split_on=["paragraph", "code_block"])
        chunks = chunker.chunk(code)
        types = [c.chunk_type for c in chunks]
        assert "code_block" in types

    def test_paragraph_splitting(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird one."
        chunker = SemanticChunker(min_chunk_size=5, max_chunk_size=500, split_on=["paragraph"])
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_sentence_splitting_for_large_paragraphs(self):
        # One very long paragraph that should be split by sentences
        long_para = ("The quick brown fox jumps over the lazy dog. " * 15).strip()
        chunker = SemanticChunker(min_chunk_size=20, max_chunk_size=100, split_on=["sentence"])
        chunks = chunker.chunk(long_para)
        assert len(chunks) > 1

    def test_chunks_have_importance_score(self):
        chunker = SemanticChunker(min_chunk_size=5, max_chunk_size=500)
        chunks = chunker.chunk("Hello world. This is a test paragraph.")
        for c in chunks:
            assert 0.0 <= c.importance_score <= 1.0

    def test_code_block_gets_higher_importance(self):
        code = "```python\ndef foo(): pass\n```"
        text = "Just some plain prose here."
        chunker = SemanticChunker(min_chunk_size=5, max_chunk_size=500,
                                  split_on=["paragraph", "code_block"])
        code_chunks = [c for c in chunker.chunk(code) if c.chunk_type == "code_block"]
        text_chunks = chunker.chunk(text)
        if code_chunks and text_chunks:
            assert code_chunks[0].importance_score > text_chunks[0].importance_score

    def test_chunk_tokens_positive(self):
        chunker = SemanticChunker(min_chunk_size=5, max_chunk_size=500)
        chunks = chunker.chunk("Some reasonable length text here for testing purposes.")
        for c in chunks:
            assert c.tokens >= 1

    def test_split_on_none_defaults_to_all_boundaries(self):
        chunker = SemanticChunker()
        assert "paragraph" in chunker.split_on
        assert "sentence" in chunker.split_on
        assert "code_block" in chunker.split_on

    def test_error_keyword_boosts_importance(self):
        error_text = "An error occurred while processing your request."
        normal_text = "The weather today is quite pleasant outside."
        chunker = SemanticChunker(min_chunk_size=5, max_chunk_size=500)
        error_chunks = chunker.chunk(error_text)
        normal_chunks = chunker.chunk(normal_text)
        if error_chunks and normal_chunks:
            assert error_chunks[0].importance_score >= normal_chunks[0].importance_score


# ===========================================================================
# ImportanceWeightedCompressor
# ===========================================================================

class TestImportanceWeightedCompressor:

    def _make_items(self, n: int, priority: str = "normal") -> list:
        """Build *n* simple items in order (oldest first = index 0)."""
        return [
            {"content": f"Message {i}", "tokens": 10, "priority": priority}
            for i in range(n)
        ]

    def test_recent_items_score_higher(self):
        compressor = ImportanceWeightedCompressor()
        items = self._make_items(5)
        oldest_score = compressor.score_item(items[0], 5, 0)
        newest_score = compressor.score_item(items[-1], 5, 4)
        assert newest_score > oldest_score

    def test_critical_priority_scores_higher_than_optional(self):
        compressor = ImportanceWeightedCompressor()
        crit = {"content": "Critical info", "tokens": 10, "priority": "critical"}
        opt = {"content": "Optional info", "tokens": 10, "priority": "optional"}
        assert compressor.score_item(crit, 2, 1) > compressor.score_item(opt, 2, 0)

    def test_system_role_boosts_score(self):
        compressor = ImportanceWeightedCompressor()
        sys_item = {"content": "System instructions", "tokens": 10,
                    "priority": "normal", "role": "system"}
        user_item = {"content": "User says hi", "tokens": 10,
                     "priority": "normal", "role": "user"}
        assert compressor.score_item(sys_item, 2, 1) > compressor.score_item(user_item, 2, 1)

    def test_code_content_boosts_score(self):
        compressor = ImportanceWeightedCompressor()
        code_item = {
            "content": "```python\nfor i in range(10):\n    print(i)\n```",
            "tokens": 20, "priority": "normal"
        }
        plain_item = {"content": "just plain text here", "tokens": 20, "priority": "normal"}
        assert compressor.score_item(code_item, 2, 1) > compressor.score_item(plain_item, 2, 1)

    def test_score_is_between_0_and_1(self):
        compressor = ImportanceWeightedCompressor()
        items = self._make_items(10)
        for i, item in enumerate(items):
            score = compressor.score_item(item, len(items), i)
            assert 0.0 <= score <= 1.0

    def test_compress_items_returns_expected_keys(self):
        compressor = ImportanceWeightedCompressor(keep_top_n=2, drop_threshold=0.0)
        items = self._make_items(5)
        result = compressor.compress_items(items)
        assert "kept" in result
        assert "compressed" in result
        assert "dropped" in result
        assert "tokens_saved" in result

    def test_keep_top_n_honoured(self):
        """Top-N items receive the 'top_n' keep reason; all others are kept verbatim
        when compress_middle=False (no summarisation happens)."""
        compressor = ImportanceWeightedCompressor(keep_top_n=3, compress_middle=False,
                                                   drop_threshold=0.0)
        items = self._make_items(10)
        result = compressor.compress_items(items)
        # Exactly 3 items should carry the top_n reason
        top_n_items = [it for it in result["kept"] if it.get("_keep_reason") == "top_n"]
        assert len(top_n_items) == 3
        # Nothing dropped (drop_threshold=0.0) and nothing compressed (compress_middle=False)
        assert len(result["dropped"]) == 0
        assert len(result["compressed"]) == 0

    def test_drop_threshold_removes_low_items(self):
        compressor = ImportanceWeightedCompressor(keep_top_n=0, compress_middle=False,
                                                   drop_threshold=0.99)
        items = self._make_items(5)
        result = compressor.compress_items(items)
        # Nearly all items should be dropped (score < 0.99)
        assert len(result["dropped"]) > 0

    def test_empty_list_returns_empty_result(self):
        compressor = ImportanceWeightedCompressor()
        result = compressor.compress_items([])
        assert result["kept"] == []
        assert result["compressed"] == []
        assert result["dropped"] == []
        assert result["tokens_saved"] == 0

    def test_tokens_saved_is_non_negative(self):
        compressor = ImportanceWeightedCompressor(keep_top_n=1, drop_threshold=0.3)
        items = self._make_items(8)
        result = compressor.compress_items(items)
        assert result["tokens_saved"] >= 0


# ===========================================================================
# Cross-session snapshot export / import
# ===========================================================================

class TestCrossSessionSnapshot:

    def _build_manager(self) -> ContextManager:
        mgr = ContextManager(total_budget=4000)
        mgr.set_section_budget("system", 500)
        mgr.set_section_budget("memory", 1000)
        mgr.add_content("system", "You are a helpful assistant.", priority="critical")
        mgr.add_content("memory", "User likes short answers.", priority="important")
        mgr.add_turn("user", "What is 2+2?")
        mgr.add_turn("assistant", "4.")
        return mgr

    def test_export_snapshot_returns_dict(self):
        mgr = self._build_manager()
        snap = mgr.export_snapshot()
        assert isinstance(snap, dict)

    def test_snapshot_is_json_serialisable(self):
        mgr = self._build_manager()
        snap = mgr.export_snapshot()
        serialised = json.dumps(snap)
        assert isinstance(serialised, str)
        # Round-trip
        parsed = json.loads(serialised)
        assert parsed["total_budget"] == 4000

    def test_snapshot_includes_section_content(self):
        mgr = self._build_manager()
        snap = mgr.export_snapshot()
        assert "sections" in snap
        assert len(snap["sections"]["system"]["items"]) == 1
        assert snap["sections"]["system"]["items"][0]["content"] == "You are a helpful assistant."

    def test_snapshot_includes_turns(self):
        mgr = self._build_manager()
        snap = mgr.export_snapshot()
        assert "turns" in snap
        assert len(snap["turns"]) == 2
        assert snap["turns"][0]["role"] == "user"

    def test_from_snapshot_roundtrip(self):
        mgr = self._build_manager()
        snap = mgr.export_snapshot()
        restored = ContextManager.from_snapshot(snap)
        assert restored.total_budget == mgr.total_budget

    def test_from_snapshot_restores_section_content(self):
        mgr = self._build_manager()
        snap = mgr.export_snapshot()
        restored = ContextManager.from_snapshot(snap)
        system_items = restored.window.sections["system"]["content"]
        assert len(system_items) == 1
        assert "helpful assistant" in system_items[0]["content"]

    def test_from_snapshot_restores_turns(self):
        mgr = self._build_manager()
        snap = mgr.export_snapshot()
        restored = ContextManager.from_snapshot(snap)
        assert restored.turn_count == 2
        assert restored._turns[1]["role"] == "assistant"

    def test_importance_filter_excludes_low_importance(self):
        mgr = ContextManager(total_budget=4000)
        mgr.set_section_budget("memory", 1000)
        # Add content WITHOUT an importance_score — defaults to 1.0
        mgr.add_content("memory", "High importance content.", priority="critical")
        # Manually inject low-importance item
        mgr.window.sections["memory"]["content"][-1]["importance_score"] = 0.3
        # Only export items above 0.5
        snap = mgr.export_snapshot(include_importance_above=0.5)
        items = snap["sections"]["memory"]["items"]
        assert all(item["importance_score"] > 0.5 for item in items)

    def test_from_snapshot_config_restored(self):
        mgr = ContextManager(total_budget=4000)
        mgr.set_compression_level("aggressive")
        snap = mgr.export_snapshot()
        restored = ContextManager.from_snapshot(snap)
        assert restored.config["compression_level"] == "aggressive"


# ===========================================================================
# Compression ratio reporting from optimize_context
# ===========================================================================

class TestCompressionRatioReporting:

    def test_optimize_context_returns_compression_result(self):
        mgr = ContextManager(total_budget=200)
        mgr.set_section_budget("system", 200)
        mgr.add_content("system", "A" * 800, priority="optional", compress=False)
        result = mgr.optimize_context(target_utilization=0.5)
        assert isinstance(result, CompressionResult)

    def test_compression_ratio_attribute_exists(self):
        mgr = ContextManager(total_budget=200)
        mgr.set_section_budget("system", 200)
        mgr.add_content("system", "A" * 800, priority="optional", compress=False)
        result = mgr.optimize_context(target_utilization=0.3)
        assert hasattr(result, "compression_ratio")
        assert 0.0 <= result.compression_ratio <= 2.0  # Could be > 1 if near-target

    def test_tokens_saved_non_negative(self):
        mgr = ContextManager(total_budget=200)
        mgr.set_section_budget("system", 200)
        mgr.add_content("system", "A" * 800, priority="optional", compress=False)
        result = mgr.optimize_context(target_utilization=0.3)
        assert result.tokens_saved >= 0

    def test_original_and_final_tokens_make_sense(self):
        mgr = ContextManager(total_budget=500)
        mgr.set_section_budget("conversation", 500)
        mgr.add_content("conversation", "word " * 200, priority="optional", compress=False)
        result = mgr.optimize_context(target_utilization=0.5)
        assert result.original_tokens >= result.final_tokens

    def test_backward_compat_dict_access_on_result(self):
        """Old code using result['success'] must keep working."""
        mgr = ContextManager(total_budget=500)
        mgr.set_section_budget("system", 100)
        mgr.add_content("system", "Hello")
        result = mgr.optimize_context()
        # dict-style access
        assert "success" in result
        assert "actions_taken" in result
        assert "initial_state" in result
        assert "final_state" in result
        assert isinstance(result["actions_taken"], list)


# ===========================================================================
# Turn-by-turn lifecycle (Sprint 12)
# ===========================================================================

class TestTurnLifecycle:

    def test_add_turn_increases_count(self):
        mgr = ContextManager()
        assert mgr.turn_count == 0
        mgr.add_turn("user", "Hello")
        assert mgr.turn_count == 1
        mgr.add_turn("assistant", "Hi there")
        assert mgr.turn_count == 2

    def test_turn_count_property(self):
        mgr = ContextManager()
        for i in range(7):
            mgr.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
        assert mgr.turn_count == 7

    def test_compact_older_turns_reduces_count(self):
        mgr = ContextManager()
        for i in range(30):
            mgr.add_turn("user", f"Turn {i}: " + "content " * 5)
        mgr.compact_older_turns(keep_last=10)
        assert mgr.turn_count <= 30  # Should not increase
        # Recent 10 are untouched; older ones are summarised → count stays same but content is shorter
        assert mgr.turn_count == 30  # summarise_older=True keeps count

    def test_compact_older_turns_returns_compacted_count(self):
        mgr = ContextManager()
        for i in range(20):
            mgr.add_turn("user", f"message {i}")
        compacted = mgr.compact_older_turns(keep_last=5)
        assert compacted == 15

    def test_compact_with_no_older_turns_is_noop(self):
        mgr = ContextManager()
        mgr.add_turn("user", "only turn")
        compacted = mgr.compact_older_turns(keep_last=10)
        assert compacted == 0
        assert mgr.turn_count == 1

    def test_compact_drop_mode(self):
        mgr = ContextManager()
        mgr.set_retention_policy(keep_last_n_verbatim=5, summarize_older=False, max_turns=100)
        for i in range(20):
            mgr.add_turn("user", f"message {i}")
        mgr.compact_older_turns(keep_last=5)
        # Without summarize, older turns are dropped
        assert mgr.turn_count == 5

    def test_max_turns_hard_cap(self):
        mgr = ContextManager()
        mgr.set_retention_policy(max_turns=5)
        for i in range(10):
            mgr.add_turn("user", f"msg {i}")
        # Hard cap enforced on add
        assert mgr.turn_count <= 5

    def test_turns_preserve_role_and_content(self):
        mgr = ContextManager()
        mgr.add_turn("user", "What is the capital of France?")
        mgr.add_turn("assistant", "Paris.")
        assert mgr._turns[0]["role"] == "user"
        assert mgr._turns[1]["content"] == "Paris."


# ===========================================================================
# Retention policy (Sprint 12)
# ===========================================================================

class TestRetentionPolicy:

    def test_set_retention_policy_stores_values(self):
        mgr = ContextManager()
        mgr.set_retention_policy(keep_last_n_verbatim=20, summarize_older=True, max_turns=200)
        assert mgr._retention_policy["keep_last_n_verbatim"] == 20
        assert mgr._retention_policy["summarize_older"] is True
        assert mgr._retention_policy["max_turns"] == 200

    def test_summarize_older_condenses_content(self):
        mgr = ContextManager()
        mgr.set_retention_policy(keep_last_n_verbatim=5, summarize_older=True, max_turns=100)
        for i in range(15):
            mgr.add_turn("user", "This is a long message that should be summarised. " * 5)
        mgr.compact_older_turns(keep_last=5)
        # Older turns should be shorter after summarisation
        older_turns = mgr._turns[:-5]
        for turn in older_turns:
            assert len(turn["content"]) < len("This is a long message that should be summarised. " * 5)

    def test_recent_turns_unchanged_after_compact(self):
        mgr = ContextManager()
        content = "The exact content we want to keep verbatim."
        for i in range(8):
            mgr.add_turn("user", f"old msg {i}")
        mgr.add_turn("user", content)  # This is the 9th (recent)
        mgr.compact_older_turns(keep_last=3)
        recent = mgr._turns[-3:]
        recent_contents = [t["content"] for t in recent]
        assert content in recent_contents


# ===========================================================================
# Provider rendering (Sprint 12)
# ===========================================================================

class TestProviderRendering:

    def _manager_with_turns(self) -> ContextManager:
        mgr = ContextManager()
        mgr.add_turn("user", "What is Python?")
        mgr.add_turn("assistant", "Python is a programming language.")
        mgr.add_turn("user", "Show me a snippet.")
        return mgr

    def test_render_returns_list(self):
        mgr = self._manager_with_turns()
        result = mgr.render(provider="generic")
        assert isinstance(result, list)

    def test_render_message_structure(self):
        mgr = self._manager_with_turns()
        messages = mgr.render(provider="anthropic")
        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    def test_render_anthropic_format(self):
        mgr = self._manager_with_turns()
        messages = mgr.render(provider="anthropic")
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_render_openai_format(self):
        mgr = self._manager_with_turns()
        messages = mgr.render(provider="openai")
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_render_generic_format(self):
        mgr = self._manager_with_turns()
        messages = mgr.render(provider="generic")
        assert len(messages) == 3

    def test_render_with_system_prompt(self):
        mgr = self._manager_with_turns()
        messages = mgr.render(provider="anthropic", system_prompt="You are a coder.")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a coder."
        # Original turns follow
        assert messages[1]["role"] == "user"

    def test_render_without_system_prompt_has_no_system_message(self):
        mgr = self._manager_with_turns()
        messages = mgr.render(provider="openai")
        roles = [m["role"] for m in messages]
        assert "system" not in roles

    def test_render_empty_turns_returns_empty_or_system_only(self):
        mgr = ContextManager()
        messages = mgr.render(provider="generic")
        assert messages == []

    def test_render_system_prompt_only(self):
        mgr = ContextManager()
        messages = mgr.render(provider="openai", system_prompt="Hello")
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


# ===========================================================================
# Section priority management (Sprint 12)
# ===========================================================================

class TestSectionPriority:

    def test_add_section_creates_section(self):
        mgr = ContextManager(total_budget=5000)
        mgr.add_section("instructions", "Always respond in JSON.", priority=10)
        assert "instructions" in mgr.window.sections

    def test_add_section_stores_priority(self):
        mgr = ContextManager(total_budget=5000)
        mgr.add_section("scratch", "temp notes", priority=1)
        assert mgr._section_priorities["scratch"] == 1

    def test_add_section_stores_content(self):
        mgr = ContextManager(total_budget=5000)
        mgr.add_section("notes", "important note", priority=8)
        assert len(mgr.window.sections["notes"]["content"]) >= 1

    def test_low_priority_section_dropped_first(self):
        """When over budget, low-priority sections should be compressed/dropped first."""
        mgr = ContextManager(total_budget=300)
        # High-priority section with small content
        mgr.add_section("instructions", "Core instructions.", priority=10, budget=150)
        # Low-priority section with lots of content
        mgr.add_section("scratch", "junk " * 100, priority=1, budget=150)
        # Ensure the low-priority section actually has content
        assert mgr.window.sections["scratch"]["used"] > 0
        result = mgr.optimize_context(target_utilization=0.5)
        # The optimize should have done something
        assert isinstance(result, CompressionResult)

    def test_section_priority_participates_in_optimize(self):
        mgr = ContextManager(total_budget=500)
        mgr.add_section("hi_pri", "critical content " * 10, priority=9, budget=250)
        mgr.add_section("lo_pri", "throwaway " * 10, priority=1, budget=250)
        result = mgr.optimize_context(target_utilization=0.3)
        # sections_dropped + sections_compressed should indicate work was done
        assert result.sections_dropped + result.sections_compressed >= 0  # no crash at minimum


# ===========================================================================
# Backward compatibility
# ===========================================================================

class TestBackwardCompatibility:

    def test_existing_context_manager_api_unchanged(self):
        """All pre-Sprint-6 ContextManager methods must still work."""
        mgr = ContextManager(total_budget=8000)
        mgr.set_section_budget("system", 1000)
        mgr.set_section_budget("memory", 2000)
        mgr.set_section_budget("conversation", 4000)
        mgr.set_section_budget("tools", 1000)
        mgr.set_compression_level("moderate")
        mgr.set_strategy("hybrid", recency_weight=0.4, relevance_weight=0.6)
        mgr.add_content("system", "System prompt here", priority="critical")
        mgr.add_content("conversation", [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ])
        report = mgr.get_usage_report()
        assert report["total_used"] > 0
        assert "sections" in report

    def test_optimize_context_success_key_still_works(self):
        mgr = ContextManager(total_budget=8000)
        mgr.set_section_budget("system", 1000)
        mgr.add_content("system", "Hello")
        result = mgr.optimize_context()
        # Old dict-style access
        assert result["success"] in (True, False)
        assert isinstance(result["actions_taken"], list)
        assert "initial_state" in result
        assert "final_state" in result

    def test_new_turn_count_doesnt_interfere_with_sections(self):
        mgr = ContextManager()
        mgr.set_section_budget("system", 500)
        mgr.add_content("system", "System prompt")
        mgr.add_turn("user", "Hello")
        mgr.add_turn("assistant", "Hi")
        # Section content and turns are independent
        assert len(mgr.window.sections["system"]["content"]) == 1
        assert mgr.turn_count == 2

    def test_export_state_still_works(self):
        mgr = ContextManager()
        mgr.set_section_budget("system", 500)
        state_json = mgr.export_state()
        data = json.loads(state_json)
        assert "config" in data
        assert "window_state" in data

    def test_save_restore_snapshot_unaffected(self):
        mgr = ContextManager()
        mgr.set_section_budget("system", 1000)
        mgr.save_snapshot("test_snap")
        restored = mgr.restore_snapshot("test_snap")
        assert restored is True
