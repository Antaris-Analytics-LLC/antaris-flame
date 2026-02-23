"""
Sprint A tests — v4.1 semantic wiring + consent + collection + PPMI.

Covers:
- ConsentStore: opt-in, revoke, persistence, thread safety
- StaticPPMILookup: load, lookup, boost_score, no-op when absent
- CollectionAgent: extract_pairs, build_payload, noise, stub transmit
- MemorySystem: max_memories cap, co-occurrence auto-init, ingest wires cooc,
  search applies co-occurrence boost, new params accepted
"""

import json
import os
import tempfile
import threading
import time

import pytest

from antaris_memory import MemorySystem, ConsentStore, CollectionAgent, StaticPPMILookup
from antaris_memory.cooccurrence import CooccurrenceIndex


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_workspace():
    """Return a fresh temporary directory for each test."""
    return tempfile.mkdtemp()


# ─────────────────────────────────────────────────────────────────────────────
# ConsentStore
# ─────────────────────────────────────────────────────────────────────────────

class TestConsentStore:

    def test_default_state_is_opted_out(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        store.load()
        assert store.is_opted_in() is False

    def test_opt_in_sets_flag(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        store.load()
        store.opt_in()
        assert store.is_opted_in() is True
        assert store.consented_at is not None
        assert store.revoked_at is None

    def test_revoke_clears_flag(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        store.opt_in()
        store.revoke()
        assert store.is_opted_in() is False
        assert store.revoked_at is not None

    def test_persist_and_reload(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        store.opt_in()
        store.save()

        store2 = ConsentStore(ws)
        store2.load()
        assert store2.is_opted_in() is True
        assert store2.consented_at == store.consented_at

    def test_revoke_persists(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        store.opt_in()
        store.revoke()
        store.save()

        store2 = ConsentStore(ws)
        store2.load()
        assert store2.is_opted_in() is False
        assert store2.revoked_at is not None

    def test_load_returns_false_when_no_file(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        result = store.load()
        assert result is False

    def test_load_returns_true_when_file_exists(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        store.opt_in()
        store.save()

        store2 = ConsentStore(ws)
        result = store2.load()
        assert result is True

    def test_status_dict(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        store.opt_in()
        status = store.status()
        assert "semantic_opt_in" in status
        assert status["semantic_opt_in"] is True
        assert "consented_at" in status

    def test_thread_safety(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        errors = []

        def toggle():
            try:
                for _ in range(50):
                    store.opt_in()
                    _ = store.is_opted_in()
                    store.revoke()
                    _ = store.status()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"

    def test_repr(self):
        ws = make_workspace()
        store = ConsentStore(ws)
        store.opt_in()
        assert "ConsentStore" in repr(store)


# ─────────────────────────────────────────────────────────────────────────────
# StaticPPMILookup
# ─────────────────────────────────────────────────────────────────────────────

class TestStaticPPMILookup:

    def test_load_returns_false_when_no_file(self):
        ws = make_workspace()
        ppmi = StaticPPMILookup(ws)
        result = ppmi.load()
        assert result is False
        assert ppmi.is_available() is False

    def test_lookup_returns_empty_when_unavailable(self):
        ws = make_workspace()
        ppmi = StaticPPMILookup(ws)
        ppmi.load()
        result = ppmi.lookup("meeting")
        assert result == {}

    def test_boost_score_returns_zero_when_unavailable(self):
        ws = make_workspace()
        ppmi = StaticPPMILookup(ws)
        ppmi.load()
        score = ppmi.boost_score("meeting reschedule", "the meeting was rescheduled")
        assert score == 0.0

    def test_load_parses_valid_json(self):
        ws = make_workspace()
        data = {"meeting": {"reschedule": 3.2, "zoom": 2.8}}
        with open(os.path.join(ws, "ppmi.json"), "w") as f:
            json.dump(data, f)

        ppmi = StaticPPMILookup(ws)
        result = ppmi.load()
        assert result is True
        assert ppmi.is_available() is True

    def test_lookup_returns_related_words(self):
        ws = make_workspace()
        data = {"meeting": {"reschedule": 3.2, "zoom": 2.8}}
        with open(os.path.join(ws, "ppmi.json"), "w") as f:
            json.dump(data, f)

        ppmi = StaticPPMILookup(ws)
        ppmi.load()
        result = ppmi.lookup("meeting")
        assert "reschedule" in result
        assert result["reschedule"] == pytest.approx(3.2)

    def test_lookup_unknown_word_returns_empty(self):
        ws = make_workspace()
        data = {"meeting": {"reschedule": 3.2}}
        with open(os.path.join(ws, "ppmi.json"), "w") as f:
            json.dump(data, f)

        ppmi = StaticPPMILookup(ws)
        ppmi.load()
        assert ppmi.lookup("unknownword") == {}

    def test_boost_score_positive_when_related(self):
        ws = make_workspace()
        data = {"meeting": {"reschedule": 5.0, "agenda": 4.0}}
        with open(os.path.join(ws, "ppmi.json"), "w") as f:
            json.dump(data, f)

        ppmi = StaticPPMILookup(ws)
        ppmi.load()
        score = ppmi.boost_score("meeting", "the meeting agenda was rescheduled")
        assert score > 0.0
        assert score <= 0.2  # capped at boost_factor

    def test_boost_score_zero_when_unrelated(self):
        ws = make_workspace()
        data = {"meeting": {"reschedule": 5.0}}
        with open(os.path.join(ws, "ppmi.json"), "w") as f:
            json.dump(data, f)

        ppmi = StaticPPMILookup(ws)
        ppmi.load()
        score = ppmi.boost_score("meeting", "unrelated completely different topic")
        assert score == 0.0

    def test_vocab_size(self):
        ws = make_workspace()
        data = {"a": {"b": 1.0}, "c": {"d": 2.0}, "e": {"f": 3.0}}
        with open(os.path.join(ws, "ppmi.json"), "w") as f:
            json.dump(data, f)

        ppmi = StaticPPMILookup(ws)
        ppmi.load()
        assert ppmi.vocab_size == 3

    def test_handles_corrupt_json_gracefully(self):
        ws = make_workspace()
        with open(os.path.join(ws, "ppmi.json"), "w") as f:
            f.write("not valid json {{{{")

        ppmi = StaticPPMILookup(ws)
        result = ppmi.load()
        assert result is False
        assert ppmi.is_available() is False


# ─────────────────────────────────────────────────────────────────────────────
# CollectionAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestCollectionAgent:

    def _make_agent(self, ws, opted_in=True, endpoint=None, interval=0.0):
        cooc = CooccurrenceIndex(ws)
        cooc.load()
        consent = ConsentStore(ws)
        consent.load()
        if opted_in:
            consent.opt_in()
        return CollectionAgent(
            workspace=ws,
            cooccurrence=cooc,
            consent=consent,
            endpoint_url=endpoint,
            interval_seconds=interval,
            min_count=1,
        )

    def test_should_run_false_when_not_opted_in(self):
        ws = make_workspace()
        agent = self._make_agent(ws, opted_in=False)
        assert agent.should_run() is False

    def test_should_run_true_when_opted_in_and_due(self):
        ws = make_workspace()
        agent = self._make_agent(ws, opted_in=True, interval=0.0)
        assert agent.should_run() is True

    def test_should_run_false_when_interval_not_elapsed(self):
        ws = make_workspace()
        agent = self._make_agent(ws, opted_in=True, interval=9999.0)
        # Simulate a recent run so the interval hasn't elapsed
        agent._last_run = time.time()
        assert agent.should_run() is False

    def test_extract_pairs_empty_when_no_data(self):
        ws = make_workspace()
        agent = self._make_agent(ws)
        pairs = agent.extract_pairs()
        assert pairs == []

    def test_extract_pairs_returns_data_after_ingest(self):
        ws = make_workspace()
        cooc = CooccurrenceIndex(ws)
        cooc.load()
        cooc.update_from_memory("meeting reschedule agenda tomorrow project")
        cooc.update_from_memory("meeting agenda confirmed tomorrow")

        consent = ConsentStore(ws)
        consent.opt_in()
        agent = CollectionAgent(
            workspace=ws,
            cooccurrence=cooc,
            consent=consent,
            min_count=1,
            interval_seconds=0.0,
        )
        pairs = agent.extract_pairs()
        assert len(pairs) > 0
        # Each pair is (word_a, word_b, count)
        for a, b, count in pairs:
            assert isinstance(a, str)
            assert isinstance(b, str)
            assert isinstance(count, int)
            assert count >= 1

    def test_noise_applied_to_counts(self):
        ws = make_workspace()
        agent = self._make_agent(ws)
        # Run noise 100 times — at least one should differ from 10
        results = [agent._apply_noise(10) for _ in range(100)]
        assert any(r != 10 for r in results), "noise should modify counts"
        assert all(r >= 1 for r in results), "noised count must be >= 1"

    def test_build_payload_structure(self):
        ws = make_workspace()
        cooc = CooccurrenceIndex(ws)
        cooc.load()
        cooc.update_from_memory("meeting reschedule project deadline planning")
        cooc.update_from_memory("meeting deadline confirmed project")

        consent = ConsentStore(ws)
        consent.opt_in()
        agent = CollectionAgent(
            workspace=ws,
            cooccurrence=cooc,
            consent=consent,
            min_count=1,
            interval_seconds=0.0,
        )
        payload = agent.build_payload()

        assert "version" in payload
        assert "noise_applied" in payload
        assert payload["noise_applied"] is True
        assert "pairs" in payload
        for pair in payload["pairs"]:
            assert "a" in pair
            assert "b" in pair
            assert "count" in pair
            assert isinstance(pair["count"], int)
            assert pair["count"] >= 1

    def test_transmit_stub_returns_true(self):
        ws = make_workspace()
        agent = self._make_agent(ws, endpoint=None)
        result = agent.transmit({"version": "1.1", "pairs": []})
        assert result is True  # Stub always succeeds

    def test_run_cycle_skips_when_not_consented(self):
        ws = make_workspace()
        agent = self._make_agent(ws, opted_in=False, interval=0.0)
        result = agent.run_cycle()
        assert result is False
        assert agent._total_cycles == 0

    def test_run_cycle_runs_when_consented(self):
        ws = make_workspace()
        cooc = CooccurrenceIndex(ws)
        cooc.update_from_memory("meeting project deadline")
        consent = ConsentStore(ws)
        consent.opt_in()
        agent = CollectionAgent(
            workspace=ws,
            cooccurrence=cooc,
            consent=consent,
            min_count=1,
            interval_seconds=0.0,
        )
        result = agent.run_cycle()
        assert result is True

    def test_stats_structure(self):
        ws = make_workspace()
        agent = self._make_agent(ws)
        stats = agent.stats()
        assert "opted_in" in stats
        assert "endpoint_configured" in stats
        assert "total_cycles" in stats
        assert "total_pairs_sent" in stats
        assert "last_error" in stats


# ─────────────────────────────────────────────────────────────────────────────
# MemorySystem v4.1 wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestMemorySystemV41:

    def test_accepts_max_memories_param(self):
        ws = make_workspace()
        mem = MemorySystem(ws, max_memories=10)
        assert mem.max_memories == 10

    def test_accepts_enable_semantic_param(self):
        ws = make_workspace()
        # enable_semantic=False (default) — no sentence-transformers needed
        mem = MemorySystem(ws, enable_semantic=False)
        assert mem.enable_semantic is False

    def test_accepts_cooccurrence_boost_weight_param(self):
        ws = make_workspace()
        mem = MemorySystem(ws, cooccurrence_boost_weight=0.5)
        assert mem.cooccurrence_boost_weight == 0.5

    def test_cooccurrence_auto_init(self):
        ws = make_workspace()
        mem = MemorySystem(ws)
        # CooccurrenceIndex should be initialized automatically
        assert mem._cooccurrence is not None

    def test_consent_store_auto_init(self):
        ws = make_workspace()
        mem = MemorySystem(ws)
        assert mem.consent is not None
        # Default: not opted in
        assert mem.consent.is_opted_in() is False

    def test_collection_agent_auto_init(self):
        ws = make_workspace()
        mem = MemorySystem(ws)
        assert mem._collection_agent is not None

    def test_ppmi_auto_init(self):
        ws = make_workspace()
        mem = MemorySystem(ws)
        assert mem._ppmi is not None
        # No ppmi.json present — not available
        assert mem._ppmi.is_available() is False

    def test_ingest_feeds_cooccurrence(self):
        ws = make_workspace()
        mem = MemorySystem(ws)
        assert mem._cooccurrence.vocab_size == 0
        mem.ingest(
            "The quarterly meeting was rescheduled to next Tuesday at noon",
            source="test",
        )
        assert mem._cooccurrence.vocab_size > 0

    def test_max_memories_cap_enforced(self):
        ws = make_workspace()
        mem = MemorySystem(ws, max_memories=3)
        # Ingest many lines
        lines = [f"This is memory entry number {i} with sufficient length" for i in range(20)]
        mem.ingest("\n".join(lines))
        assert len(mem.memories) <= 3

    def test_max_memories_none_means_unlimited(self):
        ws = make_workspace()
        mem = MemorySystem(ws, max_memories=None)
        lines = [f"This is memory entry number {i} with sufficient length" for i in range(30)]
        mem.ingest("\n".join(lines))
        assert len(mem.memories) == 30

    def test_search_returns_results(self):
        ws = make_workspace()
        mem = MemorySystem(ws)
        mem.ingest(
            "The quarterly meeting was rescheduled to Tuesday afternoon\n"
            "Project deadline confirmed for end of month budget review\n"
            "Invoice payment processed and sent to accounts department"
        )
        results = mem.search("meeting rescheduled")
        assert len(results) > 0

    def test_search_applies_cooccurrence_boost_without_error(self):
        ws = make_workspace()
        mem = MemorySystem(ws, cooccurrence_boost_weight=0.3)
        mem.ingest(
            "The quarterly meeting was rescheduled to Tuesday afternoon\n"
            "Meeting agenda includes budget and project timeline review\n"
            "Project deadline confirmed for end of month invoice review"
        )
        # Run multiple ingest cycles to build co-occurrence data
        for _ in range(3):
            mem.ingest("meeting agenda project timeline budget")

        results = mem.search("meeting agenda")
        assert len(results) > 0
        # Scores should be valid
        for entry in results:
            assert 0.0 <= entry.confidence <= 1.0

    def test_search_zero_boost_weight_still_works(self):
        ws = make_workspace()
        mem = MemorySystem(ws, cooccurrence_boost_weight=0.0)
        mem.ingest(
            "Important decision made about project architecture\n"
            "Project milestone reached and team notified of progress"
        )
        results = mem.search("project architecture")
        assert len(results) > 0

    def test_consent_accessible_on_memory_system(self):
        ws = make_workspace()
        mem = MemorySystem(ws)
        mem.consent.opt_in()
        assert mem.consent.is_opted_in() is True
        mem.consent.revoke()
        assert mem.consent.is_opted_in() is False

    def test_collection_endpoint_param(self):
        ws = make_workspace()
        mem = MemorySystem(ws, collection_endpoint="https://example.supabase.co/rest/v1/pairs")
        assert mem._collection_agent.endpoint_url == "https://example.supabase.co/rest/v1/pairs"

    def test_ppmi_boost_applied_when_file_present(self):
        ws = make_workspace()
        data = {"meeting": {"reschedule": 5.0, "agenda": 4.5}}
        with open(os.path.join(ws, "ppmi.json"), "w") as f:
            json.dump(data, f)

        mem = MemorySystem(ws)
        mem.ingest(
            "The quarterly meeting agenda was confirmed for next week\n"
            "Meeting reschedule notification sent to all participants\n"
            "Budget review and project planning session tomorrow morning"
        )
        results = mem.search("meeting agenda", explain=True)
        assert len(results) > 0
        # At least one result should have non-zero relevance
        assert any(r.relevance > 0 for r in results)

    def test_backward_compat_no_new_params(self):
        """Existing callers with no new params still work."""
        ws = make_workspace()
        mem = MemorySystem(ws)
        mem.ingest("Project meeting confirmed for tomorrow afternoon at three")
        results = mem.search("meeting")
        assert len(results) > 0
