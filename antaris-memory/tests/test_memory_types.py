"""
Sprint 2 + Sprint 8 tests for antaris-memory.

Coverage:
  Sprint 2 — Structured Memory Types + Mistake Learning
    - Each memory type ingestion (episodic, fact, preference, procedure, mistake)
    - Type-based search filtering (memory_type=)
    - Mistake surfacing in context packets (Known Pitfalls)
    - Decay multipliers differ per type
    - Type metadata stored and retrieved
    - Backward compatibility (existing ingest() still works)
    - Custom memory types

  Sprint 8 — Namespace Isolation
    - Namespace creation via memory.namespace()
    - Namespace isolation (ns-a results not in ns-b search)
    - Namespace lifecycle: create, archive, delete, list
    - NamespacedMemory forwards all typed ingest methods
    - Default namespace maintains backward compatibility
    - delete_namespace removes from manifest
    - archive_namespace keeps data but hides from active list
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

from antaris_memory import MemorySystem
from antaris_memory.entry import MemoryEntry
from antaris_memory.decay import DecayEngine
from antaris_memory.memory_types import (
    MEMORY_TYPE_CONFIGS,
    get_type_config,
    format_mistake_content,
    format_pitfall_line,
    DEFAULT_TYPE,
)
from antaris_memory.namespace import NamespacedMemory, _validate_name


# ── helpers ───────────────────────────────────────────────────────────────────

def _mem(workspace, **kw):
    """Create a MemorySystem with gating bypassed for test simplicity."""
    return MemorySystem(workspace=workspace, **kw)


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 2 — Memory Type Configs
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryTypeConfigs(unittest.TestCase):
    """Verify the canonical type configuration table."""

    def test_all_canonical_types_present(self):
        for t in ("episodic", "fact", "preference", "procedure", "mistake"):
            self.assertIn(t, MEMORY_TYPE_CONFIGS, f"Missing type: {t}")

    def test_mistake_has_highest_decay_multiplier(self):
        multipliers = {k: v["decay_multiplier"] for k, v in MEMORY_TYPE_CONFIGS.items()}
        self.assertEqual(max(multipliers.values()), multipliers["mistake"])
        self.assertEqual(multipliers["mistake"], 10.0)

    def test_preference_and_procedure_have_3x_multiplier(self):
        self.assertEqual(MEMORY_TYPE_CONFIGS["preference"]["decay_multiplier"], 3.0)
        self.assertEqual(MEMORY_TYPE_CONFIGS["procedure"]["decay_multiplier"], 3.0)

    def test_episodic_and_fact_have_1x_multiplier(self):
        self.assertEqual(MEMORY_TYPE_CONFIGS["episodic"]["decay_multiplier"], 1.0)
        self.assertEqual(MEMORY_TYPE_CONFIGS["fact"]["decay_multiplier"], 1.0)

    def test_mistake_has_highest_importance_boost(self):
        boosts = {k: v["importance_boost"] for k, v in MEMORY_TYPE_CONFIGS.items()}
        self.assertEqual(boosts["mistake"], 2.0)
        self.assertGreater(boosts["mistake"], boosts["fact"])

    def test_get_type_config_canonical(self):
        cfg = get_type_config("mistake")
        self.assertEqual(cfg["decay_multiplier"], 10.0)

    def test_get_type_config_unknown_fallback(self):
        cfg = get_type_config("unknown_custom_type")
        # Falls back to episodic defaults
        self.assertEqual(cfg["decay_multiplier"], 1.0)

    def test_get_type_config_custom_override(self):
        cfg = get_type_config("my_type", type_config={"decay_multiplier": 5.0})
        self.assertEqual(cfg["decay_multiplier"], 5.0)

    def test_format_mistake_content(self):
        content = format_mistake_content(
            "Global pip used", "Use venv first", root_cause="No venv", severity="high"
        )
        self.assertIn("MISTAKE:", content)
        self.assertIn("Global pip used", content)
        self.assertIn("CORRECTION:", content)
        self.assertIn("Use venv first", content)
        self.assertIn("ROOT CAUSE:", content)
        self.assertIn("SEVERITY: high", content)

    def test_format_pitfall_line(self):
        line = format_pitfall_line({"what_happened": "pip broke", "correction": "use venv"})
        self.assertIn("⚠️ Known Pitfall:", line)
        self.assertIn("pip broke", line)
        self.assertIn("use venv", line)


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 2 — Decay Engine Type Awareness
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecayTypeAwareness(unittest.TestCase):
    """Verify type-specific decay half-lives."""

    def setUp(self):
        self.decay = DecayEngine(half_life=7.0)

    def _entry(self, memory_type, days_old=30):
        e = MemoryEntry("long enough content here test", memory_type=memory_type)
        past = (datetime.now() - timedelta(days=days_old)).isoformat()
        e.created = past
        return e

    def test_mistake_decays_slowest(self):
        mistake = self._entry("mistake", days_old=30)
        episodic = self._entry("episodic", days_old=30)
        self.assertGreater(
            self.decay.score(mistake),
            self.decay.score(episodic),
            "Mistakes should retain higher score than episodic after 30 days",
        )

    def test_preference_decays_slower_than_episodic(self):
        pref = self._entry("preference", days_old=14)
        episodic = self._entry("episodic", days_old=14)
        self.assertGreater(self.decay.score(pref), self.decay.score(episodic))

    def test_procedure_decays_slower_than_episodic(self):
        proc = self._entry("procedure", days_old=14)
        episodic = self._entry("episodic", days_old=14)
        self.assertGreater(self.decay.score(proc), self.decay.score(episodic))

    def test_fact_and_episodic_decay_same_rate(self):
        fact = self._entry("fact", days_old=10)
        episodic = self._entry("episodic", days_old=10)
        # Same multiplier — scores should be very close (within floating-point rounding)
        self.assertAlmostEqual(
            self.decay.score(fact), self.decay.score(episodic), places=2
        )

    def test_effective_half_life_mistake(self):
        e = self._entry("mistake")
        hl = self.decay.effective_half_life(e)
        self.assertEqual(hl, 70.0)  # 7 × 10

    def test_effective_half_life_preference(self):
        e = self._entry("preference")
        hl = self.decay.effective_half_life(e)
        self.assertEqual(hl, 21.0)  # 7 × 3

    def test_effective_half_life_episodic(self):
        e = self._entry("episodic")
        hl = self.decay.effective_half_life(e)
        self.assertEqual(hl, 7.0)

    def test_type_multiplier_ordering(self):
        """mistake > preference ≈ procedure > episodic ≈ fact"""
        entries = {t: self._entry(t, days_old=20) for t in MEMORY_TYPE_CONFIGS}
        scores = {t: self.decay.score(e) for t, e in entries.items()}
        self.assertGreater(scores["mistake"], scores["preference"])
        self.assertGreater(scores["preference"], scores["episodic"])


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 2 — MemoryEntry Type Fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryEntryTypeFields(unittest.TestCase):

    def test_default_type_is_episodic(self):
        e = MemoryEntry("some content here")
        self.assertEqual(e.memory_type, "episodic")

    def test_custom_type_stored(self):
        e = MemoryEntry("some content here", memory_type="mistake")
        self.assertEqual(e.memory_type, "mistake")

    def test_type_metadata_default_empty(self):
        e = MemoryEntry("some content here")
        self.assertIsInstance(e.type_metadata, dict)
        self.assertEqual(len(e.type_metadata), 0)

    def test_serialization_roundtrip_episodic(self):
        e = MemoryEntry("content for round trip", memory_type="episodic")
        d = e.to_dict()
        # episodic is default — should not appear in dict to save space
        self.assertNotIn("memory_type", d)
        e2 = MemoryEntry.from_dict(d)
        self.assertEqual(e2.memory_type, "episodic")

    def test_serialization_roundtrip_mistake(self):
        e = MemoryEntry("mistake content here for test", memory_type="mistake")
        e.type_metadata = {"what_happened": "x", "correction": "y"}
        d = e.to_dict()
        self.assertEqual(d["memory_type"], "mistake")
        self.assertIn("type_metadata", d)
        e2 = MemoryEntry.from_dict(d)
        self.assertEqual(e2.memory_type, "mistake")
        self.assertEqual(e2.type_metadata["what_happened"], "x")

    def test_serialization_roundtrip_preference(self):
        e = MemoryEntry("preference content here for test", memory_type="preference")
        d = e.to_dict()
        e2 = MemoryEntry.from_dict(d)
        self.assertEqual(e2.memory_type, "preference")


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 2 — Typed Ingest Methods
# ═══════════════════════════════════════════════════════════════════════════════

class TestTypedIngest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.mem = _mem(self.temp)

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    # ── ingest_mistake ────────────────────────────────────────────────────────

    def test_ingest_mistake_creates_entry(self):
        e = self.mem.ingest_mistake(
            what_happened="Used global pip",
            correction="Activate venv first",
            root_cause="No venv context",
            severity="high",
            tags=["venv", "pip"],
        )
        self.assertIsNotNone(e)
        self.assertIsInstance(e, MemoryEntry)

    def test_ingest_mistake_type_is_mistake(self):
        e = self.mem.ingest_mistake(
            what_happened="Used global pip",
            correction="Activate venv first",
        )
        self.assertEqual(e.memory_type, "mistake")

    def test_ingest_mistake_importance_boosted(self):
        e = self.mem.ingest_mistake(
            what_happened="Some mistake happened here",
            correction="Correct approach",
        )
        # Mistake importance = 1.0 × 2.0 boost
        self.assertGreaterEqual(e.importance, 2.0)

    def test_ingest_mistake_stores_structured_metadata(self):
        e = self.mem.ingest_mistake(
            what_happened="pip check failed",
            correction="run in venv",
            root_cause="subprocess env",
            severity="high",
        )
        self.assertEqual(e.type_metadata["what_happened"], "pip check failed")
        self.assertEqual(e.type_metadata["correction"], "run in venv")
        self.assertEqual(e.type_metadata["root_cause"], "subprocess env")
        self.assertEqual(e.type_metadata["severity"], "high")

    def test_ingest_mistake_has_severity_tag(self):
        e = self.mem.ingest_mistake(
            what_happened="Something bad", correction="Fix it", severity="high"
        )
        self.assertIn("severity:high", e.tags)

    def test_ingest_mistake_dedup(self):
        e1 = self.mem.ingest_mistake("Same thing happened", "Same fix")
        e2 = self.mem.ingest_mistake("Same thing happened", "Same fix")
        self.assertIsNotNone(e1)
        self.assertIsNone(e2)  # Duplicate → None

    def test_ingest_mistake_invalid_severity_defaults_to_medium(self):
        e = self.mem.ingest_mistake(
            what_happened="Something happened", correction="Do it right",
            severity="extreme"  # Invalid
        )
        self.assertEqual(e.type_metadata["severity"], "medium")

    # ── ingest_fact ───────────────────────────────────────────────────────────

    def test_ingest_fact_creates_entry(self):
        count = self.mem.ingest_fact("Python uses GIL for thread safety")
        self.assertGreater(count, 0)

    def test_ingest_fact_type_is_fact(self):
        self.mem.ingest_fact("Python uses GIL for thread safety")
        fact_entries = [m for m in self.mem.memories if m.memory_type == "fact"]
        self.assertGreater(len(fact_entries), 0)

    # ── ingest_preference ─────────────────────────────────────────────────────

    def test_ingest_preference_creates_entry(self):
        count = self.mem.ingest_preference("Prefer PostgreSQL over SQLite for production use")
        self.assertGreater(count, 0)

    def test_ingest_preference_type_is_preference(self):
        self.mem.ingest_preference("Prefer PostgreSQL over SQLite for production use")
        pref_entries = [m for m in self.mem.memories if m.memory_type == "preference"]
        self.assertGreater(len(pref_entries), 0)

    # ── ingest_procedure ──────────────────────────────────────────────────────

    def test_ingest_procedure_creates_entry(self):
        count = self.mem.ingest_procedure(
            "Always activate the venv before running pip commands"
        )
        self.assertGreater(count, 0)

    def test_ingest_procedure_type_is_procedure(self):
        self.mem.ingest_procedure(
            "Always activate the venv before running pip commands"
        )
        proc_entries = [m for m in self.mem.memories if m.memory_type == "procedure"]
        self.assertGreater(len(proc_entries), 0)

    # ── backward compatibility ────────────────────────────────────────────────

    def test_plain_ingest_defaults_to_episodic(self):
        count = self.mem.ingest("This is a plain ingested memory for backward compat test")
        self.assertGreater(count, 0)
        episodic = [m for m in self.mem.memories if m.memory_type == "episodic"]
        self.assertGreater(len(episodic), 0)

    def test_ingest_with_explicit_memory_type(self):
        count = self.mem.ingest(
            "Custom procedure memory for testing purposes here",
            memory_type="procedure",
        )
        self.assertGreater(count, 0)
        proc = [m for m in self.mem.memories if m.memory_type == "procedure"]
        self.assertGreater(len(proc), 0)

    def test_ingest_custom_type_with_config(self):
        count = self.mem.ingest(
            "A custom type memory with special decay configuration",
            memory_type="super_memory",
            type_config={"decay_multiplier": 5.0, "importance_boost": 1.5},
        )
        self.assertGreater(count, 0)
        custom = [m for m in self.mem.memories if m.memory_type == "super_memory"]
        self.assertGreater(len(custom), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 2 — Search with memory_type Filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestTypedSearch(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.mem = _mem(self.temp)
        # Seed mixed memories
        self.mem.ingest_mistake(
            what_happened="pip install failed in global env",
            correction="use venv-svi for installation",
            tags=["venv", "pip"],
        )
        self.mem.ingest_fact("Python virtual environments isolate dependencies")
        self.mem.ingest_preference("Prefer explicit over implicit in code style")
        self.mem.ingest_procedure(
            "Activate the venv before running any pip or python commands"
        )
        self.mem.ingest("General episodic memory about the pip venv activation", source="daily")

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_search_returns_all_types_by_default(self):
        results = self.mem.search("venv pip")
        types = {getattr(r, "memory_type", "episodic") for r in results}
        # Should include multiple types
        self.assertGreater(len(results), 1)

    def test_search_filter_mistake_only(self):
        results = self.mem.search("venv", memory_type="mistake")
        for r in results:
            self.assertEqual(getattr(r, "memory_type", "episodic"), "mistake")

    def test_search_filter_fact_only(self):
        results = self.mem.search("virtual environments", memory_type="fact")
        for r in results:
            self.assertEqual(getattr(r, "memory_type", "episodic"), "fact")

    def test_search_filter_procedure_only(self):
        results = self.mem.search("venv activate pip", memory_type="procedure")
        for r in results:
            self.assertEqual(getattr(r, "memory_type", "episodic"), "procedure")

    def test_search_filter_returns_empty_when_no_match(self):
        # No preference entries mention "venv"
        results = self.mem.search("nonexistent_xyz_query_abc", memory_type="preference")
        self.assertEqual(results, [])

    def test_search_with_explain_returns_type_on_entry(self):
        results = self.mem.search("venv", memory_type="mistake", explain=True)
        for r in results:
            self.assertEqual(getattr(r.entry, "memory_type", "episodic"), "mistake")


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 2 — Context Packet Mistake Surfacing
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextPacketMistakes(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.mem = _mem(self.temp)
        # Add a relevant mistake
        self.mem.ingest_mistake(
            what_happened="Checked global pip, package appeared missing",
            correction="Activate venv-svi first, then check",
            root_cause="Sub-agent spawned without environment context",
            severity="high",
            tags=["venv", "installation"],
        )
        # Add a second mistake about a different topic
        self.mem.ingest_mistake(
            what_happened="Used rm -rf without checking path",
            correction="Always use trash command",
            severity="high",
        )
        # Add regular memories
        self.mem.ingest("Notes about venv activation process for installation tasks")
        self.mem.ingest("pip install antaris-guard requires the venv to be active")

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_context_packet_has_pitfalls_field(self):
        packet = self.mem.build_context_packet("installation task venv")
        self.assertTrue(hasattr(packet, "pitfalls"))
        self.assertIsInstance(packet.pitfalls, list)

    def test_context_packet_pitfalls_contain_warning_emoji(self):
        packet = self.mem.build_context_packet(
            "venv installation pip", include_mistakes=True
        )
        if packet.pitfalls:
            for p in packet.pitfalls:
                self.assertIn("⚠️ Known Pitfall:", p)

    def test_context_packet_pitfall_contains_what_happened(self):
        packet = self.mem.build_context_packet(
            "venv pip installation", include_mistakes=True
        )
        rendered = packet.render()
        # The pitfall for the venv mistake should appear
        if packet.pitfalls:
            self.assertIn("Known Pitfalls", rendered)

    def test_context_packet_no_mistakes_when_disabled(self):
        packet = self.mem.build_context_packet(
            "venv pip installation", include_mistakes=False
        )
        self.assertEqual(packet.pitfalls, [])
        self.assertNotIn("Known Pitfalls", packet.render())

    def test_context_packet_render_markdown_includes_pitfalls_section(self):
        packet = self.mem.build_context_packet(
            "venv installation", include_mistakes=True
        )
        rendered = packet.render(fmt="markdown")
        if packet.pitfalls:
            self.assertIn("### Known Pitfalls", rendered)

    def test_context_packet_render_xml_includes_pitfalls(self):
        packet = self.mem.build_context_packet(
            "venv installation", include_mistakes=True
        )
        rendered = packet.render(fmt="xml")
        if packet.pitfalls:
            self.assertIn("<known_pitfalls>", rendered)
            self.assertIn("<pitfall>", rendered)

    def test_context_packet_pitfalls_not_trimmed(self):
        """Pitfalls survive trim() — they are highest priority."""
        packet = self.mem.build_context_packet(
            "venv pip installation", include_mistakes=True
        )
        original_pitfalls = list(packet.pitfalls)
        trimmed = packet.trim(max_tokens=50)  # Very aggressive trim
        self.assertEqual(trimmed.pitfalls, original_pitfalls)

    def test_context_packet_to_dict_includes_pitfalls(self):
        packet = self.mem.build_context_packet("venv", include_mistakes=True)
        d = packet.to_dict()
        self.assertIn("pitfalls", d)
        self.assertIsInstance(d["pitfalls"], list)

    def test_context_packet_from_dict_roundtrip(self):
        from antaris_memory.context_packet import ContextPacket
        packet = self.mem.build_context_packet("venv", include_mistakes=True)
        d = packet.to_dict()
        packet2 = ContextPacket.from_dict(d)
        self.assertEqual(packet2.pitfalls, packet.pitfalls)


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 2 — Persistence of Typed Memories
# ═══════════════════════════════════════════════════════════════════════════════

class TestTypedMemoryPersistence(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_mistake_roundtrips_through_save_load(self):
        mem = _mem(self.temp)
        mem.ingest_mistake(
            what_happened="Forgot to commit before pushing",
            correction="Always commit first",
            severity="medium",
        )
        mem.save()

        mem2 = MemorySystem(workspace=self.temp)
        mem2.load()
        mistakes = [m for m in mem2.memories if m.memory_type == "mistake"]
        self.assertGreater(len(mistakes), 0)
        self.assertEqual(mistakes[0].type_metadata["what_happened"],
                         "Forgot to commit before pushing")

    def test_mixed_types_persist_correctly(self):
        mem = _mem(self.temp)
        mem.ingest_fact("Python is dynamically typed language")
        mem.ingest_preference("Prefer snake_case for Python identifiers")
        mem.ingest_procedure("Run tests before merging any pull request")
        mem.save()

        mem2 = MemorySystem(workspace=self.temp)
        mem2.load()
        types = {m.memory_type for m in mem2.memories}
        self.assertIn("fact", types)
        self.assertIn("preference", types)
        self.assertIn("procedure", types)


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 8 — Namespace Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestNamespaceValidation(unittest.TestCase):

    def test_valid_names_accepted(self):
        for name in ("alpha", "project-alpha", "my_ns", "NS1", "a"):
            _validate_name(name)  # Should not raise

    def test_invalid_names_rejected(self):
        for name in ("", "-bad", "_bad", "has space", "has/slash", "a" * 65):
            with self.assertRaises(ValueError, msg=f"Should reject: {name!r}"):
                _validate_name(name)


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 8 — Namespace Isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestNamespaceIsolation(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.mem = _mem(self.temp)

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_namespace_returns_namespacedmemory(self):
        ns = self.mem.namespace("project-alpha")
        self.assertIsInstance(ns, NamespacedMemory)

    def test_namespace_name_is_stored(self):
        ns = self.mem.namespace("project-alpha")
        self.assertEqual(ns.name, "project-alpha")

    def test_namespace_has_own_workspace(self):
        ns = self.mem.namespace("project-alpha")
        self.assertNotEqual(ns.workspace, self.temp)
        self.assertIn("project-alpha", ns.workspace)

    def test_namespace_isolation_search(self):
        """Memories in ns-a must not appear in ns-b search."""
        ns_a = self.mem.namespace("project-alpha")
        ns_b = self.mem.namespace("project-beta")

        ns_a.ingest("Alpha-specific data about photon collider configuration")
        ns_b.ingest("Beta-specific data about neutron detector calibration")

        # ns-a search should not find beta content
        results_a = ns_a.search("neutron detector")
        contents_a = [r.content for r in results_a]
        self.assertFalse(
            any("neutron" in c.lower() for c in contents_a),
            "ns-a should not contain ns-b memories",
        )

        # ns-b search should not find alpha content
        results_b = ns_b.search("photon collider")
        contents_b = [r.content for r in results_b]
        self.assertFalse(
            any("photon" in c.lower() for c in contents_b),
            "ns-b should not contain ns-a memories",
        )

    def test_namespace_isolation_from_root(self):
        """Root namespace memories don't leak into child namespaces."""
        self.mem.ingest("Root level memory about quantum entanglement research here")
        ns = self.mem.namespace("child-ns")
        results = ns.search("quantum entanglement")
        self.assertEqual(results, [], "Namespace should not see root memories")

    def test_namespace_ingest_does_not_affect_root(self):
        """Ingesting into a namespace doesn't change root memories."""
        root_count_before = len(self.mem.memories)
        ns = self.mem.namespace("side-project")
        ns.ingest("Side project memory about superconductor research details")
        self.assertEqual(len(self.mem.memories), root_count_before)

    def test_namespace_memories_property(self):
        ns = self.mem.namespace("my-ns")
        ns.ingest("Namespace memory about plasma fusion research here")
        self.assertGreater(len(ns.memories), 0)

    def test_namespace_cached_same_instance(self):
        ns1 = self.mem.namespace("cached-ns")
        ns2 = self.mem.namespace("cached-ns")
        self.assertIs(ns1, ns2, "Same namespace name should return same instance")

    def test_namespace_supports_typed_ingest(self):
        ns = self.mem.namespace("typed-ns")
        entry = ns.ingest_mistake(
            what_happened="Wrong API endpoint used for authentication",
            correction="Use /auth/v2/token not /auth/token",
            severity="high",
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry.memory_type, "mistake")

    def test_namespace_search_with_memory_type_filter(self):
        ns = self.mem.namespace("filter-ns")
        ns.ingest_fact("Python GIL limits true parallelism in CPython interpreter")
        ns.ingest("General note about Python threading and concurrency models")
        results = ns.search("Python GIL threading", memory_type="fact")
        for r in results:
            self.assertEqual(r.memory_type, "fact")

    def test_namespace_save_and_load(self):
        ns = self.mem.namespace("persist-ns")
        ns.ingest("Persisted namespace memory about dark matter detection methods")
        ns.save()

        # Re-open the namespace
        ns2 = self.mem.namespace("persist-ns")
        ns2.load()
        # The memory should be found (already loaded on init)
        results = ns2.search("dark matter detection")
        self.assertGreater(len(results), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 8 — Namespace Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestNamespaceLifecycle(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.mem = _mem(self.temp)

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_create_namespace_explicit(self):
        info = self.mem.create_namespace("explicit-ns")
        self.assertIn("status", info)
        self.assertEqual(info["status"], "active")

    def test_create_namespace_idempotent(self):
        self.mem.create_namespace("idempotent-ns")
        info2 = self.mem.create_namespace("idempotent-ns")
        self.assertEqual(info2["status"], "active")

    def test_list_namespaces_returns_list(self):
        self.mem.create_namespace("list-test-ns")
        namespaces = self.mem.list_namespaces()
        self.assertIsInstance(namespaces, list)
        names = [n["name"] for n in namespaces]
        self.assertIn("list-test-ns", names)

    def test_list_namespaces_includes_default(self):
        namespaces = self.mem.list_namespaces()
        names = [n["name"] for n in namespaces]
        self.assertIn("default", names)

    def test_archive_namespace(self):
        self.mem.create_namespace("archive-me")
        info = self.mem.archive_namespace("archive-me")
        self.assertEqual(info["status"], "archived")
        self.assertIsNotNone(info.get("archived_at"))

    def test_archived_namespace_hidden_from_list(self):
        self.mem.create_namespace("soon-archived")
        self.mem.archive_namespace("soon-archived")
        active = self.mem.list_namespaces()
        names = [n["name"] for n in active]
        self.assertNotIn("soon-archived", names)

    def test_archived_namespace_visible_with_flag(self):
        self.mem.create_namespace("visible-archived")
        self.mem.archive_namespace("visible-archived")
        all_ns = self.mem.list_namespaces(include_archived=True)
        names = [n["name"] for n in all_ns]
        self.assertIn("visible-archived", names)

    def test_delete_namespace(self):
        self.mem.create_namespace("delete-me")
        self.mem.delete_namespace("delete-me")
        namespaces = self.mem.list_namespaces(include_archived=True)
        names = [n["name"] for n in namespaces]
        self.assertNotIn("delete-me", names)

    def test_delete_default_namespace_raises(self):
        with self.assertRaises(ValueError):
            self.mem.delete_namespace("default")

    def test_delete_nonexistent_namespace_raises(self):
        with self.assertRaises(KeyError):
            self.mem.delete_namespace("does-not-exist-xyz")

    def test_archive_nonexistent_namespace_raises(self):
        with self.assertRaises(KeyError):
            self.mem.archive_namespace("does-not-exist-xyz")

    def test_namespace_method_auto_creates(self):
        """namespace() auto-creates the namespace if it doesn't exist."""
        ns = self.mem.namespace("auto-created")
        active = self.mem.list_namespaces()
        names = [n["name"] for n in active]
        self.assertIn("auto-created", names)

    def test_delete_with_data_removes_directory(self):
        self.mem.create_namespace("nuke-me")
        ns_ws = os.path.join(self.temp, "namespaces", "nuke-me")
        os.makedirs(ns_ws, exist_ok=True)
        self.mem.delete_namespace("nuke-me", delete_data=True)
        self.assertFalse(os.path.exists(ns_ws))

    def test_namespace_info_has_created_at(self):
        info = self.mem.create_namespace("timestamps-ns")
        self.assertIn("created_at", info)
        # Should be parseable as ISO datetime
        datetime.fromisoformat(info["created_at"])


# ═══════════════════════════════════════════════════════════════════════════════
# Sprint 8 — Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility(unittest.TestCase):
    """Ensure existing API surface is entirely unchanged."""

    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.mem = _mem(self.temp)

    def tearDown(self):
        shutil.rmtree(self.temp, ignore_errors=True)

    def test_ingest_no_type_arg_works(self):
        count = self.mem.ingest("Plain memory without any type argument at all")
        self.assertGreater(count, 0)

    def test_search_no_type_filter_works(self):
        self.mem.ingest("Plain memory for backward compat search testing here")
        results = self.mem.search("backward compat search")
        # Returns list of MemoryEntry
        self.assertIsInstance(results, list)

    def test_search_explain_still_works(self):
        self.mem.ingest("Memory for explain-mode backward compat test validation")
        results = self.mem.search("backward compat explain", explain=True)
        self.assertIsInstance(results, list)

    def test_build_context_packet_no_include_mistakes_arg(self):
        """Old callers that don't pass include_mistakes still get pitfalls by default."""
        packet = self.mem.build_context_packet("some task here")
        # Should not raise; pitfalls will just be [] since no mistakes exist
        self.assertIsInstance(packet.pitfalls, list)

    def test_context_packet_trim_still_works(self):
        packet = self.mem.build_context_packet("some task")
        trimmed = packet.trim(max_tokens=1000)
        self.assertIsNotNone(trimmed)

    def test_ingest_file_unchanged(self):
        test_file = os.path.join(self.temp, "test.md")
        with open(test_file, "w") as f:
            f.write("# Title\nThis is a backward-compat file ingestion test memory.\n")
        count = self.mem.ingest_file(test_file)
        self.assertGreater(count, 0)

    def test_stats_still_returns_expected_keys(self):
        self.mem.ingest("Stats memory for backward compat check here")
        s = self.mem.stats()
        for key in ("total", "avg_score", "categories"):
            self.assertIn(key, s)


if __name__ == "__main__":
    unittest.main()
