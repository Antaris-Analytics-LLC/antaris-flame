"""Tests for context_packet module — sub-agent spawn context injection."""

import json
import os
import shutil
import tempfile
import unittest

from antaris_memory import MemorySystem, ContextPacket, ContextPacketBuilder


class TestContextPacket(unittest.TestCase):
    """Test ContextPacket data structure."""

    def test_empty_packet(self):
        p = ContextPacket(task="test task")
        self.assertEqual(len(p), 0)
        self.assertFalse(bool(p))
        self.assertEqual(p.task, "test task")

    def test_packet_with_memories(self):
        p = ContextPacket(
            task="verify installation",
            memories=[
                {"content": "antaris-guard v1.1.0 installed", "score": 0.9, "source": "log"},
                {"content": "venv-svi is the active environment", "score": 0.7, "source": "config"},
            ],
        )
        self.assertEqual(len(p), 2)
        self.assertTrue(bool(p))

    def test_packet_with_environment(self):
        p = ContextPacket(
            task="test",
            environment={"venv": "venv-svi", "python": "3.11"},
        )
        self.assertTrue(bool(p))

    def test_packet_with_instructions(self):
        p = ContextPacket(
            task="test",
            instructions=["Check venv, not global pip", "Report exact versions"],
        )
        self.assertTrue(bool(p))

    def test_render_markdown(self):
        p = ContextPacket(
            task="verify guard",
            memories=[{"content": "guard is installed in venv-svi", "score": 0.9}],
            environment={"venv": "venv-svi"},
            instructions=["Check the venv"],
            metadata={"built_at": "2026-02-17T00:00:00", "total_memories_searched": 100},
        )
        md = p.render("markdown")
        self.assertIn("## Context Packet", md)
        self.assertIn("venv-svi", md)
        self.assertIn("guard is installed", md)
        self.assertIn("Check the venv", md)
        self.assertIn("100 memories", md)

    def test_render_xml(self):
        p = ContextPacket(
            task="test",
            memories=[{"content": "test memory", "score": 0.5, "source": "log"}],
            environment={"os": "linux"},
        )
        xml = p.render("xml")
        self.assertIn("<context_packet>", xml)
        self.assertIn("<environment>", xml)
        self.assertIn("<memory source=\"log\">test memory</memory>", xml)
        self.assertIn("</context_packet>", xml)

    def test_render_json(self):
        p = ContextPacket(task="test", memories=[{"content": "x", "score": 0.1}])
        j = p.render("json")
        data = json.loads(j)
        self.assertEqual(data["task"], "test")
        self.assertEqual(len(data["memories"]), 1)

    def test_to_dict_from_dict_roundtrip(self):
        p = ContextPacket(
            task="roundtrip test",
            memories=[{"content": "mem1", "score": 0.8}],
            environment={"key": "val"},
            instructions=["do this"],
            metadata={"built_at": "2026-01-01"},
        )
        d = p.to_dict()
        p2 = ContextPacket.from_dict(d)
        self.assertEqual(p.task, p2.task)
        self.assertEqual(p.memories, p2.memories)
        self.assertEqual(p.environment, p2.environment)
        self.assertEqual(p.instructions, p2.instructions)

    def test_estimated_tokens(self):
        p = ContextPacket(task="test", memories=[{"content": "a" * 400, "score": 0.5}])
        tokens = p.estimated_tokens
        self.assertGreater(tokens, 50)  # Non-trivial content

    def test_trim(self):
        memories = [{"content": f"Memory item {i} with lots of padding text to make it bigger " * 5, "score": 1.0 - i * 0.01}
                     for i in range(50)]
        p = ContextPacket(task="big task", memories=memories)
        trimmed = p.trim(500)
        self.assertLessEqual(trimmed.estimated_tokens, 500)
        self.assertLess(len(trimmed), len(p))
        self.assertIn("trimmed_to", trimmed.metadata)

    def test_trim_no_op_when_small(self):
        p = ContextPacket(task="small", memories=[{"content": "tiny", "score": 1.0}])
        trimmed = p.trim(10000)
        self.assertEqual(len(trimmed), len(p))

    def test_repr(self):
        p = ContextPacket(task="a task", memories=[{"content": "m", "score": 0.5}])
        r = repr(p)
        self.assertIn("ContextPacket", r)
        self.assertIn("memories=1", r)


    def test_render_xml_escaping(self):
        p = ContextPacket(
            task="test",
            memories=[{"content": "<script>alert('xss')</script> & other stuff", "score": 0.5, "source": "user<input>"}],
            environment={"path": "/usr/bin & /home"},
            instructions=["Handle <angle> brackets"],
        )
        xml = p.render("xml")
        self.assertNotIn("<script>", xml)
        self.assertIn("&lt;script&gt;", xml)
        self.assertIn("&amp; other", xml)
        self.assertIn("&lt;angle&gt;", xml)

    def test_min_relevance_percentage_error(self):
        """Passing min_relevance > 1.0 should raise ValueError."""
        from antaris_memory.context_packet import ContextPacketBuilder
        # Can't test fully without a MemorySystem, but we test the packet builder validation
        # indirectly via the MemorySystem method in the builder tests below

    def test_render_xml_environment_key_sanitation(self):
        """Environment keys with spaces/digits should produce valid XML."""
        p = ContextPacket(
            task="test",
            environment={"python version": "3.11", "3.11": "active", "venv-svi": "/path"},
        )
        xml = p.render("xml")
        # Should use <var name="..."> not dynamic tag names
        self.assertIn('name="python version"', xml)
        self.assertIn('name="3.11"', xml)
        self.assertNotIn("<python version>", xml)
        self.assertNotIn("<3.11>", xml)

    def test_render_default_is_markdown(self):
        p = ContextPacket(task="test", memories=[{"content": "m", "score": 0.5}])
        self.assertEqual(p.render(), p.render("markdown"))


class TestContextPacketBuilder(unittest.TestCase):
    """Test building context packets from a MemorySystem."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem = MemorySystem(self.tmpdir)
        # Ingest test memories
        self.mem.ingest("antaris-guard v1.1.0 was installed in venv-svi", source="install-log", category="installation")
        self.mem.ingest("antaris-memory v1.0.1 is the latest version", source="release-notes", category="release")
        self.mem.ingest("venv-svi contains all antaris packages", source="env-config", category="configuration")
        self.mem.ingest("Python 3.11 is used in the project", source="setup", category="configuration")
        self.mem.ingest("BM25 search returns ranked results", source="docs", category="technical")
        self.mem.ingest("The router handles model selection", source="docs", category="technical")
        self.mem.ingest("Guard blocks prompt injection attempts", source="docs", category="security")
        self.mem.ingest("Context packets solve the cold spawn problem", source="design", category="architecture")
        self.mem.save()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_build_basic(self):
        packet = self.mem.build_context_packet(task="verify antaris-guard installation")
        self.assertIsInstance(packet, ContextPacket)
        self.assertGreater(len(packet), 0)
        self.assertEqual(packet.task, "verify antaris-guard installation")

    def test_build_with_environment(self):
        packet = self.mem.build_context_packet(
            task="check guard status",
            environment={"venv": "venv-svi", "python": "3.11"},
        )
        self.assertEqual(packet.environment["venv"], "venv-svi")
        md = packet.render()
        self.assertIn("venv-svi", md)

    def test_build_with_instructions(self):
        packet = self.mem.build_context_packet(
            task="audit guard",
            instructions=["Check the venv, not global pip", "Report version numbers"],
        )
        self.assertEqual(len(packet.instructions), 2)

    def test_build_with_category_filter(self):
        packet = self.mem.build_context_packet(
            task="installation status",
            category="installation",
        )
        for mem in packet.memories:
            self.assertEqual(mem.get("category"), "installation")

    def test_build_respects_max_memories(self):
        packet = self.mem.build_context_packet(task="antaris packages", max_memories=2)
        self.assertLessEqual(len(packet), 2)

    def test_build_respects_max_tokens(self):
        packet = self.mem.build_context_packet(task="all packages", max_tokens=500)
        self.assertLessEqual(packet.estimated_tokens, 500)

    def test_build_metadata(self):
        packet = self.mem.build_context_packet(task="guard check")
        self.assertIn("built_at", packet.metadata)
        self.assertIn("total_memories_searched", packet.metadata)
        self.assertIn("results_included", packet.metadata)

    def test_build_multi_queries(self):
        packet = self.mem.build_context_packet_multi(
            task="full environment audit",
            queries=["antaris-guard installation", "Python version", "venv configuration"],
        )
        self.assertIsInstance(packet, ContextPacket)
        self.assertGreater(len(packet), 0)
        self.assertIn("queries", packet.metadata)

    def test_build_multi_deduplicates(self):
        # Same query twice should not duplicate results
        packet = self.mem.build_context_packet_multi(
            task="dedup test",
            queries=["antaris-guard", "antaris-guard"],
        )
        hashes = set()
        for mem in packet.memories:
            content = mem["content"]
            self.assertNotIn(content, hashes)
            hashes.add(content)

    def test_render_formats(self):
        packet = self.mem.build_context_packet(task="guard install check")
        md = packet.render("markdown")
        xml = packet.render("xml")
        j = packet.render("json")
        self.assertIn("## Context Packet", md)
        self.assertIn("<context_packet>", xml)
        json.loads(j)  # Should not raise

    def test_roundtrip_serialization(self):
        packet = self.mem.build_context_packet(
            task="serialize test",
            environment={"host": "mac-mini"},
        )
        d = packet.to_dict()
        restored = ContextPacket.from_dict(d)
        self.assertEqual(packet.task, restored.task)
        self.assertEqual(len(packet.memories), len(restored.memories))

    def test_standalone_builder(self):
        builder = ContextPacketBuilder(self.mem)
        packet = builder.build(task="guard check")
        self.assertIsInstance(packet, ContextPacket)

    def test_empty_search_returns_empty_packet(self):
        packet = self.mem.build_context_packet(task="zzzznonexistentzzzzz")
        self.assertEqual(len(packet), 0)

    def test_empty_memory_system(self):
        """Building a packet from an empty memory system should work."""
        empty_dir = tempfile.mkdtemp()
        try:
            empty_mem = MemorySystem(empty_dir)
            packet = empty_mem.build_context_packet(task="anything")
            self.assertEqual(len(packet), 0)
            self.assertFalse(bool(packet))
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_build_multi_with_full_params(self):
        packet = self.mem.build_context_packet_multi(
            task="full audit",
            queries=["antaris-guard", "Python version"],
            environment={"host": "mac-mini", "venv": "venv-svi"},
            instructions=["Check venv first", "Report versions"],
            max_memories=5,
            max_tokens=2000,
            min_relevance=0.05,
        )
        self.assertIsInstance(packet, ContextPacket)
        self.assertEqual(packet.environment["host"], "mac-mini")
        self.assertEqual(len(packet.instructions), 2)
        self.assertLessEqual(len(packet), 5)

    def test_build_multi_stable_dedup(self):
        """Identical content should always deduplicate regardless of process."""
        packet1 = self.mem.build_context_packet_multi(
            task="guard test",
            queries=["antaris-guard", "antaris-guard", "antaris-guard"],
        )
        packet2 = self.mem.build_context_packet_multi(
            task="guard test",
            queries=["antaris-guard", "antaris-guard", "antaris-guard"],
        )
        # Same content set both times (deterministic dedup)
        contents1 = [m["content"] for m in packet1.memories]
        contents2 = [m["content"] for m in packet2.memories]
        self.assertEqual(contents1, contents2)

    def test_min_relevance_validation(self):
        with self.assertRaises(ValueError):
            self.mem.build_context_packet(task="test", min_relevance=50)

    def test_min_relevance_filter(self):
        packet = self.mem.build_context_packet(
            task="antaris-guard installation verification",
            min_relevance=0.5,
        )
        # Should have fewer results than with low threshold
        packet_loose = self.mem.build_context_packet(
            task="antaris-guard installation verification",
            min_relevance=0.01,
        )
        self.assertLessEqual(len(packet), len(packet_loose))


if __name__ == "__main__":
    unittest.main()
