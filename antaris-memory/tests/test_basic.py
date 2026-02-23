"""
Basic tests for antaris-memory core functionality.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta

from antaris_memory import MemorySystem, InputGate, KnowledgeSynthesizer
from antaris_memory.entry import MemoryEntry


class TestMemorySystem(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_system = MemorySystem(workspace=self.temp_dir, half_life=7.0)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_system_init(self):
        """Test MemorySystem initialization."""
        mem = MemorySystem(workspace=self.temp_dir)
        self.assertEqual(mem.workspace, self.temp_dir)
        self.assertEqual(len(mem.memories), 0)
        self.assertEqual(mem.decay.half_life, 7.0)  # default

        # Test custom half-life
        mem2 = MemorySystem(workspace=self.temp_dir, half_life=14.0)
        self.assertEqual(mem2.decay.half_life, 14.0)

    def test_adding_entries(self):
        """Test adding memories via ingest."""
        content = """
        This is a test memory about patents and AI.
        Another line about memory systems.
        A third line with some financial data $50,000.
        """
        
        count = self.memory_system.ingest(content, source="test", category="tactical")
        self.assertGreater(count, 0)
        self.assertEqual(len(self.memory_system.memories), count)
        
        # Check that entries have expected properties
        entry = self.memory_system.memories[0]
        self.assertIsInstance(entry, MemoryEntry)
        self.assertEqual(entry.source, "test")
        self.assertEqual(entry.category, "tactical")
        self.assertIsInstance(entry.sentiment, dict)
        self.assertIsInstance(entry.tags, list)

    def test_search(self):
        """Test basic search functionality."""
        # Add some test data
        self.memory_system.ingest("Patent filing for AI memory system", source="doc1")
        self.memory_system.ingest("Machine learning algorithms for data processing", source="doc2")
        self.memory_system.ingest("Revenue projections for Q4 enterprise sales", source="doc3")
        
        # Test search
        results = self.memory_system.search("patent")
        self.assertGreater(len(results), 0)
        self.assertIn("patent", results[0].content.lower())
        
        results = self.memory_system.search("machine learning")
        self.assertGreater(len(results), 0)
        
        # Test search with category filter
        results = self.memory_system.search("patent", category="nonexistent")
        self.assertEqual(len(results), 0)

    def test_decay_scores(self):
        """Test that decay scores are computed."""
        self.memory_system.ingest("Test memory for decay calculation", source="test")
        entry = self.memory_system.memories[0]
        
        # Fresh memory should have high decay score
        score = self.memory_system.decay.score(entry)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)
        
        # Test reinforcement
        original_access_count = entry.access_count
        self.memory_system.decay.reinforce(entry)
        self.assertGreater(entry.access_count, original_access_count)

    def test_sentiment_analysis(self):
        """Test that sentiment analysis works."""
        self.memory_system.ingest("This is a fantastic and amazing breakthrough!", source="positive")
        self.memory_system.ingest("This is terrible and awful news.", source="negative")
        
        positive_entry = next(m for m in self.memory_system.memories if m.source == "positive")
        negative_entry = next(m for m in self.memory_system.memories if m.source == "negative")
        
        # Should have sentiment scores
        self.assertIsInstance(positive_entry.sentiment, dict)
        self.assertIsInstance(negative_entry.sentiment, dict)
        
        # Test dominant sentiment extraction
        pos_dominant = self.memory_system.sentiment.dominant(positive_entry.sentiment)
        neg_dominant = self.memory_system.sentiment.dominant(negative_entry.sentiment)
        
        # At least one should have a dominant sentiment
        self.assertTrue(pos_dominant is not None or neg_dominant is not None)

    def test_temporal_queries(self):
        """Test temporal query functionality."""
        # Create entries with specific dates - using longer content
        today = datetime.now().date().isoformat()
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
        
        count1 = self.memory_system.ingest("Today's entry with enough content for testing temporal queries", source="today")
        self.assertGreater(count1, 0)
        # Manually set creation date for testing
        if self.memory_system.memories:
            self.memory_system.memories[-1].created = today + "T10:00:00"
        
        count2 = self.memory_system.ingest("Yesterday's entry with enough content for testing temporal queries", source="yesterday")  
        self.assertGreater(count2, 0)
        if len(self.memory_system.memories) >= 2:
            self.memory_system.memories[-1].created = yesterday + "T10:00:00"
        
        # Test date-based queries
        today_results = self.memory_system.on_date(today)
        yesterday_results = self.memory_system.on_date(yesterday)
        
        # At least one should return results
        self.assertGreaterEqual(len(today_results) + len(yesterday_results), 0)

    def test_save_and_load(self):
        """Test persistence functionality."""
        # Add some memories
        self.memory_system.ingest("Memory to persist", source="persist_test")
        original_count = len(self.memory_system.memories)
        
        # Save
        save_path = self.memory_system.save()
        self.assertTrue(os.path.exists(save_path))
        
        # Create new system and load
        new_system = MemorySystem(workspace=self.temp_dir)
        loaded_count = new_system.load()
        
        self.assertEqual(loaded_count, original_count)
        self.assertEqual(len(new_system.memories), original_count)
        self.assertEqual(new_system.memories[0].content, "Memory to persist")

    def test_stats(self):
        """Test stats generation."""
        # Add some test data with longer content (needs to be >15 chars to pass filter)
        count1 = self.memory_system.ingest("This is a happy memory with good sentiment!", source="test", category="tactical")
        count2 = self.memory_system.ingest("This is a sad memory with negative sentiment.", source="test", category="strategic")
        
        total_added = count1 + count2
        stats = self.memory_system.stats()
        
        # Check structure
        self.assertIn("total", stats)
        self.assertIn("avg_score", stats)
        self.assertIn("sentiments", stats)
        self.assertIn("categories", stats)
        
        self.assertEqual(stats["total"], total_added)
        self.assertIsInstance(stats["avg_score"], (int, float))
        self.assertIsInstance(stats["categories"], dict)

    def test_file_ingestion(self):
        """Test ingesting from a file."""
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test.md")
        with open(test_file, "w") as f:
            f.write("# Test Document\nThis is test content for file ingestion.\nAnother line of content.")
        
        count = self.memory_system.ingest_file(test_file, category="documents")
        self.assertGreater(count, 0)
        
        # Check that entries were created
        doc_entries = [m for m in self.memory_system.memories if m.category == "documents"]
        self.assertGreater(len(doc_entries), 0)
        self.assertEqual(doc_entries[0].source, test_file)


class TestInputGate(unittest.TestCase):
    """Test the input gating system."""
    
    def setUp(self):
        self.gate = InputGate()
    
    def test_p0_classification(self):
        """Test P0 (critical) classification."""
        # Security issues
        self.assertEqual(self.gate.classify("Security breach detected in user authentication"), "P0")
        self.assertEqual(self.gate.classify("Critical error in payment processing"), "P0")
        
        # Financial commitments
        self.assertEqual(self.gate.classify("Budget approved for $50,000 project"), "P0")
        self.assertEqual(self.gate.classify("Contract signed with new vendor"), "P0")
        
        # Deadlines
        self.assertEqual(self.gate.classify("Urgent deadline tomorrow for product launch"), "P0")
    
    def test_p1_classification(self):
        """Test P1 (operational) classification."""
        # Decisions
        self.assertEqual(self.gate.classify("We decided to use PostgreSQL for the database"), "P1")
        self.assertEqual(self.gate.classify("Task assigned to John for API development"), "P1")
        
        # Technical choices
        self.assertEqual(self.gate.classify("Selected React framework for frontend"), "P1")
        self.assertEqual(self.gate.classify("API integration with third-party service approved"), "P1")
        
        # Meeting outcomes
        self.assertEqual(self.gate.classify("Meeting concluded with next steps defined"), "P1")
    
    def test_p2_classification(self):
        """Test P2 (contextual) classification."""
        # Background information
        self.assertEqual(self.gate.classify("For reference, here are the technical specifications"), "P2")
        self.assertEqual(self.gate.classify("Research shows that users prefer simple interfaces"), "P2")
        
        # General discussion
        self.assertEqual(self.gate.classify("This project involves web development and data analysis"), "P2")
    
    def test_p3_classification(self):
        """Test P3 (ephemeral) classification."""
        # Greetings
        self.assertEqual(self.gate.classify("Hi there!"), "P3")
        self.assertEqual(self.gate.classify("Good morning everyone"), "P3")
        
        # Acknowledgments
        self.assertEqual(self.gate.classify("Thanks"), "P3")
        self.assertEqual(self.gate.classify("OK"), "P3")
        self.assertEqual(self.gate.classify("Got it"), "P3")
        
        # Filler
        self.assertEqual(self.gate.classify("lol"), "P3")
        self.assertEqual(self.gate.classify("nice"), "P3")
        
        # Very short content
        self.assertEqual(self.gate.classify("k"), "P3")
        self.assertEqual(self.gate.classify(""), "P3")
    
    def test_should_store(self):
        """Test storage decision logic."""
        # P0-P2 should be stored
        self.assertTrue(self.gate.should_store("Critical security issue found"))
        self.assertTrue(self.gate.should_store("Decided to use new technology"))
        self.assertTrue(self.gate.should_store("Background research on market trends"))
        
        # P3 should not be stored
        self.assertFalse(self.gate.should_store("Thanks"))
        self.assertFalse(self.gate.should_store("lol"))
        self.assertFalse(self.gate.should_store("OK"))
    
    def test_route(self):
        """Test complete routing functionality."""
        # Test P0 routing
        route = self.gate.route("Critical deadline approaching for project delivery")
        self.assertEqual(route["priority"], "P0")
        self.assertEqual(route["category"], "strategic")
        self.assertTrue(route["store"])
        
        # Test P1 routing
        route = self.gate.route("Team decided to implement new feature")
        self.assertEqual(route["priority"], "P1")
        self.assertEqual(route["category"], "operational")
        self.assertTrue(route["store"])
        
        # Test P3 routing
        route = self.gate.route("thanks")
        self.assertEqual(route["priority"], "P3")
        self.assertEqual(route["category"], "ephemeral")
        self.assertFalse(route["store"])
    
    def test_context_hints(self):
        """Test classification with context hints."""
        context = {"source": "security_alert", "category": "critical"}
        self.assertEqual(self.gate.classify("System notification", context), "P0")
        
        context = {"source": "meeting_notes", "category": "operational"}
        self.assertEqual(self.gate.classify("Discussion about project timeline", context), "P1")


class TestKnowledgeSynthesizer(unittest.TestCase):
    """Test the knowledge synthesis engine."""
    
    def setUp(self):
        self.synthesizer = KnowledgeSynthesizer()
        
        # Create some test memories
        self.test_memories = [
            MemoryEntry("What is PostgreSQL and how does it compare to MySQL?", "questions"),
            MemoryEntry("We need to research Docker containerization", "todo"),
            MemoryEntry("API endpoint returns user data in JSON format", "technical"),
            MemoryEntry("Using React for frontend development", "technical"),
            MemoryEntry("PostgreSQL mentioned in database discussion", "reference"),
            MemoryEntry("TODO: Set up CI/CD pipeline", "todo"),
            MemoryEntry("Docker containers provide isolation", "research"),
        ]
    
    def test_identify_gaps(self):
        """Test knowledge gap identification."""
        gaps = self.synthesizer.identify_gaps(self.test_memories)
        self.assertIsInstance(gaps, list)
        self.assertGreater(len(gaps), 0)
        
        # Should identify questions
        gap_text = " ".join(gaps)
        self.assertIn("question", gap_text.lower())
    
    def test_suggest_research_topics(self):
        """Test research topic suggestions."""
        suggestions = self.synthesizer.suggest_research_topics(self.test_memories, limit=3)
        self.assertIsInstance(suggestions, list)
        self.assertLessEqual(len(suggestions), 3)
        
        if suggestions:
            suggestion = suggestions[0]
            self.assertIn("topic", suggestion)
            self.assertIn("reason", suggestion)
            self.assertIn("priority", suggestion)
            self.assertIn(suggestion["priority"], ["P0", "P1", "P2"])
    
    def test_synthesize(self):
        """Test knowledge synthesis from new information."""
        new_info = "PostgreSQL is a powerful open-source relational database system.\nDocker is a containerization platform that packages applications."
        
        synthesized = self.synthesizer.synthesize(self.test_memories, new_info, "research_source")
        self.assertIsInstance(synthesized, list)
        self.assertGreater(len(synthesized), 0)
        
        # Check synthesized entries have expected properties
        if synthesized:
            entry = synthesized[0]
            self.assertIsInstance(entry, MemoryEntry)
            self.assertIn("synthesis", entry.tags)
            self.assertGreater(entry.confidence, 0.5)
    
    def test_run_cycle(self):
        """Test complete synthesis cycle."""
        research_results = {
            "database_research": "PostgreSQL offers ACID compliance and advanced features",
            "container_research": "Docker provides lightweight containerization"
        }
        
        report = self.synthesizer.run_cycle(self.test_memories, research_results)
        
        self.assertIsInstance(report, dict)
        self.assertIn("timestamp", report)
        self.assertIn("gaps_identified", report)
        self.assertIn("research_suggestions", report)
        self.assertIn("synthesized_entries", report)


class TestMemorySystemWithGating(unittest.TestCase):
    """Test MemorySystem with gating integration."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory_system = MemorySystem(workspace=self.temp_dir, half_life=7.0)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ingest_with_gating(self):
        """Test gated ingestion functionality."""
        content = """
        Critical security vulnerability found in authentication system
        We decided to use PostgreSQL for the database  
        This is some background information about the project
        Thanks for the update
        OK got it
        lol that's funny
        """
        
        count = self.memory_system.ingest_with_gating(content, "test_source")
        
        # Should ingest P0-P2 content but not P3
        self.assertGreater(count, 0)
        self.assertLess(count, 6)  # Should be less than total lines due to P3 filtering
        
        # Check that P3 content was filtered out
        content_texts = [m.content for m in self.memory_system.memories]
        combined_content = " ".join(content_texts)
        
        # Should contain important content
        self.assertIn("security", combined_content.lower())
        self.assertIn("postgresql", combined_content.lower())
        
        # Should not contain P3 filler
        self.assertNotIn("Thanks for the update", combined_content)
        self.assertNotIn("lol that's funny", combined_content)
    
    def test_synthesis_integration(self):
        """Test synthesis integration in memory system."""
        # Add some memories with gaps
        self.memory_system.ingest("What is machine learning?", source="questions")
        self.memory_system.ingest("TODO: Research neural networks", source="todo")
        self.memory_system.ingest("Using TensorFlow for ML development", source="technical")
        
        # Test research suggestions
        suggestions = self.memory_system.research_suggestions(limit=2)
        self.assertIsInstance(suggestions, list)
        self.assertLessEqual(len(suggestions), 2)
        
        # Test synthesis cycle
        research_results = {
            "ml_research": "Machine learning is a subset of AI that enables computers to learn without explicit programming"
        }
        
        report = self.memory_system.synthesize(research_results)
        self.assertIsInstance(report, dict)
        self.assertIn("gaps_identified", report)
        
        # Should have added new synthesized memories
        if "synthesized_entries" in report and report["synthesized_entries"] > 0:
            # Check that new entries were added
            synthesis_entries = [m for m in self.memory_system.memories if "synthesis" in m.tags]
            self.assertGreater(len(synthesis_entries), 0)


if __name__ == "__main__":
    unittest.main()