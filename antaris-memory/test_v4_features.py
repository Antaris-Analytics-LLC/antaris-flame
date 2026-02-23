#!/usr/bin/env python3
"""
Test v0.4 features: sharding, migration, and indexing.
"""

import os
import tempfile
import shutil
import json
from antaris_memory import MemorySystem, MemoryEntry
from antaris_memory.sharding import ShardManager
from antaris_memory.migration import MigrationManager
from antaris_memory.indexing import IndexManager


def test_sharding_system():
    """Test sharding functionality."""
    print("🧪 Testing sharding system...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test memories
        memories = [
            MemoryEntry("Task completed successfully", source="test", category="work"),
            MemoryEntry("Meeting scheduled for tomorrow", source="test", category="calendar"),  
            MemoryEntry("Research findings on AI memory systems", source="test", category="research"),
            MemoryEntry("Bug fixed in production code", source="test", category="work"),
            MemoryEntry("Travel plans to conference", source="test", category="personal")
        ]
        
        # Set different dates to test date-based sharding
        memories[0].created = "2026-01-15T10:00:00"
        memories[1].created = "2026-02-15T11:00:00"
        memories[2].created = "2026-02-15T12:00:00"
        memories[3].created = "2026-02-16T09:00:00"
        memories[4].created = "2026-03-01T14:00:00"
        
        # Initialize shard manager
        shard_manager = ShardManager(tmpdir)
        
        # Shard the memories
        shard_groups = shard_manager.shard_memories(memories)
        print(f"  Created {len(shard_groups)} shards")
        
        # Save shards
        for shard_key, shard_memories in shard_groups.items():
            shard_manager.save_shard(shard_key, shard_memories)
            shard_manager.index.add_shard(shard_key, shard_memories)
        
        # Save index
        shard_manager.index.save_index()
        
        # Test loading shard
        for shard_key in shard_groups.keys():
            loaded_memories = shard_manager.load_shard(shard_key)
            print(f"  Shard {shard_key}: loaded {len(loaded_memories)} memories")
            
            # Verify content matches
            original_hashes = {m.hash for m in shard_groups[shard_key]}
            loaded_hashes = {m.hash for m in loaded_memories}
            assert original_hashes == loaded_hashes, "Loaded memories don't match original"
        
        # Test search
        results = shard_manager.search_shards("research", limit=10)
        print(f"  Search for 'research': found {len(results)} results")
        assert len(results) > 0, "Should find research-related memories"
        
        # Get stats
        stats = shard_manager.index.get_stats()
        print(f"  Stats: {stats['total_memories']} memories across {stats['total_shards']} shards")
        
        print("✅ Sharding system working correctly")


def test_indexing_system():
    """Test indexing functionality."""
    print("🧪 Testing indexing system...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test memories with different content
        memories = [
            MemoryEntry("Python programming tutorial for beginners", source="test", category="education"),
            MemoryEntry("Web development using JavaScript and React", source="test", category="tech"),
            MemoryEntry("Machine learning algorithms and data science", source="test", category="ai"),
            MemoryEntry("Project management best practices", source="test", category="business"),
            MemoryEntry("Python data analysis with pandas library", source="test", category="education")
        ]
        
        # Add tags for testing
        memories[0].tags = ["python", "tutorial", "programming"]
        memories[1].tags = ["javascript", "react", "web"]
        memories[2].tags = ["ml", "ai", "data-science"]
        memories[3].tags = ["management", "project"]
        memories[4].tags = ["python", "pandas", "data"]
        
        # Different dates for date index testing
        memories[0].created = "2026-02-10T10:00:00"
        memories[1].created = "2026-02-11T10:00:00"
        memories[2].created = "2026-02-12T10:00:00"
        memories[3].created = "2026-02-13T10:00:00"
        memories[4].created = "2026-02-14T10:00:00"
        
        # Initialize index manager
        index_manager = IndexManager(tmpdir)
        
        # Add memories to indexes
        for memory in memories:
            index_manager.add_memory(memory)
        
        # Test text search
        results = index_manager.search("python", limit=10)
        print(f"  Text search 'python': found {len(results)} results")
        assert len(results) >= 2, "Should find Python-related memories"
        
        # Test tag filtering
        results = index_manager.search("", tags=["python"], limit=10)
        print(f"  Tag search 'python': found {len(results)} results")
        assert len(results) >= 2, "Should find memories tagged with python"
        
        # Test date range search
        results = index_manager.search("", date_range=("2026-02-12", "2026-02-14"), limit=10)
        print(f"  Date range search: found {len(results)} results")
        assert len(results) >= 2, "Should find memories in date range"
        
        # Test combined search
        results = index_manager.search("data", tags=["python"], limit=10)
        print(f"  Combined search 'data' + 'python' tag: found {len(results)} results")
        assert len(results) >= 1, "Should find data-related Python memories"
        
        # Save indexes
        index_manager.save_all_indexes()
        
        # Test loading indexes
        new_index_manager = IndexManager(tmpdir)
        loaded_results = new_index_manager.search("python", limit=10)
        print(f"  After reload, 'python' search: found {len(loaded_results)} results")
        assert len(loaded_results) >= 2, "Loaded indexes should work correctly"
        
        # Get stats
        stats = index_manager.get_combined_stats()
        print(f"  Index stats: {stats['search_index']['total_memories']} memories, {stats['search_index']['total_words']} words indexed")
        
        print("✅ Indexing system working correctly")


def test_migration_system():
    """Test migration functionality."""
    print("🧪 Testing migration system...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a v0.2 format file manually
        legacy_data = {
            "version": "0.2.1",
            "saved_at": "2026-02-15T10:00:00",
            "count": 3,
            "memories": [
                {
                    "content": "Legacy memory 1",
                    "hash": "hash1",
                    "created": "2026-02-15T10:00:00",
                    "source": "test",
                    "category": "general",
                    "tags": ["legacy"],
                    "confidence": 0.8
                },
                {
                    "content": "Legacy memory 2", 
                    "hash": "hash2",
                    "created": "2026-02-15T11:00:00",
                    "source": "test",
                    "category": "work",
                    "tags": ["important"],
                    "confidence": 0.9
                },
                {
                    "content": "Legacy memory 3",
                    "hash": "hash3", 
                    "created": "2026-02-15T12:00:00",
                    "source": "test",
                    "category": "personal",
                    "tags": [],
                    "confidence": 0.7
                }
            ]
        }
        
        # Write legacy format file
        legacy_path = os.path.join(tmpdir, "memory_metadata.json")
        with open(legacy_path, "w") as f:
            json.dump(legacy_data, f, indent=2)
        
        # Initialize migration manager
        migration_manager = MigrationManager(tmpdir)
        
        # Detect version
        detected_version = migration_manager.detect_version()
        print(f"  Detected version: {detected_version}")
        assert detected_version == "0.2.1", "Should detect legacy version"
        
        # Check if migration is needed
        needs_migration = migration_manager.needs_migration("0.4.0")
        print(f"  Needs migration: {needs_migration}")
        assert needs_migration, "Should need migration from v0.2 to v0.4"
        
        # Perform migration
        result = migration_manager.migrate("0.4.0")
        print(f"  Migration result: {result['status']}")
        assert result["status"] == "success", "Migration should succeed"
        
        # Verify shards were created
        shards_dir = os.path.join(tmpdir, "shards")
        assert os.path.exists(shards_dir), "Shards directory should exist"
        
        shard_files = os.listdir(shards_dir)
        print(f"  Created {len(shard_files)} shard files")
        assert len(shard_files) > 0, "Should create shard files"
        
        # Verify index was created
        index_path = os.path.join(tmpdir, "memory_index.json")
        assert os.path.exists(index_path), "Memory index should exist"
        
        # Test that we can load the migrated data
        shard_manager = ShardManager(tmpdir)
        migrated_memories = shard_manager.get_all_memories()
        print(f"  Loaded {len(migrated_memories)} memories after migration")
        assert len(migrated_memories) == 3, "Should load all original memories"
        
        # Verify content is preserved
        content_set = {m.content for m in migrated_memories}
        expected_content = {"Legacy memory 1", "Legacy memory 2", "Legacy memory 3"}
        assert content_set == expected_content, "Memory content should be preserved"
        
        # Test rollback capability
        rollback_result = migration_manager.rollback()
        print(f"  Rollback result: {rollback_result['status']}")
        assert rollback_result["status"] == "success", "Rollback should succeed"
        
        # Verify original file is restored
        assert os.path.exists(legacy_path), "Original file should be restored"
        assert not os.path.exists(index_path), "Index file should be removed"
        
        print("✅ Migration system working correctly")


def test_v4_integration():
    """Test full v0.4 MemorySystem integration."""
    print("🧪 Testing v0.4 MemorySystem integration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize v0.4 memory system
        mem = MemorySystem(tmpdir, use_sharding=True, use_indexing=True)
        
        # Add some test content
        test_content = """
        Project kickoff meeting scheduled for Monday
        Technical architecture decisions documented
        Database schema finalized with PostgreSQL
        Frontend framework chosen: React with TypeScript
        API design completed using REST principles
        Security audit planned for next week
        Performance benchmarks established
        Code review process implemented
        Deployment pipeline configured with GitHub Actions
        Documentation wiki created for team reference
        """
        
        # Ingest content
        ingested_count = mem.ingest(test_content, source="test_integration", category="project")
        print(f"  Ingested {ingested_count} memories")
        assert ingested_count > 0, "Should ingest some memories"
        
        # Test search functionality
        results = mem.search("database PostgreSQL", limit=5)
        print(f"  Search 'database PostgreSQL': found {len(results)} results")
        assert len(results) > 0, "Should find database-related memories"
        
        # Test tag-based search
        results = mem.search("technical", tags=["technical"], limit=5)
        print(f"  Search with tags: found {len(results)} results")
        
        # Save in sharded format
        save_result = mem.save()
        print(f"  Save result: {save_result}")
        
        # Verify sharded files exist
        shards_dir = os.path.join(tmpdir, "shards")
        if os.path.exists(shards_dir):
            shard_files = os.listdir(shards_dir)
            print(f"  Created {len(shard_files)} shard files")
        
        # Test analysis features
        analysis = mem.analyze("technical architecture")
        print(f"  Analysis result: {analysis['status']}, found {analysis.get('total_memories', 0)} related memories")
        
        # Get system stats
        stats = mem.get_stats()
        print(f"  System stats: {stats['total_memories']} memories, features: {stats['features']}")
        
        # Test reload capability
        mem2 = MemorySystem(tmpdir, use_sharding=True, use_indexing=True)
        loaded_count = len(mem2.memories)
        print(f"  Reloaded {loaded_count} memories")
        assert loaded_count == len(mem.memories), "Should reload same number of memories"
        
        print("✅ v0.4 integration working correctly")


def main():
    """Run all v0.4 tests."""
    print("🚀 Testing antaris-memory v0.4 features...\n")
    
    try:
        test_sharding_system()
        print()
        
        test_indexing_system()
        print()
        
        test_migration_system()
        print()
        
        test_v4_integration()
        print()
        
        print("🎉 All v0.4 tests passed successfully!")
        print("\n📊 v0.4 Features Ready:")
        print("  ✅ Sharded storage for scalability")
        print("  ✅ Fast search indexes (text, tags, dates)")
        print("  ✅ Schema migration with backward compatibility")
        print("  ✅ Production-ready MemorySystem integration")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()