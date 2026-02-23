#!/usr/bin/env python3
"""
Antaris Memory Quickstart Example

This example shows basic usage of the antaris-memory package:
- Creating a memory system
- Adding memories from text and files
- Searching memories
- Checking decay scores and sentiment
- Temporal queries
"""

import os
import tempfile
from datetime import datetime, timedelta

# Import the memory system
from antaris_memory import MemorySystem

def main():
    print("🧠 Antaris Memory Quickstart Example")
    print("=" * 40)
    
    # Create a temporary workspace for this demo
    workspace = tempfile.mkdtemp()
    print(f"💾 Using workspace: {workspace}")
    
    # Initialize the memory system
    memory = MemorySystem(workspace=workspace, half_life=7.0)
    print(f"✅ Memory system initialized")
    
    # Add some sample memories
    print("\n📝 Adding sample memories...")
    
    sample_content = """
    We filed a patent application for our AI memory system on January 15th.
    The patent covers novel decay algorithms and sentiment-aware retrieval.
    Initial customer feedback has been overwhelmingly positive.
    Revenue projections show $500K potential in Q1 enterprise sales.
    The engineering team completed optimization work ahead of schedule.
    User testing revealed some minor UI improvements needed.
    Partnership discussions with three major tech companies are ongoing.
    """
    
    count = memory.ingest(sample_content, source="team_notes", category="strategic")
    print(f"➕ Added {count} memories from sample content")
    
    # Create and ingest a sample file
    sample_file = os.path.join(workspace, "meeting_notes.md")
    with open(sample_file, "w") as f:
        f.write("""
# Weekly Team Meeting - AI Memory Project

## Progress Updates
- Memory compression algorithms showing 60% space savings
- Sentiment analysis integration completed successfully  
- Temporal query performance improved by 40%
- Beta testing feedback compilation in progress

## Technical Achievements
- Implemented patent-pending decay mechanisms
- Added confidence scoring for enterprise reliability
- Built consolidation engine for long-term memory
- Zero external dependencies maintained as design goal

## Business Development
- Enterprise pilot program launched with Fortune 500 client
- SaaS pricing model finalized at $50/month per agent
- Partnership discussions advancing with cloud providers
- Patent application status: under review, looking positive

## Next Steps
- Finalize PyPI package publication
- Complete GitHub repository setup
- Schedule customer demo sessions
- Prepare technical documentation
        """)
    
    file_count = memory.ingest_file(sample_file, category="meetings")
    print(f"📄 Added {file_count} memories from meeting notes file")
    
    print(f"\n📊 Memory system stats:")
    stats = memory.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Demonstrate search functionality
    print("\n🔍 Search Examples:")
    
    # Search for patent-related memories
    patent_results = memory.search("patent", limit=3)
    print(f"\n'patent' search returned {len(patent_results)} results:")
    for i, result in enumerate(patent_results, 1):
        print(f"  {i}. [{result.category}] {result.content[:80]}...")
        print(f"     Source: {result.source} | Confidence: {result.confidence:.2f}")
    
    # Search with category filter
    strategic_results = memory.search("revenue", category="strategic")
    print(f"\n'revenue' in 'strategic' category: {len(strategic_results)} results")
    if strategic_results:
        result = strategic_results[0]
        print(f"  → {result.content[:100]}...")
    
    # Demonstrate decay scoring
    print(f"\n⏰ Decay Scores (fresher memories score higher):")
    for i, mem in enumerate(memory.memories[:5]):  # Show first 5
        decay_score = memory.decay.score(mem)
        print(f"  {i+1}. Score: {decay_score:.3f} | {mem.content[:60]}...")
    
    # Demonstrate sentiment analysis
    print(f"\n😊 Sentiment Analysis Examples:")
    sentiment_examples = [
        "The customer was extremely happy with our AI memory solution!",
        "Unfortunately, the server crashed during the demo presentation.",
        "The technical implementation is proceeding according to plan."
    ]
    
    for text in sentiment_examples:
        sentiment = memory.sentiment.analyze(text)
        dominant = memory.sentiment.dominant(sentiment)
        print(f"  '{text[:50]}...'")
        print(f"    → Dominant sentiment: {dominant or 'neutral'}")
        if sentiment:
            print(f"    → Scores: {dict(list(sentiment.items())[:3])}")  # Show top 3
    
    # Demonstrate temporal queries
    print(f"\n📅 Temporal Queries:")
    today = datetime.now().date().isoformat()
    yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
    
    today_memories = memory.on_date(today)
    print(f"  Memories from today ({today}): {len(today_memories)}")
    
    # Get memories from the last week
    week_ago = (datetime.now().date() - timedelta(days=7)).isoformat()
    recent_memories = memory.between(week_ago, today)
    print(f"  Memories from last week: {len(recent_memories)}")
    
    # Demonstrate narrative generation
    narrative = memory.narrative(topic="patent")
    if narrative.strip():
        print(f"\n📚 Narrative for 'patent' topic:")
        print(f"  {narrative[:200]}...")
    
    # Save the memory state
    save_path = memory.save()
    print(f"\n💾 Memory state saved to: {os.path.basename(save_path)}")
    
    # Demonstrate loading
    new_memory = MemorySystem(workspace=workspace)
    loaded_count = new_memory.load()
    print(f"🔄 Loaded {loaded_count} memories in new instance")
    
    # Demonstrate forgetting (selective memory deletion)
    print(f"\n🗑️ Selective Forgetting Example:")
    initial_count = len(new_memory.memories)
    
    # Forget memories about servers (just as an example)
    forget_result = new_memory.forget(topic="server")
    if "removed" in forget_result:
        print(f"  Forgot {forget_result['removed']} memories about 'server'")
        print(f"  Memories remaining: {len(new_memory.memories)}")
    else:
        print(f"  No memories about 'server' found to forget")
    
    print(f"\n🎉 Quickstart complete! The memory system is ready for production use.")
    print(f"   Key features demonstrated:")
    print(f"   ✓ Memory ingestion from text and files")
    print(f"   ✓ Intelligent search with decay weighting") 
    print(f"   ✓ Sentiment analysis and confidence scoring")
    print(f"   ✓ Temporal queries and narrative generation")
    print(f"   ✓ Persistence and selective forgetting")
    print(f"\n   Next steps:")
    print(f"   • Try ingesting your own files with ingest_file() or ingest_directory()")
    print(f"   • Experiment with different search queries and categories")
    print(f"   • Use consolidate() for memory optimization")
    print(f"   • Check out the full documentation for advanced features")
    
    # Clean up
    print(f"\n🧹 Cleaning up temporary files...")
    import shutil
    shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()