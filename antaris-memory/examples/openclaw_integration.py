"""
OpenClaw Integration Example

This example shows how to integrate antaris-memory into an OpenClaw agent's 
heartbeat cycle for persistent memory across sessions.

Usage:
    1. Copy this code into your OpenClaw agent
    2. Call initialize_memory() on session start
    3. Call heartbeat_memory_cycle() during heartbeat processing
    4. Call save_memory() before session end
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

# Assuming antaris-memory is installed
from antaris_memory import MemorySystem

class OpenClawMemoryManager:
    """Memory manager for OpenClaw agents."""
    
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(exist_ok=True)
        
        # Initialize memory system
        self.memory = MemorySystem(
            workspace=str(self.workspace),
            half_life=7.0,  # 7-day decay
            tag_terms=["openclaw", "agent", "task", "decision"]
        )
        
        # Track last heartbeat time
        self.last_heartbeat = None
        self.heartbeat_state_file = self.workspace / "heartbeat_state.json"
        
    def initialize_memory(self) -> Dict:
        """Load memory on session start."""
        count = self.memory.load()
        
        # Load heartbeat state
        if self.heartbeat_state_file.exists():
            with open(self.heartbeat_state_file) as f:
                state = json.load(f)
                self.last_heartbeat = state.get("last_heartbeat")
        
        return {
            "memories_loaded": count,
            "last_heartbeat": self.last_heartbeat,
            "status": "initialized"
        }
    
    def ingest_conversation(self, messages: List[Dict], session_id: str = None) -> int:
        """Ingest conversation history with gating."""
        content_lines = []
        
        for msg in messages:
            timestamp = msg.get("timestamp", datetime.now().isoformat())
            sender = msg.get("sender", "unknown")
            content = msg.get("content", "")
            
            # Format for memory ingestion
            formatted = f"[{timestamp}] {sender}: {content}"
            content_lines.append(formatted)
        
        full_content = "\n".join(content_lines)
        source = f"conversation:{session_id or 'default'}"
        
        # Use gating to automatically filter and categorize
        context = {
            "source": "conversation",
            "session_id": session_id,
            "message_count": len(messages)
        }
        
        return self.memory.ingest_with_gating(full_content, source, context)
    
    def ingest_task_outcome(self, task: str, outcome: str, success: bool) -> int:
        """Ingest task completion with high priority."""
        status = "SUCCESS" if success else "FAILED"
        content = f"TASK {status}: {task} -> {outcome}"
        
        context = {
            "category": "operational",  # Force P1 priority
            "task_type": "completion",
            "success": success
        }
        
        return self.memory.ingest_with_gating(content, "task_tracker", context)
    
    def search_memory(self, query: str, limit: int = 10) -> List[Dict]:
        """Search memory and return formatted results."""
        results = self.memory.search(query, limit=limit)
        
        return [
            {
                "content": mem.content,
                "source": mem.source,
                "category": mem.category,
                "confidence": mem.confidence,
                "tags": mem.tags,
                "created": mem.created,
                "hash": mem.hash
            }
            for mem in results
        ]
    
    def heartbeat_memory_cycle(self) -> Dict:
        """Run during OpenClaw heartbeat for autonomous processing."""
        now = datetime.now()
        report = {
            "timestamp": now.isoformat(),
            "actions": [],
            "suggestions": []
        }
        
        # Run knowledge synthesis every ~6 heartbeats (3 hours if 30min heartbeats)
        should_synthesize = (
            not self.last_heartbeat or 
            (now - datetime.fromisoformat(self.last_heartbeat)).total_seconds() > 10800  # 3 hours
        )
        
        if should_synthesize:
            synthesis_report = self.memory.synthesize()
            report["synthesis"] = synthesis_report
            report["actions"].append("knowledge_synthesis")
            
            # Generate research suggestions
            suggestions = self.memory.research_suggestions(limit=3)
            report["suggestions"].extend(suggestions)
            
            # Update heartbeat state
            self.last_heartbeat = now.isoformat()
            with open(self.heartbeat_state_file, "w") as f:
                json.dump({"last_heartbeat": self.last_heartbeat}, f)
        
        # Memory maintenance
        stats = self.memory.stats()
        if stats["total"] > 1000:  # If too many memories
            # Run consolidation to find duplicates and clusters
            consolidation = self.memory.consolidate()
            report["consolidation"] = consolidation
            report["actions"].append("consolidation")
        
        return report
    
    def save_memory(self) -> str:
        """Save memory state to disk."""
        return self.memory.save()
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics."""
        return self.memory.stats()


# Example usage in OpenClaw agent
def openclaw_agent_example():
    """Example of how to use this in an OpenClaw agent."""
    
    # Initialize memory manager (do this once per session)
    memory_manager = OpenClawMemoryManager("./my-agent-workspace")
    init_result = memory_manager.initialize_memory()
    print(f"Memory initialized: {init_result}")
    
    # Example conversation ingestion
    conversation = [
        {
            "timestamp": "2026-02-15T10:30:00",
            "sender": "user",
            "content": "We need to decide on the database technology for the new project"
        },
        {
            "timestamp": "2026-02-15T10:31:00", 
            "sender": "agent",
            "content": "I recommend PostgreSQL based on your scalability requirements"
        },
        {
            "timestamp": "2026-02-15T10:32:00",
            "sender": "user", 
            "content": "Sounds good, let's go with PostgreSQL"
        }
    ]
    
    # Ingest with automatic gating
    ingested = memory_manager.ingest_conversation(conversation, "session_123")
    print(f"Ingested {ingested} memories from conversation")
    
    # Search memory
    results = memory_manager.search_memory("database decision")
    print(f"Found {len(results)} relevant memories")
    
    # Simulate heartbeat cycle
    heartbeat_report = memory_manager.heartbeat_memory_cycle()
    print(f"Heartbeat completed: {heartbeat_report}")
    
    # Save before session ends
    saved_path = memory_manager.save_memory()
    print(f"Memory saved to: {saved_path}")


if __name__ == "__main__":
    openclaw_agent_example()