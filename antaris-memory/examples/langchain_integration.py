"""
LangChain Integration Example

This example shows how to create a LangChain BaseMemory subclass 
backed by antaris-memory for persistent, intelligent memory management.

Installation:
    pip install langchain-core

Usage:
    memory = AntarisLangChainMemory("./workspace")
    
    # Use in LangChain chains
    from langchain.chains import ConversationChain
    chain = ConversationChain(memory=memory, llm=your_llm)
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from langchain_core.memory import BaseMemory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
except ImportError:
    print("LangChain not installed. Install with: pip install langchain-core")
    # Create mock classes for demo
    class BaseMemory:
        pass
    class BaseMessage:
        def __init__(self, content):
            self.content = content

from antaris_memory import MemorySystem


class AntarisLangChainMemory(BaseMemory):
    """LangChain Memory implementation backed by antaris-memory."""
    
    def __init__(
        self,
        workspace: str,
        memory_key: str = "history",
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        return_messages: bool = False,
        max_token_limit: Optional[int] = None,
        decay_half_life: float = 7.0,
        use_gating: bool = True
    ):
        self.workspace = workspace
        self.memory_key = memory_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.return_messages = return_messages
        self.max_token_limit = max_token_limit
        self.use_gating = use_gating
        
        # Initialize antaris-memory
        self.memory_system = MemorySystem(
            workspace=workspace,
            half_life=decay_half_life,
            tag_terms=["langchain", "conversation", "context"]
        )
        
        # Load existing memories
        self.memory_system.load()
        
        # Track conversation context
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for the chain."""
        # Get current input to search for relevant context
        current_input = inputs.get("input", "")
        
        if current_input:
            # Search for relevant memories
            relevant_memories = self.memory_system.search(
                current_input, 
                limit=10,
                min_confidence=0.3
            )
            
            if self.return_messages:
                # Return as message objects
                messages = []
                for mem in relevant_memories:
                    # Parse stored conversation format
                    if ":" in mem.content and "[" in mem.content:
                        # Extract speaker and content from stored format
                        # Format: "[timestamp] Speaker: content"
                        try:
                            parts = mem.content.split("] ", 1)[1]  # Remove timestamp
                            speaker, content = parts.split(": ", 1)
                            
                            if speaker.lower() in [self.human_prefix.lower(), "human", "user"]:
                                messages.append(HumanMessage(content=content))
                            else:
                                messages.append(AIMessage(content=content))
                        except (ValueError, IndexError):
                            # Fallback for non-standard format
                            messages.append(AIMessage(content=mem.content))
                
                memory_value = messages[-20:]  # Limit to recent messages
            else:
                # Return as formatted string
                memory_lines = []
                for mem in relevant_memories[-10:]:  # Most recent relevant
                    # Clean up the stored format for display
                    content = mem.content
                    if "] " in content and ":" in content:
                        # Remove timestamp, keep speaker format
                        content = content.split("] ", 1)[1]
                    memory_lines.append(content)
                
                memory_value = "\n".join(memory_lines)
        else:
            memory_value = [] if self.return_messages else ""
        
        return {self.memory_key: memory_value}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save the context to memory."""
        # Extract human input
        human_input = inputs.get("input", "")
        
        # Extract AI output
        ai_output = outputs.get("response", outputs.get("output", ""))
        
        timestamp = datetime.now().isoformat()
        
        # Format conversation entries
        conversation_content = f"[{timestamp}] {self.human_prefix}: {human_input}\n[{timestamp}] {self.ai_prefix}: {ai_output}"
        
        source = f"langchain_session:{self.session_id}"
        
        if self.use_gating:
            # Use intelligent gating for automatic categorization
            context = {
                "source": "langchain",
                "session_id": self.session_id,
                "interaction_type": "conversation"
            }
            self.memory_system.ingest_with_gating(conversation_content, source, context)
        else:
            # Use standard ingestion
            self.memory_system.ingest(conversation_content, source, "tactical")
        
        # Auto-save periodically
        if len(self.memory_system.memories) % 10 == 0:
            self.memory_system.save()
    
    def clear(self) -> None:
        """Clear memory (start fresh session)."""
        # Start a new session rather than clearing all memory
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Optionally save current state before clearing
        self.memory_system.save()
    
    def save_memory(self) -> str:
        """Explicitly save memory to disk."""
        return self.memory_system.save()
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memory explicitly."""
        results = self.memory_system.search(query, limit=limit)
        return [
            {
                "content": mem.content,
                "source": mem.source,
                "confidence": mem.confidence,
                "category": mem.category,
                "created": mem.created
            }
            for mem in results
        ]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory_system.stats()
    
    def run_synthesis(self) -> Dict:
        """Run knowledge synthesis cycle."""
        return self.memory_system.synthesize()


# Example usage
def langchain_example():
    """Example of using AntarisLangChainMemory with LangChain."""
    
    # Initialize memory
    memory = AntarisLangChainMemory(
        workspace="./langchain-workspace",
        return_messages=False,  # Return formatted string
        use_gating=True  # Use intelligent P0-P3 filtering
    )
    
    # Simulate LangChain conversation
    print("=== LangChain + Antaris Memory Demo ===\n")
    
    # Simulate multiple conversation turns
    conversations = [
        {
            "input": "I need help planning a Python project structure",
            "response": "I'd recommend using a standard structure with src/ directory, tests/, and proper package organization. What type of project are you building?"
        },
        {
            "input": "It's a web API using FastAPI",
            "response": "Great choice! For FastAPI projects, I suggest: app/ for main code, app/routers/ for endpoints, app/models/ for data models, tests/ for testing, and requirements.txt for dependencies."
        },
        {
            "input": "What about database integration?",
            "response": "For FastAPI + database, SQLAlchemy is popular. Create app/database.py for connection, app/models/ for ORM models, and app/crud/ for database operations."
        }
    ]
    
    # Process conversations
    for i, conv in enumerate(conversations):
        print(f"--- Turn {i+1} ---")
        
        # Load relevant context (this is what LangChain would do)
        context = memory.load_memory_variables({"input": conv["input"]})
        print(f"Loaded context: {len(context['history'])} chars")
        
        # Save the conversation (this is what LangChain would do after generation)
        memory.save_context(
            {"input": conv["input"]},
            {"response": conv["response"]}
        )
        print(f"Saved conversation turn {i+1}")
        print()
    
    # Demonstrate search capabilities
    print("=== Search Demo ===")
    search_results = memory.search_memory("FastAPI project structure")
    print(f"Found {len(search_results)} relevant memories:")
    for result in search_results:
        print(f"- {result['content'][:100]}... (confidence: {result['confidence']})")
    print()
    
    # Show memory stats
    stats = memory.get_memory_stats()
    print(f"=== Memory Stats ===")
    print(f"Total memories: {stats['total']}")
    print(f"Categories: {stats['categories']}")
    print(f"Average confidence: {stats['avg_confidence']}")
    print()
    
    # Run synthesis
    synthesis = memory.run_synthesis()
    print(f"=== Knowledge Synthesis ===")
    print(f"Gaps identified: {len(synthesis.get('gaps_identified', []))}")
    print(f"Research suggestions: {len(synthesis.get('research_suggestions', []))}")
    
    # Save final state
    saved_path = memory.save_memory()
    print(f"\nMemory saved to: {saved_path}")


# Advanced usage example with custom chain
def custom_chain_example():
    """Example of building a custom chain with persistent memory."""
    
    class PersistentConversationChain:
        """Simple conversation chain with antaris-memory."""
        
        def __init__(self, workspace: str):
            self.memory = AntarisLangChainMemory(workspace)
            
        def predict(self, user_input: str) -> str:
            # Load relevant context
            context = self.memory.load_memory_variables({"input": user_input})
            relevant_history = context["history"]
            
            # Simple response generation (in real use, call your LLM here)
            if "python" in user_input.lower():
                response = "Python is a great choice! " + (
                    "Based on our previous discussion, " if relevant_history else ""
                ) + "What specific aspect would you like help with?"
            else:
                response = "I can help with that. " + (
                    "Given our conversation history, " if relevant_history else ""
                ) + "Could you provide more details?"
            
            # Save the exchange
            self.memory.save_context(
                {"input": user_input},
                {"response": response}
            )
            
            return response
    
    # Demo the custom chain
    chain = PersistentConversationChain("./custom-chain-workspace")
    
    print("=== Custom Chain Demo ===")
    inputs = [
        "I want to learn Python",
        "What about web development?",
        "How do I structure a Python web project?"
    ]
    
    for user_input in inputs:
        response = chain.predict(user_input)
        print(f"User: {user_input}")
        print(f"AI: {response}")
        print()


if __name__ == "__main__":
    langchain_example()
    print("\n" + "="*50 + "\n")
    custom_chain_example()