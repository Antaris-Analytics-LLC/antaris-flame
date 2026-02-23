# antaris-context

**Zero-dependency context window optimization for AI agents.**

Manage context windows, token budgets, turn lifecycle, and message compression without external dependencies. Integrates with `antaris-memory` for memory-informed priority boosting and `antaris-router` for adaptive budget allocation. Built for production AI agent systems that need deterministic, configurable context management.

[![PyPI](https://img.shields.io/pypi/v/antaris-context)](https://pypi.org/project/antaris-context/)
[![Tests](https://github.com/Antaris-Analytics/antaris-context/actions/workflows/tests.yml/badge.svg)](https://github.com/Antaris-Analytics/antaris-context/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange.svg)](LICENSE)

## What's New in v2.0.0

- **Turn lifecycle API** — `add_turn(role, content)`, `compact_older_turns(keep_last=20)`, `render(provider='anthropic'|'openai'|'generic')`, `set_retention_policy()`, `turn_count`
- **Provider-ready render** — `render()` produces message lists formatted for OpenAI, Anthropic, or generic clients
- **Suite integration** — `set_memory_client(client)` for memory-informed priority boosting; `set_router_hints(hints)` accepts hints from `antaris-router` and adjusts section budgets automatically
- **Pluggable summarizer** — `set_summarizer(fn)` — plug in any function to compress older turns semantically
- **`ImportanceWeightedCompressor`** — priority-aware compression with `CompressionResult` reporting
- **`SemanticChunker`** — sentence-boundary-aware text chunking with configurable overlap
- **Cross-session snapshots** — `export_snapshot(include_importance_above)`, `from_snapshot(dict)` for persistence across sessions
- 150 tests (all passing)

See [CHANGELOG.md](CHANGELOG.md) for full version history.


## OpenClaw Integration

antaris-context is purpose-built for OpenClaw agent sessions. Use it to manage the context window across multi-turn conversations — automatically compressing older turns to make room for memory recall, tool results, and new input.

```python
from antaris_context import ContextManager

ctx = ContextManager(total_budget=8000)
ctx.add_turn("user", user_input)
ctx.add_turn("assistant", agent_response)

# Before the next turn, compact to stay within budget
ctx.compact_older_turns(keep_last=10)
messages = ctx.render()  # Ready for any provider (OpenAI, Anthropic, etc.)
```

Pairs directly with antaris-memory (inject recalled memories into context budget) and antaris-router (route based on actual token count).

## Install

```bash
pip install antaris-context
```

## Quick Start

```python
from antaris_context import ContextManager

# Initialize with a preset template
manager = ContextManager(total_budget=8000, template="code_assistant")
# Templates: chatbot, agent_with_tools, rag_pipeline, code_assistant, balanced

# Add turns (conversation lifecycle)
manager.add_turn("user", "How do I add JWT auth to my Flask API?")
manager.add_turn("assistant", "Use flask-jwt-extended. Here's a minimal example...")

# Check turn count and budget usage
print(f"Turns: {manager.turn_count}")
report = manager.get_usage_report()
print(f"Used: {report['total_used']}/{report['total_budget']} tokens ({report['utilization']:.1%})")

# Compact old turns when context gets full
removed = manager.compact_older_turns(keep_last=20)
print(f"Compacted {removed} turns")

# Render for your LLM provider
messages = manager.render(provider="anthropic")       # → Anthropic message format
messages = manager.render(provider="openai")          # → OpenAI message format
messages = manager.render(provider="generic")         # → generic list of dicts
messages = manager.render(system_prompt="Be concise") # → inject system prompt

# Optimize to target utilization
result = manager.optimize_context(target_utilization=0.85)
print(f"Success: {result.success}, Actions: {result.actions_taken}")
```

## Turn Lifecycle

The v2.0 turn API mirrors how LLMs actually work:

```python
manager = ContextManager(total_budget=16000, template="agent_with_tools")

# Add turns from a conversation
for msg in conversation_history:
    manager.add_turn(msg["role"], msg["content"])

# Compact old turns before hitting the budget limit
# Keeps the most recent N turns; older turns are compressed or dropped
removed = manager.compact_older_turns(keep_last=30)

# With a pluggable summarizer:
def my_summarizer(turns: list[dict]) -> str:
    """Call your LLM to summarize old turns."""
    # ... call OpenAI/Claude/Ollama ...
    return "Summary of earlier conversation: ..."

manager.set_summarizer(my_summarizer)
manager.compact_older_turns(keep_last=20)
# Older turns are passed to my_summarizer and replaced with the summary
```

## Suite Integration

Connect `antaris-context` to the rest of the Antaris suite:

```python
from antaris_context import ContextManager
from antaris_memory import MemorySystem
from antaris_router import Router

# Memory-informed priority boosting
mem = MemorySystem("./workspace")
mem.load()
manager = ContextManager(total_budget=8000)
manager.set_memory_client(mem)
# Now optimize_context() boosts sections matching recent memory queries

# Router-driven budget adaptation
router = Router(config_path="./config")
result = router.route(user_input)
manager.set_router_hints(result.routing_hints)
# Section budgets shift based on router's complexity assessment
```

## Templates

Built-in section budget presets for common agent patterns:

```python
# List available templates
templates = ContextManager.get_available_templates()
# {
#   'chatbot':         {'system': 800,  'memory': 1500, 'conversation': 5000, 'tools': 700},
#   'agent_with_tools':{'system': 1200, 'memory': 2000, 'conversation': 3500, 'tools': 1300},
#   'rag_pipeline':    {'system': 600,  'memory': 1000, 'conversation': 4500, 'tools': 1900},
#   'code_assistant':  {'system': 1000, 'memory': 1800, 'conversation': 4000, 'tools': 1200},
#   'balanced':        {'system': 1000, 'memory': 2000, 'conversation': 4000, 'tools': 1000},
# }

# Apply at construction or later
manager = ContextManager(total_budget=8000, template="agent_with_tools")
manager.apply_template("rag_pipeline")  # Switch template mid-session
```

## Content Management

```python
# Add content with priorities
manager.add_content('system', "You are a coding assistant.", priority='critical')
manager.add_content('memory', "User prefers Python examples.", priority='important')
manager.add_content('conversation', messages, priority='normal')
manager.add_content('tools', long_debug_output, priority='optional')

# Priority levels:
# critical  → never truncated (system prompts, safety rules)
# important → removed only when necessary
# normal    → standard selection (conversation history)
# optional  → first to go when space is needed

# Add with query for relevance-based selection
manager.add_content('conversation', messages, query="JWT authentication Flask")

# Set selection strategy
manager.set_strategy('hybrid', recency_weight=0.4, relevance_weight=0.6)
manager.set_strategy('recency', prefer_high_priority=True)
manager.set_strategy('budget', approach='balanced')

# Set compression level
manager.set_compression_level('moderate')  # light, moderate, aggressive
```

## Compression

```python
from antaris_context import MessageCompressor, ImportanceWeightedCompressor, SemanticChunker

# Basic message compression
compressor = MessageCompressor('moderate')
compressed = compressor.compress_message_list(messages, max_content_length=500)
output = compressor.compress_tool_output(long_output, max_lines=20, keep_first=10, keep_last=10)
stats = compressor.get_compression_stats()
print(f"Saved {stats['bytes_saved']} bytes ({stats['compression_ratio']:.1%})")

# Priority-aware compression
iwc = ImportanceWeightedCompressor(keep_top_n=5, compress_middle=True, drop_threshold=0.1)

# Sentence-boundary chunking
chunker = SemanticChunker(min_chunk_size=100, max_chunk_size=500)
chunks = chunker.chunk(long_text)  # → list of SemanticChunk
```

## Adaptive Budgets

```python
# Track usage patterns over time
manager.track_usage()

# Get reallocation suggestions
suggestions = manager.suggest_adaptive_reallocation()
for section, budget in suggestions['suggested_budgets'].items():
    current = suggestions['current_budgets'][section]
    print(f"{section}: {current} → {budget} tokens")

# Apply automatically
manager.apply_adaptive_reallocation(auto_apply=True, min_improvement_pct=10)

# Enable continuous adaptation
manager.enable_adaptive_budgets(target_utilization=0.85)
```

## Cross-Session Snapshots

```python
# Save context state between sessions
manager.save_snapshot("pre-refactor")
snapshot_data = manager.export_snapshot(include_importance_above=0.5)

# Restore later
manager.restore_snapshot("pre-refactor")

# Or reconstruct from exported dict
manager2 = ContextManager.from_snapshot(snapshot_data)

# List saved snapshots
for name in manager.list_snapshots():
    print(name)
```

## Context Analysis

```python
analysis = manager.analyze_context()
print(f"Efficiency score: {analysis['efficiency_score']:.2f}")

for section, data in analysis['section_analysis'].items():
    print(f"{section}: {data['utilization']:.1%} — {data['status']}")

for suggestion in analysis['optimization_suggestions']:
    print(f"  - {suggestion['description']}")
```

## Configuration File

```json
{
  "compression_level": "moderate",
  "strategy": "hybrid",
  "strategy_params": {
    "recency_weight": 0.4,
    "relevance_weight": 0.6
  },
  "section_budgets": {
    "system": 1000,
    "memory": 2000,
    "conversation": 4000,
    "tools": 1000
  },
  "truncation_strategy": "oldest_first",
  "auto_compress": true
}
```

```python
manager = ContextManager(config_file="config.json")
manager.set_compression_level("aggressive")
manager.save_config("updated_config.json")
```

## Complete Agent Example

```python
from antaris_context import ContextManager

manager = ContextManager(total_budget=8000, template="code_assistant")

# System prompt
manager.add_content('system',
    "You are a coding assistant. Always provide working examples.",
    priority='critical')

# User memory
for memory in ["User is learning Python", "Prefers concise explanations"]:
    manager.add_content('memory', memory, priority='important')

# Conversation turns
current_query = "How do I add JWT auth to Flask?"
for turn in conversation_history:
    manager.add_turn(turn["role"], turn["content"])

# Compact if needed
if manager.is_over_budget():
    manager.compact_older_turns(keep_last=20)

# Optimize
result = manager.optimize_context(query=current_query, target_utilization=0.85)

# Render for your provider
messages = manager.render(provider="openai")
response = openai_client.chat.completions.create(model="gpt-4o", messages=messages)
```

## Token Estimation

Uses character-based approximation (~4 chars/token). Fast and good enough for budget management; for exact counts, integrate your model's tokenizer:

```python
# Plug in exact tokenizer (optional)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
manager._estimate_tokens = lambda text: len(enc.encode(text))
```

## What It Doesn't Do

- **No actual tokenization** — uses character-based approximation. For exact counts, plug in your tokenizer.
- **No LLM calls** — purely deterministic. The pluggable `set_summarizer()` is optional; without it, compaction is structural only.
- **No content generation** — selects, compresses, and truncates existing content. Won't paraphrase.
- **No model-specific optimization** — token estimates work generally but aren't tuned per model.
- **No distributed contexts** — manages single context windows. For multi-agent scenarios, use multiple managers.

## Performance

Benchmarks on modern hardware:

| Operation | Throughput |
|-----------|-----------|
| Token estimation | ~100K chars/sec |
| Message compression | ~50K chars/sec |
| Strategy selection | ~10K messages/sec |
| Context analysis | ~1K content items/sec |

Memory scales linearly with content size.

## Why Not tiktoken / LangChain / guidance?

tiktoken counts tokens exactly but provides no selection strategies. LangChain and guidance have heavy dependencies and non-deterministic behavior. antaris-context has zero dependencies, deterministic behavior, file-based configuration, and pluggable strategies — with optional suite integration for memory-informed and router-driven adaptation.

| Library | Dependencies | Deterministic | Strategies | Suite integration |
|---------|-------------|---------------|------------|------------------|
| **antaris-context** | ✅ None | ✅ Yes | ✅ Pluggable | ✅ memory + router |
| tiktoken | Minimal | ✅ Yes | ❌ None | ❌ |
| langchain | Heavy | ❌ No | ⚠️ Limited | ❌ |
| guidance | Heavy | ⚠️ Partial | ❌ None | ❌ |

## Running Tests

```bash
git clone https://github.com/Antaris-Analytics/antaris-context.git
cd antaris-context
python -m pytest tests/ -v
```

All 150 tests pass with zero external dependencies.

## Part of the Antaris Analytics Suite

- **[antaris-memory](https://pypi.org/project/antaris-memory/)** — Persistent memory for AI agents
- **[antaris-router](https://pypi.org/project/antaris-router/)** — Adaptive model routing with SLA enforcement
- **[antaris-guard](https://pypi.org/project/antaris-guard/)** — Security and prompt injection detection
- **antaris-context** — Context window optimization (this package)

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
---

## 🔗 Related Packages

- [**antaris-memory**](https://pypi.org/project/antaris-memory/) - Persistent memory for AI agents
- [**antaris-router**](https://pypi.org/project/antaris-router/) - Adaptive model routing
- [**antaris-guard**](https://pypi.org/project/antaris-guard/) - Security and safety
- [**antaris-pipeline**](https://pypi.org/project/antaris-pipeline/) - Agent orchestration pipeline

---

## 📞 Support

- **Documentation:** [docs.antarisanalytics.ai](https://docs.antarisanalytics.ai)
- **Email:** [dev@antarisanalytics.com](mailto:dev@antarisanalytics.com)
- **Website:** [antarisanalytics.ai](https://antarisanalytics.ai)

---

**Built with ❤️ by Antaris Analytics**  
*Deterministic infrastructure for AI agents*
