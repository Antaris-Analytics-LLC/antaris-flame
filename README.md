# antaris-flame

**Memory, safety, and context for AI agent bots — Flame tier (5,000 memories)**

[![PyPI version](https://badge.fury.io/py/antaris-flame.svg)](https://pypi.org/project/antaris-flame/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

antaris-flame is the mid-tier of the Antaris bot runtime — bundling persistent memory, safety screening, and context optimization in a single install. Capped at **5,000 memories** for production deployments.

## What's Included

- **antaris-memory** — BM25 + co-occurrence semantic search, audit logging, memory decay, recovery
- **antaris-guard** — Prompt injection detection, PII redaction, rate limiting, compliance templates
- **antaris-context** — Token budget management, context window optimization, compression strategies

## Tier Limits

| Tier | Memory Cap | Package |
|------|-----------|---------|
| Spark | 500 memories | `antaris-spark` |
| **Flame** | 5,000 memories | `antaris-flame` |
| Forge | 25,000 memories | `antaris-forge` |

Upgrade path: `pip install antaris-forge` — drop-in replacement, no code changes.

## Installation

```bash
pip install antaris-flame

# With optional semantic reranking (MiniLM, ~22MB model):
pip install antaris-flame[semantic]
```

## Quick Start

```python
from antaris_memory import MemorySystem
from antaris_guard import PromptGuard
from antaris_context import ContextManager

# Memory — capped at 5,000 entries by default
mem = MemorySystem("./workspace")
mem.load()
mem.ingest("User prefers concise answers", source="conversation")
results = mem.search("user preferences")

# Guard
guard = PromptGuard()
if not guard.is_safe(user_input):
    return  # block before reaching model

# Context
ctx = ContextManager(total_budget=8000)
ctx.set_memory_client(mem)
ctx.add_content("conversation", messages)
ctx.optimize_context()
```

## GitHub

<https://github.com/Antaris-Analytics-LLC/antaris-flame>

## License

Apache 2.0 — see [LICENSE](LICENSE)
