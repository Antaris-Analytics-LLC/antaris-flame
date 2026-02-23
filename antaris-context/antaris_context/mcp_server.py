"""
Antaris Context MCP Server

Exposes antaris-context as MCP tools for any MCP-enabled agent or Claude integration.

Tools:
  - compress_context: Compress message history to fit within a token budget
  - get_context_profile: Analyze context usage and importance distribution

Usage:
    python -m antaris_context.mcp_server
    # or
    from antaris_context.mcp_server import create_server, run_server

MCP library (mcp>=1.0.0) must be installed separately:
    pip install mcp
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Graceful degradation when mcp is not installed
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    MCP_AVAILABLE = False
    FastMCP = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Antaris Context imports (always available — zero-dependency package)
# ---------------------------------------------------------------------------
from .manager import ContextManager
from .compressor import MessageCompressor
from .profiler import ContextProfiler
from .window import ContextWindow


# ---------------------------------------------------------------------------
# Tool implementation helpers (pure functions — easy to unit-test)
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-style)."""
    return max(1, len(text) // 4)


def _compress_context_impl(
    messages: List[Dict[str, Any]],
    budget_tokens: int,
) -> Dict[str, Any]:
    """
    Core implementation for compress_context.

    Compresses a list of message dicts to fit within budget_tokens by:
      1. Trimming long content per message via MessageCompressor
      2. Dropping oldest non-system messages if still over budget

    Args:
        messages:      List of {role, content} message dicts.
        budget_tokens: Maximum total token count for the output messages.

    Returns:
        dict with keys:
            messages (list)         — Compressed message list.
            original_tokens (int)   — Estimated tokens before compression.
            compressed_tokens (int) — Estimated tokens after compression.
            messages_dropped (int)  — Number of messages removed entirely.
            compression_ratio (float) — original / compressed ratio.
    """
    if not messages:
        return {
            "messages": [],
            "original_tokens": 0,
            "compressed_tokens": 0,
            "messages_dropped": 0,
            "compression_ratio": 1.0,
        }

    # Estimate original size
    original_tokens = sum(
        _count_tokens(str(m.get("content", ""))) for m in messages
    )

    # Phase 1: compress individual message content
    compressor = MessageCompressor()
    max_content_length = max(200, (budget_tokens * 4) // max(1, len(messages)))
    compressed = compressor.compress_message_list(
        [dict(m) for m in messages],
        max_content_length=max_content_length,
    )

    # Phase 2: drop oldest non-system messages if still over budget
    messages_dropped = 0
    while True:
        total = sum(_count_tokens(str(m.get("content", ""))) for m in compressed)
        if total <= budget_tokens:
            break
        # Find first non-system message to drop
        dropped = False
        for i, m in enumerate(compressed):
            if m.get("role", "") != "system":
                compressed.pop(i)
                messages_dropped += 1
                dropped = True
                break
        if not dropped:
            # Only system messages left — truncate the last one hard
            if compressed:
                last = compressed[-1]
                content = str(last.get("content", ""))
                last["content"] = content[: budget_tokens * 4]
            break

    compressed_tokens = sum(
        _count_tokens(str(m.get("content", ""))) for m in compressed
    )
    ratio = round(original_tokens / max(1, compressed_tokens), 3)

    return {
        "messages": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "messages_dropped": messages_dropped,
        "compression_ratio": ratio,
    }


def _get_context_profile_impl(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Core implementation for get_context_profile.

    Analyzes a list of messages and returns a profile of token distribution,
    role breakdown, and importance estimates.

    Args:
        messages: List of {role, content} message dicts.

    Returns:
        dict with keys:
            total_tokens (int)         — Estimated total tokens.
            role_breakdown (dict)      — Per-role token counts and message counts.
            avg_tokens_per_message (float)
            longest_message (dict)     — {role, tokens, preview} for the longest msg.
            token_distribution (dict)  — Percentage per role.
            summary (str)              — Human-readable one-liner.
    """
    if not messages:
        return {
            "total_tokens": 0,
            "role_breakdown": {},
            "avg_tokens_per_message": 0.0,
            "longest_message": {},
            "token_distribution": {},
            "summary": "No messages to profile.",
        }

    role_breakdown: Dict[str, Dict[str, Any]] = {}
    longest_tokens = 0
    longest_msg: Dict[str, Any] = {}

    for m in messages:
        role = m.get("role", "unknown")
        content = str(m.get("content", ""))
        tokens = _count_tokens(content)

        if role not in role_breakdown:
            role_breakdown[role] = {"messages": 0, "tokens": 0}
        role_breakdown[role]["messages"] += 1
        role_breakdown[role]["tokens"] += tokens

        if tokens > longest_tokens:
            longest_tokens = tokens
            longest_msg = {
                "role": role,
                "tokens": tokens,
                "preview": content[:120] + ("…" if len(content) > 120 else ""),
            }

    total_tokens = sum(v["tokens"] for v in role_breakdown.values())
    avg = round(total_tokens / max(1, len(messages)), 1)

    token_distribution = {
        role: round(data["tokens"] / max(1, total_tokens) * 100, 1)
        for role, data in role_breakdown.items()
    }

    summary = (
        f"{len(messages)} messages, ~{total_tokens} tokens. "
        f"Roles: {', '.join(role_breakdown.keys())}."
    )

    return {
        "total_tokens": total_tokens,
        "role_breakdown": role_breakdown,
        "avg_tokens_per_message": avg,
        "longest_message": longest_msg,
        "token_distribution": token_distribution,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# MCP server factory
# ---------------------------------------------------------------------------

def create_server(name: str = "antaris-context") -> "FastMCP":  # type: ignore[return]
    """
    Create and return a configured FastMCP server instance.

    The server exposes two tools:
      • compress_context    — Compress message history to fit a token budget
      • get_context_profile — Analyze context usage and importance distribution

    Args:
        name: Server name shown to MCP clients (default "antaris-context").

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required to use the MCP server.\n"
            "Install it with: pip install mcp"
        )

    mcp = FastMCP(
        name,
        instructions=(
            "Antaris Context provides context window management tools. "
            "Use compress_context to fit messages within a token budget, and "
            "get_context_profile to analyze how tokens are distributed across roles."
        ),
    )

    # ------------------------------------------------------------------
    # Tool: compress_context
    # ------------------------------------------------------------------
    @mcp.tool(
        name="compress_context",
        description=(
            "Compress a list of chat messages to fit within a target token budget. "
            "Applies content compression to individual messages, then drops oldest "
            "non-system messages if still over budget. "
            "Returns the compressed message list plus compression statistics."
        ),
    )
    def compress_context(
        messages: List[Dict[str, Any]],
        budget_tokens: int,
    ) -> Dict[str, Any]:
        """
        Compress message history to fit within budget_tokens.

        Args:
            messages:      List of {role, content} message dicts.
            budget_tokens: Maximum total token count for the output.

        Returns:
            {messages, original_tokens, compressed_tokens,
             messages_dropped, compression_ratio}
        """
        return _compress_context_impl(messages, budget_tokens)

    # ------------------------------------------------------------------
    # Tool: get_context_profile
    # ------------------------------------------------------------------
    @mcp.tool(
        name="get_context_profile",
        description=(
            "Analyze a list of messages and return a context usage profile. "
            "Shows total token estimates, per-role breakdown, average message size, "
            "the longest message, and percentage distribution by role. "
            "Useful for deciding when and how to compress."
        ),
    )
    def get_context_profile(
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze context usage and importance distribution.

        Args:
            messages: List of {role, content} message dicts.

        Returns:
            {total_tokens, role_breakdown, avg_tokens_per_message,
             longest_message, token_distribution, summary}
        """
        return _get_context_profile_impl(messages)

    return mcp


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_server(transport: str = "stdio") -> None:
    """
    Create and run the Antaris Context MCP server.

    Args:
        transport: MCP transport — "stdio" (default), "sse", or "streamable-http".

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    server = create_server()
    server.run(transport=transport)  # type: ignore[arg-type]


def main() -> None:
    """
    CLI entry point for ``antaris-context-mcp``.

    Usage:
        antaris-context-mcp                  # stdio transport (default)
        antaris-context-mcp --transport sse  # SSE transport
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="antaris-context-mcp",
        description="Run the Antaris Context MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport to use (default: stdio).",
    )
    args = parser.parse_args()

    if not MCP_AVAILABLE:
        print(
            "ERROR: The 'mcp' package is required.\n"
            "Install it with: pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    run_server(transport=args.transport)


# ---------------------------------------------------------------------------
# __main__ support: python -m antaris_context.mcp_server
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    main()
