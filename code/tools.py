"""
tools.py — Tool schemas (OpenAI/Groq compatible) and callables for the LLM orchestrator.
"""

import json
from retriever import hackerrank_retriever, claude_retriever, visa_retriever

# ── Tool Schemas ──────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_hackerrank",
            "description": (
                "Search the HackerRank support corpus for relevant documentation. "
                "Use for tickets about HackerRank assessments, tests, candidates, interviews, "
                "subscriptions, billing, integrations, certifications, or platform issues."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query to find relevant support docs.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_claude",
            "description": (
                "Search the Claude Help Center corpus for relevant documentation. "
                "Use for tickets about Claude AI, Anthropic products, Claude API, "
                "Claude teams/enterprise, desktop/mobile apps, privacy, and data handling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query to find relevant support docs.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_visa",
            "description": (
                "Search the Visa support corpus for relevant documentation. "
                "Use for tickets about Visa cards, payments, fraud, disputes, "
                "travel, merchants, card replacement, or Visa services."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query to find relevant support docs.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": (
                "Escalate this ticket to a human support agent. Use when: "
                "(1) fraud, stolen card/identity, financial loss, or account compromise; "
                "(2) the user needs admin-level action only a human can perform; "
                "(3) the corpus does not contain enough information to answer safely; "
                "(4) ambiguous high-risk billing or legal situations; "
                "(5) sensitive or potentially harmful requests where guessing would cause harm. "
                "Do NOT use this just because a question is complex — only when human intervention is truly required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this ticket requires human escalation.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]


# ── Tool Executor ─────────────────────────────────────────────────────────────

ESCALATION_SIGNAL = "##ESCALATE##"


def execute_tool(name: str, arguments: dict) -> tuple[str, bool]:
    """
    Execute tool by name. Returns (result_string, is_escalation).
    """
    if name == "search_hackerrank":
        return hackerrank_retriever.search(arguments.get("query", "")), False
    elif name == "search_claude":
        return claude_retriever.search(arguments.get("query", "")), False
    elif name == "search_visa":
        return visa_retriever.search(arguments.get("query", "")), False
    elif name == "escalate":
        reason = arguments.get("reason", "No reason provided")
        return f"{ESCALATION_SIGNAL} {reason}", True
    else:
        return f"Unknown tool: {name}", False
