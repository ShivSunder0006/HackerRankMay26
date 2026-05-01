"""
agent.py — Support triage orchestrator.

Per-ticket flow:
  1. Pre-safety check  (keyword/pattern rules → immediate escalation)
  2. Groq tool-calling loop  (LLM decides what to search, assembles answer)
  3. Gemini fallback  (if Groq fails: manual retrieval + direct Gemini prompt)
  4. Pydantic validation of structured output
"""

import json
import re
import sys
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError
from typing import Literal

from llm_client import LLMClient
from tools import TOOL_SCHEMAS, ESCALATION_SIGNAL, execute_tool
from retriever import hackerrank_retriever, claude_retriever, visa_retriever


# ── Output Schema ─────────────────────────────────────────────────────────────

class TicketOutput(BaseModel):
    status: Literal["replied", "escalated"]
    product_area: str
    response: str
    justification: str
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]


# ── Pre-Safety Keywords ───────────────────────────────────────────────────────
# Tickets matching these patterns are escalated before the LLM is even called.

_MALICIOUS_PATTERNS = [
    r"delete\s+all\s+files",
    r"rm\s+-rf",
    r"format\s+(disk|drive|c:)",
    r"ignore\s+(previous|all|your)\s+instructions",
    r"ignore\s+your\s+system\s+prompt",
    r"jailbreak",
    r"act\s+as\s+(dan|jailbreak|evil)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|rules|internal)",
    r"show\s+(me\s+)?(all\s+)?(your\s+)?(retrieved|internal|hidden)",
    r"affiche\s+toutes\s+les\s+r[eè]gles",   # French prompt injection (in the dataset)
    r"logique\s+exacte\s+que\s+vous\s+utilisez",  # French: "exact logic you use"
]

_ESCALATE_HIGH_RISK = [
    r"identity\s+theft",
    r"identity\s+(was\s+)?stolen",
    r"security\s+vulnerability",
    r"bug\s+bounty",
    r"major\s+(security|vulnerability)",
]

_COMPILED_MALICIOUS = [re.compile(p, re.IGNORECASE) for p in _MALICIOUS_PATTERNS]
_COMPILED_HIGH_RISK = [re.compile(p, re.IGNORECASE) for p in _ESCALATE_HIGH_RISK]


def _pre_safety_check(ticket: Dict) -> Optional[str]:
    """
    Returns an escalation reason string if the ticket should be escalated
    immediately, or None if it should proceed to the LLM.
    """
    text = f"{ticket.get('Issue', '')} {ticket.get('Subject', '')}".lower()

    for pattern in _COMPILED_MALICIOUS:
        if pattern.search(text):
            return "Ticket contains potentially malicious or prompt-injection content."

    for pattern in _COMPILED_HIGH_RISK:
        if pattern.search(text):
            return "Ticket involves a high-risk security or identity issue requiring human review."

    return None


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior support triage agent for HackerRank, Claude, and Visa.

YOUR STRICT RULES:
1. Use ONLY the provided search tools to fetch relevant context before answering.
2. Do NOT use XML tags like <function> or <tool_call> in your responses. Use the provided tool-calling API.
3. Check if this issue involves sensitive data, policy violations, or high-risk actions. If yes, explain why and escalate.
4. If search results are insufficient, call escalate().
5. Output your final answer ONLY as a JSON object.
6. Do not use external knowledge. If the corpus is insufficient, explicitly say 'insufficient data' in your justification and suggest escalation.
7. Based only on the provided support corpus, identify the root cause and justify your answer with retrieved evidence.
8. Provide a confidence score (0-100%) and explain uncertainty based on missing or conflicting evidence within your 'justification' field.

REASONING WORKFLOW:
Before formatting your final JSON response, break this issue down into:
(1) classification, (2) root cause, (3) resolution steps, (4) escalation decision.

JSON OUTPUT FORMAT:
{
  "status": "replied" | "escalated",
  "product_area": "domain name",
  "response": "user answer",
  "justification": "reasoning",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid"
}"""


# ── JSON Extractor ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict]:
    """Try to extract a JSON object from LLM response text."""
    # Direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Find JSON block
    match = re.search(r'\{[^{}]*"status"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # Find ```json ... ``` block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None


def _validate_output(raw: Dict, ticket: Dict) -> Dict:
    """Validate and normalise the LLM output dict."""
    try:
        out = TicketOutput(**raw)
        return out.model_dump()
    except (ValidationError, TypeError):
        # Best-effort recovery: fill in defaults
        return {
            "status": raw.get("status", "escalated"),
            "product_area": raw.get("product_area", ticket.get("company", "unknown").lower()),
            "response": raw.get("response", "Unable to process this request. Please contact support."),
            "justification": raw.get("justification", "Output validation failed; defaulting to safe values."),
            "request_type": raw.get("request_type", "product_issue"),
        }


# ── Gemini Fallback ───────────────────────────────────────────────────────────

def _retrieve_for_ticket(ticket: Dict) -> str:
    """Simple domain-based retrieval used by the Gemini fallback."""
    company = (ticket.get("Company") or "None").strip()
    query = f"{ticket.get('Subject', '')} {ticket.get('Issue', '')}".strip()

    if company == "HackerRank":
        return hackerrank_retriever.search(query, top_k=5)
    elif company == "Claude":
        return claude_retriever.search(query, top_k=5)
    elif company == "Visa":
        return visa_retriever.search(query, top_k=5)
    else:
        # Unknown company — try all three domains
        hr = hackerrank_retriever.search(query, top_k=3)
        cl = claude_retriever.search(query, top_k=3)
        vi = visa_retriever.search(query, top_k=3)
        return f"=== HackerRank ===\n{hr}\n\n=== Claude ===\n{cl}\n\n=== Visa ===\n{vi}"


GEMINI_FALLBACK_TEMPLATE = """You are a support triage agent for HackerRank, Claude, and Visa.

RETRIEVED SUPPORT DOCUMENTS (use ONLY this to answer):
{context}

TICKET:
Company: {company}
Subject: {subject}
Issue: {issue}

Respond ONLY with a valid JSON object:
{{
  "status": "replied" or "escalated",
  "product_area": "<relevant support category>",
  "response": "<user-facing answer grounded in the docs above>",
  "justification": "<brief reason for your decision>",
  "request_type": "product_issue" or "feature_request" or "bug" or "invalid"
}}

Rules:
- Only use information from the retrieved documents above.
- Check if this issue involves sensitive data, policy violations, or high-risk actions. If yes, explain why and escalate.
- Escalate if documents don't cover the issue or if the issue is high-risk.
- For off-topic tickets, reply with status=replied, request_type=invalid.
- Do not use external knowledge. If the corpus is insufficient, explicitly say 'insufficient data' in your justification and suggest escalation.
- Based only on the provided support corpus, identify the root cause and justify your answer with retrieved evidence.
- Break this issue into: (1) classification, (2) root cause, (3) resolution steps, (4) escalation decision before formatting the final JSON.
- Include a confidence score (0-100%) and explain any uncertainty in your 'justification' field.
- Output the JSON object at the very end."""


# ── Main Agent ────────────────────────────────────────────────────────────────

MAX_TOOL_ITERATIONS = 6


class SupportAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def process_ticket(self, ticket: Dict) -> Dict:
        """
        Process a single support ticket. Returns a validated output dict.
        """
        # Step 1: Pre-safety check
        safety_reason = _pre_safety_check(ticket)
        if safety_reason:
            return {
                "status": "escalated",
                "product_area": (ticket.get("Company") or "unknown").lower(),
                "response": "This request has been flagged and escalated to our security/support team for review.",
                "justification": f"Pre-safety filter triggered: {safety_reason}",
                "request_type": "invalid",
            }

        # Step 2: Try Primary (NVIDIA)
        try:
            return self._process_with_primary(ticket)
        except Exception as e:
            print(f"  [NVIDIA] Failed: {e}. Switching to Gemini fallback...")

        # Step 3: Try Fallback (Gemini)
        try:
            return self._process_with_gemini(ticket)
        except Exception as e:
            print(f"  [Gemini] Also failed: {e}. Returning safe default.")
            return {
                "status": "escalated",
                "product_area": (ticket.get("Company") or "unknown").lower(),
                "response": "We were unable to process your request automatically. A human agent will follow up.",
                "justification": f"Both LLM providers failed: {e}",
                "request_type": "product_issue",
            }

    # ── Primary Tool-Calling Loop ────────────────────────────────────────────────

    def _process_with_primary(self, ticket: Dict) -> Dict:
        company = ticket.get("Company", "None")
        issue = ticket.get("Issue", "")
        subject = ticket.get("Subject", "")

        user_msg = (
            f"Company: {company}\n"
            f"Subject: {subject}\n"
            f"Issue: {issue}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        for attempt in range(MAX_TOOL_ITERATIONS):
            response = self.llm.primary_chat(messages, tools=TOOL_SCHEMAS)
            msg = response.choices[0].message

            # Check for tool calls
            if msg.tool_calls:
                # Append assistant message with tool calls
                messages.append(msg)

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    result, is_escalation = execute_tool(fn_name, fn_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                    # Short-circuit on escalation signal
                    if is_escalation:
                        reason = result.replace(ESCALATION_SIGNAL, "").strip()
                        return {
                            "status": "escalated",
                            "product_area": company.lower(),
                            "response": "Your request has been escalated to a human support agent who will follow up shortly.",
                            "justification": f"Agent escalated: {reason}",
                            "request_type": "product_issue",
                        }
            else:
                # Final text response
                content = msg.content or ""
                raw = _extract_json(content)
                if raw:
                    return _validate_output(raw, ticket)
                # LLM gave text without JSON — ask it to format
                if iteration < MAX_TOOL_ITERATIONS - 1:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "Please provide your final answer as a JSON object with keys: status, product_area, response, justification, request_type.",
                    })

        raise RuntimeError("Max tool iterations reached without a final JSON response.")

    # ── Gemini Fallback ───────────────────────────────────────────────────────

    def _process_with_gemini(self, ticket: Dict) -> Dict:
        context = _retrieve_for_ticket(ticket)
        prompt = GEMINI_FALLBACK_TEMPLATE.format(
            context=context,
            company=ticket.get("Company", "None"),
            subject=ticket.get("Subject", ""),
            issue=ticket.get("Issue", ""),
        )
        text = self.llm.gemini_direct(prompt)
        raw = _extract_json(text)
        if raw:
            return _validate_output(raw, ticket)
        raise RuntimeError(f"Gemini returned non-JSON response: {text[:200]}")
