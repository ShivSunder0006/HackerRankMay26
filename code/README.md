# Support Triage Agent — Code

This directory contains the full source code for the HackerRank Orchestrate support triage agent.

## Architecture

```
Ticket (CSV row)
    │
    ▼
[Pre-Safety Check]  ← keyword/regex rules → immediate escalation for malicious/injection tickets
    │
    ▼
[Groq LLM Orchestrator]  ← llama-3.3-70b-versatile with tool calling
    ├─ search_hackerrank(query)  → FAISS over 438 docs
    ├─ search_claude(query)      → FAISS over 322 docs
    ├─ search_visa(query)        → FAISS over 14 docs
    └─ escalate(reason)          → short-circuit to human
    │
    ▼  (if Groq fails after 1 retry)
[Gemini 1.5 Flash Fallback]  ← manual retrieval + direct prompt
    │
    ▼
Structured Output  →  output.csv
{status, product_area, response, justification, request_type}
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — reads CSV, runs agent, writes output |
| `agent.py` | Orchestrator loop, pre-safety checks, fallback logic |
| `llm_client.py` | Groq primary + Gemini 1.5 Flash fallback |
| `tools.py` | Tool schemas + executors for the LLM |
| `retriever.py` | FAISS-backed domain retrievers (lazy-loaded) |
| `indexer.py` | One-time index builder — run this first |
| `requirements.txt` | Pinned Python dependencies |

## Setup

### 1. Install dependencies

```bash
pip install -r code/requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` in the repo root and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

> Keys are read from environment variables only — never hardcoded.

### 3. Build the FAISS index (one-time)

```bash
python code/indexer.py
```

This walks `data/hackerrank/`, `data/claude/`, `data/visa/`, chunks all `.md` files,
embeds them with `sentence-transformers/all-MiniLM-L6-v2` (local, no API key needed),
and saves FAISS indexes to `data/index/` (gitignored).

**Estimated time:** 2–5 minutes depending on hardware.

### 4. Run the agent

```bash
python code/main.py
```

This processes all 30 tickets in `support_tickets/support_tickets.csv` and writes
predictions to `support_tickets/output.csv`.

## Design Decisions

### Why tool calling instead of plain RAG?
Tool calling gives the LLM control over *which* domain to search and *when* to escalate,
rather than applying a fixed retrieval pipeline to every ticket. This means a ticket with
`company=None` can be intelligently routed based on content, and an obviously high-risk
ticket can be escalated without wasting a retrieval round-trip.

### Why sentence-transformers (local)?
No API cost, no rate limits, deterministic embeddings. `all-MiniLM-L6-v2` is fast (384-dim)
and performs well for English support doc retrieval.

### Why FAISS IndexFlatIP?
Exact nearest-neighbour with inner product (= cosine similarity after L2 normalization).
For 774 documents this is fast enough without approximation trade-offs.

### Escalation strategy
Two layers:
1. **Pre-LLM regex rules** — catch obvious prompt-injection and high-risk keywords immediately.
2. **LLM `escalate()` tool** — the model can call this when retrieved docs aren't sufficient
   or when the situation clearly needs a human.

### Groq → Gemini fallback
Groq is retried once on transient errors (rate limits, 5xx). On second failure, the agent
falls back to Gemini 1.5 Flash using a simpler direct-prompt approach (retrieval done in Python,
context passed into the prompt). Both paths produce the same structured JSON output.

## Output Schema

`support_tickets/output.csv` columns:

| Column | Values |
|--------|--------|
| `issue` | Original ticket issue |
| `subject` | Original ticket subject |
| `company` | Original company field |
| `response` | User-facing answer |
| `product_area` | Support category |
| `status` | `replied` \| `escalated` |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` |
| `justification` | Explanation of the agent's decision |
