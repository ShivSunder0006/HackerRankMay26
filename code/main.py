"""
main.py — Entry point for the HackerRank Orchestrate support triage agent.

Usage:
    python code/main.py

Steps:
  1. Loads .env (API keys)
  2. Verifies FAISS indexes exist (runs indexer if missing)
  3. Reads support_tickets/support_tickets.csv
  4. Processes each ticket through the agent (verbose output)
  5. Writes results to support_tickets/output.csv
"""

import os
import sys
import pathlib
import subprocess
import time

import pandas as pd
from dotenv import load_dotenv

# ── Repo paths ────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
TICKETS_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV  = REPO_ROOT / "support_tickets" / "output.csv"
INDEX_DIR   = REPO_ROOT / "data" / "index"
CODE_DIR    = pathlib.Path(__file__).parent

OUTPUT_COLUMNS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type", "justification"
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_index():
    """Run indexer.py if FAISS indexes are missing."""
    required = ["hackerrank.faiss", "claude.faiss", "visa.faiss"]
    missing = [f for f in required if not (INDEX_DIR / f).exists()]
    if not missing:
        return
    print("=" * 60)
    print("FAISS indexes not found. Building indexes now...")
    print("This is a one-time step and may take a few minutes.")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(CODE_DIR / "indexer.py")],
        check=True,
    )
    print("=" * 60)
    print("Index build complete.\n")


def print_banner():
    print("\n" + "=" * 60)
    print("  HackerRank Orchestrate — Support Triage Agent")
    print("  Primary: Featherless Qwen3 0.6B")
    print("  Fallback: Gemini 1.5 Flash")
    print("=" * 60 + "\n")


def print_ticket_header(idx: int, total: int, ticket: dict):
    company = str(ticket.get("Company", "None"))
    subject = str(ticket.get("Subject", "(no subject)"))[:60]
    print(f"\n[{idx}/{total}] Company: {company} | Subject: {subject}")
    print(f"  Issue: {str(ticket.get('Issue', ''))[:120].strip()}...")


def print_result(result: dict):
    status_icon = "[OK]" if result["status"] == "replied" else "[!!]"
    print(f"  {status_icon} Status: {result['status'].upper()}")
    print(f"     Area:  {result['product_area']}")
    print(f"     Type:  {result['request_type']}")
    print(f"     Why:   {result['justification'][:100]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load environment variables
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"Warning: .env not found at {env_path}. Relying on system environment.")

    print_banner()

    # Ensure FAISS indexes exist
    ensure_index()

    # Import after dotenv is loaded
    from llm_client import LLMClient
    from agent import SupportAgent

    # Load tickets
    if not TICKETS_CSV.exists():
        print(f"ERROR: Tickets CSV not found: {TICKETS_CSV}")
        sys.exit(1)

    df = pd.read_csv(TICKETS_CSV)
    df = df.fillna("")
    total = len(df)
    print(f"Loaded {total} tickets from {TICKETS_CSV.name}\n")

    # Init agent
    llm = LLMClient()
    agent = SupportAgent(llm)

    # Process tickets
    results = []
    start_time = time.time()

    for idx, row in df.iterrows():
        ticket = row.to_dict()
        ticket_num = idx + 1

        print_ticket_header(ticket_num, total, ticket)
        t0 = time.time()

        result = agent.process_ticket(ticket)

        elapsed = time.time() - t0
        print_result(result)
        print(f"     Time:  {elapsed:.1f}s")

        results.append({
            "issue":         ticket.get("Issue", ""),
            "subject":       ticket.get("Subject", ""),
            "company":       ticket.get("Company", ""),
            "response":      result["response"],
            "product_area":  result["product_area"],
            "status":        result["status"],
            "request_type":  result["request_type"],
            "justification": result["justification"],
        })

    # Write output CSV
    out_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    out_df.to_csv(OUTPUT_CSV, index=False)

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print(f"  ✅  Processing complete in {total_time:.1f}s")
    print(f"  Tickets processed : {total}")
    replied   = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    print(f"  Replied           : {replied}")
    print(f"  Escalated         : {escalated}")
    print(f"  Output written to : {OUTPUT_CSV}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
