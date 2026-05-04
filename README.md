# HackerRank Orchestrate: AI Support Triage Agent

A high-performance, RAG-enabled support agent designed to accurately triage and resolve support tickets for **HackerRank**, **Claude**, and **Visa**. This system leverages a hybrid LLM architecture powered by **Featherless** and **Google Gemini** to provide resilient, secure, and context-aware support automation.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- [Featherless API Key](https://featherless.ai) (Primary)
- [Google Gemini API Key](https://aistudio.google.com/) (Fallback)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ShivSunder0006/HackerRankMay26.git
cd HackerRankMay26

# Install dependencies
pip install -r code/requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory (using `.env.example` as a template):
```env
NVIDIA_API_KEY=your_nvidia_nim_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Build Knowledge Base
The agent uses FAISS vector indices for retrieval. Build them from the provided markdown docs:
```bash
python code/indexer.py
```
*Indices will be saved to `data/index/`.*

### 5. Run the Agent
Execute the triage process against the support tickets:
```bash
python code/main.py
```
*Results are written to `support_tickets/output.csv`.*

---

## 🧠 Approach Overview

Our solution is built on four core pillars: **Safety**, **Retrieval**, **Resilience**, and **Structure**.

### 1. Multi-Stage Safety Pipeline
We implement a defense-in-depth strategy to handle malicious or sensitive tickets:
- **Pre-Safety Filter**: A regex-based engine intercepts prompt injections (e.g., jailbreaks, "ignore instructions") and high-risk security vulnerabilities before they reach the LLM.
- **In-Context Policy**: The system prompt enforces strict adherence to support boundaries, requiring escalation for identity theft, major bugs, or data security concerns.

### 2. Hybrid RAG Architecture
Instead of a single retrieval step, we use an **Agentic Retrieval** pattern:
- **Domain-Specific Indices**: Separate FAISS indices for HackerRank, Claude, and Visa ensure zero cross-domain contamination.
- **Dynamic Tool Calling**: The orchestrator (Llama 3.3 70B) uses specialized search tools to iteratively fetch context, allowing it to handle complex, multi-part queries that a single search might miss.

### 3. Dual-Engine Resilience
To guarantee 100% uptime during high-concurrency evaluation:
- **Primary (NVIDIA NIM)**: We use Llama 3.3 70B Instruct for its superior tool-calling speed and reasoning capabilities.
- **Fallback (Gemini 1.5 Flash)**: If the primary engine encounters rate limits or connection errors, the system automatically switches to Gemini, performing a manual retrieval step to ensure the ticket is still resolved accurately.

### 4. Structured Output & Evaluation
- **Pydantic Validation**: All LLM outputs are validated against a strict schema to ensure compatibility with the automated evaluator.
- **Justification & Confidence**: Every response includes a reasoning trace and a 0-100% confidence score, helping human supervisors understand the agent's decision-making process.

---

## 📁 Repository Structure
```
.
├── code/
│   ├── main.py          # Entry point for ticket processing
│   ├── agent.py         # Core orchestration and safety logic
│   ├── llm_client.py    # Multi-LLM provider abstraction
│   ├── retriever.py     # FAISS search implementation
│   └── indexer.py       # Knowledge base indexing script
├── data/                # Support corpora (Markdown)
├── support_tickets/     # Input tickets and output results
└── AGENTS.md            # Agent rules and log (mandatory)
```

---

## ⚖️ License
This project is part of the HackerRank Orchestrate (May 2026) Hackathon. Internal use only.
