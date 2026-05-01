# Support Triage Agent — Source Code

This directory contains the implementation of the AI Support Agent.

## Key Components

### 1. `main.py`
The orchestration entry point. It loads the `support_tickets.csv`, initializes the `SupportAgent`, and saves the results to `output.csv`.

### 2. `agent.py`
The "brain" of the system.
- **Pre-Safety**: Checks for malicious patterns.
- **Primary Loop**: Executes a tool-calling loop using Llama 3.3 70B.
- **Fallback**: Handles transitions to Gemini 1.5 Flash if needed.
- **Validation**: Enforces structured JSON output.

### 3. `llm_client.py`
A unified interface for both NVIDIA NIM and Google Gemini. Handles retries and error propagation for the failover mechanism.

### 4. `retriever.py` & `indexer.py`
The RAG subsystem.
- `indexer.py`: Converts markdown support documents into FAISS vector indices using `all-MiniLM-L6-v2`.
- `retriever.py`: Provides domain-specific semantic search capabilities.

### 5. `tools.py`
Defines the tool schemas (search functions, escalation) that the LLM can call during ticket processing.

## Logic Flow
1. **Input**: Ticket (Subject, Issue, Company).
2. **Filter**: Check for injection/security risks.
3. **Reason**: LLM searches relevant domain indices.
4. **Finalize**: Generate JSON with classification, response, and justification.
5. **Fallback**: If errors occur, use Gemini with direct context injection.
