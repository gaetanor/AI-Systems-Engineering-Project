# Legal RAG System: Italy, Estonia, Slovenia

## Project Overview
This project implements an advanced **Retrieval-Augmented Generation (RAG)** system applied to the domain of **European Civil Law**, specifically comparing **Italy, Estonia, and Slovenia**.

The system explores two competing architectural strategies to handle multi-jurisdictional legal queries, enhanced by state-of-the-art retrieval techniques like **Semantic Reranking**, **Query Expansion**, and **Context-Aware Memory**.

## 1. Architectures

### Task A: Single ReAct Agent
A single agent that operates on a **Unified Vector Index**.
- **Logic**: Uses a ReAct (Reasoning + Acting) loop to determine retrieval needs.
- **Retrieval Strategy**:
  - Extracts metadata criteria (country, law).
  - Fetches $k$ documents *per criterion*.
  - Performs a global sort to select the absolute top-k most relevant documents across all jurisdictions.

### Task B: Multi-Agent Supervisor
A hierarchical system consisting of:
- **Supervisor**: A Router Agent that classifies intent and directs queries to specialized workers.
- **Workers**: Isolated agents (Italy_Agent, Estonia_Agent, Slovenia_Agent) with dedicated Vector DBs.
- **Retrieval Strategy**:
  - Each activated agent guarantees $k$ documents.
  - Supports parallel execution and comparative synthesis.

---

## 2. Key Features

### Advanced Reasoning & Memory
- **Contextualizer Module**: Automatically detects if a user query depends on chat history and rewrites it into a standalone query. If the query is new, it **drops the history** to prevent "Context Poisoning".
- **Semantic Reranking**: A two-stage retrieval process:
  1.  **Broad Fetch**: Retrieves a large pool of candidates.
  2.  **Cross-Encoder**: Re-scores and selects the top-k strictly relevant documents.
- **Query Expansion**: Uses an LLM to refine user queries with domain-specific legal terminology before search.

### Safety & Governance
- **PII Guardrails**: Real-time sanitation of sensitive data (names, emails, credit cards) before processing.
- **Consistency Check**: Optional validation layer to ensure generated answers are supported by citations.

### Evaluation & Feedback
- **RAGAS Dashboard**: Integrated evaluation suite to measure **Faithfulness**, **Answer Relevancy**, **Context Precision**, and **Context Recall**.
- **Batch Inference**: Supports uploading a JSON test set (`question` + `ground_truth`) to automatically run the engine and calculate metrics.
- **Data Flywheel**: Users can provide explicit feedback which is logged for future fine-tuning.

---

## 📂 Repository Structure

```text
├── data/                      # Legal Corpora
├── logs/                      # System Logs
│   ├── interactions.json      # Session logs
│   └── feedback.jsonl         # User feedback collection
├── pages/                     # Streamlit Interface Pages
│   ├── 1_Chat_Interface.py    # Main Chat UI (Task A vs Task B selection)
│   └── 2_Evaluation_Dashboard.py # RAGAS Analytics & Batch Testing
├── src/                       # Core Application Logic
│   ├── core/                  # Configuration, LLM Factory, Session Logger
│   ├── engines/               # RAG Engines
│   │   ├── single_react.py    # Task A Logic 
│   │   ├── multi_supervisor.py# Task B Logic 
│   │   └── query_router.py    # Main Orchestrator
│   └── guardrails/            # PII & Security Logic
├── Settings.py                # Application Entry Point
├── Dockerfile                 # Container setup
├── requirements.txt           # Python dependencies
└── .env                       # Environment Variables