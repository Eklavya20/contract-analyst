# Contract Intelligence Analyst

An agentic RAG system that lets you chat with legal contracts. Upload any commercial agreement and ask questions in plain English. The agent finds relevant clauses, flags risks, and compares terms across documents.

This project is built as a portfolio piece to demonstrate end-to-end agentic RAG, local LLM deployment, and production-ready ML tooling.

---

## What it does

* **Chat with contracts**
  Ask questions like "What are the termination clauses?" and get grounded answers with citations.

* **Risk flagging**
  Scans contracts for high-risk language such as unlimited liability, auto-renewal, and broad IP assignment. Outputs severity levels: HIGH, MEDIUM, LOW.

* **Clause extraction**
  Extract specific clause types such as payment, liability, or non-compete across all contracts.

* **Cross-contract comparison**
  Compare how the same clause differs across agreements side by side.

* **Source transparency**
  Every answer includes references to the exact contract and chunk it came from.

---

## Architecture

```
contract-analyst/
├── ingestion/        # PDF parsing, chunking, embedding, FAISS index
│   ├── ingest.py
│   ├── faiss.index
│   └── chunks.pkl
├── agent/            # LangGraph agent with tools
│   └── agent.py
├── app/              # Streamlit UI
│   └── app.py
└── eval/             # RAG evaluation (coming soon)
```

### Ingestion pipeline

* Contracts are split into 512-character chunks with 64-character overlap using RecursiveCharacterTextSplitter
* Each chunk is embedded using `all-MiniLM-L6-v2` (384 dimensions) from sentence-transformers
* Embeddings are stored in a FAISS `IndexFlatIP` index for cosine similarity search

### Agent layer

The LangGraph StateGraph implements a ReAct loop. The LLM decides which tool to call, executes it, reasons over the result, and either calls another tool or returns a final answer.

### Tools

| Tool                  | Purpose                                  |
| --------------------- | ---------------------------------------- |
| `vector_search`       | General semantic search across contracts |
| `clause_extractor`    | Extract specific clauses with summaries  |
| `risk_flagger`        | Generate structured risk reports         |
| `contract_comparator` | Compare clauses across contracts         |

---

## Stack

| Layer           | Technology                                 |
| --------------- | ------------------------------------------ |
| LLM             | Mistral 7B via Ollama (local, no API keys) |
| Agent framework | LangGraph                                  |
| Embeddings      | sentence-transformers (all-MiniLM-L6-v2)   |
| Vector store    | FAISS                                      |
| UI              | Streamlit                                  |
| Chunking        | LangChain RecursiveCharacterTextSplitter   |

---

## Setup

### Prerequisites

* Python 3.10 or higher
* Ollama installed

### 1. Clone and create a virtual environment

```
git clone https://github.com/yourusername/contract-analyst.git
cd contract-analyst
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Pull the model

```
ollama pull mistral
```

### 4. Build the index

```
python ingestion/ingest.py
```

This step chunks and embeds contracts, then saves the FAISS index in the `ingestion/` directory.

### 5. Run the app

```
streamlit run app/app.py
```

Or run the agent directly:

```
python agent/agent.py
```

---

## Requirements

* langchain
* langchain-community
* langchain-ollama
* langgraph
* faiss-cpu
* sentence-transformers
* pdfplumber
* numpy
* streamlit
* pandas

---

## Example queries

* Flag all risks in `SaaS_Agreement.txt`
* Compare liability clauses across contracts
* What are the payment terms in `Distribution_Agreement.txt`?
* Which contracts include auto-renewal clauses?
* Show termination clauses across all contracts

---

## What I learned (Note to myself)

Some of the most important challenges were not where I expected them.

* **Chunking matters more than the model**
  If a clause is split across chunks, it becomes much harder to retrieve. Tuning chunk size (512 characters) and overlap (64 characters) improved retrieval quality more than prompt changes.

* **Agent routing becomes valuable with scale**
  A single pipeline works for small datasets, but different query types need different tools. Semantic search, structured risk reports, and comparisons require different handling.

* **Local LLMs are viable for focused use cases**
  Mistral 7B running on Ollama can reliably handle tool usage and structured outputs. Latency is higher than cloud APIs, but the tradeoff is strong: no cost, offline capability, and better data privacy.

---

## Roadmap

* Add RAG evaluation using RAGAS (faithfulness and relevance)
* Enable PDF uploads via the Streamlit UI with dynamic re-indexing
* Add concept drift detection for clause distributions
* Support the CUAD dataset (500+ annotated contracts)

---

## Acknowledgements

* CUAD Dataset (Contract Understanding Atticus Dataset)
* LangGraph for agent orchestration
* Ollama for local model serving
