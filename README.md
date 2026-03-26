# Contract Intelligence Analyst

An agentic RAG system that lets you chat with legal contracts. Upload any commercial agreement and ask questions in plain English — the agent finds the right clauses, flags risks, and compares terms across documents.

Built as a portfolio project to demonstrate end-to-end agentic RAG, local LLM deployment, and production-ready ML tooling.

---

## What it does

- **Chat with contracts** — ask "what are the termination clauses?" and get cited, grounded answers
- **Risk flagging** — scans a contract for high-risk language (unlimited liability, auto-renewal, broad IP assignment) with HIGH / MEDIUM / LOW severity
- **Clause extraction** — pull a specific clause type (payment, liability, non-compete, etc.) across all contracts at once
- **Cross-contract comparison** — see how the same clause differs between agreements side by side
- **Source transparency** — every answer cites which contract and chunk it came from

---

## Architecture

```
contract-analyst/
├── ingestion/        ← PDF parsing, chunking, embedding, FAISS index
│   ├── ingest.py
│   ├── faiss.index
│   └── chunks.pkl
├── agent/            ← LangGraph agent with 4 tools
│   └── agent.py
├── app/              ← Streamlit UI
│   └── app.py
└── eval/             ← RAGAS evaluation (coming soon)
```

**Ingestion pipeline:**
1. Contracts are split into 512-character chunks with 64-character overlap using `RecursiveCharacterTextSplitter`
2. Each chunk is embedded with `all-MiniLM-L6-v2` (384 dimensions) via `sentence-transformers`
3. Embeddings are stored in a `FAISS IndexFlatIP` index for cosine similarity search

**Agent layer:**
The LangGraph `StateGraph` implements a ReAct loop — the LLM decides which tool to call based on the query, executes it, reasons over the result, and either calls another tool or returns a final answer.

Four tools:
| Tool | Purpose |
|---|---|
| `vector_search` | General semantic search across all contracts |
| `clause_extractor` | Extract a specific clause type with plain-English summaries |
| `risk_flagger` | Structured risk report with severity tags |
| `contract_comparator` | Side-by-side comparison of a clause across all contracts |

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Mistral 7B via Ollama (fully local, no API keys) |
| Agent framework | LangGraph |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Vector store | FAISS |
| UI | Streamlit |
| Chunking | LangChain RecursiveCharacterTextSplitter |

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

### 1. Clone and create virtual environment

```bash
git clone https://github.com/yourusername/contract-analyst.git
cd contract-analyst
python -m venv venv

# Windows
.\venv\Scripts\Activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the LLM

```bash
ollama pull mistral
```

### 4. Build the index

```bash
python ingestion/ingest.py
```

This chunks and embeds the contracts and saves the FAISS index to `ingestion/`.

### 5. Run the app

```bash
streamlit run app/app.py
```

Or run the CLI agent directly:

```bash
python agent/agent.py
```

---

## Requirements

```
langchain
langchain-community
langchain-ollama
langgraph
faiss-cpu
sentence-transformers
pdfplumber
numpy
streamlit
pandas
```

---

## Example queries

```
Flag all risks in SaaS_Agreement.txt
Compare the liability clauses across contracts
What are the payment terms in the Distribution_Agreement.txt?
Which contracts have auto-renewal clauses?
What are the termination clauses across all contracts?
```

---

## What I learned building this (Note to Myself)

The interesting engineering challenges were not where I expected them.

**Chunking strategy matters more than the LLM.** A termination clause that gets split across two chunks becomes unfindable — no matter how good the model is. Getting the chunk size (512 chars) and overlap (64 chars) right made a bigger difference to retrieval quality than any prompt engineering.

**Agentic routing is genuinely useful at scale.** With a small corpus the LLM could answer from memory. The value of having four distinct tools becomes clear when you want different _types_ of answers — a semantic search answer looks completely different from a structured risk report, and forcing them through the same pipeline produces worse results for both.

**Local LLMs are production-viable for focused tasks.** Mistral 7B on Ollama handles tool-calling and structured output reliably for this domain. The latency is higher than a cloud API but the zero-cost, offline-capable, privacy-preserving tradeoff is worth it for enterprise contract analysis.

---

## Roadmap

- [ ] RAGAS evaluation suite with faithfulness and relevance scores
- [ ] PDF upload via Streamlit UI (re-index on the fly)
- [ ] Concept drift detection on clause distributions using diagnost
- [ ] Support for real CUAD dataset (500+ annotated commercial contracts)

---

## Acknowledgements

- [CUAD Dataset](https://www.atticusprojectai.org/cuad) — Contract Understanding Atticus Dataset
- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [Ollama](https://ollama.com) — local LLM serving
