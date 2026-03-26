import os
import re
import time
import pickle
import operator
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated

# ── Config ───────────────────────────────────────────────────────────────────
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH   = "ingestion/faiss.index"
CHUNKS_PATH  = "ingestion/chunks.pkl"

TOP_K        = 3

RISK_PATTERNS = [
    "unlimited liability",
    "auto-renewal",
    "automatic renewal",
    "ip assignment",
    "assigns exclusively",
    "non-compete",
    "shall not work",
    "perpetual license",
    "irrevocable",
    "indemnify and hold harmless",
    "sole discretion",
    "without cause",
    "immediately terminate",
]

# ── Load resources once at module level ──────────────────────────────────────
print("Loading embedding model and index...")
_embedder = SentenceTransformer(EMBED_MODEL)
_index    = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    _chunks = pickle.load(f)

_llm = ChatOllama(model="mistral", temperature=0)
print(f"  Ready — {_index.ntotal} chunks indexed\n")


# ── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# ── Tools ────────────────────────────────────────────────────────────────────
@tool
def vector_search(query: str) -> str:
    """
    Semantic search over all contracts.
    Returns the top matching chunks with source metadata.
    Use this first for any general question about contract content.
    """
    q_vec = _embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(q_vec, k=TOP_K)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        c = _chunks[idx]
        results.append(
            f"[{c['filename']} | chunk {c['chunk_id']} | score={score:.3f}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(results)


@tool
def clause_extractor(clause_type: str) -> str:
    """
    Extract a specific clause type from all contracts.
    Use for targeted questions like 'show me all termination clauses'
    or 'what are the payment terms across contracts'.
    clause_type examples: termination, payment, liability, ip assignment,
    non-compete, governing law, auto-renewal.
    """
    q_vec = _embedder.encode([clause_type], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(q_vec, k=6)

    chunks_text = "\n\n".join(
        f"[{_chunks[i]['filename']}]\n{_chunks[i]['text'][:300]}"
        for i in indices[0]
    )

    prompt = f"""You are a legal contract analyst. Extract all '{clause_type}' clauses from the excerpts below.
For each clause found state: contract name, exact clause text, one-line plain-English summary.

{chunks_text}"""

    response = _llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def risk_flagger(contract_name: str) -> str:
    """
    Scan a specific contract for high-risk language and flag potential issues.
    Use when asked to review a contract for risks or red flags.
    contract_name: the filename e.g. SaaS_Agreement.txt
    """
    contract_chunks = [
        c for c in _chunks if contract_name.lower() in c["filename"].lower()
    ]

    if not contract_chunks:
        available = list({c["filename"] for c in _chunks})
        return f"Contract '{contract_name}' not found. Available: {available}"

    top_chunks = contract_chunks[:2]
    full_text = "\n\n".join(c["text"][:300] for c in top_chunks)

    flags = []
    for pattern in RISK_PATTERNS:
        if pattern.lower() in full_text.lower():
            flags.append(f"  - '{pattern}' detected")

    pattern_summary = (
        "Risk patterns detected:\n" + "\n".join(flags)
        if flags else "No standard risk patterns detected."
    )

    prompt = f"""You are a legal risk analyst. Review this contract excerpt and identify:
1. Clauses that heavily favour one party
2. Missing standard protections
3. Unusual or aggressive terms
4. Auto-renewal, unlimited liability, or broad IP assignment clauses

Contract: {contract_name}
{full_text}

{pattern_summary}

Give a structured risk report with severity HIGH / MEDIUM / LOW for each finding."""

    response = _llm.invoke([HumanMessage(content=prompt)])
    return f"{pattern_summary}\n\nLLM Risk Analysis:\n{response.content}"


@tool
def contract_comparator(clause_type: str) -> str:
    """
    Compare how a specific clause type differs across all contracts.
    Use when asked to compare contracts or find differences between agreements.
    clause_type examples: termination, liability, payment, ip ownership.
    """
    q_vec = _embedder.encode([clause_type], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(q_vec, k=9)

    by_contract: dict[str, list[str]] = {}
    for i in indices[0]:
        c = _chunks[i]
        by_contract.setdefault(c["filename"], []).append(c["text"][:250])

    sections = "\n\n".join(
        f"=== {fname} ===\n" + "\n".join(texts)
        for fname, texts in by_contract.items()
    )

    prompt = f"""You are a legal analyst. Compare the '{clause_type}' clauses across these contracts.
Produce a structured summary showing:
- What each contract says about '{clause_type}'
- Key differences between contracts
- Which contract is most favourable to each party

{sections}"""

    response = _llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ── Agent graph ───────────────────────────────────────────────────────────────
tools = [vector_search, clause_extractor, risk_flagger, contract_comparator]
llm_with_tools = _llm.bind_tools(tools)

SYSTEM_PROMPT = """You are an expert legal contract analyst with deep knowledge of commercial contracts.
You have access to a database of contracts and four tools:
- vector_search: general semantic search across all contracts
- clause_extractor: extract a specific clause type (termination, payment, liability etc.)
- risk_flagger: scan a specific contract for risks and red flags
- contract_comparator: compare how a clause differs across all contracts

Always cite which contract and clause your answer comes from.
Be precise, structured, and flag any risks you notice even if not asked.
Always use tools to retrieve information — never answer from memory alone."""


def call_model(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()


# ── Chat function ─────────────────────────────────────────────────────────────
def chat(query: str, history: list = None) -> tuple[str, list]:
    """Single turn — keeps only last 2 messages to control token usage."""
    history = history or []
    history.append(HumanMessage(content=query))
    trimmed = history[-2:]

    result = app.invoke({"messages": trimmed})
    response = result["messages"][-1]
    history.append(response)
    return response.content, history


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Contract Intelligence Agent — mistral")
    print("Commands: 'quit' to exit, 'clear' to reset memory")
    print("-" * 50)

    history = []
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "clear":
            history = []
            print("Memory cleared.")
            continue

        answer, history = chat(query, history)
        print(f"\nAgent: {answer}")