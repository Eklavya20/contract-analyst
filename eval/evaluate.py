import os
import sys
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig  # ✅ FIX

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH  = "ingestion/faiss.index"
CHUNKS_PATH = "ingestion/chunks.pkl"
TOP_K       = 5
RESULTS_PATH = "eval/results.csv"

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading index and embedder...")
_embedder = SentenceTransformer(EMBED_MODEL)
_index    = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    _chunks = pickle.load(f)

print(f"  {_index.ntotal} chunks loaded\n")


# ── Golden eval dataset ───────────────────────────────────────────────────────
GOLDEN_SET = [
    {
        "question": "What is the commission rate in the Affiliate Marketing Agreement?",
        "ground_truth": "The Affiliate earns 8% commission on net sales generated through affiliate links.",
    },
    {
        "question": "How many days notice is required to terminate the Affiliate Marketing Agreement without cause?",
        "ground_truth": "Either party may terminate with 30 days written notice.",
    },
    {
        "question": "What is the annual license fee in the Software License Agreement?",
        "ground_truth": "The Licensee shall pay a $50,000 annual license fee, due January 1 each year.",
    },
    {
        "question": "What happens to IP modifications suggested by the Licensee in the Software License Agreement?",
        "ground_truth": "Any modifications suggested by Licensee and implemented by Licensor become property of Licensor.",
    },
    {
        "question": "What is the monthly rent in the Commercial Lease Agreement?",
        "ground_truth": "Base rent is $25,000 per month, increasing 3% annually on each anniversary.",
    },
    {
        "question": "How long is the lease term in the Commercial Lease Agreement?",
        "ground_truth": "The lease term is 5 years commencing April 1, 2024 and ending March 31, 2029.",
    },
    {
        "question": "What is the minimum annual purchase requirement in the Distribution Agreement?",
        "ground_truth": "The Distributor must purchase a minimum of $500,000 of products per calendar year.",
    },
    {
        "question": "What is the non-compete period after termination in the Distribution Agreement?",
        "ground_truth": "Upon termination, Distributor may not distribute competing products for 12 months.",
    },
    {
        "question": "How long do confidentiality obligations last in the NDA Agreement?",
        "ground_truth": "Confidentiality obligations survive for 3 years after disclosure.",
    },
    {
        "question": "What governing law applies to the NDA Agreement?",
        "ground_truth": "Delaware law governs. Disputes are resolved in Delaware courts.",
    },
    {
        "question": "What is the base salary in the Employment Contract?",
        "ground_truth": "The base salary is $120,000 per year, payable bi-weekly.",
    },
    {
        "question": "What are the termination without cause terms in the Employment Contract?",
        "ground_truth": "Company may terminate with 2 weeks notice and 4 weeks severance pay.",
    },
    {
        "question": "What is the monthly subscription fee in the SaaS Agreement?",
        "ground_truth": "Customer pays $5,000 per month, billed annually in advance at $60,000.",
    },
    {
        "question": "Does the SaaS Agreement have an auto-renewal clause?",
        "ground_truth": "Yes. The Agreement renews automatically for successive one-year terms unless cancelled 60 days prior.",
    },
    {
        "question": "What is the payment term in the Supply Agreement?",
        "ground_truth": "Net 45 days from delivery, with a 2% discount for payment within 10 days.",
    },
    {
        "question": "How quickly must the Supplier deliver in the Supply Agreement?",
        "ground_truth": "Supplier shall deliver within 14 days of purchase order. Time is of the essence.",
    },
    {
        "question": "What is the hourly rate in the Consulting Agreement?",
        "ground_truth": "Consultant charges $200 per hour, invoiced monthly and payable within 30 days.",
    },
    {
        "question": "Who owns the work product in the Consulting Agreement?",
        "ground_truth": "All work product is assigned to Client upon full payment, including source code, models and documentation.",
    },
    {
        "question": "What is the ownership split in the Joint Venture Agreement?",
        "ground_truth": "Each party holds 50% interest in the Joint Venture.",
    },
    {
        "question": "What does AlphaCorp contribute to the Joint Venture?",
        "ground_truth": "AlphaCorp contributes a technology platform valued at $2 million.",
    },
]


# ── Retrieval ────────────────────────────────────────────────────────────────
def retrieve(question: str) -> list[str]:
    q_vec = _embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(q_vec, k=TOP_K)

    return [
        _chunks[i]["text"]
        for i in indices[0]
        if i != -1 and "text" in _chunks[i]
    ]


# ── Generation ───────────────────────────────────────────────────────────────
def generate_answer(question: str, contexts: list[str], llm: ChatOllama) -> str:
    context_str = "\n\n".join(contexts)

    prompt = f"""You are a legal contract analyst.
Answer ONLY using the provided context.

Rules:
- 1–2 sentences maximum
- Cite the exact clause wording
- Do NOT add information not present
- If missing, say: Not found in context

Context:
{context_str}

Question: {question}
Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ── Build dataset ────────────────────────────────────────────────────────────
def build_eval_dataset(llm: ChatOllama) -> Dataset:
    print("Building eval dataset...")

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, item in enumerate(GOLDEN_SET):
        print(f"[{i+1}/{len(GOLDEN_SET)}] {item['question'][:60]}...")

        contexts = retrieve(item["question"])
        answer   = generate_answer(item["question"], contexts, llm)

        rows["question"].append(item["question"])
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(rows)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)

    print("Setting up RAGAS...")
    ollama_llm = ChatOllama(model="mistral", temperature=0)
    ollama_emb = OllamaEmbeddings(model="mistral")

    ragas_llm = LangchainLLMWrapper(ollama_llm)
    ragas_emb = LangchainEmbeddingsWrapper(ollama_emb)

    dataset = build_eval_dataset(ollama_llm)
    print(f"\nDataset built: {len(dataset)} rows\n")

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_emb

    # ✅ FIXED run_config
    run_config = RunConfig(
        max_workers=2,
        timeout=120,
    )

    print("Running evaluation...")
    results = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    # ✅ FIXED dataframe merge
    df_scores = results.to_pandas()
    df_data   = dataset.to_pandas()
    df = pd.concat([df_data, df_scores], axis=1)

    df.to_csv(RESULTS_PATH, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("RAGAS EVALUATION SUMMARY")
    print("="*50)

    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in df.columns:
            print(f"{col:<22} {df[col].mean():.3f}")

    # ── Breakdown ────────────────────────────────────────────────────────────
    print("\nPer-question breakdown:")
    print("-"*50)

    for _, row in df.iterrows():
        q = row["question"][:50]
        f = row.get("faithfulness", 0)
        ar = row.get("answer_relevancy", 0)
        print(f"{q:<52} faith={f:.2f}  rel={ar:.2f}")

    print(f"\nResults saved → {RESULTS_PATH}")
    print("Phase 3 complete.")