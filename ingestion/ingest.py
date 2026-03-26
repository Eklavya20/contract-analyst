
import os
import pickle
from pathlib import Path

import urllib.request
import json

import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Config ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512    # characters (not tokens — LangChain uses chars by default)
CHUNK_OVERLAP = 64
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, runs locally, free
INDEX_PATH    = "ingestion/faiss.index"
CHUNKS_PATH   = "ingestion/chunks.pkl"
MAX_CONTRACTS = 10     # start small — each contract is ~40k chars


# ── Step 1: Load CUAD contracts from Hugging Face ──────────────────────────
def load_cuad_texts(n=MAX_CONTRACTS) -> list[dict]:
    print("Loading contracts...")
    return _synthetic_contracts(n)

def _synthetic_contracts(n: int) -> list[dict]:
    contracts = [
        ("Affiliate_Marketing_Agreement.txt", """AFFILIATE MARKETING AGREEMENT
This Agreement is entered into as of February 1, 2024, between RetailCo Inc ("Company") and MarketPro LLC ("Affiliate").
1. APPOINTMENT. Company appoints Affiliate as a non-exclusive affiliate to promote Company products.
2. COMMISSION. Affiliate shall earn 8% commission on net sales generated through affiliate links.
3. PAYMENT TERMS. Commissions are paid monthly within 30 days of month end, minimum threshold $100.
4. TERM. This Agreement begins February 1, 2024 and continues for one year, auto-renewing annually.
5. TERMINATION WITHOUT CAUSE. Either party may terminate with 30 days written notice.
6. TERMINATION FOR CAUSE. Company may terminate immediately if Affiliate engages in fraudulent activity.
7. INTELLECTUAL PROPERTY. All Company trademarks remain property of Company. Affiliate receives limited license to use logos solely for promotion.
8. CONFIDENTIALITY. Affiliate shall not disclose commission rates or proprietary marketing data.
9. LIABILITY. Company's liability to Affiliate shall not exceed commissions earned in prior 3 months.
10. GOVERNING LAW. This Agreement is governed by the laws of California."""),

        ("Software_License_Agreement.txt", """SOFTWARE LICENSE AGREEMENT
This License Agreement is made as of March 15, 2024 between SoftTech Corp ("Licensor") and Enterprise Solutions Inc ("Licensee").
1. GRANT OF LICENSE. Licensor grants Licensee a non-exclusive, non-transferable license to use the Software.
2. RESTRICTIONS. Licensee shall not reverse engineer, decompile, or create derivative works from the Software.
3. LICENSE FEE. Licensee shall pay $50,000 annual license fee, due January 1 each year.
4. MAINTENANCE. Licensor will provide updates and bug fixes for the duration of the license.
5. INTELLECTUAL PROPERTY. All rights, title and interest in the Software remain exclusively with Licensor.
6. IP ASSIGNMENT. Any modifications suggested by Licensee and implemented by Licensor become property of Licensor.
7. TERM AND TERMINATION. License term is 3 years. Either party may terminate for material breach not cured within 45 days.
8. LIMITATION OF LIABILITY. In no event shall Licensor's liability exceed the license fees paid in the prior 12 months.
9. WARRANTY DISCLAIMER. Software is provided as-is. Licensor disclaims all implied warranties of merchantability.
10. AUDIT RIGHTS. Licensor may audit Licensee's use of Software upon 10 days notice, no more than once per year."""),

        ("Commercial_Lease_Agreement.txt", """COMMERCIAL LEASE AGREEMENT
This Lease is entered into April 1, 2024 between PropOwner Ltd ("Landlord") and TechStartup Inc ("Tenant").
1. PREMISES. Landlord leases to Tenant office space at 100 Market Street, Suite 400, San Francisco, CA.
2. TERM. Lease term is 5 years commencing April 1, 2024 and ending March 31, 2029.
3. RENT. Base rent is $25,000 per month, increasing 3% annually on each anniversary.
4. SECURITY DEPOSIT. Tenant shall deposit 3 months rent ($75,000) upon execution.
5. USE. Premises shall be used solely for general office and technology business purposes.
6. ASSIGNMENT. Tenant may not assign or sublease without Landlord's prior written consent.
7. TERMINATION FOR CAUSE. Landlord may terminate upon 30 days notice if Tenant fails to pay rent.
8. EARLY TERMINATION. Tenant may terminate after year 2 with 6 months notice and payment of 3 months penalty rent.
9. GOVERNING LAW. This Agreement shall be governed by the laws of the State of California."""),

        ("Distribution_Agreement.txt", """DISTRIBUTION AGREEMENT
This Distribution Agreement is effective May 1, 2024 between ManufactureCo ("Supplier") and DistributePro ("Distributor").
1. APPOINTMENT. Supplier appoints Distributor as exclusive distributor in the United States.
2. EXCLUSIVITY. Distributor shall not distribute competing products during the term without written consent.
3. MINIMUM PURCHASE. Distributor must purchase minimum $500,000 of products per calendar year.
4. PRICING. Supplier provides products at 40% discount to MSRP. Pricing reviewed annually.
5. PAYMENT. Net 30 days from invoice date. Late payments accrue interest at 1.5% per month.
6. TERM. Agreement is for 2 years, renewable by mutual written agreement 90 days before expiration.
7. TERMINATION WITHOUT CAUSE. Either party may terminate with 90 days written notice.
8. TERMINATION FOR CAUSE. Either party may terminate immediately upon material breach not cured within 30 days.
9. INTELLECTUAL PROPERTY. Distributor receives limited license to use Supplier trademarks solely for product promotion.
10. NON-COMPETE. Upon termination, Distributor may not distribute competing products for 12 months."""),

        ("NDA_Agreement.txt", """NON-DISCLOSURE AGREEMENT
This NDA is entered into January 1, 2024 between Acme Corp ("Disclosing Party") and Beta Inc ("Receiving Party").
1. CONFIDENTIAL INFORMATION. All proprietary business, technical and financial information disclosed is confidential.
2. OBLIGATIONS. Receiving Party shall protect confidential information with same degree of care as its own secrets.
3. EXCLUSIONS. Obligations do not apply to information already public or independently developed.
4. TERM. Confidentiality obligations survive for 3 years after disclosure.
5. TERMINATION. Either party may terminate this Agreement with 30 days written notice.
6. RETURN OF INFORMATION. Upon termination, Receiving Party shall return or destroy all confidential materials.
7. GOVERNING LAW. Delaware law governs. Disputes resolved in Delaware courts."""),

        ("Employment_Contract.txt", """EMPLOYMENT AGREEMENT
This Agreement is between TechFirm Inc and Jane Doe, effective April 1, 2024.
1. POSITION. Employee is hired as Senior Engineer reporting to the CTO.
2. COMPENSATION. Base salary $120,000 per year, payable bi-weekly.
3. BONUS. Employee eligible for annual bonus up to 20% of base salary based on performance.
4. BENEFITS. Health, dental, vision insurance and 401k with 4% company match.
5. IP ASSIGNMENT. All inventions, improvements and works created during employment are assigned exclusively to Company.
6. NON-COMPETE. Employee agrees not to work for a direct competitor for 12 months post-termination within 50 miles.
7. NON-SOLICITATION. Employee shall not solicit Company customers or employees for 24 months post-termination.
8. TERMINATION WITHOUT CAUSE. Company may terminate with 2 weeks notice and 4 weeks severance pay.
9. TERMINATION FOR CAUSE. Company may terminate immediately for gross misconduct with no severance.
10. GOVERNING LAW. This Agreement is governed by New York law."""),

        ("SaaS_Agreement.txt", """SOFTWARE AS A SERVICE AGREEMENT
This SaaS Agreement is between CloudCo ("Provider") and ClientX ("Customer") effective March 1, 2024.
1. SERVICES. Provider delivers cloud-based CRM software as described in Exhibit A.
2. SUBSCRIPTION FEE. Customer pays $5,000 per month, billed annually in advance at $60,000.
3. UPTIME SLA. Provider guarantees 99.9% monthly uptime. Credits issued for downtime exceeding SLA.
4. DATA OWNERSHIP. Customer retains ownership of all data uploaded to the platform.
5. DATA SECURITY. Provider maintains SOC 2 Type II certification and encrypts all data at rest and in transit.
6. LIMITATION OF LIABILITY. Provider's total liability shall not exceed fees paid in the prior 3 months.
7. TERMINATION FOR CAUSE. Either party may terminate immediately upon material breach not cured within 30 days.
8. AUTO-RENEWAL. Agreement renews automatically for successive one-year terms unless cancelled 60 days prior.
9. PRICE INCREASE. Provider may increase fees upon 60 days notice, not more than 5% annually."""),

        ("Supply_Agreement.txt", """SUPPLY AGREEMENT
This Supply Agreement is dated June 1, 2024 between ComponentMaker Inc ("Supplier") and AssemblyCo ("Buyer").
1. SUPPLY OBLIGATION. Supplier agrees to supply electronic components per specifications in Exhibit A.
2. PRICING. Prices fixed for 12 months. After 12 months, Supplier may adjust with 60 days notice.
3. PAYMENT. Net 45 days from delivery. 2% discount for payment within 10 days.
4. DELIVERY. Supplier shall deliver within 14 days of purchase order. Time is of the essence.
5. QUALITY. All components must meet ISO 9001 standards. Buyer may reject non-conforming goods within 30 days.
6. EXCLUSIVITY. Supplier shall not supply identical components to Buyer's direct competitors without consent.
7. FORCE MAJEURE. Neither party liable for delays caused by events beyond reasonable control.
8. TERM. Agreement is for 3 years. Either party may terminate without cause with 90 days notice.
9. LIABILITY. Supplier's liability for defective goods limited to replacement cost of defective components."""),

        ("Consulting_Agreement.txt", """CONSULTING SERVICES AGREEMENT
This Agreement is between DataInsights LLC ("Consultant") and FinanceCorp ("Client") effective July 1, 2024.
1. SERVICES. Consultant provides data analytics and ML model development services per Statement of Work.
2. FEES. Consultant charges $200 per hour. Invoices submitted monthly, payable within 30 days.
3. EXPENSES. Client reimburses pre-approved expenses within 30 days of receipt of expense report.
4. INTELLECTUAL PROPERTY. All work product created under this Agreement is assigned to Client upon full payment.
5. IP ASSIGNMENT. Consultant assigns all rights in deliverables including source code, models and documentation.
6. CONFIDENTIALITY. Consultant shall not disclose Client's proprietary data or business strategies.
7. NON-SOLICITATION. Consultant shall not solicit Client's employees for 12 months after termination.
8. TERM. Agreement continues until completion of Statement of Work or until terminated.
9. TERMINATION WITHOUT CAUSE. Client may terminate with 14 days written notice, paying for work completed.
10. LIMITATION OF LIABILITY. Consultant's liability shall not exceed total fees paid in prior 3 months."""),

        ("Partnership_Agreement.txt", """JOINT VENTURE AGREEMENT
This Agreement is between AlphaCorp and BetaVentures effective August 1, 2024 to form a joint venture.
1. PURPOSE. The Joint Venture is formed to develop and commercialize AI-powered legal software.
2. CONTRIBUTIONS. AlphaCorp contributes technology platform valued at $2M. BetaVentures contributes $2M cash.
3. OWNERSHIP. Each party holds 50% interest in the Joint Venture.
4. GOVERNANCE. Decisions require unanimous consent of both parties for major matters.
5. PROFIT DISTRIBUTION. Net profits distributed equally quarterly after reserves maintained for operations.
6. INTELLECTUAL PROPERTY. Each party retains pre-existing IP. New IP developed jointly is owned by the JV.
7. NON-COMPETE. During the term, neither party may independently develop competing AI legal software.
8. TERM. Joint Venture runs for 5 years, extendable by mutual written agreement.
9. TERMINATION FOR CAUSE. Either party may dissolve the JV upon material breach not cured within 60 days.
10. DISSOLUTION. Upon dissolution, assets distributed pro-rata after paying all outstanding obligations."""),
    ]
    return [{"filename": t[0], "text": t[1]} for t in contracts[:n]]


# ── Step 2: Chunk each contract ────────────────────────────────────────────
def chunk_contracts(contracts: list[dict]) -> list[dict]:
    """
    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then sentences, then words — keeping semantic units intact where possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
    )

    all_chunks = []
    for contract in contracts:
        raw_chunks = splitter.split_text(contract["text"])
        for i, chunk_text in enumerate(raw_chunks):
            all_chunks.append({
                "text":     chunk_text,
                "filename": contract["filename"],
                "chunk_id": i,
                "metadata": f"{contract['filename']} | chunk {i}"
            })

    print(f"  Created {len(all_chunks)} chunks from {len(contracts)} contracts")
    print(f"  Avg chunk length: {np.mean([len(c['text']) for c in all_chunks]):.0f} chars")
    return all_chunks


# ── Step 3: Embed all chunks ───────────────────────────────────────────────
def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """
    all-MiniLM-L6-v2 produces 384-dimensional vectors.
    It's small (80MB), fast on CPU, and good enough for retrieval.
    Downloads automatically on first run.
    """
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks (this takes ~1–2 min on CPU)...")

    # batch_size=64 is safe for CPU; increase if you have GPU
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # normalise → cosine sim = dot product
    )

    print(f"  Embeddings shape: {embeddings.shape}")  # (n_chunks, 384)
    return embeddings.astype("float32")


# ── Step 4: Build and save FAISS index ────────────────────────────────────
def build_index(embeddings: np.ndarray, chunks: list[dict]):
    """
    IndexFlatIP = exact inner product search (cosine sim after normalisation).
    Fine for <100k chunks. Switch to IndexIVFFlat for larger corpora.
    """
    dim = embeddings.shape[1]  # 384
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  Saved index → {INDEX_PATH}")
    print(f"  Saved chunks → {CHUNKS_PATH}")


# ── Step 5: Smoke test — retrieve chunks for a query ──────────────────────
def smoke_test(query: str = "termination without cause"):
    from sentence_transformers import SentenceTransformer

    model  = SentenceTransformer(EMBED_MODEL)
    index  = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, k=3)

    print(f"\nSmoke test — query: '{query}'")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        c = chunks[idx]
        print(f"\n  Rank {rank+1} | score={score:.3f} | {c['metadata']}")
        print(f"  {c['text'][:200]}...")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("ingestion", exist_ok=True)

    contracts  = load_cuad_texts()
    chunks     = chunk_contracts(contracts)
    embeddings = embed_chunks(chunks)
    build_index(embeddings, chunks)
    smoke_test()

    print("\nPhase 1 complete. Run: python ingestion/ingest.py")