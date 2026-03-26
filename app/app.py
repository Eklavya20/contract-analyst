import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from agent.agent import chat, _chunks, risk_flagger, clause_extractor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Intelligence Analyst",
    page_icon="📄",
    layout="wide"
)

st.title("Contract Intelligence Analyst")
st.caption("Powered by Mistral · LangGraph · FAISS")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Contracts loaded")
    contract_names = sorted({c["filename"] for c in _chunks})
    for name in contract_names:
        st.markdown(f"- {name}")

    st.divider()
    st.header("Quick actions")
    selected_contract = st.selectbox("Select contract", contract_names)

    if st.button("Flag risks in this contract"):
        with st.spinner("Analysing risks..."):
            result = risk_flagger.invoke({"contract_name": selected_contract})
        st.session_state["risk_result"] = result
        st.session_state["risk_contract"] = selected_contract

    clause_options = [
        "termination", "payment", "liability",
        "ip assignment", "non-compete", "governing law", "auto-renewal"
    ]
    selected_clause = st.selectbox("Extract clause type", clause_options)
    if st.button("Extract across all contracts"):
        with st.spinner("Extracting clauses..."):
            result = clause_extractor.invoke({"clause_type": selected_clause})
        st.session_state["clause_result"] = result
        st.session_state["clause_type"] = selected_clause

    st.divider()
    if st.button("Clear chat history"):
        st.session_state["history"] = []
        st.session_state.pop("risk_result", None)
        st.session_state.pop("clause_result", None)
        st.rerun()

# ── Main area tabs ────────────────────────────────────────────────────────────
tab_chat, tab_risk, tab_clause, tab_chunks = st.tabs([
    "Chat", "Risk report", "Clause extractor", "Source chunks"
])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # render chat history
    for msg in st.session_state["history"]:
        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    # chat input
    user_input = st.chat_input("Ask anything about the contracts...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, st.session_state["history"] = chat(
                    user_input,
                    st.session_state["history"]
                )
            st.write(answer)

    # suggested queries
    if not st.session_state["history"]:
        st.markdown("#### Try asking:")
        cols = st.columns(2)
        suggestions = [
            "What are the termination clauses across all contracts?",
            "Compare the liability clauses across contracts",
            "Which contracts have auto-renewal clauses?",
            "What are the payment terms in the Distribution_Agreement.txt?",
        ]
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"sug_{i}"):
                    with st.spinner("Thinking..."):
                        answer, st.session_state["history"] = chat(
                            suggestion,
                            st.session_state["history"]
                        )
                    st.rerun()

# ── Tab 2: Risk report ────────────────────────────────────────────────────────
with tab_risk:
    if "risk_result" in st.session_state:
        st.subheader(f"Risk report — {st.session_state['risk_contract']}")

        result = st.session_state["risk_result"]

        # split pattern summary from LLM analysis
        if "LLM Risk Analysis:" in result:
            pattern_part, llm_part = result.split("LLM Risk Analysis:", 1)
        else:
            pattern_part, llm_part = "", result

        if pattern_part.strip():
            st.markdown("#### Pattern scan")
            lines = [l for l in pattern_part.strip().splitlines() if l.strip()]
            for line in lines:
                if "detected" in line.lower():
                    st.warning(line.strip())

        if llm_part.strip():
            st.markdown("#### LLM analysis")
            st.markdown(llm_part.strip())
    else:
        st.info("Select a contract in the sidebar and click 'Flag risks' to see the report here.")

# ── Tab 3: Clause extractor ───────────────────────────────────────────────────
with tab_clause:
    if "clause_result" in st.session_state:
        st.subheader(f"'{st.session_state['clause_type']}' clauses across all contracts")
        st.markdown(st.session_state["clause_result"])
    else:
        st.info("Select a clause type in the sidebar and click 'Extract' to see results here.")

# ── Tab 4: Source chunks ──────────────────────────────────────────────────────
with tab_chunks:
    st.subheader("Browse indexed chunks")
    search_term = st.text_input("Filter chunks by keyword")

    filtered = [
        c for c in _chunks
        if not search_term or search_term.lower() in c["text"].lower()
    ]

    st.caption(f"Showing {len(filtered)} of {len(_chunks)} chunks")

    for chunk in filtered[:50]:
        with st.expander(f"{chunk['filename']} — chunk {chunk['chunk_id']}"):
            st.text(chunk["text"])