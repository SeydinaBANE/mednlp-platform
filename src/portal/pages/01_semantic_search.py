"""Streamlit page — Clinical Semantic Search (RAG)."""

import asyncio

import streamlit as st

st.set_page_config(page_title="Clinical Search", page_icon="🔍", layout="wide")

st.title("🔍 Clinical Semantic Search")
st.caption("AI-powered retrieval from clinical notes — powered by BiomedBERT + MedCPT + Claude")


# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Search settings")
    patient_id = st.text_input("Patient ID (optional)", placeholder="patient-001")
    top_k = st.slider("Max sources", min_value=1, max_value=10, value=5)
    use_streaming = st.toggle("Stream response", value=True)
    st.divider()
    st.caption("Disclaimer: AI-generated responses are for decision support only.")


# ── Query input ───────────────────────────────────────────────────────────────

query = st.text_area(
    "Clinical question",
    placeholder="What medications is the patient currently taking?",
    height=100,
)

search_btn = st.button("Search", type="primary", use_container_width=True)


# ── Results ───────────────────────────────────────────────────────────────────


async def _run_rag(q: str, pid: str | None, k: int) -> dict[str, object]:
    """Call the RAG pipeline and return the QueryResponse as a dict."""
    from src.rag.answer_generator import generate
    from src.rag.context_builder import build_context
    from src.rag.retriever import retrieve
    from src.vector_store.collections import list_note_collections

    collections = await list_note_collections()
    if not collections:
        return {"error": "No indexed collections found. Run the ingestion pipeline first."}

    collection = collections[0]
    candidates = await retrieve(q, collection, top_k=k * 3, patient_id=pid or None)
    context = build_context(candidates, max_tokens=3500)

    if not context.citations:
        return {
            "answer": "No relevant notes found.",
            "sources": [],
            "model": "none",
            "latency_ms": 0,
        }  # noqa: E501

    response = await generate(q, context)
    return {
        "answer": response.answer,
        "sources": [s.model_dump() for s in response.sources],
        "model": response.model,
        "latency_ms": response.latency_ms,
    }


if search_btn and query.strip():
    with st.spinner("Searching clinical notes…"):
        try:
            result = asyncio.run(_run_rag(query.strip(), patient_id or None, top_k))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Search failed: {exc}")
            st.stop()

    if "error" in result:
        st.warning(str(result["error"]))
        st.stop()

    st.subheader("Answer")
    st.markdown(str(result["answer"]))

    latency = result.get("latency_ms", 0)
    model = result.get("model", "")
    st.caption(f"Model: `{model}` · Latency: {latency}ms")

    sources = result.get("sources", [])
    if sources:
        st.subheader(f"Sources ({len(sources)})")
        for i, src in enumerate(sources, 1):
            note_type = src.get("note_type", "note")
            authored = str(src.get("authored_at", ""))[:10]
            with st.expander(f"[{i}] {note_type} — {authored}"):
                st.write(f"**Note ID:** `{src.get('note_id')}`")
                st.write(f"**Score:** {src.get('score', 0):.3f}")
                st.write(src.get("excerpt", ""))
elif search_btn:
    st.warning("Please enter a clinical question.")
