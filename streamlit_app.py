import streamlit as st
import requests
import os
import time

API_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

st.set_page_config(
    page_title="Finance RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 3px solid #2563eb;
        margin: 4px 0;
    }
    .pipeline-badge {
        background: #eff6ff;
        color: #1d4ed8;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .source-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
    }
    .latency-tag {
        color: #6b7280;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📁 Finance RAG Assistant")
    st.caption("Upload financial reports and ask questions in natural language.")
    
    st.divider()
    
    # Pipeline explanation
    st.markdown("**🔬 RAG Pipeline**")
    for badge in ["MultiQuery", "HyDE", "Hybrid Search", "Reranking", "Memory"]:
        st.markdown(f'<span class="pipeline-badge">{badge}</span>', unsafe_allow_html=True)
    
    st.divider()

    # Document upload
    st.markdown("**📄 Upload Document**")
    company = st.text_input("Company", value="Tesla", placeholder="e.g. Apple")
    year = st.text_input("Year", value="2023", placeholder="e.g. 2023")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Upload & Ingest", type="primary"):
        with st.spinner("Processing document..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    params={"company": company, "year": year},
                    timeout=120
                )
                if response.status_code == 200:
                    st.success(f"✅ {uploaded_file.name} ingested")
                else:
                    st.error(f"Upload failed: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()

    # Session management
    st.markdown("**🔁 Session**")
    st.caption(f"Current: `{st.session_state.get('session_id', 'default')}`")
    new_session = st.text_input("Session ID", value="default", label_visibility="collapsed")
    if st.button("Apply Session"):
        st.session_state.session_id = new_session
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("📊 [View LangFuse Traces](https://cloud.langfuse.com)", unsafe_allow_html=True)
    st.caption("All requests are traced with spans for retrieval, reranking, and generation.")

# Main area
st.markdown("## Finance Document Assistant")

# Mode selector
col1, col2 = st.columns([3, 1])
with col1:
    mode = st.radio(
        "Mode",
        ["RAG", "AGENT"],
        horizontal=True,
        help="RAG: retrieval-augmented generation with memory. Agent: ReAct agent with tools (stock prices, calculations)."
    )
with col2:
    if st.button("🗑 Clear Chat", use_container_width=True):
        try:
            requests.delete(f"{API_URL}/chat/{st.session_state.get('session_id', 'default')}", timeout=10)
        except:
            pass
        st.session_state.messages = []
        st.rerun()

st.divider()

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]
            
            cols = st.columns(3)
            with cols[0]:
                if meta.get("latency"):
                    st.caption(f"⏱ {meta['latency']:.1f}s")
            with cols[1]:
                if meta.get("chunks_used") is not None:
                    st.caption(f"📄 {meta['chunks_used']} chunks")
            with cols[2]:
                scores = meta.get("scores", {})
                if scores.get("faithfulness") is not None:
                    st.caption(f"✓ Faith: {scores['faithfulness']:.2f}")

            # Sources
            if meta.get("sources"):
                with st.expander(f"📚 View {len(meta['sources'])} sources"):
                    for i, source in enumerate(meta["sources"]):
                        st.markdown(f"""<div class="source-card">
                            <strong>Chunk {i+1}</strong> — {source.get('company', '?')} {source.get('year', '?')} · Page {source.get('page', '?')}<br>
                            <small>{source.get('content', '')}</small>
                        </div>""", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about your financial document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            try:
                if mode == "RAG":
                    response = requests.post(
                        f"{API_URL}/chat/memory",
                        params={"question": prompt, "session_id": st.session_state.session_id},
                        timeout=120
                    )
                else:
                    response = requests.post(
                        f"{API_URL}/agent/ask",
                        params={"question": prompt, "session_id": st.session_state.session_id},
                        timeout=120
                    )
                latency = time.time() - start

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    scores = data.get("scores", {})
                    chunks_used = data.get("chunks_used", len(sources))

                    st.write(answer)

                    # Metadata row
                    cols = st.columns(3)
                    with cols[0]:
                        st.caption(f"⏱ {latency:.1f}s")
                    with cols[1]:
                        st.caption(f"📄 {chunks_used} chunks")
                    with cols[2]:
                        if scores.get("faithfulness") is not None:
                            st.caption(f"✓ Faith: {scores['faithfulness']:.2f}")

                    # RAGAS scores for agent mode
                    if mode == "AGENT" and any(v is not None for v in scores.values()):
                        s1, s2 = st.columns(2)
                        with s1:
                            if scores.get("faithfulness") is not None:
                                st.metric("Faithfulness", f"{scores['faithfulness']:.2f}")
                        with s2:
                            if scores.get("answer_relevancy") is not None:
                                st.metric("Answer Relevancy", f"{scores['answer_relevancy']:.2f}")

                    # Sources
                    if sources:
                        with st.expander(f"📚 View {len(sources)} sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"""<div class="source-card">
                                    <strong>Chunk {i+1}</strong> — {source.get('company', '?')} {source.get('year', '?')} · Page {source.get('page', '?')}<br>
                                    <small>{source.get('content', '')}</small>
                                </div>""", unsafe_allow_html=True)

                    # Save to history with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": {
                            "latency": latency,
                            "chunks_used": chunks_used,
                            "scores": scores,
                            "sources": sources
                        }
                    })

                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend. Is the API running?")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The model may be processing a large document.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")