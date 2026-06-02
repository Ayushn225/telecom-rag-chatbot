import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from dotenv import load_dotenv
from rag_chain import build_chain

load_dotenv()

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]

st.set_page_config(
    page_title="Telecom Support Chat",
    page_icon="📡",
    layout="centered",
)

# ── STREAMLIT CACHING (The proper way to load in background) ──────────────────
@st.cache_resource(show_spinner=False)
def get_cached_chain():
    """Initializes and caches the vector stores and LLM setup once."""
    return build_chain()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Telecom Support")
    st.caption("Powered by RAG · Llama-3.1-8B on Groq")
    st.divider()

    st.markdown("**System Status**")
    # Using a placeholder let us swap statuses cleanly without script stutters
    status_placeholder = st.empty()
    status_placeholder.warning("⚠️ RAG Engine: Loading...")
        
    st.divider()

    st.markdown("**Sample questions**")
    st.caption("Click one to send it instantly.")
    
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []

# ── MAIN PANEL ───────────────────────────────────────────────────────────────
st.title("Customer Care Assistant")
st.caption("Ask me anything about your mobile service — connectivity, billing, SIM, roaming, and more.")

# Render existing chat logs
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Catch questions from input box or sidebar clicks
question = st.chat_input("Describe your issue…")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

# Update system status right before handling execution
try:
    chain_instance = get_cached_chain()
    status_placeholder.success("🟢 RAG Engine: Ready")
except Exception as e:
    status_placeholder.error("🔴 Pipeline Error")
    st.error(f"Failed to start RAG pipeline: {e}")
    chain_instance = None

# Execute the stream query cleanly
if question and chain_instance:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response = st.write_stream(chain_instance(question))

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()