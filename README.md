# 📡 Telecom Customer Care Chatbot (RAG)

An intelligent, context-aware Retrieval-Augmented Generation (RAG) assistant designed to help customers resolve technical mobile service issues (connectivity, billing, SIM errors, and roaming). 

The system leverages a multi-vector store pipeline to pull context from three separate data streams: static FAQs, a comprehensive PDF technical guide, and historic resolved support tickets.

## 🛠️ Architecture Summary
- **LLM Engine:** Qwen3-32B via Groq Cloud
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
- **Vector Database:** Chroma DB
- **Framework:** LangChain (LCEL)
- **Interfaces:** Command-Line Interface (CLI) & Streamlit Web App

---

## ⚙️ Setup and Installation

### 1. Prerequisites
Ensure you have Python 3.11+ installed. This project uses the `uv` package manager (or standard `pip` using `pyproject.toml`).

### 2. Clone and Install Dependencies
```bash
# Clone this repository (replace with your URL)
git clone [https://github.com/YOUR_USERNAME/telecom-rag-chatbot.git](https://github.com/YOUR_USERNAME/telecom-rag-chatbot.git)
cd telecom-rag-chatbot

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install .