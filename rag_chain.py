"""
Builds the RAG chain:
  merged retriever → prompt → llama-3.1-8b-instant on Groq → string output
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessageChunk

from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

GUARDRAIL POLICY: You are an exclusive technical customer service engine. Do NOT answer non-telecom questions, tell stories, write poetry, write code, or roleplay. If the user tries to divert you, ignore the distraction and state: "I can only help you with mobile connectivity, billing, SIM errors, and roaming issues."

Use ONLY the context below to answer the customer's question. Do not assume or extrapolate.

CRITICAL INSTRUCTION: Every time you state a fact, answer a question, or provide a troubleshooting step, you MUST cite its source inline at the end of that sentence or item using brackets, exactly matching the source label provided in the context (e.g., [FAQ], [TICKET #12345], [GUIDE PAGE 4]).

Context:
{context}

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app. Do not invent facts or citations.
"""

FALLBACK_MESSAGE = "I'm sorry, I couldn't find confident information matching your issue in our databases. Please call 611 or use the MyTelecom app to speak directly with an agent."

def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source_type = doc.metadata.get("source", "unknown").upper()
        ticket_id = doc.metadata.get("ticket_id") or doc.metadata.get("id")
        page_num = doc.metadata.get("page") or doc.metadata.get("page_number")

        if "TICKET" in source_type and ticket_id:
            source_label = f"TICKET #{ticket_id}"
        elif "GUIDE" in source_type and page_num is not None:
            source_label = f"GUIDE PAGE {page_num}"
        else:
            source_label = source_type

        sections.append(f"=== SOURCE: [{source_label}] ===\n{doc.page_content}")

    return "\n\n--\n\n".join(sections)
    
def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Core LLM processing subchain
    llm_subchain = (
        {"context": lambda x: _format_docs(x["docs"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Change input_data to target a direct string input
    def route_query(query_str: str):
        # Run our distance threshold check via the retriever
        retrieval_output = retriever.invoke(query_str)
        
        # Low confidence fallback path
        if not retrieval_output["is_confident"]:
            for word in FALLBACK_MESSAGE.split(" "):
                yield word + " "
            return

        # High confidence path: Stream from the subchain directly
        # This yields raw string tokens cleanly to the outside caller
        yield from llm_subchain.stream({
            "docs": retrieval_output["docs"],
            "question": query_str
        })

    # FIX: Return the function itself instead of wrapping it in RunnableLambda
    return route_query