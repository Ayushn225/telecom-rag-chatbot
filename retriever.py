"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_retriever(
    k_faq: int = 2,
    k_tickets: int = 2,
    k_guides: int = 2,
    distance_threshold: float = 0.8
) -> RunnableLambda:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # faq_retriever     = faq_store.as_retriever(search_kwargs={"k": k_faq})
    # tickets_retriever = tickets_store.as_retriever(search_kwargs={"k": k_tickets})
    # guides_retriever  = guides_store.as_retriever(search_kwargs={"k": k_guides})

    def retrieve_with_confidence(query: str) -> dict:
        # Run similarity search paired with structural distance metrics
        faq_results = faq_store.similarity_search_with_score(query, k=k_faq)
        ticket_results = tickets_store.similarity_search_with_score(query, k=k_tickets)
        guide_results = guides_store.similarity_search_with_score(query, k=k_guides)
        
        all_results = faq_results + ticket_results + guide_results
        
        # Pull documents and track if ANY single record beats our strict threshold
        valid_docs = []
        is_confident = False
        
        for doc, score in all_results:
            if score <= distance_threshold:
                is_confident = True
            valid_docs.append(doc)
            
        return {
            "docs": valid_docs,
            "is_confident": is_confident
        }
    
    # def retrieve(query: str) -> list[Document]:
    #     return (
    #         faq_retriever.invoke(query)
    #         + tickets_retriever.invoke(query)
    #         + guides_retriever.invoke(query)
    #     )

    return RunnableLambda(retrieve_with_confidence)
