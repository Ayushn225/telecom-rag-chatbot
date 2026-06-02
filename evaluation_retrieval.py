"""
Automated Evaluation Script for Mobile Service RAG Assistant.
Dynamically pulls 10 real resolved cases from tickets.db to check Top-3 Recall.
"""
import os
import sqlite3
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_store"
COLLECTION = "tickets"
DB_PATH = os.path.join("data", "tickets.db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_real_test_pairs() -> list[tuple[str, str]]:
    """Connects to SQLite and dynamically pulls 10 sample cases for evaluation."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Could not locate database at {DB_PATH}. Run your ingestion scripts first!")
        
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Extract 10 resolved records directly from your local production source
    rows = conn.execute(
        "SELECT description, ticket_id FROM tickets WHERE status = 'resolved' LIMIT 10"
    ).fetchall()
    conn.close()
    
    # Map into (customer_query, expected_ticket_id) evaluation pairs
    return [(row["description"], str(row["ticket_id"])) for row in rows]

def run_retrieval_evaluation(k: int = 3):
    try:
        TEST_DATASET = load_real_test_pairs()
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return

    print("Connecting to local Chroma DB ticket collection...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    tickets_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    
    hits = 0
    total_queries = len(TEST_DATASET)
    
    print(f"\n==================================================")
    print(f"   STARTING EVALUATION: MEASURING TOP-{k} RECALL")
    print(f"==================================================")
    
    for idx, (question, expected_id) in enumerate(TEST_DATASET, start=1):
        # Clean the search string to ensure vector processing matching passes smoothly
        search_query = question.strip().replace("\n", " ")
        retrieved_docs = tickets_store.similarity_search(search_query, k=k)
        
        # Pull ticket_id values from the retrieved documents metadata matching track
        retrieved_ids = [str(doc.metadata.get("ticket_id")) for doc in retrieved_docs]
        
        is_hit = str(expected_id) in retrieved_ids
        if is_hit:
            hits += 1
            status = "✅ HIT"
        else:
            status = "❌ MISS"
            
        # Truncate the display question so it looks clean in the terminal console output
        display_q = search_query if len(search_query) < 60 else search_query[:57] + "..."
        print(f"Test {idx}/{total_queries}:")
        print(f"  Question  : \"{display_q}\"")
        print(f"  Expected  : {expected_id}")
        print(f"  Retrieved : {retrieved_ids} -> {status}\n")
        
    recall_score = (hits / total_queries) * 100
    
    print("==================================================")
    print("               EVALUATION SUMMARY                 ")
    print("==================================================")
    print(f" Total Test Scenarios : {total_queries}")
    print(f" Successful Top-{k} Hits: {hits}")
    print(f" Final Top-{k} Recall : {recall_score:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    run_retrieval_evaluation(k=3)