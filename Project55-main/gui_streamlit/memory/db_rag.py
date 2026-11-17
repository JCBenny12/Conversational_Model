import sqlite3
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------- DATABASE FUNCTIONS ----------------------

def fetch_data_from_db(db_path="conversation_memory.db", table_name="conversation_memory"):
    """Fetch all rows from the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT * FROM {table_name};")
    data = cursor.fetchall()
    
    conn.close()
    return data

# ---------------------- EMBEDDING & STORAGE ----------------------

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(texts):
    """Generate embeddings for given texts."""
    return model.encode(texts).tolist()

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Stores locally
collection = chroma_client.get_or_create_collection(name="rag_data")

def store_embeddings(data):
    """Convert database records to embeddings and store them in ChromaDB."""
    texts = [str(row) for row in data]  # Convert rows to text
    embeddings = generate_embeddings(texts)
    
    for i, text in enumerate(texts):
        collection.add(
            ids=[str(i)],  # Unique identifier
            embeddings=[embeddings[i]],
            documents=[text]
        )

# ---------------------- RETRIEVAL ----------------------

def retrieve_context(query):
    """Retrieve relevant context from the vector database for a given query."""
    query_embedding = generate_embeddings([query])[0]  # Generate query embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Retrieve top 3 matches
    )
    #print(results)
    return results["documents"][0] if results["documents"] else []
