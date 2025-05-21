import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
import os
import json

# === Config ===
CHROMA_PERSIST_DIR = "./chroma_store"
CHROMA_COLLECTION_NAME = "blog_posts"
TOP_K = 5  # Number of chunks to retrieve
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_CHAT_API_URL = f"{OLLAMA_API_BASE}/api/chat"
OLLAMA_MODEL = "llama3.2:3b"  # Change to your model name

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))
    print("Using GPU for embeddings.")
else:
    print("Using CPU for embeddings.")

# === Connect to Chroma ===
# Use PersistentClient to match populate_database.py
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR
)

# Print debug info
print(f"Looking for collection in: {os.path.abspath(CHROMA_PERSIST_DIR)}")
collections = chroma_client.list_collections()
print(f"Available collections: {[c.name for c in collections]}")

# Try to get or create the collection
try:
    # Note: get_collection doesn't take metadata parameter
    collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
    print(f"Successfully found collection: {CHROMA_COLLECTION_NAME}")
except Exception as e:
    print(f"Error getting collection: {e}")
    print("Creating collection instead...")
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def embed_query(query: str):
    embedding = model.encode([query], convert_to_numpy=True)[0]
    return embedding

def retrieve_relevant_chunks(query_embedding, top_k=TOP_K):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results

def build_context(results):
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    context_chunks = []
    for doc, meta in zip(docs, metadatas):
        summary = meta.get("post_summary", "")
        filename = meta.get("filename", "unknown")
        if summary:
            context_chunks.append(f"[Summary for {filename}]: {summary}\n{doc}")
        else:
            context_chunks.append(doc)
    context = "\n\n".join(context_chunks)
    return context

def retrieve_context_for_query(query: str):
    """Get RAG context for a single query string"""
    query_embedding = embed_query(query)
    results = retrieve_relevant_chunks(query_embedding)
    context = build_context(results)
    
    # Debug the retrieval results
    print(f"Retrieved {len(results['documents'][0])} chunks")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        print(f"Chunk {i+1}: From {meta.get('filename', 'unknown')}, distance: {dist:.4f}")
        print(f"Preview: {doc[:100]}...")
    
    print(f"Total context length: {len(context)} characters")
    return context

def chat_with_rag(messages, model=None):
    """
    Handle chat request with RAG augmentation for the latest user message
    """
    if not messages:
        return {"message": {"content": "No messages provided"}}
    
    # Get the latest user message for RAG context
    user_messages = [msg for msg in messages if msg.get('role') == 'user']
    if not user_messages:
        return {"message": {"content": "No user message found"}}
    
    latest_query = user_messages[-1]['content']
    print(f"Processing query: '{latest_query}'")
    
    # Get RAG context for the latest query
    context = retrieve_context_for_query(latest_query)
    
    # Make sure we're using the requested model or default
    model_to_use = model or OLLAMA_MODEL
    
    # Default system prompt if none provided
    default_system_prompt = """You are a concise, helpful assistant based on the llama3.2:3b model embedded on Sam Spackman's blog *Data Science Magic*. You are NOT Sam Spackman himself.

Welcome each user to the site. Answer their question using the retrieved context.

- Prefer content from the context when it is relevant.
- If the context does not fully answer the question, generate additional helpful information — but do not contradict the context.
- Keep responses concise, friendly, concise, and aligned with the tone of the blog: technical but conversational and curious.
- If the context includes multiple blog posts, answer based on the most relevant post.
- Be exceptionally concise
- Do not respond in markdown format"""
    
    # Check if a system message is already present
    system_messages = [msg for msg in messages if msg.get('role') == 'system']
    system_prompt = system_messages[-1]['content'] if system_messages else default_system_prompt
    
    # Create new messages array - start with system message
    chat_messages = [{"role": "system", "content": system_prompt}]
    
    # Add all non-system messages from the original array
    # Replace the latest user message with one that includes context
    for i, msg in enumerate(messages):
        if msg.get('role') == 'system':
            continue  # Skip system messages, we already handled it
            
        if msg.get('role') == 'user' and msg['content'] == latest_query:
            # This is the latest user message, enhance it with context
            enhanced_message = {"role": "user", "content": f"Context: {context}\n\nQuestion: {msg['content']}"}
            chat_messages.append(enhanced_message)
        else:
            # Keep assistant and earlier user messages
            chat_messages.append(msg)
    
    # For debugging
    print(f"Sending request to Ollama API at: {OLLAMA_CHAT_API_URL}")
    print(f"Using model: {model_to_use}")
    
    # Call Ollama chat API
    try:
        response = requests.post(
            OLLAMA_CHAT_API_URL,
            json={
                "model": model_to_use,
                "messages": chat_messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 400  # equivalent to max_tokens
                }
            }
        )
        
        if response.status_code != 200:
            print(f"Error from Ollama chat API: {response.status_code} - {response.text}")
            return {"message": {"content": f"Error: HTTP {response.status_code} - Unable to get response from Ollama"}}
        
        # Debug response
        print(f"Response status: {response.status_code}")
        print(f"Response content type: {response.headers.get('Content-Type', 'unknown')}")
            
        result = response.json()
        return result  # Ollama's chat endpoint already returns in the format we want
    
    except Exception as e:
        print(f"Exception in chat_with_rag: {e}")
        return {"message": {"content": f"Error: {str(e)}"}}

if __name__ == "__main__":
    print("This module now supports chat format only. Use the /api/chat endpoint.")