from sentence_transformers import SentenceTransformer
import torch
import chromadb
from chromadb.config import Settings
import os
import hashlib
from typing import List
from transformers import AutoTokenizer, pipeline
import numpy as np

# === Config ===
CHUNK_SIZE = 512        # Reduced from 256 for safety
CHUNK_OVERLAP = 40      # Overlap between chunks
POSTS_DIR = "posts"  # Path to the posts folder
MAX_SEQUENCE_LENGTH = 512  # Maximum tokens for the model

# === Load the embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))
    print("Using GPU for embeddings.")
else:
    print("Using CPU for embeddings.")

# === Load HF tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# === Load summarization pipeline ===
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def generate_summary(text):
    # Limit input length for summarizer (BART max: 1024 tokens)
    max_input = 2000  # chars, adjust as needed
    input_text = text[:max_input]
    try:
        summary = summarizer(input_text, max_length=60, min_length=10, do_sample=False)[0]['summary_text']
    except Exception as e:
        print(f"Summarization failed: {e}")
        summary = ""
    return summary

# === Initialize Chroma ===
chroma_client = chromadb.PersistentClient(
    path="./chroma_store"  # Path to store your data
)
collection = chroma_client.get_or_create_collection(
    name="blog_posts",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# Helper function to compute file hash
def compute_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# === Token-aware chunker (HF version) ===
def chunk_text_tokenwise(text: str, tokenizer, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    # Encode the full text
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    
    # Ensure we're well under the model's max length to allow for special tokens
    actual_max_tokens = min(max_tokens, MAX_SEQUENCE_LENGTH - 20)  # Leave room for special tokens
    
    while start < len(input_ids):
        end = min(start + actual_max_tokens, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += actual_max_tokens - overlap
    return chunks

# === Process and chunk files ===
documents = []
metadatas = []
ids = []

# Track processed files 
processed_files = 0
skipped_files = 0
updated_files = 0

for fname in os.listdir(POSTS_DIR):
    if fname.endswith(".md"):
        file_path = os.path.join(POSTS_DIR, fname)  # Construct full file path
        file_hash = compute_file_hash(file_path)
        
        # Check if file already exists in collection and hasn't changed
        existing_entries = collection.get(
            where={"filename": fname}
        )
        
        # Check if file is already in database with the same hash
        if existing_entries["ids"] and existing_entries["metadatas"] and \
           len(existing_entries["ids"]) > 0 and \
           "file_hash" in existing_entries["metadatas"][0] and \
           existing_entries["metadatas"][0]["file_hash"] == file_hash:
            print(f"Skipping {fname} - already embedded and unchanged")
            skipped_files += 1
            continue
        
        # If we're here, either the file is new or has changed
        updated_files += 1
        processed_files += 1
        
        # 🧹 Clean old chunks for this file if it exists
        if existing_entries["ids"]:
            print(f"Updating {fname} - content has changed")
            collection.delete(where={"filename": fname})
        else:
            print(f"Processing new file: {fname}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            # --- Generate summary for the whole post ---
            summary = generate_summary(full_text)
            print(f"Summary for {fname}: {summary}")
            chunks = chunk_text_tokenwise(full_text, tokenizer)

            for i, chunk in enumerate(chunks):
                doc_id = f"{fname}-chunk-{i}"

                # Verify length of chunk before adding
                token_length = len(tokenizer.encode(chunk, add_special_tokens=True))
                if token_length <= MAX_SEQUENCE_LENGTH:
                    documents.append(chunk)
                    metadatas.append({
                        "filename": fname,
                        "chunk_index": i,
                        "token_length": token_length,
                        "file_hash": file_hash,  # Store the hash for future comparison
                        "post_summary": summary  # <-- Add summary here
                    })
                    ids.append(doc_id)
                else:
                    # Split this chunk further if it's too long
                    sub_chunks = chunk_text_tokenwise(chunk, tokenizer, max_tokens=MAX_SEQUENCE_LENGTH//2)
                    for j, sub_chunk in enumerate(sub_chunks):
                        sub_id = f"{fname}-chunk-{i}-sub-{j}"
                        sub_token_length = len(tokenizer.encode(sub_chunk, add_special_tokens=True))
                        if sub_token_length <= MAX_SEQUENCE_LENGTH:
                            documents.append(sub_chunk)
                            metadatas.append({
                                "filename": fname,
                                "chunk_index": f"{i}.{j}",
                                "token_length": sub_token_length,
                                "file_hash": file_hash,
                                "post_summary": summary  # <-- Add summary here
                            })
                            ids.append(sub_id)

if documents:
    print(f"Created {len(documents)} chunks for embedding")
    print(f"Files processed: {processed_files} (new: {processed_files - updated_files}, updated: {updated_files}, skipped: {skipped_files})")

    # === Embed and store in Chroma in batches ===
    batch_size = 16  # Reduced batch size for safety
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadata = metadatas[i:i+batch_size]
        
        # Safety check and manual embedding if needed
        safe_batch_docs = []
        safe_batch_ids = []
        safe_batch_metadata = []
        
        for j, doc in enumerate(batch_docs):
            tokens = tokenizer.encode(doc, add_special_tokens=True)
            if len(tokens) > MAX_SEQUENCE_LENGTH:
                print(f"Warning: Document {batch_ids[j]} still has {len(tokens)} tokens after chunking. Truncating.")
                tokens = tokens[:MAX_SEQUENCE_LENGTH]
                doc = tokenizer.decode(tokens, skip_special_tokens=True)
            
            safe_batch_docs.append(doc)
            safe_batch_ids.append(batch_ids[j])
            safe_batch_metadata.append(batch_metadata[j])
        
        # Now embed the safe documents
        try:
            batch_embeddings = model.encode(safe_batch_docs, convert_to_numpy=True)
            
            collection.add(
                documents=safe_batch_docs,
                embeddings=batch_embeddings.tolist(),
                metadatas=safe_batch_metadata,
                ids=safe_batch_ids
            )
            print(f"Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {str(e)}")
            # Process one by one for problem isolation
            for j, doc in enumerate(safe_batch_docs):
                try:
                    embedding = model.encode([doc], convert_to_numpy=True)[0]
                    collection.add(
                        documents=[doc],
                        embeddings=[embedding.tolist()],
                        metadatas=[safe_batch_metadata[j]],
                        ids=[safe_batch_ids[j]]
                    )
                    print(f"Successfully processed document {safe_batch_ids[j]}")
                except Exception as e2:
                    print(f"Failed on document {safe_batch_ids[j]}: {str(e2)}")
                    token_count = len(tokenizer.encode(doc, add_special_tokens=True))
                    print(f"Token count: {token_count}")
    
    print("✅ All token-aware chunks embedded and stored.")
else:
    print("✅ No new or changed files to process.")