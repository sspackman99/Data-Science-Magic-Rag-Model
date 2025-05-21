# Blog RAG API

A Retrieval-Augmented Generation (RAG) API for Sam Spackman's "Data Science Magic" blog. This system uses ChromaDB as a vector database to store and retrieve relevant blog post content, combined with an LLM to generate informative responses.

## Overview

This repository contains a system that:

1. Processes Markdown blog posts from a blog collection
2. Embeds and stores these documents in ChromaDB with metadata
3. Provides an API that retrieves relevant context when queried
4. Uses a local LLM (via Ollama) to generate responses that incorporate the retrieved information

## Components

- `populate_database.py`: Processes blog post files, chunks them, generates summaries, and stores them in ChromaDB
- `query_data.py`: Provides functions to query the vector database and generate RAG-enhanced responses
- `app.py`: Flask API server (not shown in excerpts)
- `templates/index.html`: Simple web interface for the API
- `posts/`: Directory containing blog post markdown files
- `chroma_store/`: Persistent storage for the ChromaDB vector database

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure Ollama is running locally or set the `OLLAMA_API_URL` environment variable to point to your Ollama instance
4. Run the database population script to process blog posts:
   ```bash
   python populate_database.py
   ```
5. Start the API server:
   ```bash
   python app.py
   ```

## Configuration

Key configuration variables are found at the top of `query_data.py`:

- `CHROMA_PERSIST_DIR`: Location of the ChromaDB storage directory
- `CHROMA_COLLECTION_NAME`: Name of the ChromaDB collection
- `TOP_K`: Number of chunks to retrieve for each query
- `OLLAMA_API_BASE`: URL for the Ollama API
- `OLLAMA_MODEL`: Default model to use for LLM responses

## API Usage

### Chat Endpoint

```
POST /api/chat
```

Request Body:
```json
{
  "messages": [
    {"role": "system", "content": "Your system prompt here"},
    {"role": "user", "content": "User message here"}
  ],
  "model": "llama3.2:3b"
}
```

Response Format:
```json
{
  "message": {
    "content": "LLM response here"
  }
}
```

### Health Check

```
GET /api/health
```

## How It Works

1. When a user submits a query, the system embeds the query using the SentenceTransformer model
2. The query embedding is used to search ChromaDB for the most similar text chunks
3. The top chunks and their metadata (including post summaries) are assembled into a context
4. This context is combined with the user query and sent to the LLM
5. The LLM generates a response that incorporates information from the blog posts

## Development

To add new blog posts:
1. Add markdown files to the `posts/` directory
2. Run `populate_database.py` to process and embed the new content
3. The API will automatically use the updated database for future queries