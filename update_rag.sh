#!/bin/bash

echo "Rebuilding blog-rag container..."

# Rebuild image
docker build -t blog-rag .

# Restart container
docker compose down
docker compose up -d

# Show logs
docker compose logs -f