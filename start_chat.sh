#!/bin/bash

# Set up environment
export WEBUI_PORT=5002
export PYTHONUNBUFFERED=1
export VECTOR_DB=pgvector
export PGVECTOR_DB_URL=postgresql://postgres:RagDoll42@localhost:5432/rag_db
export PGVECTOR_CREATE_EXTENSION=false
export OPENAI_API_BASE_URL=http://localhost:5000/v1
# Start Open WebUI
GLOBAL_LOG_LEVEL="DEBUG" open-webui serve \
    --port $WEBUI_PORT \
    --host 0.0.0.0 \

echo "Chat UI ready on port $WEBUI_PORT. Connects to LLM at $OPENAI_API_BASE_URL and RAG via $PGVECTOR_DB_URL."