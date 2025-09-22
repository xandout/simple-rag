#!/bin/bash

export RAG_PORT=5001
export HF_HOME=./models
export PYTHONUNBUFFERED=1


# Start FastAPI
uvicorn rag_api:app --host 0.0.0.0 --port $RAG_PORT &

echo "RAG server ready on port $RAG_PORT."