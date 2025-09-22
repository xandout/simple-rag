#!/bin/bash

# Set up environment
RAG_PORT=5001  # Matches your curl command
RAG_HOST=${RAG_HOST:-localhost}  # Use localhost or override with proxy URL (e.g., your-pod-id-5001.proxy.runpod.net)
FILE_PATH="${1:-/data/documents/default.txt}"  # Use arg or default file

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File $FILE_PATH does not exist"
    exit 1
fi

# Post file to RAG /upload endpoint
curl -s -X POST http://$RAG_HOST:$RAG_PORT/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$FILE_PATH"

echo "File $FILE_PATH uploaded to RAG server"