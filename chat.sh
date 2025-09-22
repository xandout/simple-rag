#!/bin/bash

MODEL_NAME=${MODEL_NAME:-google/gemma-3-1b-pt}  # Default; override via env

PROMPT=${@:1}
curl http://localhost:5000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "${MODEL_NAME}",
    "prompt": "${PROMPT}",
    "max_tokens": 50,
    "temperature": 0.0,
    "logprobs": false,
    "top_k": -1,
    "top_p": 1.0
  }'