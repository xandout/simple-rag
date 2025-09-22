#!/bin/bash

# Set up environment
export VLLM_PORT=5000
export HF_HOME=./models
export PYTHONUNBUFFERED=1
export MODEL_NAME=${MODEL_NAME:-google/gemma-3-1b-it}  # Default; override via env
export LORA_MODULES=${LORA_MODULES:-}  # Optional: e.g., "lora_adapter=/data/adapters/my-lora"


# Start vLLM server
vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    ${LORA_MODULES:+--lora-modules $LORA_MODULES} &

echo "LLM server ready on port $VLLM_PORT. Set MODEL_NAME and LORA_MODULES to swap models/adapters."