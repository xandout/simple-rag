#!/bin/bash

# Set up environment
export VLLM_PORT=5000
export HF_HOME=./models
export PYTHONUNBUFFERED=1
export MODEL_NAME=${MODEL_NAME:-google/gemma-3-1b-it}  # Default; override via env
export LORA_MODULES=${LORA_MODULES:-}  # Optional: e.g., "lora_adapter=/data/adapters/my-lora"

# Resolve script directory (for adapter auto-detection)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect LoRA adapter directory if not explicitly provided via LORA_MODULES
if [ -z "$LORA_MODULES" ]; then
    CANDIDATE_DIRS=(
        "$SCRIPT_DIR/../adapters/gemma3-1b-it-lora"
        "$SCRIPT_DIR/../adapters/${MODEL_NAME##*/}-lora"
        "$SCRIPT_DIR/../adapters/lora"
    )
    for d in "${CANDIDATE_DIRS[@]}"; do
        if [ -d "$d" ]; then
            export LORA_MODULES="lora_adapter=$d"
            echo "Detected LoRA adapter: $d"
            break
        fi
    done
fi


# Start vLLM server
vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    ${LORA_MODULES:+--lora-modules $LORA_MODULES} &

if [ -n "$LORA_MODULES" ]; then
    echo "LLM server ready on port $VLLM_PORT with adapters: $LORA_MODULES"
else
    echo "LLM server ready on port $VLLM_PORT (no LoRA adapters detected)."
    echo "Set LORA_MODULES or place an adapter at ../adapters/<name>-lora to auto-load."
fi