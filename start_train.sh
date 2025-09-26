#!/usr/bin/env bash

set -euo pipefail

# Resolve script directory so paths work regardless of the current working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/train_lora.py"

# Exposed defaults (override by exporting env vars before running)
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/pokemon.json}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-1b-it}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../adapters/gemma3-1b-it-lora}"
RANK_R="${R:-8}"
LORA_ALPHA_VAL="${LORA_ALPHA:-16}"
LORA_DROPOUT_VAL="${LORA_DROPOUT:-0.05}"
BATCH_SIZE_VAL="${BATCH_SIZE:-2}"
GRAD_ACCUM_VAL="${GRAD_ACCUM:-8}"
LR_VAL="${LR:-2e-4}"
EPOCHS_VAL="${EPOCHS:-1}"
MAX_LEN_VAL="${MAX_LEN:-2048}"
FP16_ENV="${FP16:-true}"

# Determine whether to pass --fp16 flag
to_lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }
FP16_LOWER="$(to_lower "$FP16_ENV")"
FP16_FLAG=""
if [[ "$FP16_LOWER" == "1" || "$FP16_LOWER" == "true" || "$FP16_LOWER" == "yes" ]]; then
  FP16_FLAG="--fp16"
fi

echo "Starting LoRA training with parameters:"
echo "  DATA_PATH        = $DATA_PATH"
echo "  MODEL_NAME       = $MODEL_NAME"
echo "  OUTPUT_DIR       = $OUTPUT_DIR"
echo "  R (rank)         = $RANK_R"
echo "  LORA_ALPHA       = $LORA_ALPHA_VAL"
echo "  LORA_DROPOUT     = $LORA_DROPOUT_VAL"
echo "  BATCH_SIZE       = $BATCH_SIZE_VAL"
echo "  GRAD_ACCUM       = $GRAD_ACCUM_VAL"
echo "  LR               = $LR_VAL"
echo "  EPOCHS           = $EPOCHS_VAL"
echo "  MAX_LEN          = $MAX_LEN_VAL"
echo "  FP16             = ${FP16_FLAG:+enabled}${FP16_FLAG:="disabled"}"

# Call the training script in the foreground
python "$PY_SCRIPT" \
  --data "$DATA_PATH" \
  --model "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --r "$RANK_R" \
  --lora_alpha "$LORA_ALPHA_VAL" \
  --lora_dropout "$LORA_DROPOUT_VAL" \
  --batch_size "$BATCH_SIZE_VAL" \
  --grad_accum "$GRAD_ACCUM_VAL" \
  --lr "$LR_VAL" \
  --epochs "$EPOCHS_VAL" \
  --max_len "$MAX_LEN_VAL" \
  ${FP16_FLAG}

