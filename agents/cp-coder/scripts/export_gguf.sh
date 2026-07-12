#!/usr/bin/env bash
# Export a trained HuggingFace model to GGUF format for Ollama deployment.
#
# Prerequisites:
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp && make -j$(nproc)
#   pip install -r llama.cpp/requirements.txt
#
# Usage:
#   bash scripts/export_gguf.sh \
#       --model ./outputs/sft/best_model \
#       --output ./outputs/gguf \
#       --quantize q4_k_m \
#       --llama-cpp ./llama.cpp

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
MODEL_DIR="./outputs/sft/best_model"
OUTPUT_DIR="./outputs/gguf"
QUANTIZE="q4_k_m"          # Options: f16, q8_0, q4_k_m, q5_k_m, q2_k, q3_k_m
LLAMA_CPP_DIR="./llama.cpp"
MODEL_NAME="cp-coder"
OLLAMA_PUSH=false

# ── Argument parsing ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_DIR="$2";      shift 2 ;;
        --output)     OUTPUT_DIR="$2";     shift 2 ;;
        --quantize)   QUANTIZE="$2";       shift 2 ;;
        --llama-cpp)  LLAMA_CPP_DIR="$2";  shift 2 ;;
        --name)       MODEL_NAME="$2";     shift 2 ;;
        --ollama-push) OLLAMA_PUSH=true;   shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== GGUF Export ==="
echo "  Model:     $MODEL_DIR"
echo "  Output:    $OUTPUT_DIR"
echo "  Quantize:  $QUANTIZE"
echo "  llama.cpp: $LLAMA_CPP_DIR"

mkdir -p "$OUTPUT_DIR"

# ── Check llama.cpp ───────────────────────────────────────────────────────
if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    echo "ERROR: llama.cpp directory not found at $LLAMA_CPP_DIR"
    echo "Clone it with: git clone https://github.com/ggerganov/llama.cpp"
    exit 1
fi

CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    # Older path
    CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert.py"
fi
if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    echo "ERROR: Cannot find convert script in $LLAMA_CPP_DIR"
    exit 1
fi

QUANTIZE_BIN="$LLAMA_CPP_DIR/llama-quantize"
if [[ ! -f "$QUANTIZE_BIN" ]]; then
    QUANTIZE_BIN="$LLAMA_CPP_DIR/quantize"
fi

# ── Step 1: Convert to F16 GGUF ──────────────────────────────────────────
F16_PATH="$OUTPUT_DIR/${MODEL_NAME}-f16.gguf"
echo ""
echo "Step 1: Converting to F16 GGUF..."
python "$CONVERT_SCRIPT" \
    "$MODEL_DIR" \
    --outtype f16 \
    --outfile "$F16_PATH"

echo "F16 GGUF saved: $F16_PATH"

# ── Step 2: Quantize ─────────────────────────────────────────────────────
if [[ "$QUANTIZE" != "f16" ]]; then
    QUANT_PATH="$OUTPUT_DIR/${MODEL_NAME}-${QUANTIZE}.gguf"
    QUANT_TYPE_UPPER="${QUANTIZE^^}"

    echo ""
    echo "Step 2: Quantizing to $QUANTIZE ..."
    if [[ -f "$QUANTIZE_BIN" ]]; then
        "$QUANTIZE_BIN" "$F16_PATH" "$QUANT_PATH" "$QUANT_TYPE_UPPER"
        echo "Quantized GGUF saved: $QUANT_PATH"
    else
        echo "WARNING: quantize binary not found at $QUANTIZE_BIN"
        echo "Build llama.cpp first: cd $LLAMA_CPP_DIR && make -j\$(nproc)"
        QUANT_PATH="$F16_PATH"
    fi
else
    QUANT_PATH="$F16_PATH"
fi

# ── Step 3: Create Modelfile for Ollama ──────────────────────────────────
MODELFILE_PATH="$OUTPUT_DIR/Modelfile"
cat > "$MODELFILE_PATH" <<EOF
FROM $QUANT_PATH

# System prompt for competitive programming assistance
SYSTEM """You are an expert competitive programmer. You solve algorithmic problems with optimal time and space complexity. Always analyze the constraints carefully, choose the right algorithm (DP, graphs, greedy, number theory, etc.), and write clean, correct code."""

# Generation parameters tuned for code generation
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 4096
PARAMETER stop "### Problem"
PARAMETER stop "<|endoftext|>"
EOF

echo ""
echo "Modelfile created: $MODELFILE_PATH"

# ── Step 4: Load into Ollama ─────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    echo ""
    echo "Step 3: Loading into Ollama as '$MODEL_NAME' ..."
    ollama create "$MODEL_NAME" -f "$MODELFILE_PATH"
    echo ""
    echo "Done! Test with:"
    echo "  ollama run $MODEL_NAME"
else
    echo ""
    echo "Ollama not found. After installing, run:"
    echo "  ollama create $MODEL_NAME -f $MODELFILE_PATH"
    echo "  ollama run $MODEL_NAME"
fi

if [[ "$OLLAMA_PUSH" == true ]]; then
    echo "Pushing to Ollama registry ..."
    ollama push "$MODEL_NAME"
fi

echo ""
echo "=== Export Complete ==="
echo "  GGUF: $QUANT_PATH"
echo "  Size: $(du -sh "$QUANT_PATH" 2>/dev/null | cut -f1 || echo 'unknown')"
