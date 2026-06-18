#!/bin/bash

OUTPUT_BASE="/app/output/full"
SCRIPT="/app/scripts/evaluate_ragas.py"
PYTHON="/app/ragas_env/bin/python"


DIRS=(
    "wiki18_IVF1024_PQ32_8bit"
    "wiki18_IVF2048_PQ32_8bit"
    "wiki18_IVF4096_PQ32_8bit"
    "wiki18_IVF8192_PQ32_8bit"
    "wiki18_IVF16384_PQ32_8bit"
)

for dir in "${DIRS[@]}"; do
    INPUT="${OUTPUT_BASE}/${dir}/intermediate_data.json"
    OUTPUT="${OUTPUT_BASE}/${dir}/ragas_scores.json"

    if [ ! -f "$INPUT" ]; then
        echo "[SKIP] $dir: intermediate_data.json not found"
        continue
    fi

    if [ -f "$OUTPUT" ]; then
        echo "[SKIP] $dir: ragas_scores.json already exists"
        continue
    fi

    echo "=========================================="
    echo "[START] $dir"
    echo "=========================================="

    "$PYTHON" "$SCRIPT" \
        --input  "$INPUT" \
        --output "$OUTPUT" \
        --max_new_tokens 512 \
        --max_samples 200

    if [ $? -eq 0 ]; then
        echo "[DONE] $dir"
    else
        echo "[ERROR] $dir failed - continuing to next"
    fi
done

echo "=========================================="
echo "All done."
echo "=========================================="
