#!/bin/bash
SCRIPT=/app/scripts/run_flashrag.py
CONFIG=/app/configs/my_config.yaml
IDX=/app/index_output
OUT=/app/output/full
NPROBE=32

declare -A INDEXES=(
	["flat"]="$IDX/e5_Flat.index"
  ["IVF1024_PQ32_8bit"]="$IDX/wiki18_IVF1024_PQ32_8bit.index"
  ["IVF2048_PQ32_8bit"]="$IDX/wiki18_IVF2048_PQ32_8bit.index"
  ["IVF4096_PQ32_8bit"]="$IDX/wiki18_IVF4096_PQ32_8bit.index"
  ["IVF8192_PQ32_8bit"]="$IDX/wiki18_IVF8192_PQ32_8bit.index"
  ["IVF16384_PQ32_8bit"]="$IDX/wiki18_IVF16384_PQ32_8bit.index"
  ["IVF8192_PQ64_8bit"]="$IDX/wiki18_IVF8192_PQ64_8bit.index"
  ["IVF8192_PQ96_8bit"]="$IDX/wiki18_IVF8192_PQ96_8bit.index"
  ["IVF8192_PQ64_4bit"]="$IDX/wiki18_IVF8192_PQ64_4bit.index"
)

for tag in "${!INDEXES[@]}"; do
  echo ""
  echo "===== Running $tag ====="

  if [ "$tag" == "flat" ]; then
    python $SCRIPT --config $CONFIG --index_path "${INDEXES[$tag]}" \
      --index_tag "$tag" --llm_quant bf16 \
      --output_dir $OUT
  else
    python $SCRIPT --config $CONFIG --index_path "${INDEXES[$tag]}" \
      --index_tag "$tag" --llm_quant bf16 --nprobe $NPROBE \
      --output_dir $OUT
  fi
done
