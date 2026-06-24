#!/bin/bash
SCRIPT=/app/scripts/run_rag.py
CONFIG=/app/configs/my_config.yaml
IDX=/app/index_output
OUT=/app/output/stage1_indexsweep_20260623
# OUT=/app/output/_smoketest

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

# nprobe = round(4 * sqrt(nlist))
# flat은 IVF가 아니므로 nprobe 없음(빈 값)
declare -A NPROBES=(
  ["flat"]=""
  ["IVF1024_PQ32_8bit"]=128
  ["IVF2048_PQ32_8bit"]=181
  ["IVF4096_PQ32_8bit"]=256
  ["IVF8192_PQ32_8bit"]=363
  ["IVF16384_PQ32_8bit"]=512
  ["IVF8192_PQ64_8bit"]=363
  ["IVF8192_PQ96_8bit"]=363
  ["IVF8192_PQ64_4bit"]=363
)

for tag in "${!INDEXES[@]}"; do
    echo ""
    echo "===== Running $tag ====="

    nprobe="${NPROBES[$tag]}"

    if [[ "$tag" == "flat" ]]; then
        python $SCRIPT \
            --config $CONFIG \
            --index_path "${INDEXES[$tag]}" \
            --index_tag "$tag" \
            --llm_quant bf16 \
            --save_dir "$OUT" \
            --test_sample_num 300
    else
        echo "  nprobe = $nprobe"
        python $SCRIPT \
            --config $CONFIG \
            --index_path "${INDEXES[$tag]}" \
            --index_tag "$tag" \
            --llm_quant bf16 \
            --nprobe "$nprobe" \
            --save_dir "$OUT" \
            --test_sample_num 300
    fi
done

