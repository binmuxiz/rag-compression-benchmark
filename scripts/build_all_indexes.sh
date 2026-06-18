#!/bin/bash
FLAT=/app/index_output/e5_Flat.index
SAVE=/app/index_output

# 그룹 A: nlist 스윕 (M=32, 8bit 고정)
for NLIST in 1024 2048 4096 8192 16384; do
  echo "=== Building IVF${NLIST}_PQ32 ==="
  python build_ivf.py --flat_index_path $FLAT --nlist $NLIST --M 32 --nbits 8 --save_dir $SAVE --metric ip
done

# 그룹 B: PQ 스윕 (nlist=8192 고정)
for M in 64 96; do
  echo "=== Building IVF8192_PQ${M} ==="
  python build_ivf.py --flat_index_path $FLAT --nlist 8192 --M $M --nbits 8 --save_dir $SAVE --metric ip
done
