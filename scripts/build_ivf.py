import faiss
import os
import argparse
import numpy as np
import gc
import time

parser = argparse.ArgumentParser()
parser.add_argument("--flat_index_path", type=str, required=True)
parser.add_argument("--nlist", type=int, required=True)
parser.add_argument("--M", type=int, default=0, help="PQ sub-vectors. If 0, use pure IVF (No Quantization)")
parser.add_argument("--nbits", type=int, default=8, help="PQ bits per sub-vector (default: 8)")
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"])
parser.add_argument("--train_size", type=int, default=0,
                    help="Train sample size. 0이면 nlist 기반 자동 설정")
parser.add_argument("--chunk_size", type=int, default=500000,
                    help="Chunk size for adding vectors (256GB면 크게 잡아도 됨)")
args = parser.parse_args()

# ============================================================
# 1. Flat 인덱스 읽기 (mmap)
# ============================================================
print("Reading flat index metadata...")
flat_index = faiss.read_index(args.flat_index_path, faiss.IO_FLAG_MMAP)

if hasattr(flat_index, "make_direct_map"):
    try:
        flat_index.make_direct_map()
    except Exception as e:
        print(f"(make_direct_map skipped: {e})")

total_vectors = flat_index.ntotal
dim = flat_index.d
print(f"Total Vectors: {total_vectors}, Dimension: {dim}")

# ============================================================
# 1-1. 안전 점검: metric + 정규화 상태
# ============================================================
flat_metric = flat_index.metric_type
print(f"[CHECK] Flat index metric_type: {flat_metric} (0=INNER_PRODUCT, 1=L2)")

requested_metric = (faiss.METRIC_INNER_PRODUCT
                    if args.metric == "ip" else faiss.METRIC_L2)
if flat_metric != requested_metric:
    print(f"[WARNING] Flat metric ({flat_metric}) != --metric {args.metric} "
          f"({requested_metric}). 확인 필요!")

sample = np.asarray(flat_index.reconstruct_n(0, 5), dtype=np.float32)
sample_norms = np.linalg.norm(sample, axis=1)
already_normalized = np.allclose(sample_norms, 1.0, atol=1e-3)
print(f"[CHECK] Sample norms: {sample_norms}")
print(f"[CHECK] Already L2-normalized: {already_normalized}")

do_normalize = (args.metric == "ip")

# ============================================================
# 1-2. train_size 자동 결정 (nlist 기반)
#       FAISS 권장: nlist당 약 39~256 샘플. 여기선 넉넉히 256배 사용.
#       (256GB 메모리라 train 샘플을 크게 써도 부담 없음)
# ============================================================
if args.train_size > 0:
    train_size = min(args.train_size, total_vectors)
else:
    # nlist * 256 을 기본 목표로, 최소 10만 ~ 최대 전체의 1/4 사이로 클램프
    target = args.nlist * 256
    target = max(target, 100000)
    target = min(target, total_vectors, total_vectors // 4 + 1)
    train_size = target
print(f"[INFO] nlist={args.nlist} → train_size={train_size} "
      f"(FAISS 권장 최소 = nlist*39 = {args.nlist*39})")

# ============================================================
# 2. 퀀타이저 / 인덱스 정의
# ============================================================
if args.metric == "ip":
    quantizer = faiss.IndexFlatIP(dim)
    metric_type = faiss.METRIC_INNER_PRODUCT
else:
    quantizer = faiss.IndexFlatL2(dim)
    metric_type = faiss.METRIC_L2

if args.M == 0:
    index = faiss.IndexIVFFlat(quantizer, dim, args.nlist, metric_type)
    file_name = f"wiki18_IVF{args.nlist}_Pure.index"
else:
    if dim % args.M != 0:
        raise ValueError(f"M({args.M})은 dim({dim})의 약수여야 합니다. (768 → 32,48,64,96,128...)")
    index = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.M, args.nbits, metric_type)
    file_name = f"wiki18_IVF{args.nlist}_PQ{args.M}_{args.nbits}bit.index"

print(f"Target index: {file_name}")

# ============================================================
# 3. 학습 (Train) — nlist에 맞춘 샘플 수 사용
# ============================================================
start = time.time()
print(f"Step 1: Sampling {train_size} vectors for training...")
train_vectors = np.asarray(flat_index.reconstruct_n(0, train_size), dtype=np.float32)
if do_normalize:
    faiss.normalize_L2(train_vectors)

print(f" -> Training (nlist={args.nlist}) ...")
index.train(train_vectors)
del train_vectors
gc.collect()
print(f" -> Training done ({time.time() - start:.0f}s)")

# ============================================================
# 4. 추가 (Add) — 청크 단위 (256GB면 chunk_size 크게)
# ============================================================
print(f"Step 2: Adding vectors in chunks (chunk_size={args.chunk_size})...")
for i in range(0, total_vectors, args.chunk_size):
    end_idx = min(i + args.chunk_size, total_vectors)
    chunk = np.asarray(flat_index.reconstruct_n(i, end_idx - i), dtype=np.float32)
    if do_normalize:
        faiss.normalize_L2(chunk)
    index.add(chunk)
    print(f" -> Added {i} ~ {end_idx} / {total_vectors} ({time.time() - start:.0f}s)")
    del chunk
    gc.collect()

# ============================================================
# 5. 저장
# ============================================================
os.makedirs(args.save_dir, exist_ok=True)
save_path = os.path.join(args.save_dir, file_name)
faiss.write_index(index, save_path)
print(f"=== Successfully Finished! Saved to {save_path} (total {time.time() - start:.0f}s) ===\n")