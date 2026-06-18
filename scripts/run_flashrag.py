import os
import time
import json
import argparse
import threading

import torch
import psutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate


# --------------------------------------------------------------------------
# Argument parsing
#   고정 설정: my_config.yaml
#   실험마다 바뀌는 것만 CLI 인자로 주입
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="my_config.yaml",
                    help="Path to the base YAML config file")
parser.add_argument("--index_path", type=str, required=True,
                    help="Path to the FAISS index for this run")
parser.add_argument("--index_tag", type=str, required=True,
                    help="Index label for bookkeeping, e.g. flat / IVF1024_PQ32_8bit")
parser.add_argument("--llm_quant", type=str, default="bf16",
                    help="LLM quantization label, e.g. bf16 / int8 / int4")
parser.add_argument("--test_sample_num", type=int, default=None,
                    help="Override test_sample_num in config (None = use config value)")
parser.add_argument("--ram_poll_interval", type=float, default=0.1,
                    help="Seconds between system-RAM samples")
parser.add_argument("--output_dir", type=str, default="/app/output",
                    help="Directory to save summary JSON (pod-local path)")
parser.add_argument("--nprobe", type=int, default=None,
                    help="IVF nprobe override. Flat은 무시됨. 미지정시 config/기본값 사용")
args = parser.parse_args()


# --------------------------------------------------------------------------
# Background sampler for peak system memory.
#   주의: 이 측정의 "의미"는 환경에 따라 다르다.
#   - Jetson(unified): CPU·GPU가 LPDDR5X 공유 → 진짜 통합 메모리 압박
#   - H200(분리형): GPU 메모리와 RAM이 별개 → 이 값은 "RAM 사용량"(주로 FAISS)
#   FAISS 인덱스 메모리는 torch CUDA allocator에 잡히지 않으므로
#   psutil 폴링으로 전체 run의 peak system RAM을 캡처한다.
# --------------------------------------------------------------------------
class MemorySampler(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self.peak_used_mb = 0.0
        self.peak_used_pct = 0.0
        self.baseline_used_mb = psutil.virtual_memory().used / (1024 ** 2)

    def run(self):
        while not self._stop_event.is_set():
            vm = psutil.virtual_memory()
            used_mb = vm.used / (1024 ** 2)
            if used_mb > self.peak_used_mb:
                self.peak_used_mb = used_mb
                self.peak_used_pct = vm.percent
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)


# --------------------------------------------------------------------------
# Config 로드
#   config_dict로 CLI 인자를 오버라이드한다.
#   FlashRAG Config는 config_dict 값이 YAML보다 우선순위가 높다.
# --------------------------------------------------------------------------
override = {"index_path": args.index_path}
if args.test_sample_num is not None:
    override["test_sample_num"] = args.test_sample_num
if args.nprobe is not None:
    # IVF 인덱스의 검색 클러스터 수. Flat에는 영향 없음.
    override["faiss_search_params"] = {"nprobe": args.nprobe}

config = Config(config_file_path=args.config, config_dict=override)

all_split = get_dataset(config)
test_data = all_split["test"]

prompt_template = PromptTemplate(
    config,
    system_prompt=(
        "Answer the question based on the given document. "
        "Only give me the answer and do not output any other words. "
        "\nThe following are given documents.\n\n{reference}"
    ),
    user_prompt="Question: {question}\nAnswer:",
)

pipeline = SequentialPipeline(config, prompt_template=prompt_template)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ── Run ──────────────────────────────────────────────────────────────────
mem_sampler = MemorySampler(interval=args.ram_poll_interval)
mem_sampler.start()

start_time = time.perf_counter()
output_dataset = pipeline.run(test_data, do_eval=True)
end_time = time.perf_counter()

mem_sampler.stop()

# ── Timing metrics ────────────────────────────────────────────────────────
# NOTE on TTFT: SequentialPipeline은 배치 단위로 처리하므로 정확한
# Time-To-First-Token을 복원할 수 없다. avg latency / QPS 만 보고한다.
total_time_sec = end_time - start_time
num_samples = len(output_dataset.pred) if output_dataset.pred is not None else 0
avg_latency_sec = total_time_sec / num_samples if num_samples > 0 else None
qps = num_samples / total_time_sec if total_time_sec > 0 else None

# ── Memory metrics ────────────────────────────────────────────────────────
peak_allocated_mb = None
peak_reserved_mb = None
if torch.cuda.is_available():
    torch.cuda.synchronize()
    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

peak_system_ram_used_mb = mem_sampler.peak_used_mb
baseline_system_ram_used_mb = mem_sampler.baseline_used_mb
peak_system_ram_pct = mem_sampler.peak_used_pct
final_system_ram_used_mb = psutil.virtual_memory().used / (1024 ** 2)

eval_result = getattr(output_dataset, "metric_score", None)

# ── 집계 평균 계산 ────────────────────────────────────────────────────────
# FlashRAG가 per-sample 리스트를 주는 경우 평균을 직접 계산.
# 이미 평균 dict면 그대로 사용.
eval_mean = None
if isinstance(eval_result, list) and len(eval_result) > 0 and isinstance(eval_result[0], dict):
    keys = eval_result[0].keys()
    n = len(eval_result)
    eval_mean = {k: sum(e.get(k, 0) for e in eval_result) / n for k in keys}
elif isinstance(eval_result, dict):
    eval_mean = eval_result  # 이미 집계된 형태


# ── Print summary ─────────────────────────────────────────────────────────
# 베이스라인 디버깅용: 예측만이 아니라 질문/정답/검색문서까지 같이 출력
# (6/4의 EM/F1 급감 원인 추적 — 프롬프트/매칭/검색 어디서 깨지는지 확인)
print("\n--- DEBUG samples (first 3: question / gold / pred / retrieved) ---")
try:
    for i in range(min(3, num_samples)):
        item = test_data[i]
        q = getattr(item, "question", None)
        gold = getattr(item, "golden_answers", None)
        pred = output_dataset.pred[i] if output_dataset.pred is not None else None
        print(f"\n[{i}] Q: {q}")
        print(f"    GOLD: {gold}")
        print(f"    PRED: {pred}")
        # 검색된 문서가 실제로 정답을 담고 있는지 눈으로 확인
        retr = getattr(item, "retrieval_result", None)
        if retr:
            first_doc = retr[0]
            doc_text = first_doc.get("contents", first_doc) if isinstance(first_doc, dict) else first_doc
            print(f"    TOP1-DOC: {str(doc_text)[:200]}...")
except Exception as e:
    print(f"(debug sample print skipped: {e})")

print("\n--- runtime summary ---")
print(f"index_tag:                    {args.index_tag}")
print(f"llm_quant:                    {args.llm_quant}")
print(f"num_samples:                  {num_samples}")
print(f"total_time_sec:               {total_time_sec:.4f}")
if avg_latency_sec is not None:
    print(f"avg_latency_sec_per_sample:   {avg_latency_sec:.4f}")
if qps is not None:
    print(f"qps:                          {qps:.4f}")
if peak_allocated_mb is not None:
    print(f"peak_gpu_allocated_mb (torch):{peak_allocated_mb:.2f}")
if peak_reserved_mb is not None:
    print(f"peak_gpu_reserved_mb  (torch):{peak_reserved_mb:.2f}")
print(f"baseline_system_ram_mb:       {baseline_system_ram_used_mb:.2f}")
print(f"peak_system_ram_mb:           {peak_system_ram_used_mb:.2f}")
print(f"peak_system_ram_pct:          {peak_system_ram_pct:.2f}%")
print(f"delta_system_ram_mb:          {peak_system_ram_used_mb - baseline_system_ram_used_mb:.2f}")
if eval_result is not None:
    print(f"flashrag_eval:                {eval_result}")

# ── Save JSON summary ─────────────────────────────────────────────────────
result_summary = {
    # co-design bookkeeping
    "config_path": args.config,
    "index_path": args.index_path,
    "index_tag": args.index_tag,
    "llm_quant": args.llm_quant,
    "retrieval_topk": config.retrieval_topk,
    "nprobe": args.nprobe,
    # timing
    "num_samples": num_samples,
    "total_time_sec": total_time_sec,
    "avg_latency_sec_per_sample": avg_latency_sec,
    "qps": qps,
    # memory
    "peak_gpu_allocated_mb_torch": peak_allocated_mb,
    "peak_gpu_reserved_mb_torch": peak_reserved_mb,
    "baseline_system_ram_mb": baseline_system_ram_used_mb,
    "peak_system_ram_mb": peak_system_ram_used_mb,
    "peak_system_ram_pct": peak_system_ram_pct,
    "delta_system_ram_mb": peak_system_ram_used_mb - baseline_system_ram_used_mb,
    "final_system_ram_mb": final_system_ram_used_mb,
    # lexical quality
    "flashrag_eval_mean": eval_mean,       # ← 집계 평균 (메인으로 볼 값)
    "flashrag_eval": eval_result,           # ← per-sample 원본 (보존)
}

os.makedirs(args.output_dir, exist_ok=True)
result_path = os.path.join(
    args.output_dir, f"summary__{args.index_tag}__{args.llm_quant}.json"
)
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result_summary, f, ensure_ascii=False, indent=2)

print(f"\nsaved summary to: {result_path}")