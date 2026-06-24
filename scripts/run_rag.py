import os
import time
import json
import yaml
import argparse
import threading

import torch
import psutil


os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

import faiss

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the base YAML config fi`le")
parser.add_argument("--index_path", type=str, required=True, help="Path to the FAISS index for this run")
parser.add_argument("--index_tag", type=str, required=True, help="Index label for bookkeeping")
parser.add_argument("--llm_quant", type=str, default="bf16", help="LLM quantization label")
parser.add_argument("--test_sample_num", type=int, default=None, help="Override test_sample_num in config  (None = use config value)")
parser.add_argument("--ram_poll_interval", type=float, default=0.1, help="Seconds between system-RAM samples")
parser.add_argument("--nprobe", type=int, default=None, help="IVF nprobe override")
parser.add_argument("--save_dir", type=str, default=None, help="Override save_dir in config (FlashRAG intermediate_data 저장 위치)")
args = parser.parse_args()

# --------------------------------------------------------------------------
# Background sampler for peak system memory.
#   - Jetson(unified): CPU·GPU가 LPDDR5X 공유 
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
if args.save_dir is not None:
    override["save_dir"] = args.save_dir 
override["save_note"] = f"{args.index_tag}__{args.llm_quant}"

config = Config(config_file_path=args.config, config_dict=override)

save_dir = config["save_dir"]
print(f"===============================[INFO] FlashRAG save_dir = {save_dir}===============================")


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


# ── Run Pipeline ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

mem_sampler = MemorySampler(interval=args.ram_poll_interval)
mem_sampler.start()
start_time = time.perf_counter()


# NOTE : SequentialPipeline은 배치 단위로 처리하므로 정확한 Time-To-First-Token을 복원할 수 없음.
# https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/pipeline/pipeline.py
pipeline = SequentialPipeline(config, prompt_template=prompt_template)

# nprobe 직접 주입
if args.nprobe is not None:
    retriever = pipeline.retriever
    inner_index = faiss.downcast_index(retriever.index)
    if hasattr(inner_index, "nprobe"):
        inner_index.nprobe = args.nprobe
        print(f"==============[INFO] nprobe set to {inner_index.nprobe}=============")
    else:
        print("==============[INFO] Flat index - nprobe 무시=============")

retrieval_peak_vram = None
generation_peak_vram = None

# [Step 1] Retrieval 구간 
# e5 임베딩은 GPU, 검색은 CPU
retrieval_start = time.perf_counter()
input_query = test_data.question
num_samples = len(input_query)
retrieval_results = pipeline.retriever.batch_search(input_query)

if torch.cuda.is_available():  # 넣어도 빼도 무관하지만 일단 안전장치로 남겨놓음..
    torch.cuda.synchronize() 
retrieval_end = time.perf_counter()
test_data.update_output("retrieval_result", retrieval_results)

# retrieval 끝난 뒤, generation 시작 전
if torch.cuda.is_available():
    torch.cuda.synchronize()
    retrieval_peak_vram = torch.cuda.max_memory_allocated() / 1024**2  # e5 구간 피크 (원하면 기록)
    torch.cuda.reset_peak_memory_stats()   # ← 여기서 리셋

# [Step 2] Generation 구간
# 프롬프트 템플릿 적용 및 Generator 실행
input_prompts = [
    pipeline.prompt_template.get_string(question=q, retrieval_result=r)
    for q, r in zip(test_data.question, test_data.retrieval_result)
]
test_data.update_output("prompt", input_prompts) 

if torch.cuda.is_available():
    torch.cuda.synchronize()
generation_start = time.perf_counter()
pred_answers = pipeline.generator.generate(input_prompts)

if torch.cuda.is_available():
    torch.cuda.synchronize()
generation_end = time.perf_counter()

if torch.cuda.is_available():
    generation_peak_vram = torch.cuda.max_memory_allocated() / 1024**2  # LLM 구간 피크만!

end_time = time.perf_counter()
mem_sampler.stop()

test_data.update_output("pred", pred_answers)
test_data = pipeline.evaluate(test_data, do_eval=True)

for i in range(10):
    print(f"GOLD: {test_data[i].golden_answers}")
    print(f"PRED: {pred_answers[i]}")
    print("---")
# print("[DEBUG] evaluate 반환:", type(test_data), test_data)
# print("[DEBUG] metric_score 속성:", getattr(test_data, "metric_score", "없음"))
# ── 집계 평균 계산 ────────────────────────────────────────────────────────
# per-sample 리스트를 반환
eval_result = getattr(test_data, "metric_score", None)

eval_mean = None
if isinstance(eval_result, list) and len(eval_result) > 0 and isinstance(eval_result[0], dict):
    keys = eval_result[0].keys()
    n = len(eval_result)
    eval_mean = {k: sum(e.get(k, 0) for e in eval_result) / n for k in keys}
elif isinstance(eval_result, dict):
    eval_mean = eval_result  # 이미 집계된 형태

# ── Memory metrics ────────────────────────────────────────────────────────
# peak_gpu_allocated_mb = None
# peak_gpu_reserved_mb = None
# if torch.cuda.is_available():
#     peak_gpu_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

baseline_ram_used_mb = mem_sampler.baseline_used_mb     # 초기 RAM 사용량 
peak_ram_used_mb = mem_sampler.peak_used_mb             # 최대 RAM 사용량
peak_ram_pct = mem_sampler.peak_used_pct                # 최대 RAM 사용률(%)
delta_ram_mb = peak_ram_used_mb - baseline_ram_used_mb
final_ram_used_mb = psutil.virtual_memory().used / (1024 ** 2)



# ── 구간별 시간 계산 ────────────────────────────────────────────────────────
total_time_sec = end_time - start_time
retrieval_time_sec = retrieval_end - retrieval_start
generation_time_sec = generation_end - generation_start
processing_time_sec = retrieval_time_sec + generation_time_sec


# 초당 쿼리
total_qps = num_samples / total_time_sec if total_time_sec > 0 else None
retrieval_qps = num_samples / retrieval_time_sec if retrieval_time_sec > 0 else None
generation_qps = num_samples / generation_time_sec if generation_time_sec > 0 else None
processing_qps = num_samples / processing_time_sec


# ── Save JSON summary ─────────────────────────────────────────────────────
result_summary = {
    "metadata": {
        "config_path": args.config,
        "index_path": args.index_path,
        "index_tag": args.index_tag,
        "llm_quant": args.llm_quant,
        "test_sample_num" : args.test_sample_num,
        "retrieval_topk": config.retrieval_topk,
        "nprobe": args.nprobe,
    },
    "runtime": {
        "num_samples": num_samples,

        "total_time_sec": total_time_sec,    # 인덱스/모델/코퍼스 로딩 + 실체 처리 시간 다 합친 시간.
        "retrieval_time_sec": retrieval_time_sec,    
        "generation_time_sec": generation_time_sec,
        "processing_time_sec": processing_time_sec,

        "total_qps": total_qps,
        "processing_qps": processing_qps,
        "retrieval_qps" : retrieval_qps,
        "generation_qps": generation_qps,
    },
    "memory": {
        # VRAM -cuda
        "retrieval_peak_vram" : retrieval_peak_vram,  # retrieval 구간 peak인데, gpu에 llm이 미리 로드되어 있어서 의미X 
        "generation_peak_vram": generation_peak_vram, # LLM(bf16) VRAM
        # RAM -psutil
        "baseline_ram_used_mb": baseline_ram_used_mb,
        "peak_ram_used_mb": peak_ram_used_mb,
        "delta_ram_mb": delta_ram_mb,                  # 인덱스 + 코퍼스 
        "final_ram_mb": final_ram_used_mb,

        "peak_ram_pct": peak_ram_pct,

    },
    "eval_mean": eval_mean, # 집계 평균
    "eval_per-sample": eval_result,    # per-sample 원본
}

# ── Save Raw JSON ─────────────────────────────────────────────────────────
output_path = os.path.join(save_dir, f"summary__{args.index_tag}__{args.llm_quant}.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result_summary, f, ensure_ascii=False, indent=2)

print(f"\n[SUCCESS] Saved summary to: {output_path}")