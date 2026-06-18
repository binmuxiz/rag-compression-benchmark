"""
evaluate_ragas.py

Evaluates RAG pipeline outputs using 4 RAGAS metrics:
  - Faithfulness
  - ResponseRelevancy (answer_relevancy)
  - ContextRecall
  - ContextPrecision
  - AnswerCorrectness   

Input : intermediate_data.json produced by rag_pipeline.py
Output: ragas_scores.json (per-sample + aggregate scores)

Usage:
    python evaluate_ragas.py \
        --input  /app/output/full/wiki18_e5_Flat/intermediate_data.json \
        --output /app/output/full/wiki18_e5_Flat/ragas_scores.json \
        --max_samples 100          # optional; omit to run on full set
"""

import argparse
import asyncio
import json
import threading
import typing as t
from pathlib import Path

import torch
from langchain_core.outputs import Generation, LLMResult
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy, AnswerCorrectness
from ragas.run_config import RunConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# LocalHuggingFaceRagasLLM  (unchanged from original)
# ---------------------------------------------------------------------------

class LocalHuggingFaceRagasLLM(BaseRagasLLM):
    def __init__(self, model, tokenizer, max_new_tokens: int = 512):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self._generate_lock = threading.Lock()

    def _prompt_to_text(self, prompt) -> str:
        if hasattr(prompt, "to_messages") and self.tokenizer.chat_template:
            messages = []
            for message in prompt.to_messages():
                role = getattr(message, "type", "user")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                elif role != "system":
                    role = "user"
                content = getattr(message, "content", "")
                if not isinstance(content, str):
                    content = str(content)
                messages.append({"role": role, "content": content})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if hasattr(prompt, "to_string"):
            return prompt.to_string()
        return str(prompt)

    def _device(self):
        hf_device_map = getattr(self.model, "hf_device_map", {})
        for device in hf_device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            return next(p.device for p in self.model.parameters() if p.device.type != "meta")
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _apply_stop(text: str, stop: t.Optional[t.List[str]]) -> str:
        if not stop:
            return text
        cut_positions = [pos for word in stop if (pos := text.find(word)) >= 0]
        return text[: min(cut_positions)] if cut_positions else text

    def _generate_one(self, prompt, temperature, stop):
        text = self._prompt_to_text(prompt)
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._device()) for k, v in inputs.items()}

        do_sample = temperature is not None and temperature > 0.01
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._apply_stop(generated_text, stop).strip()

    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n)
        with self._generate_lock:
            generations = [
                Generation(
                    text=self._generate_one(prompt, temperature, stop),
                    generation_info={"finish_reason": "stop"},
                )
                for _ in range(n)
            ]
        return LLMResult(generations=[generations])

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
        return await asyncio.to_thread(
            self.generate_text,
            prompt=prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks,
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(path: str, max_samples: t.Optional[int] = None) -> t.List[SingleTurnSample]:
    """
    Convert intermediate_data.json entries to RAGAS SingleTurnSample objects.

    Field mapping:
        question                          -> user_input
        output.pred                       -> response
        output.retrieval_result[].contents -> retrieved_contexts
        golden_answers[0]                 -> reference   (used by ContextRecall / ContextPrecision)
    """
    with open(path, "r") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    samples = []
    skipped = 0
    for item in data:
        question = item.get("question", "").strip()
        golden_answers = item.get("golden_answers", [])
        output = item.get("output", {})
        pred = output.get("pred", "").strip()
        retrieval_result = output.get("retrieval_result", [])

        # Skip entries where the model refused to answer (no useful pred)
        if not pred or not question:
            skipped += 1
            continue

        retrieved_contexts = [r["contents"] for r in retrieval_result if r.get("contents")]

        # Use the first golden answer as the reference string.
        # ContextRecall / ContextPrecision require a non-empty reference.
        reference = golden_answers[0] if golden_answers else ""

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=pred,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )
        )

    print(f"[load_samples] Loaded {len(samples)} samples, skipped {skipped}.")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG output with 4 RAGAS metrics.")
    parser.add_argument(
        "--input", required=True,
        help="Path to intermediate_data.json produced by rag_pipeline.py",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write ragas_scores.json",
    )
    parser.add_argument(
        "--model_path", default="/app/models/Llama-3.3-70B-Instruct",
        help="Path to the local HuggingFace model used as RAGAS judge LLM",
    )
    parser.add_argument(
        "--embed_model", default="intfloat/e5-base-v2",
        help="HuggingFace embedding model for ResponseRelevancy",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit evaluation to first N samples (useful for debugging)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max new tokens for the judge LLM",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------- load judge LLM ----------
    print(f"[main] Loading judge LLM from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    evaluator_llm = LocalHuggingFaceRagasLLM(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    # ---------- load embedding model ----------
    print(f"[main] Loading embedding model {args.embed_model} ...")
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=args.embed_model,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    # ---------- build metrics ----------
    # All 4 metrics use the same judge LLM.
    # ContextRecall and ContextPrecision additionally need `reference`.
    # ResponseRelevancy additionally needs embeddings.
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextRecall(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    # ---------- load data ----------
    samples = load_samples(args.input, max_samples=args.max_samples)
    if not samples:
        raise RuntimeError("No valid samples found. Check intermediate_data.json.")

    dataset = EvaluationDataset(samples=samples)

    # ---------- evaluate ----------
    print(f"[main] Running RAGAS evaluation on {len(samples)} samples ...")
    result = evaluate(
        dataset=dataset, 
        metrics=metrics,
        run_config=RunConfig(
            timeout=900,
            max_retries=2,
            max_workers=1,
        ),
    )
    print("[main] RAGAS result:", result)

    # ---------- save ----------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # RAGAS 0.4.x prints like a dict, but dict(result) can call __getitem__(0)
    # and fail with KeyError. Read the internal score mapping when available.
    scores_dict = getattr(result, "_scores_dict", None)
    if scores_dict is not None:
        scores_dict = dict(scores_dict)
    else:
        scores_dict = {}
        try:
            scores_dict.update(result.to_pandas().mean(numeric_only=True).to_dict())
        except Exception:
            pass

    # Try to attach per-sample scores if the result object supports it
    try:
        scores_dict["per_sample"] = result.to_pandas().to_dict(orient="records")
    except Exception:
        pass

    with open(out_path, "w") as f:
        json.dump(scores_dict, f, indent=2, ensure_ascii=False, default=str)

    print(f"[main] Scores saved to {out_path}")


if __name__ == "__main__":
    main()
