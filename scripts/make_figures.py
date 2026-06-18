"""
Phase 1 RAG Quantization Experiment — Figure Generator
========================================================
Fig 1: Quality (F1 & Recall) vs nlist         [Group A]
Fig 2: QPS vs nlist                            [Group A]
Fig 3: Quality (F1 & Recall) vs PQ compression [Group B]
Fig 4: QPS vs PQ compression                   [Group B]

Usage:
    python make_figures.py --input_dir /app/output/full --output_dir ./figures

각 figure는 PNG (300 dpi) 형식으로 저장됩니다.
"""

import os
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",  type=str, default="/home/cosail/eunbin/backup/output/full",
                    help="summary JSON 파일들이 있는 디렉토리")
parser.add_argument("--output_dir", type=str, default="/home/cosail/eunbin/figures",
                    help="figure 출력 디렉토리")
parser.add_argument("--llm_quant",  type=str, default="bf16",
                    help="필터링할 llm_quant 태그 (기본: bf16)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

COLOR_F1     = "#2563EB"   # blue
COLOR_RECALL = "#16A34A"   # green
COLOR_QPS    = "#DC2626"   # red
COLOR_FLAT   = "#6B7280"   # gray (baseline)

MARKER_F1     = "o"
MARKER_RECALL = "s"
MARKER_QPS    = "^"


# ── JSON 로딩 ─────────────────────────────────────────────────────────────
def load_summaries(input_dir: str, llm_quant: str) -> dict:
    """index_tag → metric dict 로 로드"""
    results = {}
    for path in sorted(Path(input_dir).glob(f"summary__*__{llm_quant}.json")):
        with open(path) as f:
            d = json.load(f)
        tag  = d["index_tag"]
        mean = d.get("flashrag_eval_mean", {})
        results[tag] = {
            "f1":     mean.get("f1",     None),
            "recall": mean.get("recall", None),
            "qps":    d.get("qps",       None),
        }
    if not results:
        raise FileNotFoundError(
            f"No summary JSON found in {input_dir} with llm_quant='{llm_quant}'. "
            f"파일명 패턴: summary__<index_tag>__{llm_quant}.json"
        )
    return results


def save_fig(fig, name: str, output_dir: str):
    p = os.path.join(output_dir, f"{name}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  saved: {p}")


# ── 데이터 정의 ───────────────────────────────────────────────────────────
# Group A: nlist sweep (M=32, nbits=8 고정)
GROUP_A_TAGS = [
    "IVF1024_PQ32_8bit",
    "IVF2048_PQ32_8bit",
    "IVF4096_PQ32_8bit",
    "IVF8192_PQ32_8bit",
    "IVF16384_PQ32_8bit",
]
GROUP_A_NLIST = [1024, 2048, 4096, 8192, 16384]

# Group B: PQ compression sweep (nlist=8192 고정)
# x축 레이블: "M×nbits" 형태로 압축 강도를 표현
# 압축 후 벡터 크기 = M * nbits bits = M * nbits / 8 bytes
# (낮을수록 강한 압축)
GROUP_B_TAGS   = [
    "IVF8192_PQ64_4bit",   # B4  32B (강한 압축)
    "IVF8192_PQ32_8bit",   # B1  32B
    "IVF8192_PQ64_8bit",   # B2  64B
    "IVF8192_PQ96_8bit",   # B3  96B (약한 압축)
]
GROUP_B_LABELS = [
    "PQ64×4bit\n(32B)",
    "PQ32×8bit\n(32B)",
    "PQ64×8bit\n(64B)",
    "PQ96×8bit\n(96B)",
]
GROUP_B_BYTES = [32, 32, 64, 96]   # M * nbits / 8 (bytes per vector), 강→약

FLAT_TAG = "flat"


# ── Figure 공통 유틸 ──────────────────────────────────────────────────────
def add_flat_hline(ax, value, color, linestyle="--", label=None, linewidth=1.4):
    """flat baseline 수평선"""
    ax.axhline(value, color=color, linestyle=linestyle,
               linewidth=linewidth, alpha=0.75, label=label)


# ══════════════════════════════════════════════════════════════════════════
# Fig 1 — Quality vs nlist (Group A)
# ══════════════════════════════════════════════════════════════════════════
def make_fig1(data: dict, output_dir: str):
    print("\n[Fig 1] Quality vs nlist")

    flat  = data.get(FLAT_TAG, {})
    flat_f1     = flat.get("f1")
    flat_recall = flat.get("recall")

    f1_vals     = [data[t]["f1"]     for t in GROUP_A_TAGS]
    recall_vals = [data[t]["recall"] for t in GROUP_A_TAGS]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(GROUP_A_NLIST, f1_vals,
            color=COLOR_F1, marker=MARKER_F1,
            linewidth=1.8, markersize=7, label="F1")
    ax.plot(GROUP_A_NLIST, recall_vals,
            color=COLOR_RECALL, marker=MARKER_RECALL,
            linewidth=1.8, markersize=7, label="Recall")

    if flat_f1 is not None:
        add_flat_hline(ax, flat_f1,     COLOR_F1,
                       label=f"Flat F1 ({flat_f1:.3f})")
    if flat_recall is not None:
        add_flat_hline(ax, flat_recall, COLOR_RECALL,
                       label=f"Flat Recall ({flat_recall:.3f})")

    ax.set_xscale("log", base=2)
    ax.set_xticks(GROUP_A_NLIST)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("nlist")
    ax.set_ylabel("Score")
    ax.set_title("Fig 1  Quality vs. nlist  (M=32, nbits=8, LLM=BF16)")
    ax.set_ylim(0, 0.75)
    ax.legend(loc="upper right", framealpha=0.85)

    save_fig(fig, "fig1_quality_vs_nlist", output_dir)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Fig 2 — QPS vs nlist (Group A)
# ══════════════════════════════════════════════════════════════════════════
def make_fig2(data: dict, output_dir: str):
    print("\n[Fig 2] QPS vs nlist")

    flat_qps = data.get(FLAT_TAG, {}).get("qps")
    qps_vals = [data[t]["qps"] for t in GROUP_A_TAGS]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(GROUP_A_NLIST, qps_vals,
            color=COLOR_QPS, marker=MARKER_QPS,
            linewidth=1.8, markersize=7, label="QPS")

    if flat_qps is not None:
        add_flat_hline(ax, flat_qps, COLOR_FLAT,
                       label=f"Flat QPS ({flat_qps:.2f})")

    ax.set_xscale("log", base=2)
    ax.set_xticks(GROUP_A_NLIST)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("nlist")
    ax.set_ylabel("Queries per Second (QPS)")
    ax.set_title("Fig 2  End-to-End QPS vs. nlist  (M=32, nbits=8, LLM=BF16)")
    ax.legend(loc="lower right", framealpha=0.85)

    save_fig(fig, "fig2_qps_vs_nlist", output_dir)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Fig 3 — Quality vs PQ compression (Group B)
# ══════════════════════════════════════════════════════════════════════════
def make_fig3(data: dict, output_dir: str):
    print("\n[Fig 3] Quality vs PQ compression")

    flat  = data.get(FLAT_TAG, {})
    flat_f1     = flat.get("f1")
    flat_recall = flat.get("recall")

    f1_vals     = [data[t]["f1"]     for t in GROUP_B_TAGS]
    recall_vals = [data[t]["recall"] for t in GROUP_B_TAGS]

    x = np.arange(len(GROUP_B_TAGS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4))

    bars_f1 = ax.bar(x - width/2, f1_vals, width,
                     color=COLOR_F1, alpha=0.85, label="F1")
    bars_rc = ax.bar(x + width/2, recall_vals, width,
                     color=COLOR_RECALL, alpha=0.85, label="Recall")

    # 값 레이블
    for bar in bars_f1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=8.5, color=COLOR_F1)
    for bar in bars_rc:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=8.5, color=COLOR_RECALL)

    if flat_f1 is not None:
        add_flat_hline(ax, flat_f1,     COLOR_F1,
                       label=f"Flat F1 ({flat_f1:.3f})")
    if flat_recall is not None:
        add_flat_hline(ax, flat_recall, COLOR_RECALL,
                       label=f"Flat Recall ({flat_recall:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_B_LABELS, fontsize=9)
    ax.set_xlabel("PQ Configuration  (nlist=8192)")
    ax.set_ylabel("Score")
    ax.set_title("Fig 3  Quality vs. PQ Compression  (nlist=8192, LLM=BF16)")
    ax.set_ylim(0, 0.75)
    ax.legend(loc="upper left", framealpha=0.85)

    # 압축 강도 방향 주석
    ax.annotate("", xy=(x[-1] + 0.6, 0.03), xytext=(x[0] - 0.6, 0.03),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.text((x[0] + x[-1]) / 2, 0.015, "higher compression →",
            ha="center", fontsize=8.5, color="gray")

    save_fig(fig, "fig3_quality_vs_pq", output_dir)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Fig 4 — QPS vs PQ compression (Group B)
# ══════════════════════════════════════════════════════════════════════════
def make_fig4(data: dict, output_dir: str):
    print("\n[Fig 4] QPS vs PQ compression")

    flat_qps = data.get(FLAT_TAG, {}).get("qps")
    qps_vals = [data[t]["qps"] for t in GROUP_B_TAGS]

    x = np.arange(len(GROUP_B_TAGS))

    fig, ax = plt.subplots(figsize=(6.5, 4))

    bars = ax.bar(x, qps_vals, color=COLOR_QPS, alpha=0.85, label="QPS")

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=8.5, color=COLOR_QPS)

    if flat_qps is not None:
        add_flat_hline(ax, flat_qps, COLOR_FLAT,
                       label=f"Flat QPS ({flat_qps:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_B_LABELS, fontsize=9)
    ax.set_xlabel("PQ Configuration  (nlist=8192)")
    ax.set_ylabel("Queries per Second (QPS)")
    ax.set_title("Fig 4  End-to-End QPS vs. PQ Compression  (nlist=8192, LLM=BF16)")
    ax.set_ylim(0, max(qps_vals) * 1.25)
    ax.legend(loc="upper left", framealpha=0.85)

    ax.annotate("", xy=(x[-1] + 0.6, 0.3), xytext=(x[0] - 0.6, 0.3),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.text((x[0] + x[-1]) / 2, 0.1, "higher compression →",
            ha="center", fontsize=8.5, color="gray")

    save_fig(fig, "fig4_qps_vs_pq", output_dir)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading summaries from: {args.input_dir}")
    data = load_summaries(args.input_dir, args.llm_quant)

    print(f"\nLoaded {len(data)} index configs:")
    for tag, m in data.items():
        print(f"  {tag:30s}  F1={m['f1']:.3f}  Recall={m['recall']:.3f}  QPS={m['qps']:.2f}")

    # 필요한 태그 존재 여부 체크
    missing = []
    for t in GROUP_A_TAGS + GROUP_B_TAGS + [FLAT_TAG]:
        if t not in data:
            missing.append(t)
    if missing:
        print(f"\n[WARNING] 다음 index_tag가 없음 — 해당 figure가 불완전할 수 있음:")
        for m in missing:
            print(f"  {m}")

    make_fig1(data, args.output_dir)
    make_fig2(data, args.output_dir)
    make_fig3(data, args.output_dir)
    make_fig4(data, args.output_dir)

    print(f"\n완료. figures → {args.output_dir}/")




