# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.utils import ensure_dir, iter_jsonl, load_json, load_yaml


RUNS = {
    "pretrain": "outputs/stage7/pretrain_long",
    "sft_full": "outputs/stage7/sft_full_long",
    "sft_lora": "outputs/stage7/sft_lora_long",
    "dpo_full": "outputs/stage7/dpo_full_long",
    "dpo_lora": "outputs/stage7/dpo_lora_long",
    "grpo_full": "outputs/stage7/grpo_full_long",
    "grpo_lora": "outputs/stage7/grpo_lora_long",
}


def load_metrics(path: str) -> List[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return []
    return [row for row in iter_jsonl(str(p)) if row.get("step") is not None and row.get("event") != "resume"]


def first_last(rows: List[Dict[str, object]], key: str):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None, None
    return values[0], values[-1]


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        if not math.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def plot_loss_ppl(pretrain_rows: List[Dict[str, object]], out: Path) -> None:
    if not pretrain_rows:
        return
    steps = [row["step"] for row in pretrain_rows]
    train_loss = [row.get("train_loss") for row in pretrain_rows]
    eval_steps = [row["step"] for row in pretrain_rows if row.get("eval_loss") is not None]
    eval_loss = [row.get("eval_loss") for row in pretrain_rows if row.get("eval_loss") is not None]
    train_ppl = [min(float(row.get("train_ppl") or 0.0), 500.0) for row in pretrain_rows]
    eval_ppl = [min(float(row.get("eval_ppl") or 0.0), 500.0) for row in pretrain_rows if row.get("eval_ppl") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, train_loss, label="train")
    if eval_steps:
        axes[0].plot(eval_steps, eval_loss, marker="o", label="eval")
    axes[0].set_title("Pretrain loss")
    axes[0].set_xlabel("step")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps, train_ppl, label="train ppl")
    if eval_steps:
        axes[1].plot(eval_steps, eval_ppl, marker="o", label="eval ppl")
    axes[1].set_title("Pretrain perplexity (clipped)")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_two_loss(rows_a, rows_b, label_a, label_b, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for rows, label in [(rows_a, label_a), (rows_b, label_b)]:
        if rows:
            ax.plot([r["step"] for r in rows], [r.get("train_loss") for r in rows], label=f"{label} train")
            ev = [r for r in rows if r.get("eval_loss") is not None]
            if ev:
                ax.plot([r["step"] for r in ev], [r.get("eval_loss") for r in ev], marker="o", label=f"{label} eval")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend()
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dpo(rows_a, rows_b, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for rows, label in [(rows_a, "Full DPO"), (rows_b, "DPO-LoRA")]:
        if rows:
            axes[0].plot([r["step"] for r in rows], [r.get("reward_margin") for r in rows], label=label)
            axes[1].plot([r["step"] for r in rows], [r.get("preference_accuracy") for r in rows], label=label)
    axes[0].set_title("Reward margin")
    axes[1].set_title("Preference accuracy")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_grpo(rows_a, rows_b, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    keys = [("reward_mean", "Reward mean"), ("reward_std", "Reward std"), ("frac_reward_zero_std", "Zero-std fraction"), ("exact_accuracy_mean", "Exact accuracy")]
    for ax, (key, title) in zip(axes.reshape(-1), keys):
        for rows, label in [(rows_a, "Full GRPO"), (rows_b, "GRPO-LoRA")]:
            if rows:
                ax.plot([r["step"] for r in rows], [r.get(key) for r in rows], label=label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_trainable_params(summaries: Dict[str, Dict[str, object]], out: Path) -> None:
    labels, values = [], []
    for key in ["pretrain", "sft_full", "sft_lora", "dpo_full", "dpo_lora", "grpo_full", "grpo_lora"]:
        summary = summaries.get(key) or {}
        val = summary.get("trainable_params") or summary.get("parameter_count")
        if val:
            labels.append(key)
            values.append(int(val))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values)
    ax.set_ylabel("trainable params")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def load_summary(run_dir: str) -> Dict[str, object]:
    for name in ["train_summary.json", "sft_summary.json", "dpo_summary.json", "grpo_summary.json"]:
        path = Path(run_dir) / name
        if path.exists():
            return load_json(str(path))
    return {}


def summarize_run(name: str, rows: List[Dict[str, object]], key: str = "train_loss") -> str:
    first, last = first_last(rows, key)
    steps = rows[-1]["step"] if rows else "N/A"
    return f"- {name}: steps `{steps}`, {key} `{fmt(first)}` -> `{fmt(last)}`"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 7 plots and interview-ready reports.")
    parser.add_argument("--out-dir", default="audit_stage7")
    args = parser.parse_args()
    audit = Path(args.out_dir)
    plots = Path("outputs/stage7/plots")
    ensure_dir(str(audit))
    ensure_dir(str(plots))

    metrics = {name: load_metrics(str(Path(path) / "metrics.jsonl")) for name, path in RUNS.items()}
    summaries = {name: load_summary(path) for name, path in RUNS.items()}

    plot_loss_ppl(metrics["pretrain"], plots / "pretrain_loss_ppl.png")
    plot_two_loss(metrics["sft_full"], metrics["sft_lora"], "Full SFT", "LoRA-SFT", plots / "sft_full_vs_lora_loss.png")
    plot_dpo(metrics["dpo_full"], metrics["dpo_lora"], plots / "dpo_full_vs_lora_margin_acc.png")
    plot_grpo(metrics["grpo_full"], metrics["grpo_lora"], plots / "grpo_reward_diagnostics.png")
    plot_trainable_params(summaries, plots / "trainable_params_compare.png")

    lines = [
        "# Stage 7 Final Report",
        "",
        "Stage 7 runs are local long-run / resume-ready demonstrations for README and interview discussion. They do not claim real LLM capability.",
        "",
        "## Run Summary",
        "",
        summarize_run("Pretrain", metrics["pretrain"], "train_loss"),
        summarize_run("Full SFT", metrics["sft_full"], "train_loss"),
        summarize_run("LoRA-SFT", metrics["sft_lora"], "train_loss"),
        summarize_run("Full DPO", metrics["dpo_full"], "train_loss"),
        summarize_run("DPO-LoRA", metrics["dpo_lora"], "train_loss"),
        summarize_run("Full GRPO", metrics["grpo_full"], "reward_mean"),
        summarize_run("GRPO-LoRA", metrics["grpo_lora"], "reward_mean"),
        "",
        "## Plots",
        "",
        "- `outputs/stage7/plots/pretrain_loss_ppl.png`",
        "- `outputs/stage7/plots/sft_full_vs_lora_loss.png`",
        "- `outputs/stage7/plots/dpo_full_vs_lora_margin_acc.png`",
        "- `outputs/stage7/plots/grpo_reward_diagnostics.png`",
        "- `outputs/stage7/plots/trainable_params_compare.png`",
        "",
    ]
    (audit / "stage7_final_report.md").write_text("\n".join(lines), encoding="utf-8")

    interview = [
        "# Stage 7 Interview Summary",
        "",
        "## 1. One-Sentence Project Intro",
        "",
        "I implemented a from-scratch mini-LLM training and post-training stack covering tokenizer/data pipeline, decoder-only Causal LM, pretraining, SFT, LoRA, DPO, GRPO, and educational quantization, with reproducible configs, metrics, checkpoints, plots, and audit reports.",
        "",
        "## 2. Technical Pipeline",
        "",
        "```mermaid",
        "flowchart LR",
        "  Data --> Tokenizer --> Pretrain --> SFT",
        "  SFT --> LoRA_SFT[LoRA-SFT]",
        "  SFT --> DPO",
        "  DPO --> DPO_LoRA[DPO-LoRA]",
        "  SFT --> GRPO",
        "  GRPO --> GRPO_LoRA[GRPO-LoRA]",
        "  SFT --> Quant[Educational Quantization]",
        "```",
        "",
        "## 3. What Each Stage Did",
        "",
        "- Stage 0.5: audited the fixed local Conda environment Python/CUDA environment.",
        "- Stage 1: built the decoder-only Causal LM with RMSNorm, RoPE, GQA, SwiGLU, and generation.",
        "- Stage 2: built tokenizer, block data pipeline, pretrain loop, checkpoints, metrics, and plots.",
        "- Stage 3: built assistant-only SFT and self-implemented LoRA.",
        "- Stage 4: built DPO and DPO-LoRA from scratch with frozen reference model.",
        "- Stage 5: built GRPO-style online sampling, rule rewards, group-relative advantage, and clipped policy loss.",
        "- Stage 6: built educational weight-only INT8/INT4, GPTQ-style, and SmoothQuant-style fake quantization.",
        "- Stage 7: ran longer resume-ready experiments and collected README/interview plots.",
        "",
        "## 4. Key Curves",
        "",
        "- Pretrain loss/PPL: `outputs/stage7/plots/pretrain_loss_ppl.png`",
        "- SFT full vs LoRA: `outputs/stage7/plots/sft_full_vs_lora_loss.png`",
        "- DPO margin/accuracy: `outputs/stage7/plots/dpo_full_vs_lora_margin_acc.png`",
        "- GRPO reward diagnostics: `outputs/stage7/plots/grpo_reward_diagnostics.png`",
        "",
        "## 5. Full vs LoRA Comparison",
        "",
        "Full fine-tuning updates all parameters and is a useful upper-bound engineering baseline. LoRA updates a small adapter subset, making training cheaper while preserving the same training loop surface and checkpoint/adapter workflow.",
        "",
        "## 6. DPO Metrics Explained",
        "",
        "DPO loss optimizes the relative log probability of chosen over rejected responses against a frozen reference. Reward margin tracks how much more the policy favors chosen outputs, and preference accuracy measures how often chosen reward exceeds rejected reward.",
        "",
        "## 7. GRPO Diagnostics Explained",
        "",
        "GRPO samples multiple completions per prompt. Reward std matters because group-relative advantage disappears when all completions get the same reward. A high zero-std fraction is a warning that the reward or sampling setup is not giving useful learning signal.",
        "",
        "## 8. Quantization Results Explained",
        "",
        "INT8 fake quant preserved loss almost exactly, INT4 degraded loss more, GPTQ-style recorded activation-weighted errors without full compensation, and SmoothQuant-style used a runtime scaling wrapper. These are educational fake-quant experiments, not production integer-kernel speedups.",
        "",
        "## 9. Limitations",
        "",
        "- Local synthetic data is not comparable to real pretraining or RLHF/RLVR corpora.",
        "- The model is small and should not be presented as generally capable.",
        "- GRPO reward functions are rule-based and sparse/dense toy signals.",
        "- Quantization lacks bit packing and real integer kernels.",
        "",
        "## 10. Interview Version",
        "",
        "I can walk through the whole LLM lifecycle end to end: how text becomes tokens, how a decoder-only transformer is trained, how SFT masks assistant tokens, how LoRA reduces trainable parameters, how DPO uses a frozen reference model, why GRPO needs reward variance, and why fake quantization can reduce estimated size without guaranteeing latency improvements.",
        "",
    ]
    (audit / "stage7_interview_summary.md").write_text("\n".join(interview), encoding="utf-8")

    readme = [
        "## Mini-LLM Long-Run Results",
        "",
        "This project implements a from-scratch mini-LLM stack for learning and portfolio demonstration. Stage 7 runs longer, resume-ready local experiments over pretraining, SFT, LoRA-SFT, DPO, DPO-LoRA, GRPO, GRPO-LoRA, and educational quantization.",
        "",
        "Key plots:",
        "",
        "- `outputs/stage7/plots/pretrain_loss_ppl.png`",
        "- `outputs/stage7/plots/sft_full_vs_lora_loss.png`",
        "- `outputs/stage7/plots/dpo_full_vs_lora_margin_acc.png`",
        "- `outputs/stage7/plots/grpo_reward_diagnostics.png`",
        "- `outputs/stage7/plots/trainable_params_compare.png`",
        "",
        "The results show engineering correctness and training dynamics, not real large-model capability.",
        "",
    ]
    (audit / "stage7_readme_section.md").write_text("\n".join(readme), encoding="utf-8")
    print("wrote Stage 7 reports and plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
