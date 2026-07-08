# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.utils import ensure_dir, iter_jsonl, load_json


PUBLIC_RUNS = {
    "pretrain": "outputs/stage8_public/pretrain_public_long",
    "sft_full": "outputs/stage8_public/sft_public_full",
    "sft_lora": "outputs/stage8_public/sft_public_lora",
    "dpo_full": "outputs/stage8_public/dpo_public_full",
    "dpo_lora": "outputs/stage8_public/dpo_public_lora",
    "grpo_full": "outputs/stage8_public/grpo_public_full",
    "grpo_lora": "outputs/stage8_public/grpo_public_lora",
}

STAGE7_RUNS = {
    "pretrain": "outputs/stage7/pretrain_long",
    "sft_full": "outputs/stage7/sft_full_long",
    "sft_lora": "outputs/stage7/sft_lora_long",
    "dpo_full": "outputs/stage7/dpo_full_long",
    "dpo_lora": "outputs/stage7/dpo_lora_long",
    "grpo_full": "outputs/stage7/grpo_full_long",
    "grpo_lora": "outputs/stage7/grpo_lora_long",
}


def load_metrics(run_dir: str) -> List[Dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    return [row for row in iter_jsonl(str(path)) if row.get("step") is not None and row.get("event") != "resume"]


def load_summary(run_dir: str) -> Dict[str, Any]:
    for name in ["train_summary.json", "sft_summary.json", "dpo_summary.json", "grpo_summary.json"]:
        path = Path(run_dir) / name
        if path.exists():
            return load_json(str(path))
    return {}


def values(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for row in rows:
        value = row.get(key)
        try:
            if value is not None and math.isfinite(float(value)):
                out.append(float(value))
        except Exception:
            pass
    return out


def first_last(rows: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float]]:
    vals = values(rows, key)
    if not vals:
        return None, None
    return vals[0], vals[-1]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}"
    except Exception:
        return str(value)


def plot_pretrain(public_rows, stage7_rows, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for rows, label in [(public_rows, "public"), (stage7_rows, "stage7 synthetic")]:
        if rows:
            axes[0].plot([r["step"] for r in rows], values(rows, "train_loss"), label=f"{label} train")
            eval_rows = [r for r in rows if r.get("eval_loss") is not None]
            if eval_rows:
                axes[0].plot([r["step"] for r in eval_rows], values(eval_rows, "eval_loss"), marker="o", label=f"{label} eval")
            ppl_vals = [min(v, 500.0) for v in values(rows, "train_ppl")]
            axes[1].plot([r["step"] for r in rows[: len(ppl_vals)]], ppl_vals, label=label)
    axes[0].set_title("Pretrain loss")
    axes[1].set_title("Train perplexity clipped at 500")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_public_pretrain_loss_ppl(public_rows, out: Path) -> None:
    if not public_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    steps = [r["step"] for r in public_rows]
    axes[0].plot(steps, values(public_rows, "train_loss"), label="train loss")
    eval_rows = [r for r in public_rows if r.get("eval_loss") is not None]
    if eval_rows:
        axes[0].plot([r["step"] for r in eval_rows], values(eval_rows, "eval_loss"), marker="o", label="eval loss")
    train_ppl = [min(v, 500.0) for v in values(public_rows, "train_ppl")]
    axes[1].plot(steps[: len(train_ppl)], train_ppl, label="train ppl")
    if eval_rows:
        eval_ppl = [min(v, 500.0) for v in values(eval_rows, "eval_ppl")]
        axes[1].plot([r["step"] for r in eval_rows[: len(eval_ppl)]], eval_ppl, marker="o", label="eval ppl")
    axes[0].set_title("Public pretrain loss")
    axes[1].set_title("Public pretrain perplexity (clipped)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_loss_pair(rows_a, rows_b, label_a, label_b, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for rows, label in [(rows_a, label_a), (rows_b, label_b)]:
        if rows:
            ax.plot([r["step"] for r in rows], values(rows, "train_loss"), label=f"{label} train")
            eval_rows = [r for r in rows if r.get("eval_loss") is not None]
            if eval_rows:
                ax.plot([r["step"] for r in eval_rows], values(eval_rows, "eval_loss"), marker="o", label=f"{label} eval")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend()
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dpo(rows_full, rows_lora, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for rows, label in [(rows_full, "Full"), (rows_lora, "LoRA")]:
        if rows:
            axes[0].plot([r["step"] for r in rows], values(rows, "reward_margin"), label=label)
            axes[1].plot([r["step"] for r in rows], values(rows, "preference_accuracy"), label=label)
    axes[0].set_title("DPO reward margin")
    axes[1].set_title("DPO preference accuracy")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_grpo(rows_full, rows_lora, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    keys = [
        ("reward_mean", "reward mean"),
        ("reward_std", "reward std"),
        ("frac_reward_zero_std", "zero-std fraction"),
        ("exact_accuracy_mean", "exact accuracy"),
    ]
    for ax, (key, title) in zip(axes.reshape(-1), keys):
        for rows, label in [(rows_full, "Full"), (rows_lora, "LoRA")]:
            if rows:
                ax.plot([r["step"] for r in rows], values(rows, key), label=label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    ensure_dir(str(out.parent))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summarize(name: str, rows: List[Dict[str, Any]], key: str) -> str:
    first, last = first_last(rows, key)
    steps = rows[-1].get("step") if rows else "N/A"
    return f"- {name}: steps `{steps}`, {key} `{fmt(first)}` -> `{fmt(last)}`"


def sample_head(path: str, max_chars: int = 700) -> str:
    p = Path(path)
    if not p.exists():
        return "N/A"
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    return text[:max_chars].replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 8 public-data comparison reports.")
    parser.add_argument("--audit-dir", default="audit_stage8")
    args = parser.parse_args()

    audit = Path(args.audit_dir)
    plots = Path("outputs/stage8_public/plots")
    ensure_dir(str(audit))
    ensure_dir(str(plots))

    public = {name: load_metrics(path) for name, path in PUBLIC_RUNS.items()}
    stage7 = {name: load_metrics(path) for name, path in STAGE7_RUNS.items()}
    summaries = {name: load_summary(path) for name, path in PUBLIC_RUNS.items()}
    dataset_meta = load_json("data/stage8_public/dataset_metadata.json") if Path("data/stage8_public/dataset_metadata.json").exists() else {}
    token_meta = load_json("data/stage8_public/processed_pretrain/metadata.json") if Path("data/stage8_public/processed_pretrain/metadata.json").exists() else {}

    plot_public_pretrain_loss_ppl(public["pretrain"], plots / "pretrain_public_loss_ppl.png")
    plot_pretrain(public["pretrain"], stage7["pretrain"], plots / "public_vs_synthetic_compare.png")
    plot_pretrain(public["pretrain"], stage7["pretrain"], plots / "pretrain_public_vs_synthetic.png")
    plot_loss_pair(public["sft_full"], public["sft_lora"], "Public Full SFT", "Public LoRA-SFT", plots / "sft_public_full_vs_lora.png")
    plot_dpo(public["dpo_full"], public["dpo_lora"], plots / "dpo_public_margin_acc.png")
    plot_grpo(public["grpo_full"], public["grpo_lora"], plots / "grpo_public_diagnostics.png")

    final_lines = [
        "# Stage 8 Public Dataset Final Report",
        "",
        "Stage 8 is a public-data subset migration/control experiment. It complements Stage 7 synthetic long-run results and does not claim real large-model capability.",
        "",
        "## Data",
        "",
        f"- Pretrain source: `{dataset_meta.get('pretrain', {}).get('source_dataset') or dataset_meta.get('pretrain', {}).get('source')}`",
        f"- Pretrain fallback: `{dataset_meta.get('pretrain', {}).get('fallback')}`",
        f"- SFT source: `{dataset_meta.get('sft', {}).get('source_dataset') or dataset_meta.get('sft', {}).get('source')}`, fallback: `{dataset_meta.get('sft', {}).get('fallback')}`",
        f"- DPO source: `{dataset_meta.get('dpo', {}).get('source_dataset') or dataset_meta.get('dpo', {}).get('source')}`, fallback: `{dataset_meta.get('dpo', {}).get('fallback')}`",
        f"- GRPO source: `{dataset_meta.get('grpo', {}).get('source_dataset') or dataset_meta.get('grpo', {}).get('source')}`",
        f"- Tokenized pretrain metadata: `{token_meta}`",
        "",
        "## Training",
        "",
        summarize("Public pretrain", public["pretrain"], "train_loss"),
        summarize("Public Full SFT", public["sft_full"], "train_loss"),
        summarize("Public LoRA-SFT", public["sft_lora"], "train_loss"),
        summarize("Public Full DPO", public["dpo_full"], "train_loss"),
        summarize("Public DPO-LoRA", public["dpo_lora"], "train_loss"),
        summarize("Public-policy Full GRPO", public["grpo_full"], "reward_mean"),
        summarize("Public-policy GRPO-LoRA", public["grpo_lora"], "reward_mean"),
        "",
        "## Plots",
        "",
        "- `outputs/stage8_public/plots/pretrain_public_loss_ppl.png`",
        "- `outputs/stage8_public/plots/public_vs_synthetic_compare.png`",
        "- `outputs/stage8_public/plots/pretrain_public_vs_synthetic.png`",
        "- `outputs/stage8_public/plots/sft_public_full_vs_lora.png`",
        "- `outputs/stage8_public/plots/dpo_public_margin_acc.png`",
        "- `outputs/stage8_public/plots/grpo_public_diagnostics.png`",
        "",
    ]
    (audit / "stage8_public_final_report.md").write_text("\n".join(final_lines), encoding="utf-8")

    training_lines = [
        "# Stage 8 Public Training Report",
        "",
        summarize("Pretrain", public["pretrain"], "train_loss"),
        summarize("SFT full", public["sft_full"], "train_loss"),
        summarize("SFT LoRA", public["sft_lora"], "train_loss"),
        summarize("DPO full", public["dpo_full"], "train_loss"),
        summarize("DPO LoRA", public["dpo_lora"], "train_loss"),
        summarize("GRPO full", public["grpo_full"], "reward_mean"),
        summarize("GRPO LoRA", public["grpo_lora"], "reward_mean"),
        "",
        "## Summaries",
        "",
        "```json",
        json.dumps(summaries, indent=2, ensure_ascii=False)[:6000],
        "```",
    ]
    (audit / "stage8_public_training_report.md").write_text("\n".join(training_lines), encoding="utf-8")

    comparison_lines = [
        "# Stage 8 Public vs Synthetic Report",
        "",
        "## Pretrain",
        "",
        summarize("Public", public["pretrain"], "train_loss"),
        summarize("Stage 7 synthetic", stage7["pretrain"], "train_loss"),
        "",
        "Public data is a more credible source for README claims because it is not generated by the project templates. Synthetic data is still useful for deterministic pipeline debugging.",
        "",
        "## SFT Naturalness",
        "",
        "Public Alpaca-format SFT is expected to be more varied than synthetic Stage 7 prompts when it is available. Sample quality should still be described as smoke/demo output from a small model.",
        "",
        "## DPO Difficulty",
        "",
        "If public preference data falls back to synthetic data, DPO accuracy saturation should not be interpreted as real preference alignment.",
        "",
        "## GRPO",
        "",
        "GRPO remains local verifiable reward data. The useful signal is reward diversity and zero-std diagnostics, not a claim of reasoning ability.",
        "",
    ]
    (audit / "stage8_public_vs_synthetic_report.md").write_text("\n".join(comparison_lines), encoding="utf-8")

    interview_lines = [
        "# Stage 8 Public Dataset Interview Update",
        "",
        "## What Changed After Stage 7",
        "",
        "Stage 8 moved the same self-built training stack onto public dataset subsets when available: TinyStories or WikiText for pretraining, Alpaca for SFT, Alpaca Farm preference data when fields are convertible, and local verifiable GRPO prompts for reward-based RL smoke tests.",
        "",
        "## What Is More Credible",
        "",
        "- Public pretraining text is less template-biased than the Stage 7 local synthetic corpus.",
        "- Public SFT data tests whether the assistant-only masking and training loop handle real instruction distributions.",
        "- Public/fallback metadata makes it clear which results depend on real datasets and which are still synthetic controls.",
        "",
        "## What Is Still Limited",
        "",
        "- The model is around the 40M-50M educational scale, not a capable LLM.",
        "- Public subsets are small and training is short compared with real pretraining.",
        "- GRPO still uses local rule rewards; exact accuracy and zero-std diagnostics are more important than sample charm.",
        "- DPO may still saturate if the converted preference pairs are easy or if fallback synthetic data is used.",
        "",
        "## README-Safe Claim",
        "",
        "I validated the same from-scratch mini-LLM pipeline on public dataset subsets with explicit fallback reporting, then compared public-data curves against the Stage 7 synthetic long-run baseline.",
        "",
        "## Sample Heads",
        "",
        f"- SFT full: `{sample_head('outputs/stage8_public/sft_public_full/samples/after.txt')}`",
        f"- DPO full: `{sample_head('outputs/stage8_public/dpo_public_full/samples/after.txt')}`",
        f"- GRPO rollout: `{sample_head('outputs/stage8_public/grpo_public_full/samples/rollout_samples.jsonl')}`",
        "",
    ]
    (audit / "stage8_public_interview_update.md").write_text("\n".join(interview_lines), encoding="utf-8")
    print("wrote Stage 8 public reports and plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
