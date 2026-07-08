# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import random
from pathlib import Path


def build_lines(num_lines: int, seed: int):
    rng = random.Random(seed)
    stories = [
        "A tiny robot reads a map and explains each step clearly.",
        "The pilot keeps altitude, checks fuel, and chooses a safe heading.",
        "A student trains a mini language model to learn the full pipeline.",
        "The dataset is small, but the engineering loop is complete.",
        "Gradient descent updates weights after the loss is computed.",
        "A decoder-only transformer predicts the next token in a sequence.",
        "RoPE rotates query and key vectors according to token position.",
        "Grouped query attention shares key and value heads across queries.",
        "RMSNorm stabilizes hidden states before attention and the MLP.",
        "SwiGLU uses a gate and an up projection before the down projection.",
    ]
    bilingual = [
        "小模型也可以帮助我们理解大模型训练流程。",
        "数据管线、分词器、模型结构和训练循环需要逐步验证。",
        "这个 toy corpus 只用于 smoke test，不代表真实训练数据。",
        "空战仿真中，智能体需要观察、规划、行动和复盘。",
        "强化学习关注奖励、策略、轨迹和稳定优化。",
    ]
    math = [
        "If x plus y equals ten, and x is four, then y is six.",
        "The loss curve should generally move down during a tiny smoke run.",
        "Perplexity is exponential of cross entropy, so large loss can overflow.",
        "A batch contains several blocks, and each block has the same context length.",
    ]
    tech_terms = [
        "tokenizer", "causal mask", "checkpoint", "tensorboard", "bf16",
        "optimizer", "AdamW", "gradient clipping", "validation split", "sampling",
    ]
    lines = []
    for idx in range(num_lines):
        template_type = idx % 6
        if template_type == 0:
            line = rng.choice(stories)
        elif template_type == 1:
            line = rng.choice(bilingual)
        elif template_type == 2:
            line = rng.choice(math)
        elif template_type == 3:
            term_a, term_b = rng.sample(tech_terms, 2)
            line = "In experiment %04d, %s is checked before %s." % (idx, term_a, term_b)
        elif template_type == 4:
            line = "Question: what does the mini model learn? Answer: it learns repeatable pipeline mechanics."
        else:
            line = "训练日志记录 step、train loss、eval loss 和 perplexity，方便后续审计。"
        lines.append(line)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local toy corpus for Stage 2 smoke tests.")
    parser.add_argument("--output", default="data/raw/toy_corpus.txt")
    parser.add_argument("--lines", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    lines = build_lines(args.lines, args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote:", out_path)
    print("lines:", len(lines))
    print("bytes:", out_path.stat().st_size)
    print("note: this toy corpus is only for pipeline validation, not model quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
