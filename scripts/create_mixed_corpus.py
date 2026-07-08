# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import random
from pathlib import Path


ENGLISH_PARAGRAPHS = [
    "A small language model is useful when the goal is to inspect every stage of the training pipeline.",
    "The engineer records inputs, outputs, metrics, checkpoints, and random seeds before changing the next variable.",
    "A decoder-only model reads tokens from left to right and learns to predict the next token under a causal mask.",
    "The aircraft changes heading slowly while the controller checks altitude, speed, and available energy.",
    "A stable experiment is easier to debug than a large experiment that fails silently after several hours.",
    "The validation split is not a leaderboard; it is a warning light for pipeline mistakes and overfitting.",
]

CHINESE_PARAGRAPHS = [
    "小规模预训练实验的目标不是获得强模型，而是验证数据、分词、模型、优化器和日志是否连贯。",
    "随机种子、配置文件、检查点和指标记录可以让实验更容易复现，也更容易被审计。",
    "因果语言模型在当前位置只能看到过去 token，不能读取未来 token。",
    "飞行器控制任务通常需要同时考虑姿态、速度、高度、能量和安全边界。",
    "强化学习实验需要关注奖励设计、策略更新、采样效率和训练稳定性。",
    "这个本地生成语料只用于 Stage 2.1 管线压测，不代表真实预训练数据质量。",
]

CODE_SNIPPETS = [
    "def clip_gradients(parameters, max_norm): return torch.nn.utils.clip_grad_norm_(parameters, max_norm)",
    "for step, batch in enumerate(loader): loss = model(batch['input_ids'], labels=batch['labels'])['loss']",
    "if scheduler_name == 'cosine': lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))",
    "tokens_seen += batch_size * context_length",
    "assert logits.shape == (batch, seq_len, vocab_size)",
]

MATH_LINES = [
    "If the cross entropy is 2.0, the perplexity is approximately exp(2.0).",
    "A warmup schedule increases the learning rate for the first few steps before decay begins.",
    "For block size 128 and batch size 16, one optimizer step observes 2048 token positions.",
    "Gradient accumulation simulates a larger batch by delaying the optimizer step.",
    "The cosine scheduler maps progress from zero to one and gradually reduces the learning rate.",
]

RL_LLM_LINES = [
    "Policy optimization compares actions, rewards, and trajectories, but this stage only performs pretraining.",
    "LoRA, DPO, and GRPO are intentionally deferred until the base training loop is reliable.",
    "A tokenizer maps text into integer ids, and the embedding table maps ids into vectors.",
    "RoPE changes query and key vectors using position-dependent rotations.",
    "Grouped query attention reduces key and value heads while keeping more query heads.",
]


def make_line(rng: random.Random, idx: int) -> str:
    choice = rng.randrange(8)
    if choice == 0:
        return rng.choice(ENGLISH_PARAGRAPHS)
    if choice == 1:
        return rng.choice(CHINESE_PARAGRAPHS)
    if choice == 2:
        return rng.choice(MATH_LINES)
    if choice == 3:
        return "Code note %05d: `%s`." % (idx, rng.choice(CODE_SNIPPETS))
    if choice == 4:
        return rng.choice(RL_LLM_LINES)
    if choice == 5:
        speed = rng.randint(180, 920)
        altitude = rng.randint(1000, 12000)
        return "Flight log %05d: speed=%d knots, altitude=%d meters, decision=hold course." % (
            idx,
            speed,
            altitude,
        )
    if choice == 6:
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        return "Math drill %05d: %d + %d = %d, and %d * %d = %d." % (
            idx,
            a,
            b,
            a + b,
            a,
            b,
            a * b,
        )
    terms = rng.sample(
        [
            "checkpoint",
            "optimizer",
            "scheduler",
            "tensorboard",
            "tokenizer",
            "causal mask",
            "validation loss",
            "gradient norm",
            "bf16",
            "random block split",
        ],
        3,
    )
    return "Experiment note %05d: inspect %s, %s, and %s before scaling." % (
        idx,
        terms[0],
        terms[1],
        terms[2],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local mixed corpus for Stage 2.1.")
    parser.add_argument("--output", default="data/raw/mixed_corpus.txt")
    parser.add_argument("--target-mb", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260708)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    target_bytes = int(args.target_mb * 1024 * 1024)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    total_bytes = 0
    idx = 0
    while total_bytes < target_bytes:
        line = make_line(rng, idx)
        lines.append(line)
        total_bytes += len((line + "\n").encode("utf-8"))
        idx += 1

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote:", out_path)
    print("lines:", len(lines))
    print("bytes:", out_path.stat().st_size)
    print("note: this mixed corpus is locally generated for pipeline hardening only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
