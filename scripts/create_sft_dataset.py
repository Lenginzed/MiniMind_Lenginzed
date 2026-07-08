# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.sft_data import write_jsonl


def make_examples(total: int, seed: int):
    rng = random.Random(seed)
    examples = []
    concepts = [
        ("tokenizer", "A tokenizer converts text into token ids and can decode ids back into text."),
        ("pretraining", "Pretraining optimizes next-token prediction on broad text before task-specific tuning."),
        ("SFT", "SFT trains on instruction and response pairs so the model follows a desired response format."),
        ("LoRA", "LoRA freezes the base model and trains small low-rank adapter matrices."),
        ("gradient checkpointing", "Gradient checkpointing saves memory by recomputing activations during backward."),
        ("scheduler", "A scheduler changes the learning rate during training, often with warmup and decay."),
        ("checkpoint", "A checkpoint stores model weights, optimizer state, step, and configuration."),
    ]
    translations = [
        ("因果语言模型", "causal language model"),
        ("奖励函数", "reward function"),
        ("检查点", "checkpoint"),
        ("分词器", "tokenizer"),
        ("梯度裁剪", "gradient clipping"),
    ]
    categories = ["concept", "math", "translation", "flight_rl", "format", "code"]
    for idx in range(total):
        cat = categories[idx % len(categories)]
        if cat == "concept":
            name, answer = rng.choice(concepts)
            inst = "Explain %s in one or two sentences." % name
            out = answer
        elif cat == "math":
            a = rng.randint(1, 50)
            b = rng.randint(1, 50)
            inst = "What is %d + %d? Show the result briefly." % (a, b)
            out = "%d + %d = %d." % (a, b, a + b)
        elif cat == "translation":
            zh, en = rng.choice(translations)
            if rng.random() < 0.5:
                inst = "Translate this Chinese technical term into English: %s" % zh
                out = "%s means %s." % (zh, en)
            else:
                inst = "解释术语：%s" % zh
                out = "%s 通常可以理解为 %s，是机器学习或控制任务中的常见概念。" % (zh, en)
        elif cat == "flight_rl":
            inst = rng.choice(
                [
                    "Why does an air-combat agent need a reward function?",
                    "What should a flight controller monitor during a maneuver?",
                    "Why is policy stability important in reinforcement learning?",
                ]
            )
            out = rng.choice(
                [
                    "A reward function turns task goals into feedback, such as safety, target progress, and energy management.",
                    "The controller should monitor altitude, speed, heading, energy, and safety limits.",
                    "Stable policies reduce erratic actions and make training easier to evaluate.",
                ]
            )
        elif cat == "format":
            inst = rng.choice(
                [
                    "Use three bullet points to explain the difference between pretraining and SFT.",
                    "用三点说明 LoRA 的优点。",
                    "List three things to log during training.",
                ]
            )
            if "LoRA" in inst:
                out = "1. 冻结 base model。\n2. 只训练低秩 adapter。\n3. 显著减少可训练参数。"
            elif "log" in inst:
                out = "1. Train loss and eval loss.\n2. Learning rate and gradient norm.\n3. Checkpoint path and tokens seen."
            else:
                out = "1. Pretraining learns next-token prediction.\n2. SFT learns instruction-response behavior.\n3. SFT usually uses curated examples."
        else:
            inst = rng.choice(
                [
                    "What does AdamW do?",
                    "What should a training checkpoint include?",
                    "Why clip gradients?",
                ]
            )
            if "AdamW" in inst:
                out = "AdamW is an optimizer that combines Adam-style adaptive updates with decoupled weight decay."
            elif "checkpoint" in inst:
                out = "It should include model weights, optimizer state, scheduler state, step, and config."
            else:
                out = "Gradient clipping limits very large updates and can improve training stability."
        examples.append({"instruction": inst, "input": "", "output": out, "category": cat})
    rng.shuffle(examples)
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local synthetic SFT dataset.")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    rows = make_examples(args.train_size + args.val_size, args.seed)
    train = rows[: args.train_size]
    val = rows[args.train_size :]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, str(out_dir / "sft_train.jsonl"))
    write_jsonl(val, str(out_dir / "sft_val.jsonl"))
    meta = {
        "train_size": len(train),
        "val_size": len(val),
        "seed": args.seed,
        "train_categories": dict(Counter(row["category"] for row in train)),
        "val_categories": dict(Counter(row["category"] for row in val)),
        "note": "Synthetic local SFT data for pipeline validation only; not real instruction-tuning data quality.",
    }
    (out_dir / "sft_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
