# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.dpo_data import write_jsonl
from minillm.utils import ensure_dir, save_json


REJECTED_TYPES = [
    "wrong_answer",
    "bad_format",
    "vague",
    "unsafe_or_unphysical",
    "off_topic",
    "hallucinated_term",
]


def concept_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    topics = [
        ("LoRA", "LoRA freezes the base model and trains small low-rank adapter matrices, so fine-tuning uses fewer trainable parameters."),
        ("DPO", "DPO compares chosen and rejected responses and optimizes the policy to increase the relative log probability of the chosen answer against a frozen reference."),
        ("SFT", "SFT trains on instruction-response pairs with labels usually applied only to the assistant response tokens."),
        ("RoPE", "RoPE rotates query and key vectors by position-dependent phases so attention can use relative position information."),
        ("GQA", "GQA uses more query heads than key-value heads, reducing KV memory while preserving several query groups."),
    ]
    topic, chosen = rng.choice(topics)
    return {
        "instruction": f"Explain {topic} in one concise paragraph.",
        "input": "",
        "chosen": chosen,
        "rejected": rejected_for(rejected_type, "concept", topic),
        "category": "concept",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def math_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    a = rng.randint(2, 49)
    b = rng.randint(2, 49)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    else:
        result = a * b
    return {
        "instruction": f"Compute {a} {op} {b}. Reply with the final integer and one short check.",
        "input": "",
        "chosen": f"{result}. Check: {a} {op} {b} = {result}.",
        "rejected": rejected_for(rejected_type, "math", str(result + rng.choice([-3, -1, 2, 5]))),
        "category": "math",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def translation_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    pairs = [
        ("Translate 'gradient checkpointing' into Chinese.", "梯度检查点"),
        ("Translate 'causal language modeling' into Chinese.", "因果语言建模"),
        ("Translate 'reward function' into Chinese.", "奖励函数"),
        ("Translate 'tokenizer vocabulary' into Chinese.", "分词器词表"),
        ("Translate 'attention head' into Chinese.", "注意力头"),
    ]
    instruction, chosen = rng.choice(pairs)
    return {
        "instruction": instruction,
        "input": "",
        "chosen": chosen,
        "rejected": rejected_for(rejected_type, "translation", chosen),
        "category": "translation",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def flight_rl_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    prompts = [
        (
            "Why does an air-combat RL agent need a reward function?",
            "A reward function turns task goals into learning signals, such as maintaining safety constraints, improving positioning, and completing objectives.",
        ),
        (
            "Explain why flight control policies must respect physical limits.",
            "Aircraft policies must respect speed, acceleration, actuator, and safety limits because actions outside those bounds are unphysical and unsafe.",
        ),
        (
            "What is a simple curriculum for an air-combat toy environment?",
            "Start with stable flight, add navigation, then add target tracking and finally constrained engagement tasks.",
        ),
    ]
    instruction, chosen = rng.choice(prompts)
    return {
        "instruction": instruction,
        "input": "",
        "chosen": chosen,
        "rejected": rejected_for(rejected_type, "flight_rl", chosen),
        "category": "flight_rl",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def format_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    topic = rng.choice(["SFT vs pretraining", "DPO pipeline checks", "tokenizer debugging", "training logs"])
    return {
        "instruction": f"Answer in exactly three bullet points about {topic}.",
        "input": "",
        "chosen": "- State the main objective.\n- Mention the key data or metric.\n- Note one limitation or risk.",
        "rejected": rejected_for(rejected_type, "format", topic),
        "category": "format",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def code_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    topics = [
        ("AdamW", "AdamW decouples weight decay from the adaptive gradient update, which often makes regularization easier to tune."),
        ("gradient clipping", "Gradient clipping limits the norm of gradients before the optimizer step, helping avoid unstable updates."),
        ("cosine scheduler", "A cosine scheduler gradually lowers the learning rate following a cosine curve after warmup."),
        ("checkpoint", "A checkpoint stores model state, optimizer state, step, and configuration so training can be resumed or evaluated."),
    ]
    topic, chosen = rng.choice(topics)
    return {
        "instruction": f"Briefly explain {topic} in a PyTorch training loop.",
        "input": "",
        "chosen": chosen,
        "rejected": rejected_for(rejected_type, "code", topic),
        "category": "code",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def rejected_for(rejected_type: str, category: str, payload: str) -> str:
    if rejected_type == "wrong_answer":
        if category == "math":
            return f"{payload}. Check: this is the exact result."
        if category == "translation":
            return "这个术语应翻译为随机梯度飞行。"
        return "The statement is false because training never uses gradients or data."
    if rejected_type == "bad_format":
        return "Here is a long paragraph without following the requested structure, numbering, or concise format at all."
    if rejected_type == "vague":
        return "It depends on many things, and the details are complicated."
    if rejected_type == "unsafe_or_unphysical":
        return "The best policy ignores constraints, uses unlimited acceleration, and rewards unsafe maneuvers."
    if rejected_type == "off_topic":
        return "My favorite color for a dashboard is blue, and the weather is pleasant today."
    if rejected_type == "hallucinated_term":
        return "Use HyperDPOFlux, AeroTokenizer Prime, and RewardNorm++ to solve it automatically."
    raise ValueError(f"unknown rejected_type: {rejected_type}")


def reason_for(rejected_type: str) -> str:
    reasons = {
        "wrong_answer": "The rejected response contains a clearly incorrect factual or mathematical answer.",
        "bad_format": "The rejected response does not follow the requested format.",
        "vague": "The rejected response is too generic to be useful.",
        "unsafe_or_unphysical": "The rejected response violates safety, physical, or task constraints.",
        "off_topic": "The rejected response does not answer the prompt.",
        "hallucinated_term": "The rejected response invents unsupported terminology.",
    }
    return reasons[rejected_type]


BUILDERS = [concept_example, math_example, translation_example, flight_rl_example, format_example, code_example]


def build_rows(count: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(count):
        builder = BUILDERS[idx % len(BUILDERS)]
        rejected_type = REJECTED_TYPES[(idx // len(BUILDERS)) % len(REJECTED_TYPES)]
        row = builder(rng, rejected_type)
        row["id"] = f"dpo_{seed}_{idx:06d}"
        rows.append(row)
    rng.shuffle(rows)
    return rows


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    return {
        "count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "rejected_type_counts": dict(Counter(row["rejected_type"] for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local synthetic DPO preference dataset.")
    parser.add_argument("--out-dir", default="data/dpo")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    train_rows = build_rows(args.train_size, args.seed)
    val_rows = build_rows(args.val_size, args.seed + 1)
    train_path = str(Path(args.out_dir) / "dpo_train.jsonl")
    val_path = str(Path(args.out_dir) / "dpo_val.jsonl")
    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)
    metadata = {
        "description": "Synthetic local preference data only for DPO pipeline validation; it is not real human preference data.",
        "seed": args.seed,
        "train_path": train_path,
        "val_path": val_path,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "format": {
            "fields": ["instruction", "input", "chosen", "rejected", "category", "rejected_type", "reason"],
            "prompt_template": "User: {instruction}\\n{input_if_any}\\nAssistant:",
        },
    }
    meta_path = str(Path(args.out_dir) / "dpo_metadata.json")
    save_json(metadata, meta_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
