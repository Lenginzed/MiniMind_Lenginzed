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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.grpo_data import write_jsonl
from minillm.utils import ensure_dir, save_json


def math_add(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(0, 49)
    b = rng.randint(0, 49)
    answer = str(a + b)
    return {
        "prompt": f"User: Compute {a} + {b}. Answer with the final integer only.\nAssistant: ",
        "answer": answer,
        "category": "math_add",
        "reward_type": "exact_integer",
        "keyword": "",
    }


def math_sub(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(10, 80)
    b = rng.randint(0, a)
    answer = str(a - b)
    return {
        "prompt": f"User: Compute {a} - {b}. Answer with the final integer only.\nAssistant: ",
        "answer": answer,
        "category": "math_sub",
        "reward_type": "exact_integer",
        "keyword": "",
    }


def math_mul_small(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(0, 12)
    b = rng.randint(0, 12)
    answer = str(a * b)
    return {
        "prompt": f"User: Compute {a} * {b}. Answer with the final integer only.\nAssistant: ",
        "answer": answer,
        "category": "math_mul_small",
        "reward_type": "exact_integer",
        "keyword": "",
    }


def format_echo(rng: random.Random) -> Dict[str, str]:
    phrase = rng.choice(["READY", "OK", "DONE", "SAFE", "PASS"])
    return {
        "prompt": f"User: Output exactly the word {phrase} and nothing else.\nAssistant: ",
        "answer": phrase,
        "category": "format_echo",
        "reward_type": "exact_text",
        "keyword": phrase,
    }


def concept_keyword(rng: random.Random) -> Dict[str, str]:
    keyword = rng.choice(["LoRA", "tokenizer", "SFT", "DPO", "reward"])
    prompts = {
        "LoRA": "User: In one short phrase, mention the adapter method LoRA.\nAssistant: ",
        "tokenizer": "User: In one short phrase, mention the component tokenizer.\nAssistant: ",
        "SFT": "User: In one short phrase, mention supervised fine-tuning as SFT.\nAssistant: ",
        "DPO": "User: In one short phrase, mention preference optimization as DPO.\nAssistant: ",
        "reward": "User: In one short phrase, mention the reward signal.\nAssistant: ",
    }
    return {
        "prompt": prompts[keyword],
        "answer": keyword,
        "category": "concept_keyword",
        "reward_type": "keyword",
        "keyword": keyword,
    }


BUILDERS = [math_add, math_sub, math_mul_small, format_echo, concept_keyword]


def build_rows(count: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(count):
        row = BUILDERS[idx % len(BUILDERS)](rng)
        row["id"] = f"grpo_{seed}_{idx:06d}"
        rows.append(row)
    rng.shuffle(rows)
    return rows


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    answers = Counter(row["answer"] for row in rows)
    return {
        "count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "reward_type_counts": dict(Counter(row["reward_type"] for row in rows)),
        "answer_top10": dict(answers.most_common(10)),
        "answer_unique": len(answers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create local synthetic GRPO/RLVR prompts.")
    parser.add_argument("--out-dir", default="data/grpo")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    train_rows = build_rows(args.train_size, args.seed)
    val_rows = build_rows(args.val_size, args.seed + 1)
    train_path = str(Path(args.out_dir) / "grpo_train.jsonl")
    val_path = str(Path(args.out_dir) / "grpo_val.jsonl")
    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)
    metadata = {
        "description": "Synthetic local reward data only for GRPO pipeline validation; it is not real RLHF/RLVR data.",
        "seed": args.seed,
        "train_path": train_path,
        "val_path": val_path,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "format": {
            "fields": ["prompt", "answer", "category", "reward_type", "keyword"],
            "prompt_template": "User: ...\\nAssistant: ",
        },
    }
    save_json(metadata, str(Path(args.out_dir) / "grpo_metadata.json"))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
