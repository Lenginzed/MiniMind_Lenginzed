from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from .tokenizer import MiniTokenizer


def load_grpo_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                {
                    "prompt": str(item.get("prompt", "")),
                    "answer": str(item.get("answer", "")),
                    "category": str(item.get("category", "unknown")),
                    "reward_type": str(item.get("reward_type", "unknown")),
                    "keyword": str(item.get("keyword", "")),
                }
            )
    if not rows:
        raise ValueError("GRPO jsonl is empty: %s" % path)
    return rows


class GRPODataset(Dataset):
    def __init__(self, path: str, tokenizer: MiniTokenizer, max_prompt_length: int) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        if self.max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive")
        self.raw_examples = load_grpo_jsonl(path)
        self.examples: List[Dict[str, object]] = []
        self.truncated_prompts = 0
        self.category_counts: Dict[str, int] = {}
        self.reward_type_counts: Dict[str, int] = {}
        bos_id = tokenizer.special_token_ids["bos_token_id"]
        if bos_id is None:
            raise ValueError("tokenizer must define bos_token_id")
        for raw in self.raw_examples:
            prompt = raw["prompt"]
            ids = [int(bos_id)] + tokenizer.encode(prompt, add_special_tokens=False)
            truncated = len(ids) > self.max_prompt_length
            if truncated:
                ids = ids[: self.max_prompt_length]
                self.truncated_prompts += 1
            category = raw["category"]
            reward_type = raw["reward_type"]
            self.category_counts[category] = self.category_counts.get(category, 0) + 1
            self.reward_type_counts[reward_type] = self.reward_type_counts.get(reward_type, 0) + 1
            self.examples.append(
                {
                    "prompt_input_ids": ids,
                    "prompt_text": prompt,
                    "answer": raw["answer"],
                    "category": category,
                    "reward_type": reward_type,
                    "keyword": raw.get("keyword", ""),
                    "truncated": truncated,
                }
            )
        if not self.examples:
            raise ValueError("all GRPO examples were skipped for %s" % path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.examples[idx]
        return {
            "prompt_input_ids": torch.tensor(item["prompt_input_ids"], dtype=torch.long),
            "prompt_text": item["prompt_text"],
            "answer": item["answer"],
            "category": item["category"],
            "reward_type": item["reward_type"],
            "keyword": item.get("keyword", ""),
        }

    def stats(self) -> Dict[str, object]:
        lengths = [len(item["prompt_input_ids"]) for item in self.examples]
        return {
            "path": self.path,
            "raw_examples": len(self.raw_examples),
            "effective_examples": len(self.examples),
            "truncated_prompts": self.truncated_prompts,
            "max_prompt_length": self.max_prompt_length,
            "prompt_length_min": min(lengths),
            "prompt_length_max": max(lengths),
            "prompt_length_mean": sum(lengths) / len(lengths),
            "category_counts": self.category_counts,
            "reward_type_counts": self.reward_type_counts,
        }


def grpo_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "prompt_input_ids": [item["prompt_input_ids"] for item in batch],
        "prompt_text": [item["prompt_text"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "category": [item["category"] for item in batch],
        "reward_type": [item["reward_type"] for item in batch],
        "keyword": [item.get("keyword", "") for item in batch],
    }


def write_jsonl(rows: Iterable[Dict[str, str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
