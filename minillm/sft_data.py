from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset

from .tokenizer import MiniTokenizer


IGNORE_INDEX = -100


def format_prompt(instruction: str, input_text: str = "") -> str:
    if input_text.strip():
        return "User: %s\n%s\nAssistant: " % (instruction.strip(), input_text.strip())
    return "User: %s\nAssistant: " % instruction.strip()


def format_full_text(instruction: str, output: str, input_text: str = "") -> str:
    return format_prompt(instruction, input_text) + output.strip()


def load_sft_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                {
                    "instruction": str(item.get("instruction", "")),
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", "")),
                    "category": str(item.get("category", "unknown")),
                }
            )
    if not rows:
        raise ValueError("SFT jsonl is empty: %s" % path)
    return rows


def encode_sft_example(
    tokenizer: MiniTokenizer,
    example: Dict[str, str],
    max_length: int,
) -> Optional[Dict[str, object]]:
    bos_id = tokenizer.special_token_ids["bos_token_id"]
    eos_id = tokenizer.special_token_ids["eos_token_id"]
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer must define bos/eos token ids")

    prompt = format_prompt(example["instruction"], example.get("input", ""))
    output = example["output"].strip()
    prompt_ids = [int(bos_id)] + tokenizer.encode(prompt, add_special_tokens=False)
    output_ids = tokenizer.encode(output, add_special_tokens=False) + [int(eos_id)]
    input_ids = prompt_ids + output_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + output_ids

    truncated = len(input_ids) > max_length
    if truncated:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    assistant_label_count = sum(1 for x in labels if x != IGNORE_INDEX)
    if assistant_label_count <= 0:
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "category": example.get("category", "unknown"),
        "truncated": truncated,
        "assistant_label_count": assistant_label_count,
    }


class SFTDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: MiniTokenizer,
        max_length: int,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.raw_examples = load_sft_jsonl(path)
        self.examples: List[Dict[str, object]] = []
        self.skipped = 0
        self.truncated = 0
        self.assistant_label_tokens = 0
        self.category_counts: Dict[str, int] = {}
        for item in self.raw_examples:
            encoded = encode_sft_example(tokenizer, item, self.max_length)
            if encoded is None:
                self.skipped += 1
                continue
            self.examples.append(encoded)
            self.truncated += int(bool(encoded["truncated"]))
            self.assistant_label_tokens += int(encoded["assistant_label_count"])
            category = str(encoded["category"])
            self.category_counts[category] = self.category_counts.get(category, 0) + 1
        if not self.examples:
            raise ValueError("all SFT examples were skipped for %s" % path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }

    def stats(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "raw_examples": len(self.raw_examples),
            "effective_examples": len(self.examples),
            "skipped_examples": self.skipped,
            "truncated_examples": self.truncated,
            "avg_assistant_label_tokens": self.assistant_label_tokens / max(1, len(self.examples)),
            "category_counts": self.category_counts,
            "max_length": self.max_length,
        }


def sft_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        length = item["input_ids"].numel()
        pad_len = max_len - length
        input_ids.append(
            torch.cat(
                [
                    item["input_ids"],
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                ]
            )
        )
        labels.append(
            torch.cat(
                [
                    item["labels"],
                    torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                ]
            )
        )
        attention_mask.append(
            torch.cat(
                [
                    item["attention_mask"],
                    torch.zeros((pad_len,), dtype=torch.long),
                ]
            )
        )
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
    }


def write_jsonl(rows: Iterable[Dict[str, str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
