from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset

from .sft_data import IGNORE_INDEX, format_prompt
from .tokenizer import MiniTokenizer


def load_dpo_jsonl(path: str) -> List[Dict[str, str]]:
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
                    "prompt": str(item.get("prompt", "")),
                    "chosen": str(item.get("chosen", "")),
                    "rejected": str(item.get("rejected", "")),
                    "category": str(item.get("category", "unknown")),
                    "rejected_type": str(item.get("rejected_type", item.get("reason", "unknown"))),
                    "reason": str(item.get("reason", "")),
                }
            )
    if not rows:
        raise ValueError("DPO jsonl is empty: %s" % path)
    return rows


def dpo_prompt(example: Dict[str, str]) -> str:
    if example.get("prompt"):
        return example["prompt"]
    return format_prompt(example["instruction"], example.get("input", ""))


def encode_prompt_response(
    tokenizer: MiniTokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> Optional[Dict[str, object]]:
    bos_id = tokenizer.special_token_ids["bos_token_id"]
    eos_id = tokenizer.special_token_ids["eos_token_id"]
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer must define bos/eos token ids")
    prompt_ids = [int(bos_id)] + tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response.strip(), add_special_tokens=False) + [int(eos_id)]
    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    truncated = len(input_ids) > max_length
    if truncated:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    label_count = sum(1 for item in labels if item != IGNORE_INDEX)
    if label_count <= 0:
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "truncated": truncated,
        "label_count": label_count,
    }


class DPODataset(Dataset):
    def __init__(self, path: str, tokenizer: MiniTokenizer, max_length: int) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.raw_examples = load_dpo_jsonl(path)
        self.examples: List[Dict[str, object]] = []
        self.skipped = 0
        self.truncated = 0
        self.chosen_label_tokens = 0
        self.rejected_label_tokens = 0
        self.category_counts: Dict[str, int] = {}
        self.rejected_type_counts: Dict[str, int] = {}
        for raw in self.raw_examples:
            prompt = dpo_prompt(raw)
            chosen = encode_prompt_response(tokenizer, prompt, raw["chosen"], self.max_length)
            rejected = encode_prompt_response(tokenizer, prompt, raw["rejected"], self.max_length)
            if chosen is None or rejected is None:
                self.skipped += 1
                continue
            self.truncated += int(bool(chosen["truncated"])) + int(bool(rejected["truncated"]))
            self.chosen_label_tokens += int(chosen["label_count"])
            self.rejected_label_tokens += int(rejected["label_count"])
            category = raw["category"]
            rejected_type = raw["rejected_type"]
            self.category_counts[category] = self.category_counts.get(category, 0) + 1
            self.rejected_type_counts[rejected_type] = self.rejected_type_counts.get(rejected_type, 0) + 1
            self.examples.append(
                {
                    "chosen": chosen,
                    "rejected": rejected,
                    "category": category,
                    "rejected_type": rejected_type,
                }
            )
        if not self.examples:
            raise ValueError("all DPO examples were skipped for %s" % path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.examples[idx]
        chosen = item["chosen"]
        rejected = item["rejected"]
        return {
            "chosen_input_ids": torch.tensor(chosen["input_ids"], dtype=torch.long),
            "chosen_labels": torch.tensor(chosen["labels"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen["attention_mask"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected["input_ids"], dtype=torch.long),
            "rejected_labels": torch.tensor(rejected["labels"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected["attention_mask"], dtype=torch.long),
        }

    def stats(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "raw_examples": len(self.raw_examples),
            "effective_examples": len(self.examples),
            "skipped_examples": self.skipped,
            "truncated_sequences": self.truncated,
            "category_counts": self.category_counts,
            "rejected_type_counts": self.rejected_type_counts,
            "avg_chosen_label_tokens": self.chosen_label_tokens / max(1, len(self.examples)),
            "avg_rejected_label_tokens": self.rejected_label_tokens / max(1, len(self.examples)),
            "max_length": self.max_length,
        }


def _pad_1d(items: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(item.numel() for item in items)
    padded = []
    for item in items:
        pad_len = max_len - item.numel()
        padded.append(torch.cat([item, torch.full((pad_len,), pad_value, dtype=item.dtype)]))
    return torch.stack(padded, dim=0)


def dpo_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    return {
        "chosen_input_ids": _pad_1d([item["chosen_input_ids"] for item in batch], pad_token_id),
        "chosen_labels": _pad_1d([item["chosen_labels"] for item in batch], IGNORE_INDEX),
        "chosen_attention_mask": _pad_1d([item["chosen_attention_mask"] for item in batch], 0),
        "rejected_input_ids": _pad_1d([item["rejected_input_ids"] for item in batch], pad_token_id),
        "rejected_labels": _pad_1d([item["rejected_labels"] for item in batch], IGNORE_INDEX),
        "rejected_attention_mask": _pad_1d([item["rejected_attention_mask"] for item in batch], 0),
    }


def write_jsonl(rows: Iterable[Dict[str, str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
