from __future__ import annotations

from pathlib import Path

import torch

from minillm.sft_data import IGNORE_INDEX, SFTDataset, encode_sft_example, sft_collate_fn
from minillm.tokenizer import MiniTokenizer


def build_tokenizer(tmp_path: Path) -> MiniTokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("User: hello\nAssistant: world\n什么是 LoRA？ LoRA freezes base weights.\n" * 50), encoding="utf-8")
    return MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)


def test_assistant_only_labels(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    example = {
        "instruction": "What is LoRA?",
        "input": "",
        "output": "LoRA freezes base weights.",
        "category": "concept",
    }
    encoded = encode_sft_example(tokenizer, example, max_length=64)
    assert encoded is not None
    labels = encoded["labels"]
    first_label = next(i for i, value in enumerate(labels) if value != IGNORE_INDEX)
    assert all(value == IGNORE_INDEX for value in labels[:first_label])
    assert any(value != IGNORE_INDEX for value in labels)
    assert labels[-1] == tokenizer.special_token_ids["eos_token_id"]


def test_sft_collate_padding_labels(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"instruction":"Short?","input":"","output":"Yes.","category":"x"}\n'
        '{"instruction":"Explain LoRA briefly.","input":"","output":"LoRA trains small adapters while the base model is frozen.","category":"x"}\n',
        encoding="utf-8",
    )
    dataset = SFTDataset(str(path), tokenizer, max_length=64)
    batch = sft_collate_fn([dataset[0], dataset[1]], int(tokenizer.special_token_ids["pad_token_id"]))
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    pad_positions = batch["attention_mask"] == 0
    assert torch.equal(batch["labels"][pad_positions], torch.full_like(batch["labels"][pad_positions], IGNORE_INDEX))


def test_sft_dataset_stats(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"instruction":"A","input":"","output":"B","category":"concept"}\n'
        '{"instruction":"C","input":"","output":"D","category":"math"}\n',
        encoding="utf-8",
    )
    dataset = SFTDataset(str(path), tokenizer, max_length=32)
    stats = dataset.stats()
    assert stats["effective_examples"] == 2
    assert stats["avg_assistant_label_tokens"] > 0
    assert stats["category_counts"]["concept"] == 1
