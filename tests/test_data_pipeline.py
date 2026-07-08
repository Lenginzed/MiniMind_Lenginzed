from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from minillm.data import (
    CausalLMBlockDataset,
    save_tokenized_splits,
    split_random_blocks,
    split_tokens,
    validate_block_size,
)
from minillm.tokenizer import MiniTokenizer


def build_tokenizer(tmp_path: Path) -> MiniTokenizer:
    corpus = tmp_path / "toy.txt"
    corpus.write_text(("hello world 小模型\n" * 100), encoding="utf-8")
    return MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)


def test_block_dataset_shape_and_labels() -> None:
    tokens = np.arange(64, dtype=np.int32)
    dataset = CausalLMBlockDataset(tokens, block_size=16)
    item = dataset[0]
    assert len(dataset) == 4
    assert item["input_ids"].shape == (16,)
    assert item["labels"].shape == (16,)
    assert torch.equal(item["input_ids"], item["labels"])


def test_train_val_split_not_empty() -> None:
    tokens = np.arange(100, dtype=np.int32)
    train, val = split_tokens(tokens, val_ratio=0.2)
    assert len(train) == 80
    assert len(val) == 20


def test_save_tokenized_splits_metadata(tmp_path: Path) -> None:
    corpus = tmp_path / "toy.txt"
    corpus.write_text(("hello world 小模型\n" * 100), encoding="utf-8")
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    out_dir = tmp_path / "processed"
    metadata = save_tokenized_splits(
        tokenizer,
        str(corpus),
        str(out_dir),
        block_size=8,
        val_ratio=0.1,
    )
    assert (out_dir / "train.npy").exists()
    assert (out_dir / "val.npy").exists()
    assert (out_dir / "metadata.json").exists()
    assert metadata["train_tokens"] > 0
    assert metadata["val_tokens"] > 0
    assert metadata["train_samples"] > 0
    assert metadata["val_samples"] > 0


def test_random_blocks_split_reproducible() -> None:
    tokens = np.arange(128, dtype=np.int32)
    train_a, val_a, meta_a = split_random_blocks(tokens, block_size=8, val_ratio=0.25, seed=42)
    train_b, val_b, meta_b = split_random_blocks(tokens, block_size=8, val_ratio=0.25, seed=42)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert meta_a == meta_b
    assert meta_a["train_blocks"] > 0
    assert meta_a["val_blocks"] > 0


def test_random_blocks_too_small_raises() -> None:
    try:
        split_random_blocks(np.arange(10, dtype=np.int32), block_size=8, val_ratio=0.1)
    except ValueError as exc:
        assert "not enough tokens" in str(exc)
        return
    raise AssertionError("split_random_blocks should reject tiny inputs")


def test_block_size_larger_than_context_raises() -> None:
    try:
        validate_block_size(block_size=128, context_length=64)
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
        return
    raise AssertionError("validate_block_size should reject block_size > context_length")
