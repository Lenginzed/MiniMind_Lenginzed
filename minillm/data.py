from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import MiniTokenizer
from .utils import ensure_dir, save_json


def encode_text_file(
    tokenizer: MiniTokenizer,
    input_path: str,
    add_special_tokens_per_line: bool = True,
) -> np.ndarray:
    token_ids = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            token_ids.extend(tokenizer.encode(line, add_special_tokens=add_special_tokens_per_line))
    if not token_ids:
        raise ValueError("input text produced no token ids")
    return np.asarray(token_ids, dtype=np.int32)


def split_tokens(
    tokens: np.ndarray,
    val_ratio: float = 0.1,
    min_val_tokens: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D array")
    if len(tokens) < 2:
        raise ValueError("need at least two tokens to split")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    val_count = max(min_val_tokens, int(len(tokens) * val_ratio))
    val_count = min(val_count, len(tokens) - 1)
    train = tokens[:-val_count]
    val = tokens[-val_count:]
    if len(train) == 0 or len(val) == 0:
        raise ValueError("train/val split produced an empty split")
    return train, val


def validate_token_ids(tokens: np.ndarray, vocab_size: int) -> None:
    if tokens.size == 0:
        raise ValueError("token array is empty")
    min_id = int(tokens.min())
    max_id = int(tokens.max())
    if min_id < 0:
        raise ValueError("token ids must be non-negative")
    if max_id >= vocab_size:
        raise ValueError(
            "token id %d exceeds vocab_size %d" % (max_id, vocab_size)
        )


def validate_block_size(block_size: int, context_length: int) -> None:
    if block_size <= 1:
        raise ValueError("block_size must be greater than 1")
    if block_size > context_length:
        raise ValueError(
            "block_size (%d) cannot exceed context_length (%d)"
            % (block_size, context_length)
        )


def split_random_blocks(
    tokens: np.ndarray,
    block_size: int,
    val_ratio: float = 0.1,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if block_size <= 1:
        raise ValueError("block_size must be greater than 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    num_blocks = len(tokens) // block_size
    if num_blocks < 2:
        raise ValueError(
            "not enough tokens for random block split: need at least %d tokens, got %d"
            % (block_size * 2, len(tokens))
        )
    usable = tokens[: num_blocks * block_size].reshape(num_blocks, block_size)
    rng = np.random.default_rng(seed)
    indices = np.arange(num_blocks)
    rng.shuffle(indices)
    val_blocks = max(1, int(num_blocks * val_ratio))
    val_blocks = min(val_blocks, num_blocks - 1)
    val_idx = np.sort(indices[:val_blocks])
    train_idx = np.sort(indices[val_blocks:])
    train = usable[train_idx].reshape(-1)
    val = usable[val_idx].reshape(-1)
    return train, val, {
        "total_blocks": int(num_blocks),
        "train_blocks": int(len(train_idx)),
        "val_blocks": int(len(val_idx)),
        "dropped_tokens": int(len(tokens) - num_blocks * block_size),
    }


class CausalLMBlockDataset(Dataset):
    def __init__(self, tokens: Iterable[int], block_size: int) -> None:
        if block_size <= 1:
            raise ValueError("block_size must be greater than 1")
        array = np.asarray(list(tokens) if not isinstance(tokens, np.ndarray) else tokens, dtype=np.int64)
        if array.ndim != 1:
            raise ValueError("tokens must be 1D")
        self.tokens = array
        self.block_size = int(block_size)
        self.num_blocks = len(self.tokens) // self.block_size
        if self.num_blocks <= 0:
            raise ValueError("not enough tokens for one block")

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_blocks:
            raise IndexError(idx)
        start = idx * self.block_size
        block = self.tokens[start : start + self.block_size]
        input_ids = torch.tensor(block, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def save_tokenized_splits(
    tokenizer: MiniTokenizer,
    input_path: str,
    out_dir: str,
    block_size: int,
    val_ratio: float = 0.1,
    split_mode: str = "contiguous",
    seed: int = 1234,
) -> Dict[str, object]:
    ensure_dir(out_dir)
    tokens = encode_text_file(tokenizer, input_path)
    validate_token_ids(tokens, tokenizer.vocab_size)
    split_mode = split_mode.lower()
    split_extra: Dict[str, int] = {}
    if split_mode == "contiguous":
        train, val = split_tokens(tokens, val_ratio=val_ratio, min_val_tokens=block_size)
    elif split_mode == "random_blocks":
        train, val, split_extra = split_random_blocks(
            tokens,
            block_size=block_size,
            val_ratio=val_ratio,
            seed=seed,
        )
    else:
        raise ValueError("split_mode must be contiguous or random_blocks")
    train_samples = int(len(train) // block_size)
    val_samples = int(len(val) // block_size)
    if train_samples <= 0 or val_samples <= 0:
        raise ValueError(
            "not enough tokens for non-empty train/val samples with block_size=%d "
            "(train_tokens=%d, val_tokens=%d)" % (block_size, len(train), len(val))
        )
    train_path = str(Path(out_dir) / "train.npy")
    val_path = str(Path(out_dir) / "val.npy")
    np.save(train_path, train)
    np.save(val_path, val)
    metadata: Dict[str, object] = {
        "input_path": input_path,
        "train_path": train_path,
        "val_path": val_path,
        "total_tokens": int(len(tokens)),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "block_size": int(block_size),
        "train_samples": train_samples,
        "val_samples": val_samples,
        "val_ratio": float(val_ratio),
        "split_mode": split_mode,
        "seed": int(seed),
        "vocab_size": tokenizer.vocab_size,
    }
    metadata.update(split_extra)
    metadata.update(tokenizer.special_token_ids)
    save_json(metadata, str(Path(out_dir) / "metadata.json"))
    return metadata


def load_token_array(path: str) -> np.ndarray:
    return np.load(path)


def load_block_datasets(
    train_path: str,
    val_path: str,
    block_size: int,
) -> Tuple[CausalLMBlockDataset, CausalLMBlockDataset]:
    train = CausalLMBlockDataset(load_token_array(train_path), block_size)
    val = CausalLMBlockDataset(load_token_array(val_path), block_size)
    return train, val
