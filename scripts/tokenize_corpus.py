# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.data import save_tokenized_splits
from minillm.tokenizer import MiniTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize a text corpus into train/val npy files.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=["contiguous", "random_blocks"], default="contiguous")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tokenizer = MiniTokenizer.load(args.tokenizer)
    metadata = save_tokenized_splits(
        tokenizer,
        args.input,
        args.out_dir,
        block_size=args.block_size,
        val_ratio=args.val_ratio,
        split_mode=args.split_mode,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("metadata:", str(Path(args.out_dir) / "metadata.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
