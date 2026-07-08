# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.tokenizer import MiniTokenizer, discover_text_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Byte-level BPE tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Text file(s), glob(s), or directories.")
    parser.add_argument("--output", required=True, help="Output tokenizer JSON path.")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    files = discover_text_files(args.input)
    tokenizer = MiniTokenizer.train_from_files(
        files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.save(args.output)
    sample = "Mini LLM smoke test: 小模型检查 tokenizer encode and decode."
    sample_ids = tokenizer.encode(sample)
    sample_text = tokenizer.decode(sample_ids)
    sample_path = str(Path(args.output).with_suffix(".sample.json"))
    Path(sample_path).write_text(
        json.dumps(
            {
                "sample": sample,
                "ids": sample_ids,
                "decoded": sample_text,
                "summary": tokenizer.summary(),
                "files": files,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("tokenizer:", args.output)
    print("vocab_size:", tokenizer.vocab_size)
    print("special_token_ids:", tokenizer.special_token_ids)
    print("sample_ids:", sample_ids[:32])
    print("sample_decoded:", sample_text)
    print("sample_file:", sample_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
