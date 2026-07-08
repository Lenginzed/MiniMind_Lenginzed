# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.generation import generate
from minillm.config import MiniLLMConfig
from minillm.model import MiniLLMForCausalLM
from minillm.tokenizer import MiniTokenizer
from minillm.utils import ensure_dir, get_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate samples from a tiny pretrain checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", default="outputs/pretrain_tiny/samples/after.txt")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = get_device(True)
    tokenizer = MiniTokenizer.load(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = MiniLLMConfig(**checkpoint["model_config"])
    lm = MiniLLMForCausalLM(config).to(device)
    lm.load_state_dict(checkpoint["model_state_dict"])
    lm.eval()

    prompts = [
        "Mini language models",
        "RoPE rotates",
        "小模型",
        "The pilot checks",
    ]
    bos_id = tokenizer.special_token_ids.get("bos_token_id")
    eos_id = tokenizer.special_token_ids.get("eos_token_id")
    lines = [
        "Tiny pretrain smoke generation. This is not a model-quality claim.",
        "checkpoint: %s" % args.checkpoint,
        "",
    ]
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if bos_id is not None:
            ids = [int(bos_id)] + ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = generate(
                lm,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=eos_id,
                do_sample=True,
            )
        text = tokenizer.decode(out[0].detach().cpu().tolist(), skip_special_tokens=True)
        lines.append("PROMPT: %s" % prompt)
        lines.append("OUTPUT: %s" % text)
        lines.append("")
    ensure_dir(str(Path(args.out).parent))
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("device:", device)
    print("wrote:", args.out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
