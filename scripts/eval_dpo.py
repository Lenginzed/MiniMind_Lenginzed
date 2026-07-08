# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.config import MiniLLMConfig
from minillm.generation import generate
from minillm.lora import apply_lora
from minillm.model import MiniLLMForCausalLM
from minillm.tokenizer import MiniTokenizer
from minillm.utils import ensure_dir, get_device, load_yaml


PROMPTS = [
    "什么是 LoRA？",
    "Explain causal language modeling.",
    "用三点解释 SFT 和预训练的区别。",
    "What does gradient checkpointing do?",
    "空战智能体为什么需要奖励函数？",
]


def build_model(config, checkpoint_path: str, device: torch.device) -> MiniLLMForCausalLM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = MiniLLMConfig(**checkpoint["model_config"])
    model = MiniLLMForCausalLM(model_config).to(device)
    lora_cfg = dict(config.get("lora", {}))
    if bool(lora_cfg.get("enabled", False)):
        model, _ = apply_lora(
            model,
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            r=int(lora_cfg.get("r", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            freeze_base=True,
        )
        model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DPO smoke generations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = get_device(True)
    tokenizer = MiniTokenizer.load(config["tokenizer_path"])
    model = build_model(config, args.checkpoint, device)
    bos_id = tokenizer.special_token_ids.get("bos_token_id")
    eos_id = tokenizer.special_token_ids.get("eos_token_id")

    lines = [
        "DPO smoke generation. This is not a real preference-alignment capability claim.",
        "",
    ]
    for prompt in PROMPTS:
        text_prompt = "User: %s\nAssistant: " % prompt
        ids = tokenizer.encode(text_prompt, add_special_tokens=False)
        if bos_id is not None:
            ids = [int(bos_id)] + ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = generate(
                model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=eos_id,
                do_sample=True,
            )
        full_ids = out[0].detach().cpu().tolist()
        completion_ids = full_ids[input_ids.shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        lines.append("PROMPT: %s" % prompt)
        lines.append("COMPLETION: %s" % completion)
        lines.append("FULL_DECODED: %s" % full_text)
        lines.append("")

    ensure_dir(str(Path(args.out).parent))
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("device:", device)
    print("wrote:", args.out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
