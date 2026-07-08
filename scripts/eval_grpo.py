# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.config import MiniLLMConfig
from minillm.grpo_data import GRPODataset
from minillm.grpo_trainer import evaluate_grpo
from minillm.lora import apply_lora
from minillm.model import MiniLLMForCausalLM
from minillm.tokenizer import MiniTokenizer
from minillm.utils import ensure_dir, get_device, load_yaml, save_json


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
    parser = argparse.ArgumentParser(description="Evaluate GRPO smoke checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = get_device(True)
    tokenizer = MiniTokenizer.load(config["tokenizer_path"])
    model = build_model(config, args.checkpoint, device)
    val_ds = GRPODataset(config["val_data_path"], tokenizer, int(config["max_prompt_length"]))
    report = evaluate_grpo(
        model,
        tokenizer,
        val_ds,
        device,
        max_new_tokens=int(config["max_new_tokens"]),
        temperature=float(config.get("temperature", 1.0)),
        top_k=int(config["top_k"]) if config.get("top_k") is not None else None,
        top_p=float(config["top_p"]) if config.get("top_p") is not None else None,
        max_examples=args.max_examples,
    )
    report["checkpoint"] = args.checkpoint
    report["note"] = "GRPO smoke eval only; this is not a real reasoning or RL alignment capability claim."
    ensure_dir(args.out_dir)
    report_path = str(Path(args.out_dir) / "eval_report.json")
    save_json(report, report_path)

    sample_path = str(Path(config["output_dir"]) / "samples" / "after.txt")
    lines = ["GRPO smoke samples. This is not a real reasoning capability claim.", ""]
    for sample in report.get("samples", []):
        lines.append("PROMPT: %s" % sample["prompt"])
        lines.append("ANSWER: %s" % sample["answer"])
        lines.append("COMPLETION: %s" % sample["completion"])
        lines.append("REWARD: %.4f" % float(sample["reward"]))
        lines.append("")
    ensure_dir(str(Path(sample_path).parent))
    Path(sample_path).write_text("\n".join(lines), encoding="utf-8")

    print("device:", device)
    print("wrote:", report_path)
    print("wrote:", sample_path)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
