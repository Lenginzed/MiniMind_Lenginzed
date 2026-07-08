# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.config import MiniLLMConfig
from minillm.generation import generate
from minillm.gptq import apply_gptq_style_quantization, collect_linear_calibration_stats
from minillm.model import MiniLLMForCausalLM
from minillm.quantization import (
    compression_report,
    estimate_model_size_bytes,
    estimate_quantized_size_bytes,
    quantize_model_weight_only,
)
from minillm.sft_data import SFTDataset, sft_collate_fn
from minillm.smoothquant import apply_smoothquant, collect_smoothquant_stats
from minillm.tokenizer import MiniTokenizer
from minillm.trainer import move_batch
from minillm.utils import autocast_context, ensure_dir, get_device, load_yaml, resolve_dtype, safe_perplexity, save_json, save_yaml, set_seed


PROMPTS = [
    "什么是 LoRA？",
    "Explain causal language modeling.",
    "用三点解释 SFT 和预训练的区别。",
    "What does gradient checkpointing do?",
    "空战智能体为什么需要奖励函数？",
]


def load_model(checkpoint_path: str, device: torch.device) -> MiniLLMForCausalLM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = MiniLLMConfig(**checkpoint["model_config"])
    model = MiniLLMForCausalLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_loader(path: str, tokenizer: MiniTokenizer, max_length: int, batch_size: int, shuffle: bool = False):
    pad_id = tokenizer.special_token_ids["pad_token_id"]
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id")
    dataset = SFTDataset(path, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda batch: sft_collate_fn(batch, int(pad_id)),
    )
    return dataset, loader


@torch.no_grad()
def evaluate_loss(model, loader, device, dtype_name: str, max_batches: int) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = move_batch(batch, device)
        with autocast_context(device, dtype_name):
            outputs = model(batch["input_ids"], labels=batch["labels"])
        loss = outputs["loss"]
        if loss is not None and torch.isfinite(loss):
            losses.append(float(loss.detach().cpu().item()))
    if was_training:
        model.train()
    if not losses:
        return float("nan")
    return float(sum(losses) / len(losses))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_forward_latency(model, batch, device, dtype_name: str, warmup: int = 2, iters: int = 8) -> Dict[str, float]:
    batch = move_batch(batch, device)
    model.eval()
    for _ in range(warmup):
        with autocast_context(device, dtype_name):
            model(batch["input_ids"], labels=batch["labels"])
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        with autocast_context(device, dtype_name):
            model(batch["input_ids"], labels=batch["labels"])
    _sync(device)
    elapsed = time.perf_counter() - start
    return {"forward_latency_ms": 1000.0 * elapsed / max(1, iters), "forward_iters": iters}


@torch.no_grad()
def measure_generate_latency(model, tokenizer: MiniTokenizer, device, max_new_tokens: int = 16, warmup: int = 1, iters: int = 3) -> Dict[str, float]:
    bos_id = tokenizer.special_token_ids.get("bos_token_id")
    eos_id = tokenizer.special_token_ids.get("eos_token_id")
    ids = tokenizer.encode("User: Explain LoRA.\nAssistant: ", add_special_tokens=False)
    if bos_id is not None:
        ids = [int(bos_id)] + ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(warmup):
        generate(model, input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50, top_p=0.9, eos_token_id=eos_id)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        generate(model, input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50, top_p=0.9, eos_token_id=eos_id)
    _sync(device)
    elapsed = time.perf_counter() - start
    return {"generate_latency_ms": 1000.0 * elapsed / max(1, iters), "generate_iters": iters, "generate_tokens": max_new_tokens}


@torch.no_grad()
def write_samples(model, tokenizer: MiniTokenizer, path: str, device: torch.device, max_new_tokens: int = 64) -> None:
    ensure_dir(str(Path(path).parent))
    bos_id = tokenizer.special_token_ids.get("bos_token_id")
    eos_id = tokenizer.special_token_ids.get("eos_token_id")
    lines = [
        "Quantization smoke samples. Educational fake quantization only; not a deployment-speed claim.",
        "",
    ]
    for prompt in PROMPTS:
        text_prompt = "User: %s\nAssistant: " % prompt
        ids = tokenizer.encode(text_prompt, add_special_tokens=False)
        if bos_id is not None:
            ids = [int(bos_id)] + ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
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
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_quant_eval(config_path: str) -> Dict[str, object]:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 20260712)))
    output_dir = config["output_dir"]
    for sub in ["samples", "logs"]:
        ensure_dir(str(Path(output_dir) / sub))
    save_yaml(config, str(Path(output_dir) / "quant_config_resolved.yaml"))
    device = get_device(bool(config.get("prefer_cuda", True)))
    dtype_name = resolve_dtype(str(config.get("dtype", "auto")))
    if device.type != "cuda":
        dtype_name = "fp32"
    tokenizer = MiniTokenizer.load(config["tokenizer_path"])
    model = load_model(config["checkpoint"], device)
    max_length = int(config.get("max_length", model.config.context_length))
    batch_size = int(config.get("batch_size", 8))
    eval_ds, eval_loader = build_loader(config["eval_data_path"], tokenizer, max_length=max_length, batch_size=batch_size)
    first_batch = next(iter(eval_loader))

    baseline_size = estimate_model_size_bytes(model)
    baseline_loss = evaluate_loss(model, eval_loader, device, dtype_name, int(config.get("eval_max_batches", 10)))
    baseline_ppl = safe_perplexity(baseline_loss)
    baseline_forward_latency = measure_forward_latency(model, first_batch, device, dtype_name)
    baseline_generate_latency = measure_generate_latency(model, tokenizer, device)

    method = str(config["method"])
    num_bits = int(config["num_bits"])
    per_channel = bool(config.get("per_channel", True))
    quant_stats: Dict[str, object]
    if method == "weight_only":
        quant_stats = quantize_model_weight_only(model, num_bits=num_bits, per_channel=per_channel)
    elif method == "gptq_style":
        calib_ds, calib_loader = build_loader(
            config["calibration_data_path"],
            tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=False,
        )
        stats = collect_linear_calibration_stats(
            model,
            calib_loader,
            target_modules=config.get("target_modules"),
            max_batches=int(config.get("calibration_max_batches", 10)),
        )
        quant_stats = apply_gptq_style_quantization(model, stats, num_bits=num_bits, per_channel=per_channel)
        quant_stats["calibration_examples"] = len(calib_ds)
    elif method == "smoothquant_style":
        calib_ds, calib_loader = build_loader(
            config["calibration_data_path"],
            tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=False,
        )
        stats = collect_smoothquant_stats(
            model,
            calib_loader,
            target_modules=config.get("target_modules"),
            max_batches=int(config.get("calibration_max_batches", 10)),
        )
        quant_stats = apply_smoothquant(
            model,
            stats,
            alpha=float(config.get("alpha", 0.5)),
            num_bits=num_bits,
            per_channel=per_channel,
        )
        quant_stats["calibration_examples"] = len(calib_ds)
    else:
        raise ValueError("unknown quantization method: %s" % method)

    quantized_size = estimate_quantized_size_bytes(model)
    size_report = compression_report(baseline_size, quantized_size)
    quant_loss = evaluate_loss(model, eval_loader, device, dtype_name, int(config.get("eval_max_batches", 10)))
    quant_ppl = safe_perplexity(quant_loss)
    quant_forward_latency = measure_forward_latency(model, first_batch, device, dtype_name)
    quant_generate_latency = measure_generate_latency(model, tokenizer, device)
    sample_path = str(Path(output_dir) / "samples" / "after.txt")
    write_samples(model, tokenizer, sample_path, device)

    report = {
        "config_path": config_path,
        "checkpoint": config["checkpoint"],
        "tokenizer_path": config["tokenizer_path"],
        "eval_data_path": config["eval_data_path"],
        "method": method,
        "num_bits": num_bits,
        "per_channel": per_channel,
        "dtype": dtype_name,
        "device": str(device),
        "eval_examples": len(eval_ds),
        "eval_max_batches": int(config.get("eval_max_batches", 10)),
        "baseline_loss": finite_or_none(baseline_loss),
        "baseline_ppl": finite_or_none(baseline_ppl),
        "quantized_loss": finite_or_none(quant_loss),
        "quantized_ppl": finite_or_none(quant_ppl),
        "loss_delta": finite_or_none(quant_loss - baseline_loss),
        "ppl_delta": finite_or_none((quant_ppl - baseline_ppl) if quant_ppl is not None and baseline_ppl is not None else None),
        "size": size_report,
        "baseline_latency": {**baseline_forward_latency, **baseline_generate_latency},
        "quantized_latency": {**quant_forward_latency, **quant_generate_latency},
        "latency_note": "Fake quantization uses dequantized floating-point F.linear and does not represent production integer-kernel speed.",
        "quantization_stats": quant_stats,
        "sample_path": sample_path,
    }
    report_path = str(Path(output_dir) / "eval_report.json")
    save_json(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate educational quantization modes.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_quant_eval(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
