# Stage 3 Source Snapshot

## `minillm/sft_data.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset

from .tokenizer import MiniTokenizer


IGNORE_INDEX = -100


def format_prompt(instruction: str, input_text: str = "") -> str:
    if input_text.strip():
        return "User: %s\n%s\nAssistant: " % (instruction.strip(), input_text.strip())
    return "User: %s\nAssistant: " % instruction.strip()


def format_full_text(instruction: str, output: str, input_text: str = "") -> str:
    return format_prompt(instruction, input_text) + output.strip()


def load_sft_jsonl(path: str) -> List[Dict[str, str]]:
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
                    "output": str(item.get("output", "")),
                    "category": str(item.get("category", "unknown")),
                }
            )
    if not rows:
        raise ValueError("SFT jsonl is empty: %s" % path)
    return rows


def encode_sft_example(
    tokenizer: MiniTokenizer,
    example: Dict[str, str],
    max_length: int,
) -> Optional[Dict[str, object]]:
    bos_id = tokenizer.special_token_ids["bos_token_id"]
    eos_id = tokenizer.special_token_ids["eos_token_id"]
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer must define bos/eos token ids")

    prompt = format_prompt(example["instruction"], example.get("input", ""))
    output = example["output"].strip()
    prompt_ids = [int(bos_id)] + tokenizer.encode(prompt, add_special_tokens=False)
    output_ids = tokenizer.encode(output, add_special_tokens=False) + [int(eos_id)]
    input_ids = prompt_ids + output_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + output_ids

    truncated = len(input_ids) > max_length
    if truncated:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    assistant_label_count = sum(1 for x in labels if x != IGNORE_INDEX)
    if assistant_label_count <= 0:
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "category": example.get("category", "unknown"),
        "truncated": truncated,
        "assistant_label_count": assistant_label_count,
    }


class SFTDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: MiniTokenizer,
        max_length: int,
    ) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.raw_examples = load_sft_jsonl(path)
        self.examples: List[Dict[str, object]] = []
        self.skipped = 0
        self.truncated = 0
        self.assistant_label_tokens = 0
        self.category_counts: Dict[str, int] = {}
        for item in self.raw_examples:
            encoded = encode_sft_example(tokenizer, item, self.max_length)
            if encoded is None:
                self.skipped += 1
                continue
            self.examples.append(encoded)
            self.truncated += int(bool(encoded["truncated"]))
            self.assistant_label_tokens += int(encoded["assistant_label_count"])
            category = str(encoded["category"])
            self.category_counts[category] = self.category_counts.get(category, 0) + 1
        if not self.examples:
            raise ValueError("all SFT examples were skipped for %s" % path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }

    def stats(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "raw_examples": len(self.raw_examples),
            "effective_examples": len(self.examples),
            "skipped_examples": self.skipped,
            "truncated_examples": self.truncated,
            "avg_assistant_label_tokens": self.assistant_label_tokens / max(1, len(self.examples)),
            "category_counts": self.category_counts,
            "max_length": self.max_length,
        }


def sft_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        length = item["input_ids"].numel()
        pad_len = max_len - length
        input_ids.append(
            torch.cat(
                [
                    item["input_ids"],
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                ]
            )
        )
        labels.append(
            torch.cat(
                [
                    item["labels"],
                    torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                ]
            )
        )
        attention_mask.append(
            torch.cat(
                [
                    item["attention_mask"],
                    torch.zeros((pad_len,), dtype=torch.long),
                ]
            )
        )
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
    }


def write_jsonl(rows: Iterable[Dict[str, str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
```

## `minillm/lora.py`

```python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        module_name: str = "",
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base_layer = base_layer
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.scaling = float(lora_alpha) / float(r)
        self.module_name = module_name
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        for param in self.base_layer.parameters():
            param.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base + update


def freeze_all_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _set_lora_trainable(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True


def apply_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> Tuple[nn.Module, Dict[str, object]]:
    target_set = set(target_modules)
    if freeze_base:
        freeze_all_parameters(model)
    replaced: List[str] = []

    def replace_children(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = "%s.%s" % (prefix, name) if prefix else name
            if name in target_set and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r=r, lora_alpha=alpha, lora_dropout=dropout, module_name=full_name))
                replaced.append(full_name)
            else:
                replace_children(child, full_name)

    replace_children(model)
    _set_lora_trainable(model)
    stats = lora_parameter_stats(model)
    stats.update(
        {
            "target_modules": sorted(target_set),
            "replaced_modules": replaced,
            "r": int(r),
            "alpha": int(alpha),
            "dropout": float(dropout),
        }
    )
    return model, stats


def lora_parameter_stats(model: nn.Module) -> Dict[str, object]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for p in [module.lora_A.weight, module.lora_B.weight]
    )
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "lora_params": int(lora_params),
        "trainable_ratio": float(trainable / total) if total else 0.0,
        "lora_module_count": sum(1 for module in model.modules() if isinstance(module, LoRALinear)),
    }


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state["%s.lora_A.weight" % name] = module.lora_A.weight.detach().cpu()
            state["%s.lora_B.weight" % name] = module.lora_B.weight.detach().cpu()
    return state


def save_lora_adapter(model: nn.Module, path: str, config: Dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora_state_dict": lora_state_dict(model), "config": config}, path)
    json_path = str(Path(path).with_suffix(".json"))
    Path(json_path).write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_lora_adapter(model: nn.Module, path: str, strict: bool = True) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    adapter_state = payload["lora_state_dict"]
    modules = dict(model.named_modules())
    missing = []
    for key, value in adapter_state.items():
        module_name, _, param_name = key.rpartition(".")
        module_name = module_name.rsplit(".", 1)[0]
        short_name = key.split(".")[-2]
        module = modules.get(module_name)
        if not isinstance(module, LoRALinear):
            missing.append(key)
            continue
        if short_name == "lora_A":
            module.lora_A.weight.data.copy_(value.to(module.lora_A.weight.device))
        elif short_name == "lora_B":
            module.lora_B.weight.data.copy_(value.to(module.lora_B.weight.device))
    if strict and missing:
        raise KeyError("missing LoRA modules for keys: %s" % missing)
    return payload.get("config", {})
```

## `minillm/sft_trainer.py`

```python
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import MiniLLMConfig
from .generation import generate
from .lora import apply_lora, lora_parameter_stats, save_lora_adapter
from .model import MiniLLMForCausalLM, count_parameters
from .sft_data import SFTDataset, sft_collate_fn
from .tokenizer import MiniTokenizer
from .trainer import build_lr_scheduler, cuda_memory_record, current_lr, move_batch
from .utils import append_jsonl, autocast_context, ensure_dir, get_device, load_yaml, resolve_dtype, safe_perplexity, save_json, save_yaml, set_seed


def load_base_model(base_checkpoint: str, device: torch.device) -> MiniLLMForCausalLM:
    checkpoint = torch.load(base_checkpoint, map_location=device)
    config = MiniLLMConfig(**checkpoint["model_config"])
    model = MiniLLMForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def sft_evaluate(model: MiniLLMForCausalLM, loader: DataLoader, device: torch.device, dtype_name: str, max_batches: int) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            batch = move_batch(batch, device)
            with autocast_context(device, dtype_name):
                outputs = model(batch["input_ids"], labels=batch["labels"])
            loss = outputs["loss"]
            if loss is not None:
                losses.append(float(loss.detach().cpu().item()))
    if was_training:
        model.train()
    return float(sum(losses) / len(losses)) if losses else float("nan")


def save_sft_checkpoint(
    path: str,
    model: MiniLLMForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: Dict[str, Any],
    best_eval_loss: float,
    mode: str,
    adapter_path: Optional[str] = None,
) -> None:
    ensure_dir(str(Path(path).parent))
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "config": config,
        "model_config": asdict(model.config),
        "best_eval_loss": float(best_eval_loss),
        "mode": mode,
        "adapter_path": adapter_path,
    }
    torch.save(payload, path)


def write_sft_samples(
    model: MiniLLMForCausalLM,
    tokenizer: MiniTokenizer,
    prompts,
    path: str,
    device: torch.device,
    max_new_tokens: int = 64,
) -> None:
    ensure_dir(str(Path(path).parent))
    was_training = model.training
    model.eval()
    bos_id = tokenizer.special_token_ids.get("bos_token_id")
    eos_id = tokenizer.special_token_ids.get("eos_token_id")
    lines = [
        "SFT smoke generation. This is not a real instruction-following capability claim.",
        "",
    ]
    for prompt in prompts:
        text_prompt = "User: %s\nAssistant: " % prompt
        ids = tokenizer.encode(text_prompt, add_special_tokens=False)
        if bos_id is not None:
            ids = [int(bos_id)] + ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
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
        decoded = tokenizer.decode(out[0].detach().cpu().tolist(), skip_special_tokens=True)
        lines.append("PROMPT: %s" % prompt)
        lines.append("OUTPUT: %s" % decoded)
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    if was_training:
        model.train()


def run_sft(config_path: str) -> Dict[str, Any]:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 1234)))
    output_dir = config["output_dir"]
    for sub in ["checkpoints", "adapters", "logs", "plots", "samples"]:
        ensure_dir(str(Path(output_dir) / sub))

    tokenizer = MiniTokenizer.load(config["tokenizer_path"])
    pad_id = tokenizer.special_token_ids["pad_token_id"]
    if pad_id is None:
        raise ValueError("tokenizer must provide pad_token_id")

    device = get_device(bool(config.get("prefer_cuda", True)))
    dtype_name = resolve_dtype(str(config.get("dtype", "auto")))
    if device.type != "cuda":
        dtype_name = "fp32"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = load_base_model(config["base_checkpoint"], device)
    lora_cfg = dict(config.get("lora", {}))
    mode = "lora" if bool(lora_cfg.get("enabled", False)) else "full"
    lora_stats: Dict[str, object] = {}
    if mode == "lora":
        model, lora_stats = apply_lora(
            model,
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            r=int(lora_cfg.get("r", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            freeze_base=True,
        )
        model.to(device)
    else:
        for param in model.parameters():
            param.requires_grad = True

    train_ds = SFTDataset(config["train_data_path"], tokenizer, int(config["max_length"]))
    val_ds = SFTDataset(config["val_data_path"], tokenizer, int(config["max_length"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda batch: sft_collate_fn(batch, int(pad_id)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda batch: sft_collate_fn(batch, int(pad_id)),
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    max_steps = int(config["training"]["max_steps"])
    scheduler = build_lr_scheduler(
        optimizer,
        str(config["training"].get("scheduler", "none")),
        int(config["training"].get("warmup_steps", 0)),
        max_steps,
    )
    use_scaler = device.type == "cuda" and dtype_name == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    metrics_path = str(Path(output_dir) / "metrics.jsonl")
    csv_path = str(Path(output_dir) / "metrics.csv")
    for path in [metrics_path, csv_path]:
        if Path(path).exists():
            Path(path).unlink()
    save_yaml(config, str(Path(output_dir) / "train_config_resolved.yaml"))
    save_json({"train": train_ds.stats(), "val": val_ds.stats()}, str(Path(output_dir) / "data_stats.json"))

    writer = SummaryWriter(log_dir=str(Path(output_dir) / "logs"))
    prompts = config.get("sample_prompts", [])
    write_sft_samples(model, tokenizer, prompts, str(Path(output_dir) / "samples" / "before.txt"), device)

    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    eval_interval = int(config["training"].get("eval_interval", 25))
    save_interval = int(config["training"].get("save_interval", 50))
    eval_batches = int(config["training"].get("eval_batches", 10))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    train_iter = iter(train_loader)
    best_eval_loss = float("inf")
    first_train_loss = None
    last_train_loss = None
    last_eval_loss = None

    csv_fields = ["step", "train_loss", "eval_loss", "train_ppl", "eval_ppl", "lr", "grad_norm", "trainable_params", "total_params"]
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step in range(1, max_steps + 1):
            losses = []
            for _ in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                batch = move_batch(batch, device)
                with autocast_context(device, dtype_name):
                    outputs = model(batch["input_ids"], labels=batch["labels"])
                    loss = outputs["loss"]
                    if loss is None:
                        raise RuntimeError("model did not return loss")
                    scaled_loss = loss / grad_accum
                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                losses.append(float(loss.detach().cpu().item()))
            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss = float(sum(losses) / len(losses))
            first_train_loss = train_loss if first_train_loss is None else first_train_loss
            last_train_loss = train_loss
            eval_loss = None
            adapter_path = str(Path(output_dir) / "adapters" / "best_adapter.pt") if mode == "lora" else None
            if step == 1 or step % eval_interval == 0 or step == max_steps:
                eval_loss = sft_evaluate(model, val_loader, device, dtype_name, eval_batches)
                last_eval_loss = eval_loss
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if mode == "lora":
                        save_lora_adapter(model, adapter_path, {**lora_cfg, **lora_stats})
                    save_sft_checkpoint(
                        str(Path(output_dir) / "checkpoints" / "best.pt"),
                        model,
                        optimizer,
                        scheduler,
                        step,
                        config,
                        best_eval_loss,
                        mode,
                        adapter_path,
                    )
            if step % save_interval == 0 or step == max_steps:
                last_adapter = str(Path(output_dir) / "adapters" / "last_adapter.pt") if mode == "lora" else None
                if mode == "lora":
                    save_lora_adapter(model, last_adapter, {**lora_cfg, **lora_stats})
                save_sft_checkpoint(
                    str(Path(output_dir) / "checkpoints" / "last.pt"),
                    model,
                    optimizer,
                    scheduler,
                    step,
                    config,
                    best_eval_loss,
                    mode,
                    last_adapter,
                )

            record = {
                "step": step,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "train_ppl": safe_perplexity(train_loss),
                "eval_ppl": safe_perplexity(eval_loss),
                "lr": current_lr(optimizer),
                "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                "trainable_params": int(trainable_params),
                "total_params": int(total_params),
            }
            append_jsonl(record, metrics_path)
            csv_writer.writerow({key: record.get(key) for key in csv_fields})
            csv_file.flush()
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("lr", record["lr"], step)
            writer.add_scalar("grad_norm", record["grad_norm"], step)
            if eval_loss is not None:
                writer.add_scalar("loss/eval", eval_loss, step)
            if step == 1 or step % int(config["training"].get("log_interval", 25)) == 0 or step == max_steps:
                print("step=%d train_loss=%.4f eval_loss=%s lr=%.6g" % (
                    step,
                    train_loss,
                    "%.4f" % eval_loss if eval_loss is not None else "None",
                    record["lr"],
                ))
    finally:
        csv_file.close()
        writer.close()

    if not Path(output_dir, "checkpoints", "last.pt").exists():
        save_sft_checkpoint(str(Path(output_dir) / "checkpoints" / "last.pt"), model, optimizer, scheduler, max_steps, config, best_eval_loss, mode)
    if not Path(output_dir, "checkpoints", "best.pt").exists():
        save_sft_checkpoint(str(Path(output_dir) / "checkpoints" / "best.pt"), model, optimizer, scheduler, max_steps, config, best_eval_loss, mode)

    write_sft_samples(model, tokenizer, prompts, str(Path(output_dir) / "samples" / "after.txt"), device)
    summary = {
        "mode": mode,
        "output_dir": output_dir,
        "base_checkpoint": config["base_checkpoint"],
        "parameter_count": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_ratio": float(trainable_params / total_params) if total_params else 0.0,
        "device": str(device),
        "dtype": dtype_name,
        "max_steps": max_steps,
        "first_train_loss": first_train_loss,
        "last_train_loss": last_train_loss,
        "last_eval_loss": last_eval_loss,
        "best_eval_loss": best_eval_loss,
        "best_eval_ppl": safe_perplexity(best_eval_loss),
        "metrics_path": metrics_path,
        "best_checkpoint": str(Path(output_dir) / "checkpoints" / "best.pt"),
        "last_checkpoint": str(Path(output_dir) / "checkpoints" / "last.pt"),
        "sample_path": str(Path(output_dir) / "samples" / "after.txt"),
        "lora": lora_stats,
    }
    save_json(summary, str(Path(output_dir) / "sft_summary.json"))
    return summary
```

## `scripts/create_sft_dataset.py`

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.sft_data import write_jsonl


def make_examples(total: int, seed: int):
    rng = random.Random(seed)
    examples = []
    concepts = [
        ("tokenizer", "A tokenizer converts text into token ids and can decode ids back into text."),
        ("pretraining", "Pretraining optimizes next-token prediction on broad text before task-specific tuning."),
        ("SFT", "SFT trains on instruction and response pairs so the model follows a desired response format."),
        ("LoRA", "LoRA freezes the base model and trains small low-rank adapter matrices."),
        ("gradient checkpointing", "Gradient checkpointing saves memory by recomputing activations during backward."),
        ("scheduler", "A scheduler changes the learning rate during training, often with warmup and decay."),
        ("checkpoint", "A checkpoint stores model weights, optimizer state, step, and configuration."),
    ]
    translations = [
        ("因果语言模型", "causal language model"),
        ("奖励函数", "reward function"),
        ("检查点", "checkpoint"),
        ("分词器", "tokenizer"),
        ("梯度裁剪", "gradient clipping"),
    ]
    categories = ["concept", "math", "translation", "flight_rl", "format", "code"]
    for idx in range(total):
        cat = categories[idx % len(categories)]
        if cat == "concept":
            name, answer = rng.choice(concepts)
            inst = "Explain %s in one or two sentences." % name
            out = answer
        elif cat == "math":
            a = rng.randint(1, 50)
            b = rng.randint(1, 50)
            inst = "What is %d + %d? Show the result briefly." % (a, b)
            out = "%d + %d = %d." % (a, b, a + b)
        elif cat == "translation":
            zh, en = rng.choice(translations)
            if rng.random() < 0.5:
                inst = "Translate this Chinese technical term into English: %s" % zh
                out = "%s means %s." % (zh, en)
            else:
                inst = "解释术语：%s" % zh
                out = "%s 通常可以理解为 %s，是机器学习或控制任务中的常见概念。" % (zh, en)
        elif cat == "flight_rl":
            inst = rng.choice(
                [
                    "Why does an air-combat agent need a reward function?",
                    "What should a flight controller monitor during a maneuver?",
                    "Why is policy stability important in reinforcement learning?",
                ]
            )
            out = rng.choice(
                [
                    "A reward function turns task goals into feedback, such as safety, target progress, and energy management.",
                    "The controller should monitor altitude, speed, heading, energy, and safety limits.",
                    "Stable policies reduce erratic actions and make training easier to evaluate.",
                ]
            )
        elif cat == "format":
            inst = rng.choice(
                [
                    "Use three bullet points to explain the difference between pretraining and SFT.",
                    "用三点说明 LoRA 的优点。",
                    "List three things to log during training.",
                ]
            )
            if "LoRA" in inst:
                out = "1. 冻结 base model。\n2. 只训练低秩 adapter。\n3. 显著减少可训练参数。"
            elif "log" in inst:
                out = "1. Train loss and eval loss.\n2. Learning rate and gradient norm.\n3. Checkpoint path and tokens seen."
            else:
                out = "1. Pretraining learns next-token prediction.\n2. SFT learns instruction-response behavior.\n3. SFT usually uses curated examples."
        else:
            inst = rng.choice(
                [
                    "What does AdamW do?",
                    "What should a training checkpoint include?",
                    "Why clip gradients?",
                ]
            )
            if "AdamW" in inst:
                out = "AdamW is an optimizer that combines Adam-style adaptive updates with decoupled weight decay."
            elif "checkpoint" in inst:
                out = "It should include model weights, optimizer state, scheduler state, step, and config."
            else:
                out = "Gradient clipping limits very large updates and can improve training stability."
        examples.append({"instruction": inst, "input": "", "output": out, "category": cat})
    rng.shuffle(examples)
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local synthetic SFT dataset.")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    rows = make_examples(args.train_size + args.val_size, args.seed)
    train = rows[: args.train_size]
    val = rows[args.train_size :]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, str(out_dir / "sft_train.jsonl"))
    write_jsonl(val, str(out_dir / "sft_val.jsonl"))
    meta = {
        "train_size": len(train),
        "val_size": len(val),
        "seed": args.seed,
        "train_categories": dict(Counter(row["category"] for row in train)),
        "val_categories": dict(Counter(row["category"] for row in val)),
        "note": "Synthetic local SFT data for pipeline validation only; not real instruction-tuning data quality.",
    }
    (out_dir / "sft_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `scripts/train_sft.py`

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.sft_trainer import run_sft


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full SFT or LoRA-SFT smoke.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = run_sft(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `scripts/eval_sft.py`

```python
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


def build_model_from_config(config, checkpoint_path: str, device: torch.device):
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
    parser = argparse.ArgumentParser(description="Evaluate SFT smoke generations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = get_device(True)
    tokenizer = MiniTokenizer.load(config["tokenizer_path"])
    model = build_model_from_config(config, args.checkpoint, device)
    bos_id = tokenizer.special_token_ids.get("bos_token_id")
    eos_id = tokenizer.special_token_ids.get("eos_token_id")
    lines = ["SFT smoke generation. This is not a real instruction-following capability claim.", ""]
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
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                eos_token_id=eos_id,
                do_sample=True,
            )
        decoded = tokenizer.decode(out[0].detach().cpu().tolist(), skip_special_tokens=True)
        lines.append("PROMPT: %s" % prompt)
        lines.append("OUTPUT: %s" % decoded)
        lines.append("")
    ensure_dir(str(Path(args.out).parent))
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("device:", device)
    print("wrote:", args.out)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `configs/sft_full.yaml`

```yaml
seed: 20260709
base_checkpoint: outputs/pretrain_stage2_hardened/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
train_data_path: data/sft/sft_train.jsonl
val_data_path: data/sft/sft_val.jsonl
output_dir: outputs/sft_full
prefer_cuda: true
dtype: auto
max_length: 128

training:
  batch_size: 16
  gradient_accumulation_steps: 1
  max_steps: 160
  eval_interval: 25
  save_interval: 50
  log_interval: 25
  eval_batches: 10
  learning_rate: 1.0e-4
  weight_decay: 0.01
  grad_clip: 1.0
  scheduler: cosine
  warmup_steps: 20

sample_prompts:
  - "什么是 LoRA？"
  - "Explain causal language modeling."
  - "用三点解释 SFT 和预训练的区别。"
  - "What does gradient checkpointing do?"
  - "空战智能体为什么需要奖励函数？"
```

## `configs/sft_lora.yaml`

```yaml
seed: 20260709
base_checkpoint: outputs/pretrain_stage2_hardened/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
train_data_path: data/sft/sft_train.jsonl
val_data_path: data/sft/sft_val.jsonl
output_dir: outputs/sft_lora
prefer_cuda: true
dtype: auto
max_length: 128

lora:
  enabled: true
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj

training:
  batch_size: 16
  gradient_accumulation_steps: 1
  max_steps: 160
  eval_interval: 25
  save_interval: 50
  log_interval: 25
  eval_batches: 10
  learning_rate: 5.0e-4
  weight_decay: 0.0
  grad_clip: 1.0
  scheduler: cosine
  warmup_steps: 20

sample_prompts:
  - "什么是 LoRA？"
  - "Explain causal language modeling."
  - "用三点解释 SFT 和预训练的区别。"
  - "What does gradient checkpointing do?"
  - "空战智能体为什么需要奖励函数？"
```

## `tests/test_sft_data.py`

```python
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
```

## `tests/test_lora.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.lora import LoRALinear, apply_lora, lora_parameter_stats


def test_lora_linear_shape_and_zero_b_equivalence() -> None:
    torch.manual_seed(0)
    base = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 3, 8)
    expected = base(x)
    wrapped = LoRALinear(base, r=2, lora_alpha=4, lora_dropout=0.0)
    actual = wrapped(x)
    assert actual.shape == (2, 3, 4)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_apply_lora_only_targets_q_v_and_freezes_base() -> None:
    config = MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
    )
    model = MiniLLMForCausalLM(config)
    model, stats = apply_lora(model, target_modules=["q_proj", "v_proj"], r=4, alpha=8, dropout=0.0)
    replaced = stats["replaced_modules"]
    assert replaced
    assert all(name.endswith("q_proj") or name.endswith("v_proj") for name in replaced)
    assert not any(name.endswith("k_proj") or name.endswith("o_proj") for name in replaced)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all("lora_A" in name or "lora_B" in name for name in trainable_names)
    stats = lora_parameter_stats(model)
    assert stats["trainable_params"] < stats["total_params"]
    assert stats["lora_module_count"] == 4
```

## `tests/test_sft_trainer_smoke.py`

```python
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.model import count_parameters
from minillm.sft_data import write_jsonl
from minillm.sft_trainer import run_sft
from minillm.tokenizer import MiniTokenizer


def setup_tiny_sft_run(tmp_path: Path, lora: bool) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("User: What is LoRA?\nAssistant: LoRA trains adapters.\n小模型 训练\n" * 100), encoding="utf-8")
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    tokenizer_path = tmp_path / "tok.json"
    tokenizer.save(str(tokenizer_path))
    train_rows = [
        {"instruction": "What is LoRA?", "input": "", "output": "LoRA trains small adapters.", "category": "concept"},
        {"instruction": "What is SFT?", "input": "", "output": "SFT uses instruction and response pairs.", "category": "concept"},
    ] * 20
    val_rows = train_rows[:8]
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(train_rows, str(train_path))
    write_jsonl(val_rows, str(val_path))
    cfg = MiniLLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        intermediate_size=64,
    )
    model = MiniLLMForCausalLM(cfg)
    base_ckpt = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": cfg.__dict__,
            "optimizer_state_dict": {},
            "step": 0,
            "best_eval_loss": 0.0,
        },
        base_ckpt,
    )
    config = {
        "seed": 123,
        "base_checkpoint": str(base_ckpt),
        "tokenizer_path": str(tokenizer_path),
        "train_data_path": str(train_path),
        "val_data_path": str(val_path),
        "output_dir": str(tmp_path / ("lora" if lora else "full")),
        "prefer_cuda": False,
        "dtype": "fp32",
        "max_length": 32,
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
            "eval_interval": 1,
            "save_interval": 1,
            "log_interval": 1,
            "eval_batches": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "scheduler": "none",
            "warmup_steps": 0,
        },
        "sample_prompts": ["What is LoRA?"],
    }
    if lora:
        config["lora"] = {"enabled": True, "r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
    config_path = tmp_path / ("lora.yaml" if lora else "full.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_full_sft_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_sft(str(setup_tiny_sft_run(tmp_path, lora=False)))
    assert summary["mode"] == "full"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] == summary["parameter_count"]
    assert Path(summary["best_checkpoint"]).exists()


def test_lora_sft_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_sft(str(setup_tiny_sft_run(tmp_path, lora=True)))
    assert summary["mode"] == "lora"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] < summary["parameter_count"]
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["output_dir"], "adapters", "best_adapter.pt").exists()
```
