# Stage 4 Source Snapshot

## `minillm/dpo_data.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset

from .sft_data import IGNORE_INDEX, format_prompt
from .tokenizer import MiniTokenizer


def load_dpo_jsonl(path: str) -> List[Dict[str, str]]:
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
                    "prompt": str(item.get("prompt", "")),
                    "chosen": str(item.get("chosen", "")),
                    "rejected": str(item.get("rejected", "")),
                    "category": str(item.get("category", "unknown")),
                    "rejected_type": str(item.get("rejected_type", item.get("reason", "unknown"))),
                    "reason": str(item.get("reason", "")),
                }
            )
    if not rows:
        raise ValueError("DPO jsonl is empty: %s" % path)
    return rows


def dpo_prompt(example: Dict[str, str]) -> str:
    if example.get("prompt"):
        return example["prompt"]
    return format_prompt(example["instruction"], example.get("input", ""))


def encode_prompt_response(
    tokenizer: MiniTokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> Optional[Dict[str, object]]:
    bos_id = tokenizer.special_token_ids["bos_token_id"]
    eos_id = tokenizer.special_token_ids["eos_token_id"]
    if bos_id is None or eos_id is None:
        raise ValueError("tokenizer must define bos/eos token ids")
    prompt_ids = [int(bos_id)] + tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response.strip(), add_special_tokens=False) + [int(eos_id)]
    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    truncated = len(input_ids) > max_length
    if truncated:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    label_count = sum(1 for item in labels if item != IGNORE_INDEX)
    if label_count <= 0:
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "truncated": truncated,
        "label_count": label_count,
    }


class DPODataset(Dataset):
    def __init__(self, path: str, tokenizer: MiniTokenizer, max_length: int) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.raw_examples = load_dpo_jsonl(path)
        self.examples: List[Dict[str, object]] = []
        self.skipped = 0
        self.truncated = 0
        self.chosen_label_tokens = 0
        self.rejected_label_tokens = 0
        self.category_counts: Dict[str, int] = {}
        self.rejected_type_counts: Dict[str, int] = {}
        for raw in self.raw_examples:
            prompt = dpo_prompt(raw)
            chosen = encode_prompt_response(tokenizer, prompt, raw["chosen"], self.max_length)
            rejected = encode_prompt_response(tokenizer, prompt, raw["rejected"], self.max_length)
            if chosen is None or rejected is None:
                self.skipped += 1
                continue
            self.truncated += int(bool(chosen["truncated"])) + int(bool(rejected["truncated"]))
            self.chosen_label_tokens += int(chosen["label_count"])
            self.rejected_label_tokens += int(rejected["label_count"])
            category = raw["category"]
            rejected_type = raw["rejected_type"]
            self.category_counts[category] = self.category_counts.get(category, 0) + 1
            self.rejected_type_counts[rejected_type] = self.rejected_type_counts.get(rejected_type, 0) + 1
            self.examples.append(
                {
                    "chosen": chosen,
                    "rejected": rejected,
                    "category": category,
                    "rejected_type": rejected_type,
                }
            )
        if not self.examples:
            raise ValueError("all DPO examples were skipped for %s" % path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.examples[idx]
        chosen = item["chosen"]
        rejected = item["rejected"]
        return {
            "chosen_input_ids": torch.tensor(chosen["input_ids"], dtype=torch.long),
            "chosen_labels": torch.tensor(chosen["labels"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen["attention_mask"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected["input_ids"], dtype=torch.long),
            "rejected_labels": torch.tensor(rejected["labels"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected["attention_mask"], dtype=torch.long),
        }

    def stats(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "raw_examples": len(self.raw_examples),
            "effective_examples": len(self.examples),
            "skipped_examples": self.skipped,
            "truncated_sequences": self.truncated,
            "category_counts": self.category_counts,
            "rejected_type_counts": self.rejected_type_counts,
            "avg_chosen_label_tokens": self.chosen_label_tokens / max(1, len(self.examples)),
            "avg_rejected_label_tokens": self.rejected_label_tokens / max(1, len(self.examples)),
            "max_length": self.max_length,
        }


def _pad_1d(items: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(item.numel() for item in items)
    padded = []
    for item in items:
        pad_len = max_len - item.numel()
        padded.append(torch.cat([item, torch.full((pad_len,), pad_value, dtype=item.dtype)]))
    return torch.stack(padded, dim=0)


def dpo_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    return {
        "chosen_input_ids": _pad_1d([item["chosen_input_ids"] for item in batch], pad_token_id),
        "chosen_labels": _pad_1d([item["chosen_labels"] for item in batch], IGNORE_INDEX),
        "chosen_attention_mask": _pad_1d([item["chosen_attention_mask"] for item in batch], 0),
        "rejected_input_ids": _pad_1d([item["rejected_input_ids"] for item in batch], pad_token_id),
        "rejected_labels": _pad_1d([item["rejected_labels"] for item in batch], IGNORE_INDEX),
        "rejected_attention_mask": _pad_1d([item["rejected_attention_mask"] for item in batch], 0),
    }


def write_jsonl(rows: Iterable[Dict[str, str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

```

## `minillm/dpo_trainer.py`

```python
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import MiniLLMConfig
from .dpo_data import DPODataset, dpo_collate_fn
from .lora import apply_lora, save_lora_adapter
from .model import MiniLLMForCausalLM
from .sft_trainer import write_sft_samples
from .tokenizer import MiniTokenizer
from .trainer import build_lr_scheduler, current_lr, move_batch
from .utils import append_jsonl, autocast_context, ensure_dir, get_device, load_yaml, resolve_dtype, save_json, save_yaml, set_seed


def sequence_logps(model: MiniLLMForCausalLM, input_ids: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(input_ids)
    logits = outputs["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    seq_logps = token_logps.sum(dim=-1)
    token_counts = mask.sum(dim=-1).clamp_min(1)
    mean_logps = seq_logps / token_counts
    return seq_logps, token_counts, mean_logps


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> Dict[str, torch.Tensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    reward_margin = chosen_rewards - rejected_rewards
    preference_accuracy = (chosen_rewards > rejected_rewards).float()
    return {
        "loss": losses.mean(),
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "reward_margin": reward_margin,
        "preference_accuracy": preference_accuracy,
        "logits": logits.detach(),
    }


def load_policy_model(checkpoint_path: str, device: torch.device) -> MiniLLMForCausalLM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = MiniLLMConfig(**checkpoint["model_config"])
    model = MiniLLMForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def trainable_stats(model: torch.nn.Module) -> Dict[str, object]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": float(trainable / total) if total else 0.0,
    }


def dpo_batch_metrics(policy_model, ref_model, batch, device, dtype_name, beta: float) -> Dict[str, torch.Tensor]:
    chosen_ids = batch["chosen_input_ids"]
    chosen_labels = batch["chosen_labels"]
    rejected_ids = batch["rejected_input_ids"]
    rejected_labels = batch["rejected_labels"]
    with torch.no_grad():
        with autocast_context(device, dtype_name):
            ref_chosen, _, _ = sequence_logps(ref_model, chosen_ids, chosen_labels)
            ref_rejected, _, _ = sequence_logps(ref_model, rejected_ids, rejected_labels)
    with autocast_context(device, dtype_name):
        policy_chosen, chosen_counts, chosen_mean = sequence_logps(policy_model, chosen_ids, chosen_labels)
        policy_rejected, rejected_counts, rejected_mean = sequence_logps(policy_model, rejected_ids, rejected_labels)
        loss_data = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    loss_data.update(
        {
            "policy_chosen_logps": policy_chosen.detach(),
            "policy_rejected_logps": policy_rejected.detach(),
            "ref_chosen_logps": ref_chosen.detach(),
            "ref_rejected_logps": ref_rejected.detach(),
            "chosen_token_count": chosen_counts.detach(),
            "rejected_token_count": rejected_counts.detach(),
            "policy_chosen_mean_logps": chosen_mean.detach(),
            "policy_rejected_mean_logps": rejected_mean.detach(),
        }
    )
    return loss_data


def mean_item(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().mean().cpu().item())


@torch.no_grad()
def evaluate_dpo(policy_model, ref_model, loader, device, dtype_name, beta: float, max_batches: int) -> Dict[str, float]:
    was_training = policy_model.training
    policy_model.eval()
    ref_model.eval()
    accum = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        batch = move_batch(batch, device)
        data = dpo_batch_metrics(policy_model, ref_model, batch, device, dtype_name, beta)
        accum.append({key: mean_item(value) for key, value in data.items() if torch.is_tensor(value)})
    if was_training:
        policy_model.train()
    if not accum:
        return {}
    keys = accum[0].keys()
    return {key: sum(item[key] for item in accum) / len(accum) for key in keys}


def save_dpo_checkpoint(path: str, policy_model, optimizer, scheduler, step: int, config: Dict[str, Any], best_eval_loss: float, mode: str, adapter_path: Optional[str]) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(
        {
            "model_state_dict": policy_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "step": int(step),
            "config": config,
            "model_config": asdict(policy_model.config),
            "best_eval_loss": float(best_eval_loss),
            "mode": mode,
            "adapter_path": adapter_path,
        },
        path,
    )


def run_dpo(config_path: str) -> Dict[str, Any]:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 1234)))
    output_dir = config["output_dir"]
    for sub in ["checkpoints", "adapters", "logs", "plots", "samples"]:
        ensure_dir(str(Path(output_dir) / sub))
    tokenizer = MiniTokenizer.load(config["tokenizer_path"])
    pad_id = tokenizer.special_token_ids["pad_token_id"]
    if pad_id is None:
        raise ValueError("tokenizer needs pad_token_id")
    device = get_device(bool(config.get("prefer_cuda", True)))
    dtype_name = resolve_dtype(str(config.get("dtype", "auto")))
    if device.type != "cuda":
        dtype_name = "fp32"

    policy_model = load_policy_model(config["policy_checkpoint"], device)
    ref_model = load_policy_model(config["reference_checkpoint"], device)
    freeze_model(ref_model)
    ref_model.eval()
    lora_cfg = dict(config.get("lora", {}))
    mode = "lora" if bool(lora_cfg.get("enabled", False)) else "full"
    lora_stats: Dict[str, object] = {}
    if mode == "lora":
        policy_model, lora_stats = apply_lora(
            policy_model,
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            r=int(lora_cfg.get("r", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            freeze_base=True,
        )
        policy_model.to(device)
    else:
        for param in policy_model.parameters():
            param.requires_grad = True
    stats = trainable_stats(policy_model)

    train_ds = DPODataset(config["train_data_path"], tokenizer, int(config["max_length"]))
    val_ds = DPODataset(config["val_data_path"], tokenizer, int(config["max_length"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda batch: dpo_collate_fn(batch, int(pad_id)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda batch: dpo_collate_fn(batch, int(pad_id)),
    )
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
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
    beta = float(config.get("beta", 0.1))
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    eval_interval = int(config["training"].get("eval_interval", 25))
    save_interval = int(config["training"].get("save_interval", 50))
    eval_batches = int(config["training"].get("eval_batches", 10))
    grad_clip = float(config["training"].get("grad_clip", 1.0))

    metrics_path = str(Path(output_dir) / "metrics.jsonl")
    csv_path = str(Path(output_dir) / "metrics.csv")
    for path in [metrics_path, csv_path]:
        if Path(path).exists():
            Path(path).unlink()
    save_yaml(config, str(Path(output_dir) / "train_config_resolved.yaml"))
    save_json({"train": train_ds.stats(), "val": val_ds.stats()}, str(Path(output_dir) / "data_stats.json"))
    writer = SummaryWriter(log_dir=str(Path(output_dir) / "logs"))

    train_iter = iter(train_loader)
    first_loss = None
    last_loss = None
    last_eval = {}
    best_eval_loss = float("inf")
    csv_fields = [
        "step", "train_loss", "eval_loss", "reward_margin", "preference_accuracy",
        "chosen_rewards", "rejected_rewards", "policy_chosen_logps", "policy_rejected_logps",
        "ref_chosen_logps", "ref_rejected_logps", "lr", "grad_norm", "trainable_params", "total_params"
    ]
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    try:
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)
        for step in range(1, max_steps + 1):
            losses = []
            last_train_data = None
            for _ in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                batch = move_batch(batch, device)
                data = dpo_batch_metrics(policy_model, ref_model, batch, device, dtype_name, beta)
                loss = data["loss"] / grad_accum
                loss.backward()
                losses.append(mean_item(data["loss"]))
                last_train_data = data
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy_model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            train_loss = float(sum(losses) / len(losses))
            first_loss = train_loss if first_loss is None else first_loss
            last_loss = train_loss
            eval_data = None
            if step == 1 or step % eval_interval == 0 or step == max_steps:
                eval_data = evaluate_dpo(policy_model, ref_model, val_loader, device, dtype_name, beta, eval_batches)
                last_eval = eval_data
                if eval_data.get("loss", float("inf")) < best_eval_loss:
                    best_eval_loss = eval_data["loss"]
                    adapter_path = str(Path(output_dir) / "adapters" / "best_adapter.pt") if mode == "lora" else None
                    if mode == "lora":
                        save_lora_adapter(policy_model, adapter_path, {**lora_cfg, **lora_stats})
                    save_dpo_checkpoint(
                        str(Path(output_dir) / "checkpoints" / "best.pt"),
                        policy_model, optimizer, scheduler, step, config, best_eval_loss, mode, adapter_path
                    )
            if step % save_interval == 0 or step == max_steps:
                adapter_path = str(Path(output_dir) / "adapters" / "last_adapter.pt") if mode == "lora" else None
                if mode == "lora":
                    save_lora_adapter(policy_model, adapter_path, {**lora_cfg, **lora_stats})
                save_dpo_checkpoint(
                    str(Path(output_dir) / "checkpoints" / "last.pt"),
                    policy_model, optimizer, scheduler, step, config, best_eval_loss, mode, adapter_path
                )
            assert last_train_data is not None
            record = {
                "step": step,
                "train_loss": train_loss,
                "eval_loss": eval_data.get("loss") if eval_data else None,
                "reward_margin": mean_item(last_train_data["reward_margin"]),
                "preference_accuracy": mean_item(last_train_data["preference_accuracy"]),
                "chosen_rewards": mean_item(last_train_data["chosen_rewards"]),
                "rejected_rewards": mean_item(last_train_data["rejected_rewards"]),
                "policy_chosen_logps": mean_item(last_train_data["policy_chosen_logps"]),
                "policy_rejected_logps": mean_item(last_train_data["policy_rejected_logps"]),
                "ref_chosen_logps": mean_item(last_train_data["ref_chosen_logps"]),
                "ref_rejected_logps": mean_item(last_train_data["ref_rejected_logps"]),
                "eval_reward_margin": eval_data.get("reward_margin") if eval_data else None,
                "eval_preference_accuracy": eval_data.get("preference_accuracy") if eval_data else None,
                "lr": current_lr(optimizer),
                "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                "trainable_params": stats["trainable_params"],
                "total_params": stats["total_params"],
            }
            append_jsonl(record, metrics_path)
            csv_writer.writerow({key: record.get(key) for key in csv_fields})
            csv_file.flush()
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("reward_margin/train", record["reward_margin"], step)
            writer.add_scalar("preference_accuracy/train", record["preference_accuracy"], step)
            if eval_data:
                writer.add_scalar("loss/eval", eval_data["loss"], step)
                writer.add_scalar("reward_margin/eval", eval_data["reward_margin"], step)
                writer.add_scalar("preference_accuracy/eval", eval_data["preference_accuracy"], step)
            if step == 1 or step % int(config["training"].get("log_interval", 25)) == 0 or step == max_steps:
                print(
                    "step=%d loss=%.4f margin=%.4f acc=%.3f eval_loss=%s"
                    % (
                        step,
                        train_loss,
                        record["reward_margin"],
                        record["preference_accuracy"],
                        "%.4f" % record["eval_loss"] if record["eval_loss"] is not None else "None",
                    )
                )
    finally:
        csv_file.close()
        writer.close()

    if not Path(output_dir, "checkpoints", "last.pt").exists():
        save_dpo_checkpoint(str(Path(output_dir) / "checkpoints" / "last.pt"), policy_model, optimizer, scheduler, max_steps, config, best_eval_loss, mode, None)
    if not Path(output_dir, "checkpoints", "best.pt").exists():
        save_dpo_checkpoint(str(Path(output_dir) / "checkpoints" / "best.pt"), policy_model, optimizer, scheduler, max_steps, config, best_eval_loss, mode, None)
    prompts = config.get("sample_prompts", [])
    write_sft_samples(policy_model, tokenizer, prompts, str(Path(output_dir) / "samples" / "after.txt"), device)
    summary = {
        "mode": mode,
        "output_dir": output_dir,
        "policy_checkpoint": config["policy_checkpoint"],
        "reference_checkpoint": config["reference_checkpoint"],
        "beta": beta,
        "parameter_count": stats["total_params"],
        "trainable_params": stats["trainable_params"],
        "trainable_ratio": stats["trainable_ratio"],
        "device": str(device),
        "dtype": dtype_name,
        "max_steps": max_steps,
        "first_train_loss": first_loss,
        "last_train_loss": last_loss,
        "best_eval_loss": best_eval_loss,
        "last_eval": last_eval,
        "metrics_path": metrics_path,
        "best_checkpoint": str(Path(output_dir) / "checkpoints" / "best.pt"),
        "last_checkpoint": str(Path(output_dir) / "checkpoints" / "last.pt"),
        "sample_path": str(Path(output_dir) / "samples" / "after.txt"),
        "lora": lora_stats,
    }
    save_json(summary, str(Path(output_dir) / "dpo_summary.json"))
    return summary

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
        full_ids = out[0].detach().cpu().tolist()
        completion_ids = full_ids[input_ids.shape[1] :]
        decoded_completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        decoded_full = tokenizer.decode(full_ids, skip_special_tokens=True)
        lines.append("PROMPT: %s" % prompt)
        lines.append("COMPLETION: %s" % decoded_completion)
        lines.append("FULL_DECODED: %s" % decoded_full)
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

## `scripts/create_dpo_dataset.py`

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.dpo_data import write_jsonl
from minillm.utils import ensure_dir, save_json


REJECTED_TYPES = [
    "wrong_answer",
    "bad_format",
    "vague",
    "unsafe_or_unphysical",
    "off_topic",
    "hallucinated_term",
]


def concept_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    topics = [
        ("LoRA", "LoRA freezes the base model and trains small low-rank adapter matrices, so fine-tuning uses fewer trainable parameters."),
        ("DPO", "DPO compares chosen and rejected responses and optimizes the policy to increase the relative log probability of the chosen answer against a frozen reference."),
        ("SFT", "SFT trains on instruction-response pairs with labels usually applied only to the assistant response tokens."),
        ("RoPE", "RoPE rotates query and key vectors by position-dependent phases so attention can use relative position information."),
        ("GQA", "GQA uses more query heads than key-value heads, reducing KV memory while preserving several query groups."),
    ]
    topic, chosen = rng.choice(topics)
    return {
        "instruction": f"Explain {topic} in one concise paragraph.",
        "input": "",
        "chosen": chosen,
        "rejected": rejected_for(rejected_type, "concept", topic),
        "category": "concept",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def math_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    a = rng.randint(2, 49)
    b = rng.randint(2, 49)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    else:
        result = a * b
    return {
        "instruction": f"Compute {a} {op} {b}. Reply with the final integer and one short check.",
        "input": "",
        "chosen": f"{result}. Check: {a} {op} {b} = {result}.",
        "rejected": rejected_for(rejected_type, "math", str(result + rng.choice([-3, -1, 2, 5]))),
        "category": "math",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def translation_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    pairs = [
        ("Translate 'gradient checkpointing' into Chinese.", "梯度检查点"),
        ("Translate 'causal language modeling' into Chinese.", "因果语言建模"),
        ("Translate 'reward function' into Chinese.", "奖励函数"),
        ("Translate 'tokenizer vocabulary' into Chinese.", "分词器词表"),
        ("Translate 'attention head' into Chinese.", "注意力头"),
    ]
    instruction, chosen = rng.choice(pairs)
    return {
        "instruction": instruction,
        "input": "",
        "chosen": chosen,
        "rejected": rejected_for(rejected_type, "translation", chosen),
        "category": "translation",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def flight_rl_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    prompts = [
        (
            "Why does an air-combat RL agent need a reward function?",
            "A reward function turns task goals into learning signals, such as maintaining safety constraints, improving positioning, and completing objectives.",
        ),
        (
            "Explain why flight control policies must respect physical limits.",
            "Aircraft policies must respect speed, acceleration, actuator, and safety limits because actions outside those bounds are unphysical and unsafe.",
        ),
        (
            "What is a simple curriculum for an air-combat toy environment?",
            "Start with stable flight, add navigation, then add target tracking and finally constrained engagement tasks.",
        ),
    ]
    instruction, chosen = rng.choice(prompts)
    return {
        "instruction": instruction,
        "input": "",
        "chosen": chosen,
        "rejected": rejected_for(rejected_type, "flight_rl", chosen),
        "category": "flight_rl",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def format_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    topic = rng.choice(["SFT vs pretraining", "DPO pipeline checks", "tokenizer debugging", "training logs"])
    return {
        "instruction": f"Answer in exactly three bullet points about {topic}.",
        "input": "",
        "chosen": "- State the main objective.\n- Mention the key data or metric.\n- Note one limitation or risk.",
        "rejected": rejected_for(rejected_type, "format", topic),
        "category": "format",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def code_example(rng: random.Random, rejected_type: str) -> Dict[str, str]:
    topics = [
        ("AdamW", "AdamW decouples weight decay from the adaptive gradient update, which often makes regularization easier to tune."),
        ("gradient clipping", "Gradient clipping limits the norm of gradients before the optimizer step, helping avoid unstable updates."),
        ("cosine scheduler", "A cosine scheduler gradually lowers the learning rate following a cosine curve after warmup."),
        ("checkpoint", "A checkpoint stores model state, optimizer state, step, and configuration so training can be resumed or evaluated."),
    ]
    topic, chosen = rng.choice(topics)
    return {
        "instruction": f"Briefly explain {topic} in a PyTorch training loop.",
        "input": "",
        "chosen": chosen,
        "rejected": rejected_for(rejected_type, "code", topic),
        "category": "code",
        "rejected_type": rejected_type,
        "reason": reason_for(rejected_type),
    }


def rejected_for(rejected_type: str, category: str, payload: str) -> str:
    if rejected_type == "wrong_answer":
        if category == "math":
            return f"{payload}. Check: this is the exact result."
        if category == "translation":
            return "这个术语应翻译为随机梯度飞行。"
        return "The statement is false because training never uses gradients or data."
    if rejected_type == "bad_format":
        return "Here is a long paragraph without following the requested structure, numbering, or concise format at all."
    if rejected_type == "vague":
        return "It depends on many things, and the details are complicated."
    if rejected_type == "unsafe_or_unphysical":
        return "The best policy ignores constraints, uses unlimited acceleration, and rewards unsafe maneuvers."
    if rejected_type == "off_topic":
        return "My favorite color for a dashboard is blue, and the weather is pleasant today."
    if rejected_type == "hallucinated_term":
        return "Use HyperDPOFlux, AeroTokenizer Prime, and RewardNorm++ to solve it automatically."
    raise ValueError(f"unknown rejected_type: {rejected_type}")


def reason_for(rejected_type: str) -> str:
    reasons = {
        "wrong_answer": "The rejected response contains a clearly incorrect factual or mathematical answer.",
        "bad_format": "The rejected response does not follow the requested format.",
        "vague": "The rejected response is too generic to be useful.",
        "unsafe_or_unphysical": "The rejected response violates safety, physical, or task constraints.",
        "off_topic": "The rejected response does not answer the prompt.",
        "hallucinated_term": "The rejected response invents unsupported terminology.",
    }
    return reasons[rejected_type]


BUILDERS = [concept_example, math_example, translation_example, flight_rl_example, format_example, code_example]


def build_rows(count: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(count):
        builder = BUILDERS[idx % len(BUILDERS)]
        rejected_type = REJECTED_TYPES[(idx // len(BUILDERS)) % len(REJECTED_TYPES)]
        row = builder(rng, rejected_type)
        row["id"] = f"dpo_{seed}_{idx:06d}"
        rows.append(row)
    rng.shuffle(rows)
    return rows


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    return {
        "count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "rejected_type_counts": dict(Counter(row["rejected_type"] for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local synthetic DPO preference dataset.")
    parser.add_argument("--out-dir", default="data/dpo")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    train_rows = build_rows(args.train_size, args.seed)
    val_rows = build_rows(args.val_size, args.seed + 1)
    train_path = str(Path(args.out_dir) / "dpo_train.jsonl")
    val_path = str(Path(args.out_dir) / "dpo_val.jsonl")
    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)
    metadata = {
        "description": "Synthetic local preference data only for DPO pipeline validation; it is not real human preference data.",
        "seed": args.seed,
        "train_path": train_path,
        "val_path": val_path,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "format": {
            "fields": ["instruction", "input", "chosen", "rejected", "category", "rejected_type", "reason"],
            "prompt_template": "User: {instruction}\\n{input_if_any}\\nAssistant:",
        },
    }
    meta_path = str(Path(args.out_dir) / "dpo_metadata.json")
    save_json(metadata, meta_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `scripts/train_dpo.py`

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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.dpo_trainer import run_dpo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full DPO or DPO-LoRA smoke training.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = run_dpo(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `scripts/eval_dpo.py`

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
        full_ids = out[0].detach().cpu().tolist()
        completion_ids = full_ids[input_ids.shape[1] :]
        decoded_completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        decoded_full = tokenizer.decode(full_ids, skip_special_tokens=True)
        lines.append("PROMPT: %s" % prompt)
        lines.append("COMPLETION: %s" % decoded_completion)
        lines.append("FULL_DECODED: %s" % decoded_full)
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

## `tests/test_dpo_data.py`

```python
from __future__ import annotations

from pathlib import Path

from minillm.dpo_data import DPODataset, dpo_collate_fn, write_jsonl
from minillm.sft_data import IGNORE_INDEX
from minillm.tokenizer import MiniTokenizer


def build_tokenizer(tmp_path: Path) -> MiniTokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "User: Explain LoRA.\nAssistant: LoRA trains low-rank adapters.\n"
            "Chosen responses are concise and correct. Rejected answers are wrong or vague.\n"
        )
        * 80,
        encoding="utf-8",
    )
    return MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)


def test_dpo_dataset_masks_prompt_and_keeps_eos(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    eos_id = tokenizer.special_token_ids["eos_token_id"]
    rows = [
        {
            "instruction": "Explain LoRA.",
            "input": "",
            "chosen": "LoRA trains low-rank adapters.",
            "rejected": "It depends.",
            "category": "concept",
            "rejected_type": "vague",
            "reason": "too vague",
        }
    ]
    path = tmp_path / "dpo.jsonl"
    write_jsonl(rows, str(path))
    ds = DPODataset(str(path), tokenizer, max_length=64)
    item = ds[0]

    for labels_name in ["chosen_labels", "rejected_labels"]:
        labels = item[labels_name].tolist()
        first_valid = next(i for i, value in enumerate(labels) if value != IGNORE_INDEX)
        assert first_valid > 0
        assert all(value == IGNORE_INDEX for value in labels[:first_valid])
        assert eos_id in labels[first_valid:]
        assert sum(value != IGNORE_INDEX for value in labels) > 0

    stats = ds.stats()
    assert stats["raw_examples"] == 1
    assert stats["effective_examples"] == 1
    assert stats["skipped_examples"] == 0
    assert stats["category_counts"]["concept"] == 1
    assert stats["rejected_type_counts"]["vague"] == 1


def test_dpo_collate_padding_labels_are_ignored(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    pad_id = tokenizer.special_token_ids["pad_token_id"]
    rows = [
        {
            "instruction": "Explain LoRA.",
            "input": "",
            "chosen": "LoRA trains adapters.",
            "rejected": "Vague.",
            "category": "concept",
            "rejected_type": "vague",
            "reason": "too vague",
        },
        {
            "instruction": "Explain DPO.",
            "input": "",
            "chosen": "DPO compares chosen and rejected completions before updating the policy.",
            "rejected": "Blue dashboard.",
            "category": "concept",
            "rejected_type": "off_topic",
            "reason": "off topic",
        },
    ]
    path = tmp_path / "dpo.jsonl"
    write_jsonl(rows, str(path))
    ds = DPODataset(str(path), tokenizer, max_length=80)
    batch = dpo_collate_fn([ds[0], ds[1]], int(pad_id))
    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["rejected_input_ids"].shape[0] == 2
    assert batch["chosen_input_ids"].shape == batch["chosen_labels"].shape
    assert batch["rejected_input_ids"].shape == batch["rejected_labels"].shape
    chosen_pad = batch["chosen_attention_mask"] == 0
    rejected_pad = batch["rejected_attention_mask"] == 0
    assert (batch["chosen_labels"][chosen_pad] == IGNORE_INDEX).all()
    assert (batch["rejected_labels"][rejected_pad] == IGNORE_INDEX).all()


def test_dpo_truncation_can_skip_when_response_is_fully_cut(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(tmp_path)
    rows = [
        {
            "instruction": "Explain LoRA with a very long prompt " * 40,
            "input": "",
            "chosen": "LoRA trains adapters.",
            "rejected": "Wrong.",
            "category": "concept",
            "rejected_type": "wrong_answer",
            "reason": "wrong",
        }
    ]
    path = tmp_path / "dpo.jsonl"
    write_jsonl(rows, str(path))
    try:
        DPODataset(str(path), tokenizer, max_length=8)
    except ValueError as exc:
        assert "all DPO examples were skipped" in str(exc)
    else:
        raise AssertionError("expected all examples to be skipped when response labels are truncated away")

```

## `tests/test_dpo_loss.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minillm.dpo_trainer import dpo_loss, sequence_logps


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, input_ids: torch.Tensor):
        return {"logits": self.logits.expand(input_ids.shape[0], -1, -1).clone()}


def test_sequence_logps_only_counts_unmasked_response_tokens() -> None:
    vocab_size = 5
    logits = torch.zeros(1, 4, vocab_size)
    logits[0, 1, 2] = 4.0
    logits[0, 2, 3] = 5.0
    model = FixedLogitModel(logits)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    labels = torch.tensor([[-100, -100, 2, 3]])

    seq_logps, token_counts, mean_logps = sequence_logps(model, input_ids, labels)
    expected = F.log_softmax(logits[0, 1], dim=-1)[2] + F.log_softmax(logits[0, 2], dim=-1)[3]
    assert seq_logps.shape == (1,)
    assert token_counts.tolist() == [2]
    assert torch.allclose(seq_logps[0], expected)
    assert torch.allclose(mean_logps[0], expected / 2)


def test_dpo_loss_finite_and_prefers_better_policy_margin() -> None:
    result = dpo_loss(
        policy_chosen_logps=torch.tensor([5.0, 4.0]),
        policy_rejected_logps=torch.tensor([1.0, 0.5]),
        ref_chosen_logps=torch.tensor([2.0, 2.0]),
        ref_rejected_logps=torch.tensor([1.5, 1.0]),
        beta=0.1,
    )
    assert torch.isfinite(result["loss"])
    assert result["chosen_rewards"].shape == (2,)
    assert result["rejected_rewards"].shape == (2,)
    assert result["preference_accuracy"].mean().item() == 1.0
    assert result["reward_margin"].mean().item() > 0

```

## `tests/test_dpo_trainer_smoke.py`

```python
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.dpo_data import write_jsonl
from minillm.dpo_trainer import freeze_model, run_dpo
from minillm.tokenizer import MiniTokenizer


def setup_tiny_dpo_run(tmp_path: Path, lora: bool) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "User: Explain LoRA.\nAssistant: LoRA trains adapters.\n"
            "User: Compute 2 + 3.\nAssistant: 5.\n"
            "Chosen answers are better than rejected answers.\n"
        )
        * 120,
        encoding="utf-8",
    )
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    tokenizer_path = tmp_path / "tok.json"
    tokenizer.save(str(tokenizer_path))
    rows = [
        {
            "instruction": "Explain LoRA.",
            "input": "",
            "chosen": "LoRA trains small adapters.",
            "rejected": "It depends.",
            "category": "concept",
            "rejected_type": "vague",
            "reason": "too vague",
        },
        {
            "instruction": "Compute 2 + 3.",
            "input": "",
            "chosen": "5. Check: 2 + 3 = 5.",
            "rejected": "8. Check: 2 + 3 = 8.",
            "category": "math",
            "rejected_type": "wrong_answer",
            "reason": "wrong answer",
        },
    ] * 16
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(rows, str(train_path))
    write_jsonl(rows[:8], str(val_path))
    cfg = MiniLLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=48,
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        intermediate_size=64,
    )
    model = MiniLLMForCausalLM(cfg)
    base_ckpt = tmp_path / "sft_best.pt"
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
        "policy_checkpoint": str(base_ckpt),
        "reference_checkpoint": str(base_ckpt),
        "tokenizer_path": str(tokenizer_path),
        "train_data_path": str(train_path),
        "val_data_path": str(val_path),
        "output_dir": str(tmp_path / ("dpo_lora" if lora else "dpo_full")),
        "prefer_cuda": False,
        "dtype": "fp32",
        "max_length": 48,
        "beta": 0.1,
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
        "sample_prompts": ["Explain LoRA."],
    }
    if lora:
        config["lora"] = {"enabled": True, "r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
    config_path = tmp_path / ("dpo_lora.yaml" if lora else "dpo_full.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_freeze_model_marks_reference_params_non_trainable() -> None:
    model = MiniLLMForCausalLM(MiniLLMConfig())
    freeze_model(model)
    assert all(not p.requires_grad for p in model.parameters())


def test_full_dpo_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_dpo(str(setup_tiny_dpo_run(tmp_path, lora=False)))
    assert summary["mode"] == "full"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] == summary["parameter_count"]
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["metrics_path"]).exists()


def test_lora_dpo_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_dpo(str(setup_tiny_dpo_run(tmp_path, lora=True)))
    assert summary["mode"] == "lora"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] < summary["parameter_count"]
    assert summary["trainable_ratio"] < 0.1
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["output_dir"], "adapters", "best_adapter.pt").exists()

```

## `configs/dpo_full.yaml`

```yaml
seed: 20260710
policy_checkpoint: outputs/sft_full/checkpoints/best.pt
reference_checkpoint: outputs/sft_full/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
train_data_path: data/dpo/dpo_train.jsonl
val_data_path: data/dpo/dpo_val.jsonl
output_dir: outputs/dpo_full
prefer_cuda: true
dtype: auto
max_length: 128
beta: 0.1

training:
  batch_size: 8
  gradient_accumulation_steps: 1
  max_steps: 140
  eval_interval: 25
  save_interval: 50
  log_interval: 25
  eval_batches: 10
  learning_rate: 5.0e-5
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

## `configs/dpo_lora.yaml`

```yaml
seed: 20260710
policy_checkpoint: outputs/sft_full/checkpoints/best.pt
reference_checkpoint: outputs/sft_full/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
train_data_path: data/dpo/dpo_train.jsonl
val_data_path: data/dpo/dpo_val.jsonl
output_dir: outputs/dpo_lora
prefer_cuda: true
dtype: auto
max_length: 128
beta: 0.1

lora:
  enabled: true
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj

training:
  batch_size: 8
  gradient_accumulation_steps: 1
  max_steps: 140
  eval_interval: 25
  save_interval: 50
  log_interval: 25
  eval_batches: 10
  learning_rate: 3.0e-4
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
