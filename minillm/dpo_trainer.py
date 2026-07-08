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
