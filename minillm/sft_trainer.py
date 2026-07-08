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
