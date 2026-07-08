from __future__ import annotations

import csv
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import MiniLLMConfig
from .data import load_block_datasets, validate_block_size
from .generation import generate
from .model import MiniLLMForCausalLM, count_parameters
from .tokenizer import MiniTokenizer
from .utils import (
    append_jsonl,
    autocast_context,
    ensure_dir,
    get_device,
    iter_jsonl,
    load_yaml,
    resolve_dtype,
    safe_perplexity,
    save_json,
    save_yaml,
    set_seed,
)


def build_model_config(config: Dict[str, Any], tokenizer: MiniTokenizer) -> MiniLLMConfig:
    model_cfg: Dict[str, Any] = {}
    model_config_path = config.get("model_config") or config.get("model_config_path")
    if model_config_path:
        loaded = load_yaml(str(model_config_path))
        model_cfg.update(loaded.get("model", loaded))
    model_cfg.update(dict(config.get("model", {})))
    model_cfg["vocab_size"] = int(tokenizer.vocab_size)
    return MiniLLMConfig(**model_cfg)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    warmup_steps: int,
    max_steps: int,
) -> Optional[LambdaLR]:
    scheduler_name = (scheduler_name or "none").lower()
    warmup_steps = max(0, int(warmup_steps))
    max_steps = max(1, int(max_steps))

    if scheduler_name == "none":
        return None

    def lr_lambda(step_index: int) -> float:
        return lr_scale_for_step(step_index, scheduler_name, warmup_steps, max_steps)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def lr_scale_for_step(
    step_index: int,
    scheduler_name: str,
    warmup_steps: int,
    max_steps: int,
) -> float:
    scheduler_name = (scheduler_name or "none").lower()
    if scheduler_name == "none":
        return 1.0
    if warmup_steps > 0 and step_index < warmup_steps:
        return max(1.0e-8, float(step_index + 1) / float(warmup_steps))
    progress_den = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step_index - warmup_steps) / progress_den))
    if scheduler_name == "linear":
        return max(0.0, 1.0 - progress)
    if scheduler_name == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError("scheduler must be one of none, cosine, linear")


def reset_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    scale: float,
) -> None:
    for group in optimizer.param_groups:
        group["initial_lr"] = base_lr
        group["lr"] = base_lr * scale


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def cuda_memory_record(device: torch.device) -> Dict[str, Optional[int]]:
    if device.type != "cuda":
        return {
            "cuda_memory_allocated": None,
            "cuda_memory_reserved": None,
            "cuda_max_memory_allocated": None,
            "cuda_max_memory_reserved": None,
        }
    torch.cuda.synchronize()
    return {
        "cuda_memory_allocated": int(torch.cuda.memory_allocated(device)),
        "cuda_memory_reserved": int(torch.cuda.memory_reserved(device)),
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


@torch.no_grad()
def evaluate(
    model: MiniLLMForCausalLM,
    loader: DataLoader,
    device: torch.device,
    dtype_name: str,
    max_batches: int,
) -> float:
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
        if loss is not None:
            losses.append(float(loss.detach().cpu().item()))
    if was_training:
        model.train()
    if not losses:
        return float("nan")
    return float(sum(losses) / len(losses))


def save_checkpoint(
    path: str,
    model: MiniLLMForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    step: int,
    config: Dict[str, Any],
    best_eval_loss: float,
    tokens_seen: int,
) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "step": int(step),
            "config": config,
            "model_config": asdict(model.config),
            "best_eval_loss": float(best_eval_loss),
            "tokens_seen": int(tokens_seen),
        },
        path,
    )


def write_sample(
    model: MiniLLMForCausalLM,
    tokenizer: MiniTokenizer,
    prompts,
    path: str,
    device: torch.device,
    max_new_tokens: int = 32,
) -> None:
    ensure_dir(str(Path(path).parent))
    was_training = model.training
    model.eval()
    lines = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        bos_id = tokenizer.special_token_ids.get("bos_token_id")
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
            eos_token_id=tokenizer.special_token_ids.get("eos_token_id"),
            do_sample=True,
        )
        text = tokenizer.decode(out[0].detach().cpu().tolist(), skip_special_tokens=True)
        lines.append("PROMPT: %s\nOUTPUT: %s\n" % (prompt, text))
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    if was_training:
        model.train()


def _existing_first_train_loss(metrics_path: str) -> Optional[float]:
    path = Path(metrics_path)
    if not path.exists():
        return None
    for row in iter_jsonl(metrics_path):
        if row.get("train_loss") is not None:
            return float(row["train_loss"])
    return None


def run_pretrain(
    config_path: str,
    resume_path: Optional[str] = None,
    max_steps_override: Optional[int] = None,
) -> Dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 1234))
    set_seed(seed)

    if max_steps_override is not None:
        config.setdefault("training", {})
        config["training"]["max_steps"] = int(max_steps_override)

    output_dir = config.get("output_dir", "outputs/pretrain_tiny")
    ensure_dir(output_dir)
    ensure_dir(str(Path(output_dir) / "checkpoints"))
    ensure_dir(str(Path(output_dir) / "logs"))
    ensure_dir(str(Path(output_dir) / "plots"))
    ensure_dir(str(Path(output_dir) / "samples"))

    tokenizer = MiniTokenizer.load(config["tokenizer_path"])
    model_config = build_model_config(config, tokenizer)
    config.setdefault("model", {})
    config["model"].update(asdict(model_config))

    block_size = int(config["data"]["block_size"])
    validate_block_size(block_size, model_config.context_length)

    device = get_device(bool(config.get("prefer_cuda", True)))
    dtype_name = resolve_dtype(str(config.get("dtype", "auto")))
    if device.type != "cuda":
        dtype_name = "fp32"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_ds, val_ds = load_block_datasets(
        config["train_data_path"],
        config["val_data_path"],
        block_size,
    )
    batch_size = int(config["training"]["batch_size"])
    if len(train_ds) < batch_size:
        raise ValueError(
            "train dataset has fewer samples (%d) than batch_size (%d)"
            % (len(train_ds), batch_size)
        )
    if len(val_ds) == 0:
        raise ValueError("validation dataset has zero samples")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = MiniLLMForCausalLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
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

    start_step = 0
    tokens_seen = 0
    best_eval_loss = float("inf")
    resume_loaded = False
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", start_step * batch_size * block_size))
        best_eval_loss = float(checkpoint.get("best_eval_loss", float("inf")))
        ckpt_max_steps = int(
            checkpoint.get("config", {})
            .get("training", {})
            .get("max_steps", max_steps)
        )
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None and ckpt_max_steps == max_steps:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif scheduler is not None:
            scheduler_name = str(config["training"].get("scheduler", "none"))
            warmup_steps = int(config["training"].get("warmup_steps", 0))
            base_lr = float(config["training"]["learning_rate"])
            reset_optimizer_lr(
                optimizer,
                base_lr,
                lr_scale_for_step(start_step, scheduler_name, warmup_steps, max_steps),
            )
            scheduler.last_epoch = start_step
        resume_loaded = True

    metrics_path = str(Path(output_dir) / "metrics.jsonl")
    csv_path = str(Path(output_dir) / "metrics.csv")
    if resume_path is None:
        for path in [metrics_path, csv_path]:
            if Path(path).exists():
                Path(path).unlink()
    else:
        append_jsonl(
            {
                "event": "resume",
                "step": start_step,
                "resume_from": resume_path,
                "tokens_seen": tokens_seen,
            },
            metrics_path,
        )

    resolved_config_path = str(Path(output_dir) / "train_config_resolved.yaml")
    save_yaml(config, resolved_config_path)
    save_json(
        {
            "parameter_count": count_parameters(model),
            "device": str(device),
            "dtype": dtype_name,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "scheduler": str(config["training"].get("scheduler", "none")),
            "warmup_steps": int(config["training"].get("warmup_steps", 0)),
            "gradient_checkpointing": bool(model_config.use_gradient_checkpointing),
            "resume_loaded": resume_loaded,
        },
        str(Path(output_dir) / "run_summary.json"),
    )

    writer = SummaryWriter(log_dir=str(Path(output_dir) / "logs"))
    prompts = config.get("sample_prompts", ["Once upon a time", "Mini language models"])
    if resume_path is None:
        write_sample(model, tokenizer, prompts, str(Path(output_dir) / "samples" / "before.txt"), device)

    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    eval_interval = int(config["training"].get("eval_interval", 20))
    save_interval = int(config["training"].get("save_interval", 50))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    eval_batches = int(config["training"].get("eval_batches", 10))
    mid_sample_written = Path(output_dir, "samples", "mid.txt").exists()

    train_iter = cycle_loader(train_loader)
    last_train_loss: Optional[float] = None
    first_train_loss: Optional[float] = _existing_first_train_loss(metrics_path)
    last_eval_loss: Optional[float] = None

    csv_exists = Path(csv_path).exists() and resume_path is not None
    csv_file = open(csv_path, "a" if resume_path is not None else "w", encoding="utf-8", newline="")
    csv_fields = [
        "event",
        "step",
        "train_loss",
        "eval_loss",
        "train_ppl",
        "eval_ppl",
        "lr",
        "grad_norm",
        "tokens_seen",
        "approximate_epoch",
        "cuda_memory_allocated",
        "cuda_memory_reserved",
        "cuda_max_memory_allocated",
        "cuda_max_memory_reserved",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not csv_exists:
        csv_writer.writeheader()

    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step + 1, max_steps + 1):
            accum_losses = []
            for _ in range(grad_accum):
                batch = move_batch(next(train_iter), device)
                tokens_seen += int(batch["input_ids"].numel())
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
                accum_losses.append(float(loss.detach().cpu().item()))

            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss = float(sum(accum_losses) / len(accum_losses))
            if first_train_loss is None:
                first_train_loss = train_loss
            last_train_loss = train_loss
            eval_loss: Optional[float] = None
            if step == 1 or step % eval_interval == 0 or step == max_steps:
                eval_loss = evaluate(model, val_loader, device, dtype_name, eval_batches)
                last_eval_loss = eval_loss
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(
                        str(Path(output_dir) / "checkpoints" / "best.pt"),
                        model,
                        optimizer,
                        scheduler,
                        step,
                        config,
                        best_eval_loss,
                        tokens_seen,
                    )

            if step % save_interval == 0 or step == max_steps:
                save_checkpoint(
                    str(Path(output_dir) / "checkpoints" / "last.pt"),
                    model,
                    optimizer,
                    scheduler,
                    step,
                    config,
                    best_eval_loss,
                    tokens_seen,
                )
            if (not mid_sample_written) and step >= max(1, max_steps // 2):
                write_sample(model, tokenizer, prompts, str(Path(output_dir) / "samples" / "mid.txt"), device)
                mid_sample_written = True

            approximate_epoch = float(tokens_seen) / float(max(1, len(train_ds) * block_size))
            mem = cuda_memory_record(device)
            record = {
                "event": "train_step",
                "step": step,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "train_ppl": safe_perplexity(train_loss),
                "eval_ppl": safe_perplexity(eval_loss),
                "lr": current_lr(optimizer),
                "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                "tokens_seen": int(tokens_seen),
                "approximate_epoch": approximate_epoch,
            }
            record.update(mem)
            append_jsonl(record, metrics_path)
            csv_writer.writerow(record)
            csv_file.flush()
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("lr", record["lr"], step)
            writer.add_scalar("grad_norm", record["grad_norm"], step)
            writer.add_scalar("tokens_seen", tokens_seen, step)
            writer.add_scalar("ppl/train", record["train_ppl"] or 0.0, step)
            if eval_loss is not None:
                writer.add_scalar("loss/eval", eval_loss, step)
                writer.add_scalar("ppl/eval", record["eval_ppl"] or 0.0, step)
            if step == 1 or step % int(config["training"].get("log_interval", 10)) == 0 or step == max_steps:
                print(
                    "step=%d train_loss=%.4f eval_loss=%s lr=%.6g tokens_seen=%d"
                    % (
                        step,
                        train_loss,
                        "%.4f" % eval_loss if eval_loss is not None else "None",
                        record["lr"],
                        tokens_seen,
                    )
                )
    finally:
        csv_file.close()
        writer.close()

    last_path = str(Path(output_dir) / "checkpoints" / "last.pt")
    if not Path(last_path).exists():
        save_checkpoint(last_path, model, optimizer, scheduler, max_steps, config, best_eval_loss, tokens_seen)
    best_path = str(Path(output_dir) / "checkpoints" / "best.pt")
    if not Path(best_path).exists():
        shutil.copyfile(last_path, best_path)

    write_sample(model, tokenizer, prompts, str(Path(output_dir) / "samples" / "after.txt"), device)
    summary = {
        "output_dir": output_dir,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "dtype": dtype_name,
        "max_steps": max_steps,
        "start_step": start_step,
        "final_step": max_steps,
        "resume_loaded": resume_loaded,
        "first_train_loss": first_train_loss,
        "last_train_loss": last_train_loss,
        "last_eval_loss": last_eval_loss,
        "best_eval_loss": best_eval_loss,
        "best_eval_ppl": safe_perplexity(best_eval_loss),
        "tokens_seen": tokens_seen,
        "scheduler": str(config["training"].get("scheduler", "none")),
        "warmup_steps": int(config["training"].get("warmup_steps", 0)),
        "gradient_checkpointing": bool(model_config.use_gradient_checkpointing),
        "metrics_path": metrics_path,
        "last_checkpoint": last_path,
        "best_checkpoint": best_path,
        "before_samples": str(Path(output_dir) / "samples" / "before.txt"),
        "after_samples": str(Path(output_dir) / "samples" / "after.txt"),
    }
    summary.update(cuda_memory_record(device))
    save_json(summary, str(Path(output_dir) / "train_summary.json"))
    return summary
