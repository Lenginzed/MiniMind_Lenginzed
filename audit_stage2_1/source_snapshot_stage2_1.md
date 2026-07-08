# Stage 2.1 Source Snapshot

## `minillm/model.py`

```python
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import MiniLLMConfig
from .modules import DecoderBlock, RMSNorm


class MiniLLMForCausalLM(nn.Module):
    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input length exceeds config.context_length")

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in parameters)
```

## `minillm/data.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import MiniTokenizer
from .utils import ensure_dir, save_json


def encode_text_file(
    tokenizer: MiniTokenizer,
    input_path: str,
    add_special_tokens_per_line: bool = True,
) -> np.ndarray:
    token_ids = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            token_ids.extend(tokenizer.encode(line, add_special_tokens=add_special_tokens_per_line))
    if not token_ids:
        raise ValueError("input text produced no token ids")
    return np.asarray(token_ids, dtype=np.int32)


def split_tokens(
    tokens: np.ndarray,
    val_ratio: float = 0.1,
    min_val_tokens: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D array")
    if len(tokens) < 2:
        raise ValueError("need at least two tokens to split")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    val_count = max(min_val_tokens, int(len(tokens) * val_ratio))
    val_count = min(val_count, len(tokens) - 1)
    train = tokens[:-val_count]
    val = tokens[-val_count:]
    if len(train) == 0 or len(val) == 0:
        raise ValueError("train/val split produced an empty split")
    return train, val


def validate_token_ids(tokens: np.ndarray, vocab_size: int) -> None:
    if tokens.size == 0:
        raise ValueError("token array is empty")
    min_id = int(tokens.min())
    max_id = int(tokens.max())
    if min_id < 0:
        raise ValueError("token ids must be non-negative")
    if max_id >= vocab_size:
        raise ValueError(
            "token id %d exceeds vocab_size %d" % (max_id, vocab_size)
        )


def validate_block_size(block_size: int, context_length: int) -> None:
    if block_size <= 1:
        raise ValueError("block_size must be greater than 1")
    if block_size > context_length:
        raise ValueError(
            "block_size (%d) cannot exceed context_length (%d)"
            % (block_size, context_length)
        )


def split_random_blocks(
    tokens: np.ndarray,
    block_size: int,
    val_ratio: float = 0.1,
    seed: int = 1234,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if block_size <= 1:
        raise ValueError("block_size must be greater than 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    num_blocks = len(tokens) // block_size
    if num_blocks < 2:
        raise ValueError(
            "not enough tokens for random block split: need at least %d tokens, got %d"
            % (block_size * 2, len(tokens))
        )
    usable = tokens[: num_blocks * block_size].reshape(num_blocks, block_size)
    rng = np.random.default_rng(seed)
    indices = np.arange(num_blocks)
    rng.shuffle(indices)
    val_blocks = max(1, int(num_blocks * val_ratio))
    val_blocks = min(val_blocks, num_blocks - 1)
    val_idx = np.sort(indices[:val_blocks])
    train_idx = np.sort(indices[val_blocks:])
    train = usable[train_idx].reshape(-1)
    val = usable[val_idx].reshape(-1)
    return train, val, {
        "total_blocks": int(num_blocks),
        "train_blocks": int(len(train_idx)),
        "val_blocks": int(len(val_idx)),
        "dropped_tokens": int(len(tokens) - num_blocks * block_size),
    }


class CausalLMBlockDataset(Dataset):
    def __init__(self, tokens: Iterable[int], block_size: int) -> None:
        if block_size <= 1:
            raise ValueError("block_size must be greater than 1")
        array = np.asarray(list(tokens) if not isinstance(tokens, np.ndarray) else tokens, dtype=np.int64)
        if array.ndim != 1:
            raise ValueError("tokens must be 1D")
        self.tokens = array
        self.block_size = int(block_size)
        self.num_blocks = len(self.tokens) // self.block_size
        if self.num_blocks <= 0:
            raise ValueError("not enough tokens for one block")

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_blocks:
            raise IndexError(idx)
        start = idx * self.block_size
        block = self.tokens[start : start + self.block_size]
        input_ids = torch.tensor(block, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def save_tokenized_splits(
    tokenizer: MiniTokenizer,
    input_path: str,
    out_dir: str,
    block_size: int,
    val_ratio: float = 0.1,
    split_mode: str = "contiguous",
    seed: int = 1234,
) -> Dict[str, object]:
    ensure_dir(out_dir)
    tokens = encode_text_file(tokenizer, input_path)
    validate_token_ids(tokens, tokenizer.vocab_size)
    split_mode = split_mode.lower()
    split_extra: Dict[str, int] = {}
    if split_mode == "contiguous":
        train, val = split_tokens(tokens, val_ratio=val_ratio, min_val_tokens=block_size)
    elif split_mode == "random_blocks":
        train, val, split_extra = split_random_blocks(
            tokens,
            block_size=block_size,
            val_ratio=val_ratio,
            seed=seed,
        )
    else:
        raise ValueError("split_mode must be contiguous or random_blocks")
    train_samples = int(len(train) // block_size)
    val_samples = int(len(val) // block_size)
    if train_samples <= 0 or val_samples <= 0:
        raise ValueError(
            "not enough tokens for non-empty train/val samples with block_size=%d "
            "(train_tokens=%d, val_tokens=%d)" % (block_size, len(train), len(val))
        )
    train_path = str(Path(out_dir) / "train.npy")
    val_path = str(Path(out_dir) / "val.npy")
    np.save(train_path, train)
    np.save(val_path, val)
    metadata: Dict[str, object] = {
        "input_path": input_path,
        "train_path": train_path,
        "val_path": val_path,
        "total_tokens": int(len(tokens)),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "block_size": int(block_size),
        "train_samples": train_samples,
        "val_samples": val_samples,
        "val_ratio": float(val_ratio),
        "split_mode": split_mode,
        "seed": int(seed),
        "vocab_size": tokenizer.vocab_size,
    }
    metadata.update(split_extra)
    metadata.update(tokenizer.special_token_ids)
    save_json(metadata, str(Path(out_dir) / "metadata.json"))
    return metadata


def load_token_array(path: str) -> np.ndarray:
    return np.load(path)


def load_block_datasets(
    train_path: str,
    val_path: str,
    block_size: int,
) -> Tuple[CausalLMBlockDataset, CausalLMBlockDataset]:
    train = CausalLMBlockDataset(load_token_array(train_path), block_size)
    val = CausalLMBlockDataset(load_token_array(val_path), block_size)
    return train, val
```

## `minillm/trainer.py`

```python
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
    model_cfg = dict(config.get("model", {}))
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
```

## `minillm/utils.py`

```python
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(record: Dict[str, Any], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_perplexity(loss: Optional[float]) -> Optional[float]:
    if loss is None or not math.isfinite(loss):
        return None
    if loss > 20:
        return float("inf")
    return float(math.exp(loss))


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def file_size_bytes(path: str) -> int:
    return os.path.getsize(path)


def resolve_dtype(dtype_name: str) -> str:
    dtype_name = (dtype_name or "auto").lower()
    if dtype_name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16"
        if torch.cuda.is_available():
            return "fp16"
        return "fp32"
    if dtype_name not in {"bf16", "fp16", "fp32"}:
        raise ValueError("dtype must be one of auto, bf16, fp16, fp32")
    return dtype_name


def autocast_context(device: torch.device, dtype_name: str):
    dtype_name = resolve_dtype(dtype_name)
    if device.type != "cuda" or dtype_name == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
```

## `scripts/create_mixed_corpus.py`

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import random
from pathlib import Path


ENGLISH_PARAGRAPHS = [
    "A small language model is useful when the goal is to inspect every stage of the training pipeline.",
    "The engineer records inputs, outputs, metrics, checkpoints, and random seeds before changing the next variable.",
    "A decoder-only model reads tokens from left to right and learns to predict the next token under a causal mask.",
    "The aircraft changes heading slowly while the controller checks altitude, speed, and available energy.",
    "A stable experiment is easier to debug than a large experiment that fails silently after several hours.",
    "The validation split is not a leaderboard; it is a warning light for pipeline mistakes and overfitting.",
]

CHINESE_PARAGRAPHS = [
    "小规模预训练实验的目标不是获得强模型，而是验证数据、分词、模型、优化器和日志是否连贯。",
    "随机种子、配置文件、检查点和指标记录可以让实验更容易复现，也更容易被审计。",
    "因果语言模型在当前位置只能看到过去 token，不能读取未来 token。",
    "飞行器控制任务通常需要同时考虑姿态、速度、高度、能量和安全边界。",
    "强化学习实验需要关注奖励设计、策略更新、采样效率和训练稳定性。",
    "这个本地生成语料只用于 Stage 2.1 管线压测，不代表真实预训练数据质量。",
]

CODE_SNIPPETS = [
    "def clip_gradients(parameters, max_norm): return torch.nn.utils.clip_grad_norm_(parameters, max_norm)",
    "for step, batch in enumerate(loader): loss = model(batch['input_ids'], labels=batch['labels'])['loss']",
    "if scheduler_name == 'cosine': lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))",
    "tokens_seen += batch_size * context_length",
    "assert logits.shape == (batch, seq_len, vocab_size)",
]

MATH_LINES = [
    "If the cross entropy is 2.0, the perplexity is approximately exp(2.0).",
    "A warmup schedule increases the learning rate for the first few steps before decay begins.",
    "For block size 128 and batch size 16, one optimizer step observes 2048 token positions.",
    "Gradient accumulation simulates a larger batch by delaying the optimizer step.",
    "The cosine scheduler maps progress from zero to one and gradually reduces the learning rate.",
]

RL_LLM_LINES = [
    "Policy optimization compares actions, rewards, and trajectories, but this stage only performs pretraining.",
    "LoRA, DPO, and GRPO are intentionally deferred until the base training loop is reliable.",
    "A tokenizer maps text into integer ids, and the embedding table maps ids into vectors.",
    "RoPE changes query and key vectors using position-dependent rotations.",
    "Grouped query attention reduces key and value heads while keeping more query heads.",
]


def make_line(rng: random.Random, idx: int) -> str:
    choice = rng.randrange(8)
    if choice == 0:
        return rng.choice(ENGLISH_PARAGRAPHS)
    if choice == 1:
        return rng.choice(CHINESE_PARAGRAPHS)
    if choice == 2:
        return rng.choice(MATH_LINES)
    if choice == 3:
        return "Code note %05d: `%s`." % (idx, rng.choice(CODE_SNIPPETS))
    if choice == 4:
        return rng.choice(RL_LLM_LINES)
    if choice == 5:
        speed = rng.randint(180, 920)
        altitude = rng.randint(1000, 12000)
        return "Flight log %05d: speed=%d knots, altitude=%d meters, decision=hold course." % (
            idx,
            speed,
            altitude,
        )
    if choice == 6:
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        return "Math drill %05d: %d + %d = %d, and %d * %d = %d." % (
            idx,
            a,
            b,
            a + b,
            a,
            b,
            a * b,
        )
    terms = rng.sample(
        [
            "checkpoint",
            "optimizer",
            "scheduler",
            "tensorboard",
            "tokenizer",
            "causal mask",
            "validation loss",
            "gradient norm",
            "bf16",
            "random block split",
        ],
        3,
    )
    return "Experiment note %05d: inspect %s, %s, and %s before scaling." % (
        idx,
        terms[0],
        terms[1],
        terms[2],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local mixed corpus for Stage 2.1.")
    parser.add_argument("--output", default="data/raw/mixed_corpus.txt")
    parser.add_argument("--target-mb", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260708)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    target_bytes = int(args.target_mb * 1024 * 1024)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    total_bytes = 0
    idx = 0
    while total_bytes < target_bytes:
        line = make_line(rng, idx)
        lines.append(line)
        total_bytes += len((line + "\n").encode("utf-8"))
        idx += 1

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote:", out_path)
    print("lines:", len(lines))
    print("bytes:", out_path.stat().st_size)
    print("note: this mixed corpus is locally generated for pipeline hardening only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `scripts/tokenize_corpus.py`

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

from minillm.data import save_tokenized_splits
from minillm.tokenizer import MiniTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenize a text corpus into train/val npy files.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=["contiguous", "random_blocks"], default="contiguous")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tokenizer = MiniTokenizer.load(args.tokenizer)
    metadata = save_tokenized_splits(
        tokenizer,
        args.input,
        args.out_dir,
        block_size=args.block_size,
        val_ratio=args.val_ratio,
        split_mode=args.split_mode,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("metadata:", str(Path(args.out_dir) / "metadata.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `scripts/train_pretrain.py`

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

from minillm.trainer import run_pretrain


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tiny Causal LM pretrain smoke.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max_steps override.")
    args = parser.parse_args()
    summary = run_pretrain(args.config, resume_path=args.resume, max_steps_override=args.max_steps)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `scripts/plot_training_curves.py`

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Windows/Anaconda can load duplicate OpenMP runtimes when matplotlib imports
# numerical backends. This plot-only process does not run training kernels.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillm.utils import ensure_dir, iter_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot train/eval loss curves from metrics.jsonl.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [row for row in iter_jsonl(args.metrics) if row.get("train_loss") is not None]
    if not rows:
        raise ValueError("metrics file is empty")
    steps = [row["step"] for row in rows]
    train_loss = [row.get("train_loss") for row in rows]
    eval_steps = [row["step"] for row in rows if row.get("eval_loss") is not None]
    eval_loss = [row.get("eval_loss") for row in rows if row.get("eval_loss") is not None]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, label="train loss", linewidth=1.5)
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval loss", marker="o", linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Tiny Pretrain Smoke Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    ensure_dir(str(Path(args.out).parent))
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("wrote:", args.out)
    print("points:", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `configs/pretrain_stage2_hardened.yaml`

```yaml
seed: 20260708
output_dir: outputs/pretrain_stage2_hardened
tokenizer_path: data/tokenizers/mixed_tokenizer.json
train_data_path: data/processed_mixed/train.npy
val_data_path: data/processed_mixed/val.npy
prefer_cuda: true
dtype: auto

data:
  block_size: 128

model:
  vocab_size: 2000
  context_length: 128
  n_layer: 3
  n_embd: 192
  n_head: 6
  n_kv_head: 2
  intermediate_size: 512
  rms_norm_eps: 1.0e-6
  rope_theta: 10000.0
  dropout: 0.0
  tie_word_embeddings: true
  use_gradient_checkpointing: true

training:
  batch_size: 16
  gradient_accumulation_steps: 1
  max_steps: 220
  eval_interval: 25
  save_interval: 50
  log_interval: 25
  eval_batches: 10
  learning_rate: 3.0e-4
  weight_decay: 0.1
  grad_clip: 1.0
  scheduler: cosine
  warmup_steps: 20

sample_prompts:
  - "Mini language models"
  - "The scheduler"
  - "小规模预训练"
  - "Flight log"
```

## `configs/pretrain_resume_smoke.yaml`

```yaml
seed: 20260708
output_dir: outputs/pretrain_resume_smoke
tokenizer_path: data/tokenizers/mixed_tokenizer.json
train_data_path: data/processed_mixed/train.npy
val_data_path: data/processed_mixed/val.npy
prefer_cuda: true
dtype: auto

data:
  block_size: 64

model:
  vocab_size: 2000
  context_length: 64
  n_layer: 2
  n_embd: 128
  n_head: 4
  n_kv_head: 2
  intermediate_size: 256
  rms_norm_eps: 1.0e-6
  rope_theta: 10000.0
  dropout: 0.0
  tie_word_embeddings: true
  use_gradient_checkpointing: false

training:
  batch_size: 16
  gradient_accumulation_steps: 1
  max_steps: 30
  eval_interval: 10
  save_interval: 10
  log_interval: 10
  eval_batches: 4
  learning_rate: 3.0e-4
  weight_decay: 0.1
  grad_clip: 1.0
  scheduler: linear
  warmup_steps: 5

sample_prompts:
  - "Mini language models"
  - "小规模预训练"
```

## `tests/test_gradient_checkpointing.py`

```python
from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM


def run_one_backward(use_checkpointing: bool) -> None:
    torch.manual_seed(11)
    config = MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
        use_gradient_checkpointing=use_checkpointing,
    )
    model = MiniLLMForCausalLM(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    outputs = model(input_ids, labels=input_ids.clone())
    loss = outputs["loss"]
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_gradient_checkpointing_true_forward_backward() -> None:
    run_one_backward(True)


def test_gradient_checkpointing_false_forward_backward() -> None:
    run_one_backward(False)
```

## `tests/test_resume_training.py`

```python
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from minillm.data import save_tokenized_splits
from minillm.tokenizer import MiniTokenizer
from minillm.trainer import run_pretrain


def test_resume_training_step_increases_and_checkpoint_loads(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("mini resume smoke test 小模型\n" * 200), encoding="utf-8")
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    processed = tmp_path / "processed"
    save_tokenized_splits(
        tokenizer,
        str(corpus),
        str(processed),
        block_size=16,
        val_ratio=0.1,
        split_mode="random_blocks",
        seed=123,
    )

    config = {
        "seed": 123,
        "output_dir": str(tmp_path / "out"),
        "tokenizer_path": str(tokenizer_path),
        "train_data_path": str(processed / "train.npy"),
        "val_data_path": str(processed / "val.npy"),
        "prefer_cuda": False,
        "dtype": "fp32",
        "data": {"block_size": 16},
        "model": {
            "vocab_size": 128,
            "context_length": 16,
            "n_layer": 1,
            "n_embd": 32,
            "n_head": 4,
            "n_kv_head": 2,
            "intermediate_size": 64,
            "dropout": 0.0,
            "use_gradient_checkpointing": False,
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
            "eval_interval": 1,
            "save_interval": 1,
            "log_interval": 1,
            "eval_batches": 1,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "scheduler": "linear",
            "warmup_steps": 1,
        },
        "sample_prompts": ["mini"],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    first = run_pretrain(str(config_path))
    assert first["final_step"] == 2
    checkpoint = Path(first["last_checkpoint"])
    assert checkpoint.exists()

    second = run_pretrain(str(config_path), resume_path=str(checkpoint), max_steps_override=4)
    assert second["resume_loaded"] is True
    assert second["start_step"] == 2
    assert second["final_step"] == 4
    loaded = torch.load(second["last_checkpoint"], map_location="cpu")
    assert loaded["step"] == 4
    assert "optimizer_state_dict" in loaded
```

## `tests/test_scheduler.py`

```python
from __future__ import annotations

import torch

from minillm.trainer import build_lr_scheduler


def test_cosine_scheduler_warmup_then_decay() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=0.1)
    scheduler = build_lr_scheduler(optimizer, scheduler_name="cosine", warmup_steps=2, max_steps=6)
    assert scheduler is not None
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[0] < lrs[1]
    assert max(lrs) <= 0.1
    assert lrs[-1] < lrs[2]


def test_none_scheduler_returns_none() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=0.1)
    assert build_lr_scheduler(optimizer, scheduler_name="none", warmup_steps=0, max_steps=10) is None
```

## `tests/test_data_pipeline.py`

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from minillm.data import (
    CausalLMBlockDataset,
    save_tokenized_splits,
    split_random_blocks,
    split_tokens,
    validate_block_size,
)
from minillm.tokenizer import MiniTokenizer


def build_tokenizer(tmp_path: Path) -> MiniTokenizer:
    corpus = tmp_path / "toy.txt"
    corpus.write_text(("hello world 小模型\n" * 100), encoding="utf-8")
    return MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)


def test_block_dataset_shape_and_labels() -> None:
    tokens = np.arange(64, dtype=np.int32)
    dataset = CausalLMBlockDataset(tokens, block_size=16)
    item = dataset[0]
    assert len(dataset) == 4
    assert item["input_ids"].shape == (16,)
    assert item["labels"].shape == (16,)
    assert torch.equal(item["input_ids"], item["labels"])


def test_train_val_split_not_empty() -> None:
    tokens = np.arange(100, dtype=np.int32)
    train, val = split_tokens(tokens, val_ratio=0.2)
    assert len(train) == 80
    assert len(val) == 20


def test_save_tokenized_splits_metadata(tmp_path: Path) -> None:
    corpus = tmp_path / "toy.txt"
    corpus.write_text(("hello world 小模型\n" * 100), encoding="utf-8")
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    out_dir = tmp_path / "processed"
    metadata = save_tokenized_splits(
        tokenizer,
        str(corpus),
        str(out_dir),
        block_size=8,
        val_ratio=0.1,
    )
    assert (out_dir / "train.npy").exists()
    assert (out_dir / "val.npy").exists()
    assert (out_dir / "metadata.json").exists()
    assert metadata["train_tokens"] > 0
    assert metadata["val_tokens"] > 0
    assert metadata["train_samples"] > 0
    assert metadata["val_samples"] > 0


def test_random_blocks_split_reproducible() -> None:
    tokens = np.arange(128, dtype=np.int32)
    train_a, val_a, meta_a = split_random_blocks(tokens, block_size=8, val_ratio=0.25, seed=42)
    train_b, val_b, meta_b = split_random_blocks(tokens, block_size=8, val_ratio=0.25, seed=42)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert meta_a == meta_b
    assert meta_a["train_blocks"] > 0
    assert meta_a["val_blocks"] > 0


def test_random_blocks_too_small_raises() -> None:
    try:
        split_random_blocks(np.arange(10, dtype=np.int32), block_size=8, val_ratio=0.1)
    except ValueError as exc:
        assert "not enough tokens" in str(exc)
        return
    raise AssertionError("split_random_blocks should reject tiny inputs")


def test_block_size_larger_than_context_raises() -> None:
    try:
        validate_block_size(block_size=128, context_length=64)
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
        return
    raise AssertionError("validate_block_size should reject block_size > context_length")
```
