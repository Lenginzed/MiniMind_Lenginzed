# Stage 2 Source Snapshot

## `minillm/tokenizer.py`

```python
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def discover_text_files(inputs: Iterable[str]) -> List[str]:
    files: List[str] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(str(p) for p in sorted(path.rglob("*.txt")))
        else:
            matched = sorted(glob.glob(str(path)))
            files.extend(matched if matched else [str(path)])
    unique = []
    seen = set()
    for file_path in files:
        if file_path not in seen:
            unique.append(file_path)
            seen.add(file_path)
    if not unique:
        raise ValueError("no text files found")
    for file_path in unique:
        if not Path(file_path).exists():
            raise FileNotFoundError(file_path)
    return unique


class MiniTokenizer:
    """Small Byte-level BPE wrapper for Stage 2 data pipeline smoke tests."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def train_from_files(
        cls,
        files: Iterable[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
    ) -> "MiniTokenizer":
        text_files = discover_text_files(files)
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train(text_files, trainer=trainer)
        bos_id = tokenizer.token_to_id(BOS_TOKEN)
        eos_id = tokenizer.token_to_id(EOS_TOKEN)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="%s $A %s" % (BOS_TOKEN, EOS_TOKEN),
            pair="%s $A %s $B %s" % (BOS_TOKEN, EOS_TOKEN, EOS_TOKEN),
            special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
        )
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str) -> "MiniTokenizer":
        return cls(Tokenizer.from_file(path))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token)

    @property
    def special_token_ids(self) -> Dict[str, Optional[int]]:
        return {
            "pad_token_id": self.token_to_id(PAD_TOKEN),
            "bos_token_id": self.token_to_id(BOS_TOKEN),
            "eos_token_id": self.token_to_id(EOS_TOKEN),
            "unk_token_id": self.token_to_id(UNK_TOKEN),
        }

    def summary(self) -> Dict[str, object]:
        data: Dict[str, object] = {"vocab_size": self.vocab_size}
        data.update(self.special_token_ids)
        return data
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
) -> Dict[str, object]:
    ensure_dir(out_dir)
    tokens = encode_text_file(tokenizer, input_path)
    train, val = split_tokens(tokens, val_ratio=val_ratio, min_val_tokens=block_size)
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
        "train_samples": int(len(train) // block_size),
        "val_samples": int(len(val) // block_size),
        "val_ratio": float(val_ratio),
        "vocab_size": tokenizer.vocab_size,
    }
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
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import MiniLLMConfig
from .data import load_block_datasets
from .generation import generate
from .model import MiniLLMForCausalLM, count_parameters
from .tokenizer import MiniTokenizer
from .utils import (
    append_jsonl,
    autocast_context,
    ensure_dir,
    get_device,
    load_json,
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


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: MiniLLMForCausalLM,
    loader: DataLoader,
    device: torch.device,
    dtype_name: str,
    max_batches: int,
) -> float:
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
    model.train()
    if not losses:
        return float("nan")
    return float(sum(losses) / len(losses))


def save_checkpoint(
    path: str,
    model: MiniLLMForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Dict[str, Any],
    best_eval_loss: float,
) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "config": config,
            "model_config": asdict(model.config),
            "best_eval_loss": float(best_eval_loss),
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
    model.train()


def run_pretrain(config_path: str) -> Dict[str, Any]:
    config = load_yaml(config_path)
    seed = int(config.get("seed", 1234))
    set_seed(seed)

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

    device = get_device(bool(config.get("prefer_cuda", True)))
    dtype_name = resolve_dtype(str(config.get("dtype", "auto")))
    if device.type != "cuda":
        dtype_name = "fp32"

    train_ds, val_ds = load_block_datasets(
        config["train_data_path"],
        config["val_data_path"],
        int(config["data"]["block_size"]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
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
    use_scaler = device.type == "cuda" and dtype_name == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    metrics_path = str(Path(output_dir) / "metrics.jsonl")
    csv_path = str(Path(output_dir) / "metrics.csv")
    for path in [metrics_path, csv_path]:
        if Path(path).exists():
            Path(path).unlink()

    resolved_config_path = str(Path(output_dir) / "train_config_resolved.yaml")
    save_yaml(config, resolved_config_path)
    save_json(
        {
            "parameter_count": count_parameters(model),
            "device": str(device),
            "dtype": dtype_name,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        },
        str(Path(output_dir) / "run_summary.json"),
    )

    writer = SummaryWriter(log_dir=str(Path(output_dir) / "logs"))
    prompts = config.get("sample_prompts", ["Once upon a time", "Mini language models"])
    write_sample(model, tokenizer, prompts, str(Path(output_dir) / "samples" / "before.txt"), device)

    max_steps = int(config["training"]["max_steps"])
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    eval_interval = int(config["training"].get("eval_interval", 20))
    save_interval = int(config["training"].get("save_interval", 50))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    eval_batches = int(config["training"].get("eval_batches", 10))

    train_iter = cycle_loader(train_loader)
    best_eval_loss = float("inf")
    last_train_loss: Optional[float] = None
    first_train_loss: Optional[float] = None

    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=["step", "train_loss", "eval_loss", "train_ppl", "eval_ppl", "lr"],
    )
    csv_writer.writeheader()

    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step in range(1, max_steps + 1):
            accum_losses = []
            for _ in range(grad_accum):
                batch = move_batch(next(train_iter), device)
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
            optimizer.zero_grad(set_to_none=True)

            train_loss = float(sum(accum_losses) / len(accum_losses))
            if first_train_loss is None:
                first_train_loss = train_loss
            last_train_loss = train_loss
            eval_loss: Optional[float] = None
            if step == 1 or step % eval_interval == 0 or step == max_steps:
                eval_loss = evaluate(model, val_loader, device, dtype_name, eval_batches)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_checkpoint(
                        str(Path(output_dir) / "checkpoints" / "best.pt"),
                        model,
                        optimizer,
                        step,
                        config,
                        best_eval_loss,
                    )

            if step % save_interval == 0 or step == max_steps:
                save_checkpoint(
                    str(Path(output_dir) / "checkpoints" / "last.pt"),
                    model,
                    optimizer,
                    step,
                    config,
                    best_eval_loss,
                )

            record = {
                "step": step,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "train_ppl": safe_perplexity(train_loss),
                "eval_ppl": safe_perplexity(eval_loss),
                "lr": float(config["training"]["learning_rate"]),
                "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            }
            append_jsonl(record, metrics_path)
            csv_writer.writerow(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "train_ppl": record["train_ppl"],
                    "eval_ppl": record["eval_ppl"],
                    "lr": record["lr"],
                }
            )
            csv_file.flush()
            writer.add_scalar("loss/train", train_loss, step)
            writer.add_scalar("ppl/train", record["train_ppl"] or 0.0, step)
            if eval_loss is not None:
                writer.add_scalar("loss/eval", eval_loss, step)
                writer.add_scalar("ppl/eval", record["eval_ppl"] or 0.0, step)
            if step == 1 or step % int(config["training"].get("log_interval", 10)) == 0:
                print(
                    "step=%d train_loss=%.4f eval_loss=%s train_ppl=%s"
                    % (
                        step,
                        train_loss,
                        "%.4f" % eval_loss if eval_loss is not None else "None",
                        "%.2f" % record["train_ppl"] if record["train_ppl"] else "None",
                    )
                )
    finally:
        csv_file.close()
        writer.close()

    last_path = str(Path(output_dir) / "checkpoints" / "last.pt")
    if not Path(last_path).exists():
        save_checkpoint(last_path, model, optimizer, max_steps, config, best_eval_loss)
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
        "first_train_loss": first_train_loss,
        "last_train_loss": last_train_loss,
        "best_eval_loss": best_eval_loss,
        "best_eval_ppl": safe_perplexity(best_eval_loss),
        "metrics_path": metrics_path,
        "last_checkpoint": last_path,
        "best_checkpoint": best_path,
        "before_samples": str(Path(output_dir) / "samples" / "before.txt"),
        "after_samples": str(Path(output_dir) / "samples" / "after.txt"),
    }
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

## `scripts/create_toy_corpus.py`

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import random
from pathlib import Path


def build_lines(num_lines: int, seed: int):
    rng = random.Random(seed)
    stories = [
        "A tiny robot reads a map and explains each step clearly.",
        "The pilot keeps altitude, checks fuel, and chooses a safe heading.",
        "A student trains a mini language model to learn the full pipeline.",
        "The dataset is small, but the engineering loop is complete.",
        "Gradient descent updates weights after the loss is computed.",
        "A decoder-only transformer predicts the next token in a sequence.",
        "RoPE rotates query and key vectors according to token position.",
        "Grouped query attention shares key and value heads across queries.",
        "RMSNorm stabilizes hidden states before attention and the MLP.",
        "SwiGLU uses a gate and an up projection before the down projection.",
    ]
    bilingual = [
        "小模型也可以帮助我们理解大模型训练流程。",
        "数据管线、分词器、模型结构和训练循环需要逐步验证。",
        "这个 toy corpus 只用于 smoke test，不代表真实训练数据。",
        "空战仿真中，智能体需要观察、规划、行动和复盘。",
        "强化学习关注奖励、策略、轨迹和稳定优化。",
    ]
    math = [
        "If x plus y equals ten, and x is four, then y is six.",
        "The loss curve should generally move down during a tiny smoke run.",
        "Perplexity is exponential of cross entropy, so large loss can overflow.",
        "A batch contains several blocks, and each block has the same context length.",
    ]
    tech_terms = [
        "tokenizer", "causal mask", "checkpoint", "tensorboard", "bf16",
        "optimizer", "AdamW", "gradient clipping", "validation split", "sampling",
    ]
    lines = []
    for idx in range(num_lines):
        template_type = idx % 6
        if template_type == 0:
            line = rng.choice(stories)
        elif template_type == 1:
            line = rng.choice(bilingual)
        elif template_type == 2:
            line = rng.choice(math)
        elif template_type == 3:
            term_a, term_b = rng.sample(tech_terms, 2)
            line = "In experiment %04d, %s is checked before %s." % (idx, term_a, term_b)
        elif template_type == 4:
            line = "Question: what does the mini model learn? Answer: it learns repeatable pipeline mechanics."
        else:
            line = "训练日志记录 step、train loss、eval loss 和 perplexity，方便后续审计。"
        lines.append(line)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local toy corpus for Stage 2 smoke tests.")
    parser.add_argument("--output", default="data/raw/toy_corpus.txt")
    parser.add_argument("--lines", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    lines = build_lines(args.lines, args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote:", out_path)
    print("lines:", len(lines))
    print("bytes:", out_path.stat().st_size)
    print("note: this toy corpus is only for pipeline validation, not model quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `scripts/train_tokenizer.py`

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

from minillm.tokenizer import MiniTokenizer, discover_text_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Byte-level BPE tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Text file(s), glob(s), or directories.")
    parser.add_argument("--output", required=True, help="Output tokenizer JSON path.")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    files = discover_text_files(args.input)
    tokenizer = MiniTokenizer.train_from_files(
        files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.save(args.output)
    sample = "Mini LLM smoke test: 小模型检查 tokenizer encode and decode."
    sample_ids = tokenizer.encode(sample)
    sample_text = tokenizer.decode(sample_ids)
    sample_path = str(Path(args.output).with_suffix(".sample.json"))
    Path(sample_path).write_text(
        json.dumps(
            {
                "sample": sample,
                "ids": sample_ids,
                "decoded": sample_text,
                "summary": tokenizer.summary(),
                "files": files,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("tokenizer:", args.output)
    print("vocab_size:", tokenizer.vocab_size)
    print("special_token_ids:", tokenizer.special_token_ids)
    print("sample_ids:", sample_ids[:32])
    print("sample_decoded:", sample_text)
    print("sample_file:", sample_path)
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
    args = parser.parse_args()

    tokenizer = MiniTokenizer.load(args.tokenizer)
    metadata = save_tokenized_splits(
        tokenizer,
        args.input,
        args.out_dir,
        block_size=args.block_size,
        val_ratio=args.val_ratio,
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
    args = parser.parse_args()
    summary = run_pretrain(args.config)
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

    rows = list(iter_jsonl(args.metrics))
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

## `scripts/eval_pretrain_smoke.py`

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
```

## `tests/test_tokenizer_pipeline.py`

```python
from __future__ import annotations

from pathlib import Path

from minillm.tokenizer import MiniTokenizer


def test_toy_tokenizer_train_load_encode_decode(tmp_path: Path) -> None:
    corpus = tmp_path / "toy.txt"
    corpus.write_text(
        "\n".join(
            [
                "Mini language models test tokenizers.",
                "小模型 tokenizer smoke test.",
                "RoPE and GQA are transformer components.",
            ]
            * 20
        ),
        encoding="utf-8",
    )
    out = tmp_path / "tok.json"
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    tokenizer.save(str(out))
    loaded = MiniTokenizer.load(str(out))
    ids = loaded.encode("Mini tokenizer 小模型")
    decoded = loaded.decode(ids)
    assert out.exists()
    assert loaded.vocab_size <= 128
    assert len(ids) > 0
    assert isinstance(decoded, str)
    specials = loaded.special_token_ids
    assert specials["pad_token_id"] is not None
    assert specials["bos_token_id"] is not None
    assert specials["eos_token_id"] is not None
    assert specials["unk_token_id"] is not None
```

## `tests/test_data_pipeline.py`

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from minillm.data import CausalLMBlockDataset, save_tokenized_splits, split_tokens
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
```

## `tests/test_pretrain_smoke.py`

```python
from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM


def test_tiny_pretrain_one_batch_forward_backward_cpu() -> None:
    torch.manual_seed(123)
    config = MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
    )
    model = MiniLLMForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (4, config.context_length))
    outputs = model(input_ids, labels=input_ids.clone())
    loss = outputs["loss"]
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    finite_grads = [torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None]
    assert finite_grads and all(finite_grads)
    optimizer.step()
```
