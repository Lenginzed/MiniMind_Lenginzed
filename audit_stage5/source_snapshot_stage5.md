# Stage 5 Source Snapshot

## `minillm/grpo_data.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from .tokenizer import MiniTokenizer


def load_grpo_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(
                {
                    "prompt": str(item.get("prompt", "")),
                    "answer": str(item.get("answer", "")),
                    "category": str(item.get("category", "unknown")),
                    "reward_type": str(item.get("reward_type", "unknown")),
                    "keyword": str(item.get("keyword", "")),
                }
            )
    if not rows:
        raise ValueError("GRPO jsonl is empty: %s" % path)
    return rows


class GRPODataset(Dataset):
    def __init__(self, path: str, tokenizer: MiniTokenizer, max_prompt_length: int) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        if self.max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive")
        self.raw_examples = load_grpo_jsonl(path)
        self.examples: List[Dict[str, object]] = []
        self.truncated_prompts = 0
        self.category_counts: Dict[str, int] = {}
        self.reward_type_counts: Dict[str, int] = {}
        bos_id = tokenizer.special_token_ids["bos_token_id"]
        if bos_id is None:
            raise ValueError("tokenizer must define bos_token_id")
        for raw in self.raw_examples:
            prompt = raw["prompt"]
            ids = [int(bos_id)] + tokenizer.encode(prompt, add_special_tokens=False)
            truncated = len(ids) > self.max_prompt_length
            if truncated:
                ids = ids[: self.max_prompt_length]
                self.truncated_prompts += 1
            category = raw["category"]
            reward_type = raw["reward_type"]
            self.category_counts[category] = self.category_counts.get(category, 0) + 1
            self.reward_type_counts[reward_type] = self.reward_type_counts.get(reward_type, 0) + 1
            self.examples.append(
                {
                    "prompt_input_ids": ids,
                    "prompt_text": prompt,
                    "answer": raw["answer"],
                    "category": category,
                    "reward_type": reward_type,
                    "keyword": raw.get("keyword", ""),
                    "truncated": truncated,
                }
            )
        if not self.examples:
            raise ValueError("all GRPO examples were skipped for %s" % path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.examples[idx]
        return {
            "prompt_input_ids": torch.tensor(item["prompt_input_ids"], dtype=torch.long),
            "prompt_text": item["prompt_text"],
            "answer": item["answer"],
            "category": item["category"],
            "reward_type": item["reward_type"],
            "keyword": item.get("keyword", ""),
        }

    def stats(self) -> Dict[str, object]:
        lengths = [len(item["prompt_input_ids"]) for item in self.examples]
        return {
            "path": self.path,
            "raw_examples": len(self.raw_examples),
            "effective_examples": len(self.examples),
            "truncated_prompts": self.truncated_prompts,
            "max_prompt_length": self.max_prompt_length,
            "prompt_length_min": min(lengths),
            "prompt_length_max": max(lengths),
            "prompt_length_mean": sum(lengths) / len(lengths),
            "category_counts": self.category_counts,
            "reward_type_counts": self.reward_type_counts,
        }


def grpo_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "prompt_input_ids": [item["prompt_input_ids"] for item in batch],
        "prompt_text": [item["prompt_text"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "category": [item["category"] for item in batch],
        "reward_type": [item["reward_type"] for item in batch],
        "keyword": [item.get("keyword", "") for item in batch],
    }


def write_jsonl(rows: Iterable[Dict[str, str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

```

## `minillm/grpo_rewards.py`

```python
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional, Tuple


INTEGER_RE = re.compile(r"[-+]?\d+")


def extract_first_integer(text: str) -> Optional[int]:
    match = INTEGER_RE.search(text or "")
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_keyword(text: str, keyword: str) -> bool:
    if not keyword:
        return False
    return normalize_text(keyword) in normalize_text(text)


def is_reasonable_length(text: str, min_chars: int = 1, max_chars: int = 160) -> bool:
    length = len((text or "").strip())
    return min_chars <= length <= max_chars


def _replacement_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = text.count("\ufffd") + text.count("锟")
    return bad / max(1, len(text))


def format_reward(completion: str, example: Dict[str, str]) -> float:
    text = completion or ""
    reward = 0.0
    if is_reasonable_length(text, 1, 160):
        reward += 0.05
    if any(ch.isalnum() for ch in text):
        reward += 0.02
    if _replacement_char_ratio(text) > 0.08:
        reward -= 0.10
    if len(text.strip()) > 220:
        reward -= 0.10
    return reward


def integer_accuracy_reward(completion: str, answer: str) -> float:
    predicted = extract_first_integer(completion)
    expected = extract_first_integer(answer)
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0


def keyword_reward(completion: str, keyword: str) -> float:
    return 1.0 if contains_keyword(completion, keyword) else 0.0


def length_penalty(completion: str, max_chars: int = 160) -> float:
    length = len((completion or "").strip())
    if length <= max_chars:
        return 0.0
    excess = min(200, length - max_chars)
    return -0.05 - 0.0005 * excess


def exact_text_reward(completion: str, answer: str) -> float:
    completion_norm = normalize_text(completion)
    answer_norm = normalize_text(answer)
    if not completion_norm or not answer_norm:
        return 0.0
    if completion_norm == answer_norm:
        return 1.0
    if completion_norm.startswith(answer_norm):
        return 0.75
    if answer_norm in completion_norm:
        return 0.5
    return 0.0


def combined_reward(completion: str, example: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
    text = completion or ""
    reward_type = str(example.get("reward_type", ""))
    category = str(example.get("category", ""))
    answer = str(example.get("answer", ""))
    keyword = str(example.get("keyword", "")) or answer

    fmt = format_reward(text, example)
    dense_length = 0.001 * min(60, len(text.strip())) if text.strip() else 0.0
    number_presence = 0.0
    exact = 0.0
    keyword_score = 0.0
    text_exact = 0.0

    if reward_type == "exact_integer" or category.startswith("math_"):
        number_presence = 0.10 if extract_first_integer(text) is not None else 0.0
        exact = integer_accuracy_reward(text, answer)
    elif reward_type == "keyword":
        keyword_score = keyword_reward(text, keyword)
    elif reward_type == "exact_text":
        text_exact = exact_text_reward(text, answer)
        keyword_score = 0.25 if contains_keyword(text, answer) else 0.0
    else:
        keyword_score = 0.5 * keyword_reward(text, keyword)

    penalty = length_penalty(text)
    total = fmt + dense_length + number_presence + exact + keyword_score + text_exact + penalty
    breakdown = {
        "format_reward": float(fmt),
        "dense_length_reward": float(dense_length),
        "number_presence_reward": float(number_presence),
        "exact_accuracy_reward": float(exact),
        "keyword_reward": float(keyword_score),
        "exact_text_reward": float(text_exact),
        "length_penalty": float(penalty),
        "total_reward": float(total),
        "exact_accuracy": float(exact > 0.0 or keyword_score >= 1.0 or text_exact >= 1.0),
        "completion_length": float(len(text.strip())),
        "completion_empty": float(len(text.strip()) == 0),
    }
    return float(total), breakdown

```

## `minillm/grpo_trainer.py`

```python
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import MiniLLMConfig
from .generation import generate
from .grpo_data import GRPODataset, grpo_collate_fn
from .grpo_rewards import combined_reward
from .lora import apply_lora, save_lora_adapter
from .model import MiniLLMForCausalLM
from .tokenizer import MiniTokenizer
from .trainer import build_lr_scheduler, current_lr
from .utils import append_jsonl, autocast_context, ensure_dir, get_device, load_yaml, resolve_dtype, save_json, save_yaml, set_seed


IGNORE_INDEX = -100


def load_policy_model(checkpoint_path: str, device: torch.device) -> MiniLLMForCausalLM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = MiniLLMConfig(**checkpoint["model_config"])
    model = MiniLLMForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def trainable_stats(model: torch.nn.Module) -> Dict[str, object]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": float(trainable / total) if total else 0.0,
    }


def _pad_1d(items: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(item.numel() for item in items)
    padded = []
    for item in items:
        pad_len = max_len - item.numel()
        padded.append(torch.cat([item, torch.full((pad_len,), pad_value, dtype=item.dtype, device=item.device)]))
    return torch.stack(padded, dim=0)


def build_prompt_completion_batch(
    prompt_ids: Sequence[Sequence[int]],
    completion_ids: Sequence[Sequence[int]],
    pad_token_id: int,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    if len(prompt_ids) != len(completion_ids):
        raise ValueError("prompt_ids and completion_ids must have the same length")
    if not prompt_ids:
        raise ValueError("batch must not be empty")
    input_tensors = []
    label_tensors = []
    completion_lengths = []
    for prompt, completion in zip(prompt_ids, completion_ids):
        prompt_list = [int(x) for x in prompt]
        completion_list = [int(x) for x in completion]
        if not completion_list:
            completion_list = [int(pad_token_id)]
        full = prompt_list + completion_list
        labels = [IGNORE_INDEX] * len(prompt_list) + completion_list
        input_tensors.append(torch.tensor(full, dtype=torch.long, device=device))
        label_tensors.append(torch.tensor(labels, dtype=torch.long, device=device))
        completion_lengths.append(len(completion_list))
    input_ids = _pad_1d(input_tensors, int(pad_token_id))
    labels = _pad_1d(label_tensors, IGNORE_INDEX)
    response_mask = labels[:, 1:] != IGNORE_INDEX
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
        "completion_token_count": torch.tensor(completion_lengths, dtype=torch.long, device=device),
    }


def token_logps_for_labels(
    model: MiniLLMForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids)
    logits = outputs["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    response_mask = shift_labels != IGNORE_INDEX
    safe_labels = shift_labels.masked_fill(~response_mask, 0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * response_mask
    return token_logps, response_mask


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    normalize: bool = True,
    eps: float = 1.0e-6,
) -> Dict[str, torch.Tensor]:
    if rewards.dim() != 1:
        raise ValueError("rewards must be a flat 1D tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if rewards.numel() % group_size != 0:
        raise ValueError("number of rewards must be divisible by group_size")
    grouped = rewards.view(-1, group_size)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, unbiased=False, keepdim=True)
    zero_std = stds <= eps
    if normalize:
        advantages = (grouped - means) / stds.clamp_min(eps)
    else:
        advantages = grouped - means
    advantages = torch.where(zero_std.expand_as(advantages), torch.zeros_like(advantages), advantages)
    return {
        "advantages": advantages.reshape(-1),
        "group_reward_mean": means.reshape(-1),
        "group_reward_std": stds.reshape(-1),
        "frac_reward_zero_std": zero_std.float().mean(),
    }


def grpo_loss(
    new_token_logps: torch.Tensor,
    old_token_logps: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> Dict[str, torch.Tensor]:
    if new_token_logps.shape != old_token_logps.shape or new_token_logps.shape != response_mask.shape:
        raise ValueError("token logps and response_mask must have the same shape")
    if advantages.dim() != 1 or advantages.shape[0] != new_token_logps.shape[0]:
        raise ValueError("advantages must have shape [batch]")
    old_token_logps = old_token_logps.detach()
    mask = response_mask.float()
    log_ratio = (new_token_logps - old_token_logps) * mask
    ratio = torch.exp(log_ratio).clamp(0.0, 10.0)
    adv = advantages.to(new_token_logps.device).float().unsqueeze(1)
    objective = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    surrogate = torch.minimum(objective, clipped)
    denom = mask.sum().clamp_min(1.0)
    loss = -(surrogate * mask).sum() / denom
    approx_kl = ((old_token_logps - new_token_logps) * mask).sum() / denom
    clip_ratio = (((ratio - 1.0).abs() > clip_epsilon).float() * mask).sum() / denom
    mean_ratio = (ratio * mask).sum() / denom
    return {
        "loss": loss,
        "approx_kl": approx_kl.detach(),
        "clip_ratio": clip_ratio.detach(),
        "mean_ratio": mean_ratio.detach(),
        "response_token_count": mask.sum(dim=1).detach(),
    }


def mean_float(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def std_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = mean_float(values)
    return float((sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5)


def rollout_batch(
    model: MiniLLMForCausalLM,
    tokenizer: MiniTokenizer,
    batch: Dict[str, object],
    device: torch.device,
    dtype_name: str,
    max_new_tokens: int,
    num_generations: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    pad_token_id: int,
) -> Dict[str, object]:
    prompt_ids_flat: List[List[int]] = []
    completion_ids_flat: List[List[int]] = []
    rewards: List[float] = []
    breakdowns: List[Dict[str, float]] = []
    rollout_rows: List[Dict[str, object]] = []
    eos_id = tokenizer.special_token_ids.get("eos_token_id")
    was_training = model.training
    model.eval()
    try:
        for prompt_index, prompt_tensor in enumerate(batch["prompt_input_ids"]):
            prompt_ids = [int(x) for x in prompt_tensor.tolist()]
            example = {
                "prompt": batch["prompt_text"][prompt_index],
                "answer": batch["answer"][prompt_index],
                "category": batch["category"][prompt_index],
                "reward_type": batch["reward_type"][prompt_index],
                "keyword": batch["keyword"][prompt_index],
            }
            for generation_index in range(num_generations):
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    generated = generate(
                        model,
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        eos_token_id=eos_id,
                        do_sample=True,
                    )
                full_ids = generated[0].detach().cpu().tolist()
                completion_ids = full_ids[len(prompt_ids) :]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                reward, breakdown = combined_reward(completion_text, example)
                prompt_ids_flat.append(prompt_ids)
                completion_ids_flat.append(completion_ids)
                rewards.append(float(reward))
                breakdowns.append(breakdown)
                rollout_rows.append(
                    {
                        "prompt": example["prompt"],
                        "answer": example["answer"],
                        "category": example["category"],
                        "reward_type": example["reward_type"],
                        "generation_index": generation_index,
                        "completion": completion_text,
                        "completion_token_count": len(completion_ids),
                        "reward": float(reward),
                        "breakdown": breakdown,
                    }
                )
    finally:
        if was_training:
            model.train()
    tensor_batch = build_prompt_completion_batch(prompt_ids_flat, completion_ids_flat, pad_token_id, device=device)
    old_training = model.training
    model.eval()
    with torch.no_grad():
        with autocast_context(device, dtype_name):
            old_logps, response_mask = token_logps_for_labels(model, tensor_batch["input_ids"], tensor_batch["labels"])
    if old_training:
        model.train()
    return {
        "prompt_ids": prompt_ids_flat,
        "completion_ids": completion_ids_flat,
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
        "breakdowns": breakdowns,
        "rollout_rows": rollout_rows,
        "input_ids": tensor_batch["input_ids"],
        "labels": tensor_batch["labels"],
        "response_mask": response_mask.detach(),
        "old_token_logps": old_logps.detach(),
        "completion_token_count": tensor_batch["completion_token_count"],
    }


@torch.no_grad()
def evaluate_grpo(
    model: MiniLLMForCausalLM,
    tokenizer: MiniTokenizer,
    dataset: GRPODataset,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    max_examples: int,
) -> Dict[str, object]:
    rewards: List[float] = []
    exacts: List[float] = []
    formats: List[float] = []
    lengths: List[float] = []
    samples: List[Dict[str, object]] = []
    eos_id = tokenizer.special_token_ids.get("eos_token_id")
    was_training = model.training
    model.eval()
    try:
        for idx in range(min(max_examples, len(dataset))):
            item = dataset[idx]
            prompt_ids = [int(x) for x in item["prompt_input_ids"].tolist()]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            out = generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_id,
                do_sample=True,
            )
            full_ids = out[0].detach().cpu().tolist()
            completion_ids = full_ids[len(prompt_ids) :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            example = {
                "prompt": item["prompt_text"],
                "answer": item["answer"],
                "category": item["category"],
                "reward_type": item["reward_type"],
                "keyword": item.get("keyword", ""),
            }
            reward, breakdown = combined_reward(completion, example)
            rewards.append(float(reward))
            exacts.append(float(breakdown["exact_accuracy"]))
            formats.append(float(breakdown["format_reward"]))
            lengths.append(float(breakdown["completion_length"]))
            if len(samples) < 12:
                samples.append({**example, "completion": completion, "reward": reward, "breakdown": breakdown})
    finally:
        if was_training:
            model.train()
    return {
        "reward_mean": mean_float(rewards),
        "reward_std": std_float(rewards),
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "exact_accuracy_mean": mean_float(exacts),
        "format_reward_mean": mean_float(formats),
        "completion_length_mean": mean_float(lengths),
        "num_examples": len(rewards),
        "samples": samples,
    }


def save_grpo_checkpoint(
    path: str,
    model: MiniLLMForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: Dict[str, Any],
    best_eval_reward: float,
    mode: str,
    adapter_path: Optional[str],
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
            "best_eval_reward": float(best_eval_reward),
            "mode": mode,
            "adapter_path": adapter_path,
        },
        path,
    )


def append_rollout_samples(rows: List[Dict[str, object]], path: str, step: int, limit: int = 8) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        for row in rows[:limit]:
            payload = {"step": step, **row}
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def run_grpo(config_path: str) -> Dict[str, Any]:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 1234)))
    output_dir = config["output_dir"]
    for sub in ["checkpoints", "adapters", "logs", "samples", "eval", "plots"]:
        ensure_dir(str(Path(output_dir) / sub))
    tokenizer = MiniTokenizer.load(config["tokenizer_path"])
    pad_id = tokenizer.special_token_ids.get("pad_token_id")
    if pad_id is None:
        raise ValueError("tokenizer must define pad_token_id")
    device = get_device(bool(config.get("prefer_cuda", True)))
    dtype_name = resolve_dtype(str(config.get("dtype", "auto")))
    if device.type != "cuda":
        dtype_name = "fp32"

    model = load_policy_model(config["policy_checkpoint"], device)
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
    stats = trainable_stats(model)

    train_ds = GRPODataset(config["train_data_path"], tokenizer, int(config["max_prompt_length"]))
    val_ds = GRPODataset(config["val_data_path"], tokenizer, int(config["max_prompt_length"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=grpo_collate_fn,
    )
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
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    eval_interval = int(config["training"].get("eval_interval", 10))
    save_interval = int(config["training"].get("save_interval", 25))
    log_interval = int(config["training"].get("log_interval", 10))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    num_generations = int(config["num_generations"])
    max_new_tokens = int(config["max_new_tokens"])
    clip_epsilon = float(config.get("clip_epsilon", 0.2))
    temperature = float(config.get("temperature", 1.0))
    top_k = config.get("top_k", None)
    top_k = int(top_k) if top_k is not None else None
    top_p = config.get("top_p", None)
    top_p = float(top_p) if top_p is not None else None
    eval_examples = int(config["training"].get("eval_examples", 24))

    metrics_path = str(Path(output_dir) / "metrics.jsonl")
    csv_path = str(Path(output_dir) / "metrics.csv")
    rollout_path = str(Path(output_dir) / "samples" / "rollout_samples.jsonl")
    for path in [metrics_path, csv_path, rollout_path]:
        if Path(path).exists():
            Path(path).unlink()
    save_yaml(config, str(Path(output_dir) / "train_config_resolved.yaml"))
    save_json({"train": train_ds.stats(), "val": val_ds.stats()}, str(Path(output_dir) / "data_stats.json"))
    writer = SummaryWriter(log_dir=str(Path(output_dir) / "logs"))

    csv_fields = [
        "step", "train_loss", "eval_reward_mean", "reward_mean", "reward_std", "reward_min", "reward_max",
        "group_reward_std_mean", "frac_reward_zero_std", "advantage_mean", "advantage_std",
        "completion_length_mean", "completion_empty_rate", "exact_accuracy_mean", "format_reward_mean",
        "clip_ratio", "approx_kl", "mean_ratio", "lr", "grad_norm", "trainable_params", "total_params",
    ]
    csv_file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    train_iter = iter(train_loader)
    best_eval_reward = float("-inf")
    first_record: Optional[Dict[str, Any]] = None
    last_record: Optional[Dict[str, Any]] = None
    last_eval: Dict[str, object] = {}
    use_scaler = device.type == "cuda" and dtype_name == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step in range(1, max_steps + 1):
            accum_losses: List[float] = []
            accum_records: List[Dict[str, float]] = []
            rollout_for_log: List[Dict[str, object]] = []
            for _ in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                rollout = rollout_batch(
                    model,
                    tokenizer,
                    batch,
                    device,
                    dtype_name,
                    max_new_tokens,
                    num_generations,
                    temperature,
                    top_k,
                    top_p,
                    int(pad_id),
                )
                rewards = rollout["rewards"]
                adv_data = compute_group_advantages(rewards, num_generations, normalize=True)
                with autocast_context(device, dtype_name):
                    new_logps, response_mask = token_logps_for_labels(model, rollout["input_ids"], rollout["labels"])
                    loss_data = grpo_loss(
                        new_logps,
                        rollout["old_token_logps"],
                        response_mask,
                        adv_data["advantages"],
                        clip_epsilon=clip_epsilon,
                    )
                    loss = loss_data["loss"] / grad_accum
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                breakdowns = rollout["breakdowns"]
                reward_values = [float(x) for x in rewards.detach().cpu().tolist()]
                advantages = adv_data["advantages"].detach().cpu().float()
                record_part = {
                    "train_loss": float(loss_data["loss"].detach().cpu().item()),
                    "reward_mean": mean_float(reward_values),
                    "reward_std": std_float(reward_values),
                    "reward_min": min(reward_values),
                    "reward_max": max(reward_values),
                    "group_reward_std_mean": float(adv_data["group_reward_std"].detach().float().mean().cpu().item()),
                    "frac_reward_zero_std": float(adv_data["frac_reward_zero_std"].detach().cpu().item()),
                    "advantage_mean": float(advantages.mean().item()),
                    "advantage_std": float(advantages.std(unbiased=False).item()),
                    "completion_length_mean": mean_float([item["completion_length"] for item in breakdowns]),
                    "completion_empty_rate": mean_float([item["completion_empty"] for item in breakdowns]),
                    "exact_accuracy_mean": mean_float([item["exact_accuracy"] for item in breakdowns]),
                    "format_reward_mean": mean_float([item["format_reward"] for item in breakdowns]),
                    "clip_ratio": float(loss_data["clip_ratio"].detach().cpu().item()),
                    "approx_kl": float(loss_data["approx_kl"].detach().cpu().item()),
                    "mean_ratio": float(loss_data["mean_ratio"].detach().cpu().item()),
                }
                accum_losses.append(record_part["train_loss"])
                accum_records.append(record_part)
                rollout_for_log.extend(rollout["rollout_rows"])

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

            def avg_record(key: str) -> float:
                return mean_float([item[key] for item in accum_records])

            eval_data = None
            if step == 1 or step % eval_interval == 0 or step == max_steps:
                eval_data = evaluate_grpo(
                    model,
                    tokenizer,
                    val_ds,
                    device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_examples=eval_examples,
                )
                last_eval = eval_data
                eval_reward = float(eval_data["reward_mean"])
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    adapter_path = str(Path(output_dir) / "adapters" / "best_adapter.pt") if mode == "lora" else None
                    if mode == "lora":
                        save_lora_adapter(model, adapter_path, {**lora_cfg, **lora_stats})
                    save_grpo_checkpoint(
                        str(Path(output_dir) / "checkpoints" / "best.pt"),
                        model,
                        optimizer,
                        scheduler,
                        step,
                        config,
                        best_eval_reward,
                        mode,
                        adapter_path,
                    )

            if step % save_interval == 0 or step == max_steps:
                adapter_path = str(Path(output_dir) / "adapters" / "last_adapter.pt") if mode == "lora" else None
                if mode == "lora":
                    save_lora_adapter(model, adapter_path, {**lora_cfg, **lora_stats})
                save_grpo_checkpoint(
                    str(Path(output_dir) / "checkpoints" / "last.pt"),
                    model,
                    optimizer,
                    scheduler,
                    step,
                    config,
                    best_eval_reward,
                    mode,
                    adapter_path,
                )

            if step == 1 or step % log_interval == 0 or step == max_steps:
                append_rollout_samples(rollout_for_log, rollout_path, step, limit=8)

            record = {
                "step": step,
                "train_loss": avg_record("train_loss"),
                "eval_reward_mean": eval_data.get("reward_mean") if eval_data else None,
                "eval_exact_accuracy_mean": eval_data.get("exact_accuracy_mean") if eval_data else None,
                "reward_mean": avg_record("reward_mean"),
                "reward_std": avg_record("reward_std"),
                "reward_min": avg_record("reward_min"),
                "reward_max": avg_record("reward_max"),
                "group_reward_std_mean": avg_record("group_reward_std_mean"),
                "frac_reward_zero_std": avg_record("frac_reward_zero_std"),
                "advantage_mean": avg_record("advantage_mean"),
                "advantage_std": avg_record("advantage_std"),
                "completion_length_mean": avg_record("completion_length_mean"),
                "completion_empty_rate": avg_record("completion_empty_rate"),
                "exact_accuracy_mean": avg_record("exact_accuracy_mean"),
                "format_reward_mean": avg_record("format_reward_mean"),
                "clip_ratio": avg_record("clip_ratio"),
                "approx_kl": avg_record("approx_kl"),
                "mean_ratio": avg_record("mean_ratio"),
                "lr": current_lr(optimizer),
                "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                "trainable_params": stats["trainable_params"],
                "total_params": stats["total_params"],
            }
            first_record = record if first_record is None else first_record
            last_record = record
            append_jsonl(record, metrics_path)
            csv_writer.writerow({key: record.get(key) for key in csv_fields})
            csv_file.flush()
            writer.add_scalar("loss/train", record["train_loss"], step)
            writer.add_scalar("reward/mean", record["reward_mean"], step)
            writer.add_scalar("reward/std", record["reward_std"], step)
            writer.add_scalar("rl/clip_ratio", record["clip_ratio"], step)
            writer.add_scalar("rl/approx_kl", record["approx_kl"], step)
            if eval_data:
                writer.add_scalar("reward/eval_mean", eval_data["reward_mean"], step)
            if step == 1 or step % log_interval == 0 or step == max_steps:
                print(
                    "step=%d loss=%.4f reward=%.4f std=%.4f zero_std=%.3f exact=%.3f kl=%.4f clip=%.3f eval=%s"
                    % (
                        step,
                        record["train_loss"],
                        record["reward_mean"],
                        record["reward_std"],
                        record["frac_reward_zero_std"],
                        record["exact_accuracy_mean"],
                        record["approx_kl"],
                        record["clip_ratio"],
                        "%.4f" % record["eval_reward_mean"] if record["eval_reward_mean"] is not None else "None",
                    )
                )
    finally:
        csv_file.close()
        writer.close()

    if not Path(output_dir, "checkpoints", "last.pt").exists():
        save_grpo_checkpoint(str(Path(output_dir) / "checkpoints" / "last.pt"), model, optimizer, scheduler, max_steps, config, best_eval_reward, mode, None)
    if not Path(output_dir, "checkpoints", "best.pt").exists():
        save_grpo_checkpoint(str(Path(output_dir) / "checkpoints" / "best.pt"), model, optimizer, scheduler, max_steps, config, best_eval_reward, mode, None)
    summary = {
        "mode": mode,
        "output_dir": output_dir,
        "policy_checkpoint": config["policy_checkpoint"],
        "parameter_count": stats["total_params"],
        "trainable_params": stats["trainable_params"],
        "trainable_ratio": stats["trainable_ratio"],
        "device": str(device),
        "dtype": dtype_name,
        "max_steps": max_steps,
        "num_generations": num_generations,
        "max_new_tokens": max_new_tokens,
        "clip_epsilon": clip_epsilon,
        "best_eval_reward": best_eval_reward,
        "first_record": first_record,
        "last_record": last_record,
        "last_eval": last_eval,
        "metrics_path": metrics_path,
        "rollout_samples_path": rollout_path,
        "best_checkpoint": str(Path(output_dir) / "checkpoints" / "best.pt"),
        "last_checkpoint": str(Path(output_dir) / "checkpoints" / "last.pt"),
        "lora": lora_stats,
    }
    save_json(summary, str(Path(output_dir) / "grpo_summary.json"))
    return summary

```

## `scripts/create_grpo_dataset.py`

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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from minillm.grpo_data import write_jsonl
from minillm.utils import ensure_dir, save_json


def math_add(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(0, 49)
    b = rng.randint(0, 49)
    answer = str(a + b)
    return {
        "prompt": f"User: Compute {a} + {b}. Answer with the final integer only.\nAssistant: ",
        "answer": answer,
        "category": "math_add",
        "reward_type": "exact_integer",
        "keyword": "",
    }


def math_sub(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(10, 80)
    b = rng.randint(0, a)
    answer = str(a - b)
    return {
        "prompt": f"User: Compute {a} - {b}. Answer with the final integer only.\nAssistant: ",
        "answer": answer,
        "category": "math_sub",
        "reward_type": "exact_integer",
        "keyword": "",
    }


def math_mul_small(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(0, 12)
    b = rng.randint(0, 12)
    answer = str(a * b)
    return {
        "prompt": f"User: Compute {a} * {b}. Answer with the final integer only.\nAssistant: ",
        "answer": answer,
        "category": "math_mul_small",
        "reward_type": "exact_integer",
        "keyword": "",
    }


def format_echo(rng: random.Random) -> Dict[str, str]:
    phrase = rng.choice(["READY", "OK", "DONE", "SAFE", "PASS"])
    return {
        "prompt": f"User: Output exactly the word {phrase} and nothing else.\nAssistant: ",
        "answer": phrase,
        "category": "format_echo",
        "reward_type": "exact_text",
        "keyword": phrase,
    }


def concept_keyword(rng: random.Random) -> Dict[str, str]:
    keyword = rng.choice(["LoRA", "tokenizer", "SFT", "DPO", "reward"])
    prompts = {
        "LoRA": "User: In one short phrase, mention the adapter method LoRA.\nAssistant: ",
        "tokenizer": "User: In one short phrase, mention the component tokenizer.\nAssistant: ",
        "SFT": "User: In one short phrase, mention supervised fine-tuning as SFT.\nAssistant: ",
        "DPO": "User: In one short phrase, mention preference optimization as DPO.\nAssistant: ",
        "reward": "User: In one short phrase, mention the reward signal.\nAssistant: ",
    }
    return {
        "prompt": prompts[keyword],
        "answer": keyword,
        "category": "concept_keyword",
        "reward_type": "keyword",
        "keyword": keyword,
    }


BUILDERS = [math_add, math_sub, math_mul_small, format_echo, concept_keyword]


def build_rows(count: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(count):
        row = BUILDERS[idx % len(BUILDERS)](rng)
        row["id"] = f"grpo_{seed}_{idx:06d}"
        rows.append(row)
    rng.shuffle(rows)
    return rows


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    answers = Counter(row["answer"] for row in rows)
    return {
        "count": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "reward_type_counts": dict(Counter(row["reward_type"] for row in rows)),
        "answer_top10": dict(answers.most_common(10)),
        "answer_unique": len(answers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create local synthetic GRPO/RLVR prompts.")
    parser.add_argument("--out-dir", default="data/grpo")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    train_rows = build_rows(args.train_size, args.seed)
    val_rows = build_rows(args.val_size, args.seed + 1)
    train_path = str(Path(args.out_dir) / "grpo_train.jsonl")
    val_path = str(Path(args.out_dir) / "grpo_val.jsonl")
    write_jsonl(train_rows, train_path)
    write_jsonl(val_rows, val_path)
    metadata = {
        "description": "Synthetic local reward data only for GRPO pipeline validation; it is not real RLHF/RLVR data.",
        "seed": args.seed,
        "train_path": train_path,
        "val_path": val_path,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "format": {
            "fields": ["prompt", "answer", "category", "reward_type", "keyword"],
            "prompt_template": "User: ...\\nAssistant: ",
        },
    }
    save_json(metadata, str(Path(args.out_dir) / "grpo_metadata.json"))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `scripts/train_grpo.py`

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

from minillm.grpo_trainer import run_grpo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GRPO / GRPO-LoRA smoke training.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = run_grpo(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `scripts/eval_grpo.py`

```python
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

```

## `tests/test_grpo_rewards.py`

```python
from __future__ import annotations

from minillm.grpo_rewards import (
    combined_reward,
    contains_keyword,
    extract_first_integer,
    integer_accuracy_reward,
    keyword_reward,
    length_penalty,
    normalize_text,
)


def test_integer_extraction_and_exact_reward() -> None:
    assert extract_first_integer("answer is -12, then 7") == -12
    assert extract_first_integer("no digits") is None
    assert integer_accuracy_reward("The answer is 19.", "19") == 1.0
    assert integer_accuracy_reward("The answer is 18.", "19") == 0.0


def test_keyword_and_normalize() -> None:
    assert normalize_text("  LoRA\nAdapter ") == "lora adapter"
    assert contains_keyword("This mentions a Tokenizer.", "tokenizer")
    assert keyword_reward("SFT is supervised fine-tuning.", "SFT") == 1.0
    assert keyword_reward("No relevant term.", "LoRA") == 0.0


def test_length_penalty() -> None:
    assert length_penalty("short", max_chars=10) == 0.0
    assert length_penalty("x" * 50, max_chars=10) < 0.0


def test_combined_reward_breakdown_for_math_and_keyword() -> None:
    math_example = {"category": "math_add", "reward_type": "exact_integer", "answer": "7", "keyword": ""}
    reward, breakdown = combined_reward("7", math_example)
    assert reward >= 1.0
    assert breakdown["exact_accuracy_reward"] == 1.0
    assert breakdown["exact_accuracy"] == 1.0

    keyword_example = {"category": "concept_keyword", "reward_type": "keyword", "answer": "LoRA", "keyword": "LoRA"}
    reward, breakdown = combined_reward("LoRA uses adapters.", keyword_example)
    assert reward >= 1.0
    assert breakdown["keyword_reward"] == 1.0
    assert breakdown["total_reward"] == reward

```

## `tests/test_grpo_advantage.py`

```python
from __future__ import annotations

import torch

from minillm.grpo_trainer import compute_group_advantages


def test_group_advantage_mean_zero_for_nonconstant_groups() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    out = compute_group_advantages(rewards, group_size=3)
    adv = out["advantages"].view(2, 3)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert out["group_reward_std"].shape == (2,)
    assert out["frac_reward_zero_std"].item() == 0.0


def test_zero_std_group_advantage_is_zero_and_recorded() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    out = compute_group_advantages(rewards, group_size=3)
    adv = out["advantages"].view(2, 3)
    assert torch.allclose(adv[0], torch.zeros(3))
    assert out["frac_reward_zero_std"].item() == 0.5


def test_invalid_group_size_raises() -> None:
    try:
        compute_group_advantages(torch.tensor([1.0, 2.0, 3.0]), group_size=2)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")

```

## `tests/test_grpo_loss.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minillm.grpo_trainer import build_prompt_completion_batch, grpo_loss, token_logps_for_labels


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, input_ids: torch.Tensor):
        return {"logits": self.logits.expand(input_ids.shape[0], -1, -1).clone()}


def test_build_prompt_completion_batch_masks_prompt() -> None:
    batch = build_prompt_completion_batch([[1, 2, 3]], [[4, 5]], pad_token_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]
    assert batch["response_mask"].tolist() == [[False, False, True, True]]
    assert batch["completion_token_count"].tolist() == [2]


def test_token_logps_for_labels_only_response_tokens() -> None:
    vocab_size = 8
    logits = torch.zeros(1, 5, vocab_size)
    logits[0, 2, 4] = 5.0
    logits[0, 3, 5] = 6.0
    model = FixedLogitModel(logits)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    token_logps, mask = token_logps_for_labels(model, input_ids, labels)
    expected = F.log_softmax(logits[0, 2], dim=-1)[4] + F.log_softmax(logits[0, 3], dim=-1)[5]
    assert mask.tolist() == [[False, False, True, True]]
    assert torch.allclose(token_logps.sum(), expected)


def test_grpo_clipped_loss_finite_and_shapes() -> None:
    old = torch.zeros(2, 3)
    new = torch.tensor([[0.1, 0.2, 0.0], [-0.1, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    advantages = torch.tensor([1.0, -1.0])
    out = grpo_loss(new, old, mask, advantages, clip_epsilon=0.2)
    assert torch.isfinite(out["loss"])
    assert out["response_token_count"].tolist() == [2.0, 1.0]
    out["loss"].backward()
    assert new.grad is not None


def test_grpo_loss_all_zero_advantage_does_not_crash() -> None:
    old = torch.zeros(2, 3)
    new = torch.zeros(2, 3, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    advantages = torch.zeros(2)
    out = grpo_loss(new, old, mask, advantages, clip_epsilon=0.2)
    assert torch.isfinite(out["loss"])
    assert abs(out["loss"].item()) < 1e-8
    out["loss"].backward()
    assert new.grad is not None

```

## `tests/test_grpo_trainer_smoke.py`

```python
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.grpo_data import write_jsonl
from minillm.grpo_trainer import run_grpo
from minillm.tokenizer import MiniTokenizer


def setup_tiny_grpo_run(tmp_path: Path, lora: bool) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "User: 1 + 1?\nAssistant: 2\n"
            "User: say LoRA\nAssistant: LoRA\n"
            "User: output OK\nAssistant: OK\n"
        )
        * 120,
        encoding="utf-8",
    )
    tokenizer = MiniTokenizer.train_from_files([str(corpus)], vocab_size=128, min_frequency=1)
    tokenizer_path = tmp_path / "tok.json"
    tokenizer.save(str(tokenizer_path))
    rows = [
        {
            "prompt": "User: Compute 1 + 1. Answer with the final integer only.\nAssistant: ",
            "answer": "2",
            "category": "math_add",
            "reward_type": "exact_integer",
            "keyword": "",
        },
        {
            "prompt": "User: Mention LoRA.\nAssistant: ",
            "answer": "LoRA",
            "category": "concept_keyword",
            "reward_type": "keyword",
            "keyword": "LoRA",
        },
        {
            "prompt": "User: Output exactly the word OK and nothing else.\nAssistant: ",
            "answer": "OK",
            "category": "format_echo",
            "reward_type": "exact_text",
            "keyword": "OK",
        },
    ] * 8
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(rows, str(train_path))
    write_jsonl(rows[:6], str(val_path))

    cfg = MiniLLMConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=40,
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        intermediate_size=64,
    )
    model = MiniLLMForCausalLM(cfg)
    ckpt = tmp_path / "sft_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": cfg.__dict__,
            "optimizer_state_dict": {},
            "step": 0,
            "best_eval_loss": 0.0,
        },
        ckpt,
    )
    config = {
        "seed": 123,
        "policy_checkpoint": str(ckpt),
        "tokenizer_path": str(tokenizer_path),
        "train_data_path": str(train_path),
        "val_data_path": str(val_path),
        "output_dir": str(tmp_path / ("grpo_lora" if lora else "grpo_full")),
        "prefer_cuda": False,
        "dtype": "fp32",
        "max_prompt_length": 28,
        "max_new_tokens": 4,
        "num_generations": 2,
        "clip_epsilon": 0.2,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
            "eval_interval": 1,
            "save_interval": 1,
            "log_interval": 1,
            "eval_examples": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "scheduler": "none",
            "warmup_steps": 0,
        },
    }
    if lora:
        config["lora"] = {"enabled": True, "r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
    config_path = tmp_path / ("grpo_lora.yaml" if lora else "grpo_full.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_full_grpo_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_grpo(str(setup_tiny_grpo_run(tmp_path, lora=False)))
    assert summary["mode"] == "full"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] == summary["parameter_count"]
    assert Path(summary["metrics_path"]).exists()
    assert Path(summary["rollout_samples_path"]).exists()
    assert Path(summary["best_checkpoint"]).exists()


def test_lora_grpo_trainer_cpu_smoke(tmp_path: Path) -> None:
    summary = run_grpo(str(setup_tiny_grpo_run(tmp_path, lora=True)))
    assert summary["mode"] == "lora"
    assert summary["max_steps"] == 2
    assert summary["trainable_params"] < summary["parameter_count"]
    assert Path(summary["metrics_path"]).exists()
    assert Path(summary["output_dir"], "adapters", "best_adapter.pt").exists()

```

## `configs/grpo_full.yaml`

```yaml
seed: 20260711
policy_checkpoint: outputs/sft_full/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
train_data_path: data/grpo/grpo_train.jsonl
val_data_path: data/grpo/grpo_val.jsonl
output_dir: outputs/grpo_full
prefer_cuda: true
dtype: auto
max_prompt_length: 96
max_new_tokens: 32
num_generations: 4
beta_kl: 0.0
clip_epsilon: 0.2
temperature: 1.0
top_k: 50
top_p: 0.95

training:
  batch_size: 4
  gradient_accumulation_steps: 1
  max_steps: 60
  eval_interval: 10
  save_interval: 25
  log_interval: 10
  eval_examples: 32
  learning_rate: 5.0e-5
  weight_decay: 0.0
  grad_clip: 1.0
  scheduler: cosine
  warmup_steps: 5

```

## `configs/grpo_lora.yaml`

```yaml
seed: 20260711
policy_checkpoint: outputs/sft_full/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
train_data_path: data/grpo/grpo_train.jsonl
val_data_path: data/grpo/grpo_val.jsonl
output_dir: outputs/grpo_lora
prefer_cuda: true
dtype: auto
max_prompt_length: 96
max_new_tokens: 32
num_generations: 4
beta_kl: 0.0
clip_epsilon: 0.2
temperature: 1.0
top_k: 50
top_p: 0.95

lora:
  enabled: true
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj

training:
  batch_size: 4
  gradient_accumulation_steps: 1
  max_steps: 50
  eval_interval: 10
  save_interval: 25
  log_interval: 10
  eval_examples: 32
  learning_rate: 1.0e-4
  weight_decay: 0.0
  grad_clip: 1.0
  scheduler: cosine
  warmup_steps: 5

```
