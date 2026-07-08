# Stage 6 Source Snapshot

## `minillm/quantization.py`

```python
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALinear


def _validate_bits(num_bits: int) -> None:
    if int(num_bits) not in {4, 8}:
        raise ValueError("num_bits must be 4 or 8")


def calculate_qparams_symmetric(
    tensor: torch.Tensor,
    num_bits: int,
    per_channel: bool = False,
    channel_dim: int = 0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    _validate_bits(num_bits)
    if tensor.numel() == 0:
        raise ValueError("cannot quantize empty tensor")
    qmax = float((2 ** (int(num_bits) - 1)) - 1)
    x = tensor.detach().float()
    if per_channel:
        channel_dim = channel_dim % x.dim()
        reduce_dims = [dim for dim in range(x.dim()) if dim != channel_dim]
        max_abs = x.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        max_abs = x.abs().max()
    scale = (max_abs / qmax).clamp_min(eps)
    return scale


def quantize_tensor_symmetric(tensor: torch.Tensor, scale: torch.Tensor, num_bits: int) -> torch.Tensor:
    _validate_bits(num_bits)
    qmax = (2 ** (int(num_bits) - 1)) - 1
    qmin = -qmax
    qtensor = torch.round(tensor.detach().float() / scale.float()).clamp(qmin, qmax)
    return qtensor.to(torch.int8)


def dequantize_tensor_symmetric(qtensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return qtensor.float() * scale.float()


def fake_quantize_tensor_symmetric(
    tensor: torch.Tensor,
    num_bits: int,
    per_channel: bool = False,
    channel_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = calculate_qparams_symmetric(tensor, num_bits, per_channel=per_channel, channel_dim=channel_dim)
    qtensor = quantize_tensor_symmetric(tensor, scale, num_bits)
    dequant = dequantize_tensor_symmetric(qtensor, scale).to(dtype=tensor.dtype, device=tensor.device)
    return dequant, qtensor, scale


class QuantizedLinear(nn.Module):
    """Educational fake-quantized Linear layer.

    The weight is stored as int8 values even for int4. INT4 bit packing and
    integer GEMM kernels are intentionally out of scope for this project stage.
    """

    def __init__(
        self,
        linear: nn.Linear,
        num_bits: int = 8,
        per_channel: bool = True,
        channel_dim: int = 0,
        module_name: str = "",
    ) -> None:
        super().__init__()
        _validate_bits(num_bits)
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.num_bits = int(num_bits)
        self.per_channel = bool(per_channel)
        self.channel_dim = int(channel_dim)
        self.module_name = module_name
        weight = linear.weight.detach()
        scale = calculate_qparams_symmetric(weight, num_bits, per_channel=per_channel, channel_dim=channel_dim)
        qweight = quantize_tensor_symmetric(weight, scale, num_bits)
        self.register_buffer("qweight", qweight.to(device=weight.device), persistent=True)
        self.register_buffer("scale", scale.to(device=weight.device), persistent=True)
        if linear.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", linear.bias.detach().clone(), persistent=True)

    @property
    def weight(self) -> torch.Tensor:
        return dequantize_tensor_symmetric(self.qweight, self.scale)

    def dequantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return dequantize_tensor_symmetric(self.qweight.to(device), self.scale.to(device)).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight(dtype=x.dtype, device=x.device)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


def _target_match(full_name: str, short_name: str, target_modules: Optional[Iterable[str]]) -> bool:
    if target_modules is None:
        return True
    targets = set(target_modules)
    return short_name in targets or full_name in targets


def quantize_model_weight_only(
    model: nn.Module,
    num_bits: int = 8,
    target_modules: Optional[Iterable[str]] = None,
    per_channel: bool = True,
) -> Dict[str, object]:
    _validate_bits(num_bits)
    stats = {
        "linear_count": 0,
        "quantized_count": 0,
        "skipped_count": 0,
        "skipped_lora_count": 0,
        "num_bits": int(num_bits),
        "per_channel": bool(per_channel),
        "quantized_modules": [],
        "skipped_modules": [],
    }

    def replace_children(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = "%s.%s" % (prefix, name) if prefix else name
            if isinstance(child, LoRALinear):
                stats["skipped_count"] += 1
                stats["skipped_lora_count"] += 1
                stats["skipped_modules"].append(full_name)
                continue
            if isinstance(child, QuantizedLinear):
                stats["skipped_count"] += 1
                stats["skipped_modules"].append(full_name)
                continue
            if isinstance(child, nn.Linear):
                stats["linear_count"] += 1
                if _target_match(full_name, name, target_modules):
                    setattr(
                        parent,
                        name,
                        QuantizedLinear(child, num_bits=num_bits, per_channel=per_channel, module_name=full_name),
                    )
                    stats["quantized_count"] += 1
                    stats["quantized_modules"].append(full_name)
                else:
                    stats["skipped_count"] += 1
                    stats["skipped_modules"].append(full_name)
            else:
                replace_children(child, full_name)

    replace_children(model)
    return stats


def estimate_model_size_bytes(model: nn.Module) -> int:
    total = 0
    seen = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is None:
            continue
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += tensor.numel() * tensor.element_size()
    return int(total)


def estimate_quantized_size_bytes(model: nn.Module) -> int:
    total = 0
    seen = set()
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            total += int(math.ceil(module.qweight.numel() * module.num_bits / 8.0))
            total += module.scale.numel() * module.scale.element_size()
            if module.bias is not None:
                total += module.bias.numel() * module.bias.element_size()
            continue
        if hasattr(module, "qweight") and hasattr(module, "num_bits"):
            qweight = getattr(module, "qweight")
            total += int(math.ceil(qweight.numel() * int(getattr(module, "num_bits")) / 8.0))
            for name, buffer in module.named_buffers(recurse=False):
                if buffer is None or name == "qweight":
                    continue
                ptr = buffer.data_ptr()
                if ptr in seen:
                    continue
                seen.add(ptr)
                total += buffer.numel() * buffer.element_size()
            continue
        for param in module.parameters(recurse=False):
            if param is None:
                continue
            ptr = param.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            total += param.numel() * param.element_size()
        for buffer in module.buffers(recurse=False):
            if buffer is None:
                continue
            ptr = buffer.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            total += buffer.numel() * buffer.element_size()
    return int(total)


def compression_report(baseline_bytes: int, quantized_bytes: int) -> Dict[str, float]:
    return {
        "baseline_size_bytes": int(baseline_bytes),
        "quantized_size_bytes": int(quantized_bytes),
        "compression_ratio": float(baseline_bytes / max(1, quantized_bytes)),
    }


def quantization_error(original: torch.Tensor, quantized: torch.Tensor) -> Dict[str, float]:
    diff = original.detach().float() - quantized.detach().float()
    return {
        "weight_mse": float(diff.pow(2).mean().cpu().item()),
        "weight_mae": float(diff.abs().mean().cpu().item()),
        "weight_max_abs_error": float(diff.abs().max().cpu().item()),
    }

```

## `minillm/gptq.py`

```python
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .lora import LoRALinear
from .quantization import (
    QuantizedLinear,
    calculate_qparams_symmetric,
    dequantize_tensor_symmetric,
    quantization_error,
    quantize_tensor_symmetric,
)


def _matches(name: str, short_name: str, targets: Optional[Iterable[str]]) -> bool:
    if targets is None:
        return True
    target_set = set(targets)
    return name in target_set or short_name in target_set


def named_plain_linears(model: nn.Module, target_modules: Optional[Iterable[str]] = None) -> Dict[str, nn.Linear]:
    modules: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, LoRALinear) or isinstance(module, QuantizedLinear):
            continue
        if isinstance(module, nn.Linear):
            short_name = name.rsplit(".", 1)[-1]
            if _matches(name, short_name, target_modules):
                modules[name] = module
    return modules


def collect_linear_calibration_stats(
    model: nn.Module,
    dataloader,
    target_modules: Optional[Iterable[str]] = None,
    max_batches: int = 10,
) -> Dict[str, Dict[str, torch.Tensor]]:
    modules = named_plain_linears(model, target_modules)
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    def make_hook(module_name: str):
        def hook(module, inputs, output):
            x = inputs[0].detach().float()
            x = x.reshape(-1, x.shape[-1]).cpu()
            if module_name not in stats:
                stats[module_name] = {
                    "hessian_diag": torch.zeros(x.shape[-1], dtype=torch.float32),
                    "act_absmax": torch.zeros(x.shape[-1], dtype=torch.float32),
                    "tokens": torch.tensor(0, dtype=torch.long),
                    "batches": torch.tensor(0, dtype=torch.long),
                }
            stats[module_name]["hessian_diag"] += x.pow(2).sum(dim=0)
            stats[module_name]["act_absmax"] = torch.maximum(stats[module_name]["act_absmax"], x.abs().amax(dim=0))
            stats[module_name]["tokens"] += x.shape[0]
            stats[module_name]["batches"] += 1

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            model(input_ids)
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    return stats


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_gptq_style_quantization(
    model: nn.Module,
    calibration_stats: Dict[str, Dict[str, torch.Tensor]],
    num_bits: int = 4,
    per_channel: bool = True,
) -> Dict[str, object]:
    """Simplified educational GPTQ-style quantization.

    This computes activation-Hessian-diagonal weighted error statistics before
    replacing Linear layers with fake-quantized layers. It does not implement
    blockwise inverse-Hessian error compensation.
    """

    modules = named_plain_linears(model, calibration_stats.keys())
    layer_errors = []
    quantized = 0
    skipped = 0
    for name, module in list(modules.items()):
        if name not in calibration_stats:
            skipped += 1
            continue
        weight = module.weight.detach().float()
        scale = calculate_qparams_symmetric(weight, num_bits, per_channel=per_channel, channel_dim=0)
        qweight = quantize_tensor_symmetric(weight, scale, num_bits)
        dequant = dequantize_tensor_symmetric(qweight, scale)
        error = quantization_error(weight, dequant)
        hdiag = calibration_stats[name]["hessian_diag"].float().to(weight.device)
        hdiag = hdiag / hdiag.mean().clamp_min(1.0e-8)
        weighted_error = ((weight - dequant.to(weight.device)).pow(2) * hdiag.view(1, -1)).mean()
        layer_errors.append(
            {
                "name": name,
                "weight_mse": error["weight_mse"],
                "weight_mae": error["weight_mae"],
                "weight_max_abs_error": error["weight_max_abs_error"],
                "weighted_error": float(weighted_error.detach().cpu().item()),
                "scale_min": float(scale.detach().float().min().cpu().item()),
                "scale_max": float(scale.detach().float().max().cpu().item()),
                "tokens": int(calibration_stats[name]["tokens"].item()),
                "num_bits": int(num_bits),
                "per_channel": bool(per_channel),
            }
        )
        _replace_module(model, name, QuantizedLinear(module, num_bits=num_bits, per_channel=per_channel, module_name=name))
        quantized += 1
    return {
        "method": "gptq_style",
        "description": "Simplified GPTQ-style weighted-error fake quantization; no inverse-Hessian block compensation.",
        "num_bits": int(num_bits),
        "per_channel": bool(per_channel),
        "calibrated_layers": len(calibration_stats),
        "quantized_count": quantized,
        "skipped_count": skipped,
        "layer_errors": layer_errors,
    }

```

## `minillm/smoothquant.py`

```python
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gptq import _replace_module, named_plain_linears
from .quantization import (
    calculate_qparams_symmetric,
    dequantize_tensor_symmetric,
    quantization_error,
    quantize_tensor_symmetric,
)


def calculate_smooth_scale(
    act_max: torch.Tensor,
    weight_max: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    act = act_max.float().clamp_min(eps)
    weight = weight_max.float().clamp_min(eps)
    scale = act.pow(alpha) / weight.pow(1.0 - alpha)
    return scale.clamp(1.0e-4, 1.0e4)


class SmoothQuantLinear(nn.Module):
    """Educational SmoothQuant-style fake-quant Linear wrapper.

    Forward computes F.linear(x / input_scale, quantized(weight * input_scale)).
    This preserves the algebraic intent while avoiding graph-level fusion.
    """

    def __init__(
        self,
        linear: nn.Linear,
        input_scale: torch.Tensor,
        num_bits: int = 8,
        per_channel: bool = True,
        module_name: str = "",
    ) -> None:
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.num_bits = int(num_bits)
        self.per_channel = bool(per_channel)
        self.module_name = module_name
        weight = linear.weight.detach().float()
        input_scale = input_scale.detach().float().reshape(-1).clamp(1.0e-4, 1.0e4).to(weight.device)
        if input_scale.numel() != weight.shape[1]:
            raise ValueError("input_scale size must match Linear in_features")
        smooth_weight = weight * input_scale.view(1, -1)
        weight_scale = calculate_qparams_symmetric(smooth_weight, num_bits, per_channel=per_channel, channel_dim=0)
        qweight = quantize_tensor_symmetric(smooth_weight, weight_scale, num_bits)
        self.register_buffer("qweight", qweight.to(device=weight.device), persistent=True)
        self.register_buffer("weight_scale", weight_scale.to(device=weight.device), persistent=True)
        self.register_buffer("input_scale", input_scale.to(device=weight.device), persistent=True)
        if linear.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", linear.bias.detach().clone(), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_scale = self.input_scale.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        x_scaled = x / input_scale
        weight = dequantize_tensor_symmetric(self.qweight.to(x.device), self.weight_scale.to(x.device)).to(dtype=x.dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x_scaled, weight, bias)


def collect_smoothquant_stats(
    model: nn.Module,
    dataloader,
    target_modules: Optional[Iterable[str]] = None,
    max_batches: int = 10,
) -> Dict[str, Dict[str, torch.Tensor]]:
    modules = named_plain_linears(model, target_modules)
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []
    for name, module in modules.items():
        weight_max = module.weight.detach().float().abs().amax(dim=0).cpu()
        stats[name] = {
            "act_absmax": torch.zeros(module.in_features, dtype=torch.float32),
            "weight_absmax": weight_max,
            "tokens": torch.tensor(0, dtype=torch.long),
            "batches": torch.tensor(0, dtype=torch.long),
        }

        def make_hook(module_name: str):
            def hook(module, inputs, output):
                x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1]).cpu()
                stats[module_name]["act_absmax"] = torch.maximum(stats[module_name]["act_absmax"], x.abs().amax(dim=0))
                stats[module_name]["tokens"] += x.shape[0]
                stats[module_name]["batches"] += 1

            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            model(input_ids)
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    return stats


def apply_smoothquant(
    model: nn.Module,
    stats: Dict[str, Dict[str, torch.Tensor]],
    alpha: float = 0.5,
    num_bits: int = 8,
    per_channel: bool = True,
) -> Dict[str, object]:
    modules = named_plain_linears(model, stats.keys())
    layer_stats = []
    quantized = 0
    skipped = 0
    for name, module in list(modules.items()):
        if name not in stats:
            skipped += 1
            continue
        act_max = stats[name]["act_absmax"]
        weight_max = stats[name]["weight_absmax"]
        smooth_scale = calculate_smooth_scale(act_max, weight_max, alpha=alpha)
        original = module.weight.detach().float()
        smooth_weight = original * smooth_scale.to(original.device).view(1, -1)
        qscale = calculate_qparams_symmetric(smooth_weight, num_bits, per_channel=per_channel, channel_dim=0)
        qweight = quantize_tensor_symmetric(smooth_weight, qscale, num_bits)
        dequant_smooth = dequantize_tensor_symmetric(qweight, qscale).to(original.device)
        reconstructed = dequant_smooth / smooth_scale.to(original.device).view(1, -1)
        error = quantization_error(original, reconstructed)
        layer_stats.append(
            {
                "name": name,
                "scale_min": float(smooth_scale.min().cpu().item()),
                "scale_max": float(smooth_scale.max().cpu().item()),
                "scale_mean": float(smooth_scale.mean().cpu().item()),
                "weight_mse": error["weight_mse"],
                "weight_mae": error["weight_mae"],
                "weight_max_abs_error": error["weight_max_abs_error"],
                "tokens": int(stats[name]["tokens"].item()),
            }
        )
        _replace_module(model, name, SmoothQuantLinear(module, smooth_scale, num_bits=num_bits, per_channel=per_channel, module_name=name))
        quantized += 1
    return {
        "method": "smoothquant_style",
        "description": "Educational SmoothQuant-style wrapper; no graph fusion or integer kernel deployment.",
        "alpha": float(alpha),
        "num_bits": int(num_bits),
        "per_channel": bool(per_channel),
        "calibrated_layers": len(stats),
        "quantized_count": quantized,
        "skipped_count": skipped,
        "layer_stats": layer_stats,
    }

```

## `scripts/quantize_eval.py`

```python
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

```

## `tests/test_quantization.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.quantization import (
    QuantizedLinear,
    calculate_qparams_symmetric,
    dequantize_tensor_symmetric,
    estimate_quantized_size_bytes,
    quantize_model_weight_only,
    quantize_tensor_symmetric,
)


def test_qparams_and_quant_dequant_shapes() -> None:
    x = torch.randn(4, 8)
    scale = calculate_qparams_symmetric(x, num_bits=8)
    assert torch.isfinite(scale).all()
    q = quantize_tensor_symmetric(x, scale, num_bits=8)
    dq = dequantize_tensor_symmetric(q, scale)
    assert q.dtype == torch.int8
    assert dq.shape == x.shape

    scale4 = calculate_qparams_symmetric(x, num_bits=4, per_channel=True, channel_dim=0)
    assert scale4.shape == (4, 1)
    q4 = quantize_tensor_symmetric(x, scale4, num_bits=4)
    assert int(q4.max()) <= 7
    assert int(q4.min()) >= -7
    assert dequantize_tensor_symmetric(q4, scale4).shape == x.shape


def test_quantized_linear_forward_shape() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(8, 4, bias=True)
    qlinear = QuantizedLinear(linear, num_bits=8, per_channel=True)
    x = torch.randn(2, 3, 8)
    y = qlinear(x)
    assert y.shape == (2, 3, 4)
    assert torch.isfinite(y).all()


def test_quantize_model_weight_only_replaces_linears() -> None:
    config = MiniLLMConfig(
        vocab_size=64,
        context_length=16,
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        intermediate_size=64,
    )
    model = MiniLLMForCausalLM(config)
    stats = quantize_model_weight_only(model, num_bits=4, per_channel=True)
    assert stats["linear_count"] > 0
    assert stats["quantized_count"] == stats["linear_count"]
    assert estimate_quantized_size_bytes(model) > 0
    out = model(torch.randint(0, config.vocab_size, (2, 8)))
    assert out["logits"].shape == (2, 8, config.vocab_size)

```

## `tests/test_gptq.py`

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.gptq import apply_gptq_style_quantization, collect_linear_calibration_stats


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, length: int = 8, count: int = 6) -> None:
        self.vocab_size = vocab_size
        self.length = length
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int):
        torch.manual_seed(idx)
        return {"input_ids": torch.randint(0, self.vocab_size, (self.length,))}


def collate(batch):
    return {"input_ids": torch.stack([item["input_ids"] for item in batch], dim=0)}


def build_model() -> MiniLLMForCausalLM:
    config = MiniLLMConfig(
        vocab_size=64,
        context_length=16,
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        intermediate_size=64,
    )
    return MiniLLMForCausalLM(config)


def test_gptq_calibration_and_quantization_forward_shape() -> None:
    model = build_model()
    loader = DataLoader(RandomTokenDataset(model.config.vocab_size), batch_size=2, collate_fn=collate)
    stats = collect_linear_calibration_stats(model, loader, max_batches=2)
    assert stats
    first = next(iter(stats.values()))
    assert "hessian_diag" in first
    assert torch.isfinite(first["hessian_diag"]).all()

    qstats = apply_gptq_style_quantization(model, stats, num_bits=4)
    assert qstats["quantized_count"] > 0
    assert qstats["layer_errors"]
    assert "weighted_error" in qstats["layer_errors"][0]
    out = model(torch.randint(0, model.config.vocab_size, (2, 8)))
    assert out["logits"].shape == (2, 8, model.config.vocab_size)
    assert torch.isfinite(out["logits"]).all()

```

## `tests/test_smoothquant.py`

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.smoothquant import apply_smoothquant, calculate_smooth_scale, collect_smoothquant_stats


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, length: int = 8, count: int = 6) -> None:
        self.vocab_size = vocab_size
        self.length = length
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int):
        torch.manual_seed(idx + 100)
        return {"input_ids": torch.randint(0, self.vocab_size, (self.length,))}


def collate(batch):
    return {"input_ids": torch.stack([item["input_ids"] for item in batch], dim=0)}


def build_model() -> MiniLLMForCausalLM:
    config = MiniLLMConfig(
        vocab_size=64,
        context_length=16,
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        intermediate_size=64,
    )
    return MiniLLMForCausalLM(config)


def test_smooth_scale_finite_for_different_alpha() -> None:
    act = torch.tensor([1.0, 4.0, 0.0])
    weight = torch.tensor([2.0, 1.0, 0.0])
    for alpha in [0.0, 0.5, 1.0]:
        scale = calculate_smooth_scale(act, weight, alpha=alpha)
        assert scale.shape == act.shape
        assert torch.isfinite(scale).all()


def test_apply_smoothquant_forward_shape_and_finite() -> None:
    model = build_model()
    loader = DataLoader(RandomTokenDataset(model.config.vocab_size), batch_size=2, collate_fn=collate)
    stats = collect_smoothquant_stats(model, loader, max_batches=2)
    assert stats
    qstats = apply_smoothquant(model, stats, alpha=0.5, num_bits=8)
    assert qstats["quantized_count"] > 0
    assert qstats["layer_stats"]
    out = model(torch.randint(0, model.config.vocab_size, (2, 8)))
    assert out["logits"].shape == (2, 8, model.config.vocab_size)
    assert torch.isfinite(out["logits"]).all()

```

## `configs/quant_int8.yaml`

```yaml
seed: 20260712
checkpoint: outputs/sft_full/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
eval_data_path: data/sft/sft_val.jsonl
output_dir: outputs/quant_int8
method: weight_only
num_bits: 8
per_channel: true
eval_max_batches: 10
batch_size: 8
dtype: auto
prefer_cuda: true

```

## `configs/quant_int4.yaml`

```yaml
seed: 20260712
checkpoint: outputs/sft_full/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
eval_data_path: data/sft/sft_val.jsonl
output_dir: outputs/quant_int4
method: weight_only
num_bits: 4
per_channel: true
eval_max_batches: 10
batch_size: 8
dtype: auto
prefer_cuda: true

```

## `configs/quant_gptq_int4.yaml`

```yaml
seed: 20260712
checkpoint: outputs/sft_full/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
eval_data_path: data/sft/sft_val.jsonl
calibration_data_path: data/sft/sft_train.jsonl
output_dir: outputs/quant_gptq_int4
method: gptq_style
num_bits: 4
per_channel: true
calibration_max_batches: 10
eval_max_batches: 10
batch_size: 8
dtype: auto
prefer_cuda: true

```

## `configs/quant_smooth_int8.yaml`

```yaml
seed: 20260712
checkpoint: outputs/sft_full/checkpoints/best.pt
tokenizer_path: data/tokenizers/mixed_tokenizer.json
eval_data_path: data/sft/sft_val.jsonl
calibration_data_path: data/sft/sft_train.jsonl
output_dir: outputs/quant_smooth_int8
method: smoothquant_style
num_bits: 8
per_channel: true
alpha: 0.5
calibration_max_batches: 10
eval_max_batches: 10
batch_size: 8
dtype: auto
prefer_cuda: true

```
