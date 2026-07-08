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
