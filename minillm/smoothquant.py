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
