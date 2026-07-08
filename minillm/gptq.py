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
