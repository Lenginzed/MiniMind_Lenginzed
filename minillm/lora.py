from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        module_name: str = "",
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base_layer = base_layer
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.scaling = float(lora_alpha) / float(r)
        self.module_name = module_name
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        for param in self.base_layer.parameters():
            param.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base + update


def freeze_all_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _set_lora_trainable(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True


def apply_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> Tuple[nn.Module, Dict[str, object]]:
    target_set = set(target_modules)
    if freeze_base:
        freeze_all_parameters(model)
    replaced: List[str] = []

    def replace_children(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = "%s.%s" % (prefix, name) if prefix else name
            if name in target_set and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r=r, lora_alpha=alpha, lora_dropout=dropout, module_name=full_name))
                replaced.append(full_name)
            else:
                replace_children(child, full_name)

    replace_children(model)
    _set_lora_trainable(model)
    stats = lora_parameter_stats(model)
    stats.update(
        {
            "target_modules": sorted(target_set),
            "replaced_modules": replaced,
            "r": int(r),
            "alpha": int(alpha),
            "dropout": float(dropout),
        }
    )
    return model, stats


def lora_parameter_stats(model: nn.Module) -> Dict[str, object]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for p in [module.lora_A.weight, module.lora_B.weight]
    )
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "lora_params": int(lora_params),
        "trainable_ratio": float(trainable / total) if total else 0.0,
        "lora_module_count": sum(1 for module in model.modules() if isinstance(module, LoRALinear)),
    }


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state["%s.lora_A.weight" % name] = module.lora_A.weight.detach().cpu()
            state["%s.lora_B.weight" % name] = module.lora_B.weight.detach().cpu()
    return state


def save_lora_adapter(model: nn.Module, path: str, config: Dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora_state_dict": lora_state_dict(model), "config": config}, path)
    json_path = str(Path(path).with_suffix(".json"))
    Path(json_path).write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_lora_adapter(model: nn.Module, path: str, strict: bool = True) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    adapter_state = payload["lora_state_dict"]
    modules = dict(model.named_modules())
    missing = []
    for key, value in adapter_state.items():
        module_name, _, param_name = key.rpartition(".")
        module_name = module_name.rsplit(".", 1)[0]
        short_name = key.split(".")[-2]
        module = modules.get(module_name)
        if not isinstance(module, LoRALinear):
            missing.append(key)
            continue
        if short_name == "lora_A":
            module.lora_A.weight.data.copy_(value.to(module.lora_A.weight.device))
        elif short_name == "lora_B":
            module.lora_B.weight.data.copy_(value.to(module.lora_B.weight.device))
    if strict and missing:
        raise KeyError("missing LoRA modules for keys: %s" % missing)
    return payload.get("config", {})
