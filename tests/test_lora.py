from __future__ import annotations

import torch
import torch.nn as nn

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.lora import LoRALinear, apply_lora, lora_parameter_stats


def test_lora_linear_shape_and_zero_b_equivalence() -> None:
    torch.manual_seed(0)
    base = nn.Linear(8, 4, bias=False)
    x = torch.randn(2, 3, 8)
    expected = base(x)
    wrapped = LoRALinear(base, r=2, lora_alpha=4, lora_dropout=0.0)
    actual = wrapped(x)
    assert actual.shape == (2, 3, 4)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_apply_lora_only_targets_q_v_and_freezes_base() -> None:
    config = MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
    )
    model = MiniLLMForCausalLM(config)
    model, stats = apply_lora(model, target_modules=["q_proj", "v_proj"], r=4, alpha=8, dropout=0.0)
    replaced = stats["replaced_modules"]
    assert replaced
    assert all(name.endswith("q_proj") or name.endswith("v_proj") for name in replaced)
    assert not any(name.endswith("k_proj") or name.endswith("o_proj") for name in replaced)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all("lora_A" in name or "lora_B" in name for name in trainable_names)
    stats = lora_parameter_stats(model)
    assert stats["trainable_params"] < stats["total_params"]
    assert stats["lora_module_count"] == 4
