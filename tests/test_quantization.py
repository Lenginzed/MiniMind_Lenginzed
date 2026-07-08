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
