from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.model import count_parameters
from minillm.modules import GQASelfAttention, repeat_kv


def tiny_config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
    )


def test_forward_logits_loss_and_backward_cpu() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    model = MiniLLMForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    outputs = model(input_ids=input_ids, labels=input_ids.clone())
    assert outputs["logits"].shape == (2, 12, config.vocab_size)
    assert outputs["loss"] is not None
    assert torch.isfinite(outputs["loss"])
    outputs["loss"].backward()
    assert model.embed_tokens.weight.grad is not None
    finite_grad_count = 0
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
            finite_grad_count += 1
    assert finite_grad_count > 0
    assert count_parameters(model) > 0


def test_loss_supports_ignore_index() -> None:
    config = tiny_config()
    model = MiniLLMForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    labels = input_ids.clone()
    labels[0, 3] = -100
    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs["loss"] is not None
    assert torch.isfinite(outputs["loss"])


def test_tied_embeddings_share_weight() -> None:
    model = MiniLLMForCausalLM(tiny_config())
    assert model.lm_head.weight is model.embed_tokens.weight


def test_untied_embeddings_do_not_share_weight() -> None:
    config = tiny_config()
    config.tie_word_embeddings = False
    model = MiniLLMForCausalLM(config)
    assert model.lm_head.weight is not model.embed_tokens.weight


def test_invalid_gqa_ratio_raises() -> None:
    try:
        MiniLLMConfig(
            vocab_size=128,
            context_length=32,
            n_layer=2,
            n_embd=64,
            n_head=6,
            n_kv_head=4,
            intermediate_size=128,
        )
    except ValueError:
        return
    raise AssertionError("MiniLLMConfig should reject n_head not divisible by n_kv_head")


def test_odd_head_dim_raises() -> None:
    try:
        MiniLLMConfig(
            vocab_size=128,
            context_length=32,
            n_layer=2,
            n_embd=60,
            n_head=4,
            n_kv_head=2,
            intermediate_size=128,
        )
    except ValueError:
        return
    raise AssertionError("MiniLLMConfig should reject odd head_dim")


def test_repeat_kv_shape_and_values() -> None:
    x = torch.arange(2 * 2 * 3 * 4).view(2, 2, 3, 4)
    repeated = repeat_kv(x, n_rep=2)
    assert repeated.shape == (2, 4, 3, 4)
    assert torch.equal(repeated[:, 0], x[:, 0])
    assert torch.equal(repeated[:, 1], x[:, 0])
    assert torch.equal(repeated[:, 2], x[:, 1])
    assert torch.equal(repeated[:, 3], x[:, 1])


def test_attention_output_shape_cpu() -> None:
    config = tiny_config()
    attn = GQASelfAttention(config)
    x = torch.randn(2, 7, config.n_embd)
    y = attn(x)
    assert y.shape == (2, 7, config.n_embd)
    assert torch.isfinite(y).all()


def test_tiny_parameter_count_exact() -> None:
    model = MiniLLMForCausalLM(tiny_config())
    assert count_parameters(model) == 82240
