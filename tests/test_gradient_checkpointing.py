from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM


def run_one_backward(use_checkpointing: bool) -> None:
    torch.manual_seed(11)
    config = MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
        use_gradient_checkpointing=use_checkpointing,
    )
    model = MiniLLMForCausalLM(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    outputs = model(input_ids, labels=input_ids.clone())
    loss = outputs["loss"]
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_gradient_checkpointing_true_forward_backward() -> None:
    run_one_backward(True)


def test_gradient_checkpointing_false_forward_backward() -> None:
    run_one_backward(False)
