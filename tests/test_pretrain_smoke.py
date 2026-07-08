from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM


def test_tiny_pretrain_one_batch_forward_backward_cpu() -> None:
    torch.manual_seed(123)
    config = MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
    )
    model = MiniLLMForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (4, config.context_length))
    outputs = model(input_ids, labels=input_ids.clone())
    loss = outputs["loss"]
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    finite_grads = [torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None]
    assert finite_grads and all(finite_grads)
    optimizer.step()
