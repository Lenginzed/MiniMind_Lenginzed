from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.modules import make_causal_mask


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


def test_make_causal_mask_blocks_future_tokens() -> None:
    mask = make_causal_mask(q_len=4, k_len=4, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask.cpu(), expected)


def test_future_token_perturbation_does_not_change_past_logits() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = MiniLLMForCausalLM(config)
    model.eval()

    prefix = torch.tensor([[5, 6, 7, 8]])
    suffix_a = torch.tensor([[9, 10, 11, 12]])
    suffix_b = torch.tensor([[100, 101, 102, 103]])
    input_a = torch.cat([prefix, suffix_a], dim=1)
    input_b = torch.cat([prefix, suffix_b], dim=1)

    with torch.no_grad():
        logits_a = model(input_a)["logits"]
        logits_b = model(input_b)["logits"]

    assert torch.allclose(logits_a[:, : prefix.shape[1], :], logits_b[:, : prefix.shape[1], :], atol=1e-5)
