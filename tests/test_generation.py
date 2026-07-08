from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM, generate
from minillm.generation import _sanitize_logits, _top_k_filter, _top_p_filter


def tiny_model() -> MiniLLMForCausalLM:
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
    return MiniLLMForCausalLM(config)


def test_generate_greedy_length_growth() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    output = generate(model, input_ids, max_new_tokens=5, do_sample=False)
    assert output.shape == (2, 9)
    assert int(output.max()) < model.config.vocab_size
    assert int(output.min()) >= 0


def test_generate_temperature_zero_is_greedy() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    output_a = generate(model, input_ids, max_new_tokens=2, temperature=0.0, do_sample=True)
    output_b = generate(model, input_ids, max_new_tokens=2, do_sample=False)
    assert torch.equal(output_a, output_b)


def test_generate_sampling_top_k_top_p_length_growth() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    output = generate(
        model,
        input_ids,
        max_new_tokens=3,
        temperature=0.8,
        top_k=16,
        top_p=0.9,
        do_sample=True,
    )
    assert output.shape == (2, 7)
    assert int(output.max()) < model.config.vocab_size
    assert int(output.min()) >= 0


def test_top_k_filter_keeps_only_k_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    filtered = _top_k_filter(logits, top_k=2)
    masked_value = torch.finfo(filtered.dtype).min
    assert filtered[0, 0].item() == masked_value
    assert filtered[0, 1].item() == masked_value
    assert filtered[0, 2].item() == 3.0
    assert filtered[0, 3].item() == 4.0


def test_top_p_filter_keeps_at_least_one_token() -> None:
    logits = torch.tensor([[10.0, 1.0, 0.0, -1.0]])
    filtered = _top_p_filter(logits, top_p=0.01)
    masked_value = torch.finfo(filtered.dtype).min
    kept = (filtered != masked_value).sum().item()
    assert kept >= 1
    assert filtered[0, 0].item() == 10.0


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int, token_id: int) -> None:
        super().__init__()
        self.config = MiniLLMConfig(vocab_size=vocab_size, context_length=16)
        self.token_id = token_id

    def forward(self, input_ids: torch.Tensor):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.config.vocab_size)
        logits[..., self.token_id] = 100.0
        return {"logits": logits.to(device=input_ids.device)}


def test_generate_eos_early_stop() -> None:
    model = ConstantLogitModel(vocab_size=16, token_id=3)
    input_ids = torch.tensor([[1, 2]])
    output = generate(model, input_ids, max_new_tokens=5, eos_token_id=3, do_sample=False)
    assert output.shape == (1, 3)
    assert output[0, -1].item() == 3


def test_sanitize_logits_handles_nan_inf() -> None:
    logits = torch.tensor([[float("nan"), float("inf"), float("-inf"), 1.0]])
    sanitized = _sanitize_logits(logits)
    assert torch.isfinite(sanitized).all()
    assert sanitized[0, 1] > sanitized[0, 3]
