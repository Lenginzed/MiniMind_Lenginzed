from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.gptq import apply_gptq_style_quantization, collect_linear_calibration_stats


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, length: int = 8, count: int = 6) -> None:
        self.vocab_size = vocab_size
        self.length = length
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int):
        torch.manual_seed(idx)
        return {"input_ids": torch.randint(0, self.vocab_size, (self.length,))}


def collate(batch):
    return {"input_ids": torch.stack([item["input_ids"] for item in batch], dim=0)}


def build_model() -> MiniLLMForCausalLM:
    config = MiniLLMConfig(
        vocab_size=64,
        context_length=16,
        n_layer=1,
        n_embd=32,
        n_head=4,
        n_kv_head=2,
        intermediate_size=64,
    )
    return MiniLLMForCausalLM(config)


def test_gptq_calibration_and_quantization_forward_shape() -> None:
    model = build_model()
    loader = DataLoader(RandomTokenDataset(model.config.vocab_size), batch_size=2, collate_fn=collate)
    stats = collect_linear_calibration_stats(model, loader, max_batches=2)
    assert stats
    first = next(iter(stats.values()))
    assert "hessian_diag" in first
    assert torch.isfinite(first["hessian_diag"]).all()

    qstats = apply_gptq_style_quantization(model, stats, num_bits=4)
    assert qstats["quantized_count"] > 0
    assert qstats["layer_errors"]
    assert "weighted_error" in qstats["layer_errors"][0]
    out = model(torch.randint(0, model.config.vocab_size, (2, 8)))
    assert out["logits"].shape == (2, 8, model.config.vocab_size)
    assert torch.isfinite(out["logits"]).all()
