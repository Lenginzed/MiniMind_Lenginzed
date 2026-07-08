from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.smoothquant import apply_smoothquant, calculate_smooth_scale, collect_smoothquant_stats


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, length: int = 8, count: int = 6) -> None:
        self.vocab_size = vocab_size
        self.length = length
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int):
        torch.manual_seed(idx + 100)
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


def test_smooth_scale_finite_for_different_alpha() -> None:
    act = torch.tensor([1.0, 4.0, 0.0])
    weight = torch.tensor([2.0, 1.0, 0.0])
    for alpha in [0.0, 0.5, 1.0]:
        scale = calculate_smooth_scale(act, weight, alpha=alpha)
        assert scale.shape == act.shape
        assert torch.isfinite(scale).all()


def test_apply_smoothquant_forward_shape_and_finite() -> None:
    model = build_model()
    loader = DataLoader(RandomTokenDataset(model.config.vocab_size), batch_size=2, collate_fn=collate)
    stats = collect_smoothquant_stats(model, loader, max_batches=2)
    assert stats
    qstats = apply_smoothquant(model, stats, alpha=0.5, num_bits=8)
    assert qstats["quantized_count"] > 0
    assert qstats["layer_stats"]
    out = model(torch.randint(0, model.config.vocab_size, (2, 8)))
    assert out["logits"].shape == (2, 8, model.config.vocab_size)
    assert torch.isfinite(out["logits"]).all()
