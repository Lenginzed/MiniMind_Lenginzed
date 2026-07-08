from __future__ import annotations

import torch

from minillm.grpo_trainer import compute_group_advantages


def test_group_advantage_mean_zero_for_nonconstant_groups() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    out = compute_group_advantages(rewards, group_size=3)
    adv = out["advantages"].view(2, 3)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert out["group_reward_std"].shape == (2,)
    assert out["frac_reward_zero_std"].item() == 0.0


def test_zero_std_group_advantage_is_zero_and_recorded() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    out = compute_group_advantages(rewards, group_size=3)
    adv = out["advantages"].view(2, 3)
    assert torch.allclose(adv[0], torch.zeros(3))
    assert out["frac_reward_zero_std"].item() == 0.5


def test_invalid_group_size_raises() -> None:
    try:
        compute_group_advantages(torch.tensor([1.0, 2.0, 3.0]), group_size=2)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")
