from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minillm.dpo_trainer import dpo_loss, sequence_logps


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, input_ids: torch.Tensor):
        return {"logits": self.logits.expand(input_ids.shape[0], -1, -1).clone()}


def test_sequence_logps_only_counts_unmasked_response_tokens() -> None:
    vocab_size = 5
    logits = torch.zeros(1, 4, vocab_size)
    logits[0, 1, 2] = 4.0
    logits[0, 2, 3] = 5.0
    model = FixedLogitModel(logits)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    labels = torch.tensor([[-100, -100, 2, 3]])

    seq_logps, token_counts, mean_logps = sequence_logps(model, input_ids, labels)
    expected = F.log_softmax(logits[0, 1], dim=-1)[2] + F.log_softmax(logits[0, 2], dim=-1)[3]
    assert seq_logps.shape == (1,)
    assert token_counts.tolist() == [2]
    assert torch.allclose(seq_logps[0], expected)
    assert torch.allclose(mean_logps[0], expected / 2)


def test_dpo_loss_finite_and_prefers_better_policy_margin() -> None:
    result = dpo_loss(
        policy_chosen_logps=torch.tensor([5.0, 4.0]),
        policy_rejected_logps=torch.tensor([1.0, 0.5]),
        ref_chosen_logps=torch.tensor([2.0, 2.0]),
        ref_rejected_logps=torch.tensor([1.5, 1.0]),
        beta=0.1,
    )
    assert torch.isfinite(result["loss"])
    assert result["chosen_rewards"].shape == (2,)
    assert result["rejected_rewards"].shape == (2,)
    assert result["preference_accuracy"].mean().item() == 1.0
    assert result["reward_margin"].mean().item() > 0
