from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minillm.grpo_trainer import build_prompt_completion_batch, grpo_loss, token_logps_for_labels


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, input_ids: torch.Tensor):
        return {"logits": self.logits.expand(input_ids.shape[0], -1, -1).clone()}


def test_build_prompt_completion_batch_masks_prompt() -> None:
    batch = build_prompt_completion_batch([[1, 2, 3]], [[4, 5]], pad_token_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]
    assert batch["response_mask"].tolist() == [[False, False, True, True]]
    assert batch["completion_token_count"].tolist() == [2]


def test_token_logps_for_labels_only_response_tokens() -> None:
    vocab_size = 8
    logits = torch.zeros(1, 5, vocab_size)
    logits[0, 2, 4] = 5.0
    logits[0, 3, 5] = 6.0
    model = FixedLogitModel(logits)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    token_logps, mask = token_logps_for_labels(model, input_ids, labels)
    expected = F.log_softmax(logits[0, 2], dim=-1)[4] + F.log_softmax(logits[0, 3], dim=-1)[5]
    assert mask.tolist() == [[False, False, True, True]]
    assert torch.allclose(token_logps.sum(), expected)


def test_grpo_clipped_loss_finite_and_shapes() -> None:
    old = torch.zeros(2, 3)
    new = torch.tensor([[0.1, 0.2, 0.0], [-0.1, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    advantages = torch.tensor([1.0, -1.0])
    out = grpo_loss(new, old, mask, advantages, clip_epsilon=0.2)
    assert torch.isfinite(out["loss"])
    assert out["response_token_count"].tolist() == [2.0, 1.0]
    out["loss"].backward()
    assert new.grad is not None


def test_grpo_loss_all_zero_advantage_does_not_crash() -> None:
    old = torch.zeros(2, 3)
    new = torch.zeros(2, 3, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    advantages = torch.zeros(2)
    out = grpo_loss(new, old, mask, advantages, clip_epsilon=0.2)
    assert torch.isfinite(out["loss"])
    assert abs(out["loss"].item()) < 1e-8
    out["loss"].backward()
    assert new.grad is not None
