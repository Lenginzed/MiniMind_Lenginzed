from __future__ import annotations

import torch

from minillm.trainer import build_lr_scheduler


def test_cosine_scheduler_warmup_then_decay() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=0.1)
    scheduler = build_lr_scheduler(optimizer, scheduler_name="cosine", warmup_steps=2, max_steps=6)
    assert scheduler is not None
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[0] < lrs[1]
    assert max(lrs) <= 0.1
    assert lrs[-1] < lrs[2]


def test_none_scheduler_returns_none() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=0.1)
    assert build_lr_scheduler(optimizer, scheduler_name="none", warmup_steps=0, max_steps=10) is None
