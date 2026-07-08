from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _top_k_filter(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)


def _top_p_filter(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits.float(), dim=-1)
    cumulative_probs = probs.cumsum(dim=-1)
    remove_mask = cumulative_probs > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove_mask, torch.finfo(sorted_logits.dtype).min)
    filtered = torch.full_like(logits, torch.finfo(logits.dtype).min)
    return filtered.scatter(dim=-1, index=sorted_indices, src=sorted_logits)


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    finite = torch.finfo(logits.dtype)
    return torch.nan_to_num(
        logits,
        nan=0.0,
        posinf=finite.max,
        neginf=finite.min,
    )


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    do_sample: bool = True,
) -> torch.Tensor:
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    was_training = model.training
    model.eval()
    generated = input_ids
    context_length = getattr(getattr(model, "config", None), "context_length", None)

    try:
        for _ in range(max_new_tokens):
            model_input = generated
            if context_length is not None and model_input.shape[1] > context_length:
                model_input = model_input[:, -context_length:]
            outputs = model(model_input)
            logits = outputs["logits"][:, -1, :]
            logits = _sanitize_logits(logits)

            if (not do_sample) or temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                safe_temperature = max(float(temperature), 1e-6)
                filtered = logits / safe_temperature
                filtered = _top_k_filter(filtered, top_k)
                filtered = _top_p_filter(filtered, top_p)
                probs = F.softmax(filtered.float(), dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                row_sums = probs.sum(dim=-1, keepdim=True)
                bad_rows = row_sums.squeeze(-1) <= 0
                if bad_rows.any():
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    sample_rows = ~bad_rows
                    if sample_rows.any():
                        sampled = torch.multinomial(
                            probs[sample_rows] / row_sums[sample_rows].clamp_min(1e-12),
                            num_samples=1,
                        )
                        next_token[sample_rows] = sampled
                else:
                    probs = probs / row_sums
                    next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
    finally:
        if was_training:
            model.train()
    return generated
