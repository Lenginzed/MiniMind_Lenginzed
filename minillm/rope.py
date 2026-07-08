from __future__ import annotations

from typing import Optional, Tuple

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    compute_dtype = torch.float32
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=compute_dtype) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=compute_dtype)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    if dtype is not None:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k tensors shaped [batch, heads, seq, head_dim]."""
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head_dim for RoPE")
    if q.shape[-1] % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if cos.shape[-1] != q.shape[-1] or sin.shape[-1] != q.shape[-1]:
        raise ValueError("cos/sin last dimension must match q/k head_dim")
    if position_ids is None:
        seq_len = q.shape[-2]
        if cos.shape[-2] < seq_len or sin.shape[-2] < seq_len:
            raise ValueError("RoPE cache is shorter than sequence length")
        cos_pos = cos[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin_pos = sin[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
    else:
        if position_ids.dim() != 2:
            raise ValueError("position_ids must have shape [batch, seq]")
        if position_ids.shape[-1] != q.shape[-2]:
            raise ValueError("position_ids seq length must match q/k seq length")
        flat_pos = position_ids.reshape(-1).to(device=cos.device)
        cos_gathered = cos.squeeze(0).squeeze(0).index_select(0, flat_pos)
        sin_gathered = sin.squeeze(0).squeeze(0).index_select(0, flat_pos)
        batch, seq_len = position_ids.shape
        cos_pos = cos_gathered.view(batch, seq_len, -1)[:, None, :, :].to(
            dtype=q.dtype, device=q.device
        )
        sin_pos = sin_gathered.view(batch, seq_len, -1)[:, None, :, :].to(
            dtype=q.dtype, device=q.device
        )

    q_embed = (q * cos_pos) + (rotate_half(q) * sin_pos)
    k_embed = (k * cos_pos) + (rotate_half(k) * sin_pos)
    return q_embed, k_embed
