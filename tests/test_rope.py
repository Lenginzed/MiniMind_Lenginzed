from __future__ import annotations

import torch

from minillm.rope import apply_rotary_pos_emb, build_rope_cache


def test_rope_shape_and_finiteness() -> None:
    batch = 2
    heads = 4
    seq_len = 8
    head_dim = 16
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    cos, sin = build_rope_cache(seq_len, head_dim, dtype=q.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.isfinite(q_rot).all()
    assert torch.isfinite(k_rot).all()


def test_rope_cache_shapes_for_different_lengths() -> None:
    for seq_len in [1, 3, 17]:
        cos, sin = build_rope_cache(seq_len, head_dim=8)
        assert cos.shape == (1, 1, seq_len, 8)
        assert sin.shape == (1, 1, seq_len, 8)
        assert torch.isfinite(cos).all()
        assert torch.isfinite(sin).all()


def test_rope_odd_head_dim_raises() -> None:
    try:
        build_rope_cache(seq_len=4, head_dim=7)
    except ValueError:
        return
    raise AssertionError("build_rope_cache should reject odd head_dim")


def test_apply_rope_cache_too_short_raises() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    cos, sin = build_rope_cache(3, 8)
    try:
        apply_rotary_pos_emb(q, k, cos, sin)
    except ValueError:
        return
    raise AssertionError("apply_rotary_pos_emb should reject too-short cache")


def test_rope_position_ids_shape() -> None:
    batch = 2
    heads = 2
    seq_len = 5
    head_dim = 8
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    cos, sin = build_rope_cache(16, head_dim, dtype=q.dtype)
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [3, 4, 5, 6, 7]])
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.isfinite(q_rot).all()
    assert torch.isfinite(k_rot).all()
