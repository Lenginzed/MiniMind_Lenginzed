from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MiniLLMConfig
from .rope import apply_rotary_pos_emb, build_rope_cache


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x_norm).to(dtype=x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads from [B, H_kv, T, D] to [B, H_q, T, D]."""
    if n_rep == 1:
        return hidden_states
    batch, n_kv_head, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, n_kv_head, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, n_kv_head * n_rep, seq_len, head_dim)


def make_causal_mask(
    q_len: int,
    k_len: int,
    device: torch.device,
) -> torch.Tensor:
    if q_len <= 0 or k_len <= 0:
        raise ValueError("q_len and k_len must be positive")
    row = torch.arange(q_len, device=device)[:, None]
    col = torch.arange(k_len, device=device)[None, :]
    return col <= row + (k_len - q_len)


class GQASelfAttention(nn.Module):
    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_rep = config.n_head // config.n_kv_head

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        cos, sin = build_rope_cache(
            config.context_length,
            self.head_dim,
            theta=config.rope_theta,
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _shape(self, x: torch.Tensor, n_head: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, n_head, self.head_dim).transpose(1, 2)

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = make_causal_mask(q_len, k_len, q.device)
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        probs = self.attn_dropout(probs)
        return torch.matmul(probs, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        if seq_len > self.config.context_length:
            raise ValueError("sequence length exceeds config.context_length")
        if hidden != self.config.n_embd:
            raise ValueError("hidden size mismatch")

        q = self._shape(self.q_proj(x), self.n_head)
        k = self._shape(self.k_proj(x), self.n_kv_head)
        v = self._shape(self.v_proj(x), self.n_kv_head)

        q, k = apply_rotary_pos_emb(q, k, self.rope_cos, self.rope_sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # TODO: add past_key_values/KV cache support after the basic model is stable.
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_output = self._manual_attention(q, k, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.resid_dropout(self.o_proj(attn_output))


class DecoderBlock(nn.Module):
    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.self_attn = GQASelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.dropout(self.mlp(self.post_attention_layernorm(x)))
        return x
