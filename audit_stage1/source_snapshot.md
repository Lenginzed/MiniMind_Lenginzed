# Stage 1 Source Snapshot

## `minillm/config.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiniLLMConfig:
    vocab_size: int = 128
    context_length: int = 32
    n_layer: int = 2
    n_embd: int = 64
    n_head: int = 4
    n_kv_head: int = 2
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    use_gradient_checkpointing: bool = False

    @property
    def max_position_embeddings(self) -> int:
        return self.context_length

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_kv_head <= 0:
            raise ValueError("n_kv_head must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
```

## `minillm/rope.py`

```python
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
```

## `minillm/modules.py`

```python
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
```

## `minillm/model.py`

```python
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MiniLLMConfig
from .modules import DecoderBlock, RMSNorm


class MiniLLMForCausalLM(nn.Module):
    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input length exceeds config.context_length")

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in parameters)
```

## `minillm/generation.py`

```python
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
```

## `scripts/smoke_model_forward.py`

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM, generate
from minillm.model import count_parameters


def build_tiny_config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
    )


def main() -> int:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = build_tiny_config()
    model = MiniLLMForCausalLM(config).to(device)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print("device:", device)
    print("tiny config:", config)
    print("parameter count:", count_parameters(model))

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs["loss"]
    logits = outputs["logits"]
    if loss is None:
        raise RuntimeError("loss was not computed")

    print("logits shape:", tuple(logits.shape))
    print("loss:", float(loss.detach().cpu().item()))
    print("loss finite:", bool(torch.isfinite(loss).detach().cpu().item()))
    loss.backward()
    print("backward: ok")
    finite_grads = []
    for param in model.parameters():
        if param.grad is not None:
            finite_grads.append(bool(torch.isfinite(param.grad).all().detach().cpu().item()))
    print("finite gradient check:", bool(finite_grads and all(finite_grads)))

    prompt = input_ids[:, :4]
    generated = generate(
        model,
        prompt,
        max_new_tokens=4,
        temperature=1.0,
        top_k=16,
        top_p=0.9,
        do_sample=True,
    )
    print("generated shape:", tuple(generated.shape))
    print("generated first row:", generated[0].detach().cpu().tolist())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print("cuda memory allocated bytes:", torch.cuda.memory_allocated(device))
        print("cuda memory reserved bytes:", torch.cuda.memory_reserved(device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `tests/test_model_shapes.py`

```python
from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.model import count_parameters
from minillm.modules import GQASelfAttention, repeat_kv


def tiny_config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
    )


def test_forward_logits_loss_and_backward_cpu() -> None:
    torch.manual_seed(0)
    config = tiny_config()
    model = MiniLLMForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    outputs = model(input_ids=input_ids, labels=input_ids.clone())
    assert outputs["logits"].shape == (2, 12, config.vocab_size)
    assert outputs["loss"] is not None
    assert torch.isfinite(outputs["loss"])
    outputs["loss"].backward()
    assert model.embed_tokens.weight.grad is not None
    finite_grad_count = 0
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
            finite_grad_count += 1
    assert finite_grad_count > 0
    assert count_parameters(model) > 0


def test_loss_supports_ignore_index() -> None:
    config = tiny_config()
    model = MiniLLMForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    labels = input_ids.clone()
    labels[0, 3] = -100
    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs["loss"] is not None
    assert torch.isfinite(outputs["loss"])


def test_tied_embeddings_share_weight() -> None:
    model = MiniLLMForCausalLM(tiny_config())
    assert model.lm_head.weight is model.embed_tokens.weight


def test_untied_embeddings_do_not_share_weight() -> None:
    config = tiny_config()
    config.tie_word_embeddings = False
    model = MiniLLMForCausalLM(config)
    assert model.lm_head.weight is not model.embed_tokens.weight


def test_invalid_gqa_ratio_raises() -> None:
    try:
        MiniLLMConfig(
            vocab_size=128,
            context_length=32,
            n_layer=2,
            n_embd=64,
            n_head=6,
            n_kv_head=4,
            intermediate_size=128,
        )
    except ValueError:
        return
    raise AssertionError("MiniLLMConfig should reject n_head not divisible by n_kv_head")


def test_odd_head_dim_raises() -> None:
    try:
        MiniLLMConfig(
            vocab_size=128,
            context_length=32,
            n_layer=2,
            n_embd=60,
            n_head=4,
            n_kv_head=2,
            intermediate_size=128,
        )
    except ValueError:
        return
    raise AssertionError("MiniLLMConfig should reject odd head_dim")


def test_repeat_kv_shape_and_values() -> None:
    x = torch.arange(2 * 2 * 3 * 4).view(2, 2, 3, 4)
    repeated = repeat_kv(x, n_rep=2)
    assert repeated.shape == (2, 4, 3, 4)
    assert torch.equal(repeated[:, 0], x[:, 0])
    assert torch.equal(repeated[:, 1], x[:, 0])
    assert torch.equal(repeated[:, 2], x[:, 1])
    assert torch.equal(repeated[:, 3], x[:, 1])


def test_attention_output_shape_cpu() -> None:
    config = tiny_config()
    attn = GQASelfAttention(config)
    x = torch.randn(2, 7, config.n_embd)
    y = attn(x)
    assert y.shape == (2, 7, config.n_embd)
    assert torch.isfinite(y).all()


def test_tiny_parameter_count_exact() -> None:
    model = MiniLLMForCausalLM(tiny_config())
    assert count_parameters(model) == 82240
```

## `tests/test_rope.py`

```python
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
```

## `tests/test_generation.py`

```python
from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM, generate
from minillm.generation import _sanitize_logits, _top_k_filter, _top_p_filter


def tiny_model() -> MiniLLMForCausalLM:
    torch.manual_seed(123)
    config = MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
    )
    return MiniLLMForCausalLM(config)


def test_generate_greedy_length_growth() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    output = generate(model, input_ids, max_new_tokens=5, do_sample=False)
    assert output.shape == (2, 9)
    assert int(output.max()) < model.config.vocab_size
    assert int(output.min()) >= 0


def test_generate_temperature_zero_is_greedy() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    output_a = generate(model, input_ids, max_new_tokens=2, temperature=0.0, do_sample=True)
    output_b = generate(model, input_ids, max_new_tokens=2, do_sample=False)
    assert torch.equal(output_a, output_b)


def test_generate_sampling_top_k_top_p_length_growth() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    output = generate(
        model,
        input_ids,
        max_new_tokens=3,
        temperature=0.8,
        top_k=16,
        top_p=0.9,
        do_sample=True,
    )
    assert output.shape == (2, 7)
    assert int(output.max()) < model.config.vocab_size
    assert int(output.min()) >= 0


def test_top_k_filter_keeps_only_k_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    filtered = _top_k_filter(logits, top_k=2)
    masked_value = torch.finfo(filtered.dtype).min
    assert filtered[0, 0].item() == masked_value
    assert filtered[0, 1].item() == masked_value
    assert filtered[0, 2].item() == 3.0
    assert filtered[0, 3].item() == 4.0


def test_top_p_filter_keeps_at_least_one_token() -> None:
    logits = torch.tensor([[10.0, 1.0, 0.0, -1.0]])
    filtered = _top_p_filter(logits, top_p=0.01)
    masked_value = torch.finfo(filtered.dtype).min
    kept = (filtered != masked_value).sum().item()
    assert kept >= 1
    assert filtered[0, 0].item() == 10.0


class ConstantLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int, token_id: int) -> None:
        super().__init__()
        self.config = MiniLLMConfig(vocab_size=vocab_size, context_length=16)
        self.token_id = token_id

    def forward(self, input_ids: torch.Tensor):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.config.vocab_size)
        logits[..., self.token_id] = 100.0
        return {"logits": logits.to(device=input_ids.device)}


def test_generate_eos_early_stop() -> None:
    model = ConstantLogitModel(vocab_size=16, token_id=3)
    input_ids = torch.tensor([[1, 2]])
    output = generate(model, input_ids, max_new_tokens=5, eos_token_id=3, do_sample=False)
    assert output.shape == (1, 3)
    assert output[0, -1].item() == 3


def test_sanitize_logits_handles_nan_inf() -> None:
    logits = torch.tensor([[float("nan"), float("inf"), float("-inf"), 1.0]])
    sanitized = _sanitize_logits(logits)
    assert torch.isfinite(sanitized).all()
    assert sanitized[0, 1] > sanitized[0, 3]
```

## `tests/test_attention_causality.py`

```python
from __future__ import annotations

import torch

from minillm import MiniLLMConfig, MiniLLMForCausalLM
from minillm.modules import make_causal_mask


def tiny_config() -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=128,
        context_length=32,
        n_layer=2,
        n_embd=64,
        n_head=4,
        n_kv_head=2,
        intermediate_size=128,
        dropout=0.0,
    )


def test_make_causal_mask_blocks_future_tokens() -> None:
    mask = make_causal_mask(q_len=4, k_len=4, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask.cpu(), expected)


def test_future_token_perturbation_does_not_change_past_logits() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = MiniLLMForCausalLM(config)
    model.eval()

    prefix = torch.tensor([[5, 6, 7, 8]])
    suffix_a = torch.tensor([[9, 10, 11, 12]])
    suffix_b = torch.tensor([[100, 101, 102, 103]])
    input_a = torch.cat([prefix, suffix_a], dim=1)
    input_b = torch.cat([prefix, suffix_b], dim=1)

    with torch.no_grad():
        logits_a = model(input_a)["logits"]
        logits_b = model(input_b)["logits"]

    assert torch.allclose(logits_a[:, : prefix.shape[1], :], logits_b[:, : prefix.shape[1], :], atol=1e-5)
```
