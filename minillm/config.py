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
