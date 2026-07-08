# Stage 1 Model Design

## Implemented Modules

The Stage 1 skeleton implements a decoder-only Causal LM in `minillm/`:

- `MiniLLMConfig` in `config.py`
- `RMSNorm`, `SwiGLU`, `GQASelfAttention`, `DecoderBlock` in `modules.py`
- RoPE cache and application helpers in `rope.py`
- `MiniLLMForCausalLM` in `model.py`
- `generate` with greedy, temperature, top-k, and top-p decoding in `generation.py`

## Transformer Block Layout

Each decoder block uses:

1. RMSNorm before attention
2. GQA self-attention with RoPE-applied q/k
3. Residual connection
4. RMSNorm before MLP
5. SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`
6. Residual connection

GQA uses `n_head` query heads and `n_kv_head` key/value heads. `n_head` must be divisible by `n_kv_head`; key/value heads are expanded with `repeat_kv`.

Attention prefers PyTorch `scaled_dot_product_attention` when available and falls back to a manual causal attention implementation otherwise.

## Forward And Loss

`MiniLLMForCausalLM.forward(input_ids, labels=None)` returns a dict:

- `logits`: `[batch, seq, vocab_size]`
- `loss`: causal LM cross entropy if labels are provided

Loss uses standard shifting:

- logits: `logits[:, :-1, :]`
- labels: `labels[:, 1:]`
- `ignore_index=-100`

Token embeddings and `lm_head` can be tied through `tie_word_embeddings=True`.

## Generation

`generate(...)` currently recomputes the full prefix each decoding step. It supports:

- greedy decoding with `do_sample=False` or `temperature=0`
- temperature scaling
- top-k filtering
- top-p nucleus filtering
- optional EOS early stop
- NaN/Inf sanitization before sampling

## Tiny Smoke Config

Stage 1 tests must use this tiny config:

- vocab_size: 128
- context_length: 32
- n_layer: 2
- n_embd: 64
- n_head: 4
- n_kv_head: 2
- intermediate_size: 128

Current skeleton parameter count: `82,240`.

## Main Target 40M-50M Config

Recommended first full project target:

- vocab_size: 20000
- context_length: 512
- n_layer: 10
- n_embd: 576
- n_head: 9
- n_kv_head: 3
- intermediate_size: 1728
- micro_batch_size: 4 for pretrain/SFT at context 512
- gradient_accumulation_steps: 8-16
- DPO micro_batch_size: 1-2
- GRPO: tiny toy run first, with very small completion counts

Current skeleton parameter count for this config: `50,239,296`.

The previous 60M idea remains a stretch target after the full loop is stable. Suggested stretch config:

- vocab_size: 24000
- context_length: 1024
- n_layer: 10
- n_embd: 640
- n_head: 10
- n_kv_head: 2
- intermediate_size: 1792

Current skeleton parameter count for this stretch config: `59,610,240`.

## Tests

Implemented tests:

- `tests/test_model_shapes.py`: forward shape, loss, backward, tied embeddings, GQA ratio validation, `repeat_kv`
- `tests/test_rope.py`: RoPE shape preservation and finite values
- `tests/test_generation.py`: greedy and sampled generation shape/length checks

Verification status in `YSJAirCombat`:

- `scripts/verify_ysj_env.py --write-audit`: passed
- `scripts/smoke_model_forward.py`: passed on CUDA, parameter count `82,240`
- `python -m py_compile ...`: passed
- `python -m pytest -q`: passed in Stage 1.1, `23 passed`

Smoke script:

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\smoke_model_forward.py
```

Pytest command once pytest is installed:

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' -m pytest -q
```

## Not Implemented Yet

- KV cache / `past_key_values`
- FlashAttention integration
- real tokenizer
- data pipeline
- training loop
- checkpoint save/resume
- LoRA
- DPO
- GRPO
- GPTQ / SmoothQuant quantization experiments
