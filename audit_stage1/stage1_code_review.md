# Stage 1 Code Review

## RoPE
- `head_dim` is validated as even in `MiniLLMConfig` and `build_rope_cache`.
- RoPE cache shape is `[1, 1, seq_len, head_dim]`, which broadcasts over batch and heads.
- `apply_rotary_pos_emb` applies rotation only on the final `head_dim` dimension of q/k and validates cache length and dimensions.
- Tests cover shape preservation, finite values, odd `head_dim`, different cache lengths, and too-short cache failure.

## GQA
- `n_head % n_kv_head == 0` is validated in config.
- q projects to `n_head`, while k/v project to `n_kv_head`; `repeat_kv` expands k/v to q head count.
- Attention output is reshaped back to `[batch, seq, n_embd]` and tested.

## Causal Mask
- SDPA path uses `attn_mask=None` with `is_causal=True`, avoiding incompatible double masking.
- Manual fallback uses `make_causal_mask`; tests verify future positions are blocked.
- A perturbation test changes future tokens and confirms earlier logits remain unchanged with `dropout=0` and `model.eval()`.

## Causal LM Loss
- Loss shifts logits with `logits[:, :-1, :]` and labels with `labels[:, 1:]`.
- `ignore_index=-100` is used and tested.
- Tests check finite loss and finite gradients after backward.

## Generation
- `do_sample=False` and `temperature=0` use greedy argmax without division by zero.
- Top-k and top-p filtering are tested directly; top-p keeps at least one token.
- NaN/Inf logits are sanitized before decoding.
- EOS early stop is tested.
- First implementation recomputes the full prefix each step; KV cache is a later TODO.

## Tied Embeddings
- `tie_word_embeddings=True` shares `lm_head.weight` and token embedding weight.
- `tie_word_embeddings=False` keeps them separate.
- Both cases are tested.

## Still Not Implemented
- KV cache / `past_key_values`
- real tokenizer
- data pipeline
- training loop / Stage 2 pretrain
- LoRA
- DPO
- GRPO
- GPTQ / SmoothQuant quantization
- FlashAttention/vLLM/DeepSpeed integrations