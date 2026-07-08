# Stage 1 Source Manifest

## `minillm/config.py`

- Lines: `51`
- Classes: MiniLLMConfig(methods: max_position_embeddings, head_dim, __post_init__)
- Functions: none
- Summary: Configuration dataclass and validation for model dimensions, GQA ratio, RoPE head_dim, dropout.

## `minillm/rope.py`

- Lines: `81`
- Classes: none
- Functions: build_rope_cache, rotate_half, apply_rotary_pos_emb
- Summary: RoPE cache construction, rotate_half, q/k rotary application and shape validation.

## `minillm/modules.py`

- Lines: `141`
- Classes: RMSNorm(methods: __init__, forward); SwiGLU(methods: __init__, forward); GQASelfAttention(methods: __init__, _shape, _manual_attention, forward); DecoderBlock(methods: __init__, forward)
- Functions: repeat_kv, make_causal_mask
- Summary: RMSNorm, SwiGLU, repeat_kv, causal mask helper, GQA self-attention, decoder block.

## `minillm/model.py`

- Lines: `69`
- Classes: MiniLLMForCausalLM(methods: __init__, _init_weights, forward)
- Functions: count_parameters
- Summary: Decoder-only Causal LM wrapper with embeddings, blocks, final norm, lm_head, tied embeddings, shifted CE loss.

## `minillm/generation.py`

- Lines: `101`
- Classes: none
- Functions: _top_k_filter, _top_p_filter, _sanitize_logits, generate
- Summary: Greedy and sampled decoding with temperature, top-k, top-p, EOS early stop, logits sanitization.

## `scripts/smoke_model_forward.py`

- Lines: `82`
- Classes: none
- Functions: build_tiny_config, main
- Summary: Tiny config forward/loss/backward/generation smoke script with CUDA memory summary.

## `tests/test_model_shapes.py`

- Lines: `119`
- Classes: none
- Functions: tiny_config, test_forward_logits_loss_and_backward_cpu, test_loss_supports_ignore_index, test_tied_embeddings_share_weight, test_untied_embeddings_do_not_share_weight, test_invalid_gqa_ratio_raises, test_odd_head_dim_raises, test_repeat_kv_shape_and_values, test_attention_output_shape_cpu, test_tiny_parameter_count_exact
- Summary: Forward/loss/backward, tied embeddings, GQA validation, parameter count, attention shape tests.

## `tests/test_rope.py`

- Lines: `64`
- Classes: none
- Functions: test_rope_shape_and_finiteness, test_rope_cache_shapes_for_different_lengths, test_rope_odd_head_dim_raises, test_apply_rope_cache_too_short_raises, test_rope_position_ids_shape
- Summary: RoPE cache construction, rotate_half, q/k rotary application and shape validation.

## `tests/test_generation.py`

- Lines: `101`
- Classes: ConstantLogitModel(methods: __init__, forward)
- Functions: tiny_model, test_generate_greedy_length_growth, test_generate_temperature_zero_is_greedy, test_generate_sampling_top_k_top_p_length_growth, test_top_k_filter_keeps_only_k_logits, test_top_p_filter_keeps_at_least_one_token, test_generate_eos_early_stop, test_sanitize_logits_handles_nan_inf
- Summary: Greedy and sampled decoding with temperature, top-k, top-p, EOS early stop, logits sanitization.

## `tests/test_attention_causality.py`

- Lines: `51`
- Classes: none
- Functions: tiny_config, test_make_causal_mask_blocks_future_tokens, test_future_token_perturbation_does_not_change_past_logits
- Summary: Causal mask and future-token perturbation tests.
