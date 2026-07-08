# Limitations And Claim Boundaries

This project is intentionally scoped as an educational mini-LLM stack. The results are useful for understanding engineering mechanics and diagnostics, but they should not be framed as proof of strong model capability.

## Model Scale

The Stage 8 public model has 45,631,296 parameters. This is large enough to exercise transformer training code, LoRA adapters, DPO log-probability logic, GRPO diagnostics, and quantization plumbing. It is not large enough, trained long enough, or trained on enough data to support claims about real general language ability.

## Data Scale

The public long run uses controlled subsets:

- WikiText-2 raw for pretraining.
- Alpaca for SFT.
- AlpacaFarm GPT-4 preference data for DPO.
- Local verifiable reward prompts for GRPO diagnostics.

These choices improve credibility over fully synthetic data, but they are still small compared with real LLM training corpora.

## SFT Output Quality

Public Alpaca SFT samples are more natural than the synthetic-only baseline in some cases. They are still unreliable and can hallucinate, drift, or fail task semantics. Treat samples as smoke-test artifacts, not proof of instruction following.

## DPO Interpretation

DPO is implemented correctly as a diagnostic training objective with a frozen reference model. Public AlpacaFarm preference accuracy stayed near `0.55` for full DPO and `0.525` for DPO-LoRA, so the result should be described as modest and noisy.

Synthetic DPO reaching `1.0` accuracy is a sign that the synthetic preference task was easy, not evidence of real alignment.

## GRPO Interpretation

GRPO is a minimal RL post-training loop with local rule rewards. It demonstrates:

- online completion sampling,
- reward breakdowns,
- group-relative advantages,
- token-level clipped policy loss,
- zero-std reward diagnostics.

It does not demonstrate real RLVR or mathematical reasoning. In Stage 8, exact accuracy stayed at `0`, and full GRPO collapsed to zero group reward standard deviation by the end.

## Quantization Interpretation

The quantization stage is educational fake quantization:

- INT4 values are not bit-packed.
- Quantized linear layers dequantize to floating-point weights.
- GPTQ-style quantization is simplified and does not implement full blockwise compensation.
- SmoothQuant-style scaling is wrapped for clarity rather than fused into an optimized graph.

The reported size compression is an estimate. The latency results should not be interpreted as production inference acceleration.

## Engineering Gaps

Current gaps that would matter for a stronger project:

- Larger and cleaner public pretraining corpus.
- Better tokenizer and longer context evaluation.
- KV cache for efficient generation.
- More robust sampling and reward design for GRPO.
- More systematic hyperparameter sweeps.
- Real packed quantization and optimized integer kernels.
- Optional future integration with production inference engines after the from-scratch path remains validated.

## Recommended Wording

Use:

- "from-scratch mini-LLM stack"
- "public-data subset long run"
- "diagnostic DPO/GRPO metrics"
- "educational quantization"
- "training dynamics and limitations"

Avoid:

- "aligned model"
- "reasoning model"
- "instruction-following model"
- "production-ready inference"
- "RLVR solved"
- "DPO solved preference alignment"
