# Model Card

This model card describes the MiniMind public-data long-run checkpoints as experimental artifacts. The checkpoints are not intended for real deployment.

## Model Summary

- Architecture: decoder-only Causal LM.
- Parameters: 45,631,296 for the Stage 8 public base/full model.
- Context length: 256 tokens.
- Vocabulary size: 12,000.
- Core modules: GQA, RoPE, SwiGLU, RMSNorm, tied token/lm head.
- Training precision: bf16 on a single local RTX 4080 SUPER 16GB environment.

## Training Data

The Stage 8 public run used controlled public subsets:

- Pretraining: WikiText-2 raw train split.
- SFT: Alpaca-format public instruction data.
- DPO: AlpacaFarm GPT-4 preference data.
- GRPO diagnostics: local generated prompts with rule-verifiable rewards.

The GRPO diagnostic data is not public RLVR.

## Intended Use

Appropriate uses:

- Education and code review.
- Demonstrating a from-scratch training/post-training stack.
- Inspecting metrics, checkpoints, logs, and limitations.
- Comparing full fine-tuning with LoRA.
- Understanding DPO/GRPO diagnostic behavior.
- Demonstrating educational fake quantization.

## Out-Of-Scope Use

Do not use these checkpoints for:

- Real user-facing question answering.
- Safety-critical, legal, medical, financial, or operational decisions.
- Claims of instruction following, alignment, or mathematical reasoning.
- Production inference benchmarking.
- Any setting where output reliability matters.

## Evaluation Summary

Public-data long-run highlights:

| Stage | Result |
| --- | --- |
| Pretrain | eval loss `9.5070 -> 4.7385`, best eval ppl `114.2674` |
| Full SFT | eval loss `4.0156`, eval ppl `55.4577` |
| LoRA-SFT | eval loss `5.2418`, eval ppl `189.0164`, 0.6687% trainable params |
| Full DPO | final eval margin `0.1615`, preference accuracy `0.5500` |
| DPO-LoRA | final eval margin `0.0582`, preference accuracy `0.5250` |
| Full GRPO diagnostic | exact accuracy stayed `0`; final zero-std fraction `1.0000` |

These metrics show training dynamics and diagnostics, not strong model capability.

## Limitations

- Small model size and short training budget.
- Public datasets are subsets, not full-scale LLM corpora.
- SFT samples can be more natural than synthetic samples but remain unreliable.
- DPO metrics are modest and noisy on public preference data.
- GRPO reward functions are local rule-based diagnostics.
- Quantization is educational fake quantization, not optimized deployment.

## Release Guidance

If checkpoints are shared outside this repository, publish them separately from the source repo and include this model card. Do not commit large checkpoint files to Git.
