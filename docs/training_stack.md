# Training Stack

This document summarizes the from-scratch mini-LLM stack implemented in this repository. The implementation favors readability, auditability, and local reproducibility over raw throughput.

## Model Core

The decoder-only model is implemented under `minillm/`.

| Area | Files | What is implemented |
| --- | --- | --- |
| Config | `minillm/config.py` | Model hyperparameters, context length, GQA ratio, RoPE theta, dropout, gradient checkpointing flag |
| Transformer modules | `minillm/modules.py` | RMSNorm, SwiGLU MLP, GQA attention, decoder block |
| RoPE | `minillm/rope.py` | Rotary frequency cache and q/k rotary application |
| Causal LM | `minillm/model.py` | Token embeddings, stacked decoder blocks, final norm, tied lm head, causal LM loss |
| Generation | `minillm/generation.py` | Greedy, temperature, top-k, top-p, EOS stopping, finite-logit handling |

The architecture used in the public long run was:

| Field | Value |
| --- | ---: |
| vocab size | 12,000 |
| context length | 256 |
| layers | 10 |
| embedding dim | 576 |
| q heads | 9 |
| kv heads | 3 |
| intermediate size | 1,728 |
| parameter count | 45,631,296 |
| gradient checkpointing | enabled |

## Data And Tokenizer

| Component | Files | Notes |
| --- | --- | --- |
| Tokenizer wrapper | `minillm/tokenizer.py` | Byte-level BPE training/loading/encode/decode and special token ids |
| LM data | `minillm/data.py` | Continuous token ids, block packing, train/val splits, metadata checks |
| SFT data | `minillm/sft_data.py` | Prompt templating, assistant-only labels, padding collate function |
| DPO data | `minillm/dpo_data.py` | Chosen/rejected examples with prompt-masked labels |
| GRPO data | `minillm/grpo_data.py` | Prompt/answer/reward metadata for online sampling |

The Stage 8 public tokenizer was trained on WikiText-2 raw text:

- 12,000 vocabulary entries.
- 2,495,217 encoded tokens.
- 9,552 train blocks and 194 validation blocks at block size 256.

## Training Loops

The project intentionally does not use Hugging Face Trainer, TRL, or PEFT for the core algorithms.

| Stage | Files | Main mechanics |
| --- | --- | --- |
| Pretrain | `minillm/trainer.py`, `scripts/train_pretrain.py` | AdamW, scheduler, warmup, grad accumulation, clipping, eval, checkpoints, resume, metrics |
| SFT | `minillm/sft_trainer.py`, `scripts/train_sft.py` | Assistant-only loss, full SFT, LoRA-SFT, sample generation, adapter save |
| DPO | `minillm/dpo_trainer.py`, `scripts/train_dpo.py` | Sequence logps, frozen reference, DPO loss, reward margin, preference accuracy |
| GRPO | `minillm/grpo_trainer.py`, `scripts/train_grpo.py` | Online completions, old/new logps, group-relative advantage, clipped objective |

## LoRA Implementation

`minillm/lora.py` implements LoRA without PEFT:

- `LoRALinear` wraps selected `nn.Linear` modules.
- Base weights are frozen in LoRA modes.
- Low-rank A/B matrices are trainable.
- Scaling uses `lora_alpha / r`.
- Target modules are explicit, typically `q_proj` and `v_proj`.
- Adapter state and config can be saved and loaded.

In the Stage 8 public runs, q/v LoRA used 307,200 trainable parameters out of 45,938,496 total parameters, or about 0.6687%.

## DPO Diagnostics

DPO computes response log probabilities only over labels that are not `-100`, so prompt tokens are excluded. The loss uses:

```text
pi_logratio  = logp_policy(chosen) - logp_policy(rejected)
ref_logratio = logp_ref(chosen) - logp_ref(rejected)
loss = -logsigmoid(beta * (pi_logratio - ref_logratio))
```

Metrics include DPO loss, chosen/rejected rewards, reward margin, preference accuracy, policy/ref log probabilities, learning rate, gradient norm, and trainable parameter counts.

## GRPO Diagnostics

GRPO samples several completions per prompt and computes local rule-based rewards. Group-relative advantages are computed within each prompt group:

```text
advantage = (reward - group_mean) / (group_std + eps)
```

If a group has near-zero reward standard deviation, its advantage is set to zero and the zero-std fraction is logged. This is important because the Stage 8 full GRPO diagnostic collapsed to zero group std by the end, which means the reward signal stopped differentiating completions.

## Quantization

Quantization modules are educational:

| File | Scope |
| --- | --- |
| `minillm/quantization.py` | Symmetric qparams, fake INT8/INT4 quantization, `QuantizedLinear`, size estimates |
| `minillm/gptq.py` | Simplified GPTQ-style calibration and weighted error reporting |
| `minillm/smoothquant.py` | SmoothQuant-style activation/weight scale collection and wrapper |

The quantized layers dequantize to floating point and call `F.linear`, so these paths validate algorithm structure and measurement plumbing, not production integer-kernel speed.
