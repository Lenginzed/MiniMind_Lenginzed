# Experiment Results

This page collects the README-ready results from Stage 7 synthetic long-run and Stage 8 public-data long-run. The Stage 8 results are more important for credibility because they use public dataset subsets and do not trivially saturate.

## Stage 8 Public Data

| Split | Source | Size |
| --- | --- | --- |
| Pretrain | `Salesforce/wikitext`, `wikitext-2-raw-v1` | 16,929 lines, 10,717,316 bytes |
| SFT | `tatsu-lab/alpaca` | 20,000 train, 2,000 val |
| DPO | `tatsu-lab/alpaca_farm`, `alpaca_gpt4_preference` | 10,000 train, 1,000 val |
| GRPO | local verifiable rewards | diagnostic only, not public RLVR |

Tokenizer and pretrain packing:

| Item | Value |
| --- | ---: |
| tokenizer vocab size | 12,000 |
| total tokens | 2,495,217 |
| train tokens | 2,445,312 |
| val tokens | 49,664 |
| train blocks | 9,552 |
| val blocks | 194 |
| block size | 256 |

## Stage 8 Public Training Results

| Stage | Steps | Initial metric | Final metric | Notes |
| --- | ---: | --- | --- | --- |
| Pretrain | 1,500 | train loss `9.5015`, eval loss `9.5070` | train loss `4.2189`, eval loss `4.7385`, best eval ppl `114.2674` | WikiText-2 subset |
| Full SFT | 1,500 | train loss `7.0393` | train loss `3.9252`, eval loss `4.0156`, eval ppl `55.4577` | Alpaca |
| LoRA-SFT | 1,500 | train loss `6.5874` | train loss `5.3978`, eval loss `5.2418`, eval ppl `189.0164` | 0.6687% trainable |
| Full DPO | 1,000 | train loss `0.6926` | train loss `1.0169`, eval margin `0.1615`, eval acc `0.5500` | AlpacaFarm preferences |
| DPO-LoRA | 1,000 | train loss `0.6929` | train loss `0.7124`, eval margin `0.0582`, eval acc `0.5250` | 0.6687% trainable |
| Full GRPO diagnostic | 150 | reward mean `0.1096`, zero_std `0.5000` | reward mean `0.2300`, zero_std `1.0000`, exact acc `0.0000` | reward diversity collapsed |
| GRPO-LoRA diagnostic | 150 | reward mean `0.1840`, zero_std `0.0000` | reward mean `0.2269`, zero_std `0.5000`, exact acc `0.0000` | retained more diversity |

## Stage 7 Synthetic Baseline

Stage 7 used locally generated synthetic data. It is useful for end-to-end validation, but less credible as a capability benchmark.

| Stage | Steps | Result |
| --- | ---: | --- |
| Pretrain | 600 | eval loss/ppl reached `0.7534` / `2.1241` |
| Full SFT | 400 | eval loss/ppl reached `0.1454` / `1.1565` |
| LoRA-SFT | 400 | train loss reached `2.0745` |
| Full DPO | 250 | preference accuracy saturated to `1.0000` |
| DPO-LoRA | 250 | loss fell near zero |
| Full GRPO | 80 | reward mean reached `0.2300`, exact accuracy stayed `0` |
| GRPO-LoRA | 80 | reward mean reached `0.1985`, exact accuracy stayed `0` |

## Synthetic vs Public Interpretation

| Metric | Stage 7 synthetic | Stage 8 public | Interpretation |
| --- | ---: | ---: | --- |
| Pretrain final eval ppl | `2.1241` | `114.2674` | Public text is harder and less templated |
| Full SFT final eval ppl | `1.1565` | `55.4577` | Alpaca is much more varied than local templates |
| Full DPO final eval accuracy | `1.0000` | `0.5500` | Public preference data does not trivially saturate |
| Full GRPO final zero_std | `1.0000` | `1.0000` | Reward diversity remains a limitation |

The public curves are less visually impressive but more useful for interviews because they expose real limitations in the pipeline.

## Full vs LoRA

| Run | Total params | Trainable params | Trainable ratio | Stage 8 result |
| --- | ---: | ---: | ---: | --- |
| Full SFT | 45,631,296 | 45,631,296 | 100% | eval loss `4.0156` |
| LoRA-SFT | 45,938,496 | 307,200 | 0.6687% | eval loss `5.2418` |
| Full DPO | 45,631,296 | 45,631,296 | 100% | eval acc `0.5500`, margin `0.1615` |
| DPO-LoRA | 45,938,496 | 307,200 | 0.6687% | eval acc `0.5250`, margin `0.0582` |

LoRA demonstrates the adapter training workflow and parameter-efficiency tradeoff. It should not be presented as matching full fine-tuning quality in these runs.

## Quantization Summary

| Method | Loss delta | Estimated compression | Caveat |
| --- | ---: | ---: | --- |
| INT8 weight-only | `+0.000553` | `1.9672x` | fake quant, no real int kernel |
| INT4 weight-only | `+0.240660` | `2.5950x` | no bit packing |
| GPTQ-style INT4 | `+0.240660` | `2.5950x` | simplified GPTQ-style, no blockwise compensation |
| SmoothQuant-style INT8 | `+0.000723` | `1.9547x` | educational wrapper |

## Plots

- Public pretrain loss/PPL: `docs/assets/pretrain_public_loss_ppl.png`
- Synthetic vs public comparison: `docs/assets/public_vs_synthetic_compare.png`
- Public SFT full vs LoRA: `docs/assets/sft_public_full_vs_lora.png`
- Public DPO margin/accuracy: `docs/assets/dpo_public_margin_acc.png`
- Public GRPO diagnostics: `docs/assets/grpo_public_diagnostics.png`
- Stage 7 pretrain loss/PPL: `outputs/stage7/plots/pretrain_loss_ppl.png`
- Stage 7 trainable params comparison: `outputs/stage7/plots/trainable_params_compare.png`

## Safe Claims

- The stack was implemented from scratch and exercised end to end.
- Public WikiText-2, Alpaca, and AlpacaFarm subsets were used for Stage 8.
- Metrics, checkpoints, samples, plots, and audit reports were generated.
- Public-data results are noisy and limited, which is part of the engineering evidence.

## Claims To Avoid

- Do not claim real instruction-following ability.
- Do not claim preference alignment.
- Do not claim mathematical reasoning.
- Do not describe GRPO as public RLVR.
- Do not claim quantization delivers production inference acceleration.
