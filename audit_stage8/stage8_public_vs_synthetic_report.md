# Stage 8 Public vs Stage 7 Synthetic Report

## Main Difference

Stage 7 used local synthetic data designed for pipeline validation. Stage 8 uses real public subsets for pretrain/SFT/DPO, while GRPO remains local reward data by design.

## Metric Comparison

- Pretrain final eval loss/ppl: Stage 7 synthetic `0.7534` / `2.1241` vs Stage 8 public `4.7385` / `114.2674`. Public is harder and more credible.
- Full SFT final eval loss/ppl: Stage 7 synthetic `0.1454` / `1.1565` vs Stage 8 public `4.0156` / `55.4577`. Public Alpaca is harder and more varied.
- DPO final eval accuracy: Stage 7 synthetic `1.0000` vs Stage 8 public `0.5500`. Public preference does not saturate, so it is a more honest diagnostic.
- GRPO zero_std final: Stage 7 full `1.0000` vs Stage 8 full `1.0000`. Both show reward-diversity limitations in this toy RL setup.

## Interpretation

- Stage 8 curves are less pretty than Stage 7, but more interview-credible because they stress the pipeline on non-template public data.
- The public run exposes limitations: high perplexity, noisy DPO, weak generation, and GRPO reward collapse.
- These limitations are useful engineering evidence, not failures to hide.
