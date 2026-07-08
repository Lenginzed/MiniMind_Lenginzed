# Stage 7 sft_full Report

- Output dir: `outputs/stage7/sft_full_long`
- Metrics: `outputs\stage7\sft_full_long\metrics.jsonl`
- base_checkpoint: `outputs/stage7/pretrain_long/checkpoints/best.pt`
- best_checkpoint: `outputs\stage7\sft_full_long\checkpoints\best.pt`
- last_checkpoint: `outputs\stage7\sft_full_long\checkpoints\last.pt`
- sample_path: `outputs\stage7\sft_full_long\samples\after.txt`
- parameter_count: `43327296`
- trainable_params: `43327296`
- trainable_ratio: `1.0`
- device: `cuda`
- dtype: `bf16`
- max_steps: `400`
- best_eval_loss: `0.14536371208960192`
- train_loss: `11.000511` -> `0.180949`
- eval_loss: `10.599199` -> `0.145364`
- train_ppl: `59904.726805` -> `1.198354`
- eval_ppl: `40102.689271` -> `1.156460`

Note: Stage 7 uses local/synthetic data and longer resume-ready runs for engineering demonstration, not real model capability claims.