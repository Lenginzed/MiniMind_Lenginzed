# Stage 7 pretrain Report

- Output dir: `outputs/stage7/pretrain_long`
- Metrics: `outputs\stage7\pretrain_long\metrics.jsonl`
- best_checkpoint: `outputs\stage7\pretrain_long\checkpoints\best.pt`
- last_checkpoint: `outputs\stage7\pretrain_long\checkpoints\last.pt`
- parameter_count: `43327296`
- device: `cuda`
- dtype: `bf16`
- max_steps: `600`
- tokens_seen: `2457600`
- best_eval_loss: `0.7533592373132706`
- train_loss: `9.089170` -> `0.740464`
- eval_loss: `9.068929` -> `0.753359`
- train_ppl: `8858.827874` -> `2.096909`
- eval_ppl: `8681.322412` -> `2.124123`

Note: Stage 7 uses local/synthetic data and longer resume-ready runs for engineering demonstration, not real model capability claims.