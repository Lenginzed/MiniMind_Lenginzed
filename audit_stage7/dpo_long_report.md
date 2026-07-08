# Stage 7 dpo_full Report

- Output dir: `outputs/stage7/dpo_full_long`
- Metrics: `outputs\stage7\dpo_full_long\metrics.jsonl`
- policy_checkpoint: `outputs/stage7/sft_full_long/checkpoints/best.pt`
- reference_checkpoint: `outputs/stage7/sft_full_long/checkpoints/best.pt`
- best_checkpoint: `outputs\stage7\dpo_full_long\checkpoints\best.pt`
- last_checkpoint: `outputs\stage7\dpo_full_long\checkpoints\last.pt`
- sample_path: `outputs\stage7\dpo_full_long\samples\after.txt`
- parameter_count: `43327296`
- trainable_params: `43327296`
- trainable_ratio: `1.0`
- device: `cuda`
- dtype: `bf16`
- max_steps: `250`
- best_eval_loss: `4.92778752274603e-06`
- train_loss: `0.693540` -> `0.000004`
- reward_margin: `-0.000517` -> `14.381209`
- preference_accuracy: `0.500000` -> `1.000000`

Note: Stage 7 uses local/synthetic data and longer resume-ready runs for engineering demonstration, not real model capability claims.