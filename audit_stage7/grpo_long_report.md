# Stage 7 grpo_full Report

- Output dir: `outputs/stage7/grpo_full_long`
- Metrics: `outputs\stage7\grpo_full_long\metrics.jsonl`
- policy_checkpoint: `outputs/stage7/sft_full_long/checkpoints/best.pt`
- best_checkpoint: `outputs\stage7\grpo_full_long\checkpoints\best.pt`
- last_checkpoint: `outputs\stage7\grpo_full_long\checkpoints\last.pt`
- rollout_samples_path: `outputs\stage7\grpo_full_long\samples\rollout_samples.jsonl`
- parameter_count: `43327296`
- trainable_params: `43327296`
- trainable_ratio: `1.0`
- device: `cuda`
- dtype: `bf16`
- max_steps: `80`
- num_generations: `4`
- max_new_tokens: `32`
- clip_epsilon: `0.2`
- best_eval_reward: `0.18500000000000008`
- reward_mean: `0.158625` -> `0.230000`
- reward_std: `0.034630` -> `0.000000`
- frac_reward_zero_std: `0.500000` -> `1.000000`
- exact_accuracy_mean: `0.000000` -> `0.000000`
- train_loss: `-0.008801` -> `0.000000`
- clip_ratio: `0.000000` -> `0.000000`
- approx_kl: `0.000060` -> `0.000206`

Note: Stage 7 uses local/synthetic data and longer resume-ready runs for engineering demonstration, not real model capability claims.