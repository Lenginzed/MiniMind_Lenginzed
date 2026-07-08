# Stage 7 grpo_lora Report

- Output dir: `outputs/stage7/grpo_lora_long`
- Metrics: `outputs\stage7\grpo_lora_long\metrics.jsonl`
- policy_checkpoint: `outputs/stage7/sft_full_long/checkpoints/best.pt`
- best_checkpoint: `outputs\stage7\grpo_lora_long\checkpoints\best.pt`
- last_checkpoint: `outputs\stage7\grpo_lora_long\checkpoints\last.pt`
- rollout_samples_path: `outputs\stage7\grpo_lora_long\samples\rollout_samples.jsonl`
- parameter_count: `43818816`
- trainable_params: `491520`
- trainable_ratio: `0.0112170990653878`
- device: `cuda`
- dtype: `bf16`
- max_steps: `80`
- num_generations: `4`
- max_new_tokens: `32`
- clip_epsilon: `0.2`
- best_eval_reward: `0.17492499999999997`
- reward_mean: `0.129125` -> `0.198500`
- reward_std: `0.051557` -> `0.011916`
- frac_reward_zero_std: `0.000000` -> `0.000000`
- exact_accuracy_mean: `0.000000` -> `0.000000`
- train_loss: `-0.409344` -> `-0.239267`
- clip_ratio: `0.000000` -> `0.000000`
- approx_kl: `-0.000225` -> `0.001190`

Note: Stage 7 uses local/synthetic data and longer resume-ready runs for engineering demonstration, not real model capability claims.