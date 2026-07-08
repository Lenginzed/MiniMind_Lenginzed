# Stage 7 dpo_lora Report

- Output dir: `outputs/stage7/dpo_lora_long`
- Metrics: `outputs\stage7\dpo_lora_long\metrics.jsonl`
- policy_checkpoint: `outputs/stage7/sft_full_long/checkpoints/best.pt`
- reference_checkpoint: `outputs/stage7/sft_full_long/checkpoints/best.pt`
- best_checkpoint: `outputs\stage7\dpo_lora_long\checkpoints\best.pt`
- last_checkpoint: `outputs\stage7\dpo_lora_long\checkpoints\last.pt`
- sample_path: `outputs\stage7\dpo_lora_long\samples\after.txt`
- parameter_count: `43818816`
- trainable_params: `491520`
- trainable_ratio: `0.0112170990653878`
- device: `cuda`
- dtype: `bf16`
- max_steps: `250`
- best_eval_loss: `0.00015368948752438882`
- train_loss: `0.693108` -> `0.000103`
- reward_margin: `0.001342` -> `9.311702`
- preference_accuracy: `0.500000` -> `1.000000`

Note: Stage 7 uses local/synthetic data and longer resume-ready runs for engineering demonstration, not real model capability claims.