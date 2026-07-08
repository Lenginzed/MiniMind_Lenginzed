# Stage 7 sft_lora Report

- Output dir: `outputs/stage7/sft_lora_long`
- Metrics: `outputs\stage7\sft_lora_long\metrics.jsonl`
- base_checkpoint: `outputs/stage7/pretrain_long/checkpoints/best.pt`
- best_checkpoint: `outputs\stage7\sft_lora_long\checkpoints\best.pt`
- last_checkpoint: `outputs\stage7\sft_lora_long\checkpoints\last.pt`
- sample_path: `outputs\stage7\sft_lora_long\samples\after.txt`
- parameter_count: `43818816`
- trainable_params: `491520`
- trainable_ratio: `0.0112170990653878`
- device: `cuda`
- dtype: `bf16`
- max_steps: `400`
- best_eval_loss: `1.9229119658470153`
- train_loss: `10.989395` -> `2.074528`
- eval_loss: `10.827685` -> `1.922912`
- train_ppl: `59242.539864` -> `7.960791`
- eval_ppl: `50396.913550` -> `6.840850`

Note: Stage 7 uses local/synthetic data and longer resume-ready runs for engineering demonstration, not real model capability claims.