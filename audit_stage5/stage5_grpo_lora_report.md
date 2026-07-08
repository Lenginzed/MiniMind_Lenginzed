# Stage 5 GRPO-LoRA Report

## Command

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\train_grpo.py --config configs\grpo_lora.yaml
```

## Setup

- Policy checkpoint: `outputs/sft_full/checkpoints/best.pt`
- LoRA target modules: `['q_proj', 'v_proj']`
- LoRA rank/alpha/dropout: 8 / 16 / 0.05
- Total params: 1,580,352
- Trainable params: 15,360
- Trainable ratio: 0.9719%
- Device/dtype: cuda / bf16
- Num generations: 4
- Max new tokens: 32
- Clip epsilon: 0.2
- Max steps: 50

## Training Metrics

- Initial/final reward_mean: 0.1272 -> 0.1397
- Initial/final reward_std: 0.0712 -> 0.0526
- Initial/final frac_reward_zero_std: 0.0000 -> 0.0000
- Initial/final exact_accuracy: 0.0000 -> 0.0000
- Initial/final train_loss: 0.070494 -> -0.048667
- Final approx_kl: -0.000720
- Final clip_ratio: 0.000000
- Final mean_ratio: 1.000813
- Best eval reward during training: 0.1672

## Eval Smoke

- Eval reward_mean/std: 0.1182 / 0.0688
- Eval exact_accuracy: 0.0000
- Eval format_reward_mean: 0.0538
- Eval completion_length_mean: 36.4000

## Artifacts

- Best checkpoint: `outputs/grpo_lora/checkpoints/best.pt`
- Last checkpoint: `outputs/grpo_lora/checkpoints/last.pt`
- Best adapter: `outputs/grpo_lora/adapters/best_adapter.pt`
- Last adapter: `outputs/grpo_lora/adapters/last_adapter.pt`
- Metrics: `outputs/grpo_lora/metrics.jsonl`
- Rollout samples: `outputs/grpo_lora/samples/rollout_samples.jsonl`
- Eval report: `outputs/grpo_lora/eval/eval_report.json`
- Samples: `outputs/grpo_lora/samples/after.txt`
