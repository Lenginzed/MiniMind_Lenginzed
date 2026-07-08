# Stage 5 Full GRPO Report

## Command

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\train_grpo.py --config configs\grpo_full.yaml
```

## Setup

- Policy checkpoint: `outputs/sft_full/checkpoints/best.pt`
- Total params: 1,564,992
- Trainable params: 1,564,992
- Trainable ratio: 100.00%
- Device/dtype: cuda / bf16
- Num generations: 4
- Max new tokens: 32
- Clip epsilon: 0.2
- Max steps: 60

## Training Metrics

- Initial/final reward_mean: 0.1114 -> 0.1800
- Initial/final reward_std: 0.0620 -> 0.0500
- Initial/final frac_reward_zero_std: 0.0000 -> 1.0000
- Initial/final exact_accuracy: 0.0000 -> 0.0000
- Initial/final train_loss: -0.423847 -> 0.000000
- Final approx_kl: -0.000237
- Final clip_ratio: 0.000000
- Final mean_ratio: 1.000247
- Final advantage_std: 0.000000
- Best eval reward during training: 0.2033

## Eval Smoke

- Eval reward_mean/std: 0.2572 / 0.2767
- Eval exact_accuracy: 0.0800
- Eval format_reward_mean: 0.0700
- Eval completion_length_mean: 82.2600

## Diagnostics

Full GRPO reached `frac_reward_zero_std=1.0` at the final step, so group advantages were zero and `grad_norm=0.0`. This is a useful Stage 5 diagnosis: the online GRPO loop works, but the current tiny model plus simple dense reward can collapse into same-reward groups and lose training signal.

## Artifacts

- Best checkpoint: `outputs/grpo_full/checkpoints/best.pt`
- Last checkpoint: `outputs/grpo_full/checkpoints/last.pt`
- Metrics: `outputs/grpo_full/metrics.jsonl`
- Rollout samples: `outputs/grpo_full/samples/rollout_samples.jsonl`
- Eval report: `outputs/grpo_full/eval/eval_report.json`
- Samples: `outputs/grpo_full/samples/after.txt`
