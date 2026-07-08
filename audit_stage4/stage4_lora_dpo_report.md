# Stage 4 DPO-LoRA Report

## Command

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\train_dpo.py --config configs\dpo_lora.yaml
```

## Setup

- Policy checkpoint: `outputs/sft_full/checkpoints/best.pt`
- Reference checkpoint: `outputs/sft_full/checkpoints/best.pt`
- Beta: 0.1
- LoRA target modules: `['q_proj', 'v_proj']`
- LoRA rank/alpha/dropout: 8 / 16 / 0.05
- LoRA modules replaced: 6
- Total params: 1,580,352
- Trainable params: 15,360
- Trainable ratio: 0.9719%
- Device/dtype: cuda / bf16
- Max steps: 140

## Metrics

- Initial train DPO loss: 0.693117
- Final train DPO loss: 0.005183
- Initial train reward margin: 0.000064
- Final train reward margin: 6.874298
- Initial train preference accuracy: 0.3750
- Final train preference accuracy: 1.0000
- Best eval DPO loss: 0.017536
- Final eval reward margin: 6.156336
- Final eval preference accuracy: 1.0000

## Artifacts

- Best checkpoint: `outputs/dpo_lora/checkpoints/best.pt`
- Last checkpoint: `outputs/dpo_lora/checkpoints/last.pt`
- Best adapter: `outputs/dpo_lora/adapters/best_adapter.pt`
- Last adapter: `outputs/dpo_lora/adapters/last_adapter.pt`
- Metrics: `outputs/dpo_lora/metrics.jsonl`
- Loss curve: `outputs/dpo_lora/plots/loss_curve.png`
- Samples: `outputs/dpo_lora/samples/after.txt`
