# Stage 4 Full DPO Report

## Command

```powershell
& '<local_python_executable>' scripts\train_dpo.py --config configs\dpo_full.yaml
```

## Setup

- Policy checkpoint: `outputs/sft_full/checkpoints/best.pt`
- Reference checkpoint: `outputs/sft_full/checkpoints/best.pt`
- Beta: 0.1
- Total params: 1,564,992
- Trainable params: 1,564,992
- Trainable ratio: 100.00%
- Device/dtype: cuda / bf16
- Max steps: 140

## Metrics

- Initial train DPO loss: 0.692218
- Final train DPO loss: 0.000044
- Initial train reward margin: 0.001869
- Final train reward margin: 12.134012
- Initial train preference accuracy: 0.5000
- Final train preference accuracy: 1.0000
- Best eval DPO loss: 0.000486
- Final eval reward margin: 12.779533
- Final eval preference accuracy: 1.0000

## Artifacts

- Best checkpoint: `outputs/dpo_full/checkpoints/best.pt`
- Last checkpoint: `outputs/dpo_full/checkpoints/last.pt`
- Metrics: `outputs/dpo_full/metrics.jsonl`
- Loss curve: `outputs/dpo_full/plots/loss_curve.png`
- Samples: `outputs/dpo_full/samples/after.txt`
