# Stage 3 Full SFT Report

## Command

```powershell
& '<local_python_executable>' scripts\train_sft.py --config configs\sft_full.yaml
```

## Run

- Base checkpoint: `outputs/pretrain_stage2_hardened/checkpoints/best.pt`
- Parameter count: `1564992`
- Trainable params: `1564992`
- Trainable ratio: `1.000000`
- Device: `cuda`
- Dtype: `bf16`
- Max steps: `160`

## Loss

- Initial train loss: `8.1192`
- Final train loss: `3.5500`
- Initial eval loss: `8.5907`
- Final/best eval loss: `3.7540`
- Final/best eval perplexity: `42.69`

## Artifacts

- Metrics: `outputs\sft_full\metrics.jsonl`
- Best checkpoint: `outputs\sft_full\checkpoints\best.pt`
- Last checkpoint: `outputs\sft_full\checkpoints\last.pt`
- Loss curve: `outputs/sft_full/plots/loss_curve.png`
- Samples: `outputs\sft_full\samples\after.txt`
