# Stage 3 LoRA-SFT Report

## Command

```powershell
& '<local_python_executable>' scripts\train_sft.py --config configs\sft_lora.yaml
```

## Run

- Base checkpoint: `outputs/pretrain_stage2_hardened/checkpoints/best.pt`
- Target modules: `['q_proj', 'v_proj']`
- Replaced modules: `['layers.0.self_attn.q_proj', 'layers.0.self_attn.v_proj', 'layers.1.self_attn.q_proj', 'layers.1.self_attn.v_proj', 'layers.2.self_attn.q_proj', 'layers.2.self_attn.v_proj']`
- r: `8`
- alpha: `16`
- dropout: `0.05`
- Total params: `1580352`
- Trainable params: `15360`
- Trainable ratio: `0.009719`
- LoRA module count: `6`
- Device: `cuda`
- Dtype: `bf16`
- Max steps: `160`

## Loss

- Initial train loss: `7.9985`
- Final train loss: `6.4172`
- Initial eval loss: `8.6357`
- Final/best eval loss: `6.6694`
- Final/best eval perplexity: `787.93`

## Artifacts

- Metrics: `outputs\sft_lora\metrics.jsonl`
- Best checkpoint: `outputs\sft_lora\checkpoints\best.pt`
- Last checkpoint: `outputs\sft_lora\checkpoints\last.pt`
- Best adapter: `outputs/sft_lora/adapters/best_adapter.pt`
- Last adapter: `outputs/sft_lora/adapters/last_adapter.pt`
- Loss curve: `outputs/sft_lora/plots/loss_curve.png`
- Samples: `outputs\sft_lora\samples\after.txt`
