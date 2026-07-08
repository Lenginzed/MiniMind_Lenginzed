# Stage 2.1 Train Report

## Hardened Run Command

```powershell
& '<local_python_executable>' scripts\train_pretrain.py --config configs\pretrain_stage2_hardened.yaml
```

## Run Summary

- Parameter count: `1564992`
- Device: `cuda`
- Dtype: `bf16`
- Max/final steps: `220`
- Scheduler: `cosine`
- Warmup steps: `20`
- Gradient checkpointing: `True`
- Tokens seen: `450560`
- CUDA max memory allocated: `101348864` bytes
- CUDA max memory reserved: `144703488` bytes

## Loss / Perplexity

- Initial train loss: `7.6506`
- Final train loss: `1.6200`
- Initial train perplexity: `2101.89`
- Final train perplexity: `5.05`
- Initial eval loss: `7.6472`
- Final eval loss: `1.6461`
- Initial eval perplexity: `2094.83`
- Final eval perplexity: `5.19`
- Initial lr: `2.9999999999999997e-05`
- Final lr: `0.0`

The loss decrease is expected on a locally generated, repetitive corpus and only validates the pipeline. It is not a claim of model capability.

## Artifacts

- Metrics: `outputs\pretrain_stage2_hardened\metrics.jsonl`
- Last checkpoint: `outputs\pretrain_stage2_hardened\checkpoints\last.pt`
- Best checkpoint: `outputs\pretrain_stage2_hardened\checkpoints\best.pt`
- Loss curve: `outputs/pretrain_stage2_hardened/plots/loss_curve.png`
- Samples: `outputs\pretrain_stage2_hardened\samples\after.txt`
- Resolved config: `outputs/pretrain_stage2_hardened/train_config_resolved.yaml`
