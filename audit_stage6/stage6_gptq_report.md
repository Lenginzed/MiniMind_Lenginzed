# Stage 6 GPTQ-Style INT4 Report

## Command

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\quantize_eval.py --config configs\quant_gptq_int4.yaml
```

## Setup

- Checkpoint: `outputs/sft_full/checkpoints/best.pt`
- Method: `gptq_style`
- Num bits: 4
- Per-channel: True
- Device/dtype: cuda / bf16
- Eval data: `data/sft/sft_val.jsonl`
- Eval max batches: 10

## Loss / Perplexity

- Baseline loss / ppl: 3.853451 / 47.1555
- Quantized loss / ppl: 4.094112 / 59.9860
- Loss delta / ppl delta: 0.240660 / 12.8305

## Size Estimate

- Estimated baseline size: 6,358,272 bytes (6.064 MiB)
- Estimated quantized size: 2,450,240 bytes (2.337 MiB)
- Compression ratio: 2.5950x

## Latency

- Baseline forward/generate latency: 3.757 ms / 67.125 ms
- Quantized forward/generate latency: 4.468 ms / 65.217 ms
- Note: Fake quantization uses dequantized floating-point F.linear and does not represent production integer-kernel speed.

## Artifacts

- Eval report: `outputs/quant_gptq_int4/eval_report.json`
- Samples: `outputs/quant_gptq_int4/samples/after.txt`

## Calibration / GPTQ-Style Details

- Calibration data: `data/sft/sft_train.jsonl`
- Calibration examples: 3000
- Calibration batches: 10
- Calibrated layers: 22
- Quantized layers: 22
- Implementation note: Simplified GPTQ-style weighted-error fake quantization; no inverse-Hessian block compensation.

Top weighted-error layers:

- `lm_head` weighted_error=0.00000984, mse=0.00000983, scale=[0.006790, 0.018412]
- `layers.1.self_attn.k_proj` weighted_error=0.00000964, mse=0.00000959, scale=[0.008057, 0.014490]
- `layers.2.self_attn.k_proj` weighted_error=0.00000902, mse=0.00000900, scale=[0.007978, 0.014588]
- `layers.2.mlp.down_proj` weighted_error=0.00000881, mse=0.00000876, scale=[0.008055, 0.012695]
- `layers.0.self_attn.k_proj` weighted_error=0.00000862, mse=0.00000867, scale=[0.007596, 0.014188]

Limitations: this is not full GPTQ. It records Hessian-diagonal weighted error and then applies fake weight-only quantization without blockwise inverse-Hessian error compensation.

