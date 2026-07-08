# Stage 6 SmoothQuant-Style INT8 Report

## Command

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\quantize_eval.py --config configs\quant_smooth_int8.yaml
```

## Setup

- Checkpoint: `outputs/sft_full/checkpoints/best.pt`
- Method: `smoothquant_style`
- Num bits: 8
- Per-channel: True
- Device/dtype: cuda / bf16
- Eval data: `data/sft/sft_val.jsonl`
- Eval max batches: 10

## Loss / Perplexity

- Baseline loss / ppl: 3.853451 / 47.1555
- Quantized loss / ppl: 3.854174 / 47.1896
- Loss delta / ppl delta: 0.000723 / 0.0341

## Size Estimate

- Estimated baseline size: 6,358,272 bytes (6.064 MiB)
- Estimated quantized size: 3,252,800 bytes (3.102 MiB)
- Compression ratio: 1.9547x

## Latency

- Baseline forward/generate latency: 5.017 ms / 58.516 ms
- Quantized forward/generate latency: 5.302 ms / 67.172 ms
- Note: Fake quantization uses dequantized floating-point F.linear and does not represent production integer-kernel speed.

## Artifacts

- Eval report: `outputs/quant_smooth_int8/eval_report.json`
- Samples: `outputs/quant_smooth_int8/samples/after.txt`

## SmoothQuant-Style Details

- Alpha: 0.5
- Calibration examples: 3000
- Calibration batches: 10
- Calibrated layers: 22
- Quantized layers: 22
- Implementation note: Educational SmoothQuant-style wrapper; no graph fusion or integer kernel deployment.

Scale stats sample:

- `layers.0.self_attn.q_proj` scale_mean=6.3243, range=[5.1274, 7.9946], mse=0.00000002
- `layers.0.self_attn.k_proj` scale_mean=6.5620, range=[5.3074, 8.1304], mse=0.00000002
- `layers.0.self_attn.v_proj` scale_mean=7.5599, range=[5.5077, 9.5853], mse=0.00000001
- `layers.0.self_attn.o_proj` scale_mean=2.4536, range=[1.6966, 3.6528], mse=0.00000002
- `layers.0.mlp.gate_proj` scale_mean=6.6476, range=[5.4762, 8.3846], mse=0.00000002
- `layers.0.mlp.up_proj` scale_mean=6.6593, range=[5.3124, 8.0823], mse=0.00000002
- `layers.0.mlp.down_proj` scale_mean=2.4842, range=[1.4683, 4.4959], mse=0.00000004
- `layers.1.self_attn.q_proj` scale_mean=6.7111, range=[4.8553, 8.7481], mse=0.00000003

Limitations: this uses an educational wrapper that applies input scaling and fake quantized smoothed weights at runtime. It does not fuse graph operations or use integer kernels.

