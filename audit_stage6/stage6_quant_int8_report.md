# Stage 6 Weight-Only INT8 Report

## Command

```powershell
& '<local_python_executable>' scripts\quantize_eval.py --config configs\quant_int8.yaml
```

## Setup

- Checkpoint: `outputs/sft_full/checkpoints/best.pt`
- Method: `weight_only`
- Num bits: 8
- Per-channel: True
- Device/dtype: cuda / bf16
- Eval data: `data/sft/sft_val.jsonl`
- Eval max batches: 10

## Loss / Perplexity

- Baseline loss / ppl: 3.853451 / 47.1555
- Quantized loss / ppl: 3.854004 / 47.1816
- Loss delta / ppl delta: 0.000553 / 0.0261

## Size Estimate

- Estimated baseline size: 6,358,272 bytes (6.064 MiB)
- Estimated quantized size: 3,232,064 bytes (3.082 MiB)
- Compression ratio: 1.9672x

## Latency

- Baseline forward/generate latency: 3.344 ms / 52.421 ms
- Quantized forward/generate latency: 4.285 ms / 60.772 ms
- Note: Fake quantization uses dequantized floating-point F.linear and does not represent production integer-kernel speed.

## Artifacts

- Eval report: `outputs/quant_int8/eval_report.json`
- Samples: `outputs/quant_int8/samples/after.txt`

## Replacement Stats

- Linear layers found: 22
- Quantized layers: 22
- Skipped layers: 0

