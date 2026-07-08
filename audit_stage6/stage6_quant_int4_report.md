# Stage 6 Weight-Only INT4 Report

## Command

```powershell
& '<local_python_executable>' scripts\quantize_eval.py --config configs\quant_int4.yaml
```

## Setup

- Checkpoint: `outputs/sft_full/checkpoints/best.pt`
- Method: `weight_only`
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

- Baseline forward/generate latency: 2.700 ms / 64.573 ms
- Quantized forward/generate latency: 5.036 ms / 73.299 ms
- Note: Fake quantization uses dequantized floating-point F.linear and does not represent production integer-kernel speed.

## Artifacts

- Eval report: `outputs/quant_int4/eval_report.json`
- Samples: `outputs/quant_int4/samples/after.txt`

## Replacement Stats

- Linear layers found: 22
- Quantized layers: 22
- Skipped layers: 0
- INT4 storage note: weights are stored in int8 tensors for fake quant execution; size estimate assumes theoretical 4-bit weight storage without bit-packing implementation.

