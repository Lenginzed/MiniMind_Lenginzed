# Stage 6 Code Review

## Quantization QParams

`calculate_qparams_symmetric` supports int8/int4, per-tensor and per-channel symmetric scales. Quantized values are clamped to signed symmetric ranges. INT4 uses int8 storage for values in -7..7, with theoretical 4-bit size estimation.

## QuantizedLinear

`QuantizedLinear` stores `qweight`, `scale`, and optional bias as buffers. Forward dequantizes weight and calls `F.linear`, so this is fake quant inference. It is correct for educational comparisons but not an int-kernel implementation.

## Model Replacement

`quantize_model_weight_only` recursively replaces ordinary `nn.Linear` modules and skips `LoRALinear`. In the Full SFT checkpoint model, 22 Linear layers were replaced, including attention projections, MLP projections, and `lm_head`.

## GPTQ-Style

`collect_linear_calibration_stats` registers hooks to collect Linear input activation statistics and Hessian diagonal approximations. `apply_gptq_style_quantization` reports Hessian-diagonal weighted error, then applies fake weight-only quantization. This is explicitly simplified and does not implement inverse-Hessian blockwise error compensation.

## SmoothQuant-Style

`collect_smoothquant_stats` records activation absmax and weight absmax per input channel. `apply_smoothquant` computes `act_max ** alpha / weight_max ** (1 - alpha)` and replaces Linear with a wrapper that divides input by scale and uses fake-quantized `weight * scale`. It is not graph-fused deployment SmoothQuant.

## Eval Script

`quantize_eval.py` evaluates baseline and quantized loss/perplexity, estimates model size, measures rough forward/generate latency, writes generation samples, and saves JSON reports. Latency is reported with a warning because fake quant often dequantizes and is not expected to speed up inference.

## TODO

Not implemented: int4 bit packing, real int kernels, AutoGPTQ/GPTQModel, bitsandbytes, AWQ, activation quantization kernels, KV-cache quantization, vLLM deployment, and production benchmarking.
