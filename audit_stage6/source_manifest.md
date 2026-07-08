# Stage 6 Source Manifest

## `minillm/quantization.py`

- Lines: 238
- Main symbols: def _validate_bits, def calculate_qparams_symmetric, def quantize_tensor_symmetric, def dequantize_tensor_symmetric, def fake_quantize_tensor_symmetric, class QuantizedLinear, def _target_match, def quantize_model_weight_only, def estimate_model_size_bytes, def estimate_quantized_size_bytes, def compression_report, def quantization_error

## `minillm/gptq.py`

- Lines: 148
- Main symbols: def _matches, def named_plain_linears, def collect_linear_calibration_stats, def _replace_module, def apply_gptq_style_quantization

## `minillm/smoothquant.py`

- Lines: 173
- Main symbols: def calculate_smooth_scale, class SmoothQuantLinear, def collect_smoothquant_stats, def apply_smoothquant

## `scripts/quantize_eval.py`

- Lines: 298
- Main symbols: def load_model, def build_loader, def evaluate_loss, def _sync, def measure_forward_latency, def measure_generate_latency, def write_samples, def finite_or_none, def run_quant_eval, def main

## `tests/test_quantization.py`

- Lines: 60
- Main symbols: def test_qparams_and_quant_dequant_shapes, def test_quantized_linear_forward_shape, def test_quantize_model_weight_only_replaces_linears

## `tests/test_gptq.py`

- Lines: 56
- Main symbols: class RandomTokenDataset, def collate, def build_model, def test_gptq_calibration_and_quantization_forward_shape

## `tests/test_smoothquant.py`

- Lines: 60
- Main symbols: class RandomTokenDataset, def collate, def build_model, def test_smooth_scale_finite_for_different_alpha, def test_apply_smoothquant_forward_shape_and_finite

## `configs/quant_int8.yaml`

- Lines: 12
- Main symbols: YAML config

## `configs/quant_int4.yaml`

- Lines: 12
- Main symbols: YAML config

## `configs/quant_gptq_int4.yaml`

- Lines: 14
- Main symbols: YAML config

## `configs/quant_smooth_int8.yaml`

- Lines: 15
- Main symbols: YAML config
