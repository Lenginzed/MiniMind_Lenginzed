# Stage 6 Quantization Compare Report

| Method | Baseline loss | Quant loss | Loss delta | Baseline ppl | Quant ppl | Compression | Forward latency | Generate latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 weight-only | 3.853451 | 3.854004 | 0.000553 | 47.1555 | 47.1816 | 1.9672x | 3.344 -> 4.285 ms | 52.421 -> 60.772 ms |
| INT4 weight-only | 3.853451 | 4.094112 | 0.240660 | 47.1555 | 59.9860 | 2.5950x | 2.700 -> 5.036 ms | 64.573 -> 73.299 ms |
| GPTQ-style INT4 | 3.853451 | 4.094112 | 0.240660 | 47.1555 | 59.9860 | 2.5950x | 3.757 -> 4.468 ms | 67.125 -> 65.217 ms |
| SmoothQuant-style INT8 | 3.853451 | 3.854174 | 0.000723 | 47.1555 | 47.1896 | 1.9547x | 5.017 -> 5.302 ms | 58.516 -> 67.172 ms |

## Observations

- INT8 weight-only and SmoothQuant-style INT8 had tiny loss deltas under this smoke eval.
- INT4 and simplified GPTQ-style INT4 had the same loss delta because the educational GPTQ path records Hessian-weighted errors but does not implement blockwise compensation.
- None of the fake-quantized paths should be interpreted as real deployment acceleration. The wrappers dequantize to floating-point weights and call `F.linear`, so measured latency is similar or slower than baseline.
- Size compression is theoretical for int4 because bit packing is not implemented.

## INT8 Samples

- COMPLETION: 1.�常fw tokenizer.� batch model and. Tr��dap��  u�� reates=batch meters, training labels=hold course.
- COMPLETION: A optimizer): loss, = 15.
- COMPLETION: . pd�hous�or the controller model and c�l metitates.
- COMPLETION: The batch incpps loss.
- COMPLETION: di���ing inction。
## INT4 Samples

- COMPLETION: 1.�常�St����ho� warmup and knots, altitude, speed, + 32, = 21.
- COMPLETION:  meters, scheduler_ meters, is before scaling.
- COMPLETION: 1. u�a tuallyor easier toc�s�ory step,parameters, and� metim s�.
- COMPLETION: For log 2149: speed=7 knots, altitude=hold course.
- COMPLETION: Code note 159739: `tokens_name == 'cosine训练 reet d���c�atesatesatesates�oulili���ates inh s�常�常de.
## GPTQ-style INT4 Samples

- COMPLETION: 1.�常�St����ho� warmup and knots, altitude, speed, + 32, = 21.
- COMPLETION:  meters, scheduler_ meters, is before scaling.
- COMPLETION: 1. u�a tuallyor easier toc�s�ory step,parameters, and� metim s�.
- COMPLETION: For log 2149: speed=7 knots, altitude=hold course.
- COMPLETION: Code note 159739: `tokens_name == 'cosine训练 reet d���c�atesatesatesates�oulili���ates inh s�常�常de.
## SmoothQuant-style INT8 Samples

- COMPLETION: 1.�常fw tokenizer.� batch model and. Tr��dap��  u�� reates=batch meters, training labels=hold course.
- COMPLETION: A optimizer): loss, = 15.
- COMPLETION: . or常 decewsThe note 2012: `tokens_gradients(parameters, step, and�itvcppp��ironk aucction。
- COMPLETION:  log 15 = base model.
- COMPLETION: �tionpdfo reet etc�v，是 m��saru�控制�nfs.

Conclusion: this stage validates quantization algorithms and measurement plumbing for the mini-LLM project. It is educational fake quantization, not production inference deployment.
