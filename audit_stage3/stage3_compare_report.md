# Stage 3 Full SFT vs LoRA-SFT Compare

## Parameter Comparison

| Run | Total Params | Trainable Params | Trainable Ratio |
| --- | ---: | ---: | ---: |
| Full SFT | 1564992 | 1564992 | 1.000000 |
| LoRA-SFT | 1580352 | 15360 | 0.009719 |

LoRA trained only `15360` parameters, about `0.97%` of the model-with-adapters parameters.

## Loss Comparison

| Run | Initial Train | Final Train | Initial Eval | Final Eval |
| --- | ---: | ---: | ---: | ---: |
| Full SFT | 8.1192 | 3.5500 | 8.5907 | 3.7540 |
| LoRA-SFT | 7.9985 | 6.4172 | 8.6357 | 6.6694 |

Full SFT adapts faster in this short smoke run because all parameters are trainable. LoRA-SFT has a much smaller trainable footprint and lower update cost, but this tiny model and synthetic data are not enough to assess real instruction-following quality.

## Sample Summary

- Full SFT sample path: `outputs/sft_full/samples/after.txt`
- LoRA-SFT sample path: `outputs/sft_lora/samples/after.txt`
- Compare plot: `outputs/sft_compare/loss_compare.png`

Both sample files are smoke artifacts. Outputs are unstable and should not be treated as real QA ability.

## Limits

- Synthetic local SFT data.
- Tiny base model and short training.
- No real instruction-following benchmark.
- No DPO/GRPO/quantization in this stage.
