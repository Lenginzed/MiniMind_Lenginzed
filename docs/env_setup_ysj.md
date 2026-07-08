# YSJAirCombat Environment Setup

## Why Use YSJAirCombat

This project should use:

```powershell
D:\anaconda3\envs\YSJAirCombat\python.exe
```

The previous audit showed that the Anaconda `base` environment has a broken PyTorch install: package metadata exists, but `import torch` fails while loading `fbgemm.dll`. That makes `base` unsuitable for CUDA validation or later training work.

`YSJAirCombat` is preferred because PyTorch imports successfully, CUDA is visible, and RTX 4080 SUPER matmul smoke tests pass.

## Verify The Environment

Run:

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\verify_ysj_env.py
```

To regenerate the Stage 0.5 audit files:

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' scripts\verify_ysj_env.py --write-audit
```

The script prints:

- Python executable
- torch version
- `torch.version.cuda`
- `torch.cuda.is_available()`
- GPU name and VRAM
- fp32/fp16/bf16 CUDA matmul checks
- PyTorch SDPA availability
- missing minimal packages

## Minimal Dependencies

Stage 1/2 should keep dependencies small:

```powershell
& 'D:\anaconda3\envs\YSJAirCombat\python.exe' -m pip install -r requirements-minimal.txt
```

Only install missing minimal packages when needed. At audit time, the environment already had most required packages, but `pytest` was missing.

## If CUDA Is Healthy

If `verify_ysj_env.py` reports that torch imports, CUDA is available, and fp32/fp16/bf16 matmul tests pass, use `YSJAirCombat` for Stage 1 and later small GPU pretrain experiments.

Do not install large training stack extras yet. Add libraries in stages:

- Stage 1/2: `numpy`, `tqdm`, `matplotlib`, `tensorboard`, `pyyaml`, `pytest`, `safetensors`
- Stage 3+: evaluate `transformers`, `datasets`, `accelerate`, `peft`, `trl`
- QLoRA/quantization stage: evaluate `bitsandbytes`

## If CUDA Is Not Healthy

Only if `YSJAirCombat` also fails `import torch` or `torch.cuda.is_available()`, create a clean `mini_llm` environment. Do not create it preemptively.

## Deferred Libraries

Do not install these for the current stage:

- `vLLM`: useful later for serving-style inference, not needed for tiny model skeletons
- `flash-attn`: Windows installation can be high-friction; PyTorch SDPA is enough for Stage 1
- `DeepSpeed`: unnecessary for single-GPU mini models at this stage
- `bitsandbytes`: wait until Stage 3/QLoRA or quantization experiments
